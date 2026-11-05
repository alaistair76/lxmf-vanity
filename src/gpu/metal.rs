//! The Metal backend (Apple GPUs, always built on macOS). The kernel is
//! compiled from source for the GPU found, when the program starts. Buffers
//! use shared storage: Apple GPUs share memory with the CPU, so nothing is
//! copied.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use super::{Backend, LaunchArgs, kernel_source};

// MTLCreateSystemDefaultDevice lives in CoreGraphics.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

const ENTRY: &str = include_str!("kernel.metal");
/// Keys per batch inversion, and GPU threads (LXMF_VANITY_GPU_THREADS
/// overrides it): the fastest pair measured on an M5 Max (154 M keys/s;
/// batch 64-256 and 32-256 K threads tried). Apple GPUs have no native
/// 64-bit integer arithmetic, so each field multiply costs more than on
/// CUDA, and a big batch amortising the inversion pays off more.
const BATCH: u64 = 256;
const DEFAULT_THREADS: u32 = 128 * 1024;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Must match InitArgs in kernel.metal.
#[repr(C)]
struct InitArgs {
    threads: u32,
}

/// Must match WalkArgs in kernel.metal.
#[repr(C)]
struct WalkArgs {
    threads: u32,
    iters: u32,
    nh0: u32,
    nh1: u32,
    nh2: u32,
    max_hits: u32,
    dump_threads: u32,
}

pub struct Metal {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    init: Pipeline,
    walk: Pipeline,
    threads: u32,
    state: Buffer,
    ns: Buffer,
}

impl Metal {
    pub fn new() -> Result<Metal, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal GPU found")?;
        let source = NSString::from_str(&kernel_source(ENTRY, BATCH));
        let options = MTLCompileOptions::new();
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| format!("Metal kernel did not compile: {}", e.localizedDescription()))?;
        let pipeline = |name: &str| -> Result<Pipeline, String> {
            let function = library.newFunctionWithName(&NSString::from_str(name)).ok_or(format!("no kernel {name}"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("Metal pipeline {name}: {}", e.localizedDescription()))
        };
        let init = pipeline("init_walk")?;
        let walk = pipeline("walk")?;
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;
        let threads = std::env::var("LXMF_VANITY_GPU_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_THREADS);
        let state = new_buffer(&device, threads as usize * 40 * 4)?;
        let ns = new_buffer(&device, threads as usize * 8)?;
        Ok(Metal { device, queue, init, walk, threads, state, ns })
    }

    /// Runs one kernel over every thread and waits for it.
    fn dispatch(&self, pipeline: &Pipeline, buffers: &[&Buffer], args: (NonNull<c_void>, usize, usize)) -> Result<(), String> {
        let commands = self.queue.commandBuffer().ok_or("no Metal command buffer")?;
        let encoder = commands.computeCommandEncoder().ok_or("no Metal compute encoder")?;
        encoder.setComputePipelineState(pipeline);
        let (bytes, len, args_index) = args;
        let mut index = 0;
        for buffer in buffers {
            if index == args_index {
                index += 1;
            }
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
            index += 1;
        }
        unsafe { encoder.setBytes_length_atIndex(bytes, len, args_index) };
        let group = pipeline.maxTotalThreadsPerThreadgroup().min(256);
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: self.threads as usize, height: 1, depth: 1 },
            MTLSize { width: group, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        commands.commit();
        commands.waitUntilCompleted();
        match commands.error() {
            Some(e) => Err(format!("Metal kernel failed: {}", e.localizedDescription())),
            None => Ok(()),
        }
    }
}

fn new_buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Result<Buffer, String> {
    let buffer = device
        .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
        .ok_or("could not allocate a Metal buffer")?;
    unsafe { std::ptr::write_bytes(buffer.contents().as_ptr() as *mut u8, 0, bytes.max(4)) };
    Ok(buffer)
}

fn buffer_from(device: &ProtocolObject<dyn MTLDevice>, words: &[u32]) -> Result<Buffer, String> {
    let buffer = new_buffer(device, words.len() * 4)?;
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), buffer.contents().as_ptr() as *mut u32, words.len()) };
    Ok(buffer)
}

fn read<T: Copy>(buffer: &Buffer, len: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const T, len).to_vec() }
}

fn bytes_of<T>(value: &T) -> (NonNull<c_void>, usize) {
    (NonNull::from(value).cast(), std::mem::size_of::<T>())
}

impl Backend for Metal {
    fn describe(&self) -> String {
        format!("{} (Metal)", self.device.name())
    }

    fn threads(&self) -> u32 {
        self.threads
    }

    fn batch(&self) -> u64 {
        BATCH
    }

    fn init(&mut self, u_prev_cur: &[u32]) -> Result<(), String> {
        let upc = buffer_from(&self.device, u_prev_cur)?;
        let args = InitArgs { threads: self.threads };
        let (ptr, len) = bytes_of(&args);
        self.dispatch(&self.init, &[&self.state, &upc], (ptr, len, 2))
    }

    fn launch(&mut self, a: &LaunchArgs) -> Result<(Vec<(u32, u64)>, Vec<u32>), String> {
        let d = &self.device;
        let ed = buffer_from(d, &a.ed)?;
        let want = buffer_from(d, &a.want)?;
        let mask = buffer_from(d, &a.mask)?;
        let hit_count = new_buffer(d, 4)?;
        let hits = new_buffer(d, a.max_hits as usize * 16)?;
        let dump_words = a.dump_threads as usize * BATCH as usize * 8;
        let dump = new_buffer(d, dump_words * 4)?;
        let q = buffer_from(d, &a.q)?;
        let args = WalkArgs {
            threads: self.threads,
            iters: a.iters,
            nh0: a.name_hash[0],
            nh1: a.name_hash[1],
            nh2: a.name_hash[2],
            max_hits: a.max_hits,
            dump_threads: a.dump_threads,
        };
        let (ptr, len) = bytes_of(&args);
        // Buffer indexes as in kernel.metal; 2 is the arguments.
        self.dispatch(&self.walk, &[&self.state, &self.ns, &ed, &want, &mask, &hit_count, &hits, &dump, &q], (ptr, len, 2))?;

        let count = (read::<u32>(&hit_count, 1)[0]).min(a.max_hits) as usize;
        let raw = read::<u64>(&hits, count * 2);
        let found = (0..count).map(|i| (raw[i * 2] as u32, raw[i * 2 + 1])).collect();
        Ok((found, read::<u32>(&dump, dump_words)))
    }
}
