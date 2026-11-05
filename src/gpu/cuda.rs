//! The CUDA backend (NVIDIA GPUs, `--features cuda`). The kernel is compiled
//! with NVRTC for the GPU found, when the program starts.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use super::{Backend, LaunchArgs, kernel_source};

const ENTRY: &str = include_str!("kernel.cu");
/// Keys per batch inversion: 64 measured fastest on a GB10 (128 was no better).
const BATCH: u64 = 64;
const BLOCK: u32 = 128;
/// GPU threads per streaming multiprocessor.
const THREADS_PER_SM: u32 = 2048;

pub struct Cuda {
    stream: Arc<CudaStream>,
    init: CudaFunction,
    walk: CudaFunction,
    threads: u32,
    state: CudaSlice<i32>,
    ns: CudaSlice<u64>,
    device: String,
}

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

impl Cuda {
    pub fn new() -> Result<Cuda, String> {
        // cudarc panics, rather than failing, when the driver library is
        // missing; turn that into an error (and keep its long message quiet).
        let quiet = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let ctx = std::panic::catch_unwind(|| CudaContext::new(0));
        std::panic::set_hook(quiet);
        let ctx = ctx
            .map_err(|_| "the NVIDIA driver (libcuda) wasn't found; set LD_LIBRARY_PATH if it is installed".to_string())?
            .map_err(err)?;
        let stream = ctx.default_stream();
        let (major, minor) = ctx.compute_capability().map_err(err)?;
        let sms = ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(err)? as u32;
        let device = format!("{} (CUDA, compute {major}.{minor}, {sms} SMs)", ctx.name().map_err(err)?);

        let opts = CompileOptions {
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(kernel_source(ENTRY, BATCH), opts).map_err(err)?;
        let module = ctx.load_module(ptx).map_err(err)?;
        let threads = sms * THREADS_PER_SM;
        Ok(Cuda {
            init: module.load_function("init_walk").map_err(err)?,
            walk: module.load_function("walk").map_err(err)?,
            state: stream.alloc_zeros::<i32>(threads as usize * 40).map_err(err)?,
            ns: stream.alloc_zeros::<u64>(threads as usize).map_err(err)?,
            stream,
            threads,
            device,
        })
    }

    fn config(&self) -> LaunchConfig {
        LaunchConfig { grid_dim: (self.threads.div_ceil(BLOCK), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }
    }
}

impl Backend for Cuda {
    fn describe(&self) -> String {
        self.device.clone()
    }

    fn threads(&self) -> u32 {
        self.threads
    }

    fn batch(&self) -> u64 {
        BATCH
    }

    fn init(&mut self, u_prev_cur: &[u32]) -> Result<(), String> {
        let upc = self.stream.clone_htod(u_prev_cur).map_err(err)?;
        let cfg = self.config();
        let mut launch = self.stream.launch_builder(&self.init);
        launch.arg(&mut self.state).arg(&upc).arg(&self.threads);
        unsafe { launch.launch(cfg) }.map_err(err)?;
        self.stream.synchronize().map_err(err)
    }

    fn launch(&mut self, a: &LaunchArgs) -> Result<(Vec<(u32, u64)>, Vec<u32>), String> {
        let s = self.stream.clone();
        let ed = s.clone_htod(&a.ed[..]).map_err(err)?;
        let want = s.clone_htod(&a.want[..]).map_err(err)?;
        let mask = s.clone_htod(&a.mask[..]).map_err(err)?;
        let q = s.clone_htod(&a.q[..]).map_err(err)?;
        let mut hit_count = s.alloc_zeros::<u32>(1).map_err(err)?;
        let mut hits = s.alloc_zeros::<u64>(a.max_hits as usize * 2).map_err(err)?;
        let mut dump = s.alloc_zeros::<u32>((a.dump_threads.max(1) as u64 * BATCH * 8) as usize).map_err(err)?;
        let cfg = self.config();
        let mut launch = s.launch_builder(&self.walk);
        launch
            .arg(&mut self.state)
            .arg(&mut self.ns)
            .arg(&self.threads)
            .arg(&a.iters)
            .arg(&ed)
            .arg(&a.name_hash[0])
            .arg(&a.name_hash[1])
            .arg(&a.name_hash[2])
            .arg(&want)
            .arg(&mask)
            .arg(&mut hit_count)
            .arg(&mut hits)
            .arg(&a.max_hits)
            .arg(&mut dump)
            .arg(&a.dump_threads)
            .arg(&q);
        unsafe { launch.launch(cfg) }.map_err(err)?;
        s.synchronize().map_err(err)?;

        let count = s.clone_dtoh(&hit_count).map_err(err)?[0].min(a.max_hits) as usize;
        let mut found = Vec::new();
        if count > 0 {
            let raw = s.clone_dtoh(&hits).map_err(err)?;
            found = (0..count).map(|i| (raw[i * 2] as u32, raw[i * 2 + 1])).collect();
        }
        let dumped = if a.dump_threads > 0 { s.clone_dtoh(&dump).map_err(err)? } else { Vec::new() };
        Ok((found, dumped))
    }
}
