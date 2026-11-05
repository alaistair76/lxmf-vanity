//! The CUDA search (built with `--features cuda`). The host picks each GPU
//! thread's starting key, computes its first two public keys on the CPU,
//! and launches `walk` (src/kernel.cu) repeatedly. Every hit is rebuilt as
//! an identity and rechecked on the CPU before it is accepted.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use curve25519_dalek::{Scalar, constants::X25519_BASEPOINT};
use ed25519_dalek::SigningKey;

use crate::{Prefix, address_of_identity, name_hash, previous_scalar, random_start_scalar, step_scalar};

const KERNEL: &str = include_str!("kernel.cu");
/// Keys per batch inversion (BATCH in kernel.cu).
const KERNEL_BATCH: u64 = 64;
const BLOCK: u32 = 128;
/// GPU threads per streaming multiprocessor.
const THREADS_PER_SM: u32 = 2048;
/// Hits recorded per launch; more than one matters only for the self-test.
const MAX_HITS: u32 = 4096;
/// Aim for launches about this long, so progress and stopping stay prompt.
const TARGET_LAUNCH: Duration = Duration::from_millis(1000);
const REPORT_INTERVAL: Duration = Duration::from_secs(10);

pub struct Gpu {
    stream: Arc<CudaStream>,
    walk: CudaFunction,
    threads: u32,
    starts: Vec<[u8; 32]>,
    seed: [u8; 32],
    state: CudaSlice<i32>,
    ns: CudaSlice<u64>,
    ed: CudaSlice<u32>,
    q: CudaSlice<u32>,
    name_hash: [u32; 3],
    pub device: String,
}

/// Big-endian words of the address bytes the prefix fixes, and their mask.
fn prefix_words(prefix: &Prefix) -> ([u32; 4], [u32; 4]) {
    let mut want = [0u8; 16];
    let mut mask = [0u8; 16];
    for i in 0..prefix.nibbles {
        let shift = if i % 2 == 0 { 4 } else { 0 };
        mask[i / 2] |= 0xF << shift;
        want[i / 2] |= ((prefix.bytes[i / 2] >> shift) & 0xF) << shift;
    }
    let words = |b: [u8; 16]| core::array::from_fn(|i| u32::from_be_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()));
    (words(want), words(mask))
}

fn le_words(b: &[u8; 32]) -> [u32; 8] {
    core::array::from_fn(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        let err = |e: &dyn std::fmt::Debug| format!("{e:?}");
        let ctx = CudaContext::new(0).map_err(|e| err(&e))?;
        let stream = ctx.default_stream();
        let (major, minor) = ctx.compute_capability().map_err(|e| err(&e))?;
        let sms = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| err(&e))? as u32;
        let device = format!("{} (compute {major}.{minor}, {sms} SMs)", ctx.name().map_err(|e| err(&e))?);

        let opts = CompileOptions {
            options: vec![format!("--gpu-architecture=compute_{major}{minor}"), format!("-DBATCH={KERNEL_BATCH}")],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL, opts).map_err(|e| err(&e))?;
        let module = ctx.load_module(ptx).map_err(|e| err(&e))?;
        let init = module.load_function("init_walk").map_err(|e| err(&e))?;
        let walk = module.load_function("walk").map_err(|e| err(&e))?;

        let threads = sms * THREADS_PER_SM;
        let starts: Vec<[u8; 32]> = (0..threads).map(|_| random_start_scalar()).collect();
        let u_prev_cur = first_public_keys(&starts);

        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).expect("no system randomness");
        let ed_pub = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
        let ed_words: Vec<u32> = (0..8).map(|i| u32::from_be_bytes(ed_pub[i * 4..i * 4 + 4].try_into().unwrap())).collect();
        let q_words = le_words(&(X25519_BASEPOINT * Scalar::from(8u8)).to_bytes());
        let nh = name_hash();
        let name_hash = [
            u32::from_be_bytes(nh[0..4].try_into().unwrap()),
            u32::from_be_bytes(nh[4..8].try_into().unwrap()),
            (nh[8] as u32) << 24 | (nh[9] as u32) << 16,
        ];

        let mut state = stream.alloc_zeros::<i32>(threads as usize * 40).map_err(|e| err(&e))?;
        let ns = stream.alloc_zeros::<u64>(threads as usize).map_err(|e| err(&e))?;
        let ed = stream.clone_htod(&ed_words).map_err(|e| err(&e))?;
        let q = stream.clone_htod(&q_words[..]).map_err(|e| err(&e))?;
        let upc = stream.clone_htod(&u_prev_cur).map_err(|e| err(&e))?;
        let cfg = LaunchConfig { grid_dim: (threads.div_ceil(BLOCK), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        let mut launch = stream.launch_builder(&init);
        launch.arg(&mut state).arg(&upc).arg(&threads);
        unsafe { launch.launch(cfg) }.map_err(|e| err(&e))?;
        stream.synchronize().map_err(|e| err(&e))?;

        Ok(Gpu { stream, walk, threads, starts, seed, state, ns, ed, q, name_hash, device })
    }

    pub fn threads(&self) -> u32 {
        self.threads
    }

    /// Keys tried per launch of `iters` batches.
    fn keys_per_launch(&self, iters: u32) -> u64 {
        self.threads as u64 * iters as u64 * KERNEL_BATCH
    }

    /// One launch. Returns the (thread, n) hits, and the public keys of the
    /// first batch of the first `dump_threads` threads.
    fn launch(&mut self, prefix: &Prefix, iters: u32, dump_threads: u32) -> Result<(Vec<(u32, u64)>, Vec<u32>), String> {
        let err = |e: &dyn std::fmt::Debug| format!("{e:?}");
        let s = &self.stream;
        let (want, mask) = prefix_words(prefix);
        let want = s.clone_htod(&want[..]).map_err(|e| err(&e))?;
        let mask = s.clone_htod(&mask[..]).map_err(|e| err(&e))?;
        let mut hit_count = s.alloc_zeros::<u32>(1).map_err(|e| err(&e))?;
        let mut hits = s.alloc_zeros::<u64>(MAX_HITS as usize * 2).map_err(|e| err(&e))?;
        let dump_len = (dump_threads.max(1) as u64 * KERNEL_BATCH * 8) as usize;
        let mut dump = s.alloc_zeros::<u32>(dump_len).map_err(|e| err(&e))?;
        let cfg = LaunchConfig { grid_dim: (self.threads.div_ceil(BLOCK), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        let mut launch = s.launch_builder(&self.walk);
        launch
            .arg(&mut self.state)
            .arg(&mut self.ns)
            .arg(&self.threads)
            .arg(&iters)
            .arg(&self.ed)
            .arg(&self.name_hash[0])
            .arg(&self.name_hash[1])
            .arg(&self.name_hash[2])
            .arg(&want)
            .arg(&mask)
            .arg(&mut hit_count)
            .arg(&mut hits)
            .arg(&MAX_HITS)
            .arg(&mut dump)
            .arg(&dump_threads)
            .arg(&self.q);
        unsafe { launch.launch(cfg) }.map_err(|e| err(&e))?;
        s.synchronize().map_err(|e| err(&e))?;

        let count = s.clone_dtoh(&hit_count).map_err(|e| err(&e))?[0].min(MAX_HITS) as usize;
        let mut found = Vec::new();
        if count > 0 {
            let raw = s.clone_dtoh(&hits).map_err(|e| err(&e))?;
            found = (0..count).map(|i| (raw[i * 2] as u32, raw[i * 2 + 1])).collect();
        }
        let dumped = if dump_threads > 0 { s.clone_dtoh(&dump).map_err(|e| err(&e))? } else { Vec::new() };
        Ok((found, dumped))
    }

    /// The identity for thread `t`'s key number `n`.
    fn identity(&self, t: u32, n: u64) -> [u8; 64] {
        let mut identity = [0u8; 64];
        identity[..32].copy_from_slice(&step_scalar(&self.starts[t as usize], n));
        identity[32..].copy_from_slice(&self.seed);
        identity
    }

    /// Search until a key matches; returns it with the number of keys tried.
    pub fn search(&mut self, prefix: &Prefix) -> Result<([u8; 64], u64), String> {
        let expected = 16f64.powi(prefix.nibbles as i32);
        let start = Instant::now();
        let mut next_report = start + REPORT_INTERVAL;
        let mut tried = 0u64;
        let mut iters = 1u32;
        loop {
            let launched = Instant::now();
            let (hits, _) = self.launch(prefix, iters, 0)?;
            let took = launched.elapsed();
            tried += self.keys_per_launch(iters);
            for (t, n) in hits {
                let identity = self.identity(t, n);
                if !prefix.matches(&address_of_identity(&identity)) {
                    return Err(format!("GPU hit (thread {t}, key {n}) failed its CPU recheck: a kernel bug"));
                }
                return Ok((identity, tried));
            }
            // Grow launches towards the target length.
            if took < TARGET_LAUNCH / 2 && iters < 1 << 16 {
                iters *= 2;
            }
            if Instant::now() >= next_report {
                let rate = tried as f64 / start.elapsed().as_secs_f64();
                let left = (expected - tried as f64).max(0.0) / rate / 60.0;
                println!("{tried} tries, {rate:.0}/s; the average is about {left:.0} min away");
                next_report += REPORT_INTERVAL;
            }
        }
    }

    /// Checks the kernel against the CPU: its public keys against plain
    /// X25519, and every hit for a short prefix against the CPU's address.
    pub fn self_test(&mut self) -> Result<(), String> {
        const DUMP_THREADS: u32 = 256;
        let any = Prefix::parse("b0").unwrap(); // about 1 key in 256
        let (hits, dumped) = self.launch(&any, 1, DUMP_THREADS)?;
        for t in 0..DUMP_THREADS {
            for i in 0..KERNEL_BATCH {
                let k = step_scalar(&self.starts[t as usize], i);
                let expected = x25519_dalek::x25519(k, x25519_dalek::X25519_BASEPOINT_BYTES);
                let base = ((t as u64 * KERNEL_BATCH + i) * 8) as usize;
                let got: Vec<u8> = dumped[base..base + 8].iter().flat_map(|w| w.to_le_bytes()).collect();
                if got != expected {
                    return Err(format!("public key mismatch: thread {t}, key {i}"));
                }
            }
        }
        if hits.len() < 100 {
            return Err(format!("only {} hits for a 2-digit prefix; expected thousands", hits.len()));
        }
        for &(t, n) in &hits {
            if !any.matches(&address_of_identity(&self.identity(t, n))) {
                return Err(format!("hit (thread {t}, key {n}) failed its CPU recheck"));
            }
        }
        println!(
            "Self-test passed: {} public keys match X25519, {} hits for 'b0' all recheck on the CPU.",
            DUMP_THREADS as u64 * KERNEL_BATCH,
            hits.len()
        );
        Ok(())
    }
}

/// u(k - 8) and u(k) for each start key k, as 16 little-endian words per
/// thread, computed across all CPU cores.
fn first_public_keys(starts: &[[u8; 32]]) -> Vec<u32> {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = starts.len().div_ceil(cores);
    let mut out = vec![0u32; starts.len() * 16];
    std::thread::scope(|s| {
        for (keys, words) in starts.chunks(chunk).zip(out.chunks_mut(chunk * 16)) {
            s.spawn(move || {
                for (k, w) in keys.iter().zip(words.chunks_mut(16)) {
                    let u = |k: [u8; 32]| le_words(&x25519_dalek::x25519(k, x25519_dalek::X25519_BASEPOINT_BYTES));
                    w[..8].copy_from_slice(&u(previous_scalar(k)));
                    w[8..].copy_from_slice(&u(*k));
                }
            });
        }
    });
    out
}
