//! The GPU search. The host picks each GPU thread's starting key, computes
//! its first two public keys on the CPU, and launches the `walk` kernel
//! (common.h) repeatedly through a backend: CUDA (kernel.cu, with
//! `--features cuda`) or Metal (kernel.metal, on macOS). Every hit is
//! rebuilt as an identity and rechecked on the CPU before it is accepted.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use curve25519_dalek::{Scalar, constants::X25519_BASEPOINT};
use ed25519_dalek::SigningKey;

use crate::{Prefix, address_of_identity, name_hash, previous_scalar, random_start_scalar, step_scalar};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(target_os = "macos")]
mod metal;

const COMMON: &str = include_str!("common.h");
/// Hits recorded per launch; more than one matters only for the self-test.
const MAX_HITS: u32 = 4096;
/// Aim for launches about this long: short enough that progress, stopping
/// and macOS's GPU watchdog are no trouble.
const TARGET_LAUNCH: Duration = Duration::from_millis(500);

/// What one launch of `walk` needs besides the per-thread state.
pub struct LaunchArgs {
    pub iters: u32,
    pub ed: [u32; 8],
    pub name_hash: [u32; 3],
    pub want: [u32; 4],
    pub mask: [u32; 4],
    pub q: [u32; 8],
    pub max_hits: u32,
    pub dump_threads: u32,
}

/// A GPU API that can run the kernels in common.h.
pub trait Backend {
    /// The GPU, for the start-up line.
    fn describe(&self) -> String;
    fn threads(&self) -> u32;
    /// Keys per batch inversion (BATCH in common.h).
    fn batch(&self) -> u64;
    /// Sets up every thread from u(k-8) and u(k): 16 little-endian words each.
    fn init(&mut self, u_prev_cur: &[u32]) -> Result<(), String>;
    /// Runs `walk` once. Returns the (thread, n) hits and the dumped public
    /// keys (BATCH x 8 words per dumped thread).
    fn launch(&mut self, args: &LaunchArgs) -> Result<(Vec<(u32, u64)>, Vec<u32>), String>;
}

/// The kernel source for one backend: its entry points with common.h
/// spliced in where they #include it.
pub fn kernel_source(entry: &str, batch: u64) -> String {
    format!("#define BATCH {batch}\n{}", entry.replace("#include \"common.h\"", COMMON))
}

fn open_backend() -> Result<Box<dyn Backend>, String> {
    #[cfg(target_os = "macos")]
    return Ok(Box::new(metal::Metal::new()?));
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    return Ok(Box::new(cuda::Cuda::new()?));
    #[cfg(not(any(feature = "cuda", target_os = "macos")))]
    Err("this build has no GPU support: on Linux, build with --features cuda".into())
}

fn le_words(b: &[u8; 32]) -> [u32; 8] {
    core::array::from_fn(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
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

pub struct GpuSearch {
    backend: Box<dyn Backend>,
    starts: Vec<[u8; 32]>,
    seed: [u8; 32],
    ed: [u32; 8],
    q: [u32; 8],
    name_hash: [u32; 3],
}

impl GpuSearch {
    /// Opens the GPU and gives each of its threads a random starting key.
    pub fn open() -> Result<GpuSearch, String> {
        let mut backend = open_backend()?;
        let starts: Vec<[u8; 32]> = (0..backend.threads()).map(|_| random_start_scalar()).collect();
        backend.init(&first_public_keys(&starts))?;

        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).expect("no system randomness");
        let ed_pub = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
        let nh = name_hash();
        Ok(GpuSearch {
            backend,
            starts,
            seed,
            ed: core::array::from_fn(|i| u32::from_be_bytes(ed_pub[i * 4..i * 4 + 4].try_into().unwrap())),
            q: le_words(&(X25519_BASEPOINT * Scalar::from(8u8)).to_bytes()),
            name_hash: [
                u32::from_be_bytes(nh[0..4].try_into().unwrap()),
                u32::from_be_bytes(nh[4..8].try_into().unwrap()),
                (nh[8] as u32) << 24 | (nh[9] as u32) << 16,
            ],
        })
    }

    pub fn describe(&self) -> String {
        format!("{}: {} GPU threads", self.backend.describe(), self.backend.threads())
    }

    fn launch(&mut self, prefix: &Prefix, iters: u32, dump_threads: u32) -> Result<(Vec<(u32, u64)>, Vec<u32>), String> {
        let (want, mask) = prefix_words(prefix);
        self.backend.launch(&LaunchArgs {
            iters,
            ed: self.ed,
            name_hash: self.name_hash,
            want,
            mask,
            q: self.q,
            max_hits: MAX_HITS,
            dump_threads,
        })
    }

    /// The identity for thread `t`'s key number `n`.
    fn identity(&self, t: u32, n: u64) -> [u8; 64] {
        let mut identity = [0u8; 64];
        identity[..32].copy_from_slice(&step_scalar(&self.starts[t as usize], n));
        identity[32..].copy_from_slice(&self.seed);
        identity
    }

    /// Searches until this or another worker sets `found`, adding the keys
    /// tried to `tries`; a match goes in `hit`.
    pub fn run(&mut self, prefix: &Prefix, found: &AtomicBool, tries: &AtomicU64, hit: &Mutex<Option<[u8; 64]>>) -> Result<(), String> {
        let per_iter = self.backend.threads() as u64 * self.backend.batch();
        let mut iters = 1u32;
        while !found.load(Ordering::Relaxed) {
            let launched = Instant::now();
            let (hits, _) = self.launch(prefix, iters, 0)?;
            let took = launched.elapsed();
            tries.fetch_add(per_iter * iters as u64, Ordering::Relaxed);
            if let Some(&(t, n)) = hits.first() {
                let identity = self.identity(t, n);
                if !prefix.matches(&address_of_identity(&identity)) {
                    return Err(format!("GPU hit (thread {t}, key {n}) failed its CPU recheck: a kernel bug"));
                }
                let mut slot = hit.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(identity);
                }
                found.store(true, Ordering::Relaxed);
                return Ok(());
            }
            // Grow launches towards the target length.
            if took < TARGET_LAUNCH / 2 && iters < 1 << 16 {
                iters *= 2;
            }
        }
        Ok(())
    }

    /// Checks the kernel against the CPU: its public keys against plain
    /// X25519, and every hit for a short prefix against the CPU's address.
    pub fn self_test(&mut self) -> Result<String, String> {
        let dump_threads = self.backend.threads().min(256);
        let batch = self.backend.batch();
        let any = Prefix::parse("b0").unwrap(); // about 1 key in 256
        let (hits, dumped) = self.launch(&any, 1, dump_threads)?;
        for t in 0..dump_threads {
            for i in 0..batch {
                let k = step_scalar(&self.starts[t as usize], i);
                let expected = x25519_dalek::x25519(k, x25519_dalek::X25519_BASEPOINT_BYTES);
                let base = ((t as u64 * batch + i) * 8) as usize;
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
        Ok(format!(
            "{} public keys match X25519, {} hits for 'b0' all recheck on the CPU.",
            dump_threads as u64 * batch,
            hits.len()
        ))
    }
}

/// u(k - 8) and u(k) for each start key k, as 16 little-endian words per
/// thread, computed across all CPU cores.
fn first_public_keys(starts: &[[u8; 32]]) -> Vec<u32> {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = starts.len().div_ceil(cores).max(1);
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
