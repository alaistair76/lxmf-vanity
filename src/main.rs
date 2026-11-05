//! Find a Reticulum identity whose LXMF address starts with a given hex prefix.
//!
//!     lxmf-vanity b00b1e5
//!
//! Reticulum derives an LXMF address from an identity like this (RNS 1.5.7,
//! Identity.py and Destination.py):
//!
//!     identity file = X25519 private key (32) + Ed25519 seed (32)
//!     public key    = X25519 public key (32) + Ed25519 public key (32)
//!     identity hash = SHA-256(public key)[..16]
//!     name hash     = SHA-256("lxmf.delivery")[..10]
//!     LXMF address  = SHA-256(name hash + identity hash)[..16]
//!
//! Each thread fixes a random Ed25519 key and walks X25519 private keys
//! k, k+8, k+16, ... Stepping by 8 keeps every key clamped, so it is used
//! as is. The public keys (u-coordinates of k*B) come from x-only
//! Montgomery differential addition, P(n+1) = P(n) + 8B with difference
//! P(n-1), and their Z coordinates are inverted a batch at a time. Every hit
//! is recomputed from scratch with x25519-dalek before it is saved.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::process::exit;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use curve25519_dalek::{Scalar, constants::X25519_BASEPOINT};
use ed25519_dalek::SigningKey;
use fiat_crypto::curve25519_64::*;
use sha2::{Digest, Sha256};

/// Public keys whose Z coordinates are inverted together.
const BATCH: usize = 256;
/// How often the main thread reports progress.
const REPORT_INTERVAL: Duration = Duration::from_secs(10);
const ASPECTS: &[u8] = b"lxmf.delivery";

#[cfg(feature = "cuda")]
mod gpu;

// ---- Field arithmetic mod 2^255 - 19, on fiat-crypto's verified primitives.

#[derive(Clone, Copy)]
struct Fe(fiat_25519_tight_field_element);

impl Fe {
    fn from_bytes(b: &[u8; 32]) -> Fe {
        let mut out = fiat_25519_tight_field_element([0; 5]);
        fiat_25519_from_bytes(&mut out, b);
        Fe(out)
    }
    fn to_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        fiat_25519_to_bytes(&mut out, &self.0);
        out
    }
    fn one() -> Fe {
        Fe(fiat_25519_tight_field_element([1, 0, 0, 0, 0]))
    }
    fn loose(self) -> fiat_25519_loose_field_element {
        let mut out = fiat_25519_loose_field_element([0; 5]);
        fiat_25519_relax(&mut out, &self.0);
        out
    }
    fn add(self, o: Fe) -> fiat_25519_loose_field_element {
        let mut out = fiat_25519_loose_field_element([0; 5]);
        fiat_25519_add(&mut out, &self.0, &o.0);
        out
    }
    fn sub(self, o: Fe) -> fiat_25519_loose_field_element {
        let mut out = fiat_25519_loose_field_element([0; 5]);
        fiat_25519_sub(&mut out, &self.0, &o.0);
        out
    }
    fn mul(self, o: Fe) -> Fe {
        mul_loose(&self.loose(), &o.loose())
    }
    fn square(self) -> Fe {
        let mut out = fiat_25519_tight_field_element([0; 5]);
        fiat_25519_carry_square(&mut out, &self.loose());
        Fe(out)
    }
    fn square_n(self, n: u32) -> Fe {
        let mut t = self;
        for _ in 0..n {
            t = t.square();
        }
        t
    }
    /// z^(p-2), the ref10 addition chain.
    fn invert(self) -> Fe {
        let z = self;
        let z2 = z.square();
        let z9 = z2.square_n(2).mul(z);
        let z11 = z9.mul(z2);
        let z2_5_0 = z11.square().mul(z9);
        let z2_10_0 = z2_5_0.square_n(5).mul(z2_5_0);
        let z2_20_0 = z2_10_0.square_n(10).mul(z2_10_0);
        let z2_40_0 = z2_20_0.square_n(20).mul(z2_20_0);
        let z2_50_0 = z2_40_0.square_n(10).mul(z2_10_0);
        let z2_100_0 = z2_50_0.square_n(50).mul(z2_50_0);
        let z2_200_0 = z2_100_0.square_n(100).mul(z2_100_0);
        let z2_250_0 = z2_200_0.square_n(50).mul(z2_50_0);
        z2_250_0.square_n(5).mul(z11)
    }
}

fn mul_loose(a: &fiat_25519_loose_field_element, b: &fiat_25519_loose_field_element) -> Fe {
    let mut out = fiat_25519_tight_field_element([0; 5]);
    fiat_25519_carry_mul(&mut out, a, b);
    Fe(out)
}

fn square_loose(a: &fiat_25519_loose_field_element) -> Fe {
    let mut out = fiat_25519_tight_field_element([0; 5]);
    fiat_25519_carry_square(&mut out, a);
    Fe(out)
}

// ---- Scalars: 256-bit little-endian integers.

/// k + 8n, as a little-endian 256-bit integer.
fn step_scalar(k: &[u8; 32], n: u64) -> [u8; 32] {
    let mut out = *k;
    let mut carry = (n as u128) * 8;
    for byte in out.iter_mut() {
        if carry == 0 {
            break;
        }
        let sum = *byte as u128 + (carry & 0xFF);
        *byte = sum as u8;
        carry = (carry >> 8) + (sum >> 8);
    }
    out
}

/// k - 8.
fn previous_scalar(k: &[u8; 32]) -> [u8; 32] {
    let mut out = *k;
    let mut borrow = 8u16;
    for byte in out.iter_mut() {
        let v = *byte as i16 - borrow as i16;
        *byte = v.rem_euclid(256) as u8;
        borrow = if v < 0 { 1 } else { 0 };
        if borrow == 0 {
            break;
        }
    }
    out
}

/// A random clamped X25519 key with room to step 2^64 times: bits 3..254
/// are neither all clear (so k - 8 stays clamped) nor so close to all set
/// that k + 8n would carry into bit 255.
fn random_start_scalar() -> [u8; 32] {
    loop {
        let mut k = [0u8; 32];
        getrandom::fill(&mut k).expect("no system randomness");
        k[0] &= 248;
        k[31] &= 127;
        k[31] |= 64;
        let high_bits_all_set = k[9..31].iter().all(|&b| b == 0xFF) && k[31] == 127;
        let low_bits_all_clear = k[..31].iter().all(|&b| b == 0) && k[31] == 64;
        if !high_bits_all_set && !low_bits_all_clear {
            return k;
        }
    }
}

// ---- Addresses.

fn name_hash() -> [u8; 10] {
    Sha256::digest(ASPECTS)[..10].try_into().unwrap()
}

/// The LXMF address for an X25519 and Ed25519 public key pair.
fn lxmf_address(name_hash: &[u8; 10], x_pub: &[u8; 32], ed_pub: &[u8; 32]) -> [u8; 16] {
    let mut public_key = [0u8; 64];
    public_key[..32].copy_from_slice(x_pub);
    public_key[32..].copy_from_slice(ed_pub);
    let identity_hash = Sha256::digest(public_key);
    let mut material = [0u8; 26];
    material[..10].copy_from_slice(name_hash);
    material[10..].copy_from_slice(&identity_hash[..16]);
    Sha256::digest(material)[..16].try_into().unwrap()
}

/// The LXMF address of a whole identity file, computed the plain way.
fn address_of_identity(identity: &[u8; 64]) -> [u8; 16] {
    let x_prv: [u8; 32] = identity[..32].try_into().unwrap();
    let seed: [u8; 32] = identity[32..].try_into().unwrap();
    let x_pub = x25519_dalek::x25519(x_prv, x25519_dalek::X25519_BASEPOINT_BYTES);
    let ed_pub = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
    lxmf_address(&name_hash(), &x_pub, &ed_pub)
}

/// A hex prefix as bytes and a mask for its last half byte.
struct Prefix {
    bytes: Vec<u8>,
    nibbles: usize,
}

impl Prefix {
    fn parse(s: &str) -> Result<Prefix, String> {
        let s = s.to_ascii_lowercase();
        if s.is_empty() || s.len() > 32 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("'{s}' isn't 1 to 32 hex digits: use 0-9 and a-f only (zero, not the letter O)."));
        }
        let padded = if s.len() % 2 == 1 { format!("{s}0") } else { s.clone() };
        let bytes = (0..padded.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&padded[i..i + 2], 16).unwrap())
            .collect();
        Ok(Prefix { bytes, nibbles: s.len() })
    }
    fn matches(&self, address: &[u8; 16]) -> bool {
        let whole = self.nibbles / 2;
        if address[..whole] != self.bytes[..whole] {
            return false;
        }
        self.nibbles % 2 == 0 || address[whole] >> 4 == self.bytes[whole] >> 4
    }
}

// ---- The search.

/// Walks X25519 public keys u(k + 8n) for n = 0, 1, 2, ...
struct KeyWalk {
    start: [u8; 32],
    next_n: u64,
    // P(n-1) and P(n), projective (X : Z).
    prev: (Fe, Fe),
    cur: (Fe, Fe),
    // u(8B) + 1 and u(8B) - 1, for differential addition.
    q_plus: fiat_25519_loose_field_element,
    q_minus: fiat_25519_loose_field_element,
}

impl KeyWalk {
    fn new(start: [u8; 32]) -> KeyWalk {
        let u = |k: [u8; 32]| Fe::from_bytes(&x25519_dalek::x25519(k, x25519_dalek::X25519_BASEPOINT_BYTES));
        let q = Fe::from_bytes(&(X25519_BASEPOINT * Scalar::from(8u8)).to_bytes());
        KeyWalk {
            start,
            next_n: 0,
            prev: (u(previous_scalar(&start)), Fe::one()),
            cur: (u(start), Fe::one()),
            q_plus: q.add(Fe::one()),
            q_minus: q.sub(Fe::one()),
        }
    }

    /// Fills `out` with the next public keys; returns the n of the first.
    fn next_batch(&mut self, out: &mut [[u8; 32]]) -> u64 {
        let first = self.next_n;
        let mut xs = [Fe::one(); BATCH];
        let mut zs = [Fe::one(); BATCH];
        let len = out.len();
        for i in 0..len {
            xs[i] = self.cur.0;
            zs[i] = self.cur.1;
            // xADD: P(n+1) = P(n) + Q, with P(n) - Q = P(n-1) and Q affine.
            let (x, z) = self.cur;
            let (xp, zp) = self.prev;
            let t1 = mul_loose(&x.sub(z), &self.q_plus);
            let t2 = mul_loose(&x.add(z), &self.q_minus);
            let x_next = zp.mul(square_loose(&t1.add(t2)));
            let z_next = xp.mul(square_loose(&t1.sub(t2)));
            self.prev = self.cur;
            self.cur = (x_next, z_next);
        }
        // Montgomery's trick: one inversion for the whole batch.
        let mut prefix = [Fe::one(); BATCH];
        let mut acc = Fe::one();
        for i in 0..len {
            prefix[i] = acc;
            acc = acc.mul(zs[i]);
        }
        let mut inv = acc.invert();
        for i in (0..len).rev() {
            let z_inv = inv.mul(prefix[i]);
            inv = inv.mul(zs[i]);
            out[i] = xs[i].mul(z_inv).to_bytes();
        }
        self.next_n += len as u64;
        first
    }
}

fn search(prefix: &Prefix, tries: &AtomicU64, found: &AtomicBool, hit: &Mutex<Option<[u8; 64]>>) {
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed).expect("no system randomness");
    let ed_pub = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
    let name_hash = name_hash();
    let mut walk = KeyWalk::new(random_start_scalar());
    let mut keys = [[0u8; 32]; BATCH];

    while !found.load(Ordering::Relaxed) {
        let first = walk.next_batch(&mut keys);
        for (i, x_pub) in keys.iter().enumerate() {
            if !prefix.matches(&lxmf_address(&name_hash, x_pub, &ed_pub)) {
                continue;
            }
            let mut identity = [0u8; 64];
            identity[..32].copy_from_slice(&step_scalar(&walk.start, first + i as u64));
            identity[32..].copy_from_slice(&seed);
            // Recompute from scratch: never save a key the fast path got wrong.
            if !prefix.matches(&address_of_identity(&identity)) {
                eprintln!("internal error: a hit failed its recheck; not saving it");
                exit(2);
            }
            let mut slot = hit.lock().unwrap();
            if slot.is_none() {
                *slot = Some(identity);
                found.store(true, Ordering::Relaxed);
            }
            return;
        }
        tries.fetch_add(BATCH as u64, Ordering::Relaxed);
    }
}

const USAGE: &str = "usage: lxmf-vanity [--gpu] <hex prefix>
       lxmf-vanity --gpu-self-test

Finds a Reticulum identity whose LXMF address starts with the prefix and
saves it to <prefix>.identity (readable only by you). --gpu searches on the
first NVIDIA GPU (needs a build with --features cuda).";

/// Searches on every CPU core; returns the identity and the keys tried.
fn search_cpu(prefix: &Prefix) -> ([u8; 64], u64) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let expected = 16f64.powi(prefix.nibbles as i32);
    println!("Searching on {threads} CPU threads; about {expected:.0} tries on average.");

    let tries = AtomicU64::new(0);
    let found = AtomicBool::new(false);
    let hit = Mutex::new(None);
    let start = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| search(prefix, &tries, &found, &hit));
        }
        let mut next_report = start + REPORT_INTERVAL;
        while !found.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
            if Instant::now() >= next_report {
                let done = tries.load(Ordering::Relaxed) as f64;
                let rate = done / start.elapsed().as_secs_f64();
                let left = (expected - done).max(0.0) / rate / 60.0;
                println!("{done:.0} tries, {rate:.0}/s; the average is about {left:.0} min away");
                next_report += REPORT_INTERVAL;
            }
        }
    });
    let identity = hit.lock().unwrap().expect("found without a hit");
    (identity, tries.load(Ordering::Relaxed))
}

#[cfg(feature = "cuda")]
fn open_gpu() -> gpu::Gpu {
    let started = Instant::now();
    let gpu = gpu::Gpu::new().unwrap_or_else(|e| {
        eprintln!("GPU unavailable: {e}");
        exit(1)
    });
    println!("{}: {} GPU threads, set up in {:.1} s.", gpu.device, gpu.threads(), started.elapsed().as_secs_f64());
    gpu
}

#[cfg(feature = "cuda")]
fn search_gpu(prefix: &Prefix) -> ([u8; 64], u64) {
    let mut gpu = open_gpu();
    println!("Searching on the GPU; about {:.0} tries on average.", 16f64.powi(prefix.nibbles as i32));
    gpu.search(prefix).unwrap_or_else(|e| {
        eprintln!("{e}");
        exit(2)
    })
}

#[cfg(not(feature = "cuda"))]
fn search_gpu(_: &Prefix) -> ([u8; 64], u64) {
    eprintln!("This build has no GPU support: cargo build --release --features cuda");
    exit(1)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flags: Vec<&str> = args.iter().filter(|a| a.starts_with("--")).map(|a| a.as_str()).collect();
    let words: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if flags.iter().any(|f| *f != "--gpu" && *f != "--gpu-self-test") {
        eprintln!("{USAGE}");
        exit(1);
    }

    if flags.contains(&"--gpu-self-test") {
        #[cfg(feature = "cuda")]
        {
            open_gpu().self_test().unwrap_or_else(|e| {
                eprintln!("Self-test FAILED: {e}");
                exit(2)
            });
            return;
        }
        #[cfg(not(feature = "cuda"))]
        search_gpu(&Prefix::parse("0").unwrap());
    }

    if words.len() != 1 {
        eprintln!("{USAGE}");
        exit(1);
    }
    let prefix = Prefix::parse(words[0]).unwrap_or_else(|e| {
        eprintln!("{e}");
        exit(1)
    });
    let path = format!("{}.identity", words[0].to_ascii_lowercase());
    if std::path::Path::new(&path).exists() {
        eprintln!("{path} already exists; not overwriting it.");
        exit(1);
    }
    println!("Looking for an LXMF address starting {}.", words[0].to_ascii_lowercase());

    let start = Instant::now();
    let (identity, tried) = if flags.contains(&"--gpu") { search_gpu(&prefix) } else { search_cpu(&prefix) };

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600) // the file holds a private key
        .open(&path)
        .unwrap_or_else(|e| {
            eprintln!("could not create {path}: {e}");
            exit(1)
        });
    file.write_all(&identity).expect("could not write the identity");

    let elapsed = start.elapsed().as_secs_f64();
    let address: String = address_of_identity(&identity).iter().map(|b| format!("{b:02x}")).collect();
    println!("Found {address}");
    println!("About {tried} tries in {elapsed:.1} s ({:.0}/s). Saved to {path}", tried as f64 / elapsed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn address_matches_reticulum_for_a_known_identity() {
        // A throwaway identity made by RNS 1.5.7; rnid reports its LXMF
        // address as b00b11fc764b58f813c3b3cc279f8b8a.
        let bytes: Vec<u8> = (0..128)
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(
                    &"e83d9ecdcdd012692583bd66f997bd0fd074d1ff05c62e892c3b5abbf1dd867801cc1353accbf59a42a1b951c71c6454cda7e4b8252ae6a5cb7fa55bb0430442"[i..i + 2],
                    16,
                )
                .unwrap()
            })
            .collect();
        let identity: [u8; 64] = bytes.try_into().unwrap();
        assert_eq!(hex(&address_of_identity(&identity)), "b00b11fc764b58f813c3b3cc279f8b8a");
    }

    #[test]
    fn walked_public_keys_match_plain_x25519_across_batches() {
        let start = random_start_scalar();
        let mut walk = KeyWalk::new(start);
        let mut keys = [[0u8; 32]; BATCH];
        for _ in 0..3 {
            let first = walk.next_batch(&mut keys);
            for (i, key) in keys.iter().enumerate() {
                let k = step_scalar(&start, first + i as u64);
                assert_eq!(*key, x25519_dalek::x25519(k, x25519_dalek::X25519_BASEPOINT_BYTES), "n = {}", first + i as u64);
            }
        }
    }

    #[test]
    fn stepped_scalars_stay_clamped() {
        let start = random_start_scalar();
        for n in [0u64, 1, 1000, u32::MAX as u64, u64::MAX / 16] {
            let k = step_scalar(&start, n);
            assert_eq!(k[0] & 7, 0);
            assert_eq!(k[31] & 0xC0, 0x40);
        }
        let p = previous_scalar(&start);
        assert_eq!(p[0] & 7, 0);
        assert_eq!(p[31] & 0xC0, 0x40);
        assert_eq!(step_scalar(&p, 1), start);
    }

    #[test]
    fn prefixes_match_whole_and_half_bytes() {
        let address = [0xb0, 0x0b, 0x1e, 0x5f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(Prefix::parse("B00B1E5").unwrap().matches(&address));
        assert!(Prefix::parse("b00b1e5f").unwrap().matches(&address));
        assert!(!Prefix::parse("b00b1e6").unwrap().matches(&address));
        assert!(Prefix::parse("BOOB").is_err());
    }
}
