# lxmf-vanity

Finds a [Reticulum](https://reticulum.network) identity whose LXMF address starts with a chosen hex prefix, such as `b00b1e5…` or `0000000000…`, on CPU cores, an NVIDIA GPU, or several GPU machines at once. The result is a standard 64-byte Reticulum identity file that you can load into Sideband, MeshChat or NomadNet.

```sh
cargo build --release
./target/release/lxmf-vanity b00b1e5
rnid -i b00b1e5.identity -H lxmf.delivery    # confirm the address
```

The identity file holds the private key, so it's written readable only by you. Keep it private, and never commit it (`.gitignore` excludes `*.identity`).

Prefixes are hex only: 0–9 and a–f (zero, not the letter O).

## How long it takes

Each extra hex digit means 16 times as many tries.

| Prefix length | Tries on average | Apple M5 Max CPU (106 M/s) | One NVIDIA GB10 (630 M/s) | Four GB10s (2.5 G/s) |
|---|---|---|---|---|
| 7 | 268 million | 2.5 s | under 1 s | under 1 s |
| 8 | 4.3 billion | 40 s | 7 s | 2 s |
| 9 | 69 billion | 11 min | 2 min | 27 s |
| 10 | 1.1 trillion | 3 h | 29 min | 7 min |
| 11 | 18 trillion | 2 days | 8 h | 2 h |
| 12 | 281 trillion | 1 month | 5 days | 31 h |
| 32 (the whole address) | 2¹²⁸ | never | never | never |

These are averages: about 1 search in 20 takes three times as long. The rates come from the measurements below.

A whole chosen address is impossible on any hardware, at about 10²¹ years even on the four-GPU setup. That's what stops anyone taking over someone else's address.

## Measurements

**Speed by method**

| Method | Hardware | Keys per second |
|---|---|---|
| Python loop with `RNS.Identity()`, 1 core | AMD Ryzen 9 7900X | 18,500 |
| Python loop, every core | AMD Ryzen 9 7900X (10 threads) | 100,000 |
| Python loop, every core | Apple M5 Max (18 cores) | 300,000 |
| `lxmf-vanity`, CPU | AMD Ryzen 9 7900X (10 threads) | 28 million |
| `lxmf-vanity`, CPU | Apple M5 Max (18 cores) | 106 million |
| `lxmf-vanity --gpu` | 1 × NVIDIA GB10 (DGX Spark) | 630 million |
| `fleet.sh` | 4 × NVIDIA GB10 | 2.5 billion |

On the same Mac, the key-walking method is about 350 times faster than generating fresh identities in Python.

**The GB10 figures are with the GPU clock limited.** The SM clock ran at 2,190 MHz of a possible 3,003 MHz, drawing about 65 W. The kernel is limited by arithmetic, so a GPU at full clock should be faster, perhaps by up to a third. That hasn't been measured.

**GPU tuning, one GB10:**

| Kernel version | Keys per second |
|---|---|
| First working version (32 keys per inversion) | 500 million |
| Plus a dedicated squaring function | 582 million |
| Plus 64 keys per inversion | 628 million |
| 128 keys per inversion | 630 million (no better, so 64 is used) |

**Real 10-digit searches on 4 × GB10.** The average is 1.1 trillion tries, about 7 minutes:

| Prefix | Time | Address found |
|---|---|---|
| `0000000000` | 6 min 9 s | `0000000000f4d61f4c8f3525a26c8193` |
| `1111111111` | 1 min 35 s | `111111111126ad771ee92dec3f19d900` |
| `2222222222` | 47 s | `22222222227948a1ad00e597df2b57ef` |
| `3333333333` | 13 min 56 s | `3333333333e0f202ce5a3c9813ceaef1` |

Times vary widely because each try is an independent chance. A search is no closer to finishing after 10 minutes than when it started.

## On an NVIDIA GPU

You need CUDA 13 and the NVIDIA driver. The kernel is compiled for your GPU when the program starts.

```sh
cargo build --release --features cuda
./target/release/lxmf-vanity --gpu-self-test     # checks the GPU against the CPU
./target/release/lxmf-vanity --gpu 0000000000
```

If the CUDA libraries aren't found, set `LD_LIBRARY_PATH=/usr/local/cuda/lib64`.

## On several GPU machines

`fleet.sh` copies the source to each machine, builds it there, and searches on all of them at once. The first to find a match wins. The others are stopped, the identity is copied back, and every remote copy is deleted. Ctrl-C stops them all.

```sh
./fleet.sh -H gpu1,gpu2,gpu3 c0ffee1234    # these machines
./fleet.sh c0ffee1234                      # the machines listed in ./hosts
./fleet.sh -n 2 c0ffee1234                 # just the first 2 of them
```

**Machines** come from, in order:
1. `-H`, a comma-separated list
2. the `LXMF_VANITY_HOSTS` environment variable
3. a `hosts` file next to the script: one ssh destination per line, as in `hosts.example`. It's git-ignored, so your machine names stay private.

**Each machine needs:**
- ssh access without a password prompt
- an NVIDIA GPU with CUDA 13
- Rust, from [rustup.rs](https://rustup.rs)

**Optional environment settings:**

| Setting | What it is | Default |
|---|---|---|
| `CUDA_LIB` | where the CUDA libraries are | `/usr/local/cuda/lib64` |
| `REMOTE_DIR` | the folder used on each machine | `lxmf-vanity` |
| `REPORT_SECONDS` | how often progress is shown | `10` |

## How it works

Reticulum derives an LXMF address from an identity like this:

```
identity      = X25519 private key (32 bytes) + Ed25519 seed (32 bytes)
identity hash = SHA-256(X25519 public key + Ed25519 public key)[:16]
LXMF address  = SHA-256(SHA-256("lxmf.delivery")[:10] + identity hash)[:16]
```

Making a fresh pair of keys for every try is slow. Instead:
- **Fixed Ed25519 key:** each CPU or GPU thread picks one random Ed25519 key and keeps it.
- **Stepping X25519 keys:** it walks X25519 private keys `k, k+8, k+16, …` from a random start. Steps of 8 keep each key valid as it is.
- **Cheap public keys:** each public key comes from the previous two by Montgomery differential addition, a handful of field multiplications instead of a full scalar multiplication.
- **Shared inversions:** the projective Z coordinates are inverted 64 at a time with Montgomery's trick.

**Checking results:**
- Every match, from the CPU or GPU, is recomputed from scratch with `x25519-dalek` before it's saved.
- `--gpu-self-test` compares thousands of GPU public keys with plain X25519, and rechecks every GPU match on the CPU.
- `cargo test` checks the address calculation against an identity made by Reticulum itself.

## Licence

MIT
