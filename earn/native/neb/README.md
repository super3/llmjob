# NEB (nebhash v2) cores

NEB (Neural Engine Byte) is a proof-of-work coin whose attempt is one integer training step of a small
network the chain owns. The CLI mines it on the cards `--card-coin N:neb` names, paid to `--neb-address`.
It is experimental and in no release yet.

The JavaScript side is in `src/shared/neb/` (the reference, the pool protocol, the address check, the CLI
run) and `src/main/nebEngine.js` and `nebCore.js` (the miner and the core interface, with a CPU core).
This folder has the GPU cores and their tools.

## One attempt

    key    = BLAKE3(80-byte header)              the nonce is bytes 76 to 79
    idx    = 16 samples of the block's dataset, from ChaCha8(key)
    g      = the sum of the 16 samples' gradients on the block's model, each entry clipped to +-256
    digest = 8 lanes: D[f mod 8] += g[f] XOR (f + 7919 * layer)
    ticket = BLAKE3(digest || key)               a share when at or under the pool's target

The network has two datasets, Kepler (13,404 samples) and TESS (7,089), and blocks alternate between
them. Both run the KEP1 network (26,000 weights of 4 bits). The pool sends each block's model with its
job. Source: github.com/SilasEigen/neb, `spec/shares.md`, `spec/nebhash-v2.md`, `spec/pool.md`.
`src/shared/neb/nebhash2.js` is our port of its C reference and replays its vectors.

## How a card mines it

A sample whose output already equals its target has a zero gradient: a dead row. On the chain's models
65% to 77% of samples are dead, so most attempts touch several live rows, and those are costly. A core
does not evaluate every nonce:

1. **Screen.** For each nonce: the key, the 16 sample indices, and the sum of one byte a sample, which says
   how many are live. About 1,130 instructions, 700 of them on the card's adder, which is what limits it.
2. **Pairs.** An attempt with at most 2 live rows is evaluated: its digest is read from a table of every
   pair of live rows, built once a model (350 MB for Kepler, 40 MB for TESS, a tenth of a second).
3. **Triples.** An attempt with exactly 3 live rows too, when its rows are in a table of row triples
   (`buildTriples`). It is rows^3 / 6 entries of 32 bytes: 21 GB for 1,600 rows, which is every row of a TESS
   model and a third of a Kepler model's, built in 1.5 to 2.4 s on an RTX 4090 and 1.9 to 5.7 s on an RX 7900
   XTX. The engine decides for each model how many rows are worth the build time (`nebEngine.triplePlan`). A
   card has one buffer for it and the two datasets take turns, since a table is good for one block.

Two scan kernels, picked by the model's share of candidates (`neb_host.inc`):

- **`nbk_lean`**, under 15% (Kepler). A thread screens all its nonces and lists the few that pass in its
  block's list; the block shares the list out afterwards. A warp's threads step together, so what one does
  for a candidate the other 31 wait for: shared out, the tickets are done by full warps.
- **`nbk_scan`**, from 15% up (TESS): rounds of screen and tickets, the candidates in shared memory.

`nbk_lean` keeps the sample bytes in the block's shared memory, where a look-up is one instruction. A live
sample whose row the triple table does not cover has the byte 17, so the sum alone says whether the tables
have an attempt, and nothing else leaves the screen.

The triple table is built a slab (one highest row) at a time, in tiles of 32 rows by 32 rows a block, so
that a block's warps read the same rows and an SM keeps them near: a build is as fast as the rows reach the
SM, not as the card adds. `neb_kernel.h` says what is shared between entries.

Everything else is skipped. That costs no luck: a nonce not tried is not a lost ticket. The rate that
earns is attempts evaluated a second, and that is what the miner shows. It moves with every model: the
share evaluated was 2.7% to 4.4% on Kepler models and 24% to 31% on TESS models on 28 Nov 2026.

The host evaluates every hit again on the reference before it sends it, and takes a card off after three
hits that are not hits.

## Files

| File | What it is |
|---|---|
| `neb_kernel.h` | All device code, with no vendor's names in it |
| `neb_host.inc`, `neb_host.h` | The host side and the kernels' wrappers, shared by CUDA and HIP through a name macro |
| `cuda/neb_cuda.cu`, `amd/neb_hip.cpp` | Each vendor's names and its timed defaults: about 30 lines each |
| `cuda/build.sh`, `cuda/build.ps1`, `amd/build.sh` | The builds: NVIDIA on Linux and on Windows, AMD on both |
| `neb_napi.c` | The N-API face (`NEB_API 1`), the same file for both |
| `ref/` | The chain's C reference, unchanged (Apache-2.0, see `ref/NOTICE`). The host uses it to evaluate each sample once a model |
| `vectors/` | The chain's vectors, a trained model for the test dataset, and two recorded pool models |
| `verify-core.js` | Checks a core against the reference on a card |
| `tools/nebbench.js` | Times a core on the recorded models, with no pool |
| `tools/nebmine.js` | Mines on a pool with the engine alone |
| `tools/nebwatch.js` | Prints the pool's models and blocks, for reading another miner's rate against its model |
| `tools/nbox.sh` | The Salad test-box script for both vendors |

The interface is at the top of `src/main/nebCore.js`.

## Building

    bash earn/native/neb/cuda/build.sh        # NVIDIA, Linux: CUDA 12.8 or newer; build/neb_cuda.node
    bash earn/native/neb/amd/build.sh         # AMD, Linux: ROCm 7; build/neb_hip.node
    powershell -ExecutionPolicy Bypass -File earn\native\neb\cuda\build.ps1    # NVIDIA, Windows: CUDA 12.8, MSVC
    bash earn/native/neb/amd/build.sh         # AMD, Windows: Git Bash, MSVC's environment, AMD's HIP SDK 7.2

The Windows builds link the C runtime in and delay-load node.exe, as the Quantus cores do, so the same file
loads in Node and in the desktop app. `.github/workflows/native-core.yml` builds all four (`neb-cuda`,
`neb-cuda-windows`, `neb-amd`, `neb-amd-windows`), checks each loads in Node with no card and carries code
for every target, and keeps it as an artifact. No Windows card has run a NEB core yet. The cores need RTX 20
or newer, or an RDNA 2, 3 or 4 card.

## Proving a core right

    node earn/native/neb/verify-core.js --addon build/neb_cuda.node
    node earn/native/neb/verify-core.js --addon build/neb_cuda.node --dataset neb-datasets.bin \
        --pool neb.rabidmining.com:3470 --address neb1…

- **vectors**: the chain's 87 KEP1 shares, digest and ticket from the card, on each model state.
- **full**: 64 random keys in full, and the live-sample count.
- **exact**, **top**: two stretches of nonces at two targets; the hits and the count evaluated must be
  exactly the reference's.
- **overflow**: more hits than asked for.
- **big**: 2^30 nonces; every hit rechecked, and the share evaluated must be what the model predicts.
- **triples**, **big3**: the same after `buildTriples`.

The second form runs the model checks on the network's dataset and the model the pool is on. Then mine
on a pool with a real address (`tools/nebmine.js`) and check shares are accepted.

## Not done

- No packaging: no installer carries the cores, and no Windows card has run one.
- The CLI does not download `neb-datasets.bin` (16 MB); it must be on disk.
- NEB cards send no row to the board or the stats file, and there is no `--coin neb` and nothing in the
  desktop app.
- No Intel Arc core. The device code needs nothing Arc lacks.
- The rows are evaluated on the host's CPU threads once a model (0.2 to 0.6 s).
