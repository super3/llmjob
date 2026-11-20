# Pearl PIP-3 (FP8): what it is, what we must change, and the plan

Written 20 November 2026 by agent f-pip3-spec. A second agent, f-pip3-gpu, ran Pearl's own FP8 miner on real
hardware the same day; its results are the section "Hardware runs, 20 November".

## Where things stand

- PIP-3 replaces Pearl's INT7 proof of work with a floating-point one. It is a hard fork: at the fork block every
  miner that is not running the new scheme stops earning.
- Only NVIDIA Hopper (compute 9.0) and Blackwell (10.0 and 12.0: B200, RTX PRO 6000, GeForce RTX 50) can mine
  after it. RTX 20/30/40, every AMD card and every Intel card cannot. No software can fix that: the verifier
  repeats one chip family's arithmetic bit for bit, and only those two families have a verifier.
- Mainnet has no fork height. Pearl aims for 4 to 8 weeks after 18 November, so 16 December to 13 January.
  Testnet forks at 67,415 and testnet2 at 150,283; testnet2 passed its fork on 18 November.
- We have a bit-exact CPU reference for Blackwell (`earn/src/shared/miner/fp8.js`). We have no FP8 GPU core, no
  pool protocol for it, and the app has no way to tell a user their card stops.

## Sources

| What | Where | State |
|---|---|---|
| PIP-3 text, merged | `pearl-research-labs/pips`, `main`, `PIPS/pip-0003.md` | Draft. Certificate v3, rank 16, names Hopper and Blackwell. Out of date. |
| PIP-3 text, updated | same repo, branch `pips/pip3_update_2609`, commit `103d7ae` (22 Sep) | "Proposed", not merged. Certificate v4, rank 32, FP10. Leaves all exact arithmetic to the whitepaper. Test vectors: "TBD". |
| Code | `pearl-research-labs/pearl`, branch `fp8-scheme`, head `1dd75a52` (18 Nov) | Runs testnet and testnet2. Not merged to `master`; v1.5.7 has none of it. |
| Consensus arithmetic | `zk-pow/src/v4/api/*.rs` on that branch | What the node runs. This is what the reference ports. |
| Reference miner | `miner/pearl-gemm` (kernels), `miner/vllm-miner` (vLLM plugin) | Python that compiles kernels at run time. Has SM90, SM100 and SM120 kernels. |
| Fork heights | `node/chaincfg/params.go` | Mainnet `Fp8ForkHeight` is unset (0 = off). |

Local clones: `C:\Users\template\code\perf-scratch\pip3-src\pearl` and `...\pips`.

Where the PIP text and the code disagree, the code is what testnet2 enforces, so this file follows the code.
Issues #104 and #303 (RTX 50 failures) are about the old INT7 miner on `master`. The `fp8-scheme` miner has its
own SM120 kernels.

## What a share is under PIP-3

A miner picks two matrices, A (m rows) and B (n rows), both k wide, and computes C' = A' · B'ᵀ, where A' and B'
are noised FP8 copies. C' is cut into tiles. Each tile hashes to a ticket. A ticket below the target wins.

### Sizes

| Thing | Rule |
|---|---|
| k | 1024 to 65536, a multiple of 32 |
| Noise rank r | exactly 32 |
| m, n | 1 to 2^24 rows each |
| Tile, h rows of A by w rows of B | h ≥ 4, w ≥ 16, h·w a multiple of 16, 256 ≤ h·w ≤ 2048 |
| Tile position | rows and columns each start at a multiple of h and w |
| Opened data per share | under 4 MB, about (h + w) · k · 1.25 bytes |

The miner chooses m, n, k, h and w and commits to them. Pearl's own miner uses 4×64 and 4×128 tiles on Hopper,
4×64 to 4×256 on RTX 50, and has a 16×32 tile only on B200. The rules allow our present 16×16.

### Storage: FP10

Each operand is committed as two planes: one INT8 value per entry, and one BF16 scale per 8 consecutive entries
of a row. That is 10 bits an entry. The real value is `int8 × scale`. Each plane is a keyed BLAKE3 Merkle tree
over chunks of 128, 256, 512 or 1024 bytes (the miner picks; Pearl's miner uses 1024, which is the tree we
already build). The operand's commitment is `BLAKE3(values_root ‖ scales_root, key)`.

### Keys and seeds

All hashes are BLAKE3 with a role label, `pearl/v4/FP8/<role>`.

```
keyA  = H("key-A",  proposed header, 76 bytes)
keyB  = H("key-B",  an ancestor's complete header, 108 bytes)
seedB = H("seed-B", HB ‖ keyB ‖ pB)
seedA = H("seed-A", HA ‖ seedB ‖ keyA ‖ pA)
pB    = k u32, r u16, quant u8 (0), device u8, n u32, hash_id u8, w u32, experts u16 (0)
pA    = m u32, hash_id u8, h u32
```

- The device byte is the chip family: 0 is Hopper ("H100"), 1 is Blackwell ("B200"). It is inside seedB, so it
  is fixed before the noise is known. RTX 50 and RTX PRO 6000 declare 1.
- B is keyed by an ancestor block up to 4 back, so one B and its noise can be reused across blocks. A is keyed by
  the block being mined.
- The header has no nonce. A miner gets a new attempt by changing A.

### Noise

Each operand gets rank-32 noise E · F. A noise line is 32 bytes of keyed BLAKE3 output for the address
`side, factor, line number`. Each byte becomes a signed integer in ±[1, 128]; the line is scaled to an L2 norm of
256 with one BF16 division and cast to E4M3. E has one line per row, F one line per column. Only A's E lines
depend on A. B's E lines and both F bases come from seedB alone.

### Row scales and the cast to FP8

For each row, from the committed planes:

1. `l2` is the row's RMS, computed in exact integers with one correctly rounded square root, then rounded to a
   multiple of 4 BF16 steps. `linf` is the largest magnitude. Both are floored at 2^-32.
2. `alpha = 448 / (linf + delta·√32 · l2)` and `beta = alpha · l2 · delta·√32 / 65536`, every step in BF16.
   `delta` is 1.0 on Hopper and 0.5 on Blackwell. 448 is the largest E4M3 value.
3. Each entry is `fma(beta, (E·F) entry, alpha · scale · int8)` in FP32, cast to E4M3 with round to nearest even.
4. "Downscale rewrite": a code smaller than `e4m3(g · alpha · l2)` is replaced by a fixed small exponent with its
   low mantissa bit set, when a coin taken from the FP32 bits of its 4-column group passes. `g` is 0.08 on
   Blackwell (0.1 when `alpha · l2 < 8`) and 0.105 on Hopper.

E4M3 is 1 sign bit, 4 exponent bits (bias 7) and 3 mantissa bits, with subnormals and no infinities.

### The FP32 sum each chip family must reproduce

This is the part that makes the fork hardware-specific.

**Blackwell (B200, RTX PRO 6000, RTX 50).** For one output cell, k is walked in ascending order in groups of 32
products. One group is one MMA instruction. It takes the running FP32 value and 32 exact products, lines them up
at the largest exponent among those 33 terms, cuts each term toward zero to 25 bits below that exponent, adds
them exactly, and cuts the result toward zero to FP32. Nothing rounds to nearest. A product more than 25 bits
below the running sum is simply lost.

**Hopper (H100, H200).** The same instruction keeps 13 bits instead of 25. Every 128 products the local
accumulator restarts from +0, and the finished window is added to the total with one normal FP32 add.

Two things follow for any GPU core:

- One accumulator per cell, walking k upward. Splitting k across accumulators, reordering k, or summing pairs
  first all give different bits. Our rotation fold and k-order search have no place here.
- The 32-product group is the instruction's own width (`mma.sync m16n8k32` on RTX 50, `tcgen05` on B200,
  `wgmma` on Hopper). The core cannot choose a different one.

Pearl's code says RTX 50's `mma.sync` gives the same bits as B200's `tcgen05`. Pearl's notes say its tests ran on
a B200. It holds on GeForce: an RTX 5080 declaring B200 mined a testnet2 block (see "Hardware runs").

### Ticket

The h×w tile of C', row by row, is cut into 16 equal runs. Each run folds into a 32-bit word:
`word = rotl32(word · 0x9E3779B1 + float bits, 13)`. The 16 words are a 64-byte message. The ticket is that
message hashed under a key derived from seedA. It wins when `ticket ≤ target · h · w · k`. There is no transcript
of intermediate values any more: only the final product is hashed.

### Jackpot policy

A winning tile must also pass four checks on the opened rows, or it is void. They exist because a float sum
drops small terms, so a miner could pick data whose result is predictable without doing the work.

1. At most 1/64 of a side's entries may be larger than `8 · delta · l2` of their row.
2. Every row's `alpha · l2` must be at least 4 on A and 8 on B.
3. At most 1/20 of the tile's h·w·k products may be small enough to be skipped, by an integer score.
4. Every row must have a mean "noise coverage" of at least 1/25, by an integer score.

Random INT8 data with a centred triangular spread and unit scales passes; Pearl's CPU test miner uses it. **A constant fill, which is what our miner uses today, fails
check 4.** Rows that are one spike, or flat, fail.

### Proof, and how a pool checks a share

A share is a `PlainProofV4`: the job tuple (ancestor header, pB, pA), up to 3 headers linking the ancestor to the
block, and Merkle openings of the tile's h rows of A and w rows of B, values and scales. The pool runs
`verify_plain_proof(header, proof, share_nbits)`. That function opens the rows, rebuilds the noise, A' and B',
replays the tile on the declared device's arithmetic, runs the policy and compares the ticket with the share
target. The zero-knowledge proof is made only for a block, by whoever submits it.

## Certain and draft

| Item | State |
|---|---|
| The arithmetic above, as testnet2 enforces it | Certain for commit `1dd75a52`. The reference passes Pearl's own pinned numbers for it. |
| Certificate version 4 | In the code and the updated text. The merged text still says 3. |
| Chip families: Hopper and Blackwell | In the code. The updated text mentions only Blackwell. Whether Hopper stays is open. |
| RTX 50 gives B200's bits | Shown on 20 November: testnet2 accepted a block from an RTX 5080, and regtest nodes accepted blocks from an RTX 5080 and an RTX PRO 6000. |
| Constants (rank 32, delta, strip `g`, policy thresholds, state window 4) | Draft. Rank was 16 in July. They can move again before mainnet. |
| Official test vectors | None. The PIP says "TBD". There is no end-to-end fixture in Pearl's repository. Ours is the RTX 5080's testnet2 proof. |
| Mainnet fork height | Not set. |
| Pool protocol for v4 shares | Nothing published by HeroMiners or Kryptex. |
| Speed and memory on RTX 50 | Measured with Pearl's kernels: 158 to 163 T-MAC/s on an RTX 5080, 242 on an RTX 5090 (see "Hardware runs"). |

## What must change in our code

### Miner core and reference

| Today | Under PIP-3 |
|---|---|
| `earn/src/shared/miner/pearlhash.js`: profile with k 2048, rank 128, `mmaType 0` (INT7), 52-byte config, 16×16 patterns | New job tuple (pB, pA), rank 32, device byte, tile sizes instead of patterns |
| `noise.js`: uniform noise and a sparse permutation | Noise lines, row scales, FP8 cast. Done in `fp8.js` |
| `reference.js`, `checkHit.js`: INT32 transcript fold, INT7 range check | Float tile fold, jackpot policy. Done in `fp8.js` (`evaluateTile`) |
| `merkle.js`: one tree per operand | Two trees per operand (values, scales) and the combined digest. The tree itself is unchanged |
| `plainProof.js`, `shareProof.js`: bincode `PlainProof` | bincode `PlainProofV4`: job bytes, ancestor chain, four openings |
| Constant operand fill with a salt | Data that passes the policy, for example pseudo-random INT8 with unit scales |
| `earn/native/src/*.cu`: INT8 IMMA, dp4a and HMMA folds for sm_61 to sm_120; `pearl_rot.cu`; HIP and SYCL cores | A new FP8 core. Nothing in the present kernels carries over except the BLAKE3 code and the host plumbing |
| `verify-hits.js`: recompute hits with `checkHit` | Same tool, calling `fp8.evaluateTile` |

The FP8 core for sm_120 needs four GPU stages, as Pearl's miner has:

1. Hash A's two planes (keyed BLAKE3 Merkle) and take the row sums that give `l2` and `linf`.
2. Draw E lines, run E·F as rank-32 FP8 MMAs, mix, cast, downscale. The same once per job for B.
3. The main GEMM with `mma.sync m16n8k32` FP8, accumulators in registers, k ascending.
4. Fold each tile, hash it, compare with the target.

Then on the CPU: re-check each hit with `fp8.evaluateTile` (which also applies the policy) and build the proof.

Worth knowing from Pearl's SM120 kernel: the block-scaled form of the instruction (`kind::mxf8f6f4` with all
scale factors 1.0) runs at twice the rate of the plain FP8 form and is stated to give the same bits.

### sm_90 and sm_100

- **sm_100 (B200)** uses the same device byte and the same reference as RTX 50, but a different instruction
  (`tcgen05`, tensor memory). It is a separate kernel.
- **sm_90 (H100)** needs `wgmma`, the 128-product windows, its own constants and its own policy replay.
- Our users have neither card. I would not build either until RTX 50 is mining, and then only if rented
  data-centre cards turn out to pay.

### Pool and protocol

Our stratum (`earn/src/shared/miner/stratum.js`, `earn/src/main/pearlMiner.js`) gets a 76-byte header, a target
and `cert_version`, and submits a base64 bincode proof. `cert_version` is parsed and never read. Needed:

- Read `cert_version` per job and pick the scheme from it. Do not hardcode a height. Pearl's guide says the same.
- The job must carry the ancestor's complete 108-byte header, and the linking headers if the pool lets B stay
  keyed to an older block. Today's notify has neither.
- Submit a `PlainProofV4`. A share is larger than now: about (h + w) · k · 1.25 bytes before compression.
- The `rank` gate and the rank-128 penalty go away for v4.
- Both pools have to publish this first. Until then the only place to test is our own testnet2 node with Pearl's
  `pearl-gateway`.

### The app: who keeps mining and what the others see

Today the coin is one manual choice for the whole rig (Settings → GPU Coin, or `--coin`). There is no per-card
coin, no backup coin and no automatic switch. A card that cannot mine shows up only as a log line.

Needed:

1. **A rule for who can mine Pearl after the fork**, in `cardFamily.js`: NVIDIA compute 12.x, plus 9.0 and 10.x
   if we build those cores. Everything else cannot.
2. **Before the fork**: a notice in the app and CLI for RTX 20/30/40, AMD and Intel users that Pearl mining ends
   for their card at the fork, with the date once Pearl sets a height.
3. **At the fork**: when a pool sends a `cert_version 4` job, cards without an FP8 core stop Pearl with a clear
   message, not a stream of rejected shares.
4. **Switch those cards to Quantus.** That needs a coin per card instead of per rig, because a rig with an RTX 50
   and an RTX 30 should mine Pearl on one and Quantus on the other. `appSession.planFor` and `startQuantus` assume
   one coin today.
5. The benchmark pages and the website's card table.

## The CPU reference

`earn/src/shared/miner/fp8.js`, with tests in `earn/test/minerFp8.test.js` (68 tests, 100% coverage, eslint
clean). Nothing in the miner requires it. `fp8.enabled(env)` is true only with `PEARL_FP8=1`, for later wiring.

It covers, for Blackwell: the E4M3 and BF16 casts, the BF16 and FP32 fused multiply-adds, keys and seeds, noise
lines, FP10 opening and exact norms, row scales, the noised cast with the downscale rewrite, the matmul, all
four policy checks, the fold, the ticket and the difficulty test. For Hopper it has the matmul and the constants
but not the policy.

It agrees with every number Pearl pins in its own tests:

- 9 matmul vectors generated by Pearl's Python model of the B200, which Pearl says was checked on B200 silicon;
- the per-device pin of one noise value, alpha, beta and FP8 code, for both B200 and H100;
- 4 noise lines; the torch vectors for FP10 opening, norms, scales and the BF16 multiply-add;
- the Python-blake3 vectors for every role label; the pB and pA encodings; the policy's kernel table and row
  constants; and the accept or reject verdicts of Pearl's policy tests, rebuilt from the same inputs.

It also agrees with consensus on a whole tile. `earn/test/fixtures/fp8Rtx5080Testnet2.json` is the proof of
testnet2 block 150,688, which an RTX 5080 mined with Pearl's miner on 20 November: both headers, the four plane
roots and the opened rows of the winning 4×64 tile, k 2048. From those, `fp8.js` derives the keys and seeds,
passes the policy, and gets the ticket `dff32409…00000000`, which is under the network target. A wrong bit
anywhere would give a random ticket, and one of those meets that target once in 57 billion tries.

Not ported: the Merkle openings of `PlainProofV4`, the MoE (mixture-of-experts) variant, the ZK proof.

## Hardware runs, 20 November

By agent f-pip3-gpu: Pearl's own FP8 miner on rented cards.

The question: does FP8 Pearl mining work today on the cards that keep Pearl after the fork, and how fast is it?

The answer: yes on every card tried. An RTX 5080 mined a testnet2 block after the fork and the network built on
it, and our CPU reference replays that block's proof to the same winning ticket. Pearl's kernels run on RTX 50
under Linux and under WSL2. Pearl's miner program does not run on a Windows host, and it gets about a ninth of
what its own kernels can do.

### What was run

Pearl's reference miner, `pearl-research-labs/pearl` branch `fp8-scheme` at `1dd75a52` (18 November):

- `miner/pearl-gemm`: the kernels. Python (CuTe DSL) that compiles for the card at run time. Four steps a
  launch: quantize A, hash and commit it, add noise, then the FP8 matrix product with the lottery inside it.
- `miner/vllm-miner`: a vLLM plugin. It mines while vLLM serves a model, on prompts of 1,024 tokens or more.
- `miner/pearl-gateway`: takes a winning proof, makes the ZK proof, submits the block to a node.
- `node`: `pearld`. No release has the fork (the newest is v1.5.7), so it was built from the branch.

`earn/tuning/salad/pip3box.sh` builds all of it on one box and runs four stages. The same script runs on Salad
and over SSH on Vast.

| Stage | What it does | What a pass means |
|---|---|---|
| gemm | Pearl's kernel tests, `miner/pearl-gemm/tests` without the slow ones | The kernels compile and match Pearl's pinned digests on this card |
| raw | `pip3raw.py`: the four steps on big random matrices, no vLLM | The rate the kernels can reach |
| e2e | Pearl's `test_e2e_regtest.py`: vLLM, gateway and `pearld --regtest` | The node's consensus verifier accepted a block this card mined |
| vload | The miner serving a model under a constant load of long prompts (`pip3load.py`) | The rate Pearl's miner credits itself as shipped |

Rates are in T-MAC/s: 1e12 of m·n·k a second, the unit Pearl's miner credits and the unit our TH/s already is.

### Results

| Card | Host | Driver | Kernel tests | Regtest block accepted | Raw rate | Power at that rate | Miner as shipped |
|---|---|---|---|---|---|---|---|
| RTX 5080 (16 GB) | Vast, Linux | 615.71.09 | 537 passed, 0 failed | Yes, as B200 | 158 to 163 | 360 W, its limit | 17.5 (Qwen3-1.7B), 5.6 (Qwen3-0.6B) |
| RTX 5090 (32 GB) | Salad, Windows with WSL2 | 621.24 and 621.14 | 537 passed, 0 failed | No: vLLM will not start | 242 to 243 | 584 to 593 W of 600 | Does not start |
| RTX PRO 6000 Max-Q (96 GB) | Vast, Linux | 595.99.02 | 537 passed, 0 failed | Yes, as B200 | 186 to 195 | 275 W, its limit | 54.9 (Qwen3-1.7B) |
| H100 SXM (80 GB) | Vast, Linux | 595.91.07 | 494 passed, 1 failed | Yes, as H100 | 413 to 434 | 675 to 690 W of 700 | 73.9 (Qwen3-1.7B) |

- Every box: CUDA 13.0 runtime from the `torch 2.13.0+cu130` wheel, vLLM 0.28.0, Python 3.12.
- Raw rate is the range of the two large shapes, 8192×8192×8192 and 8192×16384×4096, with a target no tile can
  meet. The whole pipeline is timed, not the product alone.
- All RTX 50 and PRO 6000 runs used the 4×128 tile. The H100 used 4×64, the Hopper kernel's own choice.
- The H100's one failure is `test_hit_signal.py::test_partial_n_publishes_in_bounds_tile`: "SM90 tile_n must be
  one of (128, 192)". The test asks for a tile the Hopper kernel does not have. It is not an arithmetic fault.
- The 80 tests skipped on sm_120 are for kernels it does not have: 15 for the B200-only grouped kernel, 2 for the
  16-row tile, and shapes over its shared-memory budget.

### Testnet2

The RTX 5080 mined testnet2 block 150,688 at 12:49:47 UTC on 20 November, 405 blocks after the fork:

- Hash `9a40173418fdbeec1923b8baec53215455ac9a32dd1081c05df65288574738cd`.
- Our node: "Accepted block ... via submitblock". Block 150,689, from another miner, names it as its parent. It
  had 6 confirmations five minutes later.
- It was the one block of a 40 minute solo run at 18.0 T-MAC/s.
- The winning work: Qwen3-1.7B, layer 17's `gate_up_proj`, m 2048, n 12288, k 2048, tile (392, 7), device B200.

How to mine testnet2. No pool was found, so this is solo through the gateway:

1. Build `pearld` from `fp8-scheme` and run `pearld --testnet2 --notls --rpcuser=user --rpcpass=pass
   --rpclisten=127.0.0.1:44111`. It finds peers from the DNS seeds (`reef`, `tide` and
   `stream.seeder.testnet2.pearlnetwork.net`, port 44112). A full sync took 17 minutes.
2. Run the gateway with `PEARLD_RPC_URL=http://127.0.0.1:44111`, the same user and password, and
   `PEARLD_MINING_ADDRESS` set to a taproot address with the `tprl` prefix.
3. Run `vllm serve <model> --quantization pearl` with `.venv/bin` on `PATH`, and send it long prompts.

The chip family is not a setting. The miner reads the card's compute capability and commits H100 for 9.0 and
B200 for 10.0 and 12.0.

Testnet2 at 12:55 UTC: height 150,693, difficulty 106.3, and `getmininginfo` gave a network rate of 9.3e13. If
that is in the miner's unit, all of testnet2 after the fork is about 93 T-MAC/s, less than one RTX 5080. Two of
our six peers still ran pearld 1.4.8 and sent certificate v3 blocks. The new node drops them with "certificate
version 3 is not allowed at height 150283 (require version 4)".

The proof the gateway received for that block is a whole opened tile of real RTX 5080 arithmetic that consensus
accepted. Unpacked, it is `earn/test/fixtures/fp8Rtx5080Testnet2.json`, and `earn/test/minerFp8.test.js` checks
`fp8.js` against it: the reference gets the ticket `dff32409…00000000`, under the network target. The test does
not replay the Merkle openings.

### What failed, with the exact errors

| Where | Error | Cause |
|---|---|---|
| Any box without `.venv/bin` on `PATH` | `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'` | FlashInfer builds its sampler when vLLM warms up. It needs `ninja` and `nvcc`. Not a Pearl fault. |
| Salad RTX 5090 (Windows host, WSL2) | `RuntimeError: UVA is not available` | vLLM 0.28's GPU worker needs UVA, host memory the GPU can address directly, and finds none under WSL2. So Pearl's miner cannot run on a Windows host, natively or in WSL2. The kernels are fine there: 537 tests passed and the raw rate was measured. |
| RTX 5080, Qwen3-4B | `CUDA out of memory. Tried to allocate 742.00 MiB` | The plugin holds the BF16 weights, the FP10 planes and an FP8 copy. 8 GB of weights loaded as 14.24 GiB. Qwen3-1.7B loaded as 5.1 GiB. |
| H100, kernel tests | `ValueError: SM90 tile_n must be one of (128, 192)` | One test, see above. |
| H100, first raw run | `AssertionError: committed tile must match the gemm's lottery tile` | Our benchmark committed 4×128. Fixed: it now asks the kernel. |

Nothing failed in the arithmetic. No block or proof was rejected on any card.

### Why Pearl's miner is slow, and what the kernels can do

On the RTX 5080 the miner as shipped credited 17.5 T-MAC/s at 138 W and 28% GPU use. The same kernels fed large
matrices reached 158 to 163 at 360 W.

- The miner only mines inside a served model's forward pass, so its rate is vLLM's prefill rate times the size
  of the model's layers. On the 5080 box vLLM prefilled about 12,700 tokens a second with either model, with
  the card mostly idle.
- A mined forward pass runs outside vLLM's compiled graphs, and it launches four kernels for each of the
  model's layers (112 mined layers in Qwen3-1.7B).
- Work per launch is m·n·k, so small models earn little. Qwen3-0.6B gave 5.6, Qwen3-1.7B gave 17.5.
- A 16 GB card cannot hold a model big enough to fill it: Qwen3-4B ran out of memory.
- The other boxes show the same gap: 54.9 of 186 on the RTX PRO 6000 (33,500 tokens a second, 227 W) and 73.9
  of 413 on the H100 (45,000 tokens a second, 335 W).

Launch size sets the raw rate. On the RTX 5080:

| Shape m×n×k | Whole pipeline, T-MAC/s | Product alone | Power | GPU use |
|---|---|---|---|---|
| 8192×16384×4096 | 162.9 | 177.6 | 360 W | 98% |
| 8192×8192×8192 | 158.0 | 187.9 | 360 W | 99% |
| 8192×4096×4096 | 130.9 | 181.0 | 360 W | 95% |
| 8192×6144×1024 | 70.5 | 144.9 | 241 W | 61% |
| 2048×4096×4096 | 48.4 | 129.6 | 169 W | 40% |

Two more things the runs showed:

- A target that every launch meets nearly halves the rate (90 against 158 at 8192³). Pearl's tests use such a
  target. Benchmarks must not.
- For scale: our INT7 core does 230 to 238 TH/s on an RTX 5080 at 360 W today (benchmark.md). Pearl's FP8
  kernels do 158 to 163 of the new unit on the same card. The two units pay differently, so this compares
  kernels, not income.

### The ZK proof

A solo miner must turn each winning tile into a ZK proof before the node takes the block.

| Box | Threads | Time from proof received to block submitted |
|---|---|---|
| RTX 5080 host, regtest | 112 | 43 s |
| RTX 5080 host, testnet2 | 112 | 53 s |
| H100 host, regtest | 224 | 46 s |
| RTX PRO 6000 host, regtest | 96, shared with other renters | 57 s |

The largest process during those runs held 7.5 GB (H100 box) and 8.5 GB (PRO 6000 box). Block time is 194 s, so a prover this slow on server CPUs
will lose blocks on a home PC. A pool has to take the plain proof and do this step itself.

### What our miner needs

1. **An FP8 core for sm_120.** The four steps as one CUDA core: FP10 quantize, keyed BLAKE3 commit, noise, and
   the FP8 product with the lottery in it. Declare B200. Pearl's sm_120 kernel is the model, and its bits are
   proven on GeForce: a 5080's block is on testnet2.
2. **Big launches.** Feed the core its own large random matrices, as we do now. A miner bound to a served
   model gets a ninth of the card.
3. **No WSL2, no vLLM.** Our users are on Windows. Pearl's miner cannot start there ("UVA is not available"). A
   native CUDA core like today's avoids the problem. CUDA 13 and driver 580 or newer are already what our RTX 50 core needs.
4. **A pool protocol for plain proofs.** Shares are opened tiles of about (h + w)·k·1.25 bytes. The pool must
   verify them and make the ZK proof for blocks. No pool has published one.
5. **A test gate.** Before release, mine a testnet2 block with our core through Pearl's gateway and node, as
   done here with theirs. `pip3box.sh` builds the node and gateway in about 4 minutes.
6. **A rate target.** 160 T-MAC/s on an RTX 5080 and 242 on an RTX 5090 is what Pearl's own kernels reach.
   Other miners will start from there.
7. **The app.** RTX 20, 30 and 40, AMD and Intel cards stop earning Pearl at the fork block. The spec section
   covers what they see.

### Boxes and cost

About $3.40 of the $8 budget: Vast $3.16, Salad about $0.20.

| Box | Card | Time | Cost |
|---|---|---|---|
| Vast 61636043 | RTX 5080 | 84 min | $0.42 |
| Salad g20-pip3-5090-1 and -2 (llmjob2) | RTX 5090 | 42 min | about $0.20 |
| Vast 61643766 | RTX PRO 6000 Max-Q | 38 min | $0.81 |
| Vast 61644430 | H100 SXM | 22 min | $0.90 |
| Vast 61639767, 61641313, 61642702 | two RTX 5090, one RTX PRO 6000 | 32 min | $0.39, no result: two slow downloads, one dead SSH |

Each box's cost is its time at its hourly price. The Vast account shows $3.16 spent, so about $0.64 more went on
bandwidth and storage. Every box is destroyed and every group deleted.

## Work plan

Estimates are agent working hours. Waiting on boxes, CI and pools is listed apart.

| # | Step | Agent hours | Waiting |
|---|---|---|---|
| 1 | Done 20 November: `fp8.js` checked against a whole tile from a real RTX 50, the testnet2 block's proof | 0 | none |
| 2 | FP8 host side in JS: FP10 operand generator that passes the policy, two-plane commitment, `PlainProofV4` encoder, checked by Pearl's `verify_plain_proof` | 3 | a box with `pearl_mining` built |
| 3 | sm_120 core, first bit-exact version: the four stages above, `verify-hits` at 100% | 8 to 12 | RTX 50 box, 2 to 4 h |
| 4 | Tune the sm_120 core (tile shape, m/n/k, block-scaled instruction, memory clock) against Pearl's miner | 6 to 10 | RTX 5060 to 5090 boxes, half a day |
| 5 | Solo mining on testnet2 through `pearl-gateway`, to prove blocks are accepted | 2 | a synced testnet2 node |
| 6 | Stratum v4 once a pool publishes it | 2 per pool | the pools |
| 7 | App: card rule, pre-fork notice, stop at `cert_version 4`, per-card coin with Quantus for the rest | 5 to 7 | CI |
| 8 | Benchmarks, docs, website table, release | 2 | CI and the release test |
| 9 | Optional: Hopper policy in `fp8.js` (1 h), sm_90 core (6 to 8 h), sm_100 core (6 to 8 h) | 13 to 17 | rented H100 and B200 |

Steps 1 to 3 and 7 do not depend on Pearl setting a date and can start now. Step 7 is the one every user without
an RTX 50 feels, and it does not need a GPU.

## Risks

1. **Bit-exact FP32 sums.** This is the big one. A core that is off by one bit in one cell produces shares the
   pool rejects, at full speed, with no error. It is worse than with INT7 because float sums depend on order and
   width, and the truncation is done inside the tensor-core instruction where we cannot see it. The defence is
   the reference, `verify-hits` on every card model, and checking every hit before it is submitted.
2. **RTX 50 may not match B200.** If a consumer card differs in any bit, only Pearl can fix it by adding a device.
   On 20 November an RTX 5080 and an RTX PRO 6000 mined blocks that nodes accepted, and an RTX 5090 passed
   Pearl's kernel tests (see "Hardware runs"). Other RTX 50 models are untested.
3. **The spec is still moving.** The text is a draft on a branch, the constants changed between July and
   September, and there are no official vectors. Every Pearl commit to `fp8-scheme` needs a re-read of
   `zk-pow/src/v4/api` and a rerun of our tests. Pin the commit we built against.
4. **No pool protocol yet.** We can have a perfect core and nowhere to submit to. Testnet2 solo is the fallback
   for testing, not for earning.
5. **Our speed tricks are gone.** The rotation fold, the k-order search and the per-card INT8 folds do not
   apply. The instruction fixes the order. Our lead over other miners has to be found again, in the hashing,
   the noising stage, memory traffic and clocks.
6. **Operand data must pass the policy.** A constant fill fails. A generator that is cheap on the GPU and still
   passes all four checks on every tile has to be shown with the reference before any tuning.
7. **Most of our users stop.** RTX 20/30/40, AMD and Intel are most of the fleet and of the installed base. If
   the per-card switch to Quantus is not shipped before the fork, those cards mine nothing and say nothing.
8. **Math library differences.** The policy's Gaussian table uses `exp` and `2^x` in double precision. Pearl's
   test shows no value is within 0.001 of a rounding edge, so JavaScript and Rust agree. Our tests pin the table.
9. **Share size.** Openings are tens to hundreds of kilobytes a share, more with wide tiles. 16×16 tiles keep it
   near today's size. Pools may set their own limits.

## Things not to do

- Do not hardcode a fork height. Follow `cert_version` from the job.
- Do not ship an FP8 core that has not passed `verify-hits` against `fp8.js` on that exact card model.
- Do not reuse INT7 tricks that reorder or split the sum over k.
- Do not treat `fp8.js` as final. Re-check it against Pearl's code whenever `fp8-scheme` moves.
