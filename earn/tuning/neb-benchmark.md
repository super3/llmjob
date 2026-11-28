# NEB (Neural Engine Byte) benchmarks

NEB mining is experimental and in no release. This file records what the NEB cores (`earn/native/neb`)
measured against the other miners. All rates are displayed rates: attempts evaluated a second for ours
(`earn/native/neb/README.md` says what that means), each miner's own figure for the others.

A NEB rate depends on the block's model. Blocks alternate between two datasets, Kepler and TESS, and each
model has its own share of attempts with few live rows, which is what a miner can evaluate. Compare rates
only on the same model.

## Does it pay (28 Nov 2026, about 04:30 UTC)

| | |
|---|---|
| Price | $0.484 last on SafeTrade NEB/USDT; 24 h range $0.423 to $0.60; 24 h volume about 54,800 NEB ($27.8k) |
| Network | about 9e12 H/s by rabidmining's node, 55 s a block (target 60 s), difficulty 4.9e14 |
| Reward | 20 NEB a block, 19 to the miner: about 27,400 NEB a day, about $13k |
| RTX 4090 | about 5 to 6 NEB a day at the rates below: $2.3 to $2.8 a day before power, at 430 W |

The estimate is rough: the price is thin, the network grew through the day, and Kepler models moved from
4.2% to 2.7% of attempts evaluated within three hours, which moves every miner's Kepler rate by a third.

## RTX 4090 against SRBMiner-MULTI 3.9.1

Salad PC 30ca683d (llmjob2 `t9-n1-4090-1`), driver 621.14, 450 W limit, stock clocks. Both miners on
neb.rabidmining.com:3470 with the same address, in turns of 300 s on the same PC. Fees: ours 0%, SRBMiner 1.5%.

| Race | Model (attempts with 2 or fewer live rows) | Ours | SRBMiner 3.9.1 | % of SRBMiner | Ours: power | SRBMiner: power |
|---|---|---:|---:|---:|---:|---:|
| 1 (d0ce09d0) | Kepler, 4.22% | 796 MH/s | 1,066 MH/s | 75% | 426 W | 450 W |
| 1 | Kepler, 3.75% | 715 MH/s | 952 MH/s | 75% | | |
| 1 | TESS, 25.2% | 4.00 GH/s | 7.34 GH/s | 55% | | |
| 2 (2c6c29e7) | Kepler 6c822852, 2.66% | 560 MH/s | 684 MH/s | 82% | 428 W | 450 W |
| not raced (40c2ae21) | Kepler d9dbf6c8, 3.15% | 688 MH/s | | | | |
| not raced (40c2ae21) | TESS, 30% to 32%, triple table up | 7.1 to 7.6 GH/s | | | | |

The last two rows are the final build alone on the pool for 240 s (45 accepted, none rejected): it screens
21.75 G nonces/s, which on race 2's model would be about 580 MH/s, 85% of SRBMiner. Its TESS rate is with
a triple table of every live row (16 to 18 GB, built in 3.8 to 4.4 s at the start of each TESS block).

Shares: ours 88 accepted in race 1 and 72 in race 2, none rejected, no hit refused by the CPU recheck.
SRBMiner 108 and 48, none rejected. The pool sets each session's difficulty, so the counts do not compare.
Race 2 had one TESS block of 13 s in our turn, too short to compare.

### Later on 28 Nov: the lean scan, a Kepler table and a faster build

Two other Salad PCs, both miners on rabidmining with the same address, in turns on the same PC. From the third
race on, `tools/nebwatch.js` ran beside SRBMiner and named the pool's model, so each of its readings is set
against ours on the same model. SRBMiner's figure is its API's; ours the miner's status line.

| PC, build | Kepler model (2 or fewer live) | Ours | SRBMiner 3.9.1 | % of SRBMiner | Notes |
|---|---|---:|---:|---:|---|
| 72ccff01 (400 W limit), e0e528ee | 73cfcbb5, 2.21% | 531 MH/s | 580 MH/s | 91.5% | Ours: lean scan, no Kepler table. SRBMiner with no table in memory; with one (19.6 GB) it showed 614 |
| 4da7d0eb (450 W, 87-90 C), 5abdeef3 | 0ac58523, 2.71% | 626 MH/s | 677 MH/s | 92.5% | Ours: no Kepler table |
| 4da7d0eb, f21b3f89 | 2ca5dd88, 2.83% | 693 MH/s | 702 MH/s | 98.7% | Ours with a 1,629-row table; 655 MH/s (93.3%) without. SRBMiner without |
| 4da7d0eb, 57505034 | 01bc7bb6, 2.46% | 631 MH/s | 666 MH/s | 94.7% | Ours with a 1,607-row table. SRBMiner without |
| 4da7d0eb, 57505034 | 2923e605, 2.55% | 643 to 654 MH/s | 666 MH/s | 96.5% to 98.1% | The same, two legs each |

Shares in these races: ours 17 to 45 accepted a leg, SRBMiner 23 to 45, none rejected, no hit refused by the
CPU recheck. The card under either miner: 2,385 to 2,475 MHz, 420 to 450 W, 100% busy.

What changed in ours between n1's races and these (`earn/native/neb/README.md` says how each works):

- The lean scan (`nbk_lean`) for models with few candidates: 23.2 G nonces/s against 20.7 on the recorded
  Kepler model on one PC, 24.0 to 24.8 on the pool's models.
- A triple table on Kepler models too. A card has one buffer for it and the datasets take turns; the screen
  tells from a sample's byte whether the table covers its row. Worth 7% on these models: 1,600 of 5,100 rows
  cover 3% of the attempts with three live rows, which are 2.3 times the attempts with two or fewer.
- The table build: 1.45 s for a Kepler model's 21 GB and 2.4 s for a TESS model's on PC 4da7d0eb. n1's build
  took 4.2 s and 5.5 s for tables of that size on PC 72ccff01.

What SRBMiner's numbers say, from the outside:

- It builds a triple table on Kepler models as well, when the card's memory is free: its rate on one model went
  from 580 to 614 MH/s as its memory went from 1.6 to 19.6 GB.
- Without a table its figure, divided by the model's share of attempts with two or fewer live rows, is 24.7 to
  27.1 G nonces/s on five models; ours screens 23.9 to 24.8 at the same clock and power. A plain screen would
  give the same figure on every model. So it either screens up to 12% more nonces a clock than ours on some
  models or evaluates attempts ours skips. One candidate: rows whose gradient has few entries (3% of a Kepler
  model's live rows have under 1,024 of 26,000), which could be added to a pair entry by entry.
- On TESS it showed 8.1 GH/s on a block where ours, with a table of 644 of 1,474 rows, showed 5.3. That table
  was small through a fault of ours, fixed since (the build's own tables found no room beside the buffer). No
  clean TESS pair has been raced since: TESS blocks last 5 to 30 s, shorter than a leg's start.

## RX 7900 XTX against suprminer 1.9.63

Salad PC 91701547 (llmjob2 `t9-n1-7900xtx-1`), ROCm 7.2 through WSL. Ours is the HIP core; suprminer is
its AMD (OpenCL) build, `-a neb -o stratum+tcp://neb.rabidmining.com:3470`. Same address, 300 s each.
Both 0% fee. No clock or power reading on these boxes.

| Race (2c6c29e7) | Model | Ours | suprminer 1.9.63 | % of suprminer |
|---|---|---:|---:|---:|
| 1 | Kepler 6c822852, 2.66% | 259 MH/s | 304 MH/s | 85% |

Shares: ours 33 accepted in 300 s, suprminer 26 in 210 s, none rejected. No TESS figure from this race.

### Later on 28 Nov

| PC, build | Kepler model | Ours | suprminer 1.9.63 | % of suprminer | Notes |
|---|---|---:|---:|---:|---|
| 116e18f0, d5425531 | 73cfcbb5, 2.21% | 255 MH/s | 278 MH/s | 91.7% | Ours: lean scan, no Kepler table |
| 2daa8b86, f21b3f89 | 2ca5dd88, 2.83% | 349 MH/s | 310 MH/s | 112.5% | Ours with a 1,644-row table; 326 MH/s (104.6%) without |
| 2daa8b86, 93cde7de | 962305a9, 2.48% | 300 MH/s | 271 MH/s | 110.5% | With a 1,629-row table |
| 2daa8b86, 57505034 | 01bc7bb6, 2.46% | 297 MH/s | 269 MH/s | 110% | With a 1,624-row table; the model named by nebwatch.js |
| 2daa8b86, 57505034 | 2923e605, 2.55% | 311 MH/s | 278 MH/s | 112% | The same |

The card of PC 2daa8b86 is the slower of the two: suprminer's figure by the model's share is 11.0 G nonces/s
there and 12.6 on PC 116e18f0, ours 11.4 and 11.6. The final build has not run on PC 116e18f0. suprminer
showed no sign of a triple table on Kepler models. Shares: ours 15 to 39 accepted a leg, none rejected.

On AMD the triple table takes longer: 1.9 s for a Kepler model's, 5.6 to 5.8 s for a TESS model's, and a card's
first 22 GB table waits 12 s more for its buffer. TESS blocks of 5 to 20 s often ended before the table was up.

## Bench on recorded models

`node earn/native/neb/tools/nebbench.js`: no pool, two models rabidmining sent on 28 Nov 2026
(`vectors/model-kepler-14038.hex`, 4.14% of attempts with 2 or fewer live rows; `model-tess-14037.hex`,
25.15%). "+3" is with a triple table over every row the card's memory holds.

| Card | Commit | Kepler | TESS | TESS +3 | Triple table, TESS | Pair table |
|---|---|---:|---:|---:|---|---|
| RTX 4090 | d0ce09d0 (first core, on the pool) | 796 MH/s at 4.22% | 4.00 GH/s | | | 0.86 s Kepler |
| RTX 4090 | 40c2ae21 | 883 MH/s (21.3 G nonces/s) | 4.48 GH/s | 7.03 GH/s | 21.9 GB in 5.4 s | 0.09 s Kepler, 0.014 s TESS |
| RX 7900 XTX | 2c6c29e7 | 393 MH/s (9.5 G nonces/s) | 2.10 GH/s | 2.85 GH/s | 20.4 GB in 6.7 s | 0.14 s Kepler, 0.03 s TESS |
| RTX 4090, PC 72ccff01 | 195e4e4f (n1's last) | 859 MH/s (20.7 G nonces/s) | 4.37 GH/s | 6.77 GH/s | 21.9 GB in 5.5 s | |
| RTX 4090, PC 4da7d0eb | ff75176c | 940 to 963 MH/s (22.7 to 23.3 G nonces/s) | 4.43 GH/s | 6.98 GH/s | 21.5 GB in 2.46 s | |
| RX 7900 XTX, PC 2daa8b86 | 93cde7de | 448 MH/s (10.8 G nonces/s) | 1.95 GH/s | 2.79 GH/s | 21.3 GB in 5.7 s | |

With a triple table over 1,628 of the recorded Kepler model's 4,758 live rows the RTX 4090 evaluates 1,011 to
1,027 MH/s (4.48% of 22.6 to 22.9 G nonces/s) and builds it in 1.6 s; n1's build made 861 MH/s with it and took
4.2 s (PC 72ccff01). The RX 7900 XTX: 481 MH/s, built in 2.1 s.

What moved the rates, in order (RTX 4090 unless said):

| Change | Before | After |
|---|---|---|
| Digest lanes in registers, not an indexed array, in the table builds | pair table 0.86 s | 0.17 s |
| Candidates listed per block, not through one atomic counter for the card | TESS 13.4 G nonces/s | 14.7 |
| Screen and ticket in one kernel, candidates in shared memory | TESS 14.7 | 17.3 |
| Table builds read each row's inputs once, not once an output | TESS triple table 11.7 s | 4.9 s |
| A list a thread on models with few candidates | Kepler 20.0 | 20.8 |
| BLAKE3's adds as multiply-adds | Kepler 20.0 | 20.5 |
| RX 7900 XTX: two nonces a thread between ticket steps | Kepler 8.7 | 9.5 |
| One candidate list a block, rows and tickets shared out after the screen (`nbk_lean`) | Kepler 20.7 | 23.2 |
| The same on the RX 7900 XTX | Kepler 10.1 | 11.8 |
| A live sample's byte says whether the triple table covers its row | Kepler with a table 18.3 G nonces/s, 820 MH/s | 22.9, 1,027 MH/s |
| Triple build: a slab in tiles of 32 rows by 32, a tile a block | TESS table 5.5 s | 4.4 s |
| A tile's rows in shared memory, 88 entries at a time | 4.4 s | 4.0 s |
| Eight slabs a launch; errors read once and xor tags in the factored layer's loop; the clip only where three rows can pass it | 4.0 s | 2.46 s |

Tried and not taken on 28 Nov, later (the ledger has each one's numbers): two nonces a thread with their hashes
interleaved (20.7 against 20.5); 48 warps an SM by capping registers at 40 (19.9, spills); CUDA 13.3's compiler
(the same sm_89 instructions as 12.8); 64-thread blocks; a list a block for models with many candidates too (TESS
16.9 against 17.4, and 5.2 against 6.4 on AMD); the sample bytes in shared memory for the round scan (16.9
against 17.8); a table a slab of the sums of rows b and c (a TESS table in 8.4 s against 5.5: the card's memory
is slower than its adders); two entries a multiply in the factored layer (the compiler already uses a 4-way dot
product there). On AMD the shared-memory copy of a tile's rows crashed the runtime and a stack array in the build
took a TESS table from 5.4 s to 15.2 s; both are NVIDIA-only.

Tried and not taken by n1: byte-permute rotates on CUDA (no change); ChaCha8's adds as multiply-adds (no
change); 128 or 512 threads a block on CUDA (4% and 7% slower); a bitmap in place of the byte table (the
same); evaluating 3-live attempts one by one (rows are dense: 83% of the direct entries and 10 to 17 of
the 32 outputs of the large layer are non-zero in a live row, so one attempt costs about 100 screens);
carrying the triple table from one model to the next (36% to 72% of rows change every model although
only 2 to 7 weights do).

## Where the rest of the gap is

- **RTX 4090, Kepler.** 95% to 98% of SRBMiner when it has no Kepler table, about 91% to 93% when it has one
  (its table is worth 5.7% to it, ours 7% to us). The screen runs 700 of its 1,130 instructions a nonce on
  the card's adder, which takes two a clock an SM: 24 G nonces/s is 88% of that bound at 2.4 GHz. SRBMiner's
  figure says it does more with the same clock and power. Not tried: hand-written SASS; evaluating attempts
  whose third live row has few gradient entries.
- **TESS on both cards.** Not raced cleanly. Half of a miner's shares come from TESS blocks (they are ten times
  shorter and the rate is ten times higher), and a block of 5 to 30 s spends 2.4 s (RTX 4090) or 5.7 s (RX 7900
  XTX) on the table first.
- **RX 7900 XTX.** 110% to 113% of suprminer on one PC with the Kepler table; the faster card of the two PCs
  has only run the older build (91.7%).
- **Intel Arc.** No NEB miner lists Arc. The device code needs nothing Arc lacks; there is no core yet.
