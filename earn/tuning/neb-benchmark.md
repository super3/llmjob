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

What SRBMiner's numbers say, from the outside:

- On Kepler its rate moves with the model's share exactly as ours does, so it evaluates the same attempts
  and screens nonces faster: about 25.7 G nonces/s against our 21.
- On TESS its 7.3 GH/s is what evaluating attempts with 3 live rows as well gives. With a full table of
  row triples ours does 7.0 GH/s on a recorded TESS model, but the table takes 4.9 s to build and SRBMiner
  is at its rate within one 15 s sample.

## RX 7900 XTX against suprminer 1.9.63

Salad PC 91701547 (llmjob2 `t9-n1-7900xtx-1`), ROCm 7.2 through WSL. Ours is the HIP core; suprminer is
its AMD (OpenCL) build, `-a neb -o stratum+tcp://neb.rabidmining.com:3470`. Same address, 300 s each.
Both 0% fee. No clock or power reading on these boxes.

| Race (2c6c29e7) | Model | Ours | suprminer 1.9.63 | % of suprminer |
|---|---|---:|---:|---:|
| 1 | Kepler 6c822852, 2.66% | 259 MH/s | 304 MH/s | 85% |

Shares: ours 33 accepted in 300 s, suprminer 26 in 210 s, none rejected. No TESS figure from this race.

## Bench on recorded models

`node earn/native/neb/tools/nebbench.js`: no pool, two models rabidmining sent on 28 Nov 2026
(`vectors/model-kepler-14038.hex`, 4.14% of attempts with 2 or fewer live rows; `model-tess-14037.hex`,
25.15%). "+3" is with a triple table over every row the card's memory holds.

| Card | Commit | Kepler | TESS | TESS +3 | Triple table, TESS | Pair table |
|---|---|---:|---:|---:|---|---|
| RTX 4090 | d0ce09d0 (first core, on the pool) | 796 MH/s at 4.22% | 4.00 GH/s | | | 0.86 s Kepler |
| RTX 4090 | 40c2ae21 | 883 MH/s (21.3 G nonces/s) | 4.48 GH/s | 7.03 GH/s | 21.9 GB in 5.4 s | 0.09 s Kepler, 0.014 s TESS |
| RX 7900 XTX | 2c6c29e7 | 393 MH/s (9.5 G nonces/s) | 2.10 GH/s | 2.85 GH/s | 20.4 GB in 6.7 s | 0.14 s Kepler, 0.03 s TESS |

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

Tried and not taken: byte-permute rotates on CUDA (no change); ChaCha8's adds as multiply-adds (no
change); 128 or 512 threads a block on CUDA (4% and 7% slower); a bitmap in place of the byte table (the
same); evaluating 3-live attempts one by one (rows are dense: 83% of the direct entries and 10 to 17 of
the 32 outputs of the large layer are non-zero in a live row, so one attempt costs about 100 screens);
carrying the triple table from one model to the next (36% to 72% of rows change every model although
only 2 to 7 weights do).

## Where the rest of the gap is

- **Screen speed, both vendors.** SRBMiner screens about 22% more nonces a second on the RTX 4090 and
  suprminer about 17% more on the RX 7900 XTX. Every arrangement of our kernel lands at 20 to 21 G
  nonces/s on the 4090, and the table look-ups are 3% of it: what is left is how the two hashes compile.
  Not tried: the CUDA 13 compiler, hand-placed SASS as the Pearl and Quantus cores have, OpenCL on AMD.
- **TESS triples.** The table is right (checked against the reference on a live model: 957 of 957 hits)
  and worth +57% on a TESS model, but at 4.9 s to build it pays only on blocks longer than about 10 s,
  and only cards with 16 GB or more hold most of it. A pool that sent the next TESS model during the
  Kepler block before it would let any miner build ahead; rabidmining sends it with the job.
- **Intel Arc.** No NEB miner lists Arc. The device code needs nothing Arc lacks; there is no core yet.
