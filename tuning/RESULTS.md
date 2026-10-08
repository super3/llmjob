# Tuning results

Each row is one test: the change, the card, and the hashrate of the test GPUs against the
control GPUs on the same box over the same minutes. "Verified" means it held for at least 10
minutes after warm-up with no rejected or invalid shares, and was rolled out to that box.

## Baselines (v0.5.12 release, default settings)

| Card | Box (machine) | TH/s per GPU | Source |
|---|---|---|---|
| L40S | South Africa (149491) | 290 | Oct 7-8 run |
| RTX 4070 Super | US (116195, host 410852) | 110 | Oct 8 |
| RTX 4070 Super | US (145579, host 410852) | 102 | Oct 8 |
| RTX 4090 | North Carolina (150022) | 280 | Oct 8 |
| RTX 4090 | Poland (151409) | 314 | Oct 8 |
| RTX 4090 | Japan (151594) | 319 | Oct 8 |
| RTX 5090 | Alberta (35115) | 396 | Oct 8 |
| RTX 5090 (400 W cap) | Vietnam (59404) | 349, 346, 298, 311 (four cards, all at the cap) | Oct 8 |
| RTX 5060 Ti | Finland (149979) | 97 | Oct 8 |

## Tests

Each box's tuning agent logs every test it runs in `results/<worker>.md`, its own file so
agents never edit the same lines. Verified gains are copied here.

| Date (UTC) | Card | Box | Change | Test GPUs vs control | Result |
|---|---|---|---|---|---|
| Oct 8 09:51-10:00 | RTX 4070 Super (160 W, 48 MB L2) | US 116195 | Build with `-DPEARL_ADA_COL_BATCH=1024` (1024 column offsets a launch, not 2048) | g2 116.67 TH/s vs g0/g1 release 109.72, same minutes | **+2.5 to +3.1%**, SM clock +60-75 MHz at the same 160 W, 0 invalid. Verified, rolled out to the box 10:02 |
| Oct 8 09:51-10:00 | RTX 4070 Super (160 W, 48 MB L2) | US 116195 | Build with `-DPEARL_TALL_BAND=64` (band of 64 row groups, not 16) | g3 111.26 vs 109.73 | +1.9 to +2.2%, 0 invalid. Smaller than col_batch 1024, and the two don't fit the L2 together, so not rolled out |
| Oct 8 10:05-10:15 | RTX 4070 Super (160 W, 48 MB L2) | US 116195 | Build with `-DPEARL_ADA_COL_BATCH=512` | g0 109.07 vs g1-g3 on col_batch 1024, 114.10 | 1.4% behind 1024 (+0.9% over the release). 1024 is the width for 48 MB. The rollout held: g1-g3 +2.0 / +2.6 / +2.7% over their own release rates |
| Oct 8 10:25-10:35 | RTX 4070 Super (160 W, 48 MB L2) | US 116195 | Build of 29ac5f5 (the Ada L2 rule), no defines | g0 111.56 vs g2/g3 on col_batch 1024 | +3.2% over g0's release rate, in line with cb1024: the rule picks 1024 here. Verified, all four GPUs on it from 10:51; box 453.5 vs 442.3 TH/s (+2.5%) |
| Oct 8 10:25-10:49 | RTX 4070 Super (160 W, 48 MB L2) | US 116195 | On top of the rule: `-DPEARL_OPERAND_FILL=63`, then `-DPEARL_TALL_BAND=32` | g1 vs the rule GPUs | Both level (-0.3%, +0.1%), 0 invalid. Fill 63 is pool-safe but buys nothing on Ada; a deeper band adds nothing once B' fits |

## What we've learned


- **RTX 5060 Ti (Blackwell): keep the CUDA 13 core.** It is what the release picks on this
  card. Forcing the CUDA 12.8 core cost 2.9% on Finland 149979: g1 ran 93.5 TH/s against
  96.3 on cu13, at the same clock (2737 vs 2730 MHz) and power (171 vs 169 W), while the
  control GPU held 98.7 (Oct 8, 20 minutes). So the loss is in work done per clock. It also
  means a plain on-box build (nvcc 12.8) starts about 3% behind the release on Blackwell.
  Build with `"cuda": "13"` instead: a CUDA 13.3 build of operand fill 63 ran level with the
  release on the same GPU, so that path matches it. The driver-JIT build (12.8 PTX compiled
  by the 595 driver) came out about 0.3% behind. Operand fill 63 neither raised the rate nor
  cut the power against 48 on this card.

- **Ada: the CUDA 12 and CUDA 13 cores run the same fold.** Built for sm_89 under nvcc
  12.8.93 and 13.3.73, `pearl_tile_fold_tall` comes out the same: 255 registers, no spill,
  3040 instructions, a 540-instruction chunk loop for 192 IMMA. Only predicate-register
  names differ. The North Carolina 4090 measured cu13 level with the release (+0.6%, inside
  drift). So `core: cu13` is not worth a test window on Ada.
- **Ada: an on-box build with no defines is the release.** The v0.5.12 release core's sm_89
  `pearl_tile_fold_tall` is the same as a local nvcc 12.8.93 build of this branch,
  instruction for instruction, so Ada builds can be judged against the release directly.
  Compiler flags don't move the fold: `--extra-device-vectorization`,
  `-Xptxas --allow-expensive-optimizations=true` and `-Xptxas -O2` leave its SASS unchanged.
  Moving B's copies back to m16 tile 8 (`PEARL_TALL_BPT=8`, Ada's point before v0.5.12) lost
  about 1.5% on the North Carolina 4090, which heat holds near 2205 MHz at 90 C (273.3 TH/s
  between release runs of 277.4 and 277.5). The release's copy points stand.
- **Ada cards with less L2 than a 4090: run a narrower batch.** At col_batch 2048 one launch
  of Ada's tall fold sweeps 64 MB of B' plus a 6 MB band of A'. On a 48 MB L2 (RTX 4070
  Super) B' comes back from DRAM every band, and at the power cap those DRAM watts come out
  of the SM clock. col_batch 1024 (32 MB) keeps it in the L2: +2.5 to +3.1% on 116195, with
  the SM clock 60-75 MHz higher at the same 160 W. Band 64 (fewer bands, so fewer B' re-reads)
  gave +2%. 512 was 1.4% behind 1024: narrower than the L2 needs costs more launches, each
  re-reading all 256 MB of A'. Same mechanism as Blackwell's `PEARL_TMA_L2_SHARE` rule.
  Commit 29ac5f5 makes it a rule in `pearl_host.cu` (sm_89 only): halve col_batch while one
  launch's B' plus a band of A' is more than the L2. 72 MB and up (4090) keep 2048; 48-64 MB
  (4070 Super, 4070 Ti, 4080) get 1024; 24-36 MB (4060, 4060 Ti, 4070) get 512, which is
  not measured yet.
- **Memory clock locks don't work on Vast.** `--mine-mem-clock 5001` left the memory clock
  at the driver's value on the SA L40S (149491), the Japan 4090 (151594) and the Maryland
  5090 (151626): the containers can't set clocks.
- **Blackwell: a core built on the box can run at cu13 speed through the driver's JIT.**
  Build it with `-DPEARL_FORCE_PTX_JIT=1 -gencode arch=compute_120,code=compute_120` (switch
  added in f4f6d63, off by default). The core then sets `CUDA_FORCE_PTX_JIT=1` as it loads,
  and the driver (CUDA 13.x on 580+) compiles the fold instead of nvcc 12.8's ptxas. On the
  Vietnam 5090 box (59404, driver 595.71) that build ran level with the release cu13 core
  (-0.2% vs control, hit check 400/400, 0 invalid), while the release's own cu12 core lost
  3.0% on another card of the same box over the same runs. Finland 149979 (5060 Ti, driver
  595.84) got the same: -0.3% vs cu13. Boxes rented before 09:47 UTC don't have box.sh's
  `"cuda": "13"` build option; this is how they test sm_120 knobs against the release.
- **RTX 3090 (Ampere): keep operand fill 48.** On the Bulgaria 8x 3090 box (49870, 350 W cap,
  driver 570, so the release runs its CUDA 12.8 core), fill 63 ran level (+0.1 to +0.3% against
  control over the same 10 minutes) and fill 32 slightly behind (-0.3 to -0.45%). Same order as
  on the 4090, but smaller. Build `-DPEARL_OPERAND_FILL=N` (df0fb5c) to try other values; the
  pool accepted 63 and 32 with no invalid shares. On sm_86 an on-box build (nvcc 12.8, same
  flags as CI) can be judged straight against the release. These cards already run the fold
  at about 95% of the int8 tensor peak per clock (82 SMs x 1024 MACs a clock), and most of
  them sit at the 350 W cap, so only energy per MAC can raise the rate. Two GPUs on this box
  run well under the cap at a lower clock that falls as the room warms (283-326 W, 1395-1470
  MHz, edge temp only 71-76 C). Something nvidia-smi's `[gpu]` line doesn't show holds them
  back, likely the GDDR6X temperature.
