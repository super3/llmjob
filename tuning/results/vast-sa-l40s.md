# vast-sa-l40s: L40S x3, South Africa (Vast machine 149491, host 670790, spot)

Ada, compute 8.9, driver 595.71.05, 350 W power limit per card, 48 MB L2, 142 SMs,
48 GB GDDR6 at 9001 MHz. v0.5.12 release, one miner per GPU. Max SM clock 2520 MHz.

All three cards sit at their 350 W cap at 2055-2115 MHz, 72-75 C: power-limited, not
heat-limited.

## Baseline (release default = cu12 core on Ada)

| GPU | TH/s (mean) | Window (UTC, Oct 8) | sm / mem MHz, W, C |
|---|---|---|---|
| g0 | 294.2 | 08:47-08:55 | 2115 / 9001, 350/350 W, 72 C |
| g1 | 285.7 | 08:47-08:55 | 2055 / 9001, 350/350 W, 73 C |
| g2 | 285.8 | 08:47-08:55 | 2070 / 9001, 349/350 W, 75 C |

Readings fall by 1-2% over the first ten minutes as the cards warm up.

## Tests

g2 is the test GPU; g0 and g1 are the control. Deltas are g2 against the mean of g0 and
g1 over the same minutes, and against g2's own baseline.

| Time (UTC) | Change on g2 | g2 TH/s | g0 / g1 TH/s (same minutes) | Shares g2 (good/invalid) | Verdict |
|---|---|---|---|---|---|
| 08:57 | `--mine-mem-clock 5001` | INVALID: the old release miner kept running next to the new one (box.sh orphan bug), 124-132 TH/s each | 294 / 285 | 23 / 0 at 09:01 | Invalid as a hashrate test. One thing it did show: the lock does not take on this host (`[gpu g2] mem=9001` at 09:00:14, 3 min after the start). Left in place (a no-op) because reverting would add a third miner on g2 with this box's old box.sh. |

The 08:57 switch also paused the box. g2's log line showed only the orphan's half rate, so
the supervisor read 801.7 TH/s at 09:00 instead of about 866, cut the bid cap from 0.887 to
0.828, and the box was outbid at 09:02. Until a box runs the fixed box.sh (ce1d7c9), a GPU
switch there can do the same.

## Offline checks

- **cu12 vs cu13 on Ada (static).** `pearl_kernel.cu` for sm_89 under nvcc 12.8.93 and
  13.3.73: `pearl_tile_fold_tall` is 255 registers and no spill under both, 3040 SASS
  instructions, chunk loop 540 instructions for 192 IMMA, A `.reuse` on 168 of 192. About
  110 of the 3040 lines differ, almost all predicate-register names. So `core: cu13` should
  measure level with the release default (cu12) on Ada, and is not worth a test window.
- **Ada batch width switch.** `-DPEARL_ADA_COL_BATCH=N` (bfa15c5) runs N column offsets a
  launch on Ada's tall fold instead of 2048. At 2048 a launch sweeps 64 MB of B' plus a 6 MB
  band of A', more than the L40S's 48 MB L2. `cb1024` is queued as a build here; it runs on
  g2 only once this box is on the fixed box.sh.
