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
| RTX 5060 Ti | Finland (149979) | 97 | Oct 8 |

## Tests

Each box's tuning agent logs every test it runs in `results/<worker>.md`, its own file so
agents never edit the same lines. Verified gains are copied here.

| Date (UTC) | Card | Box | Change | Test GPUs vs control | Result |
|---|---|---|---|---|---|

## What we've learned


- **RTX 5060 Ti (Blackwell): keep the CUDA 13 core.** It is what the release picks on this
  card. Forcing the CUDA 12.8 core cost 2.9% on Finland 149979: g1 ran 93.5 TH/s against
  96.3 on cu13, at the same clock (2737 vs 2730 MHz) and power (171 vs 169 W), while the
  control GPU held 98.7 (Oct 8, 20 minutes). So the loss is in work done per clock. It also
  means a core built on the box, which uses nvcc 12.8, starts about 3% behind the release on
  Blackwell cards.

- **Ada: the CUDA 12 and CUDA 13 cores run the same fold.** Built for sm_89 under nvcc
  12.8.93 and 13.3.73, `pearl_tile_fold_tall` comes out the same: 255 registers, no spill,
  3040 instructions, a 540-instruction chunk loop for 192 IMMA. Only predicate-register
  names differ. The North Carolina 4090 measured cu13 level with the release (+0.6%, inside
  drift). So `core: cu13` is not worth a test window on Ada.
- **Memory clock locks don't work on Vast.** `--mine-mem-clock 5001` left the memory clock
  at the driver's value on the SA L40S (149491), the Japan 4090 (151594) and the Maryland
  5090 (151626): the containers can't set clocks.
