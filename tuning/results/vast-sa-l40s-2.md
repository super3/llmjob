# vast-sa-l40s-2: L40S x4, South Africa (Vast machine 149491, spot)

Same host as `vast-sa-l40s.md` (the L40S x3 there). Rented again at 10:50 UTC on Oct 8 as
instance 54829455 at $0.307/GPU, on box.sh c835a90 (a settings change stops the old miner;
no reboot needed).

Ada, compute 8.9, driver 595.71.05, 350 W power limit per card, max SM clock 2520 MHz.
All four cards sit at the cap (W=349.3-350.3/350) at 2070-2100 MHz, 68-73 C, util 100,
memory 9001 MHz: power-limited, as on the x3. On Ada the release loads the CUDA 12.8 core.

## Baseline (release default, cu12 core)

10:54:36-11:04:39 UTC, 11 readings each (the miners started 10:51:36; the first 2 minutes
are skipped). The control GPUs for the tests are g0 and g3; the ratio is each GPU's TH/s
over the mean of g0 and g3 in the same minute.

| GPU | TH/s (mean) | sd | Ratio to g0/g3 | sm MHz, W, C |
|---|---|---|---|---|
| g0 | 289.85 | 0.45 | 0.9973 | 2070-2085, 349.8/350 W, 73 C |
| g1 | 288.22 | 0.44 | 0.9917 | 2070, 349.3-349.8/350 W, 72 C |
| g2 | 288.68 | 0.40 | 0.9933 | 2070-2085, 349.4-349.6/350 W, 70 C |
| g3 | 291.41 | 1.58 | 1.0027 | 2100, 349.9-350.3/350 W, 68 C |

Box total 1158.2 TH/s. Same range as the x3 on this host (286-294).

## Tests

g0 and g3 stay on the release default as the control. "vs base ratio" is the test GPU's
ratio to the g0/g3 mean in the test window over the same ratio in the baseline.

| Time (UTC) | GPU | Change | Test TH/s | g0/g3 mean, same minutes | vs base ratio | Shares (good/invalid) | Verdict |
|---|---|---|---|---|---|---|---|
