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
