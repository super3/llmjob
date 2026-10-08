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

