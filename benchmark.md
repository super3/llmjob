# RTX 20-series benchmark plan

These are the Vast hosts we'd use to benchmark the 20-series cards again. The
list is from 2026-10-06. Prices are per hour for an on-demand rental and change
often.

How the hosts were picked:

- The driver supports CUDA 12.8, which our miner needs.
- Reliability is 90% or higher.
- Cheapest first, but a host we've already tested wins when one is listed.
- At most two hosts per card.

Each host runs our miner, PeakMiner 2.17.6 and SRBMiner 3.7.1 one after
another on the same pool, so rates within a row compare directly. Don't compare
across rows: power limits and cooling differ from host to host.

| Card | Host | Machine | $/hr | Power limit | Hit check | Ours | PeakMiner | SRBMiner |
|---|---|---|---|---|---|---|---|---|
| RTX 2080 Ti | Thailand | 95392 | $0.101 | 170 W | ✅ 400/400 | 60.1 | | |
| RTX 2080 Ti | Pennsylvania | 150735 | $0.132 | 260 W | ✅ 400/400 | 83.3 | | |
| RTX 2080 | Colorado | 149439 | $0.122 | 275 W | | | | |
| RTX 2070 Super | Alberta | 31798 | $0.114 | 215 W | ✅ 343/343 | 50.5 | 56.5 | 56.0 |
| RTX 2070 | South Korea | 139007 | $0.136 | 150 W | ✅ 293/293 | 43.2 | 50.7 | 48.8 |
| RTX 2060 Super | Germany | 149900 | $0.059 | 175 W | | | | |
| RTX 2060 | Australia (6 GB) | 152547 | $0.065 | 190 W | ✅ 296/296 | | | |
| RTX 2060 | South Korea (12 GB) | 27568 | $0.109 | 184 W | ✅ 317/317 | | | 53.6 |

Rates are in TH/s. A blank cell means we have no number for that host yet.

## Where the existing numbers come from

All of them come from the kernel that shipped in v0.5.10, and every run had 0
rejected shares.

- **2080 Ti rows:** 15-minute pool runs of our miner only.
- **Other cards:** 5-minute runs of each miner.
- **Hit check:** our miner's hits verified against the reference hash.

The 2060 rows are incomplete because those tests were stopped part way through.

## Notes on availability

- **2080 Ti with all three numbers:** that host (US, machine 35928, 250 W) is
  no longer listed. There we measured 78.0 (ours), 88.5 (PeakMiner) and 86.1
  (SRBMiner). Other 2080 Ti hosts listed now:
  - Thailand 55752: $0.101, 180 W.
  - Maryland 59368: $0.135, 250 W.
- **RTX 2080:** this is the first time one has been listed. It hasn't been
  tested.
- **2070 Super:** only one host qualifies. A Washington host (134024) is
  listed, but its driver only supports CUDA 12.2.
- **2070:** this host is only offered as a 2-GPU rental. The price above is for
  both GPUs; we'd use one.
- **2060 Super:** only one host is listed, and neither host we tested before is
  available.
- **2060:** machine 27568 costs $0.002/hr more than South Korea 27564, which we
  haven't tested. We chose 27568 so a new SRBMiner run can be checked against
  the 53.6 we already have.
