# Benchmarks

Our miner against PeakMiner and SRBMiner, 5 minutes each on the same card and pool, and since 9 November against
OneZeroMiner too, a top-5 Pearl miner by downloads (NVIDIA only). WildRig Multi isn't measured: it mines NVIDIA
cards through OpenCL, which Salad's WSL2 containers don't have. "% of best" is our rate over the fastest competitor's. Vast rows ran v0.5.14 on 10–11 October; Salad rows ran v0.6.9 (October) and v0.6.13 (9–10 November) on home PCs, so compare "% of
best" between them, not TH/s. The old file with every run's notes:
[benchmark.md at 16192d68](https://github.com/super3/llmjob/blob/16192d68f40b9c7ba11048e1128e7ba2c5509dca/benchmark.md).

## Summary

| Cards | Latest Salad run | Note |
|---|---|---|
| RTX 20-series | 97–100%, median 98% (9–10 Nov, v0.6.13) | PeakMiner fastest on every PC |
| RTX 30-series | 85–95%, median 93% (9–10 Nov, v0.6.13) | OneZeroMiner fastest on most PCs |
| RTX 40-series | 87–96%, median 93% (9–10 Nov, v0.6.13) | Our gap widens at lower power limits |
| RTX 50-series | 86–97%, median 94% (9–10 Nov, v0.6.13) | 5090 trails most (86–87%) |
| GTX 16-series | 96–104% (17 Oct, v0.6.0) | 0.7–1.3 TH/s; not worth mining |
| AMD Radeon | 92–143%, median 103% against SRBMiner 3.8.2 (9–10 Nov, v0.6.13) | RDNA 4 111–143%; 7900 XTX 92–94%, the one card behind |

## RTX 20-series

Vast, 10–11 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | Thailand (170 W) | 95392 | 63.4 | 64.1 | 64.2 | 99% |
| RTX 2080 Ti | Pennsylvania (260 W) | 150735 | 89.0 | 93.2 | 92.0 | 96% |
| RTX 2080 | Colorado (275 W) | 149439 | 77.6 | 81.7 | 82.8 | 94% |
| RTX 2070 Super | Alberta (215 W) | 31798 | 60.3 | 62.4 | 64.0 | 94% |
| RTX 2070 | South Korea (150 W) | 139007 | 47.8 | 50.4 | 51.7 | 93% |
| RTX 2060 Super | Germany (175 W) | 149900 | 42.7 | 45.3 | 46.4 | 92% |
| RTX 2060 | Australia (190 W, 6 GB) | 152547 | 47.8 | 49.9 | 48.4 | 96% |
| RTX 2060 | South Korea (184 W, 12 GB) | 27568 | 49.8 | 52.9 | 52.4 | 94% |

The gap is work per clock: our clock is within 30 MHz of theirs, and we do 5–7.5% less per clock.

### Salad, 28 October (PeakMiner 2.20.3, SRBMiner 3.8.0)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | Salad (280 W) | ec9e6c87 | 96.5 | 89.0 | 94.4 | 102% |
| RTX 2080 Ti | Salad (260 W) | ef08a260 | 94.9 | 94.7 | 86.4 | 100% |
| RTX 2080 | Salad (245 W) | 83f7bc7e | 65.6 | 67.6 | 65.5 | 97% |
| RTX 2080 | Salad (129 W) | f765012b | 54.8 | 54.5 | 53.8 | 100% |
| RTX 2070 Super | Salad (215 W) | 2d10f218 | 61.6 | 62.7 | 62.4 | 98% |
| RTX 2070 Super | Salad (215 W) | ffd709c4 | 62.6 | 63.9 | 63.1 | 98% |
| RTX 2070 | Salad (175 W) | 2aaa0e3e | 48.0 | 49.3 | 47.4 | 97% |
| RTX 2070 | Salad (184 W) | 0416482b | 51.1 | 49.5 | 49.3 | 103% |
| RTX 2060 Super | Salad (200 W) | 413b2e3d | 48.1 | 49.5 | 48.2 | 97% |
| RTX 2060 Super | Salad (184 W) | ab1e6b0e | 46.9 | 47.5 | 47.3 | 99% |
| RTX 2060 | Salad (168 W, 6 GB) | 81031a48 | 46.1 | 47.3 | 45.4 | 98% |
| RTX 2060 | Salad (160 W, 6 GB) | a1fd5f37 | 47.4 | 48.9 | 46.7 | 97% |

The rows over 100% are odd PCs: one swung between 207 and 266 W, one is capped at 129 W.

### Salad, 30 October (SRBMiner 3.8.2)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | Salad (260 W) | ef08a260 | 94.8 | 95.0 | 93.6 | 99.7% |
| RTX 2080 Ti | Salad (260 W) | 671bb26b | 78.3 | 79.1 | 75.5 | 99.1% |
| RTX 2080 | Salad (245 W) | 83f7bc7e | 66.9 | 68.8 | 68.0 | 97.4% |
| RTX 2080 | Salad (260 W) | b2de4b63 | 78.2 | 79.0 | 78.0 | 99.0% |
| RTX 2070 Super | Salad (255 W) | 32c563bc | 62.1 | 63.2 | 62.4 | 98.2% |
| RTX 2070 | Salad (185 W) | 46bfd62d | 55.9 | 57.6 | 56.4 | 97.0% |
| RTX 2070 | Salad (225 W) | 1e19bb79 | 61.3 | 63.8 | 62.5 | 96.1% |
| RTX 2060 | Salad (6 GB) | 4af24fcc | 35.7 | 37.3 | 35.1 | 95.7% |
| RTX 2060 | Salad (167 W, 6 GB) | 0338617b | 46.9 | 48.3 | 45.8 | 97.3% |

### Salad, 9–10 November (v0.6.13; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 2080 Ti | Salad (300 W) | 1ff934b7 | 105.4 | 105.5 | 104.2 | 105.5 | 100% |
| RTX 2080 Ti | Salad (260 W) | 7e24f884 | 97.4 | 97.9 | 94.8 | 94.6 | 99% |
| RTX 2080 | Salad (260 W) | a5d2c787 | 79.0 | 80.9 | 76.6 | 76.0 | 98% |
| RTX 2080 | Salad (172 W) | c6b154db | 125.5 | 128.8 | 126.6 | 121.8 | 97% |
| RTX 2070 Super | Salad (215 W) | ef050aaa | 64.7 | 65.4 | 64.8 | 62.2 | 99% |
| RTX 2070 Super | Salad (215 W) | 842bcd6e | 63.9 | 65.3 | 65.2 | 61.5 | 98% |
| RTX 2070 | Salad (140 W) | b6b2b0c9 | 51.1 | 52.5 | 51.4 | 50.3 | 97% |
| RTX 2070 | Salad (175 W) | f4e53154 | 53.5 | 55.1 | 53.5 | 52.0 | 97% |
| RTX 2060 | Salad (160 W, 6 GB) | 67fbcdbf | 40.6 | 42.0 | 40.3 | 39.9 | 97% |
| RTX 2060 | Salad (160 W, 6 GB) | 92c0cd67 | 36.7 | 37.4 | 36.7 | 37.4 | 98% |

PeakMiner is fastest on every PC; OneZeroMiner is the slowest of the three here.

### Monero on the CPU, 11 October (our CPU miner against SRBMiner 3.7.5)

| CPU (CPUs the container got) | GPU | Threads | Ours | SRBMiner 3.7.5 | % of best |
|---|---|---|---|---|---|
| Core i5-11400 (11.5 of 12) | RTX 3060 | 6 | 2,457 H/s | 1,880 H/s | 131% |
| Ryzen 9 5950X (15.4 of 32) | RTX 2060 | 14 | 6,439 H/s | 4,527 H/s | 142% |

Ours used transparent huge pages and SRBMiner got none, which likely explains much of the gap.

## RTX 30-series

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 3090 Ti | Vietnam (450 W) | 21444 | 153.1 | 149.3 | 153.9 | 99% |
| RTX 3090 Ti | Oman (450 W) | 12840 | 150.4 | 150.7 | didn't start | 99.8% |
| RTX 3090 | Ukraine (290 W) | 141130 | 115.8 | 113.3 | 118.4 | 98% |
| RTX 3090 | Quebec (325 W) | 67901 | 125.1 | 123.9 | 123.0 | 101% |
| RTX 3080 Ti | Japan (330 W) | 140510 | 125.9 | 124.9 | 128.1 | 98% |
| RTX 3080 Ti | Portugal (350 W) | 56596 | 128.1 | 127.9 | 127.1 | 100% |
| RTX 3080 | Taiwan (320 W) | 146647 | 110.9 | 110.8 | 109.0 | 100% |
| RTX 3080 | Austria (290 W) | 143869 | 111.1 | 110.4 | 109.9 | 101% |
| RTX 3070 Ti | Quebec (290 W) | 137703 | 85.3 | 85.8 | 86.0 | 99% |
| RTX 3070 | Quebec (220 W) | 56506 | 72.7 | 73.3 | 75.1 | 97% |
| RTX 3070 | Quebec (180 W) | 152549 | 72.1 | 72.9 | 74.6 | 97% |
| RTX 3060 Ti | Colorado (200 W) | 151268 | 59.7 | 60.1 | 60.4 | 99% |
| RTX 3060 Ti | Argentina (180 W) | 150126 | 58.1 | 58.9 | 57.5 | 99% |
| RTX 3060 | New Brunswick (170 W) | 15303 | 50.8 | 51.4 | 51.9 | 98% |
| RTX 3060 | South Korea (170 W) | 137223 | 48.2 | 48.8 | 49.2 | 98% |

SRBMiner exited at once on the Oman host, so that row compares with PeakMiner.

### Salad, 28 October (PeakMiner 2.20.3, SRBMiner 3.8.0)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 3090 Ti | Salad (336 W) | b687d663 | 126.0 | 149.7 | 152.7 | 83% |
| RTX 3090 Ti | Salad (450 W) | ef44d3cd | 148.3 | 160.5 | 161.3 | 92% |
| RTX 3090 | Salad (350 W) | 66666c27 | 113.7 | 121.3 | 120.3 | 94% |
| RTX 3080 Ti | Salad (298 W) | 344adcb0 | 121.5 | 133.2 | 132.8 | 91% |
| RTX 3080 Ti | Salad (350 W) | 4ec4131f | 126.9 | 137.4 | 136.4 | 92% |
| RTX 3080 | Salad (285 W) | 93c805c0 | 105.8 | 114.3 | 113.4 | 93% |
| RTX 3080 | Salad (340 W) | 9fbd405b | 111.8 | 115.0 | 115.0 | 97% |
| RTX 3070 Ti | Salad (310 W) | b7d4d9a2 | 88.3 | 93.6 | 93.1 | 94% |
| RTX 3070 Ti | Salad (310 W) | cdc3c2a9 | 89.9 | 94.8 | 94.6 | 95% |
| RTX 3070 | Salad (270 W) | 46aae7db | 83.4 | 89.6 | 89.4 | 93% |
| RTX 3070 | Salad (216 W) | 22bd1d8a | 77.4 | 83.6 | 83.8 | 92% |
| RTX 3060 Ti | Salad (225 W) | abef70e6 | 65.9 | 71.2 | 70.6 | 93% |
| RTX 3060 Ti | Salad (210 W) | 3a481d90 | 66.4 | 72.5 | 71.8 | 92% |
| RTX 3060 | Salad (170 W, 12 GB) | 0771915f | 48.4 | 50.5 | 50.3 | 96% |
| RTX 3060 | Salad (170 W, 12 GB) | 85c1c681 | 50.1 | 54.3 | 54.3 | 92% |

We do 94.9–96.7% of their work per clock and run a 1.6–4.6% lower clock at the same power.

### Salad, 9–10 November (v0.6.13; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 3090 Ti | Salad (400 W) | 9f31aaad | 136.4 | 142.3 | 143.1 | 144.1 | 95% |
| RTX 3090 Ti | Salad (450 W) | 23132fc1 | 128.4 | 147.6 | 147.2 | 151.9 | 85% |
| RTX 3090 | Salad (315 W) | 36cab5e7 | 112.5 | 118.1 | 118.0 | 117.5 | 95% |
| RTX 3090 | Salad (319 W) | 219edba8 | 113.2 | 121.9 | 119.9 | 121.5 | 93% |
| RTX 3080 Ti | Salad (350 W) | c09c68c3 | 111.2 | 129.6 | 116.0 | 124.2 | 86% |
| RTX 3080 Ti | Salad (350 W) | a4519d9b | 116.0 | 121.3 | 122.0 | 122.1 | 95% |
| RTX 3080 | Salad (309 W) | 121e592d | 113.2 | 121.8 | 122.0 | 122.3 | 93% |
| RTX 3080 | Salad (320 W) | a04f3c19 | 110.4 | 118.3 | 118.6 | 119.0 | 93% |
| RTX 3070 Ti | Salad (290 W) | c203e4e5 | 82.8 | 87.5 | 87.4 | 88.7 | 93% |
| RTX 3070 Ti | Salad (290 W) | c5d9c25f | 85.1 | 90.4 | 91.4 | 91.3 | 93% |
| RTX 3070 | Salad (240 W) | ff74eb4b | 81.4 | 86.5 | 74.8 | 87.9 | 93% |
| RTX 3070 | Salad (240 W) | f0552404 | 84.0 | 88.1 | 81.6 | 88.8 | 95% |
| RTX 3060 Ti | Salad (220 W) | 9107bf3b | 67.2 | 72.0 | 67.6 | 72.4 | 93% |
| RTX 3060 Ti | Salad (200 W) | 8ccdb586 | 65.1 | 71.0 | 71.1 | 71.0 | 92% |
| RTX 3060 | Salad (170 W, 12 GB) | 6dfca07e | 49.9 | 54.1 | exited (73) | 54.3 | 92% |
| RTX 3060 | Salad (170 W, 12 GB) | 669dc77c | 49.9 | 53.7 | 53.5 | 54.0 | 92% |

A first 3080 Ti PC (`b64f0e45`) stalled for 36 minutes during our run, so it was rerun on `c09c68c3`. SRBMiner
3.8.2 exited with code 73 on one 3060. OneZeroMiner is fastest on most PCs.

## RTX 40-series

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 4090 | Australia (450 W) | 17334 | 318.7 | 316.2 | 317.7 | 100% |
| RTX 4090 | Texas (450 W) | 153411 | 314.7 | 310.8 | 314.1 | 100% |
| RTX 4090D | South Carolina (425 W) | 142279 | 276.7 | 268.0 | 277.3 | 99.8% |
| RTX 4080 Super | Florida (320 W) | 142085 | 198.0 | 196.5 | 198.9 | 99.5% |
| RTX 4080 Super | California (275 W) | 138449 | 198.1 | 196.7 | 198.6 | 99.8% |
| RTX 4080 | Japan (320 W) | 6706 | 199.8 | 198.6 | 199.0 | 100% |
| RTX 4080 | British Columbia (225 W) | 43318 | 169.7 | 158.3 | 172.9 | 98% |
| RTX 4070 Ti Super | Texas (285 W) | 142663 | 172.2 | 171.5 | 171.6 | 100% |
| RTX 4070 Ti Super | Quebec (285 W) | 36427 | 167.8 | 166.9 | 167.6 | 100% |
| RTX 4070 Ti | North Carolina (285 W) | 137392 | 155.6 | 156.2 | 155.3 | 99.6% |
| RTX 4070 Ti | France (285 W) | 138259 | 160.4 | 160.1 | 159.2 | 100% |
| RTX 4070 Super | California (209 W) | 145255 | 134.8 | 132.2 | 131.3 | 102% |
| RTX 4070 Super | California (220 W) | 153237 | 138.6 | 137.7 | 138.9 | 99.8% |
| RTX 4070 | Colorado (200 W) | 144340 | 119.2 | 118.5 | 119.7 | 99.6% |
| RTX 4060 Ti | Ohio (150 W) | 27068 | 86.0 | 84.1 | 86.6 | 99% |
| RTX 4060 Ti | Brazil (160 W) | 152073 | 88.7 | 87.7 | 88.4 | 100% |
| RTX 4060 | New Zealand (115 W) | 148383 | 60.4 | 60.1 | 58.5 | 100% |
| RTX 4060 | Australia (115 W) | 143986 | 47.0 | 47.8 | 45.7 | 98% |

### Salad, 29–30 October (PeakMiner 2.20.3, SRBMiner 3.8.0)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 4090 | Salad (450 W) | adbabd35 | 321.8 | 342.6 | 344.0 | 94% |
| RTX 4090 | Salad (450 W) | 2982dea7 | 313.3 | 331.7 | 329.3 | 94% |
| RTX 4080 | Salad (320 W) | 2ffd8ac3 | 197.9 | 209.3 | 209.3 | 95% |
| RTX 4080 | Salad (320 W) | 9850190a | 190.9 | 202.4 | 203.0 | 94% |
| RTX 4070 Ti Super | Salad (254 W) | 8445b495 | 165.0 | 175.8 | 175.4 | 94% |
| RTX 4070 Ti Super | Salad (228 W) | 42d4fae9 | 169.7 | 181.2 | 179.8 | 94% |
| RTX 4070 Ti | Salad (242 W) | cf64ae47 | 153.6 | 163.7 | 163.4 | 94% |
| RTX 4070 Ti | Salad (285 W) | 2a9e8d3d | 160.2 | 168.7 | 168.1 | 95% |
| RTX 4070 | Salad (200 W) | 24656adc | 114.2 | 121.2 | 121.1 | 94% |
| RTX 4070 | Salad (200 W) | 8e6114f1 | 122.5 | 128.6 | 128.5 | 95% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | 4138e3e4 | 91.1 | 96.0 | 95.8 | 95% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | 7e70e38b | 88.3 | 93.4 | 93.1 | 95% |
| RTX 4060 | Salad (115 W) | 4e31b49a | 59.4 | 62.0 | 61.8 | 96% |
| RTX 4060 | Salad (115 W) | 59d34bc2 | 59.7 | 62.8 | 62.6 | 95% |

Where power limits match Vast, the competitors got 4–7% faster and ours didn't.

### Salad, 9–10 November (v0.6.13; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 4090 | Salad (400 W) | 2aeb962b | 292.2 | 322.0 | 323.1 | 334.1 | 87% |
| RTX 4090 | Salad (450 W) | c1e9408c | 319.6 | 342.1 | 343.4 | 345.4 | 93% |
| RTX 4080 Super | Salad (320 W) | 5e5391f8 | 207.1 | 220.4 | 220.4 | 222.3 | 93% |
| RTX 4080 Super | Salad (256 W) | 549668e2 | 191.2 | 216.5 | 216.2 | 214.6 | 88% |
| RTX 4080 | Salad (340 W) | 6e3fff63 | 201.0 | 213.8 | 213.0 | 217.5 | 92% |
| RTX 4080 | Salad (320 W) | d43ec5f2 | 187.3 | 195.4 | 191.8 | 196.8 | 95% |
| RTX 4070 Ti Super | Salad (285 W) | b3484cf9 | 168.9 | 179.8 | 181.1 | 179.1 | 93% |
| RTX 4070 Ti Super | Salad (285 W) | 2ac34122 | 168.9 | 178.8 | 179.1 | 180.9 | 93% |
| RTX 4070 Ti | Salad (285 W) | a4683a51 | 153.3 | 176.3 | 161.7 | 163.1 | 87% |
| RTX 4070 Ti | Salad (285 W) | 61e44847 | 152.8 | 157.2 | 157.5 | 158.4 | 96% |
| RTX 4070 | Salad (160 W) | 907f31cc | 112.9 | 120.4 | 121.1 | 121.8 | 93% |
| RTX 4070 | Salad (152 W) | 884c62d6 | 109.2 | 116.7 | 117.2 | 118.1 | 92% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | ea966099 | 88.0 | 92.0 | 92.3 | 91.3 | 95% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | b6c25a5b | 89.7 | 93.7 | 94.7 | 95.2 | 94% |
| RTX 4060 | Salad (115 W) | 2caeec23 | 63.8 | 66.2 | 66.8 | 66.6 | 95% |
| RTX 4060 | Salad (115 W) | 4ad13b38 | 64.5 | 66.7 | 70.0 | 67.3 | 92% |

Our gap widens as the power limit drops: 87% on the 400 W 4090 against 93% at 450 W, 88% on the 256 W 4080 Super
against 93% at 320 W.

### rig01, native Windows, 10 November (v0.6.14; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

Every Salad row runs in WSL2; our users run Windows. This is rig01's RTX 4090 (450 W, driver 610.88) on Windows 11,
with the installed app stopped. Each miner mined 5 minutes on Kryptex (prl-us:7048) after a 60 s warm-up, two
rounds, the second in reverse order (PeakMiner, SRBMiner, OneZeroMiner, ours, then back). TH/s is each miner's own
figure, as bench.sh reads it: the competitors' API at the end of the run, ours the mean of its status line after the
first minute. The power, SM clock and temperature columns are nvidia-smi means after the first minute.

| Miner | Round 1 | Round 2 | Mean | Power | SM clock | Temp | Shares (acc/rej) |
|---|---|---|---|---|---|---|---|
| Ours (v0.6.14 CLI, the installed app's runtime and cores) | 317.3 | 317.6 | 317.5 | 449 W | 2531 MHz | 77 °C | 21/0 |
| PeakMiner 2.20.7 | 339.1 | 339.1 | 339.1 | 450 W | 2624 MHz | 77 °C | 27/0 |
| SRBMiner 3.8.2 | 339.3 | 337.9 | 338.6 | 445 W | 2619 MHz | 77 °C | 24/0 |
| OneZeroMiner 1.8.5 | 341.9 | 341.8 | 341.9 | 450 W | 2625 MHz | 77 °C | 20/0 |

Ours is 92.9% of the best, the same gap as Salad's 450 W 4090 (c1e9408c, 92.5%), so WSL2 doesn't explain it.
The competitors also land within 1% of that box. All four hold the 450 W cap with the memory at 10251 MHz, but
ours runs about 90 MHz lower on the SM clock (2531 against 2619–2625): it spends more power per clock. Per MHz it
does 0.1254 TH/s against OneZeroMiner's 0.1302, so the gap splits about evenly between clock (3.6%) and work per
clock (3.7%). SRBMiner's own rate dipped to 239 and 291 TH/s for two samples early in round 1; its end figure
didn't. No miner rejected a share.

Downloads were checked against the SHA-256 digests GitHub lists for each release asset, and SRBMiner's zip also
against its published .md5. Windows Defender quarantines SRBMiner on download, so the competitors ran from
`C:\miners\item16-1110`, rig01's excluded folder. A short Python script ran bench.sh's method, as Git Bash has no jq.

## RTX 50-series

Vast, 10 October, on the CUDA 13 core except the India 5090 (driver 570, CUDA 12.8 core).

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 5090 | South Korea (500 W) | 148959 | 380.6 | 374.0 | 384.8 | 99% |
| RTX 5090 | India (500 W) | 151861 | 362.8 | driver too old | 378.2 | 96% |
| RTX 5080 | United States (330 W) | 137656 | 198.7 | 201.2 | 200.2 | 99% |
| RTX 5080 | Poland (360 W) | 153231 | 229.8 | 228.5 | 230.4 | 99.7% |
| RTX 5070 Ti | South Korea (250 W) | 28072 | 92.0 | 93.5 | 91.0 | 98% |
| RTX 5070 Ti | South Korea (250 W) | 28852 | 92.3 | 93.5 | 92.7 | 99% |
| RTX 5070 | Colombia (250 W) | 145974 | 130.9 | 131.2 | 132.1 | 99% |
| RTX 5070 | Texas (250 W) | 147729 | 131.8 | 131.9 | 133.0 | 99% |
| RTX 5060 Ti | Finland (165 W) | 140025 | 93.3 | 92.5 | 94.3 | 99% |
| RTX 5060 Ti | Ontario (180 W) | 153080 | 95.2 | 95.4 | 88.4 | 99.8% |
| RTX 5060 | Virginia (125 W) | 151478 | 76.2 | didn't start | 76.8 | 99% |
| RTX 5060 | India (145 W) | 119163 | 77.4 | 76.4 | didn't start | 101% |

Both 5070 Ti hosts lock the clock at 1342 MHz, far below a stock card.

### Salad, 29–30 October (PeakMiner 2.20.3, SRBMiner 3.8.0)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 5090 | Salad (575 W) | 3fda699b | 404.5 | 438.4 | 440.1 | 92% |
| RTX 5090 | Salad (450 W) | 01bdc2fc | 354.8 | 394.2 | 398.9 | 89% |
| RTX 5080 | Salad (360 W) | cc442fff | 225.8 | 232.9 | 234.3 | 96% |
| RTX 5080 | Salad (360 W) | f6e3944e | 228.5 | 237.0 | 237.4 | 96% |
| RTX 5070 Ti | Salad (300 W) | ebdd97df | 186.5 | 194.7 | 195.2 | 96% |
| RTX 5070 Ti | Salad (300 W) | 770088cd | 185.8 | 195.1 | 195.5 | 95% |
| RTX 5070 | Salad (250 W) | 71aa43da | 134.2 | 141.3 | 141.3 | 95% |
| RTX 5070 | Salad (225 W) | a163e4f3 | 130.0 | 137.0 | 137.3 | 95% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 6c52b860 | 94.1 | 98.6 | 97.1 | 95% |
| RTX 5060 Ti | Salad (153 W, 16 GB) | 78d368bf | 92.7 | 97.7 | 97.5 | 95% |
| RTX 5060 | Salad (145 W) | 616220a5 | 74.0 | 77.7 | 78.1 | 95% |
| RTX 5060 | Salad (145 W) | 0db26521 | 78.5 | 77.4 | 77.7 | 101% |

### Salad, 30 October (SRBMiner 3.8.2)

| Card | Host | Machine ID | Pool | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 5090 | Salad (575 W) | 541c8dda | HeroMiners | 411.6 | 418.2 | 430.4 | 95.6% |
| RTX 5090 | Salad (437 W) | 83be133f | HeroMiners | 330.9 | 366.4 | 377.8 | 87.6% |
| RTX 5080 | Salad (360 W) | b57ed5de | HeroMiners | 213.2 | 223.7 | 229.0 | 93.1% |
| RTX 5080 | Salad (360 W) | f6e3944e | HeroMiners | 228.6 | 237.0 | 239.3 | 95.5% |
| RTX 5070 Ti | Salad (300 W) | afe54169 | HeroMiners | 186.1 | 195.9 | 196.3 | 94.8% |
| RTX 5070 Ti | Salad (300 W) | ca7ddba0 | HeroMiners | 190.5 | 200.6 | 201.3 | 94.7% |
| RTX 5070 | Salad (250 W) | cbdb57ce | HeroMiners | 127.9 | 132.8 | 133.8 | 95.6% |
| RTX 5070 | Salad (250 W) | 5599f5d3 | Kryptex | 128.3 | 133.4 | 134.0 | 95.8% |
| RTX 5060 Ti | Salad (150 W, 16 GB) | f1e711c2 | Kryptex | 92.1 | 97.9 | 86.9 | 94.1% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 3926805c | Kryptex | 96.7 | 101.0 | 101.3 | 95.5% |
| RTX 5060 | Salad (145 W) | 960ed7a7 | Kryptex | 78.6 | 76.1 | 71.8 | 103.2% |
| RTX 5060 | Salad (123 W) | dd173265 | Kryptex | 74.9 | 79.7 | 80.7 | 92.8% |

### Salad, 9–10 November (v0.6.13; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 5090 | Salad (575 W) | 4b8c60bb | 371.3 | 417.1 | 410.5 | 424.7 | 87% |
| RTX 5090 | Salad (450 W) | 2210b832 | 349.9 | 407.3 | 406.3 | 408.3 | 86% |
| RTX 5080 | Salad (360 W) | 183c762b | 224.3 | 235.2 | 225.6 | 236.1 | 95% |
| RTX 5080 | Salad (360 W) | 38fd5fd3 | 187.6 | 196.0 | 196.4 | 192.7 | 96% |
| RTX 5070 Ti | Salad (300 W) | 23e86137 | 160.7 | 167.7 | 168.9 | 167.6 | 95% |
| RTX 5070 Ti | Salad (300 W) | 51e7db39 | 187.8 | 196.0 | 196.8 | 197.7 | 95% |
| RTX 5070 | Salad (200 W) | 75cb2a16 | 124.7 | 133.4 | 133.5 | 133.7 | 93% |
| RTX 5070 | Salad (200 W) | 749878e7 | 126.2 | 134.3 | 134.5 | 136.9 | 92% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | f5eacc1c | 93.4 | 96.5 | 96.6 | 96.7 | 97% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 5ef6ad18 | 92.9 | 97.7 | 101.7 | 97.9 | 91% |
| RTX 5060 | Salad (145 W) | b784c119 | 78.9 | 83.9 | 83.9 | 84.3 | 94% |
| RTX 5060 | Salad (135 W) | 796c35b1 | 73.4 | 77.9 | 78.4 | 77.3 | 94% |

The 5090 trails most, as before.

Which core ran (#290 item 15): every box here reported a driver of 580 or newer (595.97 to 621.14; the two 5090s
621.01 and 621.14), so the app picked the CUDA 13 core on all twelve. None should have been on the CUDA 12.8 core,
which is 3.2% slower on a 5090. The bench didn't log the core itself, so a CUDA 13 core that failed to start and fell
back would not show; nothing points to that. So the 5090s' 86–87% is the CUDA 13 core's real gap on these boxes, not
an old driver. bench.sh now logs our CLI's "Pearl core:" line as "loaded ...", and bench.py keeps it per row.

## A100-class (sm_80)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| A100 PCIe 40 GB | Japan (250 W) | 146405 | 211.1 | 213.7 | 215.3 | 98% |
| A100 SXM4 40 GB | New York (400 W) | 152837 | 245.5 | 251.8 | 253.4 | 97% |
| A100 SXM4 40 GB | Mississippi (400 W) | 117301 | 246.6 | 252.8 | 253.8 | 97% |
| A30 24 GB | Australia (144 W) | 145742 | 100.5 | 90.1 | 90.8 | 111% |
| CMP 170HX | Georgia (250 W) | 147823 | 163.4 | 165.5 | 165.3 | 99% |

The A30's host lowered the power limit during the competitors' runs; at equal power ours is about 107%.

## Hopper (sm_90a)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| H100 SXM 80 GB | Czechia (575 W) | 148671 | 682.8 | 697.7 | 688.1 | 98% |
| H100 NVL 94 GB | Japan (400 W) | 29785 | 536.5 | 548.6 | 539.1 | 98% |
| H100 PCIe 80 GB | Czechia (350 W) | 147981 | 484.9 | 448.3 | 488.0 | 99% |
| H200 141 GB | — | — | — | — | — | not run: over budget |
| H200 NVL 141 GB | — | — | — | — | — | not run: over budget |

Last H200 figures (8–9 October): 99.9% on the H200 141 GB and 101% on the H200 NVL.

## Ada workstation and data-center cards (sm_89)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| L40S | Texas (350 W) | 96563 | 290.7 | 253.7 | 291.9 | 99.6% |
| L40S | Romania (350 W) | 150627 | 288.1 | 252.8 | 290.4 | 99% |
| RTX 6000 Ada | New York (300 W) | 142572 | 210.0 | 180.4 | 219.0 | 96% |
| RTX 6000 Ada | Germany (300 W) | 152478 | 216.6 | 186.6 | 225.8 | 96% |
| RTX 5000 Ada | Missouri (250 W) | 107776 | 208.4 | 185.9 | 210.7 | 99% |
| L40 | — | — | — | — | — | not run: none for rent |
| L40 | New York (250 W) | 56199 | 153.2 | 132.8 | 156.6 | 98% |
| RTX 4500 Ada | France (210 W) | 148971 | 139.7 | 134.2 | 139.6 | 100% |
| RTX 4000 Ada | Norway (130 W) | 149238 | 98.0 | 94.5 | 98.5 | 99.5% |
| RTX 4000 Ada | France (130 W) | 152701 | 96.1 | 92.4 | 97.4 | 99% |
| L4 | Czechia (72 W) | 109442 | 79.3 | 73.8 | 80.9 | 98% |
| L4 | Washington (72 W) | 152541 | 80.5 | 75.0 | 81.6 | 99% |
| RTX 2000 Ada | Romania (70 W) | 68269 | 48.3 | 46.9 | 48.7 | 99% |

## RTX PRO Blackwell (sm_120)

Vast, 10 October, on the CUDA 13 core.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX PRO 6000 Workstation | Mississippi (600 W) | 148552 | 398.7 | 389.1 | 419.2 | 95% |
| RTX PRO 6000 Server | Australia (600 W) | 139338 | 435.1 | 432.4 | 435.9 | 99.8% |
| RTX PRO 6000 Max-Q | Australia (300 W) | 142021 | 308.0 | 302.9 | 306.7 | 100% |
| RTX PRO 5000 48 GB | California (250 W) | 150611 | 211.5 | 208.3 | 214.8 | 98% |
| RTX PRO 4500 | Arizona (200 W) | 148915 | 175.7 | 171.2 | 176.0 | 99.8% |
| RTX PRO 4000 | France (145 W) | 147490 | 127.9 | 126.5 | 128.8 | 99% |
| RTX 6000D | Czechia (550 W) | 149788 | 144.6 | 144.6 | 144.6 | 100% |

## Ampere workstation and data-center cards (sm_86)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX A6000 | Vietnam (300 W) | 140998 | 121.6 | 122.3 | 124.8 | 97% |
| RTX A6000 | Romania (300 W) | 37274 | 126.1 | 125.7 | 124.9 | 100% |
| A40 | United Kingdom (300 W) | 152158 | 125.8 | 124.6 | 128.2 | 98% |
| A40 | Belgium (300 W) | 151322 | 123.2 | 122.5 | 127.3 | 97% |
| RTX A5000 | North Carolina (230 W) | 147999 | 93.7 | 92.8 | 94.0 | 99.7% |
| RTX A5000 | Italy (200 W) | 151139 | 84.6 | 83.4 | 87.7 | 97% |
| RTX A4000 | Estonia (140 W) | 145734 | 57.9 | 56.0 | 61.6 | 94% |
| RTX A4000 | Kentucky (140 W) | 29102 | 58.2 | 56.8 | 62.3 | 93% |
| RTX A2000 12 GB | Poland (70 W) | 28227 | 28.7 | 29.0 | 32.3 | 89% |
| RTX A2000 6 GB | Japan (70 W) | 152199 | 27.6 | 28.5 | 30.4 | 91% |

SRBMiner holds a higher clock than ours at the same power on the A2000s and A4000s.

### CMP 90HX on Clore, 4 November (v0.6.11, our miner only)

| Card | Power limit | TH/s after 5 min | SM clock |
|---|---|---|---|
| CMP 90HX (GPU 0) | 200 W of 250 W | 72.7 | 1485 MHz |
| CMP 90HX (GPU 1) | 200 W of 250 W | 72.7 | 1500 MHz |
| CMP 90HX (GPU 2) | 200 W of 250 W | 71.0 | 1380 MHz |

At 72 TH/s it earns about $0.056 an hour, below Clore's cheapest rent for the card.

## B200 (sm_100)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| B200 180 GB | Oregon (1000 W) | 96942 | 457.9 | 1380.6 | 1394.7 | 33% |

We mine it on `mma.sync`, which tops out near 568 TH/s here; the competitors use tcgen05.

## Tesla T4 (sm_75)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| Tesla T4 16 GB | Czechia (70 W) | 28909 | 30.1 | 32.5 | 33.1 | 91% |

## Tesla V100 (Volta, sm_70)

Vast, 10 November (PR #290 item 9). PeakMiner 2.20.7, SRBMiner 3.8.2 and OneZeroMiner 1.8.5 ran as bench.sh runs
them, on Kryptex. No release has V100 code yet, so "Ours" is this branch's core built for sm_70 alone (the dp4a fold
GTX 10 and GTX 16 cards mine on), run by `earn/native/probes/turing/pool.sh`. Not a release row.

| Card | Host | Machine ID | Ours (dp4a, this branch) | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| Tesla V100 SXM2 32 GB | Washington (300 W) | 155858 | 29.2 | exits | 40.0 | 50.6 | 58% |

- PeakMiner retires the card at start ("all GPUs retired", exit 70) twice. SRBMiner drew 289–295 W, OneZeroMiner
  292–297 W at 1410 MHz; ours 298 W at 1507 MHz.
- Ours passed verify-hits 248 of 248 and ran 29.15 TH/s for 5 minutes, about 242 MAC an SM a clock: 95% of the 256 a
  dp4a allows on Volta's 64 integer lanes an SM. So the dp4a fold is at its ceiling here. A further run of about
  12 minutes on Kryptex had 2 shares accepted and 0 rejected (none in the 5-minute run). The box cost $0.06.
- OneZeroMiner's 50.6 TH/s is about 450 MAC an SM a clock, 88% of the fp16 tensor cores' 512, so it most likely mines
  on HMMA with fp16 operands: |A'|, |B'| ≤ 63 are exact in fp16, and its fp32 sums are exact integers under 2^24
  (2048 × 63² is 8.1M). A Volta fold like that would be about +70% over dp4a.
- Why it matters: Clore listed about 1,900 V100 SXM2 cards on 4 November, 1,240 of them idle, at a median $2.06 a
  card-day. At 29 TH/s a V100 earns about $0.02 an hour, so like the CMP 90HX it doesn't pay to rent; it's for owners.
  Until this branch our core had no sm_70 code at all, so a V100 couldn't mine on it.

### The fp16 tensor-core fold, 10 November (PR #290 item 20)

Same Vast machine (155858, V100 SXM2 32 GB, 300 W). `pearl_tile_fold_hmma` is the dp4a fold's 128x128 tile on the
fp16 tensor cores: A' and B' converted to fp16 as they are staged, `mma.sync` m8n8k4 into fp32 accumulators. Each
build was raced against the dp4a fold (`-DPEARL_VOLTA_HMMA=0`) in one process (`compilers.sh` race stage, 4 s turns),
then checked with verify-hits.

| Build | TH/s | vs dp4a | SM clock | Power | verify-hits |
|---|---|---|---|---|---|
| dp4a fold (item 9) | 29.71 | — | 1530 MHz | 291 W | 248/248 (item 9) |
| wmma 16x16x16 | 35.30 | +19.2% | 1523 MHz | 295 W | 314/314 |
| m8n8k4, k outermost | 40.45 | +36.0% | 1500 MHz | 299 W | 359/359 |
| + no bank conflicts on A' | 41.45 | +39.6% | 1485 MHz | 299 W | 232/232 |
| + accumulators start at 2^23 (shipped) | 43.84 | +47.9% | 1477 MHz | 298 W | 400 verified, 0 failed |

- The shipped build's race ran 32 rounds, 32 ahead, ci95 0.2%; a copy of the dp4a build ran +0.000%. On Kryptex (`probes/turing/pool.sh`, prl-us), 20 minutes measured after a minute's warm-up: 43.85 TH/s, 3 shares accepted and 0 rejected, 1477 MHz, 299 W; its verify-hits passed 398 of 398. The box cost about $0.20 for two hours.
- 43.8 TH/s is 87% of OneZeroMiner's 50.6 here (was 58%), and ahead of SRBMiner's 40.0.
- Exactness: fp16 holds every int8, and fp32 sums of integers are exact below 2^24. (The note above that |A'| and
  |B'| stay under 64 was wrong: at fill 48 they reach 111. The sums are what matter.) The accumulators start at 2^23,
  so the readout takes the float's bits as the integer, which needs each one in [0, 2^23) at every chunk's end. At
  fill 48 the nearest end is 14 sd away (chunk 0); `pearl_config.h` has the numbers. `probes/hmma-check.cc` runs the
  fold's layout and readout on the CPU against a plain fold.
- What each step did. wmma's 16x16x16 runs its four k-steps of an 8x8 tile back to back, each waiting on the last;
  m8n8k4 with k outermost puts 16 other mma between them. A' rows 32 apart sat on the same banks, so every 16-byte
  load of A' was a 2-way conflict; swapping rows 0-3 and 4-7 of every other quad pair's tile fixes it (+3.0%). The
  readout first added a magic constant and subtracted it again (2 ops an accumulator); starting the accumulators at
  2^23 leaves only XORs (+6.0%), and 45 fewer registers (210).
- Ablations on the 41.45 build (timing only, wrong results): mma and shared loads alone 58.4 TH/s, 487 MAC an SM a
  clock, 95% of the tensor cores' 512. Without the int8-to-fp16 staging +12.0%, without the old readout +10.5%,
  without the global loads +8.7% (and the clock rises to 1530), without the barriers -3.2%.
- Tried and dropped (each against the shipped build, same race): converting and storing half the next stage between
  the two k-steps -7%; loading two stages ahead +0.3%; fragment loads of 8 or 4 halfs at a time +0.2%; tile walk
  band 4 -0.1%; 1024 column offsets a batch +0.4%. All inside the noise or worse.
- What's left: the card holds its 300 W cap at 1477 MHz, 371 MAC an SM a clock (72% of peak) against OneZeroMiner's
  ~449 at 1410 MHz. The staging (conversion and shared stores) and the global loads are the remaining ~20%. A
  128x256 tile would cut both by a quarter but fits only one block an SM; fp16 operands in global memory would drop
  the conversion but double the bytes loaded. Neither is tried.

## GTX 10-series (Pascal, sm_61, on Salad)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| GTX 1080 Ti | Salad | not run | not run | 0 | 0 | — |
| GTX 1080 | Salad (180 W) | c868132d | not run | 0 | 0 | — |
| GTX 1070 | Salad | not run | not run | 0 | 0 | — |
| GTX 1060 | Salad (6 GB) | not run | not run | 0 | 0 | — |
| GTX 1050 Ti | Salad (4 GB) | not run | not run | 0 | 0 | — |

The competitors mine nothing on Pascal; our v0.6.8 Pascal core mined 17.8 TH/s on a GTX 1080 Ti before release.

## GTX 16-series (Turing without tensor cores, sm_75, on Salad)

Salad, 17 October, v0.6.0 against PeakMiner 2.17.6 and SRBMiner 3.7.1.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| GTX 1660 Super | Salad (106 W, 6 GB) | 1535c250 | 1.1 | 1.13 | 1.13 | 97% |
| GTX 1660 Super | Salad (119 W, 6 GB) | 535c608c | 1.0 | 1.01 | 0.96 | 96% |
| GTX 1660 Ti | Salad (6 GB) | 741f9c56 | 1.2 | 1.14 | 1.16 | 104% |
| GTX 1660 Ti | Salad (88 W, 6 GB) | 30b3c19c | 1.3 | 1.28 | 1.29 | 101% |
| GTX 1660 | Salad (120 W, 6 GB) | 095beb7b | 1.1 | 1.05 | 1.04 | 104% |
| GTX 1660 | Salad (6 GB) | not run | not run | not run | not run | — |
| GTX 1650 | Salad (75 W, 4 GB) | 522e3507 | 0.7 | 0.68 | 0.69 | 99% |
| GTX 1650 | Salad (4 GB) | not run | not run | not run | not run | — |

No tensor cores, so every miner is equally slow and none is worth running on these cards.

Our miner now runs the GTX 10 cards' dp4a fold on these cards instead of the tensor fold (`PEARL_DP4A_TURING` in
pearl_config.h, on by default). Raced against the tensor fold on the same PC, Salad, 7 November. These are our own
builds on nvtest boxes, not the release files, so the rows above are unchanged:

| Card | Machine ID | Tensor fold | dp4a fold | Change | dp4a fold's clock and power |
|---|---|---|---|---|---|
| GTX 1660 Super | e4505b49 | 1.095 | 9.615 | +777% (ci95 3.3%, ahead 3 of 3 rounds) | 1895 MHz, 117 of 125 W |
| GTX 1650 | 13120675 | 0.692 | 5.654 | +717% (ci95 45%, ahead 2 of 2 rounds) | 1738 MHz, 67 of 75 W |

- Both cards run about 231 MAC an SM a clock. That is 90% of the 256 a half-rate dp4a allows, so there is little left
  to gain. The tensor fold drew only 47 W and 34 W.
- Both found the tensor fold's exact hits on three batch shapes and passed the pool's hit check.
- On Kryptex the 1660 Super mined 9.48 TH/s for 45 minutes, with 4 shares accepted and 0 rejected. The 1650 mined
  5.48 TH/s for 35 minutes on the default build, with 2 accepted and 0 rejected.
- Still not worth renting: Salad's 1660 Super costs $0.027/h at low priority, against $0.034/h for an RTX 2060 that
  mines 40–48 TH/s. The gain is for people who own these cards.

## AMD Radeon (RDNA 3 and 4, on Salad)

Salad, 30 October, v0.6.9 against SRBMiner 3.8.0. PeakMiner mines nothing on AMD.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RX 9070 XT | Salad (16 GB) | a68b3382 | 114.4 | 0 | 111.3 | 103% |
| RX 9070 XT | Salad (16 GB) | 21001ac3 | 117.4 | 0 | 117.4 | 99.9% |
| RX 9060 XT | Salad (16 GB) | f9b8be8b | 61.2 | 0 | 56.1 | 109% |
| RX 9060 XT | Salad (16 GB) | cff5c764 | 61.7 | 0 | 56.2 | 110% |
| RX 7900 XTX | Salad (24 GB) | 9a57035a | 48.8 | 0 | 53.0 | 92% |
| RX 7900 XTX | Salad (24 GB) | 45c94e40 | 53.1 | 0 | 53.9 | 99% |
| RX 7900 XT | Salad (20 GB) | cc2f03b6 | 53.1 | 0 | 51.4 | 103% |
| RX 7900 XT | Salad (20 GB) | 67c308d2 | 47.2 | 0 | 47.8 | 99% |
| RX 7800 XT | Salad (16 GB) | d0dbba63 | 30.2 | 0 | 31.2 | 97% |
| RX 7800 XT | Salad (16 GB) | 53c8eb50 | 31.2 | 0 | 30.9 | 101% |

v0.6.12 is about 13% faster on RDNA 4 than these rows; it hasn't been raced against SRBMiner yet.

### Salad, 9–10 November (v0.6.13 against SRBMiner 3.8.2)

PeakMiner and OneZeroMiner don't mine on AMD.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RX 9070 XT | Salad (16 GB) | 28f11ef2 | 131.0 | 0 | 91.5 | 0 | 143% |
| RX 9070 XT | Salad (16 GB) | c1940c4c | 120.8 | 0 | 105.9 | 0 | 114% |
| RX 9060 XT | Salad (16 GB) | d3ae7070 | 70.0 | 0 | 54.9 | 0 | 128% |
| RX 9060 XT | Salad (16 GB) | 3540c3e2 | 59.7 | 0 | 54.0 | 0 | 111% |
| RX 7900 XTX | Salad (24 GB) | 6f96bc89 | 49.8 | 0 | 54.0 | 0 | 92% |
| RX 7900 XTX | Salad (24 GB) | ae456453 | 50.8 | 0 | 54.1 | 0 | 94% |
| RX 7900 XT | Salad (20 GB) | dc8fcef5 | 46.3 | 0 | 45.4 | 0 | 102% |
| RX 7900 XT | Salad (20 GB) | 1d5cab04 | 48.4 | 0 | 47.7 | 0 | 101% |
| RX 7800 XT | Salad (16 GB) | 559a11d6 | 31.8 | 0 | 30.9 | 0 | 103% |
| RX 7800 XT | Salad (16 GB) | 16f1685c | 32.9 | 0 | 32.0 | 0 | 103% |

Every PC passed a 10-minute hit check: 400 hits verified, 0 failed, 7.94–8.05 hits a salt. The 7900 XTX is the one
AMD card where SRBMiner leads, on both PCs and in both run orders.

Rerun on the first of those PCs (`6f96bc89`), 10 November (item 7), with v0.6.13's AMD kernel (the same fold code,
built at 1c78804b): the duel (ours first, then SRBMiner, 5 minutes each on Kryptex, after a 30-minute sweep had
warmed the card) read ours 54.37 TH/s, mean after the first minute, and SRBMiner 54.40, its last figure (mean 53.89).
That is 99.9%, not the row's 92%. `hashrate.js` read 53.48 to 54.35 in three rounds, and the CLI's trace shows the
fold at 99% of each batch (10.07 of 10.17 ms; hash 0.8%, idle 0.2%). So the row's 49.8 was a low reading of this
PC, not a gap in the kernel. The row stays until a full rerun (see "Keeping this file current"). What off-chip
operand traffic costs here, from the same sweep (each build in turn, three rounds of `hashrate.js`, four of the
fold alone):

| Build | `hashrate.js` TH/s | Fold alone | Clock (stamps) |
|---|---|---|---|
| v0.6.13 | 53.48, 54.27, 54.35 | 54.8-57.3 TMAC/s | 2564 MHz |
| Operands held in the L2 (`PEARL_AMD_ABLATE=16`, timing only) | 55.59, 55.94, 55.93 (+3.3%) | +2.3 to +3.4% | 2640 MHz (+3.0%) |
| No slab staging at all (`PEARL_AMD_ABLATE=32`, timing only) | 57.19, 56.83, 56.79 (+5.4%) | +1.3 to +4.4% | 2698 MHz (+5.2%) |

Cycles a slab are the same in all three (6,699-6,745); waves wait 30-31% of the slab loop at the barrier and 0.7% on
memory. So on the 7900 XTX operands from memory cost about 3% of clock, and a kernel that drew them on chip could get
at most that, less what the draw costs.

RDNA 3's buffer loads (`PEARL_AMD_RDNA3_BUFLD`, on by default on gfx1100 and gfx1101; issue #287, item 2), 7 November:
the release and the new build taking turns on one PC, order rotating, rounds 2-5. The twin is a second copy of the
release, for the noise.

| Card | PC | Fold alone | `hashrate.js` | Twin: fold, `hashrate.js` |
|---|---|---|---|---|
| RX 7900 XTX | `57cc1801` | +1.17, +1.02, +1.25, +1.04% | +0.77, +0.76, +0.77, +0.66% (release 54.27-54.34) | −0.22 to +0.14%, 0 to +0.17% |
| RX 7800 XT | `72dd708c` | +0.81, +1.34, +0.86, +0.88% | +0.82, +0.79, +0.79, +0.75% (release 31.81-31.82) | 0 to +0.62%, 0 to +0.03% |

- TID alone (XTX): fold +0.16%, `hashrate.js` +0.04%. The gain is the buffer loads'.
- verify-hits checked 400 of 400 hits on both PCs. On Kryptex the XTX got 8 accepted shares and 0 rejected in 20
  minutes at 54.5-54.8 TH/s. The 7800 XT's default build (the shipping code) got 1 accepted and 0 rejected in 15
  minutes, but the pool reset the connection 4 times, 3 of them just after a share went out, so those shares got
  no answer. Through the stand-in pool (`verifypool:cli`) the same build's 86 shares all passed the spec's checks,
  and on another 7800 XT (`70b39a06`) it got 2 accepted, 0 rejected, with no reset in 10 minutes. So the resets
  were that PC's link to the pool.
- The 7800 XT's +0.62% twin round is one low release reading (31.765 TMAC/s against 31.97-32.01 in the others).
- Not yet raced against SRBMiner, so the rows above stay as they are.

RDNA 3's staging cuts on top of those loads (`PEARL_AMD_RDNA3_SET`: PFX, CRANK and SCHEDB, on by default on gfx1100
and gfx1101; issue #287, item 8), 7 November: item 2's build (base) and the new one taking turns on one PC, order
rotating, scored from round 2. The twin is a second copy of the base.

| Card | PC | Fold alone | `hashrate.js` | Twin: fold, `hashrate.js` |
|---|---|---|---|---|
| RX 7900 XTX | `28a12f15` | +0.71, +0.60, +0.88, +0.62% | +0.89, +1.01, +1.20% (base 55.22-55.70) | 0 to +0.13%, not run |
| RX 7800 XT | `72dd708c` | +0.83, +0.89, +0.85, +0.87% | +0.90, +0.86, +0.83, +0.93% (base 31.25-31.28) | −0.02 to +0.08%, −0.10 to +0.10% |

- The same with the slab loads pinned two slabs ahead (`PEARL_AMD_PF2PIN`, which the set leaves off) was slower on
  both: fold +0.52% and `hashrate.js` +0.38% on the XTX, +0.67% and +0.66% on the 7800 XT. CRANK and SCHEDB alone
  lost 1.22% in the fold on the 7800 XT.
- verify-hits checked 400 of 400 hits on the 7800 XT. On Kryptex the XTX got 6 accepted shares and 0 rejected in
  10 minutes at 55.6-57.1 TH/s.
- The pinned build was also raced on two more XTX PCs: on `fd4ebc44` the fold +0.87 to +1.37% and `hashrate.js` +0.36
  to +1.15% (mean +0.68%; that PC drifted, its twin read up to +0.75%), verify-hits 400 of 400 and 4 accepted, 0
  rejected in 15 minutes on Kryptex; on `c15f2792` the fold +0.24 to +0.53%, before Salad moved the box.

### Our own builds on Salad, 7 November: the RDNA 4 readout, late and in three chains (#287 item 4)

After the six RDNA 4 cuts the slab loop has no VALU but the readout's, about 88 a chunk, run after the chunk's last
WMMAs. `PEARL_AMD_RDLATE` moves it into the next slab, after that slab's fragment loads are issued (=2 splits it over
two slabs), and `PEARL_AMD_RDTREE` XORs each region's sums as three chains, not one. Both now default on with the set
(Linux gfx12). Against the release on one PC, `test/sweep.sh`: each build's fold checked bit for bit first, then
`hashrate.js` 60 s after 20 s, 4 rounds, the builds taking turns, round 1 not scored. ctl is the release under an
extra flag, the same code. Not rows: no competitor ran.

| Card, k | PC | Release (TH/s) | ctl | RDTREE | RDLATE=1 | RDLATE=2 | RDLATE=1 + RDTREE | Both (shipped) |
|---|---|---|---|---|---|---|---|---|
| RX 9070 XT, 8192 | `33042316` | 125.73–125.81 | −0.02, +0.01, +0.02% | +1.22, +1.28, +1.22% | +1.34, +1.25, +1.26% | +1.38, +1.28, +1.27% | not run | **+1.72, +1.66, +1.67%** |
| RX 9060 XT, 4096 | `f9b8be8b` | 70.91–70.92 | 0.00, 0.00, 0.00% | +1.03, +1.06, +1.04% | not run | not run | +1.62, +1.73, +1.69% | **+1.38, +1.42, +1.41%** |

- **The fold alone** (k 4096, miner operands, 4 rounds of 300 launches): on the 9070 XT both read +1.24, +1.34, +1.51%
  (ctl −0.08 to +0.19%). On the 9060 XT that timing was noisy (ctl +0.15, +1.75, +1.74%) and only the hashrate rounds
  count.
- **Clock stamps** (fold alone): a slab fell from 1888 to 1831 cycles on the 9070 XT (−3.0%) with the clock 2357 →
  2332 MHz, and from 1889 to 1831 on the 9060 XT (−3.1%) at 2700 → 2704 MHz. On both the wait at the barrier rose from
  622 to 721–722 cycles a slab while the rest fell from 1,266–1,267 to 1,109–1,110: the waves reach the barrier sooner and give
  part of that back waiting there.
- **Code** (ROCm 7.2.0): 213 VGPRs against 201, 24 KB of LDS, 7 waves by registers (6 run), no spills or scratch,
  the release's staging. On both boxes the default build's fold was byte for byte the measured flags' (md5
  `7706cb31`), and `-DPEARL_AMD_RDLATE=0 -DPEARL_AMD_RDTREE=0` gave the release's (`b6633aeb`). The Windows core (set
  off) is unchanged at 193 VGPRs and 36 KB.
- **Checks of the default build:** the fold test at k 2048, 4096 and 8192 on both chips; `verify-hits` 400 of 400 on
  the 9070 XT at k 8192 (130 salts, PC `c10c6949`) and on the 9060 XT at k 4096 (147 salts); Kryptex for 15 minutes,
  15 accepted and 0 rejected on the 9070 XT (all 15 decoded as valid k 8192 proofs, one connection) and
  7 accepted and 0 rejected on the 9060 XT (7 valid k 4096 proofs).
- **Left open:** RDLATE=1 with RDTREE read 0.24–0.31% more than the shipped =2 on the 9060 XT in every round, under the
  +0.3% median the plan asked for, on one PC, and never on a 9070 XT or a pool.
- **Cost:** a 9070 XT box for 74 minutes (10 of them a restart that went silent until `realloc`) and a 9060 XT box
  for 55, high priority.

RDNA 4's tile order (`PEARL_AMD_WALK=1`, on by default on gfx1200 and gfx1201 in the Linux core; issue #287, item 5),
7 November: supertiles about the size of what the card holds at once (4 x 12 tiles on an RX 9060 XT, 8 x 12 on an
RX 9070 XT), every other row of them backwards, against v0.6.13's order on the same PC, `hashrate.js`.

| Card, k | PC | `hashrate.js` | Fold alone |
|---|---|---|---|
| RX 9060 XT, 4096 | `9ac12ceb` | +0.38% | — |
| RX 9060 XT, 4096 | `ffa9f87b` | +1.85% | +1.91% |
| RX 9070 XT, 8192 | `c10c6949` | +0.56% | — |
| RX 9070 XT, 8192 | `1dba3ce9` | +0.75% | +0.57% |

- verify-hits checked 400 of 400, and Kryptex accepted 14 shares with none rejected.
- Dealing each supertile's blocks to one sub-tile per shader array (`=2`) read 0.04-0.20% above `=1`, under the bar.
- On an RX 7900 XTX (`45c94e40`) both were level (+0.06%, -0.04%), and the stamps put only 19% of blocks on array
  i % 12, so RDNA 3 keeps the old order. So does the Windows core, which mines k 2048.

### Our own builds on Salad, 7 November: RDNA 4's #287 items 4, 5 and 6 together

Each was raced alone against the release. Merged (exp/287-combined, 45b2c771), they were raced together on one PC per
card: base (all three off; its fold's md5 is the release's, b6633aeb, as main built on the same box), a no-op twin of
base, item 4 alone, items 4 and 5 (9060 XT only) and the merged default. `hashrate.js` 60 s after 20 s, order
rotating, round 1 not scored. Not rows: no competitor ran.

| Card, k | PC | Base (TH/s) | Twin | Item 4 | Items 4 + 5 | Items 4 + 5 + 6 |
|---|---|---|---|---|---|---|
| RX 9060 XT, 4096 | `85387649` | 70.63–71.31 | +0.06, −0.58, −0.07, −3.08, −0.55% | +1.71% (5 of 5) | **+2.20%** (5 of 5) | +1.71% (5 of 5) |
| RX 9070 XT, 8192 | `1dba3ce9` | 121.05–123.02 | −1.84, +0.03, −0.76, 0.00% | +0.92% (4 of 4) | **+1.66%** (4 of 4) | item 6 is off on gfx1201 |

- **Item 5 adds on top of item 4** on both: `hashrate.js` +0.49% (ahead in 4 of 5) and +0.74% (4 of 4); the fold alone
  +0.38% and +0.40% (5 of 5, 4 of 4).
- **Item 6 doesn't.** On top of items 4 and 5 it lost 0.47% on the 9060 XT (behind in 4 of 5; fold alone −0.39%, 0 of 5).
  On a second 9060 XT (`b1aeab8d`) the twin swung 1.4% and it read −0.12% by mean, +0.90% and +0.81% by median in
  the two settled rounds. So `PEARL_AMD_FUSE` is off by default; the gfx1200 that ships is the items 4 + 5 build.
- **Noise:** the 9060 XT's twin swung up to 3.1% from round 4 on, the 9070 XT's up to 1.8%. The bar, twice the twin's
  mean |delta|, was 1.73% and 1.31%: items 4 + 5 cleared it on both, items 4 + 5 + 6 missed by 0.02%.
- **Checks:** every build's fold bit-exact at 1024 and 4096 square (and k 8192 on the 9070 XT). The merged default:
  verify-hits 400 of 400 on both (on the 9060 XT with item 6 on), Kryptex 4 of 4 shares on the 9060 XT and 9 of 9 on
  the 9070 XT, none rejected, one connection each.
- **Cost:** three boxes, about 141 minutes, high priority.

### Our own builds on Salad, 10 November: NVIDIA ideas on AMD (PR #290 item 23)

Which of the NVIDIA miner's gains could carry to RDNA 2, 3 and 4. The desk review is in "Research notes" (NVIDIA work
on AMD). Two cheap ones were built behind switches and raced against the release on one PC each, `test/sweep.sh`:
the fold checked bit for bit first, then `hashrate.js` 60 s after 20 s, the builds taking turns, with a twin of the
release (an unused flag) for the noise. Not rows: no competitor ran.

- `PEARL_AMD_HASHQ`: the hash kernel, its hit count and the slot's event on a queue of their own, after the fold's
  event, so the next batch's fold follows this one with no hash between them (`pearl_host_hip.cpp`, Ctx). Each slot's
  fold waits for that slot's last hash first. The AMD form of the NVIDIA hash work: AMD's hash in the fold (#287 item 6)
  lost, and a stream per slot was level on an RX 9070 XT at k 2048 (23 October).
- `PEARL_AMD_RDTREE11`: RDNA 4's three XOR chains a region (`PEARL_AMD_RDTREE`) in RDNA 3's k 2048 readout.

| Card, k | PC | Release (TH/s) | Twin | `PEARL_AMD_HASHQ` | `PEARL_AMD_RDTREE11` |
|---|---|---|---|---|---|
| RX 7900 XTX, 2048 | `44d3716f` (run 1) | 51.39–51.64 | −0.15 to +0.06% | **+3.06 to +3.62%** (5 of 5) | −0.18 to −0.52% (5 of 5 behind) |
| RX 7900 XTX, 2048 | `44d3716f` (run 2, Salad gave the same PC) | 51.46–51.75 | −0.35 to +0.10% | **+2.99 to +3.71%** (6 of 6) | — |
| RX 7900 XTX, 2048 | `3bf70206` (4 rounds, then the PC restarted the container) | 55.97–56.72 | −0.13 to −0.29% | **+0.73 to +1.00%** (4 of 4) | — |
| RX 7900 XTX, 2048 | `3bf70206` (6 rounds) | 55.09–56.30 | −0.76 to +0.25% | **+0.21 to +0.62%**, first round +1.92% (6 of 6) | — |
| RX 7800 XT, 2048 | `72dd708c` | 32.62–32.66 | 0.00 to −0.03% | −0.06 to +0.03% | — |
| RX 7800 XT, 2048 | `16f1685c` | 32.84–33.01 | −0.18 to 0.00% | −0.03 to −0.27% (6 of 6 behind) | — |
| RX 9060 XT, 4096 | `8c96a2a7` | 73.14–73.23 | −0.03 to +0.04% | −0.21 to −0.36% (6 of 6 behind) | (gfx12 has RDTREE) |

- **The XTX gain is in the fold, not the hash.** `PEARL_AMD_TRACE` on `44d3716f`: the release's batch is 10.67 ms, the fold
  10.57 and the hash 0.084. With the hash on its own queue the batch is 10.33 ms and the fold itself 10.29–10.31 ms.
  That is 0.26 ms a batch from the fold, three times the hash it no longer waits for. On both 7800 XTs and the 9060 XT
  the fold got 0.1 ms slower with the hash beside it, which ate what the hash saved. This XTX runs the release at
  51.5 TH/s where item 7's XTX ran 54.4, so the size of the gain is this PC's, not the card's. The second XTX (`3bf70206`, 56.2 TH/s on the release) gained
  +0.2 to +1.0%: ahead of the release and of the twin in all 10 rounds, but its fold didn't get faster (9.68 ms either
  way); the batch lost about the hash's 0.08 ms and no more.
- **Checks.** Every build's fold bit-exact. verify-hits with the hash on its own queue: 400 of 400 on the XTX twice
  (3,646 and 3,655 hits, 8.0 a salt) and 400 of 400 on the second XTX (3,836 hits, 480 salts), 0 failed. On Kryptex the XTX got 6 accepted shares and 0 rejected in 10 minutes
  at 53.3 TH/s.
- **RDTREE11 loses** in the fold alone too (−0.25 to −0.59% a round, twin ±0.3%). RDNA 3's readout is a quarter of
  RDNA 4's share of a chunk (its WMMA is a quarter as fast), and the three chains' extra live registers cost more.
  Dropped; not in the code.
- **Shipped:** `PEARL_AMD_HASHQ` is on for gfx1100 (RX 7900) only: the default is 2. Two XTX PCs gained (+3% and
  +0.2 to +1.0%), 21 of 21 rounds ahead. No RX 7900 XT or GRE has run it. gfx1101 (RX 7800 XT, 7700 XT) and RDNA 4
  stay off. `-DPEARL_AMD_HASHQ=1` turns it on for every card, `=0` for none, and the environment variable `PEARL_AMD_HASHQ=1` (or `0`) overrides the build at run time.
- **Cost:** five boxes (one moved by Salad mid-run), about 4.5 box-hours, high priority.

## Research notes

What future speed work needs: how to measure, the limits found, and what has already been tried.

**Measuring**
- Race mode (`earn/native/probes/race.cu`, builds taking 4 s turns in one process) is the comparison to trust: a copy
  of the release reads within ±0.05–0.10%. Separate `hashrate.js` runs favour the build that runs on the warm card.
  It also runs on Windows, each build a DLL: on rig01's RTX 4090 a copy of the release read −0.015% ±0.030 over 25
  rounds.
- Compare builds on the same PC only. Salad PCs of one card differ by up to 24% on the same code, and some run far
  slower for every miner because something else is using the GPU.
- Our CLI's 5-minute mean carries about ±1.5–2.4% noise on fast cards, so 1–2 points between PCs mean nothing.
- On AMD, use 10-minute hit checks: 2-minute checks read low hits a salt for no known reason. Salad's `amd-smi`
  reports no power or clocks.
- Time a short fold on a Salad AMD PC before a race. On 7 November a 7900 XTX PC (`ec58ea79`) ran `pearl_amd_peak`'s
  plain WMMA loop at 161 MAC a CU a clock (241 on a good PC) and its 4-VALU loop at 5.6 TMAC/s, and didn't finish
  600 launches of the fold in 15 minutes.
- Cards mine at their power limit, so a saving shows up as clock, and cuts don't add: the six RDNA 4 cuts summed to
  about 8.9% alone and gave +4.3–5.4% together.

**Limits found**
- RDNA 4's WMMA does 1,009 MAC a CU a clock (98.6% of AMD's 1,024). The fold ran at 70% of that in v0.6.11 and 77% after
  KPAIR and BUFLD (4 November). Each VALU instruction beside the WMMAs costs about a cycle of WMMA time. After the six
  cuts the slab loop's only VALU is the readout's (about 88 a chunk); run beside the next slab's fragment loads and in
  three chains it gave +1.4–1.7% (#287 item 4, 7 November). Its count can't fall much further without the readout
  forms already dropped (below).
- RDNA 3's slab loads one slab ahead are nearly hidden already. The stamps of the buffer-load fold put the wait for
  them at the release fence at 0.76% and 0.79% of the slab loop on two RX 7900 XTX PCs and 1.85% on an RX 7800 XT
  (7 November), while waves wait 27-30% of it at the slab barrier for each other. Two slabs ahead
  (`PEARL_AMD_PF2PIN`) gained +0.05 to +0.42% in the fold.
- RDNA 3's WMMA does 238-241 MAC a CU a clock (93-94% of AMD's 256; `pearl_amd_peak`, RX 7900 XTX and 7800 XT,
  7 November). A VALU beside the WMMAs costs about 1 clock of WMMA time alone and 3.5-5 in a dependent chain, read
  from MAC a CU a clock, since the clock moves between kernels.
- A stock RTX 5090 loses about 8% of its clock to moving operands from L2. Mid RTX 50 cards hit their top clock first,
  so they could gain at most 2–3.3%. A byte copied from another SM's shared memory costs as much energy as one read
  from L2 (33.0 against 34.1 pJ, #287 item 7), so sharing operands between SMs by a copy saves nothing.
- On RTX 30/40 the tile hash costs about 1.3% of a tile, almost all of it the hash running past the partner's window
  (step 0): each cycle past about 2,300 after the hand-off costs a tile 0.9–1.0 cycle, and the hash releases at about
  3,150. Any edit to the hash's code makes ptxas reallocate the chunk loop, which then costs 3,383 cycles a tile.
- On RTX 30/40 the staging ring's waits cost 475 cycles a tile on a 4090 (0.46%), and all of it is that hash seam.
  With the hash stubbed out, removing every ring wait made the tile 164 cycles slower (101,381 → 101,545; #287 item
  1's builds). So lighter ring synchronization has nothing to win outside the seam (suggestion list item 4, 6 November). RTX 50's TMA
  ring hasn't been ablated, but a 5070 Ti's cycles a tile (102,989) are within 0.2% of the 4090's, and a 5090 with
  no TMA bytes at all did +0.15% ±0.3 more work a clock.
- The hash on RTX 50 was never stubbed out (the stubs are cp.async-ring only), but it can cost little more than on
  Ada: a 5070 Ti at a locked 1346 MHz runs the fold at 102,989 cycles a tile, 211 (0.2%) behind the 4090's 102,778.
  Its TMA producer (warp 4) folds only 96 IMMAs after the hand-off before it waits for the hashers, against 128 on
  Ada. Issuing just that refill at the stage's end is bounded by the 0.2%, and `PEARL_TALL_TMA_FILL_PT` 16, which
  includes it, read +0.07% on a 5090; not built. At the power limit a cycle saving comes back at about half on RTX 30
  (the paired hash: −0.30% cycles gave +0.14 to +0.22%, and −0.15% on a 3080 Ti), so the whole hash is worth under 1%
  there. (Suggestion list item 8, 6 November, from records.)
- Everything outside the fold costs about 0.2% of a card's time, so there is little to win there (suggestion list item
  10, 6 November, from records, no GPU run). On the 4090 a salt is 223 ms. Between salts the miner rewrites A' (0.34 ms, DRAM-bound: 0.15%)
  and two batch tails show (~0.07 ms each). Launch gaps are ~2 µs a batch. Hit proofs are read while the next batch
  runs. Job switches every 35–60 s cost nothing measurable. Halving the redraws (n 131072 to 262144) measured +0.2%.
  On an RX 9060 XT, `PEARL_AMD_TRACE` shows 0.13% outside the fold and hash kernels. A larger k doesn't dilute the
  redraw, because A' and a salt's work both grow with k; only n does. The one stage above 0.5% is the transcript hash
  (1.36% in the 4090's fold; 0.7–1.4% as AMD's separate kernel).
- The B200 needs a tcgen05 fold; `mma.sync` can't pass about 41% of the competitors there.
- The RTX 4090's memory clock can't be tested on any 4090 we reach (#287 item 11, 6 November). rig01's account isn't
  an Administrator (nvidia-smi -lmc, -lgc and -pl exit 4), and Vast and Salad containers can't set clocks. At stock
  the core ran 315.9 TH/s over 8 legs (0.30% spread) at 2519 MHz and 449 of 450 W, power capped in every sample, with
  memory at 10251 MHz and 2-3% busy. TH/s follows the SM clock (125.1-125.8 TH/s per GHz), and the memory clock is
  worth at most 40 W, so a lower one might give +0.5-1.5%. That is a guess until someone with admin rights runs a
  sweep (exp/287-11 has `memclock-sweep.js`, about 20 minutes).
- RDNA 2 (`dot4`, a fold for RDNA 2) is a projection, not a measurement: 14.0–41.0 TH/s, half that if its `v_dot4`
  runs at half rate. No RDNA 2 card has run it, and none can be rented: Salad has no RDNA 2 class and Vast had 0
  offers for every RX 6600 to 6950, V620 and W6800 (#287 item 10, 6 November). The only rental found is Azure's
  NGads V620, $0.47 an hour spot for a whole card.
- AMD's separate transcript hash (#287 item 6, 6 November, `PEARL_AMD_TRACE`): 0.41–0.45 ms of a 30–31 ms batch on an
  RX 9060 XT at k 4096 (1.4%; 128 MiB read at 322 GB/s), 0.23 ms of 33.6 ms on an RX 9070 XT at k 8192 (0.7%).
  Hashing inside the fold (`PEARL_AMD_FUSE`) made the fold 0.33 ms slower on the 9060 XT and 0.36 ms on the 9070 XT.
  It went in on gfx1200 first: `hashrate.js`, interleaved on one PC against two copies of the release, read +0.34% on a
  9060 XT (54b217a9: +0.30 to +0.38% a round over 5 scored rounds, copies within 0.08%), +0.36 to +0.44% in the quiet
  rounds of another (b9b8ec1e), with verify-hits 400 of 400 and 7 Kryptex shares in 30 minutes, none rejected (refverify
  7 of 7). The shipped build (default flags, gfx1200 only) passed again on a third 9060 XT (9454033f): verify-hits 400
  of 400, 3 Kryptex shares in 10 minutes, none rejected. On 7 November, on top of #287 items 4 and 5, it lost 0.47% on a
  9060 XT and read level on a second, too noisy to settle (the combined race above), so it is off by default.

**Credited against shown rate, all four miners (item 10, 10 November, stand-in pool)**

Each miner mined 10 minutes on the stand-in pool (`earn/native/probes/verify-pool.js`, run by `bench.sh` with
`VPOOL=BITS`). The stand-in checks every share as the spec's verifier does and refuses duplicates. Credited is
accepted shares × 2^(256−BITS) MACs over the run, less its first minute. Shown is the mean of the miner's own figure
after the first minute (PeakMiner: its API). Every share was accepted: 0 rejected, 0 duplicates, all miners.

| PC | Target | Miner | Shown | Credited | Shares | Credited ÷ shown |
|---|---|---|---|---|---|---|
| RTX 4090, 450 W (`1d80ae90`) | 2^210 | ours v0.6.13 | 331.4 | 346.5 | 2,945 | 1.05 ±0.02 |
| | 2^210 | PeakMiner 2.20.7 | ~355 | 334.1 | 2,836 | 0.94 |
| | 2^210 | SRBMiner 3.8.2 | 350.4 | 349.1 | 2,926 | 1.00 ±0.02 |
| | 2^210 | OneZeroMiner 1.8.5 | 351.9 | 259.6 | 2,157 | 0.74 |
| RTX 5090, 477 W (`ad3451c9`) | 2^210 | ours v0.6.13 | 346.7 | 334.7 | 2,857 | 0.97 ±0.02 |
| | 2^210 | PeakMiner | 405.2 | 379.9 | 3,251 | 0.94 |
| | 2^210 | SRBMiner | 407.1 | 399.1 | 3,343 | 0.98 ±0.02 |
| | 2^210 | OneZeroMiner | 400.7 | 272.5 | 2,295 | 0.68 |
| RTX 5090, 575 W (`756d361b`) | 2^207 | OneZeroMiner | 424.3 | 436.2 | 461 | 1.03 ±0.05 |
| | 2^207 | PeakMiner | 424.7 | 415.2 | 446 | 0.98 ±0.05 |
| | 2^207 | SRBMiner | 420.5 | 445.6 | 462 | 1.06 ±0.05 |
| RTX 5060 Ti (`a668f722`, driver 610) | 2^210 | ours v0.6.14, CUDA 13 core | 98.4 | 100.6 | 842 | 1.02 ±0.035 |

- The competitors' figures are honest. At a target near the pools' (2^207 is 64 times easier than Kryptex's 2^203)
  all three are credited what they show, within the share count's noise. SRBMiner matched at 2^210 too.
- The easy target cost OneZeroMiner a quarter of its shares and PeakMiner about 6%. At 2^210 a share comes about
  5 times a second. OneZeroMiner's own accepted count equalled the stand-in's, so it sent everything it found but
  found less. PeakMiner reported a submit latency of 1.3 s at 2^210 against 84 ms at 2^207, so its shares queue.
  That is a test artifact, not a pool loss: on Kryptex a 4090 finds a share every 26 s. Don't run this test above
  2^207 for them.
- PeakMiner, SRBMiner and OneZeroMiner all mine k = 65536, the most the spec allows at rank 128 (k ≤ 4r²). We mine
  k = 2048. A share pays the same per MAC at any k, so this costs us nothing in credit. It can matter for speed: work
  done once per region (finalize and the transcript hash) is spread over 32 times as many MACs.
- 342–345 TH/s on a 450 W 4090 is real work. The ~340 pure-IMMA ceiling was measured on rig01 at 2715 MHz in
  September. These Salad 4090s ran at 2664–2754 MHz, so their ceiling is about the same or higher. SRBMiner was
  credited 349 at 2750 MHz.
- Our own figure is honest too: 1.05, 0.97 and 1.02 on three cards, mean 1.01.
- RTX 5060 Ti (item 17's 0.73 on Kryptex): on v0.6.14 with the CUDA 13 core, the hit check passed on both cores
  (400 of 400 hits verified; cu13 1,298 hits and cu12.8 1,262 in 120 s), and the stand-in credited 1.02 of the shown
  rate. On this PC the core finds every hit and the shares carry the full rate. So the Kryptex shortfall is not in
  the core or the shares. It is somewhere between the box and the pool, or particular to the fleet's PCs.

**Pool credit against our hashrate, by cause (item 17, 10 November, pool data and logs only)**

The fleet's credit ratios, 0.97–0.98 on Kryptex and 0.93 on HeroMiners, are the pools' pay and luck, not shares we
lose. The credited work matches the hashrate we show to about 1%. A share at either pool is 2^53 MACs (difficulty
2,097,152 × 2^32), so a card showing R TH/s should send R × 10^12 / 2^53 shares a second.

| Cause | Kryptex | HeroMiners | How measured |
|---|---|---|---|
| Pool pay (fee, PPS+ terms) | 0.956 of share value | fee 0 (PROP) | Kryptex: credited ÷ ledger 0.982 (to 4 Nov, before other workers joined the wallet) ÷ work 1.027 below |
| Pool luck | none (PPS+) | 0.945 ±0.086 | HeroMiners' blocks 13–20 Oct, our mining there: 134 blocks, Σ difficulty ÷ Σ round shares |
| Orphaned blocks | none | −0.7% | 1 of 135 in that window |
| Work credited ÷ work counted | 1.023 (ledger), 1.000 ±0.020 (shown rate) | ~1.00 (0.937 ÷ 0.945 ÷ 0.993) | 125,435 Kryptex shares of 613 Salad groups against the watcher's ledger; 2,497 accepted against 2,497 expected from the boxes' own minute lines (80 box-hours, 9–10 Nov) |
| Stale shares | 0.14% fleet, 0.06% dev4090, ~0.3% Salad | penalty from 2% stale | Kryptex worker list: 13 stale of 9,213 in 24 h; dev4090 10 of 16,130 in 7 days |
| Rejected or invalid | 0 | — | 0 invalid of 9,213 in 24 h; 5 rejected of 2,502 in the Salad lines, 4 on one RX 9060 XT PC |
| Vardiff and target rounding | 0 | 0.0015% | both pools hold 2,097,152; HeroMiners' target is 0x7fff8… (1 − 2^−16 of 2^203) |
| Reconnects | ~0.03% | — | 3 reopened connections in 6.1 online worker-hours (two were bench workers), ~6 s each |
| Hits dropped at a job switch | 0.1–0.9% by card (estimate) | more: jobs every 35 s | from the code, below |

- **dev4090**, the one long run: Kryptex credited 9,049 shares on 5, 8 and 9 November, 314.5 TH/s ±1.1%, against
  316–317 shown. On Salad the ledger reads 2–3% under what Kryptex credits in every class (missed minute lines and
  each box's first minutes), so it can't see a 1% loss; the boxes' minute lines can (above).
- **The job switch.** Jobs came 81 an hour on Kryptex (33 in 25 min; 24 of them at the same height) and 122 an hour
  on HeroMiners (every 35 s; 23 of 30 at the same height), probed from rig01 on 10 November. When a job arrives the
  core finishes the batch it is waiting on and the one queued behind it, and the miner drops their hits
  (`hit.jobId !== job.jobId` in `_onHit`). The shown rate counts that work. A batch is col_batch × 8192 regions ×
  2^19 MACs, so about 2 batches a switch is 0.13% on a 4090 (28 ms batches), 0.17–0.21% on a 4060 or 3060 (37, 46 ms),
  0.10% on a 5060 Ti, and 0.4–0.9% on RTX 20 (Turing's col_batch 2048: 84–195 ms) at Kryptex's rate; half again at
  HeroMiners'. The batch times are from the code, not timed; the drop is now logged (below) so a box can count it.
- **RTX 5060 Ti on 10 November: 0.73 ±0.07.** Three boxes on v0.6.14 (cu13 core, drivers 610, 617, 621, four PCs)
  showed 95–100 TH/s and Kryptex credited 149 shares against 204 expected in 5.3 hours, steady through each run
  (368: 38 in 90 min against 57). Kryptex's counts match the miners'. The watcher already judged one at the pool's
  53 TH/s. No NVIDIA code changed between v0.6.13 and v0.6.14, and earlier cu13 5060 Ti boxes read about 0.95
  (329: 0.946 over 341 shares). Every other class was within its noise (RTX 2060 0.985 ±0.043, RX 9070 XT 1.014 ±0.041,
  4070 Ti 1.051 ±0.045). Not explained yet.

Fixes, by expected gain:
1. Find the 5060 Ti shortfall: on a 5060 Ti box, verify-hits plus a low-difficulty share count through verifypool
   (item 10's tool) against the shown rate. If a fold misses hits, it is up to 27% on that class; nothing else here
   is above 1%.
   Done on one PC (item 10, above): hits and shares were whole there (1.02 credited ÷ shown), so look between the
   box and Kryptex next.
2. Send same-height hits from the replaced job. HeroMiners runs a cryptonote-nodejs-pool fork (v1.3.439), which usually keeps recent jobs and
   rejects only a changed height; Kryptex is untested. 73–77% of switches are same-height, so it gets back most of
   the switch loss: about 0.1% on RTX 30/40/50 and 0.3–0.7% on RTX 20. Test first on one Kryptex box: HeroMiners
   penalises 50% of pay at 2–5% stale, so a pool that turns them down must not get them.
3. Shorter batches on Turing (col_batch 2048 → 512) to cut its switch loss 4×, if a race shows no launch cost.
4. A Pearl pool watchdog like Monero's ("no word from the pool in 5 minutes; reconnecting": 131 of them on Monero in
   six days on Salad). No Pearl silence was seen, so this is insurance, under 0.1%.

Changed: the miner now logs "dropping a hit for the job replaced N ms ago (same block | new block)" instead of
dropping it silently, so the switch loss can be counted on a box. Nothing it sends changed.

**What made PeakMiner faster after 2.18 (item 2, 10 November, Salad)**

Each release on one PC, against the stand-in pool (`verify-pool.js serve`, a fixed target, a new job every 30 s)
through `sharetap.py`, so every share was checked and decoded. 2.5 minutes a release, the first 45 s not counted; the
3060 ran two passes (forward, then reverse). Script: `pm.sh` in the item's scratchpad (not committed). Cost about $0.13.

| RTX 3060, `50f45594`, 170 W | Shown TH/s | SM MHz | Memory controller | VRAM | Shares ÷ shown rate | Share config |
|---|---|---|---|---|---|---|
| 2.17.6, 2.18.3, 2.19.1 | 50.2–50.6 | 1800–1837 | 39–65% | 1,334 MiB | 0.72–0.87 | m 16384, n 65536, k 8192, tile 8×32 |
| 2.19.2, 2.20.0 | 53.7–53.8 | 1890–1905 | 0% | 354 MiB | 0.96–1.04 | m 8000, n 14336, k 65536, tile 10×16 |
| 2.20.1 | 54.0 | 1897–1905 | 0% | 354 MiB | 0.93–0.94 | as 2.19.2 |
| 2.20.2 | 54.3–54.5 | 1905–1912 | 0% | 396 MiB | (its API read 0 in half the samples) | m 5248, n 261632, k 32768, tile 10×16, rows contiguous |
| 2.20.7 | 54.3 | 1905–1920 | 0% | 356 MiB | 0.94–0.96 | m 7680, n 14336, k 65536, tile 24×8 |

- **The step is 2.19.2 (19 October, "higher hashrate and better efficiency on RTX 30xx, 40xx and 50xx"): +6.6% shown
  on the 3060.** The memory controller went from 39–65% busy to 0% and VRAM from 1.3 GB to 354 MB, and the card spent
  the power it saved on clock: +4.3% SM clock at the same 170 W. The other +2.3% is work a clock (27.7 → 28.4 TH/s per
  GHz, 99% of the 3060's dense int8 peak), most likely from k 65536 (8× longer folds, so the per-attempt hash,
  noise and readout cost 8× less). 2.17.6 → 2.19.1 changed nothing on Pearl (within 0.4%). 2.20.1–2.20.7 add about 0.9%.
- **Every release sends constant operands:** every A and B byte in every decoded share is 64 (fill 64), rank 128,
  and no share was rejected. With constant A and B, A' and B' are 64 plus the low-rank noise, so a kernel can make them
  on chip from the small noise factors instead of reading them from DRAM. That is what 2.19.2 does: no DRAM traffic at
  all while mining. SRBMiner 3.8.2 and OneZeroMiner 1.8.5 show the same 0% at about 1,900 MHz on the 9–10 November 3060s.
- **Ours looks like PeakMiner before 2.19.2.** On the 9–10 November RTX 30 boxes our miner showed 32–35% memory
  controller, 1,400 MiB and 1,815–1,822 MHz against their ~1,900 MHz. On RTX 40 and 50 our memory controller is only
  4–7% busy (4060, 5060), but the clock still trails by 2–3% (2,685 against 2,760 MHz; 2,512 against 2,572). This is
  #290 item 7 (operands built on the GPU); on RTX 30 it should be worth about the 4% clock, plus whatever the longer k
  gives a clock.
- **Shares.** At a job every 30 s (Kryptex sends one every 44 s, HeroMiners every 35 s) the old releases' shares came
  to only 0.72–0.87 of their shown rate, falling over the run, while 2.19.2 and later held 0.93–1.04 (each run 190–335
  shares, so ±6–7%). So the credited gain was likely bigger than the shown 6.6%; a job switch costs the DRAM version
  much more (a 16384 × 8192 A and 65536 × 8192 B to rebuild), but a longer job interval would be needed to prove it.
- **RTX 5060 Ti (`922b3948`, 150 W, driver 621), one pass:** 2.19.1 exits at once on this driver (2.19.2 fixed "a
  crash on RTX 50xx with recent NVIDIA drivers"). 2.19.2 96.1, 2.20.0 97.1, 2.20.2 97.6, 2.20.4 98.0, 2.20.5 98.3,
  2.20.6 98.5, 2.20.7 98.5 TH/s: +2.5% in small steps, all at 0% memory controller, 492–524 MiB, 2,677–2,700 MHz.
  Configs: m 16000, n 15360, k 65536, tile 10×16 (2.19.2–2.20.0); 2.20.2 as on the 3060; m 15360, n 15360,
  k 65536, tile 24×8 from 2.20.4. Shares 0.91–1.00 of shown.
- **The memory clock lock isn't it.** 2.19.0 added `--gpu-lmem 810` and 2.20.7 tuned for a 405 or 810 MHz lock, but
  only with the flag. On Salad, 2.20.7 with `--gpu-lmem 810` logged "memory clock lock 810 MHz failed (native
  memory-control function is outside the selected NVML library)" and ran at 54.1 TH/s, the same as without it; the
  memory stayed at 7,301 MHz in every run. The flag pays where it can be set (item 1), and only because the kernel
  no longer needs DRAM.
- 2.20.7 offers zstd at login (`"encoding": ["zstd", "gzip"]`) and sends gzip when the pool says only `"type":"v2"`.

**Picking RTX 30's launch shape on each PC at startup (item 5, 10 November, Salad)**

What the miner can choose between at run time today: the CUDA 12.8 or CUDA 13 core (`PEARL_CORE_VARIANT`; picked by
itself only for RTX 50), and, on RTX 30 alone, the launch shape. RTX 30 has one core (12.8). Its shape is chosen on the
host from the card's L2 size: the flip (B' held in the L2 slice, a 128-offset batch; the 3060 Ti and the 6 MB cards) or
the band walk (A' held, band 8 or 4, a 512-offset batch). The kernel reads the band at run time and the rest is host
state, so a core can switch shapes between batches with no new build. Every other card has one fixed launch.

Race mode, the rule's shape (flip 128) against the band walk, on four RTX 3090 Ti PCs and an RTX 3080 Ti
(`nvtest`, 600–900 s):

| PC | Power, driver | Band 8 x 512 | Band 4 x 512 | Memory busy, flip / walk |
|---|---|---|---|---|
| efd04f2b | 450 W, 621.14 | −0.58% (0 of 36 rounds ahead) | −1.30% (0/36) | 11% / 18% |
| daa97471 | 450 W, 591.59 | −3.23% (0/34) | −3.92% (0/34) | 12% / 52% |
| b659e537 | 480 W, 617.42 | +0.02% (15/29; A/A sd 2.8%) | −0.76% (4/29) | 11% / 19% |
| c1a87da3 | 480 W, 621.14 | **+0.39% (25/28)** | −0.06% (10/28) | 12% / 23% |
| 594fad00 (RTX 3080 Ti) | 350 W, 617.14 | −0.53% (1/27) | −1.40% (1/27) | 9% / 17% |

On efd04f2b, batch 1024 read −0.43% and the walk without its L2 slice −0.68%; on daa97471 −3.79% and −3.39%. So the
same card's best shape moves by PC, from flip ahead by 3.2% to the walk ahead by 0.4%. The rule's pick lost on one PC
of four, by 0.4%. That is the size of what a per-PC pick can win on a 3090 Ti; it doesn't explain the 85% vs 95% PCs
in the 9–10 November table.

Built (commits on item/5): with `shapeTrialSecs` the core races its shapes while it mines, in 4 s turns ended on a
whole salt with the first second uncounted, the order reversing each round, as race.cu does. It keeps another shape
only when that beats the rule's by 0.3% on average and in two thirds of the rounds. The app runs a 150 s trial the
first time a card opens and caches the pick per card for a week (`launch-shapes.json`); `PEARL_SHAPE_TRIAL=0` turns it
off and `PEARL_SHAPE` forces one. Hits found during the trial are sent as usual.

The trial against the race, same PC: c1a87da3 picked band 8 x 512 in both runs (+0.76% and +0.43%, ahead in 10 of 10
rounds each); b659e537 kept flip 128 (band 8 −0.15% and −0.33%); the 3080 Ti kept flip 128 and read band 8 −0.47% and
band 4 −1.37%, against the race's −0.53% and −1.40%. verify-hits with the trial switching shapes: 400 of
400 on each of three builds (hits from flip, band 8 and band 4). A first version counted each turn from the switch,
while the clock was still settling, and read daa97471's −3.23% as −0.25%; the warm-up fixed it. The trial costs about
a third of 150 s at the worse shapes' rate, once a week.

**RTX 30's band-walk cards: does the flip ever win, and does the trial pick right? (item 5b, 10 November, Salad)**

On the RTX 3060, 3070, 3070 Ti and 3080 10 GB the rule picks the band walk. Each PC raced the rule against the flip
forced on (`-DPEARL_AMPERE_FLIP_SHARE=1000`, the width the trial times) and both band depths (`-DPEARL_AMPERE_BAND`;
the one equal to the rule's is a twin), plus a copy of the rule's build as a second twin (`nvtest` race mode, 900 s,
4 s turns). Then hashrate.js twice and verify-hits.js with the 150 s trial on the rule's build.

| PC | Card, power | Rule | Flip | Other band | Twins | Trial kept (flip, other band) |
|---|---|---|---|---|---|---|
| be2418c9 | 3060, 170 W | band 4 x 512 | flip 32 −1.74% (0/42) | band 8 −0.62% (1/42) | −0.02, −0.03% | band 4 twice (−1.76/−1.81, −0.66/−0.54) |
| 60cdc4aa | 3070, 347 W | band 8 x 512 | flip 64 −0.58% (0/39) | band 4 −0.17% (2/39) | −0.04, −0.03% | band 8 twice (−0.60/−0.49, −0.27/−0.05) |
| bb028be9 | 3070 Ti, 288 W | band 8 x 512 | flip 64 +0.04% (21/42) | band 4 −0.08% (11/42) | −0.01, −0.02% | band 8 twice (−0.07/+0.15, −0.09/−0.02) |
| 95be0c9b | 3070 Ti, 230 W | band 8 x 512 | flip 64 +0.01% (16/40) | band 4 −0.11% (11/40) | +0.03, −0.01% | band 8 twice (−0.03/−0.18, −0.33/−0.37) |
| 161a910f | 3080 10 GB, 320 W | band 8 x 512 | flip 64 −0.14% (0/40) | band 4 −0.88% (0/40) | −0.02, −0.03% | band 8 twice (−0.14/−0.15, −0.87/−0.88) |
| db9aa4de | 3080 10 GB, 287 W | band 8 x 512 | flip 64 −0.65% (15/40) | band 4 −0.97% (14/40) | +0.04, +0.65% | band 8 twice (−0.46/−0.33, +0.17/−0.48) |

The flip never won: it lost clearly on the 3060, 3070 and one 3080, and tied the walk on both 3070 Tis (within the
twins' spread). db9aa4de was noisy (per-round sd 2.7–4.7%, a twin read +0.65%), so its row says little, except that
the trial's +0.3% and two-thirds rule kept it from chasing the noise. On every PC the trial kept the race's winner (or
the rule, where the race tied), and on the quiet PCs its figures came within about 0.1 point of the race's.
verify-hits: 400 of 400 on every build on every PC, including the rule's build with the trial switching shapes.

The rule stays as it is: on these cards it is right or tied everywhere, so the trial buys nothing on them beyond its
cost (about a third of 150 s at up to 1.8% less, once a week). In the forced-flip build, where the flip is the
rule, the verify runs' trial moved to the walk on the 3060 (race: walk 1.74% ahead) and on db9aa4de (0.65%, noisy),
and kept the flip on the 3070 (walk 0.58% ahead), the 3070 Tis (tied) and 161a910f (0.14%). The default build never
runs the flip on these cards, but it shows the 0.3% bar sits about where the smaller gaps are.

**NVIDIA work on AMD (PR #290 item 23, 10 November, desk review, then two races)**

Every NVIDIA change that gained speed, against what the AMD core (`earn/native/amd`) already does. Ranked by what
is left to gain per hour of work. AMD cards mine at their power limit, so only energy a MAC (clock) or VALU beside
the WMMAs (work a clock) can pay. Item 7 put RDNA 3's whole operand staging at +5.4-6.9% and its off-chip share at
+3.3% (both ablations, not reachable).

| NVIDIA change (gain there) | AMD today | Left for RDNA 2 / 3 / 4 |
|---|---|---|
| Hash out of the fold's way: in the fold, 1.5 regions a lane, the paired hash (+0.14-0.22% RTX 30), "keep the GPU fed" (+0.7%) | separate hash kernel, 0.7-1.4% of a batch; in the fold (`FUSE`) lost | **raced** as `PEARL_AMD_HASHQ` (above): +3.0-3.7% on one XTX PC and +0.2-1.0% on a second, level or behind on two 7800 XTs and a 9060 XT. On for gfx1100 only |
| Readout off the matrix path: shared-XOR readout, readout before the chunk-0 guard | RDNA 4: `RDLATE` + `RDTREE` (+1.4-1.7%) | RDNA 3: **raced** `RDTREE11`, −0.2 to −0.5%. `RDLATE` needs the wrap readout and KPAIR, which RDNA 3 doesn't run; item 13 put k 4096 under +0.3% |
| Newer compiler: ptxas 13 (+3.2% on the 5090, nothing on sm_86 or sm_75) | ROCm 7.2.0 pinned (Linux CI image), HIP SDK 7.2 on Windows | ROCm 7.2.4, 7.14 and 10.1 exist (`rocm/dev-ubuntu-24.04` tags; 7.14 and 10.x images are 8-9 GB). No code; the fold's asm pins and sched_barriers were tuned on clang 22, so it can lose too. Guess −1 to +2%, 2-3 h of box plumbing. **Next to try** |
| Per-PC launch-shape trial at startup (item 5: up to +0.4-0.8% on some RTX 3090 Ti PCs) | fixed per arch: batch 128 (RDNA 3) or 256, `WALK` on gfx12, k 4096/8192 by CU count | RDNA 4's `WALK` read +0.38 to +1.85% by PC and k 8192 pays only on big gfx1201, so a trial could pick per PC. Guess +0.2-0.5%, 2-4 h (host shapes for `pearl_core.cc`'s trial). After the compiler |
| Taller tile 192x256 (+3.4% on a 5090, +4.6% on a 4090 over 128x256) | 128x256, 8 waves of 64x64 | 192 rows: 12 waves split 896 staging units unevenly, and m 131072 isn't a multiple of 192; 256x256 drops to 4 waves a SIMD in one block a WGP. Cuts staged bytes 22-33% (worth at most about a third of item 7's +5.4-6.9%). A new fold, days. Low per hour |
| Operands built on chip, no DRAM (PeakMiner 2.19.2, +6.6% on a 3060) | stored A' and B' | Ceiling +3.3% on an XTX (item 7); NVIDIA's draws cost more than that (E4 −3.6 to −5.6%), and on RDNA every VALU costs WMMA time (1 clock RDNA 4, 1-5 RDNA 3). Not worth it |
| L2 residency: RTX 30 flip and band walk, persisting L2 slice, Ada and RTX 50 batch narrowing | `WALK` (RDNA 4), batch 128 (RDNA 3); RDNA has no L2-persistence control | done where it paid; `WALK` level on RDNA 3 |
| Fill 48 (+3.6% on a 4090), k-blocked operands (+0.7%), restamp instead of redraw, hit list read only on hits | all in the AMD core (fill above 48 tried; `TILED`; portable kernels) | done |
| Longer k (RTX 50/40 lost at k 4096) | RDNA 4 k 4096/8192 shipped; RDNA 3 k 4096 under +0.3% (item 13) | done |
| Ring roles and shared bases in registers (+0.36% sm_120) | `LDS32_PIN` on RDNA 4 | RDNA 3: pinned offsets at 48-byte rows lost a little on RDNA 4; skip |
| Memory clock locked low on Blackwell | can't set clocks on Salad (WSL, no `amd-smi` power) | an owner's own Linux rig only (root, `rocm-smi --setmclk`); untested |

RDNA 2 (the `dot4` fold, which no RDNA 2 card has run): of these only host-side ones reach it (`PEARL_AMD_HASHQ`, the
compiler, a shape trial). The first thing to learn there is the tester's own rate, and whether `v_dot4_i32_i8` runs at
full rate on RDNA 2 (the projection is 14-41 TH/s).

**Tried and dropped (don't repeat without a new idea)**
"Item N" alone means issue #287 or an earlier numbered plan; "suggestion list item N" is the ten-item list worked on
6 November, which stopped on every item before any GPU time.

- RTX 5090 B' drawn on the SM (research item 8, 10 November, desk study, no card time). The "+1.81%" from 2 November
  (b68b0399, 53ddd838, 87ba3d6e) was `PEARL_TALL_SHAM_DRAW`, a timing-only stand-in: it reads each B fragment from
  shared memory again and adds one IADD3 a register, but skips the real draw's table loads and byte gathers (each B'
  byte is 48 + T[col][p0] - T[col][p1], 8 one-byte lookups a register). So +1.81% is the ceiling (skipping B' entirely
  read +7.65%). A real draw needs about 2.5x the sham's extra instructions (about 100 a warp a stage against 40), and a
  tile's table is 32 KB with only 2.7 KB of shared memory free beside the ring. Rank space doesn't help: the chunk
  length equals the rank (128), so the dense work is the same and the sums leave int8. Item 7's on-chip draws on an
  RTX 3090 lost 3.6-5.6% for the same reason. Closed without the re-race; the sham rebased onto f7ccb357 is on branch
  item/8 (be35e42b) if anyone wants it. The 5090 lead to follow instead is long k (item 7: +2.80% on a 3090 at
  k 16384 with operands in L2). Item 21 (below) raced it: long k loses on the 5090 with stored operands.
- RTX 30 B' held at a width that fills the L2 slice (#287 item 3, 6 November, race mode, `PEARL_AMPERE_FLIP_FILL` on
  exp/287-3). On an RTX 3080 10 GB it lost to the band walk: −0.44% (±0.07, 0 of 46 rounds ahead) on a 320 W PC and
  −0.55% (±0.20, 2 of 45) on a 309 W PC. Memory busy fell from 21–23% to 13–14% and the clock rose 1.1%, but TH/s per
  MHz fell 1.5–1.6%. On the same card band 4 lost 0.6–0.8%, width 256 lost 0.27% and width 1024 read +0.11% (±0.12),
  so band 8 at 512 stays. The 3080 Ti's flip read −0.14% (±0.06) against the band walk at 400 W but +1.63% (±0.27) on
  a 350 W PC that slowed to about 1300 MHz as it warmed, so it keeps the flip.
- k 4096 on NVIDIA: RTX 20 (item 9) loses 1.2–2.8% to DRAM; RTX 50 and 40 (item 8) −0.78% and −0.29% at the power cap.
- Larger k after cutting operand traffic (suggestion list item 3, 6 November, from records, no GPU run). On a 4090 the
  fold takes 102,658, 101,764 and 101,220 cycles per 2048 of k at k 2048, 4096 and 8192: about 1,900 cycles a tile are
  fixed (mostly the hash), so depth can win at most +1.64% a clock at k 16384. A 3090 with operands held in L2 turned
  +1.30% and +1.56% a clock at k 8192 and 16384 into +0.6% and +0.8% at the power cap. With today's operands depth
  loses (above), and no on-chip operand supply pays (compact tables, below), so it stops. The cp.async fold's
  `tile_srcs` offsets are 32-bit and overflow past k 16384.
- CUDA 13's ptxas for RTX 30 (sm_86) and RTX 20 (sm_75) (PR #290 research item 14, 10 November, static, no GPU
  run). The shipped `pearl_core_cu13.node` holds only sm_89 and sm_120 (cuobjdump of v0.6.14's), so RTX 30 and 20
  mine on 12.8 code; the "CUDA 13 core" Vast rows are RTX 50 and RTX PRO only. Built `pearl_kernel.cu` on rig01 with
  ptxas 12.8.93 and 13.3.73 from the same PTX, and with cicc 13.3 too. sm_75: all 24 kernels byte-identical under
  both. sm_86: 20 of 24 identical; the tall fold differs only in register and predicate names, one NOP turned into a
  no-op uniform add and one branch's sense: 3,056 instructions in the same order, 254 registers, 0 spill, 7,775
  stall cycles in both, control bits different on 4 instructions. `pearl_blake3_chunk_cvs` and
  `pearl_restamp_operand` save 2 stall cycles of 3,100 and 4,600. The 12.8 build's fold hash (35efb7e36535) is the
  release's, so the comparison is against what ships. A race could only measure noise, so none was run. Ptxas 13's
  gain is Blackwell's (+3.2% on the 5090); it doesn't reach Ampere or Turing.
- Computing only the hash word the target test reads (PR #290 research item 19, 10 November). The kernel already
  does this. Every fold hashes a region with `pearl_transcript_msw`, `pearl_hp_msw` or `pearl_hp_msw_pair`, which
  return only the top word, and runs the full hash again only for the rare region that passes the first-word test.
  What was left to try was writing the dead work out by hand: in the last BLAKE3 round, diagonals 1 and 3 don't
  feed the word that is read. A test switch (`PEARL_HASH_MSW_TRIM`, on scratch branch `exp/290-19`, commit
  546551c9) left them out. ptxas was already dropping them: the PTX loses 20 lines and 4 shuffles a tall fold, the
  machine code loses one instruction. sm_75: all 25 kernels byte-identical. Tall fold on sm_86 3,042 → 3,041
  instructions (stall cycles 7,775 → 7,778), sm_89 3,030 → 3,029 (7,704 → 7,699, same under ptxas 12.8 and 13.3),
  sm_120 3,706 → 3,705 (9,679 → 9,645). Raced on Salad, 900 s, 4 s turns, with an A/A twin of the base:

  | Card | Rounds | Base TH/s | Trim vs base | Twin vs base |
  |---|---|---|---|---|
  | RTX 3090 (420 W cap, CUDA 12.8) | 69 | 112.44 | +0.017% ± 0.040, ahead 38/69 | +0.038% ± 0.028, ahead 45/69 |
  | RTX 5070 Ti (300 W, CUDA 13.3) | 74 | 190.44 | +0.021% ± 0.038, ahead 35/74 | +0.006% ± 0.027, ahead 36/74 |

  The trim is inside the twin's spread on both. verify-hits passed 400/400 on both cards with 0 failed, and the hit
  lists matched the base on all four completeness shapes. RTX 40 (sm_89) was not raced; its code changes by the
  same one instruction. Dropped, and the switch is not on the PR: there is nothing left in the hash for the target
  test to skip.
- RTX 20: a smaller transcript hash for the L0 instruction cache (item 11), at most +0.10%.
- RTX 30/40: copy groups and readout fences (item 10); hash control bits, warp turn-taking (#18), IMMAs during the hash;
  a wider window (g6: costs 1,380 cycles a tile to free at most 355); Z6 (no clearing at all saves only 20 cycles);
  f39 (`PEARL_TALL_HASH_FMA`, −3,517 cycles).
- RTX 4090 hash tail (#287 item 1, rig01, 450 W, 6 November). The hash costs 1,397 cycles a tile here (102,778 against
  101,381 with no hash). Every form lost in race mode, 25 rounds, 0 of 25 ahead:
  - chunk 0's stage-1 refill after the chunk, so the partner runs 192 IMMAs before it waits, not 128: −0.61%. The
    knee moved from about 2,600 to 3,300 cycles, but the hash got only 61 cycles cheaper and the moved refill cost 416.
    Below the knee each cycle of hash still costs about 0.4: the partner and then the hasher run alone.
  - the same refill moved into chunk 1, with a guard before chunk 0's readout: +3,150 cycles, ptxas rebuilt the loop.
  - the hash as a `__noinline__` call: the chunk loop stays the same whatever the hash's form, but the call itself
    costs 270–340 cycles: −0.76%.
  - in that call, the adds as IMAD: the hash releases 480 cycles sooner but the tile gains only 40, because the hash
    then takes issue slots from the partner's IMMAs (−0.62%; whole-region hash only, −0.56%). A shorter all-ALU hash
    does pay: dropping a round (timing only) released 280 cycles sooner and saved 316 a tile. The hash has no ALU
    instruction to spare, so nothing is left to try there.
- Tile shape on RTX 30, 40 and 50 (suggestion list item 5, 6 November, no GPU run). The tall fold's 192 accumulators a thread (192x256, eight
  96x64 warps) is the most 255 registers allow at 256 threads. 128 lost three ways: 128x256 on the same ring −4.6% on a
  4090 (probes/README.md), twelve 64x64 warps over 192x256 at 168 registers −2.1 to −5.3% a clock (branch
  exp/rtx30-layout), and on a 5090 the 192x256 tile was +3.4% over 128x256. 160 can't keep whole regions in a warp: a
  region spans 32 rows by 64 columns, so a warp holds 64, 128 or 192 words. A 160x256 tile that splits regions between
  warps moves 11.4% more bytes a MAC than 192x256 and frees registers nothing has used to gain; the model puts it near −2%.
- RTX 30/40 unrolling and prefetch distance (suggestion list item 7, 6 November; compile and records, no GPU run). Inside a
  chunk everything is unrolled, and ptxas already keeps A two m16 tiles ahead and reloads B under the last tile. Left
  to try were the chunk loop unrolled by 2 (`PEARL_TALL_CHUNK_UNROLL`: 470 → 455 instructions a chunk, all waits,
  branches and moves; by 4 it spills) and the A fragment after each copy group loaded before it (`PEARL_TALL_A_AHEAD`:
  same instructions). Both bound under 0.3%: with no hash, removing the ring's waits (33 wait and branch instructions
  a chunk, and half those post-copy-group stalls with them) made the tile 164 cycles slower, and 15 cheap instructions
  are at most 0.7% of a chunk's energy, about +0.2% of clock. A deeper ring doesn't fit (98,688 of 101,376 bytes of
  shared) and a 4 × 64 ring lost 2% on the probe; Blackwell's streamed fragments lost 3.5–5.3% a tile on this ring.
  The two switches stay on exp/gpu10-7, off.
- RTX 50 (item 16): drawing B' on the SM (+1.81% against a +2.5% bar); TMA multicast hangs because its transaction count
  never arrives on sm_120.
- RTX 5090: two CTAs sharing B' by a copy between their shared memories (#287 item 7, 7 November,
  `PEARL_TALL_CLUSTER_PUSH` on branch exp/287-7, CUDA 13.3, 254 registers, no spill). Each CTA loads half of the pair's
  B' box and copies it to the peer (`cp.async.bulk` shared::cta to shared::cluster), so the L2 serves one box a pair. It
  is correct (verify 400/400) but loses on both counts:
  - Energy: `multicast-probe.cu`'s push arms, all held to 7500 cycles a stage at 2425 MHz (PC `380ecfca`), put a
    copied byte at 33.0 pJ and a byte from L2 at 34.1 pJ. The copy saves 2% of an L2 box (fp 0.02, −0.04 to 0.04 over 6
    rounds), against a gate of 50%.
  - Time: in race mode on the same PC, −50.50% ±0.023 against the release (49 rounds, 0 ahead; 197.3 against 398.7
    TH/s). The half gets one stage to land and the copy one more, where the release gives a whole box two. A box takes
    longer than one stage under load, and the ring has no room for a fourth buffer.
  - That PC was clock-held (2467 MHz at 409 W of 540 W): skip B' read −0.26% ±0.032, the release's copy −0.005% ±0.021.
  - At the probe's default pace (3240) the loaded arms can't keep up (a2 4807, h2 3492, p 5962 cycles a stage, PC
    `64002f7c`), so its energies aren't comparable and fp read 3.02 (1.30 to 27.12). On exp/287-7 `multicast-read.js`
    now gives no verdict then and names a pace to rerun at. One high-priority box, about 50 minutes.
- RTX 40/50: making A' and B' on the SM from their compact tables (each row's 128 noise bytes plus one pair table, 16x
  smaller than A'). Built on 29 October (branch exp/newkernel, never merged): producer warps drawing into the ring ran
  80.8% of tensor peak a clock on the 4090 against 95.75%, at 1.68 pJ a MAC against 1.42. The one 255-register layout
  with producer warps (4+4) reached only 93.25% before any draw. Drawing in the fold's own warps costs 1.1–1.6 tensor
  cycles an instruction (E4 on a 3090, item 16's sham draw), so even at half a staged byte's energy a full draw nets
  −1.3 to −3.5%, and a tile's 57 KB of tables don't fit beside the ring (2.7 KB free). Rechecked on 6 November for
  suggestion list item 1, no GPU time.
- RTX 30: a cheaper in-fold A' draw, and a long k (item 7, 10 November, race mode, RTX 3090 `9b482f0a` at 350 W,
  memory 10% busy under the release). `-DPEARL_E4_DRAWT` (branch exp/item7-gen, bench only) holds E4's dense table
  transposed, so four rows of one k index are two loads and one add, and 8 byte selects make the ring's rows: 72
  instructions a chunk a warp against E4's 134, 252 registers, no spill. Against the release (35 rounds, copy of the
  release −0.002%):
  - DRAM ablation +1.75% (1592 → 1618 MHz); E4's data path with no draw +0.88%.
  - E4's draw −3.59% (1587 MHz); the transposed draw −5.57% (1613 MHz, so it saved energy, but did less work a
    clock). Fewer instructions didn't help: its table loads are long-latency and sit in the MMA warps' stream, and
    only 6 of 8 warps draw. There are 3 registers left to load them earlier.
  - Long k, second race on the same PC (24 rounds): k 8192 with stored operands −2.71% (memory 36% busy, 1530 MHz);
    k 16384 with operands in the L2 (DRAM ablation) +2.80%. So on this PC a DRAM-free, long-k kernel could gain at
    most 2.8%, and E4's draw alone costs 4.5-6.4% against its own data path. PeakMiner's +6.6% (item 2) was on a
    3060 whose memory was 39-65% busy.
    The cp.async fold's tile offsets were 32-bit and overflowed past k 16384; item 21 (below) widened them.
  - RTX 3060 ceiling (item 7b, same builds at 1c78804b, nvtest-rtx3060-72, 170 W cap, 16 rounds, twin −0.02%):
    DRAM ablation +2.82% (±0.12, 16 of 16 ahead), memory 28.7% → 0.2% busy, 1777 → 1828 MHz; the whole gain is
    clock, TH/s per MHz unchanged. k 16384 with operands in the L2 +4.05% (±0.17, 16 of 16), 1824 MHz (+1.4% a
    clock from the long k). So the 3060's ceiling is 4.05%, higher than the 3090's 2.80%, but a draw would have to
    cost under 4% to win, and E4's draw cost 4.5-6.4% against its own data path on the 3090. Dropped on the 3060 too.
    This PC's memory was 29% busy, not PeakMiner's 39-65%; a busier 3060 could sit higher.
- Long k on NVIDIA (PR #290 research item 21, 10 November, race mode at `9ac40341`, branch item/21). Every
  competitor mines k 65536; we mine 2048. The tall fold now builds for any k from 2048 to 65536 with
  `-DPEARL_FOLD_K=K` alone (64-bit cp.async offsets past k 16384, PREGUARD off past 2048, and the DRAM ablation on
  the TMA fold too). Operands at the mainnet m and n, so A' and B' grow with k (24 GB at 65536). Against the release:

  | Card | PC | Arm | vs release | Ahead | Memory busy | Clock |
  |---|---|---|---|---|---|---|
  | RTX 5090 (570 W) | a11172a9 | k 8192 stored | −0.46% ±0.10 | 1/24 | 2.3% → 7.0% | 2600 → 2543 MHz |
  | RTX 5090 | a11172a9 | k 16384 stored | −3.53% ±0.11 | 0/24 | 20.8% | 2457 MHz |
  | RTX 5090 | a11172a9 | k 65536 stored | −13.0% ±1.0 | 0/21 | 3.2% → 68.7% | 2623 → 2229 MHz |
  | RTX 5090 | a11172a9 | k 2048, operands in L2 | +0.73% ±0.08 | 24/24 | 0.0% | 2618 MHz |
  | RTX 5090 | a11172a9 | k 16384, operands in L2 | +1.87% ±0.09 | 24/24 | 0.2% | 2594 MHz |
  | RTX 5090 | a11172a9 | k 65536, operands in L2 | +1.33% ±0.05 | 24/24 | 0.4% | 2612 → 2593 MHz |
  | RTX 3060 (185 W) | b1785794 | k 16384 stored | −2.18% ±0.12 | 0/17 | 39.5% → 96.2% | 1904 → 1840 MHz |
  | RTX 4090 (~550 W) | bdcb5688 | k 32768 stored | −29.0% ±0.02 | 0/25 | 3.0% → 100% | 2670 MHz both |

  - **Dropped:** with stored operands long k loses on both cards, more the longer k gets. A launch's B' must stay
    in the L2, so the batch narrows as k grows (2048 → 64 column offsets on the 5090) and all of A' is read once a
    launch: DRAM traffic grows with k, and at the power cap it comes out of the clock.
  - Long k's own gain is about +2.1% a clock on the 5090 (k 16384 and 65536 alike, operands in L2) and +1.2% on the
    3060 (stored). So a kernel that made its operands on chip could gain at most about 1.9% on the 5090, against
    item 8's finding that drawing B' on the SM can't pay there. The 5090's gap to PeakMiner and SRBMiner is not k.
  - **Correctness:** verify-hits passed at k 8192 (400/400), 16384 (400/400) and 65536 (309/309) on the 5090's TMA
    fold, at k 16384 (167/167) on the 3060's cp.async fold, and at k 32768 (400/400) on the 4090's, the first run
    of the 64-bit offsets (B' is 8 GB there). The 4090 at k 32768 is DRAM-bound: its batch narrows to 16 offsets. The k ≤ 16384 cubins for sm_89 and sm_120 are
    byte-identical to before. The copy of the release read +0.01% (5090) and −0.01% (3060).
  - The first 3060 box (609fef99) had another job on its GPU (100% busy, 6 GB in use before we started) and read
    16 TH/s; discarded.
  - Cost: Salad boxes, about 2 hours in all, about $1. A fourth (4090) checked k 32768.
- Shared-memory layout and fragment loads on RTX 40 and 50 (suggestion list item 6, 6 November, no GPU run). The tall fold's 64-byte
  stage rows with unit q at q ^ ((r >> 1) & 3) (SWIZZLE_64B under TMA) give conflict-free `ldmatrix` and `cp.async`.
  It loads only `LDSM.x4`, 40 a chunk a warp, 106.7 shared bytes an IMMA: the least a 96x64 warp tile can read. Its
  address math is 2 IADD and 2 LOP3 a stage on sm_89 and sm_120. Holding k-step 1's lane bases in two more
  registers, with A's base centred, cut sm_89's chunk loop from 540 to 535 instructions (254 registers, no spill).
  That's about 0.23% of a tile's energy, so about +0.07% TH/s. On sm_120 it saves nothing, because the ring roles
  sit in plain registers there. The readout's `red.shared.xor` runs 16 serial passes (8 words on 2 banks, 4 lanes a
  word). Fixing that would save the hash seam at most 50–120 cycles a tile, and it adds about as many ALU instructions.
- Registers and occupancy (suggestion list item 9, 6 November, compiled code only, no GPU). No shipped fold spills anywhere: 0 bytes of stack on
  every NVIDIA kernel and arch, 0 local loads or stores in the SASS, 0 scratch on every AMD target. The tall fold
  (254-255 registers, 98.7-99.8 KB of shared) is held to one block an SM by both. ptxas
  `--register-usage-level` 8 and 10 leave the sm_89 and sm_86 chunk loop byte for byte the same (0 and 2 make it
  longer). RDNA 3 sits at 240 VGPRs, the 6-wave cap, and its LDS allows 3 blocks anyway. RDNA 4 used 201 VGPRs (213 since #287 item 4) and could
  use 240 at the 6 waves it runs, but with the compiler told so (`-DPEARL_AMD_WAVES=6,6`) the slab loop doesn't change
  and the readout gains a move. Nothing to race.
- RDNA 3's readout as three XOR chains a region (`PEARL_AMD_RDTREE11`, PR #290 item 23, 10 November): −0.18 to
  −0.52% in `hashrate.js` and −0.25 to −0.59% in the fold alone on an RX 7900 XTX (`44d3716f`), behind in every round.
  Removed from the code. The hash on its own queue (`PEARL_AMD_HASHQ`) on RDNA 4: −0.21 to −0.36% on an RX 9060 XT
  (6 of 6 rounds); on RX 7800 XTs level or −0.03 to −0.27%. It is on for gfx1100 only (see the AMD section).
- RDNA 3 (item 13): k 4096 gained under +0.3%. Batch 128 shipped on gfx1100 and gfx1101 (+0.1 to +1.8% a round, and 64
  was slower on both chips); RDNA 4 was level at 128 and keeps 256.
- RDNA 3 load scheduling (#287 item 8, 7 November): the loads pinned two slabs ahead (`PEARL_AMD_PF2PIN`, an empty
  asm with a memory clobber keeps them at the top of the step), alone +0.05%, +0.42% and +0.25% fold alone on three
  PCs and with CRANK and SCHEDB 0.2% behind the same without the pin; CRANK and SCHEDB without PFX, −1.22%, since
  three of a pair's six loads then wait at the barrier with no WMMA to hide them. The set that shipped is above.
- AMD fold: two-lane readout regions and readout v2 (items 2 and 4); start skew (item 14); the LDS store a k-step
  earlier (item 15); `PEARL_AMD_SADDR`; a 64-byte slab; a fold with no LDS; operand fill above 48; 8 waves a SIMD; a
  split slab barrier; the keeping lane by exec mask (below RDSEL alone); batch-width and hash-kernel changes.
- k 8192 pays only on gfx1201 with 56+ CUs (+0.70% on a 9070 XT, 4 November); the 9060 XT is about 6% slower at it.
  HIP counts WGPs on RDNA (32 on a 64-CU card), so the gate takes either count.
- Unsigned or biased int8 operands (suggestion list item 2, 6 November, desk study, no card time). Fill 63 already gives nonnegative A' and
  B' (the noise is triangular on [−63, 63]) with the same bits a u8 mma would see, and it measured level with 48 or
  just below it: −0.23% on the 4090, +0.13% fold alone on a 9070 XT. The four int8 `mma.sync` type pairs compile to
  the same IMMA, registers and schedule on sm_89 and sm_86; the types are two mode bits. Any bias the operand doesn't
  already carry has to come off every accumulator before the XOR readout: one IADD per IMMA, about +8.8% energy an
  IMMA (0.51 against 5.81 nJ), or about −2.6% TH/s. A bit-flip model fitted to the fill sweep puts every bias at or
  below fill 48.
- Folding fewer k-elements a chunk through the noise's pair graph (#290 item 11, 10 November, CPU study, no card
  time). B' column t is fill + E[c][p0(t)] − E[c][p1(t)], so a chunk's 128 pairs are edges of a graph on 128
  vertices, and with fill 0 every cycle is an exact linear dependency: drop one edge, add ±its A' column to the
  cycle's other columns, and the chunk's int32 sums don't change. On real pair tables (`generatePermutationMatrix`,
  24 jobs at k 2048 and 6 at k 8192, 768 chunks) the dependencies number 21.5 a chunk (14–31). But A' is ±63, so a
  kept column can take only one dropped column's A' and stay int8, which makes the usable count a maximum
  edge-disjoint cycle packing. Solved exactly (ILP on the contracted 2-core, with a proven upper bound), it is 8.7 a
  chunk (5–15), and 16 in none of 768 chunks; a job's worst chunk is 5–8. Checked on real noise: 288 chunks gave the
  same sums with A'' within ±123. So nothing on any WMMA or `mma` fold: RDNA 3/4 and RTX 20 need 16 dropped in a
  chunk, Ada and later 32. Only the dp4a fold (GTX 10 and 16) could use it: +6.0% fewer dp4a with a per-chunk trip
  count, +3.2–3.5% with one count a job. That needs A fill 0 (B' fixed, A'' redrawn per stamp), and fill 48 was
  worth +3.6% on the 4090 by power, so the net is near zero on cards nobody rents. With fill 48 kept on B only
  balanced cycles work (3.7 a chunk, +1.8% on dp4a at best), and directed cycles for A fill 48 give 1.4. Dropped.
- Operand fill 64 and higher (research item 3, 10 November, desk study). The spec's `verify_plain_proof`
  (pearl-research-labs/pearl, `zk-pow/src/api/verify.rs`) takes A and B values in [-64, 64], "matching the ZK
  circuit's IRANGE7P1 range check". So 64 is legal and 65 and up are refused by every pool ("Matrix value out of
  range"). The 127 that measured +5.9% (probes/README.md, "Operand values") is out of reach: its gain came from
  saturation, 127 plus any positive noise clamping to 127, and A + noise can't saturate once A is at most 64. The
  fill changes no share's credit: the target and the work a share stands for (tile * k) don't read it. At 64, A'
  sits in [1, 127], so like 63 it never flips the sign bit; the bit-flip model gives both 3.500 toggles a byte
  (48: 3.489) and 64 a popcount of 3.508 against 63's 3.492. 63 measured level with 48 (-0.23% on the 4090), so
  64 is expected level too. Raced on Salad, 10 November (`race` stage, base = fill 48, an A/A twin, turns of 4 s
  for 600 s; builds 3df81e61 and b120b72e on branch item/3, which open `PEARL_OPERAND_FILL` and checkHit.js to
  [-64, 64]):

  | PC | rounds | aa (twin) | fill 63 | fill 64 |
  |---|---|---|---|---|
  | RTX 3080, 104.1 TH/s | 36 | −0.00% ±0.48 | +0.04% ±0.55 | −0.39% ±0.81 |
  | RTX 4080 SUPER, 208.0 TH/s | 37 | −0.01% ±0.02 | −0.21% ±0.02, led 0/37 | −0.29% ±0.02, led 0/37 |
  | RTX 5080, 212.3 TH/s | 35 | −0.00% ±0.01 | −0.03% ±0.03 | −0.02% ±0.04 |

  (± is the 95% interval.) verify-hits passed 400 of 400 for fill 63 and 64 on all three. Fill 64 is level on the
  3080 and 5080 and a clear −0.29% on the 4080 SUPER; the 5080 drew 2 W more at 63 and 64. Fill 48 stays, the
  [-63, 63] limit stays in the code, and nothing above 64 is legal. Dropped.
- Our AMD work carried over to NVIDIA (PR #290 research item 22, 10 November). Every AMD change that gained speed
  (git log on `earn/native/amd`, the AMD sections above), checked against the NVIDIA folds (the tall fold: TMA on
  sm_120, cp.async on sm_89 and sm_86) and the sm_86 SASS of the release's chunk loop (192 IMMA, 40 LDSM, 117 LOP3,
  14 LDGSTS, ~100 other instructions a chunk a warp). The AMD gains came from two things NVIDIA doesn't share:
  on RDNA 4 a VALU next to the WMMAs costs about a WMMA cycle (3.5-5 chained on RDNA 3), while an NVIDIA scheduler
  issues ALU work beside an IMMA that holds the tensor pipe ~16 cycles, so there an instruction costs only energy;
  and AMD's fold started further from its peak (70% of WMMA on RDNA 4 in v0.6.11). By what each would give NVIDIA:
  - Already in the NVIDIA folds, most of them first: the per-tile operand layout (AMD took
    `pearl_materialize16_tiled` from the CUDA fold), loads kept in flight across the stage barrier (PF2, PFX: the
    3-stage cp.async/TMA ring; its waits cost 0.46% a tile, all at the hash seam), one fragment load for two
    k-steps (KPAIR: `ldmatrix.x4` is already 16 bytes a lane, the least a 96x64 warp tile can read), rank at compile
    time (CRANK: no division in the chunk loop), the readout as a tree (RDTREE: 3-input LOP3 XORs), DPP lane moves
    (shuffles and `red.shared.xor`), launch bounds (TID), the tile walk and batch width sized to the L2 (WALK and
    RDNA 3's batch 128: Ampere's band, Ada's and Blackwell's width rules), and the hash inside the fold.
  - Already tried on NVIDIA and dropped: the readout moved into the next chunk (RDLATE: −2.0% in September; tiles 4
    and 5 moved, +265-293 cycles a tile), pinned lane bases (LDS32_PIN: +0.07% by estimate), k 4096 (long k is
    item 21).
  - Not built: fewer address instructions for the cp.async copies (BUFLD). On sm_86 a stage spends about 9 on
    addresses, 6 of them for A's three copies (64-bit adds); about 8 a chunk could go, ~0.3% of a chunk's energy,
    so about +0.1%. It sits in `tile_srcs`, which item 21 is widening for long k, so it waits for that.
  - Dropped by estimate: the hash in its own kernel, AMD's default shape and what GA100 ships
    (`PEARL_TALL_UNFUSED`, +2.2-4.8% there, where the fused hash cost 14% of a tile). On RTX 30/40 the
    round trip is 64 bytes a region (2^19 MACs): on a 3090 about 2% of a batch to read it back, plus more DRAM
    traffic on a card already 18-30% busy, against about 0.65% of TH/s the fused hash costs at the power cap.
    Net about −1.5%.
  - Raced: RDNA 4's tile order (WALK, every other row of supertiles backwards) as the serpentine walk on RTX 50
    (`-DPEARL_FOLD_SERPENTINE=1`; flag only, no spill under ptxas 13.3, 8 more instructions in the fold). Measured
    before only on a 5060 (+2.3% at width 2048, nothing at the 512 it runs). RTX 5090 (`8d8f5bf3`, 480 W cap,
    driver 621.14, CUDA 13.3, width 2048, race mode, 74 rounds of 4 s): aa +0.262% (±0.092, a base-position offset,
    ahead 53/74), serpentine +0.287% (±0.108, 53/74), so +0.03% against the twin: level. Memory busy 4.7 → 4.0%,
    clock +3 MHz over the twin. verify-hits 400 of 400. Nothing to ship. Mid RTX 50 cards run width 1024 or 512,
    where B' fits their L2 like the 5060's 512, so it wasn't raced there. Cost: one 5090 box, 19 minutes.
  So nothing from the AMD side is left to give NVIDIA more than about 0.1%. The NVIDIA gap is the DRAM-free,
  long-k kernel (items 2, 7, 21), which the AMD core doesn't have either.
- The hash in the fold on gfx1201 (#287 item 6): −0.50% on an RX 9070 XT at k 8192 (a68b3382, both scored rounds, copies
  within 0.06%). RDNA 3 hasn't raced it; its hash kernel is a smaller share (0.8% on a 7900 XTX) and its DRAM faster.
- Stale shares (PR #290 item 9, 10 November, pool data only). HeroMiners' lifetime counts for our wallet: 121 stale and
  10 invalid of 83,125 shares (0.15% and 0.01%). Kryptex's worker counts the same day: our 35 workers 13 stale of
  8,387 (0.16%), 0 invalid; one RX 9060 XT box had 4 of 58, the rest 0–2. PeakMiner, SRBMiner and OneZeroMiner had
  0 stale on 255–304 shares each, too few to tell 0.15% from 0. So stale shares cost at most 0.15%. The miner also
  drops a hit for a job the pool has replaced (`pearlMiner.js` `_onHit`) without counting it; a 4090 batch is about
  56 ms against jobs 35–60 s apart, so with two batches queued that is at most about 0.25%, and only if the pool
  would still take the old job, which the 0.15% stale suggests it doesn't. Not worth a change.
- Power steadiness (PR #290 item 9, 10 November, from the 9–10 November bench samples, one every 15 s after the first
  minute, 17 a miner). Every miner holds the power limit: on an RTX 2060, 3060 and 5060 PC the power's standard
  deviation was 0.02–0.94 W for all four miners, ours included, and our clock's 4.7–14.5 MHz against 3.0–15.5 for
  the others. At the same mean power we ran 1822 MHz against 1899–1904 (3060) and 2510 against 2574–2575 (5060), so
  the gap is energy a MAC (items 1, 4, 6, 12 and 13), not a power draw that swings. The 3080 Ti PC (`c09c68c3`)
  swung for SRBMiner and OneZeroMiner (sd 21 and 41 W) and held for ours and PeakMiner's (6 and 0.4 W). Sampled
  every 100 ms for 2 minutes, release v0.6.14 in the app on rig01's RTX 4090 drew 449.3 W (sd 3.5, 420–461) at
  2520–2550 MHz, power-capped in all 1,122 samples. So it doesn't swing under a second either.
- RTX 20 tuning (PR #290 item 9, 10 November, from records). Seven rounds in October (`earn/native/probes/README.md`,
  "RTX 20-series") tried compilers, launch shapes, hand-scheduled SASS, readout and hash forms, k order, a 192x256
  tile and cubin patches; what paid shipped. The 9–10 November run has us at 97–100% (median 98%) of PeakMiner on
  RTX 20, the smallest gap of any NVIDIA series. Item 14 (sm_75 with CUDA 13's ptxas) is the one idea left, and
  October's round 1 found ptxas 12.4–13.4 build the sm_75 fold to the same bytes. Not worth more time now.
- CMP 90HX (PR #290 item 9, 10 November, from records). On Clore on 4 November it ran 72 TH/s at 200 W on 50 SMs at
  about 1480 MHz: about 975 MAC an SM a clock, the same as an RTX 3080 on Salad (113 TH/s, 68 SMs). So its IMMA isn't
  cut down and it runs the RTX 30 fold like any GA102; the RTX 30 items cover it. CMP 170HX is 99% of best (Vast,
  October). Clore also lists about 350 CMP 50HX (TU102) and 40 CMP 40HX (TU106), which we have never run; neither
  is on Vast or Salad, and the business Clore account is empty, so testing them needs funding.
- B200 (PR #290 item 9, 10 November). Still 33% of best (`mma.sync` tops out near 570 TH/s; competitors 1,380 on
  tcgen05). A tcgen05 fold is a new kernel, built and tested on a card that rents for $3 or more an hour, and at
  1,380 TH/s a B200 earns about $1.07 an hour, so nobody rents one to mine. Not worth it until someone mines on
  their own B200s.

## Keeping this file current

- A change that re-measures a host here updates its row in the same PR. A number only in chat or a PR doesn't count.
- Only a run of our miner on the row's own machine replaces "Ours" and "% of best". Competitor figures stay until the
  next full rerun; a new host runs every miner.
- A run on another host or by another method doesn't change a row: record it as a new table or a research note.
- Before tagging a release, check that the intro and the Summary name the right release.
