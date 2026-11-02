# RTX benchmark

Our miner, v0.5.14, against PeakMiner 2.18.3 and SRBMiner 3.7.4 on rented
Vast hosts, one or two hosts per card. Every figure in the card tables comes
from one full rerun on 2026-10-10, 15:58–21:05 UTC, except six RTX 20-series
rows rerun on 2026-10-11 against SRBMiner 3.7.5. Each host runs all three
miners one after another on the same pool, so rates within a row compare
directly. Don't compare across rows: power limits and cooling differ from
host to host. v0.6.0 adds Monero mining on the CPU and leaves the GPU miner
as it was in v0.5.14, so these figures hold for it too. v0.6.8 adds a core for
GTX 10 cards (Pascal, below) and reads a hit's proof with one wait instead of
about 120; the new kernel code builds only for sm_61, and every other card's
kernels are as they were, so the figures hold for it as well. v0.6.8 also ships
a core for AMD cards on Linux (`pearl_core_hip.node`, loaded with `--core amd`);
its figures are under "AMD Radeon". On 2026-10-28 the RTX 20- and 30-series were
run again on Salad PCs with v0.6.9 against the newest competitors, PeakMiner
2.20.3 and SRBMiner 3.8.0. Those tables are under "On Salad, 28 October" in
their sections and leave the Vast rows alone. On 29–30 October the RTX 40- and
50-series were run the same way (under "On Salad, 29–30 October"), and the AMD
rows were rerun with the v0.6.9 release against SRBMiner 3.8.0. On 30 October
the RTX 20- and 50-series were run once more against SRBMiner 3.8.2 (under "On
Salad, 30 October, against SRBMiner 3.8.2").

## Summary

"% of best" is our rate divided by the faster of PeakMiner and SRBMiner on
the same host. 92 of the 95 rows ran. The H200 and H200 NVL weren't run
because one pass cost more than the budget left. The Vietnam L40 wasn't run
because no L40 was for rent other than the New York one.

- RTX 20-series: 92–99%. Our clock is never more than 30 MHz below the
  faster competitor's, so the gap is work per clock: we do 5–7.5% less.
  SRBMiner 3.7.5 also mines Monero on the CPU alongside, and so does ours
  from v0.6.0; PeakMiner doesn't (see the section). On Salad on 28 October,
  v0.6.9 read 97–103% (median 98%) against PeakMiner 2.20.3 and SRBMiner
  3.8.0: our RTX 20 work since v0.5.14 closed the gap. Against SRBMiner 3.8.2
  on 30 October it read 95.7–99.7% (median 97.4%), with PeakMiner fastest on
  every PC.
- RTX 30-series: 97–101%. The two 3070s trail most, at 97%. SRBMiner didn't
  start on the Oman 3090 Ti, so that row compares with PeakMiner. On Salad on
  28 October, v0.6.9 read 91–97% (median 92.5%) against PeakMiner 2.20.3 and
  SRBMiner 3.8.0, on every card: both competitors got faster here, and this is
  now our weakest series.
- RTX 40-series: 98–102%, and 99.5% or better on 15 of the 18 hosts. The
  lowest are the 225 W British Columbia 4080 and the Australia 4060, both at
  98%. On Salad on 29–30 October, v0.6.9 read 93.5–95.8% (median 94.5%)
  against PeakMiner 2.20.3 and SRBMiner 3.8.0, on every card: as on the RTX
  30, the competitors got faster and ours didn't.
- RTX 50-series: 96–101%. The India 5090 (96%) has driver 570, so it runs
  the CUDA 12.8 core. Per clock it does 3.4% less than SRBMiner there,
  against 0.6% less on the South Korea 5090 on the CUDA 13 core. Those are
  two different hosts, so this doesn't prove the core is the cause. On Salad
  on 29–30 October, v0.6.9 read 94.6–96.4% on 10 of the 12 PCs against
  PeakMiner 2.20.3 and SRBMiner 3.8.0. The 450 W 5090 read 89% and one 5060
  101%. Against SRBMiner 3.8.2 on 30 October: 87.6–103.2%, median 95.2%. The
  437 W 5090 read 87.6%, and a 145 W 5060 beat both competitors (103.2%).
- A100-class: 97–111%. The A100s are at 97–98% and the CMP 170HX at 99%. The
  A30's 111% is too high: its host lowered the power limit during
  PeakMiner's and SRBMiner's runs. Against PeakMiner at the full limit it's
  about 107%.
- Hopper: 98–99% on the H100 SXM, NVL and PCIe.
- Ada workstation and data-center cards: 96–100%. The RTX 6000 Ada trails,
  at 96% on both hosts.
- RTX PRO Blackwell: 95–100%. The RTX PRO 6000 Workstation is at 95%; the
  rest are at 98% or better.
- Ampere workstation and data-center cards: 89–100%. The RTX A4000s (93–94%)
  and A2000s (89–91%) trail most.
- B200: 33%. Its hits are correct, but it mines on `mma.sync`, which tops
  out at about 41% of the competitors on this card. Matching them needs a
  fold on the B200's own tensor instructions (tcgen05).
- GTX 10-series: 0 for PeakMiner and SRBMiner. Pearl needs RTX 20xx or newer
  (PeakMiner's docs), and both got nothing on a GTX 1080 on Salad. Our release
  has a Pascal build from v0.6.8; before the release it mined at 17.8 TH/s on a
  GTX 1080 Ti on Salad (the section's notes). The release hasn't been run on
  these cards yet.
- GTX 16-series: 0.7–1.3 TH/s for all three miners (96–104%). The cards have no tensor cores, and
  every miner's tensor instructions run on a slow shared unit. Not worth mining on Salad.
- AMD Radeon on Salad: v0.6.9's AMD core (`--core amd`) against SRBMiner 3.8.0 on the same PCs on 30 October,
  two PCs a card: 109–110% on the RX 9060 XT, 100–103% on the RX 9070 XT, 99–103% on the RX 7900 XT, 97–101% on
  the RX 7800 XT and 92–99% on the RX 7900 XTX. On 23 October, against SRBMiner 3.7.1, ours led on every card
  (102–148%). SRBMiner 3.8.0 is much faster on RDNA 4 and about the same on RDNA 3 (the section's notes).
  PeakMiner doesn't mine Pearl on AMD (0).
- Tesla T4: 91%. As on the 20-series, the clock is close (15 MHz below the
  competitors') and we do 7% less work per clock.

**Why we trail where we do.** On most hosts every miner runs at the card's
power limit. On most of the Ampere (sm_86) and Ada (sm_89) hosts, and on the
RTX PRO 6000 Workstation, SRBMiner 3.7.4 holds a higher SM clock than ours
at that same wattage, while per clock the two do about the same work. So
where SRBMiner leads on these cards, it leads by the clock. The gap is
widest on the RTX A2000s (1282 against our 1147 MHz, and 1200 against
1102), the RTX A4000s and the RTX 6000 Ada (90 MHz more on each host), the
RTX PRO 6000 Workstation (2287 against 2190), and the RTX 3070s, the Italy
RTX A5000 and the Belgium A40 (60 MHz more on each). Per clock, the two are
within 2% on every one of these cards. On the other Blackwell (sm_120)
hosts, SRBMiner's clock is at most 40 MHz above ours, and its 4% lead on the
India 5090 is work per clock. On the RTX 20-series and the T4 the clocks are
about the same, and on the A100s and H100s ours is the same or higher, so
the gap there is work per clock.

## How a host is tested

1. **Files.** Each box downloads the published v0.5.14 release files, the
   same files a user gets: the CLI, `pearl_core.node` and
   `pearl_core_cu13.node`. It checks each one against the sha256 digest
   published with the release. All three matched on every box. PeakMiner
   2.18.3 and SRBMiner 3.7.4 come from their own GitHub releases and run at
   stock; the versions here are what their APIs reported. On a compute 12.0
   card (RTX 50 and RTX PRO Blackwell) with driver 580 or newer, the CLI
   loads the CUDA 13 core, as a user's rig would. Everywhere else it loads
   the CUDA 12.8 core.
2. **Hit check.** `earn/native/probes/verify-hits.js`, from the v0.5.14 tag,
   runs the core the CLI loads for 90 s and recomputes the first 400 hits
   from scratch the way the pool's verifier would. Every one must match. It
   checks that the answers are right, not the speed. It isn't a column in
   the tables; every row passed. The H100 NVL's check ran at 15:45 UTC on
   an earlier rental of the same machine, on a different H100 NVL. Our test
   script then hung, and the rental that ran the timed runs didn't repeat
   the check. Three cards are too slow to make 400 hits in 90 s, so the
   check ran longer:
   - RTX A2000 12 GB, Poland: 258 hits in 90 s, all correct. A 161 s run
     found 481, and the first 400 matched.
   - RTX A2000 6 GB, Japan: 248 hits in 90 s, all correct. A 167 s run found
     476, and the first 400 matched.
   - Tesla T4: the 90 s check on the measured card matched all 264 of its
     hits. A 180 s check matched the first 400 of 508, but it ran on another
     T4 in the same machine, with the same driver and VBIOS.
3. **Runs.** After a 60 s warm-up that isn't measured, our miner, PeakMiner
   and SRBMiner each mine for 5 minutes on `us.pearl.herominers.com`, back
   to back on the same card. The order alternates between hosts (ours first,
   or SRBMiner first), because whoever runs first gets the coolest card.
   nvidia-smi samples the SM clock, power, temperature and use every 5 s. On
   a multi-GPU rental every miner is pinned to GPU 0, and the other card is
   checked to stay idle. It did on all four such rows.

Rates are in TH/s. For our miner it's the mean of every reading after the
first minute; each reading is the average since the one before. For
PeakMiner and SRBMiner it's the miner's own figure, from its API, at the end
of its 5 minutes. A competitor that didn't run has the reason in its cell,
and the row compares with the other one. Clocks and power in the notes are
medians of the 5 s samples after the first minute, unless a note says
otherwise.

"% of best" is computed from the unrounded rates, so dividing the table's
rounded rates can differ in the last place. It's shown as a whole percent,
or to one decimal from 99.5% to just under 100%, so a row that's behind
never shows as 100%.

After the run, every row was checked against the logs and files its host
sent back: our mean recomputed from the readings, the competitors' figures
from their API files, the hit check, the sha256 matches, the versions, the
GPU and the power limit. The check also read each run's first nvidia-smi
sample, taken before the miner starts hashing, to confirm the card was idle
and the runs didn't overlap. That is how the Spain 3090 Ti run was caught
(see the 30-series notes). Our miner had no share rejected on any row.

Vast's log API cuts every line at 500 characters. Some boxes sent their raw
files back as longer lines, so on nine rows the raw-file archive was lost:
the Ukraine 3090, the Portugal 3080 Ti, the RTX 4090D, the three A100s, and
the H100 SXM, NVL and PCIe. Their figures come from the box's own result
line, which the box computed from the same raw files before sending them.
For these rows the final check confirmed the result line against the box
log's per-minute lines (each miner's own rate and one nvidia-smi sample a
minute), and confirmed the hit check, sha256 matches, versions and run order
from the box log. It could not recompute our mean from single readings, read
the 5 s nvidia-smi samples, or see the sample at the start of each run that
would show another workload on the card. On the 4090D the cut also fell
inside SRBMiner's figure, so it's only known to be 277.30–277.39. That row
is at 99.7–99.8% either way.

A re-measure is a pool run of only our miner, as above, on the row's own
machine; the row keeps its PeakMiner and SRBMiner figures. A speed test
isn't a re-measure: `hashrate.js` runs the core alone, with no pool. Speed
tests go in the section notes and never change a row.

## How hosts are picked

- The driver supports CUDA 12.8, which our miner needs.
- Reliability is 90% or higher.
- The power limit is at least 80% of the card's stock limit, so each card is
  measured close to how it's sold.
- Vast's performance score is at least half the median for that card. This
  screens out fake hosts, like the Texas "4090s" that ran at 38 TH/s.
- Not in mainland China. Each box downloads from GitHub and mines on a US
  pool.
- Cheapest first. A host we've already tested wins when it's listed. The
  second host for a card comes from a different operator.
- One GPU per rental where possible.

This run kept each row on its own machine when that machine could be rented.
Otherwise the row moved to the cheapest verified on-demand offer of the same
GPU (and memory size, where the row names one) that met these rules, from a
different operator than the card's other host. 54 rows ran on their own
machine and 38 on new hosts. A host that couldn't run the test was swapped
for the next one (see "Hosts replaced in this run").

Exceptions:

- Own machines stay even where they no longer meet the rules. The 20-series
  hosts were picked before the power and performance-score rules existed, so
  the Thailand 2080 Ti (170 W of its 260 W default) is on the list. The
  India 5090 (151861) shows 58% reliability on Vast.
- The British Columbia RTX 4080 (43318) runs at 225 W, 70% of the card's
  320 W, and has 32 GB cards. Nevada (147894) couldn't be rented, and this
  was the only other verified RTX 4080 offer.
- The France RTX 4070 Ti (138259) isn't verified on Vast. No verified 4070
  Ti offer was left.
- Four rows are on 2-GPU boxes: the Florida 4080 Super and the Mississippi
  RTX PRO 6000 Workstation (both their own machines), the British Columbia
  4080 and the Belgium A40.

## Keeping this file current

- A change that re-measures a host in this file updates its row in the same
  PR. A number that only appears in chat or in a PR description doesn't
  count as recorded.
- Only a run of our miner on the row's own host (same machine ID) replaces
  its "Ours" and "% of best". Give the earlier figure in that section's
  notes, with the date and the commit that ran.
- PeakMiner and SRBMiner figures are fixed targets until the next full
  rerun. Don't rerun them for a row that has them. A new host runs all three
  miners.
- A run on a different host, or by a different method than "How a host is
  tested", goes in the section's notes and leaves the row alone.
- When a row changes, check that the Summary and the section heading still
  match it.
- Before tagging a release, check that the top paragraph, the Summary and
  "How a host is tested" name the right release, and that no section calls
  a shipped card unreleased.

The 2026-10-10 run on the v0.5.14 release reran all three miners on 92 of
the 95 rows, so it replaced PeakMiner's and SRBMiner's figures as well as
ours, and it moved rows to new hosts where their machine couldn't be
rented. The three rows it couldn't run have their last figures in their
section notes. The notes don't give the earlier figures of the rows that
ran; those are in git history. This run is the baseline for this file. From
here the rules above apply again.

On 2026-10-11 the founder asked for the RTX 20-series rows to be rerun
against the newest SRBMiner (3.7.5), with its Monero figures. That rerun
replaced all three miners' figures on the six rows that ran on their own
machine. The other sections still compare with SRBMiner 3.7.4.

## RTX 20-series

Six rows were rerun on 2026-10-11, 07:21–07:56 UTC, on their own machines,
against PeakMiner 2.18.3 and SRBMiner 3.7.5. The Pennsylvania 2080 Ti and
the Colorado 2080 couldn't be rerun, so they keep their 2026-10-10 figures
against SRBMiner 3.7.4.

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

- **2026-10-10 figures of the rerun rows** (ours, PeakMiner, SRBMiner 3.7.4):
  Thailand 67.8, 71.5, 72.2 (94%); Alberta 60.1, 62.1, 64.3 (93%); South
  Korea 2070 47.4, 49.7, 50.6 (94%); Germany 42.5, 44.8, 46.1 (92%);
  Australia 48.0, 50.4, 49.1 (95%); South Korea 2060 49.9, 53.8, 53.2 (93%).
  Ours moved by half a TH/s or less on every host except Thailand.
- **RTX 2080 Ti, Thailand:** the host held the card at 975–1020 MHz and
  165 W of its 170 W limit, against 1095 MHz and about 155 W the day before,
  so all three miners read 6–11% lower. The competitors lost more than we
  did, which is why the row moved from 94% to 99%.
- **RTX 2080 Ti, Pennsylvania:** its machine wasn't for rent from 07:23 to
  08:24 UTC. The stand-in, Florida (47144, 260 W), was held at 84–85 °C at
  about 155 W and 1,250–1,300 MHz, with a fan reading of 126%, so it isn't
  used: ours 39.8, PeakMiner 40.6, SRBMiner 32.8 (one low end reading; 40.9
  on average after the first minute).
- **RTX 2080, Colorado:** its machine wasn't for rent at any point from
  07:23 to 09:00 UTC, and Vast had no other RTX 2080 for rent.
- **RTX 2060 Super:** the card is held at 76–78 °C, 118–136 W of its 175 W
  limit, at 1380–1425 MHz, below the card's 1470 MHz base clock, for every
  miner.
- **RTX 2060, Australia:** our miner found no pool share in its 5 minutes
  (about 1.6 were expected). Its hit check passed 400 of 400.
- **Speed tests, 2026-10-20 (not re-measures):** a launch sweep (hashrate.js,
  no pool) on an RTX 2060 12 GB, an RTX 2070 and the Colorado 2080's own
  machine found one gain. A band of 5 column groups instead of 4 was +0.30%
  on the 2070 and +0.21% on the 2080 (both 4 MB L2; 77.76 → 77.92 TH/s), and
  -0.70% on the 2060 (3 MB). This table's rows ran v0.5.14, which mines 4 MB
  cards at band 4; the core after v0.6.6 uses 5 on them. That is the 2060
  Super, 2070 Super, 2070 and 2080 rows here, and the T4. Details are in
  earn/native/probes/README.md.
- **Fold speed tests, 2026-10-20 (not re-measures):** three fold changes,
  each raced with hashrate.js (no pool) against the build before it, 3
  rounds of 60 s, on a 2080 Ti, the South Korea 2070 row's own machine
  (139007) and a South Korea 2060 12 GB. A-direct, which loads A the way B
  is loaded, on a Vietnam 2080 Ti (152869) and the 2060 142755: 82.75 →
  84.75, 46.30 → 48.01 and 47.00 → 47.88 TH/s. Then k-lo first, which
  reorders the IMMA, on the same three: 85.22 → 86.81, 48.04 → 48.82 and
  47.91 → 48.97 TH/s. Then the lean loads, on the Thailand row's own 2080
  Ti (95392, held by its host at 167 W) and the 2060 27564: +0.2%, 50.75 →
  51.32 and 49.68 → 50.41 TH/s. Together that is about +4.6%, +6.6% and
  +5.7%. All three are in the core after v0.6.6. Details are in
  earn/native/probes/README.md.
- **Pool runs of the branch core, 2026-10-21 (not re-measures):** the
  core after v0.6.6, built from its commit rather than release files, on
  three rows' own machines, 00:31-00:41 UTC, by "How a host is tested"
  otherwise: verify-hits 400/400, a 60 s warm-up, then 5 minutes on
  us.pearl.herominers.com, ours only (probes/turing/pool.sh). Pennsylvania
  2080 Ti 94.0 TH/s (1470 MHz, 255 W, 68 °C), 100.9% of the row's best;
  Colorado 2080 82.1 (1875 MHz, 273 W, 53 °C), 99.2%; South Korea 2070
  50.8 (1515 MHz, 146 W, 65 °C), 98.3%. On v0.5.14 these rows read 89.0,
  77.6 and 47.8. The same 2070 ran at 1455-1515 MHz across the 20th and
  21st's rentals, so one run moves by a few percent with the card's heat.
  These ran before the second fold kernel below.
- **Second fold kernel, 2026-10-21 (not re-measures):** the core after
  v0.6.6 runs a second RTX 20 fold kernel, in which the second row group
  hashes, on cards with 4 MB of L2 or more, and the first on the rest.
  hashrate.js (no pool), 5 rounds of 60 s, against the same build forced to
  the first kernel: the Colorado 2080 row's own machine (149439) 81.83 →
  81.98 TH/s (+0.17%), a South Korea 2070 (148606) +0.19% and a Vietnam
  2060 Super (26278) +0.18%; the South Korea 2060 row's own machine (27568,
  3 MB) runs the first kernel either way, 54.12 → 54.12. That is the 2060
  Super, 2070 Super, 2070, 2080 and 2080 Ti rows here, and the T4. Details
  are in earn/native/probes/README.md.

### On Salad, 28 October, against the newest miners

Run from 17:04 UTC on 2026-10-28 to 00:12 UTC on the 29th, as the founder asked, on Salad PCs (home computers
that run containers under WSL2), each named by the first 8 characters of its Salad machine ID. Our release v0.6.9
ran against PeakMiner 2.20.3 and SRBMiner 3.8.0, the newest of each, all three checked against their published
checksums. Each PC ran them one after another, 5 minutes each after a 60-second warm-up of the first, on
`us.pearl.herominers.com`. A card's first PC ran ours first and its second SRBMiner first, with PeakMiner second
both times (`tuning/salad/bench.py`, `BENCH_SET=rtx`, `bench.sh` at `461026a`). There was no hit check: the
release's cores ran as users get them, and the pool rejected none of our shares. These are different PCs, tested
by a different method, so they leave the rows above alone ("Keeping this file current"): compare "% of best", not
TH/s. Salad's 2070 and 2060 classes also handed out 2070 Supers and 2060 Supers, which got rows of their own. It
has no 12 GB 2060 class.

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

- **Within 3% of the best on every PC.** The median is 98% (97–103%), up from 94% on the Vast rows. The gain is
  our core after v0.6.6 (the notes above): against SRBMiner 3.8.0, which looks unchanged on these cards since
  3.7.5, ours moved from about 95% to about 101%. PeakMiner 2.20.3 gained about 3% and is now the faster
  competitor on 11 of the 12 PCs. At the same clock it still does 1.2–3.3% more per MHz than ours on the 2080,
  2070, 2070 Super, 2060 Super and 2060.
- **The rows above 100% are each one unusual PC.**
  - ec9e6c87 (2080 Ti) swung between 207 and 266 W and 1515–1725 MHz for every miner, so all three rates are
    noisy. It reads 104.5% on all three miners' means.
  - f765012b (2080) is capped at 129 W of its 215 W default. It reads 99.2% against PeakMiner's mean.
  - 0416482b (2070) ran ours at 1575 MHz and the competitors at 1635–1650 MHz, at the same 179 W, and ours still
    hashed 3% more. The lines don't say why.
- **Held by heat:** 83f7bc7e, 2d10f218, 413b2e3d and ab1e6b0e sat at 85–88 °C below their power limits. All three
  miners ran at the same clocks there, so their ratios hold.
- **Our 2080 Ti readout is coarse.** On ef08a260, with power, clock and temperature flat and PeakMiner steady
  within ±0.2, our 15-second rate stepped between 86.7, 94.3 and 102.2 TH/s. The 5-minute mean is sound; a single
  reading is about ±8% off.
- **Shares:** ours 27 accepted, PeakMiner 18 and SRBMiner 19, against about 21.5 each expected from their rates.
  No rejects from ours or SRBMiner. PeakMiner's reject count isn't read: `bench.sh` finds no reject field in its
  API reply.
- **Skipped before running:** a 2080 Super (0dcce2f8), and three 2070 Supers once that row had its two PCs.

### On Salad, 30 October, against SRBMiner 3.8.2

Run from 13:56 to 18:10 UTC on 2026-10-30, by the same method as the 28 October run above, after SRBMiner 3.8.1 and
3.8.2 (28 and 29 October) listed small pearlhash gains on RTX 20 to 50: v0.6.9 against PeakMiner 2.20.3 and
SRBMiner 3.8.2, on `us.pearl.herominers.com` (`tuning/salad/bench.sh` at `d2de3f9`). Different PCs, except two;
they leave the rows above alone.

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

- **95.7–99.7% on every PC, median 97.4%,** against 97–103% (median 98%) on 28 October against SRBMiner 3.8.0.
  PeakMiner 2.20.3 was fastest on every PC; SRBMiner 3.8.2 was ahead of us on four of nine, by 2% at most.
- **Two PCs were the 28 October rows' own:**
  - ef08a260 (2080 Ti): ours 94.9 → 94.8, PeakMiner 94.7 → 95.0, SRBMiner 86.4 → 93.6 (+8.3%). The row went from
    100% to 99.7%, now against PeakMiner.
  - 83f7bc7e (2080): all three read 2-4% more than on 28 October, and the row stayed at 97%.
- **Our figure is noisier than theirs.** Ours is the mean of the CLI's readings after the first minute; theirs is
  each miner's own figure at the end. On fast cards our readings step by whole launches, which leaves about
  ±1.5-2.4% of noise on ours alone (research notes, 30 October), so 1-2 points between PCs are within noise.
- **4af24fcc (2060):** nvidia-smi gave no power limit, and all three miners ran slow (35-37 TH/s against 46-48 on
  the other 2060), so the PC itself was slow; the ratio is still a fair comparison.
- **Not recorded:** a 2080 box was moved off 2c215597 after 45 minutes without a result; 15ad9655 turned out to be
  a 2080 Super, which has no row; on 9f287c81 (2070 Super) no miner got work from the pool. Salad's 2070 class also
  handed out the 2070 Super 32c563bc, which got a row of its own.

### Our own builds raced on Salad, 1 November: the transcript hash and the L0 instruction cache

Research item 11, raced at 04:23 UTC on 2026-11-01 at `59ed665`, in race mode (`earn/native/probes/race.cu`: every
build in one process on one card, 4 s turns for 600 s, against the release build's turn in the same round), on
`nvtest` boxes. The idea: Turing's L0 instruction cache holds 16 KiB a scheduler, and while one warp streams the
B-direct fold's 10.9 KiB of straight-line transcript hash its partner runs the 10.3 KiB chunk loop, so both may
refetch from L1. `PEARL_BD_STUB_HASH` (timing only; a 2.1 KiB loop doing the same work, the most any smaller hash
could gain) and `PEARL_BD_ROLLED_HASH` (BLAKE3's rounds as a loop, 4.5 KiB, the same word, but 881 instructions a
region against 700). Not rows: no competitor ran.

| Card | Machine ID | stub | rolled | A/A copy |
|---|---|---|---|---|
| RTX 2080 Ti (260 W) | 6317ad20 | +0.10% ±0.07 (25/34 ahead) | −0.42% (1/34) | +0.02% ±0.05 |
| RTX 2070 (225 W) | 1e19bb79 | −0.01% ±0.02 (13/34) | −0.39% (0/34) | −0.01% ±0.02 |

- **Dropped:** the stub had to gain at least +0.5% on both cards. Shrinking the hash's code to a fifth buys at most
  0.1%, so instruction fetch is not what limits the RTX 20 fold, and the rolled hash only pays for its extra
  instructions. The code is reverted.
- **Cost:** two high-priority boxes for about 13 minutes each, about $0.04.

### Our own builds raced on Salad, 1 November: would k 4096 pay on RTX 20?

Research item 9, raced at 09:41 UTC on 2026-11-01 at `7accd79`, in race mode, on `nvtest` boxes. A share may declare
k 4096, which halves the hashes and tile seams a MAC (on RDNA 4 it gained about 1%, in the AMD section). The B-direct
fold holds a band of B' in the L2 and streams A' past it, and the band rule (`pearl_bd_band_for`) halves the band
when k doubles, so A' bytes a MAC double. That cost needs no k 4096 fold to price: each arm here is today's k 2048
fold built with `-DPEARL_BD_BAND` at the band k 4096 would take, so it reads at least what k 4096 would lose to
DRAM. Not rows: no competitor ran.

| Card | Machine ID | Band today | Arm | vs release | Rounds ahead | Memory busy | Clock |
|---|---|---|---|---|---|---|---|
| RTX 2080 Ti (250 W) | dbcb4344 | 8 | band 4 (k 4096's) | −1.24% ±0.07 | 0/48 | 11% → 19% | 1360 → 1346 MHz |
| RTX 2070 Super (172 W) | bc68007a | 5 | band 2 (k 4096's) | −2.76% ±0.02 | 0/45 | 16% → 35.5% | 1676 → 1640 MHz |
| RTX 2070 Super (172 W) | bc68007a | 5 | band 3 (a better rule) | −1.65% ±0.07 | 0/45 | 16% → 25% | 1676 → 1649 MHz |

- **Dropped:** k 4096 could win back at most about 0.9% (half of the 1–2 points the tile hash and hand-off cost in
  the fold's static model), and it had to clear +0.5%. The DRAM cost alone is 1.2% on the 5.5 MB L2 and 1.7–2.8% on
  the 4 MB one, so no RTX 20 card gains. The 2080 Ti's copy of the release read −0.01% ±0.05. The 3 MB RTX 2060 was
  not raced: the hashrate runs under `PEARL_BD_WALK` in `pearl_config.h` already put its band 2 1.1% behind band 4.
  No code was written for it.
- **Cost:** two high-priority boxes for 12 and 16 minutes, about $0.04.

### Monero on the CPU

SRBMiner 3.7.5 mines Monero (RandomX) on the CPU and Pearl on the GPU in one
process. Its fee is 0.85% on RandomX and 2% on Pearl. PeakMiner 2.18.3
doesn't mine Monero: it rejects `-c randomx` and `-c monero`. Each rental
above also ran SRBMiner for 5 minutes on both at once and 5 minutes on
Monero alone, on HeroMiners' Monero pool (port 1111). The pool accepted 14
shares and rejected none.

| Card | Host | CPU (CPUs the container got) | Monero alone | Monero with PRL | SRBMiner PRL alone → with Monero |
|---|---|---|---|---|---|
| RTX 2080 Ti | Thailand | EPYC 7B13 (17.6 of 256) | 504 H/s | 556 H/s | 64.2 → 44.0 |
| RTX 2080 Ti | Florida (stand-in) | Ryzen 9 5900X (11.5 of 24) | 4,133 H/s | 3,869 H/s | 32.8 (see above) → 40.1 |
| RTX 2070 Super | Alberta | Ryzen 5 5500 (11.5 of 12) | 2,677 H/s | 2,509 H/s | 64.0 → 64.2 |
| RTX 2070 | South Korea | Xeon E5-2680 v4 (13.4 of 56) | 3,672 H/s | 3,760 H/s | 51.7 → 50.5 |
| RTX 2060 Super | Germany | Core i5-10400 (7.7 of 8) | 1,845 H/s | 1,810 H/s | 46.4 → 45.3 |
| RTX 2060 | Australia | Xeon E5-2699 v3 (34.6 of 72) | 4,128 H/s | 4,209 H/s | 48.4 → 49.4 |
| RTX 2060 | South Korea (12 GB) | Xeon E5-2680 v4 (9.0 of 56) | 2,399 H/s | 2,575 H/s | 52.4 → 52.6* |

- These are rented containers, not desktops. None allowed huge pages or
  CPU tuning (MSR), which a tuned desktop can have, so the Monero rates are
  low for these CPUs.
- SRBMiner ignores the container's CPU quota when it picks its thread
  count. It ran 2–15 times more threads than CPUs on Thailand,
  Florida and both South Korea hosts. On Thailand (256 threads on 17.6
  CPUs) that cut its Pearl rate by 31% and the pool saw no Monero share.
  On Alberta, the South Korea 2070, Germany and Australia the Pearl rate
  moved by 2.4% or less.
- \* On the South Korea 2060 the GPU sat at 88% use during the dual run.
  SRBMiner's own average over the run was 45.3 TH/s, so the end figure
  overstates it.

Our CPU miner, new in v0.6.0, ran on 2026-10-11 on two other rentals, not
the rows above, built from its PR branch. Each ran ours and SRBMiner 3.7.5
for 5 minutes on Monero alone, one after the other, with the same thread
count and pool. "% of best" is ours divided by SRBMiner's rate. PeakMiner
doesn't mine Monero.

| CPU (CPUs the container got) | GPU | Threads | Ours | SRBMiner 3.7.5 | % of best |
|---|---|---|---|---|---|
| Core i5-11400 (11.5 of 12) | RTX 3060 | 6 | 2,457 H/s | 1,880 H/s | 131% |
| Ryzen 9 5950X (15.4 of 32) | RTX 2060 | 14 | 6,439 H/s | 4,527 H/s | 142% |

- Ours is the mean of its readings over the run. SRBMiner's is its own
  1-minute figure at the end.
- Ours used transparent huge pages for its dataset. SRBMiner asks for
  reserved 2 MB huge pages, couldn't get enough on either box, and ran
  without them. That likely explains much of the gap. Neither box let
  SRBMiner reserve huge pages, so how they compare when it can is
  untested. XMRig wasn't run.
- With both mining, our Pearl rate moved by under 1%: 44.23 → 44.03 TH/s
  on the 3060 and 42.21 → 41.90 on the 2060. Across all our test runs the
  pool accepted 34 of our Monero shares and rejected none.

## RTX 30-series

Run on 2026-10-10, 15:58–18:51 UTC; the Vietnam 3090 Ti and the 3070 Ti ran
at 20:24–20:49 UTC. Three rows are on their own machine: the Ukraine 3090,
the Portugal 3080 Ti and the 180 W Quebec 3070. The other twelve machines
couldn't be rented, so those rows are on new hosts.

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

- **New hosts,** with the row's earlier host and machine ID:
  - RTX 3090 Ti: Vietnam (21444) for Greece (152830), and Oman (12840) for
    Washington (151121).
  - RTX 3090: Quebec (67901) for Quebec (152641).
  - RTX 3080 Ti: Japan (140510) for Japan (137807).
  - RTX 3080: Taiwan (146647) for Kentucky (29108), and Austria (143869) for
    Washington (25433).
  - RTX 3070 Ti: Quebec (137703) for Ontario (43435).
  - RTX 3070 (220 W): Quebec (56506) for Quebec (148988).
  - RTX 3060 Ti: Colorado (151268) for Japan (137800), and Argentina
    (150126) for New Zealand (144095).
  - RTX 3060: New Brunswick (15303) for Poland (149975), and South Korea
    (137223) for Vietnam (138808).
- **Power limit:** the Ukraine 3090's host now sets 290 W, 83% of the card's
  350 W. Before this run the row listed 390 W.
- **RTX 3090 Ti, Oman:** SRBMiner exited at once, wrote nothing and left the
  card idle, so the row compares with PeakMiner. SRBMiner ran normally on
  the Vietnam 3090 Ti, so the problem is this host, not the card.
- **RTX 3090 Ti, Spain run dropped:** an earlier run on Spain (147387) read
  ours 148.6, PeakMiner 146.9 and SRBMiner 72.7. At the start of all three
  runs that card was already at 311–346 W and 100% use before the miner
  under test was hashing, so another workload was on the GPU. SRBMiner's
  half rate most likely came from sharing it. The run was dropped. The
  Vietnam box checked that the card was idle before the warm-up and before
  each miner.
- **Heat-limited hosts, dropped:** the first 3080 host, Utah (140965), ran
  at 93 C and 690–840 MHz and read ours 51.2, PeakMiner 47.0 and SRBMiner
  51.6. The first 3060 host, United States (120972), ran at 88–91 C and
  about 1480–1560 MHz, at 131–135 W of 170 W, and read 40.5, 41.8 and 42.9.
  Both rows were rerun on hosts that hold their clocks.
- **RTX 3070, Quebec (180 W):** at the start of our run and PeakMiner's, the
  card was already at 179 W and 99% use before the miner was hashing, so
  another workload uses this card at times. The measured windows look clean:
  per clock, all three miners match the 220 W 3070 within 0.4%. The row is
  kept.
- **SRBMiner's end figure:** on the Quebec 3090 it was a one-sample dip,
  123.05 against 126.1–126.3 just before. Against SRBMiner's mean after the
  first minute (126.34) the row would read 99.1%. On the Colorado 3060 Ti,
  against SRBMiner's mean after the first minute (60.82), it would read
  98.1%.
- **RTX 3060 Ti, Argentina:** the host locks the clock at 1560 MHz, so all
  three miners ran there, below the power limit.
- **Ukraine 3090 and Portugal 3080 Ti:** their raw files were lost (see "How
  a host is tested"). The Portugal machine had an earlier rental stopped
  mid-run, as did the Spain 3090 Ti and the 180 W Quebec 3070 machines, and
  both of those showed another workload on the card. Portugal's
  start-of-run samples are lost, so that can't be ruled out there, but its
  per-clock rates match the Japan 3080 Ti's.

### On Salad, 28 October, against the newest miners

Run from 17:08 UTC on 2026-10-28 to 00:39 UTC on the 29th, by the same method as the RTX 20-series' Salad run:
v0.6.9 against PeakMiner 2.20.3 and SRBMiner 3.8.0, on Salad PCs, 5 minutes each after a warm-up, a card's first
PC with ours first and its second with SRBMiner first. These are different PCs and leave the rows above alone.

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

- **91–97% on every healthy PC, median 92.5%,** down from 97–101% on the Vast rows. It fell on all eight cards
  and against both competitors, which are within about 1% of each other on every PC. Where a power limit matches
  a Vast row, the competitors are 5–7% faster than there and ours is flat to 3% slower:
  - 3090 Ti at 450 W: ours −1.4%, PeakMiner +7.5%, SRBMiner +4.8%.
  - 3080 Ti at 350 W: ours −0.9%, both competitors +7.4%.

  So most of the drop is their gain, though on different PCs.
- **The gap has two parts.**
  - **Per clock:** ours does 94.9–96.7% of the best competitor's work on all 15 PCs, 95.8% on average.
  - **Clock at the power limit:** ours runs a lower clock than they do at the same power, usually 1.6–4.6% lower,
    and 14% lower on the 336 W 3090 Ti.

  On the two PCs where something other than power set the clock (9fbd405b, 0771915f), the clock gap disappears
  and ours reaches 96–97%.
- **b687d663 (3090 Ti):** the host caps it at 336 W, 70% of its 480 W default. Ours ran at 1560 MHz against
  1800–1815 MHz at the same 335 W. Its rate per clock matches the 450 W 3090 Ti's (80.8 against 81.0 TH/s per
  GHz), so the cap explains the low row, not another workload.
- **66666c27 (3090):** not limited by power. All three miners drew 280–313 W of 350 W and stopped at 1455–1470
  MHz, so the host caps the clock. Ours drew 22 W more, ran 3–4 °C hotter and settled at 1410 MHz.
- **9fbd405b (3080):** the competitors stopped at 297 W of 340 W, at 1680–1695 MHz. Ours reached the limit at
  1710 MHz, our best row at 97%. PeakMiner reported no accepted share there at a normal rate, where about 4 were
  expected.
- **0771915f (3060):** ran at 84–86 °C with all three miners at the same clock and power, so only the per-clock
  gap shows (96%).
- **Heat-limited PC, dropped:** be39e9c4 (3090, 420 W) sat at 92–93 °C, 630–930 MHz and about 210 W, with
  unsteady rates. It read ours 53.7, PeakMiner 71.1 and SRBMiner 68.5 (75%). A third 3090 wasn't run.
- **No pool work, not recorded:** on d5b5d769 (3060 Ti) PeakMiner logged "authorize failed … Connection timed out"
  and SRBMiner sat idle at 11 W. 3a481d90 took its place.
- **Shares:** ours 45 accepted, against about 53 expected at the competitors' rate. No rejects from ours or
  SRBMiner.

### Our own builds raced on Salad, 31 October: holding B' in the L2

Run from 22:47 UTC on 2026-10-31 to 00:57 on 1 November, on `nvtest` boxes (`tuning/salad`), in race mode
(`earn/native/probes/race.cu`): every build in one process on one card, taking turns of 4 s (1 s of warm-up, then
whole salts) for 600 s, each scored against the release build's turn in the same round. These are builds of our own
core against each other, not rows: no competitor ran. They decided research item 1, the Ampere flip, which ships in
`f421841` (`PEARL_AMPERE_FLIP`): hold the B' of a narrow batch in the persisting L2 slice and stream A', instead of
holding a band of A' and reading B' again every band.

| Card | Machine ID | Batch | vs release | ±95% | Rounds ahead | Memory busy | Clock |
|---|---|---|---|---|---|---|---|
| RTX 3060 Ti (200 W) | 997ccdd4 | 64 | +1.92% | 0.09 | 35/35 | 37% → 21% | 1730 → 1762 MHz |
| RTX 3090 (350 W) | db36e38b | 128 | +0.66% | 0.04 | 35/35 | 15% → 10% | 1622 → 1632 MHz |
| RTX 3090 (297.5 W), shipped build | 33af5d90 | 128 | +1.52% | 0.37 | 49/49 | 16% → 10% | 1487 → 1509 MHz |
| RTX 3060 Ti (156 W), shipped build | — | 64 | +3.62% | 0.91 | 41/42 | 20% → 14% | 1167 → 1196 MHz |
| RTX 3080 10 GB (320 W), flip forced on | — | 64 | −0.71% | 0.07 | 0/44 | 22% → 19% | 1623 → 1634 MHz |
| RTX 3080 10 GB (320 W) | cd2691f0 | 512 in strips of 64 | +0.03% | 0.05 | 19/34 | 22% → 21% | 1696 → 1700 MHz |
| RTX 3060 (153 W) | 21b77e7f | 512 in strips of 32 | −0.63% | 0.11 | 0/34 | 33% → 37% | 1754 → 1748 MHz |

- **Why only some cards:** the batch is as wide as the card's largest persisting slice allows (2 MiB of B' at 64
  offsets, 4 MiB at 128). The 3060 Ti and the 6 MB cards fit a batch wide enough that A' read once a batch costs
  less than B' read once a band; the 3080 10 GB and the 3060 do not. The shipped rule (`pearl_ampere_flip_for`)
  takes the flip where its modelled DRAM bytes are at most 80% of the band walk's: the 3060 Ti and the 3080 12 GB,
  3080 Ti, 3090 and 3090 Ti. Forced on a 3080 10 GB (`-DPEARL_AMPERE_FLIP_SHARE=200`), the shipped flip read
  −0.71%, behind in all 44 rounds. The 3060 row is the strip walk at the width the flip would take there, a
  stand-in for the narrow batch: on the 3060 Ti the same strips read +1.48%, the narrow batch +1.92%.
- **A tighter limit gains more:** the 3090 capped at 297.5 W gained +1.52% where the 350 W one gained +0.66%; at
  the cap, the watts DRAM no longer takes go to the SM clock. The 156 W 3060 Ti gained +3.62%, but it drew only
  108-111 W at 1167-1200 MHz and 74 °C, so heat or the host held it below its limit, and its A/A copy read
  +0.50% ±0.77: take that row as a direction, not a size.
- **The race's own noise:** a copy of the release raced against it read −0.05% to +0.04% in the six races at full
  power that had one (±0.05 to ±0.10 at 95%), +0.14% ±0.36 on the 297.5 W 3090, whose clock moved more, and
  +0.50% ±0.77 on the 156 W 3060 Ti. The
  old 2048-offset batch, known to lose 0.2–0.9% to 512, read −0.84% and trailed in all 28 rounds.
- **The old method agrees, with its bias:** on the same 3060 Ti, `hashrate.js` in separate 60 s runs gave the
  strip walk (512 in strips of 64) −0.6% in the first round, when the release ran first on a cold card (58 °C,
  1775 MHz), and +1.5 to +1.7% in the three rounds after. Race mode put the same build at +1.24% ±0.08.
- **3090, repeated:** +0.68% and +0.66% in two more races at 128 offsets.
- **Tried and dropped,** in the same races:
  - tiles from a counter (`PEARL_TALL_CTR`) with the flip: 3090 +0.14% against +0.68% without; on the 3060 Ti's
    strip walk −1.22% against +1.24% without;
  - 64 offsets on the 3090: −0.29%;
  - half of a 256-offset batch's B' held: −1.9 to −2.2%, memory 27–33%;
  - a 256 to 2048-offset batch walked in strips whose B' the slice holds: 3060 Ti +1.79% (256), +1.48% (512),
    +0.41% (2048); 3090 +0.52%, +0.45%, −0.31%. Below the narrow batch on both cards;
  - narrower strips on the 3080 10 GB (32 offsets) −1.31%, and on the 3060 (16 offsets) −2.95%.
- **Correctness:** `verify-hits.js` checked 400 of 400 hits on the 3090's narrow flips (128, with and without the
  counter), the strip builds on the 3060 Ti, 3090 and 3060, and the shipped build on the 297.5 W 3090 and the
  156 W 3060 Ti, where it ran the flip at 128 and 64 offsets. At the same batch width the 3090's four narrow and
  half-held builds found exactly the release's 2,640 hits, and the strip builds the release's 620 (3060 Ti and
  3090). Builds at different widths can't be compared batch by batch; `compilers.sh` says so now.
- **Pool runs of the shipped build** (`pool.sh`, Kryptex, 10 minutes after a 1-minute warm-up, our CLI alone): the
  297.5 W 3090 read 118.78 TH/s with 3 shares accepted and 0 rejected; the 156 W 3060 Ti 45.11 TH/s with 8 accepted
  and 0 rejected.
- **Cost:** nine high-priority boxes for 15–75 minutes each, about $0.60 at Salad's list prices.

### Our own builds raced on Salad, 1 November: copy groups and a readout fence

Research item 10, raced at 03:07 UTC on 2026-11-01 at `db1c5fd`, in race mode as above (600 s, turns of 4 s), on
`nvtest` boxes: `PEARL_TALL_COPY_RUN` (the cp.async copies' addresses from running pointers, and the next A tile
loaded above the copy group: IMMAs a chunk with no cover, 6 to 2) and `PEARL_TALL_RO_FENCE` (the per-chunk readout
spread between `pmevent.mask 0` fences), 2x2 against the release build. Not rows: no competitor ran.

| Card | Machine ID | copy | fence | both | A/A copy |
|---|---|---|---|---|---|
| RTX 3060 Ti (200 W, the flip on) | 3da34f5f | +0.10% ±0.04 (29/35 ahead) | −2.37% (0/35) | −2.39% (0/35) | not run |
| RTX 4070 Super (220 W) | 271b82b6 | +0.01% ±0.15 (16/28) | −2.57% (0/28) | −2.69% (0/28) | +0.00% ±0.15 |

- **Dropped:** the pass rule was at least +0.3% with the interval clear of 0 on both cards. The copy arm's six-to-two
  cut in uncovered IMMAs buys at most 0.1%: the other warp on the scheduler already covers those waits. The fences
  cost 2.4-2.7% on both: they stop ptxas interleaving the readout's LOP3 with anything, so it costs more than the clump
  it replaced. The code is reverted; the verify stage was not run.
- **Cost:** two high-priority boxes for about 15 minutes each, about $0.06.

### Our own builds raced on Salad, 1 November: what the tile hash costs

Research item 12, raced at 05:39 UTC on 2026-11-01 at `015c02e`, in race mode as above, on `nvtest` boxes. Timing-only
stand-ins in the tall fold's hasher branch (they refuse to mine): `nohash` (`PEARL_TALL_STUB_SPIN=1`: no hash at all,
the like-for-like reference), `s0`/`s0s` (spin 3,000 / 1,700 cycles from the hand-off), and `s1` (`PEARL_TALL_STUB_ALU`:
the hash's issue count from 816 bytes of code). Not rows: no competitor ran.

| Card | Machine ID | no hash | spin 1,700 | spin 3,000 | ALU burst | hash ablation | A/A copy |
|---|---|---|---|---|---|---|---|
| RTX 4070 (142 W of 200, 2,112 MHz) | 60ddb754 | +1.69% (19/20) | +1.79% | +1.31% | +0.49% | +2.14% | −0.16% ±0.13 |
| RTX 3060 Ti (147 W of 200, 1,473 MHz) | b1a5c182 | +1.42% (24/25) | +1.16% | +0.50% | +0.27% | not run | not run |

- **The tile hash costs RTX 30 and 40 about 1.4-1.7%,** far more than round 4's +0.15-0.2% estimate. Both PCs ran
  below their power limits, so this is per clock.
- **Most of it is issue contention:** the ALU burst, which keeps the hash's instruction count but not its code or its
  waits, keeps almost all the cost. Waiting through the partner's window costs little (spin 1,700 matches no hash).
  The existing hash ablation reads high because it also reschedules the chunk loop. **Wrong, as step 0 below
  shows:** the burst runs about as long as the hash, so it pays the same run past the window. Contention is
  under 0.1 point.
- **So no cheap fix follows.** The matching fix is SASS control bits that let the hasher yield, which needs an
  sm_86/sm_89 port of the sm_75 patcher; items 17 and 18 (readout pacing, warp turn-taking) aim at the same cost.
  The stand-ins stay in the code, off, as their gauge.
- **Cost:** two high-priority boxes for about 13 minutes each, about $0.05.

### Our own builds on Salad, 2 November: where the tile hash's cost comes from (step 0)

Run at 03:27 UTC on 2026-11-02 at `4c78475` on `nvtest` boxes. This was not race mode: bench ran 15 builds in turn,
20 s each, for 5 rounds, with `PEARL_TALL_CYC` tile cycles and the new `PEARL_TALL_HASHCYC` stamps. Those stamps time
the hasher from the hand-off to its hash's end and to its release of stage 0. Each card ran at its stock power limit
the whole time.

Arms:
- `base`: the real hash.
- `s1`: no hash.
- `s1200` to `s4600`: spins of that many cycles.
- `a1` to `a5`: ALU bursts.
  - `a1`: the hash's instruction count.
  - `a2`: one chain.
  - `a3`: adds as IMAD, on the FMA pipe.
  - `a4`: `a1` plus 32 IMMAs.
  - `a5`: IMAD only.
- `base0` and `s1x`: `base` and `s1` without stamps.

Not rows: no competitor ran.

| Card | Machine ID | Tile, no hash | Hash cost | Hash ends / releases | Past the window | Contention | Stamps' own cost |
|---|---|---|---|---|---|---|---|
| RTX 3060 Ti (200 W) | 9db901eb | 101,648 cycles | +1,330 (1.31%) | 3,068 / 3,140 | 1,299 (1.28%) | +31 (0.03%) | +219 (0.22%) |
| RTX 4070 Super (220 W) | 6d99dc2c | 101,561 cycles | +1,396 (1.37%) | 3,116 / 3,188 | 1,333 (1.31%) | +63 (0.06%) | +303 (0.30%) |

What each spin length costs a tile, against no hash. The release cycle comes first, then the cost:

| Card | 1,476 | 1,940 | 2,276 | 2,628 | 3,220 | 4,820 |
|---|---|---|---|---|---|---|
| RTX 3060 Ti | +311 | +464 | +502 | +809 | +1,375 | +2,990 |
| RTX 4070 Super | +324 | +553 | +568 | +804 | +1,371 | +2,979 |

Round-to-round spread was about 12 cycles a tile on the 3060 Ti and 2 on the 4070 Super. Each spin's stamp read 156 to
180 cycles above its length on both cards. That is the stamp's own offset, so the stamps time what they should.

- **The cost is the hash running past the partner's window, not contention.** Past about 2,300 cycles after the
  hand-off, each cycle the hasher runs costs a tile 0.9-1.0 cycle on both cards. Below that it costs about 0.3. The
  hash releases at about 3,150, so about 0.8 point is the hash's length past 2,300.
  - The burst (`a1`) costs what a spin of the same length costs: −41 cycles on the 3060 Ti, +62 on the 4070 Super.
  - That overturns item 12's reading above.
- **Dropped: hash control bits (the SASS yield and stall bits).** The rule was to drop them if the 3060 Ti's
  contention came in under 0.5 point. It is 0.03 point. Yield would also make the hash longer, and each extra cycle
  costs about one.
- **Dropped: #18, warp turn-taking.** The rule was to drop it if the partner kept under about 80% of its rate beside
  the burst.
  - On the 4070 Super, the burst's +62 cycles mean the partner had not finished its window's work by the release. It
    therefore ran at under about 75% of its rate. The model puts it at 31-43%.
  - The 3060 Ti can't settle it: its contention is inside the noise, which gives only a lower bound (at least 46%).
- **Dropped: IMMAs during the hash.** The 32-IMMA arm (`a4`) released about 1,400 cycles later than `a1` and cost
  what it added. The tensor pipe has no idle time at the seam.
- **Parked: #17, readout pacing.** It was to ride the control-bit patch path, which is now not being built. On its own
  it would need that 5-6 h path, and a hand-patched binary to ship, for +0.15-0.45% at 15-20% odds.
- **What helps instead: a shorter hash.** The adds-as-IMAD burst (`a3`) has the same count as `a1` but splits it
  between the ALU and FMA pipes. It released about 300 cycles sooner and cost 130-175 fewer cycles a tile, even
  with more contention. The one-chain burst (`a2`) ran almost twice as long.
  - So the hasher is limited by the ALU pipe and by its own length.
  - The fixes that follow are a shorter hash phase (word-major transcript records, or more of the hash on the FMA
    pipe) or a wider window (six 32-k stages).
- **The stamps cost a fixed 220-300 cycles a tile.** `s1` minus `s1x` matches `base` minus `base0`, even though `s1`
  releases well inside the window. So the cost doesn't come through the hash's length. It is the same in every
  stamped arm and cancels in the comparisons above.
- **Spill:** the box logs don't print registers. Locally, under CUDA 12.6 and 12.8.1 (the box ran 12.8.93), every
  arm keeps 254/255 registers with no spill and the chunk loop unchanged.
- **Cost:** two high-priority boxes for about 30 minutes each, about $0.15.

## RTX 40-series

Run on 2026-10-10, 16:04–19:46 UTC. Ten rows are on their own machine; eight
moved to new hosts.

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

- **New hosts,** with the row's earlier host and machine ID:
  - RTX 4090: Texas (153411) for British Columbia (152545).
  - RTX 4080: Japan (6706) for Taiwan (149135), and British Columbia (43318)
    for Nevada (147894).
  - RTX 4070 Ti Super: Quebec (36427) for United States (43741).
  - RTX 4070 Ti: North Carolina (137392) for United Kingdom (149163), and
    France (138259) for Delaware (39901).
  - RTX 4070: Colorado (144340) for New York (19053).
  - RTX 4060 Ti: Ohio (27068) for Ontario (37799).
- **Power limit:** the Australia 4060's host now sets 115 W. Before this
  run the row listed 120 W.
- **Runs replaced by the row's own machine:** the Australia 4090 and the
  California 4080 Super first ran on new hosts while their machines couldn't
  be rented. Illinois (20820) read ours 306.3, PeakMiner 307.0 and SRBMiner
  277.2, a dip in SRBMiner's last reading (its own 1-minute average was
  297.5). Arizona (151147) read 203.1, 202.1 and 203.2. Both machines came
  back, at 19:25 and 19:00 UTC, and the rows are those runs.
- **RTX 4090D:** its raw files were lost and SRBMiner's figure was cut (see
  "How a host is tested"). The machine couldn't be rented again before the
  run ended. A rerun on a new, unverified host, Tanzania (70632), didn't
  start its container within 10 minutes and was dropped. The row stays on
  its own machine.
- **RTX 4060, Australia:** the host runs the card either locked at 1995 MHz
  or power-capped near 2600 MHz. In this run it held 1995 MHz for all three
  miners.

### On Salad, 29–30 October, against PeakMiner 2.20.3 and SRBMiner 3.8.0

Run from 22:45 UTC on 2026-10-29 to 03:30 UTC on the 30th, by the same method as the RTX 20- and 30-series' Salad
runs: v0.6.9 against PeakMiner 2.20.3 and SRBMiner 3.8.0, on Salad PCs, 5 minutes each after a warm-up, a card's
first PC with ours first and its second with SRBMiner first (`tuning/salad/bench.py` at `f60a455`). These are
different PCs and leave the rows above alone. Salad's 4090 class can hand out a 4090D, its 4080 class a 4080
Super and its 4070 class a 4070 Super, but none did. SRBMiner 3.8.1 (28 October, 20:00 UTC) and 3.8.2
(29 October, 13:48 UTC) were already out when this run started. Their notes list a minor pearlhash gain on RTX 20
to 50, so SRBMiner's figures here and in the RTX 50 run may be a little below its newest.

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

- **93.5–95.8% on every PC, median 94.5%,** down from 98–102% on the Vast rows. PeakMiner and SRBMiner are within
  about 1% of each other on every PC. Where a power limit matches a Vast row, the competitors are 4–7% faster than
  there and ours is flat to 3% slower, on different PCs:
  - 4090 at 450 W: ours +0.3%, best competitor +6.9% (means of both PCs against both Vast rows).
  - 4080 at 320 W: ours −2.7%, best competitor +3.6% (against Japan).
  - 4070 at 200 W: ours −0.7%, best competitor +4.3%.
  - 4060 at 115 W: ours −1.4%, best competitor +3.8% (against New Zealand).
- **Power limits** are the host's, read from `nvidia-smi`; two were 253.65 W and 242.25 W, rounded in the table.
  Drivers were 610–617.
- **Box moved:** the first 4070 Ti Super box was given PC `8123f436`, then moved to `8445b495` before it ran.
  Six boxes waited out Salad's 30 minutes with no PC and were started again; none ran long.
- These boxes don't record clocks, so the gap can't be split into per-clock work and clock, as on the RTX 30.

## RTX 50-series

Run on 2026-10-10, 16:02–17:48 UTC. Every row ran on the CUDA 13 core except
the India 5090, whose driver 570 gets the CUDA 12.8 core. Seven rows are on
their own machine; five moved to new hosts.

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

- **New hosts,** with the row's earlier host and machine ID:
  - RTX 5090: South Korea (148959) for South Korea (145183).
  - RTX 5080: United States (137656) for New Jersey (152066).
  - RTX 5070 Ti: South Korea (28072) for South Korea (27661).
  - RTX 5070: Texas (147729) for Poland (31379).
  - RTX 5060 Ti: Finland (140025) for Virginia (151123).
- **Competitors that didn't run:** on the India 5090, PeakMiner logged
  "driver too old for this CUDA runtime"; it needs driver 580 or newer on
  these cards. PeakMiner on the Virginia 5060 and SRBMiner on the India 5060
  exited at once with an empty log and left the card idle. Each of these
  rows compares with the competitor that ran.
- **RTX 5060, Virginia:** SRBMiner is the only comparison, and its end
  figure (76.75) is a dip. Against its mean after the first minute (77.47)
  the row would read 98.4%.
- **RTX 5060 Ti, Ontario:** SRBMiner ran below the power limit, at about
  166 W and 2497 MHz, and read 88.4. PeakMiner is the comparison.
- **RTX 5080, United States:** heat held the card at 87–88 C and about
  286–298 W of its 330 W limit for every miner. Both competitors' end
  figures were below their means after the first minute (PeakMiner 203.1,
  SRBMiner 202.7). Against those the row would read about 98%.
- **RTX 5070 Ti:** both hosts lock the SM clock at 1342 MHz, even at idle,
  so all three miners ran at 86–98 W of 250 W. The comparison is fair, but
  the card runs far below a stock 5070 Ti.

### On Salad, 29–30 October, against PeakMiner 2.20.3 and SRBMiner 3.8.0

Run from 03:30 to 07:29 UTC on 2026-10-30, by the same method as the RTX 40-series' Salad run above. Every PC had
driver 591 or newer, so all ran the CUDA 13 core. These are different PCs and leave the rows above alone.

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

- **94.6–96.4% on 10 of the 12 PCs,** down from 96–101% on the Vast rows. Where a power limit matches a Vast row,
  the competitors are 2–7% faster than there and ours is −1% to +2%, on different PCs:
  - 5080 at 360 W: ours −1.1%, best competitor +2.4% (means of both PCs against Poland).
  - 5070 at 250 W: ours +2.2%, best competitor +6.6% (against the mean of Colombia and Texas).
- **RTX 5090 trails most:** 92% at 575 W and 89% at 450 W. On Vast at 500 W it was 99% on the CUDA 13 core.
- **RTX 5060 `0db26521`:** the one PC in either series where ours beat both competitors (101%). The other 5060,
  at the same 145 W, read 95%.
- **Boxes restarted:** three waited out Salad's 30 minutes with no PC and were started again; none ran long.

### On Salad, 30 October, against SRBMiner 3.8.2

Run from 18:10 to 22:21 UTC on 2026-10-30 by the same method, against PeakMiner 2.20.3 and SRBMiner 3.8.2. The
founder retired HeroMiners during the run: boxes started from 20:30 UTC mined on Kryptex
(`prl-us.kryptex.network:7048`, `bench.sh` at `cce2aaf`), the earlier ones on `us.pearl.herominers.com` (`d2de3f9`).
All three miners of a row share its pool. Different PCs, except one; they leave the rows above alone.

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

- **87.6–103.2%, median 95.2%.** SRBMiner 3.8.2 was the faster competitor on 10 of the 12 PCs, PeakMiner on two.
- **f6e3944e (5080) was the 29–30 October row's own PC:** ours 228.5 → 228.6, PeakMiner 237.0 → 237.0, SRBMiner
  237.4 → 239.3 (+0.8%). The row went from 96.3% to 95.5%.
- **RTX 5090 at 437 W: 87.6%,** our lowest NVIDIA row. At 575 W it read 95.6%.
- **RTX 5060 960ed7a7: 103.2%,** ours ahead of both. With 0db26521 (101%) on 29–30 October, that's two 145 W 5060
  PCs past 100%, while 616220a5 at the same 145 W read 95% and dd173265 at 123 W 92.8%. What separates them isn't
  known yet.
- **The pools agree:** the two 5070 rows, one on each pool, read 95.6% and 95.8%.
- **Not recorded:** on bf32527c (5070 Ti) no miner got work from the pool.
- Drivers 596-621. The same noise caveat as the RTX 20 run applies to ours.

### Our own builds raced on Salad, 1 November: would k 4096 pay on RTX 50 and RTX 40?

Research item 8, raced at 10:51 UTC on 2026-11-01 at `aa63e33`, in race mode (`name=lib.so:K` gives a build its own
k and profile, so every rate is MACs a second), built with CUDA 13.3.1 as the CUDA 13 core is. A share may declare
k 4096: the tall fold's tile hash, hand-off and seam then cover twice the MACs, but a tile's operand panels double,
so the batch narrows to keep B' in the L2 and A' is read twice as often. RTX 50's TMA fold reaches k 4096 only with
`PEARL_TALL_PREGUARD=0` (its readout before the hand-off guard has no wrap), so that build is raced too. Not rows: no
competitor ran.

| Card | Machine ID | Arm | vs release | Rounds ahead | Memory busy | Clock |
|---|---|---|---|---|---|---|
| RTX 5090 (480 W) | 43dc224b | k 4096, PREGUARD off, batch 1024 | −0.78% ±0.35 | 3/30 | 4.9% → 9.5% | 2266 → 2226 MHz |
| RTX 5090 (480 W) | 43dc224b | k 2048, PREGUARD off | −0.05% ±0.24 | 17/30 | 4.9% → 4.5% | 2266 → 2268 MHz |
| RTX 5090 (480 W) | 43dc224b | k 2048, batch 1024 | +0.04% ±0.25 | 12/30 | 4.9% → 4.0% | 2266 MHz |
| RTX 4090 (400 W) | 08d26781 | k 4096, batch 512 | −0.29% ±0.05 | 2/50 | 2.9% → 5.4% | 2484 → 2467 MHz |

- **Dropped at the power limit:** k 4096 does more work a clock, about +1.0% on the 5090 and +0.4% on the 4090 (rate
  over clock), but its DRAM traffic doubles, and at the cap those watts come out of the SM clock. The narrower batch
  alone costs nothing (the 5090's batch-1024 arm), so it is k's own A' traffic. The copy of the release read
  +0.06% ±0.15 on the 5090 and −0.02% ±0.03 on the 4090. The 4090's −0.29% fits bench.exe's +0.1% at k 4096 in
  `pearl_config.h` on a different PC.
- **Left open:** a card held at a fixed clock below its power limit keeps the per-clock gain, at more watts. Salad
  cannot hold a clock, so that needs a rig run.
- **Correctness:** `verify-hits.js` 400 of 400 at k 4096 on both cards (the 5090's PREGUARD-off TMA fold, the 4090's
  cp.async fold), and at k 2048 on the 5090's PREGUARD-off and batch-1024 builds.
- **Cost:** two high-priority boxes for 15 minutes each, about $0.15.

### Our own builds raced on Salad, 1 November: what moving the operands costs a stock 5090

Research item 16 (the RTX 50 fold rewrite), step 1, raced at 21:19 UTC on 2026-11-01 at `933a79a`, built with CUDA
13.3.1, on `nvtest-rtx5090-11`: a stock RTX 5090 (PC `cbd4eba1`, driver 621.01, power limit 600 W, its default). The
rewrite only pays if, at the power cap, moving fewer operand bytes lets the clock rise. `PEARL_TALL_AB_SKIP` (timing
only; the host refuses to mine with it) stops the TMA producer loading A' (1), B' (2) or both (3) after each block's
first tile, so the warps fold the same MMAs, readout and hash on what is already in shared memory, with about 99.8% of
the operand bytes gone. Race mode, 6 s turns after 2 s warm-up for 1,200 s, clock and power logged each turn. Not
rows: no competitor ran.

| Arm | vs release | ±95% | Rounds ahead | Clock | Per clock | Power |
|---|---|---|---|---|---|---|
| release | 432.46 TH/s | | | 2613 MHz | | 600.0 W in every turn |
| copy of the release | +0.02% | 0.02 | 20/40 | +0.00% | +0.01% | 600.0 W |
| skip A' (12 KB a stage) | +4.19% | 0.02 | 40/40 | +4.28% | −0.08% | 600.0 W |
| skip B' (16 KB a stage) | +5.20% | 0.03 | 40/40 | +5.32% | −0.12% | 600.0 W |
| skip both | +8.33% | 0.03 | 40/40 | +8.16% | +0.15% | 600.0 W |

- **Moving the operands costs a stock 5090 about 8% of its clock.** The card sits at its 600 W limit in every turn, and
  without the operand traffic the same work runs 214 MHz faster; work per clock moves +0.15%, inside the 0.3% noise.
  So it is energy, about 0.3% of clock for each KB a stage moves, and almost none of it is waiting. The bytes come
  from L2, not DRAM (memory busy 2.0% against 0.0%): this is the cost of L2-to-SM traffic and shared-memory fills.
- **The rewrite passes its step-1 gate** (2% at the cap): a fold that makes its operands on the SM instead of moving
  them could win back part of the 8%. How much depends on what generating them costs, which step 2 prices.
- The build stage's register and spill lines didn't reach the box's log. A spill would slow a skip arm, not speed it,
  so it could only understate the result.
- **Cost:** one high-priority box for about 22 minutes, about $0.15.
- **The cluster build hangs on a 5090.** `PEARL_TALL_CLUSTER=1` (two CTAs sharing B' by TMA multicast; compiled by CI,
  never run on a card) went into a race at 21:44 UTC the same day, at `933a79a`, on another stock 5090 (PC
  `6509e943`, 575 W of a default 575 W), against the release and the B' skip. The race wrote its audit lines and then
  no turn at all for 21 minutes, while the card sat at 250-255 W and 2910 MHz: a kernel spinning, not folding (the
  fold draws the full 575 W). A refused cluster launch would have returned an error, so the clusters were granted
  and something in the multicast or the cluster-wide EMPTY barrier never completes on sm_120. Which build hung isn't
  known; the release runs fine in every other race. Before any multicast work on RTX 50 (item 16's step 2), a
  standalone feed probe has to show multicast works and what it saves. The box was stopped; about $0.12.
- **The mid cards gain less: they hit their top clock first.** The same skip arms raced at 22:09 UTC at `9e6d18b`,
  900 s each, all three cards at their stock limits:

| Card | PC | Release | Skip B' | Skip both | Clock, release to skip both | Power, skip both |
|---|---|---|---|---|---|---|
| RTX 5090 (600 W) | `cbd4eba1` | 432.46 TH/s | +5.20% | +8.33% | 2613 → 2827 MHz | 600 W, the limit |
| RTX 5080 (360 W) | `973fc20d` | 231.18 TH/s | +2.70% ±0.03 | +3.29% ±0.04 | 2820 → 2910 MHz | 331.5 W |
| RTX 5070 Ti (300 W) | `2a6a0ba5` | 192.23 TH/s | +1.70% ±0.05 | +1.95% ±0.05 | 2812 → 2865 MHz | 262.6 W |
| RTX 5060 (145 W) | `b90400a2` | 78.34 TH/s | +2.54% ±0.07 | +2.84% ±0.07 | 2687 → 2760 MHz | 128.6 W |

  - Each ran 37 rounds, ahead in every one. The release's copy read +0.008%, +0.005% and +0.12%.
  - The 5080, 5070 Ti and 5060 run the release within 2-3% of their top clock. With the operand traffic gone they
    reach it, below their power limits, so an energy saving can win them at most 2-3.3%.
  - Only the 5090 stays power-limited all the way, with the full 8%. For item 16 that means the prize is mostly the
    5090's (and any card held to a lower limit), and a rewrite that costs work per clock would lose on the mid cards.
  - Three high-priority boxes for about 18 minutes each, about $0.30.
- **Step 2: neither way to move fewer operands works on a 5090.** At 00:10 UTC on 2 November, at `b68b039`, on
  `nvtest-rtx5090-12` (stock 5090, PC `9888e456`, 600 W of a default 600 W, at the limit in every turn):
  - **Drawing B' on the SM from a small table.** `PEARL_TALL_SHAM_DRAW` leaves B' unloaded and reads each B
    fragment a second time from shared memory with one IADD3 a register, the work a real draw would add (it leaves out
    the draw's table load and tile-seam wait, so it reads high). Against the release's 401.84 TH/s at 2440 MHz:
    skip B' +7.65% ±0.10 (2621 MHz); the sham draw **+1.81% ±0.11** (2587 MHz), ahead in all 37 rounds; the copy of
    the release −0.02%. The clock comes back, but the extra reads cost about 4% of work per clock. The draw had to
    clear +2.5% to be worth building, so it is dropped.
  - **Sharing B' between two CTAs by TMA multicast.** `multicast-probe.cu` (a `compilers.sh` stage) launched 85
    two-CTA clusters on all 170 SMs (the cluster query returned success for every arm). The arms that load their own
    boxes, only rank 0's, or nothing all ran; the arm that multicasts B' to both CTAs never finished, and its
    watchdog never fired, so the stuck wait is the multicast's receiver on rank 1. It differs from the plain-load
    arm only in its load: the multicast's bytes or its transaction count never arrive on this card. That is the
    cluster build's hang, and CUTLASS keeps RTX 50 to one-CTA clusters. Every multicast option is dropped.
  - **Item 16 stops here.** The operand traffic does cost a stock 5090 about 8%, but on-SM generation costs more work
    than it saves, multicast doesn't complete, and the mid cards would gain 2-3% at most. One high-priority box for
    about 20 minutes, about $0.14.

## A100-class (sm_80)

Run on 2026-10-10, 16:06–16:46 UTC. Four rows are on their own machine; one
moved.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| A100 PCIe 40 GB | Japan (250 W) | 146405 | 211.1 | 213.7 | 215.3 | 98% |
| A100 SXM4 40 GB | New York (400 W) | 152837 | 245.5 | 251.8 | 253.4 | 97% |
| A100 SXM4 40 GB | Mississippi (400 W) | 117301 | 246.6 | 252.8 | 253.8 | 97% |
| A30 24 GB | Australia (144 W) | 145742 | 100.5 | 90.1 | 90.8 | 111% |
| CMP 170HX | Georgia (250 W) | 147823 | 163.4 | 165.5 | 165.3 | 99% |

- **New host,** with the row's earlier host and machine ID:
  - A100 SXM4: Mississippi (117301) for California (152356), which had no
    offer.
- **A100s:** all three miners ran at the power limit. In the per-minute
  samples we held 15–30 MHz more than SRBMiner, so its 2–3% lead is work per
  clock. The raw files of all three A100 rows were lost (see "How a host is
  tested").
- **A30:** the host now enforces 144 W; before this run the row listed
  155 W. It also lowered the limit during the competitors' runs. Ours had
  144 W throughout. PeakMiner dropped to 129–136 W from 166 s on, and its
  end figure was taken at 135 W; at 144 W it read 94.0–94.3. SRBMiner ran
  at 140 W. Against PeakMiner's 144 W reading, ours is about 107%, not 111%.
- **CMP 170HX:** the card reports 64 GB, where a stock card has 8 GB, so
  this is an unlocked card.

## Hopper (sm_90a)

Run on 2026-10-10, 16:01–16:21 UTC. Two rows are on their own machine and
the H100 SXM moved. The H200 and H200 NVL weren't run.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| H100 SXM 80 GB | Czechia (575 W) | 148671 | 682.8 | 697.7 | 688.1 | 98% |
| H100 NVL 94 GB | Japan (400 W) | 29785 | 536.5 | 548.6 | 539.1 | 98% |
| H100 PCIe 80 GB | Czechia (350 W) | 147981 | 484.9 | 448.3 | 488.0 | 99% |
| H200 141 GB | — | — | — | — | — | not run: over budget |
| H200 NVL 141 GB | — | — | — | — | — | not run: over budget |

- **New host,** with the row's earlier host and machine ID:
  - H100 SXM: Czechia (148671) for New York (153443, 700 W), which had no
    offer. Czechia sets 575 W, 82% of the card's 700 W.
- **Clocks:** in the per-minute samples all three miners ran at about the
  same clock on the H100 SXM (1320–1410 MHz) and the H100 NVL
  (1140–1155 MHz), so PeakMiner's 2% lead there is work per clock. On the
  H100 PCIe we held 1155–1230 MHz to SRBMiner's 1110–1140.
- **H100 PCIe:** PeakMiner's end figure, 448.3, is a dip below its mean
  after the first minute (462.5). SRBMiner is the faster competitor either
  way.
- **Raw files:** lost on all three rows (see "How a host is tested", which
  also covers the H100 NVL's hit check).
- **Not run, over budget:**
  - H200 141 GB: the only verified offers were 4- and 8-GPU boxes, from
    $20.00/h, so one pass cost more than the budget. Its last figures, from
    Saudi Arabia (700 W, 131919): ours 719.9 on 2026-10-09 with PR #256's
    build at `466ccf0`, which has v0.5.14's Hopper fold, against PeakMiner
    2.17.6 at 718.6 and SRBMiner 3.7.1 at 720.4 on 2026-10-08: 99.9%.
  - H200 NVL 141 GB: its machine had no offer, and the cheapest verified one
    ($4.26/h) cost more than the budget left. Its last figures, from Quebec
    (600 W, 153365): ours 682.9 on 2026-10-09 with `466ccf0`, against
    PeakMiner 2.17.6 at 669.8 and SRBMiner 3.7.1 at 675.3 on 2026-10-08:
    101%.

## Ada workstation and data-center cards (sm_89)

Run on 2026-10-10, 16:22–17:46 UTC. The New York L40 ran at 20:45–21:05 UTC,
once its machine could be rented. Eight rows are on their own machine and
four moved; the Vietnam L40 wasn't run.

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

- **New hosts,** with the row's earlier host and machine ID:
  - L40S: Romania (150627) for Taiwan (151291).
  - RTX 6000 Ada: Germany (152478) for Germany (146274).
  - RTX 5000 Ada: Missouri (107776) for Minnesota (151333).
  - L4: Czechia (109442) for Czechia (116596).
- **L40, Vietnam: not run.** Its machine (114318) had no offer, and searches
  from 20:22 to 21:05 UTC found no L40 for rent other than New York. Its
  last figures, from Vietnam (300 W, 114318): ours 172.9 on 2026-10-08 with
  PR #256's build at `1040668`, which has v0.5.14's core for these cards,
  against PeakMiner 2.17.6 at 166.5 and SRBMiner 3.7.1 at 170.4 on
  2026-10-07: 101%.
- **L40:** half the L40S's int8 tensor rate per SM, so its rates are lower.
  The New York host sets 250 W of the card's 300 W.
- **RTX 4500 Ada:** the card ran at 88 C for every miner, and SRBMiner drew
  about 191 W of 210 W, so heat likely held this run.

## RTX PRO Blackwell (sm_120)

Run on 2026-10-10, 16:21–18:23 UTC, on the CUDA 13 core. Four rows are on
their own machine; three moved.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX PRO 6000 Workstation | Mississippi (600 W) | 148552 | 398.7 | 389.1 | 419.2 | 95% |
| RTX PRO 6000 Server | Australia (600 W) | 139338 | 435.1 | 432.4 | 435.9 | 99.8% |
| RTX PRO 6000 Max-Q | Australia (300 W) | 142021 | 308.0 | 302.9 | 306.7 | 100% |
| RTX PRO 5000 48 GB | California (250 W) | 150611 | 211.5 | 208.3 | 214.8 | 98% |
| RTX PRO 4500 | Arizona (200 W) | 148915 | 175.7 | 171.2 | 176.0 | 99.8% |
| RTX PRO 4000 | France (145 W) | 147490 | 127.9 | 126.5 | 128.8 | 99% |
| RTX 6000D | Czechia (550 W) | 149788 | 144.6 | 144.6 | 144.6 | 100% |

- **New hosts,** with the row's earlier host and machine ID:
  - RTX PRO 6000 Server: Australia (139338) for California (150951).
  - RTX PRO 6000 Max-Q: Australia (142021) for Illinois (153399).
  - RTX PRO 4000: France (147490) for Pennsylvania (150214).
- **RTX PRO 6000 Max-Q:** SRBMiner's end figure is a dip in its last sample:
  it read 312.9–314.4 from 60 s to 290 s, then 306.7. Against its mean
  after the first minute (313.18) the row would read 98.4%.
- **RTX 6000D:** Vast's name for a cut-down card with 156 SMs and 84 GB. All
  three miners sat at 2422 MHz, just under its 2430 MHz top clock, using
  234–254 W of 550 W, so they tie.
- **RTX PRO 5000 48 GB:** the host sets 250 W, 83% of the card's 300 W.

## Ampere workstation and data-center cards (sm_86)

Run on 2026-10-10, 16:21–18:26 UTC. Six rows are on their own machine; four
moved.

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

- **New hosts,** with the row's earlier host and machine ID:
  - RTX A6000: Vietnam (140998) for Italy (47281), and Romania (37274) for
    Kansas (38333).
  - A40: Belgium (151322) for Belgium (150273).
  - RTX A4000: Estonia (145734) for Kazakhstan (147920).
- **RTX A5000, Italy:** the host sets 200 W, 87% of the card's 230 W.
- **RTX A4000s:** both cards ran at 89–93 C.

## B200 (sm_100)

Run on 2026-10-10, 16:47–17:06 UTC, on the row's own machine. The CLI loads
the CUDA 12.8 core on this compute 10.0 card.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| B200 180 GB | Oregon (1000 W) | 96942 | 457.9 | 1380.6 | 1394.7 | 33% |

- **Why it's slow:** the B200 runs the TMA tall fold that RTX 50 cards mine
  on, built for sm_100. That fold uses `mma.sync`, and a probe on this host
  on 2026-10-09 topped out at about 568 TH/s with it, about 41% of SRBMiner.
  PeakMiner and SRBMiner ran at the 1000 W limit at about 1490–1510 MHz and
  still did three times our rate, so they use the B200's own tensor
  instructions (tcgen05). Matching them needs a tcgen05 fold.
- **Clocks and power:** ours ran at 1965 MHz and about 852 W, under the
  1000 W limit.

## Tesla T4 (sm_75)

Run on 2026-10-10, 16:44–17:04 UTC, on the row's own machine. The CLI loads
the CUDA 12.8 core on this compute 7.5 card.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| Tesla T4 16 GB | Czechia (70 W) | 28909 | 30.1 | 32.5 | 33.1 | 91% |

- **Clocks and power:** all three miners ran at the card's 70 W limit, which
  is its default and maximum: ours at 825 MHz, the competitors at 840 MHz.
- **Hit check:** the longer check ran on another T4 in the same machine (see
  "How a host is tested").
- **Band depth:** this row ran v0.5.14, which mines the T4's 4 MB L2 at a band
  of 4 column groups. The core after v0.6.6 uses 5 on 4 MB cards, on the
  strength of an RTX 2070 and 2080 (see the RTX 20-series notes). It also
  runs the second RTX 20 fold kernel on 4 MB cards (same notes). No T4 has
  run either.

## GTX 10-series (Pascal, sm_61, on Salad)

Pearl can't be mined on these cards with PeakMiner or SRBMiner, so both are 0. PeakMiner's README says
"Pearl (PRL) requires RTX 20xx or newer": since its v2.11.0 the network enforces rank-128 consensus rules that
its Pascal kernels can't mine. SRBMiner's docs list Pascal among its architectures but say nothing about
pearlhash on it.

One card was measured, a GTX 1080 on Salad PC `c868132d` (180 W), on 17 October with `tuning/salad/bench.sh`
(5 minutes each on `us.pearl.herominers.com` after a 60-second warm-up). PeakMiner 2.17.6 quit within 15
seconds, twice, with no shares. SRBMiner 3.7.1 ran the full 5 minutes at 0 H/s with no shares, the card idle
at 8 W and 139 MHz. The other rows are taken as 0 from that and PeakMiner's docs, without a run. Until v0.6.8
our released miner had no Pascal build (its lowest target was sm_75, and its kernel uses int8 tensor cores, which
Pascal lacks). v0.6.8 adds one, a fold on Pascal's `dp4a` instruction in its CUDA 12.8 core, but the release
hasn't been run on these cards by "How a host is tested", so "Ours" is "not run". The same core was measured
before the release; see the notes under the table.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| GTX 1080 Ti | Salad | not run | not run | 0 | 0 | — |
| GTX 1080 | Salad (180 W) | c868132d | not run | 0 | 0 | — |
| GTX 1070 | Salad | not run | not run | 0 | 0 | — |
| GTX 1060 | Salad (6 GB) | not run | not run | 0 | 0 | — |
| GTX 1050 Ti | Salad (4 GB) | not run | not run | 0 | 0 | — |

- **Our Pascal core, before its release (17–18 October).** PR #261 adds a fold on Pascal's `dp4a` instruction
  for sm_61, released in v0.6.8. These runs came before the release and by a different method from "How a host
  is tested" (`tuning/box.sh` on a mining box, the CLI picking its own HeroMiners host, rates from the miner's
  minute lines), so they stay here and the rows read "not run" until the release is run on these cards.
- **GTX 1080 Ti, Salad PC `af9f7a6e` (250 W), commit `d425085`:** the hit check verified 156 of 156 hits, 24
  salts in 90 s, 20.4 TH/s at the end on a cool card. Mining: **17.8 TH/s**, the mean of the minute samples
  over 5 minutes after a 1-minute warm-up (17.5–18.0), the card at 100%, 214–222 W, 82–83 °C and 1594–1620
  MHz (1911 MHz when cool). The pool accepted its first share at 4 minutes, 0 rejected.
- **Same PC, commit `4c48c3d`** (the first version: operands row-major rather than in 32-byte k-blocks, and
  one GPU wait per read of a hit rather than one per hit): 139 of 139 hits, 22 salts, on the warm card.
  Mining: 17.5 TH/s by the same method (17.3–17.6), at 236–237 W, 82 °C and 1620 MHz. So `d425085` is at most
  about 2% faster.
- **A first 1080 Ti PC (`2afc1bcc`, 200 W) isn't a result.** Both commits verified every hit there (7 of 7,
  three checks) but saw one salt in 90 s, about 0.9 TH/s. Its GPU ran at 93–95% and 160–186 W even while only
  the release was on it, which has no code that runs on compute 6.1, so another load on that PC was using the
  card.

## GTX 16-series (Turing without tensor cores, sm_75, on Salad)

Run on 17 October 2026, 20:09–22:33 UTC. These cards are rented on Salad, not Vast: each row is one Salad PC
(a home computer), named by the first 8 characters of its Salad machine ID, with the power limit from
`nvidia-smi`. Salad doesn't say where a PC is. Each PC ran PeakMiner 2.17.6, SRBMiner 3.7.1 and our release
v0.6.0 one after another, 5 minutes each after a 60-second unmeasured warm-up, on `us.pearl.herominers.com`,
PeakMiner and SRBMiner alternating which went first (`tuning/salad/bench.sh`).

All three miners run at the same speed on every card, 0.7–1.3 TH/s, with the card drawing only 28–58 W.
These cards are sm_75, which our release is built for, but they have no tensor cores: the tensor
instructions every miner uses run on a slow shared unit, about 25–30 int8 multiply-adds per clock per SM
against 1,024 on an RTX 2080 Ti. No miner found a share in 5 minutes (at 1 TH/s one share takes about 2.5
hours). A dp4a path could reach 5–9 TH/s, but even that earns less than Salad's cheapest price for these
cards ($0.015–0.02/hr), so it isn't being built. Our CLI prints its rate to one decimal, so "% of best" is
only good to about ±5%.

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

- **GTX 1660 Ti** rows came from Salad's "GTX 1660" class, which handed out 7 Tis in 10 PCs. The 741f9c56
  PC's driver doesn't report a power limit.
- **The second GTX 1660** wasn't found in 10 PCs. Of the two plain 1660s, one (e3babbec) couldn't keep a
  connection to HeroMiners (PeakMiner's authorize timed out, SRBMiner had a rate for 15 seconds, ours never
  got a job), so it isn't a result.
- **The second GTX 1650** wasn't run: the benchmark was stopped after the first, since every 16-series card
  gives the same answer.

## AMD Radeon (RDNA 3 and 4, on Salad)

Run on 18 October 2026, 00:04–02:28 UTC. These are the AMD cards Salad rents. Each row is one Salad PC (a home
computer that runs containers in WSL2), named by the first 8 characters of its Salad machine ID. Each PC ran
SRBMiner 3.7.1 and then PeakMiner 2.17.6, 5 minutes each after a 60-second unmeasured warm-up, on
`us.pearl.herominers.com` (`tuning/salad/bench.sh` at `8da98db`, on the `rocm/dev-ubuntu-24.04:7.2` image).
Our releases were CUDA only until v0.6.8, which ships the AMD core below for Linux (`pearl_core_hip.node`,
loaded with `--core amd`). Its release build is the `ubuntu-latest-hip` job of `.github/workflows/native-core.yml`,
in the same image and with the same compiler (ROCm 7.2.0) as the boxes' builds below, so the device code is the
code these rows ran. The host code differs only in two guards and an environment variable's parsing (the notes).

Every row was rerun from 20:24 UTC on 22 October to 00:19 UTC on 23 October, as the founder asked: our
AMD core, before its release, and SRBMiner 3.7.1 on the same PC, 5 minutes each after a 60-second warm-up, the order
alternating between PCs, after a hit check of the core (10 minutes on the first five PCs, 2 on the last five;
`tuning/salad/bench.sh` at `f78d37a`, the same image and pool). PeakMiner wasn't run, as asked: it mines nothing
on AMD (below), so its 0 stands. Two of the ten PCs are the 18 October rows' own; Salad didn't hand out the
others, so those rows moved to new PCs, and the 18 October figures are in the notes.

Every row was run again between about 06:45 and 10:45 UTC on 23 October, as the founder asked, after work on our AMD core:
our core and SRBMiner 3.7.1 on the same PC, 5 minutes each after a 60-second warm-up of the first, the order
alternating between PCs, after a 5- or 10-minute hit check of the core (`tuning/salad/amdtest.sh`'s `duel` task,
the same method and pool as on 22 October). PeakMiner wasn't run; its 0 stands. Salad handed out the 22 October
rows' own PC only once (RX 7900 XT `67c308d2`), so the other rows moved to new PCs, and the 22 October figures are
in the notes. "% of best" is ours divided by SRBMiner's rate, since PeakMiner mines nothing on these cards.

Every row was run again from 07:29 to 11:36 UTC on 30 October, with the v0.6.9 release's AMD core against SRBMiner
3.8.0 (3.8.1 and 3.8.2 list no change for pearlhash on AMD): the core built on each PC from the v0.6.9 commit (`d28faf2`) with ROCm 7.2 and run by the CLI
from source, after a 2-minute hit check, then both miners 5 minutes each after a 60-second warm-up, the order
alternating between PCs (`tuning/salad/bench.py` and `bench.sh` at `f60a455`, the same image and pool). PeakMiner
wasn't run; its 0 stands. Three rows are on the 23 October rows' own PCs (`67c308d2`, `cc2f03b6`, `d0dbba63`); the
others moved to new PCs, and the 23 October figures are in the notes.

SRBMiner mines Pearl on every one of these cards, through AMD's OpenCL. PeakMiner started on all ten PCs and
quit within 15 seconds with no hashrate and no shares: its README lists AMD support only for CSD,
Alphanumeric and Quantus, not Pearl, so it's 0.

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

- **The 30 October rerun, on the v0.6.9 release.** "Ours" is the CLI's mean hashrate after the first minute of
  its 5; SRBMiner 3.8.0's is its own figure at the end. Every PC passed the 2-minute hit check, 0 failed.

  | Card | PC | Copy | Hits a salt | First | Ours | SRBMiner | Ours ÷ SRBMiner | 23 October (ours / SRBMiner 3.7.1) |
  |---|---|---|---|---|---|---|---|---|
  | RX 9070 XT | `a68b3382` | 559 GB/s | 7.89 (1,538 in 195 salts) | ours | 114.45 | 111.35 | 102.8% | new PC, in place of `83151dba` (88.64 / 69.27) |
  | RX 9070 XT | `21001ac3` | 480 GB/s | 7.82 (1,564 in 200) | SRBMiner | 117.35 | 117.43 | 99.9% | new PC, in place of `77904d36` (109.88 / 92.43) |
  | RX 9060 XT | `f9b8be8b` | 260 GB/s | 7.64 (764 in 100) | ours | 61.22 | 56.10 | 109.1% | new PC, in place of `d324328f` (63.86 / 43.24) |
  | RX 9060 XT | `cff5c764` | 267 GB/s | 7.72 (811 in 105) | SRBMiner | 61.74 | 56.19 | 109.9% | new PC, in place of `0a4940aa` (59.28 / 43.22) |
  | RX 7900 XTX | `9a57035a` | 647 GB/s | 7.51 (668 in 89) | ours | 48.78 | 52.99 | 92.1% | new PC, in place of `bd6ca862` (46.92 / 45.78) |
  | RX 7900 XTX | `45c94e40` | 747 GB/s | 7.54 (686 in 91) | SRBMiner | 53.14 | 53.87 | 98.7% | new PC, in place of `2c8dc7f3` (44.24 / 41.87) |
  | RX 7900 XT | `cc2f03b6` | 672 GB/s | 7.57 (689 in 91) | ours | 53.11 | 51.42 | 103.3% | the same PC: 51.65 / 50.09 |
  | RX 7900 XT | `67c308d2` | 622 GB/s | 7.41 (600 in 81) | SRBMiner | 47.21 | 47.85 | 98.7% | the same PC: 47.20 / 46.88 |
  | RX 7800 XT | `d0dbba63` | 492 GB/s | 6.94 (361 in 52) | ours | 30.22 | 31.21 | 96.8% | the same PC: 32.84 / 31.80 |
  | RX 7800 XT | `53c8eb50` | 494 GB/s | 6.94 (368 in 53) | SRBMiner | 31.16 | 30.87 | 100.9% | new PC, in place of `56ae1be5` (33.39 / 32.38) |

  - **SRBMiner 3.8.0 closed most of our lead on RDNA 4.** It read 111–117 on the 9070 XT PCs (69–92 on 23
    October) and 56 on both 9060 XT PCs (43), on different PCs. Ours read 114–117 and 61–62 there, in line with
    the 23 October code. On RDNA 3, on the three PCs both runs used, SRBMiner moved −2% to +3% and ours −8% to
    +3%; the 7000 cards are now within a few percent either way.
  - **Hidden group name:** Salad holds a group named `bench-rx9070xt-1` from an earlier run that it can't show or
    delete (HTTP 404), so the first box took `bench-rx9070xt-1r1`.
  - **Slow PC, moved:** the second 9060 XT box got `d324328f` (a 23 October row's PC) and hadn't finished 42
    minutes later, so it moved to a new PC. That PC isn't a row.

- **SRBMiner's figure** is its own hashrate at the end of the run, read from its API. After the first 30
  seconds every run stayed within about 4% of it, apart from one RX 7800 XT (`56ae1be5`, 28.7–33.0). The pool
  accepted its shares on 8 of the 10 PCs, 0 rejected; the other two found none in 5 minutes, which is
  normal at their rates (a share is 2^53 hashes, about 3 minutes at 50 TH/s).
- **No power figures.** Salad's `amd-smi` (mounted at `/opt/rocm-wsl`) answered `static --json` with nothing
  and `monitor` with only its header line, so the Host column has no power limit and the box logged no draw
  or clocks. `rocminfo` named every card; AMD's OpenCL listed each as one device.
- **SRBMiner's own checks don't work here, but mining does.** `--list-devices` printed nothing on every PC.
  On the first PC SRBMiner's warm-up run quit after 15 seconds with no output, and the measured run after
  it mined normally.
- **Hosts not used:** an RX 7900 XT (`8e4144bb`) ran SRBMiner at 43.4 TH/s for 15 seconds, then at 0 with no
  shares for the rest of the run (it lost the pool or the GPU stalled), so it isn't a result; the next two
  7900 XTs held 44–48. Salad took another 7900 XT PC (`8dae4b18`) back before its image downloaded. The
  "AMD RX 7900 XTX" class once handed out a PC with an NVIDIA RTX 2080 Ti (`4f7066fe`), which was skipped.
- **The 23 October rerun.** PR #261's HIP core after a night of work on it (below), built on each PC with ROCm
  7.2 from the commit in the table, run by the CLI from source. "Ours" is the CLI's mean hashrate after the first
  minute of its 5; SRBMiner's is its own figure at the end, as before. Each PC first ran the core's fold check
  (`pearl_amd_foldtest` at 2048 × 2048) and hit check (`verify-hits.js`, 300 or 600 s): every one passed, with 400
  hits checked and 0 failed. Both miners had shares accepted on every row they found one, none rejected.

  | Card | PC | Clock | Copy | Hits a salt | First | Ours | SRBMiner | Ours ÷ SRBMiner | Code | 22 October (ours / SRBMiner) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | RX 9070 XT | `83151dba` | 2400 MHz | 569 GB/s | 8.03 (3,058 in 381 salts) | SRBMiner | 88.64 | 69.27 | 128.0% | `189bc49` | new PC, in place of `51a4b5c2` (15.8 / 49.0); an 18 October row PC (SRBMiner 79.4) |
  | RX 9070 XT | `77904d36` | 2460 MHz | 561 GB/s | 8.00 (3,758 in 470) | ours | 109.88 | 92.43 | 118.9% | `534e285` | new PC, in place of `e412934f` (92.8 / 92.9) |
  | RX 9060 XT | `d324328f` | 2780 MHz | 279 GB/s | 8.05 (4,387 in 545; 600 s) | ours | 63.86 | 43.24 | 147.7% | `d24ffbf` | new PC, in place of `733bb3d9` (22.0 / 41.1) |
  | RX 9060 XT | `0a4940aa` | 2700 MHz | 260 GB/s | 7.90 (1,968 in 249) | SRBMiner | 59.28 | 43.22 | 137.2% | `534e285` | new PC, in place of `377fd8c1` (22.2 / 43.7) |
  | RX 7900 XTX | `bd6ca862` | 2304 MHz | 729 GB/s | 7.81 (1,593 in 204) | SRBMiner | 46.92 | 45.78 | 102.5% | `636169b` | new PC, in place of `fd4ebc44` (45.5 / 50.1) |
  | RX 7900 XTX | `2c8dc7f3` | 2482 MHz | 731 GB/s | 7.94 (1,517 in 191) | ours | 44.24 | 41.87 | 105.7% | `c7180a3` | new PC, in place of `44d3716f` (47.9 / 52.1) |
  | RX 7900 XT | `67c308d2` | 2175 MHz | 549 GB/s | 7.83 (1,573 in 201) | SRBMiner | 47.20 | 46.88 | 100.7% | `c7180a3` | the same PC: 44.6 / 47.3 |
  | RX 7900 XT | `cc2f03b6` | 2219 MHz | 648 GB/s | 8.00 (3,561 in 445; 600 s) | ours | 51.65 | 50.09 | 103.1% | `e63fd15` | new PC, in place of `dc8fcef5` (41.7 / 44.4) |
  | RX 7800 XT | `d0dbba63` | 2254 MHz | 499 GB/s | 7.96 (2,237 in 281; 600 s) | ours | 32.84 | 31.80 | 103.3% | `259bf7b` | new PC, in place of `3a0284e3` (29.2 / 31.3) |
  | RX 7800 XT | `56ae1be5` | 2254 MHz | 502 GB/s | 7.81 (1,117 in 143) | SRBMiner | 33.39 | 32.38 | 103.1% | `259bf7b` | new PC, in place of `75fd6379` (29.0 / 31.7); an 18 October row PC (SRBMiner 32.4) |

  - **Code.** Each row ran the same default code for its chip as the branch head `fd16a8c`. The lanes compared
    the compiled fold, hash and reference kernels: gfx1100 and gfx1101 are the same from `259bf7b` to `fd16a8c`,
    gfx1200 and gfx1201 the same from `d24ffbf` to `fd16a8c` (at most register names differ), and the host
    (`pearl_host_hip.cpp`) and the CLI haven't changed since `9d8c430`, apart from two guards added after the
    run (a profile that isn't whole tiles, and a batch-width override above the cap), neither on the mining path;
    the device code compiled for all four chips is the same with them. For the release, `PEARL_AMD_COL_BATCH` is
    read with a hand-written digit loop instead of `atoi`, which under glibc 2.38 tied the addon to Ubuntu 24.04;
    unset, as in every row, it changes nothing.
  - **What changed in the core.** The RX 9060 XT's gap was its cache: the fold read A' and B' row by row, and a
    batch's B' (8 MB) didn't fit the chip's L2, so the fold waited on memory (15.4 TMAC/s at the mainnet batch
    against 49.3 at a quarter of it). Reading them in per-tile order (`a2c3da4`, the default since `d27b281`) took
    the fold to 52.1 and the card from 22 to about 50 TH/s. On RDNA 4 the fold's slab barrier then waited for every
    load in flight and dropped the cache, about 10% of its time; since `d24ffbf` it waits only for shared memory,
    the next slabs' loads stay in flight, and the result is read out with DPP moves (RDNA 4 only): 10–17% faster than
    the old path on five PCs. On RDNA 3 the same loads ahead and DPP readout (`259bf7b`, gfx11 only) add 2–4%. A
    hit's proof is read with one wait instead of about 120 (`22dfa93`). Variants that measured no faster were
    removed.
  - **Earlier rows of this rerun, on code since changed.** RX 9060 XT `a777486d` 51.15 against SRBMiner 40.31
    (`d27b281`) and `3540c3e2` 49.80 against 41.84 (`7f65179`), and RX 9070 XT `4c48f1b8` 100.34 against 95.46
    (`8067ab5`) and `f7baf9ac` 100.86 against 94.27 (`9d8c430`): all on the RDNA 4 path before `d24ffbf`, the first
    three also on a host that ran each batch's hash on its own stream (`d4733d8`, taken out in `9d8c430` after it
    measured no faster).
  - **The margins on the 7000 cards are thin** (100.7–105.7%), and these cards slow as they heat: on `bd6ca862`
    ours ran at 51.16 on the cold card and its batches slowed from 22.2 to 23.6 ms over 3 minutes, for both miners
    alike, so which miner goes first matters by a few percent. SRBMiner was slow on `83151dba` (69.3 against 79.4
    there on 18 October and 92–95 on other 9070 XT PCs); ours ran at 81.3 there on the old RDNA 4 path.
  - **A slow PC is the PC.** A 7900 XTX PC, `99389558`, isn't a row: it copied memory at 102 GB/s (729–759 on the
    others), and both miners crawled (ours 3.56, SRBMiner 1.15 at the end). The core's timing trace
    (`PEARL_AMD_TRACE=1`) put each batch's fold at 260 ms of GPU time (21 ms normally) with the GPU idle 0.03 ms a
    batch: the GPU itself ran 12 times slower, so something else on the PC was using it. That fits `51a4b5c2`'s
    32% on 22 October.

- **The 22 October rerun.** PR #261's HIP core, built on each PC from `f78d37a`, run by the CLI from source.
  "Ours" is the CLI's mean hashrate after the first minute of its 5; SRBMiner's is its own figure at the end, as
  on 18 October. Each PC first ran the core's hit check (`verify-hits.js`, the way the pool checks a hit; 600 s on
  the first five PCs, 120 s on the last five): every one passed, with up to 400 hits checked and 0 failed. The clock is the card's top clock as HIP reports
  it; the copy is device memory read and written.

  | Card | PC | Clock | Copy | Hits a salt | First | Ours | SRBMiner | 18 October |
  |---|---|---|---|---|---|---|---|---|
  | RX 7900 XTX | `fd4ebc44` | 2304 MHz | 753 GB/s | 7.97 (2,877 in 361 salts) | ours | 45.5 | 50.1 | new PC, in place of `d3b480e3` (SRBMiner 52.8) |
  | RX 7900 XTX | `44d3716f` | 2371 MHz | 758 GB/s | 7.99 (2,997 in 375) | SRBMiner | 47.9 | 52.1 | SRBMiner 53.5 |
  | RX 7900 XT | `67c308d2` | 2175 MHz | 627 GB/s | 7.97 (2,902 in 364) | SRBMiner | 44.6 | 47.3 | SRBMiner 46.8 |
  | RX 7900 XT | `dc8fcef5` | 2075 MHz | 622 GB/s | 7.99 (2,829 in 354) | ours | 41.7 | 44.4 | new PC, in place of `2c7775f7` (SRBMiner 44.6) |
  | RX 9070 XT | `51a4b5c2` | 2400 MHz | 552 GB/s | 7.82 (1,917 in 245) | ours | 15.8 | 49.0 | new PC, in place of `83151dba` (SRBMiner 79.4) |
  | RX 9070 XT | `e412934f` | 2460 MHz | 558 GB/s | 7.81 (1,219 in 156; 120 s) | SRBMiner | 92.8 | 92.9 | new PC, in place of `3d6cb276` (SRBMiner 85.1) |
  | RX 9060 XT | `733bb3d9` | 2620 MHz | 268 GB/s | 6.53 (248 in 38; 120 s) | ours | 22.0 | 41.1 | new PC, in place of `e235d465` (SRBMiner 42.2) |
  | RX 9060 XT | `377fd8c1` | 2780 MHz | 264 GB/s | 6.53 (248 in 38; 120 s) | SRBMiner | 22.2 | 43.7 | new PC, in place of `3540c3e2` (SRBMiner 40.7) |
  | RX 7800 XT | `3a0284e3` | 2213 MHz | 490 GB/s | 6.94 (340 in 49; 120 s) | ours | 29.2 | 31.3 | new PC, in place of `56ae1be5` (SRBMiner 32.4) |
  | RX 7800 XT | `75fd6379` | 2213 MHz | 451 GB/s | 6.89 (317 in 46; 120 s) | SRBMiner | 29.0 | 31.7 | new PC, in place of `b9d70479` (SRBMiner 32.4) |

  - **RDNA 3:** our core ran at 91–94% of SRBMiner on all six RX 7900 XTX, 7900 XT and 7800 XT PCs, whichever
    miner went first.
  - **RDNA 4 splits by chip.** On the RX 9070 XT `e412934f` ours matched SRBMiner (92.8 against 92.9). On the RX
    9060 XT ours held 21–24 TH/s all through on both PCs against SRBMiner's 41–44, 51–54%, with no slow start,
    so the gap is in our core on that chip (gfx1200, which HIP counts as 16 CUs against the 9070 XT's 32), not in
    the PCs.
  - **The RX 9070 XT on `51a4b5c2` is 32%.** Our core ran at 1.6–3.6 TH/s for the first 3½ minutes of its 5, then
    at 34.4–35.7 to the end; SRBMiner held 48–50 all through. So at its full rate ours was about 71% of SRBMiner
    there, and the PC is slow for both: SRBMiner got 79–85 on the 18 October 9070 XT PCs, and ours 93.8 on
    `f7baf9ac`. The same PC ran ours at 35.2 in the morning's test. The slow start hasn't been seen on any other PC,
    and its cause isn't known.
  - An RX 9070 XT PC, `470832a4`, isn't a row: neither miner got work from the pool there (0 for 5 minutes each),
    though the core passed its hit check (8.03 hits a salt, 6,104 in 760 salts).
  - Before the rerun, the rows' "Ours" was each card's best figure from the morning's tests on other PCs, marked †:
    93.8 on both 9070 XT rows and 50.4 on both 7900 XTX rows, "not run" on the rest.
  - The first 7900 XTX PC, `ded998b6`, isn't a row: Salad took it back 30 s into SRBMiner's run. Before that its
    hit check found 8.03 hits a salt (3,230 in 402 salts), ours ran at 50.0 for 5 minutes (49.9–50.4, 1 share
    accepted, 0 rejected), and SRBMiner read 55.0 and 54.7 TH/s at 15 and 30 s.
- **Our AMD core before its release (22 October).** PR #261's HIP core (`earn/native/amd`), unchanged, at commit
  `5aa0069` (`2d540fe` on `51a4b5c2` and `ded998b6`; the core's sources are the same in both). It ran by a
  different method from "How a host is tested" (`tuning/salad/amdtest.py`: built on the box with ROCm 7.2;
  `hashrate.js` for 60 s after a 10 s warm-up; then the CLI from source on `prl-us.kryptex.network:7048`), on
  PCs other than the rows'. Each card's best figure was in the rows, marked †, as the founder asked, until the
  rerun above replaced them. Each PC passed the fold check
  (`pearl_amd_foldtest` at 2048 × 2048: wmma11 on gfx1100, wmma12 on gfx1201) and verified every hit it
  checked:

  | Card | PC | Hit check | Ours | Mining on Kryptex |
  |---|---|---|---|---|
  | RX 9070 XT | `f7baf9ac` | 400 of 2202 hits, 0 failed; 277 salts in 240 s | **93.8** | 92–94 TH/s for 31 min, 25 accepted, 0 rejected |
  | RX 9070 XT | `51a4b5c2` | 353 of 353, 0 failed; 51 salts in 120 s | 35.2 | not run |
  | RX 7900 XTX | `ded998b6` | 400 of 634, 0 failed; 84 salts in 120 s | **50.4** | not run |
  | RX 7900 XTX | `68d7c53e` | 400 of 469, 0 failed; 66 salts in 120 s | 38.3 | 40–41 TH/s for 50 min, 9 accepted, 0 rejected |

  - **The PC matters as much as the card.** `51a4b5c2` copied device memory at 226 GB/s, against 564 on
    `f7baf9ac`. The two 7900 XTX PCs differ by 24% on the same code.
  - **Kryptex's own count.** On `f7baf9ac` it counted 25 shares in 1,893 s, 119 TH/s at 2^53 hashes a share. On
    `68d7c53e` it counted 9 in 3,000 s, 27.0 TH/s, where 13.6 shares were expected at the CLI's rate; 9 or fewer
    happens about 13% of the time. Over both runs it counted 34 shares against 33.2 expected.
  - **Hits a salt, the one check of work done that doesn't rest on the core's own count.** The target makes 8
    a salt. `f7baf9ac` found 7.95, but the other three PCs found 6.92 (`51a4b5c2`), 7.55 (`ded998b6`) and 7.11
    (`68d7c53e`) in these 2-minute checks. The 10-minute checks of the 22 October rerun found 7.82–8.03 on all
    seven PCs that ran one, `ded998b6` and `51a4b5c2` included, so the card searches at the rate it reports; the
    low counts come from short checks, and the rerun's 2-minute checks on its last five PCs found 6.53–7.81 again
    (both 9060 XT PCs the same 248 hits in 38 salts). What makes them low isn't known.
  - These ran on other PCs than SRBMiner's; the rerun above compares the two on the same PCs.

### Our own builds on Salad, 1 November: the readout probe for two-lane regions

Run from 01:16 to 01:35 UTC on 2026-11-01 on `amdtest` boxes, at `86ac8f6`: the default core against
`PEARL_AMD_READOUT_DROP`, a timing-only build whose RDNA 4 fold skips the readout's cross-lane steps (its hits are
wrong, and it refuses to mine). It times the most that two-lane 32x8 hash regions (research item 2) could gain,
before anyone builds them. `test/sweep.sh`, `hashrate.js` 60 s after 20 s, the builds taking turns, 4 rounds; the
first round is not scored. Not rows: no competitor ran.

| Card | PC | Release (TH/s) | Probe | Scored rounds | Fold alone |
|---|---|---|---|---|---|
| RX 9070 XT | `d36593f1` | 110.8–111.6 | 112.6–113.1 | +1.48%, +1.50%, +1.62% | +3.0% |
| RX 9060 XT | `f9b8be8b` | 61.4–61.5 | 62.4 | +1.61%, +1.61%, +1.64% | +2.0% |

- **The gate passed:** at least +1.0% in all three scored rounds on both cards, so two-lane regions get built. The
  design review had put the gain at +0.8–0.9%; the readout costs more than it estimated.
- **The 9060 XT's first PC** timed the probe's fold at 11.7 TMAC/s against 59.6 for the default, once, before
  Salad moved the box to `f9b8be8b`, where the two timed 59.2 and 58.1. That PC wasn't used.
- **Cost:** two high-priority boxes for about 15 minutes each, about $0.08.

**Then the real thing, and why it was dropped.** Two-lane regions in the fold itself (`911d2ab`, Stage A: fold test
only), at 03:00 UTC on 1 November, on `amdtest-rdna4-10` (9070 XT) and `amdtest-rdna4-11` (9060 XT): both forms
checked bit for bit against a 32x8 reference on the card, then fold-only timing, 4 rounds of 20 launches at 131072 x
4096, the builds taking turns:

| Card | Register form, by round | Store form |
|---|---|---|
| RX 9070 XT | −0.23%, +0.32%, +0.21%, about level | −32 to −33% |
| RX 9060 XT | +0.12%, −0.02%, −0.73%, about level | −0.65 to −1.26% |

- CI's census on ROCm 7.2.0 shows the fold built as designed: 20 fewer VALU a readout copy in the slab loop
  (257 to 217), DPP ops 24 to 3, `v_permlanex16` 12 to 0, a 1-instruction keep step, no scratch, 212 VGPRs and
  still 6 waves a SIMD. The VALU it saved don't turn into time, where the probe that removed the same steps gained
  2-3% of fold time. What the probe removed beyond the VALU is every cross-lane dependency in the chunk loop, which
  no correct readout can.
- Under the decision's rule (median under +0.5% on both cards) the item stops: `911d2ab` is reverted, and the
  probe stays as the gate for the next readout idea.

**Readout v2, the next readout idea, and why it was dropped too.** Each lane XORs its words into the spare bytes of
the B staging rows with non-returning `ds_xor`, and the 8 lanes of a region combine there (`74f02de`, research item
4): no cross-lane VALU in the chunk loop at all. Raced at 04:38 UTC on 1 November on `amdtest-rdna4-12` (9070 XT,
PC `c1940c4c`) and `amdtest-rdna4-13` (9060 XT, `54b217a9`): every build checked bit for bit at 2048 x 4096 first
(256 blocks, so later blocks reuse shared memory; v2 and its 32-bit-atomic variant v2w passed), then fold-only
timing, 4 rounds of 100 launches, the probe in the same rounds as the ceiling:

| Card | v2 | v2w | probe |
|---|---|---|---|
| RX 9070 XT | −0.86%, −0.60%, −0.48% | about −2.2% | +1.49%, +1.40%, +1.45% |
| RX 9060 XT | −0.62%, −0.37%, −0.87% | about −2.1% | +1.24%, +1.81%, +1.54% |

- Both correct readouts lose a little to today's on both cards, while the probe reproduces its gain on new PCs.
  So combining the 8 lanes' words costs at least what today's cross-lane steps cost, however it is done; the
  probe's gain comes only from not combining, which a miner can't skip. The boxes were stopped before the hashrate
  rounds, which a fold-time loss can't turn into +1%. `74f02de` is reverted.
- **Cost:** two high-priority boxes for about 8 minutes each, about $0.05.

**k 4096, which passed.** Research item 5, step 1 (`5567f0e`): a k 4096 build of the fold (its readout rotates the
kept words every 16 chunks, as the spec's `mine.rs` does) against today's k 2048 core, each with its own profile k,
at 05:33 UTC on 1 November on `amdtest-rdna4-14` (9070 XT, PC `5ecd7586`) and `amdtest-rdna4-15` (9060 XT,
`f9b8be8b`). At rank 128 credit per MAC is the same at any k, so the MAC rate is the fair figure. Every build
checked bit for bit at its own k first; `hashrate.js` 60 s after 20 s, the builds taking turns, round 1 not scored:

| Card | Release (TH/s) | k 4096 (TH/s) | Scored rounds | Fold alone | New readout alone, k 2048 |
|---|---|---|---|---|---|
| RX 9070 XT | 112.4-112.7 | 113.64-113.68 | +1.11%, +1.03%, +1.05% | −5.0 to −5.5% | −0.36 to −0.41% |
| RX 9060 XT | 61.4 | 62.14-62.18 | +1.22%, +1.14%, +1.24% | level (−0.09 to +0.22%) | −0.37 to −0.40% |

- **It passed the gate** (+0.5% in all 3 scored rounds on both cards). The fold alone is level on the 9060 XT and 5%
  slower on the 9070 XT, but the separate hash kernel's work halves, and in the miner's own loop that wins on both.
- **Checks at k 4096:** `verify-hits` re-verified 400 of 400 hits with the k 4096 profile, as a pool would (145 and
  77 salts), and the reference fold matched. Registers unchanged (198 VGPRs, no spills, 6 waves a SIMD).
- **Next:** step 2, a per-card mining profile in the app so RDNA 4 cards mine at k 4096, then pool runs.
- **Cost:** two high-priority boxes for about 21 minutes each, about $0.12.

**k 4096 in the app (item 5, step 2).** RDNA 4 cards mine k 4096 on Linux (`c3ec3a5`, `7accd79`), checked on 1
November on `amdtest-rdna4-16` (RX 9070 XT) and `amdtest-rdna4-19` (RX 9060 XT), at `7accd79`. Shares went to the
pools through `sharetap.py`, and every one was then checked by the spec's own `verify_plain_proof` (`refverify.rs`).

| Check | RX 9070 XT | RX 9060 XT |
|---|---|---|
| Fold test at k 4096 and k 2048 | pass (`d36593f1`) | pass (`733bb3d9`) |
| `verify-hits` through the app's path, k 4096 | 400 of 400, 88 salts | 400 of 400, 50 salts |
| Stand-in pool, both k on one connection | 215/215 and 200/200 | 137/137 and 101/101 |
| Our CLI against the stand-in | 417/417 compressed, 178/178 raw (180 KB) | not run |
| Kryptex, k 4096 | 1 h: 33 accepted, 0 rejected, 1 connection | 1 h on `a4e51ee5`: 21/21, 1 connection |
| HeroMiners, k 4096 (`a68b3382`) | 30 min: 26 accepted, 0 rejected, 1 connection | not run |
| Kryptex, both k on one connection (`a68b3382`) | 30 min: 12/12 and 9/9, 0 unanswered | not run |
| The spec's verifier on every captured share | 33/33, 26/26, 400/400 | 36/36, 21/21, 400/400 |

Against today's release on the same PC, `test/sweep.sh` and `hashrate.js` 60 s after 10 s, the builds taking turns:

| Card | PC | One job | A job every 60 s | A job every 35 s | Fold alone |
|---|---|---|---|---|---|
| RX 9070 XT | `a68b3382` | +0.47, +0.70, +0.75%; again +0.72, +0.76, +0.75% | +0.76, +0.75, +0.75% | +0.75, +0.76, +0.74% | +0.7 to −0.5% |
| RX 9060 XT | `a4e51ee5` | +1.28, +1.31, +1.30% | +1.28, +1.31, +1.28% | not run | +0.12 to −0.15% |

- **Job switching costs nothing measurable:** a k 4096 job's operands and trees are twice as large, but with a new job
  every 35 or 60 s (HeroMiners' and Kryptex's median gaps here) the gain is the same as on one job.
- **The 9070 XT gains less than in step 1** (+0.75% here against +1.03 to +1.11% on `5ecd7586`); the 9060 XT matches
  it. Both are above the +0.5% gate.
- **One PC's hang-ups, traced:** on `733bb3d9` Kryptex closed the 9060 XT's connection 8 times in 90 minutes, each
  time with a k 4096 share waiting, so 8 of 36 shares got no reply (27 accepted, 1 stale). All 36 pass the spec's
  verifier. A k 2048 control on another 9060 XT PC (`f029e99c`) got 34 of 34 on one connection, and k 4096 on a third
  (`a4e51ee5`) got 21 of 21 on one connection, as did the 9070 XT on both pools, so the hang-ups point to that PC,
  not to k 4096 (k 2048 never ran on it). The miner's k fallback only counts hang-ups before the first accepted
  share, and wouldn't have caught it.
- **Salad:** three restarts went silent and needed a new PC (`realloc`), and two extra boxes created by a script
  mistake were deleted within a minute, before they mined.
- **Cost:** about 3 hours of a 9070 XT and 5 hours of a 9060 XT, high priority, about $1.20.

**RDNA 3: k 4096, batch width and the hash's share (research item 13, step 0).** No code change: at 11:36 UTC on
1 November, at `d9c70e8`, on two RX 7900 XTX boxes, `amdtest-rdna3-5` (PC `45c94e40`) and `amdtest-rdna3-6`
(`ded998b6`; its first PC, `57cc1801`, took 15 minutes over a 13-second fold timing and was replaced). `test/sweep.sh`:
the fold alone 3 rounds of 600 launches, then `hashrate.js` 4 rounds of 60 s after 20 s, the builds taking turns;
batch 128 ran last in each round, at run time (`PEARL_AMD_COL_BATCH`). Against the release in the same round:

| Arm | `45c94e40` (53.2-53.3 TH/s) | `ded998b6` (55.5-55.6 TH/s) |
|---|---|---|
| k 4096 | +0.45, +0.19, +0.47, +0.24% | −0.02, +0.29, +0.11, +0.20% |
| Batch 128 | +0.64, +0.58, +0.60, +0.47% | +1.44, +1.84, +1.46, +1.60% |
| Batch 512 | −0.24, −0.49, −0.13, −0.34% | −1.02, −0.83, −0.88, −0.79% |
| k 4096's readout at k 2048 | (built at k 4096 by mistake) | −0.23, −0.25, −0.18, −0.07% |

- **k 4096 is not for RDNA 3:** +0.34% and +0.15% on average, under the +0.3% it needed in every round on both PCs.
  Its fold alone at the same MACs read +0.08 to +0.25% on one PC and −0.74 to −0.96% on the other, against −0.14 to
  −0.54% for its readout alone at k 2048: no tile seam to win on RDNA 3.
- **Batch 128 won every round on both PCs** (+0.57% and +1.58% on average). Checked next as a build, with
  `verify-hits` at that width and on an RX 7800 XT, before it becomes RDNA 3's default.
- **The hash kernel's share** (`PEARL_AMD_TRACE`, 45c94e40): 0.15-0.17 ms of a 20.6 ms batch, about 0.8%, under the 1%
  that a fused fold and hash would need on RDNA 3.
- **Cost:** two high-priority boxes for about 35 and 50 minutes, about $0.25.

**Batch 128 as a build, and the ablations (item 13, step 1a).** At 12:36 UTC on 1 November, at `2cc4403`, on
`amdtest-rdna3-6` (RX 7900 XTX, the same PC `ded998b6`), `amdtest-rdna3-10` (RX 7800 XT, `16f1685c`) and
`amdtest-rdna4-16` (RX 9070 XT, `26b0a4ff`). Batch widths as builds (`-DPEARL_AMD_COL_BATCH`), `hashrate.js` 60 s
after 20 s, the builds taking turns, against the release's 256 in the same round:

| Card | Batch 128 | Batch 64 |
|---|---|---|
| RX 7900 XTX (55.5-55.6 TH/s) | +1.44, +1.67, +1.84, +1.62% | −0.77, −0.20, −0.16, −0.74% |
| RX 7800 XT (32.3-32.5 TH/s) | +0.12, +0.37, +0.25, +0.22% | −0.22, −0.12, −0.12, −0.12% |
| RX 9070 XT, k 4096 (107.3-107.4 TH/s) | −0.10, −0.04, −0.01% | not run |

- **Batch 128 becomes the default on gfx1100 and gfx1101** (`amd_col_batch`): ahead in every round on three PCs of
  two chips, and 64 slower on both. `verify-hits` at 128 checked 400 of 400 hits on the XTX (97 salts) and 387 of
  387 on the 7800 XT. RDNA 4 is level, as a 9060 XT was before, and keeps 256; gfx1102, gfx1150 and gfx1151 weren't
  measured and keep 256 too.
- **The ablations** (`PEARL_AMD_ABLATE`, timing only: their transcripts are wrong on purpose), the fold alone on the
  miner's operands (`SWEEP_OPERANDS=miner`), 4 rounds of 600 launches, scored from round 2, against the release:

| Card | No slab barrier (2) | No LDS fragment loads (8) | Operands in L2 (16) | PFX+TID | PFX+TID in L2 |
|---|---|---|---|---|---|
| RX 7900 XTX | +1.50, +1.92, +1.32% | +6.53, +6.76, +6.42% | +2.48, +2.86, +2.45% | −0.41, +0.07, −0.72% | +2.04, +2.73, +2.24% |
| RX 9070 XT, k 4096 | +5.77, +5.86, +5.80% | not valid (below) | | | |

- Every XTX build kept 6 waves a SIMD with no spills (240 VGPRs; 203 with bit 8, 219 with PFX+TID).
- **Item 14 (start skew on RDNA 3) passes its cheap gate:** the barrier is 1.3-1.9% of fold time, above the 0.5% it
  needed. Its real gate, whether the blocks on a WGP run in step, needs the clock stamps. The barrier costs RDNA 4
  three times as much.
- **Bit 8 on RDNA 4 is not a reading:** with no fragment loaded from LDS, the compiler dropped the LDS stores too
  (LDS 0 bytes, 8 waves against the release's 6, 170 VGPRs), so its +26 to +29% is no operand staging at a higher
  occupancy. It needs a real LDS read kept per slab before it can gate item 15.
- Operands held in L2 still gain about 2.7% over PFX+TID alone, so PFX doesn't hide RDNA 3's memory latency.
- **Cost:** three high-priority boxes for 25 to 35 minutes each, about $0.25.

**Item 15's gate, and the clock stamps (item 13, step 1b).** At 15:03 UTC on 1 November, at `3f67113`. Bit 8 now
keeps one real LDS fragment read a slab, so the compiler keeps the stores and loads that feed it. Each run first
reads every variant's registers and counts its fold's LDS stores, global loads and LDS loads in the assembly, and
drops any variant that differs from the release where it shouldn't.

| Card | Build | VGPRs | Waves | LDS | LDS stores, global loads, LDS loads | Kept |
|---|---|---|---|---|---|---|
| RX 9070 XT (`d7f2cec4`) | release | 198 | 6 | 36864 | 9, 12, 24 | yes |
| | bit 8 | 188 | 6 | 36864 | 9, 12, 3 | yes |
| | stamps | 218 | 6 | 36864 | 9, 13, 24 | no: VGPRs |
| RX 7900 XTX (`06139310`) | release | 240 | 6 | 36864 | 9, 12, 48 | yes |
| | stamps | 245 | 5 | 36864 | 9, 13, 48 | no: a wave less |

- **Item 15 passes its gate on RDNA 4:** with LDS fragment loads cut from 24 a slab to 3, the 9070 XT's fold at k 4096
  ran +11.88%, +5.99% and +8.10% faster (scored rounds of 600 launches, miner operands), against a 3% kill line. It's
  an upper bound: fragments that stop changing may let the card clock up, and this PC's release wandered from 103.4
  to 106.8 TMAC/s.
- **The stamps don't fit yet:** 20 more VGPRs on RDNA 4 and 5 on RDNA 3, where that costs a wave a SIMD. Every
  stamped build was dropped, so neither item 14's lockstep gate nor item 15 in cycles was read. They need to live in
  SGPRs.
- The XTX's PC folded at 25.3 TMAC/s, less than half the 58.5 of `ded998b6`, and its box was stopped.
- **Cost:** two high-priority boxes for about 8 minutes each, about $0.05.

**Item 15, built and dropped.** `PEARL_AMD_LDS_AHEAD` (`d6cb96c`) moved the next slab's LDS store and barrier one
k-step earlier, after k-step 0's WMMAs, so the next slab's fragments load from LDS while k-step 1 runs: form 1 loads
k-step 0's fragments early, form 2 both. The stamps moved to SGPRs in the same commit. Raced at 17:09 UTC on
1 November against the release on the same PC, at k 4096. First, every build's registers and assembly
(`test/variant-check.sh`):

- **Both forms fit:** 207 and 221 VGPRs against the release's 198, at 6 waves, no spills, LDS 36864, the release's
  LDS stores and global loads. 16 WMMAs ran after each slab barrier before an LDS wait, against the release's 0.
- **Both are bit-exact** against the CPU at k 4096 on both cards.
- On a 7900 XTX both forms compiled to the release (gfx11 is left alone).
- **The SGPR stamps still don't fit:** 221 VGPRs on gfx1201 against 198, and 244 at 5 waves on gfx1100.

Fold alone on the miner's operands, 600 launches a round, scored from round 2; then `hashrate.js` 60 s after 20 s:

| Card | PC | Form 1, fold | Form 2, fold | Form 1, hashrate | Form 2, hashrate |
|---|---|---|---|---|---|
| RX 9070 XT | `4c48f1b8` | −2.80, −2.74, −2.84% | −3.99, −3.98, −4.14% | not run | not run |
| RX 9060 XT | `b3e43d09` | −1.91, −1.69, −1.79% | −3.77, −3.35, −3.32% | −1.81, −0.41% | −2.33% |

- **Dropped:** slower on both cards in every round, so the boxes were stopped early and the code is reverted.
  Loading the fragments early is not what costs time. Moving the barrier and the next slab's store one k-step earlier
  gives the global loads that feed it less time to arrive (about 1.5 slabs instead of 2), and the barrier waits on
  them. The fragment-load ablation's +6 to +12% stays as a reading of what LDS waits cost, not of what this ordering
  could win.
- **Cost:** two high-priority boxes for about 15 minutes each, plus a 5-minute check, about $0.12.

**Item 14, built and dropped.** `PEARL_AMD_SKEW` (`a7fb599`) delayed each block of a launch's first wave once, by
its place on its WGP (a per-WGP counter; gfx12 takes the engine from `s_sendmsg_rtn`, its HW_ID1 has none), so the
three blocks a WGP holds would reach their slab barriers out of step. Its gate, clock stamps showing the blocks run
in step, couldn't be read (the stamps don't fit the registers), so the skew itself was raced, at 19:30 UTC on
1 November. The waits are percents of a slab, set at run time; `ctl` waits the same at every place, the same delay
with no skew.

- **It ran as designed:** on a 7900 XTX (`28a12f15`) the skew build was 236 VGPRs at 6 waves and its probe put 144
  blocks on 48 WGPs, three each, in places 0, 1 and 2; on a 9070 XT (`aa09f97d`) 209 VGPRs against 198, still 6
  waves (raced with that noted), 96 blocks on 32 WGPs as designed. Bit-exact at 1024 x 1024 and 2048 x 4096.
- **It helps one generation of blocks, then fades.** The fold alone, scored rounds 2-4, against the release:

| Card | One generation, best against `ctl` | Eight generations (`ctl`) | The mined launch |
|---|---|---|---|
| RX 7900 XTX | +1.09 points (0:8:17), +0.99 (0:14:38) | −0.15 to −0.46% (−0.23%) | −0.28 to −0.50% (one −6.6% clock dip) |
| RX 9070 XT, k 4096 | +5 to +10 points, rounds noisy | −0.84 to −1.80% (−1.28%) | −0.63 to −0.93% (rounds 2-3) |

- **Dropped:** by eight generations every wait reads the same as `ctl`, and at the launch the miner runs (about 114
  generations on the XTX) every wait is slower. The barriers pull the blocks back into step. On the 9070 XT the
  build itself is about 1% slower even as `ctl`, from its different schedule. The boxes were stopped before the
  hashrate rounds, and the code is reverted.
- **Cost:** two high-priority boxes for about 8 and 5 minutes, plus a 1-minute first run on the 9070 XT, about $0.05.

Salad's prices per hour on 17 October (batch / low / medium / high): RX 9070 XT $0.070 / 0.110 / 0.155 /
0.196; RX 9060 XT $0.049 / 0.075 / 0.100 / 0.126; RX 7900 XTX $0.066 / 0.100 / 0.134 / 0.168; RX 7900 XT
$0.054 / 0.088 / 0.122 / 0.156; RX 7800 XT $0.042 / 0.072 / 0.102 / 0.132.

## Hosts replaced in this run

These hosts were dropped, and the row ran on the next one by the same
rules.

- Didn't start its container within 10 minutes: RTX 3070 Ti Pennsylvania
  (136798, unverified, the only 3070 Ti offer at the time), RTX 3060 Romania
  (140517), RTX 4090 Finland (34105) and RTX 4080 Super California (143795).
- CUDA found no GPU, though nvidia-smi listed it, so the hit check and all
  three miners failed: RTX 4070 Ti Super Arizona (38773).
- Setup failed while installing packages: A100 SXM4 California (151502).
- Outbound port 1200 was blocked, so no miner could reach the pool: RTX 5070
  Quebec (37540) and L4 Brazil (151023).
- Went offline during the PeakMiner run: RTX 5070 France (152291). The RTX
  5070 row is on the third host tried, Texas (147729).
- Another workload was on the GPU: RTX 3090 Ti Spain (147387).
- Held by heat: RTX 3080 Utah (140965) and RTX 3060 United States (120972).
- Our test script failed, not the host, and the row moved to another
  machine: RTX 3060 Utah (131146); H100 SXM Czechia (140932), which was no
  longer listed by then and was also below the power rule at 525 W of
  700 W; RTX 4080 Super Indiana (143955), also below the power rule at
  250 W of 320 W; and L40S Taiwan (39563). The RTX 3090 (California,
  151414), H100 PCIe (South Africa, 149333) and RTX 6000 Ada (California,
  137575) went back to their own machines. Several other rows hit the same
  failure and were rerun on the same machine.
- Runs that finished on a new host and were replaced when the row's own
  machine came back (the Australia 4090 and the California 4080 Super) are
  in the 40-series notes.
