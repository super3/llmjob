# RTX benchmark

Our miner, v0.5.14, against PeakMiner 2.18.3 and SRBMiner 3.7.4 on rented
Vast hosts, one or two hosts per card. Every figure in the card tables comes
from one full rerun on 2026-10-10, 15:58–21:05 UTC, except six RTX 20-series
rows rerun on 2026-10-11 against SRBMiner 3.7.5. Each host runs all three
miners one after another on the same pool, so rates within a row compare
directly. Don't compare across rows: power limits and cooling differ from
host to host. v0.6.0 adds Monero mining on the CPU and leaves the GPU miner
as it was in v0.5.14, so these figures hold for it too.

## Summary

"% of best" is our rate divided by the faster of PeakMiner and SRBMiner on
the same host. 92 of the 95 rows ran. The H200 and H200 NVL weren't run
because one pass cost more than the budget left. The Vietnam L40 wasn't run
because no L40 was for rent other than the New York one.

- RTX 20-series: 92–99%. Our clock is never more than 30 MHz below the
  faster competitor's, so the gap is work per clock: we do 5–7.5% less.
  SRBMiner 3.7.5 also mines Monero on the CPU alongside, and so does ours
  from v0.6.0; PeakMiner doesn't (see the section).
- RTX 30-series: 97–101%. The two 3070s trail most, at 97%. SRBMiner didn't
  start on the Oman 3090 Ti, so that row compares with PeakMiner.
- RTX 40-series: 98–102%, and 99.5% or better on 15 of the 18 hosts. The
  lowest are the 225 W British Columbia 4080 and the Australia 4060, both at
  98%.
- RTX 50-series: 96–101%. The India 5090 (96%) has driver 570, so it runs
  the CUDA 12.8 core. Per clock it does 3.4% less than SRBMiner there,
  against 0.6% less on the South Korea 5090 on the CUDA 13 core. Those are
  two different hosts, so this doesn't prove the core is the cause.
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
  has no Pascal build either; an unreleased one mined at 17.8 TH/s on a GTX
  1080 Ti on Salad (the section's notes).
- GTX 16-series: 0.7–1.3 TH/s for all three miners (96–104%). The cards have no tensor cores, and
  every miner's tensor instructions run on a slow shared unit. Not worth mining on Salad.
- AMD Radeon on Salad: SRBMiner 32–85 TH/s, from the RX 7800 XT to the RX 9070 XT. PeakMiner doesn't
  mine Pearl on AMD (0), and our release is CUDA only. An unreleased AMD core ran at 91–94% of SRBMiner on
  the same PCs on the RX 7900 XTX (45.5–47.9 TH/s) and RX 7900 XT (41.7–44.6). On an RX 9070 XT it was 32%
  of SRBMiner on one slow PC (15.8 against 49.0, after a slow start), and it mined at 93.8 TH/s on another PC
  without SRBMiner (the section's notes).
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
at 8 W and 139 MHz. The other rows are taken as 0 from that and PeakMiner's docs, without a run. Our released miner
has no Pascal build (its lowest target is sm_75, and its kernel uses int8 tensor cores, which Pascal lacks), so
"Ours" is "no build". An unreleased Pascal build was measured afterwards; see the notes under the table.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| GTX 1080 Ti | Salad | not run | no build | 0 | 0 | — |
| GTX 1080 | Salad (180 W) | c868132d | no build | 0 | 0 | — |
| GTX 1070 | Salad | not run | no build | 0 | 0 | — |
| GTX 1060 | Salad (6 GB) | not run | no build | 0 | 0 | — |
| GTX 1050 Ti | Salad (4 GB) | not run | no build | 0 | 0 | — |

- **Our Pascal core, not released (17–18 October).** PR #261 adds a fold on Pascal's `dp4a` instruction for
  sm_61. It isn't in a release, and it ran by a different method from "How a host is tested" (`tuning/box.sh`
  on a mining box, the CLI picking its own HeroMiners host, rates from the miner's minute lines), so the rows
  stay "no build" until a release has it.
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
Our released miner is CUDA only.

The RX 7900 XTX and RX 7900 XT rows and one RX 9070 XT row were rerun on 22 October, from 20:24 UTC, as the
founder asked: our unreleased AMD core and SRBMiner 3.7.1 on the same PC, 5 minutes each after a 60-second
warm-up, the order alternating between PCs, after a 10-minute hit check of the core (`tuning/salad/bench.sh` at
`f78d37a`, the same image and pool). PeakMiner wasn't run, as asked: it mines nothing on AMD (below), so its 0
stands. Two of the five PCs are the 18 October rows' own; Salad didn't hand out the others, so those rows moved to
new PCs. The other rows aren't rerun yet. Their "Ours" is the core's best figure from other Salad PCs (marked †, see the
notes), put in the rows as the founder asked; that "% of best" compares across PCs, so read it loosely. Cards
the core hasn't run on say "not run".

SRBMiner mines Pearl on every one of these cards, through AMD's OpenCL. PeakMiner started on all ten PCs and
quit within 15 seconds with no hashrate and no shares: its README lists AMD support only for CSD,
Alphanumeric and Quantus, not Pearl, so it's 0.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RX 9070 XT | Salad (16 GB) | 51a4b5c2 | 15.8 | 0 | 49.0 | 32% |
| RX 9070 XT | Salad (16 GB) | 3d6cb276 | 93.8 † | 0 | 85.1 | 110% |
| RX 9060 XT | Salad (16 GB) | e235d465 | not run | 0 | 42.2 | — |
| RX 9060 XT | Salad (16 GB) | 3540c3e2 | not run | 0 | 40.7 | — |
| RX 7900 XTX | Salad (24 GB) | fd4ebc44 | 45.5 | 0 | 50.1 | 91% |
| RX 7900 XTX | Salad (24 GB) | 44d3716f | 47.9 | 0 | 52.1 | 92% |
| RX 7900 XT | Salad (20 GB) | 67c308d2 | 44.6 | 0 | 47.3 | 94% |
| RX 7900 XT | Salad (20 GB) | dc8fcef5 | 41.7 | 0 | 44.4 | 94% |
| RX 7800 XT | Salad (16 GB) | 56ae1be5 | not run | 0 | 32.4 | — |
| RX 7800 XT | Salad (16 GB) | b9d70479 | not run | 0 | 32.4 | — |

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
- **The 22 October rerun.** PR #261's HIP core, built on each PC from `f78d37a`, run by the CLI from source.
  "Ours" is the CLI's mean hashrate after the first minute of its 5; SRBMiner's is its own figure at the end, as
  on 18 October. Each PC first ran the core's hit check (`verify-hits.js`, 600 s, the way the pool checks a
  hit): every one passed, with 400 hits checked and 0 failed. The clock is the card's top clock as HIP reports
  it; the copy is device memory read and written.

  | Card | PC | Clock | Copy | Hits a salt | First | Ours | SRBMiner | 18 October |
  |---|---|---|---|---|---|---|---|---|
  | RX 7900 XTX | `fd4ebc44` | 2304 MHz | 753 GB/s | 7.97 (2,877 in 361 salts) | ours | 45.5 | 50.1 | new PC, in place of `d3b480e3` (SRBMiner 52.8) |
  | RX 7900 XTX | `44d3716f` | 2371 MHz | 758 GB/s | 7.99 (2,997 in 375) | SRBMiner | 47.9 | 52.1 | SRBMiner 53.5 |
  | RX 7900 XT | `67c308d2` | 2175 MHz | 627 GB/s | 7.97 (2,902 in 364) | SRBMiner | 44.6 | 47.3 | SRBMiner 46.8 |
  | RX 7900 XT | `dc8fcef5` | 2075 MHz | 622 GB/s | 7.99 (2,829 in 354) | ours | 41.7 | 44.4 | new PC, in place of `2c7775f7` (SRBMiner 44.6) |
  | RX 9070 XT | `51a4b5c2` | 2400 MHz | 552 GB/s | 7.82 (1,917 in 245) | ours | 15.8 | 49.0 | new PC, in place of `83151dba` (SRBMiner 79.4) |

  - On the RDNA 3 cards our core ran at 91–94% of SRBMiner on every PC, whichever miner went first.
  - **The RX 9070 XT on `51a4b5c2` is 32%.** Our core ran at 1.6–3.6 TH/s for the first 3½ minutes of its 5, then
    at 34.4–35.7 to the end; SRBMiner held 48–50 all through. So at its full rate ours was about 71% of SRBMiner
    there, and the PC is slow for both: SRBMiner got 79–85 on the 18 October 9070 XT PCs, and ours 93.8 on
    `f7baf9ac`. The same PC ran ours at 35.2 in the morning's test. The slow start hasn't been seen on any other PC,
    and its cause isn't known.
  - An RX 9070 XT PC, `470832a4`, isn't a row: neither miner got work from the pool there (0 for 5 minutes each),
    though the core passed its hit check (8.03 hits a salt, 6,104 in 760 salts).
  - Before the rerun, both 7900 XTX rows had "Ours" 50.4 †, from `ded998b6`.
  - The first 7900 XTX PC, `ded998b6`, isn't a row: Salad took it back 30 s into SRBMiner's run. Before that its
    hit check found 8.03 hits a salt (3,230 in 402 salts), ours ran at 50.0 for 5 minutes (49.9–50.4, 1 share
    accepted, 0 rejected), and SRBMiner read 55.0 and 54.7 TH/s at 15 and 30 s.
- **Our AMD core, not released (22 October).** PR #261's HIP core (`earn/native/amd`), unchanged, at commit
  `5aa0069` (`2d540fe` on `51a4b5c2` and `ded998b6`; the core's sources are the same in both). It ran by a
  different method from "How a host is tested" (`tuning/salad/amdtest.py`: built on the box with ROCm 7.2;
  `hashrate.js` for 60 s after a 10 s warm-up; then the CLI from source on `prl-us.kryptex.network:7048`), on
  PCs other than the rows'. Each card's best figure is in the rows, marked †, as the founder asked on 22 October;
  by the rules above it would stay out of them until it ran on the row's own PC. Each PC passed the fold check
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
    seven PCs, `ded998b6` and `51a4b5c2` included, so the card searches at the rate it reports; the low counts came from the short
    checks. What made them low isn't known.
  - Against SRBMiner on the rows' PCs: 110% of the best 9070 XT row, but on different PCs, so this doesn't show
    which miner is faster. The 7900 XTX figure here was replaced by the rerun's.

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
