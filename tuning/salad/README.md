# PRL mining on SaladCloud

`watcher.py` rents GPUs on SaladCloud, runs our miner on them, and keeps a ledger of what they cost and
what they mine. It runs one round a minute. Each round it reads every group's state, makes one log
query for all groups and one pool call, applies the rules below, and writes three files into its data
directory:

- `state.json`: the ledger (every group it created, spend, PRL mined, shares, readings).
- `events.log`: one line per action or notable change (created, got a PC, first share, reallocated,
  deleted, alerts) plus a `REPORT` line every `report_min` minutes (10 by default).
- `report.md`: the status table.

`keeper.sh` restarts the watcher if it stops. `config.json` in the data directory is re-read every round.

## Running it

```sh
export PRL_WALLET=prl1...            # payout address
export SALAD_API_KEY=...             # not needed in a Claude cloud session: its proxy adds the key
export SALAD_MINE_DIR=/some/dir      # default: tuning/salad/run/ (gitignored)
cp tuning/salad/config.example.json "$SALAD_MINE_DIR/config.json"
setsid nohup bash tuning/salad/keeper.sh >/dev/null 2>&1 </dev/null &
tail -F "$SALAD_MINE_DIR/events.log"
```

`SALAD_ORG` and `SALAD_PROJECT` default to `llmjob` and `default`.

Config keys:

| Key | Meaning |
|---|---|
| `max_active` | Most groups alive at once. Salad's quota caps it too, so set it above the quota and a raised quota is used at once (live: 10, the quota; 9 while one slot ran benchmarks on 17–18 October). |
| `search` | Run the hourly search. |
| `search_token` | Change it to run a search now. |
| `restart_token` | Change it after a top-up: restarts groups Salad stopped for lack of credit. |
| `credit_start`, `credit_offset` | Credit left = start + offset − spend. Set the offset when the portal shows a different balance. |
| `box_ver` | Miner release for every group. A running group on an older release is moved to it in place, by its command alone, as Gemma is: a box that is mining keeps its PC, where Salad usually restarts it within minutes (since 21 October; before, new groups only). `box_ver` never lowers a release. `ver_override` maps a group name to a release: to try one on a single group first, or to take that group back from a bad one, since it can lower it. On 21 October v0.6.7 was a bad one here: its Linux cores need glibc 2.38, and the boxes' Ubuntu 22.04 image has 2.35, so the first box moved to it ran no miner. |
| `wallet` | Payout address, if `PRL_WALLET` isn't set. |
| `report_min` | Minutes between `REPORT` lines. |
| `min_margin` | The margin every class and PC must clear (0.10). |
| `max_new_per_hour` | Most new groups an hour (3). |
| `max_per_class` | Most groups of one class (1). |
| `search_every_min` | Minutes between searches while a slot is empty (60). |
| `retry_after_min` | Minutes before a class released for want of a PC is tried again (60). |
| `retry_failed_h` | Hours before a class whose group failed is tried again (24). |
| `no_pc_streak`, `no_pc_hold_min` | A class whose last 3 groups got no PC isn't tried for 180 min. |
| `upgrade_gain` | How much more an hour a class must be expected to make before it takes the weakest group's slot (0.05). `null` turns swaps off. |
| `upgrade_every_min` | Minutes between swap checks while every slot is taken (15). |
| `swap_wait_min` | Minutes a swapped slot is expected to earn nothing while the new group gets a PC and starts (20). Part of a swap's cost. |
| `yield` | The share of the network-rate estimate the rules count on when they judge a margin: `"auto"` for the credit ratio of the pool each box mines on (below; for a new group, the pool it would start on), or a number for every box. Live: `"auto"` since 19 October (it was `0.9`, under what the pools credit: 92% and 93%). Until 22 October `"auto"` meant HeroMiners' ratio for every box, about 0.93 while every box mined on Kryptex at 0.97, so the rules left and deleted boxes that cleared the bar: 63 of 125 margin moves and deletions in the 48 hours before. |
| `yield_min_blocks`, `yield_floor` | The credit ratio is used once the pool has credited us in 10 blocks, and never below 0.5. |
| `bad_pc_hold_h` | Hours a PC that a group left stays on the shared bad list (24). |
| `rate_days` | Days a PC's measured rate counts toward its class's rate (7). |
| `drop_count`, `drop_min`, `drop_hours` | A PC Salad takes back within `drop_min` minutes of handing it over, `drop_count` times in `drop_hours`, goes on the bad list for every box (3, 10, 24). |
| `drift` | How far the ledger and the pool may disagree before a `DRIFT` event (0.15). |
| `trader` | Railway ids of the SafeTrade autoseller: `{"project", "service", "environment"}`. Unset: no SafeTrade line. |
| `safetrade` | Unused since 20 October: a hand-typed snapshot of the trader's 13 October 17:00Z log (46.79 PRL sold, 1.74 held). It is not a ledger. On 24 October it was read as one and made 130.8 PRL look missing; the trader's own STATUS line (178.486 PRL sold by 23 October 02:00Z) accounts for every PRL the pools had paid by then. |
| `xmr_wallet`, `xmr_groups` | Monero on the box's CPU (see [Monero](#monero-on-the-cpu)): the address, and the groups that mine it (`["*"]` for every group). Unset: no Monero, and the table is as before. |
| `xmr_ver`, `xmr_cpu`, `xmr_memory`, `xmr_threads` | A Monero box runs release `v0.6.0` or later on 8 vCPUs and 8192 MB. `xmr_threads` 0 lets the CLI choose (it keeps one CPU free for Pearl). |
| `kryptex_xmr_pay` | What Kryptex credits for Monero work, as a share of the network's pay for it (0.94, measured 21 October). The report's XMR column values each box's Kryptex work at it (see [Monero](#monero-on-the-cpu)). |
| `xmr_kx`, `xmr_kx_host`, `xmr_kx_ver` | Monero to Kryptex's PPS+ pool (`xmr-us.kryptex.network:7029`) instead of HeroMiners', with release `v0.6.6` or later (the first with `--xmr-pool`). A running group is switched by its command alone (see [Monero](#monero-on-the-cpu)). `false`: HeroMiners, and every command is as before. |
| `kryptex_groups`, `kryptex_host`, `kryptex_ver`, `kryptex_pay`, `kryptex_token` | How many groups mine Pearl on Kryptex instead of HeroMiners (see [Kryptex](#kryptex-pps)), its stratum (`prl-us.kryptex.network:7048`), the release a Kryptex box needs at least (v0.6.1, the first with `--pool`) and the share of each share's value it pays (0.94, measured; used until Kryptex has credited 1 PRL). Change `kryptex_token` to restart the test after it stopped. 0: none, and the table is as before. |
| `kx_in_place` | While fewer than `kryptex_groups` are on Kryptex, also switch boxes that are mining on HeroMiners, by updating their command: Salad usually restarts each one on the same PC (see [Kryptex](#kryptex-pps)). `false`: a mining box keeps its pool until its next PC. |
| `llm`, `llm_ver` | Gemma beside mining on every box (see [Gemma](#gemma-beside-mining)), with release `llm_ver` (v0.6.5) or later. `false`: mining only, and every command is as before. |

Tests: `python3 -m unittest discover -s tuning/salad`. They fake Salad, the pool and the clock.

## Rules

**Margin** is profit as a share of earnings: (earnings − price) / earnings, as the first account
measured it. A 2060 at 40 TH/s earning $0.047/hr on a $0.030/hr PC is +36%, not +56%. The bar is
10% (`min_margin`) everywhere: for the search and priority moves on a class's measured rate, and for
each PC on its own readings.

**A class's rate** is the median TH/s of the PCs it has mined on here in the last `rate_days`, each PC
at the median of its last 3 readings after warm-up, capped and slow PCs included. Before a class has mined, it's the
benchmark in `KNOWN`. The search, priority moves and swaps all use it. On 12 October the 3070 measured
77 TH/s against a 59 benchmark. With `yield` 0.9 the benchmark cleared 10% only at batch, so a second
3070 was released for want of a batch PC although 3070s clear low easily; the 2080 measured 64 against 72.

**Earnings** are TH/s × the network's $ per TH-hr × `yield`. The network's rate is what the work is
worth if the pool paid us exactly our share of every block. It doesn't: on 12 October, after 25 blocks,
the pool had credited 83% of that estimate, with the fee, orphaned blocks and luck all in the gap.
With `yield: "auto"` the rules count on that credit ratio. On that day it would have dropped the three
5080s and three 4080s at medium (+15–25% on the network's rate, −2% to +9% on 83% of it) and kept the
2080 Tis and the 3070, so it's set to `1.0` until we decide. The table always uses the credit ratio.

1. **Search** hourly at :01, and every `search_every_min` while a slot is empty. Take classes with a known rate that clear a 10% margin at the cheapest
   priority with a GPU free. Classes that paid before come first, then by profit per hour. Create up to
   `max_new_per_hour` one-replica groups an hour (however the search was started), inside `max_active`
   and the quota. A class gets `max_per_class` groups, but a second only while every group of it is on a
   PC clearing the margin, and not within `retry_failed_h` of one of its groups failing. A class whose
   group failed (rules 4 and 5) is tried again `retry_failed_h` later. It used to be barred for good: on
   19 October that had barred 7 of 21 classes, the 4070 and 2070 by a single PC that sent no mining
   reading a week before. A class whose groups only got no PC (released) is tried again `retry_after_min` later, but when its last
   `no_pc_streak` groups never got a PC at all, it isn't tried for `no_pc_hold_min` after the last of them.
   Until 22 October it only went after every other class and still took any slot nothing else wanted: that
   day (UTC), while the watcher ran, every slot it was allowed was full for 722–723 minutes, in 324 of them with
   such a group holding a slot, and 3 times a class that gets PCs was refused for about 5 minutes while one did
   (waits while the account was full can't be seen: no search ran then). When a
   class's latest ended group never got a PC, its next group starts at the priority its latest group with a PC
   got it at, not at batch: on 14 October 4080s and 5080s only got PCs at medium or high, and each new group
   spent 30 min climbing from batch first. If that priority no longer clears the margin, it starts at the
   dearest cheaper one that does and has a GPU free: on 19 October 5090s were only tried at high (+1% after
   PRL fell), while 13 were free at medium (+18%). Besides the classes
   measured on Salad, the search knows benchmark rates for cards not tried there yet (see `KNOWN`);
   rule 5 judges each of their PCs on its own readings.
2. **No PC at a priority within 10 min:** move to the next priority that still clears 10%. If none is
   left, delete the group. The 10 minutes start once the image is ready: while the group is `pending`
   it isn't looking for a PC, and Salad refuses a priority change (`pending_update_in_progress`).
3. **Had a PC, lost it, no other within 20 min:** delete.
4. **No share for max(10 min, 8 expected share gaps) after 20 min on a PC:** leave the PC.
5. **Only PCs that clear 10%.** Skip the miner's first 2 minutes (warm-up). From then on, every round,
   the median of the last 3 readings (one a minute) must clear the margin at the PC's price. The first
   time it doesn't, leave the PC. The median since 23 October: with the mean, one bad minute moved good
   boxes (24 of the 277 margin leaves in 12.6 days whose readings were kept; on 20 October 5070-31 left a
   PC at 133 TH/s for one reading of 3 between two of 133). A PC that really slows shows it in 2 of 3
   readings a minute later. A PC with no mining reading 10 min after we got it is left too, and so is one
   whose newest reading is 6 min old (its miner stopped). Readings come from Salad's log service, which
   sometimes fails for minutes at a time (6 in a row once), so both clocks count only up to the last
   log read that worked, and neither leaves a PC the pool got a share from in the last 6 min. After 10
   min of failed reads there's one `WARN` event. When box.sh starts again on a PC the box mines on in this
   stint (Salad restarted it there), both clocks and the share clock start again from that line, as on a new
   PC, once a PC (a `RESTARTED` event), since 22 October. In 12 days of events, 4 of the 39 "no mining reading
   for 6 min" moves came just after such a line (one of them a box that couldn't mine); 4 more came after a
   restart on the same PC with no line before the rule fired, and the other 31 had no restart (OPEN.md #55).
6. **"outbound port N blocked"** in the logs: reallocate, and put the PC on the bad list (below) for
   boxes that check the same port. Before it starts, box.sh checks the port of the box's pool:
   HeroMiners' 1200, or Kryptex's own host and port (`kryptex_host`, 7048) on a Kryptex box. So a
   Kryptex box keeps a PC that blocks only 1200, and a HeroMiners box one that blocks only 7048. A PC
   found blocking both stays listed for both: on 19 October e8c084ab blocked 1200 for a HeroMiners box,
   then 7048 for the first Kryptex box with the new check. Until
   19 October the PC stayed off the list, and Salad handed the same blocked PCs out again: on 18–19 October
   one cost six groups a move each, and two groups were deleted on a PC another group had already
   found blocked. Every box checked 1200 then, so Kryptex boxes also left PCs they could have mined on.
7. **Credit runs out:** Salad stops every group, and creating one returns `no_credits_available`. The
   watcher leaves stopped groups alone and stops creating. A group that stops after it had a PC is
   taken as this case. Bump `restart_token` after a top-up.
8. **Swap the weakest slot.** With every slot taken, every `upgrade_every_min`, compare the group making
   the least on its PC (profit an hour on its last 3 readings) with the best class the search could create
   that has had a PC on this account (a working box isn't given up for a class Salad may never place),
   and not while that class's latest group got no PC. That class is expected to make its measured rate
   at the priority its latest group got a PC at, or its free priority if that's higher, counted no
   cheaper than low: batch PCs are seldom handed out, and a group that gets none moves up. Switching
   costs the weak box's profit for `swap_wait_min` and the new box's bill for its first 5 minutes,
   taken off the first hour's gain. If what's left is `upgrade_gain` or more, delete the weak group
   (released) and create the new one. One swap per check. On 12 October three swaps priced 4080s at low,
   where they never got a PC; they got PCs at medium, about $0.04/hr less than the swap expected.
9. It only acts on groups in its own ledger.

**Leaving a PC** means reallocating to another PC if the class has paid before, and deleting the group
if it hasn't. The PC goes on a shared **bad list** for `bad_pc_hold_h`, and so does a PC that blocks the
pool port (rule 6): a group that gets a PC another group left is moved off it at once, and that move
doesn't count toward its 3. (On 12 October one 2080 Ti
PC that overheated to 66 TH/s was handed to a second group two hours later.) Since 23 October a PC left for
its margin (rule 5) keeps the TH/s it was left at, and a group whose own price and pool that rate clears 10%
at keeps it (a `NOTE ... keeps PC` event) and is judged on its own readings from there. Before, a PC left at
high was refused at medium too. A PC off the bad list (after 24 h) is still left the same way, uncounted,
when the rate a group of the class last measured there in `rate_days` fails 10% at this group's price and
pool: in the 12.2 days of events to 23 October, 114 of 558 margin leaves were off PCs a group of the class had
left for its margin 1–7 days before, each one a counted move or a deletion. A PC Salad takes back isn't
bad, unless it takes it back within `drop_min` (10) minutes of handing it over `drop_count` (3) times in
`drop_hours` (24): since 23 October that puts it on the bad list for every box, with a `NOTE PC ... goes on the
bad list` event. In the 14 days to then, 281 stints under 10 minutes ended with Salad taking the PC back,
billed $4.05 with almost nothing mined; most were a PC's only one, and this rule would have saved $0.28 of it
(26 stints, 21 of them on 3323a855, which went to 3090 Ti groups 19 times in 38 hours on 17–19 October, and
6a33a882) while blocking one longer stint. A class has **paid before** if it paid on the first account, or once a group of it here
clears the margin over 10 readings with shares at the pool. Reallocations are capped at 3 a group, shared
by rules 4 and 5; the next one deletes it, as released: running out of moves means bad PCs, not a bad
class, so the class isn't barred from new groups. A move off a PC that blocks the pool port (rule 6) is the
PC's fault and doesn't count, up to 6 a group, since 22 October (4060-54 was deleted on its fourth PC for one).
The search also sizes its room by Salad's own count of replicas in use, and stops at the first create Salad
refuses for its quota.
A deleted group counts as **released** when it never got going (no PC, or lost it) and **failed** when
it mined badly on its first PC before the class had paid (no shares, under cost, never mined); a blocked
port never fails a group. Its deletion event carries the box's last log
lines, so it says why.

**Spend** counts the group's price for every minute its instance is `running`, which is all Salad
bills. A group waiting for a PC, or one whose instance is still downloading or starting, isn't billed.
Until 24 October it also counted downloading and creating: against Salad's own running time (its
system logs, 91 groups on 23 October) that read 2.4% high, about 29 s a PC. Not yet checked against
the portal's balance. **PRL mined** adds each `[mining g0]` reading (one a
minute) as TH/s × 1 min × PRL per TH-hr at that moment.

## Monero on the CPU

Release v0.6.0 can also mine Monero (XMR) on the CPU, paid to `xmr_wallet` on HeroMiners' Monero pool (0.9% fee,
payouts from 0.01 XMR). The groups in `xmr_groups` get it. A new group starts with it; a running group is switched
by a PATCH of its command and resources, then a restart of its instance: a running instance keeps the command it
started with until it restarts. It comes back on the new version about 2 minutes later, possibly on another PC. Taking a group out of
`xmr_groups` (or clearing `xmr_wallet`) switches it back to Pearl alone, on 2 vCPUs.

- **vCPUs are free on a GPU group.** Salad's GPU price includes the vCPUs and RAM, so a Monero box asks for 8 vCPUs
  and 8 GB instead of 2 and 4. With 2, the CLI could run one thread at most, and with the Pearl miner using about
  one CPU the container would reach its CPU limit, which slows every process in it, Pearl's included. On 8 the CLI
  runs up to 7 threads (fewer when the CPU's L3 cache is small) at low priority, and one CPU stays free for Pearl.
  The cost is that fewer PCs have 8 vCPUs and 8 GB to spare, so a group may wait longer for one.
- **The box command** is box.sh at the same pinned commit, with three more edits made by `sed`: it downloads
  `llmjob-xmr-worker` from the release, copies it beside each GPU's CLI, and adds `--xmr-address` to the default
  flags.
- **Earnings** come from the pool, by worker (`<group>-g0`). The pool counts every accepted hash, so the XMR a box
  earned is its new hashes times what a hash earns (the last block's reward over the network difficulty, less the
  fee). The price is CoinGecko's. The address's pool balance also counts any other rig that mines to it.
  A read 3 times off the last one is set aside until every read agrees on it for 20 minutes (on 22 October at least 15
  HeroMiners reads gave a hash about 8 to 2,200 times too little), a chart bucket valued 3 times off the median of
  all buckets is valued again at it, and a failed read waits 5 minutes (CoinGecko answers 429 every minute).
- **The rules count Monero too** (since 22 October): each box's earnings in the keep, leave, climb, search and
  swap rules add what the median Monero box makes, from the boxes the Monero pools saw in the last 15 minutes
  (about $0.002/hr; a box's own figure is a 30-minute average that, on a new PC, still holds the old one). Before,
  they judged every box on Pearl alone: in the 48 hours to 9 pm EDT on 22 October, 54 of the 124 margin moves
  and deletions would have cleared 10% with Monero counted, but all 54 also clear at Kryptex's ratio alone; with
  both, 79 clear, so Monero saves 16 beyond the ratio fix. Monero's speed is the table's kH/s column and what each box has earned is
  its XMR column.
- **On Kryptex** (`xmr_kx`, since 20 October). The command adds `--xmr-pool xmr-us.kryptex.network:7029`, which
  needs release v0.6.6. A running group is switched by its command alone, as Gemma is: a box that is mining keeps
  its PC, where Salad usually restarts it with the new command within minutes. The vCPUs and RAM don't change.
  Kryptex's Monero pool is PPS+: it pays for each share, not by its luck, less 1% (HeroMiners takes 0.9%). It pays
  out from 0.1 XMR unless the address's threshold is lowered on Kryptex's site (0.01 at least); HeroMiners pays from
  0.01. A box's XMR there is counted from its worker's chart (since 21 October; see
  [Checks against the pool](#checks-against-the-pool)). A box's kH/s is its rate at each pool that saw it in the
  last 15 minutes: both for a few minutes after a switch.
- **What Kryptex pays for Monero** (`kryptex_xmr_pay`, 94%). From 6 pm on 20 October to 10 am on 21 October EDT,
  Kryptex credited 94% of the network's pay for the work its charts showed, each 10-minute bucket valued at the
  difficulty then (HeroMiners' difficulty chart). The ratio held between 93% and 95% all night: its 1% fee and
  about 5% more. The credit kept pace with the work. No lag showed, though the balance reports a `maturation_time`
  of 2 hours. Until 21 October a box's XMR there was its worker's 30-minute rate over the time since the last read,
  at the network's pay less 1%, starred: about 5% high, and up to 15 minutes behind the box.

## Gemma beside mining

With config `llm` on, each box runs the miner in auto mode (`--mode auto`) and serves the LLMJob network's jobs
(no `--no-serve`), with release `llm_ver` (v0.6.5) or later. The model is the CLI's default, Gemma-4-E4B (Q4_K_M,
4.6 GB). On since 10:23 pm EDT on 19 October, as asked: within two minutes every live group had the new
command (`GEMMA ON` events), and no box was moved.

- **The miner comes first.** From release v0.6.1 the CLI makes only the VRAM plan before the miner starts; it
  downloads Gemma and starts llama-server once the miner is running, so a first run's download costs no mining
  time. If Gemma can't start (no room: it needs about 4.6 GB of VRAM beside the miner's 2.6 GB, or the disk or
  memory is short), the miner carries on alone. On a card that could serve a bigger model with the GPU to itself
  (a 32 GB 5090), auto mode mines until a job comes and then hands the whole card to that model; every other card
  we rent co-runs Gemma.
- **The switch costs a restart, not a PC.** Turning `llm` on or off only updates each group's command. I expected
  a box that is mining to keep the command it started with until its next PC. Most don't: about 6 minutes after
  the update, Salad restarted 5 of the 8 boxes mining then on the same PC with the new command. Each lost a few
  minutes of mining (the image start and the miner's warm-up), but none was moved. Salad didn't restart the other
  three (3060-45, 3060-47, 2060-13): they keep mining without Gemma and run it from their next PC. A group waiting
  for a PC starts that way.
- **Mining pauses for a prompt.** While the LLM answers a prompt the CLI stops mining on that card, and starts
  again 10 seconds after the last one ends (10 minutes at most). The server's speed test is such a prompt: once
  when a node first shows up, then at most once a day, and each Salad box is a new node on each PC. The CLI prints
  no hashrate meanwhile, and box.sh used to repeat the miner's last line, which the watcher reads as a dead miner
  (left after 6 minutes). Now box.sh says `paused for a prompt` from the CLI's own pause lines, and the watcher:
  - shows the box as 🧠 in the report, and logs `PAUSED` when a pause starts and `RESUMED` with its length;
  - counts the "no mining reading" and "no share" rules from the end of the pause, so a pause doesn't cost a PC,
    counting at most 15 minutes of one pause (the CLI resumes after 10, so a longer one is a miner gone quiet);
  - starts the margin rule's 2-minute warm-up again from the first reading after a pause, so the card coming back
    up to speed isn't judged.
- **Our numbers stay right.** A paused minute sends no reading, so it earns nothing in the ledger and nothing is
  averaged into the box's TH/s or its class's measured rate; PRL and XMR come from the pools' own counts anyway.
  The pool's 1-hour hashrate does dip by what a pause cost.

## Kryptex (PPS+)

HeroMiners pays each block's reward across the shares since the last few blocks, so what we're due
depends on its luck and its orphaned blocks: it has credited 82–95% of our network-rate estimate.
Kryptex's PPS+ pays a fixed amount for each accepted share (the network's reward per share less its
fee), plus a share of the blocks' transaction fees. So what a box has earned is known from its accepted
shares. Its advertised Pearl fee is 3%, but it was measured paying **94%** of a share's value
(`block_reward × 2097152 / network_difficulty × 0.94`; the release session's test on 17 October, with the
same 94.0% on three other Kryptex wallets). It pays out from 1 PRL. At HeroMiners' recent 92% that's about
2 points more; the main gain is knowing the pay.

Kryptex's worker rows (`/v3/miner/workers`) have no `hashrate` or `shares` field, though `openapi.yaml`
lists them. They have `avg_hashrate_24h`, `avg_hashrate_3h` and `avg_hashrate_30m`, strings in H/s; the
watcher reads the 30-minute one. Their `valid` count includes the last finished 10-minute bucket twice, so
it isn't used.

**Status: on from 17 October with release v0.6.1: one group, then 5 (`kryptex_groups` 5) from 18 October, and every
slot (`kryptex_groups` 10, the same as `max_active`) from 10:02 pm EDT on 19 October.** A box mining on HeroMiners
then kept its PC and would move to Kryptex when it next got one. At 10:35 pm, as asked, the last four were
switched with `kx_in_place`, by updating their commands. Salad restarted three (3070-53, -55, -56) on the same
PC within 2 minutes, and they mine on Kryptex. It didn't restart 3060-45, even after two calls to its restart
endpoint (both HTTP 202, and nothing happened): Salad lists that instance as already on the new version. So
3060-45 keeps mining on HeroMiners on its PC and starts on Kryptex on its next one. The first try
redirected HeroMiners' names to Kryptex through `/etc/hosts`, which is read-only inside Salad containers:
on 17 October that box stopped at `cannot create /etc/hosts: Read-only file system`, before box.sh ran.
Release v0.6.1 (PR #265) added `--pool host:port` to the CLI, tested on an RTX 4090 for 26 minutes
against Kryptex (316 TH/s, 51 accepted, 0 rejected, all 51 credited). The miner logs in to
`*.kryptex.network` as `wallet/worker` by itself, and from v0.6.1 counts a share only when the pool
accepts it.

Instructions for finishing it on a machine with a GPU are in [KRYPTEX.md](KRYPTEX.md).

What's built for it, with `kryptex_groups` boxes marked **(K)** in the table while some box is still on
HeroMiners (since every box went to Kryptex on 20 October, the mark told nothing, so it's left off then):

- **Which groups:** new groups start on Kryptex while fewer than `kryptex_groups` are on it, and while
  fewer are, HeroMiners groups waiting for a PC are switched by updating their command, the cheapest
  first. Not one still preparing its image (`pending`): Salad refuses an update then
  (`pending_update_in_progress`). An update Salad refuses is tried again in 15 minutes, and the next group
  takes the slot. A Kryptex group
  stays on Kryptex when it loses its PC and starts there on its next one. A box that is mining is never
  moved to switch it. With `kx_in_place` on, if there are still too few on Kryptex after that, boxes mining
  on HeroMiners are switched by their command too, the cheapest first: Salad usually restarts the box on the
  same PC within a few minutes, and it comes back on Kryptex. When Salad doesn't, the check below finds the box
  still on HeroMiners and books its PC there; it starts on Kryptex on its next PC. Kryptex groups beyond the count go back to HeroMiners
  once they have no PC.
- **Which pool a box really mines on:** a switch by command alone can miss. If Salad gives a waiting
  group a PC just as its command changes, the box can start with the old command, and it keeps it. So
  each round the watcher looks at both pools' worker rows: a share at the pool the ledger doesn't
  expect, from 5 minutes after the box got its PC and after its last switch (the PC it left can still
  send some; 15 minutes after a switch in place, since the old miner runs until Salad restarts it), means
  the box mines there. That PC's readings then count to that pool (in the ledger and
  the hourly hashrate check), its last share and hashrate come from that pool's row, a `WARN` says so
  once, and the group's next PC starts on the pool its command names. The report's Kryptex hashrate sums
  the boxes that mine there. Kryptex's rows are read while any group is on Kryptex, was switched in the
  last 2 hours, or has a box whose PC mines there. If Salad hands the group the same PC again, the new box there starts on the group's
  command. When the test stops, a Kryptex box already seen mining on HeroMiners
  isn't moved. The ledger also follows the command Salad has for each group: an update that timed out
  can still go through, and the PC the box is on keeps the pool its readings were booked to.
- **Why not move a mining box:** until 18 October new groups started on HeroMiners (on the first try two
  new Kryptex groups in a row got no PC at medium in 10 minutes, though Salad picks a PC by class and
  priority, not by command), and the HeroMiners box earning least was moved to a new PC whenever the
  Kryptex box had none. On 18 October, while PCs were scarce, that gave up a working PC every 15
  minutes: Salad took back the Kryptex box's PC, a mining box was moved over and waited for a new one,
  and 15 minutes later the next box was moved.
- **Why a move and not a restart,** when a Kryptex box with a PC has to go back (it failed, or the test
  stopped): Salad's instance restart answers 202 but doesn't restart the box. On 17 October two restarts
  of a box whose command had been switched left its miner running (its uptime kept counting) on the old
  command, though the instance reported the new version. The move isn't counted toward the group's 3.
- **When Kryptex doesn't take a box's work:** a Kryptex box that would be left for sending no shares or
  no readings, while no other box's work has reached Kryptex in 30 minutes, goes back to HeroMiners
  instead, on a new PC, and the PC stays off the bad list (the pool is the likelier cause). "Reached
  Kryptex" is by Kryptex's own count (its worker rows' last share): the miner's log can't tell which
  pool accepted a share. When Kryptex's API hasn't answered for 2 minutes, the miners' own accepted
  counts since its last answer stand in, for the box itself as for the others: only on PCs whose readings
  count to Kryptex, and on each PC from 5 minutes in (by then a box that started on HeroMiners would
  have been seen; a box that left within 5 minutes never was). So an outage of the API alone isn't blamed on Kryptex, and one of its pool too
  still is. If another box's work did reach Kryptex, or the box was seen mining on HeroMiners at
  that PC, the PC is the likelier cause: the box is left as usual and stays on Kryptex. A new PC that never sends a reading is common here (46 times by
  18 October), and with new groups starting on Kryptex, two of those would stop the test. Two Kryptex failures stop it: Kryptex
  boxes with a PC are moved to start on HeroMiners at once, and every group stays on HeroMiners until
  config `kryptex_token` changes. **A Kryptex box checks Kryptex's port** before it starts (rule 6). A PC
  that blocks it is left and listed for Kryptex boxes, and the box stays on Kryptex, while some box's
  work reached Kryptex in the last 30 minutes. If none did, Kryptex itself may be down, so it counts as
  a Kryptex failure, as a box with no shares would: an outage then still stops the test after two,
  instead of every Kryptex box leaving PC after PC. Until 19 October box.sh checked only HeroMiners'
  port 1200, so a PC that blocked 7048 showed up as no shares, and Kryptex boxes left PCs that blocked
  only 1200. An earlier check (proof from the miner's `connecting to` line) was tried on 17 October and
  taken out: that line comes after box.sh's first 12 lines on a box with Monero, so it never arrived, and
  the extra path let boxes trade places on Kryptex until their groups were deleted. The port check
  doesn't switch pools, so it can't do that.
- **How:** a Kryptex box runs release `kryptex_ver` (v0.6.1) or later, and its command puts
  `--pool prl-us.kryptex.network:7048` into box.sh's default flags, the same way Monero adds
  `--xmr-address`. Monero stays on HeroMiners. Going back to HeroMiners drops the flag; the box keeps the
  release it would have had anyway (v0.6.0 with Monero).
- **Reading it:** Kryptex's API (`pool.kryptex.com/openapi.yaml`): each worker's last share and
  30-minute hashrate (`avg_hashrate_30m`, H/s) every round, and the wallet's balance and payouts every 5
  minutes. Kryptex lists payouts 15 to a page, and every page is read (`kx_paid`): until 22 October only the
  first was, so the paid total fell as old payouts left it (16.655 of 23.452 PRL that day), and the report's
  Kryptex margins read about 25 points low. The miner's own share count still feeds the
  rules, as for every box.
- **Credit:** each pool has its own credit ratio. PRL mined while a box is on Kryptex is counted apart
  (`prl_kx`), so a switched box's earlier PRL stays with HeroMiners. HeroMiners' ratio is what it
  credited over the PRL mined there; Kryptex's is what it has credited (balance and payouts) over the PRL
  mined there, or `kryptex_pay` (94%) before 1 PRL. Each box's profit counts each part at its pool's ratio.
  The rules judge each PC at `yield`: with `"auto"`, at the ratio of its own pool (since 22 October).
- **In the report:** a Kryptex row in the pool table (its hashrate is the sum of our Kryptex workers' now,
  not an hour's average; a box that just switched counts once Kryptex has a row for it), and a line saying what it has credited against its boxes' estimate.
- **The hourly drift check** compares HeroMiners only with the PRL mined there.

## Salad API notes

Base `https://api.salad.com/api/public/organizations/<org>`, key in the `Salad-Api-Key` header.

- **Cloudflare refuses Python's default user agent** (`Python-urllib`) with error 1010. Send your own.
- **A group whose state Salad doesn't answer** (a 502 or a timeout on `GET /containers/<name>` or its
  instances) is left alone that round: no rule or Kryptex switch acts on last round's state. Until 18
  October a failed group read still ran the rules, which crashed the round on a box that had a PC.
- **There is no balance endpoint.** The public API covers containers, queues, quotas, inference
  endpoints, GPU classes, webhooks, logs and availability. The credit balance is only in the portal, so
  the ledger counts down from the top-up and takes corrections.
- `GET /gpu-classes`: a price for each priority (batch, low, medium, high), CPU and RAM included.
- `POST /availability/sce-gpu-availability` with the container's resources gives
  `available_gpu_<priority>` counts. They aren't reliable for busy cards.
- `GET /quotas`: a new org gets 10 replicas. Stopped groups count against it: with 10 stopped groups,
  creating one failed with `created_replicas_quota_exceeded`, although `container_replicas_used`
  read 0 once Salad had stopped them for lack of credit. A raise goes through "Request Increase" in
  the portal, usually answered within 2 business days.
- Create errors are HTTP 400 with a type: `container_group_update_exception` (the `; curl` command),
  `created_replicas_quota_exceeded`, `cannot_start_container_group_with_current_status` (start while
  pending) and `no_credits_available` ("Entitlement check failed"). A 408 is `connection_timeout`.
- A group created with `autostart_policy: true` can still end up `stopped` once its image is ready;
  `POST .../start` starts it.
- Salad keeps a deleted group's name for a while: creating a group with it an hour later failed with
  `name_conflict`, though the old group no longer showed in the list. The watcher skips every name in its
  ledger, live or deleted, then tries the next suffix (`-2`, `-3`) up to 20 refused names. Skipping only
  live names once ran out of retries: after 21 deleted 4080 groups, `-2` to `-21` were all still held.
- Changing a running group's environment (a new version) makes Salad stop the instance and move it
  to another PC.
- `GET .../containers/<name>` shows `priority` at the top level, but a change goes in
  `container.priority`: `PATCH` with `Content-Type: application/merge-patch+json` and
  `{"container":{"priority":"medium"}}`. Refused while the group is `pending`.
- `GET .../instances` returns `{"instances": [...]}` with `id`, `machine_id` and `state`
  (`allocating`, `downloading`, `creating`, `running`, `stopping`).
  `POST .../instances/<id>/reallocate` moves it to another PC.
- `POST /log-entries`: `page_size` must be 1 to 100. A call takes 1 to 10 s and sometimes returns 408 or
  500. A query per group made a round take 11 minutes, so the watcher reads all container lines in one
  query a round.
- **When more lines match than `page_size`, which ones come back isn't reliable**, whatever
  `sort_order` says. Asking for the last 12 lines of a group over 40 minutes returned lines from 19:43
  to 19:54 and missed 19:57 to 20:02; asking for 100 returned all 34. Only a reply with fewer lines
  than `page_size` is complete, so the watcher splits a window that comes back full and reads each half. The first
  account ran `(log contains "A" OR log contains "B")` fine, so the one 408 here was probably the
  API's usual timeout. `resource.type = "deployment_controller"` returns Salad's system events.
- Salad can move an instance to another PC without a gap in `running`, so a new PC shows only as a
  new `machine_id`.
- **A class can hand out a card it doesn't name.** The "GTX 1660" class mostly handed out 1660 Tis, the
  "GTX 1070, 1080, 1080Ti" class gave a 1080 Ti on 2 of 16 PCs, and on 18 October the "AMD RX 7900 XTX" class
  gave a PC with an NVIDIA RTX 2080 Ti. Read the card from the box, not the class.
- **Some PCs share the GPU with another load.** A GTX 1080 Ti PC on 17 October ran at 93–95% and 160–186 W
  while nothing of ours could run on it, and our miner got about a twentieth of its rate on an idle 1080 Ti.
- **AMD PCs run containers in WSL2** with the GPU behind `/dev/dxg`. Salad mounts its own `amd-smi` at
  `/opt/rocm-wsl`, but on our boxes it answered `static --json` with nothing and `monitor` with only a
  header, so there's no power reading. `rocminfo` and AMD's OpenCL see the card.

## AMD test boxes (amdtest.py)

`amdtest.py` runs test boxes for the AMD (HIP) Pearl core in `earn/native/amd` on Salad's AMD classes. It
manages only the groups `amdtest-rdna3` and `amdtest-rdna4`, or those with `-2` to `-99` after them
(`amdtest-rdna4-2`), and refuses any other name. Each box runs
`amdtest.sh` on the `rocm/dev-ubuntu-24.04:7.2` image (high priority, 4 vCPU, 8 GB by default): it installs
git, Node 22 and node-addon-api, fetches the repo at `BASE_SHA`, applies `PATCH_GZ_B64` if set, runs the tasks
in `TASKS`, prints `[amdtest] HH:MM:SS <tag> ...` lines and sleeps. The tasks are listed at the top of
`amdtest.sh` (card, hip, check, build, foldtest, verify, profile, hashrate, mine, srb, duel, cpuchecks, run).
`duel` is benchmark.md's AMD method on the box's PC: the first miner warms the card for 60 s, then ours (the
CLI) and SRBMiner 3.7.1 run 5 minutes each on HeroMiners; it ends with ours' mean after the first minute and
SRBMiner's own last figure, the two numbers the table compares. A task can carry its own settings, so one run
can compare them on one PC: `hashrate+PEARL_AMD_TRACE=1+PEARL_AMD_COL_BATCH=128:60:10`.

```sh
python3 tuning/salad/amdtest.py create amdtest-rdna3 7900xtx --sha <pushed sha> --tasks card,hip,build,foldtest
python3 tuning/salad/amdtest.py set-env amdtest-rdna3 --tasks build,verify:120,hashrate:60:10   # next run, same PC
python3 tuning/salad/amdtest.py set-env amdtest-rdna3 --tasks build,duel:srb   # SRBMiner first, then ours
python3 tuning/salad/amdtest.py wait amdtest-rdna3 --until 'all done' --since 14:05
python3 tuning/salad/amdtest.py logs amdtest-rdna3 --since 30m
python3 tuning/salad/amdtest.py status amdtest-rdna3
```

What it found on 22 October:

- HIP works through Salad's WSL2 bridge. On an RX 7900 XTX and an RX 9070 XT the process loads
  `/usr/lib/wsl/lib/libdxcore.so`, Salad's `/opt/rocm-host/lib/librocdxg.so.1.2.2` and the image's ROCm 7.2.0
  HIP and HSA runtimes; there is no `/dev/kfd` or `/dev/dri`. HIP counts WGPs as CUs (48 on the 7900 XTX,
  32 on the 9070 XT; rocminfo says 96 and 64).
- A PATCH of `container.environment_variables` replaces the whole map; it does not merge, and a null is
  stored as a value. `set-env` sends the full map.
- After a PATCH, `set-env` restarts the instance (`POST .../instances/<id>/restart`). Of eight runs
  started by a PATCH, two printed: one with a restart (same PC, 30 s later) and one without (8 minutes
  later, on another PC). The four given a `PATCH_GZ_B64` (4,788, 27,516, 58,964 and 123,900 characters)
  printed nothing for 10 to 20 minutes, although Salad showed a new running instance, and so did the next
  run without a patch on each of those two PCs. `realloc` fixed it: the box printed from a new PC a minute
  later. Why is not known yet. So: push the change and pass `--sha` rather than send a patch, and `realloc`
  a box that prints nothing for 10 minutes.
- Salad's API accepts a 262,144-character variable and refuses 1,048,576 (HTTP 400).
- A restart without a patch can go silent too. At 14:53 and 14:54 a `set-env --sha` with no patch, after a
  clean run, printed nothing for 9 to 10 minutes on both boxes (PCs 45c94e40 and 51a4b5c2). `realloc` gave
  each a new PC that printed `boot` within a minute. On 23 October it went the same way two times in three on
  RX 9070 XT boxes (f6f1e2e2 at 05:08, f7baf9ac at 06:24; 4c48f1b8 at 05:37 restarted fine). So to move on
  to a new PC anyway, `set-env --no-restart` and then `realloc` is the reliable way to start a run.
- A box's log can stop mid-run while the box keeps running. An RX 7900 XTX box (b47ff4a8) printed some lines
  of a sweep, skipped others, and printed nothing after 07:10 on 23 October, while the other lane's box logged
  normally. `realloc` moved it on.
- Don't `delete` a box to change its class. After amdtest-rdna3 was deleted at 15:57, creating it again
  failed with `name_conflict` every 30 s until at least 16:22, 25 minutes later. Salad holds a deleted
  name (see "Salad API notes" above). On 23 October `amdtest-rdna4` was still held about 8 hours after its
  delete, so the tool now also takes a numbered name (`amdtest-rdna4-2`). `delete` only when the box is done
  for good.

## AMD benchmark (bench.py)

`BENCH_SET=amd` fills benchmark.md's AMD Radeon table: two PCs per class, our miner and SRBMiner on the same
PC, 5 minutes each after a 60 s warm-up, on `us.pearl.herominers.com:1200`. PeakMiner has no Pearl kernel for
AMD and isn't run, so its column reads "not run" and "% of best" is ours against SRBMiner. No release has an AMD
build, so each box (`bench.sh` on the ROCm image, high priority, 4 vCPU, 8 GB) uses `amdtest.sh`'s setup to
build the HIP core at `BENCH_BASE_SHA` (default `BENCH_COMMIT`) for its card, prints the PC's device copy rate,
runs `verify-hits.js` for `BENCH_VERIFY_SECS` (600), then runs the CLI from source and SRBMiner. That commit
must be the full 40-character SHA: the box `git fetch`es it, and git finds no ref for a short one, so the runner
refuses one. The two PCs of a class take turns going first: after a recorded PC the next gets a new group, since
a moved one keeps its command. A core that doesn't build or a failed hit check leaves "Ours" saying so and still
runs SRBMiner. A PC takes about 30 minutes once it has the image; the runner gives it 40 minutes plus the hit
check (50 at 600 s). If Salad moves a box to another PC by itself, that PC's row is built from its own lines only.
Salad sometimes answers a delete with 202 and keeps the group, so the runner looks at each deleted group every 5
minutes and deletes it again until Salad answers 404, with a note if it's still there after 10. A lane finishes
only once its groups are gone.

Up to three lanes run at once, one Salad slot each, each with its own files (`bench-amd-a.json`,
`bench-amd-a_status.md`, `bench-amd-a.log`) and groups (`bench-a-rx9070xt-1`, ...). Lane a takes the RX 9070 XT
and 9060 XT, lane b the 7900 XTX and 7900 XT, and lane c the 7800 XT (until 22 October lane a had it too). A PC
takes about 25 minutes with the 10-minute hit check and about 17 with a 2-minute one (`BENCH_VERIFY_SECS=120`),
more when PCs are slow to come or get skipped. A lane needs a free replica: the watcher's `max_active` plus the
lanes must stay within Salad's quota, and lowering `max_active` doesn't end a group, so a lane waits until one ends.

```sh
export SALAD_MINE_DIR=<a folder for the runs> PRL_WALLET=<prl1...> BENCH_SET=amd BENCH_COMMIT=$(git rev-parse HEAD)
BENCH_LANE=a nohup python3 tuning/salad/bench.py > $SALAD_MINE_DIR/bench-amd-a.out 2>&1 &
BENCH_LANE=b nohup python3 tuning/salad/bench.py > $SALAD_MINE_DIR/bench-amd-b.out 2>&1 &
BENCH_LANE=c nohup python3 tuning/salad/bench.py > $SALAD_MINE_DIR/bench-amd-c.out 2>&1 &
python3 tuning/salad/bench.py table      # every lane's rows, in benchmark.md's shape
```

Each status file has the lane's table and, under it, a line per PC: its copy rate, the hit check, and hits a
salt (hits / salts; the check's target makes 8).

## The container

- Image `nvidia/cuda:12.2.2-base-ubuntu22.04`. The 12.8.1 image sat in "preparing" for 45+ min. The
  PCs run driver 610 and the miner brings its own CUDA runtime, so the image's CUDA doesn't matter.
- Resources `{"cpu":2,"memory":4096,"storage_amount":10737418240,"gpu_classes":[<id>]}`, 1 replica,
  `restart_policy` `always`, `autostart_policy` true.
- The command installs curl and runs `tuning/box.sh` at commit `f0369f9`, with the worker name and
  payout address put in by `sed`. Salad refuses `; curl` (HTTP 400, no detail), so curl comes after
  `&&`.
- box.sh mines with release v0.5.13 and prints `[mining g0] HH:MM:SS <TH/s> TH/s · <n> accepted · ...`
  once a minute. A PC that blocks the pool port prints `[run] FAIL: outbound port 1200 blocked` (on a
  Kryptex box, the watcher's `sed` makes it check and print Kryptex's port, 7048).
- Workers are named `biz-<card>` (a second group of a card gets `-2`). The pool shows `<worker>-g0`.
- The miner can't lock the memory clock on Salad: its log says the user has no permission.
- After the miner dies, box.sh keeps printing its last `[mining g0]` line every minute, uptime and all.
  The watcher ignores a line whose uptime ("up 1h 23m") hasn't moved, so a dead miner earns nothing in
  the ledger and its PC is left 6 min after the last real reading. Rate and share count alone aren't
  enough: a slow card's rate and shares can stay the same for minutes while it mines.
- Image preparation took from 1 to 60+ min on the first account, with the same image. A PC came within
  about a minute when one was free, and the first share 1–3 min after that.

## Checks against the pool

The ledger is an estimate: PRL from the miners' own TH/s, spend from list prices. Once an hour the
watcher checks it against the pool and writes a `DRIFT` event when they disagree by more than `drift`:

- **Hashrate:** the pool's 1-hour average for the wallet against the miners' readings over the same hour.
- **Ledger:** the pool's count of our hashes, at today's PRL per TH-hr, against the ledger's PRL.
- **Credit:** what the pool has credited (paid + unlocked + unconfirmed) against that work. The pool
  pays a share of each block it finds, so with few blocks credit runs well above or below the work;
  this fires only when it's short by more than 2 / √blocks too.

The report is **one table** with seven columns (five without a Monero address), so each row fits on one line in a chat window:

```
**9:49 pm EDT, Mon, Oct 19** · PRL $1.92 · 7/10 mining · Salad $282.17* left (643* h)

| Box | TH/s | kH/s | $/hr | Profit/hr (margin) | PRL | XMR |
|---|---|---|---|---|---|---|
| ⛏️ 3070-53 med | 78 | 4.8 | 0.08 | +0.02* (+20%*) | 0.569* | 0.000067* |
| ⛏️ 3060-46 low (K) | 47 | 0.0 | 0.05 | +0.01* (+14%*) | 0.141 | 0.000021* |
| ⏳ 5070-29 med (K) | — | — | — | — | 0.651 | 0.000052* |
| Others ×781 | — | — | — | — | 155.386* | 0.009229* |
| **All** | **410** | **20.8** | **0.44** | **+0.09\* (+17%\*)** | **157.402\*** | **0.009489\*** |

| Pool | Hashrate (1h) | Confirmed | Pending | Paid |
|---|---|---|---|---|
| ⛏️ PRL HeroMiners | 365 TH/s | 0.871 | 1.331 | 149.210 |
| ⛏️ PRL Kryptex | 140 TH/s | 0.968 | 1.719 | 3.269 |
| 🪙 XMR HeroMiners | 30.7 kH/s | 0.009922 | 0.000000 | 0.000000 |
```

That's the report the new code gave from the live ledger at 9:49 pm on 19 October with 7 of its 10 box rows left
out, so the All row counts boxes not shown. Its PRL total, 157.402, was within 0.04 of what the two pools had
credited the wallet (Paid + Confirmed + Pending: 151.412 and about 5.95).

- **The first line** gives only what the table doesn't: the time (New York, on the 12-hour clock: EDT or EST; it was
  24-hour time until 20 October), the date in US style ("Mon, Oct 19"), the PRL price in cents, how many boxes are
  mining out of the slots (`max_active`, or Salad's quota if lower), and the Salad credit left with how many hours it
  lasts at the All row's $/hr (no hours while nothing bills). Profit an hour, the margin and the profit so far are in
  the **All** row.
- **Every slot has a row.** A slot with no group (the search found no class it could use yet) shows as `⬜ empty`, so the table
  always has `max_active` box rows.
- **Profit/hr** has the margin (profit ÷ earnings) in brackets beside it.
- Inside bold text the estimate stars are written `\*`, or Markdown reads them as emphasis and the bold breaks.
- **TH/s and kH/s** are each box's speed on both coins: Pearl on the GPU (the miner's 10-minute average) and
  Monero on the CPU (as the Monero pool measures it; — for a box that doesn't mine it).
- **Others** is one row for every group that has ended, with its count and what they earned; the events say which
  were released and which failed.
- **A box row** holds its state, its group name without `biz-rtx` and its priority (`med` for medium), joined
  by no-break spaces so it doesn't wrap. Boxes are sorted by profit an hour, best first. The PC id is in the
  events, not the table.
- **PRL** is what the box has earned by its pools' own counts (since 20 October; it replaced Profit $, a dollar
  estimate the user found didn't help):
  - **On Kryptex**, its accepted shares from its worker's chart there (10-minute buckets, read every 5 minutes),
    each at the network's pay for a share when first seen, times what Kryptex pays a share (`kryptex_pay`, 94%).
    PPS+ pays a set amount a share, so this is exact but for its share of block fees. No star.
  - **On HeroMiners**, the work HeroMiners counted for its worker (its share difficulty, `hashes`), valued the same
    way, times HeroMiners' **credit ratio** (credited ÷ estimate once it has credited us in 10 blocks). HeroMiners
    pays each block across recent shares, so what a share earns follows its luck: starred.
  - A group that ended before 20 October has only the ledger's estimate (its speed at the network's rate, at each
    pool's ratio), starred. Work a worker of the same name did before the group began isn't counted.
- **XMR** is what the box has earned in Monero by its pools' own counts:
  - **On Kryptex** (since 21 October), the work Kryptex accepted from its worker, from the worker's chart there
    (10-minute buckets, read every 5 minutes; a bucket's hashrate times 600 is its accepted hashes). Each bucket is
    valued at the network's pay for a hash when first seen, times what Kryptex pays for it (`kryptex_xmr_pay`, 94%;
    see [Monero](#monero-on-the-cpu)). PPS+ pays a set amount for the work, so this is exact but for its share of
    block fees. No star. The chart reaches back 2 days, so the first read counted all the work since Monero went to
    Kryptex on 20 October. Up to 5 minutes of work just before a box ends can be missed.
  - **On HeroMiners**, the box's Monero work as HeroMiners counted it, at the network's pay less its 0.9% fee:
    starred, since HeroMiners pays by its luck. A box that mined on both is starred for its HeroMiners part.
  - A group that ended before 21 October has its rate on Kryptex over time there instead (see
    [Monero](#monero-on-the-cpu)), starred.
- **The pool table** is what each pool reports for the account, with no stars: its own hashrate (1-hour average,
  which still counts a box that lost its PC in the last hour), the coins confirmed (unlocked, paid out from
  1 PRL or 0.01 XMR), pending (in blocks not yet confirmed) and paid. The XMR pool's figures count any other rig
  mining to the address. Locked-in and not-yet-locked rows were here before; they were dropped on 14 October.
  Each pool's name links to its page for our address (since 20 October): HeroMiners' `?wallet=` lookup on
  pearl.herominers.com and monero.herominers.com, and Kryptex's `/prl/miner/stats/` and `/xmr/miner/stats/`
  pages. The **XMR Kryptex** row (since 20 October) is Kryptex's Monero pool for our Monero address, read every
  5 minutes: its workers' 30-minute hashrate, the coins confirmed (paid out from 0.1 XMR), pending and paid. It
  shows whenever Monero is on, even with no box mining there yet.
- **Nothing under the tables.** Until 20 October a line there gave what Salad had cost (Profit $ counted it),
  what each credited PRL cost to mine and the credit left. The credit left is now in the first line; the other
  two were dropped on 20 October as not helpful (for a few minutes the three were rows under the All row, and
  the cost per PRL sat beside the price). Before that line, a paragraph also gave each pool's credit ratio and
  payout threshold, what SafeTrade sold and holds, and the XMR our boxes had mined. The pool table shows the
  coins; the credit ratios still set Profit/hr and the HeroMiners part of PRL, and are in the 15-minute notes when
  they move.
- **The key** is here, not in the report (it was a line under the table, repeated every 15 minutes, until
  20 October): ⛏️ mining · 🧠 mining paused while Gemma answers a prompt · ⏳ waiting for a PC · ⏹️ stopped ·
  ⬜ empty slot · `*` our own count or estimate,
  or entered by hand (the rest comes from an API: Salad, the pools, prlscan, CoinGecko, or the trader's log)
  · (K) mines Pearl on Kryptex, the rest on HeroMiners (shown only while some box is on HeroMiners).

A box on a new PC with no reading yet (under 15 minutes) is **starting**: it shows — for Profit/hr and
Margin, and the total leaves it out until its first reading. Its bill still counts in $/hr. Before
this, two boxes changing PCs at once took the total from +$0.41 to +$0.07 for a few minutes. A box paused
for a prompt (🧠) with no reading in 10 minutes shows the same way, since 22 October: Gemma's speed test on a
new PC can run past 10 minutes, and 3060-70 showed a margin of −3180% from its Monero alone, which took the
All row's margin below zero. A box with no Pearl reading shows its profit an hour without a margin.

## SafeTrade row

When `trader` is set, the watcher reads the newest `STATUS` line of the autoseller (`trader/`) from its
Railway logs, at most every 5 minutes. The report used it for Profit $ (sold PRL at what it sold for), with
sales entered by hand in config `safetrade` standing in, until Profit $ went on 20 October; it listed the sales
under the table until then too. Nothing in the report uses it now.

It needs a Railway project token for the production environment, set as `RAILWAY_TOKEN` or added by the
cloud session's network proxy for `backboard.railway.com`. Without one, Railway answers "Not Authorized"
and there's one `WARN` event.

## Safety net

**Strays.** Salad sometimes answers a delete with 202 and keeps the group, and a create that errors can
still make one. Such a group bills while nothing watches it, and it takes a replica from the quota, so
a new group can't start. Every 5 minutes the watcher lists the groups on Salad and deletes any `biz-`
group that isn't live in its ledger and has been there for 10 minutes (`SWEPT` in the events). The 10
minutes give a normal delete time to finish.

`guard.py` is for when the watcher isn't running, for example because the machine it ran on went away
while the groups keep billing on Salad. It keeps no state. It stops every running `biz-` group that is
at least an hour old, has a running PC, and from which neither pool has seen work in the last hour:
HeroMiners, or Kryptex for a box mining Pearl there (its 30-minute hashrate or a share in the hour). If
HeroMiners can't be read, it stops nothing; if Kryptex can't be read, it leaves the boxes on Kryptex
(`--pool` in their command) alone and checks the rest. The watcher never lets that happen, so a hit means
nothing is watching. Stopping is reversible. Run it
hourly (`--dry-run` only reports):

```sh
PRL_WALLET=prl1... python3 -I tuning/salad/guard.py
```

It isn't running on a schedule yet. Two tries with a Claude Routine failed: a fresh session without
the repo had to download the script, which its permission check blocks, and a Routine bound to a
session with the repo fired into a new empty session instead. The Routine is paused. The better home
is a Railway cron service next to the trader: Root Directory `/tuning/salad`, start command
`python3 guard.py`, hourly, with `SALAD_API_KEY` and `PRL_WALLET` set.

## Money

- **Every figure we count or estimate ourselves has a `*`** in `report.md`; a number without one comes
  straight from an API (Salad, the pool, prlscan, CoinGecko) or from the trader's log. Starred: the
  Salad credit left (Salad's API has no balance or billing, so the watcher adds each
  group's price for every minute it has a PC; on 14 October it was $1.01 high over $100), PRL earned on HeroMiners
  and before 20 October, XMR earned, and profit and margin. Not starred: TH/s, kH/s, $/hr, PRL earned on Kryptex
  (its own share count at its set pay a share), the pool's totals and prices. Set `credit_offset` from the portal now and then so the credit count stays right.
- $/TH-hr = 1e12 × 3600 / (difficulty × 2^48) × reward_grains / 1e8 × PRL price, from
  `api.prlscan.com/v1/blocks?limit=1` and `/v1/market/prl`.
- One pool unit/s = 419/93644 TH/s (a 5090's 24-hour pool average against its miner's 419 TH/s); one
  share = 2^21 units. A 4080 at 199 TH/s finds a share about every 47 s, a 2060 at 40 TH/s about every
  4 min. One 2060 went 11 min without a share while mining steadily, which is why rule 4 waits 8 gaps:
  a working miner goes that long without a share about 1 time in 3000.

## Classes measured on Salad

From the first Salad account (personal, Oct 9–10 2026). TH/s is the median of a group's readings at
the priority that worked; margins at PRL $1.37–1.42.

| Result | Class | Priority, $/hr | TH/s | Margin |
|---|---|---|---|---|
| Paid | RTX 2060 | low, 0.030 | 40 | +32% |
| Paid | RTX 2080 | low, 0.060 | 72 | +25% |
| Paid | RTX 3080 | low, 0.087 | 107 | +25% |
| Paid | RTX 4080 | medium, 0.190 | 199 | +14% |
| Paid | RTX 5080 | medium, 0.223 | 223 | +11% |
| Thin | RTX 4070 | medium, 0.123 | 89–120 | about +5% |
| Thin | RTX 4070 Ti | medium, 0.160 | 152 | +5% |
| Thin | RTX 3090 Ti | medium, 0.160 | 152 | +5% |
| Thin | RTX 5070 | medium, 0.137 | 115–128 | 0 to +4% |
| Thin | RTX 3060 Ti | low, 0.047 | 45 | +6% |
| Lost money | RTX 3080 Ti | low, 0.105 | 70 | under cost |
| Lost money | RTX 3090 | low, 0.117 | 93–121 | +17% on one PC, under cost on one capped at 262 W |
| Lost money | RTX 4070 Ti Super | medium, 0.170 | 137 | under cost |
| No share | RTX 3070 Ti | batch, 0.060 | | |

5060, 5060 Ti, 5070 Ti, 4090 and 5090 got no PC at a price that pays. A 4090 at high landed on a 300 W
PC.

Batch rarely gets a PC for 40- and 50-series cards; low works for older cards. The same class varies
about ±15% between PCs, because owners cap the card's power. Salad has AMD classes, but the miner is
CUDA-only. SRBMiner mines Pearl on them at 32–85 TH/s (benchmark.md's AMD section).

What went wrong on the first account:

- A 4090 sat 20+ min at low and at medium with about 100 reported free, then got a PC at high in a
  minute. That's why rule 2 climbs the priorities.
- A 3090 waited 2 hours at batch for a PC, holding a quota slot.
- A 4080's PC blocked port 1200 (rule 6). A 4070 Super's PC later couldn't reach github.com to fetch
  the miner, though it had mined at +44% earlier: when a class that pays stops, the PC is the problem.
- The credit ran out on Oct 10 at 07:43 UTC. Salad stopped every group at once, and none started again
  on its own.
- Rates swing a lot between PCs of one class: a 4070 did 120 TH/s on one PC and 57 on another, a 3070
  76 then 59, a 3060 Ti 61 then 45 (on a PC at 85 °C). That's why rule 5 leaves a PC at once.

The first account spent $24.43 and mined 20.65 PRL over about a day, about $1.18 a PRL, worth about
+$4.3 at PRL $1.40.

## The business account

- Oct 10, 19:33: the first 2080 group got a 2080 SUPER at 85–87 °C, held to 750–900 MHz and 115 of
  250 W: 33 TH/s. The first 2060 got a card reporting no power limit at 55 W, likely a laptop: 22
  TH/s. Both were moved to other PCs.
- Salad's "RTX 2060" class includes the CMP 40HX mining card (110 W). Our miner did 0.8 TH/s on it,
  and rule 5 moved the group off it 5 minutes after the miner started.

v0.5.14 adds an L2 rule worth about +0.5 to 1.7% on power-capped RTX 40 cards. Try it on one group
(`ver_override`) before switching `box_ver`.
