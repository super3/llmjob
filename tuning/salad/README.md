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
| `max_active` | Most groups alive at once. Salad's quota caps it too, so set it above the quota and a raised quota is used at once (live: 30, with a quota of 10). |
| `search` | Run the hourly search. |
| `search_token` | Change it to run a search now. |
| `restart_token` | Change it after a top-up: restarts groups Salad stopped for lack of credit. |
| `credit_start`, `credit_offset` | Credit left = start + offset − spend. Set the offset when the portal shows a different balance. |
| `box_ver` | Miner release for new groups. `ver_override` maps a group name to a release, to try one on a single group. |
| `wallet` | Payout address, if `PRL_WALLET` isn't set. |
| `report_min` | Minutes between `REPORT` lines. |
| `min_margin` | The margin every class and PC must clear (0.10). |
| `max_new_per_hour` | Most new groups an hour (3). |
| `max_per_class` | Most groups of one class (1). |
| `search_every_min` | Minutes between searches while a slot is empty (60). |
| `retry_after_min` | Minutes before a class released for want of a PC is tried again (60). |
| `no_pc_streak`, `no_pc_hold_min` | A class whose last 3 groups got no PC goes last for 180 min. |
| `upgrade_gain` | How much more an hour a class must be expected to make before it takes the weakest group's slot (0.05). `null` turns swaps off. |
| `upgrade_every_min` | Minutes between swap checks while every slot is taken (15). |
| `swap_wait_min` | Minutes a swapped slot is expected to earn nothing while the new group gets a PC and starts (20). Part of a swap's cost. |
| `yield` | The share of the network-rate estimate the rules count on when they judge a margin: `"auto"` for the pool's credit ratio (below), or a number. Live: `1.0` for now. |
| `yield_min_blocks`, `yield_floor` | The credit ratio is used once the pool has credited us in 10 blocks, and never below 0.5. |
| `bad_pc_hold_h` | Hours a PC that a group left stays on the shared bad list (24). |
| `rate_days` | Days a PC's measured rate counts toward its class's rate (7). |
| `drift` | How far the ledger and the pool may disagree before a `DRIFT` event (0.15). |
| `trader` | Railway ids of the SafeTrade autoseller: `{"project", "service", "environment"}`. Unset: no SafeTrade line. |
| `safetrade` | SafeTrade sales entered by hand (`sold`, `received`, `fees`, `usdt`, `prl`), used until the trader's logs can be read. |
| `xmr_wallet`, `xmr_groups` | Monero on the box's CPU (see [Monero](#monero-on-the-cpu)): the address, and the groups that mine it (`["*"]` for every group). Unset: no Monero, and the table is as before. |
| `xmr_ver`, `xmr_cpu`, `xmr_memory`, `xmr_threads` | A Monero box runs release `v0.6.0` or later on 8 vCPUs and 8192 MB. `xmr_threads` 0 lets the CLI choose (it keeps one CPU free for Pearl). |

Tests: `python3 -m unittest discover -s tuning/salad`. They fake Salad, the pool and the clock.

## Rules

**Margin** is profit as a share of earnings: (earnings − price) / earnings, as the first account
measured it. A 2060 at 40 TH/s earning $0.047/hr on a $0.030/hr PC is +36%, not +56%. The bar is
10% (`min_margin`) everywhere: for the search and priority moves on a class's measured rate, and for
each PC on its own readings.

**A class's rate** is the median TH/s of the PCs it has mined on here in the last `rate_days`, each PC
on its last 3 readings after warm-up, capped and slow PCs included. Before a class has mined, it's the
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
   PC clearing the margin, and never after one of its groups failed. Each class is tried once on an
   account, as on the first one: a class whose group failed (rules 4 to 6) isn't created again. A class
   whose groups only got no PC (released) is tried again `retry_after_min` later, but when its last
   `no_pc_streak` groups never got a PC at all, it goes after every other class for `no_pc_hold_min`. When a
   class's latest ended group never got a PC, its next group starts at the priority its latest group with a PC
   got it at, not at batch: on 14 October 4080s and 5080s only got PCs at medium or high, and each new group
   spent 30 min climbing from batch first. Besides the classes
   measured on Salad, the search knows benchmark rates for cards not tried there yet (see `KNOWN`);
   rule 5 judges each of their PCs on its own readings.
2. **No PC at a priority within 10 min:** move to the next priority that still clears 10%. If none is
   left, delete the group. The 10 minutes start once the image is ready: while the group is `pending`
   it isn't looking for a PC, and Salad refuses a priority change (`pending_update_in_progress`).
3. **Had a PC, lost it, no other within 20 min:** delete.
4. **No share for max(10 min, 8 expected share gaps) after 20 min on a PC:** leave the PC.
5. **Only PCs that clear 10%.** Skip the miner's first 2 minutes (warm-up). From then on, every round,
   the last 3 readings (one a minute) must clear the margin at the PC's price. The first time they
   don't, leave the PC. A PC with no mining reading 10 min after we got it is left too, and so is one
   whose newest reading is 6 min old (its miner stopped). Readings come from Salad's log service, which
   sometimes fails for minutes at a time (6 in a row once), so both clocks count only up to the last
   log read that worked, and neither leaves a PC the pool got a share from in the last 6 min. After 10
   min of failed reads there's one `WARN` event.
6. **"port 1200 blocked"** in the logs: reallocate.
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
if it hasn't. The PC goes on a shared **bad list** for `bad_pc_hold_h`: a group that gets a PC another
group left is moved off it at once, and that move doesn't count toward its 3. (On 12 October one 2080 Ti
PC that overheated to 66 TH/s was handed to a second group two hours later.) A PC Salad takes back isn't
bad. A class has **paid before** if it paid on the first account, or once a group of it here
clears the margin over 10 readings with shares at the pool. Reallocations are capped at 3 a group, shared
by rules 4 to 6; the next one deletes it, as released: running out of moves means bad PCs, not a bad
class, so the class isn't barred from new groups.
A deleted group counts as **released** when it never got going (no PC, or lost it) and **failed** when
it mined badly (no shares, under cost, port blocked). Its deletion event carries the box's last log
lines, so it says why.

**Spend** counts the group's price for every minute it has a PC (instance downloading, creating or
running). A group waiting for a PC isn't billed. The first account counted only `running` minutes.
Neither way has been checked against the portal. **PRL mined** adds each `[mining g0]` reading (one a
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
- **The rules** still judge every box on Pearl alone. Monero's earnings count in the table's Profit/hr, Margin and
  Profit $, with a 🪙 row for what the boxes have mined, but they don't keep a slow PC or a weak class.

## Salad API notes

Base `https://api.salad.com/api/public/organizations/<org>`, key in the `Salad-Api-Key` header.

- **Cloudflare refuses Python's default user agent** (`Python-urllib`) with error 1010. Send your own.
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

## The container

- Image `nvidia/cuda:12.2.2-base-ubuntu22.04`. The 12.8.1 image sat in "preparing" for 45+ min. The
  PCs run driver 610 and the miner brings its own CUDA runtime, so the image's CUDA doesn't matter.
- Resources `{"cpu":2,"memory":4096,"storage_amount":10737418240,"gpu_classes":[<id>]}`, 1 replica,
  `restart_policy` `always`, `autostart_policy` true.
- The command installs curl and runs `tuning/box.sh` at commit `f0369f9`, with the worker name and
  payout address put in by `sed`. Salad refuses `; curl` (HTTP 400, no detail), so curl comes after
  `&&`.
- box.sh mines with release v0.5.13 and prints `[mining g0] HH:MM:SS <TH/s> TH/s · <n> accepted · ...`
  once a minute. A PC that blocks the pool port prints `[run] FAIL: outbound port 1200 blocked`.
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

The report is **one table** with six columns, so each row fits on one line in a chat window:

```
**22:40 UTC, Wed 14 Oct** · PRL $1.882 · 7 of 10 mining · **+$0.335*/hr** · +26%* margin · **profit so far +$36.42***

| Box | TH/s | $/hr | Profit/hr | Margin | Profit $ |
|---|---|---|---|---|---|
| ⛏️ 3090ti-4 med | 150 | 0.160 | +0.066* | +29%* | +1.17* |
| ⏳ 5070-22 med | — | — | — | — | +0.00* |
| Others ×206 🔁201 ❌5 | — | — | — | — | +34.47* |
| **All** | **867** | **0.954** | **+0.335*** | **+26%*** | **+36.42*** |
| 🔒 Locked in · 46.793* PRL sold | | | | | +7.39* |
| 📈 Not yet locked · 45.791* PRL | | | | | +27.59* |
| 🪙 Monero · 0.002598* XMR | | | | | +1.44* |
```

- **The first line** answers "how is it going": the time, the PRL price, how many boxes are mining, profit an
  hour, the margin, and the profit so far.
- **A box row** holds its state, its group name without `biz-rtx` and its priority (`med` for medium), joined
  by no-break spaces so it doesn't wrap. Boxes are sorted by profit an hour, best first. The PC id is in the
  events, not the table.
- **Profit $** counts the PRL we expect the pool to credit for the box's speed (the network-rate estimate times
  the pool's **credit ratio**, credited ÷ estimate once the pool has credited us in 10 blocks), plus its Monero,
  less its Salad bill. Sold PRL counts at what it sold for and the rest at today's price, so the column adds up to
  the three rows under the total: 🔒 **locked in** (USDT received after fees, less the sold PRL at the average
  cost to mine), 📈 **not yet locked** (the unsold PRL at today's price, less the same cost; it moves with the
  price and the pool's luck) and 🪙 Monero. Before 14 October, Profit $ valued sold PRL at today's price, which
  put it about $22 above what the pool and SafeTrade showed.
- **Under the table**, one line gives the key (what ⛏️ ⏳ ⏹️ and `*` mean), and one more gives the money trail:
  what Salad cost and what each PRL cost to mine, what's at the pool, what SafeTrade sold and holds, Monero at
  the pool, and the Salad credit left.

A box on a new PC with no reading yet (under 15 minutes) is **starting**: it shows — for Profit/hr and
Margin, and the total leaves it out until its first reading. Its bill still counts in $/hr. Before
this, two boxes changing PCs at once took the total from +$0.41 to +$0.07 for a few minutes.

Every PRL carries the average cost, so spending now and selling hours later still gives one number.
The cost is per credited PRL, not per estimated PRL: at 83% that's $1.35, not $1.12, and locked in on
26.24 PRL sold for 35.53 USDT is +$0.22, not +$6.22.

## SafeTrade row

When `trader` is set, the watcher reads the newest `STATUS` line of the autoseller (`trader/`) from its
Railway logs, at most every 5 minutes, and uses it for the 🔒 and 📈 rows and the SafeTrade part of the line under the table.
Until it can, sales entered by hand in config `safetrade` stand in.

It needs a Railway project token for the production environment, set as `RAILWAY_TOKEN` or added by the
cloud session's network proxy for `backboard.railway.com`. Without one, Railway answers "Not Authorized"
and there's one `WARN` event.

## Safety net

`guard.py` is for when the watcher isn't running, for example because the machine it ran on went away
while the groups keep billing on Salad. It keeps no state. It stops every running `biz-` group that is
at least an hour old, has a running PC, and from which the pool has seen no work in the last hour. The
watcher never lets that happen, so a hit means nothing is watching. Stopping is reversible. Run it
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
  straight from an API (Salad, the pool, prlscan, CoinGecko) or from the trader's log. Starred: Spent $
  and the credit left (Salad's API has no balance or billing, so the watcher adds each group's price for
  every minute it has a PC; on 14 October it was $1.01 high over $100), PRL expected, profit and margin,
  XMR mined, and SafeTrade entered by hand. Not starred: TH/s, $/hr, the pool's totals and prices. Set `credit_offset` from the portal now and then so the credit count stays right.
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
CUDA-only.

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
