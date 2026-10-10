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
| `max_active` | Most groups alive at once. The quota caps it too. |
| `search` | Run the hourly search. |
| `search_token` | Change it to run a search now. |
| `restart_token` | Change it after a top-up: restarts groups Salad stopped for lack of credit. |
| `credit_start`, `credit_offset` | Credit left = start + offset − spend. Set the offset when the portal shows a different balance. |
| `box_ver` | Miner release for new groups. `ver_override` maps a group name to a release, to try one on a single group. |
| `wallet` | Payout address, if `PRL_WALLET` isn't set. |
| `report_min` | Minutes between `REPORT` lines. |
| `min_margin` | The margin every class and PC must clear (0.10). |

Tests: `python3 -m unittest discover -s tuning/salad`. They fake Salad, the pool and the clock.

## Rules

**Margin** is profit as a share of earnings: (earnings − price) / earnings, as the first account
measured it. A 2060 at 40 TH/s earning $0.047/hr on a $0.030/hr PC is +36%, not +56%. The bar is
10% (`min_margin`) everywhere: for the search and priority moves on a class's measured rate, and for
each PC on its own readings.

1. **Search** hourly at :01. Take classes with a known rate that clear a 10% margin at the cheapest
   priority with a GPU free. Classes that paid before come first, then by profit per hour. Create up to
   3 one-replica groups an hour (however the search was started), inside `max_active` and the quota. Each class is tried once on an
   account, as on the first one: a class whose group failed (rules 4 to 6) isn't created again. A class
   whose groups only got no PC (released) is tried again in a search at least an hour later.
2. **No PC at a priority within 10 min:** move to the next priority that still clears 10%. If none is
   left, delete the group. The 10 minutes start once the image is ready: while the group is `pending`
   it isn't looking for a PC, and Salad refuses a priority change (`pending_update_in_progress`).
3. **Had a PC, lost it, no other within 20 min:** delete.
4. **No share for max(10 min, 8 expected share gaps) after 20 min on a PC:** leave the PC.
5. **Only PCs that clear 10%.** Skip the miner's first 2 minutes (warm-up). From then on, every round,
   the last 3 readings (one a minute) must clear the margin at the PC's price. The first time they
   don't, leave the PC. A PC with no mining reading 10 min after we got it is left too.
6. **"port 1200 blocked"** in the logs: reallocate.
7. **Credit runs out:** Salad stops every group, and creating one returns `no_credits_available`. The
   watcher leaves stopped groups alone and stops creating. A group that stops after it had a PC is
   taken as this case. Bump `restart_token` after a top-up.
8. It only acts on groups in its own ledger.

**Leaving a PC** means reallocating to another PC if the class has paid before, and deleting the group
if it hasn't. A class has **paid before** if it paid on the first account, or once a group of it here
clears the margin over 10 readings with shares at the pool. Reallocations are capped at 3 a group, shared
by rules 4 to 6; the next one deletes it.
A deleted group counts as **released** when it never got going (no PC, or lost it) and **failed** when
it mined badly (no shares, under cost, port blocked). Its deletion event carries the box's last log
lines, so it says why.

**Spend** counts the group's price for every minute it has a PC (instance downloading, creating or
running). A group waiting for a PC isn't billed. The first account counted only `running` minutes.
Neither way has been checked against the portal. **PRL mined** adds each `[mining g0]` reading (one a
minute) as TH/s × 1 min × PRL per TH-hr at that moment.

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
- Changing a running group's environment (a new version) makes Salad stop the instance and move it
  to another PC.
- `GET .../containers/<name>` shows `priority` at the top level, but a change goes in
  `container.priority`: `PATCH` with `Content-Type: application/merge-patch+json` and
  `{"container":{"priority":"medium"}}`. Refused while the group is `pending`.
- `GET .../instances` returns `{"instances": [...]}` with `id`, `machine_id` and `state`
  (`allocating`, `downloading`, `creating`, `running`, `stopping`).
  `POST .../instances/<id>/reallocate` moves it to another PC.
- `POST /log-entries`: `page_size` must be 1 to 100. A call takes 1 to 10 s and sometimes returns 408 or
  500. A query with `OR` timed out (408), so the watcher reads all container lines in one query and
  pages when a round has more than 100. A query per group made a round take 11 minutes. The first
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
- Image preparation took from 1 to 60+ min on the first account, with the same image. A PC came within
  about a minute when one was free, and the first share 1–3 min after that.

## Money

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
