# Mining PRL on SaladCloud

These scripts rent Salad GPUs one container group at a time. Each group runs `tuning/box.sh`, the same script as our Vast and Clore boxes. A group is kept only while it earns more than it costs. They were built and tested on a personal Salad account on Oct 9-10.

## Setup

1. Copy `config.example.json` to `config.json` and fill it in:
   - `org` and `project`: the Salad organization and project names.
   - `prefix`: starts every group and worker name, like `biz-rtx4080`. Pick one no other account uses, so the pool's worker list doesn't mix them.
   - `wallet`: the PRL address the boxes mine to. The watcher reads the pool stats for it.
   - `credit`: the prepaid credit you start with. The watcher counts down from it, because Salad's API has no balance.
   - `box_ref`: the commit of `tuning/box.sh` that boxes run.
   - `miner_version`: the release the boxes mine with.
2. Put the API key in the environment as `SALAD_API_KEY`. Nothing reads it from a file, and nothing prints it.
3. Run `python3 search.py` to see which classes clear a 10% margin right now. It only reads.
4. Run `./monitor.sh`. It starts `watch_all.py` and `hourly.py`, restarts either one that stops, and prints the lines worth reading.

`config.json`, `groups/` (one JSON file per group, with its spend, PRL and events) and the logs are gitignored.

## What each script does

- `salad.py`: the API calls.
- `search.py`: the classes worth renting, at the cheapest priority with a GPU free that clears 10%. Uses the TH/s we measured on Salad where we have it, otherwise the benchmark figure.
- `hourly.py`: at 90 s past each hour, creates a one-replica group for up to 3 classes not tried before, within the replica quota. `NOW=1` runs one round.
- `watch_all.py`: checks every group about once a minute. Logs a summary and the credit left every 15 rounds.

## The watcher's rules

1. **No PC at a priority within 10 minutes:** move up to the next priority that still clears 10%. If none is left, delete the group. Salad's free counts aren't reliable for busy cards.
2. **Lost its PC and got no other within 20 minutes:** delete the group.
3. **No share after 20 minutes on a PC:** the window is 8 expected share gaps, at least 10 minutes. If the class has paid before, the PC is the problem: move to another PC, up to 3 times. Otherwise delete the group.
4. **Earning less than it costs:** judged after 20 minutes mining, on at least 10 minutes of the miner's own rate from its log. Two checks in a row means move to another PC if the class has paid before (up to 3 times), else delete.
5. **"port 1200 blocked" in the box's log:** move to another PC, up to 3 times.
6. **Stopped groups are left alone.** Salad stops every group when the credit runs out, and they don't start again on their own. Start them by hand after a top-up.

## What we measured (Oct 9-10, PRL about $1.37-1.42)

| Class | Priority | $/hr | TH/s | Result |
|---|---|---|---|---|
| RTX 2060 | low | 0.030 | 40 | +32% |
| RTX 2080 | low | 0.060 | 72 | +25% |
| RTX 3080 | low | 0.087 | 107 | +25% |
| RTX 4080 | medium | 0.190 | 199 | +14% |
| RTX 5080 | medium | 0.223 | 223 | +11% |
| RTX 3060 Ti | low | 0.047 | 45 | +6% |
| RTX 4070 | medium | 0.123 | 89-120 | about +5%, depends on the PC |
| RTX 4070 Ti | medium | 0.160 | 152 | +5% |
| RTX 3090 Ti | medium | 0.160 | 152 | +5% |
| RTX 5070 | medium | 0.137 | 115-128 | 0 to +4% |
| RTX 3080 Ti | low | 0.105 | 70 | earned less than it cost |
| RTX 3090 | low | 0.117 | 93-121 | paid on one PC, not on another |
| RTX 4070 Ti Super | medium | 0.170 | 137 | earned less than it cost |
| RTX 3070 Ti | batch | 0.060 | | no share |

Batch rarely gets a PC for 40- and 50-series cards. The 5060, 5060 Ti, 5070 Ti, 4090 and 5090 got no PC at a price that pays. The same class varies about 15% from PC to PC, because owners cap the card's power. Salad has AMD classes, but the miner runs on CUDA only.

## Things that went wrong

- **The 12.8.1 CUDA image sat in "preparing" for 45+ minutes.** The 12.2.2 base image starts at once. The miner brings its own CUDA runtime.
- **Salad refuses a command containing "; curl".** It returns HTTP 400 with no detail, so curl follows `&&`.
- **A priority change is refused while a group is "pending"** (preparing its image).
- **The log API takes about 10 s a call and sometimes returns 408 or 500.** The watcher makes one query a round for all groups. A query per group made a round take 11 minutes.
- **A recreated group mines under the same worker name**, so only pool shares after the group was created count.
