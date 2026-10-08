# Tuning mining on rented Vast boxes

The goal is to earn more per dollar of rent on the boxes we rent on Vast, on-demand and
spot (interruptible). Earnings scale with hashrate, so every gain in TH/s on a box we already
pay for is profit.

## How a box is tuned

Each box runs [`box.sh`](box.sh). It starts one miner per GPU (`PEARL_GPU_INDEX`), each
under its own pool worker name (`<worker>-g<N>`), on the published v0.5.12 release. Every
2 minutes it fetches `tuning/control/<worker>.json` from this branch and restarts only the
GPUs whose settings changed. Everything else keeps mining.

What a control file can set, per GPU or as the box default:

| Setting | What it does |
|---|---|
| `"core": "release"` | The CLI's own pick of its CUDA 12 or CUDA 13 core (the default) |
| `"core": "cu12"` / `"cu13"` | Force one of the release's two cores |
| `"core": "build:<name>"` | A core built on the box from this branch, listed under `"builds"` with a commit and `-D` defines |
| `"flags"` | Extra CLI flags, e.g. `--mine-mem-clock 0` |
| `"env"` | Extra environment for the miner |

A built core runs only after it passes the hit check on its GPU (400 hits recomputed from
scratch, the way the pool verifies them). Settings whose miner keeps exiting are dropped and
the GPU goes back to the release default.

## Rules for a test

1. **Change one or two GPUs at a time.** The box's other GPUs are the control group: same
   host, same power and cooling, measured over the same minutes.
2. **Measure after warm-up.** Ignore each miner's first 2 minutes, then compare at least
   10 minutes of readings from the test GPUs against the control GPUs.
3. **A gain counts when it holds.** At least +1% over the control GPUs across the whole
   window, with no rejected or invalid shares at the pool. Then roll it out to the rest of
   the box and record it in [`RESULTS.md`](RESULTS.md).
4. **Single-GPU boxes have no control group,** so they only take settings already verified
   on the same card elsewhere, compared before and after over at least 10 minutes each.
5. **Code changes go on this branch** under `earn/native`, are built on the box through
   `"builds"`, and must pass the hit check and `npm test` in `earn/`.
6. **Write down what didn't work too,** so no one tries it again.

## Files

- [`box.sh`](box.sh): the script each rented box runs.
- `control/<worker>.json`: each box's live settings. These steer running boxes; they are an
  experiment log, not part of the miner.
- [`RESULTS.md`](RESULTS.md): what was tried on which card, and the result.
