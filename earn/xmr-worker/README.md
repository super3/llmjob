# RandomX CPU worker (Monero)

`llmjob-xmr-worker` hashes RandomX on the CPU for LLMJob Earn's opt-in Monero
miner. It is our own small program around
[tevador/RandomX](https://github.com/tevador/RandomX) (BSD-3-Clause), the
library Monero itself verifies blocks with. It has no pool code, no wallet
code and no fee: the app sends it jobs and submits the shares it finds
(`src/main/xmrEngine.js`, `src/shared/xmr/`).

The app runs it as a child process, never inside its own. A crash here, or an
antivirus deleting or killing this file, stops the CPU miner; the app logs it,
restarts it a few times, and the GPU miner carries on.

## Protocol

One JSON object per line each way, on stdin and stdout. The top of
`src/xmr_worker.cpp` lists the commands and events, and
`src/shared/xmr/worker.js` is the app's side. The worker says `"protocol": 1`
in its first line; the app refuses a worker with another number, so a stale
copy from an older release is never run.

## What it does

- Checks itself before mining: seven of RandomX's own test vectors (v1, v2 and
  a v2 commitment), hashed in light mode with the flags mining will use (JIT,
  hardware AES, the Argon2 build). A wrong answer stops it before any share.
- Builds the 2080 MiB dataset on its threads, in chunks so a quit is quick, and
  again when a job brings a new seed (every 2048 Monero blocks, about 2.8 days).
  After each build it checks one hash against light mode.
- Tries huge pages (hugetlbfs on Linux, large pages on Windows) and falls back.
  On Linux without reserved huge pages it asks for transparent huge pages for
  the dataset. The log says which it got.
- Mines at the priority the app asks for: `idle` (`SCHED_IDLE` threads on
  Linux, the idle priority class on Windows), `low` (nice 19, or below-normal)
  or `normal`. The app asks for low on Linux and idle on Windows
  (`defaultPriority` in `src/shared/xmr/worker.js` says why). The GPU miner
  is kept whole by the free CPU the thread plan always leaves, not by the
  priority.
- Compares each hash with the job's target itself and reports only shares, each
  checked again in light mode before it goes out.
- Exits when stdin closes, so it never outlives the app (on Linux also when the
  app dies, via `PR_SET_PDEATHSIG`).

Monero's v17 fork will change the proof of work to RandomX v2 with
commitments. The worker already does both (`"variant": "v2"`, `"commit": true`
on a job); turning them on is an entry in `ALGOS` in
`src/shared/xmr/stratum.js` once pools name the new algorithm.

## Build

```bash
scripts/build-xmr-worker.sh            # build and self-test
scripts/build-xmr-worker.sh --stage    # also copy to vendor/native/
```

It fetches RandomX at a pinned commit (in the script), then builds with CMake.
Linux gives one static binary that runs on any x86-64 distribution; Windows
(Git Bash with Visual Studio's CMake) gives an `.exe` with the static MSVC
runtime. CI builds both in `miner-build.yml`.

Other options, for trying it by hand:

```bash
llmjob-xmr-worker --self-test              # the known-answer check, then exit
llmjob-xmr-worker --threads 6 --bench 60   # mine a fixed job, no pool, for 60 s
llmjob-xmr-worker --help
```
