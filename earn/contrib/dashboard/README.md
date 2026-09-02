# gpu-dashboard

A full-screen terminal view of one rig: mode, hashrate, shares, last share, the
served model, generation and prefill throughput, pool windows, an earnings
estimate, and GPU/MEM/PWR/TEMP/FAN bars. Arrow keys switch the node between
`AUTO`, `MINE`, `LLM` and `OFF`.

```
 Device 0 [ GeForce RTX 5090 ] PCIe 4@16x RX 602 KiB/s  MODE   AUTO (MINING)   MINE   LLM   OFF

 HASHRATE 120.7 TH/s  worker rig-1  last share 25s  |  POOL  [10m] 121 TH/s  [1h] 125 TH/s
 MINER 0.5.0    PRL $0.33    EST   $0.07 / hr    $1.59 / day    $47.55 / month
 LLM  READY port 8000  mining - a request wakes the model (~5s)
```

## This is contrib, not a supported feature

It is a 300-line bash script written for one setup and shared because it is
useful, not because it is portable. It requires **Linux with systemd**, an
**NVIDIA card with `nvidia-smi`**, and it drives the node through a systemd unit
it expects to own. It reads the node's state by running `systemctl`, `ps` and
`journalctl` and by polling the gate's HTTP endpoint — it is scraping, not an
API, so a change to a log line can break a panel.

If you want a dashboard that works everywhere, the right shape is a `--watch`
flag rendering in-process from state the miner already holds, with no scraping.
That does not exist yet.

## Install

```sh
sudo cp llmjob-earn.default.example /etc/default/llmjob-earn
sudo "$EDITOR" /etc/default/llmjob-earn      # at minimum EARN_ADDRESS and EARN_BIN
sudo ./install.sh
```

`install.sh` installs the unit and the script, starts the node, and prints where
it got to. `uninstall.sh` reverses it. Neither touches an existing dashboard
without backing it up first.

Then run it wherever you want it visible — it is a normal foreground program:

```sh
gpu-dashboard                                    # in a terminal
tmux -L gpu new-session -d -x 128 -y 40 gpu-dashboard   # detached, 128 cols
```

It wants **at least 128 columns**. Narrower and the model row wraps.

## How mode switching works

Arrow keys move the highlight; the switch fires once the selection settles.
Switching rewrites `EARN_MODE` in `/etc/default/llmjob-earn` and restarts the
unit — systemd re-reads an `EnvironmentFile` only at exec, so a restart is what
makes a mode change take effect. `OFF` stops the unit and records the choice, so
a reboot does not resurrect the previous mode.

`AUTO` shows a sub-state — `AUTO (MINING)`, `AUTO (LLM)`, `AUTO (SWITCHING)` —
read from the gate's `/health`. Polling that endpoint is safe: probes
deliberately do not count as inference demand, so watching the node cannot keep
it awake.

## Known rough edges

- Hashrate, shares and throughput come from `journalctl` greps of human-readable
  log lines. A wording change upstream breaks them silently, showing `?` or `--`
  rather than an error. Prefer `--stats-file` if you are building something.
- `MINER` shows the version of the binary the *unit* runs. If that cannot be
  determined it falls back to `EARN_FALLBACK_BIN` and marks the value with `?`,
  because that may be a different build from the one running.
- The four modes map onto one process, so `MINE` and `AUTO` differ only in
  whether the gate is listening. Switching between them costs a restart (~3 s)
  even though the miner itself is unchanged.
- Pool columns and the earnings estimate call HeroMiners and CoinGecko directly.
  They read `?` when either is unreachable, and they are only meaningful if you
  mine to that pool.
