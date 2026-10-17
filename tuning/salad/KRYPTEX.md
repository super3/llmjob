# Kryptex support: instructions for a session with a GPU

Goal: let the Salad watcher run some boxes on Kryptex's PPS+ pool for Pearl, so each box's pay is known
from its accepted shares. Work on PR #261, branch `claude/amazing-newton-mu7lgq`. Follow `CLAUDE.md`
(tests before and after, `git status` in replies, PR title and description updated on every push).

## Where things stand (17 October 2026)

- The watcher side is built and tested (`tuning/salad/watcher.py`, `KX_API`, `kx_sync`, `kx_switch`,
  `kx_failed`, `kx_pool`, the `(K)` mark, PRL counted per pool as `prl_kx`). It is switched off live:
  config `kryptex_groups` is 0. See "Kryptex (PPS+)" in `tuning/salad/README.md`.
- What blocks it: the miner can't be pointed at Kryptex on a Salad box.
  - The CLI only takes a HeroMiners region (`--region`, `earn/src/shared/config.js` `REGIONS`).
  - The watcher tried to redirect the HeroMiners names with `/etc/hosts` and a local `socat` forward.
    `/etc/hosts` is read-only in Salad containers, so the box command stops before box.sh runs.
  - Salad boxes download the CLI from a GitHub release (`tuning/box.sh` pulls
    `llmjob-earn-cli-linux` from `releases/download/$VER`), so a new CLI flag needs a release before
    boxes can use it.
- Unknown: whether Kryptex accepts our miner's protocol at all. Nobody has connected to it yet. The
  Claude cloud session that built the watcher can't reach port 7048. A machine with a GPU can.

## Kryptex facts (from `pool.kryptex.com/prl` and its API, 17 October)

- Stratum: `prl-us.kryptex.network:7048` (plain TCP), SSL on 8048. Other regions: `prl`, `prl-eu`,
  `prl-br`, `prl-sg`, `prl-hk`, `prl-ru`, `prl-ae`, all `.kryptex.network`.
- Login: "wallet/worker" (slash), or the wallet alone. `solo:wallet` is SOLO mining; don't use it.
- PPS+ fee 3%, SOLO 1%, payout from 1 PRL. Static share difficulty 2097152, the same as HeroMiners'
  (`job_id` ends in `_2097152` there).
- Kryptex lists `pearl-source` among the miners on the pool, so Pearl's reference stratum likely works.
- API: `https://pool.kryptex.com/openapi.yaml`. The watcher uses
  `/prl/api/v3/miner/workers/{wallet}`, `/prl/api/v1/miner/balance/{wallet}`,
  `/prl/api/v1/miner/payouts/{wallet}` and `/prl/api/v1/pool/info`.

## Our miner's protocol (what Kryptex has to accept)

`earn/src/shared/miner/stratum.js`, captured from HeroMiners:

- One message to log in: `{"id":1,"method":"mining.authorize","params":{"wallet":"prl1…","worker":"name"}}`.
  The params are an object, not the classic `["wallet.worker","x"]` array.
- Jobs: `mining.notify` with object params `{job_id, header, target, height, cert_version}`.
- Shares: `mining.submit` with `{job_id, plain_proof, hs}`, the proof as base64
  (`serializeProof`). A comment there says Kryptex may want a different container ("protobuf"); that
  was never checked.

## Step 1: find out whether Kryptex takes our protocol (no code change)

On the GPU machine:

```sh
W=<the PRL wallet, from PRL_WALLET or tuning/salad config>
printf '{"id":1,"method":"mining.authorize","params":{"wallet":"%s","worker":"probe"}}\n' "$W" \
  | timeout 15 nc prl-us.kryptex.network 7048
printf '{"id":1,"method":"mining.authorize","params":{"wallet":"%s/probe","worker":"probe"}}\n' "$W" \
  | timeout 15 nc prl-us.kryptex.network 7048
```

- Good sign: `{"id":1,"result":true,...}` followed by a `mining.notify` with the same object shape as
  HeroMiners'. Note which login form worked.
- If the reply is a different shape (array params, a subscribe step first, binary/protobuf), write
  down exactly what came back. The miner needs a Kryptex dialect in `stratum.js`; scope it before
  building anything else.

## Step 2: add `--pool host:port` to the CLI (in `earn/`)

The override already exists inside: `settings.endpoint`, read by `resolveEndpoint()` in
`earn/src/shared/config.js` (the desktop app uses it). Wire a flag to it:

1. `earn/src/shared/cliArgs.js`
   - add `'--pool'` to `VALUE_FLAGS`, and a line in `USAGE` (for example
     `--pool <host:port>  Mine on this stratum endpoint instead of a HeroMiners region`);
   - in `buildSettings`, validate it with `splitEndpoint` (host and a port 1–65535, else an error like
     the region's) and set `settings.endpoint`;
   - when `--pool` is given, region detection must not run and must not override it. In
     `earn/src/cli/earn-cli.js`, `if (!settings.regionProvided) settings.region = await detectRegion();`
     (line ~783) should be skipped when `settings.endpoint` is set, and the `pool:` log line (~804)
     should show the endpoint without a region label.
2. Login format: if step 1 showed Kryptex wants `wallet/worker`, make `buildAuthorize` send that when
   the endpoint isn't HeroMiners' (or add `--pool-login slash`). Keep HeroMiners' message exactly as
   it is: it's the captured golden fixture in the tests.
3. Check other HeroMiners-only code paths still behave with another pool: `earn/src/shared/balance.js`
   (HeroMiners balance API), `poolLookup.js` (DNS fallback chain, fine for any host), the stats file
   and the network-board report (they may show a HeroMiners region).
4. Tests: `cd earn && npm test` before and after. Add tests for the flag (valid, missing port, bad
   port, with and without `--region`) and, if you add one, the Kryptex login.

## Step 3: prove it on the GPU

Run the CLI from source against Kryptex for 20–30 minutes:

```sh
cd earn && node src/cli/earn-cli.js -a "$W" -w kx-test --mode mining --no-serve --no-update \
  --pool prl-us.kryptex.network:7048
```

- Accepted shares should climb at about one per 47 s per 200 TH/s, with no rejects.
- Check Kryptex sees the worker:
  `curl -s https://pool.kryptex.com/prl/api/v3/miner/workers/$W` (look for `kx-test` and its
  `hashrate` and `last_share`), and the balance at `/prl/api/v1/miner/balance/$W` should start to grow
  within minutes (PPS+ credits per share).
- Write down the TH/s, accepted and rejected counts, and Kryptex's figures. They go in the PR
  description and in the README's Kryptex section.

## Step 4: release, then switch the watcher over

1. Release v0.6.1 the repo's usual way (version in `earn/package.json`, a `v0.6.1` tag, which runs
   `.github/workflows/miner-build.yml`; check `benchmark.md` first, per `CLAUDE.md`). Ask the user
   before tagging: a release reaches every user.
2. In `tuning/salad/watcher.py`, replace the `/etc/hosts` + `socat` redirect in `cmd_for(...,
   kryptex=...)` with a `--pool` flag, the same way Monero adds `--xmr-address`: a `sed` edit that puts
   `--pool <kryptex_host>` into box.sh's default `"flags"`. Remove `HM_REGIONS` and the socat install.
   Kryptex boxes need the new release: when `kx` is on, use at least v0.6.1, the way `box()` bumps the
   version for Monero (`xmr_ver`).
3. Update the tests in `TestKryptex` (`tuning/salad/test_watcher.py`): the command test should check
   for `--pool prl-us.kryptex.network:7048` and no socat or hosts file. Run
   `python3 -m unittest discover -s tuning/salad` (all must pass).
4. Turn it on in the live config with `kryptex_groups: 1`. The watcher switches the weakest-earning
   HeroMiners box: it updates the command and moves the box to a new PC (Salad's instance restart
   doesn't restart anything). Watch for `KRYPTEX ON`, the box's `[minerlog g0]` lines, and its first
   shares. If Kryptex rejects the work, the box goes back to HeroMiners by itself, and two such
   failures stop the test until `kryptex_token` changes.
5. After a few hours, compare Kryptex's credited PRL with the box's estimate (the report's Kryptex
   line). Expect about 97% against HeroMiners' 92%. If it holds, raise `kryptex_groups` to 3.

## Things not to do

- Don't touch org `development1`, Vast or Clore. Only org `llmjob`.
- Never print the Salad key, the wallet's SafeTrade keys or any proxy login.
- Don't change HeroMiners' stratum messages: every live box and user depends on them.
