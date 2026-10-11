#!/usr/bin/env python3
# A safety net for when watcher.py isn't running, for example because the machine it ran on went away.
# It keeps no state: it reads Salad and the pool, and stops every biz- group that is on a running PC but
# from which the pool has seen no work in the last hour, while the group is at least an hour old.
# The watcher never lets that happen (it leaves a PC that isn't mining within 10 min, and deletes a group
# after 3 tries), so a hit means nothing is watching. Stopping is reversible: start the group again from
# the portal, or bump restart_token once the watcher runs. Run it hourly, e.g. from a Routine:
#   PRL_WALLET=prl1... python3 -I tuning/salad/guard.py [--dry-run]
import importlib.util, os, sys, tempfile

os.environ.setdefault("SALAD_MINE_DIR", tempfile.mkdtemp())   # the watcher module wants one; nothing is written
spec = importlib.util.spec_from_file_location("watcher", os.path.join(os.path.dirname(os.path.abspath(__file__)), "watcher.py"))
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


def check(wallet, dry=False):
    """Returns (lines to print, names stopped). Acts on nothing when Salad or the pool can't be read."""
    s, j = W.http("GET", W.CB)
    if s != 200 or not isinstance(j, dict):
        return [f"Salad: listing groups failed (HTTP {s}); did nothing"], []
    s, pj = W.http("GET", f"https://pearl.herominers.com/api/stats_address?address={wallet}&longpoll=false", timeout=30)
    if s != 200 or not isinstance(pj, dict):
        return [f"Pool: stats failed (HTTP {s}); did nothing"], []
    ws = pj.get("workers") or []
    if isinstance(ws, dict):
        ws = [dict(v, name=k) for k, v in ws.items()]
    workers = {w["name"]: w for w in ws if isinstance(w, dict) and w.get("name")}
    t = W.now()
    out, stopped, checked = [], [], 0
    for g in j.get("items") or []:
        name = g.get("name") or ""
        if not name.startswith("biz-") or (g.get("current_state") or {}).get("status") != "running":
            continue
        checked += 1
        try:
            age = t - W.parse_iso(g.get("create_time") or "")
        except (AttributeError, TypeError, ValueError):
            continue
        if age < 3600:
            continue
        si, ji = W.http("GET", f"{W.CB}/{name}/instances")
        if si != 200 or not isinstance(ji, dict):
            out.append(f"{name}: instances failed (HTTP {si}); left alone")
            continue
        if not any(i.get("state") == "running" and i.get("machine_id") for i in ji.get("instances") or []):
            continue
        w = workers.get(f"{name}-g0") or {}
        try:
            if float(w.get("hashrate_1h") or 0) > 0 or float(w.get("hashrate") or 0) > 0:
                continue
        except (TypeError, ValueError):
            continue
        why = "on a running PC, but the pool has seen no work from it in the last hour"
        if dry:
            out.append(f"WOULD STOP {name}: {why}")
            continue
        s, _ = W.http("POST", f"{W.CB}/{name}/stop")
        out.append(f"STOPPED {name}: {why} (HTTP {s})" if 200 <= s < 300 else f"STOP FAILED {name}: HTTP {s}")
        if 200 <= s < 300:
            stopped.append(name)
    out.append(f"Checked {checked} running biz- group{'s' if checked != 1 else ''}; stopped {len(stopped)}"
               + (" (dry run)" if dry else ""))
    return out, stopped


def main():
    wallet = os.environ.get("PRL_WALLET", "")
    if not wallet.startswith("prl1"):
        sys.exit("Set PRL_WALLET to the payout address")
    lines, _ = check(wallet, dry="--dry-run" in sys.argv)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
