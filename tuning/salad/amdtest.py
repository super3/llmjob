#!/usr/bin/env python3
"""Test boxes for the AMD (HIP) Pearl core on Salad's AMD classes. Manages only the groups amdtest-rdna3 and
amdtest-rdna4 (it refuses any other name, so it can never touch the mining fleet's biz- groups). Each box runs
amdtest.sh, which prints "[amdtest] HH:MM:SS <tag> ..." lines; `logs` and `wait` read them back.

  amdtest.py create NAME CLASS --sha SHA [--tasks LIST] [--env K=V ...] [--patch-from DIR] [--cpu 4 --mem 8192]
                                [--retry SECS]
  amdtest.py set-env NAME [K=V ...] [--sha SHA] [--tasks LIST] [--unset K ...] [--patch-from DIR | --no-patch]
                          [--no-restart]
  amdtest.py status NAME
  amdtest.py logs NAME [--since 30m] [--any-machine] [--grep REGEX]
  amdtest.py wait NAME --until REGEX [--since 2m] [--timeout 480]
  amdtest.py patch DIR [--sha SHA]     size of the patch --patch-from would send
  amdtest.py start|stop|realloc|delete NAME

CLASS is Salad's name ("AMD RX 7900 XTX (24GB)") or a short one: 7900xtx, 7900xt, 7800xt, 9070xt, 9060xt.
The box's command never changes: it fetches amdtest.sh at SCRIPT_SHA or BASE_SHA and runs it, so `set-env` is
how a new run starts. It updates the environment and then restarts the instance (--no-restart leaves that to
Salad). See "AMD test boxes" in README.md for how long that took. --patch-from DIR sends DIR's uncommitted and
untracked changes against BASE_SHA as PATCH_GZ_B64, so a change can be tried without pushing it. But on 22
October no box given a patch printed anything (README.md, "AMD test boxes"): push and use --sha instead.

Uses watcher.py's Salad client. The key is added by the proxy (or SALAD_API_KEY); nothing here prints it or any
environment variable of the box.
"""
import argparse, base64, datetime as dt, gzip, importlib.util, os, re, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("SALAD_MINE_DIR", tempfile.mkdtemp())   # never the live watcher's directory
_spec = importlib.util.spec_from_file_location("watcher", os.path.join(HERE, "watcher.py"))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

NAMES = ("amdtest-rdna3", "amdtest-rdna4")
IMAGE = "rocm/dev-ubuntu-24.04:7.2"
CLASSES = {"7900xtx": "AMD RX 7900 XTX (24GB)", "7900xt": "AMD RX 7900 XT (20GB)", "7800xt": "AMD RX 7800 XT (16GB)",
           "9070xt": "AMD RX 9070 XT (16GB)", "9060xt": "AMD RX 9060 XT (16GB)"}
TAG = "[amdtest]"

# The container command. It never changes; the environment says what to run. dash, not bash.
BOOT = r"""say() { echo "[amdtest] $(date -u +%T) $*"; }
say "boot container started"
if [ -z "$BASE_SHA" ]; then say "boot no BASE_SHA; idle"; exec sleep infinity; fi
command -v curl >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq --no-install-recommends curl ca-certificates >/dev/null 2>&1; }
u="https://raw.githubusercontent.com/super3/llmjob/${SCRIPT_SHA:-$BASE_SHA}/tuning/salad/amdtest.sh"
n=0
until curl -fsSL "$u" -o /root/amdtest.sh; do n=$((n+1)); if [ $n -ge 5 ]; then say "boot FAIL fetch $u"; exec sleep infinity; fi; sleep 20; done
bash /root/amdtest.sh 2>&1 | tee -a /root/amdtest.log
say "boot amdtest.sh exited"
exec sleep infinity"""


def check_name(name):
    if name not in NAMES:
        sys.exit(f"amdtest.py manages only {', '.join(NAMES)}, not {name!r}")
    return name


def class_name(arg):
    """A class name from Salad's name or a short one (7900xtx). Exact short names only, so 7900xt is never
    read as 7900xtx."""
    key = re.sub(r"[^a-z0-9]", "", arg.lower().replace("amd", "").replace("rx", ""))
    return CLASSES.get(key, arg)


def parse_kv(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"not K=V: {it!r}")
        k, v = it.split("=", 1)
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            sys.exit(f"bad variable name {k!r}")
        out[k] = v
    return out


def body(name, class_id, env, cpu=4, mem=8192, storage_gb=10, priority="high"):
    """The create request: one replica, high priority, the ROCm image, BOOT as the command."""
    return {"name": name, "display_name": name, "replicas": 1, "restart_policy": "never", "autostart_policy": True,
            "container": {"image": IMAGE, "priority": priority,
                          "resources": {"cpu": cpu, "memory": mem, "gpu_classes": [class_id],
                                        "storage_amount": storage_gb * 1024 ** 3},
                          "command": ["/bin/sh", "-c", BOOT], "environment_variables": env}}


def env_patch(current, sets, unsets):
    """The update body for new variables. Salad replaces the whole environment_variables map on a PATCH (it does
    not merge it, and stores a null as a value: found 22 October), so this sends the full map: the current one
    with sets applied and unsets left out."""
    env = {k: v for k, v in (current or {}).items() if v is not None and k not in unsets}
    env.update(sets)
    return {"container": {"environment_variables": env}}


def since_t(s, now):
    """A start time from "30m", "2h", "90s", "13:20" or "13:20:30" (UTC, today, or yesterday if that is in the future) or ISO."""
    m = re.match(r"^(\d+)([smh])$", s)
    if m:
        return now - int(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[m.group(2)]
    m = re.match(r"^(\d\d?):(\d\d)(?::(\d\d))?$", s)
    if m:
        d = dt.datetime.fromtimestamp(now, dt.timezone.utc).replace(hour=int(m.group(1)), minute=int(m.group(2)),
                                                                     second=int(m.group(3) or 0), microsecond=0)
        t = d.timestamp()
        return t - 86400 if t > now + 60 else t
    return w.parse_iso(s)


def pick_lines(items, machine=None, grep=None):
    """Salad log items into (time, machine, text) for amdtest lines, oldest first, without repeats. machine: only
    that PC's lines (None: every PC)."""
    out, seen = [], set()
    for it in sorted(items, key=lambda it: it.get("time") or ""):
        text = (it.get("text_log") or "").strip()
        mach = ((it.get("resource") or {}).get("labels") or {}).get("machine_id") or ""
        if TAG not in text or (machine and mach != machine):
            continue
        text = text[text.index(TAG):]
        if grep and not re.search(grep, text):
            continue
        key = (it.get("time"), mach, text)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def fmt_line(t, mach, text):
    return f"{(mach or '?')[:8]} {text}"


def make_patch(repo, base):
    """gzip+base64 of every change in repo's working tree against base, untracked files included, using a
    throwaway index so the repo's own index is left alone. Returns (b64, raw diff bytes)."""
    idx = tempfile.NamedTemporaryFile(delete=False)
    idx.close()
    try:
        env = dict(os.environ, GIT_INDEX_FILE=idx.name)
        subprocess.run(["git", "-C", repo, "read-tree", base], env=env, check=True)
        subprocess.run(["git", "-C", repo, "add", "-A"], env=env, check=True)
        diff = subprocess.run(["git", "-C", repo, "diff", "--cached", "--binary", base], env=env, check=True,
                              capture_output=True).stdout
    finally:
        os.unlink(idx.name)
    return base64.b64encode(gzip.compress(diff, 9)).decode(), diff


def head(repo):
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


# Salad calls -------------------------------------------------------------------------------------------------

def class_id(name):
    s, j = w.http("GET", f"{w.BASE}/gpu-classes")
    ids = {c["name"]: c["id"] for c in (j or {}).get("items", [])} if s == 200 else {}
    if name not in ids:
        sys.exit(f"no Salad class {name!r}; AMD classes: {sorted(n for n in ids if 'AMD' in n)}")
    return ids[name]


def group(name):
    s, j = w.http("GET", f"{w.CB}/{name}")
    return j if s == 200 else None


def instances(name):
    s, j = w.http("GET", f"{w.CB}/{name}/instances")
    return ((j or {}).get("instances") or (j or {}).get("items") or []) if s == 200 else []


def current(name):
    """The instance on a PC now (downloading, creating or running), or None."""
    on = [i for i in instances(name) if i.get("machine_id") and i.get("state") in ("downloading", "creating",
                                                                                      "running")]
    return on[0] if on else None


def read_logs(name, start, end, depth=0):
    """Every log line of the group in [start, end]. Salad returns at most 100 a call, so a full window is split
    in half and read again (as watcher.py does)."""
    q = f'resource.type = "container" AND resource.labels.container_group_name = "{name}"'
    b = {"start_time": w.iso(start), "end_time": w.iso(end), "query": q, "page_size": 100, "sort_order": "asc"}
    for attempt in range(3):   # the log service times out now and then (HTTP 408)
        s, j = w.http("POST", f"{w.BASE}/log-entries", b, timeout=90)
        if s == 200:
            break
        sys.stderr.write(f"log-entries: HTTP {s} {str(j)[:120]}\n")
        time.sleep(5)
    else:
        return []
    items = (j or {}).get("items") or []
    if len(items) < 100 or depth >= 6 or end - start < 2:
        return items
    mid = start + (end - start) / 2
    return read_logs(name, start, mid, depth + 1) + read_logs(name, mid, end, depth + 1)


def env_summary(env):
    """Variables as K=V, with a patch or script shown by its size only."""
    out = []
    for k in sorted(env or {}):
        v = env[k]
        out.append(f"{k}=<{len(v)} chars>" if k.endswith("_GZ_B64") else f"{k}={v}")
    return " ".join(out)


def cmd_create(a):
    name = check_name(a.name)
    cls = class_name(a.cls)
    env = parse_kv(a.env)
    env.setdefault("WORKER", name)
    if a.sha:
        env["BASE_SHA"] = a.sha
    if a.tasks:
        env["TASKS"] = a.tasks
    if "BASE_SHA" not in env:
        sys.exit("needs --sha (a pushed commit)")
    if a.patch_from:
        env["PATCH_GZ_B64"], diff = make_patch(a.patch_from, env["BASE_SHA"])
        print(f"patch: {len(diff)} bytes, {len(env['PATCH_GZ_B64'])} chars as gzip+base64")
    b = body(name, class_id(cls), env, a.cpu, a.mem, a.storage_gb, a.priority)
    t_end = time.time() + min(a.retry, 520)
    while True:
        s, j = w.http("POST", w.CB, b)
        if 200 <= s < 300:
            print(f"created {name} on {cls}, {a.priority}, {a.cpu} vCPU, {a.mem} MB: {env_summary(env)}")
            # Salad sometimes stops a new group at once, before it has a PC (amdtest-rdna4, 22 October). Start it.
            time.sleep(20)
            if ((group(name) or {}).get("current_state") or {}).get("status") == "stopped":
                s, _ = w.http("POST", f"{w.CB}/{name}/start")
                print(f"it was stopped at once; started it again: HTTP {s}")
            return
        msg = str(j)[:300]
        print(f"{time.strftime('%H:%M:%S', time.gmtime())}Z create {name}: HTTP {s} {msg}", flush=True)
        if "quota" not in msg or time.time() + 60 > t_end:
            sys.exit(1)
        time.sleep(60)


def cmd_set_env(a):
    name = check_name(a.name)
    g = group(name)
    if not g:
        sys.exit(f"no group {name}")
    cur = (g.get("container") or {}).get("environment_variables") or {}
    sets = parse_kv(a.env)
    if a.sha:
        sets["BASE_SHA"] = a.sha
    if a.tasks:
        sets["TASKS"] = a.tasks
    unsets = list(a.unset or [])
    if a.patch_from:
        base = sets.get("BASE_SHA") or cur.get("BASE_SHA")
        sets["PATCH_GZ_B64"], diff = make_patch(a.patch_from, base)
        print(f"patch: {len(diff)} bytes against {base[:12]}, {len(sets['PATCH_GZ_B64'])} chars as gzip+base64")
    elif a.no_patch:
        unsets.append("PATCH_GZ_B64")
    # A value that is already set doesn't make a new version, and Salad wouldn't restart the box, so a run can
    # always be repeated: RUN_ID changes every time.
    sets["RUN_ID"] = str(int(time.time()))
    s, j = w.http("PATCH", f"{w.CB}/{name}", env_patch(cur, sets, unsets), ctype="application/merge-patch+json")
    if not 200 <= s < 300:
        sys.exit(f"set-env {name}: HTTP {s} {str(j)[:300]}")
    new = ((j or {}).get("container") or {}).get("environment_variables") or {}
    print(f"set-env {name} at {time.strftime('%H:%M:%S', time.gmtime())}Z: {env_summary(new)}")
    if a.restart:
        inst = current(name)
        if inst:
            rs, rj = w.http("POST", f"{w.CB}/{name}/instances/{inst.get('instance_id') or inst.get('id')}/restart")
            print(f"restart: HTTP {rs} {'' if 200 <= rs < 300 else str(rj)[:200]}")
        else:
            print("restart: no instance on a PC")


def cmd_status(a):
    name = check_name(a.name)
    g = group(name)
    if not g:
        print(f"{name}: no such group")
        return
    cs = g.get("current_state") or {}
    c = g.get("container") or {}
    res = c.get("resources") or {}
    print(f"{name}: {cs.get('status')} ({cs.get('description') or ''}), version {g.get('version')}, "
          f"{g.get('priority') or c.get('priority')}, {res.get('cpu')} vCPU, {res.get('memory')} MB, image {c.get('image')}")
    print(f"  env: {env_summary(c.get('environment_variables'))}")
    if cs.get("status") == "stopped":
        print("  stopped: `amdtest.py start` it (Salad stops a new group before a PC sometimes)")
    for i in instances(name):
        print(f"  instance {i.get('instance_id') or i.get('id')}: {i.get('state')}, machine {i.get('machine_id')}, "
              f"version {i.get('version')}, updated {i.get('update_time')}")


def cmd_logs(a):
    name = check_name(a.name)
    t = time.time()
    mach = None
    if not a.any_machine:
        inst = current(name)
        mach = inst.get("machine_id") if inst else None
    lines = pick_lines(read_logs(name, since_t(a.since, t), t), mach, a.grep)
    for ln in lines:
        print(fmt_line(*ln))
    if not lines:
        print(f"(no {TAG} lines since {a.since}" + (f" from machine {mach[:8]}" if mach else "") + ")")


def cmd_wait(a):
    """Polls the logs every 30 s and prints new lines until one matches --until, or --timeout (at most 520 s)."""
    name = check_name(a.name)
    t0 = time.time()
    start = since_t(a.since, t0)
    shown = set()
    while True:
        t = time.time()
        for ln in pick_lines(read_logs(name, start, t)):
            if ln in shown:
                continue
            shown.add(ln)
            print(fmt_line(*ln), flush=True)
            if re.search(a.until, ln[2]):
                return
        if t - t0 > min(a.timeout, 520):
            print(f"(no line matched {a.until!r} in {int(t - t0)} s)")
            sys.exit(1)
        time.sleep(30)


def cmd_patch(a):
    base = a.sha or head(a.dir)
    b64, diff = make_patch(a.dir, base)
    print(f"patch against {base[:12]}: {len(diff)} bytes, {len(b64)} chars as gzip+base64")


def cmd_simple(a):
    name = check_name(a.name)
    if a.cmd == "delete":
        s, j = w.http("DELETE", f"{w.CB}/{name}")
    elif a.cmd == "realloc":
        inst = current(name) or (instances(name) or [None])[0]
        if not inst:
            sys.exit(f"{name}: no instance to reallocate")
        s, j = w.http("POST", f"{w.CB}/{name}/instances/{inst.get('instance_id') or inst.get('id')}/reallocate")
    else:
        s, j = w.http("POST", f"{w.CB}/{name}/{a.cmd}")
    print(f"{a.cmd} {name}: HTTP {s} {'' if 200 <= s < 300 else str(j)[:300]}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("name")
    c.add_argument("cls")
    c.add_argument("--sha")
    c.add_argument("--tasks")
    c.add_argument("--env", nargs="*", default=[])
    c.add_argument("--patch-from")
    c.add_argument("--cpu", type=int, default=4)
    c.add_argument("--mem", type=int, default=8192)
    c.add_argument("--storage-gb", type=int, default=10)
    c.add_argument("--priority", default="high", choices=w.PRIS)
    c.add_argument("--retry", type=int, default=0, help="seconds to keep retrying a quota error (at most 520)")
    c.set_defaults(fn=cmd_create)
    e = sub.add_parser("set-env")
    e.add_argument("name")
    e.add_argument("env", nargs="*")
    e.add_argument("--sha")
    e.add_argument("--tasks")
    e.add_argument("--unset", nargs="*", default=[])
    e.add_argument("--patch-from")
    e.add_argument("--no-patch", action="store_true")
    e.add_argument("--no-restart", dest="restart", action="store_false",
                   help="don't restart the instance; wait for Salad to (slow: often 8+ min, or not at all)")
    e.set_defaults(fn=cmd_set_env)
    s = sub.add_parser("status")
    s.add_argument("name")
    s.set_defaults(fn=cmd_status)
    lg = sub.add_parser("logs")
    lg.add_argument("name")
    lg.add_argument("--since", default="30m")
    lg.add_argument("--any-machine", action="store_true")
    lg.add_argument("--grep")
    lg.set_defaults(fn=cmd_logs)
    wt = sub.add_parser("wait")
    wt.add_argument("name")
    wt.add_argument("--until", required=True)
    wt.add_argument("--since", default="2m")
    wt.add_argument("--timeout", type=int, default=480)
    wt.set_defaults(fn=cmd_wait)
    pt = sub.add_parser("patch")
    pt.add_argument("dir")
    pt.add_argument("--sha")
    pt.set_defaults(fn=cmd_patch)
    for n in ("start", "stop", "realloc", "delete"):
        x = sub.add_parser(n)
        x.add_argument("name")
        x.set_defaults(fn=cmd_simple)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
