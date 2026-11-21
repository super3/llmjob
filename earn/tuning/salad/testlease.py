#!/usr/bin/env python3
"""Leases for Salad test boxes in the test org llmjob2, and the reaper that cleans up after agents.

Why: agents' test boxes were deleted while in use, and left running (and billed) when agents were stopped. A
lock was a bare folder with a free-text owner file, group names weren't tied to locks, and deletes weren't
checked. Now:

  * A lock is locks/salad-<slot> (slot 5..9), taken atomically with mkdir, with lease.json written straight after:
        {"agent", "slot", "groups": [{"name", "state": "creating|live|deleting"}], "heartbeat_utc"}
  * A group is named t<slot>-<agent>-<card>-<n> (t5-q4nv30-3090-1). amdtest.py refuses to create or change one
    unless the lease for that slot belongs to that agent. It writes the group as "creating" before the POST and
    "live" after, and every amdtest.py command on it refreshes the heartbeat.
  * A delete is verified: DELETE, then poll until Salad answers 404 (retrying 429 and 5xx). Only then is the group
    taken out of the lease.
  * Agents never take over a stale lock. The reaper is the only thing that frees one.
  * The box's hard cap (amdtest.py's BOX_MAX_SECS, 3 h) can only be raised by a coordinator's approval file,
    locks/approvals/<group>.json. An agent's `extend` only writes a request.

  testlease.py take --agent NAME [--slot N]       take a free lock (refuses one that exists, stale or not)
  testlease.py release --agent NAME --slot N      free your lock; refused while the lease still lists groups
  testlease.py heartbeat --agent NAME             refresh your lease's heartbeat (amdtest.py does it on its own)
  testlease.py show                               every lock, its lease and heartbeat age
  testlease.py extend --group G --hours N --reason "..."      agent: request a longer cap (then ask the coordinator)
  testlease.py approve --group G --until 2026-11-22T03:00Z|+6h [--reason "..."]   coordinator only
  testlease.py reap [--live | --dry-run]          the coordinator's 15-minute report runs this

Locks live in SALAD_LOCKS_DIR, or LOCKS_DIR below.
"""
import argparse, contextlib, datetime as dt, importlib.util, json, os, re, shutil, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))

# Timers. Change them here. -------------------------------------------------------------------------------------
HEARTBEAT_STALE_SECS = 10 * 60     # a lease whose heartbeat is older than this is stale (founder, 21 Nov)
JOB_IDLE_SECS = 10 * 60            # ... and a box with no job line in this long is idle; both together: reap
DEFAULT_CAP_SECS = 3 * 3600        # the box's hard cap without an approval (amdtest.py's BOX_MAX_SECS)
DELETE_TIMEOUT_SECS = 300          # how long a verified delete waits for the 404
DELETE_POLL_SECS = 10
# Day one (21-22 Nov 2026): the reaper only logs what it would do. After a day of clean dry runs in reaper.log,
# set this to True (or pass --live) so it stops and deletes for real.
REAP_LIVE_DEFAULT = False

ORG = "llmjob2"                    # the test org; the reaper never looks anywhere else
SLOTS = range(5, 10)
LOCKS_DIR = os.environ.get("SALAD_LOCKS_DIR") or (
    r"C:\Users\template\AppData\Local\Temp\claude\18634d96a113c206\5675ce09-95f3-4f49-ac47-74a8f5730983"
    r"\scratchpad\locks")
AGENT_RE = re.compile(r"[a-z0-9]{1,16}")
NAME_RE = re.compile(r"t([5-9])-([a-z0-9]{1,16})-([a-z0-9]{1,12})-([1-9][0-9]?)")
TAGS = ("[amdtest]", "[nvtest]")


def locks_dir():
    return os.environ.get("SALAD_LOCKS_DIR") or LOCKS_DIR


def now_s():
    return time.time()


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d)(:\d\d)?(\.\d+)?(Z|[+-]\d\d:\d\d)?$", s or "")
    if not m:
        raise ValueError(f"not a UTC time: {s!r}")
    base, sec, _, tz = m.groups()
    tz = "+00:00" if tz in (None, "Z") else tz
    return dt.datetime.fromisoformat(base + (sec or ":00") + tz).timestamp()


def parse_name(name):
    """(slot, agent, card, n) for a t<slot>-<agent>-<card>-<n> name, else None."""
    m = NAME_RE.fullmatch(name or "")
    return (int(m.group(1)), m.group(2), m.group(3), int(m.group(4))) if m else None


def lock_dir(slot):
    return os.path.join(locks_dir(), f"salad-{slot}")


def lease_path(slot):
    return os.path.join(lock_dir(slot), "lease.json")


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, obj):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    for i in range(5):   # Windows refuses a replace while a reader has the file open; try again briefly
        try:
            return os.replace(tmp, path)
        except PermissionError:
            if i == 4:
                raise
            time.sleep(0.2)


def read_lease(slot):
    lease = read_json(lease_path(slot))
    return lease if isinstance(lease, dict) and isinstance(lease.get("groups"), list) else None


@contextlib.contextmanager
def mutex(slot, wait=10.0):
    """A short lock around a lease's read-modify-write, so amdtest.py and the reaper don't lose each other's
    changes. A mutex older than 60 s is left from a crash and is broken."""
    m = os.path.join(lock_dir(slot), "lease.mutex")
    t_end = time.time() + wait
    while True:
        try:
            os.mkdir(m)
            break
        except FileNotFoundError:   # no lock folder, so no lease to guard; never make the folder (that's a take)
            yield
            return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(m) > 60:
                    os.rmdir(m)
                    continue
            except OSError:
                pass
            if time.time() > t_end:
                raise SystemExit(f"lease for salad-{slot} is busy; try again")
            time.sleep(0.1)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.rmdir(m)


def held_by(agent):
    return [s for s in SLOTS if (read_lease(s) or {}).get("agent") == agent]


# Agent side --------------------------------------------------------------------------------------------------

def take(agent, slot=None, now=None):
    """Takes a free lock for agent: mkdir is the atomic step, lease.json follows at once. Refuses an existing lock
    even when its lease is stale (the reaper frees those), and a second lock for an agent that holds one."""
    if not AGENT_RE.fullmatch(agent or ""):
        raise SystemExit(f"agent name must be 1-16 lowercase letters or digits, not {agent!r}")
    have = held_by(agent)
    if have:
        raise SystemExit(f"{agent} already holds salad-{have[0]}; hold at most one lock")
    os.makedirs(locks_dir(), exist_ok=True)
    for s in ([slot] if slot else SLOTS):
        if s not in SLOTS:
            raise SystemExit(f"slot must be 5-9, not {s}")
        try:
            os.mkdir(lock_dir(s))
        except FileExistsError:
            if slot:
                raise SystemExit(f"salad-{s} is taken (stale or not, only the reaper frees it)")
            continue
        t = now if now is not None else now_s()
        with open(os.path.join(lock_dir(s), "owner"), "w", encoding="utf-8") as f:
            f.write(f"{agent} (testlease, {iso(t)})\n")
        write_json(lease_path(s), {"agent": agent, "slot": s, "groups": [], "heartbeat_utc": iso(t)})
        return s
    raise SystemExit("no free lock in salad-5..9")


def require_owner(name):
    """(slot, agent) when name is a t-name whose slot's lease belongs to its agent; SystemExit otherwise."""
    p = parse_name(name)
    if not p:
        raise SystemExit(f"{name!r} is not a test-box name: use t<slot>-<agent>-<card>-<n>, e.g. t5-q4nv30-3090-1")
    slot, agent = p[0], p[1]
    lease = read_lease(slot)
    if not lease:
        raise SystemExit(f"no lease in salad-{slot}: take the lock first (testlease.py take --agent {agent} "
                         f"--slot {slot})")
    if lease.get("agent") != agent:
        raise SystemExit(f"salad-{slot} belongs to {lease.get('agent')!r}, not {agent!r}")
    return slot, agent


def heartbeat(slot, now=None):
    with mutex(slot):
        lease = read_lease(slot)
        if not lease:
            return False
        lease["heartbeat_utc"] = iso(now if now is not None else now_s())
        write_json(lease_path(slot), lease)
        return True


def heartbeat_agent(agent, now=None):
    return [s for s in held_by(agent) if heartbeat(s, now)]


def set_state(slot, name, state, now=None):
    """Adds or updates a group in the lease (and refreshes the heartbeat)."""
    with mutex(slot):
        lease = read_lease(slot)
        if not lease:
            raise SystemExit(f"no lease in salad-{slot}")
        for g in lease["groups"]:
            if g.get("name") == name:
                g["state"] = state
                break
        else:
            lease["groups"].append({"name": name, "state": state})
        lease["heartbeat_utc"] = iso(now if now is not None else now_s())
        write_json(lease_path(slot), lease)


def update_group(slot, name, **fields):
    with mutex(slot):
        lease = read_lease(slot)
        for g in (lease or {}).get("groups", []):
            if g.get("name") == name:
                for k, v in fields.items():
                    if v is None:
                        g.pop(k, None)
                    else:
                        g[k] = v
                write_json(lease_path(slot), lease)
                return True
    return False


def remove_group(slot, name):
    """Takes a group out of the lease. Returns how many are left."""
    with mutex(slot):
        lease = read_lease(slot)
        if not lease:
            return 0
        lease["groups"] = [g for g in lease["groups"] if g.get("name") != name]
        write_json(lease_path(slot), lease)
        return len(lease["groups"])


def free_lock(slot):
    shutil.rmtree(lock_dir(slot), ignore_errors=True)


def release(agent, slot):
    lease = read_lease(slot)
    if not lease or lease.get("agent") != agent:
        raise SystemExit(f"salad-{slot} is not {agent}'s")
    if lease["groups"]:
        raise SystemExit(f"salad-{slot} still lists {[g['name'] for g in lease['groups']]}: delete them first "
                         "(amdtest.py delete NAME verifies the delete)")
    free_lock(slot)


# Approvals for a longer hard cap -------------------------------------------------------------------------------

def approval_path(group):
    return os.path.join(locks_dir(), "approvals", f"{group}.json")


def request_path(group):
    return os.path.join(locks_dir(), "requests", f"{group}.json")


def request_extension(group, hours, reason, now=None):
    """What an agent runs: it writes a request only. The box keeps the normal cap until the coordinator approves."""
    if not parse_name(group):
        raise SystemExit(f"{group!r} is not a test-box name")
    if not reason:
        raise SystemExit("give a --reason")
    t = now if now is not None else now_s()
    write_json(request_path(group), {"group": group, "hours": hours, "reason": reason, "requested_utc": iso(t)})
    return request_path(group)


def approve(group, until, reason="", now=None):
    """The coordinator's command: writes locks/approvals/<group>.json. until: an ISO UTC time or +Nh."""
    t = now if now is not None else now_s()
    m = re.fullmatch(r"\+(\d+(?:\.\d+)?)h", until or "")
    u = t + float(m.group(1)) * 3600 if m else parse_iso(until)
    if u <= t:
        raise SystemExit("--until is in the past")
    req = read_json(request_path(group)) or {}
    obj = {"group": group, "until_utc": iso(u), "reason": reason or req.get("reason") or "",
           "approved_by": "coordinator"}
    write_json(approval_path(group), obj)
    return obj


def approval(group, now=None):
    """The group's approval if it is valid now (written by the coordinator, not expired), else None."""
    a = read_json(approval_path(group))
    if not isinstance(a, dict) or a.get("approved_by") != "coordinator" or a.get("group") != group:
        return None
    try:
        until = parse_iso(a.get("until_utc"))
    except ValueError:
        return None
    return a if until > (now if now is not None else now_s()) else None


def cap_secs(group, now=None):
    """The hard cap a box made (or restarted) now should get: the normal 3 h, or up to the approval's end when
    that is later."""
    t = now if now is not None else now_s()
    a = approval(group, t)
    return max(DEFAULT_CAP_SECS, int(parse_iso(a["until_utc"]) - t)) if a else DEFAULT_CAP_SECS


# Salad -------------------------------------------------------------------------------------------------------

def default_http():
    os.environ.setdefault("SALAD_MINE_DIR", tempfile.mkdtemp())   # never the live watcher's directory
    spec = importlib.util.spec_from_file_location("watcher", os.path.join(HERE, "watcher.py"))
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    return w.http


class Salad:
    """The few calls the lease code needs, for one org. http(method, url, body=None, **kw) -> (status, json)."""
    API = "https://api.salad.com/api/public/organizations"

    def __init__(self, http, org=ORG, project="default"):
        self.http, self.org = http, org
        self.cb = f"{self.API}/{org}/projects/{project}/containers"

    def list_groups(self):
        """(ok, items). ok is False when the listing failed or doesn't look like one."""
        s, j = self.http("GET", self.cb)
        items = (j or {}).get("items") if isinstance(j, dict) else None
        return (s == 200 and isinstance(items, list)), (items or [])

    def get_status(self, name):
        return self.http("GET", f"{self.cb}/{name}")[0]

    def stop(self, name):
        return self.http("POST", f"{self.cb}/{name}/stop")[0]

    def delete(self, name):
        return self.http("DELETE", f"{self.cb}/{name}")[0]

    def has_job_line(self, name, start, end):
        """True when the group logged an [amdtest] or [nvtest] line in [start, end], False when it didn't, None
        when the logs couldn't be read (then nothing is reaped)."""
        q = f'resource.type = "container" AND resource.labels.container_group_name = "{name}"'
        b = {"start_time": iso(start), "end_time": iso(end), "query": q, "page_size": 100, "sort_order": "desc"}
        s, j = self.http("POST", f"{self.API}/{self.org}/log-entries", b, timeout=90)
        if s != 200 or not isinstance(j, dict):
            return None
        return any(t in (it.get("text_log") or "") for it in j.get("items") or [] for t in TAGS)


def retryable(status):
    return status == 0 or status == 429 or status >= 500


def verified_delete(api, name, sleep=time.sleep, clock=time.time, timeout=DELETE_TIMEOUT_SECS):
    """DELETE, then poll until Salad answers 404. 429, 5xx and network errors are retried. (ok, detail)."""
    t_end = clock() + timeout
    while True:
        s = api.delete(name)
        if 200 <= s < 300 or s == 404:
            break
        if not retryable(s) or clock() > t_end:
            return False, f"DELETE {name}: HTTP {s}"
        sleep(DELETE_POLL_SECS)
    while True:
        s = api.get_status(name)
        if s == 404:
            return True, f"{name} is gone (404)"
        if clock() > t_end:
            return False, f"{name} still answers HTTP {s} after {timeout} s"
        sleep(DELETE_POLL_SECS)


def delete_leased(api, name, sleep=time.sleep, clock=time.time):
    """An agent's delete of its own group: marked deleting, deleted, checked gone, then out of the lease. The lock
    stays theirs (release it with `testlease.py release`); the reaper frees it if they go quiet."""
    slot, _ = require_owner(name)
    set_state(slot, name, "deleting")
    ok, detail = verified_delete(api, name, sleep, clock)
    if ok:
        left = remove_group(slot, name)
        detail += f"; salad-{slot} lists {left} group(s) now"
    return ok, detail


# Reaper ------------------------------------------------------------------------------------------------------

def reap(api, live=None, now=None, sleep=time.sleep, clock=time.time, log=None):
    """One pass. Returns the list of records it logged. See the module docstring and README.md, "Test-box
    leases". It acts only on t5-..t9- groups in llmjob2 that their lease lists, and only when the lease's heartbeat
    is stale and the box has logged no job line for JOB_IDLE_SECS: first pass stop, next pass a verified delete.
    It frees a lock only when its lease is stale and lists no groups."""
    live = REAP_LIVE_DEFAULT if live is None else live
    t = now if now is not None else now_s()
    log = log or os.path.join(locks_dir(), "reaper.log")
    recs = []

    def rec(action, **kw):
        r = {"utc": iso(t), "action": action, "dry_run": not live, **kw}
        recs.append(r)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")

    if api.org != ORG:
        rec("abort", reason=f"org {api.org} is not {ORG}")
        return recs
    ok, items = api.list_groups()
    if not ok:
        rec("abort", reason="Salad listing failed; did nothing")
        return recs
    listed = {g.get("name"): g for g in items if g.get("name")}

    for name in sorted(listed):
        p = parse_name(name)
        if not p:
            rec("report", group=name, reason="not a t<slot>- name; never touched")
            continue
        slot, agent = p[0], p[1]
        lease = read_lease(slot)
        entry = next((g for g in (lease or {}).get("groups", []) if g.get("name") == name), None)
        if not lease or lease.get("agent") != agent or not entry:
            rec("report", group=name, reason="no lease lists it; never deleted")
            continue
        try:
            hb_age = t - parse_iso(lease.get("heartbeat_utc"))
        except ValueError:
            hb_age = float("inf")
        if hb_age <= HEARTBEAT_STALE_SECS:
            if entry.get("reaper_stopped_utc") and live:
                update_group(slot, name, reaper_stopped_utc=None)   # the agent is back; start the count again
            rec("keep", group=name, reason=f"heartbeat {int(hb_age)} s old")
            continue
        active = api.has_job_line(name, t - JOB_IDLE_SECS, t)
        if active is None:
            rec("keep", group=name, reason="logs unreadable; not reaped")
            continue
        if active:
            rec("keep", group=name, reason=f"heartbeat stale ({int(hb_age)} s) but the box logged a job line "
                                          f"in the last {JOB_IDLE_SECS} s")
            continue
        status = ((listed[name].get("current_state") or {}).get("status")) or ""
        if not entry.get("reaper_stopped_utc"):
            if not live:
                rec("stop", group=name, agent=agent, heartbeat_age_s=int(hb_age))
                continue
            s = api.stop(name)
            if 200 <= s < 300 or status == "stopped":
                update_group(slot, name, reaper_stopped_utc=iso(t))
                rec("stop", group=name, agent=agent, heartbeat_age_s=int(hb_age), http=s)
            else:
                rec("stop_failed", group=name, http=s)
            continue
        if not live:
            rec("delete", group=name, agent=agent)
            continue
        update_group(slot, name, state="deleting")
        ok, detail = verified_delete(api, name, sleep, clock)
        if ok:
            remove_group(slot, name)
            rec("delete", group=name, agent=agent, detail=detail)
        else:
            rec("delete_failed", group=name, detail=detail)

    # Leases: groups they list that Salad no longer has, and stale empty leases (the only way a stale lock is freed).
    for slot in SLOTS:
        if not os.path.isdir(lock_dir(slot)):
            continue
        lease = read_lease(slot)
        if not lease:
            rec("report", slot=slot, reason="lock without lease.json (old style); left alone")
            continue
        try:
            stale = t - parse_iso(lease.get("heartbeat_utc")) > HEARTBEAT_STALE_SECS
        except ValueError:
            stale = True
        if not stale:
            continue
        left = 0
        for g in lease["groups"]:
            if g.get("name") in listed:
                left += 1
                continue
            s = api.get_status(g.get("name"))
            if s == 404:
                if live:
                    remove_group(slot, g.get("name"))
                rec("forget", slot=slot, group=g.get("name"), reason=f"listed as {g.get('state')}; Salad says 404")
            else:
                left += 1
        if left == 0:
            if live:
                free_lock(slot)
            rec("free_lock", slot=slot, agent=lease.get("agent"))

    for kind, sub in (("approval", "approvals"), ("request", "requests")):
        d = os.path.join(locks_dir(), sub)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not f.endswith(".json"):
                continue
            obj = read_json(os.path.join(d, f)) or {}
            group = f[:-5]
            extra = {"valid": approval(group, t) is not None} if kind == "approval" else {}
            rec(kind, group=group, **{k: v for k, v in obj.items() if k != "group"}, **extra)
    return recs


def summary(recs):
    lines = []
    for r in recs:
        if r["action"] == "keep":
            continue
        what = r.get("group") or f"salad-{r.get('slot')}"
        more = r.get("reason") or r.get("detail") or r.get("until_utc") or ""
        lines.append(f"{'would ' if r['dry_run'] and r['action'] in ('stop', 'delete', 'forget', 'free_lock') else ''}"
                     f"{r['action']} {what}" + (f": {more}" if more else ""))
    kept = sum(r["action"] == "keep" for r in recs)
    mode = "LIVE" if recs and not recs[0]["dry_run"] else "dry run"
    return f"reaper ({mode}): {kept} kept" + "".join(f"\n  {ln}" for ln in lines)


# CLI ---------------------------------------------------------------------------------------------------------

def main(argv=None, http=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("take")
    x.add_argument("--agent", required=True)
    x.add_argument("--slot", type=int)
    x = sub.add_parser("release")
    x.add_argument("--agent", required=True)
    x.add_argument("--slot", type=int, required=True)
    x = sub.add_parser("heartbeat")
    x.add_argument("--agent", required=True)
    sub.add_parser("show")
    x = sub.add_parser("extend")
    x.add_argument("--group", required=True)
    x.add_argument("--hours", type=float, required=True)
    x.add_argument("--reason", required=True)
    x = sub.add_parser("approve")
    x.add_argument("--group", required=True)
    x.add_argument("--until", required=True)
    x.add_argument("--reason", default="")
    x = sub.add_parser("reap")
    g = x.add_mutually_exclusive_group()
    g.add_argument("--live", dest="live", action="store_true", default=None)
    g.add_argument("--dry-run", dest="live", action="store_false")
    a = p.parse_args(argv)
    if a.cmd == "take":
        s = take(a.agent, a.slot)
        print(f"took salad-{s} for {a.agent}; name groups t{s}-{a.agent}-<card>-<n>")
    elif a.cmd == "release":
        release(a.agent, a.slot)
        print(f"freed salad-{a.slot}")
    elif a.cmd == "heartbeat":
        print(f"heartbeat: {['salad-%d' % s for s in heartbeat_agent(a.agent)] or 'no lease held'}")
    elif a.cmd == "show":
        t = now_s()
        for s in SLOTS:
            if not os.path.isdir(lock_dir(s)):
                print(f"salad-{s}: free")
                continue
            lease = read_lease(s)
            if not lease:
                print(f"salad-{s}: taken, no lease.json (old style)")
                continue
            try:
                age = f"{int(t - parse_iso(lease.get('heartbeat_utc')))} s ago"
            except ValueError:
                age = "unknown"
            print(f"salad-{s}: {lease.get('agent')}, heartbeat {age}, groups "
                  f"{[(g.get('name'), g.get('state')) for g in lease['groups']]}")
    elif a.cmd == "extend":
        path = request_extension(a.group, a.hours, a.reason)
        print(f"request written to {path}. It does nothing yet: message the coordinator and ask for approval. "
              f"Without an approval the box keeps its {DEFAULT_CAP_SECS // 3600} h cap.")
    elif a.cmd == "approve":
        obj = approve(a.group, a.until, a.reason)
        print(f"approved {a.group} until {obj['until_utc']}. It takes effect when the box is next created or "
              f"restarted (amdtest.py create or set-env), which reads the approval into BOX_MAX_SECS.")
    elif a.cmd == "reap":
        print(summary(reap(Salad(http or default_http()), a.live)))


if __name__ == "__main__":
    main()
