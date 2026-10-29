#!/usr/bin/env python3
# Runs benchmark.md's GTX 10- and 16-series benchmark on Salad: one high-priority box at a time, card class by
# card class from the top of the tables, two PCs per card. Each box runs bench.sh (PeakMiner, SRBMiner and our
# miner, 5 minutes each after a warm-up) and prints "[bench]" lines, which this reads from Salad's logs. Every 15
# minutes it writes both tables, filled so far, with what the box is doing now, to bench_status.md and a STATUS
# line to bench.log. State is in bench.json, so a restart carries on. Uses the watcher's Salad client; groups are
# named "bench-...", which the watcher's sweep and guard.py leave alone (they only touch "biz-").
# BENCH_SET=amd runs the Radeon cards instead (our HIP core against SRBMiner), in one run or two lanes.
# BENCH_SET=rtx reruns the RTX 20- and 30-series rows on Salad PCs, one lane a series.
# `bench.py table [STATE...]` prints the set's whole table from its lanes' state files.
import json, os, re, sys, time, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("watcher", os.path.join(HERE, "watcher.py"))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

# BENCH_SET picks the cards: "gtx" (the GTX 10- and 16-series), "amd" (the Radeon cards) or "rtx" (the RTX 20- and
# 30-series). Each set has its own state, status and log files, so two runs never share a place in a plan.
SET = os.environ.get("BENCH_SET", "gtx")
# BENCH_LANE (AMD: "a", "b" or "c"; RTX: "20" or "30") splits the set into runs that go at the same time, one Salad
# slot each. Each lane has its own files and group names and takes its own classes. No lane: the whole set in one
# run. The RX 7800 XT was lane a's last class until 22 October; it has a lane of its own so a third slot can take it.
LANE = os.environ.get("BENCH_LANE", "")
LANES = {"amd": {"a": ["RX 9070 XT", "RX 9060 XT"], "b": ["RX 7900 XTX", "RX 7900 XT"], "c": ["RX 7800 XT"]},
         "rtx": {"20": ["RTX 2080 Ti", "RTX 2080", "RTX 2070 Super", "RTX 2070", "RTX 2060 Super", "RTX 2060"],
                 "30": ["RTX 3090 Ti", "RTX 3090", "RTX 3080 Ti", "RTX 3080", "RTX 3070 Ti", "RTX 3070", "RTX 3060 Ti",
                        "RTX 3060"]}}.get(SET, {})
NAME = f"bench{'' if SET == 'gtx' else '-' + SET}{'-' + LANE if LANE else ''}"
STATE, STATUS, LOG = (os.path.join(w.D, NAME + x) for x in (".json", "_status.md", ".log"))
COMMIT = os.environ.get("BENCH_COMMIT", "")   # the commit whose bench.sh the boxes run (pinned, like box.sh)
SRB_EXTRA = os.environ.get("BENCH_SRB_EXTRA", "")   # extra SRBMiner flags, e.g. --gpu-sensors-disable
# AMD set: the commit whose HIP core the boxes build (our miner there; the full SHA, which git fetch needs), and
# how long the hit check runs.
BASE_SHA = os.environ.get("BENCH_BASE_SHA") or COMMIT
VERIFY_SECS = int(os.environ.get("BENCH_VERIFY_SECS") or 600)
# Salad's AMD PCs run containers in WSL2 with the GPU behind /dev/dxg; the image brings its own ROCm (7.1 or later).
AMD_IMAGE = "rocm/dev-ubuntu-24.04:7.2"
MIN = 60
# Lanes share whatever Salad quota the watcher leaves. With room for one more box only, the lane that frees a
# replica took it back within the minute and the other waited: on 28 October the RTX 20 lane waited 55 minutes
# while the 30 lane ran three PCs. So a lane the quota turns away flags it in its own bench-<set>-<lane>.waiting.json
# until its box gets a PC, and a lane about to start a new box leaves the freed replica to the lane that has waited
# longest. A flag counts for TURN_MIN minutes after its lane's last refused try, so a lane that stopped holds no one
# back.
TURN_MIN = 3
HOSTS = 2               # PCs per card, as in benchmark.md
MAX_TRIES = 10          # boxes per class before moving on (the 1070/1080/1080 Ti class may not show all three)
NO_PC_MIN = 30          # minutes to wait for a PC before moving on
RUN_MIN = 45            # minutes from a PC to "all done" before moving the box on
STATUS_MIN = 15

# Salad's class, the rows it can fill (benchmark.md's order) and how long our miner runs (90 s where the release
# has no build for the card, 300 s where it does).
PLAN = [
    ("GTX 1070, 1080, 1080Ti (8 GB)", ["GTX 1080 Ti", "GTX 1080", "GTX 1070"], 90),
    ("GTX 1060 (6 GB)", ["GTX 1060"], 90),
    ("GTX 1050 Ti (4 GB)", ["GTX 1050 Ti"], 90),
    ("GTX 1660 Super (6 GB)", ["GTX 1660 Super"], 300),
    ("GTX 1660 (6 GB)", ["GTX 1660"], 300),
    ("GTX 1650 (4 GB)", ["GTX 1650"], 300),
]
# Cards a class also hands out that aren't rows of their own: recorded in their own row when they turn up, but not
# waited for. Salad's "GTX 1660" class gave a GTX 1660 Ti on 17 October.
EXTRA = {"GTX 1660 (6 GB)": ["GTX 1660 Ti"]}
SECTIONS = [("GTX 10-series (Pascal, sm_61, on Salad)", ["GTX 1080 Ti", "GTX 1080", "GTX 1070", "GTX 1060", "GTX 1050 Ti"]),
            ("GTX 16-series (Turing without tensor cores, sm_75, on Salad)",
             ["GTX 1660 Super", "GTX 1660 Ti", "GTX 1660", "GTX 1650"])]
MEM = {"GTX 1060": "6 GB", "GTX 1050 Ti": "4 GB", "GTX 1660 Super": "6 GB", "GTX 1660 Ti": "6 GB", "GTX 1660": "6 GB",
       "GTX 1650": "4 GB"}
if SET == "amd":
    # Our miner and SRBMiner, 5 minutes each. No release has an AMD build, so ours is the HIP core at BASE_SHA,
    # built on the box and run through the CLI from source after a hit check (bench.sh). PeakMiner has no Pearl
    # kernel for AMD and isn't run.
    PLAN = [("AMD RX 9070 XT (16GB)", ["RX 9070 XT"], 300), ("AMD RX 9060 XT (16GB)", ["RX 9060 XT"], 300),
            ("AMD RX 7900 XTX (24GB)", ["RX 7900 XTX"], 300), ("AMD RX 7900 XT (20GB)", ["RX 7900 XT"], 300),
            ("AMD RX 7800 XT (16GB)", ["RX 7800 XT"], 300)]
    EXTRA = {}
    SECTIONS = [("AMD Radeon (RDNA 3 and 4, on Salad)", ["RX 9070 XT", "RX 9060 XT", "RX 7900 XTX", "RX 7900 XT",
                                                         "RX 7800 XT"])]
    MEM = {"RX 9070 XT": "16 GB", "RX 9060 XT": "16 GB", "RX 7900 XTX": "24 GB", "RX 7900 XT": "20 GB",
           "RX 7800 XT": "16 GB"}
    # On a PC that has the image: setup 1-4 min (apt is the slow part), build under a minute, the hit check,
    # SRBMiner's device list up to 1 min, then the warm-up and two 5-minute runs, about 12 min with the pauses
    # between them. 16-19 min plus the check in all; 40 plus the check leaves 20 min for the image to download.
    # 50 with the default 10-minute check.
    RUN_MIN = 40 + -(-VERIFY_SECS // 60)
if SET == "rtx":
    # benchmark.md's RTX 20- and 30-series rows, rerun on Salad PCs with all three miners at their newest (bench.sh).
    # Salad has no 2060 Super or 2070 Super class, so those rows fill only if another class hands one out, and only
    # a 12 GB RTX 3060 class is run, as benchmark.md's 3060 rows are 12 GB cards. It has no 3050 row.
    PLAN = [("RTX 2080 Ti (11 GB)", ["RTX 2080 Ti"], 300), ("RTX 2080 (8 GB)", ["RTX 2080"], 300),
            ("RTX 2070 (8 GB)", ["RTX 2070"], 300), ("RTX 2060 (6 GB)", ["RTX 2060"], 300),
            ("RTX 3090 Ti (24 GB)", ["RTX 3090 Ti"], 300), ("RTX 3090 (24 GB)", ["RTX 3090"], 300),
            ("RTX 3080 Ti (12 GB)", ["RTX 3080 Ti"], 300), ("RTX 3080 (10 GB)", ["RTX 3080"], 300),
            ("RTX 3070 Ti (8 GB)", ["RTX 3070 Ti"], 300), ("RTX 3070 (8 GB)", ["RTX 3070"], 300),
            ("RTX 3060 Ti (8 GB)", ["RTX 3060 Ti"], 300), ("RTX 3060 (12 GB)", ["RTX 3060"], 300)]
    EXTRA = {"RTX 2070 (8 GB)": ["RTX 2070 Super"], "RTX 2060 (6 GB)": ["RTX 2060 Super"]}
    SECTIONS = [("RTX 20-series (on Salad)", LANES["20"]), ("RTX 30-series (on Salad)", LANES["30"])]
    MEM = {"RTX 2060": "6 GB", "RTX 3060": "12 GB"}
ALL_SECTIONS = SECTIONS          # every row of the set, for `table`
if LANE in LANES:
    PLAN = [p for p in PLAN if p[1][0] in LANES[LANE]]
    # A section none of the lane's rows are in (RTX: the other series) isn't written at all.
    SECTIONS = [(t, [r for r in rows if r in LANES[LANE]]) for t, rows in SECTIONS
                if any(r in LANES[LANE] for r in rows)]
# A probe rather than a table: BENCH_CLASSES (Salad class names, ";" between them, as some names hold commas) runs
# only those classes of the set, and BENCH_HOSTS how many PCs per card (benchmark.md's 2 by default).
ONLY = [c.strip() for c in os.environ.get("BENCH_CLASSES", "").split(";") if c.strip()]
if ONLY:
    PLAN = [p for p in PLAN if p[0] in ONLY]
HOSTS = int(os.environ.get("BENCH_HOSTS") or HOSTS)


def log(msg):
    line = f"{time.strftime('%H:%M', time.gmtime(w.now()))}Z {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def row_for(card, rows):
    """The table row a card name is, the whole name without "NVIDIA GeForce" or "AMD Radeon", so neither
    "GTX 1080 Ti" nor "GTX 1660 Ti" is read as the card without the Ti, nor "RX 7900 XTX" as "RX 7900 XT"."""
    c = re.sub(r"^(NVIDIA\s+)?(GEFORCE\s+)?(AMD\s+)?(RADEON\s+)?", "", card.strip().upper())
    return next((r for r in rows if r.upper() == c), None)


def hit_check(body):
    """bench.sh's "hitcheck PASS|FAIL secs=600 hits=2202 verified=400 failed=0 salts=277" line, which can end in
    "why=..." (the first failed hit's reason) or "error=..." (no result from verify-hits.js)."""
    m = re.match(r"hitcheck (PASS|FAIL)\b(.*)", body)
    if not m:
        return None
    rest, out = m.group(2), {"pass": m.group(1) == "PASS"}
    free = re.search(r"\b(why|error)=(.*)$", rest)
    if free:
        out[free.group(1)] = free.group(2).strip()
        rest = rest[:free.start()]
    for k, v in re.findall(r"(\w+)=(\S+)", rest):
        out[k] = int(v) if v.isdigit() else None
    return out


def parse(lines):
    """A box's "[bench]" lines into what it did: the card, each miner's samples, end, exit, and whether it's done.
    On AMD also the PC's copy rate, the hit check, the core's commit and any miner the box left out, and why."""
    out = {"card": None, "miners": {}, "done": False, "fail": None, "phase": None, "copy": None, "hitcheck": None,
           "core": None, "skipped": {}}
    for text in lines:
        m = re.search(r"\[bench\] \S+ (.*)", text)
        if not m:
            continue
        body = m.group(1).strip()
        if body.startswith("copy "):
            c = re.match(r"copy (\d+(?:\.\d+)?) GB/s", body)
            out["copy"] = float(c.group(1)) if c else None
        elif body.startswith("hitcheck "):
            out["hitcheck"] = hit_check(body)
        elif body.startswith("skip "):
            s = re.match(r"skip (\w+): (.*)", body)
            if s:
                out["skipped"][s.group(1)] = s.group(2)
        elif body.startswith("fetch ok "):
            out["core"] = body.split()[2]
        elif body.startswith("card "):
            f = [x.strip() for x in body[5:].split(",")]
            out["card"] = {"name": f[0], "driver": f[1] if len(f) > 1 else "", "power": f[2] if len(f) > 2 else "",
                           "mem": f[4] if len(f) > 4 else ""}
        elif body.startswith("FAIL "):
            out["fail"] = body[5:]
        elif body == "all done":
            out["done"] = True
        elif body.startswith("start "):
            out["phase"] = body
        elif body.startswith("sample "):
            s = re.match(r"sample (\w+) t=(\d+) hs=(\S+) acc=(\S+) rej=(\S+)", body)
            if s:
                name, t, hs = s.group(1), int(s.group(2)), s.group(3)
                mm = out["miners"].setdefault(name, {"samples": []})
                try:
                    v = float(hs)
                    th = v / 1e12 if v > 1e9 else v
                except ValueError:
                    th = None
                mm["samples"].append({"t": t, "th": th, "acc": s.group(4), "rej": s.group(5)})
                out["phase"] = f"{name} {t}s"
        elif body.startswith("exited "):
            e = re.match(r"exited (\w+) t=(\d+): (.*)", body)
            if e:
                out["miners"].setdefault(e.group(1), {"samples": []})["exited"] = e.group(3)[:200]
    return out


def figure(m, mean_after=None):
    """A miner's TH/s: its own figure at the end (PeakMiner, SRBMiner), or the mean after the first minute (ours)."""
    if not m:
        return None, "not run"
    # Only after the first minute, and only a miner that had a rate for at least half of that: one on a PC that
    # loses the pool shows a rate for its first seconds and 0 after (SRBMiner on a GTX 1660, 17 October).
    late = [x for x in m.get("samples", []) if x["t"] >= 60]
    good = [x["th"] for x in late if x["th"] is not None and x["th"] > 0]
    if good and len(good) * 2 >= len(late):
        return (sum(good) / len(good) if mean_after is not None else good[-1]), None
    return None, ("didn't start" if m.get("exited") and not m.get("samples") else "no rate")


def quota_refused(s, j):
    """Salad's answer when the organization's container replicas are all in use ("created_replicas_quota_exceeded"
    on a create, "replicas_quota_exceeded" on a start)."""
    return s == 400 and "quota_exceeded" in str((j or {}).get("type", "") if isinstance(j, dict) else j)


def cell(v, why):
    return f"{v:.1f}" if v is not None else why


def pc_note(row, h):
    """The line under the AMD table for one PC: its device copy rate, the hit check, and which miner went first.
    Hits a salt is hits / salts; the hit check's target makes it 8."""
    parts = [f"copy {h['copy']:.0f} GB/s" if h.get("copy") else "copy not measured"]
    hc = h.get("hitcheck")
    if hc:
        s = f"hit check {'PASS' if hc['pass'] else 'FAIL'}"
        if hc.get("hits") is not None:
            s += (f", {hc.get('verified')} of {hc['hits']} hits verified, {hc.get('failed')} failed, "
                  f"{hc.get('salts')} salts in {hc.get('secs')} s")
            if hc.get("salts"):
                s += f", {hc['hits'] / hc['salts']:.2f} hits a salt"
        if hc.get("why") or hc.get("error"):
            s += f" ({hc.get('why') or hc.get('error')})"
        parts.append(s)
    else:
        parts.append("no hit check")
    if h.get("order"):
        parts.append(("ours" if h["order"].startswith("ours") else "SRBMiner") + " ran first")
    return f"- `{h['machine'][:8]}` ({row}): " + "; ".join(parts) + "."


def render(st, sections=None):
    """Both tables, in benchmark.md's shape, with what's filled so far. On AMD PeakMiner isn't run, so "% of best"
    is ours against SRBMiner, and a line for each PC under the table gives its copy rate and hit check."""
    out = []
    for title, rows in sections or SECTIONS:
        out += [f"## {title}", "", "| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |",
                "|---|---|---|---|---|---|---|"]
        notes = [pc_note(r, h) for r in rows for h in st["results"].get(r, []) if SET == "amd"]
        for r in rows:
            hosts = st["results"].get(r, [])
            extra = any(r in e for e in EXTRA.values())
            for i in range(len(hosts) if extra else HOSTS):    # a card no class is for: only the hosts it had
                mem = f", {MEM[r]}" if r in MEM else ""
                if i >= len(hosts):
                    out.append(f"| {r} | Salad (— W{mem}) | — | not run | not run | not run | — |")
                    continue
                h = hosts[i]
                # nvidia-smi says "[N/A]" on a PC whose driver doesn't report a power limit.
                p = re.sub(r"\.0+ W$", " W", h["power"]) if re.match(r"\d", h.get("power") or "") else "— W"
                o, ow = h["ours"]
                pk, pw = h["peak"]
                sr, sw = h["srb"]
                best = max([x for x in (pk, sr) if x] or [0])
                pct = f"{o / best * 100:.0f}%" if o and best else "—"
                out.append(f"| {r} | Salad ({p}{mem}) | {h['machine'][:8]} | {cell(o, ow)} | {cell(pk, pw)} | "
                           f"{cell(sr, sw)} | {pct} |")
        out.append("")
        if notes:
            out += notes + [""]
    return "\n".join(out)


def table(paths):
    """The set's whole table from several state files (two lanes'), their results put together."""
    st = {"results": {}}
    for p in paths:
        for row, hosts in ((w.load(p, None) or {}).get("results") or {}).items():
            st["results"].setdefault(row, []).extend(hosts)
    return render(st, ALL_SECTIONS)


class Bench:
    def __init__(self):
        self.st = w.load(STATE, None) or {"cls": 0, "tries": 0, "group": None, "results": {}, "seen": [],
                                         "notes": [], "status_t": 0}
        self.classes = {}

    def save(self):
        w.save(STATE, self.st)

    def class_id(self, name):
        if not self.classes:
            s, j = w.http("GET", f"{w.BASE}/gpu-classes")
            self.classes = {c["name"]: c["id"] for c in (j or {}).get("items", [])} if s == 200 else {}
        return self.classes.get(name)

    def plan(self):
        return PLAN[self.st["cls"]] if self.st["cls"] < len(PLAN) else None

    def full(self, rows):
        return all(len(self.st["results"].get(r, [])) >= HOSTS for r in rows)

    def order(self):
        """Which miner goes first, and so gets the warm-up. GTX: by box. AMD and RTX: by how many PCs of the class
        are recorded, so its two PCs take turns (each gets a group of its own, see move_on). RTX alternates ours
        first and SRBMiner first, as benchmark.md's "How a host is tested" does."""
        n = sum(len(self.st["results"].get(r, [])) for r in self.plan()[1])
        if SET == "amd":
            return "ours,srb" if n % 2 == 0 else "srb,ours"
        if SET == "rtx":
            return "ours,peak,srb" if n % 2 == 0 else "srb,peak,ours"
        return "peak,srb,ours" if self.st["tries"] % 2 == 0 else "srb,peak,ours"

    def command(self, ours_secs, order):
        raw = f"https://raw.githubusercontent.com/super3/llmjob/{COMMIT}/tuning/salad"
        url = f"{raw}/bench.sh"
        extra = f" SRB_EXTRA='{SRB_EXTRA}'" if SRB_EXTRA else ""
        if SET == "amd":
            # bench.sh builds and runs our core with amdtest.sh's functions, from the same commit, beside it.
            return (f"apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl ca-certificates >/dev/null 2>&1 && "
                    f"curl -fsSL {url} -o /root/bench.sh && curl -fsSL {raw}/amdtest.sh -o /root/amdtest.sh && "
                    f"W={self.wallet} ORDER={order} OURS_SECS={ours_secs} BASE_SHA={BASE_SHA} "
                    f"VERIFY_SECS={VERIFY_SECS}{extra} bash /root/bench.sh 2>&1 | tee -a /root/bench.log")
        return (f"apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl ca-certificates >/dev/null 2>&1 && "
                f"curl -fsSL {url} -o /root/bench.sh && W={self.wallet} ORDER={order} OURS_SECS={ours_secs}{extra} "
                f"bash /root/bench.sh 2>&1 | tee -a /root/bench.log")

    def create(self):
        cls, rows, ours = self.plan()
        cid = self.class_id(cls)
        if not cid:
            self.note(f"{cls}: no such class on Salad; skipped")
            self.next_class()
            return
        slug = re.sub(r"[^a-z0-9]", "", cls.split("(")[0].lower().replace("amd", "")) if SET == "amd" else \
            w.slug(cls) if SET == "rtx" else w.slug(cls).replace("rtx", "gtx")
        bump = self.st.get("bump", 0)       # names Salad holds but can't show (below), passed over
        name = f"bench-{LANE + '-' if LANE else ''}{slug}-{self.st['tries'] + 1}{f'r{bump}' if bump else ''}"[:60]
        order = self.order()
        # The HIP build on an AMD box needs amdtest's 4 vCPUs and 8 GB.
        cpu, mem = (4, 8192) if SET == "amd" else (2, 4096)
        body = {"name": name, "display_name": name, "replicas": 1, "restart_policy": "never",
                "autostart_policy": True,
                "container": {"image": AMD_IMAGE if SET == "amd" else w.IMAGE, "priority": "high",
                              "resources": {"cpu": cpu, "memory": mem, "gpu_classes": [cid],
                                            "storage_amount": 10737418240},
                              "command": ["/bin/sh", "-c", self.command(ours, order)]}}
        s, j = w.http("POST", w.CB, body)
        t = w.now()
        if s == 400 and "name_conflict" in str(j):
            # A create that errored can still have made the group: on 28 October Salad's gateway answered one with
            # a 504 and made it. One we can see is ours, so it's watched rather than left billing. One Salad holds
            # under the name but can't show or delete (404) is passed over with the next name; that one also held a
            # replica of the quota, which only Salad can give back.
            g, _ = w.http("GET", f"{w.CB}/{name}")
            if 200 <= g < 300:
                self.st.update(group=name, created=t, pc_t=None, machine=None, instance=None, log_t=t - 120,
                               lines=[], order=order)
                log(f"ADOPTED {name} ({cls}): a create that errored had made it")
                return
            self.st["bump"] = bump + 1
            self.note(f"{name}: Salad has a group by that name that it can't show or delete (HTTP {g}); "
                      "the next box takes another name")
            return
        if not 200 <= s < 300:
            log(f"FAIL create {name}: HTTP {s} {str(j)[:200]}")
            self.waiting(quota_refused(s, j))
            return
        # A lane's waiting flag stays until its box gets a PC: at 21:26 on 28 October the RTX 20 lane's create went
        # through, Salad then stopped the box for want of a replica, and the RTX 30 lane, seeing no flag, took it.
        self.st.update(group=name, created=t, pc_t=None, machine=None, instance=None, log_t=t - 120, lines=[],
                       order=order)
        log(f"CREATED {name} ({cls}, high)")

    def wait_file(self, lane):
        return os.path.join(w.D, f"bench-{SET}-{lane}.waiting.json")

    def waiting(self, on):
        """Flag (or clear) that Salad's quota turned this lane away, for the set's other lanes (TURN_MIN)."""
        self.st["waiting"] = bool(on)
        if not LANE:
            return
        p = self.wait_file(LANE)
        if on:
            w.save(p, {"since": (w.load(p, None) or {}).get("since") or w.now(), "last": w.now()})
        elif os.path.exists(p):
            os.remove(p)

    def ahead(self):
        """The lane to leave a free replica to, or None: of the lanes the quota turned away in the last TURN_MIN
        minutes, the one that has waited longest, unless that's this lane."""
        t = w.now()
        fresh = {}
        for lane in LANES:
            v = w.load(self.wait_file(lane), None)
            if v and t - v.get("last", 0) <= TURN_MIN * MIN:
                fresh[lane] = v
        first = min(fresh, key=lambda lane: (fresh[lane].get("since", 0), lane), default=None)
        return None if first == LANE else first

    def note(self, msg):
        self.st["notes"].append(f"{time.strftime('%d %b %H:%M', time.gmtime(w.now()))}Z {msg}")
        log(f"NOTE {msg}")

    def delete(self):
        g = self.st["group"]
        if g:
            s, _ = w.http("DELETE", f"{w.CB}/{g}")
            log(f"DELETED {g}: HTTP {s}")
            if s != 404:              # looked at again until Salad says it's gone (check_gone)
                t = w.now()
                self.st.setdefault("gone", {})[g] = {"t": t, "try": t, "said": False}
        self.st["group"] = None

    def check_gone(self):
        """Salad sometimes answers a delete with 202 and keeps the group (watcher.py's sweep), and a delete can
        fail outright. Such a box sleeps on its PC at high, billing and holding a slot, and the watcher's sweep
        only looks at "biz-" names. So 5 minutes after each delete the group is looked at again, and deleted
        again until Salad answers 404. A note says so once it's still there 10 minutes after the first delete."""
        t = w.now()
        gone = self.st.setdefault("gone", {})
        for g, d in list(gone.items()):
            if t - d["try"] < 5 * MIN:
                continue
            s, _ = w.http("GET", f"{w.CB}/{g}")
            if s == 404:
                del gone[g]
                log(f"GONE {g}")
                continue
            if s != 200:
                continue                  # no answer: the next tick asks again
            s, _ = w.http("DELETE", f"{w.CB}/{g}")
            d["try"] = t
            log(f"DELETED {g} again: HTTP {s}")
            if not d["said"] and t - d["t"] >= 10 * MIN:
                d["said"] = True
                self.note(f"{g} still on Salad {(t - d['t']) / MIN:.0f} min after its delete; deleting it again "
                          "every 5 min")

    def next_class(self):
        self.delete()
        self.st["cls"] += 1
        self.st["tries"] = 0

    def move_on(self, why):
        """Done with this PC: another PC in the same class, or the next class."""
        cls, rows, _ = self.plan()
        self.st["tries"] += 1
        if self.full(rows) or self.st["tries"] >= MAX_TRIES:
            if not self.full(rows):
                self.note(f"{cls}: stopped after {self.st['tries']} boxes with rows still empty")
            self.next_class()
            return
        if SET in ("amd", "rtx") and why == "done":
            # A moved instance keeps its group's command, so the next PC gets a new group: the other order.
            log(f"NEW GROUP for the next PC of {cls}, so the other miner goes first")
            self.delete()
            return
        inst = self.st.get("instance")
        s = 0
        if inst:
            s, _ = w.http("POST", f"{w.CB}/{self.st['group']}/instances/{inst}/reallocate")
        if 200 <= s < 300 or s == 404:
            log(f"MOVED {self.st['group']} to a new PC ({why})")
            t = w.now()
            self.st.update(pc_t=None, machine=None, instance=None, log_t=t - 30, lines=[], created=t)
        else:
            log(f"reallocate refused (HTTP {s}); new box instead ({why})")
            self.delete()

    def read(self):
        t = w.now()
        body = {"start_time": w.iso(self.st["log_t"]), "end_time": w.iso(t), "page_size": 100, "sort_order": "asc",
                "query": f'resource.type = "container" AND resource.labels.container_group_name = "{self.st["group"]}"'}
        s, j = w.http("POST", f"{w.BASE}/log-entries", body, timeout=90)
        if s != 200:
            return
        items = sorted((j or {}).get("items") or [], key=lambda it: it.get("time") or "")
        seen = set(self.st["lines"])
        for it in items:
            mach = ((it.get("resource") or {}).get("labels") or {}).get("machine_id")
            text = it.get("text_log") or ""
            if "[bench]" in text and (not self.st.get("machine") or mach == self.st["machine"]):
                key = f"{it.get('time')}|{text.strip()}"
                if key not in seen:
                    self.st["lines"].append(key)
                    seen.add(key)
        if items:
            self.st["log_t"] = max(self.st["log_t"], w.parse_iso(items[-1]["time"]) - 30)

    def stopped(self):
        s, j = w.http("GET", f"{w.CB}/{self.st['group']}")
        return s == 200 and ((j or {}).get("current_state") or {}).get("status") == "stopped"

    def tick(self):
        if not self.plan():
            return False
        cls, rows, ours = self.plan()
        if self.full(rows):
            self.next_class()
            return True
        if not self.st["group"]:
            other = self.ahead()
            if other:
                # The lane that has waited longest goes first; the others leave the freed replica to it.
                if self.st.get("yielded") != other:
                    log(f"YIELD to lane {other}: the quota turned it away, so it gets the next free replica")
                self.st["yielded"] = other
                return True
            self.st["yielded"] = None
            self.create()
            return True
        t = w.now()
        s, j = w.http("GET", f"{w.CB}/{self.st['group']}/instances")
        insts = (j or {}).get("instances") or (j or {}).get("items") or [] if s == 200 else []
        on = [i for i in insts if i.get("machine_id") and i.get("state") in ("downloading", "creating", "running")]
        if on:
            if self.st.get("waiting"):
                self.waiting(False)
            m = on[0]["machine_id"]
            if m != self.st.get("machine"):
                # A new PC, by our move or by Salad's own, which shows only as a new machine_id (README, "Salad API
                # notes"): its row is built from its own lines only. Reading back 2 minutes catches its first
                # lines; read() drops the old PC's.
                old = self.st.get("machine")
                self.st.update(machine=m, instance=on[0].get("instance_id") or on[0].get("id"), pc_t=t, lines=[],
                               log_t=t - 120)
                log(f"PC {self.st['group']}: {m[:8]}" + (f" (was {old[:8]})" if old else ""))
                if m in self.st["seen"]:
                    self.move_on(f"PC {m[:8]} was benchmarked already")
                    return True
        elif not self.st.get("pc_t") and t - self.st.get("start_t", 0) >= MIN and self.stopped():
            # Salad sometimes stops a new group before it gets a PC, as the watcher sees too. Start it again, every
            # minute: until 28 October every 5, and a start the quota refused cost 5 minutes each time.
            s, j = w.http("POST", f"{w.CB}/{self.st['group']}/start")
            self.st["start_t"] = t
            log(f"STARTED {self.st['group']} again: it was stopped with no PC (HTTP {s})")
            if quota_refused(s, j):
                # Held back by the quota, not by the class having no PC: that wait doesn't count to NO_PC_MIN.
                self.st["created"] = t
                self.waiting(True)
        elif not self.st.get("pc_t") and t - self.st["created"] >= NO_PC_MIN * MIN:
            self.note(f"{cls}: no PC at high in {NO_PC_MIN} min")
            self.next_class()
            return True
        if not self.st.get("machine"):
            return True
        self.read()
        r = parse(x.split("|", 1)[1] for x in self.st["lines"])
        if r["card"]:
            row = row_for(r["card"]["name"], rows + EXTRA.get(cls, []))
            if row is None or len(self.st["results"].get(row, [])) >= HOSTS:
                self.st["seen"].append(self.st["machine"])
                self.move_on(f"{r['card']['name']}: " + ("not a card in this class's rows" if row is None
                                                        else f"{row} has its {HOSTS} hosts"))
                return True
        if r["fail"]:
            self.note(f"{self.st['machine'][:8]}: {r['fail']}")
            self.st["seen"].append(self.st["machine"])
            self.move_on(r["fail"])
            return True
        if r["done"] and r["card"]:
            row = row_for(r["card"]["name"], rows + EXTRA.get(cls, []))
            res = {"machine": self.st["machine"], "card": r["card"]["name"], "power": r["card"]["power"],
                   "driver": r["card"]["driver"], "group": self.st["group"],
                   "ours": figure(r["miners"].get("ours"), mean_after=60),
                   "peak": figure(r["miners"].get("peak")), "srb": figure(r["miners"].get("srb")),
                   "exits": {k: v.get("exited") for k, v in r["miners"].items() if v.get("exited")}}
            if res["ours"][0] is None and res["ours"][1] in ("didn't start", "no rate", "not run") and ours == 90:
                res["ours"] = (None, "no build")
            if SET == "amd":
                res.update(copy=r["copy"], hitcheck=r["hitcheck"], core=r["core"], order=self.st.get("order"))
                if r["skipped"].get("ours"):     # the core didn't build, or the hit check failed
                    res["ours"] = (None, r["skipped"]["ours"])
                    # bench.sh left ours out of the order too, so SRBMiner went first.
                    res["order"] = ",".join(x for x in (res["order"] or "").split(",") if x and x != "ours") or None
            if ours != 90 and all(res[k][0] is None for k in ("ours", "peak", "srb")):
                # Where our release has a build, a PC on which no miner gets work points at its network, not the
                # card: try another PC instead of filling the row with it.
                self.note(f"{self.st['machine'][:8]} ({r['card']['name']}): no miner got work from the pool; "
                          "not recorded")
                self.st["seen"].append(self.st["machine"])
                self.move_on("no miner got work")
                return True
            self.st["results"].setdefault(row, []).append(res)
            self.st["seen"].append(self.st["machine"])
            log(f"RESULT {row} on {self.st['machine'][:8]}: ours {cell(*res['ours'])}, PeakMiner {cell(*res['peak'])}, "
                f"SRBMiner {cell(*res['srb'])}")
            self.status(force=True)
            self.move_on("done")
            return True
        if self.st.get("pc_t") and t - self.st["pc_t"] >= RUN_MIN * MIN:
            self.note(f"{self.st['machine'][:8]}: not done {RUN_MIN} min after getting the PC")
            self.st["seen"].append(self.st["machine"])
            self.move_on("too slow")
        self.st["phase"] = r["phase"] or ("setting up" if not r["card"] else "downloading miners")
        self.st["card"] = r["card"]["name"] if r["card"] else None
        return True

    def now_line(self):
        p = self.plan()
        if not p:
            left = sorted(self.st.get("gone") or {})
            return "All classes done." + (f" Waiting for Salad to remove {', '.join(left)}." if left else "")
        g = self.st.get("group")
        if not g:
            if self.st.get("yielded"):
                return f"Next: {p[0]}, once lane {self.st['yielded']} has its box (it was waiting for a replica)."
            return f"Next: {p[0]}." + (" Salad's quota is full; trying every minute." if self.st.get("waiting") else "")
        if not self.st.get("machine"):
            return f"Now: {g} ({p[0]}, high) waiting for a PC since {time.strftime('%H:%M', time.gmtime(self.st['created']))}Z."
        return (f"Now: {g} on PC {self.st['machine'][:8]}" + (f" ({self.st.get('card')})" if self.st.get("card") else "")
                + f", {self.st.get('phase') or 'setting up'}.")

    def status(self, force=False):
        t = w.now()
        if not force and t - self.st.get("status_t", 0) < STATUS_MIN * MIN:
            return
        self.st["status_t"] = t
        text = render(self.st) + "\n" + self.now_line() + "\n"
        if self.st["notes"]:
            text += "\nNotes:\n" + "\n".join(f"- {n}" for n in self.st["notes"][-8:]) + "\n"
        with open(STATUS, "w") as f:
            f.write(text)
        log("STATUS")

    def run(self):
        self.wallet = os.environ.get("PRL_WALLET") or w.load(w.CFG, {}).get("wallet", "")
        if not self.wallet.startswith("prl1") or not COMMIT:
            sys.exit("needs PRL_WALLET (or the watcher's config wallet) and BENCH_COMMIT")
        if LANE and LANE not in LANES:
            sys.exit("BENCH_LANE is a, b or c with BENCH_SET=amd, or 20 or 30 with BENCH_SET=rtx")
        if SET == "amd" and not re.fullmatch(r"[0-9a-f]{40}", BASE_SHA):
            # A short one is fine in the raw URL, but git fetch on the box finds no such ref, and every box fails.
            sys.exit("BENCH_BASE_SHA (or BENCH_COMMIT) must be a full 40-character commit: the box git-fetches it")
        while True:
            try:
                going = self.tick()
                self.check_gone()
                self.status()
                self.save()
                if not going and not self.st.get("gone"):    # done once Salad has removed every group too
                    self.status(force=True)
                    self.save()
                    log("FINISHED")
                    return
            except Exception as e:  # keep going; the next tick tries again
                log(f"ERROR {type(e).__name__}: {e}")
            time.sleep(MIN)


if __name__ == "__main__":
    if sys.argv[1:2] == ["table"]:
        # Default: the set's lane files where it has lanes (bench-amd-a.json, bench-rtx-20.json, ...), else its own.
        print(table(sys.argv[2:] or ([os.path.join(w.D, f"bench-{SET}-{x}.json") for x in LANES] if LANES
                                     else [STATE])))
    else:
        Bench().run()
