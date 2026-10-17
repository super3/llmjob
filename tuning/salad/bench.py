#!/usr/bin/env python3
# Runs benchmark.md's GTX 10- and 16-series benchmark on Salad: one high-priority box at a time, card class by
# card class from the top of the tables, two PCs per card. Each box runs bench.sh (PeakMiner, SRBMiner and our
# miner, 5 minutes each after a warm-up) and prints "[bench]" lines, which this reads from Salad's logs. Every 15
# minutes it writes both tables, filled so far, with what the box is doing now, to bench_status.md and a STATUS
# line to bench.log. State is in bench.json, so a restart carries on. Uses the watcher's Salad client; groups are
# named "bench-...", which the watcher's sweep and guard.py leave alone (they only touch "biz-").
import json, os, re, sys, time, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("watcher", os.path.join(HERE, "watcher.py"))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

STATE, STATUS, LOG = (os.path.join(w.D, f) for f in ("bench.json", "bench_status.md", "bench.log"))
COMMIT = os.environ.get("BENCH_COMMIT", "")   # the commit whose bench.sh the boxes run (pinned, like box.sh)
MIN = 60
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
SECTIONS = [("GTX 10-series (Pascal, sm_61, on Salad)", ["GTX 1080 Ti", "GTX 1080", "GTX 1070", "GTX 1060", "GTX 1050 Ti"]),
            ("GTX 16-series (Turing without tensor cores, sm_75, on Salad)", ["GTX 1660 Super", "GTX 1660", "GTX 1650"])]
MEM = {"GTX 1060": "6 GB", "GTX 1050 Ti": "4 GB", "GTX 1660 Super": "6 GB", "GTX 1660": "6 GB", "GTX 1650": "4 GB"}


def log(msg):
    line = f"{time.strftime('%H:%M', time.gmtime(w.now()))}Z {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def row_for(card, rows):
    """The table row a card name belongs to, longest name first so "GTX 1080 Ti" isn't read as "GTX 1080"."""
    c = card.upper()
    for r in sorted(rows, key=len, reverse=True):
        if re.search(r"\b" + re.escape(r.upper()) + r"\b", c):
            return r
    return None


def parse(lines):
    """A box's "[bench]" lines into what it did: the card, each miner's samples, end, exit, and whether it's done."""
    out = {"card": None, "miners": {}, "done": False, "fail": None, "phase": None}
    for text in lines:
        m = re.search(r"\[bench\] \S+ (.*)", text)
        if not m:
            continue
        body = m.group(1).strip()
        if body.startswith("card "):
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
    good = [x["th"] for x in m.get("samples", []) if x["th"] is not None and x["th"] > 0]
    if mean_after is not None:
        good = [x["th"] for x in m.get("samples", []) if x["th"] and x["t"] >= mean_after]
        if good:
            return sum(good) / len(good), None
    elif good:
        return good[-1], None
    return None, ("didn't start" if m.get("exited") and not m.get("samples") else "no rate")


def cell(v, why):
    return f"{v:.1f}" if v is not None else why


def render(st):
    """Both tables, in benchmark.md's shape, with what's filled so far."""
    out = []
    for title, rows in SECTIONS:
        out += [f"## {title}", "", "| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |",
                "|---|---|---|---|---|---|---|"]
        for r in rows:
            hosts = st["results"].get(r, [])
            for i in range(HOSTS):
                mem = f", {MEM[r]}" if r in MEM else ""
                if i >= len(hosts):
                    out.append(f"| {r} | Salad (— W{mem}) | — | not run | not run | not run | — |")
                    continue
                h = hosts[i]
                p = re.sub(r"\.0+ W$", " W", h["power"]) if h.get("power") else "— W"
                o, ow = h["ours"]
                pk, pw = h["peak"]
                sr, sw = h["srb"]
                best = max([x for x in (pk, sr) if x] or [0])
                pct = f"{o / best * 100:.0f}%" if o and best else "—"
                out.append(f"| {r} | Salad ({p}{mem}) | {h['machine'][:8]} | {cell(o, ow)} | {cell(pk, pw)} | "
                           f"{cell(sr, sw)} | {pct} |")
        out.append("")
    return "\n".join(out)


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

    def command(self, ours_secs):
        order = "peak,srb,ours" if self.st["tries"] % 2 == 0 else "srb,peak,ours"
        url = f"https://raw.githubusercontent.com/super3/llmjob/{COMMIT}/tuning/salad/bench.sh"
        return (f"apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl ca-certificates >/dev/null 2>&1 && "
                f"curl -fsSL {url} -o /root/bench.sh && W={self.wallet} ORDER={order} OURS_SECS={ours_secs} "
                f"bash /root/bench.sh 2>&1 | tee -a /root/bench.log")

    def create(self):
        cls, rows, ours = self.plan()
        cid = self.class_id(cls)
        if not cid:
            self.note(f"{cls}: no such class on Salad; skipped")
            self.next_class()
            return
        name = f"bench-{w.slug(cls).replace('rtx', 'gtx')}-{self.st['tries'] + 1}"[:60]
        body = {"name": name, "display_name": name, "replicas": 1, "restart_policy": "never",
                "autostart_policy": True,
                "container": {"image": w.IMAGE, "priority": "high",
                              "resources": {"cpu": 2, "memory": 4096, "gpu_classes": [cid],
                                            "storage_amount": 10737418240},
                              "command": ["/bin/sh", "-c", self.command(ours)]}}
        s, j = w.http("POST", w.CB, body)
        if not 200 <= s < 300:
            log(f"FAIL create {name}: HTTP {s} {str(j)[:200]}")
            return
        t = w.now()
        self.st.update(group=name, created=t, pc_t=None, machine=None, instance=None, log_t=t - 120, lines=[])
        log(f"CREATED {name} ({cls}, high)")

    def note(self, msg):
        self.st["notes"].append(f"{time.strftime('%d %b %H:%M', time.gmtime(w.now()))}Z {msg}")
        log(f"NOTE {msg}")

    def delete(self):
        g = self.st["group"]
        if g:
            s, _ = w.http("DELETE", f"{w.CB}/{g}")
            log(f"DELETED {g}: HTTP {s}")
        self.st["group"] = None

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

    def tick(self):
        if not self.plan():
            return False
        cls, rows, ours = self.plan()
        if self.full(rows):
            self.next_class()
            return True
        if not self.st["group"]:
            self.create()
            return True
        t = w.now()
        s, j = w.http("GET", f"{w.CB}/{self.st['group']}/instances")
        insts = (j or {}).get("instances") or (j or {}).get("items") or [] if s == 200 else []
        on = [i for i in insts if i.get("machine_id") and i.get("state") in ("downloading", "creating", "running")]
        if on:
            m = on[0]["machine_id"]
            if m != self.st.get("machine"):
                self.st.update(machine=m, instance=on[0].get("instance_id") or on[0].get("id"), pc_t=t)
                log(f"PC {self.st['group']}: {m[:8]}")
                if m in self.st["seen"]:
                    self.move_on(f"PC {m[:8]} was benchmarked already")
                    return True
        elif not self.st.get("pc_t") and t - self.st["created"] >= NO_PC_MIN * MIN:
            self.note(f"{cls}: no PC at high in {NO_PC_MIN} min")
            self.next_class()
            return True
        if not self.st.get("machine"):
            return True
        self.read()
        r = parse(x.split("|", 1)[1] for x in self.st["lines"])
        if r["card"]:
            row = row_for(r["card"]["name"], rows)
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
            row = row_for(r["card"]["name"], rows)
            res = {"machine": self.st["machine"], "card": r["card"]["name"], "power": r["card"]["power"],
                   "driver": r["card"]["driver"], "group": self.st["group"],
                   "ours": figure(r["miners"].get("ours"), mean_after=60),
                   "peak": figure(r["miners"].get("peak")), "srb": figure(r["miners"].get("srb")),
                   "exits": {k: v.get("exited") for k, v in r["miners"].items() if v.get("exited")}}
            if res["ours"][0] is None and res["ours"][1] in ("didn't start", "no rate", "not run") and ours == 90:
                res["ours"] = (None, "no build")
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
            return "All classes done."
        g = self.st.get("group")
        if not g:
            return f"Next: {p[0]}."
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
        while True:
            try:
                going = self.tick()
                self.status()
                self.save()
                if not going:
                    self.status(force=True)
                    self.save()
                    log("FINISHED")
                    return
            except Exception as e:  # keep going; the next tick tries again
                log(f"ERROR {type(e).__name__}: {e}")
            time.sleep(MIN)


if __name__ == "__main__":
    Bench().run()
