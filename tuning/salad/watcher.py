#!/usr/bin/env python3
# PRL mining on SaladCloud. Runs one round a minute: reads group state, one log query for every
# group, the pool, then applies the rules in README.md and writes state.json (the ledger), report.md
# (the table) and events.log (one line per action) into the data directory: $SALAD_MINE_DIR, or
# run/ next to this file. config.json there is re-read every round, so limits change without a restart.
# The Salad key is sent as the Salad-Api-Key header from $SALAD_API_KEY when it is set; in a Claude
# cloud session the network proxy adds it instead.
import json, os, re, sys, time, urllib.request, urllib.error, datetime as dt, traceback

D = os.environ.get("SALAD_MINE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")
STATE, CFG, EV, REPORT = (os.path.join(D, f) for f in ("state.json", "config.json", "events.log", "report.md"))
ORG = os.environ.get("SALAD_ORG", "llmjob")
PROJECT = os.environ.get("SALAD_PROJECT", "default")
BASE = f"https://api.salad.com/api/public/organizations/{ORG}"
CB = f"{BASE}/projects/{PROJECT}/containers"
IMAGE = "nvidia/cuda:12.2.2-base-ubuntu22.04"
BOX = "https://raw.githubusercontent.com/super3/llmjob/f0369f92651c2b3b1e24245e50cfe89b02281592/tuning/box.sh"
PRIS = ["batch", "low", "medium", "high"]
UNIT_TH = 419 / 93644          # TH/s per pool unit/s
SHARE_UNITS = 2 ** 21
MIN = 60

# Measured TH/s on Salad (the median of a group's readings) and whether the class paid on the first account.
# Thin classes have a rate but never paid there; one that pays here counts as paid from then on.
KNOWN = {
    "RTX 2060 (6 GB)": (40, True), "RTX 2080 (8 GB)": (72, True), "RTX 3080 (10 GB)": (107, True),
    "RTX 4080 (16 GB)": (199, True), "RTX 5080 (16 GB)": (223, True),
    "RTX 4070 (12 GB)": (105, False), "RTX 4070 Ti (12 GB)": (152, False), "RTX 3090 Ti (24 GB)": (152, False),
    "RTX 5070 (12 GB)": (121, False), "RTX 3060 Ti (8 GB)": (45, False),
}
DEFAULT_CFG = {"max_active": 1, "search": True, "search_token": 0, "restart_token": 0,
               "credit_start": 100.0, "credit_offset": 0.0, "box_ver": "v0.5.13", "ver_override": {}, "report_min": 10,
               "min_margin": 0.10,
               "wallet": os.environ.get("PRL_WALLET", "")}


def now(): return time.time()
def hm(t=None): return time.strftime("%H:%M", time.gmtime(t or now()))
def iso(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?(Z|[+-]\d\d:\d\d)?", s)
    base, frac, tz = m.groups()
    tz = "+00:00" if tz in (None, "Z") else tz
    return dt.datetime.fromisoformat(base + tz).timestamp() + (float(frac) if frac else 0)


def http(method, url, body=None, ctype="application/json", timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", ctype)
    req.add_header("Accept", "application/json")
    # Cloudflare in front of api.salad.com refuses Python's default user agent (error 1010).
    req.add_header("User-Agent", "llmjob-salad-watcher/1.0")
    if url.startswith(BASE) and os.environ.get("SALAD_API_KEY"):
        req.add_header("Salad-Api-Key", os.environ["SALAD_API_KEY"])
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else None)
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(txt)
        except Exception:
            return e.code, {"raw": txt[:500]}
    except Exception as e:
        return 0, {"error": str(e)[:300]}


def event(msg):
    with open(EV, "a") as f:
        f.write(f"{hm()}Z {msg}\n")


def err(msg):
    sys.stderr.write(f"{iso(now())} {msg}\n"); sys.stderr.flush()


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def slug(cls):
    return "rtx" + re.sub(r"[^a-z0-9]", "", cls.split("(")[0].lower().replace("rtx", ""))


def cmd_for(worker, wallet, ver):
    sed = f"sed -e 's/^WORKER=__WORKER__$/WORKER={worker}/' -e 's/^W=prl1.*$/W={wallet}/'"
    if ver != "v0.5.13":
        sed += f" -e 's/^VER=v0.5.13;/VER={ver};/'"
    return ("apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl ca-certificates >/dev/null 2>&1 && "
            f"curl -fsSL {BOX} | {sed} > /root/box.sh; bash /root/box.sh 2>&1 | tee -a /root/box.log")


class W:
    def __init__(self):
        self.st = load(STATE, None) or {"groups": {}, "created": now(), "credits_out": False, "seen": {},
                                        "search_hour": None, "search_token": 0, "restart_token": 0,
                                        "last_round": None, "last_log_end": None, "last_report_q": None,
                                        "last_credit_alert": 0, "classes": {}, "classes_t": 0, "mkt": None, "mkt_t": 0,
                                        "pool": {}, "api_fail": 0}
        self.cfg = DEFAULT_CFG

    # ---------- data ----------
    def market(self):
        st = self.st
        if st["mkt"] and now() - st["mkt_t"] < 300:
            return
        s1, b = http("GET", "https://api.prlscan.com/v1/blocks?limit=1", timeout=30)
        s2, p = http("GET", "https://api.prlscan.com/v1/market/prl", timeout=30)
        if s1 == 200 and s2 == 200:
            it = b["items"][0]
            prl_th_hr = 1e12 * 3600 / (float(it["difficulty"]) * 2 ** 48) * float(it["reward_grains"]) / 1e8
            price = float(p["price_usd"])
            st["mkt"] = {"price": price, "prl_th_hr": prl_th_hr, "usd_th_hr": prl_th_hr * price}
            st["mkt_t"] = now()
        else:
            err(f"market fetch failed {s1} {s2}")

    def classes(self):
        st = self.st
        if st["classes"] and now() - st["classes_t"] < 3600:
            return
        s, j = http("GET", f"{BASE}/gpu-classes")
        if s == 200:
            st["classes"] = {c["name"]: {"id": c["id"], "prices": {p["priority"]: float(p["price"]) for p in c["prices"]}}
                             for c in j["items"]}
            st["classes_t"] = now()
        else:
            err(f"gpu-classes failed {s} {j}")

    def margin(self, cls, pri, rate=None):
        """The margin a class should make at a priority, at its measured rate."""
        return margin_of((rate or KNOWN[cls][0]) * self.st["mkt"]["usd_th_hr"], self.st["classes"][cls]["prices"][pri])

    def paid(self, g_or_cls):
        cls = g_or_cls["class"] if isinstance(g_or_cls, dict) else g_or_cls
        return KNOWN[cls][1] or cls in self.st.setdefault("paid_classes", [])

    def last_lines(self, g, n=5):
        """The box's last log lines, so a deletion says why. One query, only when a group is deleted."""
        t = now()
        body = {"start_time": iso(t - 2400), "end_time": iso(t), "page_size": 12, "sort_order": "desc",
                "query": f'resource.type = "container" AND resource.labels.container_group_name = "{g["name"]}"'}
        s, j = http("POST", f"{BASE}/log-entries", body, timeout=60)
        if s != 200:
            return [f"(log query failed: HTTP {s})"]
        lines = [(it.get("text_log") or "").strip()[:160] for it in (j.get("items") or [])]
        return [l for l in lines if l][:n][::-1] or ["(no container output)"]

    def logs(self, active):
        """One query for every group: all container lines since the last successful query (5 min overlap).
        Salad returns at most 100 lines a page, so a busy round reads up to 5 pages, oldest first."""
        st = self.st
        t = now()
        start = max(min(g["created"] for g in active) - 60, (st["last_log_end"] or 0) - 300, t - 1800)
        items = []
        for page in range(5):
            body = {"start_time": iso(start), "end_time": iso(t), "query": 'resource.type = "container"',
                    "page_size": 100, "sort_order": "asc"}
            s, j = http("POST", f"{BASE}/log-entries", body, timeout=90)
            if s != 200:
                err(f"log-entries failed {s} {str(j)[:200]}")
                break
            got = j.get("items") or []
            items += got
            if len(got) < 100:
                st["last_log_end"] = t
                break
            start = parse_iso(got[-1]["time"])   # the next page starts at this page's last line
            st["last_log_end"] = start
        out = []
        seen = st["seen"]
        for it in items:
            lab = (it.get("resource") or {}).get("labels") or {}
            name, text, tm = lab.get("container_group_name"), it.get("text_log") or "", it.get("time")
            if not name or not tm:
                continue
            key = f"{tm}|{name}|{text[:100]}"
            if key in seen:
                continue
            seen[key] = t
            out.append((parse_iso(tm), name, lab.get("machine_id"), text))
        for k in [k for k, v in seen.items() if t - v > 7200]:
            del seen[k]
        out.sort()
        return out

    def pool(self):
        wallet = self.cfg["wallet"]
        s, j = http("GET", f"https://pearl.herominers.com/api/stats_address?address={wallet}&longpoll=false", timeout=30)
        res = {}
        if s != 200 or not isinstance(j, dict):
            return res
        ws = j.get("workers") or []
        if isinstance(ws, dict):
            ws = [dict(v, name=k) for k, v in ws.items()]
        for w in ws:
            if isinstance(w, dict) and w.get("name"):
                res[w["name"]] = w
        return res

    # ---------- actions ----------
    def delete(self, g, kind, reason):
        why = self.last_lines(g) if g["had_pc"] else []
        s, j = http("DELETE", f"{CB}/{g['name']}")
        if 200 <= s < 300 or s == 404:
            g.update(ended=now(), end_kind=kind, end_reason=reason, has_pc=False)
            event(f"DELETED {g['name']} ({g['class']}, {g['pri']}): {reason} [{'released' if kind == 'released' else 'failed'}]"
                  + "".join(f"\n    WHY {g['name']}: {l}" for l in why))
        else:
            err(f"delete {g['name']} failed {s} {j}")

    def reallocate(self, g, reason):
        if g["reallocs"] >= 3:
            self.delete(g, "failed", f"{reason}; already reallocated 3 times")
            return
        if not g.get("instance"):
            return
        s, j = http("POST", f"{CB}/{g['name']}/instances/{g['instance']}/reallocate")
        if 200 <= s < 300:
            g["reallocs"] += 1
            g.setdefault("left", []).append(g["machine"])
            event(f"REALLOCATED {g['name']} off PC {(g['machine'] or '')[:8]} ({reason}); reallocation {g['reallocs']} of 3")
            g.update(machine=None, pc_since=None, lost_since=now(), profit_fail=0, has_pc=False)
        else:
            err(f"reallocate {g['name']} failed {s} {j}")

    def set_priority(self, g, pri):
        s, j = http("PATCH", f"{CB}/{g['name']}", {"container": {"priority": pri}}, ctype="application/merge-patch+json")
        if 200 <= s < 300:
            old = g["pri"]
            g.update(pri=pri, price=self.st["classes"][g["class"]]["prices"][pri], pri_since=now())
            event(f"PRIORITY {g['name']}: no PC at {old} in 10 min, moved to {pri} (${g['price']:.3f}/hr, "
                  f"{self.margin(g['class'], pri) * 100:+.0f}% margin)")
            return True
        err(f"patch {g['name']} -> {pri} refused {s} {str(j)[:200]}")
        return False

    def create(self, cls, pri):
        st = self.st
        active_names = {g["name"] for g in st["groups"].values() if not g["ended"]}
        base = "biz-" + slug(cls)
        name, i = base, 2
        while name in active_names:
            name, i = f"{base}-{i}", i + 1
        ver = self.cfg["ver_override"].get(name, self.cfg["box_ver"])
        body = {"name": name, "display_name": name, "replicas": 1, "restart_policy": "always", "autostart_policy": True,
                "container": {"image": IMAGE, "priority": pri,
                              "resources": {"cpu": 2, "memory": 4096, "storage_amount": 10737418240,
                                            "gpu_classes": [st["classes"][cls]["id"]]},
                              "command": ["/bin/sh", "-c", cmd_for(name, self.cfg["wallet"], ver)]}}
        s, j = http("POST", CB, body)
        if 200 <= s < 300:
            rate, paid = KNOWN[cls]
            t = now()
            key = f"{name}@{int(t)}"
            st["groups"][key] = {
                "key": key, "name": name, "class": cls, "rate": rate, "paid": paid, "pri": pri,
                "price": st["classes"][cls]["prices"][pri], "created": t, "pri_since": t, "ver": ver,
                "had_pc": False, "has_pc": False, "pc_since": None, "machine": None, "instance": None, "lost_since": None,
                "reallocs": 0, "spend": 0.0, "prl": 0.0, "pc_min": 0.0, "mined_min": 0, "readings": [],
                "acc": {}, "last_share": None, "profit_fail": 0, "last_check": 0, "status": "pending",
                "ended": None, "end_kind": None, "end_reason": None, "blocked": [], "cards": [], "pool_ths": None,
                "last_start": 0, "first_share_said": False}
            event(f"CREATED {name}: {cls} at {pri} (${st['classes'][cls]['prices'][pri]:.3f}/hr, "
                  f"{self.margin(cls, pri) * 100:+.0f}% margin at {rate} TH/s), miner {ver}")
            return True
        text = json.dumps(j)
        if "no_credits_available" in text:
            if not st["credits_out"]:
                event("ALERT Salad says no credits are available; not creating groups until a top-up")
            st["credits_out"] = True
        else:
            err(f"create {name} failed {s} {text[:300]}")
            event(f"CREATE FAILED {name} ({cls}, {pri}): HTTP {s} {text[:200]}")
        return False

    # ---------- the round ----------
    def round(self):
        st, t = self.st, now()
        self.cfg = dict(DEFAULT_CFG, **load(CFG, {}))
        self.market()
        self.classes()
        if not st["mkt"] or not st["classes"]:
            return
        dt_s = min(t - st["last_round"], 1800) if st["last_round"] else 0
        st["last_round"] = t
        active = [g for g in st["groups"].values() if not g["ended"]]

        # Restart groups that Salad stopped for lack of credit, once asked to.
        if self.cfg["restart_token"] != st["restart_token"]:
            st["restart_token"] = self.cfg["restart_token"]
            st["credits_out"] = False
            for g in active:
                if g["status"] == "stopped":
                    s, j = http("POST", f"{CB}/{g['name']}/start")
                    event(f"RESTART {g['name']}: HTTP {s}")

        # Group and instance state.
        for g in active:
            s, j = http("GET", f"{CB}/{g['name']}")
            if s == 404:
                g.update(ended=t, end_kind="released", end_reason="group no longer exists on Salad", has_pc=False)
                event(f"GONE {g['name']}: the group no longer exists on Salad")
                continue
            if s != 200:
                st["api_fail"] += 1
                if st["api_fail"] in (5, 30):
                    event(f"WARN Salad API failing for {st['api_fail']} calls in a row: HTTP {s} {str(j)[:150]}")
                continue
            st["api_fail"] = 0
            cs = j.get("current_state") or {}
            g["status"] = cs.get("status", "?")
            g["state_desc"] = cs.get("description")
            si, ji = http("GET", f"{CB}/{g['name']}/instances")
            insts = (ji or {}).get("instances") or (ji or {}).get("items") or [] if si == 200 else []
            on = [i for i in insts if i.get("state") in ("downloading", "creating", "running") and i.get("machine_id")]
            g["inst_state"] = ",".join(i.get("state", "?") for i in insts) or "-"
            if si != 200:
                g["_skip"] = True
                continue
            g["_skip"] = False
            g["has_pc"] = bool(on)
            if on:
                g["instance"] = on[0].get("instance_id") or on[0].get("id")
                g["_machine"] = on[0]["machine_id"]
            else:
                ins = insts[0] if insts else {}
                g["instance"] = ins.get("instance_id") or ins.get("id") or g.get("instance")
                g["_machine"] = None

        # Billing: price for every minute a group has a PC.
        for g in active:
            if g.get("has_pc") and not g["ended"]:
                g["spend"] += g["price"] * dt_s / 3600
                g["pc_min"] += dt_s / 60

        # One log query for every group.
        by_name = {g["name"]: g for g in active if not g["ended"]}
        if by_name:
            for tm, name, mach, text in self.logs(list(by_name.values())):
                g = by_name.get(name)
                if not g or tm < g["created"]:
                    continue
                m = re.search(r"\[mining g(\d+)\] \d\d:\d\d:\d\d\s+([0-9.]+) TH/s · ([0-9,]+) accepted", text)
                if m:
                    ths, acc = float(m.group(2)), int(m.group(3).replace(",", ""))
                    g["readings"].append([tm, mach, ths, acc])
                    g["readings"] = [r for r in g["readings"] if t - r[0] < 7200]
                    if ths > 0:
                        g["prl"] += ths / 60 * st["mkt"]["prl_th_hr"]
                        g["mined_min"] += 1
                    prev = g["acc"].get(mach)
                    if (prev is None and acc > 0) or (prev is not None and acc > prev) or (prev is not None and acc < prev and acc > 0):
                        g["last_share"] = max(g["last_share"] or 0, tm)
                    g["acc"][mach] = acc
                    continue
                text = text.strip()
                if "port 1200 blocked" in text:
                    if mach not in g["blocked"]:
                        g["blocked"].append(mach)
                        event(f"PORT BLOCKED {name} on PC {(mach or '')[:8]}: {text.strip()[:120]}")
                elif text.startswith("[card]") and mach not in g["cards"]:
                    g["cards"].append(mach)
                    event(f"CARD {name} PC {(mach or '')[:8]}: {text.strip()[7:160]}")
                elif re.match(r"\[run\] (FAIL|g\d miner exited)", text) or text.startswith("[run] release"):
                    event(f"LOG {name} PC {(mach or '')[:8]}: {text.strip()[:200]}")

        # Pool: shares and pool-side hashrate.
        pw = self.pool() if by_name else {}
        for g in by_name.values():
            w = pw.get(f"{g['name']}-g0")
            if not w:
                continue
            ls = w.get("lastShare") or w.get("last_share")
            try:
                ls = float(ls)
                ls = ls / 1000 if ls > 1e12 else ls
                if ls >= g["created"]:
                    g["last_share"] = max(g["last_share"] or 0, ls)
            except (TypeError, ValueError):
                pass
            try:
                g["pool_ths"] = float(w.get("hashrate") or 0) * UNIT_TH
            except (TypeError, ValueError):
                pass
        for g in by_name.values():
            if g["last_share"] and not g["first_share_said"]:
                g["first_share_said"] = True
                event(f"FIRST SHARE {g['name']} at {hm(g['last_share'])}Z")

        # Rules.
        for g in by_name.values():
            if g.get("_skip"):
                continue
            self.rules(g, t)

        # Hourly search at :01, or when asked through config.search_token.
        tm = time.gmtime(t)
        hour = time.strftime("%Y-%m-%dT%H", tm)
        asked = self.cfg["search_token"] != st["search_token"]
        if self.cfg["search"] and not st["credits_out"] and (asked or (tm.tm_min >= 1 and st["search_hour"] != hour)):
            st["search_hour"], st["search_token"] = hour, self.cfg["search_token"]
            self.search()

        # Credit warning and the report tick, every report_min minutes.
        self.write_report()
        q = int(t // (60 * self.cfg["report_min"]))
        if st["last_report_q"] != q:
            st["last_report_q"] = q
            event("REPORT")

    def rules(self, g, t):
        st = self.st
        if g["status"] == "stopped":
            if st["credits_out"]:
                return
            if not g["had_pc"] and t - g["last_start"] > 300:
                g["last_start"] = t
                s, j = http("POST", f"{CB}/{g['name']}/start")
                g["pri_since"] = t
                event(f"START {g['name']}: it was stopped before getting a PC; POST /start -> HTTP {s}")
            elif g["had_pc"]:
                st["credits_out"] = True
                event(f"ALERT Salad stopped {g['name']} after it had a PC ({g.get('state_desc') or 'no reason given'}). "
                      "Credit may have run out; leaving stopped groups alone until a top-up.")
            return
        if g["status"] == "failed" and not g.get("failed_said"):
            g["failed_said"] = True
            event(f"WARN {g['name']} shows status failed: {g.get('state_desc')}")

        mach = g.pop("_machine", None)
        if g["has_pc"] and mach in g.get("left", []):
            return  # the PC it was just moved off still shows; Salad bills it, but it isn't a new PC
        if g["has_pc"]:
            if mach != g["machine"]:
                first = not g["had_pc"]
                g.update(machine=mach, pc_since=t, lost_since=None, profit_fail=0, had_pc=True)
                event(f"{'GOT PC' if first else 'NEW PC'} {g['name']} ({g['class']}, {g['pri']}): PC {mach[:8]}")
        else:
            if g["had_pc"]:
                if g["machine"] is not None:
                    event(f"LOST PC {g['name']}: PC {g['machine'][:8]} gone (instance {g.get('inst_state', '-')})")
                    g.update(machine=None, pc_since=None)
                if g["lost_since"] is None:
                    g["lost_since"] = t
                if t - g["lost_since"] >= 20 * MIN:
                    self.delete(g, "released", "lost its PC and got no other in 20 min")
            elif g["status"] == "pending":
                g["pri_since"] = t  # preparing the image: not looking for a PC yet, and a priority change is refused
            elif t - g["pri_since"] >= 10 * MIN:
                nxt = next((p for p in PRIS[PRIS.index(g["pri"]) + 1:] if self.margin(g["class"], p) >= self.cfg["min_margin"]), None)
                if nxt is None:
                    self.delete(g, "released", f"no PC at {g['pri']} and no higher priority clears "
                                               f"{self.cfg['min_margin']:.0%}")
                else:
                    self.set_priority(g, nxt)
            return

        on = t - g["pc_since"]
        # Rule 6: the PC blocks the pool port.
        if g["machine"] in g["blocked"]:
            self.reallocate(g, "PC blocks outbound port 1200")
            return
        # This PC's readings (one a minute). A PC that isn't mining 10 min after we got it is left.
        cur = [r for r in g["readings"] if r[1] == g["machine"] and r[2] > 0 and r[0] >= g["pc_since"] - 5 * MIN]
        if not cur and on >= 10 * MIN:
            self.leave(g, f"no mining reading {on / 60:.0f} min after getting PC {g['machine'][:8]}")
            return
        # Rule 4: no share after 20 min on a PC.
        if on >= 20 * MIN:
            gap = SHARE_UNITS / (g["rate"] / UNIT_TH)
            limit = max(10 * MIN, 8 * gap)
            ref = max(g["pc_since"], g["last_share"] or 0)
            if t - ref >= limit:
                self.leave(g, f"no share for {(t - ref) / 60:.0f} min on PC {g['machine'][:8]}")
                return
        # Rule 5: only PCs that clear the margin. After the miner's first 2 minutes (warm-up), the last 3
        # readings must clear min_margin at the PC's price, every round; the first time they don't, the PC is left.
        post = [r for r in cur if r[0] >= cur[0][0] + 2 * MIN] if cur else []
        if len(post) >= 3:
            usd, bar = st["mkt"]["usd_th_hr"], self.cfg["min_margin"]
            avg = sum(r[2] for r in post[-3:]) / 3
            if margin_of(avg * usd, g["price"]) < bar:
                self.leave(g, f"earns ${avg * usd:.3f}/hr at {avg:.0f} TH/s on PC {g['machine'][:8]} against its "
                              f"${g['price']:.3f}/hr ({margin_of(avg * usd, g['price']) * 100:+.0f}%, under {bar:.0%})")
                return
            ten = post[-10:]
            shared = (g["last_share"] or 0) >= g["pc_since"]   # the pool accepts its work, not just the miner's figure
            if not self.paid(g) and shared and len(ten) == 10 and margin_of(sum(r[2] for r in ten) / 10 * usd, g["price"]) >= bar:
                st["paid_classes"].append(g["class"])
                event(f"PAID {g['name']}: {g['class']} clears {bar:.0%} here ({avg:.0f} TH/s); it counts as paid now")

    def leave(self, g, why):
        """Stop paying for this PC: another PC if the class has paid before, otherwise delete the group."""
        if self.paid(g):
            self.reallocate(g, why)
        else:
            self.delete(g, "failed", why)

    def search(self):
        st = self.st
        active = [g for g in st["groups"].values() if not g["ended"]]
        s, q = http("GET", f"{BASE}/quotas")
        quota = (q or {}).get("container_groups_quotas", {}).get("container_replicas_quota", 10) if s == 200 else 10
        room = min(self.cfg["max_active"], quota) - len(active)
        if room <= 0:
            return
        # Each class is tried once on this account, as on the first one, except that a class whose groups only
        # ended for want of a PC (released) is tried again in a search at least an hour after.
        t = now()
        skip = {g["class"] for g in st["groups"].values()
                if not g["ended"] or g["end_kind"] != "released" or t - g["ended"] < 3600}
        cands = []
        for cls, (rate, _) in KNOWN.items():
            if cls not in st["classes"] or cls in skip:
                continue
            body = {"cpu": 2, "memory": 4096, "storage_amount": 10737418240, "gpu_classes": [st["classes"][cls]["id"]]}
            s, a = http("POST", f"{BASE}/availability/sce-gpu-availability", body)
            if s != 200:
                continue
            for p in PRIS:
                if (a.get(f"available_gpu_{p}") or 0) > 0 and self.margin(cls, p) >= self.cfg["min_margin"]:
                    profit = rate * st["mkt"]["usd_th_hr"] - st["classes"][cls]["prices"][p]
                    cands.append((not self.paid(cls), -profit, cls, p))
                    break
        cands.sort()
        made = 0
        for _, _, cls, p in cands:
            if made >= min(3, room) or st["credits_out"]:
                break
            if self.create(cls, p):
                made += 1
        if not made:
            event(f"SEARCH found nothing to create (room {room}, {len(cands)} candidates)")

    # ---------- report ----------
    def write_report(self):
        st, t = self.st, now()
        mk = st["mkt"]
        price, usd = mk["price"], mk["usd_th_hr"]
        rows, act_rows = [], []
        all_ths = all_cost = all_earn = 0.0
        mining = 0
        ended = [g for g in st["groups"].values() if g["ended"]]
        for g in sorted((g for g in st["groups"].values() if not g["ended"]), key=lambda g: g["created"]):
            cur = [r for r in g["readings"] if r[1] == g["machine"] and r[2] > 0 and r[0] >= t - 10 * MIN] if g["has_pc"] else []
            ths = sum(r[2] for r in cur) / len(cur) if cur else 0.0
            cost = g["price"] if g["has_pc"] else 0.0
            earn = ths * usd
            state = "⏹️" if g["status"] == "stopped" else "⛏️" if g["has_pc"] else "⏳"
            if g["has_pc"] and cur:
                mining += 1
            all_ths += ths; all_cost += cost; all_earn += earn
            prof = g["prl"] * price - g["spend"]
            host = g["name"] + (f" · PC {g['machine'][:8]}" if g["machine"] else "")
            short = g["class"].split(" (")[0]
            rows.append(f"| ⚡ {short} ×1 (Salad, {g['pri']}) | {host} | {state} | {ths:.0f} | {cost:.3f} | "
                        f"{earn - cost:+.3f} | {pct(earn - cost, earn)} | {g['prl']:.3f} | {prof:+.2f} |")
        if ended:
            rel = sum(1 for g in ended if g["end_kind"] == "released")
            fail = len(ended) - rel
            eprl = sum(g["prl"] for g in ended)
            esp = sum(g["spend"] for g in ended)
            rows.append(f"| Others ×{len(ended)} | 🔁 {rel} released / ❌ {fail} failed | ended | — | — | — | — | "
                        f"{eprl:.3f} | {eprl * price - esp:+.2f} |")
        spend = sum(g["spend"] for g in st["groups"].values())
        prl = sum(g["prl"] for g in st["groups"].values())
        rows.append(f"| **All** | {mining} GPU{'s' if mining != 1 else ''} mining | | **{all_ths:.0f}** | **{all_cost:.3f}** | "
                    f"**{all_earn - all_cost:+.3f}** | **{pct(all_earn - all_cost, all_earn)}** | **{prl:.3f}** | "
                    f"**{prl * price - spend:+.2f}** |")
        credit = self.cfg["credit_start"] + self.cfg["credit_offset"] - spend
        hours = credit / all_cost if all_cost > 0 else None
        lines = [f"PRL ${price:.3f} · ${usd:.6f}/TH-hr · {iso(t)}", "",
                 "| Box | Machine / host | State | TH/s | $/hr | Profit/hr | Margin | PRL | Profit $ |",
                 "|---|---|---|---|---|---|---|---|---|", *rows, "",
                 "⛏️ on a PC · ⏳ waiting for a PC · ⏹️ stopped. TH/s is the miner's average over the last 10 min on its "
                 "current PC (0 while it starts). $/hr is what Salad bills while a group has a PC (a waiting group costs "
                 "nothing). Profit $ is PRL mined at today's price minus spend.", "",
                 f"- Spent ${spend:.2f}; mined {prl:.3f} PRL.",
                 (f"- Each PRL cost ${spend / prl:.3f} to mine (break-even sale price); it's worth ${price:.3f} today."
                  if prl > 0 else "- No PRL mined yet, so no cost per PRL."),
                 f"- Salad credit left: ${credit:.2f} of ${self.cfg['credit_start'] + self.cfg['credit_offset']:.2f} (our count)"
                 + (f"; lasts {hours:.0f} h at ${all_cost:.3f}/hr." if hours is not None else "; nothing is billing now.")]
        if st["credits_out"]:
            lines.append("- ⚠️ Credit ran out: Salad stopped the groups. They restart after a top-up.")
        with open(REPORT + ".tmp", "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(REPORT + ".tmp", REPORT)
        if hours is not None and hours < 12 and t - st["last_credit_alert"] > 3600:
            st["last_credit_alert"] = t
            event(f"ALERT credit ${credit:.2f} lasts about {hours:.1f} h at ${all_cost:.3f}/hr")


def margin_of(earn, price):
    """Profit as a share of earnings, as the first account measured it: (earn - price) / earn."""
    return (earn - price) / earn if earn > 0 else float("-inf")


def pct(a, b):
    return f"{a / b * 100:+.0f}%" if b > 0 else "—"


def main():
    os.makedirs(D, exist_ok=True)
    w = W()
    w.cfg = dict(DEFAULT_CFG, **load(CFG, {}))
    if not re.match(r"^prl1[0-9a-z]{50,}$", w.cfg["wallet"]):
        sys.exit("No payout address: set PRL_WALLET or \"wallet\" in config.json")
    if not w.st.get("started_said"):
        w.st["started_said"] = True
        event("WATCHER started")
    else:
        event("WATCHER resumed")
    while True:
        t0 = now()
        try:
            w.round()
        except Exception:
            err(traceback.format_exc())
        save(STATE, w.st)
        time.sleep(max(5, 60 - (now() - t0)))


if __name__ == "__main__":
    main()
