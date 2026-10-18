#!/usr/bin/env python3
# PRL mining on SaladCloud. Runs one round a minute: reads group state, one log query for every
# group, the pool, then applies the rules in README.md and writes state.json (the ledger), report.md
# (the table) and events.log (one line per action) into the data directory: $SALAD_MINE_DIR, or
# run/ next to this file. config.json there is re-read every round, so limits change without a restart.
# The Salad key is sent as the Salad-Api-Key header from $SALAD_API_KEY when it is set; in a Claude
# cloud session the network proxy adds it instead.
import json, os, re, sys, time, urllib.request, urllib.error, datetime as dt, traceback
from zoneinfo import ZoneInfo

D = os.environ.get("SALAD_MINE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")
STATE, CFG, EV, REPORT = (os.path.join(D, f) for f in ("state.json", "config.json", "events.log", "report.md"))
ORG = os.environ.get("SALAD_ORG", "llmjob")
RAILWAY = "https://backboard.railway.com/graphql/v2"
PROJECT = os.environ.get("SALAD_PROJECT", "default")
BASE = f"https://api.salad.com/api/public/organizations/{ORG}"
CB = f"{BASE}/projects/{PROJECT}/containers"
IMAGE = "nvidia/cuda:12.2.2-base-ubuntu22.04"
BOX = "https://raw.githubusercontent.com/super3/llmjob/f0369f92651c2b3b1e24245e50cfe89b02281592/tuning/box.sh"
PRIS = ["batch", "low", "medium", "high"]
UNIT_TH = 419 / 93644          # TH/s per pool unit/s
SHARE_UNITS = 2 ** 21
MIN = 60
START_MIN = 5   # minutes a new PC is billed before its miner sends a reading
NB = "\u00a0"   # no-break space: keeps a table cell on one line
EST = "*"  # marks a figure that is our own count or estimate, or entered by hand, not read from an API


def star(cell):
    """A figure we count or estimate ourselves gets EST; a dash stays a dash."""
    return cell if cell == "—" else cell + EST

# Measured TH/s on Salad (the median of a group's readings) and whether the class paid on the first account.
# Thin classes have a rate but never paid there; one that pays here counts as paid from then on.
KNOWN = {
    "RTX 2060 (6 GB)": (40, True), "RTX 2080 (8 GB)": (72, True), "RTX 3080 (10 GB)": (107, True),
    "RTX 4080 (16 GB)": (199, True), "RTX 5080 (16 GB)": (223, True),
    "RTX 4070 (12 GB)": (105, False), "RTX 4070 Ti (12 GB)": (152, False), "RTX 3090 Ti (24 GB)": (152, False),
    "RTX 5070 (12 GB)": (121, False), "RTX 3060 Ti (8 GB)": (45, False),
    # Not tried on Salad yet: benchmark.md or our Vast boxes (the rates #255's search used). Salad PCs often run
    # power-capped under these, so rule 5 judges each PC on its own readings within minutes. Left out: the 3080 Ti,
    # 3090 and 4070 Ti Super (lost money on the first account) and the 3070 Ti (no share).
    "RTX 2070 (8 GB)": (48, False), "RTX 2080 Ti (11 GB)": (79, False), "RTX 3060 (12 GB)": (49, False),
    "RTX 3070 (8 GB)": (59, False), "RTX 4060 (8 GB)": (55, False), "RTX 4060 Ti (16 GB)": (89, False),
    "RTX 4090 (24 GB)": (315, False), "RTX 5060 (8 GB)": (76, False), "RTX 5060 Ti (16 GB)": (94, False),
    "RTX 5070 Ti (16 GB)": (174, False), "RTX 5090 (32 GB)": (418, False),
}
DEFAULT_CFG = {"max_active": 1, "search": True, "search_token": 0, "restart_token": 0,
               "credit_start": 100.0, "credit_offset": 0.0, "box_ver": "v0.5.13", "ver_override": {}, "report_min": 10,
               "min_margin": 0.10, "max_new_per_hour": 3, "max_per_class": 1,
               "search_every_min": 60, "retry_after_min": 60, "no_pc_streak": 3, "no_pc_hold_min": 180, "drift": 0.15,
               "upgrade_gain": 0.05, "upgrade_every_min": 15, "swap_wait_min": 20,
               "yield": "auto", "yield_min_blocks": 10, "yield_floor": 0.5, "bad_pc_hold_h": 24, "rate_days": 7,
               "trader": None, "safetrade": None,
               "wallet": os.environ.get("PRL_WALLET", ""),
               # Monero on the box's CPU (release v0.6.0 on): the groups named in xmr_groups ("*" for every group)
               # mine it to xmr_wallet. Salad bills a GPU group the same for any vCPU and RAM, so they ask for
               # xmr_cpu vCPUs and xmr_memory MB; the CLI picks the threads (xmr_threads 0) and keeps one CPU free.
               "xmr_wallet": "", "xmr_groups": [], "xmr_ver": "v0.6.0", "xmr_cpu": 8, "xmr_memory": 8192,
               "xmr_threads": 0,
               # Kryptex's PPS+ pool for Pearl: kryptex_groups groups mine there instead of HeroMiners, with release
               # kryptex_ver or later (the first with --pool). New groups start there while fewer are, and groups
               # waiting for a PC are switched (see kx_sync). Two Kryptex boxes that mine no shares, while no other
               # box's work reaches Kryptex, stop the test until kryptex_token changes.
               "kryptex_groups": 0, "kryptex_host": "prl-us.kryptex.network:7048", "kryptex_ver": "v0.6.1",
               "kryptex_pay": 0.94, "kryptex_token": 0}
XMR_FEE = 0.009   # HeroMiners' Monero pool fee
KX_API = "https://pool.kryptex.com/prl/api"


def now(): return time.time()


def share_t(row):
    """A pool worker row's last share, in seconds (HeroMiners' lastShare, Kryptex's last_share; either can be in ms),
    or 0 for none."""
    try:
        ls = float((row or {}).get("lastShare") or (row or {}).get("last_share") or 0)
    except (TypeError, ValueError):
        return 0.0
    return ls / 1000 if ls > 1e12 else ls
def hm(t=None): return time.strftime("%H:%M", time.gmtime(t or now()))
def iso(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
# The report's clock is New York time (EDT in summer, EST in winter), where the user is.
NY = ZoneInfo("America/New_York")


def stamp(t):
    d = dt.datetime.fromtimestamp(t, NY)
    return f"{d:%H:%M %Z}, {d:%a} {d.day} {d:%b}"


def parse_iso(s):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?(Z|[+-]\d\d:\d\d)?", s)
    base, frac, tz = m.groups()
    tz = "+00:00" if tz in (None, "Z") else tz
    return dt.datetime.fromisoformat(base + tz).timestamp() + (float(frac) if frac else 0)


def http(method, url, body=None, ctype="application/json", timeout=60, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
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


def vnum(ver):
    """A release tag as numbers, for comparing: "v0.6.0" -> (0, 6, 0)."""
    return tuple(int(x) for x in re.findall(r"\d+", ver))


def cmd_for(worker, wallet, ver, xmr=None, threads=0, kryptex=None):
    sed = f"sed -e 's/^WORKER=__WORKER__$/WORKER={worker}/' -e 's/^W=prl1.*$/W={wallet}/'"
    if ver != "v0.5.13":
        sed += f" -e 's/^VER=v0.5.13;/VER={ver};/'"
    flags = []
    if kryptex:
        # Kryptex: Pearl on its stratum (release v0.6.1 on). The miner logs in there as "wallet/worker" by itself.
        flags.append(f"--pool {kryptex}")
    if xmr:
        # Monero: box.sh also downloads the release's CPU worker and puts it beside each GPU's CLI.
        flags.append(f"--xmr-address {xmr}" + (f" --cpu-threads {threads}" if threads else ""))
        sed += (" -e 's/ pearl_core_cu13.node; do/ pearl_core_cu13.node llmjob-xmr-worker; do/'"
                " -e 's|cp $D/rel/cli $R/|cp $D/rel/cli $D/rel/llmjob-xmr-worker $R/ \\&\\& chmod +x $R/llmjob-xmr-worker|'")
    if flags:
        # Both go in box.sh's default flags (a control file for the box would override them; the biz- boxes have none).
        sed += f" -e 's|\"flags\":\"\"}}|\"flags\":\"{' '.join(flags)}\"}}|'"
    return (f"apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl ca-certificates >/dev/null 2>&1 && "
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
        return margin_of((rate or self.rate(cls)) * self.usd(), self.st["classes"][cls]["prices"][pri])

    def rate(self, cls):
        """The TH/s a new group of this class should make: the median of the PCs it has mined on here in the
        last rate_days (each PC's rate on its last 3 readings after warm-up, capped and slow PCs included), or
        the benchmark in KNOWN before any has."""
        pcs = sorted(r for r, _ in self.st.get("pc_rates", {}).get(cls, {}).values())
        if not pcs:
            return KNOWN[cls][0]
        n = len(pcs)
        return pcs[n // 2] if n % 2 else (pcs[n // 2 - 1] + pcs[n // 2]) / 2

    def record_rates(self, t):
        """Keep each PC's latest rate, by class, for rate(). The first time, fill it from the readings in the
        ledger (the last 2 hours of each group's PCs)."""
        st = self.st
        if "pc_rates" not in st:
            st["pc_rates"] = {}
            for g in st["groups"].values():
                by = {}
                for r in g.get("readings", []):
                    if r[2] > 0 and r[1]:
                        by.setdefault(r[1], []).append(r)
                for mach, rs in by.items():
                    post = [r for r in rs if r[0] >= rs[0][0] + 2 * MIN][-3:]
                    if len(post) == 3:
                        st["pc_rates"].setdefault(g["class"], {})[mach] = [sum(r[2] for r in post) / 3, post[-1][0]]
        for g in st["groups"].values():
            r = None if g["ended"] else self.pc_rate(g)
            if r is not None:
                st["pc_rates"].setdefault(g["class"], {})[g["machine"]] = [r, t]
        for pcs in st["pc_rates"].values():
            for m in [m for m, (_, seen) in pcs.items() if t - seen >= 86400 * self.cfg["rate_days"]]:
                del pcs[m]

    def credit_ratio(self):
        """The share of our estimated PRL that the pool has credited (paid, unlocked and unconfirmed), once it has
        credited us in yield_min_blocks blocks; 1.0 before that. Kept within [yield_floor, 1]. The estimate is the
        miners' speed at the network's rate, so this ratio carries the pool's fee, orphaned blocks and its luck."""
        pt = self.st.get("pool_tot")
        if not pt or pt["blocks"] < self.cfg["yield_min_blocks"]:
            return 1.0
        prl = sum(g.get("prl", 0.0) - g.get("prl_kx", 0.0) for g in self.st["groups"].values())
        if prl < 1:
            return 1.0
        cred = pt["paid"] + pt["balance"] + pt["unconfirmed"]
        return min(1.0, max(self.cfg["yield_floor"], cred / prl))

    def kx_ratio(self):
        """The same for Kryptex: what it has credited (balance and payouts) over its boxes' estimate. PPS+ credits
        each share as it is accepted, so this holds from the first PRL; before that, the 94% of a share's value that it was
        measured to pay on 17 October (its advertised fee is 3%)."""
        kt = self.st.get("kx_tot")
        prl = sum(g.get("prl_kx", 0.0) for g in self.st["groups"].values())
        if not kt or prl < 1:
            return self.cfg["kryptex_pay"]
        return min(1.0, max(self.cfg["yield_floor"], (kt["total"] + kt["paid"]) / prl))

    def ratio(self, g):
        """The credit ratio for a group's pool now."""
        return self.kx_ratio() if g.get("kx") else self.credit_ratio()

    def credited(self, g, hy=None, ky=None):
        """The PRL a group's pools should have credited: its HeroMiners part and its Kryptex part, each at its
        pool's ratio."""
        hy = self.credit_ratio() if hy is None else hy
        ky = self.kx_ratio() if ky is None else ky
        k = g.get("prl_kx", 0.0)
        return (g.get("prl", 0.0) - k) * hy + k * ky

    def usd(self):
        """$ per TH-hr the rules count on: the network's rate times config "yield", a number, or "auto" for the
        pool's credit ratio."""
        y = self.cfg.get("yield", "auto")
        return self.st["mkt"]["usd_th_hr"] * (self.credit_ratio() if y == "auto" else float(y))

    def paid(self, g_or_cls):
        cls = g_or_cls["class"] if isinstance(g_or_cls, dict) else g_or_cls
        return KNOWN[cls][1] or cls in self.st.setdefault("paid_classes", [])

    def last_lines(self, g, n=5):
        """The box's last log lines, so a deletion says why. Only when a group is deleted."""
        t = now()
        items = self.read_logs(t - 2400, t, f'resource.type = "container" AND '
                                             f'resource.labels.container_group_name = "{g["name"]}"')
        if items is None:
            return ["(log query failed)"]
        items.sort(key=lambda it: it.get("time") or "")
        lines = [(it.get("text_log") or "").strip()[:160] for it in items]
        return [l for l in lines if l][-n:] or ["(no container output)"]

    def read_logs(self, start, end, query, depth=0):
        """Every log line in [start, end]. Salad returns at most 100 a call, and when more match, which 100
        it returns isn't reliably the newest or the oldest, whatever sort_order says. So a window that comes
        back full is split in half and each half read again. None if a call fails."""
        body = {"start_time": iso(start), "end_time": iso(end), "query": query, "page_size": 100, "sort_order": "asc"}
        s, j = http("POST", f"{BASE}/log-entries", body, timeout=90)
        if s != 200:
            err(f"log-entries failed {s} {str(j)[:200]}")
            return None
        items = j.get("items") or []
        if len(items) < 100:
            return items
        if depth >= 5 or end - start < 2:
            err(f"log-entries: {iso(start)}..{iso(end)} still has 100+ lines after {depth} splits; some may be missed")
            return items
        mid = start + (end - start) / 2
        a = self.read_logs(start, mid, query, depth + 1)
        b = self.read_logs(mid, end, query, depth + 1) if a is not None else None
        return None if b is None else a + b

    def logs(self, active):
        """One query for every group: all container lines since the last successful read (2 min overlap).
        A busy window is split (see read_logs); a failed read is tried again next round, up to 30 min back."""
        st = self.st
        t = now()
        start = max(min(g["created"] for g in active) - 60, (st["last_log_end"] or 0) - 120, t - 1800)
        items = self.read_logs(start, t, 'resource.type = "container"')
        if items is None:
            ok = st.get("last_log_ok") or t
            if t - ok >= 10 * MIN and not st.get("log_warned"):
                st["log_warned"] = True
                event(f"WARN Salad log reads have failed for {(t - ok) / 60:.0f} min; PCs whose pool shares keep "
                      "coming stay, and the ledger catches up when reads work again (up to 30 min back)")
            return []
        st["last_log_end"] = st["last_log_ok"] = t
        st["log_warned"] = False
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
        ps = j.get("stats") or {}
        unc = [b for b in (j.get("unconfirmed") or []) if isinstance(b, dict)]
        try:
            self.st["pool_tot"] = {
                "t": now(), "paid": float(ps.get("paid") or 0) / 1e8, "balance": float(ps.get("balance") or 0) / 1e8,
                "unconfirmed": sum(float(b.get("reward") or 0) for b in unc) / 1e8,
                # Blocks we've been credited in: pending ones, and unlocked ones ("height:hash:..." entries).
                "blocks": len(unc) + sum(1 for x in (j.get("unlocked") or []) if isinstance(x, str) and ":" in x),
                "hashes": float(ps.get("hashes") or 0), "ths_1h": float(ps.get("hashrate_1h") or 0) * UNIT_TH}
        except (TypeError, ValueError):
            pass
        ws = j.get("workers") or []
        if isinstance(ws, dict):
            ws = [dict(v, name=k) for k, v in ws.items()]
        for w in ws:
            if isinstance(w, dict) and w.get("name"):
                res[w["name"]] = w
        return res

    def kx_pool(self, workers=True):
        """Kryptex: each worker's shares (when a box is on Kryptex), keyed by name ("<group>-g0"), and what the wallet
        holds and has been paid, at most every 5 minutes, failed tries included, so an outage costs one try in 5."""
        wallet, st, t = self.cfg["wallet"], self.st, now()
        res = {}
        if workers:
            s, j = http("GET", f"{KX_API}/v3/miner/workers/{wallet}", timeout=30)
            if s == 200 and isinstance(j, dict):
                st["kx_read_t"] = t
                for x in j.get("results") or []:
                    if isinstance(x, dict) and x.get("worker"):
                        res[x["worker"].split("/")[-1].split(".")[-1]] = x
        kt = st.get("kx_tot") or {}
        if t - max(kt.get("t", 0), st.get("kx_tot_try", 0)) < 5 * MIN:
            return res
        st["kx_tot_try"] = t
        s1, b = http("GET", f"{KX_API}/v1/miner/balance/{wallet}", timeout=30)
        s2, p = http("GET", f"{KX_API}/v1/miner/payouts/{wallet}", timeout=30)
        if s1 != 200 or not isinstance(b, dict) or s2 != 200 or not isinstance(p, dict):
            return res
        try:
            kt = {"t": t, "total": float(b.get("total") or 0), "confirmed": float(b.get("confirmed") or 0),
                  "unconfirmed": float(b.get("unconfirmed") or 0),
                  "paid": sum(float(x.get("received") or 0) for x in p.get("results") or [] if isinstance(x, dict))}
        except (TypeError, ValueError):
            return res
        st["kx_tot"] = kt
        return res

    def trader(self):
        """The SafeTrade autoseller's newest STATUS line (trader/trader.py), read from its Railway logs at most every
        5 min. Needs config "trader": {"project", "service", "environment"} (Railway ids) and a Railway token: in
        $RAILWAY_TOKEN, or added by the network proxy. A failed read keeps the last status."""
        cfg, st, t = self.cfg.get("trader"), self.st, now()
        if not cfg or t - st.get("trader_t", 0) < 300:
            return
        st["trader_t"] = t
        tok = os.environ.get("RAILWAY_TOKEN")
        # A project token goes in Project-Access-Token; an account or team token is a bearer token.
        hdrs = ({"Project-Access-Token": tok} if cfg.get("token_kind", "project") == "project" else
                {"Authorization": f"Bearer {tok}"}) if tok else {}

        def gql(query, variables):
            s, j = http("POST", RAILWAY, {"query": query, "variables": variables}, timeout=30, headers=hdrs)
            if s != 200 or not isinstance(j, dict) or j.get("errors"):
                if not st.get("trader_warned"):
                    st["trader_warned"] = True
                    event(f"WARN can't read the trader's Railway logs: HTTP {s} {str((j or {}).get('errors') or j)[:150]}")
                return None
            return j.get("data")

        d = gql("query($i: DeploymentListInput!) { deployments(input: $i, first: 5) "
                "{ edges { node { id status createdAt } } } }",
                {"i": {"projectId": cfg["project"], "serviceId": cfg["service"], "environmentId": cfg["environment"]}})
        if d is None:
            return
        for edge in (d.get("deployments") or {}).get("edges") or []:
            logs = gql("query($id: String!) { deploymentLogs(deploymentId: $id, limit: 1000) { message timestamp } }",
                       {"id": edge["node"]["id"]})
            if logs is None:
                return
            found = None
            for line in logs.get("deploymentLogs") or []:
                msg = (line.get("message") or "").strip()
                if msg.startswith("STATUS "):
                    try:
                        found = (json.loads(msg[7:]), line.get("timestamp"))
                    except ValueError:
                        pass
            if found:
                st["trader_status"] = dict(found[0], seen=found[1])
                st["trader_warned"] = False
                return

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

    def sweep(self, t):
        """Delete groups on Salad that the ledger doesn't count as live. Salad sometimes answers a delete with
        202 and keeps the group, and a create that errors can still make one. Such a group bills while nothing
        watches it and takes a replica from the quota, so new groups can't start. A stray is deleted once it
        has been seen for 10 min, which gives a normal delete time to finish. Only names starting "biz-"."""
        st = self.st
        if t - st.get("sweep_t", 0) < 5 * MIN:
            return
        st["sweep_t"] = t
        s, j = http("GET", CB)
        if s != 200:
            return
        live = {g["name"] for g in st["groups"].values() if not g["ended"]}
        names = {c.get("name") for c in (j or {}).get("items") or []}
        stray = st.setdefault("stray", {})
        for name in list(stray):
            if name not in names or name in live:
                del stray[name]
        for name in sorted(n for n in names if n and n.startswith("biz-") and n not in live):
            first = stray.setdefault(name, {"t": t, "said": False})
            if t - first["t"] < 10 * MIN:
                continue
            s, j = http("DELETE", f"{CB}/{name}")
            if not first["said"]:
                first["said"] = True
                event(f"SWEPT {name}: on Salad for {(t - first['t']) / MIN:.0f} min but not live in the ledger "
                      f"(a delete that didn't take, or a create that errored); deleted, HTTP {s}")
            elif not (200 <= s < 300 or s == 404):
                err(f"sweep delete {name} failed {s} {j}")

    def reallocate(self, g, reason, count=True, gone_ok=False):
        """Ask Salad for another PC. count=False for a PC another group already left: that move isn't the
        group's fault, so it doesn't use up one of its 3. gone_ok: an instance Salad no longer has (404) counts as
        moved, since the group starts a new one anyway (for a pool switch, which needs only a fresh start)."""
        if count and g["reallocs"] >= 3:
            # Running out of moves means bad PCs (slow, capped, blocked port), not a bad class: leave() only
            # reallocates classes that have paid. Released, so the class isn't barred from new groups.
            self.delete(g, "released", f"{reason}; already reallocated 3 times")
            return
        if not g.get("instance"):
            return False
        s, j = http("POST", f"{CB}/{g['name']}/instances/{g['instance']}/reallocate")
        if 200 <= s < 300 or (gone_ok and s == 404):
            g["reallocs"] += 1 if count else 0
            g.setdefault("left", []).append(g["machine"])
            event(f"REALLOCATED {g['name']} off PC {(g['machine'] or '')[:8]} ({reason}); "
                  + (f"reallocation {g['reallocs']} of 3" if count else "not counted toward its 3"))
            g.update(machine=None, pc_since=None, lost_since=now(), profit_fail=0, has_pc=False)
            return True
        err(f"reallocate {g['name']} failed {s} {j}")
        return False

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

    def wants_xmr(self, name):
        """Whether config asks this group to mine Monero on its CPU too."""
        gs = self.cfg["xmr_groups"]
        return bool(self.cfg["xmr_wallet"]) and ("*" in gs or name in gs)

    def box(self, name, xmr, kx=False):
        """The release, start command and CPU/RAM for a group's box. Monero needs release xmr_ver or later, and
        Kryptex kryptex_ver or later."""
        ver = self.cfg["ver_override"].get(name, self.cfg["box_ver"])
        host = self.cfg["kryptex_host"] if kx else None
        if kx and vnum(ver) < vnum(self.cfg["kryptex_ver"]):
            ver = self.cfg["kryptex_ver"]
        if not xmr:
            return ver, ["/bin/sh", "-c", cmd_for(name, self.cfg["wallet"], ver, kryptex=host)], {"cpu": 2, "memory": 4096}
        if vnum(ver) < vnum(self.cfg["xmr_ver"]):
            ver = self.cfg["xmr_ver"]
        cmd = cmd_for(name, self.cfg["wallet"], ver, self.cfg["xmr_wallet"], self.cfg["xmr_threads"], host)
        return ver, ["/bin/sh", "-c", cmd], {"cpu": self.cfg["xmr_cpu"], "memory": self.cfg["xmr_memory"]}

    def xmr_sync(self, active, t):
        """Turn Monero on or off in a running group when config asks for it. The PATCH makes a new version of the
        group, but a running instance keeps the command it started with, so the box is then restarted. It comes
        back on the new version about 2 minutes later, possibly on another PC: the first restart (13 October)
        moved the box. A refused update is tried again after 10 minutes."""
        for g in active:
            want = self.wants_xmr(g["name"])
            if g["ended"] or want == bool(g.get("xmr")) or t - g.get("xmr_try", 0) < 600:
                continue
            g["xmr_try"] = t
            ver, command, res = self.box(g["name"], want, g.get("kx"))
            s, j = http("PATCH", f"{CB}/{g['name']}", {"container": {"command": command, "resources": res}},
                        ctype="application/merge-patch+json")
            if 200 <= s < 300:
                g["xmr"], g["ver"] = want, ver
                rs = None
                if g.get("has_pc") and g.get("instance"):
                    rs, _ = http("POST", f"{CB}/{g['name']}/instances/{g['instance']}/restart")
                event(f"MONERO {'ON' if want else 'OFF'} {g['name']}: miner {ver} on {res['cpu']} vCPUs and "
                      f"{res['memory'] // 1024} GB; " + ("box restarted" if rs and 200 <= rs < 300 else
                                                         "Salad dropped its instance (the PC can't take the new size); it starts "
                                                         "that way on its next PC" if rs == 404 else
                                                         f"restart refused (HTTP {rs})" if rs else
                                                         "it starts that way on its next PC"))
            else:
                event(f"WARN Monero {'on' if want else 'off'} for {g['name']} refused: HTTP {s} {str(j)[:150]}")

    def kx_want(self):
        """How many groups config asks to have on Kryptex: kryptex_groups, or 0 once the test has stopped, until
        config kryptex_token changes."""
        st, cfg = self.st, self.cfg
        if st.setdefault("kx_token", cfg["kryptex_token"]) != cfg["kryptex_token"]:
            st["kx_token"], st["kx_fails"] = cfg["kryptex_token"], 0
        return 0 if st.get("kx_fails", 0) >= 2 else cfg["kryptex_groups"]

    def kx_sync(self, active, t):
        """Keeps kryptex_groups groups on Kryptex without moving a box that is mining. New groups start on Kryptex
        while fewer are (see create), a Kryptex group stays on Kryptex when it loses its PC, and while fewer than
        the count are on Kryptex, HeroMiners groups waiting for a PC are switched by updating their command, the
        cheapest first. Until 18 October the HeroMiners box earning least was moved over whenever the Kryptex box
        had no PC: each time Salad took back the Kryptex box's PC, a working PC was given up, every 15 minutes
        while PCs were scarce. Kryptex groups beyond the count go back to HeroMiners once they have no PC. When
        the test has stopped, Kryptex isn't taking their work, so those with a PC are moved to a new PC at once."""
        want = self.kx_want()
        stopped = self.st.get("kx_fails", 0) >= 2
        # A group whose state read failed this round (_skip) counts by its last known state, so one failed read
        # doesn't switch another group in; it is only never switched itself that round.
        live = [g for g in active if not g["ended"]]
        kx = [g for g in live if g.get("kx")]
        if len(kx) < want:
            # Not a group still preparing its image: Salad refuses an update then (pending_update_in_progress).
            waiting = sorted((g for g in live if not g.get("kx") and not g["has_pc"]
                              and g["status"] not in ("pending", "failed") and self.kx_ready(g, t)
                              and g.get("kx_failed_tok") != self.st["kx_token"]),   # failed there: not again
                             key=lambda g: g["price"])
            need = want - len(kx)
            for g in waiting:
                if need <= 0:
                    break
                if self.kx_switch(g, True, t, "waiting for a PC"):   # one Salad refuses is tried again in 15 min
                    need -= 1
        extra = len(kx) - want
        for g in sorted(kx, key=lambda g: (g["has_pc"], -(g.get("kx_since") or g["created"]))):
            if extra <= 0:
                break
            if not self.kx_ready(g, t) or (g["has_pc"] and not stopped):
                continue
            why = ("the Kryptex test stopped after 2 boxes mined no shares there" if stopped else
                   "more Kryptex groups than config asks")
            if self.kx_switch(g, False, t, why):
                extra -= 1

    def kx_works(self, g, t):
        """Whether Kryptex took another box's work in the last 30 minutes (kx_proof). A Kryptex box that sends
        nothing then points at its PC, not at Kryptex: a new PC that never sends a reading is common here (46 times
        by 18 October), and once new groups started on Kryptex, two of those would have stopped the test."""
        return any(o is not g and self.kx_proof(o, t) >= t - 30 * MIN for o in self.st["groups"].values())

    def kx_proof(self, g, t):
        """When Kryptex last took this group's work, as far as can be told: Kryptex's own count (kx_share), or, for
        the time since Kryptex's API last answered when it hasn't for 2 minutes, the miner's own count (kx_log_proof).
        So an outage of Kryptex's API alone isn't blamed on Kryptex, and one of its pool too still is."""
        p, read = g.get("kx_share") or 0, self.st.get("kx_read_t", 0)
        if t - read >= 2 * MIN:
            lp = self.kx_log_proof(g)
            if lp > read:
                p = max(p, lp)
        return p

    def kx_log_proof(self, g):
        """The last accepted share the miner's own log shows on a PC whose readings count to Kryptex (on_kx, so a PC
        seen mining on HeroMiners later doesn't count), from 5 minutes into each PC (its pc_since, or its first
        reading for a PC it has left): before that pc_pool can't have seen a box that started on HeroMiners, and a
        box that left within 5 minutes was never seen at all. A PC seen mining on Kryptex counts from the start."""
        last, acc, first, seen = 0, {}, {}, g.get("pc_pool") or {}
        for r in g["readings"]:
            tm, m, n = r[0], r[1], r[3]
            start = g.get("pc_since") if m == g.get("machine") and g.get("pc_since") else first.setdefault(m, tm)
            if n > acc.get(m, 0) and self.on_kx(g, m, tm) and (seen.get(m) is True or tm >= start + 5 * MIN):
                last = max(last, tm)
            acc[m] = n
        return last

    def on_kx(self, g, mach, t):
        """Whether what a group's box mines on PC `mach` at time `t` goes to Kryptex: the pool seen taking that
        PC's shares (pc_pool), else the pool before the group's last switch for the PC it left or a time before
        the switch, else the pool its command names."""
        seen = (g.get("pc_pool") or {}).get(mach)
        if seen is not None:
            return seen
        if g.get("sw_t") and (mach == g.get("sw_from") or t < g["sw_t"]):
            return bool(g.get("sw_prev"))
        return bool(g.get("kx"))

    def pc_pool(self, g, kx_row, hm_row):
        """Which pool a group's box mines on at its PC, by which pool has its shares. A switch by command alone can
        miss: if Salad gives a waiting group a PC as its command changes, the box can start with the old command,
        and a running box keeps the command it started with. A share at the pool the ledger doesn't expect, from 5
        minutes after the box got this PC and after its last switch (the PC it left can still send some), means the
        box mines there: this PC's readings count to that pool, and a WARN says so once. The group keeps its
        command, so its next PC starts on the pool the ledger has."""
        m, since = g.get("machine"), g.get("pc_since")
        if (not (g.get("has_pc") and m and since) or g.get("_machine") not in (None, m)
                or m in (g.get("pc_pool") or {})):
            return
        on = not g.get("kx")
        if share_t(kx_row if on else hm_row) < max(since, g.get("sw_t") or 0) + 5 * MIN:
            return
        seen = {k: v for k, v in (g.get("pc_pool") or {}).items() if any(r[1] == k for r in g["readings"])}
        g["pc_pool"] = dict(seen, **{m: on})
        a, b = ("Kryptex", "HeroMiners") if on else ("HeroMiners", "Kryptex")
        event(f"WARN {g['name']} mines Pearl on {a} on PC {m[:8]}, though its command is for {b}: it got the PC as "
              f"its command changed. This PC's readings count to {a}; its next PC starts on {b}")

    def kx_follow(self, g, j, t):
        """Keep the ledger's pool for a group the same as the command Salad has for it. An update that timed out can
        still go through (http answers 0 then), and the next PC would start on the other pool than the ledger says."""
        cmd = " ".join(str(c) for c in ((j.get("container") or {}).get("command") or []))
        if "box.sh" not in cmd or ("--pool " in cmd) == bool(g.get("kx")) or t - g.get("sw_t", 0) < 2 * MIN:
            return
        on = not g.get("kx")
        # The PC it is on, or just left, keeps the pool its readings are booked to: the box there started on the
        # command before this one.
        m = g.get("machine") or (g.get("left") or [None])[-1]
        g.update(kx=on, kx_since=t if on else None, sw_t=t, sw_from=m, sw_prev=self.on_kx(g, m, t),
                 kx_share=None if on else g.get("kx_share"))
        self.st["kx_any_t"] = t
        event(f"WARN {g['name']}: Salad has the {'Kryptex' if on else 'HeroMiners'} command for it, though the ledger "
              f"said {'HeroMiners' if on else 'Kryptex'} (an update that timed out went through); the ledger follows "
              "Salad")

    def kx_ready(self, g, t):
        """Whether a group can be switched this round: its state was read, it didn't just change PC (rules() hasn't
        seen the new one yet, so the move would name the old PC), and Salad didn't refuse its last move lately."""
        if g.get("_skip") or (g["has_pc"] and g.get("_machine") not in (None, g.get("machine"))):
            return False
        return t - g.get("kx_try", 0) >= 15 * MIN

    def kx_switch(self, g, on, t, why):
        """Point a group's box at Kryptex (on) or back at HeroMiners: update its command, then, if it has a PC, move
        it to a new PC, where it starts with the new command. Salad's instance restart doesn't do it: on 17 October
        it answered 202 twice and the box kept running its old command. The move isn't counted toward the group's
        3, and the PC it leaves doesn't go on the bad list."""
        ver, command, _ = self.box(g["name"], bool(g.get("xmr")), on)
        s, j = http("PATCH", f"{CB}/{g['name']}", {"container": {"command": command}},
                    ctype="application/merge-patch+json")
        if not 200 <= s < 300:
            g["kx_try"] = t   # tried again in 15 minutes
            event(f"WARN Kryptex {'on' if on else 'off'} for {g['name']} refused: HTTP {s} {str(j)[:150]}")
            return False
        # A box seen mining on the target pool already (pc_pool) stays on its PC.
        moving = bool(g["has_pc"] and g.get("instance")) and self.on_kx(g, g.get("machine"), t) != on
        if moving and not self.reallocate(g, f"to start on {'Kryptex' if on else 'HeroMiners'}", count=False, gone_ok=True):
            # Still on its PC with the command it started with: put the group's command back to match, and try again
            # in 15 minutes.
            g["kx_try"] = t
            _, back, _ = self.box(g["name"], bool(g.get("xmr")), bool(g.get("kx")))
            sb, _ = http("PATCH", f"{CB}/{g['name']}", {"container": {"command": back}},
                         ctype="application/merge-patch+json")
            if 200 <= sb < 300:
                event(f"WARN Kryptex {'on' if on else 'off'} for {g['name']}: Salad refused to move it to a new PC; "
                      "it stays where it was")
                return False
            # Salad keeps the new command, so the box starts on the new pool at its next PC: the ledger follows,
            # and this PC's readings stay with the old pool.
            moving = False
            event(f"WARN Kryptex {'on' if on else 'off'} for {g['name']}: Salad refused to move it and to put its "
                  f"command back (HTTP {sb}); it mines on {'HeroMiners' if on else 'Kryptex'} here and starts on "
                  f"{'Kryptex' if on else 'HeroMiners'} at its next PC")
        # Readings the box still sends from the PC it left or stays on, or from before now, count to the pool that
        # box mines on: it keeps the command it started with. A group without a PC can have just left one (rules()
        # reallocated it), whose box may still send a line.
        m = (g.get("machine") if not moving else None) or (g.get("left") or [None])[-1]
        prev = self.on_kx(g, m, t)
        already = bool(g["has_pc"]) and not moving and prev == on
        g.update(kx=on, ver=ver, kx_since=t if on else None, sw_t=t, sw_from=m, sw_prev=prev,
                 kx_share=None if on else g.get("kx_share"))
        self.st["kx_any_t"] = t
        event(f"KRYPTEX {'ON' if on else 'OFF'} {g['name']} ({why}): Pearl on {'Kryptex' if on else 'HeroMiners'}; "
              + ("moved to a new PC, where it starts that way" if moving else
                 "it mines that way on this PC already" if already else "it starts that way on its next PC"))
        return True

    def kx_failed(self, g, why):
        """A Kryptex box that sends no shares or no readings while no other box's work reaches Kryptex points at
        Kryptex, not at the PC: it goes back to HeroMiners on a new PC, and the PC stays off the bad list. Two of
        these stop the test."""
        st, t = self.st, now()
        if not self.kx_switch(g, False, t, f"on Kryptex: {why}"):
            return False
        st["kx_fails"] = st.get("kx_fails", 0) + 1
        g["kx_failed_tok"] = st.get("kx_token")       # not this one again until kryptex_token changes
        if st["kx_fails"] == 2:
            event("WARN Kryptex test stopped: 2 Kryptex boxes mined no shares. Boxes stay on HeroMiners until "
                  "config kryptex_token changes")
        return True

    def xmr_market(self):
        """Monero's price and what one accepted hash earns at the pool (the network's reward per hash, less the
        pool's fee), at most every 5 minutes."""
        st = self.st
        if st.get("xmr_mkt") and now() - st["xmr_mkt"]["t"] < 300:
            return
        s1, j = http("GET", "https://monero.herominers.com/api/stats", timeout=30)
        s2, p = http("GET", "https://api.coingecko.com/api/v3/simple/price?ids=monero&vs_currencies=usd", timeout=30)
        try:
            per_hash = float(j["lastblock"]["reward"]) / 1e12 / float(j["network"]["difficulty"]) * (1 - XMR_FEE)
            price = float(p["monero"]["usd"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            err(f"Monero market fetch failed {s1} {s2}")
            return
        st["xmr_mkt"] = {"t": now(), "price": price, "per_hash": per_hash, "usd_khs_hr": per_hash * 1000 * 3600 * price}

    def xmr_pool(self, groups, t):
        """Each Monero box's CPU rate and the XMR it has earned, from its worker at the pool ("<group>-g0"). The
        pool counts every accepted hash, so the XMR earned is the new hashes times what a hash earns."""
        st, mk = self.st, self.st.get("xmr_mkt")
        s, j = http("GET", f"https://monero.herominers.com/api/stats_address?address={self.cfg['xmr_wallet']}"
                           "&longpoll=false", timeout=30)
        if s != 200 or not isinstance(j, dict) or not mk:
            return
        ps = j.get("stats") or {}
        try:
            unc = [b for b in (j.get("unconfirmed") or []) if isinstance(b, dict)]
            st["xmr_tot"] = {"balance": float(ps.get("balance") or 0) / 1e12, "paid": float(ps.get("paid") or 0) / 1e12,
                             "pending": sum(float(b.get("reward") or 0) for b in unc) / 1e12,
                             "khs_1h": float(ps.get("hashrate_1h") or 0) / 1000}
        except (TypeError, ValueError):
            pass
        ws = {w["name"]: w for w in (j.get("workers") or []) if isinstance(w, dict) and w.get("name")}
        for g in groups:
            w = ws.get(f"{g['name']}-g0")
            if not w:
                continue
            try:
                hashes, khs = float(w.get("hashes") or 0), float(w.get("hashrate") or 0) / 1000
            except (TypeError, ValueError):
                continue
            last = g.get("xmr_hashes", 0.0)
            g["xmr_coins"] = g.get("xmr_coins", 0.0) + (hashes - last if hashes >= last else hashes) * mk["per_hash"]
            g["xmr_hashes"], g["xmr_khs"], g["xmr_seen"] = hashes, khs, t

    def create(self, cls, pri, taken=(), tries=0):
        """Create a group of `cls` at `pri`. Salad keeps a deleted group's name for a while (name_conflict),
        so a name it refuses is skipped and the next suffix tried, up to 20 refused names. Every name in the
        ledger, live or deleted, is skipped without asking and doesn't count toward the 20: a deleted group's
        name is often still held, and after 20 deleted 4080 groups the retries ran out before a free name."""
        st = self.st
        taken = {g["name"] for g in st["groups"].values()} | set(taken)
        base = "biz-" + slug(cls)
        name, i = base, 2
        while name in taken:
            name, i = f"{base}-{i}", i + 1
        xmr = self.wants_xmr(name)
        # On Kryptex while fewer groups than config asks are (see kx_sync).
        kx = sum(1 for g in st["groups"].values() if not g["ended"] and g.get("kx")) < self.kx_want()
        ver, command, res = self.box(name, xmr, kx)
        res.update(storage_amount=10737418240, gpu_classes=[st["classes"][cls]["id"]])
        body = {"name": name, "display_name": name, "replicas": 1, "restart_policy": "always", "autostart_policy": True,
                "container": {"image": IMAGE, "priority": pri, "resources": res, "command": command}}
        s, j = http("POST", CB, body)
        if 200 <= s < 300:
            rate, paid = round(self.rate(cls)), KNOWN[cls][1]
            t = now()
            key = f"{name}@{int(t)}"
            st["groups"][key] = {
                "key": key, "name": name, "class": cls, "rate": rate, "paid": paid, "pri": pri,
                "price": st["classes"][cls]["prices"][pri], "created": t, "pri_since": t, "ver": ver,
                "had_pc": False, "has_pc": False, "pc_since": None, "machine": None, "instance": None, "lost_since": None,
                "reallocs": 0, "spend": 0.0, "prl": 0.0, "pc_min": 0.0, "mined_min": 0, "readings": [],
                "acc": {}, "last_share": None, "profit_fail": 0, "last_check": 0, "status": "pending",
                "ended": None, "end_kind": None, "end_reason": None, "blocked": [], "cards": [], "pool_ths": None,
                "last_start": 0, "first_share_said": False, "xmr": xmr, "kx": kx, "kx_since": t if kx else None}
            event(f"CREATED {name}: {cls} at {pri} (${st['classes'][cls]['prices'][pri]:.3f}/hr, "
                  f"{self.margin(cls, pri) * 100:+.0f}% margin at {rate} TH/s), miner {ver}"
                  + (f", Monero on {res['cpu']} vCPUs" if xmr else "") + (", Pearl on Kryptex" if kx else ""))
            return True
        text = json.dumps(j)
        if "name_conflict" in text and tries < 20:
            return self.create(cls, pri, taken | {name}, tries + 1)
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
        bad = st.setdefault("bad_pcs", {})
        for m in [m for m, b in bad.items() if t - b["t"] >= 3600 * self.cfg["bad_pc_hold_h"]]:
            del bad[m]
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
                g["_skip"] = True   # no rule or switch acts on last round's state
                continue
            st["api_fail"] = 0
            self.kx_follow(g, j, t)
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

        self.sweep(t)

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
                m = re.search(r"\[mining g(\d+)\] \d\d:\d\d:\d\d\s+([0-9.]+) TH/s · ([0-9,]+) accepted(?:.*?up (\S+ \S+))?", text)
                if m:
                    ths, acc, up = float(m.group(2)), int(m.group(3).replace(",", "")), m.group(4)
                    # box.sh repeats the miner's last line after the miner dies. Its uptime ("up 1h 23m") then
                    # stops moving; a live miner's moves every minute, though its rate and share count can repeat.
                    # A repeat isn't a reading: it earns nothing, and the stale-reading rule sees the miner stopped.
                    same = [r for r in g["readings"] if r[1] == mach]
                    if up and same and len(same[-1]) > 4 and same[-1][4] == up:
                        continue
                    g["readings"].append([tm, mach, ths, acc, up])
                    g["readings"] = [r for r in g["readings"] if t - r[0] < 7200]
                    # The pool this reading was mined on. One from the PC a box left at its last switch, or from
                    # before the switch, was mined on the pool it was on before.
                    on_kx = self.on_kx(g, mach, tm)
                    if ths > 0:
                        g["prl"] += ths / 60 * st["mkt"]["prl_th_hr"]
                        if on_kx:   # the part mined on Kryptex, which has its own credit ratio
                            g["prl_kx"] = g.get("prl_kx", 0.0) + ths / 60 * st["mkt"]["prl_th_hr"]
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
                elif g.get("kx") and text.startswith("[minerlog"):
                    # A Kryptex box's first miner lines show whether Kryptex took its login.
                    event(f"LOG {name} PC {(mach or '')[:8]}: {text.strip()[:200]}")

        # Pool: shares and pool-side hashrate.
        pw = self.pool() if by_name else {}
        # Workers while a group is on Kryptex, was switched in the last 2 hours (its box may mine there: pc_pool), or
        # has a box whose PC mines there.
        live_kx = any(g.get("kx") or t - (g.get("sw_t") or 0) < 2 * 3600
                      or (g["has_pc"] and self.on_kx(g, g.get("machine"), t)) for g in by_name.values())
        kw = (self.kx_pool(workers=live_kx) if live_kx or any(g.get("prl_kx") for g in st["groups"].values())
              else {})
        self.trader()
        for g in by_name.values():
            x, w = kw.get(f"{g['name']}-g0"), pw.get(f"{g['name']}-g0")
            self.pc_pool(g, x, w)
            ls = share_t(x)
            if ls >= g["created"]:
                # Kryptex's own count is the proof it takes a box's work (see leave and kx_works): the miner's log
                # can't tell which pool accepted a share.
                g["kx_share"] = max(g.get("kx_share") or 0, ls)
            # The hashrate is from the row of the pool it mines on now, or none: a box that just switched has no row
            # there yet, and the other pool's figure would show under this one.
            g["pool_ths"] = None
            if self.on_kx(g, g.get("machine"), t):   # the pool it mines on now: last share and hashrate from there
                if x:
                    if ls >= g["created"]:
                        g["last_share"] = max(g["last_share"] or 0, ls)
                    try:
                        # The live rows have no "hashrate" (openapi.yaml lists one): avg_hashrate_30m, a string in H/s.
                        g["pool_ths"] = float(x.get("avg_hashrate_30m") or 0) / 1e12
                    except (TypeError, ValueError):
                        pass
                continue
            if not w:
                continue
            ls = share_t(w)
            if ls >= g["created"]:
                g["last_share"] = max(g["last_share"] or 0, ls)
            try:
                g["pool_ths"] = float(w.get("hashrate") or 0) * UNIT_TH
            except (TypeError, ValueError):
                pass
        # Monero: switch groups on or off as config asks, then read the CPU workers at the pool.
        self.xmr_sync(list(by_name.values()), t)
        self.kx_sync(list(by_name.values()), t)
        xg = [g for g in by_name.values() if g.get("xmr")]
        if xg:
            self.xmr_market()
            self.xmr_pool(xg, t)
        for g in by_name.values():
            if g["last_share"] and not g["first_share_said"]:
                g["first_share_said"] = True
                event(f"FIRST SHARE {g['name']} at {hm(g['last_share'])}Z")

        # Rules.
        for g in by_name.values():
            if g.get("_skip"):
                continue
            self.rules(g, t)

        self.record_rates(t)
        self.critic(t)

        # Hourly search at :01, or when asked through config.search_token.
        tm = time.gmtime(t)
        hour = time.strftime("%Y-%m-%dT%H", tm)
        asked = self.cfg["search_token"] != st["search_token"]
        # Also every search_every_min while a slot is empty: waiting for a PC costs nothing.
        slots = min(self.cfg["max_active"], st.get("quota") or self.cfg["max_active"])
        empty = sum(1 for g in st["groups"].values() if not g["ended"]) < slots
        often = empty and t - st.get("last_search", 0) >= 60 * self.cfg["search_every_min"]
        if self.cfg["search"] and not st["credits_out"] and (asked or often or (tm.tm_min >= 1 and st["search_hour"] != hour)):
            st["search_hour"], st["search_token"], st["last_search"] = hour, self.cfg["search_token"], t
            self.search()
        # With every slot taken, every upgrade_every_min, see whether a better class should take the weakest one's.
        elif self.cfg["search"] and not empty and t - st.get("last_upgrade", 0) >= 60 * self.cfg["upgrade_every_min"]:
            self.upgrade(t)

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
                # A new box there starts on the group's command, whatever the box before it mined on.
                (g.get("pc_pool") or {}).pop(mach, None)
                if g.get("sw_from") == mach:
                    g["sw_from"] = None
                event(f"{'GOT PC' if first else 'NEW PC'} {g['name']} ({g['class']}, {g['pri']}): PC {mach[:8]}")
                # A PC that another group left lately (slow, capped, not mining) is left again at once.
                bad = st.setdefault("bad_pcs", {}).get(mach)
                if bad and t - bad["t"] < 3600 * self.cfg["bad_pc_hold_h"]:
                    self.reallocate(g, f"{bad['group']} left this PC at {hm(bad['t'])}Z: {bad['why']}", count=False)
                    return
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
        # Readings come from Salad's log service, which sometimes fails for minutes at a time. Missing readings
        # then say nothing about the miner, so these two rules count only up to the last log read that worked,
        # and leave nothing while the pool still gets shares from this PC (in the last 6 min).
        seen_to = min(t, st.get("last_log_ok") or t)
        ls = g["last_share"] or 0
        sharing = ls >= g["pc_since"] and t - ls < 6 * MIN
        cur = [r for r in g["readings"] if r[1] == g["machine"] and r[2] > 0 and r[0] >= g["pc_since"] - 5 * MIN]
        if not cur and seen_to - g["pc_since"] >= 10 * MIN and not sharing:
            self.leave(g, f"no mining reading {on / 60:.0f} min after getting PC {g['machine'][:8]}")
            return
        # A miner that stops (the card is taken back, the driver fails) stops sending readings: leave the PC
        # once the newest is 6 min old.
        if cur and seen_to - cur[-1][0] >= 6 * MIN and not sharing:
            self.leave(g, f"no mining reading for {(t - cur[-1][0]) / 60:.0f} min on PC {g['machine'][:8]}")
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
            usd, bar = self.usd(), self.cfg["min_margin"]
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

    def critic(self, t):
        """Once an hour, checks the ledger against the pool and writes a DRIFT event when they disagree by more
        than `drift` (15%): the miners' reported hashrate against what the pool saw in the last hour; the ledger's
        PRL against what the pool's count of our work is worth; and what the pool has credited against that work,
        when the gap is more than its luck explains."""
        st, pt = self.st, self.st.get("pool_tot")
        hour = int(t // 3600)
        if not pt or t - pt["t"] > 300 or st.get("critic_hour") == hour:
            return
        st["critic_hour"] = hour
        bar = self.cfg["drift"]
        # Hashrate. Readings are one a minute a group, so their sum over an hour / 60 is the average TH/s.
        # After a switch, an hour of readings mixes both pools, so the check waits an hour.
        if t - st["created"] >= 3600 and t - st.get("kx_any_t", 0) >= 3600:
            miner = sum(r[2] for g in st["groups"].values() for r in g["readings"]
                        if r[0] > t - 3600 and not self.on_kx(g, r[1], r[0])) / 60
            if miner > 0 and abs(pt["ths_1h"] / miner - 1) > bar:
                event(f"DRIFT hashrate: the pool saw {pt['ths_1h']:.0f} TH/s over the last hour; the miners reported "
                      f"{miner:.0f} ({pct(pt['ths_1h'] - miner, miner)})")
        # Work: the pool's count of our hashes, at today's PRL per TH-hr (the ledger used the rate at each minute).
        prl = sum(g["prl"] - g.get("prl_kx", 0.0) for g in st["groups"].values())   # what was mined on HeroMiners
        work = pt["hashes"] * UNIT_TH / 3600 * st["mkt"]["prl_th_hr"]
        if prl >= 1 and abs(work / prl - 1) > bar:
            event(f"DRIFT ledger: the pool's count of our work is worth {work:.3f} PRL; the ledger says {prl:.3f} "
                  f"({pct(work - prl, prl)})")
        # Credit: the pool pays a share of each block it finds, so with few blocks it runs well above or below
        # the work. Two standard deviations of that luck is about 2 / sqrt(blocks).
        cred = pt["paid"] + pt["balance"] + pt["unconfirmed"]
        luck = 2 / pt["blocks"] ** 0.5 if pt["blocks"] else 1.0
        if work >= 1 and cred / work < 1 - max(bar, luck):
            event(f"DRIFT credit: the pool has credited {cred:.3f} PRL for work worth {work:.3f} "
                  f"({pct(cred - work, work)}, more than luck explains over {pt['blocks']} blocks)")

    def pc_rate(self, g):
        """The miner's average TH/s on this PC's last 3 readings after warm-up, or None while there aren't 3."""
        if not g.get("has_pc") or not g.get("machine") or not g.get("pc_since"):
            return None
        cur = [r for r in g["readings"] if r[1] == g["machine"] and r[2] > 0 and r[0] >= g["pc_since"] - 5 * MIN]
        post = [r for r in cur if r[0] >= cur[0][0] + 2 * MIN] if cur else []
        if len(post) < 3:
            return None
        return sum(r[2] for r in post[-3:]) / 3

    def pc_margin(self, g):
        """The margin on this PC's last 3 readings after warm-up, or None while there aren't 3."""
        rate = self.pc_rate(g)
        return None if rate is None else margin_of(rate * self.usd(), g["price"])

    def unplaced(self, cls, t):
        """True when the class's last no_pc_streak groups all ended without ever getting a PC, the latest within
        no_pc_hold_min: the search then tries it after every other class, so it stops taking a slot it can't fill."""
        ended = sorted((g for g in self.st["groups"].values() if g["class"] == cls and g["ended"]),
                       key=lambda g: g["ended"], reverse=True)[:self.cfg["no_pc_streak"]]
        return (len(ended) == self.cfg["no_pc_streak"] and all(g["end_kind"] == "released" and not g["had_pc"] for g in ended)
                and t - ended[0]["ended"] < 60 * self.cfg["no_pc_hold_min"])

    def leave(self, g, why):
        """Stop paying for this PC: another PC if the class has paid before, otherwise delete the group. The PC goes
        on the bad list, so another group that gets it leaves it at once (for bad_pc_hold_h)."""
        # A Kryptex box that has had Kryptex shares and then stops, stops while other boxes' work reaches Kryptex,
        # or mines on HeroMiners at this PC (pc_pool) has a PC problem: it is left as usual below.
        if (g.get("kx") and why.startswith(("no share", "no mining reading"))
                and self.kx_proof(g, now()) < (g.get("kx_since") or g["created"]) and not self.kx_works(g, now())
                and (g.get("pc_pool") or {}).get(g.get("machine")) is not False and self.kx_failed(g, why)):
            return
        if g.get("machine"):
            self.st.setdefault("bad_pcs", {})[g["machine"]] = {"t": now(), "group": g["name"], "why": why}
        if self.paid(g):
            self.reallocate(g, why)
        else:
            self.delete(g, "failed", why)

    def search(self):
        st = self.st
        active = [g for g in st["groups"].values() if not g["ended"]]
        s, q = http("GET", f"{BASE}/quotas")
        quota = (q or {}).get("container_groups_quotas", {}).get("container_replicas_quota", 10) if s == 200 else 10
        st["quota"] = quota
        room = min(self.cfg["max_active"], quota) - len(active)
        t = now()
        this_hour = sum(1 for g in st["groups"].values() if g["created"] >= t - t % 3600)
        room = min(room, self.cfg["max_new_per_hour"] - this_hour)   # however the search was started
        if room <= 0:
            return
        made = 0
        for *_, cls, p in self.candidates(active, t):
            if made >= room or st["credits_out"]:
                break
            if self.create(cls, p):
                made += 1
        if not made:
            event(f"SEARCH found nothing to create (room {room})")

    def candidates(self, active, t):
        """Classes a new group may use, each at its cheapest free priority from start_pri that clears the margin, best first:
        classes not held for want of PCs, then paid classes, then by expected profit an hour."""
        st = self.st
        # Each class is tried once on this account, as on the first one, except that a class whose groups only
        # ended for want of a PC (released) is tried again retry_after_min after.
        skip = {g["class"] for g in st["groups"].values()
                if not g["ended"] or g["end_kind"] != "released" or t - g["ended"] < 60 * self.cfg["retry_after_min"]}
        # A class whose groups are all on a PC clearing the margin may get another group, up to max_per_class.
        for cls in {g["class"] for g in active}:
            mine = [g for g in active if g["class"] == cls]
            if len(mine) < self.cfg["max_per_class"] and all((self.pc_margin(g) or -1) >= self.cfg["min_margin"] for g in mine) \
                    and not any(g["ended"] and g["end_kind"] == "failed" for g in st["groups"].values() if g["class"] == cls):
                skip.discard(cls)
        cands = []
        for cls in KNOWN:
            if cls not in st["classes"] or cls in skip:
                continue
            rate = self.rate(cls)
            body = {"cpu": 2, "memory": 4096, "storage_amount": 10737418240, "gpu_classes": [st["classes"][cls]["id"]]}
            s, a = http("POST", f"{BASE}/availability/sce-gpu-availability", body)
            if s != 200:
                continue
            for p in PRIS[PRIS.index(self.start_pri(cls)):]:
                if (a.get(f"available_gpu_{p}") or 0) > 0 and self.margin(cls, p) >= self.cfg["min_margin"]:
                    profit = rate * self.usd() - st["classes"][cls]["prices"][p]
                    cands.append((self.unplaced(cls, t), not self.paid(cls), -profit, cls, p))
                    break
        return sorted(cands)

    def start_pri(self, cls):
        """The cheapest priority a new group of `cls` starts at. Batch, unless the class's latest ended group never
        got a PC: then the priority its latest group with a PC got it at. On 14 October 4080s and 5080s only got
        PCs at medium or high, and each new group spent 30 min climbing from batch first."""
        gs = [g for g in self.st["groups"].values() if g["class"] == cls]
        ended = max((g for g in gs if g["ended"]), key=lambda g: g["ended"], default=None)
        placed = max((g for g in gs if g["had_pc"]), key=lambda g: g["created"], default=None)
        if ended is None or ended["had_pc"] or placed is None:
            return "batch"
        return placed["pri"]

    def upgrade(self, t):
        """With every slot taken, swap the group making the least for a class expected to make upgrade_gain $/hr
        more, after the cost of switching. Only classes that have had a PC here, and not one whose latest group
        got none. A class is expected to make its measured rate at the priority its latest group got a PC at,
        or its free priority if that's higher, and never cheaper than low: batch PCs are seldom handed out.
        Switching costs the old box's profit for swap_wait_min (getting a PC, starting the miner) and the new
        box's bill for its first START_MIN minutes; that is taken off the first hour's gain. One swap per check."""
        st = self.st
        st["last_upgrade"] = t
        this_hour = sum(1 for g in st["groups"].values() if g["created"] >= t - t % 3600)
        if self.cfg["upgrade_gain"] is None or st["credits_out"] or this_hour >= self.cfg["max_new_per_hour"]:
            return
        u = self.usd()
        active = [g for g in st["groups"].values() if not g["ended"]]
        mining = [(self.pc_rate(g) * u - g["price"], g) for g in active if self.pc_rate(g) is not None]
        if not mining:
            return
        worst, g = min(mining, key=lambda x: x[0])
        # Each class's latest group, and the priority its latest group with a PC got it at.
        latest, placed = {}, {}
        for x in sorted(st["groups"].values(), key=lambda x: x["created"]):
            latest[x["class"]] = x
            if x["had_pc"]:
                placed[x["class"]] = x["pri"]
        best = None
        for unplaced, _, _, cls, p in self.candidates(active, t):
            # Only a class Salad has placed for us, and not while its latest try got no PC.
            if unplaced or cls == g["class"] or cls not in placed or (latest[cls]["ended"] and not latest[cls]["had_pc"]):
                continue
            price = st["classes"][cls]["prices"][PRIS[max(PRIS.index(p), PRIS.index(placed[cls]), 1)]]
            expect = self.rate(cls) * u - price
            switch = max(worst, 0) * self.cfg["swap_wait_min"] / 60 + price * START_MIN / 60
            if best is None or expect - switch > best[0] - best[1]:
                best = (expect, switch, cls, p)
        if best is None or best[0] - best[1] - worst < self.cfg["upgrade_gain"]:
            return
        expect, switch, cls, p = best
        event(f"UPGRADE {g['name']} makes ${worst:+.3f}/hr on its PC; a {cls} should make ${expect:+.3f}/hr "
              f"(switching costs about ${switch:.3f})")
        self.delete(g, "released", f"swapped for a {cls}, expected to make ${expect - worst:.3f}/hr more")
        if g["ended"]:
            self.create(cls, p)

    # ---------- report ----------
    def write_report(self):
        """One table: a row per box, a total, then where the profit stands (locked in, not yet locked, Monero),
        with the details in two lines under it. Every row fits on one line in a chat window."""
        st, t = self.st, now()
        mk = st["mkt"]
        # The table counts on what the pool credits: the network's rate times its credit ratio. The rules may
        # use a different yield (config "yield"); the table shows what we expect to be paid.
        hy, ky = self.credit_ratio(), self.kx_ratio()
        yg = lambda g: ky if g.get("kx") else hy   # each box at its own pool's credit ratio
        price = mk["price"]
        # Monero on the CPU: its earnings count in each box's profit, once config has an address.
        xon, xm = bool(self.cfg["xmr_wallet"]), st.get("xmr_mkt") or {}
        xusd, xprice = xm.get("usd_khs_hr", 0.0), xm.get("price", 0.0)
        groups = list(st["groups"].values())
        spend = sum(g["spend"] for g in groups)
        est = sum(g["prl"] for g in groups)
        prl = sum(self.credited(g, hy, ky) for g in groups)
        y = prl / est if est > 0 else hy
        cost_prl = spend / prl if prl > 0 else 0.0
        coins = sum(g.get("xmr_coins", 0.0) for g in groups)
        # SafeTrade from the trader's logs, or from config "safetrade" (entered by hand) until those can be read.
        ts = st.get("trader_status") or self.cfg.get("safetrade")
        sold = (ts or {}).get("sold", 0)
        net = (ts or {}).get("received", 0) - (ts or {}).get("fees", 0)
        h = "" if (ts or {}).get("t") else EST  # entered by hand: we can't check it
        unsold = prl - sold
        # A PRL is worth what it sold for if it sold, today's price if not, so the boxes' Profit $ adds up to
        # locked in + not yet locked + Monero instead of valuing sold PRL at today's price.
        per_prl = (net + max(unsold, 0) * price) / prl if ts and prl > 0 else price

        rows = []
        all_ths = all_cost = all_earn = run_cost = all_khs = 0.0
        mining = 0
        live = sorted((g for g in groups if not g["ended"]), key=lambda g: g["created"])
        for g in live:
            cur = [r for r in g["readings"] if r[1] == g["machine"] and r[2] > 0 and r[0] >= t - 10 * MIN] if g["has_pc"] else []
            ths = sum(r[2] for r in cur) / len(cur) if cur else 0.0
            cost = g["price"] if g["has_pc"] else 0.0
            khs = g.get("xmr_khs", 0.0) if g.get("xmr") and g["has_pc"] and t - g.get("xmr_seen", 0) < 15 * MIN else 0.0
            earn = ths * mk["usd_th_hr"] * yg(g) + khs * xusd
            state = "⏹️" if g["status"] == "stopped" else "⛏️" if g["has_pc"] else "⏳"
            # A box on a new PC with no reading yet is starting: billed, but left out of the profit total.
            start = bool(g["has_pc"] and not cur and g.get("pc_since") and t - g["pc_since"] < 15 * MIN)
            if g["has_pc"] and cur:
                mining += 1
            all_ths += ths; all_cost += cost; all_earn += earn; all_khs += khs
            run_cost += 0.0 if start else cost
            prof = self.credited(g, hy, ky) * per_prl + g.get("xmr_coins", 0.0) * xprice - g["spend"]
            # One cell for the box: state, group name less "biz-rtx" and priority, joined by no-break spaces.
            pri = "med" if g["pri"] == "medium" else g["pri"]
            box = NB.join([state, re.sub(r"^biz-(rtx)?", "", g["name"]), pri] + (["(K)"] if g.get("kx") else []))
            # Each coin's speed (Pearl on the GPU, Monero on the CPU), then the bill, profit and margin.
            kc = [f"{khs:.1f}" if g.get("xmr") else "—"] if xon else []
            if not g["has_pc"]:
                cells = ["—"] + (["—"] if xon else []) + ["—", "—"]
            elif start:
                cells = ["0"] + kc + [f"{cost:.2f}", "—"]
            else:
                cells = [f"{ths:.0f}"] + kc + [f"{cost:.2f}", with_margin(earn - cost, earn)]
            cells = " | ".join(cells)
            rows.append((0.0 if start or not g["has_pc"] else earn - cost, f"| {box} | {cells} | {prof:+.2f}{EST} |"))
        # Most profitable an hour first; groups waiting for a PC make nothing, so they sit above any losing.
        lines = [r for _, r in sorted(rows, key=lambda x: -x[0])]
        ended = [g for g in groups if g["ended"]]
        if ended:
            # Groups that have ended, as one row with their count; the events say which were released or failed.
            eprof = (sum(self.credited(g, hy, ky) for g in ended) * per_prl + sum(g.get("xmr_coins", 0.0) for g in ended) * xprice
                     - sum(g["spend"] for g in ended))
            lines.append(f"| Others{NB}×{len(ended)} |"
                         + " — |" * (4 if xon else 3) + f" {eprof:+.2f}{EST} |")
        total = prl * per_prl + coins * xprice - spend
        hr_all = all_earn - run_cost
        lines.append(f"| **All** | {bold(f'{all_ths:.0f}')} |" + (f" {bold(f'{all_khs:.1f}')} |" if xon else "")
                     + f" {bold(f'{all_cost:.2f}')} | {bold(with_margin(hr_all, all_earn))} | {bold(f'{total:+.2f}{EST}')} |")
        # What each pool reports for the account: its own hashrate (1-hour average) and the coins it holds and paid.
        pt, xt = st.get("pool_tot"), st.get("xmr_tot") if xon else None
        # Kryptex shows once a live box is on it or some PRL was mined there; an ended group that never mined
        # there (its switch failed) doesn't count.
        kt = st.get("kx_tot") if any((g.get("kx") and not g["ended"]) or g.get("prl_kx") for g in groups) else None
        pool = []
        if pt:
            pool.append(f"| ⛏️ PRL{NB + 'HeroMiners' if kt else ''} | {pt.get('ths_1h', 0):.0f} TH/s | {pt['balance']:.3f} "
                        f"| {pt['unconfirmed']:.3f} | {pt['paid']:.3f} |")
        if kt:
            # Kryptex gives each worker's hashrate now, not an account average: the sum over our Kryptex boxes.
            kths = sum(g.get("pool_ths") or 0 for g in live if g["has_pc"] and self.on_kx(g, g.get("machine"), now()))
            pool.append(f"| ⛏️ PRL{NB}Kryptex | {kths:.0f} TH/s | {kt['confirmed']:.3f} | {kt['unconfirmed']:.3f} "
                        f"| {kt['paid']:.3f} |")
        if xt:
            pool.append(f"| 🪙 XMR | {xt.get('khs_1h', 0):.1f} kH/s | {xt['balance']:.6f} | {xt.get('pending', 0):.6f} "
                        f"| {xt['paid']:.6f} |")
        if pool:
            pool = ["", "| Pool | Hashrate (1h) | Confirmed | Pending | Paid |", "|---|---|---|---|---|", *pool]

        n_live = len(live)
        # Just what the table doesn't show: profit an hour, margin and profit so far are in its All row.
        head = f"**{stamp(t)}** · PRL ${price:.3f} · {mining}/{n_live} mining"
        credit = self.cfg["credit_start"] + self.cfg["credit_offset"] - spend
        hours = credit / all_cost if all_cost > 0 else None
        # What doesn't fit on a row, in two lines: the key, then the money trail.
        key = ("⛏️ mining · ⏳ waiting for a PC · ⏹️ stopped · * our own count or estimate, or entered by hand; the rest "
               "comes from an API (Salad, the pool, prlscan, CoinGecko) or the trader's log. TH/s is Pearl on the GPU"
               + (", kH/s Monero on the CPU" if xon else "") + "; $/hr is Salad's bill. Profit $ counts sold PRL at what "
               "it sold for and the rest at today's price" + (", plus Monero" if xon else "") + "."
               + (" (K) mines Pearl on Kryptex (PPS+), the rest on HeroMiners." if kt else ""))
        trail = [f"Spent ${spend:.2f}{EST} on Salad"]
        if prl > 0:
            trail[0] += (f" for {prl:.3f}{EST} PRL" + (f" ({est:.3f}{EST} at the network's rate × {y:.0%}{EST} credited)"
                                                        if y < 1 else "") + f", ${cost_prl:.3f}{EST} each")
        if pt:
            cred = pt["paid"] + pt["balance"] + pt["unconfirmed"]
            hest = sum(g["prl"] - g.get("prl_kx", 0.0) for g in groups)
            trail.append(f"the PRL pool{' HeroMiners' if kt else ''} has credited {cred / hest * 100 if hest > 0 else 0:.0f}%{EST} of our "
                         f"estimate over {pt['blocks']} blocks, and pays out from 1 PRL"
                         + (f" (from {self.cfg['yield_min_blocks']} blocks on the table counts on that share)"
                            if hy == 1 and pt["blocks"] < self.cfg["yield_min_blocks"] else ""))
        if kt:
            kest = sum(g.get("prl_kx", 0.0) for g in groups)
            trail.append(f"Kryptex has credited {kt['total'] + kt['paid']:.3f} PRL for {kest:.3f}{EST} at the network's "
                         f"rate ({ky:.0%}{EST}; {self.cfg['kryptex_pay']:.0%} expected), "
                         "and pays out from 1 PRL")
        if ts:
            when = (f"at the {time.strftime('%H:%M', time.gmtime(ts['t']))}Z run" if ts.get("t") else "entered by hand")
            trail.append(f"SafeTrade {when}: {sold:.3f}{h} PRL sold for {net:.2f}{h} USDT net"
                         + (f" (avg {ts['received'] / sold:.4f}{h})" if sold else "")
                         + f", holds {ts.get('usdt', 0):.2f}{h} USDT and {ts.get('prl', 0) + ts.get('prl_locked', 0):.3f}{h} PRL"
                         + (f"; last run sold {ts['last'].get('sold', 0):g} PRL at {ts['last'].get('average')}" if ts.get("last") else ""))
        if xon:
            trail.append(f"our boxes have mined {coins:.6f}{EST} XMR (${coins * xprice:.2f}{EST} at ${xprice:.0f}); the XMR "
                         "pool pays out from 0.01 and counts any other rig mining to the address")
        trail.append(f"Salad credit left ${credit:.2f}{EST} of ${self.cfg['credit_start'] + self.cfg['credit_offset']:.2f}{EST}"
                     + (f", {hours:.0f}{EST} h at ${all_cost:.3f}/hr" if hours is not None else ", nothing billing now")
                     + " (our count: set config credit_offset from the portal now and then)")
        cols = ["Box", "TH/s"] + (["kH/s"] if xon else []) + ["$/hr", "Profit/hr (margin)", "Profit $"]
        lines = [head, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols), *lines, *pool, "",
                 key, "", ". ".join(x[0].upper() + x[1:] for x in trail) + "."]
        if st["credits_out"]:
            lines += ["", "⚠️ Credit ran out: Salad stopped the groups. They restart after a top-up."]
        with open(REPORT + ".tmp", "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(REPORT + ".tmp", REPORT)
        if hours is not None and hours < 12 and t - st["last_credit_alert"] > 3600:
            st["last_credit_alert"] = t
            event(f"ALERT credit ${credit:.2f} lasts about {hours:.1f} h at ${all_cost:.3f}/hr")


def bold(text):
    """Markdown bold. The estimate stars inside are escaped, or Markdown reads them as more emphasis
    and "**+0.27* (+24%*)**" doesn't come out bold."""
    return "**" + text.replace("*", "\\*") + "**"


def with_margin(profit, earn):
    """Profit an hour with its margin in brackets, or alone when there's no margin to show."""
    m = star(pct(profit, earn))
    return f"{profit:+.2f}{EST}" + (f" ({m})" if m != "—" else "")


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
