#!/usr/bin/env python3
# PRL mining on SaladCloud. Runs one round a minute: reads group state, one log query for every
# group, the pool, then applies the rules in README.md and writes state.json (the ledger), report.md
# (the table) and events.log (one line per action) into the data directory: $SALAD_MINE_DIR, or
# run/ next to this file. config.json there is re-read every round, so limits change without a restart.
# The Salad key is sent as the Salad-Api-Key header from $SALAD_API_KEY when it is set; in a Claude
# cloud session the network proxy adds it instead. With $SAFETRADE_API_KEY and $SAFETRADE_API_SECRET set,
# the report also shows our SafeTrade account (read-only calls).
import base64, hashlib, hmac, json, math, os, re, sys, time, urllib.parse, urllib.request, urllib.error, datetime as dt, traceback
from zoneinfo import ZoneInfo
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:   # then a box has no node identity of its own: each PC is a new node, as before 24 October
    Ed25519PrivateKey = None

D = os.environ.get("SALAD_MINE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")
STATE, CFG, EV, REPORT = (os.path.join(D, f) for f in ("state.json", "config.json", "events.log", "report.md"))
ORG = os.environ.get("SALAD_ORG", "llmjob")
RAILWAY = "https://backboard.railway.com/graphql/v2"
SAFETRADE = "https://safetrade.com/api/v2/trade"
STABLE = ("usdc", "dai", "fdusd")   # coins worth a dollar that the USDT from sales may be changed into before withdrawal
PROJECT = os.environ.get("SALAD_PROJECT", "default")
BASE = f"https://api.salad.com/api/public/organizations/{ORG}"
CB = f"{BASE}/projects/{PROJECT}/containers"
IMAGE = "nvidia/cuda:12.2.2-base-ubuntu22.04"
# AMD boxes: the ROCm image the AMD core was built and tested in on Salad (amdtest.py), with its HIP runtime. The AMD
# core ships from v0.6.8, so an AMD box runs that release or a later one.
AMD_IMAGE, AMD_VER = "rocm/dev-ubuntu-24.04:7.2", "v0.6.8"
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
    # AMD, from v0.6.8's AMD core: the mean of its two PCs in benchmark.md (23 October). At Salad's prices then, the
    # 9070 XT clears 10% at batch and on fast PCs at low, the 9060 XT at batch, the 7900 XT at batch only; the
    # 7900 XTX and 7800 XT clear it at none, so they aren't here.
    "AMD RX 9070 XT (16GB)": (99, False), "AMD RX 9060 XT (16GB)": (61, False), "AMD RX 7900 XT (20GB)": (49, False),
}
DEFAULT_CFG = {"max_active": 1, "search": True, "search_token": 0, "restart_token": 0,
               "credit_start": 100.0, "credit_offset": 0.0, "box_ver": "v0.5.13", "ver_override": {}, "report_min": 10,
               "min_margin": 0.10, "max_new_per_hour": 3, "max_per_class": 1,
               "search_every_min": 60, "retry_after_min": 60, "retry_failed_h": 24, "no_pc_streak": 3, "no_pc_hold_min": 180, "drift": 0.15,
               "upgrade_gain": 0.05, "upgrade_every_min": 15, "swap_wait_min": 20,
               "yield": "auto", "yield_min_blocks": 10, "yield_floor": 0.5, "bad_pc_hold_h": 24, "rate_days": 7, "min_rate_pcs": 3,
               # The search, and a group waiting for a PC when it moves up a priority, price a class at this share of
               # its stored PCs (0.35: 35% of them are slower), not the median (see pick_margin).
               "search_pct": 0.35,
               "drop_count": 3, "drop_min": 10, "drop_hours": 24,
               "trader": None, "safetrade": None,
               # The Solana wallet that tops up Salad's credit: SafeTrade withdrawals to it are "to Salad", not bank.
               "salad_wallet": "3v3e89FeHu2wUFdqAwN5MFeMUB6roQrxGJzQis5Qqy9d",
               "wallet": os.environ.get("PRL_WALLET", ""),
               # Monero on the box's CPU (release v0.6.0 on): the groups named in xmr_groups ("*" for every group)
               # mine it to xmr_wallet. Salad bills a GPU group the same for any vCPU and RAM, so they ask for
               # xmr_cpu vCPUs and xmr_memory MB; the CLI picks the threads (xmr_threads 0) and keeps one CPU free.
               "xmr_wallet": "", "xmr_groups": [], "xmr_ver": "v0.6.0", "xmr_cpu": 8, "xmr_memory": 8192,
               "xmr_threads": 0,
               # With xmr_kx on, Monero goes to Kryptex's PPS+ pool (xmr_kx_host) instead of HeroMiners', with release
               # xmr_kx_ver or later (the first with --xmr-pool). A running group is switched by its command alone (see
               # xmr_kx_sync).
               "xmr_kx": False, "xmr_kx_host": "xmr-us.kryptex.network:7029", "xmr_kx_ver": "v0.6.6",
               # What Kryptex credits for Monero work, as a share of the network's pay for it: 94% of what its charts
               # showed from 20 October 6 pm to 21 October 10 am EDT, at the difficulty then (its 1% fee and more).
               "kryptex_xmr_pay": 0.94,
               # Kryptex's PPS+ pool for Pearl: kryptex_groups groups mine there instead of HeroMiners, with release
               # kryptex_ver or later (the first with --pool). New groups start there while fewer are, and groups
               # waiting for a PC are switched (see kx_sync). Two Kryptex boxes that mine no shares, while no other
               # box's work reaches Kryptex, stop the test until kryptex_token changes.
               "kryptex_groups": 0, "kryptex_host": "prl-us.kryptex.network:7048", "kryptex_ver": "v0.6.1",
               "kryptex_pay": 0.94, "kryptex_token": 0,
               # With kx_in_place on, boxes mining on HeroMiners are switched too, by their command alone: Salad
               # restarts a running box on its PC when its group's command changes (seen 20 October), so it keeps
               # the PC and mines on Kryptex a few minutes later.
               "kx_in_place": False,
               # Gemma beside mining (release llm_ver on): the box runs the CLI in auto mode, which fetches and starts
               # the LLM only once the miner has started, and serves the network's jobs. Mining pauses on the card
               # while the LLM answers a prompt (the server's speed test is one); box.sh says so, and the rules wait.
               "llm": False, "llm_ver": "v0.6.5"}
XMR_FEE = 0.009   # HeroMiners' Monero pool fee
KX_API = "https://pool.kryptex.com/prl/api"
# Each pool's own page for an address, linked from the report's pool table. HeroMiners' home page looks up
# ?wallet= (its home.html); Kryptex's miner page is a path.
HM_PAGE = "https://{coin}.herominers.com/?wallet={addr}"
KX_PAGE = "https://pool.kryptex.com/{coin}/miner/stats/{addr}"
# Kryptex's Monero pool, read for the report's XMR Kryptex row.
KXX_API = "https://pool.kryptex.com/xmr/api"


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


def past(t):
    """A past time for an event, as HH:MMZ, with its day when that wasn't today (UTC): the bad list keeps
    entries for days, and "left this PC at 15:54Z" in an event at 15:04Z read as a time still to come."""
    tm = time.gmtime(t)
    if tm[:3] == time.gmtime(now())[:3]:
        return time.strftime("%H:%MZ", tm)
    return f"{tm.tm_mday} {time.strftime('%b %H:%MZ', tm)}"
def iso(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
# The report's clock is New York time (EDT in summer, EST in winter), where the user is.
NY = ZoneInfo("America/New_York")


def clock(t):
    """New York time on the 12-hour clock, which the user reads more easily than 24-hour time: 8:05 pm EDT."""
    d = dt.datetime.fromtimestamp(t, NY)
    return f"{d.hour % 12 or 12}:{d:%M} {'am' if d.hour < 12 else 'pm'} {d:%Z}"


PORTAL = "https://portal-api.salad.com/api/portal"


def portal_balance(user, password):
    """Salad's credit balance in cents, as the portal shows it: log in (a session cookie), then read it. Returns
    (HTTP status, cents, dollars bought) or (status, error, None). Salad's public API has no balance or billing;
    salad_ledger.py reads the rest."""
    import http.cookiejar
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def call(method, path, body=None):
        req = urllib.request.Request(PORTAL + path, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "User-Agent": "llmjob-salad-watcher/1.0"})
        try:
            with op.open(req, timeout=30) as r:
                txt = r.read().decode()
                return r.status, json.loads(txt) if txt.strip() else None
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")[:200]
        except Exception as e:
            return 0, str(e)[:200]

    s, j = call("POST", "/users/login", {"email": user, "password": password})
    if s != 204:
        return s, j, None
    s, j = call("GET", f"/organizations/{ORG}/billing-profile/credits-balance")
    if not (s == 200 and isinstance(j, dict) and "amount" in j):
        return s or 0, j, None
    return s, j["amount"], bought(call)


def bought(call):
    """All the credit ever added, in dollars: the total of the credit grants in Metronome (Salad's billing service),
    less voided ones, read as the portal's credits page reads them. None if the read fails. Until 28 October it
    added up the credit purchase invoices, but a crypto top-up makes a grant and no invoice: the $5 bought with USDT
    that day left the heading at "/$500" (grants: $400 + $100 + $5, and a voided $400)."""
    try:
        s, j = call("POST", f"/organizations/{ORG}/dashboard-sessions/credits")
        tok = urllib.parse.parse_qs(urllib.parse.urlsplit(j["embed_url"]).query)["token"][0]
        q = "query { Customer { CreditGrants { amount_granted voided_at } } }"
        req = urllib.request.Request("https://graphql.metronome.com/v1/graphql", method="POST",
                                     data=json.dumps({"query": q}).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "authorization": f"Bearer {tok}", "User-Agent": "llmjob-salad-watcher/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            grants = json.loads(r.read().decode())["data"]["Customer"][0]["CreditGrants"]
        return sum(float(g["amount_granted"]) for g in grants if not g.get("voided_at")) / 100
    except Exception:
        return None


COIN_DP = {"pearl": 3, "xmr": 6}   # decimals a coin shows in the report


def coin_name(coin):
    """A SafeTrade currency as the report names it: PRL for "pearl", else its code in capitals."""
    return "PRL" if coin == "pearl" else coin.upper()


def short_clock(t, ref):
    """A time for a table cell: 2:00 am on ref's New York day, else just its date (Oct 22). The heading has the zone."""
    d, r = dt.datetime.fromtimestamp(t, NY), dt.datetime.fromtimestamp(ref, NY)
    if d.date() == r.date():
        return f"{d.hour % 12 or 12}:{d:%M}{NB}{'am' if d.hour < 12 else 'pm'}"
    return f"{d:%b}{NB}{d.day}"


def stamp(t):
    """The report's time and date in US style, as the user asked: 8:09 pm EDT, Mon, Oct 19."""
    d = dt.datetime.fromtimestamp(t, NY)
    return f"{clock(t)}, {d:%a}, {d:%b} {d.day}"


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


# Every file is UTF-8, named: Python on Windows (before 3.15) opens text in the locale's code page, cp1252, which
# has no ⛏️ or ⏳, so the first report there would have crashed the watcher.
def event(msg):
    with open(EV, "a", encoding="utf-8") as f:
        f.write(f"{hm()}Z {msg}\n")


def err(msg):
    sys.stderr.write(f"{iso(now())} {msg}\n"); sys.stderr.flush()


def load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def median3(readings):
    """The median TH/s of the last 3 readings ([t, machine, ths, ...])."""
    return sorted(r[2] for r in readings[-3:])[1]


def slug(cls):
    """A class's part of its groups' names: "RTX 4060 (8 GB)" -> rtx4060, "AMD RX 9070 XT (16GB)" -> rx9070xt."""
    if cls.startswith("AMD "):
        return re.sub(r"[^a-z0-9]", "", cls[4:].split("(")[0].lower())
    return "rtx" + re.sub(r"[^a-z0-9]", "", cls.split("(")[0].lower().replace("rtx", ""))


def is_amd(name):
    """Whether a group is an AMD box, from its name (slug)."""
    return name.startswith("biz-rx")


def kx_paid(api, addr, most=100):
    """Everything Kryptex has paid an address: its payouts list comes 15 to a page, newest first, so every page
    is read, while the answer has a "next". Before 22 October only the first page was summed, so the paid total
    dropped as old payouts left it, and with it the Kryptex credit ratio. None if a page fails (the caller keeps
    its last total), or past `most` pages, rather than a total short of some payouts."""
    paid = 0.0
    for page in range(1, most + 1):
        s, p = http("GET", f"{api}/v1/miner/payouts/{addr}" + (f"?page={page}" if page > 1 else ""), timeout=30)
        if s != 200 or not isinstance(p, dict):
            return None
        try:
            paid += sum(float(x.get("received") or 0) for x in p.get("results") or [] if isinstance(x, dict))
        except (TypeError, ValueError):
            return None
        if not p.get("next") or not p.get("results"):
            return paid
    return None


def vnum(ver):
    """A release tag as numbers, for comparing: "v0.6.0" -> (0, 6, 0)."""
    return tuple(int(x) for x in re.findall(r"\d+", ver))


def cmd_ver(cmd):
    """The release a box command starts: box.sh pins v0.5.13 unless the command's sed puts another in (cmd_for)."""
    m = re.search(r"VER=v[\d.]+;/VER=(v[\d.]+);/", cmd)
    return m.group(1) if m else "v0.5.13"


def node_step(node):
    """Shell that writes a box's LLMJob node identity where the CLI keeps it (~/.local/share/llmjob-earn/node.json),
    before box.sh starts the CLI. Each PC is a fresh container, so every PC was a new node, and the server's speed
    test (1,024 tokens, once a day a node) paused mining there for 2–10 min: 304 of 327 pauses on 20–24 October,
    $0.21–0.26 a day of Pearl (OPEN.md #11). With one identity a group, the test comes once a day a group."""
    js = json.dumps({"nodeId": node["nodeId"], "publicKey": node["publicKey"], "secretKey": node["secretKey"],
                     "name": None, "connected": False, "createdAt": node["createdAt"]}, separators=(",", ":"))
    return (f"d=${{HOME:-/root}}/.local/share/llmjob-earn; mkdir -p $d && printf '%s' '{js}' > $d/node.json "
            f"&& chmod 600 $d/node.json; ")


def cmd_for(worker, wallet, ver, xmr=None, threads=0, kryptex=None, llm=False, xmr_pool=None, amd=False, node=None):
    sed = f"sed -e 's/^WORKER=__WORKER__$/WORKER={worker}/' -e 's/^W=prl1.*$/W={wallet}/'"
    if ver != "v0.5.13":
        sed += f" -e 's/^VER=v0.5.13;/VER={ver};/'"
    flags = []
    if kryptex:
        # Kryptex: Pearl on its stratum (release v0.6.1 on). The miner logs in there as "wallet/worker" by itself.
        flags.append(f"--pool {kryptex}")
        # box.sh checks HeroMiners' port 1200 before it starts; a Kryptex box checks Kryptex's host and port instead,
        # so it doesn't leave a PC that blocks only a port it never uses.
        host, port = kryptex.rsplit(":", 1)
        sed += (f" -e 's|for h in us us2 ca de; do timeout 5 bash -c \"exec 3<>/dev/tcp/$h.pearl.herominers.com/1200\"|"
                f"for h in {host}; do timeout 5 bash -c \"exec 3<>/dev/tcp/$h/{port}\"|'"
                f" -e 's|outbound port 1200 blocked|outbound port {port} blocked|'")
    if xmr:
        # Monero: box.sh also downloads the release's CPU worker and puts it beside each GPU's CLI. --xmr-pool (release
        # v0.6.6 on) sends it to that pool instead of HeroMiners'; box.sh checks no Monero port.
        flags.append(f"--xmr-address {xmr}" + (f" --xmr-pool {xmr_pool}" if xmr_pool else "")
                     + (f" --cpu-threads {threads}" if threads else ""))
        sed += (" -e 's/ pearl_core_cu13.node; do/ pearl_core_cu13.node llmjob-xmr-worker; do/'"
                " -e 's|cp $D/rel/cli $R/|cp $D/rel/cli $D/rel/llmjob-xmr-worker $R/ \\&\\& chmod +x $R/llmjob-xmr-worker|'")
    if llm:
        # Gemma beside mining: auto mode instead of mining only, and serving the network's jobs. While the LLM answers
        # a prompt the CLI stops mining on the card and prints no hashrate, so box.sh would repeat the miner's last
        # line, which reads as a dead miner: it says "paused for a prompt" instead, from the CLI's own pause lines.
        sed += (" -e 's/ --mode mining -w \\$WORKER-g\\$g --no-serve / --mode auto -w $WORKER-g$g /'"
                " -e '/^    line=\\$(tr -d/a\\    p=$(grep -aoE \"mining (paused|resumed)\" $D/g$g.log | tail -1); "
                "[ \"$p\" = \"mining paused\" ] && line=\"paused for a prompt\"'")
    if amd:
        # An AMD box (v0.6.8 on): the AMD core beside the CLI, and --core amd. box.sh reads its card with nvidia-smi,
        # which the ROCm image doesn't have: the card line comes from rocminfo, the box has one card, and the
        # 5-minute card stats print nothing. Its build tools are CUDA's, so they aren't installed.
        flags.append("--core amd")
        sed += (" -e 's#^  nvidia-smi --query-gpu=index,name,driver_version.*$#  echo \"[card] 0, $(rocminfo 2>/dev/null "
                "| grep -m1 -oE \"AMD Radeon[ A-Za-z0-9]+\"), $(rocminfo 2>/dev/null | grep -m1 -oE \"gfx[0-9a-f]+\")\"#'"
                " -e 's#^N=.*nvidia-smi -L.*$#N=1#' -e 's#^CC=.*nvidia-smi.*$#CC=0#'"
                " -e 's#^  declare -A dl; for f in llmjob-earn-cli-linux #  declare -A dl; for f in pearl_core_hip.node "
                "llmjob-earn-cli-linux #'"
                " -e 's#release) cp \\$D/rel/pearl_core.node #release) cp $D/rel/pearl_core_hip.node $D/rel/pearl_core.node #'"
                " -e 's#\\&\\& nvidia-smi --query-gpu=index,clocks.sm#\\&\\& true --query-gpu=index,clocks.sm#'"
                " -e 's#^if \\[ ! -f \\$D/tools \\]; then#if false; then#'")
    if flags:
        # Both go in box.sh's default flags (a control file for the box would override them; the biz- boxes have none).
        sed += f" -e 's|\"flags\":\"\"}}|\"flags\":\"{' '.join(flags)}\"}}|'"
    return (f"{node_step(node) if node else ''}"
            f"apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl ca-certificates >/dev/null 2>&1 && "
            f"curl -fsSL {BOX} | {sed} > /root/box.sh; bash /root/box.sh 2>&1 | tee -a /root/box.log")


class W:
    def __init__(self):
        self.st = load(STATE, None) or {"groups": {}, "created": now(), "credits_out": False, "seen": {},
                                        "search_hour": None, "search_token": 0, "restart_token": 0,
                                        "last_round": None, "last_log_end": None, "last_report_q": None,
                                        "last_credit_alert": 0, "classes": {}, "classes_t": 0, "mkt": None, "mkt_t": 0,
                                        "pool": {}, "api_fail": 0}
        self.cfg = DEFAULT_CFG
        self.quota_full = False   # set when Salad refuses a create for its replica quota; the search stops on it

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
            # A pool's work unit (share difficulty 1) is 2^32 hashes, so one share at 2^21 is 2^53 hashes.
            st["mkt"] = {"price": price, "prl_th_hr": prl_th_hr, "usd_th_hr": prl_th_hr * price,
                         "prl_unit": prl_th_hr * 2 ** 32 / (1e12 * 3600)}
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
            # Salad bills a running group at today's price, not the one it was created at (its 1 November 2026 rise
            # applied to running instances), so each group takes a new price here, for spend and its margin.
            for name, g in st["groups"].items():
                new = st["classes"].get(g["class"], {}).get("prices", {}).get(g["pri"])
                if not g["ended"] and new is not None and abs(new - g["price"]) > 1e-9:
                    event(f"PRICE {name}: {g['class']} at {g['pri']} now ${new:.3f}/hr (was ${g['price']:.3f})")
                    g["price"] = new
        else:
            err(f"gpu-classes failed {s} {j}")

    def margin(self, cls, pri, rate=None, kx=None):
        """The margin a class should make at a priority, at its measured rate, on Kryptex or not (kx; None: the pool
        a new group would start on)."""
        return margin_of((rate or self.rate(cls)) * self.usd(kx) + self.xmr_usd(cls=cls),
                         self.st["classes"][cls]["prices"][pri])

    def pick_margin(self, cls, pri, kx=None):
        """The margin a class must clear at a priority before the search picks it there, or a waiting group moves up
        to it: at a PC slower than search_pct of its stored PCs, not its median one. At the median about a third of
        the PCs Salad handed out failed: RTX 4060s at medium (median 59.7 TH/s, +12–13%; 57 or less fails) used 7
        counted moves on 23–24 October and one was deleted (OPEN.md #69)."""
        return self.margin(cls, pri, self.rate(cls, self.cfg["search_pct"]), kx)

    def xmr_usd(self, g=None, cls=None):
        """What Monero on a box's CPU earns an hour, for the rules. For group g (0 unless it mines Monero), its own
        figure once the pools' 30-minute averages hold only this PC (own_khs). A box whose Monero doesn't reach the
        pool then counts 0, and so does one the pools haven't seen at all 45 minutes in. Before that, and for a new
        group of class cls, the median of the own figures of that class's PCs in the last rate_days (xmr_rates), or
        with none stored, the median of the live boxes the pools saw in the last 15 minutes. The rules counted Pearl
        alone until 22 October, so boxes that cleared 10% with their Monero (about $0.002 a box) were left."""
        if not (g.get("xmr") if g is not None else self.wants_xmr("*")):
            return 0.0
        t, xusd = now(), (self.st.get("xmr_mkt") or {}).get("usd_khs_hr", 0.0)
        if g is not None and g.get("has_pc") and g.get("pc_since"):
            own = self.own_khs(g, t)
            if own is not None:
                return own * xusd
            since = max(g["pc_since"], g.get("run_since") or 0, g.get("restart_t") or 0, self.pause_end(g))
            # A box the pools never saw here has no reads at all, and took the median for as long as it stayed: 4060-60
            # on 23 October, an hour on its PC with no Monero at either pool, was judged +14% against its +10%. A new
            # box shows at the pools within 30 minutes (3060-75 that evening), so after 45 with none it counts 0.
            if t - since >= 45 * MIN and not any(seen >= since for seen, _ in g.get("xmr_own") or []):
                return 0.0
        # By class: a PC's CPU goes with its owner's card. The fleet median, which the 3060s' 3–5 kH/s pulled up while
        # 4060s made 0.5–1.3, credited a new 4060 about $0.0038/hr against its $0.002, so on 23 October 4060-60 kept a
        # PC another 4060 had left at 58 TH/s and left it 8 minutes later at +9% (OPEN.md #65).
        mine = sorted(k for k, _ in self.st.get("xmr_rates", {}).get(g["class"] if g is not None else cls, {}).values())
        if mine:
            n = len(mine)
            return (mine[n // 2] if n % 2 else (mine[n // 2 - 1] + mine[n // 2]) / 2) * xusd
        khs = sorted(k for k in (self.box_khs(x, t) for x in self.st["groups"].values()
                                 if not x["ended"] and x.get("xmr") and x["has_pc"]) if k > 0)
        n = len(khs)
        if not n:
            return 0.0
        return (khs[n // 2] if n % 2 else (khs[n // 2 - 1] + khs[n // 2]) / 2) * xusd

    def own_khs(self, g, t):
        """A box's own Monero kH/s on its PC, once the pools' 30-minute averages hold only this PC: the mean of its
        last 3 reads taken 30 minutes or more after it got the PC, restarted or last paused (xmr_own), the latest
        within 15 minutes. None before that."""
        since = max(g["pc_since"], g.get("run_since") or 0, g.get("restart_t") or 0, self.pause_end(g))
        own = [k for seen, k in g.get("xmr_own") or [] if seen >= since + 30 * MIN][-3:]
        if own and t - g["xmr_own"][-1][0] < 15 * MIN:
            return sum(own) / len(own)
        return None

    def box_khs(self, g, t):
        """A box's Monero kH/s as HeroMiners' pay values it (xmr_mkt's usd_khs_hr is HeroMiners' net pay): its
        HeroMiners figure, plus its Kryptex one times what Kryptex pays against that (kryptex_xmr_pay over
        1 - XMR_FEE, 0.94 against 0.991), each only when the pool saw the box in the 15 minutes before t."""
        hm = g.get("xmr_khs") or 0.0 if t - (g.get("xmr_seen") or 0) < 15 * MIN else 0.0
        kx = g.get("kxx_khs") or 0.0 if t - (g.get("kxx_seen") or 0) < 15 * MIN else 0.0
        return hm + kx * self.cfg["kryptex_xmr_pay"] / (1 - XMR_FEE)

    def rate(self, cls, q=0.5):
        """The TH/s a new group of this class should make: the median of the PCs it has mined on here in the
        last rate_days (each PC's rate on its last 3 readings after warm-up, capped and slow PCs included), or
        the benchmark in KNOWN before any has. With q, the rate that share of them are slower than (between
        neighbours), for pick_margin."""
        pcs = sorted(r for r, _ in self.st.get("pc_rates", {}).get(cls, {}).values())
        if not pcs:
            return KNOWN[cls][0]
        n = len(pcs)
        i, f = divmod(q * (n - 1), 1)
        i = int(i)
        med = pcs[i] * (1 - f) + pcs[i + 1] * f if f else pcs[i]
        # A PC a group left for its margin is stored at its slow rate, so with few PCs a class's rate can be one bad
        # PC's: one RTX 5090 left at 131 TH/s would make the search expect 131 for every 5090, not 418, and never
        # create one (OPEN.md #62). Until min_rate_pcs PCs, a slower median doesn't lower the benchmark.
        return med if n >= self.cfg["min_rate_pcs"] else max(med, KNOWN[cls][0])

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
            # Each new Monero read of the box, for its own figure in xmr_usd.
            seen = max(g.get("xmr_seen") or 0, g.get("kxx_seen") or 0)
            if not g["ended"] and g.get("xmr") and g.get("has_pc") and seen:
                own = g.setdefault("xmr_own", [])
                if not own or own[-1][0] < seen:
                    own.append([seen, self.box_khs(g, seen)])
                    del own[:-6]
                # Its own figure on this PC, by class, for a new box's estimate (xmr_usd).
                k = self.own_khs(g, t) if g.get("pc_since") and g.get("machine") else None
                if k is not None:
                    st.setdefault("xmr_rates", {}).setdefault(g["class"], {})[g["machine"]] = [k, t]
        for pcs in list(st["pc_rates"].values()) + list(st.get("xmr_rates", {}).values()):
            for m in [m for m, (_, seen) in pcs.items() if t - seen >= 86400 * self.cfg["rate_days"]]:
                del pcs[m]
        # A name no live group has used for rate_days lets its node identity go.
        keys = st.get("node_keys", {})
        for n in [n for n, k in keys.items() if t - k.get("used", 0) >= 86400 * self.cfg["rate_days"]]:
            del keys[n]

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
        # The estimate when Kryptex's totals were read: while its API fails the totals stay put but the estimate
        # grows, which would lower the ratio about 1% an hour.
        prl = (kt or {}).get("prl_kx") or sum(g.get("prl_kx", 0.0) for g in self.st["groups"].values())
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

    def usd(self, kx=None):
        """$ per TH-hr the rules count on: the network's rate times config "yield", a number, or "auto" for the
        credit ratio of the pool the box mines on: Kryptex's for kx True, HeroMiners' for False, and for None the
        pool a new group would start on. Until 22 October "auto" was HeroMiners' ratio for every box, 0.93 against
        Kryptex's 0.97 while every box mined on Kryptex, so the rules left and deleted boxes that cleared the bar:
        63 of 125 margin moves and deletions in the 48 hours before."""
        y = self.cfg.get("yield", "auto")
        if y != "auto":
            return self.st["mkt"]["usd_th_hr"] * float(y)
        if kx is None:
            kx = self.new_on_kx()
        return self.st["mkt"]["usd_th_hr"] * (self.kx_ratio() if kx else self.credit_ratio())

    def new_on_kx(self):
        """Whether a group created now would mine on Kryptex: while fewer groups than config asks are there."""
        return sum(1 for g in self.st["groups"].values() if not g["ended"] and g.get("kx")) < self.kx_want()

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
        paid = kx_paid(KX_API, wallet)
        if s1 != 200 or not isinstance(b, dict) or paid is None:
            return res
        try:
            kt = {"t": t, "total": float(b.get("total") or 0), "confirmed": float(b.get("confirmed") or 0),
                  "unconfirmed": float(b.get("unconfirmed") or 0), "paid": paid,
                  "prl_kx": sum(g.get("prl_kx", 0.0) for g in st["groups"].values())}
        except (TypeError, ValueError):
            return res
        st["kx_tot"] = kt
        return res

    def prl_unit(self):
        """The PRL the network pays for one unit of pool work (2^32 hashes) now, before any pool's cut."""
        mk = self.st.get("mkt") or {}
        return mk.get("prl_unit") or mk.get("prl_th_hr", 0.0) * 2 ** 32 / (1e12 * 3600)

    def hm_count(self, g, w):
        """The Pearl work HeroMiners has counted for a box's worker ("hashes": the difficulty of the shares it
        accepted), each new unit valued at the network's pay for it when first seen."""
        if not w or not self.st.get("mkt"):
            return
        try:
            units = float(w.get("hashes") or 0)
        except (TypeError, ValueError):
            return
        last = g.get("hm_units")
        if last is None:
            # A row an earlier group of the same name left: only work after this group started counts.
            last = units if share_t(w) < g["created"] else 0.0
        g["hm_gross"] = g.get("hm_gross", 0.0) + (units - last if units >= last else units) * self.prl_unit()
        g["hm_units"] = units

    def kx_count(self, groups, t):
        """Each Kryptex box's accepted shares, from its worker's chart at Kryptex (10-minute buckets), at most every
        5 minutes. PPS+ pays a set amount a share, so the shares, each bucket valued at the network's pay for a
        share when first seen, give what the box has earned there. A bucket still filling is read again."""
        st = self.st
        for g in st["groups"].values():
            if g["ended"] and "kx_b" in g:
                del g["kx_b"]   # its totals stay; the buckets were only kept to read a filling one again
        if t - st.get("kx_count_t", 0) < 5 * MIN or not st.get("mkt"):
            return
        st["kx_count_t"] = t
        per_share = self.prl_unit() * 2 ** 21
        for g in groups:
            # On Kryptex now, or in the last hour, or mined there before these counts began and not read yet.
            if not (g.get("kx") or t - (g.get("sw_t") or 0) < 3600
                    or (g["has_pc"] and self.on_kx(g, g.get("machine"), t)) or (g.get("prl_kx") and "kx_gross" not in g)):
                continue
            s, c = http("GET", f"{KX_API}/v1/miner/chart/{self.cfg['wallet']}/{g['name']}-g0?days=2", timeout=30)
            if s != 200 or not isinstance(c, list):
                continue
            b = g.setdefault("kx_b", {})
            for p in c:
                try:
                    ts, v = str(int(p["timestamp"])), int(p.get("valid") or 0)
                except (KeyError, TypeError, ValueError):
                    continue
                if int(ts) / 1000 + 600 <= g["created"]:
                    continue   # a bucket that ended before the group began: an earlier group of the same name
                if v > b.get(ts, [0])[0]:
                    b[ts] = [v, b[ts][1] if ts in b else per_share]
            g["kx_shares"] = sum(v for v, _ in b.values())
            g["kx_gross"] = sum(v * ps for v, ps in b.values())

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

    def credit_count(self):
        """Our own count of the Salad credit left: what was bought, less each group's price for every minute it had a
        PC, moved to Salad's figure at its last read (credit_fix). The report shows it, starred, only when Salad's
        balance hasn't been read for an hour."""
        spend = sum(g["spend"] for g in self.st["groups"].values())
        return self.cfg["credit_start"] + self.cfg["credit_offset"] - spend + self.st.get("credit_fix", 0.0)

    def report_read_due(self, key, t):
        """Whether a read for the report is due: one each report period (report_min, the quarter hour), in its first
        round, which writes that period's report, so the report shows a read from its own minute. A failed read is
        tried again 5 minutes later. The caller marks a good read with mark_report_read."""
        st = self.st
        if st.get(key + "_q") == int(t // (60 * self.cfg["report_min"])) or t - st.get(key + "_try", 0) < 300:
            return False
        st[key + "_try"] = t
        return True

    def mark_report_read(self, key, t):
        self.st[key + "_q"] = int(t // (60 * self.cfg["report_min"]))

    def salad_balance(self):
        """Salad's own credit balance, read from the portal once a report (report_read_due) when $SALAD_USER and
        $SALAD_PASS are set. Each read also sets our count to it, so the count only drifts while reads fail. A failed
        read warns once and keeps the last."""
        user, pw, st, t = os.environ.get("SALAD_USER"), os.environ.get("SALAD_PASS"), self.st, now()
        if not user or not pw or not self.report_read_due("bal", t):
            return
        s, cents, paid = portal_balance(user, pw)
        if s != 200 or not isinstance(cents, (int, float)):
            if not st.get("bal_warned"):
                st["bal_warned"] = True
                event(f"WARN can't read Salad's balance from the portal: HTTP {s} {str(cents)[:150]}")
            return
        usd = cents / 100
        st["credit_fix"] = st.get("credit_fix", 0.0) + usd - self.credit_count()
        st["salad_bal"] = {"t": t, "usd": usd}
        if paid:
            st["salad_bought"] = paid
        st["bal_warned"] = False
        self.mark_report_read("bal", t)

    def safetrade(self):
        """Our SafeTrade account, where the pools pay and the trader sells, for its rows in the pool table: each coin
        held and in open orders, and all we have sold (fills of our own sell orders) with the USDT it brought after
        fees. Each new sale is a SOLD event. Read once a report (report_read_due), with GET calls only, when
        $SAFETRADE_API_KEY and $SAFETRADE_API_SECRET are set. A failed read keeps the last one, and the table says
        when that was."""
        key, sec = os.environ.get("SAFETRADE_API_KEY"), os.environ.get("SAFETRADE_API_SECRET")
        st, t = self.st, now()
        if not key or not sec or not self.report_read_due("st", t):
            return

        def get(path):
            # Signed with HMAC-SHA256 of a millisecond nonce and the key, on the real clock (SafeTrade checks it).
            n = str(int(time.time() * 1000))
            sig = hmac.new(sec.encode(), (n + key).encode(), hashlib.sha256).hexdigest()
            s, j = http("GET", SAFETRADE + path, timeout=30,
                        headers={"X-Auth-Apikey": key, "X-Auth-Nonce": n, "X-Auth-Signature": sig})
            if s != 200 or not isinstance(j, list):
                if not st.get("st_warned"):
                    st["st_warned"] = True
                    event(f"WARN can't read SafeTrade {path.split('?')[0]}: HTTP {s} {str(j)[:150]}")
                return None
            return j

        bal = get("/account/balances/spot")
        if bal is None:
            return
        fills = []
        for page in range(1, 21):   # every fill we have, newest first, 100 a page
            got = get(f"/market/trades?limit=100&page={page}")
            if got is None:
                return
            fills += got
            if len(got) < 100:
                break
        coins = {}
        for b in bal:
            try:
                coins[str(b["currency"]).lower()] = [float(b.get("balance") or 0), float(b.get("locked") or 0)]
            except (KeyError, TypeError, ValueError):
                pass
        # The first read only notes the newest fill, so the account's history isn't announced as new sales.
        # sold: coin -> [amount, USDT after fees, the fees]. bought: a stablecoin bought with USDT (the founder changed
        # 110.27 USDT to USDC on 14 October and withdrew it) -> [USDT spent, its trading fee].
        sold, bought, new, last = {}, {}, {}, st.get("st_fill_id")
        for f in fills:
            # A fill's "side" is the taker's; "order_side" is our own order's.
            m = str(f.get("market") or "")
            if not m.endswith("usdt"):
                continue
            try:
                fid, amt, got = int(f["id"]), float(f["amount"]), float(f["total"])
                fee = float(f.get("fee") or 0)
            except (KeyError, TypeError, ValueError):
                continue
            if f.get("order_side") == "buy" and m[:-4] in STABLE:
                b = bought.setdefault(m[:-4], [0.0, 0.0])
                b[0] += got
                b[1] += fee   # in the coin bought, about a dollar each
                continue
            if f.get("order_side") != "sell":
                continue
            fee = fee if f.get("fee_currency") == "usdt" else 0.0
            coin = "pearl" if m[:-4] in ("pearl", "prl") else m[:-4]   # SafeTrade's PRL is "pearl" since 20 Oct
            s = sold.setdefault(coin, [0.0, 0.0, 0.0])
            s[0] += amt
            s[1] += got - fee
            s[2] += fee
            if last is not None and fid > last:
                o = new.setdefault(f.get("order_id"), [coin, 0.0, 0.0])
                o[1] += amt
                o[2] += got - fee
        for coin, amt, usd in reversed(list(new.values())):
            event(f"SOLD {amt:.{COIN_DP.get(coin, 4)}f} {coin_name(coin)} on SafeTrade for ${usd:.2f} USDT after fees "
                  f"(${usd / amt:.2f} each)")
        # What reached the account: every deposit of each coin, so the table can set it beside what the pools say they
        # paid. On 26 October the 124 PRL deposits came to 196.743, the pools' paid to the thousandth.
        recv = {}
        for coin in ("pearl", "xmr"):
            total = 0.0
            for page in range(1, 21):
                got = get(f"/account/deposits?currency={coin}&limit=100&page={page}")
                if got is None:
                    return
                for d in got:
                    if str(d.get("status") or "").lower() not in ("rejected", "failed", "canceled", "cancelled", "errored"):
                        try:
                            total += float(d.get("amount") or 0)
                        except (TypeError, ValueError):
                            pass
                if len(got) < 100:
                    break
            recv[coin] = total
        # What left: every withdrawal that went through, by currency -> [amount, network fee]. On 26 October, 103.26
        # USDC (fee 7) on 14 October and 206.28 USDT (fee 3) on 22 October, both on Ethereum. One to the Salad
        # top-up wallet (salad_wallet) is keyed "<currency>>salad": it paid for credit and never reached the bank.
        out, topup = {}, self.cfg.get("salad_wallet")
        for page in range(1, 21):
            got = get(f"/account/withdraws?limit=100&page={page}")
            if got is None:
                return
            for x in got:
                if str(x.get("status") or x.get("state") or "").lower() not in ("succeed", "success", "done", "accepted",
                                                                                "processing", "confirming", "collected"):
                    continue
                try:
                    cur = str(x.get("currency_id") or x.get("currency")).lower()
                    o = out.setdefault(cur + (">salad" if topup and x.get("rid") == topup else ""), [0.0, 0.0])
                    o[0] += float(x.get("amount") or 0)
                    o[1] += float(x.get("fee") or 0)
                except (TypeError, ValueError):
                    pass
            if len(got) < 100:
                break
        ids = []
        for f in fills:
            try:
                ids.append(int(f["id"]))
            except (KeyError, TypeError, ValueError):
                pass
        st["st_fill_id"] = max(ids + [last or 0])
        st["st_acct"] = {"t": t, "coins": coins, "sold": sold, "recv": recv, "bought": bought, "out": out}
        st["st_warned"] = False
        self.mark_report_read("st", t)

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

    def ghost_line(self, g, tm, mach, text):
        """A log line from a group already deleted. Salad's delete stops the copy it lists, but a first copy it no
        longer listed (OPEN.md #23) mined on: 16 min for 3060-50 on 21 Oct, and over 10 min for 3060-71 on 23 Oct.
        Salad then answers 404 for the group and the instance, so nothing here can stop it, and the report no longer
        shows it. Said once a PC, for mining lines 5 min or more after the delete whose uptime moves (box.sh repeats
        a dead miner's last line, uptime and all)."""
        m = re.search(r"\[mining g\d+\] \d\d:\d\d:\d\d\s+([0-9.]+) TH/s · [0-9,]+ accepted.*?up (\S+ \S+)", text)
        if not m or not mach or tm - g["ended"] < 5 * MIN or mach in g.setdefault("ghost_said", []):
            return
        ups = g.setdefault("ghost_up", {})
        if ups.get(mach) in (None, m.group(2)):
            ups[mach] = m.group(2)
            return
        g["ghost_said"].append(mach)
        event(f"WARN {g['name']}: deleted at {past(g['ended'])}, but PC {mach[:8]} still mines for it "
              f"({float(m.group(1)):g} TH/s, up {m.group(2)}); Salad no longer lists it, so it can't be stopped from here")

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
            g["left_t"] = now()
            event(f"REALLOCATED {g['name']} off PC {(g['machine'] or '')[:8]} ({reason}); "
                  + (f"reallocation {g['reallocs']} of 3" if count else "not counted toward its 3"))
            g.update(machine=None, pc_since=None, run_since=None, lost_since=now(), profit_fail=0, has_pc=False)
            return True
        err(f"reallocate {g['name']} failed {s} {j}")
        return False

    def next_pri(self, g):
        """The cheapest priority above the group's that still clears the margin (pick_margin), or None."""
        return next((p for p in PRIS[PRIS.index(g["pri"]) + 1:]
                     if self.pick_margin(g["class"], p, kx=bool(g.get("kx"))) >= self.cfg["min_margin"]), None)

    def set_priority(self, g, pri, why="no PC at {old} in 10 min"):
        s, j = http("PATCH", f"{CB}/{g['name']}", {"container": {"priority": pri}}, ctype="application/merge-patch+json")
        if 200 <= s < 300:
            old = g["pri"]
            g.update(pri=pri, price=self.st["classes"][g["class"]]["prices"][pri], pri_since=now())
            event(f"PRIORITY {g['name']}: {why.format(old=old)}, moved to {pri} (${g['price']:.3f}/hr, "
                  f"{self.margin(g['class'], pri, kx=bool(g.get('kx'))) * 100:+.0f}% margin)")
            return True
        err(f"patch {g['name']} -> {pri} refused {s} {str(j)[:200]}")
        return False

    def wants_xmr(self, name):
        """Whether config asks this group to mine Monero on its CPU too."""
        gs = self.cfg["xmr_groups"]
        return bool(self.cfg["xmr_wallet"]) and ("*" in gs or name in gs)

    def box(self, name, xmr, kx=False):
        """The release, start command and CPU/RAM for a group's box. Monero needs release xmr_ver or later (xmr_kx_ver
        on Kryptex), Kryptex kryptex_ver or later, and Gemma (config "llm") llm_ver or later."""
        ver = self.cfg["ver_override"].get(name, self.cfg["box_ver"])
        host = self.cfg["kryptex_host"] if kx else None
        llm, amd = self.llm_on(name), is_amd(name)
        node = self.node_key(name) if llm else None
        if amd and vnum(ver) < vnum(AMD_VER):
            ver = AMD_VER
        if kx and vnum(ver) < vnum(self.cfg["kryptex_ver"]):
            ver = self.cfg["kryptex_ver"]
        if llm and vnum(ver) < vnum(self.cfg["llm_ver"]):
            ver = self.cfg["llm_ver"]
        if not xmr:
            return (ver, ["/bin/sh", "-c", cmd_for(name, self.cfg["wallet"], ver, kryptex=host, llm=llm, amd=amd,
                                                   node=node)],
                    {"cpu": 2, "memory": 4096})
        if vnum(ver) < vnum(self.cfg["xmr_ver"]):
            ver = self.cfg["xmr_ver"]
        xpool = self.cfg["xmr_kx_host"] if self.xmr_kx(xmr) else None
        if xpool and vnum(ver) < vnum(self.cfg["xmr_kx_ver"]):
            ver = self.cfg["xmr_kx_ver"]
        cmd = cmd_for(name, self.cfg["wallet"], ver, self.cfg["xmr_wallet"], self.cfg["xmr_threads"], host, llm, xpool, amd,
                      node)
        return ver, ["/bin/sh", "-c", cmd], {"cpu": self.cfg["xmr_cpu"], "memory": self.cfg["xmr_memory"]}

    def node_key(self, name):
        """The LLMJob node identity for a group's box (node_step): an Ed25519 key in the CLI's form (tweetnacl's
        64-byte secret key, the id the first 16 hex of the public key's sha256), made the first time and kept by
        group name, so a group keeps it across its PCs and restarts. None without the cryptography module."""
        keys = self.st.setdefault("node_keys", {})
        k = keys.get(name)
        if k is None:
            if Ed25519PrivateKey is None:
                return None
            sk = Ed25519PrivateKey.generate()
            seed = sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                    serialization.NoEncryption())
            pub = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            pk = base64.b64encode(pub).decode()
            k = keys[name] = {"nodeId": hashlib.sha256(pk.encode()).hexdigest()[:16], "publicKey": pk,
                              "secretKey": base64.b64encode(seed + pub).decode(),
                              "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now()))}
        k["used"] = now()
        return k

    def llm_on(self, name):
        """Whether a group's box runs Gemma: config "llm", except on an AMD box, where the CLI's local LLM has no
        build for the card."""
        return bool(self.cfg["llm"]) and not is_amd(name)

    def xmr_kx(self, xmr):
        """Whether a box's command sends its Monero to Kryptex: it mines Monero and config xmr_kx is on."""
        return bool(xmr) and bool(self.cfg["xmr_kx"])

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
                g["xmr"], g["ver"], g["llm"], g["xmr_kx"] = want, ver, self.llm_on(g["name"]), self.xmr_kx(want)
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

    def llm_sync(self, active, t):
        """Turn Gemma on or off in a group when config "llm" changes, by updating its command only. A box that is
        mining keeps its PC: Salad usually restarts it there with the new command within minutes (19 October: 5 of 8,
        after 6 minutes); one it doesn't restart runs Gemma from its next PC, as a group waiting for a PC does from
        its first. A group still preparing its image is left for a later round (Salad refuses updates then),
        and a refused update is tried again after 10 minutes."""
        for g in active:
            want = self.llm_on(g["name"])
            if (g["ended"] or bool(g.get("llm")) == want or g.get("status") == "pending"
                    or t - g.get("llm_try", 0) < 600):
                continue
            g["llm_try"] = t
            ver, command, _ = self.box(g["name"], bool(g.get("xmr")), bool(g.get("kx")))
            s, j = http("PATCH", f"{CB}/{g['name']}", {"container": {"command": command}},
                        ctype="application/merge-patch+json")
            if 200 <= s < 300:
                g["llm"], g["ver"], g["xmr_kx"] = want, ver, self.xmr_kx(g.get("xmr"))
                event(f"GEMMA {'ON' if want else 'OFF'} {g['name']}: miner {ver}; "
                      + ("it keeps this PC, where Salad usually restarts it that way within minutes" if g["has_pc"]
                         else "it starts that way on its next PC"))
            else:
                event(f"WARN Gemma {'on' if want else 'off'} for {g['name']} refused: HTTP {s} {str(j)[:150]}")

    def xmr_kx_sync(self, active, t):
        """Send each Monero group's CPU mining to Kryptex's pool, or back to HeroMiners', when config xmr_kx changes,
        by updating its command only, as llm_sync does: a box that is mining keeps its PC, where Salad usually
        restarts it with the new command within minutes; one waiting for a PC starts that way at its next. A group
        still preparing its image is left for a later round, and a refused update is tried again after 10 minutes."""
        for g in active:
            want = self.xmr_kx(g.get("xmr"))
            if (g["ended"] or bool(g.get("xmr_kx")) == want or g.get("status") == "pending"
                    or t - g.get("xmr_kx_try", 0) < 600):
                continue
            g["xmr_kx_try"] = t
            ver, command, _ = self.box(g["name"], bool(g.get("xmr")), bool(g.get("kx")))
            s, j = http("PATCH", f"{CB}/{g['name']}", {"container": {"command": command}},
                        ctype="application/merge-patch+json")
            pool = "Kryptex" if want else "HeroMiners"
            if 200 <= s < 300:
                g["xmr_kx"], g["ver"], g["llm"] = want, ver, self.llm_on(g["name"])
                event(f"MONERO ON {pool.upper()} {g['name']}: miner {ver}; "
                      + ("it keeps this PC, where Salad usually restarts it that way within minutes" if g["has_pc"]
                         else "it starts that way on its next PC"))
            else:
                event(f"WARN Monero to {pool} for {g['name']} refused: HTTP {s} {str(j)[:150]}")

    def ver_sync(self, active, t):
        """Move each group to the release config gives it (box_ver, or ver_override for that group) when that is newer
        than the one it runs, by updating its command only, as llm_sync does: a box that is mining keeps its PC, where
        Salad usually restarts it with the new release within minutes; one waiting for a PC starts that way at its
        next. box_ver never lowers a release; ver_override does, for the group it names, to take it back from a bad
        one: v0.6.7's Linux cores need glibc 2.38, and the boxes' Ubuntu 22.04 has 2.35, so on 21 October the first
        box moved to it ran no miner. Until then box_ver reached new groups only. A group still preparing its image is
        left for a later round, and a refused update is tried again after 10 minutes."""
        for g in active:
            if g["ended"] or g.get("status") == "pending" or t - g.get("ver_try", 0) < 600:
                continue
            ver, command, _ = self.box(g["name"], bool(g.get("xmr")), bool(g.get("kx")))
            old = g.get("ver") or "v0"
            if vnum(ver) == vnum(old) or (vnum(ver) < vnum(old) and g["name"] not in self.cfg["ver_override"]):
                continue
            g["ver_try"] = t
            s, j = http("PATCH", f"{CB}/{g['name']}", {"container": {"command": command}},
                        ctype="application/merge-patch+json")
            if 200 <= s < 300:
                g["ver"], g["llm"], g["xmr_kx"] = ver, self.llm_on(g["name"]), self.xmr_kx(g.get("xmr"))
                event(f"RELEASE {g['name']}: miner {old} -> {ver}; "
                      + ("it keeps this PC, where Salad usually restarts it that way within minutes" if g["has_pc"]
                         else "it starts that way on its next PC"))
            else:
                event(f"WARN release {ver} for {g['name']} refused: HTTP {s} {str(j)[:150]}")

    def paused(self, g, t):
        """Whether mining is paused on the box now while its LLM answers a prompt: box.sh said so in the last 5
        minutes (the logs come a minute or two late), and the pause began under 15 minutes ago."""
        ps = g.get("pause_since")
        return bool(ps) and t - (g.get("paused_t") or 0) < 5 * MIN and t - ps < 15 * MIN

    def pause_end(self, g):
        """When mining last stood paused for a prompt on this PC, or 0. One pause counts for 15 minutes at most: the
        CLI resumes after 10 whatever the LLM is doing, so a longer one is a miner that isn't coming back."""
        pt, ps = g.get("paused_t") or 0, g.get("pause_since")
        return min(pt, ps + 15 * MIN) if ps else pt

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
            # Then, if config asks, boxes mining on HeroMiners, cheapest first, on the PCs they have.
            mining = sorted((g for g in live if not g.get("kx") and g["has_pc"] and self.cfg["kx_in_place"]
                             and g["status"] not in ("pending", "failed") and self.kx_ready(g, t)
                             and g.get("kx_failed_tok") != self.st["kx_token"]),
                            key=lambda g: g["price"])
            for g in mining:
                if need <= 0:
                    break
                if self.kx_switch(g, True, t, "mining on HeroMiners", in_place=True):
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
        # After a switch in place the box mines on the old pool until Salad restarts it, which took about 6 minutes
        # on 20 October: its old pool's shares count as staying there only after 15.
        wait = 15 * MIN if g.get("sw_in_place") and g["sw_in_place"] == g.get("sw_t") else 5 * MIN
        if share_t(kx_row if on else hm_row) < max(since, g.get("sw_t") or 0) + wait:
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
        # The release too, so ver_sync judges the command Salad has.
        g.update(kx=on, kx_since=t if on else None, sw_t=t, sw_from=m, sw_prev=self.on_kx(g, m, t),
                 kx_share=None if on else g.get("kx_share"), ver=cmd_ver(cmd))
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

    def kx_switch(self, g, on, t, why, in_place=False):
        """Point a group's box at Kryptex (on) or back at HeroMiners: update its command, then, if it has a PC, move
        it to a new PC, where it starts with the new command. Salad's instance restart doesn't do it: on 17 October
        it answered 202 twice and the box kept running its old command. The move isn't counted toward the group's
        3, and the PC it leaves doesn't go on the bad list. In place (kx_in_place), the command alone: Salad restarts
        the box on its PC with it a few minutes later."""
        ver, command, _ = self.box(g["name"], bool(g.get("xmr")), on)
        s, j = http("PATCH", f"{CB}/{g['name']}", {"container": {"command": command}},
                    ctype="application/merge-patch+json")
        if not 200 <= s < 300:
            g["kx_try"] = t   # tried again in 15 minutes
            event(f"WARN Kryptex {'on' if on else 'off'} for {g['name']} refused: HTTP {s} {str(j)[:150]}")
            return False
        # The command carries config's Gemma and Monero pool settings too.
        g["llm"], g["xmr_kx"] = self.llm_on(g["name"]), self.xmr_kx(g.get("xmr"))
        # A box seen mining on the target pool already (pc_pool) stays on its PC.
        moving = (not in_place and bool(g["has_pc"] and g.get("instance"))
                  and self.on_kx(g, g.get("machine"), t) != on)
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
        # In place, the box mines on its PC with the old command until Salad restarts it there: readings from now on
        # count to the new pool, so the PC isn't kept with the old one.
        g.update(kx=on, ver=ver, kx_since=t if on else None, sw_t=t, sw_from=None if in_place else m, sw_prev=prev,
                 kx_share=None if on else g.get("kx_share"), sw_in_place=t if in_place else None)
        self.st["kx_any_t"] = t
        event(f"KRYPTEX {'ON' if on else 'OFF'} {g['name']} ({why}): Pearl on {'Kryptex' if on else 'HeroMiners'}; "
              + ("moved to a new PC, where it starts that way" if moving else
                 "it mines that way on this PC already" if already else
                 "switched by its command; it keeps this PC, where Salad usually restarts it that way within minutes"
                 if in_place else
                 "it starts that way on its next PC"))
        return True

    def pool_port(self, kx):
        """The port a box's port check tries: Kryptex's for a Kryptex box, HeroMiners' 1200 otherwise."""
        return int(self.cfg["kryptex_host"].rsplit(":", 1)[1]) if kx else 1200

    @staticmethod
    def bad_ports(entry):
        """The ports a bad-list entry is for, or None when the PC is bad for every box (slow, capped, not mining)."""
        if entry is None:
            return None
        return entry.get("ports") or ([entry["port"]] if "port" in entry else None)

    @staticmethod
    def bad_ths(entry):
        """The TH/s a PC was left at for its margin, or None when it was left for another reason. Entries from
        before 23 October have it only in their reason ("earns $0.056/hr at 42 TH/s on PC ...")."""
        if entry.get("ths") is not None:
            return entry["ths"]
        m = re.match(r"earns \$[\d.]+/hr at ([\d.]+) TH/s", entry.get("why", ""))
        return float(m.group(1)) if m else None

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
        pool's fee), at most every 5 minutes, and 5 minutes after a failed read (CoinGecko answers 429 when asked
        every minute)."""
        st = self.st
        if (st.get("xmr_mkt") and now() - st["xmr_mkt"]["t"] < 300) or now() - st.get("xmr_try", 0) < 300:
            return
        s1, j = http("GET", "https://monero.herominers.com/api/stats", timeout=30)
        s2, p = http("GET", "https://api.coingecko.com/api/v3/simple/price?ids=monero&vs_currencies=usd", timeout=30)
        try:
            per_hash = float(j["lastblock"]["reward"]) / 1e12 / float(j["network"]["difficulty"]) * (1 - XMR_FEE)
            price = float(p["monero"]["usd"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            err(f"Monero market fetch failed {s1} {s2}")
            st["xmr_try"] = now()
            return
        # HeroMiners' stats sometimes answer far off: on 22 October at least 15 reads gave a hash about 8 to 2,200
        # times too little, each valuing the Kryptex chart buckets first seen then, and the rules' Monero, at almost
        # nothing.
        # A hash's pay and the price don't move 3 times in minutes, so such a read is set aside and the last one
        # kept. Only figures that every read agrees on for 20 minutes are taken as real: a bad answer can last a few
        # minutes, and at 02:01Z on 23 October one was taken because the next round's read gave the same figures.
        old = st.get("xmr_mkt")
        if per_hash <= 0 or price <= 0 or (old and not (near(per_hash, old["per_hash"], 3) and near(price, old["price"], 3))):
            odd = st.get("xmr_odd")
            if not (odd and near(per_hash, odd["per_hash"], 1.25) and near(price, odd["price"], 1.25)):
                odd = st["xmr_odd"] = {"per_hash": per_hash, "price": price, "since": now()}
            if now() - odd.get("since", now()) < 20 * MIN:
                err(f"Monero market read set aside: a hash {per_hash:.3g} XMR at ${price}, against "
                    f"{(old or {}).get('per_hash', 0):.3g} at ${(old or {}).get('price', 0)}")
                return
        st.pop("xmr_odd", None)
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

    def kx_xmr(self, groups, t):
        """Kryptex's Monero pool for our Monero address, at most every 5 minutes, failed tries included: for the
        report's XMR Kryptex row, its workers' 30-minute hashrate and what it holds and has paid; for each Monero box,
        its worker's ("<group>-g0") rate there and the work Kryptex accepted from it. That work is from the worker's
        chart, in 10-minute buckets (a bucket's hashrate times 600 is its accepted hashes), each bucket valued at the
        network's pay for a hash when first seen. A bucket still filling is read again."""
        st, addr = self.st, self.cfg["xmr_wallet"]
        for g in st["groups"].values():
            if g["ended"] and "kxx_b" in g:
                del g["kxx_b"]   # its total stays; the buckets were only kept to read a filling one again
        if t - max((st.get("kxx_tot") or {}).get("t", 0), st.get("kxx_try", 0)) < 5 * MIN:
            return
        st["kxx_try"] = t
        s1, b = http("GET", f"{KXX_API}/v1/miner/balance/{addr}", timeout=30)
        paid = kx_paid(KXX_API, addr)
        s3, ws = http("GET", f"{KXX_API}/v3/miner/workers/{addr}", timeout=30)
        rows = (ws.get("results") or []) if s3 == 200 and isinstance(ws, dict) else []
        # Each box's rate there, read even when the balance or payouts fail: the rules count each box's Monero
        # from these (xmr_usd), and they went stale after 15 minutes of balance failures.
        byw = {x["worker"]: x for x in rows if isinstance(x, dict) and x.get("worker")}
        for g in groups:
            x = byw.get(f"{g['name']}-g0")
            if x:
                try:
                    g["kxx_khs"], g["kxx_seen"] = float(x.get("avg_hashrate_30m") or 0) / 1000, t
                except (TypeError, ValueError):
                    pass
        if s1 != 200 or not isinstance(b, dict) or paid is None:
            return
        try:
            st["kxx_tot"] = {"t": t, "confirmed": float(b.get("confirmed") or 0),
                             "unconfirmed": float(b.get("unconfirmed") or 0), "paid": paid,
                             "khs": sum(float(x.get("avg_hashrate_30m") or 0) for x in rows if isinstance(x, dict)) / 1000}
        except (TypeError, ValueError):
            pass
        mk = st.get("xmr_mkt")
        # The pay buckets already hold, across every box: a bucket valued 3 times off it was valued from a bad market
        # read (xmr_market), and is valued again at it; so is a new bucket while the market read is off.
        pays = sorted(ph for x in st["groups"].values() for _, ph in (x.get("kxx_b") or {}).values())
        ref = pays[len(pays) // 2] if len(pays) >= 10 else None
        for g in groups:
            if not mk or not ((g.get("xmr_kx") and g.get("had_pc")) or "kxx_b" in g):
                continue
            s, c = http("GET", f"{KXX_API}/v1/miner/chart/{addr}/{g['name']}-g0?days=2", timeout=30)
            if s != 200 or not isinstance(c, list):
                continue
            if "kxx_b" not in g:
                # The chart holds all its Kryptex work (Monero went there on 20 October, inside its 2 days), so the
                # estimates from its rate there, kept in xmr_coins until 21 October, give way to it: xmr_coins is
                # HeroMiners' count alone again, at today's pay.
                g["xmr_coins"] = g.get("xmr_hashes", 0.0) * mk["per_hash"]
            b = g.setdefault("kxx_b", {})
            per_hash = mk["per_hash"] / (1 - XMR_FEE)
            if ref and not near(per_hash, ref, 3):
                per_hash = ref
            for v in b.values():
                if ref and not near(v[1], ref, 3):
                    v[1] = ref
            for p in c:
                try:
                    ts, h = str(int(p["timestamp"])), float(p.get("hashrate") or 0) * 600
                except (KeyError, TypeError, ValueError):
                    continue
                if int(ts) / 1000 + 600 <= g["created"]:
                    continue   # a bucket that ended before the group began: an earlier group of the same name
                if h > b.get(ts, [0])[0]:
                    b[ts] = [h, b[ts][1] if ts in b else per_hash]
            g["kxx_gross"] = sum(h * ph for h, ph in b.values())

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
        kx = self.new_on_kx()
        ver, command, res = self.box(name, xmr, kx)
        res.update(storage_amount=10737418240, gpu_classes=[st["classes"][cls]["id"]])
        body = {"name": name, "display_name": name, "replicas": 1, "restart_policy": "always", "autostart_policy": True,
                "container": {"image": AMD_IMAGE if is_amd(name) else IMAGE, "priority": pri, "resources": res,
                              "command": command}}
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
                "last_start": 0, "first_share_said": False, "xmr": xmr, "kx": kx, "kx_since": t if kx else None,
                "llm": self.llm_on(name), "xmr_kx": self.xmr_kx(xmr)}
            event(f"CREATED {name}: {cls} at {pri} (${st['classes'][cls]['prices'][pri]:.3f}/hr, "
                  f"{self.margin(cls, pri, kx=kx) * 100:+.0f}% margin at {rate} TH/s), miner {ver}"
                  + (f", Monero on {res['cpu']} vCPUs" + (" to Kryptex" if self.xmr_kx(xmr) else "") if xmr else "")
                  + (", Pearl on Kryptex" if kx else "")
                  + (", Gemma" if self.llm_on(name) else ""))
            return True
        text = json.dumps(j)
        if "name_conflict" in text and tries < 20:
            return self.create(cls, pri, taken | {name}, tries + 1)
        if "quota_exceeded" in text:
            self.quota_full = True   # the search stops here: every other create this round would be refused too
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
        drops = st.setdefault("pc_drops", {})
        for m in [m for m, d in drops.items() if t - d[-1] >= 3600 * self.cfg["drop_hours"]]:
            del drops[m]
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
            # A group asks for one replica, so a second running instance is billed with no rule judging it. None
            # seen up to 23 October, but nothing would have shown one: say so once for each set of PCs.
            multi = sorted(i["machine_id"][:8] for i in on)
            if len(multi) > 1 and g.get("multi_said") != multi:
                g["multi_said"] = multi
                event(f"WARN {g['name']}: Salad runs {len(multi)} instances of it, on PCs {', '.join(multi)}; only "
                      f"{on[0]['machine_id'][:8]} is judged")
            g["has_pc"] = bool(on)
            # Salad bills an instance only while it is running ("billed per second only for container instances in
            # the 'running' state"), not while it downloads or starts: has_pc counts those too, for the rules.
            g["_run"] = any(i.get("state") == "running" and i.get("machine_id") for i in insts)
            if on:
                g["instance"] = on[0].get("instance_id") or on[0].get("id")
                g["_machine"] = on[0]["machine_id"]
            else:
                ins = insts[0] if insts else {}
                g["instance"] = ins.get("instance_id") or ins.get("id") or g.get("instance")
                g["_machine"] = None

        self.sweep(t)

        # Billing: price for every minute a group's instance is running. Counting downloading and creating as well
        # read 2.4% over Salad's own running time (system logs, 91 groups on 23 Oct): about 29 s a PC, $0.22-0.27 a
        # day, so "Salad $X left" read low (OPEN.md #66).
        for g in active:
            if g.get("_run") and not g["ended"]:
                g["spend"] += g["price"] * dt_s / 3600
                g["pc_min"] += dt_s / 60

        # One log query for every group.
        by_name = {g["name"]: g for g in active if not g["ended"]}
        # Groups deleted in the last 2 hours, for boxes that keep mining after it (see ghost_line).
        gone = {g["name"]: g for g in st["groups"].values()
                if g.get("ended") and t - g["ended"] < 2 * 3600 and g["name"] not in by_name}
        if by_name:
            for tm, name, mach, text in self.logs(list(by_name.values())):
                g = by_name.get(name)
                if not g or tm < g["created"]:
                    if name in gone:
                        self.ghost_line(gone[name], tm, mach, text)
                    continue
                if re.search(r"\[mining g\d+\] \d\d:\d\d:\d\d\s+paused for a prompt", text):
                    # Mining stopped on the card while its LLM answers a prompt (box.sh says so rather than repeat
                    # the miner's last line). It earns nothing meanwhile, and the rules wait (paused, pause_end).
                    if mach == g.get("machine"):
                        g["paused_t"] = max(g.get("paused_t") or 0, tm)
                        if not g.get("pause_since"):
                            g["pause_since"] = tm
                            event(f"PAUSED {name}: mining stopped on PC {(mach or '')[:8]} while Gemma answers a prompt")
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
                    # Salad can start a second copy of a group on a new PC and list only that one while the first
                    # keeps mining (OPEN.md #23: 3060-50 on 21 Oct, 3060-71 on 23 Oct, each then deleted on the new
                    # copy's rate). The first copy's lines still come: a live reading from another PC 5 min or
                    # more after the group got the listed one. It is counted as mined, but no rule judges it.
                    if (mach and g.get("machine") and mach != g["machine"] and g.get("has_pc")
                            and tm - (g.get("pc_since") or tm) >= 5 * MIN and mach not in g.setdefault("ghost_said", [])):
                        g["ghost_said"].append(mach)
                        event(f"WARN {name}: PC {mach[:8]} still mines for it ({ths:g} TH/s, up {up}), though Salad "
                              f"lists it on PC {g['machine'][:8]}: two copies, and only {g['machine'][:8]} is judged")
                    g["readings"].append([tm, mach, ths, acc, up])
                    if g.get("pause_since") and mach == g.get("machine") and tm > (g.get("paused_t") or 0):
                        event(f"RESUMED {name}: mining again on PC {(mach or '')[:8]} after "
                              f"{(tm - g['pause_since']) / 60:.0f} min paused for Gemma")
                        g["pause_since"] = None   # mining again
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
                blocked = re.search(r"outbound port (\d+) blocked", text)
                if blocked:
                    if mach not in g["blocked"]:
                        g["blocked"].append(mach)
                        g.setdefault("blocked_port", {})[mach] = int(blocked.group(1))   # which pool's port it tried
                        event(f"PORT BLOCKED {name} on PC {(mach or '')[:8]}: {text.strip()[:120]}")
                elif text.startswith("[card]") and mach not in g["cards"]:
                    g["cards"].append(mach)
                    event(f"CARD {name} PC {(mach or '')[:8]}: {text.strip()[7:160]}")
                elif re.match(r"\[run\] (FAIL|g\d miner exited)", text) or text.startswith("[run] release"):
                    event(f"LOG {name} PC {(mach or '')[:8]}: {text.strip()[:200]}")
                    # box.sh starting again on a PC the box mines on in this stint: Salad restarted it there (a
                    # command update, or its own restart). The box is quiet while it starts, so the reading and share
                    # rules count from here, as on a new PC, once a PC. In 12 days of events, 4 of the 39 "no mining
                    # reading for 6 min" moves came just after such a line (one of them a box that couldn't mine).
                    # Readings from an earlier stint on the same PC don't count: that is a first start.
                    if (text.startswith("[run] release") and mach and mach == g.get("machine") and g.get("restart_pc") != mach
                            and any(r[1] == mach and (g.get("pc_since") or 0) - 5 * MIN <= r[0] < tm for r in g["readings"])):
                        g["restart_t"], g["restart_pc"] = tm, mach
                        event(f"RESTARTED {name} on PC {mach[:8]}: Salad started the box again there; its readings "
                              "and shares count from now, as on a new PC")
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
        self.safetrade()
        self.salad_balance()
        self.kx_count(list(by_name.values()), t)
        for g in by_name.values():
            x, w = kw.get(f"{g['name']}-g0"), pw.get(f"{g['name']}-g0")
            self.hm_count(g, w)
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
        self.llm_sync(list(by_name.values()), t)
        self.xmr_kx_sync(list(by_name.values()), t)
        self.ver_sync(list(by_name.values()), t)
        xg = [g for g in by_name.values() if g.get("xmr")]
        if xg:
            self.xmr_market()
            self.xmr_pool(xg, t)
            self.kx_xmr(xg, t)
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
                # Salad's reason when it refuses: in the 13 days to 23 October it answered 400 to 345 of 1,314 starts,
                # and 102 groups needed two or more, but the log never said why.
                why = "" if s in (200, 202) else f" {str(j)[:200]}"
                event(f"START {g['name']}: it was stopped before getting a PC; POST /start -> HTTP {s}{why}")
            elif g["had_pc"]:
                st["credits_out"] = True
                event(f"ALERT Salad stopped {g['name']} after it had a PC ({g.get('state_desc') or 'no reason given'}). "
                      "Credit may have run out; leaving stopped groups alone until a top-up.")
            return
        if g["status"] == "failed" and not g.get("failed_said"):
            g["failed_said"] = True
            event(f"WARN {g['name']} shows status failed: {g.get('state_desc')}")

        mach = g.pop("_machine", None)
        # The PC it was just moved off can still show for a few minutes; Salad bills it, but it isn't a new PC.
        # Only that PC, and only for 10 minutes: until 22 October any PC the group had ever left was skipped, for
        # good, so 2070-3, handed back a bad PC it had left hours before, billed at 0 TH/s for 4.5 hours with no
        # rule run on it (no bad-list check, no reading rule, no delete).
        if g["has_pc"] and g.get("left") and mach == g["left"][-1] and t - (g.get("left_t") or 0) < 10 * MIN:
            return
        if g["has_pc"]:
            if mach != g["machine"]:
                first = not g["had_pc"]
                g.update(machine=mach, pc_since=t, run_since=None, lost_since=None, profit_fail=0, had_pc=True,
                         pause_since=None, paused_t=None, restart_t=None)
                # A new box there starts on the group's command, whatever the box before it mined on.
                (g.get("pc_pool") or {}).pop(mach, None)
                if g.get("sw_from") == mach:
                    g["sw_from"] = None
                event(f"{'GOT PC' if first else 'NEW PC'} {g['name']} ({g['class']}, {g['pri']}): PC {mach[:8]}")
                # A PC that another group left lately (slow, capped, not mining) is left again at once, unless it was
                # left for its rate and that rate clears the margin at this group's price and pool: until 22 October
                # a PC left at high was refused at a cheaper priority too: 21 of 681 bad-list moves to 23 October.
                bad = st.setdefault("bad_pcs", {}).get(mach)
                if (bad and t - bad["t"] < 3600 * self.cfg["bad_pc_hold_h"]
                        and self.pool_port(g.get("kx")) in (self.bad_ports(bad) or [self.pool_port(g.get("kx"))])):
                    ths, bar = self.bad_ths(bad), self.cfg["min_margin"]
                    if ths is not None and margin_of(ths * self.usd(self.on_kx(g, mach, t)) + self.xmr_usd(g), g["price"]) >= bar:
                        event(f"NOTE {g['name']} keeps PC {mach[:8]}: {bad['group']} left it at {ths:.0f} TH/s, which "
                              f"clears {bar:.0%} at this group's ${g['price']:.3f}/hr")
                    else:
                        self.reallocate(g, f"{bad['group']} left this PC at {past(bad['t'])}: {bad['why']}", count=False)
                        return
                # A bad-list entry lasts bad_pc_hold_h (24 h), but the PC's measured rate is kept rate_days (7): a PC
                # whose rate there fails the margin at this group's price and pool is left too, the same way.
                else:
                    seen = st.get("pc_rates", {}).get(g["class"], {}).get(mach)
                    if seen and t - seen[1] < 86400 * self.cfg["rate_days"]:
                        bar = self.cfg["min_margin"]
                        mg = margin_of(seen[0] * self.usd(self.on_kx(g, mach, t)) + self.xmr_usd(g), g["price"])
                        if mg < bar:
                            self.reallocate(g, f"PC {mach[:8]} made {seen[0]:.0f} TH/s {(t - seen[1]) / 3600:.0f} h ago, "
                                               f"{under(mg, bar)} at this group's ${g['price']:.3f}/hr, under {bar:.0%}",
                                            count=False)
                            return
        else:
            if g["had_pc"]:
                if g["machine"] is not None:
                    event(f"LOST PC {g['name']}: PC {g['machine'][:8]} gone (instance {g.get('inst_state', '-')})")
                    self.note_drop(g, t)
                    g.update(machine=None, pc_since=None, run_since=None)
                if g["lost_since"] is None:
                    g["lost_since"] = t
                if t - g["lost_since"] >= 20 * MIN:
                    # As a group that never got a PC: a higher priority that still clears the margin first, with 20
                    # more minutes there. Most of these never had a PC that mined: on 23 October 2070-38, 5060ti-147
                    # and 148 and the AMD box rx9070xt-3 were each turned off bad-list PCs, then deleted at their
                    # first priority 20 minutes later.
                    nxt = self.next_pri(g)
                    if nxt is None:
                        self.delete(g, "released", "lost its PC and got no other in 20 min")
                    elif self.set_priority(g, nxt, "no PC since losing one 20 min ago"):
                        g["lost_since"] = t
            elif g["status"] == "pending":
                g["pri_since"] = t  # preparing the image: not looking for a PC yet, and a priority change is refused
            elif t - g["pri_since"] >= 10 * MIN:
                nxt = self.next_pri(g)
                if nxt is None:
                    self.delete(g, "released", f"no PC at {g['pri']} and no higher priority clears "
                                               f"{self.cfg['min_margin']:.0%}")
                else:
                    self.set_priority(g, nxt)
            return

        # The clocks below start when the container runs on the PC, not when Salad hands the PC over: Salad first
        # downloads the image there (has_pc counts downloading and creating), and bills nothing until it runs. The
        # AMD image is 1.16 GB against 31 MB for the NVIDIA one: on 28 October rx9060xt was left 10 minutes after
        # getting PC e9687de9 with no container output at all, and its class barred for a day as failed.
        if g.get("_run", True):
            if not g.get("run_since"):
                # A group already on its PC before this rule came in has no run_since at all: its clock stays.
                g["run_since"] = g["pc_since"] if "run_since" not in g else t
        else:
            if t - g["pc_since"] >= 30 * MIN:
                why = f"not running {(t - g['pc_since']) / 60:.0f} min after getting PC {g['machine'][:8]} (image still loading)"
                self.st.setdefault("bad_pcs", {})[g["machine"]] = {"t": t, "group": g["name"], "why": why}
                self.reallocate(g, why, count=False)   # the PC's line, not the class's fault
            return
        start = max(g["pc_since"], g["run_since"])
        on = t - start
        # Rule 6: the PC blocks the pool port. It goes on the bad list like any PC a group leaves: Salad hands the
        # same PCs out again. On 18–19 October one blocked PC cost six groups a move each, and two groups were
        # deleted on a PC another group had already found blocked.
        # A box checks the port of the pool its command names: 1200 for HeroMiners, Kryptex's own for a Kryptex box.
        if g["machine"] in g["blocked"]:
            port = (g.get("blocked_port") or {}).get(g["machine"], 1200)
            why = f"PC blocks outbound port {port}"
            # Kryptex's port unreachable while nothing shows Kryptex took any box's work in 30 min may be Kryptex
            # itself: that counts as a Kryptex failure (back to HeroMiners), as a box with no shares would.
            reached = self.kx_works(g, t) or self.kx_proof(g, t) >= t - 30 * MIN
            if port == self.pool_port(True) and g.get("kx") and not reached and self.kx_failed(g, why):
                return
            # The PC goes on the bad list for boxes that check the same port; a box on the other pool may use it. A PC
            # can block both: on 19 October e8c084ab blocked 1200 for a HeroMiners box and then 7048 for a Kryptex box.
            bad = st.setdefault("bad_pcs", {})
            old = bad.get(g["machine"])
            held = old is not None and t - old["t"] < 3600 * self.cfg["bad_pc_hold_h"]
            if not (held and self.bad_ports(old) is None):   # a PC already bad for every box stays so
                ports = sorted({port} | set((self.bad_ports(old) or []) if held else []))
                why = "PC blocks outbound port" + ("s " if len(ports) > 1 else " ") + " and ".join(map(str, ports))
                bad[g["machine"]] = {"t": t, "group": g["name"], "why": why, "ports": ports}
            # The PC's fault, not the group's: the move doesn't use one of its 3 (on 22 October 4060-54 was deleted
            # on its fourth PC for a blocked port). Up to 6 such moves; after that a group that keeps landing on
            # blocked PCs counts them, so it can't move for ever.
            g["port_moves"] = g.get("port_moves", 0) + 1
            self.reallocate(g, f"PC blocks outbound port {port}", count=g["port_moves"] > 6)
            return
        # This PC's readings (one a minute). A PC that isn't mining 10 min after we got it is left.
        # Readings come from Salad's log service, which sometimes fails for minutes at a time. Missing readings
        # then say nothing about the miner, so these two rules count only up to the last log read that worked,
        # and leave nothing while the pool still gets shares from this PC (in the last 6 min).
        seen_to = min(t, st.get("last_log_ok") or t)
        ls = g["last_share"] or 0
        sharing = ls >= g["pc_since"] and t - ls < 6 * MIN
        cur = [r for r in g["readings"] if r[1] == g["machine"] and r[2] > 0 and r[0] >= g["pc_since"] - 5 * MIN]
        # Time mining stood paused for a prompt doesn't count against a box: these three rules count from its end.
        pend = self.pause_end(g)
        # After a restart on this PC (restart_t) they run as on a new PC from then: earlier readings don't count.
        rs = g.get("restart_t") or 0
        cur = [r for r in cur if r[0] > rs]
        if not cur and seen_to - max(start, pend, rs) >= 10 * MIN and not sharing:
            self.leave(g, f"no mining reading {(t - rs) / 60:.0f} min after its restart on PC {g['machine'][:8]}" if rs
                       else f"no mining reading {on / 60:.0f} min after getting PC {g['machine'][:8]}" if start == g["pc_since"]
                       else f"no mining reading {on / 60:.0f} min after it started on PC {g['machine'][:8]} "
                            f"({(start - g['pc_since']) / 60:.0f} min loading first)")
            return
        # A miner that stops (the card is taken back, the driver fails) stops sending readings: leave the PC
        # once the newest is 6 min old.
        if cur and seen_to - max(cur[-1][0], pend) >= 6 * MIN and not sharing:
            self.leave(g, f"no mining reading for {(t - cur[-1][0]) / 60:.0f} min on PC {g['machine'][:8]}")
            return
        # Rule 4: no share after 20 min on a PC.
        if on >= 20 * MIN:
            gap = SHARE_UNITS / (g["rate"] / UNIT_TH)
            limit = max(10 * MIN, 8 * gap)
            ref = max(start, g["last_share"] or 0, pend, rs)
            if t - ref >= limit:
                self.leave(g, f"no share for {(t - ref) / 60:.0f} min on PC {g['machine'][:8]}")
                return
        # Rule 5: only PCs that clear the margin. After the miner's first 2 minutes (warm-up), the median of the last
        # 3 readings must clear min_margin at the PC's price, every round; the first time it doesn't, the PC is left.
        # The median, not the mean: in the 12.6 days to 23 October, 24 of 277 margin leaves whose readings were kept
        # came from one reading 5% or more under the other two (one minute at 3 TH/s between two at 133), where the
        # median cleared the bar. A PC that really slows shows it in 2 of 3 readings a minute later.
        # A pause for a prompt starts the warm-up again, from the first reading after it: those can be low.
        back = next((r[0] for r in cur if r[0] > pend), cur[0][0]) if cur else 0
        post = [r for r in cur if r[0] >= back + 2 * MIN] if cur else []
        if len(post) >= 3:
            # At the pool the box's PC mines on, which can differ from its command's for a while (see on_kx).
            # Its Monero counts too (xmr_usd).
            usd, bar, xmr = self.usd(self.on_kx(g, g["machine"], t)), self.cfg["min_margin"], self.xmr_usd(g)
            ths = median3(post)
            if margin_of(ths * usd + xmr, g["price"]) < bar:
                self.leave(g, f"earns ${ths * usd + xmr:.3f}/hr at {ths:.0f} TH/s" + (f" (Monero ${xmr:.3f} of it)" if xmr else "")
                              + f" on PC {g['machine'][:8]} against its ${g['price']:.3f}/hr "
                              f"({under(margin_of(ths * usd + xmr, g['price']), bar)}, under {bar:.0%})",
                           ths=ths)
                return
            # A group still on its PC has cleared the bar every round since its warm-up there (the first round it
            # doesn't, it leaves), so after 2 h on it its count of moves starts again. The count never reset until
            # 24 October: 3060-72 mined at +17–20% for hours and was deleted on the first slow PC it got, its 3 moves
            # used on earlier days (OPEN.md #68).
            if g["reallocs"] and t - max(g["pc_since"], rs) >= 2 * 3600:
                event(f"MOVES {g['name']}: 2 h above {bar:.0%} on PC {g['machine'][:8]}, so its count of moves starts "
                      f"again (it had used {g['reallocs']} of 3)")
                g["reallocs"] = 0
            ten = post[-10:]
            shared = (g["last_share"] or 0) >= g["pc_since"]   # the pool accepts its work, not just the miner's figure
            if not self.paid(g) and shared and len(ten) == 10 and margin_of(sum(r[2] for r in ten) / 10 * usd + xmr, g["price"]) >= bar:
                st["paid_classes"].append(g["class"])
                event(f"PAID {g['name']}: {g['class']} clears {bar:.0%} here ({ths:.0f} TH/s); it counts as paid now")

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
        """The median TH/s of the miner's last 3 readings on this PC after warm-up, as rule 5 judges it, or None while
        there aren't 3."""
        if not g.get("has_pc") or not g.get("machine") or not g.get("pc_since"):
            return None
        cur = [r for r in g["readings"] if r[1] == g["machine"] and r[2] > 0 and r[0] >= g["pc_since"] - 5 * MIN]
        # The readings in the 2 minutes after a pause for a prompt are a warm-up, as rule 5 treats them: one dip
        # (0.1 TH/s on 3060-70 at 00:55Z on 23 October) could otherwise stand as the PC's rate for 7 days.
        pend = self.pause_end(g)
        back = next((r[0] for r in cur if r[0] > pend), None) if pend else None
        post = [r for r in cur if r[0] >= cur[0][0] + 2 * MIN and not (back and back <= r[0] < back + 2 * MIN)] if cur else []
        if len(post) < 3:
            return None
        return median3(post)

    def pc_margin(self, g):
        """The margin at pc_rate, or None while there aren't 3 readings after warm-up."""
        rate = self.pc_rate(g)
        return None if rate is None else margin_of(rate * self.usd(self.on_kx(g, g["machine"], now())) + self.xmr_usd(g),
                                                   g["price"])

    def unplaced(self, cls, t):
        """True when the class's last no_pc_streak groups all ended without ever getting a PC, the latest within
        no_pc_hold_min: the search then leaves the class out, so it stops taking a slot it can't fill."""
        ended = sorted((g for g in self.st["groups"].values() if g["class"] == cls and g["ended"]),
                       key=lambda g: g["ended"], reverse=True)[:self.cfg["no_pc_streak"]]
        return (len(ended) == self.cfg["no_pc_streak"] and all(g["end_kind"] == "released" and not g["had_pc"] for g in ended)
                and t - ended[0]["ended"] < 60 * self.cfg["no_pc_hold_min"])

    def note_drop(self, g, t):
        """A PC Salad takes back within drop_min of handing it over is a drop; drop_count of them in drop_hours put
        it on the bad list for every box (bad_pc_hold_h). On 17-19 October 3323a855 was handed to 3090 Ti groups 19
        times in 38 hours and taken back within minutes nearly every time, each stint billed with nothing mined. A
        PC taken back once isn't bad: in 14 days to 23 October such short stints cost $0.29/day, but holding a PC
        after 3 in a day saved $0.02/day of it and blocked one longer stint."""
        mach, since = g["machine"], g.get("pc_since")
        if not since or t - since >= 60 * self.cfg["drop_min"]:
            return
        st = self.st
        drops = [d for d in st.setdefault("pc_drops", {}).get(mach, []) if t - d < 3600 * self.cfg["drop_hours"]] + [t]
        st["pc_drops"][mach] = drops
        if len(drops) < self.cfg["drop_count"]:
            return
        del st["pc_drops"][mach]
        bad = st.setdefault("bad_pcs", {})
        if bad.get(mach) and self.bad_ports(bad[mach]) is None and self.bad_ths(bad[mach]) is None:
            return   # already bad for every box
        why = (f"Salad took it back within {self.cfg['drop_min']} min of handing it over {len(drops)} times in "
               f"{self.cfg['drop_hours']} h")
        bad[mach] = {"t": t, "group": g["name"], "why": why}
        event(f"NOTE PC {mach[:8]} goes on the bad list: {why}")

    def leave(self, g, why, ths=None):
        """Stop paying for this PC: another PC if the class has paid before, otherwise delete the group. The PC goes
        on the bad list, so another group that gets it leaves it at once (for bad_pc_hold_h); with `ths`, the rate
        it was left at, only a group that rate doesn't clear the margin for."""
        # A Kryptex box that has had Kryptex shares and then stops, stops while other boxes' work reaches Kryptex,
        # or mines on HeroMiners at this PC (pc_pool) has a PC problem: it is left as usual below.
        if (g.get("kx") and why.startswith(("no share", "no mining reading"))
                and self.kx_proof(g, now()) < (g.get("kx_since") or g["created"]) and not self.kx_works(g, now())
                and (g.get("pc_pool") or {}).get(g.get("machine")) is not False and self.kx_failed(g, why)):
            return
        if g.get("machine"):
            entry = {"t": now(), "group": g["name"], "why": why}
            if ths is not None:
                entry["ths"] = ths
                # And it's the PC's rate now (pc_rates), for the 7-day check and the class's rate: a PC left at its
                # first judgment never had one recorded, so the check missed most of the PCs it was for (60% had none).
                self.st.setdefault("pc_rates", {}).setdefault(g["class"], {})[g["machine"]] = [ths, now()]
            self.st.setdefault("bad_pcs", {})[g["machine"]] = entry
        if self.paid(g):
            self.reallocate(g, why)
        else:
            # Left for its rate, the PC was at fault and the lists above bar it (24 h bad list, 7 days in pc_rates),
            # and rate() drops a class once 3 of its PCs are slow: so the class isn't held 24 h as failed. On 23
            # October a 9070 XT left at 46 TH/s on PC f6f1e2e2 (others ran 88-111) held the class out for a day
            # (OPEN.md #67). A group that never mined or shared still fails, and holds its class.
            self.delete(g, "released" if ths is not None else "failed", why)

    def search(self):
        st = self.st
        active = [g for g in st["groups"].values() if not g["ended"]]
        s, q = http("GET", f"{BASE}/quotas")
        c = (q or {}).get("container_groups_quotas", {}) if s == 200 else {}
        quota = c.get("container_replicas_quota", 10)
        st["quota"] = quota
        # Salad's own count of replicas in use, when it gives one: on 22 October it counted one more than we had
        # at times all day, so every search sent creates it then refused (created_replicas_quota_exceeded).
        used = c.get("container_replicas_used")
        room = min(self.cfg["max_active"] - len(active), quota - (len(active) if used is None else used))
        t = now()
        this_hour = sum(1 for g in st["groups"].values() if g["created"] >= t - t % 3600)
        room = min(room, self.cfg["max_new_per_hour"] - this_hour)   # however the search was started
        if room <= 0:
            return
        made = 0
        self.quota_full = False
        for *_, cls, p in self.candidates(active, t):
            if made >= room or st["credits_out"] or self.quota_full:
                break
            # Priced at the pool a new group starts on; a create can fill the last Kryptex slot, so check again.
            if made and self.pick_margin(cls, p) < self.cfg["min_margin"]:
                continue
            if self.create(cls, p):
                made += 1
        if not made:
            event(f"SEARCH found nothing to create (room {room})")

    def candidates(self, active, t):
        """Classes a new group may use, each at its cheapest free priority from start_pri that clears the margin, best first:
        paid classes, then by expected profit an hour. A class whose last no_pc_streak groups got no PC is left out
        for no_pc_hold_min (unplaced). When no priority from
        start_pri up clears it, the dearest cheaper one that does and has a GPU free: on 19 October 5090s started at
        high, where their last PC came from, which cleared 1% at a lower PRL price, so with 13 free at medium (+18%)
        no 5090 was tried."""
        st = self.st
        # A class is tried again retry_after_min after a group of it ended for want of a PC (released), and
        # retry_failed_h after one failed. A failure used to bar the class for good: on 19 October 7 of 21
        # classes were barred, two by a single PC that never mined a week before.
        def held(g):
            wait = 60 * self.cfg["retry_after_min"] if g["end_kind"] == "released" else 3600 * self.cfg["retry_failed_h"]
            return t - g["ended"] < wait
        skip = {g["class"] for g in st["groups"].values() if not g["ended"] or held(g)}
        # A class whose groups are all on a PC clearing the margin may get another group, up to max_per_class.
        for cls in {g["class"] for g in active}:
            mine = [g for g in active if g["class"] == cls]
            if len(mine) < self.cfg["max_per_class"] and all((self.pc_margin(g) or -1) >= self.cfg["min_margin"] for g in mine) \
                    and not any(g["ended"] and g["end_kind"] == "failed" and held(g)
                                for g in st["groups"].values() if g["class"] == cls):
                skip.discard(cls)
        cands = []
        for cls in KNOWN:
            # Until 22 October such a class only went last, and still took any slot nothing else wanted: that day
            # every allowed slot was full for 722-723 min, in 324 of them with such a group holding one, and 3 times
            # a class that gets PCs was refused while one did.
            if cls not in st["classes"] or cls in skip:
                continue
            # But a class held only for tries cheaper than the priority it last got a PC at is tried again at that
            # priority or above, once it clears: 4060-32 waited 64 min on 19 October for tries at low.
            order = None
            if self.unplaced(cls, t):
                pl = PRIS.index(self.start_pri(cls))
                streak = sorted((x for x in st["groups"].values() if x["class"] == cls and x["ended"]),
                                key=lambda x: x["ended"], reverse=True)[:self.cfg["no_pc_streak"]]
                if not all(PRIS.index(x["pri"]) < pl for x in streak):
                    continue
                order = PRIS[pl:]
            rate = self.rate(cls)
            body = {"cpu": 2, "memory": 4096, "storage_amount": 10737418240, "gpu_classes": [st["classes"][cls]["id"]]}
            s, a = http("POST", f"{BASE}/availability/sce-gpu-availability", body)
            if s != 200:
                continue
            start = PRIS.index(self.start_pri(cls))
            for p in order or PRIS[start:] + PRIS[:start][::-1]:
                if (a.get(f"available_gpu_{p}") or 0) > 0 and self.pick_margin(cls, p) >= self.cfg["min_margin"]:
                    profit = rate * self.usd() + self.xmr_usd(cls=cls) - st["classes"][cls]["prices"][p]
                    cands.append((not self.paid(cls), -profit, cls, p))
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
        u = self.usd()   # for the class a new group would be
        active = [g for g in st["groups"].values() if not g["ended"]]
        mining = [(self.pc_rate(g) * self.usd(self.on_kx(g, g["machine"], t)) + self.xmr_usd(g) - g["price"], g) for g in active
                  if self.pc_rate(g) is not None]
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
        for _, _, cls, p in self.candidates(active, t):
            # Only a class Salad has placed for us, and not while its latest try got no PC.
            if cls == g["class"] or cls not in placed or (latest[cls]["ended"] and not latest[cls]["had_pc"]):
                continue
            price = st["classes"][cls]["prices"][PRIS[max(PRIS.index(p), PRIS.index(placed[cls]), 1)]]
            expect = self.rate(cls) * u + self.xmr_usd(cls=cls) - price
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
        """A heading, then one table: a row per box with what it has earned in PRL and XMR, and a total, the pools'
        own figures under it. Every row fits on one line in a chat window."""
        st, t = self.st, now()
        mk = st["mkt"]
        # Profit/hr counts on what the pool credits: the network's rate times its credit ratio. The rules may
        # use a different yield (config "yield"); the table shows what we expect to be paid.
        hy, ky = self.credit_ratio(), self.kx_ratio()
        yg = lambda g: ky if g.get("kx") else hy   # each box at its own pool's credit ratio
        price = mk["price"]
        # Monero on the CPU: its earnings count in each box's profit an hour, once config has an address.
        xon, xm = bool(self.cfg["xmr_wallet"]), st.get("xmr_mkt") or {}
        xusd = xm.get("usd_khs_hr", 0.0)
        groups = list(st["groups"].values())
        spend = sum(g["spend"] for g in groups)

        def coins(g):
            """The PRL a box has earned by its pools' own counts, and whether that's starred: its Kryptex shares
            at what Kryptex pays a share, exact under PPS+ (but for its share of block fees), and its HeroMiners
            work at HeroMiners' credit ratio, which follows that pool's luck. A group that ended before the pools'
            counts were kept (20 October) has only the ledger's estimate."""
            if "hm_units" not in g and "kx_gross" not in g:
                return self.credited(g, hy, ky), g.get("prl", 0.0) > 0
            h = g.get("hm_gross", 0.0) * hy
            return g.get("kx_gross", 0.0) * self.cfg["kryptex_pay"] + h, h > 0

        def prl_cell(gs, fmt=str):
            got = [coins(g) for g in gs]
            return fmt(f"{sum(c for c, _ in got):.3f}" + (EST if any(s for _, s in got) else ""))

        def xmr_coins(g):
            """The XMR a box has earned, and whether that's starred: its Kryptex work from its worker's chart (see
            kx_xmr) at what Kryptex pays (kryptex_xmr_pay), exact under PPS+ but for its share of block fees, and its
            Monero work as HeroMiners counted it, at the network's pay less its fee, which HeroMiners pays by its
            luck. A group that ended before the charts were read (21 October) has its Kryptex rate over time there."""
            h = g.get("xmr_coins", 0.0)
            return g.get("kxx_gross", 0.0) * self.cfg["kryptex_xmr_pay"] + h, h > 0

        def xmr_cell(gs, fmt=str):
            got = [xmr_coins(g) for g in gs]
            return [fmt(f"{sum(c for c, _ in got):.6f}" + (EST if any(s for _, s in got) else ""))] if xon else []

        rows = []
        all_ths = all_cost = all_earn = run_cost = all_khs = 0.0
        mining = 0
        live = sorted((g for g in groups if not g["ended"]), key=lambda g: g["created"])
        # (K) tells Kryptex boxes from HeroMiners ones, so it shows only while some box is on HeroMiners.
        kmark = any(not g.get("kx") for g in live)
        for g in live:
            cur = [r for r in g["readings"] if r[1] == g["machine"] and r[2] > 0 and r[0] >= t - 10 * MIN] if g["has_pc"] else []
            ths = sum(r[2] for r in cur) / len(cur) if cur else 0.0
            cost = g["price"] if g["has_pc"] else 0.0
            # Its rate at each Monero pool that saw it in the last 15 minutes: one, or both just after a switch.
            khs = sum(g.get(k, 0.0) for k, seen in (("xmr_khs", "xmr_seen"), ("kxx_khs", "kxx_seen"))
                      if g.get("xmr") and g["has_pc"] and t - (g.get(seen) or 0) < 15 * MIN)
            earn = ths * mk["usd_th_hr"] * yg(g) + (self.box_khs(g, t) * xusd if g.get("xmr") and g["has_pc"] else 0.0)
            state = ("⏹️" if g["status"] == "stopped" else ("🧠" if self.paused(g, t) else "⛏️") if g["has_pc"]
                     else "⏳")
            # A box on a new PC with no reading yet is starting: billed, but left out of the profit total. So is one
            # paused for a prompt: Gemma's speed test on a new PC can take 10 minutes or more, and with only Monero
            # earning meanwhile its margin read -3180% on 22 October and turned the All row's margin negative.
            start = bool(g["has_pc"] and not cur and (state == "🧠" or (g.get("pc_since") and
                                                                         t - g["pc_since"] < 15 * MIN)))
            if g["has_pc"] and cur:
                mining += 1
            all_ths += ths; all_cost += cost; all_earn += earn; all_khs += khs
            run_cost += 0.0 if start else cost
            # One cell for the box: state, group name less "biz-rtx" and priority, joined by no-break spaces.
            pri = "med" if g["pri"] == "medium" else g["pri"]
            box = NB.join([state, re.sub(r"^biz-(rtx)?", "", g["name"]), pri] + (["(K)"] if kmark and g.get("kx") else []))
            # Each coin's speed (Pearl on the GPU, Monero on the CPU), the bill, profit and margin, then each coin
            # earned so far.
            kc = [f"{khs:.1f}" if g.get("xmr") else "—"] if xon else []
            if not g["has_pc"]:
                cells = ["—"] + (["—"] if xon else []) + ["—", "—"]
            elif start:
                cells = ["0"] + kc + [f"{cost:.2f}", "—"]
            else:
                # No margin without a Pearl reading: Monero alone makes a margin of thousands of percent below zero.
                cells = [f"{ths:.0f}"] + kc + [f"{cost:.2f}", with_margin(earn - cost, earn if ths > 0 else 0)]
            cells = " | ".join(cells + [prl_cell([g])] + xmr_cell([g]))
            rows.append((0.0 if start or not g["has_pc"] else earn - cost, f"| {box} | {cells} |"))
        # Most profitable an hour first; groups waiting for a PC make nothing, so they sit above any losing.
        lines = [r for _, r in sorted(rows, key=lambda x: -x[0])]
        # A row for every slot, so a slot the search couldn't fill shows as empty instead of vanishing.
        slots = max(len(live), min(self.cfg["max_active"], st.get("quota") or self.cfg["max_active"]))
        empty = slots - len(live)
        lines += [f"| ⬜{NB}empty |" + " — |" * (6 if xon else 4)] * empty
        ended = [g for g in groups if g["ended"]]
        if ended:
            # Groups that have ended, as one row with their count and what they earned; the events say which were
            # released or failed. Called "Others" until 24 October, which read like other people's miners (OPEN.md #71).
            lines.append(f"| Ended{NB}×{len(ended)} |" + " — |" * (4 if xon else 3)
                         + " " + " | ".join([prl_cell(ended)] + xmr_cell(ended)) + " |")
        hr_all = all_earn - run_cost
        lines.append(f"| **All** | {bold(f'{all_ths:.0f}')} |" + (f" {bold(f'{all_khs:.1f}')} |" if xon else "")
                     + f" {bold(f'{all_cost:.2f}')} | {bold(with_margin(hr_all, all_earn))} | "
                     + " | ".join([prl_cell(groups, bold)] + xmr_cell(groups, bold)) + " |")
        # Salad's own balance when it was read in the last hour (no star), else our count (starred).
        sb = st.get("salad_bal")
        credit, cmark = (sb["usd"], "") if sb and t - sb["t"] < 3600 else (self.credit_count(), EST)
        hours = credit / all_cost if all_cost > 0 else None
        # Out of all the credit bought, when the portal's invoices have been read: $202.37/$500.
        b = st.get("salad_bought")
        of_bought = "" if not b else f"/${b:.0f}" if b == round(b) else f"/${b:.2f}"
        # What each pool reports for the account: its own hashrate (1-hour average) and the coins it holds and paid.
        pt, xt = st.get("pool_tot"), st.get("xmr_tot") if xon else None
        # Kryptex shows once a live box is on it or some PRL was mined there; an ended group that never mined
        # there (its switch failed) doesn't count.
        kt = st.get("kx_tot") if any((g.get("kx") and not g["ended"]) or g.get("prl_kx") for g in groups) else None
        pool = []
        # Each pool's name links to that pool's page for our address.
        wallet, xwallet = self.cfg["wallet"], self.cfg.get("xmr_wallet")
        sa = st.get("st_acct")

        if pt:
            name = f"[PRL{NB + 'HeroMiners' if kt else ''}]({HM_PAGE.format(coin='pearl', addr=wallet)})"
            pool.append(f"| ⛏️ {name} | {pt.get('ths_1h', 0):.0f} TH/s | {pt['balance']:.3f} "
                        f"| {pt['unconfirmed']:.3f} | {pt['paid']:.3f} |")
        if kt:
            # Kryptex gives each worker's hashrate now, not an account average: the sum over our Kryptex boxes.
            kths = sum(g.get("pool_ths") or 0 for g in live if g["has_pc"] and self.on_kx(g, g.get("machine"), now()))
            pool.append(f"| ⛏️ [PRL{NB}Kryptex]({KX_PAGE.format(coin='prl', addr=wallet)}) | {kths:.0f} TH/s "
                        f"| {kt['confirmed']:.3f} | {kt['unconfirmed']:.3f} | {kt['paid']:.3f} |")
        if xt:
            pool.append(f"| 🪙 [XMR HeroMiners]({HM_PAGE.format(coin='monero', addr=xwallet)}) | {xt.get('khs_1h', 0):.1f} kH/s "
                        f"| {xt['balance']:.6f} | {xt.get('pending', 0):.6f} | {xt['paid']:.6f} |")
        kxx = st.get("kxx_tot") if xon else None
        if kxx:
            pool.append(f"| 🪙 [XMR Kryptex]({KX_PAGE.format(coin='xmr', addr=xwallet)}) | {kxx.get('khs', 0):.1f} kH/s "
                        f"| {kxx['confirmed']:.6f} | {kxx['unconfirmed']:.6f} | {kxx['paid']:.6f} |")
        if pool:
            pool = ["", "| Pool | Hashrate (1h) | Confirmed | Pending | Paid |", "|---|---|---|---|---|", *pool]
        # SafeTrade in one table, a row a line. A coin a row in the order a coin moves: what the pools paid (Paid), what
        # reached SafeTrade, what waits there to sell (held and in open orders), what sold, its trading fees and the USDT
        # it brought after them. Then a row for each currency withdrawn, with its fees (for a stablecoin bought with USDT
        # first, that trade's fee too) and the USDT it took. Last, Left: the USDT still there and every fee paid. So the
        # USDT column is a ledger: on 26 October +$357.11 + $6.42 − $110.27 − $209.28 = $43.98, the account's balance.
        # Until 26 October the coins were a 💵 row under each coin's pools, with the sales in Paid. No table without a
        # SafeTrade key, or with a read from before 24 Oct 10:40 am EDT (last-day sales only).
        if sa and "sold" in sa:
            read = f"{NB}(read{NB}{short_clock(sa['t'], t)})" if t - sa["t"] >= 15 * MIN else ""
            # Both pools' paid, Kryptex's even with no box on it now (its row is hidden then, its payouts aren't).
            kxp = st.get("kx_tot")
            paid = {"pearl": (pt["paid"] if pt else 0.0) + (kxp["paid"] if kxp else 0.0),
                    "xmr": (xt["paid"] if xt else 0.0) + (kxx["paid"] if kxx else 0.0)}
            rows, fees = [], 0.0
            for coin in ["pearl"] + (["xmr"] if xon else []):
                dp, (b, l) = COIN_DP[coin], sa["coins"].get(coin, [0.0, 0.0])
                n, usd, fee = (sa["sold"].get(coin, []) + [0.0, 0.0, 0.0])[:3]
                r = (sa.get("recv") or {}).get(coin)
                rows.append(f"| {coin_name(coin)} | {paid[coin]:.{dp}f} | " + ("—" if r is None else f"{r:.{dp}f}")
                            + f" | {b + l:.{dp}f} | {n:.{dp}f} | ${fee:.2f} | +${usd:.2f} |")
                fees += round(fee, 2)   # the shown cents, so the column adds up on screen
            # Each withdrawal shows what reached the bank: the amount sent, with the network's fee (taken on top)
            # and, for a stablecoin bought with USDT, that trade's fee in Fees. Total out adds them up.
            sent, out_fees = 0.0, 0.0
            outs = sa.get("out") or {}
            for cur, (amt, wfee) in sorted(outs.items(), key=lambda kv: kv[0] != "usdc"):
                if cur.endswith(">salad"):
                    continue    # below, after the bank's total
                if cur == "usdt":
                    fee = wfee
                elif cur in (sa.get("bought") or {}):
                    fee = sa["bought"][cur][1] + wfee
                else:
                    continue   # a coin withdrawn as itself never became USDT
                rows.append(f"| {cur.upper()} out | | | | | ${fee:.2f} | −${amt:.2f} |")
                sent += round(amt, 2)       # the shown cents, so the column adds up on screen
                out_fees += round(fee, 2)
            if sent:
                rows.append(f"| **Total out** | | | | | {bold(f'${out_fees:.2f}')} | {bold(f'−${sent:.2f}')} |")
            fees += out_fees
            # Withdrawals to the Salad top-up wallet, which paid for credit: a row of their own, out of Total out.
            for cur, (amt, wfee) in sorted(outs.items()):
                if cur.endswith(">salad"):
                    rows.append(f"| {cur.split('>')[0].upper()} to Salad | | | | | ${wfee:.2f} | −${amt:.2f} |")
                    fees += round(wfee, 2)
            left = sum(sa["coins"].get("usdt", [0.0, 0.0]))
            pool += ["", f"| 💵 [SafeTrade](https://safetrade.com){read} | Paid | Received | Waiting | Sold | Fees | USDT |",
                     "|---|---|---|---|---|---|---|", *rows,
                     f"| **Left** | | | | | {bold(f'${fees:.2f}')} | {bold(f'${left:.2f}')} |"]

        # Just what the table doesn't show: profit an hour, margin and profit so far are in its All row. The Salad
        # credit left (with how long it lasts at the All row's $/hr: days, or hours under a day) ends the line. Until 20 October it was under
        # the tables with what Salad had cost and what each PRL cost, and before that in a paragraph that also gave
        # each pool's credit ratio and payout threshold, SafeTrade's sales and the XMR mined; the key to the
        # symbols is in the README.
        head = (f"**{stamp(t)}** · PRL ${price:.2f} · {mining}/{slots} mining · Salad ${credit:.2f}{cmark}{of_bought} left"
                + (f" ({lasts(hours, cmark)})" if hours is not None else ""))
        cols = ["Box", "TH/s"] + (["kH/s"] if xon else []) + ["$/hr", "Profit/hr (margin)", "PRL"] + (["XMR"] if xon else [])
        lines = [head, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols), *lines, *pool]
        if st["credits_out"]:
            lines += ["", "⚠️ Credit ran out: Salad stopped the groups. They restart after a top-up."]
        with open(REPORT + ".tmp", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(REPORT + ".tmp", REPORT)
        if hours is not None and hours < 12 and t - st["last_credit_alert"] > 3600:
            st["last_credit_alert"] = t
            event(f"ALERT credit ${credit:.2f} lasts about {hours:.1f} h at ${all_cost:.3f}/hr")


def lasts(hours, mark=""):
    """How long the credit lasts, as the founder asked on 24 Oct: whole days, or hours when under a day ("363 h"
    read poorly)."""
    days = round(hours / 24)
    if hours < 24 or days < 1:
        return f"{hours:.0f}{mark} h"
    return f"{days}{mark} day{'s' if days != 1 else ''}"


def bold(text):
    """Markdown bold. The estimate stars inside are escaped, or Markdown reads them as more emphasis
    and "**+0.27* (+24%*)**" doesn't come out bold."""
    return "**" + text.replace("*", "\\*") + "**"


def with_margin(profit, earn):
    """Profit an hour with its margin in brackets, or alone when there's no margin to show."""
    m = star(pct(profit, earn))
    return f"{profit:+.2f}{EST}" + (f" ({m})" if m != "—" else "")


def near(a, b, f):
    """Whether a and b are both positive and within a factor f of each other."""
    return a > 0 and b > 0 and 1 / f < a / b < f


def margin_of(earn, price):
    """Profit as a share of earnings, as the first account measured it: (earn - price) / earn."""
    return (earn - price) / earn if earn > 0 else float("-inf")


def under(m, bar):
    """A margin under the bar, as a message shows it: whole points, or one decimal rounded down where whole points
    would read as the bar (a box was left as "+10%, under 10%" on 22 October)."""
    return f"{m * 100:+.0f}%" if round(m * 100) < round(bar * 100) else f"{math.floor(m * 1000) / 10:+.1f}%"


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
