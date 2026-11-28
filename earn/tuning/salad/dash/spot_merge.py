"""Fleet spot checks into the official rows (founder, 18 Nov): a card keeps its two newest races. A spot check on the
current versions replaces the card's older official row (both official rows are from 15 Nov, so the second one
goes first); two spot checks replace both. Every row carries the versions it was raced on (r["ver"]), so the
dashboard and the site colour a figure only when its own row is behind, not the whole column."""
import re

OFFICIAL_VER = {"ours": "0.7.3", "peak": "2.20.9", "srb": "3.8.8", "ozm": "1.8.6"}


def _cell(v, default):
    """A table cell and the version it raced: "143.0 (3.9.0)" -> ("143.0", "3.9.0"), "143.0" -> ("143.0", default)."""
    m = re.match(r"^(.*?)\s*\(([0-9]+\.[0-9]+\.[0-9]+)\)$", v)
    return (m.group(1), m.group(2)) if m else (v, default)


def spot_rows(md):
    """The 'Fleet spot checks' table in benchmark.md: one dict a race, oldest first."""
    i = md.index("## Fleet spot checks: SRBMiner ")
    sec = md[i:md.index("\n## ", i + 5)]
    srbv = re.search(r"SRBMiner ([0-9.]+)", sec.splitlines()[0]).group(1)
    base_ours = (re.search(r"races it inside production boxes: v([0-9.]+)", sec.replace("\n", " ")) or [None, "0.7.3"])[1]
    out = []
    for line in sec.splitlines():
        if not line.startswith("| RTX") and not line.startswith("| RX") and not line.startswith("| Arc"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        name = re.sub(r"\s*\(.*\)$", "", c[0]).replace("SUPER", "Super")
        mem = re.search(r"\((\d+ GB)", c[0])
        # The tag is a release ("(v0.7.5)") or the commit of a pre-release build ("(35a6a708)", 20 Nov); either one
        # becomes the row's version, so a build counts as behind the release.
        om = re.match(r"^([0-9.]+)(?: \(v?([0-9a-f.]+)\))?$", c[3])
        # A competitor cell may name the version it raced, "143.0 (3.9.0)" (the 24 Nov rerun on PeakMiner 2.21.0 and
        # SRBMiner 3.9.0); without one the row raced the section's SRBMiner and the official PeakMiner and OneZero.
        pk, pkv = _cell(c[4], OFFICIAL_VER["peak"])
        sr, srv = _cell(c[5], srbv)
        oz, ozv = _cell(c[6], OFFICIAL_VER["ozm"])
        out.append({"name": name, "mem": mem.group(1) if mem else None, "machine": c[2], "ours": om.group(1),
                    "peak": pk, "srb": sr, "ozm": oz, "pct": c[7],
                    "ver": {"ours": om.group(2) or base_ours, "peak": pkv, "srb": srv, "ozm": ozv}})
    return out


def base(card):
    return card.split(" · ")[0]


def merge(rows, md):
    """rows: bench_json's official rows (card already 'RTX 3060 · 12 GB' style). Returns the merged list."""
    for r in rows:
        if r["series"] != "Intel Arc":
            r["ver"] = dict(OFFICIAL_VER)
    spots = spot_rows(md)
    out = []
    i = 0
    while i < len(rows):
        r = rows[i]
        j = i
        while j < len(rows) and rows[j]["series"] == r["series"] and base(rows[j]["card"]) == base(r["card"]):
            j += 1
        group = rows[i:j]
        mine = [s for s in spots if s["name"] == base(r["card"])][-len(group):] if group else []
        keep = group[:len(group) - len(mine)]
        for s in reversed(mine):   # newest first
            out.append({"key": "", "series": r["series"], "count": r["count"], "card": group[0]["card"],
                        "host": "Salad (spot check)", "machine": s["machine"], "ours": s["ours"], "peak": s["peak"],
                        "srb": s["srb"], "ozm": s["ozm"], "pct": s["pct"], "ver": s["ver"], "spot": True})
        out.extend(keep)
        i = j
    # Each series' count line again, over the merged rows.
    by = {}
    for r in out:
        if r["pct"].endswith("%"):
            by.setdefault(r["series"], []).append(float(r["pct"].rstrip("%")))
    for r in out:
        ps = sorted(by.get(r["series"], []))
        if ps and r["series"] != "Intel Arc":
            n = len(ps)
            med = ps[n // 2] if n % 2 else (ps[n // 2 - 1] + ps[n // 2]) / 2
            r["count"] = f"{sum(p > 100 for p in ps)} of {n} above 100%, median {med:.1f}%"
    for k, r in enumerate(out):
        r["key"] = str(k)
    return out
