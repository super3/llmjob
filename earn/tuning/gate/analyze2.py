"""analyze2.py <runs.jsonl> [<runs.jsonl> ...] : the follow-up's rig01 tables (markdown), one a power limit.

Each arm's rate is hashrate.js's times the share of the block's eight warps that fold (an arm named m4... folds
with four, m7... with seven: the host counts every region of a launch, and the warps that do not fold leave theirs
undone). Change is against the controls ("base", the release fold) on either side of the run, and against the
no-copy k 65536 fold ("nck") on either side. Work a clock is TH/s per GHz, and its share of the tensor peak
(1024 MAC an SM a clock: 131.072 TH/s a GHz on the 4090's 128 SMs). "Cycles a chunk" is 6144 over that share: the
SM clocks one chunk of a 192x256 tile takes, 6144 at the peak.
"""
import json
import statistics
import sys
from collections import OrderedDict, defaultdict

PEAK = 131.072
NAMES = OrderedDict([
    ("base", "Release fold, k 2048 (control)"),
    ("nck", "No copies, k 65536 (the base of both measurements)"),
    ("kro", "Readout at the tile's last chunk only"),
    ("krg", "Ring waits and arrivals off"),
    ("khs", "Tile hash off"),
    ("kall", "Readout, ring and hash all off"),
    ("kbs", "B fragments reloaded a tile: 80 more ldmatrix a chunk"),
    ("kbz1", "The same, readout moved into the next chunk's first k-step"),
    ("m4k", "4 warps fold (one a scheduler), 4 leave"),
    ("m4p0k", "4 fold, 4 producers that spin on their barrier, no work"),
    ("m4n0k", "4 fold, 4 producers that nap 100 ns between tests, no work"),
    ("m4n1k", "4 fold, 4 producers, 1.03 instructions an IMMA (9 a unit)"),
    ("m4n2k", "4 fold, 4 producers, 1.95 an IMMA (17 a unit)"),
    ("m4n3k", "4 fold, 4 producers, 3.67 an IMMA (32 a unit)"),
    ("m4p2k", "4 fold, 4 spinning producers, 1.95 an IMMA"),
    ("m7k", "7 warps fold, 1 leaves"),
    ("m7n0k", "7 fold, 1 producer, no work"),
    ("m7n1k", "7 fold, 1 producer for all seven, 1.00 an IMMA"),
    ("t4k", "128x256 tile, 8 warps of 64x64, no producer"),
    ("t12n0k", "128x256, 8 fold, 4 napping producer warps, no work"),
    ("t12n1k", "128x256, 8 fold, 4 producers, 1.03 an IMMA (11 a unit)"),
    ("t12n2k", "128x256, 8 fold, 4 producers, 1.97 an IMMA (21 a unit)"),
    ("t12n3k", "128x256, 8 fold, 4 producers, 3.00 an IMMA (32 a unit)"),
    ("t12p2k", "128x256, 8 fold, 4 spinning producers, 1.97 an IMMA"),
    ("t9n2k", "128x256, 8 fold, 1 producer warp, 1.97 an IMMA"),
])


def factor(arm):
    return 0.5 if arm.startswith("m4") else 0.875 if arm.startswith("m7") else 1.0


def load(paths):
    by_pl = defaultdict(list)
    for path in paths:
        rows = []
        for line in open(path, encoding="utf-8"):
            try:
                j = json.loads(line)
            except Exception:
                continue
            r = j.get("r") or {}
            ok = "th" in r and not j.get("app")
            row = {"arm": j["arm"], "pl": j.get("pl"), "ok": ok, "t": j.get("t")}
            if ok:
                f = factor(j["arm"])
                row.update(th=r["th"] * f, raw=r["th"], mhz=r["mhz"], w=r["watts"], c=r["tempC"], sd=r.get("sd", 0) * f)
            else:
                row["err"] = str(r.get("error", ""))[:120]
            rows.append(row)
        for name in ("base", "nck"):
            idx = [i for i, r in enumerate(rows) if r["arm"] == name and r["ok"]]
            for i, r in enumerate(rows):
                if not r["ok"]:
                    continue
                near = [c for c in ([c for c in idx if c < i][-1:] + [c for c in idx if c > i][:1]) if c != i]
                r["ref_" + name] = statistics.mean(rows[c]["th"] for c in near) if near else None
                r["refg_" + name] = statistics.mean(rows[c]["th"] / rows[c]["mhz"] for c in near) if near else None
        for r in rows:
            by_pl[round(r["pl"])].append(r)
    # A run with no no-copy fold beside it in its own file is set against the mean of that limit's.
    for pl, rows in by_pl.items():
        v = [r["th"] for r in rows if r["ok"] and r["arm"] == "nck"]
        if v:
            for r in rows:
                if r["ok"] and not r.get("ref_nck"):
                    r["ref_nck"] = statistics.mean(v)
    return by_pl


def main():
    by_pl = load(sys.argv[1:])
    for pl in sorted(by_pl, reverse=True):
        rows = by_pl[pl]
        arms = OrderedDict()
        for r in rows:
            arms.setdefault(r["arm"], []).append(r)
        order = [a for a in NAMES if a in arms] + [a for a in arms if a not in NAMES]
        for name in ("base", "nck"):
            v = [r["th"] for r in arms.get(name, []) if r["ok"]]
            if v:
                print(f"<!-- {pl} W: {name} {len(v)} runs, {min(v):.2f} to {max(v):.2f} TH/s, range {(max(v) - min(v)) / statistics.mean(v) * 100:.2f}% -->")
        print(f"| Arm at {pl} W | Runs | TH/s | vs release | vs no-copy k 65536 | SM clock | W | °C | TH/s per GHz | Of the tensor peak | Cycles a chunk |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in order:
            rs = [r for r in arms[arm] if r["ok"]]
            bad = [r for r in arms[arm] if not r["ok"]]
            if not rs:
                print(f"| `{arm}` {NAMES.get(arm, arm)} | 0 | no rate ({bad[0].get('err', '')}) | | | | | | | | |")
                continue
            th = statistics.mean(r["th"] for r in rs)
            mhz = statistics.mean(r["mhz"] for r in rs)
            g = statistics.mean(r["th"] / r["mhz"] for r in rs) * 1000
            db = [(r["th"] / r["ref_base"] - 1) * 100 for r in rs if r.get("ref_base")]
            dn = [(r["th"] / r["ref_nck"] - 1) * 100 for r in rs if r.get("ref_nck")]
            sb = "" if arm == "base" or not db else f"{statistics.mean(db):+.2f}%" + (
                " (" + ", ".join(f"{x:+.2f}" for x in db) + ")" if len(db) > 1 else "")
            sn = "" if arm in ("base", "nck") or not dn else f"{statistics.mean(dn):+.2f}%"
            fr = g / PEAK
            print(f"| `{arm}` {NAMES.get(arm, arm)} | {len(rs)} | {th:.1f} | {sb} | {sn} | {mhz:.0f} MHz | "
                  f"{statistics.mean(r['w'] for r in rs):.0f} | {statistics.mean(r['c'] for r in rs):.0f} | {g:.1f} | "
                  f"{fr * 100:.1f}% | {6144 / fr:.0f} |")
        print()


main()
