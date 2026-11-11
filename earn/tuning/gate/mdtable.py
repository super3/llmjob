"""mdtable.py salad <log.txt> [<log2.txt> ...] | rig <runs.jsonl> ... : markdown tables for benchmark.md.

salad: the competitors' rows and one row an arm of each race in the logs given (a second log is a second pass on
the same PC: its arms are listed after the first's, against that pass's own base).
rig: per power limit, one row an arm: mean TH/s, change against the neighbouring controls (mean, and each run),
clock, watts, temperature, TH/s per GHz.
"""
import json
import math
import re
import statistics
import sys
from collections import OrderedDict, defaultdict

NAMES = {
    "base": "Release fold (control)", "aa": "Twin: a copy of the control",
    "dram": "DRAM ablation (copies kept, source in L2)", "tdram": "DRAM ablation (boxes kept, source in L2)",
    "sk3": "Copies gated off, A' and B'", "sk1": "Copies gated off, A' alone", "sk2": "Copies gated off, B' alone",
    "tsk3": "Boxes skipped, A' and B'", "tsk1": "Boxes skipped, A' alone", "tsk2": "Boxes skipped, B' alone",
    "sk3p": "Copies gated off, no operand redraw", "tsk3p": "Boxes skipped, no operand redraw",
    "nc": "Copies compiled out", "ncp": "Copies compiled out, no operand redraw",
    "br": "Release fold, stages as a loop", "ncpr": "No copies, stages as a loop",
    "cbase": "Release fold on the cp.async ring", "cncp": "cp.async ring, copies compiled out",
    "cncpr": "cp.async ring, no copies, stages as a loop",
    "sh05": "Sham 0.5 an IMMA, runs at the k-step ends", "sh1": "Sham 1 an IMMA, runs at the k-step ends",
    "sh2": "Sham 2 an IMMA, runs at the k-step ends", "csh1": "Sham 1 an IMMA, runs at the k-step ends",
    "si05r": "Sham 0.5 an IMMA, between the mma", "si1r": "Sham 1 an IMMA, between the mma",
    "si2r": "Sham 2 an IMMA, between the mma", "csi05r": "Sham 0.5 an IMMA, between the mma",
    "csi1r": "Sham 1 an IMMA, between the mma", "csi2r": "Sham 2 an IMMA, between the mma",
    "csi1": "Sham 1 an IMMA, between the mma, stages unrolled",
    "sb1r": "Sham 1 an IMMA, asked for between m16 tiles", "csb1r": "Sham 1 an IMMA, asked for between m16 tiles",
    "sa05r": "ALU only, 0.5 an IMMA", "sa1r": "ALU only, 1 an IMMA", "sa2r": "ALU only, 2 an IMMA",
    "csa1r": "ALU only, 1 an IMMA",
    "sa0r": "The sham's stage-end release alone (no ALU)", "sa1rb": "ALU only, 1 an IMMA, no release",
    "si0r": "Sham's loads, stores and release, no ALU", "si1rb": "Sham 1 an IMMA, no release",
    "si05rb": "Sham 0.5 an IMMA, no release",
    "spn0": "Spin kernel beside the no-copy fold, idle", "spn05": "Spin kernel, 0.5 an IMMA's worth",
    "spn1": "Spin kernel, 1 an IMMA's worth", "spn2": "Spin kernel, 2 an IMMA's worth",
    "dramk": "DRAM ablation, long k", "tdramk": "DRAM ablation, long k", "sk3k": "Copies gated off, long k",
    "tsk3k": "Boxes skipped, long k", "nck": "No copies, long k", "cnck": "cp.async ring, no copies, long k",
    "ncprk": "No copies, stages as a loop, long k", "cncprk": "cp.async ring, no copies, loop, long k",
    "si05rk": "Sham 0.5 an IMMA, long k", "si1rk": "Sham 1 an IMMA, long k", "si2rk": "Sham 2 an IMMA, long k",
    "csi05rk": "Sham 0.5 an IMMA, long k", "csi1rk": "Sham 1 an IMMA, long k", "csi2rk": "Sham 2 an IMMA, long k",
    "si1rbk": "Sham 1 an IMMA, no release, long k", "sb1rk": "Sham 1 an IMMA, between m16 tiles, long k",
    "peak": "PeakMiner 2.20.7", "srb": "SRBMiner 3.8.2", "ozm": "OneZeroMiner 1.8.5", "ours": "Ours, v0.6.15 (CLI)",
}


def kv(text):
    return dict(m.groups() for m in re.finditer(r"(\w+)=([^\s]+)", text))


def salad(paths):
    for n, path in enumerate(paths):
        samples, credit, legs, order = defaultdict(list), {}, defaultdict(dict), []
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.search(r"\[bench\] \S+ sample (\w+) t=(\d+) hs=(\S+) acc=(\S+) rej=(\S+) gpu=(\S+)", line)
            if m:
                try:
                    samples[m.group(1)].append((int(m.group(2)), float(m.group(3)) / 1e12,
                                                [float(x) for x in m.group(6).split(",")[:6]]))
                except ValueError:
                    pass
                continue
            m = re.search(r"\[bench\] \S+ credit (\w+) (.*)", line)
            if m:
                credit[m.group(1)] = kv(m.group(2))
                continue
            m = re.search(r"race LEG round=(\d+) (\S+) (.*)", line)
            if m:
                if m.group(2) not in order:
                    order.append(m.group(2))
                legs[m.group(2)][int(m.group(1))] = {k: float(v) for k, v in kv(m.group(3)).items()
                                                     if re.fullmatch(r"-?[\d.]+", v)}
        print(f"\n<!-- {path} -->")
        if samples:
            print("| Miner (its own figure) | TH/s | SM clock | W | °C | TH/s per GHz | Credited by the stand-in |")
            print("|---|---|---|---|---|---|---|")
            for name, ss in samples.items():
                ss = [s for s in ss if s[0] >= 60]
                if not ss:
                    continue
                th = statistics.mean(s[1] for s in ss)
                g = list(zip(*[s[2] for s in ss]))
                mhz = statistics.mean(g[1])
                print(f"| {NAMES.get(name, name)} | {th:.1f} | {mhz:.0f} MHz | {statistics.mean(g[0]):.0f} | "
                      f"{statistics.mean(g[2]):.0f} | {th / mhz * 1000:.1f} | {credit.get(name, {}).get('ths60', '-')} |")
        if legs:
            base = legs["base"]
            bg = statistics.mean(v["th"] / v["mhz"] for v in base.values())
            bm = statistics.mean(v["mhz"] for v in base.values())
            print(f"\n| Arm ({len(base)} rounds) | TH/s | vs control | ±95% | Ahead | SM clock | Clock | W | °C | TH/s per GHz | Per clock |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for name in order:
                ls = legs[name]
                th = statistics.mean(v["th"] for v in ls.values())
                r = [ls[k]["th"] / base[k]["th"] - 1 for k in ls if k in base]
                d = statistics.mean(r) * 100
                ci = 1.96 * statistics.stdev(r) / math.sqrt(len(r)) * 100 if len(r) > 1 else 0.0
                mhz = statistics.mean(v["mhz"] for v in ls.values())
                g = statistics.mean(v["th"] / v["mhz"] for v in ls.values())
                dec = 2 if th < 100 else 1
                if name == "base":
                    print(f"| `{name}` {NAMES.get(name, name)} | {th:.{dec}f} | | | | {mhz:.0f} MHz | | "
                          f"{statistics.mean(v['w'] for v in ls.values()):.0f} | {statistics.mean(v['c'] for v in ls.values()):.0f} | {g * 1000:.{dec}f} | |")
                    continue
                print(f"| `{name}` {NAMES.get(name, name)} | {th:.{dec}f} | {d:+.2f}% | {ci:.2f} | "
                      f"{sum(1 for x in r if x > 0)}/{len(r)} | {mhz:.0f} | {(mhz / bm - 1) * 100:+.1f}% | "
                      f"{statistics.mean(v['w'] for v in ls.values()):.0f} | {statistics.mean(v['c'] for v in ls.values()):.0f} | "
                      f"{g * 1000:.{dec}f} | {(g / bg - 1) * 100:+.1f}% |")


def rig(paths):
    by_pl = defaultdict(list)
    for path in paths:
        rows = []
        for line in open(path, encoding="utf-8"):
            try:
                j = json.loads(line)
            except Exception:
                continue
            r = j.get("r") or {}
            rows.append({"arm": j["arm"], "pl": j.get("pl"), "bad": bool(j.get("app")) or "th" not in r, **r})
        ctl = [i for i, r in enumerate(rows) if r["arm"] == "base" and not r["bad"]]
        for i, r in enumerate(rows):
            if r["bad"]:
                continue
            near = [c for c in ([c for c in ctl if c < i][-1:] + [c for c in ctl if c > i][:1]) if c != i]
            r["ref"] = statistics.mean(rows[c]["th"] for c in near) if near else None
            r["refg"] = statistics.mean(rows[c]["th"] / rows[c]["mhz"] for c in near) if near else None
            by_pl[round(r["pl"])].append(r)
    for pl in sorted(by_pl, reverse=True):
        arms = OrderedDict()
        for r in by_pl[pl]:
            arms.setdefault(r["arm"], []).append(r)
        base = arms["base"]
        bth = [r["th"] for r in base]
        bm = statistics.mean(r["mhz"] for r in base)
        print(f"\n<!-- {pl} W: control {len(base)} runs, {min(bth):.2f} to {max(bth):.2f}, range {(max(bth) - min(bth)) / statistics.mean(bth) * 100:.2f}% -->")
        print(f"| Arm at {pl} W | Runs | TH/s | vs control (each run) | SM clock | Clock | W | °C | TH/s per GHz | Per clock |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for arm, rs in arms.items():
            th = statistics.mean(r["th"] for r in rs)
            mhz = statistics.mean(r["mhz"] for r in rs)
            d = [(r["th"] / r["ref"] - 1) * 100 for r in rs if r.get("ref")]
            g = [(r["th"] / r["mhz"] / r["refg"] - 1) * 100 for r in rs if r.get("refg")]
            w = statistics.mean(r["watts"] for r in rs)
            c = statistics.mean(r["tempC"] for r in rs)
            if arm == "base":
                print(f"| `{arm}` {NAMES.get(arm, arm)} | {len(rs)} | {th:.1f} | range {(max(bth) - min(bth)) / statistics.mean(bth) * 100:.2f}% | {mhz:.0f} MHz | | {w:.0f} | {c:.0f} | {th / mhz * 1000:.1f} | |")
                continue
            each = ", ".join(f"{x:+.2f}" for x in d)
            print(f"| `{arm}` {NAMES.get(arm, arm)} | {len(rs)} | {th:.1f} | {statistics.mean(d):+.2f}% ({each}) | {mhz:.0f} | "
                  f"{(mhz / bm - 1) * 100:+.1f}% | {w:.0f} | {c:.0f} | {th / mhz * 1000:.1f} | {statistics.mean(g):+.1f}% |")


if sys.argv[1] == "salad":
    salad(sys.argv[2:])
else:
    rig(sys.argv[2:])
