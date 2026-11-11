"""mkrecord.py: the record's tables from the boxes' logs (logs-<card>-final.txt and friends in this folder).

  python mkrecord.py pipe      pipe use by stream and card (short runs)
  python mkrecord.py peak      what a full pipe is, by card
  python mkrecord.py long      clock, watts and rate by count, by card (long runs)
  python mkrecord.py steps     one to four draw units and the pad streams, by card
  python mkrecord.py miners    the miners and the release on each PC
"""
import json
import re
import statistics
import sys
from collections import OrderedDict, defaultdict

RUN = re.compile(r"\[probe\] (nvcc[\d.]+) (short|long) (\S+)\s+pipe\s+([\d.]+)% \(([\d.]+)-([\d.]+)\)\s+([\d.]+) cyc/IMMA "
                 r"(\d+) ms x(\d+) \| ([\d.]+) T-MAC/s by count, counter (\d+) MHz, (\d+) SMs, (\d+) of (\d+) warps"
                 r"(?: \| (.*))?")
SMI = re.compile(r"(\d+) MHz \((\d+)-(\d+)\) ([\d.]+) W \(max ([\d.]+)\) of (\d+) W, (\d+) C, util (\d+)%, reasons (\S+), (\d+) samples")
# card key, log file, toolkit, column title
CARDS = OrderedDict([
    ("3090", ("logs-3090-final.txt", "nvcc12.8", "RTX 3090")),
    ("3060", ("logs-3060-final.txt", "nvcc12.8", "RTX 3060")),
    ("4070", ("logs-4070-final.txt", "nvcc12.8", "RTX 4070")),
    ("5090", ("logs-5090-final.txt", "nvcc13.3", "RTX 5090, 13.3")),
    ("5090b", ("logs-5090-final.txt", "nvcc12.8", "RTX 5090, 12.8")),
    ("5070ti", ("logs-5070ti-final.txt", "nvcc13.3", "RTX 5070 Ti, 13.3")),
    ("5070", ("logs-5070-final.txt", "nvcc13.3", "RTX 5070, 13.3")),
])
STUDY = {"pure": "99.93", "base": "100.00", "rd1": "98.72", "rd1a": "97.91", "rd2": "99.67", "rd2a": "99.71", "rd3": "99.68",
         "d4e": "99.52", "d4m": "99.61", "d4h": "98.44", "d4s": "98.44", "d4mg": "97.68", "d4r": "98.80",
         "fullns": "95.15", "full": "94.71", "full3": "96.97", "fulle": "96.50"}
NAMES = OrderedDict([
    ("pure", "IMMA only"),
    ("base", "+ fragment loads (kept in every row below)"),
    ("rd1", "+ readout, one run of 96 XOR3"),
    ("rd1a", "the same with a shared XOR"),
    ("rd2", "+ readout, three runs of 32"),
    ("rd2a", "the same with a shared XOR"),
    ("rd3", "+ readout, six runs of 16"),
    ("d4m", "+ 4 draw units a k-step, after the 24th IMMA"),
    ("d4e", "the same at the k-step's end"),
    ("d4h", "the same in two halves"),
    ("d4s", "one unit every 12 IMMA"),
    ("d4mg", "two units' steps grouped by kind"),
    ("d4r", "+ 4 units that read their offsets first"),
    ("fullns", "loads, three-run readout, 4 such units"),
    ("full", "the same with one shared add and test a k-step"),
    ("full3", "the same with 3 units"),
    ("fulle", "the same, the 4 units at the k-step's end"),
])
LONGS = OrderedDict([
    ("pure", "IMMA only"), ("base", "+ fragment loads"), ("rd1", "+ readout, one run"), ("rd2a", "+ readout, three runs"),
    ("d4m", "+ 4 draw units"), ("d4r", "+ 4 units, offsets first"), ("full3", "all together, 3 units"),
    ("full", "all together, 4 units"), ("baser", "loads, hashed bytes"), ("rd1r", "one-run readout, hashed bytes"),
    ("rd2ar", "three-run readout, hashed bytes"), ("d4mr", "4 draw units, hashed bytes"),
    ("d1m", "1 draw unit"), ("d2m", "2 draw units"),
])
STEPS = OrderedDict([
    ("base", "loads only"), ("d1m", "1 draw unit a k-step"), ("d2m", "2 draw units"), ("d3m", "3 draw units"),
    ("d4m", "4 draw units"), ("d2r", "2 units, offsets first"), ("d3r", "3 units, offsets first"),
    ("d4r", "4 units, offsets first"), ("rd1c4", "one-run readout, 4 chains"), ("rd1", "one-run readout, 8 chains"),
    ("rd1c16", "one-run readout, 16 chains"), ("rd2c4", "three-run readout, 4 chains"),
    ("rd2a", "three-run readout, 8 chains"), ("rd2c16", "three-run readout, 16 chains"),
    ("p1k0", "an independent add every 2 IMMA"), ("p2k0", "an independent add an IMMA"),
    ("p4k0", "2 independent adds an IMMA"), ("p2k4", "a chained add an IMMA"),
    ("p1k1", "an XOR3 of two earlier sums an IMMA"), ("p2k1", "2 such XOR3 an IMMA"),
    ("p1k2", "a shared load and half an XOR an IMMA"), ("p1k3", "a shared store an IMMA"),
])


def load(path):
    d = dict(short=defaultdict(lambda: defaultdict(list)), long=defaultdict(lambda: defaultdict(list)),
             own=defaultdict(dict), built=defaultdict(dict), samples=defaultdict(list), hr=None, card=None, sms=None)
    try:
        f = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in f:
        line = line.rstrip()
        m = RUN.search(line)
        if m:
            r = dict(pipe=float(m.group(4)), lo=float(m.group(5)), hi=float(m.group(6)), cyc=float(m.group(7)),
                     macs=float(m.group(10)), ctr=int(m.group(11)), sms=int(m.group(12)))
            d["sms"] = r["sms"]
            if m.group(15):
                s = SMI.search(m.group(15))
                if s:
                    r.update(mhz=int(s.group(1)), w=float(s.group(4)), lim=int(s.group(6)), c=int(s.group(7)),
                             why=s.group(9))
            d[m.group(2)][m.group(1)][m.group(3)].append(r)
            continue
        m = re.search(r"\[probe\] (nvcc[\d.]+) short (\S+)\s+own cycles an mma, by warp: (.*)", line)
        if m:
            d["own"][m.group(1)][m.group(2)] = m.group(3)
            continue
        m = re.search(r"\[probe\] (nvcc[\d.]+) built (\S+) regs=(\S+) spill=(\S+) (.*)", line)
        if m:
            d["built"][m.group(1)][m.group(2)] = m.group(5)
            continue
        m = re.search(r"\[bench\] \S+ sample (\w+) t=(\d+) hs=(\S+) acc=\S+ rej=\S+ gpu=(\S+)", line)
        if m:
            try:
                d["samples"][m.group(1)].append((int(m.group(2)), float(m.group(3)) / 1e12,
                                                 [float(x) for x in m.group(4).split(",")[:3]]))
            except ValueError:
                pass
            continue
        m = re.search(r"\[probe\] hashrate (\S+) (\S+) .*?(\{.*\})", line)
        if m:
            try:
                d["hr"] = json.loads(m.group(3))
            except ValueError:
                pass
            continue
        m = re.search(r"\[probe\] card (NVIDIA.*)", line)
        if m:
            d["card"] = m.group(1)
    return d


def flag(why):
    bits = set()
    for w in why.split("/"):
        try:
            v = int(w, 16)
        except ValueError:
            continue
        if v & 0x4:
            bits.add("power cap")
        if v & 0x20 or v & 0x40:
            bits.add("heat")
    return ", ".join(sorted(bits)) or "none"


def main():
    what = sys.argv[1]
    data = {k: load(v[0]) for k, v in CARDS.items()}
    cols = [k for k in CARDS if data[k] and data[k]["short"].get(CARDS[k][1])]
    if what == "pipe":
        print("| Stream | RTX 4090 (the study) | " + " | ".join(CARDS[k][2] for k in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for s, label in NAMES.items():
            row = []
            for k in cols:
                rs = data[k]["short"][CARDS[k][1]].get(s, [])
                if not rs:
                    row.append("")
                    continue
                v = statistics.mean(r["pipe"] for r in rs)
                if k in ("5070ti", "5070"):
                    b = statistics.mean(r["pipe"] for r in data[k]["short"][CARDS[k][1]]["base"])
                    row.append(f"{v / b * 100:.1f}%")
                else:
                    row.append(f"{v:.2f}%")
            print(f"| {label} | {STUDY.get(s, '')}% | " + " | ".join(row) + " |")
    elif what == "peak":
        rows = [("m0", "One warp alone on the SM"), ("w4", "4 warps (one a scheduler)"), ("pure", "8 warps (two)"),
                ("w12", "12 warps (three), 32 IMMA a k-step"), ("w16", "16 warps (four), 24 IMMA a k-step"),
                ("w4k", "4 warps with fragment loads"), ("base", "8 warps with loads"), ("w12k", "12 warps with loads"),
                ("w16k", "16 warps with loads"), ("m04", "Warps 0 and 4 only: each warp's own cycles an IMMA"),
                ("m01", "Warps 0 and 1 only: each warp's own")]
        cols = [k for k in cols if k != "5090b"]
        print("| Cycles an IMMA a scheduler | " + " | ".join(CARDS[k][2].split(",")[0] for k in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for s, label in rows:
            row = []
            for k in cols:
                rs = data[k]["short"][CARDS[k][1]].get(s, [])
                if not rs:
                    row.append("")
                elif s in ("m04", "m01"):
                    o = data[k]["own"][CARDS[k][1]].get(s, "")
                    row.append(" / ".join(x.split(":")[1] for x in o.split()))
                else:
                    row.append(f"{rs[0]['cyc']:.2f} ({rs[0]['pipe']:.2f}%)")
            print(f"| {label} | " + " | ".join(row) + " |")
    elif what == "long":
        for k in cols:
            tk = CARDS[k][1]
            if not data[k]["long"].get(tk):
                continue
            full = data[k]["sms"] * 1024e-6
            print(f"\n{CARDS[k][2]} ({data[k]['sms']} SMs; {data[k]['card']}):\n")
            print("| Stream | Pipe use | Clock, MHz (the SM's own counter) | Watts | Flag | T-MAC/s by count |")
            print("|---|---|---|---|---|---|")
            for s, label in LONGS.items():
                for i, r in enumerate(data[k]["long"][tk].get(s, [])):
                    if "mhz" not in r:
                        continue
                    print(f"| {label}{' (again, last)' if i else ''} | {r['pipe']:.2f}% | {r['mhz']} ({r['ctr']}) | "
                          f"{r['w']:.0f} of {r['lim']} | {flag(r['why'])} | {r['macs']:.1f} |")
            hr = data[k]["hr"]
            if hr and tk == CARDS[k][1] and k != "5090b":
                print(f"| Our release v0.6.15, hashrate.js (a mining rate) | {hr['th'] / (hr['mhz'] * full) * 100:.1f}% of "
                      f"clock x SMs x 1024 | {hr['mhz']} | {hr['watts']} | | {hr['th']:.1f} TH/s (sd {hr['sd']}) |")
            for name, ss in data[k]["samples"].items():
                if k == "5090b":
                    break
                ss = [x for x in ss if x[0] >= 60]
                if not ss:
                    continue
                th = statistics.mean(x[1] for x in ss)
                w = statistics.mean(x[2][0] for x in ss)
                mhz = statistics.mean(x[2][1] for x in ss)
                who = "OneZeroMiner 1.8.5" if name == "ozm" else "Our release v0.6.15, CLI"
                print(f"| {who}, stand-in pool (its own figure) | {th / (mhz * full) * 100:.1f}% of clock x SMs x 1024 | "
                      f"{mhz:.0f} | {w:.0f} | | {th:.1f} TH/s |")
    elif what == "steps":
        extra = OrderedDict([("5090c", ("logs-5090-look2.txt", "nvcc13.3", "RTX 5090 (450 W PC), 13.3")),
                             ("5090d", ("logs-5090-look2.txt", "nvcc12.8", "RTX 5090 (450 W PC), 12.8")),
                             ("3060p", ("logs-3060-pads.txt", "nvcc12.8", "RTX 3060 (pads)")),
                             ("4070p", ("logs-4070-pads.txt", "nvcc12.8", "RTX 4070 (pads)")),
                             ("5070b", ("logs-5070-final.txt", "nvcc12.8", "RTX 5070, 12.8"))])
        allc = OrderedDict(list(CARDS.items()) + list(extra.items()))
        dd = dict(data)
        for k, v in extra.items():
            dd[k] = load(v[0])
        cols = [k for k in allc if dd[k] and dd[k]["short"].get(allc[k][1])]
        print("| Stream | " + " | ".join(allc[k][2] for k in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for s, label in STEPS.items():
            row = []
            for k in cols:
                rs = dd[k]["short"][allc[k][1]].get(s, [])
                b = dd[k]["built"][allc[k][1]].get(s, "")
                m = re.search(r"other=(\d+)", b)
                im = re.search(r"imma=(\d+)", b)
                per = f", {int(m.group(1)) / int(im.group(1)):.2f}" if m and im else ""
                row.append(f"{statistics.mean(r['pipe'] for r in rs):.2f}%{per}" if rs else "")
            print(f"| {label} | " + " | ".join(row) + " |")
    elif what == "miners":
        for path in sys.argv[2:]:
            d = load(path)
            if not d:
                continue
            print(path, d["card"])
            for name, ss in d["samples"].items():
                ss = [x for x in ss if x[0] >= 60]
                if ss:
                    print(f"  {name}: {statistics.mean(x[1] for x in ss):.1f} TH/s, {statistics.mean(x[2][0] for x in ss):.0f} W, "
                          f"{statistics.mean(x[2][1] for x in ss):.0f} MHz, {statistics.mean(x[2][2] for x in ss):.0f} C, n {len(ss)}")
            if d["hr"]:
                print("  hashrate.js:", {k: d["hr"].get(k) for k in ("th", "sd", "n", "mhz", "watts", "tempC", "k")})


main()
