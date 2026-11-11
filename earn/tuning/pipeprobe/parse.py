"""parse.py <log.txt> [--md]: a pipe-probe box's log (salad.py logs output) as tables.

Card line, the miners (bench.sh samples from 60 s in: mean of each miner's own figure, watts, clock), hashrate.js on
the release core, then the probe: builds (registers, instructions an IMMA), short runs (pipe use), long runs (pipe
use, clock, watts, rate by count). Rates by count are streams that compute nothing: they are not mining rates.
"""
import json
import re
import statistics
import sys
from collections import OrderedDict, defaultdict

NAMES = OrderedDict([
    ("pure", "IMMA only"),
    ("base", "+ fragment loads (4 ldmatrix a k-step)"),
    ("rd1", "+ readout, one run of 96 XOR3 (own word)"),
    ("rd1a", "the same with a shared XOR"),
    ("rd2", "+ readout, three runs of 32 (own word)"),
    ("rd2a", "the same with a shared XOR"),
    ("rd3", "+ readout, six runs of 16"),
    ("d4e", "+ 4 draw units a k-step, at its end"),
    ("d4m", "+ 4 draw units a k-step, after the 24th IMMA"),
    ("d4h", "the same in two halves"),
    ("d4s", "one unit every 12 IMMA"),
    ("d4mg", "two units' steps grouped by kind"),
    ("d4r", "+ 4 units that read their offsets first"),
    ("fullns", "loads, three-run readout, 4 such units"),
    ("full", "the same with one shared add and test a k-step"),
    ("full3", "the same with 3 units"),
    ("fulle", "the same, 4 units at the k-step's end"),
    ("baser", "fragment loads, hashed bytes"),
    ("rd1r", "one-run readout, hashed bytes"),
    ("rd2ar", "three-run readout, hashed bytes"),
    ("d4mr", "4 draw units, hashed bytes"),
])
PEAK = OrderedDict([("m0", "one warp on the SM"), ("w4", "4 warps (one a scheduler)"), ("pure", "8 warps (two)"),
                    ("w8a32", "8 warps, 32 IMMA a k-step"), ("w8a24", "8 warps, 24 IMMA a k-step"),
                    ("w12", "12 warps (three), 32 IMMA a k-step"), ("w16", "16 warps (four), 24 IMMA a k-step"),
                    ("w4k", "4 warps with loads"), ("base", "8 warps with loads"), ("w12k", "12 warps with loads"),
                    ("w16k", "16 warps with loads")])
RUN = re.compile(r"\[probe\] (nvcc[\d.]+) (short|long) (\S+)\s+pipe\s+([\d.]+)% \(([\d.]+)-([\d.]+)\)\s+([\d.]+) cyc/IMMA "
                 r"(\d+) ms x(\d+) \| ([\d.]+) T-MAC/s by count, counter (\d+) MHz, (\d+) SMs, (\d+) of (\d+) warps"
                 r"(?: \| (.*))?")
SMI = re.compile(r"(\d+) MHz \((\d+)-(\d+)\) ([\d.]+) W \(max ([\d.]+)\) of (\d+) W, (\d+) C, util (\d+)%, reasons (\S+), (\d+) samples")


def main():
    path = sys.argv[1]
    card = []
    built = defaultdict(dict)
    shorts = defaultdict(lambda: defaultdict(list))
    longs = defaultdict(lambda: defaultdict(list))
    own = defaultdict(dict)
    samples = defaultdict(list)
    credit, hr, loaded, misc = {}, None, None, []
    sms = None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip()
        m = RUN.search(line)
        if m:
            tk, kind, name = m.group(1), m.group(2), m.group(3)
            d = dict(pipe=float(m.group(4)), lo=float(m.group(5)), hi=float(m.group(6)), cyc=float(m.group(7)),
                     ms=int(m.group(8)), n=int(m.group(9)), macs=float(m.group(10)), ctr=int(m.group(11)),
                     sms=int(m.group(12)), run=int(m.group(13)), warps=int(m.group(14)))
            sms = d["sms"]
            if m.group(15):
                s = SMI.search(m.group(15))
                if s:
                    d.update(mhz=int(s.group(1)), mhz_lo=int(s.group(2)), mhz_hi=int(s.group(3)), w=float(s.group(4)),
                             wmax=float(s.group(5)), lim=int(s.group(6)), c=int(s.group(7)), util=int(s.group(8)),
                             why=s.group(9), ns=int(s.group(10)))
            (longs if kind == "long" else shorts)[tk][name].append(d)
            continue
        m = re.search(r"\[probe\] (nvcc[\d.]+) short (\S+)\s+own cycles an mma, by warp: (.*)", line)
        if m:
            own[m.group(1)][m.group(2)] = m.group(3)
            continue
        m = re.search(r"\[probe\] (nvcc[\d.]+) built (\S+) regs=(\S+) spill=(\S+) (.*)", line)
        if m:
            built[m.group(1)][m.group(2)] = (m.group(3), m.group(4), m.group(5))
            continue
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
            credit[m.group(1)] = dict(x.split("=", 1) for x in m.group(2).split() if "=" in x)
            continue
        m = re.search(r"\[probe\] hashrate (\S+) (\S+) .*?(\{.*\})", line)
        if m:
            try:
                hr = (m.group(1), m.group(2), json.loads(m.group(3)))
            except ValueError:
                misc.append(line[line.find("[probe]"):])
            continue
        if "[bench]" in line and " loaded " in line:
            loaded = line[line.find("loaded "):]
        if re.search(r"\[probe\] (card|hashrate|toolkit2|.*NOT built|.*builds done|FAIL|WARN)", line) or "[bench]" in line and (
                " card " in line or "FAIL" in line or "exited" in line or " ready " in line):
            card.append(line[line.find("[nvtest]") + 9:][:300])
    for c in card:
        print(c)
    if loaded:
        print(loaded)
    full = sms * 1024e-6 if sms else None   # T-MAC/s per MHz at 16 cycles an IMMA a scheduler, 4 schedulers an SM
    if samples:
        print("\nMiners on the stand-in pool (their own figures, samples from 60 s in):")
        print("| Miner | n | TH/s | W | MHz | C | TH/s over clock x SMs x 1024 | credited ths60 |")
        print("|---|---|---|---|---|---|---|---|")
        for name, ss in samples.items():
            ss = [s for s in ss if s[0] >= 60]
            if not ss:
                continue
            th = statistics.mean(s[1] for s in ss)
            g = list(zip(*[s[2] for s in ss]))
            mhz = statistics.mean(g[1])
            print(f"| {name} | {len(ss)} | {th:.1f} | {statistics.mean(g[0]):.0f} | {mhz:.0f} | {statistics.mean(g[2]):.0f} | "
                  f"{(th / (mhz * full) * 100 if full else 0):.1f}% | {credit.get(name, {}).get('ths60', '-')} |")
    if hr:
        j = hr[2]
        print(f"\nhashrate.js {hr[0]} {hr[1]}: {j.get('th')} TH/s sd {j.get('sd')} n {j.get('n')}, {j.get('mhz')} MHz, "
              f"{j.get('watts')} W, {j.get('tempC')} C, k {j.get('k')}, shape {j.get('shape')}, device {j.get('device')}"
              + (f", {j['th'] / (j['mhz'] * full) * 100:.1f}% of clock x SMs x 1024" if full and j.get("mhz") else ""))
    for m in misc:
        print(m[:300])
    for tk in shorts:
        print(f"\n== {tk}: {sms} SMs; a full pipe at 16 cycles an IMMA is {full:.4f} T-MAC/s per MHz")
        print("\nWhat a full pipe is (IMMA only unless said):")
        print("| Warps | Cycles an IMMA a scheduler | Use at 16 | A warp's own cycles an IMMA |")
        print("|---|---|---|---|")
        for k, label in PEAK.items():
            for d in shorts[tk].get(k, [])[:1]:
                print(f"| {label} | {d['cyc']:.3f} | {d['pipe']:.2f}% ({d['lo']:.2f}-{d['hi']:.2f}) | {own[tk].get(k, '')} |")
        print("\nWarp pairs (which share a pipe):")
        for k in ("m01", "m02", "m03", "m04", "m05", "m06", "m07"):
            for d in shorts[tk].get(k, [])[:1]:
                print(f"  {k}: {own[tk].get(k, '')}   (use {d['pipe']:.2f}%)")
        print("\nStreams:")
        print("| Stream | Regs | Other instr an IMMA | Longest run | Pipe use (short) | Long: pipe use | MHz | W of limit | C | "
              "By count T-MAC/s | pipe x clock x SMs |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, label in NAMES.items():
            b = built[tk].get(k)
            per = re.search(r"per_imma=([\d.]+)", b[2]).group(1) if b and "per_imma" in b[2] else "?"
            lr = re.search(r"longest_run=(\d+)", b[2]).group(1) if b and "longest_run" in b[2] else "?"
            sh = shorts[tk].get(k, [])
            lg = longs[tk].get(k, [])
            if not sh and not lg:
                continue
            s_pipe = "/".join(f"{d['pipe']:.2f}%" for d in sh) or "-"
            if not lg:
                print(f"| {label} | {b[0] if b else '?'} | {per} | {lr} | {s_pipe} | | | | | | |")
            for d in lg:
                if "mhz" in d:
                    print(f"| {label} | {b[0] if b else '?'} | {per} | {lr} | {s_pipe} | {d['pipe']:.2f}% | {d['mhz']} "
                          f"({d['mhz_lo']}-{d['mhz_hi']}) | {d['w']:.0f} of {d['lim']} (max {d['wmax']:.0f}) | {d['c']} | "
                          f"{d['macs']:.1f} | {d['pipe'] / 100 * d['mhz'] * full:.1f} |  reasons {d['why']} ctr {d['ctr']}")
                else:
                    print(f"| {label} | {b[0] if b else '?'} | {per} | {lr} | {s_pipe} | {d['pipe']:.2f}% | no card samples | | | "
                          f"{d['macs']:.1f} | |")
        nb = [f"{k}: {v[2][:120]}" for k, v in built[tk].items() if "NOT" in v[2]]
        for x in nb:
            print("not built:", x)


main()
