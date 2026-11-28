"""sasscount.py: SASS instructions per nonce of a scan kernel, by opcode and by pipe, from nvdisasm -c output.

  python sasscount.py <file.sass> <kernel> --trips 2,4,21,3 [--half 0,3,4] [--skip 5] [--list]

Loops are found from backward branches (BRA to an earlier label). The outermost one is the per-nonce loop and
counts once; --trips gives the trip counts of the loops inside it, in the order their heads appear. --list prints
the loops and the forward branches inside the per-nonce loop, numbered; --half N makes the region forward branch N
skips count half (a pass-only block of the two-pass loop), --skip N makes it count zero (the rare hit path).

Pipes are those of Turing to Blackwell as Nsight Compute names them: IMAD (every IMAD form) goes to the FMA pipe
(fmaheavy on Ampere/Ada), IADD3, LOP3, SHF, PRMT, SEL, ISETP, LEA, MOV and friends to the ALU pipe.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict

INS = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)(.*?);")
LABEL = re.compile(r"^(\.L_x_\d+):")
BR = re.compile(r"`\((\.L_x_\d+)\)")

ALU = {"IADD3", "LOP3", "SHF", "PRMT", "SEL", "ISETP", "LEA", "MOV", "IABS", "IMNMX", "PLOP3", "P2R", "R2P",
       "BMSK", "SGXT", "FLO", "LOP", "SHL", "SHR", "IADD", "FSEL", "FSETP", "CSET", "CSETP", "VIMNMX"}
FMA = {"IMAD", "IMUL", "FFMA", "FMUL", "FADD", "HFMA2", "HADD2", "HMUL2", "IDP", "IDP4A"}
BRANCH = {"BRA", "CALL", "EXIT", "BSSY", "BSYNC", "YIELD", "WARPSYNC", "RET", "BAR", "NOP", "BREAK", "JMP"}
LSU = {"LDL", "STL", "LDG", "STG", "LDS", "STS", "ATOMG", "ATOM", "RED", "ATOMS", "LD", "ST"}


def pipe(op):
    base = op.split(".")[0]
    if base == "IMAD" and ".WIDE" in op:
        return "fma.wide"
    if base == "IMAD" and ".HI" in op:
        return "fma.hi"
    if base in FMA:
        return "fma"
    if base == "VIADD":
        return "viadd"
    if base in ALU:
        return "alu"
    if base in BRANCH:
        return "branch"
    if base in LSU:
        return "lsu"
    if base == "LDC":
        return "ldc"
    if base.startswith("U"):
        return "uniform"
    return "other:" + base


def parse(path, kernel):
    lines = open(path, encoding="utf-8-sig", errors="replace").read().splitlines()
    on = False
    ins, labels = [], {}
    for ln in lines:
        if ln.startswith("//----") and ".text." in ln:
            on = (".text." + kernel + " ") in ln or ln.rstrip("- ").endswith(".text." + kernel)
            continue
        if not on:
            continue
        m = LABEL.match(ln)
        if m:
            labels[m[1]] = len(ins)
            continue
        m = INS.match(ln)
        if m:
            b = BR.search(m[4])
            ins.append({"op": m[3], "pred": (m[2] or "").strip(), "target": b[1] if b else None, "text": ln.strip()})
    return ins, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("kernel")
    ap.add_argument("--trips", default="")
    ap.add_argument("--half", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    ins, labels = parse(a.sass, a.kernel)
    if not ins:
        sys.exit("no instructions for kernel " + a.kernel)
    loops = []
    for i, x in enumerate(ins):
        if x["op"] == "BRA" and x["target"] in labels and labels[x["target"]] <= i:
            loops.append((labels[x["target"]], i))
    loops.sort()
    outer = loops[0]
    inner = [l for l in loops[1:] if outer[0] <= l[0] and l[1] <= outer[1]]
    fwd = []
    for i, x in enumerate(ins):
        if outer[0] <= i <= outer[1] and x["op"] == "BRA" and x["target"] in labels and labels[x["target"]] > i:
            fwd.append((i, labels[x["target"]], x["pred"]))
    if a.list:
        print("per-nonce loop: %d..%d (%d instructions)" % (outer[0], outer[1], outer[1] - outer[0] + 1))
        for n, l in enumerate(inner):
            print("loop %d: %d..%d (%d)" % (n, l[0], l[1], l[1] - l[0] + 1))
        for n, f in enumerate(fwd):
            print("fwd %d: %s BRA at %d -> %d (skips %d)" % (n, f[2] or "always", f[0], f[1], f[1] - f[0] - 1))
    trips = [float(t) for t in a.trips.split(",") if t]
    if len(trips) != len(inner):
        sys.exit("%d inner loops, %d trip counts given (use --list)" % (len(inner), len(trips)))
    freq = [0.0] * len(ins)
    for i in range(outer[0], outer[1] + 1):
        f = 1.0
        for t, l in zip(trips, inner):
            if l[0] <= i <= l[1]:
                f *= t
        freq[i] = f
    for n in [int(x) for x in a.half.split(",") if x]:
        s, e, _ = fwd[n]
        for i in range(s + 1, e):
            freq[i] *= 0.5
    for n in [int(x) for x in a.skip.split(",") if x]:
        s, e, _ = fwd[n]
        for i in range(s + 1, e):
            freq[i] = 0.0
    by_op, by_pipe = Counter(), Counter()
    for i, x in enumerate(ins):
        if freq[i]:
            by_op[x["op"]] += freq[i]
            by_pipe[pipe(x["op"])] += freq[i]
    total = sum(by_pipe.values())
    print("%s %s: %.0f instructions a nonce" % (a.sass.split("\\")[-1].split("/")[-1], a.kernel, total))
    for p, n in sorted(by_pipe.items(), key=lambda kv: -kv[1]):
        print("  pipe %-10s %8.0f  %5.1f%%" % (p, n, 100 * n / total))
    for op, n in by_op.most_common(40):
        print("    %-24s %8.0f" % (op, n))


if __name__ == "__main__":
    main()
