"""lat.py <cubin> <sass>: what ptxas left between an instruction and the first later reader of what it wrote, in
cycles of stall, by kind of writer. Straight-line order through BAR (it falls through), stopping at branches and
labels. The smallest figure a kind shows is an upper bound on the latency ptxas honours for it."""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spread import read as read5, CTL, REG
read = lambda *a: read5(*a)[:4]


def dest(i):
    """Registers the instruction writes: its first operand, unless it is a store or control."""
    if i.op in ("STS", "ST", "STG", "STL", "BAR", "BRA", "NOP", "EXIT", "RET", "CALL", "YIELD", "WARPSYNC", "BSSY",
                "BSYNC", "BREAK", "BMOV", "RED"):
        return set()
    t = i.text
    if t.startswith("@"):
        t = t.split(None, 1)[1]
    body = t.split(None, 1)[1] if " " in t else ""
    m = REG.match(body.strip())
    if not m:
        return set()
    mnem = t.split()[0]
    w = 3 if i.op in ("IMMA", "LDSM") or "128" in mnem else 1 if ("64" in mnem or "WIDE" in mnem or i.op == "CS2R") else 0
    if m.group(1) != "R":
        w = 0
    out = {m.group(1) + str(q) for q in range(int(m.group(2)), int(m.group(2)) + w + 1)}
    if i.op in ("ISETP", "IADD3") and m.group(1) == "R":   # an IADD3 may also write carry predicates
        for mm in REG.finditer(body.split(",")[1] if "," in body else ""):
            if mm.group(1) == "P":
                out.add("P" + mm.group(2))
    return out


def main():
    b, off, size, ins = read(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "pearl")
    best = collections.defaultdict(lambda: (10 ** 9, None))
    hist = collections.defaultdict(collections.Counter)
    waits = collections.Counter()
    for x, i in enumerate(ins):
        if i.op == "IMMA":
            waits["IMMA wait %d wsb %s rsb %s" % (bin(i.c["wait"]).count("1"), i.c["wsb"] != 7, i.c["rsb"] != 7)] += 1
        d = dest(i)
        if not d:
            continue
        cyc = 0
        for y in range(x, min(x + 400, len(ins))):
            j = ins[y]
            if y > x:
                if j.label or (j.op in CTL and j.op != "BAR"):
                    break
                if j.regs & d:
                    # a reader (or an overwrite: either way the first later instruction naming it)
                    reads = True
                    key = (i.op + (".sb" if i.c["wsb"] != 7 else ""), j.op + (".w" if j.c["wait"] else ""))
                    hist[key][cyc] += 1
                    if cyc < best[key][0]:
                        best[key] = (cyc, "%x -> %x" % (i.addr, j.addr))
                    break
            cyc += j.c["stall"]
    for k, v in sorted(waits.items()):
        print(k, v)
    for key in sorted(best):
        h = hist[key]
        n = sum(h.values())
        if n >= 3:
            low = sorted(h)[:4]
            print("%-14s -> %-12s least %3d (%s), %4d pairs, lowest %s" % (key[0], key[1], best[key][0], best[key][1], n,
                                                                    " ".join("%d:%d" % (c, h[c]) for c in low)))


if __name__ == "__main__":
    main()
