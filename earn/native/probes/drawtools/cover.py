"""cover.py <cubin> <sass>: for every shared-memory load (LDS, LDSM) and global load, how its first reader is made to
wait for it. 'own': the reader waits on the load's own scoreboard, or an instruction between them does. 'later':
an instruction between them (or the reader) waits on the scoreboard of a later load of the same kind, which
covers this one only if loads finish in the order they were issued. 'none': neither; then the stall distance is
printed, which is all that protects the read."""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spread import read as read5, CTL
read = lambda *a: read5(*a)[:4]
from lat import dest

SH = ("LDS", "LDSM")


def main():
    b, off, size, ins = read(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "pearl")
    res = collections.Counter()
    none = collections.defaultdict(list)
    for x, i in enumerate(ins):
        if i.op not in SH and i.op != "LDG":
            continue
        d = dest(i)
        pipe = SH if i.op in SH else ("LDG",)
        cyc, how, armed = 0, None, set()
        if i.c["wsb"] != 7:
            armed.add(("own", i.c["wsb"]))
        found = False
        for y in range(x, min(x + 600, len(ins))):
            j = ins[y]
            if y > x:
                if j.label or (j.op in CTL and j.op != "BAR"):
                    break
                for kind, k in armed:
                    if j.c["wait"] >> k & 1:
                        how = "own" if kind == "own" or how == "own" else (how or "later")
                if j.regs & d:
                    found = True
                    break
                if j.op in pipe and j.c["wsb"] != 7:
                    armed.add(("later", j.c["wsb"]))
            cyc += j.c["stall"]
        if not found:
            res[i.op + " no reader in the stretch"] += 1
            continue
        res["%s %s" % (i.op, how or "none")] += 1
        if not how:
            none[i.op].append((cyc, "%x -> %x %s" % (i.addr, ins[y].addr, ins[y].op)))
    for k, v in sorted(res.items()):
        print(k, v)
    for op, lst in none.items():
        lst.sort()
        print(op, "unprotected, by stall distance:", lst[:12])


if __name__ == "__main__":
    main()
