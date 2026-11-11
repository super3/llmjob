"""sim.py <cubin> <sass> [Ls Lm Lb issue]: a small model of one scheduler running the two loop copies of a compiled
draw fold side by side, to say what share of the tensor pipe the compiled order can hold. Not a measurement.

The model. A scheduler holds two warps, one from each loop copy (the first copy's warps are 0-3, the second's
4-7). A warp issues its instructions in order; after one, its next may issue no sooner than the stall count ptxas
wrote. An IMMA also needs the pipe, which takes one every 16 cycles. An instruction that waits on a scoreboard
issues only when every load of its warp that set that scoreboard has returned (Ls cycles for a shared load, Lm
for ldmatrix, 4 Ls for a global one). The scheduler issues `issue` instructions a cycle at most, turn about. At a
whole-block barrier (BAR.SYNC 0) a warp waits for the other and both go on Lb cycles later; a barrier of a
copy's own four warps is passed with no wait. Pipe use is 16 cycles an IMMA over the cycles taken.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spread import read
from ctl import read_text


def loops(ins, sass, fn_sub="pearl"):
    """(first, last) instruction index of every loop with 100 IMMA or more: a label and a later branch to it."""
    labels, pending, inside = {}, [], False
    addr_of = {i.addr: x for x, i in enumerate(ins)}
    for line in read_text(sass).splitlines():
        if re.match(r"\s*\.section\s", line):
            inside = ".text." in line and fn_sub in line
            continue
        if not inside:
            continue
        m = re.match(r"\s*(\.L_x_\d+):", line)
        if m:
            pending.append(m.group(1))
            continue
        m = re.search(r"/\*([0-9a-f]{4,})\*/", line)
        if m and pending:
            for lab in pending:
                labels[lab] = addr_of[int(m.group(1), 16)]
            pending = []
    out = []
    for x, i in enumerate(ins):
        if i.op == "BRA":
            m = re.search(r"\.L_x_\d+", i.text)
            if m and m.group(0) in labels and labels[m.group(0)] < x:
                top = labels[m.group(0)]
                if sum(1 for j in ins[top:x] if j.op == "IMMA") >= 100:
                    out.append((top, x))
    return out


def run(streams, passes, Ls, Lm, Lb, issue, phase=0):
    """Two warps, each looping over its stream `passes` times. Returns (cycles, IMMA issued)."""
    n = [len(s) for s in streams]
    pc = [0, 0]
    ready = [0, phase]
    pend = [dict(), dict()]      # scoreboard -> list of return times
    at_bar = [None, None]        # time the warp reached a block barrier
    pipe = 0
    done = [0, 0]
    imma = 0
    t = 0
    turn = 0
    limit = passes * max(n) * 400
    while (done[0] < passes or done[1] < passes) and t < limit:
        issued = 0
        for k in range(2):
            w = (turn + k) % 2
            if done[w] >= passes or issued >= issue:
                continue
            i = streams[w][pc[w]]
            if at_bar[w] is not None:
                continue
            if ready[w] > t:
                continue
            c = i.c
            if c["wait"]:
                busy = False
                for sb in range(6):
                    if c["wait"] >> sb & 1 and any(r > t for r in pend[w].get(sb, ())):
                        busy = True
                if busy:
                    continue
            if i.op == "IMMA":
                if pipe > t:
                    continue
                pipe = t + 16
                imma += 1
            if i.op == "BAR" and " 0x0" in i.text:
                at_bar[w] = t
                issued += 1
                continue
            if c["wsb"] != 7:
                lat = Lm if i.op == "LDSM" else 4 * Ls if i.op == "LDG" else Ls
                lst = [r for r in pend[w].get(c["wsb"], []) if r > t]
                lst.append(t + lat)
                pend[w][c["wsb"]] = lst
            ready[w] = t + max(1, c["stall"])
            pc[w] += 1
            if pc[w] == n[w]:
                pc[w] = 0
                done[w] += 1
            issued += 1
        if at_bar[0] is not None and at_bar[1] is not None:
            for w in range(2):
                ready[w] = t + Lb
                at_bar[w] = None
                pc[w] += 1
                if pc[w] == n[w]:
                    pc[w] = 0
                    done[w] += 1
        elif at_bar[0] is not None and done[1] >= passes or at_bar[1] is not None and done[0] >= passes:
            break
        turn ^= 1
        t += 1
    return t, imma


def main():
    cubin, sass = sys.argv[1:3]
    Ls = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    Lm = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    Lb = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    issue = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    b, off, size, ins, info = read(cubin, sass, "pearl")
    lp = loops(ins, sass)
    if len(lp) < 2:
        print("%s: %d loops found" % (os.path.basename(cubin), len(lp)))
        return
    streams = [ins[a:z + 1] for a, z in lp[:2]]
    t, imma = run(streams, 6, Ls, Lm, Lb, issue)
    solo = []
    for s in streams:
        ts, im = run([s, s], 6, Ls, Lm, Lb, issue)
        solo.append(16.0 * im / ts)
    print("%-10s loops %s IMMA %s: model pipe use %.2f%% (two of the first copy %.2f%%, two of the second %.2f%%)" % (
        os.path.basename(cubin).replace(".cubin", ""), ["%x-%x" % (ins[a].addr, ins[z].addr) for a, z in lp[:2]],
        [sum(1 for j in s if j.op == "IMMA") for s in streams], 1600.0 * imma / t, 100 * solo[0], 100 * solo[1]))


if __name__ == "__main__":
    main()
