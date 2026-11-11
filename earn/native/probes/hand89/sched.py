"""sched.py <base.cubin> <base.sass> <nvdisasm> <out dir> <list file>: timing-only streams for the question "what
does a plain instruction cost the tensor pipe, by where it sits and how it is marked?". Each stream replaces the
loop function of a build of hprobe.cu (asm89.patch) with: warps 0-3 (group A) and warps 4-7 (group B, the other
warp of each scheduler) each looping over 40 IMMA, with plain adds after chosen IMMA, every stall count and yield
flag set here and not by ptxas. THEY COMPUTE NOTHING REAL: the sums are of constant fragments, the check would
fail, and no figure from them is a mining rate.

A group's spec: gaps (plain adds after each of the 40 IMMA), s (an add's stall), y (an add's yield flag), iy and
istall (an IMMA's yield flag and its stall when no add follows), pad (least cycles from one IMMA to the warp's
next, kept by the last add's stall; 0 for none), bar (a block barrier after the 40), mult (loop passes are
chunks * 4 * mult: with mult 2 a group is still running when the other ends, so the other's time is all in
company).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402

A_FRAG, B_FRAG, ACC, PLAIN, CNT = 4, 8, 24, 184, 200


def group(tag, gaps, s=1, y=True, iy=True, istall=8, pad=8, bar=False, mult=1, lasty=None, kind="add", ru=0, ist1=1, first=None, pat=None):
    out = [Ins("IMAD.SHL.U32 R%d, R3, 0x%x, RZ" % (CNT, 4 * mult), label=".G" + tag)]
    body = []
    na = 0
    for i in range(40):
        g = gaps[i]
        if pat:
            # pat: the stalls written out, the IMMA's first and then each step's.
            g, ist1, pad = len(pat) - 1, pat[0], 0
        text = "IMMA.16832.S8.S8 R%d, R%d.ROW, R%d.COL, R%d" % (ACC + 4 * i, A_FRAG, B_FRAG + 2 * (i % 8), ACC + 4 * i)
        reuse = 1 if ru and i % 8 < 7 else 0     # ru: A's fragment kept in the reuse latch, as ptxas marks it
        if g == 0:
            body.append(Ins(text, stall=istall, yld=iy, reuse=reuse))
            continue
        # ist1: the IMMA's stall when plain steps follow, so the first of them issues that many cycles behind it.
        body.append(Ins(text, stall=ist1, yld=iy, reuse=reuse))
        for j in range(g):
            r = PLAIN + na % 16
            na += 1
            st, yl = s, y
            if j == g - 1:
                st = max(s, pad - ist1 - (g - 1) * s) if pad else s
                if st > 11:
                    # nvdisasm knows no stall over 11 with the yield flag on an add: the rest goes on the IMMA.
                    body[-j - 1].stall += st - 11
                    st = 11
                if lasty is not None:
                    yl = lasty
            if pat:
                st = pat[1 + j]
            if first == "add" and j == 0:
                body.append(Ins("IADD3 R%d, R%d, 0x1, RZ" % (r, r), stall=st, yld=yl))
            elif kind == "x3":
                # A plain step with three register sources, two of them sums made 20 IMMA earlier.
                a = ACC + 4 * ((i + 20) % 40) + 2 * (j & 1)
                body.append(Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (r, r, a, a + 1), stall=st, yld=yl))
            elif kind == "p2":
                # Two register sources, neither a sum.
                body.append(Ins("PRMT R%d, R%d, 0x6420, R%d" % (r, r, PLAIN + (na + 7) % 16), stall=st, yld=yl))
            else:
                body.append(Ins("IADD3 R%d, R%d, 0x1, RZ" % (r, r), stall=st, yld=yl))
    # The loop's test and count sit after IMMA 35, the branch after the last IMMA. The test reads the count
    # before the add writes it (a reader wants its value 4 cycles old or more), and the two take their cycles
    # out of the stall of the instruction before them.
    idx = [i for i, x in enumerate(body) if x.text.startswith("IMMA")][36]
    prev = body[idx - 1]
    left = max(1, prev.stall - 2)
    if prev.stall >= 3:
        prev.stall = 1
    body[idx:idx] = [Ins("ISETP.NE.AND P0, PT, R%d, 0x1, PT" % CNT, stall=1),
                     Ins("IADD3 R%d, R%d, -0x1, RZ" % (CNT, CNT), stall=left)]
    body[0].label = ".L" + tag
    out += body
    if bar:
        out.append(Ins("BAR.SYNC.DEFER_BLOCKING 0x0", stall=1))
    out.append(Ins("@P0 BRA .L" + tag, stall=5))
    return out


def stream(a, b):
    c = [Ins("S2R R0, SR_TID.X", wsb=0, stall=1),
         Ins("IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x184]", stall=1)]
    for r in range(A_FRAG, A_FRAG + 4):
        c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, 0x%x" % (r, 0x01020304 + r)))
    for r in range(B_FRAG, B_FRAG + 16):
        c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, 0x%x" % (r, 0x02010403 + r)))
    for r in range(ACC, ACC + 160, 2):
        c.append(Ins("CS2R R%d, SRZ" % r))
    for r in range(PLAIN, PLAIN + 16, 2):
        c.append(Ins("CS2R R%d, SRZ" % r))
    c.append(Ins("SHF.R.U32.HI R2, RZ, 0x7, R0", wait=(0,), stall=6))
    c.append(Ins("ISETP.NE.AND P0, PT, R2, RZ, PT", stall=13, yld=False))
    c.append(Ins("@P0 BRA .GB", stall=5))
    c += group("A", **a)
    c.append(Ins("BRA .END", stall=5))
    c += group("B", **b)
    c.append(Ins("NOP", stall=1, label=".END"))
    return c


def gaps_clump(n):
    return [0] * 39 + [n]


def gaps_first(n, per=1):
    return [per if i < n else 0 for i in range(40)]


def variants():
    v = {}
    plain = dict(gaps=[0] * 40)
    v["n0"] = (plain, plain)
    for n in (1, 2, 4, 8, 16, 32, 64):
        v["c1_%d" % n] = (dict(gaps=gaps_clump(n)), plain)
    for n in (4, 8, 16, 32, 64):
        v["c2_%d" % n] = (dict(gaps=gaps_clump(n), s=2), plain)
        v["cn_%d" % n] = (dict(gaps=gaps_clump(n), y=False), plain)
    for n in (8, 32):
        v["cn2_%d" % n] = (dict(gaps=gaps_clump(n), s=2, y=False), plain)
    for n in (16, 32, 64):
        # Each group's time with the other still running.
        v["c1_%d_a" % n] = (dict(gaps=gaps_clump(n)), dict(gaps=[0] * 40, mult=2))
        v["c1_%d_b" % n] = (dict(gaps=gaps_clump(n), mult=2), plain)
    for n in (8, 16, 32, 40):
        v["s1_%d" % n] = (dict(gaps=gaps_first(n)), plain)
    for g in (2, 3, 4, 5, 6, 7):
        v["sg_%d" % g] = (dict(gaps=[g] * 40), plain)
        v["bg_%d" % g] = (dict(gaps=[g] * 40), dict(gaps=[g] * 40))
    v["bg_1"] = (dict(gaps=[1] * 40), dict(gaps=[1] * 40))
    for g in (2, 3):
        v["t2_%d" % g] = (dict(gaps=[g] * 40, s=2), plain)
        v["bt2_%d" % g] = (dict(gaps=[g] * 40, s=2), dict(gaps=[g] * 40, s=2))
    for g in (2, 4, 7):
        v["bn_%d" % g] = (dict(gaps=[g] * 40, y=False), dict(gaps=[g] * 40, y=False))
        v["bl_%d" % g] = (dict(gaps=[g] * 40, lasty=False), dict(gaps=[g] * 40, lasty=False))
    # No pad: an IMMA's adds at stall 1 and nothing keeping the warp's next IMMA 8 cycles off.
    for g in (1, 2, 4):
        v["bp0_%d" % g] = (dict(gaps=[g] * 40, pad=0), dict(gaps=[g] * 40, pad=0))
    # A pad of 16: the warp's own next IMMA is held a whole IMMA period off.
    for g in (2, 4):
        v["bp16_%d" % g] = (dict(gaps=[g] * 40, pad=16), dict(gaps=[g] * 40, pad=16))
    # A barrier every 40 IMMA.
    for g in (0, 1, 2, 3, 4):
        v["bb_%d" % g] = (dict(gaps=[g] * 40, bar=True), dict(gaps=[g] * 40, bar=True))
    v["bbc_32"] = (dict(gaps=gaps_clump(32), bar=True), dict(gaps=gaps_clump(32), bar=True))
    v["bbh_32"] = (dict(gaps=[0] * 19 + [32] + [0] * 20, bar=True), dict(gaps=gaps_clump(32), bar=True))
    # The IMMA's own marks, IMMA only.
    for st in (1, 2, 4, 11):
        v["i%d" % st] = (dict(gaps=[0] * 40, istall=st), dict(gaps=[0] * 40, istall=st))
    v["iy"] = (dict(gaps=[0] * 40, iy=False), dict(gaps=[0] * 40, iy=False))
    v["iy_a"] = (dict(gaps=[0] * 40, iy=False), plain)
    return v


def variants2():
    """Hold 4: does a plain step cost by what it reads? Both warps, g steps after every IMMA."""
    v = {}
    for g in (3, 5):
        for kind in ("add", "p2", "x3"):
            for ru in (0, 1):
                d = dict(gaps=[g] * 40, kind=kind, ru=ru)
                v["k%s_%d%s" % (kind, g, "u" if ru else "")] = (d, dict(d))
    for ru in (0, 1):
        d = dict(gaps=[3] * 40, kind="x3", ru=ru, bar=True)
        v["kx3b_3%s" % ("u" if ru else "")] = (d, dict(d))
    return v


def variants4():
    """Hold 6: the map. Both warps, the same two-register steps after every IMMA, every stall written out."""
    pats = []
    pats += [(a, 8 - a) for a in (2, 3, 5, 6, 7)] + [(5, 1), (6, 1)]
    pats += [(a, 1, 7 - a) for a in (2, 3, 4, 5, 6)] + [(5, 1, 1), (4, 1, 1), (5, 2, 1)]
    pats += [(4, 1, 1, 1), (5, 1, 1, 2), (5, 1, 2, 1), (5, 2, 1, 1)]
    pats += [(1, 1, 1, 1, 4), (2, 1, 1, 1, 3), (3, 1, 1, 1, 2), (4, 1, 1, 1, 1), (5, 1, 1, 1, 1), (3, 1, 1, 1, 1)]
    pats += [(1, 1, 1, 1, 1, 3), (2, 1, 1, 1, 1, 2), (3, 1, 1, 1, 1, 1), (4, 1, 1, 1, 1, 1), (5, 1, 1, 1, 1, 1)]
    pats += [(1,) + (1,) * 5 + (2,), (2,) + (1,) * 6, (3,) + (1,) * 6, (5,) + (1,) * 6]
    pats += [(1,) * 8, (2,) + (1,) * 7, (5,) + (1,) * 7]
    v = {}
    for p in pats:
        d = dict(gaps=[len(p) - 1] * 40, kind="p2", pat=p)
        v["m" + "".join(str(x) for x in p)] = (d, dict(d))
    return v


def variants3():
    """Hold 5: the same steps issued later behind the IMMA (w cycles), or with a one-register step first."""
    v = {}
    for kind, g in (("p2", 3), ("x3", 3), ("x3", 5)):
        for w in (2, 3, 4, 5, 6):
            d = dict(gaps=[g] * 40, kind=kind, ist1=w)
            v["w%s_%d_%d" % (kind, g, w)] = (d, dict(d))
        d = dict(gaps=[g + 1] * 40, kind=kind, first="add")
        v["f%s_%d" % (kind, g)] = (d, dict(d))
    for kind in ("p2", "x3"):
        for g in (1, 2):
            d = dict(gaps=[g] * 40, kind=kind)
            v["k%s_%d" % (kind, g)] = (d, dict(d))
        d = dict(gaps=[1] * 40, kind=kind, ist1=4)
        v["w%s_1_4" % kind] = (d, dict(d))
        d = dict(gaps=gaps_clump(32), kind=kind, bar=True)
        v["c%s_32b" % kind] = (d, dict(d))
    return v


def main():
    cubin, sass, nvd, outdir, listfile = sys.argv[1:6]
    text = asm89.read_text(sass)
    os.makedirs(outdir, exist_ok=True)
    lines = ["# sched.py's timing-only streams (they compute nothing real). k 8192, 64 tiles a block."]
    for name, (a, b) in ({"2": variants2, "3": variants3, "4": variants4}.get(sys.argv[6] if len(sys.argv) > 6 else "", variants)()).items():
        code = stream(a, b)
        n, room = asm89.patch(cubin, text, os.path.join(outdir, "sc_" + name + ".cubin"), code, nvd)
        lines.append("r1@sc_%s sc_%s 8192 64" % (name, name))
        print("sc_%s: %d instructions of %d" % (name, n, room))
    open(listfile, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
