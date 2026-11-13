"""fold.py: the draw fold's chunk loop written by hand for sm_89, instruction by instruction, with its own register
assignment, stall counts and scoreboards. It replaces the loop function ptxas compiled from fold_h.cuh (asm89.patch)
and computes the same transcripts: hprobe's CPU check judges it.

  python fold.py <base.cubin> <base.sass> <nvdisasm> <out.cubin> [key=value ...]

What the loop does is fold_h.cuh's PD_LOOPFN (the third design, a barrier every k-step): 40 IMMA a k-step into 160
sums a thread, A's fragments tile by tile and B's a k-step ahead by LDSM from the stage, one unit of A and one of B
drawn into the stage two k-steps ahead, every tile's sums XORed into its regions' words once a chunk.

What is different is the order. A k-step is 40 IMMA and everything else sits in the gaps behind them, a few
instructions to a gap (key cap), placed here and not by a compiler:
  - a fragment load sits 7 or more IMMA ahead of its first reader and has its own scoreboard, so no IMMA waits;
  - a draw unit's steps follow each other a gap or more apart (table reads two gaps ahead of the subtracts);
  - a tile's readout is one chain, one XOR3 a gap, started 10 IMMA after the tile's last (key rsh0/rsh1 shift it);
  - the chunk's bookkeeping (stage buffers turned, offsets pointer, counter) is in the last gaps of k-step 3.

The defaults are the best form that passed the CPU check on rig01's RTX 4090 (11 November 2026): 96.0 to 96.3% of
the tensor pipe at 2538 to 2548 MHz, where the same fold as ptxas compiles it holds 81.7%. What got it there,
each measured (probes/README.md, "The loop written by hand"):
  - ist=5: an IMMA that has plain steps behind it gets a stall of 5. A step that reads two or more registers costs
    the pipe when it issues 1 to 4 cycles behind its warp's IMMA and nothing from 5 (91.3% to 95.2%);
  - sf=1: a gap's table reads and one-register steps go first, and the IMMA's stall shrinks by their count;
  - ub=20, rsh1=4, py=0: unit B starts at gap 20, warps 4 to 7 read out 4 IMMA later than warps 0 to 3, and the
    plain steps' yield flag is off (85% to 91% before ist);
  - hold=1: read scoreboards on the loads, stores and atomics whose address or value register is overwritten
    soon after. Without them two placements in 45 computed wrong sums. The latency checker here does not model
    that hazard: every build must pass hprobe's CPU check before its figure counts.

Registers (R1 and R252-R254 are the kernel's): A's two fragment sets R4-R11, B's eight R12-R27, the sums R28-R187,
draw temporaries R188-R219, offsets R220-R227, lane constants and counters R228-R249.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402

TABA, SJB, SJA, STB, STA, STGB, STGA, TR, SA = 0x8000, 0x210, 0x150, 0x2100, 0x1500, 0xD000, 0x11200, 0x15100, 0xA0
AF, BF, ACC = [4, 8], 12, 28
D, E = [188, 192, 196, 200], [204, 208, 212, 216]
OA, OB = 220, 224
TABBL, TABAL, DSTBL, FRAGBL = 228, 229, 230, 231
AD, AFR = [232, 233, 234], [235, 236, 237]
TRL, TRW, ROT, CH, LATER0, IDXB, POFF = 238, 239, 240, 241, 242, 243, 244
X, T0, ONE = [246, 247, 248], 249, 0
# Scoreboards: A's fragment loads, B's (one for all four), the holds (read scoreboards, see key hold), the
# draw's two pairs of table reads, the offsets.
SB_AF, SB_B0, SB_B1, SB_HOLD, SB_DX, SB_DY, SB_OFF = 0, 1, 1, 2, 3, 4, 5

CFG = dict(cap=3, cap0=5, cap3=5, pad=8, ua=1, ub=20, ua1=None, ub1=None, rsh0=0, rsh1=4, ly=1, py=0, iy=1, ru=0, ist=5, pol=2, sf=1, hold=1, lead=2, bstall=1, rst=1, pst=2,
           # r1: see unit(). ah: the readout's atomics are held too (ah=0 with hold=1 is the form that once
           # computed 8 regions of 10,240 wrong).
           r1=1, ah=1,
           # Timing only (each computes wrong results on purpose): pieces left out.
           nodraw=0, noread=0, nold=0, nost=0, nobar=0, noldsm=0)


def acc(mb, nb):
    return ACC + 4 * (8 * mb + nb)


def muladd(d, a, imm, c=255):
    cs = "RZ" if c == 255 else "R%d" % c
    if c == 255 and imm & (imm - 1) == 0:
        return "IMAD.SHL.U32 R%d, R%d, 0x%x, RZ" % (d, a, imm)
    return "%s R%d, R%d, 0x%x, %s" % ("IMAD.IADD" if imm == 1 else "IMAD", d, a, imm, cs)


def addi(d, a, imm):
    return "IADD3 R%d, R%d, %s0x%x, RZ" % (d, a, "-" if imm < 0 else "", abs(imm))


def andi(d, a, imm):
    return "LOP3.LUT R%d, R%d, 0x%x, RZ, 0xc0, !PT" % (d, a, imm)


def shr(d, a, n):
    return "SHF.R.U32.HI R%d, RZ, 0x%x, R%d" % (d, n, a)


def mov(d, s):
    return "IMAD.MOV.U32 R%d, RZ, RZ, R%d" % (d, s)


# ---------------------------------------------------------------- what an instruction reads and writes
def rw(text):
    """(reads, writes, kind) of an instruction's text; registers as numbers, the predicate P0 as 'P0'."""
    t = text.replace(".reuse", "")
    reads, writes = set(), set()
    m = re.match(r"@!?P(\d) (.*)$", t)
    if m:
        reads.add("P" + m[1])
        t = m[2]
    op, _, rest = t.partition(" ")
    regs = [int(x) for x in re.findall(r"(?<!U)R(\d+)", rest)]     # uniform registers (URn) are not tracked

    def span(r, n):
        return set(range(r, r + n))
    if op == "R2UR":
        return reads | {regs[0]}, writes, "store"
    if op.startswith("IMMA"):
        d, a, b, c = regs
        return reads | span(a, 4) | span(b, 2) | span(c, 4), span(d, 4), "imma"
    if op.startswith("LDG.E.64"):
        return reads | span(regs[1], 2), span(regs[0], 2), "load"
    if op.startswith(("LDSM", "LDS.128", "LDG")):
        return reads | (span(regs[1], 2) if op.startswith("LDG") else {regs[1]}), span(regs[0], 4), "load"
    if op.startswith("STS.128"):
        return reads | {regs[0]} | span(regs[1], 4), writes, "store"
    if op.startswith("CCTL"):
        return reads | span(regs[0], 2), writes, "store"
    if op == "STS":
        return reads | set(regs), writes, "store"
    if op == "LDS":
        return reads | {regs[1]}, {regs[0]}, "load"
    if op == "LDL":
        return reads | {regs[1]}, {regs[0]}, "load"
    if op == "STL":
        return reads | set(regs), writes, "store"
    if op.startswith("ATOMS"):
        return reads | set(regs), writes, "store"
    if op.startswith("ISETP"):
        return reads | set(regs), {"P0"}, "setp"
    if op.startswith("CS2R"):
        return reads, span(regs[0], 2), "alu"
    if op.startswith("S2R"):
        return reads, {regs[0]}, "load"
    if op.startswith("IMAD.WIDE"):
        return reads | set(regs[1:]), span(regs[0], 2), "alu"
    if op.startswith(("BRA", "BAR", "NOP", "ULDC", "DEPBAR")):
        return reads, writes, "ctl"
    if op.startswith("SYNCS.PHASECHK"):
        # sm_120's mbarrier wait (rfold.py w8m): reads the parity register, writes a predicate, on a scoreboard.
        p = re.match(r"P(\d),", rest)
        return reads | {regs[-1]}, {"P" + p[1]}, "load"
    if op.startswith("SYNCS.ARRIVE"):
        return reads, writes, "store"
    if op.startswith("VOTEU"):
        # rfold.py w8mu: reads a predicate, writes a uniform one (not tracked).
        p = re.search(r"\bP(\d)\b", rest)
        return reads | ({"P" + p[1]} if p else set()), writes, "ctl"
    assert regs, text
    return reads | set(regs[1:]), {regs[0]}, "alu"


LAT = dict(alu=5, imma=28, setp=13)


def check(code, name, fix=False):
    """Walks straight-line code summing stalls: every register read must be `LAT` cycles behind its writer, or
    wait on the scoreboard its writer set. With fix, the stall of the instruction before a too-early reader
    is raised instead (prologue and epilogue). Returns the count of problems."""
    last, t, bad = {}, 0, 0
    for i, ins in enumerate(code):
        if ins.text.startswith(("BRA", "@P0 BRA", "@!P0 BRA")):
            t += ins.stall
            continue
        reads, writes, kind = rw(ins.text)
        for sb in ins.wait:
            # A wait on a scoreboard covers every load that set it before: their registers are ready from here.
            for r, (tw, k, s0) in list(last.items()):
                if k == "load" and s0 == sb:
                    last[r] = (tw, "done", None)
        for r in reads:
            if r not in last or last[r][1] == "done":
                continue
            tw, k, sb = last[r]
            if k == "load":
                ok = (sb is not None and sb in ins.wait) or t - tw >= 100
                need = 0
            else:
                need = LAT[k] - (t - tw)
                ok = need <= 0
            if not ok:
                if fix and need > 0 and i > 0:
                    code[i - 1].stall += need
                    assert code[i - 1].stall <= 11, (name, code[i - 1])
                    t += need
                else:
                    bad += 1
                    if bad <= 10:
                        print("%s: %r reads R%s %d cycles after its %s writer" % (name, ins, r, t - tw, k))
        for w in writes:
            last[w] = (t, kind, ins.wsb)
        t += ins.stall
    return bad


# ---------------------------------------------------------------- the prologue
def prologue():
    S = list(range(188, 220))
    c = [Ins("S2R R0, SR_TID.X", wsb=SB_OFF), Ins("ULDC.64 UR6, c[0x0][0x118]")]
    for r in range(ACC, ACC + 160, 2):
        c.append(Ins("CS2R R%d, SRZ" % r))
    c.append(Ins("CS2R R%d, SRZ" % X[0]))
    c.append(Ins("CS2R R%d, SRZ" % X[2]))
    c.append(Ins(andi(S[0], 0, 0xf), wait=(SB_OFF,)))          # l16
    c.append(Ins(shr(S[1], 0, 4)))
    c.append(Ins(shr(S[2], 0, 5)))                             # warp
    c.append(Ins(andi(S[3], 0, 7)))                            # lane & 7
    c.append(Ins(shr(S[4], 0, 3)))
    c.append(Ins(andi(S[19], 0, 3)))                           # lane & 3
    c.append(Ins(andi(S[1], S[1], 1)))                         # hL
    c.append(Ins(andi(S[4], S[4], 1)))                         # lane bit 3
    c.append(Ins(andi(S[5], S[2], 3)))                         # warp & 3
    c.append(Ins(addi(S[6], S[0], 6)))
    c.append(Ins(muladd(TABBL, S[0], 0x10)))                   # table B: 16 * l16
    c.append(Ins(shr(S[17], S[2], 2)))                         # warp >> 2
    c.append(Ins(muladd(S[20], S[19], 0x280)))                 # (lane & 3) * 640
    c.append(Ins(shr(S[6], S[6], 4)))                          # 1 for places 10-15
    c.append(Ins(muladd(S[9], S[1], 8, S[2])))                 # the lane's k-pair of a unit: warp + 8 hL
    c.append(Ins(muladd(S[12], S[4], 8, S[3])))
    c.append(Ins(muladd(S[13], S[1], 0x10)))
    c.append(Ins(muladd(S[15], S[1], 8, S[3])))
    c.append(Ins(muladd(S[16], S[4], 0x10)))
    c.append(Ins(muladd(S[8], S[6], 0x80)))
    c.append(Ins(muladd(S[21], S[5], 0xa00, S[20])))
    c.append(Ins(addi(S[10], TABBL, STGB)))
    c.append(Ins(muladd(S[14], S[5], 0x80, S[13])))
    c.append(Ins(muladd(S[18], S[17], 0xa0, S[16])))
    c.append(Ins(muladd(IDXB, S[9], 0x10)))
    c.append(Ins("IADD3 R%d, -R%d, 0x0, R%d" % (S[7], S[8], TABBL)))      # 16 * wA
    c.append(Ins(muladd(S[22], S[1], 0x40, S[21])))
    c.append(Ins(muladd(DSTBL, S[9], SJB, S[10])))
    c.append(Ins(addi(S[14], S[14], STGB)))
    c.append(Ins(addi(S[18], S[18], STGA)))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, 0x1" % ONE))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, c[0x0][0x184]" % T0))
    c.append(Ins(addi(TABAL, S[7], TABA)))
    c.append(Ins(addi(S[11], S[7], STGA)))
    c.append(Ins(addi(TRL, S[22], TR)))
    c.append(Ins(muladd(FRAGBL, S[12], SJB, S[14])))
    c.append(Ins(muladd(AFR[0], S[15], SJA, S[18])))
    c.append(Ins("IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x170]" % (POFF, IDXB, ONE)))
    c.append(Ins(shr(S[23], T0, 4)))
    c.append(Ins(muladd(AD[0], S[9], SJA, S[11])))
    c.append(Ins(mov(TRW, TRL)))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % CH))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % ROT))
    c.append(Ins(addi(AFR[1], AFR[0], STA)))
    c.append(Ins(addi(AFR[2], AFR[0], 2 * STA)))
    c.append(Ins(addi(LATER0, S[23], -1)))
    c.append(Ins(addi(AD[1], AD[0], STA)))
    c.append(Ins(addi(AD[2], AD[0], 2 * STA)))
    # The first k-step's fragments (stage 0, drawn by the kernel before the call) and the offsets of the stage
    # k-step 0 draws.
    c.append(Ins("LDSM.16.MT88.4 R%d, [R%d]" % (AF[0], AFR[0]), wsb=SB_AF))
    for p in range(4):
        c.append(Ins("LDSM.16.MT88.4 R%d, [R%d%s]" % (BF + 4 * p, FRAGBL, "+0x%x" % (32 * p) if p else ""),
                     wsb=SB_B0 if p < 2 else SB_B1))
    c.append(Ins("LDG.E.128 R%d, [R%d.64+0x400]" % (OB, POFF), wsb=SB_OFF))
    c.append(Ins("LDG.E.128 R%d, [R%d.64+0x500]" % (OA, POFF), wsb=SB_OFF))
    c.append(Ins("ISETP.GE.U32.AND P0, PT, R%d, 0x4, PT" % S[2], stall=13, yld=False))
    c.append(Ins("@P0 BRA .G1", stall=5))
    return c


# ---------------------------------------------------------------- one chunk of one group
def unit(s, which, cfg):
    """The draw unit of k-step s as micro-groups [(instructions, least gaps after the group before)]."""
    if which == "A":
        q, tab, off, ldg = D, TABAL, OA, (s + 3) * 0x200 + 0x100
        d0, d1 = "[R%d]" % AD[(s + 2) % 3], "[R%d+0x%x]" % (AD[(s + 2) % 3], SA)
    else:
        q, tab, off, ldg = E, TABBL, OB, (s + 3) * 0x200
        b = (s & 1) * STB
        d0 = "[R%d+0x%x]" % (DSTBL, b) if b else "[R%d]" % DSTBL
        d1 = "[R%d+0x%x]" % (DSTBL, b + 0x100)
    # r1: every reader of a loaded register waits on the load's scoreboard itself (the sixth round's first rule:
    # a step right behind a waiting step can read the register before the load has written it). r1=0 is the
    # loop as the fifth round measured it, with only the first of each four waiting.
    r1 = cfg["r1"]
    adds = [Ins(muladd(q[j], tab, 1, off + j), wait=(SB_OFF,) if j == 0 or r1 else ()) for j in range(4)]
    lds = [Ins("LDS.128 R%d, [R%d]" % (q[j], q[j]), wsb=SB_DX if j < 2 else SB_DY) for j in range(4)]
    lds.append(Ins("LDG.E.128 R%d, [R%d.64+0x%x]" % (off, POFF, ldg), wsb=SB_OFF))
    dx = [Ins("IADD3 R%d, -R%d, 0x3f3f3f3f, R%d" % (q[0] + i, q[1] + i, q[0] + i),
              wait=(SB_DX,) if i == 0 or r1 else ()) for i in range(4)]
    dy = [Ins("IADD3 R%d, -R%d, 0x3f3f3f3f, R%d" % (q[2] + i, q[3] + i, q[2] + i),
              wait=(SB_DY,) if i == 0 or r1 else ()) for i in range(4)]
    pr = []
    for h in range(2):          # the first store's four words, then the second's
        for i in range(2):
            x, y = q[0] + 2 * h + i, q[2] + 2 * h + i
            pr.append(Ins("PRMT R%d, R%d, 0x5140, R%d" % (q[1 + 2 * h] + 2 * i, x, y)))
            pr.append(Ins("PRMT R%d, R%d, 0x7362, R%d" % (q[1 + 2 * h] + 2 * i + 1, x, y)))
    st = [Ins("STS.128 %s, R%d" % (d0, q[1])), Ins("STS.128 %s, R%d" % (d1, q[3]))]
    if which == "A" and s == 3 and cfg["hold"]:
        st = [Ins(i.text, rsb=SB_HOLD) for i in st]     # A's buffer register is turned at the chunk's end
    if cfg["nold"]:
        # Timing only: no table read and no offsets read; the words are whatever the registers hold.
        adds = [Ins(i.text) for i in adds]
        lds = [Ins(mov(q[j], q[j])) for j in range(4)] + [Ins(mov(off, off))]
        dx, dy = [Ins(i.text) for i in dx], [Ins(i.text) for i in dy]
    if cfg["nost"]:
        st = [Ins(mov(q[1], q[1])), Ins(mov(q[3], q[3]))]
    return [(adds, 0), (lds, 1), (dx, None), (dy, 0), (pr, 1), (st, 1)]


def chunk(grp, cfg):
    """Gaps 0..159 of a chunk (gap g: behind IMMA g), each a list of (Ins, wrapped). A wrapped item belongs to
    the readout of the chunk before; the epilogue runs those once more after the last chunk."""
    G = [[] for _ in range(160)]
    cap = cfg["cap"]

    def put(g, ins, wrapped=False, front=False):
        (G[g].insert(0, (ins, wrapped)) if front else G[g].append((ins, wrapped)))

    for s in range(4) if not cfg["noldsm"] else ():
        b = 40 * s
        # hold: a load or store reads its address register for as long as the memory unit keeps it waiting.
        # The chunk's end overwrites A's buffer registers and the offsets pointer a few gaps behind their last
        # users, so those users set a read scoreboard and the overwriting moves wait on it (ptxas does the
        # same). Without it two placements in 45 computed wrong sums for tile 0 of warps 0-3: the load of the
        # next chunk's first fragment read a register the turn had already changed.
        held = SB_HOLD if s == 3 and cfg["hold"] else None
        for m in range(1, 5):
            put(b + 8 * (m - 1), Ins("LDSM.16.MT88.4 R%d, [R%d+0x%x]" % (AF[(m + s) & 1], AFR[s % 3], 32 * m),
                                     wsb=SB_AF, rsb=held if m == 4 else None))
        put(b + 32, Ins("LDSM.16.MT88.4 R%d, [R%d]" % (AF[(s + 1) & 1], AFR[(s + 1) % 3]), wsb=SB_AF, rsb=held))
        bn = STB - (s & 1) * STB
        for p in range(4):
            o = bn + 32 * p
            put(b + 33 + 2 * p, Ins("LDSM.16.MT88.4 R%d, [R%d%s]" % (BF + 4 * p, FRAGBL, "+0x%x" % o if o else ""),
                                    wsb=SB_B0 if p < 2 else SB_B1))
    # The readout's words of this chunk: set in k-step 1, after the last wrapped use.
    ep = [Ins(shr(2, CH, 4)), Ins(muladd(3, CH, 4)), Ins("IADD3 R2, -R2, 0x0, R%d" % LATER0), Ins(andi(3, 3, 0x3c)),
          Ins(muladd(2, 2, 0xd)), Ins(muladd(TRW, TRL, 1, 3)), Ins(andi(ROT, 2, 0x1f))]
    for i, ins in enumerate(ep):
        put(40 + i, ins)
    # The readout: tile t's chain of 16, its rotate and its XOR into the word, one a gap.
    rsh = cfg["rsh1"] if grp else cfg["rsh0"]
    for t in range(5) if not cfg["noread"] else ():
        x = X[t % 3]
        g0 = 120 + 8 * t + 10 + rsh
        steps = [Ins("LOP3.LUT R%d, R%d, R%d, RZ, 0x3c, !PT" % (x, acc(t, 0), acc(t, 0) + 1)),
                 Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (x, x, acc(t, 0) + 2, acc(t, 0) + 3))]
        for nb in range(1, 8):
            steps.append(Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (x, x, acc(t, nb), acc(t, nb) + 1)))
            steps.append(Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (x, x, acc(t, nb) + 2, acc(t, nb) + 3)))
        steps.append(Ins("SHF.L.W.U32.HI R%d, R%d, R%d, R%d" % (x, x, ROT, x)))
        imm = 128 * ((grp * 5 + t) >> 1)
        steps.append(Ins("ATOMS.XOR RZ, [R%d%s], R%d" % (TRW, "+0x%x" % imm if imm else "", x),
                         rsb=SB_HOLD if cfg["hold"] and cfg["ah"] else None))
        if cfg["hold"] and cfg["ah"]:
            # The atomic XOR keeps reading its value register until every lane's word is done (four lanes share
            # a word). The next tile that takes the same register starts its chain by overwriting it: it waits.
            steps[0].wait = (SB_HOLD,)
        for j, ins in enumerate(steps):
            g = g0 + j * cfg["rst"]
            put(g % 160, ins, wrapped=g >= 160)
            # A sum is read 2 IMMA or more behind the IMMA that made it and before the next one that adds to it.
            if j < 16:
                nb = j // 2
                first, nxt = 120 + 8 * t + nb + 2, 160 + 8 * t + nb - 1
                assert first <= g <= nxt, (t, j, g)
    assert max(120 + 8 * 4 + 10 + rsh + 17 * cfg["rst"] - 160, 0) < 40, "the readout must end in k-step 0"
    # The chunk's end, in k-step 3's last gaps: A's stage buffers turn by one, the offsets move on, the count.
    e = 153
    hw = (SB_HOLD,) if cfg["hold"] else ()
    put(e + 0, Ins(mov(T0, AFR[0])))
    put(e + 0, Ins(mov(2, AD[0])))
    put(e + 0, Ins(addi(IDXB, IDXB, 0x800)))
    put(e + 1, Ins(mov(AFR[0], AFR[1]), wait=hw))
    put(e + 1, Ins(mov(AD[0], AD[1]), wait=hw))
    put(e + 1, Ins(addi(CH, CH, 1)))
    put(e + 2, Ins(mov(AFR[1], AFR[2]), wait=hw))
    put(e + 2, Ins(mov(AD[1], AD[2]), wait=hw))
    # The offsets pointer: its last users are the k-step's two LDG, which set the offsets' scoreboard.
    put(e + 2, Ins("IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x170]" % (POFF, IDXB, ONE),
                   wait=(SB_OFF,) if cfg["hold"] else ()))
    put(e + 3, Ins(mov(AFR[2], T0), wait=hw))
    put(e + 3, Ins(mov(AD[2], 2), wait=hw))
    put(e + 3, Ins("ISETP.GE.U32.AND P0, PT, R%d, c[0x0][0x184], PT" % CH))
    # The draw: each unit's groups go into the first gaps with room, from the unit's start.
    ua = cfg["ua1"] if grp and cfg["ua1"] is not None else cfg["ua"]
    ub = cfg["ub1"] if grp and cfg["ub1"] is not None else cfg["ub"]
    for s in range(4):
        b = 40 * s
        for which, start in (("A", ua), ("B", ub)) if not cfg["nodraw"] else ():
            g, prev = b + start, None
            for items, least in unit(s, which, cfg):
                if prev is not None:
                    g = max(g, prev + (cfg["lead"] if least is None else least))
                for ins in items:
                    # K-steps 3 and 0 also hold the tiles' readout chains: their gaps take more (keys cap3, cap0).
                    while len(G[g]) >= (cfg["cap3"] if s == 3 else cfg["cap0"] if s == 0 else cap):
                        g += 1
                    put(g, ins)
                prev = g
            # The stores are ahead of the k-step's barrier, and in k-step 3 ahead of the turn of A's buffers.
            assert g <= b + (32 if s == 3 else 38), "k-step %d's unit %s ends in gap %d" % (s, which, g - b)
    return G


def emit(G, grp, cfg):
    """A group's chunk loop and its epilogue."""
    out = []
    pad = cfg["pad"]
    for g in range(160):
        s, i = divmod(g, 40)
        mb, nb = divmod(i, 8)
        wait = []
        if nb == 0:
            wait.append(SB_AF)
        if mb == 0 and nb == 0:
            wait.append(SB_B0)
        if mb == 0 and nb == 4:
            wait.append(SB_B1)
        items = [ins for ins, _w in G[g]]
        n = len(items)
        nsingle = 0
        if cfg["sf"]:
            # sf: a gap's loads and one-register steps go first, right behind the IMMA (only a step that reads
            # two or more registers has to keep its distance), and the IMMA's stall shrinks by their count.
            def single(ins):
                r, _w, kind = rw(ins.text)
                # Not the fragment loads: they overwrite registers the IMMA just ahead may still be reading.
                return (kind == "load" and not ins.text.startswith("LDSM")) or (kind == "alu" and len(r) <= 1)
            items = [x for x in items if single(x)] + [x for x in items if not single(x)]
            nsingle = sum(1 for x in items if single(x))
        # pol: where a gap's spare cycles go. 0: behind the last plain step (the IMMA's stall is key ist).
        # 1: behind the IMMA, the steps at stall 1 (a two-register step issued 5 cycles behind its IMMA costs
        # the pipe nothing; issued 1 to 4 behind it costs up to 2 cycles: measured). 2: the IMMA's stall is
        # ist and nothing is added behind the steps.
        if cfg["pol"] == 1:
            ist_g, last_g = max(1, pad - n), 1
        elif cfg["pol"] == 2:
            ist_g, last_g = cfg["ist"], 1
        else:
            ist_g, last_g = cfg["ist"], max(1, pad - cfg["ist"] - (n - 1))
        if cfg["sf"]:
            ist_g = max(1, ist_g - nsingle) if nsingle < n else 1
        imma = Ins("IMMA.16832.S8.S8 R%d, R%d.ROW, R%d.COL, R%d" % (acc(mb, nb), AF[(mb + s) & 1], BF + 2 * nb,
                                                                 acc(mb, nb)),
                   stall=pad if not items else ist_g, yld=bool(cfg["iy"]), wait=wait,
                   # ru: as ptxas marks them, a tile's IMMA keep A's fragment in the reuse latch (all but the
                   # tile's last), so seven of eight do not read it from the register file again.
                   reuse=1 if cfg["ru"] and nb < 7 else 0,
                   label=".L%d" % grp if g == 0 else None)
        out.append(imma)
        for k, ins in enumerate(items):
            ins = Ins(ins.text, ins.stall, bool(cfg["py"]), ins.wsb, ins.rsb, ins.wait)
            if k == len(items) - 1:
                ins.stall = max(ins.stall, last_g)
                ins.yld = bool(cfg["ly"])
            if cfg["ru"] >= 2 and ins.text.startswith("PRMT") and "0x5140" in ins.text and k + 1 < len(items)                     and items[k + 1].text.startswith("PRMT") and "0x7362" in items[k + 1].text:
                ins.reuse = 5      # both sources stay latched for the pair's second PRMT, as ptxas does
            out.append(ins)
        if i == 39 and not cfg["nobar"]:
            out.append(Ins("BAR.SYNC.DEFER_BLOCKING 0x0", stall=cfg["bstall"]))
    out.append(Ins("@!P0 BRA .L%d" % grp, stall=5))
    loop_len = len(out)
    # After the last chunk: the wrapped part of the readout once more.
    ep = []
    for g in range(160):
        for ins, wrapped in G[g]:
            if wrapped:
                ep.append(Ins(ins.text, cfg["pst"], ins.yld, ins.wsb, ins.rsb, ins.wait))
    return out, loop_len, ep


def build(cfg):
    pro = prologue()
    bad = check(pro, "prologue", fix=True)
    code = list(pro)
    for grp in (0, 1):
        G = chunk(grp, cfg)
        loop, n, ep = emit(G, grp, cfg)
        body = loop[:n]
        # Two passes of the loop behind the prologue, then the epilogue: every read's distance from its writer.
        bad += check(pro + body + body, "group %d loop" % grp)
        check(body + ep, "group %d epilogue" % grp, fix=True)
        if grp == 1:
            code.append(Ins("NOP", label=".G1"))
        code += loop + ep
        if grp == 0:
            code.append(Ins("BRA .END", stall=5))
    code.append(Ins("NOP", label=".END"))
    return code, bad


def main():
    cubin, sass, nvd, out = sys.argv[1:5]
    cfg = dict(CFG)
    for kv in sys.argv[5:]:
        k, v = kv.split("=")
        assert k in cfg, k
        cfg[k] = None if v == "None" else int(v)
    code, bad = build(cfg)
    if bad:
        print("%d latency problems: not written" % bad)
        sys.exit(1)
    text = asm89.read_text(sass)
    regs, _preds = asm89.clobbers(text)
    used = set()
    for ins in code:
        _r, w, _k = rw(ins.text)
        used |= {x for x in w if x != "P0"}
    assert used <= regs, "writes registers the kernel keeps: %s" % sorted(used - regs)
    n, room = asm89.patch(cubin, text, out, code, nvd)
    plain = sum(1 for i in code if not i.text.startswith("IMMA"))
    print("%s: %d instructions of %d (%d IMMA)" % (os.path.basename(out), n, room, n - plain - 3))


if __name__ == "__main__":
    main()
