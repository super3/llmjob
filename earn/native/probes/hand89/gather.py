"""gather.py: a TIMING-ONLY stream with the instruction mix of the "gather" draw (S/algo/proto.js, the maths
agent's form): each warp builds its own mma fragments with ldmatrix.trans used as a gather on a static table.
No stage, no store, no barrier, nothing shared between warps. IT COMPUTES NOTHING REAL: the table it reads is
whatever the probe's kernel left in shared memory, so its sums are wrong on purpose and no figure from it is a
mining rate. It prices the form: can its instructions sit among the IMMA and keep the tensor pipe full?

  python gather.py <base.cubin> <base.sass> <nvdisasm> <out.cubin> [key=value ...]

A k-step of a warp (mt m16 tiles by eight n8 tiles, 8 mt IMMA):
  - one 16-byte load of the lane's four table offsets of the next k-step (A's two k halves, B's two), and four
    shifts (the real kernel's host would store them ready: the shifts make this stream a little dearer);
  - a gather for each k half of each m16 tile and of each pair of n8 tiles: one LDSM.x4.trans whose 32 lanes
    each address one 16-byte table entry (16 p for the lane's k, p the rank index, so the banks collide as the
    real ones would), two IADD3 (fill + plus - minus), two PRMT (the two fragment words);
  - A's fragments of a tile are built while the tile before runs, into the other of two register sets; B's of
    the next k-step are built in the last gaps, each pair behind its last IMMA of this k-step.
The readout is each lane's XOR of its own sums once a chunk (two chains, one XOR3 a gap) into the lane's own
word in shared memory. The probe counts 160 IMMA a chunk: with mt=6 (192) multiply its pipe figure by 1.2.

Keys: mt (6 or 5), conf (1: offsets from the probe's random table; 0: every lane its own row, no bank shared
inside a matrix), noread, nold (the gathers' LDSM replaced by register moves), pad, lead (gaps from a gather's
LDSM to its adds), py/iy/ly (yield flags of plain instructions, IMMA, the last instruction of a gap).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402
import fold  # noqa: E402
from fold import addi, andi, mov, muladd, shr  # noqa: E402

CFG = dict(mt=6, conf=1, noread=0, nold=0, pad=8, lead=2, py=1, iy=1, ly=1, ru=0, ist=1, pol=0, bshift=0, ashift=0)
AF, BF, ACC = [4, 8], 12, 28
OQ, POFF, CNT, TRL = [240, 244], 248, 250, 251
XR, W = [2, 3], 0
SB_OFF, SB_RD = 4, 5
TR = 0x15100


def build(cfg):
    mt = cfg["mt"]
    ni, tot = 8 * mt, 32 * mt
    tq = list(range(ACC + 4 * ni, 240, 4))            # temporaries: whatever the sums leave
    G = [[] for _ in range(tot)]
    live = []                                         # (first gap, last gap) of each gather's temporaries
    seq = [0]

    def gather(g_ld, g_add, g_prm, oreg, goff, d1, d2):
        live.append((g_ld, g_prm, len(live)))
        k = len(live) - 1
        sb = seq[0] % 4
        seq[0] += 1
        return k, sb, g_ld, g_add, g_prm, oreg, goff, d1, d2

    plans = []
    lead = cfg["lead"]
    for s in range(4):
        b = ni * s
        cur, nxt = OQ[s & 1], OQ[(s + 1) & 1]
        if cfg["conf"] in (1, 3):
            G[b + 2].append(Ins("LDG.E.128 R%d, [R%d.64+0x%x]" % (nxt, POFF, (s + 1) * 0x200), wsb=SB_OFF))
            for q in range(4):
                G[b + 6 + q // 2].append(Ins(shr(nxt + q, nxt + q, 4), wait=(SB_OFF,) if q == 0 else ()))
            if cfg["conf"] == 3:       # a check on the addresses: nothing over a group's 128 entries
                for q in range(4):
                    G[b + 8 + q // 2].append(Ins(andi(nxt + q, nxt + q, 0x7f0)))
        for m in range(1, mt):                         # tile m's fragments, during tile m - 1
            a = AF[(m + (mt & 1) * s) & 1]
            for h in range(2):
                g0 = b + 8 * (m - 1) + h + cfg["ashift"]
                plans.append(gather(g0, g0 + lead, g0 + lead + 2, cur + h, 2048 * m, a + 2 * h, a + 2 * h + 1))
        a = AF[((mt & 1) * (s + 1)) & 1]               # the next k-step's first tile, during the last tile
        for h in range(2):
            g0 = b + ni - 12 + h
            plans.append(gather(g0, g0 + lead, g0 + 4, nxt + h, 0, a + 2 * h, a + 2 * h + 1))
        for p in range(4):                             # the next k-step's B, each pair behind its last IMMA
            for h in range(2):
                g_prm = b + ni - 6 + 2 * p + h
                g0 = g_prm - 3 - cfg["bshift"]
                plans.append(gather(g0, g0 + lead, g_prm, nxt + 2 + h, 2048 * (6 + p), BF + 4 * p + h,
                                    BF + 4 * p + 2 + h))
    # Temporaries: a quad from the LDSM's gap to the PRMT's. The quads go round in the order the LDSM are
    # issued, with a count of quads that divides the loop's gathers, so the next pass of the loop meets the
    # same assignment. Scoreboards 0-3 go round the same way, so an add waits on its own load only.
    order = sorted(live)
    pool = next(n for n in range(len(tq), 0, -1) if len(order) % n == 0)
    quad = {k: tq[n % pool] for n, (_a, _b, k) in enumerate(order)}
    for g_ld, g_prm, k in order:
        for g2, _p2, k2 in order:
            if k2 != k and quad[k2] == quad[k]:
                assert not any(g_ld < g2 + w <= g_prm for w in (-tot, 0, tot)),                     "two gathers in flight share temporaries (gaps %d and %d)" % (g_ld, g2)
    for n, (g_ld, g_prm, k) in enumerate(order):
        plans[k] = (plans[k][0], n % 4) + plans[k][2:] + (quad[k],)
    for k, sb, g_ld, g_add, g_prm, oreg, goff, d1, d2, q in plans:
        ld = "LDSM.16.MT88.4 R%d, [R%d%s]" % (q, oreg, "+0x%x" % goff if goff else "")
        if cfg["nold"]:
            G[g_ld % tot].append(Ins(mov(q, oreg)))
            w = ()
        else:
            G[g_ld % tot].append(Ins(ld, wsb=sb))
            w = (sb,)
        G[g_add % tot].append(Ins("IADD3 R%d, -R%d, 0x3f3f3f3f, R%d" % (q, q + 2, q), wait=w))
        G[g_add % tot].append(Ins("IADD3 R%d, -R%d, 0x3f3f3f3f, R%d" % (q + 1, q + 3, q + 1)))
        G[g_prm % tot].append(Ins("PRMT R%d, R%d, 0x6420, R%d" % (d1, q, q + 1)))
        G[g_prm % tot].append(Ins("PRMT R%d, R%d, 0x7531, R%d" % (d2, q, q + 1)))
    if not cfg["noread"]:
        for t in range(mt):
            x = XR[t & 1]
            g0 = 3 * ni + 8 * t + 10
            for j in range(16):
                r = ACC + 4 * (8 * t + j // 2) + 2 * (j & 1)
                if j == 0 and t < 2:
                    ins = "LOP3.LUT R%d, R%d, R%d, RZ, 0x3c, !PT" % (x, r, r + 1)
                else:
                    ins = "LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (x, x, r, r + 1)
                G[(g0 + j) % tot].append(Ins(ins))
        e = 3 * ni + 8 * (mt - 1) + 26
        G[(e - 4) % tot].append(Ins("LDS R%d, [R%d]" % (W, TRL), wsb=SB_RD))
        G[(e + 0) % tot].append(Ins("SHF.L.W.U32.HI R%d, R%d, 0xd, R%d" % (W, W, W), wait=(SB_RD,)))
        G[(e + 1) % tot].append(Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (W, W, XR[0], XR[1])))
        G[(e + 2) % tot].append(Ins("STS [R%d], R%d" % (TRL, W)))
    G[tot - 12].append(Ins("ISETP.NE.AND P0, PT, R%d, 0x1, PT" % CNT))
    G[tot - 11].append(Ins(addi(CNT, CNT, -1)))

    # ---- the prologue
    S = tq
    c = [Ins("S2R R0, SR_TID.X", wsb=SB_RD), Ins("ULDC.64 UR6, c[0x0][0x118]")]
    for r in list(range(4, ACC + 4 * ni, 2)) + [2] + list(range(tq[0], 240, 2)):
        c.append(Ins("CS2R R%d, SRZ" % r))
    c.append(Ins(andi(S[0], 0, 0xf), wait=(SB_RD,)))
    c.append(Ins(shr(S[0] + 1, 0, 4)))
    c.append(Ins(muladd(S[0] + 2, 0, 0x20)))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, 0x1" % (S[1])))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, c[0x0][0x184]" % CNT))
    c.append(Ins(andi(S[0] + 1, S[0] + 1, 1)))
    c.append(Ins(addi(TRL, S[0] + 2, TR)))
    c.append(Ins(muladd(S[0] + 3, S[0] + 1, 0x20, S[0])))
    c.append(Ins(muladd(S[0] + 3, S[0] + 3, 0x10)))
    c.append(Ins("IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x170]" % (POFF, S[0] + 3, S[1])))
    if cfg["conf"] in (1, 3):
        c.append(Ins("LDG.E.128 R%d, [R%d.64]" % (OQ[0], POFF), wsb=SB_OFF))
        for q in range(4):
            c.append(Ins(shr(OQ[0] + q, OQ[0] + q, 4), wait=(SB_OFF,) if q == 0 else ()))
        if cfg["conf"] == 3:
            for q in range(4):
                c.append(Ins(andi(OQ[0] + q, OQ[0] + q, 0x7f0)))
    elif cfg["conf"] in (4, 5):
        # Rank indices whose residues mod 8 are the lane's own row number, so a matrix's rows share no bank,
        # except that row 7 (conf 4), or rows 6 and 7 (conf 5), take the residue of row 0 (and 1): one or two
        # two-way collisions in every matrix. The searched order of k leaves about one row of 8 colliding.
        nrow = 7 if cfg["conf"] == 4 else 6
        c.append(Ins(andi(S[2], 0, 7)))
        c.append(Ins("ISETP.GE.U32.AND P0, PT, R%d, 0x%x, PT" % (S[2], nrow), stall=13, yld=False))
        c.append(Ins("@P0 " + addi(S[2], S[2], -nrow)))
        for q in range(8):
            c.append(Ins(addi(OQ[0] + q, 0, 256 * q + 1)))
        for q in range(8):
            c.append(Ins(muladd(OQ[0] + q, OQ[0] + q, 0x2545f491)))
        for q in range(8):
            c.append(Ins(shr(OQ[0] + q, OQ[0] + q, 23)))
        for q in range(8):
            c.append(Ins(andi(OQ[0] + q, OQ[0] + q, 0x78)))
        for q in range(8):
            c.append(Ins(muladd(OQ[0] + q, OQ[0] + q, 1, S[2])))
        for q in range(8):
            c.append(Ins(muladd(OQ[0] + q, OQ[0] + q, 0x10)))
    elif cfg["conf"] == 2:
        # No offsets table: every lane's eight rank indices made here from its thread number by a multiply,
        # the same every k-step. Rows of a matrix collide in the banks as chance has it, and every address
        # is inside its group's table whatever the probe's offsets hold.
        for q in range(8):
            c.append(Ins(addi(OQ[0] + q, 0, 256 * q + 1)))
        for q in range(8):
            c.append(Ins(muladd(OQ[0] + q, OQ[0] + q, 0x2545f491)))
        for q in range(8):
            c.append(Ins(shr(OQ[0] + q, OQ[0] + q, 23)))
        for q in range(8):
            c.append(Ins(andi(OQ[0] + q, OQ[0] + q, 0x7f)))
        for q in range(8):
            c.append(Ins(muladd(OQ[0] + q, OQ[0] + q, 0x10)))
    else:
        for q in range(8):
            c.append(Ins(muladd(OQ[0] + q, 0, 0x10)))
    c.append(Ins("CS2R R%d, SRZ" % S[0]))
    c.append(Ins("CS2R R%d, SRZ" % (S[0] + 2)))
    fold.check(c, "prologue", fix=True)
    # ---- the loop
    loop = []
    pad = cfg["pad"]
    for g in range(tot):
        s, i = divmod(g, ni)
        mb, nb = divmod(i, 8)
        items = G[g]
        n = len(items)
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
        loop.append(Ins("IMMA.16832.S8.S8 R%d, R%d.ROW, R%d.COL, R%d" % (
            ACC + 4 * i, AF[(mb + (mt & 1) * s) & 1], BF + 2 * nb, ACC + 4 * i), stall=pad if not items else ist_g,
            yld=bool(cfg["iy"]), reuse=1 if cfg["ru"] and nb < 7 else 0, label=".L0" if g == 0 else None))
        for k, ins in enumerate(items):
            ins = Ins(ins.text, ins.stall, bool(cfg["py"]), ins.wsb, ins.rsb, ins.wait)
            if k == len(items) - 1:
                ins.stall = max(ins.stall, last_g)
                ins.yld = bool(cfg["ly"])
            loop.append(ins)
    loop.append(Ins("@P0 BRA .L0", stall=5))
    bad = fold.check(loop + loop, "loop")
    return c + loop, bad, max(len(x) for x in G), sum(len(x) for x in G)


def main():
    cubin, sass, nvd, out = sys.argv[1:5]
    cfg = dict(CFG)
    for kv in sys.argv[5:]:
        k, v = kv.split("=")
        assert k in cfg, k
        cfg[k] = int(v)
    code, bad, most, plain = build(cfg)
    if bad:
        print("%d latency problems: not written" % bad)
        sys.exit(1)
    text = asm89.read_text(sass)
    n, room = asm89.patch(cubin, text, out, code, nvd)
    print("%s: %d instructions of %d; a chunk is %d IMMA and %d others (%.2f an IMMA), %d at most in a gap" % (
        os.path.basename(out), n, room, 32 * cfg["mt"], plain, plain / (32.0 * cfg["mt"]), most))


if __name__ == "__main__":
    main()
