"""price.py: TIMING-ONLY streams that price one step of a fold's draw at a time, for sm_89. Every stream computes
wrong results on purpose; none is a mining rate.

  python price.py <base.cubin> <base.sass> <nvdisasm> <out.cubin> <spec> [gfold key=value ...]

The skeleton is gfold.py's loop with ready fragments (its key direct=1, run with gprobe's fake 35): a k-step is 48
IMMA, ten LDSM (one a fragment, from tables whose bytes are spread as noised operands are), one 16-byte load of
offsets from global memory, and the readout. That is the floor every fold shares: operands that are already there.
<spec> adds N more of one step a warp a k-step, spread over the k-step's gaps, e.g. "prmt=8" or, for a design's
whole mix, "ldsm_sc=7,iadd3=14,prmt=14,sts16=4,bar_k=1" ("" is the skeleton alone):

  ldsm_seq   LDSM.x4.trans, the eight lanes of a matrix read eight entries in a row (as a stage's rows are read)
  ldsm_sc    the same as a gather: every lane another entry, no two of a matrix in the same banks
  ldsm_col   a gather whose matrix has three two-way bank collisions (chance gives 2.76 rows of 8)
  lds4       a 4-byte shared load, every lane its own word (4 bytes apart: every bank once)
  lds16      a 16-byte shared load, every lane its own 16 bytes
  sts16      a 16-byte shared store, every lane its own 16 bytes (it writes back what the table holds)
  prmt       PRMT of two registers of operand-like bytes
  iadd3      IADD3 of two registers and a constant (the draw's subtract)
  add        IADD3 of one register and a constant
  xor3       LOP3 of three registers
  ldg_l1     a 16-byte global load of bytes the SM's cache holds (the k-step's own offsets again)
  ldg_l2     a 16-byte global load of bytes no SM has read lately (a walk through the 33 MB of B's tables)
  bar_k      a barrier at every k-step's end (the count is ignored)
  bar_c      a barrier at every chunk's end
  bar_n      a barrier at the end of every N-th chunk (N a power of two)
  nop        a NOP that holds its warp for 11 cycles (two to a gap make the warp miss its turn at the pipe)

"kind=N/C" puts the N steps C to a gap.

A load of the chunk's last k-step sets a scoreboard the loop's end waits for; the others set none, so that no
step of the skeleton waits for a step that is being priced.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402
import fold  # noqa: E402
from fold import addi, andi, muladd, shr  # noqa: E402
import gfold  # noqa: E402

# Registers the skeleton leaves free (gfold.layout: the five quads start at 220; the kernel keeps R1, R253, R254).
Q0, Q1, ADDR = 220, 224, 236        # operand-like bytes; the loads' sink; a lane's own table address
PG = 228                            # 228, 229: the walk's pointer
SCR = [230, 231, 235]               # sinks of the ALU steps
T_OCT, RSEQ, R4L, T_LANE, RSC, RCOL = 232, 233, 234, 237, 238, 239
KINDS = ("ldsm_seq", "ldsm_sc", "ldsm_col", "lds4", "lds16", "sts16", "prmt", "iadd3", "add", "xor3", "ldg_l1",
         "ldg_l2", "bar_k", "bar_c", "bar_n", "nop")
CAP = 4


def parse(spec):
    out = []
    for kv in spec.split(","):
        if not kv:
            continue
        k, v = kv.split("=")
        assert k in KINDS, k
        v, _, c = v.partition("/")
        out.append((k, int(v), int(c) if c else 1))
    return out


def prologue(cfg, R, spec):
    """gfold's prologue for ready fragments, with the address registers the priced steps read."""
    pro = gfold.prologue(cfg, R)
    S = R["quads"][0]
    at = next(n for n, x in enumerate(pro) if x.text == muladd(S, S, 0x10))
    add = [Ins(andi(T_LANE, S, 7)),                          # t = lane & 7
           Ins(muladd(T_OCT, S, 5)),
           Ins(shr(RCOL, T_LANE, 1)),
           Ins(addi(T_OCT, T_OCT, 3)),
           Ins(muladd(RCOL, T_LANE, 5, RCOL)),               # 5 t + (t >> 1): residues 0,5,3,0,6,3,1,6
           Ins(andi(T_OCT, T_OCT, 15)),                      # the lane's octet of entries: (5 lane + 3) & 15
           Ins(andi(RCOL, RCOL, 7)),
           Ins(muladd(RSC, T_OCT, 8, T_LANE)),
           Ins(muladd(RCOL, T_OCT, 8, RCOL)),
           Ins(muladd(RSEQ, S, 0x10)),                       # 16 lane
           Ins(muladd(R4L, S, 4)),                           # 4 lane
           Ins(muladd(RSC, RSC, 0x10)),
           Ins(muladd(RCOL, RCOL, 0x10))]
    if any(x[0] == "ldg_l2" for x in spec):
        # Where this warp's walk begins: a place in B's tables taken from the clock, 512 bytes a step.
        add += [Ins("CS2R R%d, SR_CLOCKLO" % SCR[0]),
                Ins("IMAD.MOV.U32 R%d, RZ, RZ, 0x1" % SCR[2]),
                Ins(shr(SCR[0], SCR[0], 3)),
                Ins(andi(SCR[0], SCR[0], 0x7fff)),
                Ins(muladd(SCR[0], SCR[0], 0x200, RSEQ)),
                Ins("IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x168]" % (PG, SCR[0], SCR[2]))]
    pro[at:at] = add
    # gfold's prologue ends by clearing the third and fourth quad: the fourth holds address registers here.
    keep = []
    for x in pro:
        if x.text in ("CS2R R%d, SRZ" % (Q0 + 12), "CS2R R%d, SRZ" % (Q0 + 14)):
            continue
        if x.text == "CS2R R%d, SRZ" % (Q0 + 8) and any(y[0] == "ldg_l2" for y in spec):
            continue
        keep.append(x)
    return keep


def step(kind, j, s, n, R, last):
    """The j-th priced step of its kind in k-step s."""
    sb = gfold.SB_RD if last else None
    tab = 2048 * ((5 * j + 3 * s) % 12)
    off = "+0x%x" % tab if tab else ""
    a, b = Q0 + (j + s) % 4, Q0 + (j + s + 1 + j % 3) % 4
    d = SCR[(j + s) % 3]
    if kind in ("ldsm_seq", "ldsm_sc", "ldsm_col"):
        reg = dict(ldsm_seq=RSEQ, ldsm_sc=RSC, ldsm_col=RCOL)[kind]
        return Ins("LDSM.16.MT88.4 R%d, [R%d%s]" % (Q1, reg, off), wsb=sb)
    if kind == "lds4":
        return Ins("LDS R%d, [R%d%s]" % (Q1 + j % 4, R4L, off), wsb=sb)
    if kind == "lds16":
        return Ins("LDS.128 R%d, [R%d%s]" % (Q1, RSEQ, off), wsb=sb)
    if kind == "sts16":
        return Ins("STS.128 [R%d], R%d" % (ADDR, Q0))
    if kind == "prmt":
        return Ins("PRMT R%d, R%d, 0x%s, R%d" % (d, a, ("6420", "7531", "5140", "7362")[j % 4], b))
    if kind == "iadd3":
        return Ins("IADD3 R%d, -R%d, 0x3f3f3f3f, R%d" % (d, a, b))
    if kind == "add":
        return Ins(addi(d, a, 0x10 + 16 * j))
    if kind == "xor3":
        return Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (d, a, Q0 + (j + s + 1) % 4, Q0 + (j + s + 2) % 4))
    if kind == "nop":
        return Ins("NOP", stall=11)
    if kind == "ldg_l1":
        return gfold.ldg(Q1, R["poff"], 0x200 * (s + 1) if s < 3 else 0, wsb=sb)
    if kind == "ldg_l2":
        return gfold.ldg(Q1, PG, 0x200 * (n * s + j), wsb=sb)
    raise ValueError(kind)


def extras(G, cfg, R, spec):
    mt = cfg["mt"]
    ni, tot = 8 * mt, 32 * mt
    for ki, (kind, n, clump) in enumerate(spec):
        if kind in ("bar_k", "bar_c", "bar_n"):
            continue
        for s in range(4):
            for j in range(n):
                # "kind=N/C": the N steps go C to a gap (the default is one), the gaps spread over the k-step.
                jc = j // clump
                g = ni * s + (ni * jc * clump + 5 * ki + 3) // n + 3
                while len(G[g % tot]) >= max(CAP, clump) and j % clump == 0:
                    g += 1
                G[g % tot].append((step(kind, j, s, n, R, s == 3), False))
        if kind == "ldg_l2":
            G[3 * ni + 9].append((Ins(addi(PG, PG, 0x800 * n)), False))


def build(cfg, spec):
    gfold.UR = cfg["ur"]
    R = gfold.layout(cfg["mt"], 0)
    pro = prologue(cfg, R, spec)
    bad = gfold.check(pro, "prologue", fix=True)
    G, iwait = gfold.chunk(cfg, R)
    extras(G, cfg, R, spec)
    loop, ep, most = gfold.emit(G, iwait, cfg)
    kinds = {x[0]: x[1] for x in spec}
    if "bar_n" in kinds:
        # A barrier every N chunks (N a power of two): the test rides in the chunk's last k-step.
        G[3 * 8 * cfg["mt"] + 10].append((Ins(andi(SCR[1], R["cnt"], kinds["bar_n"] - 1)), False))
        G[3 * 8 * cfg["mt"] + 12].append((Ins("ISETP.NE.AND P2, PT, R%d, 0x0, PT" % SCR[1]), False))
        loop, ep, most = gfold.emit(G, iwait, cfg)
        loop.insert(len(loop) - 1, Ins("@!P2 BAR.SYNC.DEFER_BLOCKING 0x0", stall=1))
    if "bar_k" in kinds or "bar_c" in kinds:
        # A barrier ahead of the first IMMA of every k-step but the chunk's first, and at the chunk's end.
        ni, out, seen = 8 * cfg["mt"], [], 0
        for x in loop:
            if x.text.startswith("IMMA"):
                if seen and seen % ni == 0 and "bar_k" in kinds:
                    out.append(Ins("BAR.SYNC.DEFER_BLOCKING 0x0", stall=1))
                seen += 1
            if "BRA" in x.text:
                out.append(Ins("BAR.SYNC.DEFER_BLOCKING 0x0", stall=1))
            out.append(x)
        loop = out
    body = [x for x in loop if "BRA" not in x.text and not x.text.startswith("BAR")]
    bad += gfold.check(pro + body + body + body, "loop")
    bad += gfold.check(body + ep, "epilogue", fix=True)
    # Every load that set a scoreboard has landed before the function returns.
    ep.append(Ins(addi(R["tmp"], R["tmp"], 0), wait=(0, 1, 2, 3, gfold.SB_OFF, gfold.SB_RD)))
    ep.append(Ins("NOP", stall=8))
    return pro + loop + ep, bad, most, sum(len(x) for x in G)


def main():
    cubin, sass, nvd, out, spec = sys.argv[1:6]
    cfg = dict(gfold.CFG)
    cfg["direct"] = 1
    for kv in sys.argv[6:]:
        k, v = kv.split("=")
        assert k in cfg, k
        cfg[k] = int(v, 0)
    spec = parse(spec)
    code, bad, most, plain = build(cfg, spec)
    if bad:
        print("%d latency problems: not written" % bad)
        sys.exit(1)
    text = asm89.read_text(sass)
    regs, preds = asm89.clobbers(text)
    used = set()
    for ins in code:
        if "BRA" in ins.text or ins.text.startswith("ULDC"):
            continue
        _r, w, _k = fold.rw(ins.text)
        used |= {x for x in w if not isinstance(x, str)}
    assert used <= regs and 1 in preds, "writes registers the kernel keeps: %s" % sorted(used - regs)
    n, room = asm89.patch(cubin, text, out, code, nvd)
    tot = 32 * cfg["mt"]
    print("%s: %d instructions of %d; a chunk is %d IMMA and %d others (%.2f an IMMA), %d at most in a gap" % (
        os.path.basename(out), n, room, tot, plain, plain / float(tot), most))


if __name__ == "__main__":
    main()
