"""cfold.py: the copy fold's chunk loop written by hand for sm_89, instruction by instruction, with its own
registers, stall counts and scoreboards. It replaces the loop function ptxas compiled from cfold_h.cuh
(asm89.patch) and computes the same transcripts: cprobe's CPU check judges it.

  python cfold.py <base.cubin> <base.sass> <nvdisasm> <out.cubin> [key=value ...]

What the loop does is cfold_h.cuh's: 8 mt IMMA a k-step into 32 mt sums a thread, A's fragments tile by tile and
B's a k-step ahead by LDSM from the stage, every tile's sums XORed into its regions' words once a chunk, a
barrier a k-step. It is hand89/fold.py's loop with the draw taken out: where that loop drew two units a k-step
(54 instructions), this one copies 3 or 4 units of 16 bytes a lane from global memory, where A' and B' lie
ready in the stage's own layout.

The copy unit has two forms (key cp):
  0  one LDG.E.128 and one STS.128. With la=1 the load is issued a k-step ahead of its store, into one of two
     sets of quads, so a slow load has a whole k-step before anything waits for it (mt 5 only: 32 registers).
  1  one LDGSTS.E.BYPASS.128, the release fold's cp.async: no register, one instruction. K-step s issues stage
     s + lead and commits the group (LDGDEPBAR); the k-step's end waits until all but the last lead - 2 groups
     have landed (DEPBAR.LE). The global address takes only a 12-bit offset, so each source pointer moves on
     once a k-step.

  2  TIMING ONLY, no copy at all: the staged DRAW's instruction mix in the copy units' places, at the least a
     block can draw. A unit is two address adds, two 16-byte shared loads (stand-ins for the table words at the
     plus and the minus rank index), four IADD3, four PRMT and one 16-byte shared store; two global loads a
     k-step stand in for the offsets. It prices a draw fold at 3.5 stores a warp a k-step before one is built.

Rules kept (build-notes, "Rules for hand loops, each from a fault"; check() enforces the first):
  - a reader of a loaded register waits on the load's scoreboard itself, or issues 8 cycles or more behind a
    step that did;
  - nothing waits on a load's scoreboard right behind the load;
  - before the loop returns, one step waits on every scoreboard a load set, and the copies are drained;
  - a register a load, a store or a copy still reads is not overwritten: the reader sets a read scoreboard and
    the writer waits on it (key hold);
  - an IMMA with plain steps behind it gets stall 5 (key ist), loads and one-register steps first (sf).

Timing-only keys (each computes wrong results on purpose): nocopy (the stage keeps what the tile's prologue
copied), fix (every k-step copies the tile's k-step 2 again: the source stays in the SM's cache), nost (loads
with no stores), nobar, noread.

Registers (R0, R1 and R252-R254 are the kernel's): R2-R3 scratch, A's two fragment sets R4-R11, B's eight
R12-R27, the sums from R28, then (cp 0) eight quads, then the pointers and lane constants.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402
import fold  # noqa: E402
from fold import addi, andi, muladd, shr, mov  # noqa: E402

CFG = dict(mt=5, cp=0, la=1, sd=2, lead=3, g0=2, g1=10, g2=18, g3=26, gs1=0, a2g1=0,
           pad=8, ist=5, sf=1, py=0, ly=1, iy=1, rsh0=0, rsh1=4, pst=2, bstall=1, hold=1, ur=4,
           # cp 1: ca=1 copies through the SM's cache (LDGSTS.E.128; 0 is the release's BYPASS); lst the stall
           # behind each LDGSTS; lpad never-run LDS ahead of a k-step's first LDGSTS, as ptxas pads them.
           ca=0, lst=1, lpad=0,
           # Timing only (each computes wrong results on purpose). nodep: the k-step's end does not wait for
           # the copies.
           nocopy=0, fix=0, nost=0, nobar=0, noread=0, nodep=0, dm=0)
AF, BF, ACC = [4, 8], 12, 28
# Scoreboards. 0 is the copies' (LDGDEPBAR sets it, DEPBAR.LE waits on it).
SB_AF, SB_B, SB_HOLD, SB_Q = 1, 2, 3, [4, 5]
LAT = dict(alu=5, imma=28, setp=13)
SETTLE = 8
PRED = 0                    # the predicate of the loop's branch (one the compiled function writes)
CHUNKS = "c[0x0][0x170]"    # the kernel's third parameter


def layout(cfg):
    """Shared-memory places (cfold_h.cuh) and register numbers for mt m16 tiles a warp."""
    mt = cfg["mt"]
    ga = 4 * mt
    L = dict(GA=ga, SJB=528, SJA=16 * ga + 16, RECB=8192, RECA=256 * ga)
    L["STB"], L["STA"] = 16 * L["SJB"], 16 * L["SJA"]
    L["STGB"], L["STGA"] = 0, 4 * L["STB"]
    L["TR"] = L["STGA"] + 4 * L["STA"]
    L["PTR"] = L["TR"] + mt * 2048 + 192
    L["SCR0"] = L["TR"] + mt * 2048 + 512
    base = ACC + 32 * mt
    if cfg["cp"] == 0:
        nset = 2 if cfg["la"] == 1 else 1
        L["Q"] = [[base + 16 * st + 4 * u for u in range(4)] for st in range(nset)]
        base += 16 * nset
    names = ["PB", "PBh", "PA1", "PA1h", "PA2", "PA2h", "FRAGBL", "AFR", "DSTB", "DSTA1", "DSTA2", "TRW", "ROT",
             "CH", "LATER0"]
    for i, n in enumerate(names):
        L[n] = base + i
    base += len(names)
    # The readout's three chain registers. Short of registers (192 sums and quads), two of them are R2 and R3:
    # the chunk's bookkeeping uses those in k-step 1 only, and the chains run from k-step 3 into k-step 0.
    L["X"] = [base, base + 1, base + 2] if base + 3 <= 252 else [2, 3, base]
    assert L["X"][2] < 252, "no registers left: mt %d with cp %d la %d" % (mt, cfg["cp"], cfg["la"])
    if cfg["cp"] == 2:
        # A 16-byte load or store takes four registers from a multiple of four (one from R238 is "an illegal
        # instruction" when it runs, though it assembles and disassembles).
        L["ZR"] = L["X"][2] + 1
        base = (L["ZR"] + 4) & ~3
        L["DA"], L["DB"], L["SCR"] = base, base + 4, base + 8
        assert base + 12 <= 252, "no registers left for the draw mix"
    assert all(q % 4 == 0 for st in L.get("Q", []) for q in st) and not (L["PB"] | L["PA1"] | L["PA2"]) & 1
    return L


def acc(mb, nb):
    return ACC + 4 * (8 * mb + nb)


def mem(base, off=0):
    return "[R%d+0x%x]" % (base, off) if off else "[R%d]" % base


def ldg(dst, base, off, ur, **k):
    """A 16-byte global load: asm89's LDG with the uniform register the kernel was compiled with."""
    text = "LDG.E.128 R%d, [R%d.64%s]" % (dst, base, "+0x%x" % off if off else "")
    lo, hi = asm89.assemble(text)
    lo = (lo & ~(0xFF << 32)) | (ur << 32)
    return Ins(text, raw=(lo, hi), **k)


def rw(text):
    """fold.rw, and the forms this loop adds."""
    t = text.replace(".reuse", "")
    pre = set()
    m = re.match(r"@!?P(\d) (.*)$", t)
    if m:
        pre.add("P" + m[1])
        t = m[2]
    op, _, rest = t.partition(" ")
    regs = [int(x) for x in re.findall(r"(?<![A-Z])R(\d+)", rest)]
    if t.startswith("@!PT "):
        return set(), set(), "ctl"
    if op.startswith("LDGSTS"):
        return pre | {regs[0], regs[1], regs[1] + 1}, set(), "store"
    if op == "LDGDEPBAR" or op.startswith("DEPBAR"):
        return pre, set(), "ctl"
    if op == "LDS.64":
        return pre | {regs[1]}, {regs[0], regs[0] + 1}, "load"
    if op.startswith("ISETP"):
        return pre | set(regs), {"P" + re.match(r"ISETP\S* P(\d)", t)[1]}, "setp"
    return fold.rw(text)


def check(code, name, fix=False):
    """Walks straight-line code summing stalls (gfold.check). A register an ALU step, an IMMA or a compare wrote is
    read LAT cycles or more behind it. A register a load wrote is read by an instruction that waits on the load's
    scoreboard, or SETTLE cycles or more behind one that did. With fix, the stall of the instruction before a
    too-early reader is raised instead. Time is the sum of the stalls, and a warp's IMMA are 16 cycles apart or
    more (the tensor pipe takes one in 16 cycles). Returns the count of problems."""
    last, t, bad, t_imma = {}, 0, 0, -16
    for i, ins in enumerate(code):
        if "BRA" in ins.text:
            t += ins.stall
            continue
        reads, writes, kind = rw(ins.text)
        if kind == "imma":
            t = max(t, t_imma + 16)
            t_imma = t
        for sb in ins.wait:
            for r, (tw, k, s0, ts) in list(last.items()):
                if k == "load" and s0 == sb and ts is None:
                    last[r] = (tw, k, s0, t)
        for r in reads:
            if r not in last:
                continue
            tw, k, sb, ts = last[r]
            if k == "load":
                ok = ts is not None and (sb in ins.wait or t - ts >= SETTLE)
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
            last[w] = (t, kind, ins.wsb, None)
        t += ins.stall
    return bad


# ---------------------------------------------------------------- the prologue
def prologue(cfg, L):
    mt, ur = cfg["mt"], cfg["ur"]
    ga = L["GA"]
    T = [ACC + i for i in range(32)]            # temporaries: sum registers, zeroed below
    (tid, lane7, hL, b3, l16, warp, w3, wr, lane3, q, w1, wb1, jU, t0, t1, t2, t3, t4, t5, t6, t7, j2, oB, oA1, oA2,
     zero) = T[:26]
    q0, q1 = SB_Q
    c = [Ins("S2R R%d, SR_TID.X" % tid, wsb=q0),
         Ins("ULDC.64 UR%d, c[0x0][0x118]" % ur, raw=(0x0000460000007ab9 | (ur << 16), 0xa00)),
         Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % zero),
         Ins("LDS.64 R%d, %s" % (L["PA1"], mem(zero, L["PTR"])), wsb=q1),
         Ins("LDS.64 R%d, %s" % (L["PB"], mem(zero, L["PTR"] + 8)), wsb=q1),
         Ins("NOP", stall=8)]
    w0 = (q0,)
    c.append(Ins(andi(lane7, tid, 7), wait=w0))
    c.append(Ins(shr(hL, tid, 4), wait=w0))
    c.append(Ins(shr(q, tid, 3), wait=w0))
    c.append(Ins(andi(l16, tid, 0xf), wait=w0))
    c.append(Ins(shr(warp, tid, 5), wait=w0))
    c.append(Ins(andi(lane3, tid, 3), wait=w0))
    c.append(Ins(andi(hL, hL, 1)))
    c.append(Ins(andi(b3, q, 1)))
    c.append(Ins(andi(q, q, 3)))
    c.append(Ins(andi(w3, warp, 3)))
    c.append(Ins(shr(wr, warp, 2)))
    c.append(Ins(andi(w1, warp, 1)))
    c.append(Ins(shr(wb1, warp, 1)))
    c.append(Ins(andi(wb1, wb1, 1)))
    c.append(Ins(muladd(jU, hL, 8, warp)))
    # B's fragment rows and A's.
    c.append(Ins(muladd(t0, b3, 8, lane7)))
    c.append(Ins(muladd(t1, w3, 0x80)))
    c.append(Ins(muladd(t1, hL, 0x10, t1)))
    c.append(Ins(muladd(L["FRAGBL"], t0, L["SJB"], t1)))
    c.append(Ins(muladd(t2, hL, 8, lane7)))
    c.append(Ins(muladd(t3, wr, 32 * mt)))
    c.append(Ins(muladd(t3, b3, 0x10, t3)))
    c.append(Ins(muladd(L["AFR"], t2, L["SJA"], t3)))
    c.append(Ins(addi(L["AFR"], L["AFR"], L["STGA"])))
    # The copy units' words in a buffer and in a k-step's record.
    c.append(Ins(muladd(t4, l16, 0x10)))
    c.append(Ins(muladd(L["DSTB"], jU, L["SJB"], t4)))
    c.append(Ins(muladd(L["DSTA1"], jU, L["SJA"], t4)))
    c.append(Ins(addi(L["DSTA1"], L["DSTA1"], L["STGA"])))
    c.append(Ins(andi(t5, lane7, ga - 17)))
    c.append(Ins(muladd(t5, t5, 0x10)))
    c.append(Ins(addi(t5, t5, 0x100)))
    c.append(Ins(muladd(j2, w3, 4, q)))
    c.append(Ins(muladd(L["DSTA2"], j2, L["SJA"], t5)))
    c.append(Ins(addi(L["DSTA2"], L["DSTA2"], L["STGA"])))
    c.append(Ins(muladd(oB, jU, 512, t4)))
    c.append(Ins(muladd(oA1, jU, 16 * ga, t4)))
    c.append(Ins(muladd(oA2, j2, 16 * ga, t5)))
    # The pointers: the tile's two bases from shared memory, the lane's word added (the host keeps these adds
    # from carrying). cp 1: they start at the record of the stage k-step 0 issues.
    first = (2 if cfg["fix"] else cfg["lead"]) if cfg["cp"] == 1 else 0
    w1s = (q1,)
    c.append(Ins("IADD3 R%d, R%d, 0x%x, R%d" % (L["PB"], L["PB"], first * L["RECB"], oB), wait=w1s))
    c.append(Ins("IADD3 R%d, R%d, 0x%x, R%d" % (L["PA2"], L["PA1"], first * L["RECA"], oA2), wait=w1s))
    c.append(Ins(mov(L["PA2h"], L["PA1h"]), wait=w1s))
    c.append(Ins("IADD3 R%d, R%d, 0x%x, R%d" % (L["PA1"], L["PA1"], first * L["RECA"], oA1), wait=w1s))
    # The transcript words, the chunk count, the group.
    c.append(Ins(muladd(t6, lane3, 2 * mt * 64)))
    c.append(Ins(muladd(t6, w3, 8 * mt * 64, t6)))
    c.append(Ins(muladd(t6, hL, 0x40, t6)))
    c.append(Ins(addi(L["TRW"], t6, L["TR"] + 60)))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, %s" % (t7, CHUNKS)))
    c.append(Ins(shr(t7, t7, 4)))
    c.append(Ins(addi(L["LATER0"], t7, -1)))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % L["CH"]))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % L["ROT"]))
    if cfg["cp"] == 2:
        c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % L["ZR"]))
    c.append(Ins("ISETP.GE.U32.AND P%d, PT, R%d, 0x4, PT" % (PRED, warp)))
    # The first k-step's fragments (stage 0, copied by the tile's prologue before the call).
    c.append(Ins("LDSM.16.MT88.4 R%d, %s" % (AF[0], mem(L["AFR"])), wsb=SB_AF))
    for p in range(4):
        c.append(Ins("LDSM.16.MT88.4 R%d, %s" % (BF + 4 * p, mem(L["FRAGBL"], 32 * p)), wsb=SB_B))
    if cfg["cp"] == 0 and cfg["la"] and not cfg["nocopy"]:
        # Stage 2's loads, which k-step 0 stores (la 2: of the units whose store falls in the next k-step).
        gl = [cfg["g0"], cfg["g1"], cfg["g2"], cfg["g3"]]
        for u, (P, so, _d, _do, _st, rec) in enumerate(units(0, cfg, L, every=True)):
            if cfg["la"] == 1:
                c.append(ldg(L["Q"][1][u], P, 2 * rec + so, ur, wsb=SB_Q[1]))
            elif min(gl[u], gl[u] + cfg["gs1"]) + cfg["sd"] >= 8 * mt:
                c.append(ldg(L["Q"][0][u], P, 2 * rec + so, ur, wsb=SB_Q[u & 1]))
    for r in range(ACC, ACC + 32 * mt, 2):
        c.append(Ins("CS2R R%d, SRZ" % r))
    for x in L["X"]:
        c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % x))
    c.append(Ins("@P%d BRA .G1" % PRED, stall=5))
    return c


def units(grp, cfg, L, every=False):
    """A group's copy units of a k-step: (source pointer, source offset, stage pointer, its offset, a buffer's
    bytes, a record's bytes). Two of B, one of A, and A's second in warps 0 to 3 (key a2g1: in all)."""
    u = [(L["PB"], 0, L["DSTB"], 0, L["STB"], L["RECB"]), (L["PB"], 256, L["DSTB"], 0x100, L["STB"], L["RECB"]),
         (L["PA1"], 0, L["DSTA1"], 0, L["STA"], L["RECA"])]
    if grp == 0 or cfg["a2g1"] or every:
        u.append((L["PA2"], 0, L["DSTA2"], 0, L["STA"], L["RECA"]))
    return u


# ---------------------------------------------------------------- one chunk of one group
def chunk(grp, cfg, L):
    """Gaps 0 .. 32 mt - 1 of a chunk (gap g: behind IMMA g), each a list of (Ins, wrapped). A wrapped item belongs
    to the readout of the chunk before; the epilogue runs those once more after the last chunk."""
    mt, ur = cfg["mt"], cfg["ur"]
    ni, tot = 8 * mt, 32 * mt
    G = [[] for _ in range(tot)]
    hold = SB_HOLD if cfg["hold"] else None
    hw = (SB_HOLD,) if cfg["hold"] else ()

    def put(g, ins, wrapped=False):
        G[g].append((ins, wrapped))

    for s in range(4):
        b = ni * s
        for m in range(1, mt):
            put(b + 8 * (m - 1), Ins("LDSM.16.MT88.4 R%d, %s" % (AF[(m + (mt & 1) * s) & 1],
                                                              mem(L["AFR"], s * L["STA"] + 32 * m)), wsb=SB_AF))
        put(b + ni - 8, Ins("LDSM.16.MT88.4 R%d, %s" % (AF[((mt & 1) * (s + 1)) & 1],
                                                     mem(L["AFR"], ((s + 1) & 3) * L["STA"])), wsb=SB_AF))
        for p in range(4):
            put(b + ni - 7 + 2 * p, Ins("LDSM.16.MT88.4 R%d, %s" % (
                BF + 4 * p, mem(L["FRAGBL"], ((s + 1) & 3) * L["STB"] + 32 * p)), wsb=SB_B))
    # The copies.
    un = units(grp, cfg, L)
    gl = [g + (cfg["gs1"] if grp else 0) for g in (cfg["g0"], cfg["g1"], cfg["g2"], cfg["g3"])][:len(un)]
    fix, cp, la, lead = cfg["fix"], cfg["cp"], cfg["la"], cfg["lead"]
    assert max(gl) + (4 if cp or la == 2 else cfg["sd"]) < ni - 8 and min(gl) >= 1
    if cp == 0 and la == 2:
        # Nothing waits on a scoreboard right behind a load that set it: a store and another unit's load on
        # the same scoreboard are never in the same gap.
        for u in range(len(un)):
            for v in range(len(un)):
                assert (u & 1) != (v & 1) or (gl[u] + cfg["sd"]) % ni != gl[v], (u, v)
    for s in range(4) if not cfg["nocopy"] else ():
        b = ni * s
        for u, (P, so, D, do, st, rec) in enumerate(un):
            if cp == 2:
                # TIMING ONLY: one store's share of a draw unit. The table words' stand-ins are the lane's own
                # words of two other stage buffers (16 bytes at the lane's place, as a table read is).
                a, bq = L["DA"], L["DB"]
                g = b + gl[u]
                dm = cfg["dm"]          # pieces left out, to find a fault or to price a piece
                if not dm & 1:
                    put(g - 1, Ins(muladd(a, D, 1, L["ZR"]), wait=hw))
                    put(g - 1, Ins(muladd(bq, D, 1, L["ZR"]), wait=hw))
                ra, rb = (D, D) if dm & 1 else (a, bq)
                if not dm & 2:
                    put(g, Ins("LDS.128 R%d, %s" % (a, mem(ra, ((s + 1) & 3) * st + do)), wsb=SB_Q[0], wait=hw))
                    put(g, Ins("LDS.128 R%d, %s" % (bq, mem(rb, ((s + 3) & 3) * st + do)), wsb=SB_Q[0], wait=hw))
                for i in range(4) if not dm & 4 else ():
                    put(g + 2, Ins("IADD3 R%d, -R%d, 0x3f3f3f3f, R%d" % (a + i, bq + i, a + i),
                                   wait=() if dm & 2 else (SB_Q[0],)))
                for i in range(2) if not dm & 8 else ():
                    put(g + 3, Ins("PRMT R%d, R%d, 0x5140, R%d" % (bq + 2 * i, a + 2 * i, a + 2 * i + 1)))
                    put(g + 3, Ins("PRMT R%d, R%d, 0x7362, R%d" % (bq + 2 * i + 1, a + 2 * i, a + 2 * i + 1)))
                if not cfg["nost"]:
                    # Into the scratch buffers behind the transcripts (cfold_h.cuh, CP_SCR): the stage keeps
                    # the real operand bytes the tile's prologue copied.
                    scr = L["SCR0"] + (0 if st == L["STB"] else L["STB"] - L["STGA"])
                    put(g + 4, Ins("STS.128 %s, R%d" % (mem(D, scr + do), bq), rsb=hold))
            elif cp == 0 and la == 2:
                # One set of quads: the store sits sd gaps behind its load, in this k-step or the next, and
                # the load fetches the stage that k-step stores. The quad's next load waits for the store to
                # have read it (hold).
                gs = gl[u] + cfg["sd"]
                nxt = gs >= ni
                assert gs - ni < gl[u] - 1, "unit %d's store must come before its quad's next load" % u
                stage = 2 if fix else s + 2 + nxt
                put(b + gl[u], ldg(L["Q"][0][u], P, stage * rec + so, ur, wsb=SB_Q[u & 1], wait=hw,
                                   rsb=hold if s == 3 and not fix else None))
                if not cfg["nost"]:
                    put((b + gs) % tot, Ins("STS.128 %s, R%d" % (mem(D, ((s + nxt + 2) & 3) * st + do), L["Q"][0][u]),
                                            wait=(SB_Q[u & 1],), rsb=hold))
            elif cp == 0:
                # la 1: the set loaded in k-step s is stored in k-step s + 1, so k-step s stores the other one.
                ld_set, st_set = (s & 1, (s + 1) & 1) if la else (0, 0)
                sb_ld, sb_st = (SB_Q[ld_set], SB_Q[st_set]) if la else (SB_Q[u & 1], SB_Q[u & 1])
                stage = 2 if fix else (s + 3 if la else s + 2)
                put(b + gl[u], ldg(L["Q"][ld_set][u], P, stage * rec + so, ur, wsb=sb_ld,
                                   rsb=hold if s == 3 and not fix else None))
                if not cfg["nost"]:
                    put(b + gl[u] + cfg["sd"], Ins("STS.128 %s, R%d" % (mem(D, ((s + 2) & 3) * st + do),
                                                                       L["Q"][st_set][u]), wait=(sb_st,)))
            else:
                for _ in range(cfg["lpad"] if u == 0 else 0):
                    put(b + gl[u], Ins("@!PT LDS RZ, [RZ]", raw=(0x00000000fffff984, 0x800)))
                put(b + gl[u], Ins("LDGSTS.E%s.128 %s, [R%d.64%s]" % (
                    "" if cfg["ca"] else ".BYPASS", mem(D, ((s + lead) & 3) * st + do), P, "+0x%x" % so if so else ""),
                    rsb=hold, stall=cfg["lst"]))
        if cp == 2 and not cfg["dm"] & 16:
            # The offsets' stand-ins: two 16-byte global loads a k-step from a place that does not move.
            put(b + 5, ldg(L["SCR"], L["PA1"], 0, ur, wsb=SB_Q[1]))
            put(b + 21, ldg(L["SCR"], L["PB"], 0, ur, wsb=SB_Q[1]))
        if cp == 1:
            # The k-step's group, then the pointers on to the next k-step's record (behind the copies that
            # still read them).
            put(b + max(gl), Ins("LDGDEPBAR", wsb=0))
            for i, (P, rec) in enumerate(sorted({(x[0], x[5]) for x in un}) if not fix else ()):
                put(b + max(gl) + 2 + i, Ins(addi(P, P, rec), wait=hw))
    if cp == 0 and not fix and not cfg["nocopy"]:
        # The chunk's end: the pointers on by four records, behind k-step 3's loads.
        for i, (P, rec) in enumerate(sorted({(x[0], x[5]) for x in un})):
            put(tot - 6 + i, Ins(addi(P, P, 4 * rec), wait=hw))
    # The readout's words of this chunk: set in k-step 1, after the last wrapped use.
    CH, TRW, ROT, LATER0 = L["CH"], L["TRW"], L["ROT"], L["LATER0"]
    ep = [Ins(shr(2, CH, 4)), Ins(addi(3, TRW, 4)), Ins("IADD3 R2, -R2, 0x0, R%d" % LATER0), Ins(andi(3, 3, 0x3c)),
          Ins(muladd(2, 2, 0xd)), Ins("LOP3.LUT R%d, R%d, 0x3f, R3, 0xba, !PT" % (TRW, TRW), wait=hw),
          Ins(andi(ROT, 2, 0x1f))]
    for i, ins in enumerate(ep):
        put(ni + i, ins)
    # The readout: tile t's chain of 16, its rotate and its XOR into the word, one a gap.
    rsh = cfg["rsh1"] if grp else cfg["rsh0"]
    b3 = 3 * ni
    for t in range(mt) if not cfg["noread"] else ():
        x = L["X"][t % 3]
        g0 = b3 + 8 * t + 10 + rsh
        steps = [Ins("LOP3.LUT R%d, R%d, R%d, RZ, 0x3c, !PT" % (x, acc(t, 0), acc(t, 0) + 1)),
                 Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (x, x, acc(t, 0) + 2, acc(t, 0) + 3))]
        for nb in range(1, 8):
            steps.append(Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (x, x, acc(t, nb), acc(t, nb) + 1)))
            steps.append(Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (x, x, acc(t, nb) + 2, acc(t, nb) + 3)))
        steps.append(Ins("SHF.L.W.U32.HI R%d, R%d, R%d, R%d" % (x, x, ROT, x)))
        imm = 128 * ((grp * mt + t) >> 1)
        steps.append(Ins("ATOMS.XOR RZ, %s, R%d" % (mem(TRW, imm), x), rsb=hold))
        if cfg["hold"]:
            # The atomic XOR keeps reading its value register until every lane's word is done (four lanes share
            # a word). The next tile that takes the same register starts its chain by overwriting it: it waits.
            steps[0].wait = (SB_HOLD,)
        for j, ins in enumerate(steps):
            g = g0 + j
            put(g % tot, ins, wrapped=g >= tot)
            if j < 16:
                # A sum is read 2 IMMA or more behind the IMMA that made it and before the next one that adds to it.
                nb = j // 2
                assert b3 + 8 * t + nb + 2 <= g <= tot + 8 * t + nb - 1, (t, j, g)
    assert 8 * (mt - 1) + 10 + rsh + 17 - ni < ni - 8, "the readout must end in k-step 0"
    put(tot - 6, Ins(addi(CH, CH, 1)))
    put(tot - 4, Ins("ISETP.GE.U32.AND P%d, PT, R%d, %s, PT" % (PRED, CH, CHUNKS)))
    return G


def emit(G, grp, cfg, L):
    """A group's chunk loop and its epilogue."""
    mt = cfg["mt"]
    ni, tot = 8 * mt, 32 * mt
    out = []
    for g in range(tot):
        s, i = divmod(g, ni)
        mb, nb = divmod(i, 8)
        wait = []
        if nb == 0:
            wait.append(SB_AF)
        if mb == 0 and nb in (0, 4):
            wait.append(SB_B)
        items = [ins for ins, _w in G[g]]
        n, nsingle = len(items), 0
        if cfg["sf"]:
            # sf: a gap's global loads and one-register steps go first, right behind the IMMA (only a step that
            # reads two or more registers has to keep its distance), and the IMMA's stall shrinks by their count.
            def single(ins):
                r, _w, kind = rw(ins.text)
                return (kind == "load" and not ins.text.startswith("LDSM")) or (kind == "alu" and len(r) <= 1)
            items = [x for x in items if single(x)] + [x for x in items if not single(x)]
            nsingle = sum(1 for x in items if single(x))
        ist_g = cfg["ist"]
        if cfg["sf"]:
            ist_g = max(1, ist_g - nsingle) if nsingle < n else 1
        out.append(Ins("IMMA.16832.S8.S8 R%d, R%d.ROW, R%d.COL, R%d" % (acc(mb, nb), AF[(mb + (mt & 1) * s) & 1],
                                                                    BF + 2 * nb, acc(mb, nb)),
                       stall=cfg["pad"] if not items else ist_g, yld=bool(cfg["iy"]), wait=wait,
                       label=".L%d" % grp if g == 0 else None))
        for k, ins in enumerate(items):
            ins = Ins(ins.text, ins.stall, bool(cfg["py"]), ins.wsb, ins.rsb, ins.wait, raw=ins.raw)
            if k == len(items) - 1:
                ins.yld = bool(cfg["ly"])
            out.append(ins)
        if i == ni - 1:
            if cfg["cp"] == 1 and not cfg["nocopy"] and not cfg["nodep"]:
                # Every group but the last lead - 2 has landed before any warp goes on.
                out.append(Ins("DEPBAR.LE SB0, 0x%x" % (cfg["lead"] - 2), stall=4, yld=False))
            if not cfg["nobar"]:
                out.append(Ins("BAR.SYNC.DEFER_BLOCKING 0x0", stall=cfg["bstall"]))
    out.append(Ins("@!P%d BRA .L%d" % (PRED, grp), stall=5))
    # After the last chunk: the wrapped part of the readout once more.
    ep = []
    for g in range(tot):
        for ins, wrapped in G[g]:
            if wrapped:
                ep.append(Ins(ins.text, cfg["pst"], ins.yld, ins.wsb, ins.rsb, ins.wait, raw=ins.raw))
    return out, ep


def build(cfg):
    assert cfg["lead"] in (2, 3) and cfg["mt"] in (5, 6)
    asm89.LDGSTS_UR[0] = cfg["ur"]
    L = layout(cfg)
    pro = prologue(cfg, L)
    bad = check(pro, "prologue", fix=True)
    code = list(pro)
    plain = 0
    for grp in (0, 1):
        G = chunk(grp, cfg, L)
        loop, ep = emit(G, grp, cfg, L)
        body = loop[:-1]
        plain += len(body) - 32 * cfg["mt"]
        bad += check(pro + body + body, "group %d loop" % grp)
        check(body + ep, "group %d epilogue" % grp, fix=True)
        if grp == 1:
            code.append(Ins("NOP", label=".G1"))
        code += loop + ep
        if grp == 0:
            code.append(Ins("BRA .END", stall=5))
    code.append(Ins("NOP", label=".END"))
    # Before the return: the copies drained, and one step that waits for every load.
    if cfg["cp"] == 1 and not cfg["nocopy"]:
        code.append(Ins("DEPBAR.LE SB0, 0x0", stall=4, yld=False))
    code.append(Ins(addi(L["CH"], L["CH"], 0), wait=(SB_AF, SB_B, SB_HOLD) + tuple(SB_Q)))
    code.append(Ins("NOP", stall=8))
    return code, bad, plain


def main():
    cubin, sass, nvd, out = sys.argv[1:5]
    cfg = dict(CFG)
    for kv in sys.argv[5:]:
        k, v = kv.split("=")
        assert k in cfg, k
        cfg[k] = int(v, 0)
    code, bad, plain = build(cfg)
    if bad:
        print("%d latency problems: not written" % bad)
        sys.exit(1)
    text = asm89.read_text(sass)
    regs, preds = asm89.clobbers(text)
    used = set()
    for ins in code:
        if "BRA" in ins.text or ins.text.startswith("ULDC"):
            continue
        _r, w, _k = rw(ins.text)
        used |= {x for x in w if not isinstance(x, str)}
    assert used <= regs and PRED in preds, "writes registers the kernel keeps: %s" % sorted(used - regs)
    n, room = asm89.patch(cubin, text, out, code, nvd)
    tot = 32 * cfg["mt"]
    print("%s: %d instructions of %d; a chunk is %d IMMA and %.1f others a warp (%.2f an IMMA)" % (
        os.path.basename(out), n, room, tot, plain / 2.0, plain / 2.0 / tot))


if __name__ == "__main__":
    main()
