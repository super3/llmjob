"""rfold.py: the rotation fold's chunk loop written by hand for sm_89. It replaces the loop function ptxas compiled
from gfold_h.cuh (built with -DGP_STAGE=24576) and computes the same transcripts from gprobe's `rot` offsets:
gprobe's CPU check judges it.

  python rfold.py <base.cubin> <base.sass> <nvdisasm> <out.cubin> [key=value ...]

Why this form. Measured on rig01's RTX 4090 (probes/README.md, "The price of a step, and the rotation fold"):
under the card's cap the clock falls with the count of shared-memory steps a k-step (LDSM, LDS, STS: about one
unit each, 16 MHz) and hardly with ALU steps (a tenth of a unit). The gather fold has 20 LDSM and 80 ALU steps a
warp a k-step because every warp makes all of its fragments itself. But the four warps of a row band fold the
same rows: an A fragment (an m16 tile's four words of a lane) is the same in all four.

What the loop does. Every warp takes the four k-steps of a chunk in an order of its own (any order inside a chunk
gives the same sums), so that at the chunk's first turn the four warps of a band fold four different k-steps.
At that turn a warp makes its k-step's six A fragments with gathers, as the gather fold does, uses them, and
stores each with one 16-byte store. At its other three turns it folds k-steps another warp of its band made at
the first turn, and an A fragment is one 16-byte load from the stage. B's fragments are gathered by every warp
at every turn, as in the gather fold. A warp a k-step: 11 LDSM, 4.5 LDS, 1.5 STS and 44 ALU steps for the draw
(the gather fold: 20 LDSM and 80 ALU steps).

What keeps a load behind its store (sync=4, band=1, the defaults). The four warps of a row band meet at a
barrier of their own at every k-step's end (bar.sync 1, 128 for band 0 and 2, 128 for band 1; the bands do not
wait for each other). Tiles 1 to 5 are stored in the first turn and loaded in turns 1 to 3. Tile 0, which a
turn loads in the turn before, is stored in the chunk's last turn (the first chunk's in the prologue, ahead of
a barrier). So every load has a barrier between it and the store that wrote its fragment, and another between
it and the store that overwrites it. Without barriers the loop is faster and wrong: nothing then orders a load
after its store (12 to 54 of 64 tiles failed the check).

Two things the barriers made necessary, each measured. One barrier for all eight warps costs 3 points of pipe
more than one a band. And warps a barrier has put in step all gather in the same gaps: with bank collisions
that is more than shared memory serves, so the first five of B's eight LDSM go out early and two gaps apart
(bsp=1: 96.0% of the pipe against 93.8%).

Registers: as gfold.py's, but R0 is the lane's place in the warp's own part of the stage and the transcript
word is worked in the scratch register. P2 is the warp's row band.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402
import fold  # noqa: E402
from fold import addi, andi, muladd, shr  # noqa: E402
import gfold  # noqa: E402
from gfold import ACC, AF, BF, SB_OFF, SB_RD, XR, gather_ins, ldg  # noqa: E402

CFG = dict(mt=6, pad=8, ist=5, py=0, iy=1, ly=1, ru=0, lead=3, sf=0, hold=1, pst=2, rmar=5, roff=11, pol=2, ur=4,
           # sync=4: a barrier at every k-step's end, and tile 0's store in the turn before its first use: every
           # load of a stored fragment then has a barrier between it and the store before and the store after.
           # sync=0: only a barrier every barn chunks (a power of two; 0 for none), tile 0's store in gap st0:
           # correct only while the warps of a band stay within 39 IMMA of each other (they do not).
           sync=4, barn=16, st0=1,
           # band=1: the k-step barriers are one for each row band (128 threads each): only the four warps
           # that share fragments wait for each other.
           band=1, bsp=1,
           # lead: the gaps between a gather's LDSM and its IADD3; plead: between its IADD3 and its PRMT. Three and
           # one hold 0.5 points more of the pipe than two and two; py=0 (the plain steps' yield flag off) 0.2.
           plead=1,
           # Timing only (wrong results on purpose): nost, no stores; nold, the gathers' LDSM are moves.
           nost=0, nold=0)
SB_ST = 3                   # the stage loads' scoreboard; the gathers' LDSM go round 0-2
# bsp: the gaps of a turn's eight B gathers' LDSM, in the turns 0 to 2 and in the chunk's last turn (where
# the next chunk's first tile is gathered in gaps 37 and 38). Their PRMT stay in gaps 43 to 50.
BLD = ([28, 30, 34, 37, 39, 44, 45, 46], [28, 30, 34, 42, 43, 44, 45, 46])
SW = 0                      # R0: where the lane's fragments of the warp's first turn go


def layout(mt):
    R = gfold.layout(mt, 0)
    R["w"] = R["tmp"]
    R["stg"] = R["tr"] + 256 * 64 + 256            # GP_STG
    return R


def prologue(cfg, R, tail):
    """As gfold's, with the stage address. `tail`: the quad and scoreboard the loop gave the chunk's last three
    gathers, whose ends lie in the next chunk's first gaps."""
    mt = cfg["mt"]
    q = R["quads"]
    S = q[0]
    c = [Ins("S2R R0, SR_TID.X", wsb=SB_RD),
         Ins("ULDC.64 UR%d, c[0x0][0x118]" % gfold.UR, raw=(0x0000460000007ab9 | (gfold.UR << 16), 0xa00))]
    for r in list(range(AF[0], ACC + 32 * mt, 2)) + [XR[0]]:
        c.append(Ins("CS2R R%d, SRZ" % r))
    c.append(Ins(andi(S, 0, 0x1f), wait=(SB_RD,)))                            # lane
    c.append(Ins(shr(S + 1, 0, 5), wait=(SB_RD,)))                            # warp
    c.append(Ins(muladd(R["trw"], 0, 0x40), wait=(SB_RD,)))                   # the thread's transcript ...
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, c[0x0][0x178]" % R["cnt"]))       # chunks
    c.append(Ins(muladd(S + 2, R["cnt"], 0x800)))                             # a warp's offsets: 4 chunks + 1
    c.append(Ins(addi(S + 2, S + 2, 0x200)))                                  # k-steps of 512 bytes
    c.append(Ins("IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x170]" % (R["poff"], S + 1, S + 2)))
    c.append(Ins(shr(S + 3, S + 1, 2)))                                       # row band
    if cfg["band"]:
        c.append(Ins("ISETP.NE.AND P2, PT, R%d, 0x0, PT" % (S + 3)))
    c.append(Ins("IADD3 R%d, R%d, 0x0, R%d" % (q[1], S + 1, S + 3)))
    c.append(Ins(andi(q[1], q[1], 3)))                                        # the k-step of the first turn
    c.append(Ins(muladd(q[1], q[1], 512 * mt)))
    c.append(Ins(muladd(S + 3, S + 3, 2048 * mt, q[1])))
    c.append(Ins(muladd(S, S, 0x10)))                                         # 16 lane
    c.append(Ins(addi(R["trw"], R["trw"], R["tr"] + 60)))                     # ... at its word 15
    c.append(Ins("NOP", stall=4))
    c.append(Ins("IADD3 R%d, R%d, 0x0, R%d" % (R["poff"], R["poff"], S)))     # the host keeps this from carrying
    c.append(Ins("IADD3 R%d, R%d, 0x%x, R%d" % (SW, S + 3, R["stg"], S)))
    cur = R["oq"][0]
    c.append(ldg(cur, R["poff"], 0, wsb=SB_OFF))
    c.append(Ins("NOP", stall=8))
    wl = (SB_OFF,)
    jobs = [(cur + h, 0, AF[0] + 2 * h, AF[0] + 2 * h + 1) for h in range(2)]
    jobs += [(cur + 2 + h, 2048 * p, BF + 4 * p + h, BF + 4 * p + 2 + h) for p in range(4) for h in range(2)]
    tq = [t[0] for t in tail]
    free = [x for x in q if x not in tq]
    for part, quads in ((jobs[0:5], q), (jobs[5:7], free)):
        part = [gather_ins(quads[n], a, o, d1, d2, n % 4, cfg, wait_ld=wl) for n, (a, o, d1, d2) in enumerate(part)]
        c += [g[0] for g in part]
        for g in part:
            c += g[1]
        for g in part:
            c += g[2]
    if cfg["sync"] == 4 and not cfg["nost"]:
        # The first chunk's tile 0 goes to the stage here, and the warps meet before any of them loads.
        c.append(Ins("STS.128 [R%d], R%d" % (SW, AF[0])))
        c.append(Ins("NOP", stall=8))
        c += barrier(cfg)
    last = [gather_ins(tail[n][0], a, o, d1, d2, tail[n][1], cfg, wait_ld=wl) for n, (a, o, d1, d2) in enumerate(jobs[7:])]
    c += [g[0] for g in last]
    for n, g in enumerate(last):
        if tail[n][2]:                  # its IADD3 lie in the chunk's last gaps: the loop does not redo them
            c += g[1]
    return c


def chunk(cfg, R):
    """Gaps 0 .. 32 mt - 1 of a chunk, each a list of (Ins, wrapped), the IMMA that wait, and the quad and
    scoreboard of the chunk's last three gathers."""
    mt = cfg["mt"]
    assert mt % 2 == 0
    ni, tot = 8 * mt, 32 * mt
    G = [[] for _ in range(tot)]
    iwait = {}

    def put(g, ins, wrapped=None):
        G[g % tot].append((ins, g >= tot if wrapped is None else wrapped))

    plans = []          # (LDSM gap, IADD3 gap, PRMT gap, address register, offset, d1, d2, rsb)
    lead, plead = cfg["lead"], cfg["plead"]
    hold = SB_RD if cfg["hold"] else None
    hw = (SB_RD,) if cfg["hold"] else ()
    for s in range(4):
        b = ni * s
        cur, nxt = R["oq"][s & 1], R["oq"][(s + 1) & 1]
        # The next turn's record. The load overwrites the set the turn before last read: it waits for that
        # turn's last readers (hold). In the chunk's last turn it is the last user of the pointer.
        put(b + 2, ldg(nxt, R["poff"], 0x200 * (s + 1), wsb=SB_OFF, rsb=hold if s == 3 else None, wait=hw))
        # The record's first readers issue 8 cycles or more behind three IMMA that wait for the load.
        first = b + (min(BLD[0]) if cfg["bsp"] else ni - 11) - 2
        for g in (first - 2, first - 1, first):
            iwait.setdefault(g, []).append(SB_OFF)
        for m in range(1, mt):                         # tile m's fragment, during tile m - 1
            a = AF[m & 1]
            if s == 0:                                 # the first turn: made here
                for h in range(2):
                    g0 = b + 8 * (m - 1) + h
                    plans.append((g0, g0 + lead, g0 + lead + plead, cur + h, 2048 * m, a + 2 * h, a + 2 * h + 1,
                                  hold if m == mt - 1 else None))
            else:                                      # a later turn: another warp's, from the stage
                put(b + 8 * (m - 1) + 1, Ins("LDS.128 R%d, [R%d+0x%x]" % (a, cur, 512 * m), wsb=SB_ST,
                                             rsb=hold if m == mt - 1 else None), False)
                for g in (b + 8 * m, b + 8 * m + 1):
                    iwait.setdefault(g % tot, []).append(SB_ST)
        a = AF[0]                                      # the next turn's first tile
        if s == 3:                                     # the next chunk's first turn: made here
            for h in range(2):
                g0 = b + ni - 11 + h
                plans.append((g0, g0 + lead, g0 + lead + plead, nxt + h, 0, a + 2 * h, a + 2 * h + 1, None))
        else:
            put(b + ni - 7, Ins("LDS.128 R%d, [R%d]" % (a, nxt), wsb=SB_ST), False)
            for g in (b + ni, b + ni + 1):
                iwait.setdefault(g % tot, []).append(SB_ST)
        for p in range(4):                             # the next turn's B: pair p's PRMT at r 8mt-5+2p (+1),
            for h in range(2):                         # two IMMA behind the last IMMA that reads the pair
                n = 2 * p + h
                g0 = b + ni - 9 + n
                if cfg["bsp"]:
                    # bsp: the LDSM of the first B gathers go out early and apart (the quads allow five
                    # gathers under way), so that the warps a barrier has put in step do not all gather in
                    # the same eight gaps: with bank collisions that is more than shared memory serves.
                    g_ld = b + BLD[s == 3][n]
                    plans.append((g_ld, g_ld + lead, g0 + lead + plead, nxt + 2 + h, 2048 * p, BF + 4 * p + h,
                                  BF + 4 * p + 2 + h, None))
                else:
                    plans.append((g0, g0 + lead, g0 + lead + plead, nxt + 2 + h, 2048 * p, BF + 4 * p + h,
                                  BF + 4 * p + 2 + h, None))
    # The first turn's stores: tile m's four words, a gap behind its last PRMT (tile 0's, made at the end of
    # the chunk before, in gap st0). The set is written again 9 gaps later or more.
    for m in range(mt) if not cfg["nost"] else ():
        g = 8 * (m - 1) + lead + plead + 2 if m else (tot - 9 + lead + plead if cfg["sync"] == 4 else cfg["st0"])
        put(g, Ins("STS.128 [R%d%s], R%d" % (SW, "+0x%x" % (512 * m) if m else "", AF[m & 1])), False)
    # A gather takes the lowest quad that is free (a quad is free from the gap behind its last PRMT); the
    # chunk before left its last gathers under way, so the chunk is walked twice and the second walk kept.
    # The scoreboards go round 0-2 in the order the LDSM are issued.
    order = sorted(range(len(plans)), key=lambda k: plans[k][0])
    assert len({plans[k][0] for k in order}) == len(order), "two LDSM in one gap"
    nq = len(R["quads"])
    free_at = [-tot] * nq
    for walk in range(2):
        took = []
        for k in order:
            g_ld, g_prm = plans[k][0] + walk * tot, plans[k][2] + walk * tot
            qi = next((i for i in range(nq) if free_at[i] <= g_ld), None)
            assert qi is not None, "more than %d gathers under way in gap %d" % (nq, plans[k][0])
            free_at[qi] = g_prm + 1
            took.append(qi)
    use, assign = {}, []
    for n, k in enumerate(order):
        g_ld, g_add, g_prm, areg, aoff, d1, d2, rsb = plans[k]
        assert g_ld + lead <= g_add < g_prm
        q, sb = R["quads"][took[n]], n % 3
        for (a0, a1) in use.get(q, []):
            assert not any(a0 <= g_prm + w and g_ld + w <= a1 for w in (-tot, 0, tot)), (q, g_ld)
        use.setdefault(q, []).append((g_ld, g_prm))
        ld, adds, prm = gather_ins(q, areg, aoff, d1, d2, sb, cfg, rsb=rsb)
        put(g_ld, ld, wrapped=False)
        for x in adds:
            put(g_add, x, wrapped=False)
        for x in prm:
            put(g_prm, x, wrapped=False)
        assign.append((q, sb, g_add < tot))
    assert [plans[k][0] for k in order[-3:]] == [tot - 4, tot - 3, tot - 2]
    # The readout, as gfold's: two chains of XOR3, one step of each a gap, from r roff of the last turn.
    b3 = 3 * ni
    busy = set(range(b3 + ni - 9, tot + 3))
    end = b3
    for ch in range(2):
        g = b3 + cfg["roff"]
        first = True
        for t in range(ch, mt, 2):
            for j in range(16):
                nb = j // 2
                made, again = b3 + 8 * t + nb, tot + 8 * t + nb          # the sum's last IMMA, and its next
                g = max(g, made + cfg["rmar"])
                while g in busy:
                    g += 1
                assert g < again, "tile %d's sum %d is read in gap %d, behind its next IMMA" % (t, nb, g)
                r = ACC + 4 * (8 * t + nb) + 2 * (j & 1)
                if first:
                    ins = Ins("LOP3.LUT R%d, R%d, R%d, RZ, 0x3c, !PT" % (R["xr"][ch], r, r + 1))
                    first = False
                else:
                    ins = Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (R["xr"][ch], R["xr"][ch], r, r + 1))
                put(g, ins)
                g += 1
        end = max(end, g)
    assert end > tot, "the readout's tail is taken to lie in the next chunk"
    trw, w, xr = R["trw"], R["w"], R["xr"]
    put(end - 4, Ins("LDS R%d, [R%d]" % (w, trw), wsb=SB_RD), wrapped=True)
    put(end + 0, Ins("SHF.L.W.U32.HI R%d, R%d, 0xd, R%d" % (w, w, w), wait=(SB_RD,)), wrapped=True)
    put(end + 1, Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (w, w, xr[0], xr[1])), wrapped=True)
    put(end + 2, Ins("STS [R%d], R%d" % (trw, w), rsb=hold), wrapped=True)
    # The next word of the sixteen: the pointer is 64-aligned, so its low six bits are the word. The scratch
    # register is the word the store is still reading: its next writer waits for the store.
    put(end + 4, Ins(addi(w, trw, 4), wait=hw), wrapped=True)
    put(end + 5, Ins(andi(w, w, 0x3c)), wrapped=True)
    put(end + 6, Ins("LOP3.LUT R%d, R%d, 0x3f, R%d, 0xba, !PT" % (trw, trw, w), wait=hw), wrapped=True)
    assert end + 6 - tot < ni - 16
    # The chunk's count and the offsets pointer, in quiet gaps of the last turn; and the barrier's test.
    put(b3 + 6, Ins(addi(R["cnt"], R["cnt"], -1)), wrapped=False)
    put(b3 + 7, Ins("ISETP.NE.AND P1, PT, R%d, 0x0, PT" % R["cnt"]), wrapped=False)
    put(b3 + 14, Ins(addi(R["poff"], R["poff"], 0x800), wait=hw), wrapped=False)
    if cfg["barn"] and cfg["sync"] != 4:
        assert cfg["barn"] & (cfg["barn"] - 1) == 0
        put(b3 + 9, Ins(andi(w, R["cnt"], cfg["barn"] - 1)), wrapped=False)
        put(b3 + 10, Ins("ISETP.NE.AND P2, PT, R%d, 0x0, PT" % w), wrapped=False)
    return G, iwait, assign[-3:]


def barrier(cfg):
    """The k-step barrier: all eight warps, or (band) the four of the warp's row band; P2 is set for band 1."""
    if not cfg["band"]:
        return [Ins("BAR.SYNC.DEFER_BLOCKING 0x0", stall=1)]
    return [Ins("@!P2 BAR.SYNC.DEFER_BLOCKING 0x1, 0x80", stall=1), Ins("@P2 BAR.SYNC.DEFER_BLOCKING 0x2, 0x80", stall=1)]


def build(cfg):
    gfold.UR = cfg["ur"]
    R = layout(cfg["mt"])
    G, iwait, tail = chunk(cfg, R)
    pro = prologue(cfg, R, tail)
    bad = gfold.check(pro, "prologue", fix=True)
    loop, ep, most = gfold.emit(G, iwait, cfg)
    if cfg["sync"] == 4:
        ni, out, seen = 8 * cfg["mt"], [], 0
        for x in loop:
            if x.text.startswith("IMMA"):
                if seen and seen % ni == 0:
                    out += barrier(cfg)
                seen += 1
            if "BRA" in x.text:
                out += barrier(cfg)
            out.append(x)
        loop = out
    elif cfg["barn"]:
        loop.insert(len(loop) - 1, Ins("@!P2 BAR.SYNC.DEFER_BLOCKING 0x0", stall=1))
    body = [x for x in loop if "BRA" not in x.text and "BAR" not in x.text]
    bad += gfold.check(pro + body + body + body, "loop")
    bad += gfold.check(body + ep, "epilogue", fix=True)
    # Every load has landed before the function returns.
    ep.append(Ins(addi(R["tmp"], R["tmp"], 0), wait=(0, 1, 2, 3, SB_OFF, SB_RD)))
    ep.append(Ins("NOP", stall=8))
    return pro + loop + ep, bad, most, G


def main():
    cubin, sass, nvd, out = sys.argv[1:5]
    cfg = dict(CFG)
    for kv in sys.argv[5:]:
        k, v = kv.split("=")
        assert k in cfg, k
        cfg[k] = int(v, 0)
    code, bad, most, G = build(cfg)
    plain = sum(len(x) for x in G)
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
    assert used <= regs and 2 in preds, "writes registers the kernel keeps: %s" % sorted(used - regs)
    # The loop's own predicate is P1 where the compiled function writes P1; a build whose function writes
    # another one (with the hash in the kernel it is P0) gets that one, since the kernel may keep the rest.
    if 1 not in preds:
        lp = min(preds - {2})
        for ins in code:
            if ins.text.startswith("ISETP.NE.AND P1, "):
                ins.text = "ISETP.NE.AND P%d, " % lp + ins.text[len("ISETP.NE.AND P1, "):]
            elif ins.text.startswith("@P1 BRA "):
                ins.text = "@P%d BRA " % lp + ins.text[len("@P1 BRA "):]
    n, room = asm89.patch(cubin, text, out, code, nvd)
    if cfg["band"]:
        asm89.set_barriers(out, 3)
    tot = 32 * cfg["mt"]
    kinds = {}
    for g in G:
        for ins, _w in g:
            op = ins.text.split(" ")[0].split(".")[0]
            kinds[op] = kinds.get(op, 0) + 1
    print("%s: %d instructions of %d; a chunk is %d IMMA and %d others (%.2f an IMMA), %d at most in a gap; %s" % (
        os.path.basename(out), n, room, tot, plain, plain / float(tot), most,
        ", ".join("%s %d" % kv for kv in sorted(kinds.items(), key=lambda kv: -kv[1]))))


if __name__ == "__main__":
    main()
