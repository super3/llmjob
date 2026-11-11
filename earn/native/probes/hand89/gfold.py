"""gfold.py: the gather fold's chunk loop written by hand for sm_89, instruction by instruction, with its own
registers, stall counts and scoreboards. It replaces the loop function ptxas compiled from gfold_h.cuh
(asm89.patch) and computes the same transcripts: gprobe's CPU check judges it.

  python gfold.py <base.cubin> <base.sass> <nvdisasm> <out.cubin> [key=value ...]

What the loop does (gfold_h.cuh has the layouts): a k-step is 8 mt IMMA into 32 mt sums a thread. Every fragment
is made by the warp itself with a gather: one LDSM.x4.trans on a static table at the lane's offset, two IADD3
(fill + plus - minus) and two PRMT. A's fragments of an m16 tile are built while the tile before runs, into the
other of two register sets; B's of the next k-step are built in the k-step's last gaps, each pair two IMMA
behind the last IMMA that reads the pair it replaces. Nothing is stored, nothing is shared, no warp waits for
another. The lane's XOR of its own sums goes into its own transcript word once a chunk.

Where things sit (gap g: behind IMMA g of the chunk; 8 mt gaps a k-step, r = the gap in its k-step):
  - the next k-step's four offsets: one LDG.128 at r 2. The IMMA at r 33 to 35 wait on its scoreboard, so the
    gathers that read them from r 37 never run ahead of the load;
  - tile m's two gathers at r 8(m-1) and +1 (LDSM), +2 (IADD3), +4 (PRMT); the next k-step's first tile at
    r 8mt-11, -10; B's eight from r 8mt-9, one a gap. At most five plain steps in a gap;
  - the readout: two chains of XOR3 (even and odd tiles), one step of each a gap, from r 11 of the chunk's last
    k-step, a sum read five IMMA or more behind the IMMA that made it. They pause while B's gathers run and end
    in the next chunk's first k-step, before each sum's next IMMA. The pass of the first chunk XORs zeros into
    word 15; after the last chunk the tail runs once more.

Rules this loop keeps, each from a measurement on rig01's RTX 4090 (probes/README.md):
  - an IMMA with plain steps behind it gets stall 5 (key ist): a step that reads two or more registers costs
    the pipe when it issues 1 to 4 cycles behind its warp's IMMA and nothing from 5;
  - an instruction that reads a register a load writes waits on that load's scoreboard ITSELF, or issues a
    gap or more behind one that did. A wait on the instruction just ahead is not enough: the old gather
    stream shifted a loaded quad in place with only the first shift waiting, and the other three read the
    registers before the load had written them (the "illegal memory access" of 11 November);
  - a load, store or LDSM keeps reading its address and value registers while the memory unit holds it: what
    overwrites one of them later waits on a read scoreboard the reader set (key hold).

Registers (R1, R253, R254 are the kernel's): R0 the transcript word, R2-R3 the readout's chains, A's two
fragment sets R4-R11, B's eight R12-R27, the sums from R28, then five quads of gather temporaries, the two
sets of offsets, the offsets pointer (two), the chunk count, the transcript pointer and a scratch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402
import fold  # noqa: E402
from fold import addi, andi, muladd, shr  # noqa: E402

CFG = dict(mt=6, pad=8, ist=5, py=1, iy=1, ly=1, ru=0, lead=2, sf=0, hold=1, pst=2, rmar=5, roff=11,
           # pol=0: a gap with plain steps is padded to `pad` cycles behind its last step, so every gap is
           # as long as an empty one (the default, 2, adds nothing: the gap is ist + its steps).
           pol=2,
           # Timing only (each computes wrong results on purpose): pieces left out.
           noread=0, nold=0, noalu=0, nogather=0, dbg=0, pw=0,
           # Timing only, no draw at all: what a fold costs whose operands are ready in shared memory. direct:
           # a fragment is one LDSM from the table as it stands (ten a k-step). cu: copy units a k-step (a
           # 16-byte load from global memory and a 16-byte store to shared, as a stage's refill). xalu: extra
           # three-register steps a k-step.
           direct=0, cu=0, xalu=0,
           # dg=1 is the DIRECT fold (gfold_h.cuh built with GP_DIRECT): no draw, A' and B' ready in global memory
           # in the fragments' own layout, a fragment one 16-byte load a lane. ur: the uniform register pair
           # the loads' address base goes in (the one the compiled loop function uses: 4 gather, 14 direct).
           dg=0, ur=4,
           # pf: the direct fold touches its operands ahead of time. 1: every warp loads each of its ten
           # fragments of the next k-step into scratch registers (the loads that count then find the lines
           # in the SM's cache: 68% of the pipe without, 99% with). pfk: how many k-steps ahead (two is too
           # far: 73%).
           pf=0, pfk=1)
W, XR, AF, BF, ACC = 0, [2, 3], [4, 8], 12, 28
# Scoreboards: the gathers' LDSM go round 0-3; the offsets load; the transcript word's load and the holds.
SB_OFF, SB_RD = 4, 5
FILL4 = 0x3f3f3f3f
UR = 4                      # the uniform register pair the compiled function loads the address base into
LAT = dict(alu=5, imma=28, setp=13)
SETTLE = 8                  # cycles behind a waiter from which a loaded register may be read without waiting


def layout(mt, dg=0):
    """Register numbers and shared-memory places for mt m16 tiles a warp."""
    q0 = ACC + 32 * mt
    if dg:
        # The direct fold's kernel keeps R0, R3, R240-R242, R248, R249 and R252 through the call.
        assert mt == 6
        return dict(quads=[q0 + 12], oq=[q0 + 12, q0 + 12], w=q0, xr=[q0 + 1, q0 + 2], pa=q0 + 4, pb=q0 + 6, cnt=q0 + 8,
                    trw=q0 + 9, tmp=q0 + 10, s=q0 + 12, tid=q0 + 16, poff=q0 + 4, tr=0, ptr=256 * 64 + 192)
    r = dict(quads=[q0 + 4 * i for i in range(5)], oq=[q0 + 20, q0 + 24], poff=q0 + 28, cnt=q0 + 30, trw=q0 + 31,
             tmp=q0 + 32, tr=2048 * (2 * mt + 16), w=W, xr=XR)
    assert r["tmp"] <= 252
    return r


def ldg(dst, base, off, size=128, **k):
    """A 16-byte global load (8 with size=64): asm89's LDG with the uniform register the kernel was compiled
    with."""
    text = "LDG.E.%d R%d, [R%d.64%s]" % (size, dst, base, "+0x%x" % off if off else "")
    lo, hi = asm89.assemble(text)
    lo = (lo & ~(0xFF << 32)) | (UR << 32)
    return Ins(text, raw=(lo, hi), **k)


def gather_ins(q, areg, aoff, d1, d2, sb, cfg, wait_ld=(), rsb=None):
    """(LDSM, the two IADD3, the two PRMT) of one gather into quad q, fragment words d1 and d2."""
    if cfg["nold"]:
        ld = Ins(fold.mov(q, areg), wait=wait_ld)
        w = ()
    else:
        ld = Ins("LDSM.16.MT88.4 R%d, [R%d%s]" % (q, areg, "+0x%x" % aoff if aoff else ""), wsb=sb, rsb=rsb,
                 wait=wait_ld)
        w = (sb,)
    adds = [Ins("IADD3 R%d, -R%d, 0x%x, R%d" % (q, q + 2, FILL4, q), wait=w),
            Ins("IADD3 R%d, -R%d, 0x%x, R%d" % (q + 1, q + 3, FILL4, q + 1), wait=w)]
    prm = [Ins("PRMT R%d, R%d, 0x6420, R%d" % (d1, q, q + 1)), Ins("PRMT R%d, R%d, 0x7531, R%d" % (d2, q, q + 1))]
    return ld, adds, prm


def check(code, name, fix=False):
    """Walks straight-line code summing stalls. A register an ALU step, an IMMA or a compare wrote is read LAT
    cycles or more behind it. A register a load wrote is read by an instruction that waits on the load's
    scoreboard, or SETTLE cycles or more behind one that did. With fix, the stall of the instruction before a
    too-early reader is raised instead. Returns the count of problems."""
    last, t, bad = {}, 0, 0
    for i, ins in enumerate(code):
        if "BRA" in ins.text:
            t += ins.stall
            continue
        reads, writes, kind = fold.rw(ins.text)
        for sb in ins.wait:
            # A wait on a scoreboard covers every load that set it before: their registers are settled from here.
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


def prologue(cfg, R):
    mt = cfg["mt"]
    q = R["quads"]
    S = q[0]
    c = [Ins("S2R R0, SR_TID.X", wsb=SB_RD),
         Ins("ULDC.64 UR%d, c[0x0][0x118]" % UR, raw=(0x0000460000007ab9 | (UR << 16), 0xa00))]
    for r in list(range(AF[0], ACC + 32 * mt, 2)) + [XR[0]]:
        c.append(Ins("CS2R R%d, SRZ" % r))
    c.append(Ins(andi(S, 0, 0x1f), wait=(SB_RD,)))                            # lane
    c.append(Ins(shr(S + 1, 0, 5), wait=(SB_RD,)))                            # warp
    c.append(Ins(muladd(R["trw"], 0, 0x40), wait=(SB_RD,)))                   # the thread's transcript ...
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, c[0x0][0x178]" % R["cnt"]))       # chunks
    c.append(Ins(muladd(S + 2, R["cnt"], 0x800)))                             # a warp's offsets: 4 chunks + 1
    c.append(Ins(addi(S + 2, S + 2, 0x200)))                                  # k-steps of 512 bytes
    c.append(Ins("IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x170]" % (R["poff"], S + 1, S + 2)))
    c.append(Ins(muladd(S, S, 0x10)))                                         # 16 lane
    c.append(Ins(addi(R["trw"], R["trw"], R["tr"] + 60)))                     # ... at its word 15
    c.append(Ins("NOP", stall=4))
    c.append(Ins("IADD3 R%d, R%d, 0x0, R%d" % (R["poff"], R["poff"], S)))     # the host keeps this from carrying
    cur = R["oq"][0]
    c.append(ldg(cur, R["poff"], 0, wsb=SB_OFF))
    wl = (SB_OFF,)
    if cfg["pw"] == 1:
        # Four steps that each wait for the load and write its registers back as they are, then a pause: the
        # gathers behind read registers an ALU step wrote.
        for j in range(4):
            c.append(Ins(andi(cur + j, cur + j, 0xfffffff0), wait=(SB_OFF,)))
        c.append(Ins("NOP", stall=6))
        wl = ()
    elif cfg["pw"] == 2:
        c.append(Ins("NOP", stall=8))
    if cfg["direct"]:
        c.append(Ins("NOP", stall=8))
        c.append(Ins("LDSM.16.MT88.4 R%d, [R%d]" % (AF[0], cur), wsb=2, wait=(SB_OFF,)))
        for p in range(4):
            c.append(Ins("LDSM.16.MT88.4 R%d, [R%d%s]" % (BF + 4 * p, cur + 2, "+0x%x" % (2048 * p) if p else ""), wsb=3,
                         wait=(SB_OFF,)))
        # The copy unit's store writes back what this read: the tables stay as they are.
        c.append(Ins(fold.mov(q[4], cur + 2), wait=(SB_OFF,)))
        c.append(Ins("NOP", stall=6))
        c.append(Ins("LDS.128 R%d, [R%d]" % (q[0], q[4]), wsb=SB_RD))
        c.append(Ins("NOP", stall=8))
        for j in range(4):
            c.append(Ins(andi(q[0] + j, q[0] + j, 0xffffffff), wait=(SB_RD,)))
        for j in (8, 10, 12, 14):
            c.append(Ins("CS2R R%d, SRZ" % (q[0] + j)))
        return c
    # The first k-step's fragments: A's first tile and B's four pairs. The loop builds a k-step's fragments in
    # the k-step before and the last three gathers of B end in the next k-step's first gaps (pair 2's second
    # half and pair 3's first: their PRMT; pair 3's second: its IADD3 and PRMT). So the first seven are made
    # whole here and the last three are left where the loop expects them, in its own quads and scoreboard.
    jobs = [(cur + h, 0, AF[0] + 2 * h, AF[0] + 2 * h + 1) for h in range(2)]
    jobs += [(cur + 2 + h, 2048 * p, BF + 4 * p + h, BF + 4 * p + 2 + h) for p in range(4) for h in range(2)]
    for part in (jobs[0:5], jobs[5:7]):
        part = [gather_ins(q[n], a, o, d1, d2, n % 4, cfg, wait_ld=wl) for n, (a, o, d1, d2) in enumerate(part)]
        c += [g[0] for g in part]
        for g in part:
            c += g[1]
        for g in part:
            c += g[2]
    tail = [gather_ins(q[2 + n], a, o, d1, d2, 1 + n, cfg, wait_ld=wl) for n, (a, o, d1, d2) in enumerate(jobs[7:])]
    c += [g[0] for g in tail]
    c += tail[0][1] + tail[1][1]
    return c


def prologue_dg(cfg, R):
    """The direct fold's: the tile's two pointers from shared memory, the warp's and the lane's part added, the
    first k-step's five loads."""
    mt = cfg["mt"]
    S, tid, pa, pb, trw = R["s"], R["tid"], R["pa"], R["pb"], R["trw"]
    c = [Ins("S2R R%d, SR_TID.X" % tid, wsb=SB_RD),
         Ins("ULDC.64 UR%d, c[0x0][0x118]" % UR, raw=(0x0000460000007ab9 | (UR << 16), 0xa00))]
    for r in range(AF[0], ACC + 32 * mt, 2):
        c.append(Ins("CS2R R%d, SRZ" % r))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % R["xr"][0]))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % R["xr"][1]))
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % S))
    c.append(Ins("NOP", stall=5))
    c.append(Ins("LDS.64 R%d, [R%d+0x%x]" % (pa, S, R["ptr"]), wsb=SB_OFF))
    c.append(Ins("LDS.64 R%d, [R%d+0x%x]" % (pb, S, R["ptr"] + 8), wsb=SB_OFF))
    c.append(Ins(andi(S + 1, tid, 0x1f), wait=(SB_RD,)))                      # lane
    c.append(Ins(shr(S + 2, tid, 5), wait=(SB_RD,)))                          # warp
    c.append(Ins(muladd(trw, tid, 0x40), wait=(SB_RD,)))                      # the thread's transcript ...
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, c[0x0][0x178]" % R["cnt"]))       # chunks
    c.append(Ins(muladd(S + 1, S + 1, 0x10)))                                 # 16 lane
    c.append(Ins(shr(S + 3, S + 2, 2)))                                       # row band
    c.append(Ins(andi(S + 2, S + 2, 3)))                                      # column slot
    c.append(Ins(addi(trw, trw, R["tr"] + 60)))                               # ... at its word 15
    c.append(Ins(muladd(S + 3, S + 3, 512 * mt, S + 1)))                      # the band's tiles, the lane's word
    c.append(Ins(muladd(S + 2, S + 2, 0x800, S + 1)))                         # the slot's pairs, the lane's word
    c.append(Ins("NOP", stall=8))
    c.append(Ins("IADD3 R%d, R%d, 0x0, R%d" % (pa, pa, S + 3), wait=(SB_OFF,)))   # the host keeps these from
    c.append(Ins("IADD3 R%d, R%d, 0x0, R%d" % (pb, pb, S + 2), wait=(SB_OFF,)))   # carrying
    c.append(Ins("NOP", stall=8))
    c.append(ldg(AF[0], pa, 0, wsb=2))
    for p in range(4):
        c.append(ldg(BF + 4 * p, pb, 512 * p, wsb=3, rsb=SB_RD if p == 3 else None))
    c.append(Ins("NOP", stall=8))
    c.append(Ins(addi(pb, pb, -0x8000), wait=(SB_RD,)))                       # a chunk behind: see chunk()
    return c


def chunk(cfg, R):
    """Gaps 0 .. 32 mt - 1 of a chunk, each a list of (Ins, wrapped). A wrapped item belongs to the readout of
    the chunk before; the epilogue runs those once more after the last chunk. Also the IMMA that wait."""
    mt = cfg["mt"]
    ni, tot = 8 * mt, 32 * mt
    G = [[] for _ in range(tot)]
    iwait = {}

    def put(g, ins, wrapped=None):
        G[g % tot].append((ins, g >= tot if wrapped is None else wrapped))

    plans = []          # (LDSM gap, IADD3 gap, PRMT gap, address register, offset, d1, d2, rsb)
    lead = cfg["lead"]
    hold = SB_RD if cfg["hold"] else None
    hw = (SB_RD,) if cfg["hold"] else ()
    if cfg["dg"]:
        # The direct fold: tile m's fragment is loaded two IMMA behind the last IMMA of the tile that used the
        # same register set, the next k-step's first tile and B's four pairs likewise. Each load has a
        # scoreboard and its first two IMMA wait on it. pa is the chunk's first k-step; pb is a chunk behind
        # until gap 4 (pair 3's load of the last k-step falls in the next chunk's gap 1), so the first chunk's
        # pass of that load fetches the first k-step's pair 3 again.
        pa, pb = R["pa"], R["pb"]
        for s in range(4):
            b = ni * s
            for m in range(1, mt):
                put(b + 8 * (m - 1) + 1, ldg(AF[(m + (mt & 1) * s) & 1], pa, 1024 * mt * s + 512 * m, wsb=m & 1), False)
                for g in (b + 8 * m, b + 8 * m + 1):
                    iwait.setdefault(g % tot, []).append(m & 1)
            put(b + ni - 7, ldg(AF[((mt & 1) * (s + 1)) & 1], pa, 1024 * mt * (s + 1), wsb=2,
                                rsb=hold if s == 3 else None), False)
            for g in (b + ni, b + ni + 1):
                iwait.setdefault(g % tot, []).append(2)
            for p in range(4):
                put(b + ni - 5 + 2 * p, ldg(BF + 4 * p, pb, 8192 * (s + 1) + 512 * p, wsb=3,
                                            rsb=hold if s == 3 and p == 3 else None), False)
                for g in (b + ni + 2 * p, b + ni + 2 * p + 1):
                    iwait.setdefault(g % tot, []).append(3)
            for j in range(mt + 4) if cfg["pf"] else ():
                # Gaps 5 on: pb is the chunk's own from gap 4. The results are never read, but the loads
                # have a scoreboard: the loop's end waits for them too, or one would land on a register
                # of the kernel's after the return (it did: a fault in long runs).
                ahead = s + cfg["pfk"]
                off = 1024 * mt * ahead + 512 * j if j < mt else 8192 * ahead + 512 * (j - mt)
                put(b + 5 + 3 * j, ldg(R["s"], pa if j < mt else pb, off, wsb=SB_OFF), False)
        put(tot - 4, Ins(addi(pa, pa, 4096 * mt), wait=hw), False)
        put(tot + 4, Ins(addi(pb, pb, 0x8000), wait=hw), False)
    for s in range(4) if not cfg["dg"] else ():
        b = ni * s
        cur, nxt = R["oq"][s & 1], R["oq"][(s + 1) & 1]
        # The next k-step's offsets. The load overwrites the set the k-step before last read: it waits for
        # that k-step's last two LDSM (hold). In the chunk's last k-step it is the last user of the pointer.
        if not cfg["dbg"] & 1:
            put(b + 2, ldg(nxt, R["poff"], 0x200 * (s + 1), wsb=SB_OFF, rsb=hold if s == 3 else None, wait=hw))
        for g in (b + ni - 15, b + ni - 14, b + ni - 13):
            iwait.setdefault(g, []).append(SB_OFF)
        for m in range(1, mt):                         # tile m's fragments, during tile m - 1
            a = AF[(m + (mt & 1) * s) & 1]
            for h in range(2):
                g0 = b + 8 * (m - 1) + h
                plans.append((g0, g0 + lead, g0 + lead + 2, cur + h, 2048 * m, a + 2 * h, a + 2 * h + 1,
                              hold if m == mt - 1 else None))
        a = AF[((mt & 1) * (s + 1)) & 1]               # the next k-step's first tile, two IMMA behind the
        for h in range(2):                             # last IMMA of the tile that uses the same set
            g0 = b + ni - 11 + h
            plans.append((g0, g0 + lead, g0 + lead + 2, nxt + h, 0, a + 2 * h, a + 2 * h + 1, None))
        for p in range(4):                             # the next k-step's B: pair p's PRMT at r 8mt-5+2p (+1),
            for h in range(2):                         # two IMMA behind the last IMMA that reads the pair
                g0 = b + ni - 9 + 2 * p + h
                plans.append((g0, g0 + lead, g0 + lead + 2, nxt + 2 + h, 2048 * p, BF + 4 * p + h,
                              BF + 4 * p + 2 + h, None))
    if cfg["direct"]:
        plans = []
        q = R["quads"]
        for s in range(4):
            b = ni * s
            cur, nxt = R["oq"][s & 1], R["oq"][(s + 1) & 1]
            for m in range(1, mt):
                a = AF[(m + (mt & 1) * s) & 1]
                put(b + 8 * (m - 1) + 1, Ins("LDSM.16.MT88.4 R%d, [R%d+0x%x]" % (a, cur, 2048 * m), wsb=m & 1), False)
                for g in (b + 8 * m, b + 8 * m + 1):
                    iwait.setdefault(g % tot, []).append(m & 1)
            a = AF[((mt & 1) * (s + 1)) & 1]
            put(b + ni - 7, Ins("LDSM.16.MT88.4 R%d, [R%d]" % (a, nxt), wsb=2), False)
            for g in (b + ni, b + ni + 1):
                iwait.setdefault(g % tot, []).append(2)
            for p in range(4):
                put(b + ni - 5 + 2 * p, Ins("LDSM.16.MT88.4 R%d, [R%d%s]" % (
                    BF + 4 * p, nxt + 2, "+0x%x" % (2048 * p) if p else ""), wsb=3), False)
                for g in (b + ni + 2 * p, b + ni + 2 * p + 1):
                    iwait.setdefault(g % tot, []).append(3)
            for j in range(cfg["cu"]):
                g = b + 9 + (ni - 20) * j // max(1, cfg["cu"])
                put(g, ldg(q[1], R["poff"], 0x200 * (s + 1) if s < 3 else 0), False)
                put(g + 1, Ins("STS.128 [R%d], R%d" % (q[4], q[0])), False)
            for j in range(cfg["xalu"]):
                g = b + 4 + (ni - 14) * j // max(1, cfg["xalu"])
                x = q[2] + (j & 3)
                put(g, Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (x, x, q[3] + 2, q[3] + 3)), False)
    # Quads and scoreboards go round in the order the LDSM are issued.
    order = sorted(range(len(plans)), key=lambda k: plans[k][0])
    nq = len(R["quads"])
    assert len(plans) % nq == 0 and len(plans) % 4 == 0
    use = {}
    for n, k in enumerate(order):
        g_ld, g_add, g_prm, areg, aoff, d1, d2, rsb = plans[k]
        q = R["quads"][n % nq]
        for (a0, a1) in use.get(q, []):                # the quad is free from the gap behind its last PRMT
            assert not any(a0 <= g_prm + w and g_ld + w <= a1 for w in (-tot, 0, tot)), (q, g_ld)
        use.setdefault(q, []).append((g_ld, g_prm))
        ld, adds, prm = gather_ins(q, areg, aoff, d1, d2, n % 4, cfg, rsb=rsb)
        if cfg["nogather"]:                # timing only: no gathers at all, or (noalu) their LDSM alone
            continue
        put(g_ld, ld, wrapped=False)
        for x in adds if not cfg["noalu"] else ():
            put(g_add, x, wrapped=False)
        for x in prm if not cfg["noalu"] else ():
            put(g_prm, x, wrapped=False)
    # The readout. Chain 0 takes the even tiles and chain 1 the odd ones, one XOR3 of each a gap, in the
    # gaps from r roff of the last k-step that B's gathers leave alone.
    b3 = 3 * ni
    busy = set(range(b3 + ni - 9, tot + 3))
    end = b3
    for ch in range(2) if not cfg["noread"] else ():
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
    if not cfg["noread"]:
        assert end > tot, "the readout's tail is taken to lie in the next chunk"
        trw, tmp = R["trw"], R["tmp"]
        w, xr = R["w"], R["xr"]
        put(end - 4, Ins("LDS R%d, [R%d]" % (w, trw), wsb=SB_RD), wrapped=True)
        put(end + 0, Ins("SHF.L.W.U32.HI R%d, R%d, 0xd, R%d" % (w, w, w), wait=(SB_RD,)), wrapped=True)
        put(end + 1, Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (w, w, xr[0], xr[1])), wrapped=True)
        put(end + 2, Ins("STS [R%d], R%d" % (trw, w), rsb=hold), wrapped=True)
        # The next word of the sixteen: the pointer is 64-aligned, so its low six bits are the word.
        put(end + 3, Ins(addi(tmp, trw, 4)), wrapped=True)
        put(end + 4, Ins(andi(tmp, tmp, 0x3c)), wrapped=True)
        put(end + 5, Ins("LOP3.LUT R%d, R%d, 0x3f, R%d, 0xba, !PT" % (trw, trw, tmp), wait=hw), wrapped=True)
        assert end + 5 - tot < ni - 16
    # The chunk's count and the offsets pointer, in quiet gaps of the last k-step.
    put(b3 + 6, Ins(addi(R["cnt"], R["cnt"], -1)), wrapped=False)
    put(b3 + 7, Ins("ISETP.NE.AND P1, PT, R%d, 0x0, PT" % R["cnt"]), wrapped=False)
    if not cfg["dbg"] & 2 and not cfg["dg"]:
        put(b3 + 14, Ins(addi(R["poff"], R["poff"], 0x800), wait=hw), wrapped=False)
    return G, iwait


def emit(G, iwait, cfg, bfalt=None):
    """`bfalt`: a second set of sixteen registers for B's fragments, read in a chunk's second and fourth k-step
    (rfold.py's bfa: the next turn's fragments are then loaded while this turn's are still read)."""
    mt = cfg["mt"]
    ni, tot = 8 * mt, 32 * mt
    out, most = [], 0
    for g in range(tot):
        s, i = divmod(g, ni)
        mb, nb = divmod(i, 8)
        items = [ins for ins, _w in G[g]]
        most = max(most, len(items))
        nsingle = 0
        if cfg["sf"]:
            # sf: a gap's loads and one-register steps go first, right behind the IMMA, and the IMMA's stall
            # shrinks by their count (only a step that reads two or more registers has to keep its distance).
            def single(x):
                r, _w, kind = fold.rw(x.text)
                return kind == "load" or (kind == "alu" and len(r) <= 1)
            items = [x for x in items if single(x)] + [x for x in items if not single(x)]
            nsingle = sum(1 for x in items if single(x))
        ist = cfg["ist"]
        if cfg["sf"]:
            ist = max(1, ist - nsingle) if nsingle < len(items) else 1
        a = AF[(mb + (mt & 1) * s) & 1]
        bf = bfalt if bfalt is not None and s & 1 else BF
        out.append(Ins("IMMA.16832.S8.S8 R%d, R%d.ROW, R%d.COL, R%d" % (ACC + 4 * i, a, bf + 2 * nb, ACC + 4 * i),
                       stall=cfg["pad"] if not items else ist, yld=bool(cfg["iy"]), wait=iwait.get(g, ()),
                       reuse=1 if cfg["ru"] and nb < 7 else 0, label=".L0" if g == 0 else None))
        for k, x in enumerate(items):
            x = Ins(x.text, x.stall, bool(cfg["py"]), x.wsb, x.rsb, x.wait, raw=x.raw)
            if k == len(items) - 1:
                x.yld = bool(cfg["ly"])
                if cfg["pol"] == 0:
                    x.stall = max(x.stall, cfg["pad"] - ist - (len(items) - 1))
            out.append(x)
    out.append(Ins("@P1 BRA .L0", stall=5))
    # After the last chunk: the readout's tail once more.
    ep = []
    for g in range(tot):
        for x, wrapped in G[g]:
            if wrapped:
                ep.append(Ins(x.text, cfg["pst"], x.yld, x.wsb, x.rsb, x.wait, raw=x.raw))
    return out, ep, most


def build(cfg):
    global UR
    UR = cfg["ur"]
    R = layout(cfg["mt"], cfg["dg"])
    pro = prologue_dg(cfg, R) if cfg["dg"] else prologue(cfg, R)
    bad = check(pro, "prologue", fix=True)
    G, iwait = chunk(cfg, R)
    loop, ep, most = emit(G, iwait, cfg)
    body = loop[:-1]
    bad += check(pro + body + body + body, "loop")
    bad += check(body + ep, "epilogue", fix=True)
    if cfg["dg"]:
        # The direct fold returns through R4, which is a fragment register: the last chunk's loads of the
        # k-step that never comes are still on their way and would land on the return address. One step
        # waits for every load, then the return address is written (asm89.patch).
        ep.append(Ins(addi(R["tmp"], R["tmp"], 0), wait=(0, 1, 2, 3, SB_OFF, SB_RD)))
        ep.append(Ins("NOP", stall=8))
    # dbg (looking for a fault, TIMING ONLY): 1 no offsets load in the loop, 2 the pointer does not move,
    # 8 no loop at all, 16 no epilogue, 4 no prologue gathers' loads (nold covers the loop's).
    if cfg["dbg"] & 8:
        loop = []
    if cfg["dbg"] & 16:
        ep = []
    return pro + loop + ep, bad, most, sum(len(x) for x in G)


def main():
    cubin, sass, nvd, out = sys.argv[1:5]
    cfg = dict(CFG)
    for kv in sys.argv[5:]:
        k, v = kv.split("=")
        assert k in cfg, k
        cfg[k] = int(v, 0)
    code, bad, most, plain = build(cfg)
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
