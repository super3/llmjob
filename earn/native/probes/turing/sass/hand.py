#!/usr/bin/env python3
"""Write the hand-scheduled Turing SASS for the fold's inner loop.

usage: hand.py kernels.cuasm out.cuasm [--list]

kernels.cuasm is CuAssembler's text of the nvcc-built kernels.cubin. Each
sass_hand* kernel there is a copy of sass_ptxas (same parameters, same
attributes); this replaces its code with a schedule written here, and writes
the result to out.cuasm for CuAssembler to assemble. --list prints the
generated code of every variant instead.

The loop (one k-step of the B-direct fold, four a loop pass, as ptxas's is):
  - 128 IMMA.8816 a k-step into the 64x64 warp tile's 128 accumulators: the
    64 k-lo IMMA first, then the 64 k-hi, so the two IMMA that write one
    accumulator pair are 64 IMMA (256 issue cycles) apart. ptxas puts them
    as close as one apart and then waits on a scoreboard that a later IMMA
    also set, which drains the warp's tensor pipe; nothing here waits on an
    IMMA issued less than 32 IMMA earlier.
  - B: four LDG.128 (ld.global.nc) a k-step, fragment-ordered as the fold's,
    D k-steps ahead into four rotating 16-register buffers. Each k-step's
    loads get their own scoreboard, so the first IMMA of a k-step waits for
    its own B and nothing issued later.
  - A: four LDSM.x4 a k-step from the fold's two swizzled 16 KB stages, one
    k-step ahead, double-buffered.
  - Every IMMA issues 4 cycles after the one before (the pipe's rate, one
    IMMA.8816 every 4 cycles a scheduler); loads and loop control sit in the
    3 free issue cycles between IMMA.

Scoreboards: 0 LDSM data, 1 late operand reads (the last IMMA of a k-step,
every LDG and LDSM), then either 2-3 IMMA results (two groups of 32
alternating) and 4-5 LDG data by k-step parity (D <= 2), or 2-5 LDG data by
k-step mod 4 with no IMMA scoreboard (D = 3: IMMA results are then trusted to
its fixed latency, which ptxas itself relies on at 36 cycles).

Variants (kernel: D, IMMA scoreboards):
  sass_hand1  1, yes   the fold's prefetch distance: the schedule alone
  sass_hand2  2, yes
  sass_hand3  3, no
  sass_hceil  the same loop with no loads at all: a hand-written ceiling

A checker runs on every variant before anything is written: it walks the
prologue, three loop passes and the epilogue and fails on any register read
before its producer is done (fixed latency by summed stall counts, variable
latency by scoreboard), any register overwritten before a pending reader has
read it, and any scoreboard wait that would drain the IMMA pipe.
"""
import re
import sys

# ---------------------------------------------------------------- registers
ACC0 = 2           # acc[mb][nb][c] = R(2 + mb*32 + nb*4 + c): out index i is R(2+i)
ABUF = 132         # A[b][mb][r] = R(132 + b*16 + mb*4 + r), b in 0..1
BBUF = 164         # B[b][j][r] = R(164 + b*16 + j*4 + r), b in 0..3
Q, RR, BW = 228, 230, 232          # 64-bit: B load bases, B + this warp's offset
QO, RO, QNO, IT, TL, CGN, INC = 234, 235, 236, 237, 238, 239, 240
CUR = (None, 241, 242, 243)        # LDSM address of k-step t (1..3), this stage
NXT0 = 244                         # LDSM address of k-step 0, the other stage
TMP, TMP2, CTA, T0 = 245, 246, 247, 248
MAXREG = 249
PARAM = dict(A=0x160, B=0x168, out=0x170, cyc=0x178, tiles=0x180, ncg=0x184, kmask=0x188)
FRAG = 0x8000      # one 16-column block of B': k/32 * 512 bytes
KSTEP_B = 0x200    # one k-step of a block: 512 bytes
TILE_B = 0x80000   # one column group: 16 blocks
STAGE = 0x4000     # one A stage

VARIANTS = {
    'sass_hand1': dict(d=1, imma_sb=True, loads=True),
    'sass_hand2': dict(d=2, imma_sb=True, loads=True),
    'sass_hand3': dict(d=3, imma_sb=False, loads=True),
    'sass_hceil': dict(d=2, imma_sb=True, loads=False),
}


def acc(mb, nb, u):
    return ACC0 + mb * 32 + nb * 4 + 2 * u


def areg(b, mb, r):
    return ABUF + b * 16 + mb * 4 + r


def breg(b, j, r=0):
    return BBUF + b * 16 + j * 4 + r


class Ins:
    def __init__(self, text, stall=1, wait=(), rsb=None, wsb=None, yld=False, drain=False):
        self.text, self.stall, self.wait, self.rsb, self.wsb, self.yld = text, stall, set(wait), rsb, wsb, yld
        self.label, self.drain = None, drain

    def ctrl(self):
        # As ptxas 12.8 writes them: no stall over 12, and the yield flag set
        # on every stall over 9 (nvdisasm rejects 13-15 on ALU ops).
        assert 0 <= self.stall <= 12, self.text
        if self.stall > 9:
            self.yld = True
        b = ''.join(str(i) if i in self.wait else '-' for i in range(6))
        return '[B%s:R%s:W%s:%s:S%02d]' % (b, '-' if self.rsb is None else self.rsb,
                                         '-' if self.wsb is None else self.wsb,
                                         'Y' if self.yld else '-', self.stall)


def lbl(name):
    i = Ins(None)
    i.label = name
    return i


# ---------------------------------------------------------------- the loop
def immas(abuf, bbuf):
    """The 128 IMMA of a k-step: (acc pair, A reg, B reg, B reused next)."""
    out = []
    for ph in (0, 1):
        for nb in range(8):
            j, h = nb >> 1, nb & 1
            b = breg(bbuf, j, 2 * h + ph)
            for mb in range(4):
                for u in range(2):
                    out.append([acc(mb, nb, u), areg(abuf, mb, u + 2 * ph), b])
    for i, x in enumerate(out):
        x.append(i + 1 < len(out) and out[i + 1][2] == x[2])
    return out


def ldg_sb(v, s):
    """Scoreboard of the B loads for k-step s (mod 4)."""
    return 4 + s % 2 if v['imma_sb'] else 2 + s % 4


def kstep(v, t, name):
    """K-step t (0..3) of a loop pass: k-step s = 4i + t, A buffer t % 2,
    B buffer t. Returns the instruction list."""
    d, loads = v['d'], v['loads']
    slots = {}   # IMMA index -> instructions issued after it

    def at(p, ins):
        slots.setdefault(p, []).append(ins)

    if loads:
        # B for k-step s + d into buffer (t + d) % 4, from Q, or from RR where
        # the k-step may be the next tile's (see the loop control).
        tb = (t + d) % 4
        # Loads past the pass (t >= 4 - d) go from RR, which points at the
        # pass's k-step 4 (the next tile's k-step 0 in a tile's last pass).
        base, k0 = (Q, t) if t < 4 - d else (RR, t - (4 - d))
        for j in range(4):
            off = k0 * KSTEP_B + j * FRAG
            addr = '[R%d+0x%x]' % (base, off) if off else '[R%d]' % base
            at(7 + 8 * j, Ins('LDG.E.128.CONSTANT.SYS R%d, %s' % (breg(tb, j), addr),
                              wait=(1,) if j == 0 else (), rsb=1, wsb=ldg_sb(v, t + d)))
        # A for k-step s + 1 into the other buffer; k-step 0 of the next pass
        # reads the other stage.
        ab = (t + 1) % 2
        src = CUR[t + 1] if t < 3 else NXT0
        for mb in range(4):
            off = mb * 0x800
            addr = '[R%d+0x%x]' % (src, off) if off else '[R%d]' % src
            at(39 + 8 * mb, Ins('LDSM.16.M88.4 R%d, %s' % (areg(ab, mb, 0), addr), rsb=1, wsb=0))
    if t == 3:
        loop_control(v, at, name)

    out = []
    seq = immas(t % 2, t)
    for i, (c, a, b, reuse) in enumerate(seq):
        wait = set()
        g = i // 32
        if i == 0 and loads:
            wait |= {0, ldg_sb(v, t)}
        if v['imma_sb'] and i % 32 == 0:
            wait.add(2 + g % 2)
        ins = Ins('IMMA.8816.S8.S8 R%d, R%d.ROW, R%d%s.COL, R%d' % (c, a, b, '.reuse' if reuse else '', c),
                  stall=4, wait=wait, wsb=(2 + g % 2) if v['imma_sb'] else None,
                  rsb=1 if i == len(seq) - 1 else None)
        out.append(ins)
        extra = slots.pop(i, [])
        if extra:
            ins.stall = 1
            for k, x in enumerate(extra):
                x.stall = 1 if k + 1 < len(extra) else max(1, 3 - (len(extra) - 1))
            out.extend(extra)
    assert not slots, slots
    return out


def loop_control(v, at, name):
    """End of a loop pass, spread over k-step 3 (see the module doc). it counts
    passes in the tile (16 a tile); qo is the B offset of this pass's first
    load, qno the next tile's base. RR is B + this pass's k-step 4, which the
    loads that run past the pass use: Q + (4 - d) k-steps, except in a tile's
    last pass, where it is the next tile's k-step 0. (Every 32-bit offset
    stays non-negative: IMAD.WIDE.U32 widens it unsigned.)"""
    d = v['d']
    at(66, Ins('ISETP.EQ.AND P0, PT, R%d, 0xf, PT' % IT))
    at(66, Ins('IADD3 R%d, R%d, 0x1, RZ' % (TMP, CGN)))
    at(67, Ins('IADD3 R%d, R%d, R%d, RZ' % (QO, QO, INC)))
    at(67, Ins('IADD3 R%d, R%d, 0x1, RZ' % (IT, IT)))
    at(71, Ins('ISETP.EQ.AND P1, PT, R%d, c[0x0][0x%x], PT' % (TMP, PARAM['ncg'])))
    at(71, Ins('@P0 IADD3 R%d, R%d, 0x%x, RZ' % (QO, QNO, d * KSTEP_B)))
    at(72, Ins('@P0 IMAD.MOV.U32 R%d, RZ, RZ, RZ' % IT))
    at(72, Ins('@P0 IADD3 R%d, R%d, -0x1, RZ' % (TL, TL)))
    at(76, Ins('@P1 IMAD.MOV.U32 R%d, RZ, RZ, RZ' % TMP))
    at(77, Ins('ISETP.EQ.AND P2, PT, R%d, 0xf, PT' % IT))
    at(79, Ins('@P0 IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (CGN, TMP)))
    at(81, Ins('@P0 IMAD.SHL.U32 R%d, R%d, 0x80000, RZ' % (QNO, CGN)))
    at(83, Ins('IADD3 R%d, R%d, 0x%x, RZ' % (TMP2, QO, (4 - d) * KSTEP_B)))
    at(85, Ins('SEL R%d, R%d, R%d, P2' % (RO, QNO, TMP2)))
    # Q, RR and the LDSM addresses are read late by this pass's loads: wait
    # for those reads (scoreboard 1) before rewriting them.
    at(86, Ins('IMAD.WIDE.U32 R%d, R%d, 0x1, R%d' % (Q, QO, BW), wait=(1,)))
    at(88, Ins('IMAD.WIDE.U32 R%d, R%d, 0x1, R%d' % (RR, RO, BW)))
    for k, r in enumerate((CUR[1], CUR[2], CUR[3], NXT0)):
        at(89 + k, Ins('LOP3.LUT R%d, R%d, 0x4000, RZ, 0x3c, !PT' % (r, r)))
    at(94, Ins('ISETP.NE.AND P3, PT, R%d, RZ, PT' % TL))


def loop_body(v, name):
    out = [lbl('.L_%s_loop' % name)]
    for t in range(4):
        out += kstep(v, t, name)
    out[-1].stall = 1
    out.append(Ins('@P3 BRA `(.L_%s_loop)' % name, stall=5))
    return out


# ------------------------------------------------------- prologue, epilogue
def prologue(v, name):
    d, loads = v['d'], v['loads']
    o = []
    I = lambda *a, **k: o.append(Ins(*a, **k))
    I('IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]', stall=2)
    I('S2R R0, SR_TID.X', wsb=0)
    I('S2R R%d, SR_CTAID.X' % CTA, wsb=0, stall=2)
    # Stage this block's 256-byte A rows, two 128-byte chunks, into the two
    # stages, swizzled as the fold stores them. Thread: row tid>>3 (+32 a
    # slot), unit tid&7. Temporaries live in B buffers 0-1 and 3.
    T = 212
    I('SHF.R.U32.HI R%d, RZ, 0x3, R0' % T, wait=(0,), stall=2)
    I('LOP3.LUT R%d, R0, 0x7, RZ, 0xc0, !PT' % (T + 1), stall=2)
    I('IMAD.MOV.U32 R%d, RZ, RZ, 0x1' % (T + 2), stall=4)
    I('IMAD.SHL.U32 R%d, R%d, 0x100, RZ' % (T + 3, T), stall=6)
    I('IMAD R%d, R%d, 0x10, R%d' % (T + 3, T + 1, T + 3), stall=6)
    I('IMAD R%d, R%d, 0x8000, R%d' % (T + 3, CTA, T + 3), stall=6)
    I('IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x%x]' % (T + 4, T + 3, T + 2, PARAM['A']), stall=6)
    I('LOP3.LUT R%d, R0, R%d, RZ, 0x3c, !PT' % (T + 6, T), stall=6)
    I('LOP3.LUT R%d, R%d, 0x7, RZ, 0xc0, !PT' % (T + 6, T + 6), stall=6)
    I('IMAD.SHL.U32 R%d, R%d, 0x10, RZ' % (T + 6, T + 6), stall=6)
    I('IMAD R%d, R%d, 0x80, R%d' % (T + 7, T, T + 6), stall=6)
    for i in range(8):
        off = (i & 3) * 0x2000 + (i >> 2) * 0x80
        addr = '[R%d+0x%x]' % (T + 4, off) if off else '[R%d]' % (T + 4)
        I('LDG.E.128.CONSTANT.SYS R%d, %s' % (BBUF + 4 * i, addr), rsb=2, wsb=1, stall=2)
    for i in range(8):
        off = (i >> 2) * STAGE + (i & 3) * 0x1000
        addr = '[R%d+0x%x]' % (T + 7, off) if off else '[R%d]' % (T + 7)
        I('STS.128 %s, R%d' % (addr, BBUF + 4 * i), wait=(1,) if i == 0 else (), rsb=2, stall=2)
    I('BAR.SYNC 0x0', stall=5)
    # ldmatrix lane address, k-step 0, stage 0 (the fold's aLane0):
    # wr*8192 + (lane&15)*128 + (((lane>>4)^lane)&1)*16 + (lane&6)*16
    I('LOP3.LUT R%d, R0, 0x1f, RZ, 0xc0, !PT' % (T + 8), stall=2)
    I('LOP3.LUT R%d, R0, 0xf, RZ, 0xc0, !PT' % (T + 9), stall=2)
    I('LOP3.LUT R%d, R0, 0x80, RZ, 0xc0, !PT' % (T + 10), stall=4)
    I('SHF.R.U32.HI R%d, RZ, 0x4, R%d' % (T + 11, T + 8), stall=2)
    I('LOP3.LUT R%d, R%d, 0x6, RZ, 0xc0, !PT' % (T + 12, T + 8), stall=4)
    I('LOP3.LUT R%d, R%d, R%d, RZ, 0x3c, !PT' % (T + 11, T + 11, T + 8), stall=6)
    I('LOP3.LUT R%d, R%d, 0x1, RZ, 0xc0, !PT' % (T + 11, T + 11), stall=6)
    I('IMAD.SHL.U32 R%d, R%d, 0x10, RZ' % (T + 11, T + 11), stall=2)
    I('IMAD.SHL.U32 R%d, R%d, 0x10, RZ' % (T + 12, T + 12), stall=2)
    I('IMAD.SHL.U32 R%d, R%d, 0x80, RZ' % (T + 9, T + 9), stall=2)
    I('IMAD.SHL.U32 R%d, R%d, 0x40, RZ' % (T + 10, T + 10), stall=6)
    I('IADD3 R%d, R%d, R%d, R%d' % (T + 13, T + 9, T + 10, T + 11), stall=6)
    I('IADD3 R%d, R%d, R%d, RZ' % (T + 13, T + 13, T + 12), stall=6)
    aL = T + 13
    I('LOP3.LUT R%d, R%d, 0x20, RZ, 0x3c, !PT' % (CUR[1], aL), stall=1)
    I('LOP3.LUT R%d, R%d, 0x40, RZ, 0x3c, !PT' % (CUR[2], aL), stall=1)
    I('LOP3.LUT R%d, R%d, 0x60, RZ, 0x3c, !PT' % (CUR[3], aL), stall=1)
    I('LOP3.LUT R%d, R%d, 0x4000, RZ, 0x3c, !PT' % (NXT0, aL), stall=2)
    # Bw = B + wc * 0x20000 + lane * 16
    I('SHF.R.U32.HI R%d, RZ, 0x5, R0' % TMP, stall=6)
    I('LOP3.LUT R%d, R%d, 0x3, RZ, 0xc0, !PT' % (TMP, TMP), stall=6)
    I('IMAD.SHL.U32 R%d, R%d, 0x20000, RZ' % (TMP, TMP), stall=6)
    I('IMAD R%d, R%d, 0x10, R%d' % (TMP, T + 8, TMP), stall=6)
    I('IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x%x]' % (BW, TMP, T + 2, PARAM['B']), stall=6)
    # This block's first column group: ctaid % ncg, by subtraction.
    I('IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (TMP2, CTA), stall=6)
    o.append(lbl('.L_%s_mod' % name))
    I('ISETP.GE.U32.AND P0, PT, R%d, c[0x0][0x%x], PT' % (TMP2, PARAM['ncg']), stall=12)
    I('@P0 IADD3 R%d, R%d, -c[0x0][0x%x], RZ' % (TMP2, TMP2, PARAM['ncg']), stall=6)
    I('@P0 BRA `(.L_%s_mod)' % name, stall=5)
    # qcur (in T+14) = cg0 << 19; cgn = cg0 + 1 wrapped; qno = cgn << 19
    I('IMAD.SHL.U32 R%d, R%d, 0x80000, RZ' % (T + 14, TMP2), stall=1)
    I('IADD3 R%d, R%d, 0x1, RZ' % (CGN, TMP2), stall=6)
    I('ISETP.EQ.AND P1, PT, R%d, c[0x0][0x%x], PT' % (CGN, PARAM['ncg']), stall=12)
    I('@P1 IMAD.MOV.U32 R%d, RZ, RZ, RZ' % CGN, stall=6)
    I('IMAD.SHL.U32 R%d, R%d, 0x80000, RZ' % (QNO, CGN), stall=1)
    I('IADD3 R%d, R%d, 0x%x, RZ' % (QO, T + 14, d * KSTEP_B), stall=6)
    I('IADD3 R%d, R%d, 0x%x, RZ' % (RO, QO, (4 - d) * KSTEP_B), stall=1)
    I('IMAD.MOV.U32 R%d, RZ, RZ, RZ' % IT, stall=1)
    I('IMAD.MOV.U32 R%d, RZ, RZ, c[0x0][0x%x]' % (TL, PARAM['tiles']), stall=1)
    I('IMAD.MOV.U32 R%d, RZ, RZ, c[0x0][0x%x]' % (TMP, PARAM['kmask']), stall=6)
    I('ISETP.EQ.AND P2, PT, R%d, 0x3f, PT' % TMP, stall=1)
    I('IMAD.MOV.U32 R%d, RZ, RZ, RZ' % INC, stall=12)
    I('@P2 IMAD.MOV.U32 R%d, RZ, RZ, 0x800' % INC, stall=1)
    I('IMAD.WIDE.U32 R%d, R%d, 0x1, R%d' % (Q, QO, BW), stall=1)
    I('IMAD.WIDE.U32 R%d, R%d, 0x1, R%d' % (RR, RO, BW), stall=1)
    I('IMAD.WIDE.U32 R%d, R%d, 0x1, R%d' % (T + 14, T + 14, BW), stall=6)   # tile 0's B (T+14, T+15)
    # Zero the accumulators.
    for p in range(64):
        I('CS2R R%d, SRZ' % (ACC0 + 2 * p), stall=1)
    if loads:
        # B for k-steps 0..d-1 into buffers 0..d-1, after the staging stores
        # have read buffers 0-1 (scoreboard 2), and A for k-step 0.
        for k in range(d):
            for j in range(4):
                off = k * KSTEP_B + j * FRAG
                addr = '[R%d+0x%x]' % (T + 14, off) if off else '[R%d]' % (T + 14)
                I('LDG.E.128.CONSTANT.SYS R%d, %s' % (breg(k, j), addr),
                  wait=(2,) if k == 0 and j == 0 else (), rsb=1, wsb=ldg_sb(v, k), stall=2)
        for mb in range(4):
            off = mb * 0x800
            addr = '[R%d+0x%x]' % (aL, off) if off else '[R%d]' % aL
            I('LDSM.16.M88.4 R%d, %s' % (areg(0, mb, 0), addr), rsb=1, wsb=0, stall=2)
    else:
        # The ceiling's operands: any values, set once, never loaded.
        for r in list(range(ABUF, ABUF + 32)) + list(range(BBUF, BBUF + 64)):
            I('IMAD R%d, R0, 0x%x, R0' % (r, 0x01010101 + 2 * r), wait=(2,) if r == ABUF else (), stall=1)
    I('CS2R R%d, SR_CLOCKLO' % T0, stall=6)
    return o


def epilogue(v, name):
    o = []
    I = lambda *a, **k: o.append(Ins(*a, **k))
    t1 = QNO   # QNO:IT, free after the loop and read by no load
    I('CS2R R%d, SR_CLOCKLO' % t1, stall=2)
    # Every load and every IMMA done (the IMMA by their fixed latency where
    # they set no scoreboard: the padding below).
    I('IMAD.MOV.U32 R%d, RZ, RZ, 0x1' % CGN, wait=(0, 1, 2, 3, 4, 5), stall=12, drain=True)
    I('LOP3.LUT P4, RZ, R0, 0x1f, RZ, 0xc0, !PT', stall=12)
    I('SHF.R.U32.HI R%d, RZ, 0x5, R0' % TL, stall=12)
    I('IMAD.SHL.U32 R%d, R%d, 0x10, RZ' % (TL, TL), stall=12)
    I('IMAD R%d, R%d, 0x80, R%d' % (TL, CTA, TL), stall=6)
    I('IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x%x]' % (QO, TL, CGN, PARAM['cyc']), stall=6)
    I('@!P4 STG.E.64.SYS [R%d], R%d' % (QO, T0), stall=2)
    I('@!P4 STG.E.64.SYS [R%d+0x8], R%d' % (QO, t1), stall=2)
    I('IMAD.SHL.U32 R%d, R0, 0x4, RZ' % TMP, stall=6)
    I('IMAD R%d, R%d, 0x20000, R%d' % (TMP, CTA, TMP), stall=6)
    I('IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x%x]' % (CUR[1] + 1, TMP, CGN, PARAM['out']), stall=6)
    base = CUR[1] + 1   # R242:R243
    for i in range(128):
        addr = '[R%d+0x%x]' % (base, i * 0x400) if i else '[R%d]' % base
        I('STG.E.SYS %s, R%d' % (addr, ACC0 + i), stall=1 if i == 127 else 2)
    I('EXIT', stall=5)
    o.append(lbl('.L_%s_end' % name))
    I('BRA `(.L_%s_end)' % name, stall=0, yld=True)
    return o


def kernel(name):
    v = VARIANTS[name]
    pro, body, epi = prologue(v, name), loop_body(v, name), epilogue(v, name)
    check(v, name, pro, body, epi)
    return pro + body + epi


# ---------------------------------------------------------------- checker
FIXED_LAT = 6       # integer ALU, IMAD, CS2R: issue to result, generous
PRED_LAT = 12       # a predicate written by ISETP/LOP3 to its first use (ptxas: Y:S12)
IMMA_LAT = 36       # the longest IMMA result ptxas 12.8 leaves unguarded
VAR = ('LDG', 'LDSM', 'S2R', 'LDS')
LATE_READ = ('LDG', 'LDSM', 'STS', 'STG', 'IMMA')


def regs_of(text):
    """(predicate guard, op, dst regs, src regs, dst preds, src preds)."""
    m = re.match(r'(@!?(P\d))?\s*(\S+)\s*(.*)$', text)
    guard, op, rest = m.group(2), m.group(3), m.group(4)
    ops = [x.strip() for x in rest.split(',')] if rest else []
    base = op.split('.')[0]
    dst, src, pdst, psrc = set(), set(), set(), set()
    if guard:
        psrc.add(guard)

    def regs(s, width=1):
        out = set()
        for mm in re.finditer(r'\bR(\d+)\b', s):
            out |= set(range(int(mm.group(1)), int(mm.group(1)) + width))
        return out

    if base in ('STS', 'STG'):
        w = 4 if '.128' in op else (2 if '.64' in op else 1)
        src |= regs(ops[0], 2 if base == 'STG' else 1) | regs(ops[1], w)
    elif base in ('BAR', 'BRA', 'EXIT', 'NOP'):
        pass
    elif base == 'ISETP':
        pdst.add(ops[0])
        for x in ops[2:]:
            src |= regs(x)
            psrc |= set(re.findall(r'\bP\d\b', x))
    elif base == 'LOP3' and ops[0].startswith('P'):
        pdst.add(ops[0])
        for x in ops[2:]:
            src |= regs(x)
    else:
        w = 4 if (base in ('LDG', 'LDSM') and ('.128' in op or base == 'LDSM')) else \
            (2 if base in ('IMMA', 'CS2R') or '.WIDE' in op else 1)
        dst |= regs(ops[0], w)
        for k, x in enumerate(ops[1:]):
            if base == 'IMMA' and k == 2:
                src |= regs(x, 2)
            elif '[' in x and base == 'LDG':
                src |= regs(x, 2)
            elif base == 'IMAD' and '.WIDE' in op and k == 2:
                src |= regs(x, 2)
            else:
                src |= regs(x)
            psrc |= set(re.findall(r'\bP\d\b', x)) - {'PT'}
    return guard, base, dst, src, pdst, psrc


def check(v, name, pro, body, epi):
    seq = [i for i in pro + body * 3 + epi if i.text]
    cyc = 0
    ready = {}            # reg -> (kind, info): ('fix', cycle) | ('sb', sb, idx) | ('imma', cycle, sb)
    pready = {}           # pred -> cycle
    readers = {}          # reg -> list of (idx, op, rsb, cycle)
    pending = {s: [] for s in range(6)}   # sb -> (issue index, set as write) outstanding
    last_imma = []        # (idx, cycle, rsb) of IMMA, for in-order operand reads
    waited = {}           # sb -> index of last wait
    errs = []
    for idx, ins in enumerate(seq):
        guard, op, dst, src, pdst, psrc = regs_of(ins.text)
        for s in ins.wait:
            # A wait on a scoreboard an IMMA set less than 32 IMMA ago drains the pipe.
            for p, w in pending[s]:
                if w and seq[p].text.startswith('IMMA') and not ins.drain:
                    n = sum(1 for q in range(p + 1, idx) if seq[q].text.startswith('IMMA'))
                    if n < 31:
                        errs.append('%d: %s waits SB%d set by an IMMA %d IMMA back' % (idx, ins.text, s, n))
            pending[s] = []
            waited[s] = idx
        for r in src:
            st = ready.get(r)
            if st is None:
                continue
            if st[0] == 'fix' and cyc < st[1]:
                errs.append('%d: %s reads R%d %d cycles early' % (idx, ins.text, r, st[1] - cyc))
            if st[0] == 'sb' and waited.get(st[1], -1) < st[2]:
                errs.append('%d: %s reads R%d before SB%d (set at %d)' % (idx, ins.text, r, st[1], st[2]))
            if st[0] == 'imma' and cyc < st[1] and not (st[2] is not None and waited.get(st[2], -1) > st[3]):
                errs.append('%d: %s reads R%d %d cycles after its IMMA' % (idx, ins.text, r, cyc - st[1] + IMMA_LAT))
        for p in psrc:
            if p in pready and cyc < pready[p]:
                errs.append('%d: %s reads %s %d cycles early' % (idx, ins.text, p, pready[p] - cyc))
        for r in dst:
            for (ri, rop, rsb, rcyc) in readers.get(r, []):
                if ri == idx:
                    continue
                ok = rsb is not None and waited.get(rsb, -1) > ri
                if not ok and rop == 'IMMA':
                    # in-order operand reads: a later IMMA's read scoreboard, waited on
                    ok = any(j > ri and s is not None and waited.get(s, -1) > j for (j, c, s) in last_imma[-200:])
                    ok = ok or (op == 'IMMA') or cyc - rcyc >= IMMA_LAT
                if not ok and rop not in LATE_READ:
                    ok = True
                if not ok:
                    errs.append('%d: %s overwrites R%d before %s at %d has read it' % (idx, ins.text, r, seq[ri].text, ri))
            readers[r] = []
            st = ready.get(r)
            if st and st[0] == 'sb' and waited.get(st[1], -1) < st[2] and op != 'IMMA':
                errs.append('%d: %s overwrites R%d still being loaded' % (idx, ins.text, r))
        if op in LATE_READ:
            for r in src:
                readers.setdefault(r, []).append((idx, op, ins.rsb, cyc))
        if op in VAR and ins.wsb is None:
            errs.append('%d: %s has no write scoreboard' % (idx, ins.text))
        for r in dst:
            if op in VAR:
                ready[r] = ('sb', ins.wsb, idx)
            elif op == 'IMMA':
                ready[r] = ('imma', cyc + IMMA_LAT, ins.wsb, idx)
            else:
                ready[r] = ('fix', cyc + FIXED_LAT)
        for p in pdst:
            pready[p] = cyc + PRED_LAT
        if op == 'IMMA':
            last_imma.append((idx, cyc, ins.rsb))
        if ins.rsb is not None:
            pending[ins.rsb].append((idx, False))
        if ins.wsb is not None:
            pending[ins.wsb].append((idx, True))
        cyc += ins.stall
    if errs:
        sys.stderr.write('%s: %d hazards\n  %s\n' % (name, len(errs), '\n  '.join(errs[:30])))
        sys.exit(1)


# ---------------------------------------------------------------- splice
def splice(lines, name, code):
    """Replace kernel `name`'s instructions in a cuasm file's lines."""
    head = '.section\t.text.%s,' % name
    s = next(i for i, l in enumerate(lines) if l.strip().startswith(head.strip()) or
             l.startswith('\t' + head))
    start = next(i for i in range(s, len(lines)) if lines[i].strip() == '.text.%s:' % name)
    size = next(l for l in lines[s:start] if '.size' in l)
    end_label = re.search(r'\((\.L_\w+) -', size).group(1)
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == end_label + ':')
    for i in range(s, start):
        if 'SHI_REGISTERS' in lines[i]:
            lines[i] = re.sub(r'SHI_REGISTERS=\d+', 'SHI_REGISTERS=%d' % (MAXREG + 3), lines[i])
    out, addr = [], 0
    for ins in code:
        if ins.label:
            out.append('  %s:' % ins.label)
            continue
        out.append('      %s         /*%04x*/                   %s ;' % (ins.ctrl(), addr, ins.text))
        addr += 16
    while addr % 128:
        out.append('      [B------:R-:W-:Y:S00]         /*%04x*/                   NOP ;' % addr)
        addr += 16
    return lines[:start + 1] + out + lines[end:]


def norm(text):
    """An instruction as nvdisasm prints it, spacing and label names aside."""
    t = re.sub(r'`\(\.L_\w+\)', 'LABEL', text)
    return re.sub(r'\s+', ' ', t.replace(' ;', '').strip())


def verify(path):
    """Compare every variant's code with a disassembly of the assembled cubin."""
    pat = re.compile(r'\s*(\[B[0-5-]{6}:R[0-5-]:W[0-5-]:[Y-]:S\d\d\])\s+/\*[0-9a-f]+\*/\s+(.*?)\s*;?\s*$')
    lines = open(path).read().split('\n')
    bad = 0
    for name in VARIANTS:
        want = [(i.ctrl(), norm(i.text)) for i in kernel(name) if i.text]
        s = next(k for k, l in enumerate(lines) if l.strip().startswith('.section') and
                 '.text.%s,' % name in l)
        got = []
        for l in lines[s + 1:]:
            if re.match(r'\s*\.section\s', l):
                break
            m = pat.match(l)
            if m and m.group(2).strip() not in ('NOP', 'NOP ;'):
                got.append((m.group(1), norm(m.group(2))))
        diff = [(k, w, g) for k, (w, g) in enumerate(zip(want, got)) if w != g]
        if len(want) != len(got) or diff:
            bad += 1
            print('[sass] asm-check %s FAIL: %d vs %d instructions, first difference %s'
                  % (name, len(want), len(got), diff[:1]))
        else:
            print('[sass] asm-check %s ok: %d instructions match' % (name, len(want)))
    sys.exit(1 if bad else 0)


def main():
    if '--verify' in sys.argv:
        verify(sys.argv[sys.argv.index('--verify') + 1])
        return
    if len(sys.argv) < 3 and '--list' not in sys.argv:
        sys.exit(__doc__)
    if '--list' in sys.argv:
        for name in VARIANTS:
            code = kernel(name)
            print('==== %s' % name)
            for i in code:
                print(i.label + ':' if i.label else '  %s %s ;' % (i.ctrl(), i.text))
        return
    lines = open(sys.argv[1]).read().split('\n')
    for name in VARIANTS:
        lines = splice(lines, name, kernel(name))
    open(sys.argv[2], 'w').write('\n'.join(lines))


if __name__ == '__main__':
    main()
