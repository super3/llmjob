"""t96gen.py: the hand-written k loop of the RTX 20 fold with a 96 x 64 warp tile (src/pearl_t96.cu).

  python3 -I t96gen.py <base cubin> <nvdisasm> <out cubin>      (build_t96.py runs it)
  python3 -I t96gen.py --model                                  (the loop on a model of a warp; needs numpy)

pearl_t96.cu compiles a function, pearl_t96_loop, that folds one tile of one thread in C. ptxas cannot keep its
192 sums and its operands in 255 registers, so that code spills. This file replaces the function's code with a
loop written here: 192 sums, 40 operand registers (every operand in one register quad, loaded once), four
addresses, a counter, the transcript's address, the flags and the readout's chain: 244 registers of the 246 the
compiled function writes.

What a tile is (pearl_t96.cu has the contract): 64 k-steps of k 32, each its low k16 half then its high one. A
half is 96 IMMA.8816, row block by row block: sum[i][j] += a[i] * b[j], i 0 .. 11, j 0 .. 7. After every fourth
k-step the XOR of the 192 sums is stored as the chunk's transcript word.

Loads. A quad is four fragments of ONE half (four row blocks of A, or four column blocks of B), 16 bytes a
lane. Quad q of half h of k-step s is at <base> + 0x10000 q + 0x400 s + 0x200 h. A's quads 0 and 1 of a half are
free after their 32 IMMA and are loaded again 8 IMMA later for the next k-step; A's quad 2 and B's two quads are
free at the half's end and are loaded 8 to 16 IMMA into the other half.

From one tile to the next. The last k-step's loads ahead read the NEXT tile's first k-step, so a tile does not
start by waiting for seven loads from the L2 or from DRAM: they land while the warp hashes or waits at the
tile's barrier. The kernel says which tile is next: before the call it stores, in the thread's own transcript
rows 13, 14 and 15 (free until the loop writes them), the next tile's offsets in A' and B' and two flags: 1,
nothing was loaded for this tile (a block's first: the entry loads it), and 2, there is a next tile. The seven
quads loaded ahead stay in registers across the kernel's own code between two calls, so they are registers
that code never names (checked in facts()). With no next tile the last k-step loads nothing ahead: a k-step
after it would be past the end of B'.

The code, in order:
  entry  the three words from shared memory, the four addresses (64-bit: the operands' bases from the kernel's
         parameters plus the thread's offsets), the sums' zeroes, and for a block's first tile the seven loads
         k-step 0 needs first.
  P      a k-step. Counts, steps the addresses, then: to L when 63 k-steps are done, to P again inside a
         chunk, on into Q at a chunk's start.
  Q      a chunk's first k-step: P with the readout of the chunk before in its low half. Each sum pair is
         XORed into the chain just before the IMMA that next writes it (it was last written 96 IMMA earlier),
         and the word is stored early in the high half.
  L      the last k-step: its loads ahead read the next tile (when there is one), and the last chunk's readout
         runs after the tile's last IMMA has landed.
  The compiled function's last word (the RET) is kept, and L ends by branching to it.

Rules the card taught (benchmark.md, 20 November, goal/f-tu-asm), all kept here:
  - an ALU result is read no sooner than 4 cycles later, a predicate by a branch no sooner than 12;
  - a load has a read scoreboard, and whatever next writes its address register waits on it;
  - the way out waits on every scoreboard but the loads' own: what is still in flight then lands in
    registers nothing else writes before the next call, whose first IMMA of each half waits for them;
  - the last IMMA of a k-step have a write scoreboard, which the way out waits on.

Checks before a file is written: what the loop takes for granted of the compiled kernel (where the arguments
are, the registers and predicates the compiled function writes, the room, no word the driver may rewrite, the
constants in t96_shape); the loop run on a model of a warp, instruction by instruction, against the sums
worked out directly (--model, also run by the build when numpy is there); and every word read back through
nvdisasm and compared with the text asked for. Standard library only (numpy for the model).
"""
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sm75patch  # noqa: E402

SEC = 'pearl_t96_fold'
LOOP_NAME = 'pearl_t96_loop'
RZ = 255
CTRL_SHIFT = 41
K, PLANE, TRB, SMEM = 2048, 0x10000, 2048, 12 * 2048
KSTEPS = K // 32
# Where the compiled kernel hands the loop its arguments, and where the loop returns to. ptxas chooses these, so
# facts() reads them from the listing (bind()); the values here are one compile's, for --model.
R_TR, R_BO, R_AO = 0, 251, 250
RET_TO = 0xa40
C_AP, C_BP = 0x160, 0x168
# The transcript rows the kernel leaves the next tile's offsets and the flags in (byte offsets from R_TR).
S_AO2, S_BO2, S_FL = 13 * 128, 14 * 128, 15 * 128
# The loop's registers. Everything else the caller may hold a value in across the call.
RA, RB, RA2, RB2 = 2, 4, 6, 8       # the addresses of A' and B' for this k-step, and the next tile's (pairs)
R_T, R_CNT, R_TRP, R_FL = 10, 11, 12, 13   # the readout's chain, k-steps done, the next word's address, flags
R_T2 = 246                          # the readout's second chain (key rch=2): the pair no sum uses
# A_Q[half][quad], B_Q[half][quad]. The seven the last k-step loads ahead for the next tile are KEEP: they hold
# their values from one call to the next, so they are registers the kernel's own code never names (R52 up).
A_Q = [[52, 56, 60], [72, 76, 16]]
B_Q = [[64, 68], [20, 24]]
KEEP = list(range(52, 80))
PAIRS = [14] + list(range(28, 52, 2)) + list(range(80, 248, 2))
ACC = [[PAIRS[8 * i + j] for j in range(8)] for i in range(12)]
# Four predicates the compiled function writes (bind()): inside a chunk (L: a next tile); not yet 63 k-steps;
# the addresses' two carries.
P_IN, P_LAST, P_C0, P_C1 = 0, 2, 3, 4
SB_IMMA, SB_RD = 0, 1               # the k-step's last IMMA; loads' and the store's reads
OPT = {}                            # tuning keys (key=value on the command line), see opt()


def opt(name, default):
    return int(OPT.get(name, default))


def bind(tr, bo, ao, ret_to, preds):
    """Sets what ptxas chose: the arguments' registers, the address the loop returns to, four predicates."""
    global R_TR, R_BO, R_AO, RET_TO, P_IN, P_LAST, P_C0, P_C1
    R_TR, R_BO, R_AO, RET_TO = tr, bo, ao, ret_to
    P_IN, P_LAST, P_C0, P_C1 = preds


def ctrl(stall=1, yld=True, wsb=None, rsb=None, wait=(), reuse=0):
    c = (stall & 15) | ((1 if yld else 0) << 4) | ((7 if wsb is None else wsb) << 5) | \
        ((7 if rsb is None else rsb) << 8) | (reuse << 17)
    for w in wait:
        c |= 1 << (11 + w)
    return c


def sections(b):
    shoff, = struct.unpack_from('<Q', b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', b, 0x3A)
    sh = [struct.unpack_from('<IIQQQQIIQQ', b, shoff + i * shentsize) for i in range(shnum)]
    stro = sh[shstrndx][4]
    return {b[stro + s[0]:b.index(b'\0', stro + s[0])].decode(): (s[4], s[5]) for s in sh}


def listing(text, sec):
    """([(addr, text)], {label: addr}) of one section of an nvdisasm -c listing."""
    t = text.split('\n')
    i = [k for k, l in enumerate(t) if l.split()[:1] == ['.section'] and '.text.' + sec + ',' in l][0]
    out, labels, pend = [], {}, []
    for l in t[i + 1:]:
        if l.split()[:1] == ['.section']:
            break
        m = re.match(r'\s*(\.L_x_\d+|\$[\w$]+):', l)
        if m:
            pend.append(m[1])
        m = re.search(r'/\*([0-9a-f]{4,6})\*/\s+(.*?)\s*;', l)
        if m:
            a = int(m[1], 16)
            out.append((a, re.sub(r'\s+', ' ', m[2].strip())))
            for p in pend:
                labels[p] = a
            pend = []
    return out, labels


def regs(s):
    return [int(x) for x in re.findall(r'\bR(\d+)\b', s)]


def writes(txt):
    """(registers, predicates) one instruction of the compiled function writes. Only the forms that function
    has are known; any other raises."""
    t = re.sub(r'^@!?P\d\s+', '', txt)
    op, _, rest = t.partition(' ')
    ops = [o.strip() for o in re.split(r',\s*(?![^\[]*\])', rest)] if rest else []
    base = op.split('.')[0]
    if base in ('STL', 'STS', 'WARPSYNC', 'BRA', 'RET', 'NOP'):
        return [], []
    if base == 'ISETP':
        return [], [int(ops[0][1])]
    d = regs(ops[0])[0]
    if base in ('IMMA', 'CS2R'):
        return [d, d + 1], []
    if base == 'LDG':
        assert '.128' in op
        return [d, d + 1, d + 2, d + 3], []
    if base in ('LDL', 'IMAD', 'LOP3', 'IADD3', 'MOV'):
        ps = [int(o[1]) for o in ops[1:3] if re.fullmatch(r'P\d', o)] if base == 'IADD3' and '.X' not in op else []
        return [d], ps
    raise ValueError('t96gen does not know what "%s" writes' % txt)


# ------------------------------------------------------------------------------------------------ the instructions
# An item is a dict: lo, hi (the word, without its control field), text (as nvdisasm prints it), c (control field
# or, for an IMMA, None: assemble() sets its stall by what follows), and sem (what the model runs).
def ins(lo, hi, text, c, sem):
    return {'lo': lo, 'hi': hi, 'text': text, 'c': c, 'sem': sem}


def imma(dst, a, b, reuse, wsb=None, wait=()):
    it = ins(0x7237 | dst << 16 | a << 24 | b << 32, 0x5400 | dst,
             'IMMA.8816.S8.S8 R%d, R%d%s.ROW, R%d.COL, R%d' % (dst, a, '.reuse' if reuse else '', b, dst), None,
             ('imma', dst, a, b))
    it['imma'] = {'wsb': wsb, 'wait': wait, 'reuse': 1 if reuse else 0}
    return it


def ldg(q, base, off, wsb, el=False, p=None):
    """A quad from global memory; el: with L1 evict-last, as the release loads B'; p: only when predicate p."""
    o = '' if off == 0 else '+0x%x' % off
    return ins(0x0381 | (7 if p is None else p) << 12 | q << 16 | base << 24 | (off & 0xFFFFFF) << 40,
               0x2E6D00 if el else 0x1E6D00,
               '%sLDG.E.%s128.CONSTANT.SYS R%d, [R%d%s]' % ('' if p is None else '@P%d ' % p, 'EL.' if el else '',
                                                           q, base, o),
               ctrl(opt('lst', 2), True, wsb, SB_RD), ('ldg', q, base, off, p))


def lds(dst, base, off, wsb):
    return ins(0x7984 | dst << 16 | base << 24 | off << 40, 0x1800,
               'LDS.U R%d, [R%d+0x%x]' % (dst, base, off), ctrl(2, True, wsb, SB_RD), ('lds', dst, base, off))


def lop3(dst, a, b, c, stall=3, wait=()):
    return ins(0x7212 | dst << 16 | a << 24 | b << 32, 0x078E9600 | c,
               'LOP3.LUT R%d, R%d, R%d, %s, 0x96, !PT' % (dst, a, b, 'RZ' if c == RZ else 'R%d' % c),
               ctrl(stall, False, wait=wait), ('xor3', dst, a, b, c))


def lop3p(p, a, imm):
    return ins(0x7812 | 0xFF << 16 | a << 24 | imm << 32, 0x0780C0FF | p << 17,
               'LOP3.LUT P%d, RZ, R%d, 0x%x, RZ, 0xc0, !PT' % (p, a, imm), ctrl(1, bool(opt('ay', 1))), ('pand', p, a, imm))


def isetp_ne(p, a, imm):
    return ins(0x780C | a << 24 | imm << 32, 0x03F05270 | p << 17,
               'ISETP.NE.AND P%d, PT, R%d, 0x%x, PT' % (p, a, imm), ctrl(1, bool(opt('ay', 1))), ('pne', p, a, imm))


def cs2r(dst):
    return ins(0x7805 | dst << 16, 0x1FF00, 'CS2R R%d, SRZ' % dst, ctrl(1, True), ('zero2', dst))


def sts(base, src):
    return ins(0x7388 | base << 24 | src << 32, 0x800, 'STS [R%d], R%d' % (base, src), ctrl(2, True, rsb=SB_RD),
               ('sts', base, src))


def iadd3(dst, a, imm, carry=None, wait=()):
    hi = 0x07FFE0FF if carry is None else 0x0700E0FF | (0x81 | carry[0] << 1 | carry[1] << 4) << 16
    return ins(0x7810 | dst << 16 | a << 24 | (imm & 0xFFFFFFFF) << 32, hi,
               'IADD3 R%d, %sR%d, 0x%x, RZ' % (dst, '' if carry is None else 'P%d, P%d, ' % carry, a, imm),
               ctrl(opt('ast', 2), bool(opt('ay', 1)), wait=wait), ('add', dst, a, imm, carry))


def iadd3_c(dst, a, coff, carry, wait=()):
    """dst = a + c[0x0][coff], with the two carries out."""
    return ins(0x7A10 | dst << 16 | a << 24 | (coff >> 2) << 40, 0x0700E0FF | (0x81 | carry[0] << 1 | carry[1] << 4) << 16,
               'IADD3 R%d, P%d, P%d, R%d, c[0x0][0x%x], RZ' % (dst, carry[0], carry[1], a, coff),
               ctrl(2, True, wait=wait), ('addc', dst, a, coff, carry))


def iaddx_c(dst, coff, carry):
    """dst = c[0x0][coff + 4] + the two carries: the high word of the address whose low word is at coff."""
    return ins(0x7A10 | dst << 16 | 0xFF << 24 | ((coff + 4) >> 2) << 40, 0x007E04FF | carry[0] << 23 | carry[1] << 13,
               'IADD3.X R%d, RZ, c[0x0][0x%x], RZ, P%d, P%d' % (dst, coff + 4, carry[0], carry[1]), ctrl(2, True),
               ('addxc', dst, coff, carry))


def iaddx(dst, a, carry):
    """dst = a + the two carries."""
    return ins(0x7210 | dst << 16 | a << 24 | 0xFF << 32, 0x007E04FF | carry[0] << 23 | carry[1] << 13,
               'IADD3.X R%d, R%d, RZ, RZ, P%d, P%d' % (dst, a, carry[0], carry[1]), ctrl(opt('ast', 2), bool(opt('ay', 1))),
               ('addx', dst, a, carry))


def mov(dst, src):
    return ins(0x000000FFFF007224 | dst << 16, 0x078E0000 | src, 'IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (dst, src),
               ctrl(2, True), ('mov', dst, src))


def movi(dst, imm, wait=()):
    return ins(0xFF007424 | dst << 16 | imm << 32, 0x078E00FF, 'IMAD.MOV.U32 R%d, RZ, RZ, 0x%x' % (dst, imm),
               ctrl(2, True, wait=wait), ('movi', dst, imm))


def warpsync():
    return ins(0xFFFFFFFF00007948, 0x03800000, 'WARPSYNC 0xffffffff', ctrl(4, True), ('nop',))


def bra(label, p=None, neg=False, wait=()):
    return {'bra': label, 'p': p, 'neg': neg, 'wait': wait}


def label(name):
    return {'label': name}


# ------------------------------------------------------------------------------------------------ the loop
def kstep(kind):
    """One k-step. kind: 'P', 'Q' (the chunk before's readout in the low half) or 'L' (the last: its loads ahead
    read the next tile's first k-step when there is one, and the last chunk's readout runs behind the high
    half)."""
    out = []
    ahead = kind != 'L'
    el = bool(opt('el', 1))

    def lda(h, q, nxt):
        if nxt and not ahead:
            assert A_Q[h][q] in KEEP
            out.append(ldg(A_Q[h][q], RA2, 0x10000 * q + 0x200 * h, 2 + h, p=P_IN))
        else:
            out.append(ldg(A_Q[h][q], RA, 0x10000 * q + 0x200 * h + (0x400 if nxt else 0), 2 + h))

    def ldb(h, q, nxt):
        if nxt and not ahead:
            assert B_Q[h][q] in KEEP
            out.append(ldg(B_Q[h][q], RB2, 0x10000 * q + 0x200 * h, 4 + h, el=el, p=P_IN))
        else:
            out.append(ldg(B_Q[h][q], RB, 0x10000 * q + 0x200 * h + (0x400 if nxt else 0), 4 + h, el=el))

    for h in (0, 1):
        for n in range(96):
            i, j = n // 8, n % 8
            x = ACC[i][j]
            # Key dro=1 (timing only): the same XORs in every other k-step too, into the chain nobody reads then.
            if (kind == 'Q' or (kind == 'P' and opt('dro', 0))) and h == 0:
                # Keys rch and rst: the XORs in one chain or two (R_T and R_T2, joined early in the high half),
                # and their stall.
                ch = [R_T, R_T2][:opt('rch', 1)]
                out.append(lop3(ch[n % len(ch)], x, x + 1, RZ if n < len(ch) else ch[n % len(ch)],
                                stall=opt('rst', 3), wait=(SB_RD,) if n == 0 else ()))
            out.append(imma(x, A_Q[h][i // 4] + i % 4, B_Q[h][j // 4] + j % 4, reuse=j != 7,
                            wsb=SB_IMMA if h == 1 and n >= opt('sb0', 80) else None, wait=(2 + h, 4 + h) if n == 0 else ()))
            # The other half's last quads, 8 to 16 IMMA into this one: for this k-step's high half, or the next
            # k-step's low half.
            if n == opt('la2', 7):
                lda(1 - h, 2, h == 1)
            if n == opt('lb0', 11):
                ldb(1 - h, 0, h == 1)
            if n == opt('lb1', 15):
                ldb(1 - h, 1, h == 1)
            # This half's first two quads of A, 8 IMMA after their last row, for the next k-step.
            if n == opt('la0', 39):
                lda(h, 0, True)
            if n == opt('la1', 71):
                lda(h, 1, True)
            # L: whether there is a next tile, well before the first load that asks.
            if kind == 'L' and h == 0 and n == 3:
                out.append(lop3p(P_IN, R_FL, 2))
            if h == 1 and kind == 'Q':
                if n == 1 and opt('rch', 1) == 2:
                    out.append(lop3(R_T, R_T, R_T2, RZ, stall=2))
                if n == 4:
                    out.append(sts(R_TRP, R_T))
                if n == 24:
                    out.append(iadd3(R_TRP, R_TRP, 0x80, wait=(SB_RD,)))
            if h == 1 and ahead:
                if n == 63:
                    out.append(iadd3(R_CNT, R_CNT, 1))
                # The addresses step once every load of this k-step has gone out and read its register.
                st = opt('as', 72)
                if n == st:
                    out.append(iadd3(RA, RA, 0x400, carry=(P_C0, P_C1), wait=(SB_RD,)))
                if n == st + 2:
                    out.append(iaddx(RA + 1, RA + 1, (P_C0, P_C1)))
                if n == st + 4:
                    out.append(iadd3(RB, RB, 0x400, carry=(P_C0, P_C1)))
                if n == st + 6:
                    out.append(iaddx(RB + 1, RB + 1, (P_C0, P_C1)))
                if n == opt('pn', 79) and kind == 'P':
                    out.append(lop3p(P_IN, R_CNT, 3))
                    out.append(isetp_ne(P_LAST, R_CNT, KSTEPS - 1))
    if kind == 'L':
        # The last chunk's readout, after the tile's last IMMA. The first XOR waits for the scoreboarded IMMA
        # (the k-step's last 16; the unit finishes them in order, so every earlier one has landed too) and for
        # the last store's read of the chain. Until 20 November these XORs ran inside the high half, each 12
        # IMMA behind its pair's last write with no wait: on an RTX 2060 a warp's last word then came out
        # wrong now and then (the startup check: 32 words wrong).
        # Four chains, so that a chain's next XOR is 4 cycles after its last: R_T and three registers of A's
        # last quad, which no IMMA reads any more.
        ch = [R_T, A_Q[1][2], A_Q[1][2] + 1, A_Q[1][2] + 2]
        for n in range(96):
            y = ACC[n // 8][n % 8]
            out.append(lop3(ch[n % 4], y, y + 1, RZ if n < 4 else ch[n % 4], stall=1 if n < 95 else 4,
                            wait=(SB_IMMA, SB_RD) if n == 0 else ()))
        out.append(lop3(R_T, ch[0], ch[1], ch[2], stall=4))
        out.append(lop3(R_T, R_T, ch[3], RZ, stall=12))
        out.append(sts(R_TRP, R_T))
        # Keys dum and dus (timing only): dum times 80 XORs of dead registers at stall dus behind the tile, eight
        # chains in turn, to price work a warp does between two tiles (a hash, say) by its count and its stalls.
        if opt('dum', 0):
            sc = [A_Q[1][2] + i for i in range(4)] + [B_Q[1][0] + i for i in range(4)]
            out.append(movi(R_CNT, 0))
            out.append(label('D'))
            for n in range(80):
                out.append(lop3(sc[n % 8], sc[n % 8], sc[n % 8], RZ, stall=opt('dus', 1)))
                if n == 20:
                    out.append(iadd3(R_CNT, R_CNT, 1))
                if n == 40:
                    out.append(isetp_ne(P_LAST, R_CNT, opt('dum', 0)))
            out.append(bra('D', P_LAST))
    return out


def program():
    el = bool(opt('el', 1))
    zero = [cs2r(ACC[i][j]) for i in range(12) for j in range(8)]
    prog = [lds(R_FL, R_TR, S_FL, SB_IMMA), lds(RA2, R_TR, S_AO2, SB_IMMA), lds(RB2, R_TR, S_BO2, SB_IMMA),
            iadd3_c(RA, R_AO, C_AP, (P_IN, P_LAST)), iadd3_c(RB, R_BO, C_BP, (P_C0, P_C1)),
            mov(R_TRP, R_TR), movi(R_CNT, 0),
            iaddx_c(RA + 1, C_AP, (P_IN, P_LAST)), iaddx_c(RB + 1, C_BP, (P_C0, P_C1))]
    prog += zero[:4]
    prog += [iadd3_c(RA2, RA2, C_AP, (P_IN, P_LAST), wait=(SB_IMMA,)), iadd3_c(RB2, RB2, C_BP, (P_C0, P_C1))]
    prog += zero[4:8]
    prog += [iaddx_c(RA2 + 1, C_AP, (P_IN, P_LAST)), iaddx_c(RB2 + 1, C_BP, (P_C0, P_C1))]
    prog += zero[8:16]
    prog += [lop3p(P_IN, R_FL, 1)]                 # nothing was loaded for this tile: the entry loads it
    prog += zero[16:]
    prog += [bra('GO', P_IN, neg=True)]
    prog += [ldg(A_Q[0][q], RA, 0x10000 * q, 2) for q in range(3)]
    prog += [ldg(B_Q[0][q], RB, 0x10000 * q, 4, el=el) for q in range(2)]
    prog += [ldg(A_Q[1][q], RA, 0x10000 * q + 0x200, 3) for q in range(2)]
    prog += [label('GO'), warpsync()]
    prog += [label('P')] + kstep('P') + [bra('L', P_LAST, neg=True), bra('P', P_IN)]
    prog += [label('Q')] + kstep('Q') + [bra('P')]
    # The way out waits for the last IMMA and for every load's and the store's reads. Not for the loads
    # themselves: those are the next tile's, into KEEP.
    # The return address goes in R2 and R3, where the compiled function's RET reads it (ptxas sets them in
    # more than one way, so the loop sets both itself); the loads that read them as A's address have read
    # them by then (the wait).
    prog += [label('L')] + kstep('L') + [movi(2, RET_TO, wait=(SB_RD,)), movi(3, 0),
                                         bra('RET', wait=(SB_IMMA, SB_RD))]
    return prog


def used(prog):
    """(registers written, predicates written) by the loop."""
    rw, pw = set(), set()
    for it in prog:
        s = it.get('sem')
        if not s:
            continue
        k = s[0]
        if k in ('imma', 'zero2'):
            rw |= {s[1], s[1] + 1}
        elif k == 'ldg':
            rw |= set(range(s[1], s[1] + 4))
        elif k in ('xor3', 'mov', 'movi', 'addx', 'addxc', 'lds'):
            rw.add(s[1])
        elif k in ('add', 'addc'):
            rw.add(s[1])
            pw |= set(s[4] or ())
        elif k in ('pand', 'pne'):
            pw.add(s[1])
    return rw, pw


def assemble(prog, entry, ret_at):
    """[(lo, hi, text)] from the function's first word; branch targets are absolute addresses in the text."""
    items = [it for it in prog if 'label' not in it]
    labels, n = {'RET': (ret_at - entry) // 16}, 0
    for it in prog:
        if 'label' in it:
            labels[it['label']] = n
        else:
            n += 1
    words = []
    for idx, it in enumerate(items):
        nxt = items[idx + 1] if idx + 1 < len(items) else None
        if 'bra' in it:
            rel = (labels[it['bra']] - (idx + 1)) * 16
            p = 7 if it['p'] is None else (8 | it['p']) if it['neg'] else it['p']
            lo = 0x0947 | p << 12 | (rel & 0xFFFFFFFF) << 32
            hi = 0x03800000 | (0x3FFFF if rel < 0 else 0) | ctrl(5, True, wait=it['wait']) << CTRL_SHIFT
            pt = '' if it['p'] is None else '@%sP%d ' % ('!' if it['neg'] else '', it['p'])
            words.append((lo, hi, '%sBRA 0x%x' % (pt, entry + labels[it['bra']] * 16)))
            continue
        c = it['c']
        if c is None:
            m = it['imma']
            nx_imma = nxt is not None and 'imma' in nxt
            nx_ro = nxt is not None and nxt.get('sem', ('',))[0] == 'xor3'
            c = ctrl(opt('ist', 4) if nx_imma else 1 if nx_ro else 2, bool(opt('iy', 1)), m['wsb'], wait=m['wait'],
                     reuse=m['reuse'])
        words.append((it['lo'], it['hi'] | c << CTRL_SHIFT, it['text']))
    return words


# ------------------------------------------------------------------------------------------------ the model
def model(prog, seed=1):
    """Runs the loop on a model of one warp (32 lanes, the instructions' meanings, loads at once) over random
    operands laid out as t96_aux_mat lays them out, for two tiles one after the other, as a block's first and
    last: the first loads its own first quads and the second's ahead; the second finds them in KEEP. Between
    the two calls every register but KEEP and the arguments is forgotten, which is more than the kernel's own
    code does. Compares the 16 transcript words of every lane of both tiles with the sums worked out directly
    and returns the count of wrong words; raises if the loop reads a register it never wrote or loads from
    outside an operand. The operands' bases are put so that every address carries into its high word."""
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, size=(192, K), dtype=np.int64)
    b = rng.integers(-128, 128, size=(128, K), dtype=np.int64)

    def lay(mat):
        rows = mat.shape[0]
        out = np.zeros(rows // 32 * PLANE, dtype=np.int64)
        r = np.arange(rows)[:, None]
        kk = np.arange(K)[None, :]
        at = (r >> 5) * PLANE + (kk >> 4) * 512 + ((r & 7) * 4 + ((kk >> 2) & 3)) * 16 + ((r >> 3) & 3) * 4 + (kk & 3)
        out[at] = mat
        return out
    lane = np.arange(32, dtype=np.int64)
    base = {C_AP: 0x7FFFF8000, C_BP: 0x3FFFF4000}
    mem = {C_AP: lay(a), C_BP: lay(b)}
    M32 = 0xFFFFFFFF
    items = [it for it in prog if 'label' not in it]
    labels, n = {}, 0
    for it in prog:
        if 'label' in it:
            labels[it['label']] = n
        else:
            n += 1
    smem = {}
    R = {}
    bad = 0
    for call in (0, 1):
        ao, bo = lane * 16 + call * 3 * PLANE, lane * 16 + call * 2 * PLANE
        tr = 0x1000 + call * 0x2000 + lane * 4
        R = {r: v for r, v in R.items() if r in KEEP}
        R.update({R_AO: ao, R_BO: bo, R_TR: tr, RZ: np.zeros(32, dtype=np.int64)})
        for ln in range(32):
            # The first tile: nothing loaded for it, and the second is next. The second: no next (its own offsets).
            smem[int(tr[ln]) + S_AO2] = int(lane[ln]) * 16 + 3 * PLANE
            smem[int(tr[ln]) + S_BO2] = int(lane[ln]) * 16 + 2 * PLANE
            smem[int(tr[ln]) + S_FL] = 3 if call == 0 else 0
        P = {}
        pc, steps, immas = 0, 0, 0

        def rd(r):
            if r not in R:
                raise AssertionError('the loop reads R%d before writing it (word %d, call %d)' % (r, pc, call))
            return R[r]

        def frag(v):                                # a lane's word as its four signed bytes
            v = v & M32
            out = np.stack([(v >> (8 * i)) & 0xFF for i in range(4)], axis=1)
            return np.where(out > 127, out - 256, out)
        while True:
            steps += 1
            assert steps < 200000, 'the loop does not end'
            it = items[pc]
            if 'bra' in it:
                take = True
                if it['p'] is not None:
                    take = P[it['p']] != it['neg']
                if take:
                    if it['bra'] == 'RET':
                        break
                    pc = labels[it['bra']]
                else:
                    pc += 1
                continue
            s = it['sem']
            k = s[0]
            if k == 'imma':
                am8 = frag(rd(s[2])).reshape(8, 16)                       # row g, k 4 t + byte
                bm8 = frag(rd(s[3])).reshape(8, 16)                       # column g, k 4 t + byte
                d = am8 @ bm8.T                                           # [row][column]
                c0, c1 = rd(s[1]), rd(s[1] + 1)
                g, t = lane >> 2, lane & 3
                R[s[1]] = (c0 + d[g, 2 * t]) & M32
                R[s[1] + 1] = (c1 + d[g, 2 * t + 1]) & M32
                immas += 1
            elif k == 'ldg':
                if s[4] is None or P[s[4]]:
                    cb = R[('base', s[2])]
                    addr = ((rd(s[2] + 1) << 32) | rd(s[2])) + s[3] - base[cb]
                    buf = mem[cb]
                    assert (addr >= 0).all() and (addr + 16 <= len(buf)).all(), \
                        'a load outside its operand (word %d, call %d)' % (pc, call)
                    for i in range(4):
                        v = np.zeros(32, dtype=np.int64)
                        for byte in range(4):
                            v |= (buf[addr + 4 * i + byte] & 0xFF) << (8 * byte)
                        R[s[1] + i] = v
            elif k == 'lds':
                R[s[1]] = np.array([smem[int(x) + s[3]] for x in rd(s[2])], dtype=np.int64)
            elif k == 'xor3':
                R[s[1]] = rd(s[2]) ^ rd(s[3]) ^ rd(s[4])
            elif k == 'zero2':
                R[s[1]] = np.zeros(32, dtype=np.int64)
                R[s[1] + 1] = np.zeros(32, dtype=np.int64)
            elif k == 'sts':
                for ln in range(32):
                    smem[int(rd(s[1])[ln])] = int(rd(s[2])[ln])
            elif k in ('add', 'addc'):
                other = s[3] if k == 'add' else base[s[3]] & M32
                v = rd(s[2]) + other
                if s[4]:
                    assert (v >> 32).all() == (v >> 32).any(), 'lanes disagree on a carry: the model keeps one flag'
                    P[s[4][0]] = bool((v >> 32).any())
                    P[s[4][1]] = False
                R[s[1]] = v & M32
                if k == 'addc':
                    R[('base', s[1])] = s[3]
            elif k in ('addx', 'addxc'):
                other = rd(s[2]) if k == 'addx' else base[s[2]] >> 32
                R[s[1]] = (other + int(P[s[3][0]]) + int(P[s[3][1]])) & M32
            elif k == 'mov':
                R[s[1]] = rd(s[2]).copy()
            elif k == 'movi':
                R[s[1]] = np.full(32, s[2], dtype=np.int64)
            elif k == 'pand':
                P[s[1]] = bool((rd(s[2])[0] & s[3]) != 0)
            elif k == 'pne':
                P[s[1]] = bool(rd(s[2])[0] != s[3])
            pc += 1
        assert immas == KSTEPS * 192, 'the loop ran %d IMMA, a tile is %d' % (immas, KSTEPS * 192)
        for ln in range(32):
            g, t = ln >> 2, ln & 3
            rows = [96 * call + g + 8 * i for i in range(12)]
            cols = [64 * call + 2 * t + e + 8 * j for j in range(8) for e in (0, 1)]
            acc = np.zeros((12, 16), dtype=np.int64)
            for c in range(16):
                ks = slice(128 * c, 128 * c + 128)
                acc += a[rows][:, ks] @ b[cols][:, ks].T
                w = 0
                for v in (acc & M32).flatten():
                    w ^= int(v)
                if smem.get(int(tr[ln]) + c * 128) != w:
                    bad += 1
    return bad


# ------------------------------------------------------------------------------------------------ the build
class Refuse(Exception):
    pass


def facts(b, text):
    """Reads what ptxas chose from the listing (bind()) and checks what the loop takes for granted of the
    compiled kernel. Returns (section offset, entry, ret_at, the program): the function's first word and the
    first of its last two (the return), as offsets in the section."""
    rows, labels = listing(text, SEC)
    at = dict(rows)
    fn = [l for l in labels if l.startswith('$') and LOOP_NAME in l]
    if len(fn) != 1:
        raise Refuse('%d functions named %s in the kernel, not 1' % (len(fn), LOOP_NAME))
    entry = labels[fn[0]]
    rets = [a for a, t in rows if a > entry and t.startswith('RET.REL.NODEC')]
    if len(rets) != 1:
        raise Refuse('the function does not end in one RET')
    ret = rets[0]
    if not at[ret].startswith('RET.REL.NODEC R2 '):
        raise Refuse('the function does not return through R2 as expected')
    calls = [a for a, t in rows if t.startswith('CALL.REL.NOINC') and LOOP_NAME in t]
    if len(calls) != 1 or calls[0] >= entry:
        raise Refuse('the kernel does not call the function from one place')
    ret_to = calls[0] + 16
    if not any(re.fullmatch(r'MOV R\d+, 0x%x' % ret_to, t) for _a, t in rows):
        raise Refuse('no return address 0x%x in the kernel: the call is not the form the hand loop returns from' % ret_to)
    body = [(a, t) for a, t in rows if entry <= a <= ret]
    txt = '\n'.join(t for _a, t in body)
    found = []
    for what, pat in (('the transcript\'s address', r'IMAD R\d+, R\d+, 0x80, R(\d+)\b'),
                      ('B\'s base and offset', r'IADD3 R\d+, P\d, P\d, R\d+(?:\.reuse)?, c\[0x0\]\[0x%x\], R(\d+)' % C_BP),
                      ('A\'s base and offset', r'IADD3 R\d+, P\d, P\d, R\d+(?:\.reuse)?, c\[0x0\]\[0x%x\], R(\d+)' % C_AP)):
        got = set(re.findall(pat, txt))
        if len(got) != 1:
            raise Refuse('the compiled function does not take %s in one register' % what)
        found.append(int(got.pop()))
    rw, pw = set(), set()
    for _a, t in body:
        r, p = writes(t)
        rw |= set(r)
        pw |= set(p)
    if len(pw) < 4:
        raise Refuse('the compiled function writes %d predicates, the hand loop needs four' % len(pw))
    bind(found[0], found[1], found[2], ret_to, sorted(pw)[:4])
    prog = program()
    mr, mp = used(prog)
    if mr - rw:
        raise Refuse('the hand loop writes registers the compiled function does not: %s' % sorted(mr - rw))
    if mp - pw:
        raise Refuse('the hand loop writes predicates the compiled function does not: %s' % sorted(mp - pw))
    if {R_TR, R_BO, R_AO} & mr or len({R_TR, R_BO, R_AO}) != 3:
        raise Refuse('an argument\'s register is written by the hand loop')
    # KEEP holds the next tile's first operands from one call to the next: the kernel's own code must not name
    # those registers at all (a wide instruction names up to four from its first).
    named = set()
    for a, t in rows:
        if not entry <= a <= ret:
            for r in regs(t):
                named |= set(range(r, r + 4))
    if named & set(KEEP):
        raise Refuse('the kernel\'s own code names registers the hand loop keeps between calls: %s'
                     % sorted(named & set(KEEP)))
    secs = sections(b)
    off, size = secs['.text.' + SEC]
    sm75patch.SECTION = ('.text.' + SEC).encode()
    _o, _s, fold, hs = sm75patch.fold_section(b)
    for where, what in sm75patch.load_time_rewrites(b, hs, fold):
        if entry - 16 < where < ret + 16:
            raise Refuse('the driver may rewrite 0x%x (%s), inside the function' % (where, what))
    g = secs.get('.nv.global.init')
    shape = struct.unpack_from('<4I', b, g[0]) if g and g[1] == 16 else None
    if shape != (K, PLANE, TRB, SMEM):
        raise Refuse('t96_shape is %s, the hand loop is written for %s' % (shape, (K, PLANE, TRB, SMEM)))
    print('t96gen: arguments in R%d (transcript), R%d (A), R%d (B); returns to 0x%x; predicates %s'
          % (R_TR, R_AO, R_BO, RET_TO, ' '.join('P%d' % x for x in (P_IN, P_LAST, P_C0, P_C1))))
    return off, entry, ret, prog


def build(base, nvdisasm, out):
    b = bytearray(open(base, 'rb').read())
    r = subprocess.run([nvdisasm, '-c', base], capture_output=True)
    if r.returncode != 0:
        raise Refuse('nvdisasm cannot read the compiled kernel')
    off, entry, ret_at, prog = facts(b, r.stdout.decode('utf-8', 'replace'))
    words = assemble(prog, entry, ret_at)
    room = (ret_at - entry) // 16
    if len(words) > room:
        raise Refuse('the loop is %d words and the function has room for %d' % (len(words), room))
    hot = sum(1 for it in prog[prog.index({'label': 'P'}):prog.index({'label': 'L'})] if 'label' not in it)
    try:
        import numpy  # noqa: F401
        bad = model(prog)
        if bad:
            raise Refuse('the loop on the model of a warp gets %d transcript words wrong' % bad)
        print('t96gen: the loop on the model of a warp, two tiles: 1024 of 1024 transcript words right')
    except ImportError:
        print('t96gen: no numpy here, the model of a warp was not run')
    nop = (0x7918, 0, 'NOP')
    words = words + [nop] * (room - len(words))
    for i, (lo, hi, _t) in enumerate(words):
        struct.pack_into('<QQ', b, off + entry + 16 * i, lo, hi)
    tmp = out + '.rb'
    open(tmp, 'wb').write(b)
    r = subprocess.run([nvdisasm, '-c', tmp], capture_output=True)
    os.remove(tmp)
    if r.returncode != 0:
        raise Refuse('nvdisasm cannot read the patched kernel: ' + r.stderr.decode('utf-8', 'replace')[:200])
    rows, labels = listing(r.stdout.decode('utf-8', 'replace'), SEC)
    got = dict(rows)
    lab_at = {v: k for k, v in labels.items()}
    bad = 0
    for i, (_lo, _hi, want) in enumerate(words):
        a = entry + 16 * i
        g = got.get(a, '<nothing>')
        if 'BRA 0x' in want:
            target = int(want.split('BRA ')[1], 16)
            m = re.search(r'`\((\.L_x_\d+)\)', g)
            ok = m is not None and labels.get(m[1]) == target and g[:g.index('BRA')] == want[:want.index('BRA')]
        else:
            ok = g == want
        if not ok:
            bad += 1
            if bad <= 12:
                print('t96gen: %04x: asked "%s", nvdisasm reads "%s"' % (a, want, g))
    del lab_at
    if bad:
        raise Refuse('%d words do not read back as asked' % bad)
    open(out, 'wb').write(b)
    print('t96gen: %d words of %d used, the two k-steps of the loop are %d (%.1f KB); every word read back'
          % (len([w for w in words if w[2] != 'NOP']), room, hot, hot * 16 / 1024))


def main():
    args = [a for a in sys.argv[1:] if '=' not in a]
    OPT.update(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
    if args == ['--model']:
        bad = model(program())
        print('t96gen: model, %d transcript words wrong of 1024' % bad)
        return 1 if bad else 0
    if len(args) != 3:
        print(__doc__)
        return 2
    try:
        build(*args)
    except Refuse as e:
        print('t96gen: FAILED: %s' % e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
