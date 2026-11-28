"""full75.py: the sm_75 kernel's body from the first round's layer to the last S-boxes, written by hand.

  python full75.py <in.cubin built with -DQC_HAND75=2> <out.cubin> [key=value ...]

It replaces qc_hand_loopall's body (qc_kernel.cuh) with these blocks, each scheduled on its own:

  ent     the elements into their registers, the counters; on to S
  Ppro    partial round 0: S-box, the two half sums, element 0's product
  Ploop   partial rounds 1..21, the diagonal products of the round before riding beside each S-box (sm89/hand.py)
  Pepi    the last round's eleven products with the terminal constants
  S, Sx   twelve S-boxes in place; leave after the last ones
  E, Ex   the external layer with its constants, then the step counter: on to S, again to E (the first
          permutation's last layer is followed by the second one's first), or to the partial rounds (after
          layers 3 and 12)
  exit    the results into the registers the caller reads, and the return address

The arithmetic is gen75.py's. The external layer keeps every sum as three words (two 32-bit halves and a count) and
adds up to three of them a step with two-carry IADD3s; an output is folded as (w0 - w2) + (w1 + w2) f, exact but
for the high word passing 2^32 (odds about 2^-27 an output, as the compiled kernel's own fold).

Registers: the twelve elements keep the registers they arrive in; between an element's last read and its next
write in a block its registers are lent to that block's temporaries. The interface is read from the markers
qc_mark leaves in the compiled function. Every instruction is encoded from a ptxas instruction of the same shape,
read back through nvdisasm, and the patched function is simulated from the cubin against ref_all().

Options (key=value): patch75.py's lat, plat, stall, ybit, limit; sqmul=1 squares as four-product multiplies;
plimit=N the scheduler's cap on live carry predicates (4); sbatch=N S-boxes scheduled together (3); imx=N bit 1 the sums' single carries, 2 the layer's two-input counts, 4 the layer's high-word folds as IMAD.X
(multiplier pipe), 8 the layer's other counts, 16 its outputs' counts and 32 sum2's counts as pairs of IMAD.X; anch=N anchors, so a multiply's high product takes the middle sum's high word as its addend;
sum2=1 the partial rounds' sum in 18 ops; kf=1 (with sum2) the partial rounds' constant as a thirteenth value of that
sum, with the tables in constant bank 3 rewritten to match (kf_consts); kt=1 (with kf) the terminal constants too,
through the layer before the partial rounds; dl=1 (with those) the sum's low word joins a diagonal product in its
fold, not as an addend; ep=N the partial rounds' counter and compare N instructions ahead of the branch; sm=1 schedule_smooth's order; ylast=1 the S-box's output last in the
partial rounds' sum; bank=1 registers chosen by bank (allocate); reuse=1 operand reuse flags where the next instruction
reads the same register in the same slot, rpair=N moving such an instruction up to N places (reuse_pairs); nmul=1 and
nsq=1 the multiply and square with a negated WIDE addend (muln, sqrn), xpa=1 their x1 + pt on the ALU; wf=1 each
layer output folded by one WIDE into its home pair; hx=N products' high-word corrections as IMAD.X (hxon); fr=1 (a
kernel built with -DQC_HAND75=3, qc_hand_all) two more blocks, F before the steps (the first round's two S-boxes and
layer, from the two moving elements and the line's constants in constant bank 0, into the homes) and Z after them
(element 0 of the last layer into the registers the caller reads), so ent and exit copy nothing; Sanch, Panch and
the like: an option for one kind of block (ProgAll.o).
"""
import os
import random
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'sm89'))
sys.path.insert(0, HERE)
import enc  # noqa: E402
import gen  # noqa: E402
import hand  # noqa: E402
import iface as IF  # noqa: E402
import patch as P89  # noqa: E402
import ref  # noqa: E402
import sass  # noqa: E402
import gen75  # noqa: E402
import patch75  # noqa: E402

hx = gen.hx

OPT = patch75.OPT
OPT.update(sqmul=0, imx=0, plimit=4, sbatch=3, anch=0, ldcs=1, slow=0, bra=5, land=9, lend=1, ulat=1, dmx=0, sq2=0,
           sum2=0, kf=0, sm=0, smc=2, ylast=0, kt=0, dl=0, ep=0, reuse=0, bank=0, rpair=0, nmul=0, nsq=0, xpa=0, wf=0, hx=0, fr=0, sq3=0)
P = ref.P
RC_SEQ = [ref._arr('QC_RC_SEQ')[12 * i:12 * i + 12] for i in range(9)]
RC_P22 = ref._arr('QC_RC_P22')
assert len(RC_P22) == 22 and len(RC_SEQ[8]) == 12


# ------------------------------------------------------------------ what the function computes
def ext(s, rc):
    y = [0] * 12
    for g in (0, 4, 8):
        a, b, c, d = s[g:g + 4]
        y[g:g + 4] = [2 * a + 3 * b + c + d, a + 2 * b + 3 * c + d, a + b + 2 * c + 3 * d, 3 * a + b + c + 2 * d]
    return [(y[g + k] + y[k] + y[k + 4] + y[k + 8] + rc[g + k]) % P for g in (0, 4, 8) for k in range(4)]


def ref_first(xa, xb, c):
    """qc_first_layer(sbox(xa), sbox(xb), c)."""
    a, b = ref.sbox(xa % P), ref.sbox(xb % P)
    v2 = a + b
    v = [v2 + a, v2 + b, v2]
    out = []
    for g in range(3):
        out += [(2 * v[g] + c[4 * g]) % P, (v[g] + c[4 * g + 1]) % P, (v[g] + c[4 * g + 2]) % P,
                (3 * v[g] + c[4 * g + 3]) % P]
    return out


def ref_ext0(s):
    """qc_ext0: element 0 of the layer without constants."""
    m = [2 * s[4 * g] + 3 * s[4 * g + 1] + s[4 * g + 2] + s[4 * g + 3] for g in range(3)]
    return (2 * m[0] + m[1] + m[2]) % P


def ref_all(s):
    s = [v % P for v in s]
    for k in range(1, 17):
        j = k - 9 if k >= 9 else k
        if k > 1:
            s = ext(s, RC_SEQ[j])
        if j == 4:
            x = s[0]
            for r in range(22):
                x = ref.partial(s, ref.sbox(x), RC_P22[r])
            s[0] = x
            for i in range(1, 12):
                s[i] = (s[i] + ref.RC_T0[i]) % P
        if j != 8:
            s = [ref.sbox(v) for v in s]
    return s


# ------------------------------------------------------------------ instructions sass.py does not parse
class Ldc:
    """LDC.64 Rd, c[0x3][Rb+off]."""
    def __init__(self, text):
        m = re.fullmatch(r'LDC\.64 R(\d+), c\[0x3\]\[R(\d+)(?:\+0x([0-9a-f]+))?\]', text)
        self.text, self.op, self.pred, self.target = text, 'LDC.64', None, None
        self.d, self.b, self.off = int(m[1]), int(m[2]), int(m[3] or '0', 16)
        self.dregs, self.dpreds, self.uregs, self.upreds = [self.d, self.d + 1], [], [self.b], []
        self.srcs = []


def mkop(text):
    t = re.sub(r'\s+', ' ', text.strip().rstrip(';').strip())
    if t.startswith('LDC.64'):
        return Ldc(t)
    return sass.Op(t)


class Machine(sass.Machine):
    def run(self, op):
        if isinstance(op, Ldc):
            a = self.r[op.b] + op.off
            self.r[op.d], self.r[op.d + 1] = self.const[(3, a)], self.const[(3, a + 4)]
            return None
        if op.op == 'MOV' and op.pred is None:
            self.setr(op.args[0].n, self.get(op.args[1]))
            return None
        return super().run(op)


# ------------------------------------------------------------------ the interface, from qc_mark's immediates
def find_iface(cubin):
    body, labels, entry, ret, retreg = IF.function(cubin)
    ops = []
    for r in body:
        try:
            ops.append(mkop(r['text']))
        except Exception:
            ops.append(None)
    nin = 2 if OPT['fr'] else 12
    inp = [[None, None] for _ in range(nin)]
    origin, zreg = {}, None
    for op in ops:                                  # the entry: copies and the input markers
        if op is None and OPT['fr']:
            continue
        if op is None or op.op.startswith(('BRA', 'U')):
            break
        if op.op in ('IMAD.MOV.U32', 'MOV') and op.args[-1].kind == 'r':
            origin[op.args[0].n] = origin.get(op.args[-1].n, op.args[-1].n)
            continue
        if op.op == 'IMAD' and op.args[2].kind == 'i' and 0x1001 <= op.args[2].n < 0x1200:
            k = op.args[2].n - 0x1001
            inp[(k % 0x100) // 2][k // 0x100] = origin.get(op.args[3].n, op.args[3].n)
            zreg = origin.get(op.args[1].n, op.args[1].n)
            origin[op.args[0].n] = op.args[0].n     # a new value
            continue
        for d in op.dregs:
            origin[d] = d
    assert all(x is not None for p in inp for x in p), inp
    # the exit: from the first output marker to the RET, values followed through the copies
    first = next(i for i, op in enumerate(ops) if op is not None and op.op == 'IMAD' and op.args[2].kind == 'i'
                 and 0x2001 <= op.args[2].n < 0x2200)
    tok, tail = {}, ops[first:]
    for op in tail:
        assert op is not None and op.pred is None, 'the exit is not straight-line'
        if op.op.startswith('RET'):
            break
        if op.op == 'IMAD' and op.args[2].kind == 'i' and 0x2001 <= op.args[2].n < 0x2200:
            k = op.args[2].n - 0x2001
            tok[op.args[0].n] = ((k % 0x100) // 2, k // 0x100)
        elif op.op in ('IMAD.MOV.U32', 'MOV') and op.args[-1].kind == 'r':
            if op.args[-1].n in tok:
                tok[op.args[0].n] = tok[op.args[-1].n]
            else:
                tok.pop(op.args[0].n, None)
        else:
            for d in op.dregs:
                tok.pop(d, None)
    out = [[[], []] for _ in range(1 if OPT['fr'] else 12)]
    for r, (i, h) in tok.items():
        out[i][h].append(r)
    assert all(o[0] and o[1] for o in out), out
    # the return address: the instructions that make the RET's pair, back to immediates or live-in registers
    need, retfix, keep = {retreg, retreg + 1}, [], set()
    for op in reversed(tail):
        if op.op.startswith('RET'):
            continue
        if set(op.dregs) & need:
            assert op.op in ('IMAD.MOV.U32', 'MOV') and len(op.dregs) == 1, op.text
            retfix.insert(0, op.text)
            need -= set(op.dregs)
            need |= set(op.uregs)
    # ptxas may load the return address as an immediate anywhere in the function; what the exit still needs is
    # either such a load (repeated here) or a register that is live on entry (kept)
    for rr in sorted(need):
        defs = [op for op in ops[:first] if op is not None and rr in op.dregs]
        if defs and defs[-1].op == 'MOV' and defs[-1].args[1].kind == 'i' and defs[-1].pred is None:
            retfix.insert(0, defs[-1].text)
        else:
            assert not defs, 'R%d, which the return needs, is computed in the function' % rr
            keep.add(rr)
    regs_w, preds_w, urs = set(), set(), set()
    for r, op in zip(body, ops):
        if op is not None:
            regs_w.update(op.dregs)
            preds_w.update(op.dpreds)
        m = re.match(r'(?:@!?P\d )?U[A-Z0-9.]+ UR(\d+)', r['text'])
        if m:
            urs.add(int(m[1]))
            if '.64' in r['text'].split(' ')[0] or 'WIDE' in r['text']:
                urs.add(int(m[1]) + 1)
    cbase = None
    if OPT['fr']:
        # fr=1: the line's constants are the kernel's parameter (QcLaunch.c at its offset 0), which ptxas reads
        # from constant bank 0 in this function, as it does z (QcLaunch.zero, at offset 0x88)
        zl = [r['text'] for r in body
              if re.fullmatch(r'(IMAD\.MOV\.U32|MOV) R%d, (RZ, RZ, )?c\[0x0\]\[0x[0-9a-f]+\]' % zreg, r['text'])]
        assert zl, 'fr=1: z is not read from the parameter block'
        cbase = int(re.search(r'c\[0x0\]\[0x([0-9a-f]+)\]', zl[0])[1], 16) - 0x88
        seen = set(int(x, 16) for r in body for x in re.findall(r'c\[0x0\]\[0x([0-9a-f]+)\]', r['text']))
        assert {cbase + 8 * i for i in range(12)} & seen, 'fr=1: no constant read at 0x%x' % cbase
    return dict(inp=inp, out=out, entry=entry, ret=ret, retreg=retreg, retfix=retfix, keep=sorted(keep),
                clob=sorted(regs_w), cpreds=sorted(preds_w), urs=sorted(urs), n=len(body), cbase=cbase)


# ------------------------------------------------------------------ the program
class ProgAll(gen75.Prog75):
    sub = 0
    cnt_out = None

    def fold(self, L0, L1, M0, u, T, pa, pb, out=None, **kw):
        if self.cnt_out is None or out is None:
            if self.hxon():
                return self.fold_hx(L0, L1, M0, u, T, pa, pb, out=out, **kw)
            return super().fold(L0, L1, M0, u, T, pa, pb, out=out, **kw)
        assert not kw
        # sq3=1: a full round's S-box hands the layer its output as three words. The fold stops after its two
        # sums: lo1 + hi1 f + c f^2 - 2 f, with the high sum's carries c counted into a third word (one add
        # where the fold's two corrections were). The layer takes the - 2 f off (ext).
        k1, k2, c1, c2 = self.pred(), self.pred(), self.pred(), self.pred()
        lo, hi = out
        ca = ('!' + pa) if pa else 'PT'
        cb = ('!' + pb) if pb else 'PT'
        if T is None:
            self.emit('IADD3.X %s, %s, %s, %s, -0x1, ~%s, %s, %s' % (lo, k1, k2, L0, u, ca, cb))
        else:
            self.emit('IADD3.X %s, %s, %s, %s, ~%s, ~%s, %s, %s' % (lo, k1, k2, L0, u, T, ca, cb))
        self.emit('IADD3.X %s, %s, %s, %s, %s, %s, %s, %s' % (hi, c1, c2, L1, M0, u, k1, k2))
        self.emit('IADD3.X %s, RZ, RZ, RZ, %s, %s' % (self.cnt_out, c1, c2))
        return lo, hi

    prod = 0

    def hxon(self):
        """hx=N: bit k set, product k's high-word correction (hi' - 2 + c1 + c2) as two IMAD.X on the multiplier
        pipe, -2 from a register: k = 0..3 the S-box's x^2, x^3, x^4, x^7, 4 a diagonal product. One ALU op fewer
        and one instruction more a product; for blocks where the multiplier has room (Shx, Phx)."""
        return bool(self.o('hx') >> self.prod & 1)

    def hicorr(self, hi, hi1, c1, c2, m):
        t = '%' + self.tmp('n')
        self.emit('IMAD.X %s, %s, 0x1, %s, %s' % (t, hi1, m, c1))
        self.emit('IMAD.X %s, %s, 0x1, RZ, %s' % (hi, t, c2))

    def fold_hx(self, L0, L1, M0, u, T, pa, pb, out=None, plus=None):
        k1, k2, c1, c2 = self.pred(), self.pred(), self.pred(), self.pred()
        lo1, hi1 = '%' + self.tmp('f'), '%' + self.tmp('f')
        lo, hi = out if out else ('%' + self.tmp('o'), '%' + self.tmp('o'))
        ca = ('!' + pa) if pa else 'PT'
        cb = ('!' + pb) if pb else 'PT'
        if plus is not None:
            assert T is None
            self.emit('IADD3.X %s, %s, %s, %s, %s, ~%s, %s, %s' % (lo1, k1, k2, L0, plus, u, ca, cb))
        elif T is None:
            self.emit('IADD3.X %s, %s, %s, %s, -0x1, ~%s, %s, %s' % (lo1, k1, k2, L0, u, ca, cb))
        else:
            self.emit('IADD3.X %s, %s, %s, %s, ~%s, ~%s, %s, %s' % (lo1, k1, k2, L0, u, T, ca, cb))
        self.emit('IADD3.X %s, %s, %s, %s, %s, %s, %s, %s' % (hi1, c1, c2, L1, M0, u, k1, k2))
        self.hicorr(hi, hi1, c1, c2, '%m2' if plus is None else '%m1')
        self.emit('IADD3.X %s, %s, %s, RZ, !%s, !%s' % (lo, lo1, '-0x2' if plus is None else '-0x3', c1, c2))
        return lo, hi

    def o(self, key):
        """An option, or its value for this kind of block when one is given: Sanch=0 turns the anchors off in the
        S block only, Panch=6 sets them for the partial rounds (S, P, E: the block's first letter)."""
        return OPT.get(self.block[:1] + key, OPT[key])

    def emit(self, text):
        # sub: the scheduler orders one sub-block at a time. Given twelve independent S-boxes at once it starts
        # them all, and their products do not fit the registers.
        super().emit(text)
        self.ops[-1].sub = self.sub

    def foldn(self, l0, l1, m0, m1, t, p, out=None):
        """The fold of nmul / nsq: the value is [l0 - m1 + 1 - t - p] + [l1 + m0 + m1 - 1 + t] f (t the low
        chain's no-borrow, p the middle chain's last carry). The low bracket lies in [-2^32, 2^32], so with 2^32
        added it carries once at most (twice only for l0 = 2^32 - 1, m1 = 0: odds 2^-64); the high sum then takes
        t and that carry, and what is left is the usual lo1 + (hi1 - 2) f + c f^2."""
        k1, c1, c2 = self.pred(), self.pred(), self.pred()
        lo1 = '%' + self.tmp('f')
        if self.cnt_out is not None and out is not None:
            lo, hi = out
            self.emit('IADD3.X %s, %s, %s, ~%s, RZ, !%s, !%s' % (lo, k1, l0, m1, t, p))
            self.emit('IADD3.X %s, %s, %s, %s, %s, %s, %s, %s' % (hi, c1, c2, l1, m0, m1, t, k1))
            self.emit('IADD3.X %s, RZ, RZ, RZ, %s, %s' % (self.cnt_out, c1, c2))
            return lo, hi
        hi1 = '%' + self.tmp('f')
        lo, hi = out if out else ('%' + self.tmp('o'), '%' + self.tmp('o'))
        self.emit('IADD3.X %s, %s, %s, ~%s, RZ, !%s, !%s' % (lo1, k1, l0, m1, t, p))
        self.emit('IADD3.X %s, %s, %s, %s, %s, %s, %s, %s' % (hi1, c1, c2, l1, m0, m1, t, k1))
        if self.hxon():
            self.hicorr(hi, hi1, c1, c2, '%m2')
        else:
            self.emit('IADD3.X %s, %s, -0x2, RZ, %s, %s' % (hi, hi1, c1, c2))
        self.emit('IADD3.X %s, %s, -0x2, RZ, !%s, !%s' % (lo, lo1, c1, c2))
        return lo, hi

    def muln(self, a, b, out=None):
        """nmul=1: a b = (L - H) + (M + H) f with H = a1 b1 riding the middle chain and taken off the low product
        by a WIDE with a negated addend (IMAD.WIDE.U32.X ..., ~H, !p1: a0 b0 - H - p1, the middle chain's first
        carry folded in), so no ALU op forms a sum before the fold: 4 WIDE + 4 ALU (mul: 4 + 5)."""
        a0, a1 = a
        b0, b1 = b
        H = self.wide(a1, b1)
        p1 = self.pred()
        M1 = self.wide(a1, b0, H, cout=p1)
        p2 = self.pred()
        M2 = self.wide(a0, b1, M1, cout=p2)
        t = self.pred()
        Lp = self.tmp('w')
        self.pairs.add(Lp)
        self.emit('IMAD.WIDE.U32.X %%%s, %s, %s, %s, ~%%%s, !%s' % (Lp, t, a0, b0, H, p1))
        return self.foldn('%%%s.0' % Lp, '%%%s.1' % Lp, '%%%s.0' % M2, '%%%s.1' % M2, t, p2, out=out)

    def sqrn(self, x, out=None):
        """nsq=1: sqr2's square with H = x1 (x1 + pt) riding the middle product and taken off x0^2 by a negated
        addend (IMAD.WIDE.U32 ..., -H: x0^2 - H, carry out = no borrow): 3 WIDE + 5 ALU (sqr2: 3 + 6)."""
        x0, x1 = x
        b, pt = '%' + self.tmp('b'), self.pred()
        self.emit('IADD3 %s, %s, %s, %s, RZ' % (b, pt, x0, x0))
        xp = '%' + self.tmp('s')
        if self.o('xpa'):
            # xpa=1: x1 + pt as an ALU op (sm_120's cubin has no IMAD.X with an immediate to take the form from)
            self.emit('IADD3.X %s, %s, RZ, RZ, %s, !PT' % (xp, x1, pt))
        else:
            self.emit('IMAD.X %s, %s, 0x1, RZ, %s' % (xp, x1, pt))
        H = self.wide(x1, xp)
        p1 = self.pred()
        M = self.wide(b, x1, H, cout=p1)
        t = self.pred()
        Lp = self.tmp('w')
        self.pairs.add(Lp)
        self.emit('IMAD.WIDE.U32 %%%s, %s, %s, %s, -%%%s' % (Lp, t, x0, x0, H))
        return self.foldn('%%%s.0' % Lp, '%%%s.1' % Lp, '%%%s.0' % M, '%%%s.1' % M, t, p1, out=out)

    def mul(self, a, b, out=None):
        if self.o('nmul'):
            return self.muln(a, b, out=out)
        a0, a1 = a
        b0, b1 = b
        M = self.wide(a1, b0)
        pm = self.pred()
        M2 = self.wide(a0, b1, M, cout=pm)
        L = self.wide(a0, b0)
        if self.o('anch'):
            # The middle sum's high word rides in the high product's addend: it is copied (on the multiplier pipe,
            # which has room) to an anchor, an even register whose odd neighbour holds zero, so the pair is
            # (m1, 0). That saves the ALU op that added it; the ALU is the pipe that binds.
            anc = self.tmp('anc')
            self.emit('IMAD.MOV.U32 %%%s.0, RZ, RZ, %%%s.1' % (anc, M2))
            Hq = self.wide(a1, b1, anc)
            return self.fold('%%%s.0' % L, '%%%s.1' % L, '%%%s.0' % M2, '%%%s.0' % Hq, '%%%s.1' % Hq, pm, None,
                             out=out)
        H = self.wide(a1, b1)
        u, cu = '%' + self.tmp('u'), self.pred()
        self.emit('IADD3 %s, %s, %%%s.0, %%%s.1, RZ' % (u, cu, H, M2))
        return self.fold('%%%s.0' % L, '%%%s.1' % L, '%%%s.0' % M2, u, '%%%s.1' % H, pm, cu, out=out)

    def dmul(self, a, i, L, H, out=None):
        """gen75's product with q5-nv40's carry: X's carry out of 2^64 is f^2 = f - 1, and its + f goes into Y as
        the second product's carry-in (free), so only - 1 is left for the fold. Four ALU ops for every constant."""
        self.prod = 4
        if not OPT['dmx']:
            return super().dmul(a, i, L, H, out=out)
        D, DP = ref.D[i], ref.DP[i]
        d0, d1, e0, e1 = D & 0xFFFFFFFF, D >> 32, DP & 0xFFFFFFFF, DP >> 32
        CX, CY = gen.CXY[i]
        a0, a1 = a
        dl = OPT['dl'] and L == 'L'
        # dl=1: the sum's low word is added in the fold (its free slot) and not as the first product's addend.
        # An addend WIDE holds the ALU longer than a plain one (2.4-3.1 cycles against 2.0, ub75.py). Only for
        # the pair L itself, whose high word is zero with sum2.
        Pp = self.wide(a0, hx(d0)) if dl else self.wide(a0, hx(d0), L)
        pcx = self.pred() if CX else None
        X = self.wide(a1, hx(e0), Pp, cout=pcx) if CX else self.wide(a1, hx(e0), Pp)
        Q = self.wide(a0, hx(d1), H)
        pcy = self.pred() if CY else None
        if CX:
            Y = self.tmp('w')
            self.pairs.add(Y)
            self.emit('IMAD.WIDE.U32.X %%%s, %s%s, %s, %%%s, %s' % (Y, pcy + ', ' if CY else '', a1, hx(e1), Q, pcx))
        else:
            Y = self.wide(a1, hx(e1), Q, cout=pcy) if CY else self.wide(a1, hx(e1), Q)
        return self.fold('%%%s.0' % X, '%%%s.1' % X, '%%%s.0' % Y, '%%%s.1' % Y, None, pcy, pcx, out=out,
                         plus='%L.0' if dl else None)

    def sqr2(self, x):
        """A square in three WIDE with the doubled-out bit folded into a factor: 2 x0 x1 = b x1 + pt x1 f with
        b = 2 x0 mod f, and pt x1 f^2 + x1^2 f^2 = x1 (x1 + pt) f^2. x1 + pt is made on the multiplier pipe; it
        wraps only for x1 = 2^32 - 1 (odds 2^-32, a slip like the fold's)."""
        x0, x1 = x
        b, pt = '%' + self.tmp('b'), self.pred()
        self.emit('IADD3 %s, %s, %s, %s, RZ' % (b, pt, x0, x0))
        xp = '%' + self.tmp('s')
        self.emit('IMAD.X %s, %s, 0x1, RZ, %s' % (xp, x1, pt))
        M = self.wide(b, x1)
        L = self.wide(x0, x0)
        if self.o('anch'):
            anc = self.tmp('anc')
            self.emit('IMAD.MOV.U32 %%%s.0, RZ, RZ, %%%s.1' % (anc, M))
            Hq = self.wide(x1, xp, anc)
            return self.fold('%%%s.0' % L, '%%%s.1' % L, '%%%s.0' % M, '%%%s.0' % Hq, '%%%s.1' % Hq, None, None)
        H = self.wide(x1, xp)
        u, cu = '%' + self.tmp('u'), self.pred()
        self.emit('IADD3 %s, %s, %%%s.0, %%%s.1, RZ' % (u, cu, H, M))
        return self.fold('%%%s.0' % L, '%%%s.1' % L, '%%%s.0' % M, u, '%%%s.1' % H, cu, None)

    def sbox(self, x, out=None, cnt=None):
        sq = (lambda v: self.mul(v, v)) if self.o('sqmul') else self.sqrn if self.o('nsq') else             self.sqr2 if self.o('sq2') else self.sqr
        self.prod = 0
        x2 = sq(x)
        self.prod = 1
        x3 = self.mul(x2, x)
        self.prod = 2
        x4 = sq(x2)
        self.cnt_out = cnt
        self.prod = 3
        try:
            return self.mul(x3, x4, out=out)
        finally:
            self.cnt_out = None
            self.prod = 4

    def sum12(self, terms, S=None):
        if not OPT['imx'] & 1:
            return super().sum12(terms, S)
        hv, gen.HANDV = gen.HANDV, gen.HANDV | 4
        try:
            return super().sum12(terms, S)
        finally:
            gen.HANDV = hv

    # ---- the external layer: a value is ('x', lo, hi) or ('q', w0, w1, w2)
    def sums2(self, lo_terms, hi_terms, k=None):
        """sum2=1 (q5-nv40): the sum of twelve values as L.0 + w H in 18 ALU ops against sum12's 24. L.0 is the
        low words' sum mod 2^32 and its carries go into the high words' sum H as carry-ins, so L's high word stays
        zero (written once on entry) and H is below 13 * 2^32 + 13 (no constant's chain is that close to its
        limit: gen.CXY is the same for 14 * 2^32)."""
        idx = [[0, 1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11]]
        for k_, ix in enumerate(idx):
            ls = [lo_terms[i] for i in ix] if k_ == 0 else ['%L.0'] + [lo_terms[i] for i in ix]
            hs = [hi_terms[i] for i in ix] if k_ == 0 else ['%H.0'] + [hi_terms[i] for i in ix]
            if k is not None and k_ == 5:
                # kf=1: the round's constant is the thirteenth value (a uniform register goes second: enc.py)
                ls, hs = [ls[0], k[0], ls[1]], [hs[0], k[1], hs[1]]
            p1, p2, q1, q2 = self.pred(), self.pred(), self.pred(), self.pred()
            if len(ls) == 3:
                self.emit('IADD3 %%L.0, %s, %s, %s' % (p1, p2, ', '.join(ls)))
                self.emit('IADD3.X %%H.0, %s, %s, %s, %s, %s' % (q1, q2, ', '.join(hs), p1, p2))
                if OPT['imx'] & 32:
                    self.count2('%H.1', [] if k_ == 0 else ['%H.1'], q1, q2)
                    continue
                self.emit('IADD3.X %%H.1, RZ, RZ, %s, %s, %s' % ('RZ' if k_ == 0 else '%H.1', q1, q2))
            else:
                self.emit('IADD3 %%L.0, %s, %s, RZ' % (p1, ', '.join(ls)))
                self.emit('IADD3.X %%H.0, %s, %s, RZ, %s, !PT' % (q1, ', '.join(hs), p1))
                if OPT['imx'] & 1:
                    self.emit('IMAD.X %%H.1, RZ, RZ, %%H.1, %s' % q1)
                else:
                    self.emit('IADD3.X %%H.1, RZ, RZ, %%H.1, %s, !PT' % q1)

    def count2(self, dst, terms, q1, q2):
        """dst = up to three terms + two carries, as two IMAD.X: a count has no carry out, so it can ride the
        multiplier pipe, which idles through the layer and has room in the partial rounds."""
        c = list(terms)
        tmp = '%' + self.tmp('n')
        if len(c) >= 2:
            self.emit('IMAD.X %s, %s, 0x1, %s, %s' % (tmp, c[0], c[1], q1))
        else:
            self.emit('IMAD.X %s, RZ, RZ, %s, %s' % (tmp, c[0] if c else 'RZ', q1))
        if len(c) == 3:
            self.emit('IMAD.X %s, %s, 0x1, %s, %s' % (dst, tmp, c[2], q2))
        else:
            self.emit('IMAD.X %s, RZ, RZ, %s, %s' % (dst, tmp, q2))

    def step(self, *ins, out3=None):
        n = len(ins)
        w = out3 or tuple('%' + self.tmp('q') for _ in range(3))
        cols = [[v[1 + j] if len(v) > 1 + j else None for v in ins] for j in range(3)]
        pad = lambda c: ', '.join((c + ['RZ', 'RZ'])[:3])
        if n == 3:
            p1, p2, q1, q2 = self.pred(), self.pred(), self.pred(), self.pred()
            self.emit('IADD3 %s, %s, %s, %s' % (w[0], p1, p2, pad(cols[0])))
            self.emit('IADD3.X %s, %s, %s, %s, %s, %s' % (w[1], q1, q2, pad(cols[1]), p1, p2))
            if OPT['imx'] & 8:
                self.count2(w[2], [c for c in cols[2] if c], q1, q2)
            else:
                self.emit('IADD3.X %s, %s, %s, %s' % (w[2], pad([c or 'RZ' for c in cols[2]]), q1, q2))
        else:
            p1, q1 = self.pred(), self.pred()
            self.emit('IADD3 %s, %s, %s' % (w[0], p1, pad(cols[0])))
            third = [c for c in cols[2] if c]
            if third and (OPT['imx'] & 8 or (OPT['imx'] & 2 and all(v[0] == 'e' for v in ins))):
                # two inputs and one carry in cannot carry twice: one carry out, and the count is one IMAD.X
                self.emit('IADD3.X %s, %s, %s, %s, !PT' % (w[1], q1, pad(cols[1]), p1))
                if len(third) == 2:
                    self.emit('IMAD.X %s, %s, 0x1, %s, %s' % (w[2], third[0], third[1], q1))
                else:
                    self.emit('IMAD.X %s, RZ, RZ, %s, %s' % (w[2], third[0], q1))
            elif third:
                q2 = self.pred()
                self.emit('IADD3.X %s, %s, %s, %s, %s, !PT' % (w[1], q1, q2, pad(cols[1]), p1))
                self.emit('IADD3.X %s, %s, %s, %s' % (w[2], pad(third), q1, q2))
            else:
                self.emit('IADD3.X %s, %s, %s, %s, !PT' % (w[1], q1, pad(cols[1]), p1))
                if OPT['imx'] & 2:
                    self.emit('IMAD.X %s, RZ, RZ, RZ, %s' % (w[2], q1))
                else:
                    self.emit('IADD3.X %s, RZ, RZ, RZ, %s, !PT' % (w[2], q1))
        return ('q',) + tuple(w)

    def fold2(self, o, dst, off=0):
        """A three-word value ('q', w0, w1, w2) into two words dst: (w0 - w2) + (w1 + w2 - off) f, as ext's fold
        without imx bit 4 (or one WIDE, wf=1, when there is no offset and (w0, w1) is a pair)."""
        if OPT['wf'] and not off and o[1].endswith('.0') and o[2] == o[1][:-1] + '1' and \
                re.fullmatch(r'%s\d+l', dst[0]):
            self.emit('IMAD.WIDE.U32 %s, %s, -0x1, %s' % (dst[0], o[3], o[1][:-2]))
            return
        pb = self.pred()
        self.emit('IADD3 %s, %s, %s, -%s, RZ' % (dst[0], pb, o[1], o[3]))
        self.emit('IADD3.X %s, %s, -0x%x, %s, %s, !PT' % (dst[1], o[2], 1 + off, o[3], pb))

    def first(self, xa, xb, S, cbase):
        """fr=1: the first round's two S-boxes (elements 0 and 4) and its layer (qc_first_layer), into the homes."""
        self.sub = 0
        a = ('x',) + tuple(self.sbox(xa))
        b = ('x',) + tuple(self.sbox(xb))

        def cst(i):
            return ('x', 'c[0x0][0x%x]' % (cbase + 8 * i), 'c[0x0][0x%x]' % (cbase + 8 * i + 4))
        self.sub = 1
        v2 = self.step(a, b)
        v = [self.step(v2, a), self.step(v2, b), v2]
        for g in range(3):
            self.sub = 2 + g
            vg = v[g]
            # a constant goes in the second slot, the one an IADD3 takes it in
            self.fold2(self.step(vg, cst(4 * g), vg), S[4 * g])
            self.fold2(self.step(vg, cst(4 * g + 1)), S[4 * g + 1])
            self.fold2(self.step(vg, cst(4 * g + 2)), S[4 * g + 2])
            self.fold2(self.step(self.step(vg, vg, vg), cst(4 * g + 3)), S[4 * g + 3])
        self.sub = 0

    def ext0(self, S, C, dst):
        """fr=1: element 0 of the last layer, 2 m0 + m1 + m2 with m = 2 x0 + 3 x1 + x2 + x3 (qc_ext0), into dst.
        With sq3 the last S-boxes' outputs are three words, each 2 f high: 56 f in all, taken off in the fold."""
        x = [('e', S[i][0], S[i][1], C[i]) if C else ('x', S[i][0], S[i][1]) for i in range(12)]
        m = []
        for g in range(3):
            self.sub = g
            a = self.step(x[4 * g], x[4 * g + 1])
            t = self.step(x[4 * g + 1], x[4 * g + 2], x[4 * g + 3])
            m.append(self.step(a, a, t))
        self.sub = 3
        o = self.step(self.step(m[0], m[0], m[1]), m[2])
        self.fold2(o, dst, 56 if C else 0)
        self.sub = 0

    def ext(self, S, C=None):
        # C (sq3=1): the elements arrive as three words, each 2 f too high. An output is 28 elements' worth, and the
        # layer's constants in bank 3 are 56 f lower to match (sq3_consts).
        # ('e': an element with its count. Two of them added give one carry, so that step's count is one IMAD.X
        # with imx bit 2, as it was without counts.)
        x = [('e', S[i][0], S[i][1], C[i]) if C else ('x', S[i][0], S[i][1]) for i in range(12)]
        assert not C or OPT['imx'] & 4, 'sq3=1 needs imx bit 4'

        def rc(i):
            # each constant is loaded where it is used, so it holds its two registers only briefly
            name = self.tmp('rc')
            self.pairs.add(name)
            self.emit('LDC.64 %%%s, c[0x3][%%rcp%s]' % (name, '+0x%x' % (8 * i) if i else ''))
            return ('x', '%%%s.0' % name, '%%%s.1' % name)
        y = [None] * 12
        for g in (0, 4, 8):
            self.sub = g // 4
            a, b, c, d = x[g:g + 4]
            t01, t23 = self.step(a, b), self.step(c, d)
            u, w = self.step(t01, t23, b), self.step(t01, t23, d)
            y[g], y[g + 1], y[g + 2], y[g + 3] = self.step(u, t01), self.step(u, c, c), self.step(w, t23), \
                self.step(w, a, a)
        for k in range(4):
            self.sub = 3 + k
            col = self.step(y[k], y[k + 4], y[k + 8])
            for g in (0, 4, 8):
                pb = self.pred()
                if OPT['imx'] & 4:
                    # the count is kept one low (w2 - 1), so the high word's fold w1 + (w2 - 1) + (no borrow) is
                    # one IMAD.X on the multiplier pipe; the low word takes ~(w2 - 1) = -w2
                    # w2 = 0 is the one case this gets wrong (the low word then reports a borrow there is
                    # not, and the high word comes out one low). It needs the sum of an output's terms, 28
                    # elements' worth plus a constant, to stay under 2^64: every S-box output of the round
                    # below about 2^59, odds under 2^-60 a layer. It is a slip like the fold's own, far rarer,
                    # and the host checks every hit.
                    c, r = rc(g + k), y[g + k]
                    p1, p2, q1, q2 = self.pred(), self.pred(), self.pred(), self.pred()
                    if OPT['wf']:
                        # wf=1: the output's fold as one WIDE, w2 (2^32 - 1) + (w0 : w1), written straight into
                        # the element's home pair (2^64 = 2^32 - 1 mod p). The sum passing 2^64 is dropped: odds about
                        # w2 2^-32, the fold's usual slip.
                        wp = self.tmp('wq')
                        self.pairs.add(wp)
                        w2 = '%' + self.tmp('q')
                        self.emit('IADD3 %%%s.0, %s, %s, %s, %s, %s' % (wp, p1, p2, r[1], col[1], c[1]))
                        self.emit('IADD3.X %%%s.1, %s, %s, %s, %s, %s, %s, %s' % (wp, q1, q2, r[2], col[2], c[2],
                                                                            p1, p2))
                        if OPT['imx'] & 16:
                            self.count2(w2, [r[3], col[3]], q1, q2)
                        else:
                            self.emit('IADD3.X %s, %s, %s, RZ, %s, %s' % (w2, r[3], col[3], q1, q2))
                        self.emit('IMAD.WIDE.U32 %s, %s, -0x1, %%%s' % (S[g + k][0], w2, wp))
                        continue
                    w0, w1, w2 = ('%' + self.tmp('q') for _ in range(3))
                    self.emit('IADD3 %s, %s, %s, %s, %s, %s' % (w0, p1, p2, r[1], col[1], c[1]))
                    self.emit('IADD3.X %s, %s, %s, %s, %s, %s, %s, %s' % (w1, q1, q2, r[2], col[2], c[2], p1, p2))
                    if OPT['imx'] & 16:
                        self.count2(w2, [r[3], col[3], '%m1'], q1, q2)
                    else:
                        self.emit('IADD3.X %s, %s, -0x1, %s, %s, %s' % (w2, r[3], col[3], q1, q2))
                    self.emit('IADD3.X %s, %s, %s, ~%s, RZ, !PT, !PT' % (S[g + k][0], pb, w0, w2))
                    self.emit('IMAD.X %s, %s, 0x1, %s, %s' % (S[g + k][1], w2, w1, pb))
                    continue
                o = self.step(y[g + k], col, rc(g + k))
                self.emit('IADD3 %s, %s, %s, -%s, RZ' % (S[g + k][0], pb, o[1], o[3]))
                self.emit('IADD3.X %s, %s, -0x1, %s, %s, !PT' % (S[g + k][1], o[2], o[3], pb))


# ------------------------------------------------------------------ an order that keeps both pipes fed (sm=1)
# What one instruction holds, in cycles of an SM sub-partition (ub75.py on an RTX 3070, 28 Nov 2026): any ALU op
# the ALU for 2, whatever its inputs and carries; a WIDE the multiplier for 4 and the ALU for 2 (2.33 with an
# addend, beside adds); IMAD.X and IMAD.MOV the multiplier for 2. The two pipes run side by side.
def pipe_time(text, unit):
    if unit == 'W':
        return (2.0 if text.rstrip().endswith('RZ') else 2.33), 4.0
    if unit == 'M':
        return 0.0, 2.0
    if unit == 'A' and not text.startswith('LDC'):
        return 2.0, 0.0
    return 0.0, 0.0


def schedule_smooth(ops, carried):
    """sm=1: hand.schedule_block's job with another aim. Six warps share a scheduler, so a warp issues about one
    instruction in thirteen cycles and never waits on its own results; what costs time is a run of WIDEs (the ALU
    idles behind the multiplier) or a run of adds (the multiplier idles). So the next op is the ready one that
    keeps the multiplier's share of the recent work nearest the block's own ratio (smc: how many cycles of lead
    or lag are remembered), inside the same caps on live registers and predicates. Latency is left to timing()."""
    n = len(ops)
    info = [cells(o.text) for o in ops]
    preds = [set() for _ in range(n)]
    last_def, readers = {}, {}
    for i, (op, d, u) in enumerate(info):
        for c in u:
            base = c.split('.')[0]
            for key in (c, base + '.0', base + '.1') if '.' not in c and not c.startswith('$') else (c,):
                if key in last_def:
                    preds[i].add(last_def[key])
            readers.setdefault(c, []).append(i)
        for c in d:
            if c in last_def:
                preds[i].add(last_def[c])               # write after write
            for r in readers.get(c, []):
                if r != i:
                    preds[i].add(r)                     # write after read
            base = c.split('.')[0]
            for r in readers.get(base, []):
                if r != i:
                    preds[i].add(r)
            last_def[c] = i
            readers[c] = []
            if '.' in c:
                readers[base] = []
    tail = [i for i in range(n) if ops[i].unit == 'B' or ops[i].text.startswith(('ISETP', 'UIADD3', 'IADD3 %k'))]
    for i in tail:
        for j in range(n):
            if j not in tail:
                preds[i].add(j)
    cost = [pipe_time(o.text, o.unit) for o in ops]

    def grp(c):
        return c if c.startswith('$') else c.split('.')[0]
    uses_left, size = {}, {}
    for i, (op, d, u) in enumerate(info):
        for gn in set(grp(c) for c in u):
            uses_left[gn] = uses_left.get(gn, 0) + 1
        for c in d:
            gn = grp(c)
            if gn not in carried and gn.lstrip('$') not in carried:
                size[gn] = 2 if ('.' in c) else 1
    ta, tm = sum(c[0] for c in cost), sum(c[1] for c in cost)
    rho = tm / ta if ta else 0.0
    live, done, order = set(), [False] * n, []
    remaining = set(range(n))
    debt, cap = 0.0, float(OPT['smc'])      # multiplier time ahead of its share, forgotten beyond cap cycles
    while remaining:
        best, bkey = None, None
        for i in remaining:
            if any(not done[p] for p in preds[i]):
                continue
            if ops[i].unit == 'B' and len(remaining) > 1:
                continue
            _op, d_, u_ = info[i]
            dg, ug = set(grp(c) for c in d_), set(grp(c) for c in u_)
            grow = sum(size.get(g, 0) for g in dg if g not in live and g in size and not g.startswith('$'))
            shrink = sum(size.get(g, 0) for g in ug if g in live and uses_left.get(g, 0) == 1 and not g.startswith('$'))
            pg = sum(1 for g in dg if g.startswith('$') and g not in live and g in size)
            ps = sum(1 for g in ug if g.startswith('$') and g in live and uses_left.get(g, 0) == 1)
            press = sum(size[x] for x in live if not x.startswith('$'))
            plive = sum(1 for x in live if x.startswith('$'))
            over = (press + grow - shrink > hand.LIMIT[0]) or (plive + pg - ps > hand.PLIMIT[0])
            da, dm = cost[i]
            dev = abs(debt + dm - rho * da)
            # On a tie the earliest op in program order: that finishes one product before it starts the next, so
            # few values are live and a product's adds run beside the next one's WIDEs.
            key = (over, (grow - shrink) + 4 * (pg - ps) if over else 0, round(dev, 3), i)
            if bkey is None or key < bkey:
                best, bkey = i, key
        i = best
        debt = max(-cap, min(cap, debt + cost[i][1] - rho * cost[i][0]))
        done[i] = True
        remaining.discard(i)
        _op, d_, u_ = info[i]
        for g in set(grp(c) for c in u_):
            uses_left[g] = uses_left.get(g, 0) - 1
            if uses_left[g] <= 0:
                live.discard(g)
        for c in d_:
            if grp(c) in size and uses_left.get(grp(c), 0) > 0:
                live.add(grp(c))
        order.append(ops[i])
    return order


GLOBAL1 = ['cnt', 'rcp', 'k']
GLOBALX = ['m1']                 # -1, only with imx bit 16 (or hx)
GLOBALH = ['m2']                 # -2, only with hx
GLOBAL2 = ['L', 'H']
URP, URD0, URD1 = 6, 4, 5        # the partial rounds' constant pointer and data (what the compiled function uses)


def build(c, cbase=None):
    """c: constant offsets in bank 3 (seq, p22, rct). cbase (fr=1): the parameter block in bank 0."""
    g = ProgAll()
    S = [('%%s%dl' % i, '%%s%dh' % i) for i in range(12)]
    C = ['%%s%dc' % i for i in range(12)] if OPT['sq3'] else None
    g.pairs.update(GLOBAL2)

    def sums(y):
        if OPT['sum2']:
            kk = ('UR%d' % URD0, 'UR%d' % URD1) if OPT['kf'] else None
            lo, hi = [y[0]] + [S[i][0] for i in range(1, 12)], [y[1]] + [S[i][1] for i in range(1, 12)]
            if OPT['ylast']:
                # ylast=1: the S-box's output goes into the sum last, so the other eleven are added as their
                # products finish and the sum does not wait for the S-box, the longest chain of the round
                lo, hi = lo[1:] + lo[:1], hi[1:] + hi[:1]
            g.sums2(lo, hi, kk)
            if kk:
                return 'L', 'H'
            return g.add64('L', 'UR%d' % URD0), g.add64('H', 'UR%d' % URD1)
        g.sum12([y[0]] + [S[i][0] for i in range(1, 12)], 'L')
        g.sum12([y[1]] + [S[i][1] for i in range(1, 12)], 'H')
        return g.add64('L', 'UR%d' % URD0), g.add64('H', 'UR%d' % URD1)

    g.block = 'ent'
    if OPT['imx'] & 16 or any(OPT.get(k) for k in ('hx', 'Shx', 'Phx')):
        g.emit('IADD3 %m1, RZ, -0x1, RZ')       # -1 in a register: an IMAD.X takes no immediate addend
    if any(OPT.get(k) for k in ('hx', 'Shx', 'Phx')):
        g.emit('IADD3 %m2, RZ, -0x2, RZ')
    if OPT['sum2']:
        g.emit('IADD3 %L.1, RZ, RZ, RZ')      # sums2 never writes the low sum's high word
    g.emit('IADD3 %cnt, RZ, 0x0, RZ')
    g.emit('IADD3 %%rcp, RZ, 0x%x, RZ' % (c['seq'] + 2 * 0x60))
    if OPT['fr']:
        g.block = 'F'
        g.first(('%xa0', '%xa1'), ('%xb0', '%xb1'), S, cbase)
        g.sub = 9                                 # after every sub-block of F
        g.emit('BRA .S')
        g.sub = 0
        g.block = 'Z'
        g.ext0(S, C, ('%hlo', '%hhi'))
    g.block = 'Ppro'
    g.emit('UMOV UR%d, 0x%x' % (URP, c['p22']))
    g.emit('ULDC.64 UR%d, c[0x3][UR%d]' % (URD0, URP))
    y = g.sbox(S[0])
    L0, H0 = sums(y)
    g.dmul(y, 0, L0, H0, out=S[0])
    g.emit('IADD3 %k, RZ, 0x1, RZ')
    g.emit('UIADD3 UR%d, UR%d, 0x8, URZ' % (URP, URP))
    g.block = 'Ploop'
    g.emit('ULDC.64 UR%d, c[0x3][UR%d]' % (URD0, URP))
    for i in range(1, 12):
        g.dmul(S[i], i, 'L', 'H', out=S[i])
    y = g.sbox(S[0])
    L0, H0 = sums(y)
    g.dmul(y, 0, L0, H0, out=S[0])
    g.emit('UIADD3 UR%d, UR%d, 0x8, URZ' % (URP, URP))
    g.emit('IADD3 %k, %k, 0x1, RZ')
    g.emit('ISETP.NE.AND $q, PT, %k, 0x16, PT')
    g.emit('@$q BRA .LOOP')
    g.block = 'Pepi'
    for i in range(1, 12):
        g.sub = (i - 1) // 4
        if OPT['kt']:
            g.dmul(S[i], i, 'L', 'H', out=S[i])
            continue
        Li = g.add64('L', 'c[0x3][0x%x]' % (c['rct'] + 8 * i))
        Hi = g.add64('H', 'c[0x3][0x%x]' % (c['rct'] + 8 * i + 4))
        g.dmul(S[i], i, Li, Hi, out=S[i])
    g.sub = 0
    g.block = 'S'
    for i in range(12):
        g.sub = i // OPT['sbatch']
        g.sbox(S[i], out=S[i], cnt=C[i] if C else None)
    g.sub = 0
    g.block = 'Sx'
    g.emit('ISETP.NE.AND $pe, PT, %cnt, 0xf, PT')
    g.emit('@!$pe BRA .EXIT')
    g.block = 'E'
    g.ext(S, C)
    g.sub = 0
    g.block = 'Ex'
    g.emit('IADD3 %cnt, %cnt, 0x1, RZ')
    g.emit('IADD3 %rcp, %rcp, 0x60, RZ')
    g.emit('ISETP.NE.AND $xa, PT, %cnt, 0x3, PT')
    g.emit('ISETP.NE.AND $xb, PT, %cnt, 0xc, PT')
    g.emit('ISETP.NE.AND $xc, PT, %cnt, 0x7, PT')
    g.emit('@!$xa BRA .P')
    g.emit('@!$xb BRA .P')
    g.emit('@$xc BRA .S')
    g.emit('IADD3 %%rcp, RZ, 0x%x, RZ' % c['seq'])
    if C:
        # the one layer that follows a layer (its constants are the table's first row, which sq3_consts leaves
        # alone): its inputs are two words, so their counts are zero
        for i in range(12):
            g.emit('IADD3 %s, RZ, 0x0, RZ' % C[i])
    g.emit('BRA .E')
    return g, S


ORDER = ['ent', 'Ppro', 'Ploop', 'Pepi', 'S', 'Sx', 'E', 'Ex', 'exit']
LABEL = {'.P': 'Ppro', '.LOOP': 'Ploop', '.S': 'S', '.E': 'E', '.EXIT': 'exit'}
FIXED = ('ent', 'Sx', 'Ex', 'exit')
PP = ['Ppro'] + ['Ploop'] * 21 + ['Pepi']
SE = ['S', 'Sx', 'E', 'Ex']
PATH = ['ent'] + SE * 3 + PP + SE * 4 + ['E', 'Ex'] + SE * 4 + PP + SE * 3 + ['S', 'Sx', 'exit']


def layout():
    """fr=1 adds block F (the first round, after ent) and block Z (the last layer's element 0, before exit)."""
    fr = bool(OPT['fr'])
    ORDER[:] = ['ent'] + (['F'] if fr else []) + ['Ppro', 'Ploop', 'Pepi', 'S', 'Sx', 'E', 'Ex'] + \
        (['Z'] if fr else []) + ['exit']
    LABEL['.EXIT'] = 'Z' if fr else 'exit'
    PATH[:] = ['ent'] + (['F'] if fr else []) + SE * 3 + PP + SE * 4 + ['E', 'Ex'] + SE * 4 + PP + SE * 3 + \
        ['S', 'Sx'] + (['Z'] if fr else []) + ['exit']


HOMEPAIR = re.compile(r's(\d+)l\.([01])$')


def cells(text):
    if text.startswith('LDC.64'):
        m = re.match(r'LDC\.64 %(\w+), c\[0x3\]\[%(\w+)', text)
        return 'LDC.64', [m[1] + '.0', m[1] + '.1'], [m[2]]
    op, d, u = _cells(text)
    if op.startswith('IMAD.WIDE') and d and HOMEPAIR.match(d[0]):
        # wf=1: a WIDE written to an element's home, whose two registers are a pair (body_code checks that)
        d = [('s%sl' if HOMEPAIR.match(c)[2] == '0' else 's%sh') % HOMEPAIR.match(c)[1] if HOMEPAIR.match(c) else c
             for c in d]
    return op, d, u


_cells = hand.cells


# ------------------------------------------------------------------ registers
def allocate(blocks, pre, avail, npred=7):
    """blocks: {name: ops in order}. pre: {home cell: register}. Returns {name: register or predicate}."""
    loc = dict(pre)
    pool = [r for r in avail if r not in pre.values()]
    for n in GLOBAL2:
        r = next(r for r in pool if r % 2 == 0 and r + 1 in pool)
        loc[n] = r
        pool.remove(r)
        pool.remove(r + 1)
    hxany = any(OPT.get(k) for k in ('hx', 'Shx', 'Phx'))
    for n in GLOBAL1 + (GLOBALX if OPT['imx'] & 16 or hxany else []) + (GLOBALH if hxany else []):
        r = next(r for r in sorted(pool, key=lambda r: ((r ^ 1) in pool, r)))
        loc[n] = r
        pool.remove(r)
    anchors = []
    for _ in range(max(OPT['anch'], OPT.get('Sanch', 0), OPT.get('Panch', 0))):
        # An anchor is an even register whose odd neighbour holds zero. (A WIDE cannot write its pair to an odd
        # register instead, which would have saved the copy: an RTX 2060 SUPER calls that an illegal instruction.)
        r = next(r for r in pool if r % 2 == 0 and r + 1 in pool)
        anchors.append(r)
        pool.remove(r)
        pool.remove(r + 1)
    loc['__anchors'] = anchors
    homes = set(pre)
    glob = homes | set(GLOBAL1 + GLOBALX + GLOBALH) | set(n + h for n in GLOBAL2 for h in ('.0', '.1')) | set(GLOBAL2)

    def gname(c):
        return c if c.startswith('$') else c.split('.')[0]

    peak = 0
    for name, ops in blocks.items():
        info = [cells(o.text) for o in ops]
        n = len(ops)
        busy = {r: [] for r in pool}
        for h in homes:                             # a home is lent out between its last read and its next write
            reads = [i for i, (_o, d, u) in enumerate(info) if h in u]
            writes = [i for i, (_o, d, u) in enumerate(info) if h in d]
            if h in IO_CELLS and not reads and not writes:
                iv = []                             # fr=1: an input or the result, used only in F or Z
            elif h in IO_CELLS and not reads:
                iv = [(writes[0], n)]
            elif h in IO_CELLS:
                iv = [(0, max(reads))]
            elif h in CNT_CELLS and not writes:
                # a count (sq3) lives from its S-box to its last read in the layer
                iv = [(0, max(reads))] if reads else []
            elif not writes or not OPT['lend']:
                iv = [(0, n)]
            else:
                before = [i for i in reads if i <= writes[0]]
                iv = ([(0, before[-1])] if before else []) + [(writes[0], n)]
            busy.setdefault(pre[h], []).extend(iv)
        pbusy = {p: [] for p in range(npred)}
        abusy = {r: [] for r in anchors}
        groups = {}
        for i, (_o, d, u) in enumerate(info):
            for c in d + u:
                gn = gname(c)
                if gn in glob or c in glob:
                    continue
                if gn not in groups:
                    groups[gn] = [i, i]
                groups[gn][1] = i
        for gn in groups:
            if groups[gn][0] == groups[gn][1]:
                groups[gn][1] += 2

        def free(tbl, r, a, b):
            return all(b < x or a > y for x, y in tbl[r])

        # bank=1: the register file has two banks (register number mod 2), and an instruction that reads two or
        # three registers from one bank waits for the second read (4090: an IADD3 with three even registers 3.55
        # cycles against 2.7; beside a WIDE, two from one bank cost a cycle). A single is put in the bank its
        # readers' other registers use least, counting those already placed (the rest count it when placed).
        rd = {}
        if OPT['bank']:
            for i, o in enumerate(ops):
                srcs = src_cells(o.text)
                for k, (c, pair) in enumerate(srcs):
                    if c is None or pair:
                        continue
                    rd.setdefault(gname(c), []).append((i, k, srcs))

        def reg_of(c):
            gn = gname(c)
            if c in loc:
                return loc[c]
            if gn not in loc or gn.startswith('$') or isinstance(loc[gn], list):
                return None
            return loc[gn] + (1 if c.endswith('.1') else 0)

        def bank_cost(gn, a, b):
            cost = [0, 0]
            for i, k, srcs in rd.get(gn, ()):
                if not (a <= i <= b):
                    continue
                seen = set()
                for j, (c, pair) in enumerate(srcs):
                    if j == k or c is None:
                        continue
                    r = reg_of(c)
                    if r is None:
                        continue
                    seen.add(r)
                    if pair:
                        seen.add(r + 1)
                for par in (0, 1):
                    m = sum(1 for r in seen if r % 2 == par)
                    cost[par] += m * m
            return cost

        for gn in sorted(groups, key=lambda x: (groups[x][0], x)):
            a, b = groups[gn]
            if gn.startswith('$'):
                p = next((p for p in range(npred) if free(pbusy, p, a, b)), None)
                if p is None:
                    raise RuntimeError('out of predicates in %s at %d' % (name, a))
                pbusy[p].append((a, b))
                loc[gn] = p
            elif gn.startswith('anc'):
                r = next((r for r in anchors if free(abusy, r, a, b)), None)
                if r is None:
                    raise RuntimeError('out of anchors in %s at %d' % (name, a))
                abusy[r].append((a, b))
                loc[gn] = r
            elif gn in g_pairs[0]:
                r = next((r for r in sorted(busy) if r % 2 == 0 and r + 1 in busy and free(busy, r, a, b)
                          and free(busy, r + 1, a, b)), None)
                if r is None:
                    raise RuntimeError('out of register pairs in %s at %d (%s)' % (name, a, gn))
                busy[r].append((a, b))
                busy[r + 1].append((a, b))
                loc[gn] = r
            else:
                # a single goes where its pair partner is taken anyway, so whole pairs stay free
                bc = bank_cost(gn, a, b) if OPT['bank'] else [0, 0]
                r = next((r for r in sorted(busy, key=lambda r: (bc[r % 2], (r ^ 1) in busy and free(busy, r ^ 1, a, b), r))
                          if free(busy, r, a, b)), None)
                if r is None:
                    raise RuntimeError('out of registers in %s at %d (%s)' % (name, a, gn))
                busy[r].append((a, b))
                loc[gn] = r
        peak = max(peak, max(r for r in busy if busy[r]) if any(busy.values()) else 0)
    return loc


def src_cells(text):
    """The first three source operands of a symbolic instruction: (cell or None, is a WIDE's pair addend)."""
    t = re.sub(r'^@!?\$\w+ ', '', text)
    op, _, rest = t.partition(' ')
    if not rest or op == 'BRA' or op.startswith(('U', 'LDC', 'ISETP')):
        return []
    args = [a.strip() for a in rest.split(',')]
    nd = 1 + sass.npreds(op, [None] * len(args))
    out = []
    for j, a in enumerate(args[nd:nd + 3]):
        a = a.lstrip('-~!')
        if a.startswith('%'):
            c = a[1:]
            pair = op.startswith('IMAD.WIDE') and j == 2
            out.append(('%' + c if False else c, pair))
        else:
            out.append((None, False))
    return out


g_pairs = [set()]
CNT_CELLS = set()
CNT_REGS = [None]
IO_CELLS = set()


# ------------------------------------------------------------------ timing
def ur_ops(text):
    """(written uniform registers, latency) and read ones."""
    m = re.match(r'U[A-Z0-9.]+ UR(\d+)', text)
    wr, lat = [], 0
    rest = text
    if m:
        k = int(m[1])
        wr = [k, k + 1] if text.startswith('ULDC.64') else [k]
        # 13 after a uniform op: with 6 (sm89/hand.py's figure) the load right after the pointer's UMOV read the
        # old pointer on an RTX 2060 SUPER; doubling both figures made it right
        lat = (30 if text.startswith('ULDC') else 13) * OPT['ulat']
        rest = text.split(',', 1)[1] if ',' in text else ''
    return wr, lat, [int(x) for x in re.findall(r'UR(\d+)', rest)]


def timing(code, idx):
    """code: [text, unit, block]. Returns stalls and wait masks / write scoreboards (for the LDC loads)."""
    n = len(code)
    ops = [None if 'BRA' in c[0] else mkop(c[0]) for c in code]
    stalls, wait, wsb = [1] * n, [0] * n, [7] * n
    for name, ix in idx.items():                    # the loads' scoreboards, within their block
        ldc, k = {}, 0
        for i in ix:
            op = ops[i]
            if op is None:
                wait[i] |= 0x3f if any(isinstance(ops[j], Ldc) for j in ix) else 0
                continue
            for r in op.uregs:
                if r in ldc:
                    wait[i] |= 1 << ldc[r]
            for r in op.dregs:
                if r in ldc and not isinstance(op, Ldc):
                    wait[i] |= 1 << ldc[r]
                    del ldc[r]
            if isinstance(op, Ldc):
                wsb[i] = k % 6
                ldc[op.d] = ldc[op.d + 1] = k % 6
                k += 1
        if name == 'Ex':
            for i in ix[:2]:
                wait[i] |= 0x3f
    plat = OPT.get('plat') or max(5, OPT['lat'])
    path = [i for b in PATH for i in idx[b]]
    for _pass in range(200):
        t, wr, wp, wu, bumps = 0, {}, {}, {}, 0
        for pos, i in enumerate(path):
            need, text, unit = t, code[i][0], code[i][1]
            op = ops[i]
            if op is None:
                m = re.match(r'@!?P(\d) BRA', text)
                if m and int(m.group(1)) in wp:
                    need = max(need, wp[int(m.group(1))][0] + max(13, OPT['lat']))
            else:
                addend = set()
                if op.op.startswith('IMAD.WIDE') and op.srcs[2].kind == 'r' and op.srcs[2].n != 255:
                    addend = {op.srcs[2].n, op.srcs[2].n + 1}
                for r in op.uregs:
                    if r in wr and wr[r][1] != 'L':
                        need = max(need, wr[r][0] + patch75.gen_lat(wr[r][1], unit, r in addend))
                for p in op.upreds:
                    if p in wp:
                        need = max(need, wp[p][0] + plat)
                # write after write: a register or predicate is reused, and the later write must land last (a WIDE's
                # result lands two cycles later than an ALU op's)
                lw = hand.LATW
                for r in op.dregs:
                    if r in wr and wr[r][1] != 'L':
                        need = max(need, wr[r][0] + lw[wr[r][1]] - lw.get(unit, 4) + 1)
                for p in op.dpreds:
                    if p in wp:
                        need = max(need, wp[p][0] + lw[wp[p][1]] - lw.get(unit, 4) + 1)
            uw, ulat, ur = ur_ops(text)
            for x in ur:
                if x in wu:
                    need = max(need, wu[x])
            if need > t:
                add, k = need - t, pos - 1
                while add > 0:
                    assert k >= 0, 'cannot place a wait'
                    take = min(15 - stalls[path[k]], add)
                    stalls[path[k]] += take
                    add -= take
                    k -= 1
                bumps += 1
                t = need
            if op is not None:
                for r in op.dregs:
                    wr[r] = (t, 'L' if isinstance(op, Ldc) else unit)
                for p in op.dpreds:
                    wp[p] = (t, unit)
            for x in uw:
                wu[x] = t + ulat
            t += stalls[i]
        # A branch waits at least bra cycles, and a block that ends in one lets what it wrote land first (land
        # cycles), as sm89/hand.py's timing does at its loop's back edge.
        for name, ix in idx.items():
            if not ix:
                continue
            last = ix[-1]
            for i in ix:
                if 'BRA' in code[i][0] and stalls[i] < OPT['bra']:
                    stalls[i] = OPT['bra']
                    bumps += 1
            if 'BRA' in code[last][0] and OPT['land']:
                tail, k = 0, len(ix) - 1
                while k >= 0 and tail < OPT['land']:
                    tail += stalls[ix[k]]
                    k -= 1
                # ix[k + 1 ..] cover `tail` cycles; instructions from ix[k + 1] on need land cycles after them
                acc = 0
                for j in range(len(ix) - 1, -1, -1):
                    acc += stalls[ix[j]]
                    if acc >= OPT['land']:
                        break
                    short = OPT['land'] - acc
                    if j < len(ix) - 1 or True:
                        add = min(15 - stalls[last], short)
                        if add > 0:
                            stalls[last] += add
                            acc += add
                            bumps += 1
        if not bumps:
            return stalls, wait, wsb
    raise RuntimeError('timing did not settle')


# ------------------------------------------------------------------ the whole thing
def encode_ldc(text, tmpl):
    op = Ldc(text)
    w = tmpl[0] | tmpl[1] << 64
    w = enc.setf(w, 16, 8, op.d)
    w = enc.setf(w, 24, 8, op.b)
    w = enc.setf(w, 40, 14, op.off >> 2)
    return w & ((1 << 64) - 1), w >> 64


def kf_consts(c):
    """kf=1: the partial rounds add their constant to the sum, so every element gets it, not only element 0. Then
    element i of 1..11 is held with an offset d_i that is known: d_i' = D_i d_i + rc after each round. The table
    holds K_r = rc_r - (d_1 + .. + d_11), which leaves element 0 exact, and the terminal constants take the last
    offsets off. Returns {offset in constant bank 3: u64}."""
    assert OPT['sum2'], 'kf=1 needs sum2=1'
    d, out = [0] * 12, {}
    lo, hi = c['seq'], c['seq'] + 8 * 9 * 12
    if OPT['kt']:
        # kt=1: the elements also enter the partial rounds with offsets e_i, chosen so that the offset left after
        # the last round is the terminal constant itself: d_i(22) = D_i^22 e_i + sum_r D_i^(21-r) rc_r = RC_T0[i].
        # e_i is added by the layer before the partial rounds (row 4 of the layer table, used nowhere else), so
        # the eleven 64-bit adds after the last round are not needed.
        for i in range(1, 12):
            s = 0
            for r in range(22):
                s = (ref.D[i] * s + RC_P22[r]) % P
            d[i] = (ref.RC_T0[i] - s) * pow(pow(ref.D[i], 22, P), P - 2, P) % P
            out[c['seq'] + 8 * (4 * 12 + i)] = (RC_SEQ[4][i] + d[i]) % P
    for r in range(22):
        out[c['p22'] + 8 * r] = (RC_P22[r] - sum(d[1:])) % P
        d = [0] + [(ref.D[i] * d[i] + RC_P22[r]) % P for i in range(1, 12)]
    if OPT['kt']:
        assert all(d[i] == ref.RC_T0[i] for i in range(1, 12))
    else:
        for i in range(1, 12):
            out[c['rct'] + 8 * i] = (ref.RC_T0[i] - d[i]) % P
        assert not any(lo <= o < hi for o in out), 'the patched constants overlap the layer table'
    return out


def sq3_consts(c, other):
    """sq3=1: every layer but the one that follows a layer takes S-box outputs that are 2 f too high (fold), 56 f
    an output. Rows 1..8 of the layer's table are lowered by that, on top of what kt=1 put there (other). Returns
    {offset in constant bank 3: u64}."""
    out = {}
    for j in range(1, 9):
        for i in range(12):
            o = c['seq'] + 8 * (12 * j + i)
            out[o] = (other.get(o, RC_SEQ[j][i]) - (56 << 32)) % P
    return out


CONSTS = [None]


def reuse_bits(code, idx):
    """reuse=1: the operand reuse flags ptxas sets and the hand code left clear. Slot k (a, b, c) of instruction i
    is flagged when instruction i + 1, in the same straight run (no label at it, no branch at either), reads the
    same register in the same slot and i does not write it: the warp then takes it from the reuse cache, not the
    register file (fewer bank reads and less energy; the value is the same one, as nothing writes it in between).
    A WIDE's third slot is a pair, so only a WIDE after a WIDE shares it."""
    import sass
    starts = {v[0] for v in idx.values() if v}
    ops = []
    for x in code:
        t = x[0]
        if 'BRA' in t or t.startswith(('LDC', 'ULDC', 'UMOV', 'UIADD3', 'ISETP')):
            ops.append(None)
            continue
        try:
            ops.append(sass.Op(t))
        except Exception:
            ops.append(None)
    bits = [0] * len(code)
    for i in range(len(code) - 1):
        a, c = ops[i], ops[i + 1]
        if a is None or c is None or (i + 1) in starts or a.pred or c.pred:
            continue
        for k in range(3):
            if k >= len(a.srcs) or k >= len(c.srcs):
                break
            x, y = a.srcs[k], c.srcs[k]
            if x.kind != 'r' or y.kind != 'r' or x.n == 255 or x.n != y.n:
                continue
            wa, wc = 'WIDE' in a.op, 'WIDE' in c.op
            if k == 2 and (wa or wc) and not (wa and wc):
                continue
            regs = {x.n, x.n + 1} if (k == 2 and wa) else {x.n}
            if regs & set(a.dregs):
                continue
            bits[i] |= 1 << k
    return bits


def finish(cubin, out, info, code, cpatch=None):
    """Times, encodes and writes code ([text, unit, block] in ORDER) over the function, and reads it back."""
    idx = {b: [i for i, x in enumerate(code) if x[2] == b] for b in ORDER}
    stalls, wait, wsb = timing(code, idx)
    stalls[-2] = max(stalls[-2], 9)
    stalls[-1] = 5
    entry, ret = info['entry'], info['ret']
    room = (ret - entry) // 16
    assert len(code) <= room, 'the hand code has %d instructions, the function %d' % (len(code), room)
    labs = {k: entry + 16 * idx[v][0] for k, v in LABEL.items()}
    labs['.RET'] = ret
    rows, _labels, _e = IF.listing(cubin)
    T = patch75.Templates([rows])
    ldc_t = next((r['lo'], r['hi'] & ~enc.CTRL_MASK) for r in rows if r['text'].startswith('LDC.64'))
    b = bytearray(open(cubin, 'rb').read())
    if cpatch:
        # nothing outside this function may read constant bank 3 through a register, or at the patched offsets
        for r in rows:
            if entry <= r['addr'] <= ret or 'c[0x3]' not in r['text']:
                continue
            m = re.search(r'c\[0x3\]\[0x([0-9a-f]+)\]', r['text'])
            assert m and int(m[1], 16) & ~7 not in cpatch, 'the kernel reads a patched constant: ' + r['text']
        secs = IF.sections(bytes(b))
        coff = secs[[n for n in secs if n.startswith('.nv.constant3')][0]][0]
        for o, v in cpatch.items():
            struct.pack_into('<Q', b, coff + o, v)
    off, _size = P89.text_section(b)
    base = rows[0]['addr']
    rbits = reuse_bits(code, idx) if OPT['reuse'] else [0] * len(code)
    want = {}
    for i in range(room):
        a = entry + 16 * i
        if i < len(code):
            x = code[i][0]
            if 'BRA' in x:
                lo, hi = P89.bra(x, a, labs)
            elif x.startswith('LDC.64'):
                lo, hi = encode_ldc(x, ldc_t)
            else:
                lo, hi = T.get(x)
            s = max(stalls[i], OPT['stall'], OPT['ldcs'] if x.startswith('LDC') else 0)
            # slow=mask (diagnostic): every instruction of these blocks waits the longest
            bn = code[i][2]
            if OPT['slow'] & ((1 if bn[0] == 'P' else 2 if bn == 'S' else 4 if bn == 'E' else 8)
                           | {'Ppro': 16, 'Ploop': 32, 'Pepi': 64}.get(bn, 0)):
                s = 15
            # No scoreboard but the LDC loads': ptxas gives ULDC none either (checked in the sm_75 and sm_86
            # cubins: every ULDC.64 there has W- and no reader waits), it is timed by its latency (30 above).
            ct = (s & 15) | ((1 if s <= OPT['ybit'] else 0) << 4) | (wsb[i] << 5) | (7 << 8) | (wait[i] << 11) |                 (rbits[i] << 17)
            hi = (hi & ~enc.CTRL_MASK) | (ct << 41)
            want[a] = x
        else:
            lo, hi = 0x7918, ((7 << 5) | (7 << 8)) << 41
        struct.pack_into('<QQ', b, off + a - base, lo, hi)
    open(out, 'wb').write(b)
    # read back
    rows2, labels2, _ = IF.listing(out)
    got = {r['addr']: r['text'].replace('.reuse', '') for r in rows2}
    bad = 0
    for a, x in want.items():
        h = got.get(a, '?')
        if 'BRA' in x:
            m = re.search(r'\((\.L_x_\d+)\)', h)
            ok = bool(m) and labels2.get(m[1]) == labs[x.split()[-1]] and x.split('BRA')[0] == h.split('BRA')[0]
        else:
            ok = h == x
        if not ok:
            bad += 1
            if bad < 8:
                print('MISMATCH %05x asked %r got %r' % (a, x, h))
    assert bad == 0, '%d instructions read back differently' % bad
    per = {bn: (len(idx[bn]), sum(stalls[i] for i in idx[bn])) for bn in ORDER}
    return stalls, per, rows2, labels2


def body_code(cubin):
    """The interface, the constants and the body's code ([text, unit, block]) up to the exit."""
    IF.NVDISASM[0] = os.environ.get('NVDISASM', 'nvdisasm')
    info = find_iface(cubin)
    const = IF.const_bank3(cubin)
    c = dict(seq=patch75.find_u64(const, [v for row in RC_SEQ for v in row]), p22=patch75.find_u64(const, RC_P22),
             rct=patch75.find_u64(const, ref.RC_T0))
    assert {URP, URD0, URD1} <= set(info['urs']), info['urs']
    gen.Prog = ProgAll
    hand.gen_lat = patch75.gen_lat
    hand.cells = cells
    hand.LIMIT[0] = OPT['limit']
    hand.PLIMIT[0] = OPT['plimit']
    CONSTS[0] = c
    assert not OPT['dl'] or (OPT['sum2'] and OPT['kf'] and OPT['kt'] and OPT['dmx']), 'dl=1 needs sum2, kf, kt, dmx'
    layout()
    g, S = build(c, info['cbase'])
    g_pairs[0] = g.pairs
    # The elements arrive in whatever registers ptxas chose (all odd ones here, which would leave few aligned
    # pairs), so they are moved to the top 24 free registers on the way in and to the caller's on the way out.
    avail = [r for r in info['clob'] if r not in info['keep'] and r != 1]
    top = sorted(avail)[-24:]
    home = [(top[2 * i], top[2 * i + 1]) for i in range(12)]
    assert not OPT['wf'] or all(h[0] % 2 == 0 and h[1] == h[0] + 1 for h in home), 'wf=1 needs paired homes'
    pre = {}
    for i in range(12):
        pre[S[i][0][1:]], pre[S[i][1][1:]] = home[i]
    CNT_CELLS.clear()
    CNT_REGS[0] = None
    if OPT['sq3']:
        CNT_REGS[0] = sorted(avail)[-36:-24]
        for i in range(12):
            pre['s%dc' % i] = CNT_REGS[0][i]
            CNT_CELLS.add('s%dc' % i)
    IO_CELLS.clear()
    if OPT['fr']:
        io = {'xa0': info['inp'][0][0], 'xa1': info['inp'][0][1], 'xb0': info['inp'][1][0], 'xb1': info['inp'][1][1],
              'hlo': info['out'][0][0][0], 'hhi': info['out'][0][1][0]}
        assert not set(io.values()) & set(r for h in home for r in h), 'fr=1: an input or the result is in a home'
        pre.update(io)
        IO_CELLS.update(io)
    mv = [] if OPT['fr'] else \
        [(home[i][h], info['inp'][i][h]) for i in range(12) for h in range(2) if home[i][h] != info['inp'][i][h]]
    busy_mv = set(d for d, _ in mv) | set(s_ for _, s_ in mv)
    ent = [gen.V('IMAD.MOV.U32 R%d, RZ, RZ, R%d' % m, 'ent')
           for m in hand.parallel_moves(mv, [r for r in avail if r not in busy_mv])]
    for o in ent:
        o.sub = 0
    g.ops = ent + g.ops
    carried = set(pre) | set(GLOBAL1 + GLOBALX + GLOBALH) | set(GLOBAL2)
    blocks = {}
    for b in ORDER[:-1]:
        ops = [o for o in g.ops if o.block == b]
        if b in FIXED:
            blocks[b] = ops
        else:
            blocks[b] = []
            sched = schedule_smooth if OPT.get(b[:1] + 'sm', OPT['sm']) else hand.schedule_block
            for sub in sorted(set(o.sub for o in ops)):
                blocks[b] += sched([o for o in ops if o.sub == sub], carried)
        # the scheduler does not see uniform registers: the constant's pointer and load go first
        first = [o for o in blocks[b] if o.text.startswith('UMOV')] + [o for o in blocks[b] if o.text.startswith('ULDC')]
        blocks[b] = first + [o for o in blocks[b] if o not in first]
        if OPT['ep'] and b == 'Ploop':
            # ep=N: the loop's counter and compare go N instructions ahead of the branch. A branch waits 13
            # cycles for its predicate; at the very end of the block all six warps of a scheduler sit out that
            # wait together, 42 times a nonce.
            book = [o for o in blocks[b] if o.text.startswith(('IADD3 %k,', 'ISETP'))]
            rest = [o for o in blocks[b] if o not in book]
            at = max(0, len(rest) - 2 - OPT['ep'])
            blocks[b] = rest[:at] + book + rest[at:]
    loc = allocate(blocks, pre, avail)
    for r in loc['__anchors']:
        z = gen.V('IADD3 R%d, RZ, 0x0, RZ' % (r + 1), 'ent')
        blocks['ent'].append(z)
    if not OPT['fr']:
        blocks['ent'].append(gen.V('BRA .S', 'ent'))      # the first three steps have no partial rounds
    code = [[hand.lower(o.text, loc), o.unit, b] for b in ORDER[:-1] for o in blocks[b]]
    if OPT['rpair']:
        code = reuse_pairs(code, OPT['rpair'])
    return info, const, code, home, avail


def reuse_pairs(code, window):
    """rpair=N: after registers are placed, an instruction up to N places later that reads one of this one's
    source registers in the same slot is moved up to follow it, where nothing in between depends on it or on what
    it writes, so reuse=1 can flag the shared operand (one register-file read fewer, often a bank conflict fewer).
    Within a block only; loads, uniform ops, compares and branches stay where they are."""
    import sass
    def parse(t):
        if 'BRA' in t or t.startswith(('LDC', 'ULDC', 'UMOV', 'UIADD3', 'ISETP')):
            return None
        try:
            return sass.Op(t)
        except Exception:
            return None
    out = list(code)
    ops = [parse(x[0]) for x in out]
    n = len(out)
    i = 0
    while i < n - 1:
        a = ops[i]
        if a is None or a.pred:
            i += 1
            continue
        wa = 'WIDE' in a.op
        srcs = [(k, x.n) for k, x in enumerate(a.srcs[:3]) if x.kind == 'r' and x.n != 255 and x.n not in a.dregs
                and not (k == 2 and wa and x.n + 1 in a.dregs)]
        nxt = ops[i + 1]
        def shares(c):
            return c is not None and not c.pred and any(k < len(c.srcs) and c.srcs[k].kind == 'r' and c.srcs[k].n == r
                                                       and (k != 2 or wa == ('WIDE' in c.op)) for k, r in srcs)
        if not srcs or shares(nxt):
            i += 1
            continue
        for j in range(i + 2, min(n, i + 1 + window)):
            if out[j][2] != out[i][2] or ops[j] is None:
                break
            c = ops[j]
            if not shares(c):
                continue
            ok = True
            for k in range(i + 1, j):
                m = ops[k]
                if m is None or out[k][2] != out[i][2]:
                    ok = False
                    break
                if (set(c.uregs) & set(m.dregs) or set(c.dregs) & (set(m.uregs) | set(m.dregs))
                        or set(c.upreds) & set(m.dpreds) or set(c.dpreds) & (set(m.upreds) | set(m.dpreds))):
                    ok = False
                    break
            if ok:
                out.insert(i + 1, out.pop(j))
                ops.insert(i + 1, ops.pop(j))
                break
        i += 1
    return out


def run(cubin, out, **opt):
    OPT.update(opt)
    info, const, code, home, avail = body_code(cubin)
    # the exit: every register the caller reads gets its element, then the return address
    moves = []
    if CNT_REGS[0] and not OPT['fr']:
        # sq3=1: the last S-boxes' outputs are three words; the fold's two corrections are made here
        for i in range(12):
            lo, hi, cn = home[i][0], home[i][1], CNT_REGS[0][i]
            code.append(['IADD3.X R%d, R%d, -0x2, R%d, !PT, !PT' % (hi, hi, cn), 'A', 'exit'])
            code.append(['IADD3.X R%d, P0, R%d, ~R%d, RZ, PT, !PT' % (lo, lo, cn), 'A', 'exit'])
    for i in range(0 if OPT['fr'] else 12):
        for h in range(2):
            for dst in info['out'][i][h]:
                if dst != home[i][h]:
                    moves.append((dst, home[i][h]))
    if OPT['fr']:
        for h in range(2):
            for dst in info['out'][0][h][1:]:
                moves.append((dst, info['out'][0][h][0]))
    used = set(d for d, _ in moves) | set(s for _, s in moves)
    for d, s in hand.parallel_moves(moves, [r for r in avail if r not in used]):
        code.append(['IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (d, s), 'M', 'exit'])
    for x in info['retfix']:
        code.append([x, gen.unit_of(x), 'exit'])
    code.append(['BRA .RET', 'B', 'exit'])
    cpatch = kf_consts(CONSTS[0]) if OPT['kf'] else {}
    if OPT['sq3']:
        cpatch.update(sq3_consts(CONSTS[0], cpatch))
    cpatch = cpatch or None
    stalls, per, rows2, labels2 = finish(cubin, out, info, code, cpatch)
    if cpatch:
        const = IF.const_bank3(out)
    entry, ret = info['entry'], info['ret']
    # simulate the patched function from the cubin
    body = [r for r in rows2 if entry <= r['addr'] <= ret]
    sops = [mkop(r['text']) for r in body]
    at = {r['addr']: i for i, r in enumerate(body)}
    rnd = random.Random(11)
    for _trial in range(3):
        s = [rnd.randrange(P) for _ in range(12)]
        cc = dict(const)
        if OPT['fr']:
            # fr=1: elements 0 and 4 in, and the line's constants in the parameter block
            lc = [rnd.randrange(P) for _ in range(12)]
            for i in range(12):
                cc[(0, info['cbase'] + 8 * i)], cc[(0, info['cbase'] + 8 * i + 4)] = lc[i] & 0xFFFFFFFF, lc[i] >> 32
            s = s[:2]
        m = Machine(cc)
        for r in range(256):
            m.r[r] = 0xA000 + r
        for i in range(len(s)):
            m.r[info['inp'][i][0]], m.r[info['inp'][i][1]] = s[i] & 0xFFFFFFFF, s[i] >> 32
        i, steps = 0, 0
        while not sops[i].op.startswith('RET'):
            if m.run(sops[i]) == 'BRA':
                i = at[labels2[re.search(r'\((\.L_x_\d+)\)', sops[i].target)[1]]]
            else:
                i += 1
            steps += 1
            assert steps < 400000
        w = [ref_ext0(ref_all(ref_first(s[0], s[1], lc)))] if OPT['fr'] else ref_all(s)
        for i in range(len(w)):
            for lo in info['out'][i][0]:
                for hi in info['out'][i][1]:
                    assert (m.r[lo] | m.r[hi] << 32) % P == w[i], 'simulation: element %d is wrong' % i
        for r in info['keep']:
            assert m.r[r] == 0xA000 + r, 'kept register R%d was written' % r
    return code, stalls, per, steps


def form(text):
    """An instruction's class for counting: a1/a2/a3 an ALU op by how many inputs it adds (registers, immediates,
    constants; RZ and carries do not count), w / wa a WIDE without and with an addend, m other IMADs, ld a load,
    u uniform, b branch."""
    t = re.sub(r'^@!?P\d ', '', text)
    op, _, rest = t.partition(' ')
    if 'BRA' in op:
        return 'b'
    if op.startswith('U'):
        return 'u'
    if op.startswith('LDC'):
        return 'ld'
    a = [x.strip() for x in rest.split(',')]
    if op.startswith('IMAD.WIDE'):
        k = 1 + sass.npreds(op, a)
        return 'w' if a[k + 2] == 'RZ' else 'wa'
    if op.startswith('IMAD'):
        return 'm'
    if op.startswith('IADD3'):
        k = 1 + sass.npreds(op, a)
        return 'a%d' % max(1, sum(1 for x in a[k:k + 3] if x != 'RZ'))
    return 'a1'


def mix(code):
    """{form: how many run a call}, along PATH."""
    from collections import Counter
    per = {bn: Counter(form(x[0]) for x in code if x[2] == bn) for bn in ORDER}
    tot = Counter()
    for bn in PATH:
        tot.update(per[bn])
    return per, tot


if __name__ == '__main__':
    kv = dict(a.split('=') for a in sys.argv[3:])
    code, stalls, per, steps = run(sys.argv[1], sys.argv[2], **{k: int(v) for k, v in kv.items()})
    from collections import Counter
    print('instructions %d; a nonce half runs %d' % (len(code), steps))
    pm, tot = mix(code)
    for bn in ORDER:
        u = Counter(x[1] for x in code if x[2] == bn)
        print('%-6s %4d instructions, %5d cycles for one warp, %s %s' % (bn, per[bn][0], per[bn][1], dict(u),
                                                                        dict(sorted(pm[bn].items()))))
    print('a call runs', dict(sorted(tot.items())))
