"""gen.py: build the hand-written partial-round function for sm_89 (qc_hand_loop's replacement).

Ops are written as SASS text with virtual names: %name is a 32-bit value, %name.0 / %name.1 the halves of a
register pair %name (a pair operand is written %name). $name is a predicate. Blocks are scheduled one at a time
(list scheduling that keeps the multiplier and ALU pipes level), then registers are allocated over the whole
function and the control fields worked out from the latencies ptxas itself keeps.
"""
import os
import re
import ref

# QC_HANDV (a bit mask, for trials): 1 the squares' top carry, 2 the diagonal multiplies' e when X cannot carry,
# 4 the sums' last carry, each on the multiplier pipe (IMAD.X) in place of the ALU.
# The default, 3205 = 1 + 4 + 128 + 1024 + 2048: carries that need no carry-out on IMAD.X (1, 4), products folded
# with w^2 = w - 1 (128, 1024) and the 18-op sum (2048). 0 is q5-rtx40h's first loop.
HANDV = int(os.environ.get('QC_HANDV') or '3205')

EPS = '-0x1'
MFOLD = int(os.environ.get('QC_MFOLD', '2'))


def hx(v):
    v &= 0xFFFFFFFF
    return '-0x%x' % ((1 << 32) - v) if v >= 0x80000000 else '0x%x' % v


class V:
    """One op: text with virtual names, and which block it is in."""
    def __init__(self, text, block):
        self.text, self.block = text, block
        self.unit = unit_of(text)


def unit_of(t):
    op = t.split(' ')[0] if not t.startswith('@') else t.split(' ')[1]
    if op.startswith('IMAD.WIDE'):
        return 'W'
    if op.startswith('IMAD'):
        return 'M'
    if op.startswith(('BRA', 'RET')):
        return 'B'
    if op.startswith('U'):
        return 'U'
    return 'A'


NAME = re.compile(r'([%$])([A-Za-z_][A-Za-z0-9_]*)(\.[01])?')


class Prog:
    def __init__(self):
        self.ops = []
        self.block = 'main'
        self.n = 0
        self.pairs = set()

    def tmp(self, base='t'):
        self.n += 1
        return '%s%d' % (base, self.n)

    def emit(self, text):
        self.ops.append(V(text, self.block))

    # ---- arithmetic building blocks; values are (lo, hi) tuples of operand strings
    def wide(self, a, b, c=None, cout=None, cin=None, X=False):
        d = self.tmp('w')
        self.pairs.add(d)
        cs = ('%' + c) if c else 'RZ'
        if X:
            self.emit('IMAD.WIDE.U32.X %%%s, %s, %s, %s, %s' % (d, a, b, cs, cin))
        elif cout:
            self.emit('IMAD.WIDE.U32 %%%s, %s, %s, %s, %s' % (d, cout, a, b, cs))
        else:
            self.emit('IMAD.WIDE.U32 %%%s, %s, %s, %s' % (d, a, b, cs))
        return d

    def pred(self):
        return '$' + self.tmp('p')

    def reduce(self, pairlo, h0, h1, out=None):
        """(lo, hi) of r = (pairlo.0, pairlo.1) + h0 2^64 + h1 2^96 mod p (the QC_RED 3 tail, carry-in fixed)."""
        c, b = self.pred(), self.pred()
        E = self.wide(h0, EPS, pairlo, cout=c)
        lo, hi = out if out else ('%' + self.tmp('r'), '%' + self.tmp('r'))
        self.emit('IADD3.X %s, %s, %%%s.0, ~%s, RZ, !%s, !PT' % (lo, b, E, h1, c))
        self.emit('IADD3.X %s, RZ, -0x1, %%%s.1, %s, %s' % (hi, E, b, c))
        return lo, hi

    def sqr(self, x, out=None):
        x0, x1 = x
        T = self.wide(x0, x1)
        p0 = self.pred()
        t2 = self.tmp('d')
        self.pairs.add(t2)
        lo2 = '%' + self.tmp('r')
        self.emit('IADD3 %s, %s, %%%s.0, %%%s.0, RZ' % (lo2, p0, T, T))
        self.emit('IADD3.X %%%s.0, %s, %%%s.1, %%%s.1, RZ, %s, !PT' % (t2, p0, T, T, p0))
        self.emit(('IMAD.X %%%s.1, RZ, RZ, RZ, %s' if HANDV & 1 else 'IADD3.X %%%s.1, RZ, RZ, RZ, %s, !PT') % (t2, p0))
        R = self.wide(x0, x0)
        p1 = self.pred()
        self.emit('IADD3 %%%s.1, %s, %%%s.1, %s, RZ' % (R, p1, R, lo2))
        H = self.wide(x1, x1, t2, cin=p1, X=True)
        return self.reduce(R, '%%%s.0' % H, '%%%s.1' % H, out)

    def mul(self, a, b, out=None):
        a0, a1 = a
        b0, b1 = b
        M = self.wide(a1, b0)
        pm = self.pred()
        M2 = self.wide(a0, b1, M, cout=pm)
        Lp = self.wide(a0, b0)
        pq = self.pred()
        self.emit('IADD3 %%%s.1, %s, %%%s.1, %%%s.0, RZ' % (Lp, pq, Lp, M2))
        Hh = self.wide(a1, b1)
        px = self.pred()
        hlo, hhi = '%' + self.tmp('r'), '%' + self.tmp('r')
        self.emit('IADD3.X %s, %s, %%%s.0, %%%s.1, RZ, %s, !PT' % (hlo, px, Hh, M2, pq))
        self.emit('IADD3.X %s, %%%s.1, RZ, RZ, %s, %s' % (hhi, Hh, px, pm))
        return self.reduce(Lp, hlo, hhi, out)

    def sbox(self, x, out=None):
        mul = (self.mulf if HANDV & 1024 else self.mulw) if HANDV & 128 else self.mul
        x2 = self.sqr(x)
        x3 = mul(x2, x)
        x4 = self.sqr(x2)
        return mul(x3, x4, out)

    # ---- QC_HANDV 128: products folded with w^2 = w - 1 (w = 2^32, p = w^2 - w + 1) on the ALU, in place of the
    # multiply by EPS. A value a0 + a1 w times b0 + b1 w is (a0 b0 - a1 b1) + (a0 b1 + a1 b0 + a1 b1) w, so the
    # high product rides in the cross products' addend chain and only words are left to add and subtract. A carry
    # k out of the w digit is k w^2 = k w - k: add k to that digit, take k from the low one. The folds' own wraps
    # (a digit within 2 of 0 or 2^32 when a carry lands) are left, at about 2^-31 a multiply, as the EPS tail's are.
    def mulw(self, a, b, out=None):
        a0, a1 = a
        b0, b1 = b
        P2 = self.wide(a1, b1)
        pa = self.pred()
        M1 = self.wide(a1, b0, P2, cout=pa)
        pb = self.pred()
        C1 = self.wide(a0, b1, M1, cout=pb)
        P0 = self.wide(a0, b0)
        g1, g2, h1, h2, j1, j2 = [self.pred() for _ in range(6)]
        t0, t1, t2, r0a = ['%' + self.tmp('f') for _ in range(4)]
        r0, r1 = out if out else ('%' + self.tmp('r'), '%' + self.tmp('r'))
        # digit 0: c00 - p20 - C11 - pa - pb = t0 + (G - 2) w
        self.emit('IADD3.X %s, %s, %s, %%%s.0, ~%%%s.0, ~%%%s.1, !%s, !%s' % (t0, g1, g2, P0, P2, C1, pa, pb))
        # digit 1: c01 - p21 + C10 + (G - 2) = t1 + (H - 1) w - 1
        self.emit('IADD3.X %s, %s, %s, %%%s.1, ~%%%s.1, %%%s.0, %s, %s' % (t1, h1, h2, P0, P2, C1, g1, g2))
        # ... + C11, with (H - 1) folded in: t1 + C11 + H - 2 = t2 + (J - 1) w
        self.emit('IADD3.X %s, %s, %s, %s, -0x2, %%%s.1, %s, %s' % (t2, j1, j2, t1, C1, h1, h2))
        jj = j1 if HANDV & 256 else j2          # which of the two carries says "two"
        self.emit(('IMAD.X %s, RZ, RZ, %s, %s' if HANDV & 512 else 'IADD3.X %s, RZ, RZ, %s, %s, !PT') % (r1, t2, jj))
        self.emit('IADD3.X %s, %s, -0x1, RZ, !%s, !%s' % (r0a, t0, h1, h2))
        self.emit('IADD3.X %s, %s, -0x1, RZ, !%s, !PT' % (r0, r0a, jj))
        return r0, r1

    def fold(self, L0, L1, M0, u, T, pa, pb, out=None):
        """q5-nv2030's four-instruction fold (sm75/gen75.py): (lo, hi) of [L0 - u - T - pa - pb] + [L1 + M0 + u] w."""
        k1, k2, c1, c2 = self.pred(), self.pred(), self.pred(), self.pred()
        lo1, hi1 = '%' + self.tmp('f'), '%' + self.tmp('f')
        lo, hi = out if out else ('%' + self.tmp('o'), '%' + self.tmp('o'))
        self.emit('IADD3.X %s, %s, %s, %s, ~%s, ~%s, %s, %s' % (lo1, k1, k2, L0, u, T, '!' + pa, '!' + pb))
        self.emit('IADD3.X %s, %s, %s, %s, %s, %s, %s, %s' % (hi1, c1, c2, L1, M0, u, k1, k2))
        self.emit('IADD3.X %s, %s, -0x2, RZ, %s, %s' % (hi, hi1, c1, c2))
        self.emit('IADD3.X %s, %s, -0x2, RZ, !%s, !%s' % (lo, lo1, c1, c2))
        return lo, hi

    def mulf(self, a, b, out=None):
        """4 WIDE + 5 ALU: the middle sum chained, u = low(a1 b1) + high(middle), then fold."""
        a0, a1 = a
        b0, b1 = b
        M = self.wide(a1, b0)
        pm = self.pred()
        M2 = self.wide(a0, b1, M, cout=pm)
        L = self.wide(a0, b0)
        H = self.wide(a1, b1)
        u, cu = '%' + self.tmp('u'), self.pred()
        self.emit('IADD3 %s, %s, %%%s.0, %%%s.1, RZ' % (u, cu, H, M2))
        return self.fold('%%%s.0' % L, '%%%s.1' % L, '%%%s.0' % M2, u, '%%%s.1' % H, pm, cu, out)

    def dmulw(self, a, i, L, H, out=None):
        """D_i * a + (L + 2^32 H) mod p: X = a0 d0 + L + a1 e0 and Y = a0 d1 + H + a1 e1 as dmul has them, then
        X + Y w = [x0 - y1 - cx - cy] + [x1 + y0 + y1 + cx] w, folded on the ALU."""
        D, DP = ref.D[i], ref.DP[i]
        d0, d1, e0, e1 = D & 0xFFFFFFFF, D >> 32, DP & 0xFFFFFFFF, DP >> 32
        CX, CY = CXY[i]
        a0, a1 = a
        Q = self.wide(a0, hx(d1), H)
        pcy = self.pred() if CY else None
        Y = self.wide(a1, hx(e1), Q, cout=pcy) if CY else self.wide(a1, hx(e1), Q)
        Pp = self.wide(a0, hx(d0), L)
        pcx = self.pred() if CX else None
        X = self.wide(a1, hx(e0), Pp, cout=pcx) if CX else self.wide(a1, hx(e0), Pp)
        g, h1, h2 = self.pred(), self.pred(), self.pred()
        t0, t1 = '%' + self.tmp('f'), '%' + self.tmp('f')
        r0, r1 = out if out else ('%' + self.tmp('o'), '%' + self.tmp('o'))
        if CX and CY:
            gx = self.pred()
            pr = (g, gx) if HANDV & 256 else (gx, g)
            self.emit('IADD3.X %s, %s, %s, %%%s.0, -0x1, ~%%%s.1, !%s, !%s' % (t0, pr[0], pr[1], X, Y, pcx, pcy))
        elif CX or CY:
            self.emit('IADD3.X %s, %s, %%%s.0, ~%%%s.1, RZ, !%s, !PT' % (t0, g, X, Y, pcx or pcy))
        else:
            self.emit('IADD3.X %s, %s, %%%s.0, ~%%%s.1, RZ, PT, !PT' % (t0, g, X, Y))
        # digit 1 + 1 = x1 + y0 + y1 + cx + (no borrow) = t1 + H w
        self.emit('IADD3.X %s, %s, %s, %%%s.1, %%%s.0, %%%s.1, %s, %s' % (t1, h1, h2, X, Y, Y, pcx if CX else '!PT', g))
        self.nd = getattr(self, 'nd', 0) + 1
        if HANDV & 4096 and self.nd % MFOLD == 0:
            # the same two sums as four IMAD.X (%m1 holds -1 and %one 1): 8 multiplier cycles for 4 ALU cycles
            u, v = '%' + self.tmp('f'), '%' + self.tmp('f')
            self.emit('IMAD.X %s, %%m1, %%one, %s, %s' % (u, t1, h1))
            self.emit('IMAD.X %s, RZ, RZ, %s, %s' % (r1, u, h2))
            self.emit('IMAD.X %s, %%m1, %%one, %s, !%s' % (v, t0, h1))
            self.emit('IMAD.X %s, %%m1, %%one, %s, !%s' % (r0, v, h2))
            return r0, r1
        self.emit('IADD3.X %s, %s, -0x1, RZ, %s, %s' % (r1, t1, h1, h2))
        self.emit('IADD3.X %s, %s, -0x2, RZ, !%s, !%s' % (r0, t0, h1, h2))
        return r0, r1

    def dmul(self, a, i, L, H, out=None):
        """D_i * a + (L + 2^32 H) mod p with D' (qc_dmul); L, H pair names."""
        D, DP = ref.D[i], ref.DP[i]
        d0, d1, e0, e1 = D & 0xFFFFFFFF, D >> 32, DP & 0xFFFFFFFF, DP >> 32
        CX, CY = CXY[i]
        a0, a1 = a
        Q = self.wide(a0, hx(d1), H)
        pcy = self.pred() if CY else None
        Y = self.wide(a1, hx(e1), Q, cout=pcy) if CY else self.wide(a1, hx(e1), Q)
        Pp = self.wide(a0, hx(d0), L)
        pcx = self.pred() if CX else None
        X = self.wide(a1, hx(e0), Pp, cout=pcx) if CX else self.wide(a1, hx(e0), Pp)
        pk = self.pred()
        self.emit('IADD3 %%%s.1, %s, %%%s.1, %%%s.0, RZ' % (X, pk, X, Y))
        e = '%' + self.tmp('e')
        if not CX and HANDV & 2:
            self.emit('IMAD.X %s, RZ, RZ, %%%s.1, %s' % (e, Y, pk))
        else:
            self.emit('IADD3.X %s, RZ, RZ, %%%s.1, %s, %s' % (e, Y, pk, pcx if CX else '!PT'))
        c, b = self.pred(), self.pred()
        E = self.wide(e, EPS, X, cout=c)
        lo, hi = out if out else ('%' + self.tmp('o'), '%' + self.tmp('o'))
        if CY:
            self.emit('IADD3.X %s, %s, %%%s.0, -0x2, RZ, !%s, !%s' % (lo, b, E, pcy, c))
        else:
            self.emit('IADD3.X %s, %s, %%%s.0, -0x1, RZ, !%s, !PT' % (lo, b, E, c))
        self.emit('IADD3.X %s, RZ, -0x1, %%%s.1, %s, %s' % (hi, E, b, c))
        return lo, hi

    def sum12(self, terms, S=None):
        """A 64-bit pair holding the sum of twelve 32-bit operands (into pair S when given)."""
        S = S or self.tmp('S')
        self.pairs.add(S)
        t = list(terms)
        assert len(t) == 12
        groups = [t[0:3], t[3:5], t[5:7], t[7:9], t[9:11], t[11:12]]
        for g, take in enumerate(groups):
            srcs = list(take) if g == 0 else ['%%%s.0' % S] + list(take)
            p1, p2 = self.pred(), self.pred()
            if len(srcs) == 3:
                self.emit('IADD3 %%%s.0, %s, %s, %s' % (S, p1, p2, ', '.join(srcs)))
                cin = '%s, %s' % (p1, p2)
            else:
                self.emit('IADD3 %%%s.0, %s, %s, RZ' % (S, p1, ', '.join(srcs)))
                cin = '%s, !PT' % p1
            if g == 0:
                self.emit('IADD3.X %%%s.1, RZ, RZ, RZ, %s' % (S, cin))
            elif len(srcs) == 2 and HANDV & 4:
                self.emit('IMAD.X %%%s.1, RZ, RZ, %%%s.1, %s' % (S, S, p1))
            else:
                self.emit('IADD3.X %%%s.1, RZ, RZ, %%%s.1, %s' % (S, S, cin))
        return S

    def sums2(self, lo_terms, hi_terms):
        """The sum of twelve values as L.0 + w H: L.0 the low words' sum mod 2^32 (L.1 stays zero, written once
        in the prologue), its carries taken as carry-ins by the high words' sum H (below 13 * 2^32 + 13). 18 ALU
        ops against sum12's 24 for the two columns."""
        idx = [[0, 1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11]]
        for g, ix in enumerate(idx):
            ls = [lo_terms[i] for i in ix] if g == 0 else ['%L.0'] + [lo_terms[i] for i in ix]
            hs = [hi_terms[i] for i in ix] if g == 0 else ['%H.0'] + [hi_terms[i] for i in ix]
            p1, p2, q1, q2 = self.pred(), self.pred(), self.pred(), self.pred()
            if len(ls) == 3:
                self.emit('IADD3 %%L.0, %s, %s, %s' % (p1, p2, ', '.join(ls)))
                self.emit('IADD3.X %%H.0, %s, %s, %s, %s, %s' % (q1, q2, ', '.join(hs), p1, p2))
                cin = '%s, %s' % (q1, q2)
            else:
                self.emit('IADD3 %%L.0, %s, %s, RZ' % (p1, ', '.join(ls)))
                self.emit('IADD3.X %%H.0, %s, %s, RZ, %s, !PT' % (q1, ', '.join(hs), p1))
                cin = '%s, !PT' % q1
            if len(ls) != 3 and HANDV & 4:
                self.emit('IMAD.X %%H.1, RZ, RZ, %%H.1, %s' % q1)
            else:
                self.emit('IADD3.X %%H.1, RZ, RZ, %s, %s' % ('RZ' if g == 0 else '%H.1', cin))

    def add64(self, S, lo, hi=None):
        """S + (lo, hi-word count) as a new pair; lo an operand, hi None (zero)."""
        N = self.tmp('S')
        self.pairs.add(N)
        p = self.pred()
        self.emit('IADD3 %%%s.0, %s, %%%s.0, %s, RZ' % (N, p, S, lo))
        if hi is None and HANDV & 4:
            self.emit('IMAD.X %%%s.1, RZ, RZ, %%%s.1, %s' % (N, S, p))
            return N
        self.emit('IADD3.X %%%s.1, %%%s.1, RZ, RZ, %s, !PT' % (N, S, p) if hi is None else
                  'IADD3.X %%%s.1, %%%s.1, %s, RZ, %s, !PT' % (N, S, hi, p))
        return N


def _cxy():
    """Which of the diagonal multiply's two chains can pass 2^64 (L, H < 13 * 2^32), from the constants."""
    out = []
    for i in range(12):
        D, DP = ref.D[i], ref.DP[i]
        d0, d1, e0, e1 = D & 0xFFFFFFFF, D >> 32, DP & 0xFFFFFFFF, DP >> 32
        m = (1 << 32) - 1
        out.append((m * d0 + m * e0 + 14 * (1 << 32) >= 1 << 64, m * d1 + m * e1 + 14 * (1 << 32) >= 1 << 64))
    return out


CXY = _cxy()
