"""gen75.py: the field arithmetic of the hand-written sm_75 (RTX 20) partial rounds. No reduction multiply.

sm89/gen.py's routines end every product with r2 * EPS + (r1:r0) as an IMAD.WIDE with a 64-bit addend. An addend
WIDE runs 0.76 a clock a SM on Turing against 1.0 without one (ledger, 21 Nov), and the compiled kernel sits at 98%
of the multiplier pipe because of them. Here a product is folded on the ALU with p's own shape: with f = 2^32,
f^2 = f - 1 and f^3 = -1 (mod p), so

    w0 + w1 f + w2 f^2 + w3 f^3 = (w0 - w2 - w3) + (w1 + w2) f.

Each routine below writes its value as  [L0 - U - T - carries] + [L1 + M0 + U] f  with U = u + cu f one 32-bit sum,
and then runs the same four instructions (fold):

    lo' = L0 + ~u + ~T + !pa + !pb      -> k (0..2)    L0 - u - T - pa - pb = lo' - (2 - k) f
    hi' = L1 + M0 + u + k               -> c (0..2)
    hi  = hi' - 2 + c
    lo  = lo' - c

which is exact but for hi leaving [0, f) or lo borrowing in the last two (odds about 2^-30 a product; the kernel is
QC_EXACT 0 and the host checks every hit). pa and pb take the two carries that weigh -1: the middle sum's carry out
of 2^64 (f^3) and cu (cu f^2 at f is cu f - cu; the + cu f cancels against the - cu f of lo's bracket).

  multiply   4 WIDE (one with an addend) + 5 ALU      (sm89/gen.py: 5 WIDE + 5)
  square     3 WIDE (none)               + 7 ALU      (4 WIDE + 6)
  diagonal   4 WIDE (chained)            + 4 or 5 ALU (5 WIDE + 4)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'sm89'))
import gen  # noqa: E402
import ref  # noqa: E402

hx = gen.hx


class Prog75(gen.Prog):
    def fold(self, L0, L1, M0, u, T, pa, pb, out=None):
        """(lo, hi) of [L0 - u - T - pa - pb] + [L1 + M0 + u] f. T an operand or None (zero); pa, pb predicates or
        None."""
        k1, k2, c1, c2 = self.pred(), self.pred(), self.pred(), self.pred()
        lo1, hi1 = '%' + self.tmp('f'), '%' + self.tmp('f')
        lo, hi = out if out else ('%' + self.tmp('o'), '%' + self.tmp('o'))
        ca = ('!' + pa) if pa else 'PT'
        cb = ('!' + pb) if pb else 'PT'
        if T is None:
            self.emit('IADD3.X %s, %s, %s, %s, -0x1, ~%s, %s, %s' % (lo1, k1, k2, L0, u, ca, cb))
        else:
            self.emit('IADD3.X %s, %s, %s, %s, ~%s, ~%s, %s, %s' % (lo1, k1, k2, L0, u, T, ca, cb))
        self.emit('IADD3.X %s, %s, %s, %s, %s, %s, %s, %s' % (hi1, c1, c2, L1, M0, u, k1, k2))
        self.emit('IADD3.X %s, %s, -0x2, RZ, %s, %s' % (hi, hi1, c1, c2))
        self.emit('IADD3.X %s, %s, -0x2, RZ, !%s, !%s' % (lo, lo1, c1, c2))
        return lo, hi

    def mul(self, a, b):
        a0, a1 = a
        b0, b1 = b
        M = self.wide(a1, b0)
        pm = self.pred()
        M2 = self.wide(a0, b1, M, cout=pm)
        L = self.wide(a0, b0)
        H = self.wide(a1, b1)
        u, cu = '%' + self.tmp('u'), self.pred()
        self.emit('IADD3 %s, %s, %%%s.0, %%%s.1, RZ' % (u, cu, H, M2))
        return self.fold('%%%s.0' % L, '%%%s.1' % L, '%%%s.0' % M2, u, '%%%s.1' % H, pm, cu)

    def sqr(self, x):
        x0, x1 = x
        # 2 x0 x1 = (2 x0 mod f) x1 + pt x1 f, pt the bit doubled out
        b, pt = '%' + self.tmp('b'), self.pred()
        self.emit('IADD3 %s, %s, %s, %s, RZ' % (b, pt, x0, x0))
        M = self.wide(b, x1)
        L = self.wide(x0, x0)
        H = self.wide(x1, x1)
        xp = '%' + self.tmp('s')
        self.emit('SEL %s, %s, RZ, %s' % (xp, x1, pt))
        u, cu1, cu2 = '%' + self.tmp('u'), self.pred(), self.pred()
        self.emit('IADD3 %s, %s, %s, %%%s.0, %%%s.1, %s' % (u, cu1, cu2, H, M, xp))
        return self.fold('%%%s.0' % L, '%%%s.1' % L, '%%%s.0' % M, u, '%%%s.1' % H, cu1, cu2)

    def dmul(self, a, i, L, H, out=None):
        """D_i * a + (L + f H) mod p, L and H pair names, with D' = D f: X = a0 d0 + a1 e0 + L, Y = a0 d1 + a1 e1
        + H, value X + Y f."""
        D, DP = ref.D[i], ref.DP[i]
        d0, d1, e0, e1 = D & 0xFFFFFFFF, D >> 32, DP & 0xFFFFFFFF, DP >> 32
        CX, CY = gen.CXY[i]
        a0, a1 = a
        Q = self.wide(a0, hx(d1), H)
        pcy = self.pred() if CY else None
        Y = self.wide(a1, hx(e1), Q, cout=pcy) if CY else self.wide(a1, hx(e1), Q)
        Pp = self.wide(a0, hx(d0), L)
        pcx = self.pred() if CX else None
        X = self.wide(a1, hx(e0), Pp, cout=pcx) if CX else self.wide(a1, hx(e0), Pp)
        if CX:
            u, cu = '%' + self.tmp('u'), self.pred()
            self.emit('IADD3.X %s, %s, %%%s.1, RZ, RZ, %s, !PT' % (u, cu, Y, pcx))
        else:
            u, cu = '%%%s.1' % Y, None
        return self.fold('%%%s.0' % X, '%%%s.1' % X, '%%%s.0' % Y, u, None, pcy, cu, out=out)
