"""full75.py: the sm_75 kernel's body from the first partial rounds to the last S-boxes, written by hand.

  python full75.py <in.cubin built with -DQC_HAND75=2> <out.cubin> [key=value ...]

It replaces qc_hand_loopall's body (qc_kernel.cuh) with these blocks, each scheduled on its own:

  ent     the counters
  Ppro    partial round 0: S-box, the two half sums, element 0's product
  Ploop   partial rounds 1..21, the diagonal products of the round before riding beside each S-box (sm89/hand.py)
  Pepi    the last round's eleven products with the terminal constants
  S, Sx   twelve S-boxes in place; leave after the last ones
  E, Ex   the external layer with its constants, then the step counter: on to S, again to E (the first
          permutation's last layer is followed by the second one's first), or to the partial rounds
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
(multiplier pipe); anch=N anchors, so a multiply's high product takes the middle sum's high word as its addend.
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

OPT = patch75.OPT
OPT.update(sqmul=0, imx=0, plimit=4, sbatch=3, anch=0, ldcs=1, slow=0)
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


def ref_all(s):
    s = [v % P for v in s]
    for k in range(4, 17):
        j = k - 9 if k >= 9 else k
        if k > 4:
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
    inp = [[None, None] for _ in range(12)]
    origin, zreg = {}, None
    for op in ops:                                  # the entry: copies and the input markers
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
    out = [[[], []] for _ in range(12)]
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
    return dict(inp=inp, out=out, entry=entry, ret=ret, retreg=retreg, retfix=retfix, keep=sorted(keep),
                clob=sorted(regs_w), cpreds=sorted(preds_w), urs=sorted(urs), n=len(body))


# ------------------------------------------------------------------ the program
class ProgAll(gen75.Prog75):
    sub = 0

    def emit(self, text):
        # sub: the scheduler orders one sub-block at a time. Given twelve independent S-boxes at once it starts
        # them all, and their products do not fit the registers.
        super().emit(text)
        self.ops[-1].sub = self.sub

    def mul(self, a, b, out=None):
        a0, a1 = a
        b0, b1 = b
        M = self.wide(a1, b0)
        pm = self.pred()
        M2 = self.wide(a0, b1, M, cout=pm)
        L = self.wide(a0, b0)
        if OPT['anch']:
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

    def sbox(self, x, out=None):
        sq = (lambda v: self.mul(v, v)) if OPT['sqmul'] else self.sqr
        x2 = sq(x)
        x3 = self.mul(x2, x)
        x4 = sq(x2)
        return self.mul(x3, x4, out=out)

    def sum12(self, terms, S=None):
        if not OPT['imx'] & 1:
            return super().sum12(terms, S)
        hv, gen.HANDV = gen.HANDV, gen.HANDV | 4
        try:
            return super().sum12(terms, S)
        finally:
            gen.HANDV = hv

    # ---- the external layer: a value is ('x', lo, hi) or ('q', w0, w1, w2)
    def step(self, *ins, out3=None):
        n = len(ins)
        w = out3 or tuple('%' + self.tmp('q') for _ in range(3))
        cols = [[v[1 + j] if len(v) > 1 + j else None for v in ins] for j in range(3)]
        pad = lambda c: ', '.join((c + ['RZ', 'RZ'])[:3])
        if n == 3:
            p1, p2, q1, q2 = self.pred(), self.pred(), self.pred(), self.pred()
            self.emit('IADD3 %s, %s, %s, %s' % (w[0], p1, p2, pad(cols[0])))
            self.emit('IADD3.X %s, %s, %s, %s, %s, %s' % (w[1], q1, q2, pad(cols[1]), p1, p2))
            self.emit('IADD3.X %s, %s, %s, %s' % (w[2], pad([c or 'RZ' for c in cols[2]]), q1, q2))
        else:
            p1, q1 = self.pred(), self.pred()
            self.emit('IADD3 %s, %s, %s' % (w[0], p1, pad(cols[0])))
            third = [c for c in cols[2] if c]
            if third:
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

    def ext(self, S):
        x = [('x', S[i][0], S[i][1]) for i in range(12)]

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
                    c, r = rc(g + k), y[g + k]
                    p1, p2, q1, q2 = self.pred(), self.pred(), self.pred(), self.pred()
                    w0, w1, w2 = ('%' + self.tmp('q') for _ in range(3))
                    self.emit('IADD3 %s, %s, %s, %s, %s, %s' % (w0, p1, p2, r[1], col[1], c[1]))
                    self.emit('IADD3.X %s, %s, %s, %s, %s, %s, %s, %s' % (w1, q1, q2, r[2], col[2], c[2], p1, p2))
                    self.emit('IADD3.X %s, %s, -0x1, %s, %s, %s' % (w2, r[3], col[3], q1, q2))
                    self.emit('IADD3.X %s, %s, %s, ~%s, RZ, !PT, !PT' % (S[g + k][0], pb, w0, w2))
                    self.emit('IMAD.X %s, %s, 0x1, %s, %s' % (S[g + k][1], w2, w1, pb))
                    continue
                o = self.step(y[g + k], col, rc(g + k))
                self.emit('IADD3 %s, %s, %s, -%s, RZ' % (S[g + k][0], pb, o[1], o[3]))
                self.emit('IADD3.X %s, %s, -0x1, %s, %s, !PT' % (S[g + k][1], o[2], o[3], pb))


GLOBAL1 = ['cnt', 'rcp', 'k']
GLOBAL2 = ['L', 'H']
URP, URD0, URD1 = 6, 4, 5        # the partial rounds' constant pointer and data (what the compiled function uses)


def build(c):
    """c: constant offsets in bank 3 (seq, p22, rct)."""
    g = ProgAll()
    S = [('%%s%dl' % i, '%%s%dh' % i) for i in range(12)]
    g.pairs.update(GLOBAL2)

    def sums(y):
        g.sum12([y[0]] + [S[i][0] for i in range(1, 12)], 'L')
        g.sum12([y[1]] + [S[i][1] for i in range(1, 12)], 'H')
        return g.add64('L', 'UR%d' % URD0), g.add64('H', 'UR%d' % URD1)

    g.block = 'ent'
    g.emit('IADD3 %cnt, RZ, 0x0, RZ')
    g.emit('IADD3 %%rcp, RZ, 0x%x, RZ' % (c['seq'] + 5 * 0x60))
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
        Li = g.add64('L', 'c[0x3][0x%x]' % (c['rct'] + 8 * i))
        Hi = g.add64('H', 'c[0x3][0x%x]' % (c['rct'] + 8 * i + 4))
        g.dmul(S[i], i, Li, Hi, out=S[i])
    g.block = 'S'
    for i in range(12):
        g.sub = i // OPT['sbatch']
        g.sbox(S[i], out=S[i])
    g.sub = 0
    g.block = 'Sx'
    g.emit('ISETP.NE.AND $pe, PT, %cnt, 0xc, PT')
    g.emit('@!$pe BRA .EXIT')
    g.block = 'E'
    g.ext(S)
    g.sub = 0
    g.block = 'Ex'
    g.emit('IADD3 %cnt, %cnt, 0x1, RZ')
    g.emit('IADD3 %rcp, %rcp, 0x60, RZ')
    g.emit('ISETP.NE.AND $p4, PT, %cnt, 0x4, PT')
    g.emit('ISETP.NE.AND $p9, PT, %cnt, 0x9, PT')
    g.emit('@!$p9 BRA .P')
    g.emit('@$p4 BRA .S')
    g.emit('IADD3 %%rcp, RZ, 0x%x, RZ' % c['seq'])
    g.emit('BRA .E')
    return g, S


ORDER = ['ent', 'Ppro', 'Ploop', 'Pepi', 'S', 'Sx', 'E', 'Ex', 'exit']
LABEL = {'.P': 'Ppro', '.LOOP': 'Ploop', '.S': 'S', '.E': 'E', '.EXIT': 'exit'}
FIXED = ('ent', 'Sx', 'Ex', 'exit')
PP = ['Ppro'] + ['Ploop'] * 21 + ['Pepi']
SE = ['S', 'Sx', 'E', 'Ex']
PATH = ['ent'] + PP + SE * 4 + ['E', 'Ex'] + SE * 4 + PP + SE * 3 + ['S', 'Sx', 'exit']


def cells(text):
    if text.startswith('LDC.64'):
        m = re.match(r'LDC\.64 %(\w+), c\[0x3\]\[%(\w+)', text)
        return 'LDC.64', [m[1] + '.0', m[1] + '.1'], [m[2]]
    return _cells(text)


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
    for n in GLOBAL1:
        r = next(r for r in sorted(pool, key=lambda r: ((r ^ 1) in pool, r)))
        loc[n] = r
        pool.remove(r)
    anchors = []
    for _ in range(OPT['anch']):
        r = next(r for r in pool if r % 2 == 0 and r + 1 in pool)
        anchors.append(r)
        pool.remove(r)
        pool.remove(r + 1)
    loc['__anchors'] = anchors
    homes = set(pre)
    glob = homes | set(GLOBAL1) | set(n + h for n in GLOBAL2 for h in ('.0', '.1')) | set(GLOBAL2)

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
            if not writes:
                iv = [(0, n)]
            else:
                before = [i for i in reads if i <= writes[0]]
                iv = ([(0, before[-1])] if before else []) + [(writes[0], n)]
            busy[pre[h]] = iv
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
                r = next((r for r in sorted(busy, key=lambda r: ((r ^ 1) in busy and free(busy, r ^ 1, a, b), r))
                          if free(busy, r, a, b)), None)
                if r is None:
                    raise RuntimeError('out of registers in %s at %d (%s)' % (name, a, gn))
                busy[r].append((a, b))
                loc[gn] = r
        peak = max(peak, max(r for r in busy if busy[r]) if any(busy.values()) else 0)
    return loc


g_pairs = [set()]


# ------------------------------------------------------------------ timing
def ur_ops(text):
    """(written uniform registers, latency) and read ones."""
    m = re.match(r'U[A-Z0-9.]+ UR(\d+)', text)
    wr, lat = [], 0
    rest = text
    if m:
        k = int(m[1])
        wr = [k, k + 1] if text.startswith('ULDC.64') else [k]
        lat = 30 if text.startswith('ULDC') else 6
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


def run(cubin, out, **opt):
    OPT.update(opt)
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
    g, S = build(c)
    g_pairs[0] = g.pairs
    # The elements arrive in whatever registers ptxas chose (all odd ones here, which would leave few aligned
    # pairs), so they are moved to the top 24 free registers on the way in and to the caller's on the way out.
    avail = [r for r in info['clob'] if r not in info['keep'] and r != 1]
    top = sorted(avail)[-24:]
    home = [(top[2 * i], top[2 * i + 1]) for i in range(12)]
    pre = {}
    for i in range(12):
        pre[S[i][0][1:]], pre[S[i][1][1:]] = home[i]
    mv = [(home[i][h], info['inp'][i][h]) for i in range(12) for h in range(2) if home[i][h] != info['inp'][i][h]]
    busy_mv = set(d for d, _ in mv) | set(s_ for _, s_ in mv)
    ent = [gen.V('IMAD.MOV.U32 R%d, RZ, RZ, R%d' % m, 'ent')
           for m in hand.parallel_moves(mv, [r for r in avail if r not in busy_mv])]
    for o in ent:
        o.sub = 0
    g.ops = ent + g.ops
    carried = set(pre) | set(GLOBAL1) | set(GLOBAL2)
    blocks = {}
    for b in ORDER[:-1]:
        ops = [o for o in g.ops if o.block == b]
        if b in FIXED:
            blocks[b] = ops
        else:
            blocks[b] = []
            for sub in sorted(set(o.sub for o in ops)):
                blocks[b] += hand.schedule_block([o for o in ops if o.sub == sub], carried)
        # the scheduler does not see uniform registers: the constant's pointer and load go first
        first = [o for o in blocks[b] if o.text.startswith('UMOV')] + [o for o in blocks[b] if o.text.startswith('ULDC')]
        blocks[b] = first + [o for o in blocks[b] if o not in first]
    loc = allocate(blocks, pre, avail)
    for r in loc['__anchors']:
        z = gen.V('IADD3 R%d, RZ, 0x0, RZ' % (r + 1), 'ent')
        blocks['ent'].append(z)
    code = [[hand.lower(o.text, loc), o.unit, b] for b in ORDER[:-1] for o in blocks[b]]
    # the exit: every register the caller reads gets its element, then the return address
    moves = []
    for i in range(12):
        for h in range(2):
            for dst in info['out'][i][h]:
                if dst != home[i][h]:
                    moves.append((dst, home[i][h]))
    used = set(d for d, _ in moves) | set(s for _, s in moves)
    for d, s in hand.parallel_moves(moves, [r for r in avail if r not in used]):
        code.append(['IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (d, s), 'M', 'exit'])
    for x in info['retfix']:
        code.append([x, gen.unit_of(x), 'exit'])
    code.append(['BRA .RET', 'B', 'exit'])
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
    off, _size = P89.text_section(b)
    base = rows[0]['addr']
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
            if OPT['slow'] & (1 if bn[0] == 'P' else 2 if bn == 'S' else 4 if bn == 'E' else 8):
                s = 15
            ct = (s & 15) | ((1 if s <= OPT['ybit'] else 0) << 4) | (wsb[i] << 5) | (7 << 8) | (wait[i] << 11)
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
    # simulate the patched function from the cubin
    body = [r for r in rows2 if entry <= r['addr'] <= ret]
    sops = [mkop(r['text']) for r in body]
    at = {r['addr']: i for i, r in enumerate(body)}
    rnd = random.Random(11)
    for _trial in range(3):
        s = [rnd.randrange(P) for _ in range(12)]
        m = Machine(const)
        for r in range(256):
            m.r[r] = 0xA000 + r
        for i in range(12):
            m.r[info['inp'][i][0]], m.r[info['inp'][i][1]] = s[i] & 0xFFFFFFFF, s[i] >> 32
        i, steps = 0, 0
        while not sops[i].op.startswith('RET'):
            if m.run(sops[i]) == 'BRA':
                i = at[labels2[re.search(r'\((\.L_x_\d+)\)', sops[i].target)[1]]]
            else:
                i += 1
            steps += 1
            assert steps < 400000
        w = ref_all(s)
        for i in range(12):
            for lo in info['out'][i][0]:
                for hi in info['out'][i][1]:
                    assert (m.r[lo] | m.r[hi] << 32) % P == w[i], 'simulation: element %d is wrong' % i
        for r in info['keep']:
            assert m.r[r] == 0xA000 + r, 'kept register R%d was written' % r
    per = {bn: (len(idx[bn]), sum(stalls[i] for i in idx[bn])) for bn in ORDER}
    return code, stalls, per, steps


if __name__ == '__main__':
    kv = dict(a.split('=') for a in sys.argv[3:])
    code, stalls, per, steps = run(sys.argv[1], sys.argv[2], **{k: int(v) for k, v in kv.items()})
    from collections import Counter
    print('instructions %d; a nonce half runs %d' % (len(code), steps))
    for bn in ORDER:
        u = Counter(x[1] for x in code if x[2] == bn)
        print('%-6s %4d instructions, %5d cycles for one warp, %s' % (bn, per[bn][0], per[bn][1], dict(u)))
