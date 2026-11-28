"""full.py: hand-written full rounds for sm_89, in place of the compiled bodies of qc_hand_fulls<WHICH, R0, N>.

  python full.py <cubin (carryfix applied, built with -DQC_HAND89=2)> <out.cubin>

Each function is straight-line code for N rounds: twelve S-boxes (gen.py's square and fold multiply), then the
external layer on three-word values (lo, hi, top) with the round constants as immediates, then one IMAD.WIDE a
value to fold the top word back (top * EPS + (hi:lo)).

The layer: M4 on each group of four (nine adds), the column sums across the three groups, then value + column +
constant. An add of two or three values is one IADD3 for the low words, one IADD3.X for the high words with the low
add's carries as its carry-ins, and one op for the top words with the high add's carries (an IMAD.X when the add
has two inputs and at most one top word, so it rides on the multiplier). No value is ever a pair of 64-bit sums.

The compiled function's registers are found from its marks (qc_kernel.cuh): each element goes in through
IMAD.WIDE R, z, QC_MARK_IN + 2 i, <its pair> and out through the same with QC_MARK_OUT.
"""
import os
import random
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enc  # noqa: E402
import gen  # noqa: E402
import hand  # noqa: E402
import iface as IF  # noqa: E402
import patch  # noqa: E402
import ref  # noqa: E402
import sass  # noqa: E402

MARK_IN, MARK_OUT = 0x51ED0001, 0x0A11CE01
P = ref.P
RC_F1 = ref._arr('QC_RC_F1')
assert len(RC_F1) == 48 and len(ref.RC_T) == 60
LEAD = float(os.environ.get('QC_FLEAD', '3'))
PLIM = int(os.environ.get('QC_FPLIM', '6'))
FTOP = int(os.environ.get('QC_FTOP', '0'))     # 1, 2: top-word sums on IMAD.X (see add3w)
FN = re.compile(r'^\$(\S*qc_hand_fulls\S*):')
ARGS = re.compile(r'qc_hand_fullsILi(\d+)ELi(\d+)ELi(\d+)E')


def rc_of(which, r):
    t = ref.RC_T if which else RC_F1
    return t[12 * r:12 * r + 12]


# ------------------------------------------------------------------ the reference
def ext(s):
    def m4(x):
        return [2 * x[0] + 3 * x[1] + x[2] + x[3], x[0] + 2 * x[1] + 3 * x[2] + x[3],
                x[0] + x[1] + 2 * x[2] + 3 * x[3], 3 * x[0] + x[1] + x[2] + 2 * x[3]]
    m = [m4(s[4 * g:4 * g + 4]) for g in range(3)]
    col = [m[0][k] + m[1][k] + m[2][k] for k in range(4)]
    return [(m[g][k] + col[k]) % P for g in range(3) for k in range(4)]


def ref_fulls(s, which, r0, n):
    s = [x % P for x in s]
    for r in range(r0, r0 + n):
        s = ext([pow(x, 7, P) for x in s])
        s = [(x + c) % P for x, c in zip(s, rc_of(which, r))]
    return s


# ------------------------------------------------------------------ the compiled functions
def functions(cubin):
    """{name: (entry address, rows of its body up to its RET)} and all rows, from nvdisasm."""
    r = subprocess.run([IF.NVDISASM[0], '-hex', '-c', cubin], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    L = r.stdout.decode('utf-8', 'replace').splitlines()
    rows, starts, pend = [], [], None
    for i, l in enumerate(L):
        m = FN.match(l.strip())
        if m:
            pend = m[1]
            continue
        m = IF.LINE.match(l)
        if m:
            rows.append(dict(addr=int(m[1], 16), text=re.sub(r'\s+', ' ', m[2]), lo=int(m[3], 16),
                             hi=int(IF.HIGH.match(L[i + 1])[1], 16)))
            if pend:
                starts.append((pend, len(rows) - 1))
                pend = None
    out = {}
    for name, k in starts:
        e = next(j for j in range(k, len(rows)) if rows[j]['text'].startswith('RET.REL'))
        out[name] = (rows[k]['addr'], rows[k:e + 1])
    return out, rows


def find_iface(body):
    ops = []
    for r in body:
        try:
            ops.append(sass.Op(r['text'].replace('.reuse', '')))
        except Exception:
            ops.append(None)
    rnd = random.Random(7)
    m = sass.Machine({})
    vals = rnd.sample(range(1 << 20, 1 << 31), 255)
    for k in range(255):
        m.r[k] = vals[k]
    inv = {v: k for k, v in enumerate(vals)}
    inp = {}
    for op in ops:
        if op is None:
            continue
        if op.op.startswith(('BRA', 'RET', 'CALL')):
            break
        if op.op == 'IMAD.WIDE.U32' and len(op.srcs) == 3 and op.srcs[1].kind == 'i' and \
                0 <= op.srcs[1].n - MARK_IN < 24 and op.srcs[2].kind == 'r':
            a = op.srcs[2].n
            inp[(op.srcs[1].n - MARK_IN) // 2] = [inv[m.r[a]], inv[m.r[a + 1]]]
            if len(inp) == 12:
                break
        try:
            m.run(op)
        except Exception:
            for d in op.dregs:
                m.r[d] = rnd.getrandbits(32) | 1 << 31       # an op the simulator lacks: its result is no input
    assert len(inp) == 12, sorted(inp)
    # outputs: from the first out mark to the RET, simulated; each mark's pair gets two tokens in place of its
    # value, and the registers holding a token at the RET are that word's (ptxas moves and XOR-swaps them)
    first = next(k for k, op in enumerate(ops) if op is not None and op.op == 'IMAD.WIDE.U32' and len(op.srcs) == 3
                 and op.srcs[1].kind == 'i' and 0 <= op.srcs[1].n - MARK_OUT < 24)
    m = sass.Machine({})
    for k in range(255):
        m.r[k] = vals[k]
    TOK = 0x7E000000
    for op in ops[first:]:
        if op is None:
            continue
        if op.op.startswith('RET'):
            break
        if op.op == 'IMAD.WIDE.U32' and len(op.srcs) == 3 and op.srcs[1].kind == 'i' and                 0 <= op.srcs[1].n - MARK_OUT < 24:
            i = (op.srcs[1].n - MARK_OUT) // 2
            m.r[op.dregs[0]], m.r[op.dregs[1]] = TOK + 2 * i, TOK + 2 * i + 1
            continue
        try:
            m.run(op)
        except Exception:
            for d in op.dregs:
                m.r[d] = rnd.getrandbits(30) | 1 << 31
    out = []
    for i in range(12):
        lo = [r for r in range(255) if m.r[r] == TOK + 2 * i]
        hi = [r for r in range(255) if m.r[r] == TOK + 2 * i + 1]
        assert lo and hi, i
        n = max(len(lo), len(hi))
        out.append([[lo[k % len(lo)], hi[k % len(hi)]] for k in range(n)])
    ret = body[-1]
    retreg = int(re.match(r'RET\.REL\.NODEC R(\d+)', ret['text'])[1])
    regs_w, retfix, keep = set(), [], set()
    for op in ops:
        if op is not None:
            regs_w.update(op.dregs)
    for rr in (retreg, retreg + 1):
        last = None
        for op in ops:
            if op is not None and rr in op.dregs:
                last = op
        if last is not None:
            assert last.op in ('IMAD.MOV.U32', 'MOV') and len(last.dregs) == 1, last.text
            # with one call site ptxas writes the return address inside the callee (MOV Rk, <address>) and moves
            # it to the RET's register at the end: that MOV is part of the protocol too
            for src in last.uregs:
                w = [op for op in ops[:ops.index(last)] if op is not None and src in op.dregs]
                if w:
                    assert w[-1].op == 'MOV' and w[-1].srcs[0].kind == 'i', w[-1].text
                    retfix.append(w[-1].text)
                else:
                    keep.add(src)
            retfix.append(last.text)
        else:
            keep.add(rr)
    return dict(inp=[inp[i] for i in range(12)], out=out, retfix=retfix, keep=sorted(keep), clob=sorted(regs_w))


# ------------------------------------------------------------------ the program
class FProg(gen.Prog):
    def add3w(self, ops, pair=None):
        """Sum of two or three (lo, hi, top) values (top None for a two-word value or a constant; a constant's
        words are immediates and it must come second). Returns (lo, hi, top) names; lo and hi are the halves of
        the pair `pair` when given."""
        n = len(ops)
        if pair:
            self.pairs.add(pair)
            dlo, dhi = '%%%s.0' % pair, '%%%s.1' % pair
        else:
            dlo, dhi = '%' + self.tmp('l'), '%' + self.tmp('h')
        dtop = '%' + self.tmp('t')
        p1, q1 = self.pred(), self.pred()
        lo = [o[0] for o in ops] + ['RZ'] * (3 - n)
        hi = [o[1] for o in ops] + ['RZ'] * (3 - n)
        tops = [o[2] for o in ops if o[2]]
        if n == 3:
            p2, q2 = self.pred(), self.pred()
            self.emit('IADD3 %s, %s, %s, %s' % (dlo, p1, p2, ', '.join(lo)))
            self.emit('IADD3.X %s, %s, %s, %s, %s, %s' % (dhi, q1, q2, ', '.join(hi), p1, p2))
            if FTOP & 2 and tops:
                # the top words' sum as two IMAD.X (%one holds 1): off the ALU, which binds
                u = '%' + self.tmp('t')
                self.emit('IMAD.X %s, %s, %%one, %s, %s' % (u, tops[0], tops[1] if len(tops) > 1 else 'RZ', q1))
                self.emit('IMAD.X %s, %s, %%one, %s, %s' % (dtop, u, tops[2] if len(tops) > 2 else 'RZ', q2))
            else:
                self.emit('IADD3.X %s, %s, %s, %s' % (dtop, ', '.join(tops + ['RZ'] * (3 - len(tops))), q1, q2))
        else:
            self.emit('IADD3 %s, %s, %s' % (dlo, p1, ', '.join(lo)))
            self.emit('IADD3.X %s, %s, %s, %s, !PT' % (dhi, q1, ', '.join(hi), p1))
            if len(tops) < 2:
                self.emit('IMAD.X %s, RZ, RZ, %s, %s' % (dtop, tops[0] if tops else 'RZ', q1))
            elif FTOP & 1:
                self.emit('IMAD.X %s, %s, %%one, %s, %s' % (dtop, tops[0], tops[1], q1))
            else:
                self.emit('IADD3.X %s, %s, %s, RZ, %s, !PT' % (dtop, tops[0], tops[1], q1))
        return dlo, dhi, dtop

    def layer(self, y, rc):
        m = []
        for g in range(3):
            x0, x1, x2, x3 = [(a, b, None) for a, b in y[4 * g:4 * g + 4]]
            t01 = self.add3w([x0, x1])
            t23 = self.add3w([x2, x3])
            t0123 = self.add3w([t01, t23])
            t01123 = self.add3w([t0123, x1])
            t01233 = self.add3w([t0123, x3])
            y3 = self.add3w([t01233, x0, x0])
            y1 = self.add3w([t01123, x2, x2])
            y0 = self.add3w([t01123, t01])
            y2 = self.add3w([t01233, t23])
            m.append([y0, y1, y2, y3])
        new = [None] * 12
        for k in range(4):
            col = self.add3w([m[0][k], m[1][k], m[2][k]])
            for g in range(3):
                c = rc[4 * g + k]
                Z = self.tmp('Z')
                if c:
                    z = self.add3w([m[g][k], (gen.hx(c & 0xFFFFFFFF), gen.hx(c >> 32), None), col], pair=Z)
                else:
                    z = self.add3w([m[g][k], col], pair=Z)
                E = self.wide(z[2], gen.EPS, Z)
                new[4 * g + k] = ('%%%s.0' % E, '%%%s.1' % E)
        return new


def build(info, which, r0, n):
    g = FProg()
    g.carried, g.pre, g.block = set(), {}, 'main'
    s = []
    for i in range(12):
        g.pre['x%dl' % i], g.pre['x%dh' % i] = info['inp'][i]
        s.append(('%%x%dl' % i, '%%x%dh' % i))
    bounds = []
    if FTOP:
        g.emit('IADD3 %one, RZ, 0x1, RZ')
    diag = int(os.environ.get('QC_FDIAG', '0'))
    if diag == 1:
        # diagnostic (wrong results): no rounds, each input through one IMAD.WIDE by one with a zero addend
        new = []
        for x in s:
            E = g.wide(x[0], '0x1', None)
            F = g.wide(x[1], '0x1', None)
            new.append(('%%%s.0' % E, '%%%s.0' % F))
        g.final = new
        return g, [len(g.ops)]
    for r in range(r0, r0 + n):
        s = g.layer([g.sbox(x) for x in s], rc_of(which, r))
        bounds.append(len(g.ops))
    g.final = s
    return g, bounds


# ------------------------------------------------------------------ order, registers, timing
def order(ops):
    """Two queues in the order written, merged by each pipe's share of the block, the multiplies LEAD ahead."""
    n = len(ops)
    info = [hand.cells(o.text) for o in ops]
    last_def, readers, npred, succ = {}, {}, [0] * n, [[] for _ in range(n)]

    def dep(a, b):
        if a != b and b not in succ[a]:
            succ[a].append(b)
            npred[b] += 1
    for i, (op, d, u) in enumerate(info):
        for c in u:
            base = c.split('.')[0]
            for key in ((c, base + '.0', base + '.1') if '.' not in c and not c.startswith('$') else (c,)):
                if key in last_def:
                    dep(last_def[key], i)
            readers.setdefault(c, []).append(i)
        for c in d:
            if c in last_def:
                dep(last_def[c], i)
            for r in readers.get(c, []):
                dep(r, i)
            last_def[c] = i
            readers[c] = []
    cls = ['W' if o.unit == 'W' else 'A' for o in ops]
    q = {'W': [i for i in range(n) if cls[i] == 'W'], 'A': [i for i in range(n) if cls[i] == 'A']}
    tot = {k: len(v) for k, v in q.items()}
    placed = {'W': 0, 'A': 0}
    done = [False] * n
    uses_left = {}
    for op, d, u in info:
        for c in set(u):
            if c.startswith('$'):
                uses_left[c] = uses_left.get(c, 0) + 1
    live = set()
    out = []
    while len(out) < n:
        cand = {}
        for k in 'WA':
            for i in q[k][:16]:
                if npred[i] == 0:
                    cand[k] = i
                    break
        best = None
        for k, i in cand.items():
            deficit = tot[k] * (len(out) + 1) / float(n) - placed[k] + (LEAD if k == 'W' else 0)
            newp = sum(1 for c in info[i][1] if c.startswith('$') and c not in live)
            freed = sum(1 for c in set(info[i][2]) if c.startswith('$') and uses_left.get(c) == 1)
            key = (len(live) + newp - freed > PLIM, -deficit, i)
            if best is None or key < best[0]:
                best = (key, k, i)
        assert best, 'stuck at %d' % len(out)
        _, k, i = best
        q[k].remove(i)
        placed[k] += 1
        done[i] = True
        for c in set(info[i][2]):
            if c.startswith('$'):
                uses_left[c] -= 1
                if uses_left[c] == 0:
                    live.discard(c)
        for c in info[i][1]:
            if c.startswith('$') and uses_left.get(c, 0) > 0:
                live.add(c)
        for x in succ[i]:
            npred[x] -= 1
        out.append(ops[i])
    return out


def timing(texts, units):
    n = len(texts)
    ops = [sass.Op(t) for t in texts]
    t = [0] * n
    avail_r, avail_p = {}, {}
    cur = 0
    for i in range(n):
        ready = cur + 1 if i else 0
        op, u = ops[i], units[i]
        for kind, r, role in hand.use_roles(op):
            tbl = avail_r if kind == 'r' else avail_p
            if r in tbl:
                tr, pu, pc, dk = tbl[r]
                ready = max(ready, tr + hand.lat2(pc, dk, hand.op_class(op.op), role))
        for r in op.dregs:
            if r in avail_r:
                tr, pu = avail_r[r][:2]
                ready = max(ready, tr + hand.LATW[pu] - hand.LATW[u] + 1)
        cur = ready
        t[i] = cur
        pc = hand.op_class(op.op)
        for k, r in enumerate(op.dregs):
            avail_r[r] = (cur, u, pc, ('lo', 'hi')[k] if len(op.dregs) == 2 else 'r')
        for p in op.dpreds:
            avail_p[p] = (cur, u, pc, 'p')
    return [max(1, min(15, (t[i + 1] - t[i]) if i + 1 < n else 9)) for i in range(n)]


def make(info, which, r0, n):
    g, bounds = build(info, which, r0, n)
    seq, a = [], 0
    for b in bounds:                      # a round at a time: the next round's S-boxes need this one's folds
        seq += order(g.ops[a:b])
        a = b
    keep = set(info['keep']) | {1}
    avail = [r for r in info['clob'] if r not in keep]
    final = g.final
    g.final = [(x.split('.')[0], x.split('.')[0]) for p in final for x in p]     # allocate() wants group names
    loc = hand.allocate(seq, g, avail)
    g.final = final
    texts = [hand.lower(o.text, loc) for o in seq]
    units = [o.unit for o in seq]
    moves = []
    for k, p in enumerate(g.final):
        cell = [re.match(r'%(\w+)\.([01])', x) for x in p]
        srcs = [loc[c[1]] + int(c[2]) for c in cell]
        for dst in info['out'][k]:
            for h in range(2):
                if dst[h] != srcs[h]:
                    moves.append((dst[h], srcs[h]))
    moves = sorted(set(moves))
    used = set(d for d, _ in moves) | set(s for _, s in moves)
    for d, s in hand.parallel_moves(moves, [r for r in avail if r not in used]):
        texts.append('IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (d, s))
        units.append('M')
    nbody = len(texts)
    for x in info['retfix']:
        texts.append(x)
        units.append(gen.unit_of(x))
    stalls = timing(texts, units)
    stalls[-1] = 9
    return texts, stalls, units, nbody


def check(texts, info, which, r0, n, trials=6):
    rnd = random.Random(11)
    ops = [sass.Op(t) for t in texts]
    bad = 0
    for _ in range(trials):
        s = [rnd.randrange(1 << 64) for _ in range(12)]
        m = sass.Machine({})
        for i in range(12):
            m.r[info['inp'][i][0]], m.r[info['inp'][i][1]] = s[i] & 0xFFFFFFFF, s[i] >> 32
        for op in ops:
            m.run(op)
        want = ref_fulls(s, which, r0, n)
        for k in range(12):
            for lo, hi in info['out'][k]:
                if (m.r[lo] | m.r[hi] << 32) % P != want[k]:
                    bad += 1
    return bad


def run(cubin, out):
    fns, rows = functions(cubin)
    assert len(fns) == 4, sorted(fns)
    T = enc.Templates([rows])
    b = bytearray(open(cubin, 'rb').read())
    off, _size = patch.text_section(b)
    base = rows[0]['addr']
    want = {}
    only = os.environ.get('QC_FONLY')
    for k, name in enumerate(sorted(fns)):
        if only and str(k) not in only:
            continue
        entry, body = fns[name]
        which, r0, n = [int(x) for x in ARGS.search(name).groups()]
        info = find_iface(body)
        texts, stalls, units, nbody = make(info, which, r0, n)
        bad = 0 if os.environ.get('QC_FDIAG') else check(texts, info, which, r0, n)
        assert bad == 0, '%s: simulation: %d mismatches' % (name, bad)
        ret = body[-1]['addr']
        room = (ret - entry) // 16
        code = list(zip(texts, stalls)) + [('BRA .RET', 9)]
        assert len(code) <= room, (name, len(code), room)
        for i in range(room):
            a = entry + 16 * i
            if i < len(code):
                x, s = code[i]
                lo, hi = patch.bra(x, a, {'.RET': ret}) if 'BRA' in x else T.get(x)
                hi = (hi & ~enc.CTRL_MASK) | (patch.ctrl(min(s, 15), s < 4) << 41)
                want[a] = x
            else:
                lo, hi = 0x7918, patch.ctrl(0, False) << 41
            struct.pack_into('<QQ', b, off + a - base, lo, hi)
        from collections import Counter
        c = Counter(units[:nbody])
        print('%s: which %d rounds %d..%d: %d instructions of %d (W %d, ALU %d, IMAD %d); simulation ok' % (
            'qc_hand_fulls', which, r0, r0 + n - 1, len(code), room, c['W'], c['A'], c['M']))
    open(out, 'wb').write(b)
    got = {r['addr']: r['text'].replace('.reuse', '') for r in functions(out)[1]}
    nbad = 0
    for a, x in want.items():
        h = got.get(a, '?')
        ok = (h.split('BRA')[0] == x.split('BRA')[0]) if 'BRA' in x else h == x
        if not ok:
            nbad += 1
            if nbad < 10:
                print('MISMATCH %05x asked %r got %r' % (a, x, h))
    assert nbad == 0, nbad
    return out


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
