"""hand.py: the whole hand-written qc_hand_loop: build (gen.py), schedule, allocate, time, check by simulation.

  python hand.py <iface.json> <cubin> [out.txt]

Writes the instruction list (text and control fields) as JSON lines to out.txt for patch.py.
"""
import json
import random
import re
import sys

import gen
import ref
import sass

NAME = gen.NAME
RC_BASE = 0x1e8           # QC_RC_P in constant bank 3 (checked against the cubin by patch.py)
RCT_BASE = 0x290          # QC_RC_T[0], right after QC_RC_P (so RC_P[21] reads RC_T[0][0])


def build(iface):
    g = gen.Prog()
    inp = iface['inp']
    S = [('%%s%dl' % i, '%%s%dh' % i) for i in range(12)]     # s_i homes (loop-carried)
    X0 = ('%x0l', '%x0h')                                      # out0 (loop-carried)
    g.carried = set(n[1:] for p in S[1:] + [X0] for n in p) | {'L', 'H', 'k'}
    g.pairs.update(['L', 'H'])
    g.pre = {}
    for i in range(1, 12):
        g.pre[S[i][0][1:]] = inp[i][0]
        g.pre[S[i][1][1:]] = inp[i][1]
    g.pre['in0l'], g.pre['in0h'] = inp[0]
    # ---- prologue: trip 0
    g.block = 'pro'
    g.emit('UMOV UR4, 0x%x' % RC_BASE)
    g.emit('ULDC.64 UR6, c[0x3][UR4]')
    y = g.sbox(('%in0l', '%in0h'))
    g.sum12([y[0]] + [S[i][0] for i in range(1, 12)], 'L')
    g.sum12([y[1]] + [S[i][1] for i in range(1, 12)], 'H')
    L0 = g.add64('L', 'UR6')
    H0 = g.add64('H', 'UR7')
    g.dmul(y, 0, L0, H0, out=X0)
    g.emit('IADD3 %k, RZ, 0x1, RZ')
    g.emit('UIADD3 UR4, UR4, 0x8, URZ')
    # ---- loop: trips 1..21 (the DMs of the trip before, then this trip's S-box, sums and element 0)
    g.block = 'loop'
    g.emit('ULDC.64 UR6, c[0x3][UR4]')
    for i in range(1, 12):
        g.dmul(S[i], i, 'L', 'H', out=S[i])
    y = g.sbox(X0)
    g.sum12([y[0]] + [S[i][0] for i in range(1, 12)], 'L')
    g.sum12([y[1]] + [S[i][1] for i in range(1, 12)], 'H')
    L0 = g.add64('L', 'UR6')
    H0 = g.add64('H', 'UR7')
    g.dmul(y, 0, L0, H0, out=X0)
    g.emit('UIADD3 UR4, UR4, 0x8, URZ')
    g.emit('IADD3 %k, %k, 0x1, RZ')
    g.emit('ISETP.NE.AND $q, PT, %k, 0x16, PT')
    g.emit('@$q BRA .LOOP')
    # ---- epilogue: the last DMs with the terminal constants
    g.block = 'epi'
    outs = []
    for i in range(1, 12):
        Li = g.add64('L', 'c[0x3][0x%x]' % (RCT_BASE + 8 * i))
        Hi = g.add64('H', 'c[0x3][0x%x]' % (RCT_BASE + 8 * i + 4))
        outs.append(g.dmul(S[i], i, Li, Hi))
    g.final = [X0] + outs
    return g


# ------------------------------------------------------------------ operands of a virtual op
def cells(text):
    """(defs, uses) as lists of cells: 'n.0'/'n.1' for pair halves, 'n' for 32-bit names, '$n' predicates.
    The destination is the first operand, then predicate destinations (count from sass.npreds)."""
    t = text
    pred_guard = None
    m = re.match(r'@(!?)\$(\w+) (.*)$', t)
    if m:
        pred_guard, t = '$' + m[2], m[3]
    op, _, rest = t.partition(' ')
    args = [a.strip() for a in rest.split(',')] if rest and not rest.startswith('.') else []
    if op == 'BRA':
        args = []
    defs, uses = [], []
    if pred_guard:
        uses.append(pred_guard)
    if not args:
        return op, defs, uses
    nd = 1 + sass.npreds(op, [None] * len(args))
    if op.startswith('ISETP'):
        nd = 1
    if op.startswith('U'):
        nd = 0
    wide = op.startswith('IMAD.WIDE')
    for j, a in enumerate(args):
        a = a.lstrip('-~!')
        mm = NAME.fullmatch(a)
        if not mm:
            continue
        kind, name, half = mm[1], mm[2], mm[3]
        if kind == '$':
            c = ['$' + name]
        elif half:
            c = [name + half]
        else:
            # a whole pair: a WIDE's destination or its addend
            c = [name + '.0', name + '.1'] if (wide and (j == 0 or j == len(args) - 1 - (1 if op.endswith('.X') else 0))) \
                else [name]
        (defs if j < nd else uses).extend(c)
    return op, defs, uses


LAT = {('A', 'A'): 4, ('W', 'A'): 5, ('M', 'A'): 5, ('A', 'M'): 7, ('W', 'M'): 4, ('M', 'M'): 4,
       ('A', 'W'): 7, ('W', 'W'): 6, ('M', 'W'): 4, ('U', 'A'): 4, ('U', 'W'): 4, ('U', 'U'): 4, ('A', 'B'): 4}


def lat(pu, cu, cell, op_text):
    """Cycles from a producer of unit pu to a consumer of unit cu reading `cell`."""
    if cell.startswith('$'):
        return 5 if (pu == 'W' or cu in 'WM') else 4
    base = LAT.get((pu, cu), 8)
    if cu == 'W' and pu in 'AW':
        # the addend (third source) of a WIDE: ptxas keeps 9 after an ALU op and 8 after a WIDE
        return base + 2
    return base


# ------------------------------------------------------------------ scheduling
def schedule_block(ops, carried):
    """Orders one block's ops: keeps the FMA and ALU pipes level and avoids stalls."""
    n = len(ops)
    info = [cells(o.text) for o in ops]
    preds = [set() for _ in range(n)]
    last_def, readers = {}, {}
    for i, (op, d, u) in enumerate(info):
        for c in u:
            base = c.split('.')[0]
            for key in (c, base + '.0', base + '.1') if '.' not in c and not c.startswith('$') else (c,):
                if key in last_def:
                    preds[i].add((last_def[key], key))
            readers.setdefault(c, []).append(i)
        for c in d:
            if c in last_def:
                preds[i].add((last_def[c], None))       # WAW
            for r in readers.get(c, []):
                if r != i:
                    preds[i].add((r, None))             # WAR
            base = c.split('.')[0]
            for r in readers.get(base, []):
                if r != i:
                    preds[i].add((r, None))
            last_def[c] = i
            readers[c] = []
            if '.' in c:
                readers[base] = [x for x in readers.get(base, []) if False]
    # the loop's bookkeeping (counter, compare, branch, the constant pointer's step) goes after everything else
    tail = [i for i in range(n) if ops[i].unit == 'B' or ops[i].text.startswith(('ISETP', 'UIADD3', 'IADD3 %k'))]
    for i in tail:
        for j in range(n):
            if j not in tail:
                preds[i].add((j, None))
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p, _c in preds[i]:
            succ[p].append(i)
    height = [0] * n
    for i in reversed(range(n)):
        h = 0
        for s in succ[i]:
            h = max(h, height[s] + 4)
        height[i] = h + (4 if ops[i].unit == 'W' else 1)
    # register pressure: groups (pairs count 2) defined in this block and not yet fully read
    def grp(c):
        return c if c.startswith('$') else c.split('.')[0]
    uses_left = {}
    for i, (op, d, u) in enumerate(info):
        for gn in set(grp(c) for c in u):
            uses_left[gn] = uses_left.get(gn, 0) + 1
    size = {}
    for i, (op, d, u) in enumerate(info):
        for c in d:
            gn = grp(c)
            if gn not in carried and gn.lstrip('$') not in carried:
                size[gn] = 2 if ('.' in c) else 1
    live = set()
    done = [False] * n
    t_issue = [0] * n
    order, t, fma, alu = [], 0, 0.0, 0.0
    remaining = set(range(n))
    branch = [i for i in range(n) if ops[i].unit == 'B']
    while remaining:
        best, bkey = None, None
        for i in remaining:
            if any(not done[p] for p, _ in preds[i]):
                continue
            if i in branch and len(remaining) > 1:
                continue
            ready = t + 1
            for p, c in preds[i]:
                if c is not None:
                    ready = max(ready, t_issue[p] + lat(ops[p].unit, ops[i].unit, c, ops[i].text))
            stall = ready - (t + 1)
            u = ops[i].unit
            # pipe balance: prefer the op whose pipe is behind
            if u == 'W':
                bal = (fma + 2) - (alu + 1) * 1.0
            elif u == 'M':
                bal = (fma + 1) - alu
            elif u == 'A':
                bal = alu + 1 - fma
            else:
                bal = -100
            op_, d_, u_ = info[i]
            grow = sum(size.get(grp(c), 0) for c in set(grp(c) for c in d_) if grp(c) not in live and grp(c) in size)
            shrink = sum(size.get(gn, 0) for gn in set(grp(c) for c in u_) if gn in live and uses_left.get(gn, 0) == 1)
            press = sum(size[x] for x in live if not x.startswith('$'))
            pg = sum(size.get(grp(c), 0) for c in set(grp(c) for c in d_) if grp(c).startswith('$') and grp(c) not in live)
            ps = sum(1 for gn in set(grp(c) for c in u_) if gn.startswith('$') and gn in live and uses_left.get(gn, 0) == 1)
            plive = sum(1 for x in live if x.startswith('$'))
            gr = grow - pg
            sh = shrink - ps
            over = (press + gr - sh > LIMIT[0]) or (plive + pg - ps > 5)
            usesp = any(gn.startswith('$') and gn in live for gn in set(grp(c) for c in u_))
            net = (gr - sh) + (0 if usesp else 4 * (pg - ps))
            key = (over, net if over else 0, stall > 0, stall, bal > 1.5, -height[i], bal)
            if bkey is None or key < bkey:
                best, bkey = i, key
        i = best
        ready = t + 1 if order else 0
        for p, c in preds[i]:
            if c is not None:
                ready = max(ready, t_issue[p] + lat(ops[p].unit, ops[i].unit, c, ops[i].text))
        t = ready
        t_issue[i] = t
        u = ops[i].unit
        fma += 2 if u == 'W' else (1 if u == 'M' else 0)
        alu += 1 if u in 'WA' else 0
        done[i] = True
        remaining.discard(i)
        op_, d_, u_ = info[i]
        for gn in set(grp(c) for c in u_):
            uses_left[gn] = uses_left.get(gn, 0) - 1
            if uses_left[gn] <= 0:
                live.discard(gn)
        for c in d_:
            if grp(c) in size and uses_left.get(grp(c), 0) > 0:
                live.add(grp(c))
        order.append(ops[i])
    return order


# ------------------------------------------------------------------ register allocation
def allocate(seq, g, avail, npred=7):
    """seq: the whole function's ops in order. Returns {cell or pair name: physical register} and preds."""
    info = [cells(o.text) for o in seq]
    groups = {}                       # group name -> [first, last] position

    def gname(c):
        if c.startswith('$'):
            return c
        return c.split('.')[0]

    loop_idx = [i for i, o in enumerate(seq) if o.block == 'loop']
    lo_l, hi_l = (loop_idx[0], loop_idx[-1]) if loop_idx else (0, 0)
    for i, (op, d, u) in enumerate(info):
        for c in d + u:
            gn = gname(c)
            if gn not in groups:
                groups[gn] = [i, i]
            groups[gn][1] = max(groups[gn][1], i)
            groups[gn][0] = min(groups[gn][0], i)
    for gn in list(groups):
        if gn in g.carried or gn.lstrip('$') in g.carried:
            groups[gn][0] = min(groups[gn][0], lo_l)
            groups[gn][1] = max(groups[gn][1], hi_l)
        if gn in g.pre or gn in ('in0l', 'in0h'):
            groups[gn][0] = 0
    for gn in groups:
        if groups[gn][1] == groups[gn][0]:
            groups[gn][1] += 2               # a dead def still lands later: keep its register a while
    # final values: live to the end
    for p in g.final:
        for n in p:
            groups[n[1:]][1] = len(seq) + 1
    busy = {r: [] for r in avail}
    pbusy = {p: [] for p in range(npred)}
    loc = {}

    def free(r, a, b, tbl):
        return all(b < x or a >= y for x, y in tbl[r])

    # pre-coloured first
    for gn, r in g.pre.items():
        if gn in groups:
            a, b = groups[gn]
            assert free(r, a, b, busy), gn
            busy[r].append((a, b))
            loc[gn] = r
    for gn in sorted(groups, key=lambda x: (groups[x][0], x)):
        if gn in loc:
            continue
        a, b = groups[gn]
        if gn.startswith('$'):
            for p in range(npred):
                if free(p, a, b, pbusy):
                    pbusy[p].append((a, b))
                    loc[gn] = p
                    break
            else:
                raise RuntimeError('out of predicates at %d' % a)
            continue
        if gn in g.pairs:
            for r in avail:
                if r % 2 == 0 and r + 1 in busy and free(r, a, b, busy) and free(r + 1, a, b, busy):
                    busy[r].append((a, b))
                    busy[r + 1].append((a, b))
                    loc[gn] = r
                    break
            else:
                raise RuntimeError('out of register pairs at %d (%s)' % (a, gn))
        else:
            # prefer odd registers for singles, so pairs keep their even slots
            for r in sorted(avail, key=lambda r: (r % 2 == 0, r)):
                if free(r, a, b, busy):
                    busy[r].append((a, b))
                    loc[gn] = r
                    break
            else:
                raise RuntimeError('out of registers at %d (%s)' % (a, gn))
    return loc


def lower(text, loc):
    def rep(m):
        kind, name, half = m[1], m[2], m[3]
        if kind == '$':
            return 'P%d' % loc['$' + name]
        r = loc[name]
        if half:
            r += int(half[1])
        return 'R%d' % r
    return NAME.sub(rep, text)


def timing(texts, units, blocks):
    """Stall counts from the latencies, per instruction (physical registers)."""
    n = len(texts)
    ops = [sass.Op(t) if not t.endswith('.LOOP') and not t.endswith('.RET') else None for t in texts]
    t = [0] * n
    avail_r, avail_p = {}, {}       # reg -> (time ready, producer unit, writer completion)
    cur = 0
    for i in range(n):
        if i and blocks[i] != blocks[i - 1]:
            # a block boundary (the loop head or its exit): let everything land first
            m = max([v[0] for v in avail_r.values()] + [v[0] for v in avail_p.values()] + [cur + 1])
            cur = max(cur + 1, m)
            avail_r, avail_p = {}, {}
        ready = cur + 1 if i else 0
        op = ops[i]
        u = units[i]
        if op is not None:
            for j, r in enumerate(op.uregs):
                if r in avail_r:
                    tr, pu = avail_r[r]
                    role_c = op.op.startswith('IMAD.WIDE') and j >= 2
                    lt = gen_lat(pu, u, role_c)
                    ready = max(ready, tr + lt)
            for p in op.upreds:
                if p in avail_p:
                    tr, pu = avail_p[p]
                    ready = max(ready, tr + (5 if (pu == 'W' or u in 'WM') else 4))
            # write after write: land after the earlier writer
            for r in op.dregs:
                if r in avail_r:
                    tr, pu = avail_r[r]
                    ready = max(ready, tr + LATW[pu] - LATW[u] + 1)
        else:
            # the loop's branch: its predicate (a branch reads it late in the pipe; 13 is what ptxas leaves at most)
            m = re.match(r'@!?P(\d) BRA', texts[i])
            if m and int(m.group(1)) in avail_p:
                ready = max(ready, avail_p[int(m.group(1))][0] + 13)
        cur = ready
        t[i] = cur
        if op is not None:
            for r in op.dregs:
                avail_r[r] = (cur, u)
            for p in op.dpreds:
                avail_p[p] = (cur, u)
    stalls = []
    for i in range(n):
        s = (t[i + 1] - t[i]) if i + 1 < n else 1
        stalls.append(max(1, s))
    # the loop's back edge: everything the body wrote must land before the body starts again (9 covers every
    # latency above), so the branch waits out what is still in flight
    ib = [i for i in range(n) if 'BRA .LOOP' in texts[i]][0]
    lb = [i for i in range(n) if blocks[i] == 'loop']
    out_ = max(t[j] + 9 for j in lb if j != ib) - t[ib]
    stalls[ib] = max(5, min(15, out_))
    rem, k = out_ - stalls[ib], ib - 1
    while rem > 0:
        add = min(15 - stalls[k], rem)
        stalls[k] += add
        rem -= add
        k -= 1
    # the end of the function: the same for the caller (the final branch to the RET carries it)
    stalls[-1] = max(stalls[-1], 9)
    return stalls, t


LIMIT = [30]
LATW = {'A': 4, 'W': 6, 'M': 5, 'U': 4, 'B': 1, 'L': 8}


def gen_lat(pu, cu, role_c):
    base = LAT.get((pu, cu), 8)
    if cu == 'W' and pu in 'AW' and role_c:
        return base + 2
    return base


def make(iface_path, cubin, out_path=None, seed=1):
    iface = json.load(open(iface_path))
    g = build(iface)
    blocks = ['pro', 'loop', 'epi']
    seq = []
    for b in blocks:
        seq += schedule_block([o for o in g.ops if o.block == b], g.carried)
    keep = set(iface['keep']) | {1}
    avail = [r for r in iface['clob'] if r not in keep]
    loc = allocate(seq, g, avail)
    texts = [lower(o.text, loc) for o in seq]
    units = [o.unit for o in seq]
    bl = [o.block for o in seq]
    # exit: moves into the output registers (a parallel move), then the return address
    moves = []
    for k, p in enumerate(g.final):
        srcs = (loc[p[0][1:]], loc[p[1][1:]])
        for dst in iface['out'][k]:
            for h in range(2):
                if dst[h] != srcs[h]:
                    moves.append((dst[h], srcs[h]))
    mv = parallel_moves(moves, [r for r in avail if all(r != d for d, _ in moves) and
                                all(r != s for _, s in moves)])
    for d, s in mv:
        texts.append('IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (d, s))
        units.append('M')
        bl.append('exit')
    for x in iface['retfix']:
        texts.append(x)
        units.append(gen.unit_of(x))
        bl.append('exit')
    if gen.HANDV & 16:
        # diagnostic: no arithmetic, only the inputs copied to the outputs and the return (wrong results)
        mv = parallel_moves([(d[h], iface['inp'][k][h]) for k in range(12) for d in iface['out'][k] for h in range(2)
                             if d[h] != iface['inp'][k][h]], [r for r in avail if r not in
                             set(x for p in iface['inp'] for x in p) | set(x for o in iface['out'] for p in o for x in p)])
        texts = ['IMAD.MOV.U32 R%d, RZ, RZ, R%d' % m for m in mv] + list(iface['retfix'])
        units = ['M'] * len(texts)
        bl = ['loop'] + ['exit'] * (len(texts) - 1)
        return texts, [15] * len(texts), bl, list(range(len(texts))), g, loc
    stalls, t = timing(texts, units, bl)
    if gen.HANDV & 8:
        stalls = [15] * len(stalls)     # diagnostic: every instruction waits the longest
    return texts, stalls, bl, t, g, loc


def parallel_moves(moves, temps):
    """Orders (dst, src) register moves so no source is overwritten before it is read."""
    moves = [m for m in moves]
    out = []
    pending = list(moves)
    while pending:
        dsts = set(d for d, _ in pending)
        srcs = [s for _, s in pending]
        ready = [m for m in pending if m[0] not in srcs]
        if ready:
            for m in ready:
                out.append(m)
                pending.remove(m)
            continue
        # a cycle: break it through a temp
        d, s = pending[0]
        tmp = temps[0]
        out.append((tmp, s))
        pending = [(dd, tmp if ss == s else ss) for dd, ss in pending]
    return out


def simulate(texts, iface, const, s):
    m = sass.Machine(const)
    for i in range(12):
        m.r[iface['inp'][i][0]] = s[i] & 0xFFFFFFFF
        m.r[iface['inp'][i][1]] = s[i] >> 32
    for r in iface['keep']:
        m.r[r] = 0xABC0 + r
    ops = [sass.Op(t.replace('.LOOP', '`(.L)')) for t in texts]
    loop_start = None
    i = 0
    steps = 0
    while i < len(ops):
        r = m.run(ops[i])
        if r == 'BRA':
            i = LOOPI[0]
        else:
            i += 1
        steps += 1
        assert steps < 20000
    return m


LOOPI = [0]


def check(texts, bl, iface, const, trials=20):
    LOOPI[0] = bl.index('loop')
    rnd = random.Random(9)
    bad = 0
    for _ in range(trials):
        s = [rnd.randrange(ref.P) for _ in range(12)]
        m = simulate(texts, iface, const, s)
        want = ref.hand_loop(s)
        for k in range(12):
            for lo, hi in iface['out'][k]:
                got = (m.r[lo] | m.r[hi] << 32)
                if got % ref.P != want[k]:
                    bad += 1
        for x in iface['retfix']:
            d, s = re.match(r'(?:IMAD\.MOV\.U32|MOV) R(\d+), (?:RZ, RZ, )?(\S+)$', x).groups()
            want_v = 0xABC0 + int(s[1:]) if s.startswith('R') else int(s, 16)
            assert m.r[int(d)] == want_v, x
    return bad


if __name__ == '__main__':
    import iface as IF
    texts, stalls, bl, t, g, loc = make(sys.argv[1], sys.argv[2])
    const = IF.const_bank3(sys.argv[2])
    print('instructions', len(texts), 'loop', bl.count('loop'), 'cycles/loop trip (1 warp)',
          t[len(bl) - bl[::-1].index('loop') - 1] - t[bl.index('loop')])
    from collections import Counter
    print('loop units', Counter(u for u, b in zip([gen.unit_of(x) for x in texts], bl) if b == 'loop'))
    print('simulated mismatches:', check(texts, bl, json.load(open(sys.argv[1])), const))
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'w') as f:
            for x, s, b in zip(texts, stalls, bl):
                f.write(json.dumps([x, s, b]) + '\n')
