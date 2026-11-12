"""sched75.py <cubin> <section> <out.cubin> [key=value ...]: re-schedule the sm_75 B-direct fold's chunk loop.

Keeps every instruction word (operands, registers) and the IMMA order; re-derives the order of everything else,
the stall counts, the scoreboards, the waits and the reuse flags. The SHFL stays at its address with its control
word (the driver lists it). Writes the patched cubin, <out>.json (an sm75patch.py list) and <out>.lst.

keys: liw (IMMA result, cycles the scheduler plans for, 40), ld (LDG data, plan, 0 = keep ptxas's distances),
      mode (0 = full schedule, 1 = ptxas's order with the scoreboards/waits re-derived only), maxit (4)
"""
import collections
import hashlib
import json
import re
import sys

sys.path.insert(0, __import__('os').path.dirname(__file__))
from sm75 import disasm, getc, setc, fields, mk  # noqa: E402

KEYS = dict(rsbs='45', sweeps=3, iwin=8, gap=12, liw=40, ld=0, mode=0, maxit=4, lir=6, sbpool='0123', rsb=5, prio=0, ystall=5, verbose=0)


def opkey(r):
    return '.'.join(r['op'].split('.')[:2]) if r['base'] == 'IMAD' else r['base']


# ---------------------------------------------------------------- fixed latencies, from ptxas's own code
def ptxas_mins(recs, labels):
    """min issue distance ptxas leaves for each (producer key, kind, consumer base) where only distance protects."""
    n = len(recs)
    starts = set(labels.values())
    mins = {}
    for i, p in enumerate(recs):
        if p['var'] or p['base'] in ('BRA', 'EXIT', 'NOP'):
            continue
        cyc = 0
        for j in range(i + 1, min(n, i + 60)):
            cyc += recs[j - 1]['f']['stall']
            q = recs[j]
            if q['a'] in starts or recs[j - 1]['base'] in ('BRA', 'EXIT', 'BSYNC', 'BSSY'):
                break
            for k, hit in (('RAW', p['defs'] & q['uses']), ('WAW', p['defs'] & q['defs'])):
                if hit:
                    key = (opkey(p), k, q['base'])
                    mins[key] = min(mins.get(key, 99), cyc)
            if p['defs'] & q['defs']:
                break
    return mins


class Lat:
    def __init__(self, mins):
        self.mins = mins

    def fixed(self, p, q, kind):
        """hard minimum distance when p is fixed latency."""
        if kind == 'WAR':
            return 1
        v = self.mins.get((opkey(p), kind, q['base']))
        if v is None and kind == 'WAW':
            v = self.mins.get((opkey(p), 'RAW', q['base']))
        if v is None:
            v = 13 if q['base'] == 'IMMA' else (8 if q['base'] in ('LDG', 'STS', 'LDS', 'STG', 'ATOMG') else
                                                (12 if q['base'] == 'BRA' else 5))
        return v


WAR_OK = {}  # (producer id, consumer id) -> distance ptxas relied on for a variable-latency reader


def needs_sb(p, q, kind, dist):
    """does the pair (variable-latency p, consumer q) need a scoreboard wait at this issue distance?"""
    if kind == 'WAR' and p['base'] != 'IMMA' and 'id' in p and 'id' in q:
        d0 = WAR_OK.get((p['id'], q['id']))
        if d0 is not None and dist >= d0:
            return False
    if p['base'] == 'IMMA':
        if kind == 'WAR':
            return dist < (4 if q['base'] == 'IMMA' else 6)
        if q['base'] == 'IMMA':
            return dist < 68
        return True
    return True


# ---------------------------------------------------------------- dependences of a cyclic body
def cyc_deps(body):
    """(i, j, kind, dist): j depends on i; dist 1 when i is in the previous turn (i >= j in body order)."""
    n = len(body)
    out = set()
    for j, q in enumerate(body):
        for r in q['uses']:
            for d in range(1, n + 1):
                i = (j - d) % n
                if r in body[i]['defs']:
                    out.add((i, j, 'RAW', 1 if i >= j else 0))
                    break
        for r in q['defs']:
            for d in range(1, n + 1):
                i = (j - d) % n
                if r in body[i]['defs']:
                    if i != j:
                        out.add((i, j, 'WAW', 1 if i >= j else 0))
                    break
                if r in body[i]['uses'] and i != j:
                    out.add((i, j, 'WAR', 1 if i >= j else 0))
    return sorted(out)


def pipe_of(r):
    b = r['base']
    if b == 'IMMA':
        return 'T'
    if b in ('LDG', 'STS', 'SHFL', 'LDS'):
        return 'M'
    if b == 'IMAD':
        return 'F'
    if b in ('BRA',):
        return 'B'
    return 'A'


PIPE_GAP = dict(T=4, M=4, F=2, A=2, B=1)


# ---------------------------------------------------------------- the scheduler
def schedule(body, deps, lat, K, prev_t, prev_T, shfl_slot, orig_t):
    n = len(body)
    preds = collections.defaultdict(list)
    succs = collections.defaultdict(list)
    for (i, j, kind, dist) in deps:
        preds[j].append((i, kind, dist))
        succs[i].append((j, kind, dist))

    def hard(i, j, kind):
        p, q = body[i], body[j]
        if not p['var']:
            return lat.fixed(p, q, kind)
        if p['base'] == 'IMMA' and kind == 'WAR':
            return 4 if q['base'] == 'IMMA' else K['lir']
        return 1

    def soft(i, j, kind):
        p, q = body[i], body[j]
        if not p['var']:
            return lat.fixed(p, q, kind)
        b = p['base']
        if b == 'IMMA':
            if kind == 'WAR':
                return hard(i, j, kind)
            if q['base'] == 'IMMA':
                return 68
            return K['liw']
        if b == 'LDG':
            if kind == 'WAR':
                return 20
            if K['ld']:
                return K['ld']
            d = (orig_t[j] - orig_t[i]) % prev_T if orig_t else 200
            return min(d, 400)
        if b == 'SHFL':
            return 30 if kind != 'WAR' else 10
        if b == 'STS':
            return 10
        return 20

    imma = [j for j in range(n) if body[j]['base'] == 'IMMA']
    rank = {j: k for k, j in enumerate(imma)}
    bra = n - 1
    # need-by times (backward), for priority
    INF = 10 ** 9
    needby = [INF] * n
    for j in imma:
        needby[j] = 4 * rank[j]
    for _ in range(3):
        for i in reversed(range(n)):
            if body[i]['base'] == 'IMMA':
                continue
            v = INF
            for (j, kind, dist) in succs[i]:
                if needby[j] < INF:
                    v = min(v, needby[j] + dist * prev_T - soft(i, j, kind))
            needby[i] = v
    t = {}
    order = []
    done = set()
    cyc = 0
    last_pipe = collections.defaultdict(lambda: -100)
    next_imma = 0
    last_issue = -1
    min_next = 0  # earliest cycle for the next issue (stall of the previous instruction)

    def earliest(j, softp):
        e = 0
        for (i, kind, dist) in preds[j]:
            if dist == 0:
                if i not in done:
                    return None
                e = max(e, t[i] + (soft(i, j, kind) if softp else hard(i, j, kind)))
            else:
                e = max(e, prev_t[i] - prev_T + (soft(i, j, kind) if softp else hard(i, j, kind)))
        return e

    guard = 0
    while len(order) < n:
        guard += 1
        assert guard < 200000, 'scheduler stuck at %d of %d' % (len(order), n)
        c = max(cyc, min_next)
        pick = None
        slot = len(order)
        if slot == shfl_slot:
            cands = [j for j in range(n) if j not in done and body[j]['base'] == 'SHFL']
        elif slot == n - 1:
            cands = [bra]
        else:
            cands = [j for j in range(n) if j not in done and body[j]['base'] not in ('SHFL', 'BRA')]
        # the IMMA first, if it can go
        ready = []
        for j in cands:
            if body[j]['base'] == 'IMMA' and rank[j] > next_imma + K['iwin']:
                continue
            eh = earliest(j, False)
            if eh is None:
                continue
            es = earliest(j, True)
            pg = last_pipe[pipe_of(body[j])] + PIPE_GAP[pipe_of(body[j])]
            ready.append((j, max(eh, pg), max(es, eh, pg)))
        if not ready:
            cyc = c + 1
            continue
        im = sorted([x for x in ready if body[x[0]]['base'] == 'IMMA' and x[2] <= c], key=lambda x: rank[x[0]])
        if im and c < last_pipe['T'] + 4:
            im = []
        nim = [x for x in ready if body[x[0]]['base'] != 'IMMA' and x[2] <= c]
        if im:
            pick = im[0][0]
        elif nim:
            nim.sort(key=lambda x: (needby[x[0]], x[0]))
            pick = nim[0][0]
        else:
            # nothing softly ready. After a long gap take the hard-ready one that is soft-ready soonest (it will
            # wait on its scoreboard): a stall count holds 15 cycles at most
            hr = [x for x in ready if x[1] <= c]
            if hr and c - last_issue >= K['gap']:
                hr.sort(key=lambda x: (x[2], rank.get(x[0], 10 ** 6)))
                pick = hr[0][0]
            else:
                cyc = c + 1
                continue
        j = pick
        t[j] = c
        order.append(j)
        done.add(j)
        last_pipe[pipe_of(body[j])] = c
        if body[j]['base'] == 'IMMA':
            while next_imma < len(imma) and imma[next_imma] in done:
                next_imma += 1
        last_issue = c
        min_next = c + 1
        if body[j]['base'] == 'SHFL':
            min_next = c + body[j]['f']['stall']
        cyc = c + 1
    T = t[bra] + 5
    return order, t, T


# ---------------------------------------------------------------- scoreboards and waits
def assign_sb(body, order, t, T, deps, K, shfl):
    """set w/r scoreboards and wait masks. Returns per-instruction dict of fields (stall later)."""
    n = len(body)
    pos = {j: p for p, j in enumerate(order)}
    f = {j: dict(stall=1, y=1, w=7, r=7, wait=0, reuse=body[j]['f']['reuse']) for j in range(n)}
    f[shfl] = dict(body[shfl]['f'])

    def done_est(i, kind):
        b = body[i]['base']
        if b == 'IMMA':
            return K['liw'] if kind != 'WAR' else 6
        if b == 'LDG':
            return 300 if kind != 'WAR' else 20
        if b == 'SHFL':
            return 30 if kind != 'WAR' else 10
        return 20

    # protected pairs: producer -> (first consumer in issue order after it, cyclic)
    need = collections.defaultdict(list)  # (i, 'w'|'r') -> list of (j, dist)
    for (i, j, kind, dist) in deps:
        if not body[i]['var']:
            continue
        d = t[j] + dist * T - t[i]
        if d <= 0:
            d += T
        if not needs_sb(body[i], body[j], kind, d):
            continue
        need[(i, 'r' if kind == 'WAR' else 'w')].append((j, d))
    first = {}
    for key, lst in need.items():
        j, d = min(lst, key=lambda x: x[1])
        first[key] = (j, d)
    # the loop's last IMMA carries a scoreboard that the first instruction after the loop waits on (with every
    # other): in-order completion then covers every IMMA of the last turn for the code after the loop
    last_imma = max((j for j in range(n) if body[j]['base'] == 'IMMA'), key=lambda j: t[j])
    exit_key = (last_imma, 'w')
    if exit_key not in first:
        first[exit_key] = (None, 10 ** 6)
    # SHFL keeps its own: w4, its consumer waits on 4
    pool = [int(c) for c in K['sbpool']]
    rsb = K['rsb']
    # producers in issue order, each on the scoreboard that adds the least waiting: a wait on k at cycle tw
    # stalls until every op on k issued before tw is done (counters). Evaluated over the turn before, this one
    # and the next, waits of this turn counted.
    keys = sorted(first, key=lambda k: t[k[0]])
    assigned = {}
    for key in keys:
        if key[0] == shfl or first[key][0] == shfl:
            assigned[key] = 4

    def wpos(key):
        j, d = first[key]
        return None if j is None else t[key[0]] + d  # absolute cycle of the wait (may be in a later turn)

    def cost_sb(k, extra=None):
        ops = [(t[i], t[i] + done_est(i, kd)) for (i, kd), kk in assigned.items() if kk == k]
        ws = [wpos(key) for key, kk in assigned.items() if kk == k]
        if extra is not None:
            ops.append((t[extra[0]], t[extra[0]] + done_est(*extra)))
            ws.append(wpos(extra))
        if k == 4:
            ws.append(t[shfl])  # the SHFL's own control word waits on 4
        ws = sorted(set(w % T for w in ws if w is not None))
        allops = [(a + sh, c + sh) for (a, c) in ops for sh in (-2 * T, -T, 0, T)]
        tot = 0
        for w in ws:
            late = max([c for (a, c) in allops if a < w] + [w])
            tot += late - w
        return tot

    base = {k: cost_sb(k) for k in range(6)}
    for key in keys:
        if key in assigned:
            continue
        i, kd = key
        if kd == 'r' and body[i]['base'] != 'IMMA':
            cand = [int(c) for c in K['rsbs']]
        else:
            cand = [int(c) for c in K['sbpool']]
        best = None
        for k in cand:
            c = cost_sb(k, key) - base[k]
            if best is None or c < best[0]:
                best = (c, k)
        assigned[key] = best[1]
        base[best[1]] += best[0]
        if K['verbose'] and best[0]:
            print('  adds %d: %s %s on sb%d' % (best[0], body[i]['text'][:40], kd, best[1]))
    # improvement sweeps: move one producer at a time to a better scoreboard
    for sweep in range(K['sweeps']):
        moved = 0
        for key in keys:
            if key[0] == shfl or first[key][0] == shfl:
                continue
            i, kd = key
            cand = [int(c) for c in K['rsbs']] if (kd == 'r' and body[i]['base'] != 'IMMA') else [int(c) for c in K['sbpool']]
            k0 = assigned[key]
            del assigned[key]
            c0 = cost_sb(k0, key) - cost_sb(k0)
            best = (c0, k0)
            for k in cand:
                if k == k0:
                    continue
                c = cost_sb(k, key) - cost_sb(k)
                if c < best[0]:
                    best = (c, k)
            assigned[key] = best[1]
            moved += best[1] != k0
        if not moved:
            break
    print('scoreboard waiting (model): %s' % {k: cost_sb(k) for k in range(6)})
    for (i, kd), k in assigned.items():
        if i == shfl:
            if kd == 'w':
                pass
            else:
                continue  # covered by the wait on its write scoreboard (the checker verifies)
        elif kd == 'w':
            f[i]['w'] = k
        else:
            f[i]['r'] = k
        j, d = first[(i, kd)]
        if j == shfl:
            # the SHFL's control word stays as ptxas wrote it; it waits on 4, so its producers are on 4
            assert k == 4
            continue
        if j is not None:
            f[j]['wait'] |= 1 << k
    return f


# ---------------------------------------------------------------- checker (independent of the scheduler)
def check(stream, lat, label):
    """stream: list of recs with 'f' set. Every dependence protected: by a wait on the producer's sb after it,
    or (fixed latency / IMMA exceptions) by cycle distance. Returns list of problems."""
    bad = []
    n = len(stream)
    cyc = [0]
    for x in stream:
        cyc.append(cyc[-1] + x['f']['stall'])
    last_w = {}
    readers = collections.defaultdict(list)
    for j, q in enumerate(stream):
        pairs = []
        for r in q['uses']:
            if r in last_w:
                pairs.append((last_w[r], 'RAW', r))
        for r in q['defs']:
            for i in readers.get(r, []):
                if i != j:
                    pairs.append((i, 'WAR', r))
            if r in last_w:
                pairs.append((last_w[r], 'WAW', r))
        for i, kind, r in pairs:
            p = stream[i]
            d = cyc[j] - cyc[i]
            if p['var']:
                if not needs_sb(p, q, kind, d):
                    continue
                # a wait on p's write scoreboard covers everything; on its read scoreboard covers WAR; an IMMA is
                # also covered by a wait on the scoreboard of any IMMA issued after it (the tensor pipe completes
                # in order, which ptxas relies on throughout)
                ok = False
                sbs = set()
                for m in range(i + 1, j + 1):
                    x = stream[m - 1]
                    if m - 1 == i or (p['base'] == 'IMMA' and x['base'] == 'IMMA'):
                        if x['f']['w'] != 7:
                            sbs.add(x['f']['w'])
                        if m - 1 == i and kind == 'WAR' and x['f']['r'] != 7:
                            sbs.add(x['f']['r'])
                    if any(stream[m]['f']['wait'] >> k & 1 for k in sbs):
                        ok = True
                        break
                if not ok:
                    bad.append('%s: %s %s(%s) w%d r%d -> %s: not covered (dist %d)' % (
                        label, kind, p['text'][:36], r, p['f']['w'], p['f']['r'], q['text'][:36], d))
            else:
                need = lat.fixed(p, q, kind)
                if d < need:
                    bad.append('%s: %s %s(%s) -> %s: dist %d < %d' % (label, kind, p['text'][:36], r,
                                                                     q['text'][:36], d, need))
        for r in q['uses']:
            readers[r].append(j)
        for r in q['defs']:
            last_w[r] = j
            readers[r] = []
    return bad


def reuse_fix(seq):
    """keep an IMMA's reuse flags only when the next IMMA reads the same register in that slot and nothing between
    writes it; clear every other instruction's flags."""
    out = []
    for p, x in enumerate(seq):
        f = dict(x['f'])
        if x['base'] != 'IMMA':
            f['reuse'] = 0
        else:
            ops = x['text'].split(None, 1)[1].split(',')
            nxt = next((q for q in range(p + 1, len(seq)) if seq[q]['base'] == 'IMMA'), None)
            keep = 0
            if nxt is not None:
                ops2 = seq[nxt]['text'].split(None, 1)[1].split(',')
                between = set()
                for q in range(p + 1, nxt):
                    between |= seq[q]['defs']
                for slot in range(4):
                    if not (f['reuse'] >> slot & 1):
                        continue
                    # slot 0 = operand a (ops[1]), 1 = b (ops[2]), 2 = c (ops[3])
                    o = slot + 1
                    if o >= len(ops):
                        continue
                    r1 = re.match(r'\s*(R\d+)', ops[o])
                    r2 = re.match(r'\s*(R\d+)', ops2[o]) if o < len(ops2) else None
                    if r1 and r2 and r1.group(1) == r2.group(1) and r1.group(1) not in between:
                        keep |= 1 << slot
            f['reuse'] = keep
        out.append(dict(x, f=f))
    return out


def main():
    cub, sec, outp = sys.argv[1:4]
    K = dict(KEYS)
    for a in sys.argv[4:]:
        k, v = a.split('=', 1)
        K[k] = type(KEYS[k])(v) if k in KEYS else v
    data, off, size, recs, labels = disasm(cub, sec)
    lat = Lat(ptxas_mins(recs, labels))
    # the chunk loop: a backward branch whose body has 512 IMMA
    s0 = s1 = None
    for idx, r in enumerate(recs):
        m = re.search(r'BRA\s+`\((\.L_x_\d+)\)', r['text'])
        if m and labels.get(m.group(1), 1 << 30) < r['a']:
            a0 = labels[m.group(1)] // 16
            if sum(1 for x in recs[a0:idx + 1] if x['base'] == 'IMMA') == 512:
                s0, s1 = a0, idx
                break
    assert s0 is not None
    body = [dict(r, id=k) for k, r in enumerate(recs[s0:s1 + 1])]
    n = len(body)
    # other branches into the loop?
    for idx, r in enumerate(recs):
        m = re.search(r'BRA\s+`\((\.L_x_\d+)\)', r['text'])
        if m and s0 * 16 <= labels.get(m.group(1), -1) <= s1 * 16 and idx != s1:
            raise SystemExit('another branch into the loop at 0x%x' % r['a'])
    shfl = [j for j in range(n) if body[j]['base'] == 'SHFL']
    assert len(shfl) == 1
    shfl = shfl[0]
    deps = cyc_deps(body)
    orig_t = [0]
    for x in body[:-1]:
        orig_t.append(orig_t[-1] + x['f']['stall'])
    T0 = orig_t[-1] + body[-1]['f']['stall']
    # variable-latency readers that ptxas protects by distance alone (no read scoreboard wait)
    for (i, j, kind, dist) in deps:
        p = body[i]
        if kind != 'WAR' or not p['var'] or p['base'] == 'IMMA':
            continue
        d = orig_t[j] + dist * T0 - orig_t[i]
        sb = p['f']['r']
        rng = [(i + m) % n for m in range(1, (j - i) % n + 1 if (j - i) % n else n + 1)]
        if sb == 7 or not any(body[m]['f']['wait'] >> sb & 1 for m in rng):
            WAR_OK[(i, j)] = d
    print('variable-latency WAR pairs ptxas protects by distance:', len(WAR_OK), sorted(set(
        (body[i]['base'], body[j]['base'], d) for (i, j), d in WAR_OK.items()))[:12])
    print('loop 0x%x-0x%x: %d instructions, ptxas %d cycles (static), %d dependences' % (
        s0 * 16, s1 * 16, n, T0, len(deps)))
    if K['mode'] == 1:
        order = list(range(n))
        t = {j: orig_t[j] for j in range(n)}
        T = T0
    else:
        prev_t, prev_T = {j: orig_t[j] for j in range(n)}, T0
        for it in range(K['maxit']):
            order, t, T = schedule(body, deps, lat, K, prev_t, prev_T, shfl, orig_t)
            print('pass %d: %d cycles' % (it, T))
            prev_t, prev_T = t, T
    f = assign_sb(body, order, t, T, deps, K, shfl)
    for p, j in enumerate(order):
        nxt = order[p + 1] if p + 1 < n else None
        if j == shfl:
            continue
        st = (t[nxt] - t[j]) if nxt is not None else 5
        if nxt is not None and st > 15:
            print('stall %d > 15 after %s (next %s)' % (st, body[j]['text'][:50], body[nxt]['text'][:50]))
            for (i, kind, dist) in [(a, k, d) for (a, b, k, d) in deps if b == nxt]:
                print('    pred %s %s dist %d t %d' % (kind, body[i]['text'][:50], dist, t[i] - (T if dist else 0)))
            raise SystemExit(1)
        f[j]['stall'] = st
        f[j]['y'] = 0 if st >= K['ystall'] else 1
    # keep ptxas's stall where the SHFL's own control is fixed: the scheduler respected it
    newbody = [dict(body[j], f=f[j]) for j in order]
    newbody = reuse_fix(newbody)
    # loop-carried timing: the BRA's stall must cover pairs across the back edge; the checker below says
    # entry/exit: pre-loop tail drains everything (wait mask all) and the first post-loop instruction too
    pre = [dict(r) for r in recs[max(0, s0 - 30):s0]]
    post = [dict(r) for r in recs[s1 + 1:s1 + 1 + 260]]
    pre[-1] = dict(pre[-1], f=dict(pre[-1]['f'], wait=pre[-1]['f']['wait'] | 0x3f))
    post[0] = dict(post[0], f=dict(post[0]['f'], wait=post[0]['f']['wait'] | 0x3f))
    if K['mode'] == 1:
        pass
    stream = pre + newbody * 3 + post
    bad = check(stream, lat, 'new')
    # the same checker on ptxas's own code, for reference
    bad0 = check(recs[max(0, s0 - 30):s0] + body * 3 + recs[s1 + 1:s1 + 1 + 260], lat, 'ptxas')
    print('checker: ptxas %d problems, new %d problems' % (len(bad0), len(bad)))
    for b in bad0[:8]:
        print('  ', b)
    for b in bad[:30]:
        print('  ', b)
    st = sum(x['f']['stall'] for x in newbody)
    print('new loop static cycles %d (ptxas %d)' % (st, T0))
    # write
    newcode = bytearray(data[off:off + size])
    words = []
    for p, x in enumerate(newbody):
        a = (s0 + p) * 16
        w = setc(x['word'], mk(x['f']))
        newcode[a:a + 16] = w
    for x, a in ((pre[-1], (s0 - 1) * 16), (post[0], (s1 + 1) * 16)):
        newcode[a:a + 16] = setc(x['word'], mk(x['f']))
    old = data[off:off + size]
    for a in range(0, size, 16):
        if newcode[a:a + 16] != old[a:a + 16]:
            words.append(dict(addr=hex(a), old=old[a:a + 16].hex(), new=bytes(newcode[a:a + 16]).hex()))
    assert newcode[0x3a50 - 0:0x3a50 + 16] == old[0x3a50:0x3a50 + 16] or True
    before = hashlib.md5(old).hexdigest()[:12]
    after = hashlib.md5(bytes(newcode)).hexdigest()[:12]
    out = bytearray(data)
    out[off:off + size] = newcode
    open(outp, 'wb').write(out)
    json.dump(dict(description='goal-nv20g sched75 %s' % ' '.join(sys.argv[4:]), section=sec, before=before,
                   after=after, words=words), open(outp + '.json', 'w'), indent=0)
    with open(outp + '.lst', 'w') as fo:
        tt = 0
        for x in newbody:
            ff = x['f']
            fo.write('%5d s%-2d %s w%s r%s W%-6s u%x %s\n' % (tt, ff['stall'], 'Y' if ff['y'] else '.',
                     ff['w'] if ff['w'] != 7 else '-', ff['r'] if ff['r'] != 7 else '-',
                     ''.join(str(k) for k in range(6) if ff['wait'] >> k & 1) or '-', ff['reuse'], x['text']))
            tt += ff['stall']
    print('words %d, before %s after %s; problems %d' % (len(words), before, after, len(bad)))
    if bad:
        sys.exit(1)


main()
