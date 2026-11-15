"""climb.py <cubin> <section> <out.cubin> [key=value]: ptxas's chunk loop, same order, stalls, yield and reuse bits;
only scoreboard numbers and wait masks change. Moves: drop a wait bit; move a producer's write or read scoreboard
to another number (and add that number to the waits that then need it). A move is kept only when sched75's
independent checker finds every dependence covered over three turns plus the code before and after, and the
two-warp model (sim2's) gets faster. keys: L (IMMA result latencies for the model, '32,40'), iters (40),
moves (removals only: 'rm'; both: 'all')."""
import sys, os, re, json, hashlib, random, collections
D = r'C:\Users\template\AppData\Local\Temp\claude\18634d96a113c206\5675ce09-95f3-4f49-ac47-74a8f5730983\scratchpad\g-nv20i\earn\native\probes\turing\sched'
sys.path.insert(0, D)
src = open(os.path.join(D, 'sched75.py')).read().rsplit('\nmain()', 1)[0]
g = {'__file__': os.path.join(D, 'sched75.py'), '__name__': 'sched75'}
exec(compile(src, 'sched75', 'exec'), g)
disasm, setc, mk, Lat, ptxas_mins, cyc_deps, needs_sb = (g[k] for k in (
    'disasm', 'setc', 'mk', 'Lat', 'ptxas_mins', 'cyc_deps', 'needs_sb'))
WAR_OK = g['WAR_OK']
cub, sec, outp = sys.argv[1:4]
opt = dict(a.split('=', 1) for a in sys.argv[4:])
LS = [int(x) for x in opt.get('L', '32,40').split(',')]
ITERS = int(opt.get('iters', 40))
MOVES = opt.get('moves', 'all')
data, off, size, recs, labels = disasm(cub, sec)
lat = Lat(ptxas_mins(recs, labels))
for idx, r in enumerate(recs):
    m = re.search(r'BRA\s+`\((\.L_x_\d+)\)', r['text'])
    if m and labels.get(m.group(1), 1 << 30) < r['a']:
        a0 = labels[m.group(1)] // 16
        if sum(1 for x in recs[a0:idx + 1] if x['base'] == 'IMMA') == 512:
            s0, s1 = a0, idx
            break
body = [dict(r, id=k, f=dict(r['f'])) for k, r in enumerate(recs[s0:s1 + 1])]
n = len(body)
deps = cyc_deps(body)
orig_t = [0]
for x in body[:-1]:
    orig_t.append(orig_t[-1] + x['f']['stall'])
T0 = orig_t[-1] + body[-1]['f']['stall']
for (i, j, kind, dist) in deps:
    p = body[i]
    if kind != 'WAR' or not p['var'] or p['base'] == 'IMMA':
        continue
    d = orig_t[j] + dist * T0 - orig_t[i]
    sb = p['f']['r']
    rng = [(i + m) % n for m in range(1, (j - i) % n + 1 if (j - i) % n else n + 1)]
    if sb == 7 or not any(body[m]['f']['wait'] >> sb & 1 for m in rng):
        WAR_OK[(i, j)] = d
pre = [dict(r) for r in recs[max(0, s0 - 30):s0]]
post = [dict(r) for r in recs[s1 + 1:s1 + 1 + 260]]
P = len(pre)
shfl = [j for j in range(n) if body[j]['base'] == 'SHFL'][0]


def problems(bd):
    """uncovered pairs as (producer body index or None, consumer body index or None, kind)."""
    stream = pre + bd * 3 + post
    out = []
    cyc = [0]
    for x in stream:
        cyc.append(cyc[-1] + x['f']['stall'])
    last_w = {}
    readers = collections.defaultdict(list)

    def bi(s):
        return (s - P) % n if P <= s < P + 3 * n else None
    for j, q in enumerate(stream):
        pairs = []
        for r in q['uses']:
            if r in last_w:
                pairs.append((last_w[r], 'RAW'))
        for r in q['defs']:
            for i in readers.get(r, []):
                if i != j:
                    pairs.append((i, 'WAR'))
            if r in last_w:
                pairs.append((last_w[r], 'WAW'))
        for i, kind in pairs:
            p = stream[i]
            d = cyc[j] - cyc[i]
            if p['var']:
                if not needs_sb(p, q, kind, d):
                    continue
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
                    out.append((bi(i), bi(j), kind))
            else:
                if d < lat.fixed(p, q, kind):
                    out.append((bi(i), bi(j), 'FIX'))
        for r in q['uses']:
            readers[r].append(j)
        for r in q['defs']:
            last_w[r] = j
            readers[r] = []
    return out


def sim(bd, phases=4, turns=16):
    """sim2's model: two warps share one tensor pipe (an IMMA every 4 cycles) and one issue slot; counters."""
    recs2 = [dict(s=x['f']['stall'], y=bool(x['f']['y']), w=None if x['f']['w'] == 7 else x['f']['w'],
                  r=None if x['f']['r'] == 7 else x['f']['r'], W=[k for k in range(6) if x['f']['wait'] >> k & 1],
                  imma=x['base'] == 'IMMA', op=x['base']) for x in bd]
    tot = []
    for LIW in LS:
        rnd = random.Random(1)
        for phase in range(phases):
            NW = 2
            warps = []
            for k in range(NW):
                start = int(rnd.random() * n) if k else 0
                warps.append(dict(pc=start, ready=0, pend=collections.defaultdict(list), n=0, y=True))
            pipe_free = 0
            c = 0
            immas = 0
            last = 0
            total = turns * n
            while warps[0]['n'] + warps[1]['n'] < total:
                order = (last, 1 - last) if warps[last]['y'] else (1 - last, last)
                for k in order:
                    w = warps[k]
                    if w['ready'] > c:
                        continue
                    x = recs2[w['pc']]
                    if any(t > c for sb in x['W'] for t in w['pend'][sb]):
                        continue
                    if x['imma'] and pipe_free > c:
                        continue
                    for sb in x['W']:
                        w['pend'][sb] = []
                    if x['imma']:
                        pipe_free = c + 4
                        immas += 1
                        lw, lr = LIW, 8
                    elif x['op'] == 'LDG':
                        lw, lr = 200, 12
                    elif x['op'] == 'SHFL':
                        lw, lr = 30, 8
                    else:
                        lw, lr = 20, 8
                    if x['w'] is not None:
                        w['pend'][x['w']].append(c + lw)
                    if x['r'] is not None:
                        w['pend'][x['r']].append(c + lr)
                    w['ready'] = c + max(1, x['s'])
                    w['y'] = x['y']
                    w['pc'] = (w['pc'] + 1) % n
                    w['n'] += 1
                    last = k
                    break
                c += 1
            tot.append(immas * 4 / c)
    return 100 * sum(tot) / len(tot)


def clone(bd):
    return [dict(x, f=dict(x['f'])) for x in bd]


def rm_pass(bd, sites=None):
    """drop every wait bit the checker does not need (sites: body indices to try, default all)."""
    for j in (sites if sites is not None else range(n)):
        if j == shfl:
            continue
        for k in range(6):
            if bd[j]['f']['wait'] >> k & 1:
                bd[j]['f']['wait'] &= ~(1 << k)
                if problems(bd):
                    bd[j]['f']['wait'] |= 1 << k
    return bd


assert not problems(body), 'checker fails on ptxas loop'
s_ptx = sim(body)
print('ptxas loop: model %.3f%%' % s_ptx, flush=True)
cur = rm_pass(clone(body))
s_cur = sim(cur)
print('after removing unneeded waits: model %.3f%%' % s_cur, flush=True)
dep_of = collections.defaultdict(set)
for (i, j, kind, dist) in deps:
    dep_of[j].add(i)


def blockers(bd):
    """(producer, field) pairs on a waited scoreboard, issued shortly before a wait that does not need them."""
    out = []
    for j, x in enumerate(bd):
        W = x['f']['wait']
        if not W:
            continue
        for k in range(6):
            if not W >> k & 1:
                continue
            for d in range(1, 30):
                i = (j - d) % n
                if orig_t[j] - orig_t[i] + (T0 if i > j else 0) > 70:
                    break
                for fld in 'wr':
                    if bd[i]['f'][fld] == k and i != shfl and i not in dep_of[j]:
                        dd = orig_t[j] - orig_t[i] + (T0 if i > j else 0)
                        out.append(((40 if fld == 'w' and bd[i]['base'] == 'IMMA' else 8) - dd, i, fld))
    best = {}
    for w, i, fld in out:
        if w > 0:
            best[(i, fld)] = max(best.get((i, fld), 0), w)
    return [k for k, _ in sorted(best.items(), key=lambda kv: -kv[1])[:int(opt.get('top', 14))]]


if MOVES == 'all':
    for it in range(ITERS):
        best = None
        cands = blockers(cur)
        for (i, fld) in cands:
            k0 = cur[i]['f'][fld]
            for k1 in range(6):
                if k1 == k0:
                    continue
                bd = clone(cur)
                bd[i]['f'][fld] = k1
                fixed = set()
                ok = True
                for _ in range(4):
                    pr = problems(bd)
                    if not pr:
                        break
                    for (a, b, kind) in pr:
                        if a is None or b is None or kind == 'FIX' or b == shfl:
                            ok = False
                            break
                        sb = bd[a]['f']['r'] if kind == 'WAR' and bd[a]['f']['r'] != 7 else bd[a]['f']['w']
                        if sb == 7:
                            ok = False
                            break
                        bd[b]['f']['wait'] |= 1 << sb
                        fixed.add(b)
                    if not ok:
                        break
                if not ok or problems(bd):
                    continue
                near = {(i + d) % n for d in range(1, 40)}
                rm_pass(bd, sorted(fixed | {j for j in near if bd[j]['f']['wait'] >> k0 & 1}))
                s = sim(bd)
                if s > s_cur + 0.02 and (best is None or s > best[0]):
                    best = (s, bd, '%s%d %s -> %s%d' % (fld, k0, bd[i]['text'][:44], fld, k1))
        if best is None:
            print('no better move; stop', flush=True)
            break
        s_cur, cur = best[0], best[1]
        print('iter %d: %s  model %.3f%%' % (it, best[2], s_cur), flush=True)

assert not problems(cur)
print('final model %.3f%% (ptxas %.3f%%); long run: ptxas %.3f%%, new %.3f%%' % (
    s_cur, s_ptx, sim(body, 8, 40), sim(cur, 8, 40)))
nw = sum(bin(x['f']['wait']).count('1') for x in cur)
print('wait bits %d (ptxas %d)' % (nw, sum(bin(x['f']['wait']).count('1') for x in body)))
newcode = bytearray(data[off:off + size])
for p, x in enumerate(cur):
    a = (s0 + p) * 16
    newcode[a:a + 16] = setc(x['word'], mk(x['f']))
old = data[off:off + size]
words = [dict(addr=hex(a), old=old[a:a + 16].hex(), new=bytes(newcode[a:a + 16]).hex())
         for a in range(0, size, 16) if newcode[a:a + 16] != old[a:a + 16]]
assert all(int(w['addr'], 16) != (s0 + shfl) * 16 for w in words)
before, after = hashlib.md5(old).hexdigest()[:12], hashlib.md5(bytes(newcode)).hexdigest()[:12]
out = bytearray(data)
out[off:off + size] = newcode
open(outp, 'wb').write(out)
json.dump(dict(description='goal-nv20i climb: ptxas loop, scoreboards/waits only ' + ' '.join(sys.argv[4:]),
               section=sec, before=before, after=after, words=words), open(outp + '.json', 'w'), indent=0)
with open(outp + '.loop', 'w') as fo:
    for x in cur:
        f = x['f']
        fo.write('%04x s%-2d %s w%s r%s W%-6s u%x %s\n' % (x['a'], f['stall'], 'Y' if f['y'] else '.',
                 f['w'] if f['w'] != 7 else '-', f['r'] if f['r'] != 7 else '-',
                 ''.join(str(k) for k in range(6) if f['wait'] >> k & 1) or '-', f['reuse'], x['text']))
print('words %d, before %s after %s' % (len(words), before, after))
