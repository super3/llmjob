"""hrcheck.py: for hr.py's new loop, every kept instruction's sources reach the same writer as in ptxas's loop
(gates excepted), and the distance (in IMMAs) from each chain LOP3's register-file sum to the IMMA that wrote it."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location('hr', 'hr.py')
src = open('hr.py', encoding='utf-8').read().replace('\nmain()\n', '\n')
hr = {'__file__': os.path.abspath('hr.py'), '__name__': 'hr'}
exec(compile(src, 'hr.py', 'exec'), hr)
from ana import loop
for sec in ('pearl_t75_tile_fold_wmma_rg', 'pearl_t75_tile_fold_wmma'):
    d, o, rs = loop('b0/base.cubin', sec)
    seq, words, items, sums = hr['build'](rs, True)
    n = len(rs)
    ro = {i for i, x in enumerate(rs) if x['base'] == 'LOP3' and not x['text'].startswith('@')}
    def reach(entries, j, r):
        for k in range(1, n + 1):
            e = entries[(j - k) % n]
            if r in e[1]:
                return e[0]
        return None
    old = [(('ro' if i in ro else i), hr['defs_n'](x), x) for i, x in enumerate(rs)]
    new = []
    for e in seq:
        if e['kind'] == 'keep':
            new.append((e['i'], hr['defs_n'](rs[e['i']]), rs[e['i']]))
        elif e['kind'] in ('chain', 'fin'):
            new.append(('chain' if e['kind'] == 'chain' else 'fin', {e['rd']}, None))
        else:
            new.append(('nop', set(), None))
    oldpos = {i: j for j, (i, _, _) in enumerate(old)}
    bad = 0
    for j, (i, dd, x) in enumerate(new):
        if not isinstance(i, int):
            continue
        uses = {int(u[1:]) for u in x['uses'] if re.fullmatch(r'R\d+', u)}
        for r in uses:
            a = reach(old, oldpos[i], r)
            b = reach(new, j, r)
            if a != b:
                if x['base'] == 'PRMT' and a == 'ro':
                    continue
                if a == 'ro' and b == 'fin' and r == 224:
                    continue
                bad += 1
                print('DIFF', sec[-3:], hex(x['a']), x['text'], 'R%d' % r, a, b)
    # distance from register-file sum reads to their IMMA writer, in IMMAs issued between
    dmin = 999
    imma_before = []
    c = 0
    for e in seq:
        imma_before.append(c)
        if e['kind'] == 'keep' and rs[e['i']]['base'] == 'IMMA':
            c += 1
    wpos = {}
    for j, e in enumerate(seq):
        if e['kind'] == 'keep':
            wpos[e['i']] = j
    for j, e in enumerate(seq):
        if e['kind'] != 'chain':
            continue
        for r in (e['rb'],) + ((e['rc'],) if e['item']['cached'] is None else ()):
            if r == 255:
                continue
            w = [wi for (rr, wi, p) in sums if rr == r and wi in wpos and wpos[wi] < j]
            w = max(w, key=lambda wi: wpos[wi])
            dist = imma_before[j] - imma_before[wpos[w]] - 1
            dmin = min(dmin, dist)
    print(sec, 'kept-reader differences:', bad, '| fewest IMMAs between a sum\'s writer and its register-file read:', dmin)
