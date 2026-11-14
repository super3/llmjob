"""freeregs.py: registers with no live value across the chain span of hr.py's new loop (both sections)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open('hr.py', encoding='utf-8').read().replace('\nmain()\n', '\n')
hr = {'__file__': os.path.abspath('hr.py'), '__name__': 'hr'}
exec(compile(src, 'hr.py', 'exec'), hr)
from ana import loop
for sec in ('pearl_t75_tile_fold_wmma_rg', 'pearl_t75_tile_fold_wmma'):
    d, o, rs = loop('b0/base.cubin', sec)
    seq, words, items, sums = hr['build'](rs, True)
    n = len(seq)
    du = []
    for e in seq:
        if e['kind'] == 'keep':
            x = rs[e['i']]
            du.append((hr['defs_n'](x), {int(u[1:]) for u in x['uses'] if re.fullmatch(r'R\d+', u)}))
        elif e['kind'] in ('chain', 'fin'):
            du.append(({e['rd']}, {r for r in (e['ra'], e['rb'], e['rc']) if r != 255}))
        else:
            du.append((set(), set()))
    # backward liveness over the cyclic sequence
    live_in = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for j in range(n - 1, -1, -1):
            out = live_in[(j + 1) % n]
            new = (out - du[j][0]) | du[j][1]
            if new != live_in[j]:
                live_in[j] = new
                changed = True
    first = min(j for j, e in enumerate(seq) if e['kind'] == 'chain')
    fin = [j for j, e in enumerate(seq) if e['kind'] == 'fin'][0]
    busy = set()
    for j in range(first, fin + 1):
        busy |= live_in[j]
        busy |= {r for r in du[j][0] if seq[j]['kind'] == 'keep'}
    free = [r for r in range(0, 251) if r not in busy]
    print(sec, 'chain span', first, fin, 'free registers:', free)
