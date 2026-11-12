"""miss.py <cubin> <section>...: in the chunk loop, IMMA a/b reads served by the reuse cache (one entry per slot
and bank), and IMMAs whose uncached a/b reads share a bank (a likely extra cycle)."""
import re, sys
sys.path.insert(0, '.')
from sm75 import disasm
from reuse2 import slots
def loop(cub, sec):
    data, off, size, recs, labels = disasm(cub, sec)
    for idx, r in enumerate(recs):
        m = re.search(r'BRA\s+`\((\.L_x_\d+)\)', r['text'])
        if m and labels.get(m.group(1), 1 << 30) < r['a']:
            a0 = labels[m.group(1)] // 16
            if sum(1 for x in recs[a0:idx + 1] if x['base'] == 'IMMA') == 512:
                return recs[a0:idx + 1]
for cub in sys.argv[2:]:
    body = loop(cub, sys.argv[1])
    cache = {}
    hit = miss = conf = 0
    for turn in range(2):
        for x in body:
            if x['base'] != 'IMMA':
                for d in x['defs']:
                    if re.fullmatch(r'R\d+', d):
                        r = int(d[1:]); cache = {k: v for k, v in cache.items() if v != r}
                continue
            sl = slots(x); unc = []
            for s in (0, 1):
                r = sl[s]
                if cache.get((s, r % 2)) == r:
                    hit += turn
                else:
                    miss += turn; unc.append(r % 2)
            # C pair reads both banks
            if turn and len(unc) == 2 and unc[0] == unc[1]:
                conf += 1
            for s in (0, 1):
                r = sl[s]
                if x['f']['reuse'] >> s & 1:
                    cache[(s, r % 2)] = r
                else:
                    cache.pop((s, r % 2), None)
            for d in x['defs']:
                if re.fullmatch(r'R\d+', d):
                    r = int(d[1:]); cache = {k: v for k, v in cache.items() if v != r}
    print('%-40s hits %d misses %d same-bank uncached a+b %d' % (cub[-40:], hit, miss, conf))
