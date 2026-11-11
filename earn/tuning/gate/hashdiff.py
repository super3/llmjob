import re, sys, collections
sys.path.insert(0, '.')
src = open('sassstat.py').read().replace('main()\n', '', 1) if False else None
import importlib.util
spec = importlib.util.spec_from_file_location('ss', 'sassstat.py')
# sassstat runs main() on import; re-implement the small part needed
LINE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?);"); LABEL = re.compile(r"^(\.L_x_\d+):")
def kernel(path, fn):
    out, labels, on = [], {}, False
    for line in open(path, encoding='utf-8', errors='replace'):
        if line.startswith('//---') and '.text.' in line:
            on = ('.text.' + fn + ' ') in line; continue
        if not on: continue
        m = LABEL.match(line)
        if m: labels[m.group(1)] = len(out); continue
        m = LINE.match(line)
        if m:
            t = m.group(2).strip(); p = ''
            if t.startswith('@'): p, t = t.split(None, 1)
            out.append((p, t.split()[0], t))
    return out, labels
def region(path, fn):
    k, labels = kernel(path, fn)
    imma = [i for i, x in enumerate(k) if x[1].startswith('IMMA')]
    first, last = imma[0], imma[-1]
    tg = lambda t: (re.search(r'\.L_x_\d+', t) or [None])[0] if re.search(r'\.L_x_\d+', t) else None
    close = head = None
    for i in range(last, len(k)):
        if k[i][1].startswith('BRA'):
            t = labels.get(tg(k[i][2]) or '')
            if t is not None and t <= first: close, head = i, t; break
    hand = next(i for i in range(close, len(k)) if k[i][1].startswith('BAR.SYNC'))
    for i in range(hand, len(k)):
        if k[i][1].startswith('BRA') and not k[i][0]:
            t = labels.get(tg(k[i][2]) or '')
            if t is not None and t <= head: tend = i; break
    return k[head:close+1], k[hand:tend+1]
fn = sys.argv[1]
base_loop, base_hash = region(sys.argv[2], fn)
hb = collections.Counter(x[1] for x in base_hash); lb = collections.Counter(x[1].split('.')[0] for x in base_loop)
for p in sys.argv[3:]:
    loop, h = region(p, fn)
    hc = collections.Counter(x[1] for x in h); lc = collections.Counter(x[1].split('.')[0] for x in loop)
    dh = {k: hc[k] - hb[k] for k in set(hc) | set(hb) if hc[k] != hb[k]}
    dl = {k: lc[k] - lb[k] for k in set(lc) | set(lb) if lc[k] != lb[k]}
    print(p.split('/')[-2], 'hash diff', dict(sorted(dh.items())), '| loop diff', dict(sorted(dl.items())))
print('base hash', dict(sorted(hb.items())))
print('base loop', dict(sorted(lb.items())))
