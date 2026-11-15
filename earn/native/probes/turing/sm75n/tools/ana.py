"""ana.py <cubin> [section]: goal-nv20n. The t75 chunk loop: where reuse2's bank rule adds IMMA reuse flags that
ptxas left off, and register-bank pressure (reads by bank) around the readout."""
import os
import re
import sys

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(S, "g-nv20n", "earn", "native", "probes", "turing", "sched"))
import sm75  # noqa: E402
import reuse2  # noqa: E402

sm75.DIS = os.path.join(S, "nv50", "dl", "cuda133", "bin", "nvdisasm.exe")


def loop(cub, sec):
    data, off, size, recs, labels = sm75.disasm(cub, sec)
    for idx, r in enumerate(recs):
        m = re.search(r'BRA\s+`\((\.L_x_\d+)\)', r['text'])
        if m and labels.get(m.group(1), 1 << 30) < r['a']:
            a0 = labels[m.group(1)] // 16
            if sum(1 for x in recs[a0:idx + 1] if x['base'] == 'IMMA') == 512:
                return data, off, recs[a0:idx + 1]
    raise SystemExit('no loop')


def main():
    cub = sys.argv[1]
    sec = sys.argv[2] if len(sys.argv) > 2 else 'pearl_t75_tile_fold_wmma_rg'
    data, off, rs = loop(cub, sec)
    new = reuse2.reuse_bank(rs)
    add = 0
    for i, (x, y) in enumerate(zip(rs, new)):
        if x['base'] != 'IMMA':
            continue
        a, b = x['f']['reuse'], y['f']['reuse']
        if a != b:
            add += 1
            # non-IMMA instructions between this IMMA and the next IMMA
            j = i + 1
            between = []
            while j < len(rs) and rs[j]['base'] != 'IMMA':
                between.append(rs[j]['base'])
                j += 1
            print('%04x ptxas u%x bank-rule u%x  %-55s | %s' % (x['a'], a, b, x['text'][:55], ' '.join(between)))
    print('IMMAs where the bank rule differs:', add)
    # IMMAs whose uncached A and B are in one bank
    same = 0
    for x in rs:
        if x['base'] != 'IMMA':
            continue
        sl = reuse2.slots(x)
        u = x['f']['reuse']
        if not (u & 1) and not (u & 2) and sl[0] % 2 == sl[1] % 2:
            same += 1
    print('IMMAs with A and B uncached in one bank:', same)
    lop = [x for x in rs if x['base'] == 'LOP3']
    tri = 0
    for x in lop:
        sl = [s for s in reuse2.slots(x) if s is not None and s != 255]
        if len(sl) == 3 and len({s % 2 for s in sl}) == 1:
            tri += 1
            print('LOP3 three in one bank: %04x %s' % (x['a'], x['text']))
    print('LOP3:', len(lop), 'with all three sources in one bank:', tri)


if __name__ == "__main__":
    main()
