"""hr.py <cubin> <outdir> [--nocache]: goal-nv20n hand readout for the t75 chunk loop (both fold sections).

ptxas's readout: 68 LOP3 (a 3-input xor tree over the 128 sums) interleaved with the readout k-step, two register
reads in one bank each (two register-file cycles). Priced: the reads are the cost (rz +1.8%), and one read fewer a
LOP3 (c0) is half of it.

This rewrite drops ptxas's 68 readout LOP3 and folds the 128 sums in two chains (R252, even bank; R251, odd bank).
Each chain LOP3 is placed right after the IMMA that next reads the pair it folds as C (the next chunk's IMMA). That
IMMA gets a reuse flag on C, and the LOP3 takes one half of the pair from slot c (the reuse cache), the other half
and its chain register from the register file, in different banks: one register-file cycle. When that IMMA writes
the pair in place, the LOP3 still reads the old sum: it issues right after the IMMA, long before the IMMA's result
lands, and the cached half is the value the IMMA read. A final LOP3 writes R252 ^ R251 into R224 where ptxas wrote
the readout; NOPs fill the slots left. The 8 PRMT gates' tie to the fold becomes RZ (their value never depended on
it). Every other instruction keeps its order, registers and control bits, except stalls (a dropped LOP3's stall goes
to the instruction before it; an IMMA's stall is split with the LOP3 after it) and waits (a dropped LOP3's wait
moves to the next instruction that is not a chain LOP3).

--nocache: the same chains with no reuse flag (both halves from the register file), to price the cache.
--y0: the chain LOP3s with the yield bit clear (as ptxas left it on about half its readout LOP3s).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUR = os.path.join(S, "g-nv20n", "earn", "native", "probes", "turing")
sys.path.insert(0, os.path.join(TUR, "sched"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sm75  # noqa: E402
import reuse2  # noqa: E402
from ana import loop  # noqa: E402

TA, TB = 252, 251          # chain registers: TA even bank, TB odd bank (R0-R252 are allocated; neither is touched)
FINAL = 224
YB = 1
PY = sys.executable


def regs3(x):
    return [r for r in reuse2.slots(x)]


def imma_dc(x):
    ops = x['text'].split(None, 1)[1].split(',')
    d = int(re.search(r'R(\d+)', ops[0]).group(1))
    c = int(re.search(r'R(\d+)', ops[3]).group(1))
    return d, c


def defs_n(x):
    return {int(r[1:]) for r in x['defs'] if re.fullmatch(r'R\d+', r)}


def lop_word(tmpl, rd, ra, rb, rc, ctrl):
    w = bytearray(tmpl)
    w[2], w[3], w[4], w[8] = rd, ra, rb, rc
    return sm75.setc(bytes(w), ctrl)


def nop_word(ctrl):
    return sm75.setc((0x7918).to_bytes(8, 'little') + bytes(8), ctrl)


def plan(rs, cache=True):
    n = len(rs)
    ro = [i for i, x in enumerate(rs) if x['base'] == 'LOP3' and not x['text'].startswith('@')]
    roset = set(ro)
    shfl = [i for i, x in enumerate(rs) if x['base'] == 'SHFL'][0]
    fin = [i for i in ro if FINAL in defs_n(rs[i])][-1]          # ptxas's last write of the readout value
    # the sums: reads by readout LOP3s whose reaching definition is an IMMA
    sums = []
    for p in ro:
        for r in regs3(rs[p]):
            if r is None or r == 255:
                continue
            w = None
            for k in range(1, n + 1):
                q = (p - k) % n
                if r in defs_n(rs[q]):
                    w = q
                    break
            if rs[w]['base'] == 'IMMA':
                assert w < p, (p, r, w)
                sums.append((r, w, p))
    assert len(sums) == 128, len(sums)
    assert len({(r, w) for r, w, _ in sums}) == 128
    # for each sum: the next IMMA after its writer that reads it as C, with no write to it in between, before fin
    nxt = {}
    for r, w, p in sums:
        hit = None
        for q in range(w + 1, fin):
            x = rs[q]
            if x['base'] == 'IMMA':
                d, c = imma_dc(x)
                if r in (c, c + 1):
                    hit = q
                    break
            if r in defs_n(x) and q not in roset:
                break
        nxt[(r, w)] = hit
    # items: one chain LOP3 per IMMA that reads two sums as C (cached half + register-file half); the rest fall back
    by_i = {}
    for (r, w), q in nxt.items():
        if q is not None:
            by_i.setdefault(q, []).append(r)
    items = []      # (position after which it goes, cached reg or None, [rf regs])
    left = []
    for q, rr in sorted(by_i.items()):
        d, c = imma_dc(rs[q])
        if sorted(rr) == [c, c + 1]:
            items.append([q, None, [c, c + 1]])
        else:
            left += [(r, q) for r in rr]
    for (r, w), q in nxt.items():
        if q is None:
            left.append((r, None))
    # leftovers: pair them in order of their original reader, placed after the later original reader
    lp = sorted(((dict((r2, p2) for r2, w2, p2 in sums if r2 == r)[r], r) for r, _ in left))
    while lp:
        a = lp.pop(0)
        b = lp.pop(0) if lp else None
        rr = [a[1]] + ([b[1]] if b else [])
        at = (b or a)[0]
        items.append([at, None, rr])
    items.sort(key=lambda it: it[0])
    # chains: alternate; cache the half whose bank differs from the chain register's
    out = []
    ca = cb = None
    for k, (q, _, rr) in enumerate(items):
        chain = TA if k % 2 == 0 else TB
        cached = None
        rf = list(rr)
        if cache and rs[q]['base'] == 'IMMA' and len(rr) == 2:
            # chain TA (even) wants the odd half from the register file, so the even half is cached; TB the reverse
            cached = [r for r in rr if r % 2 == chain % 2][0]
            rf = [r for r in rr if r != cached]
        out.append(dict(after=q, chain=chain, cached=cached, rf=rf))
    return out, ro, fin, shfl, sums


def build(rs, cache=True):
    items, ro, fin, shfl, sums = plan(rs, cache)
    roset = set(ro)
    after = {}
    for it in items:
        after.setdefault(it['after'], []).append(it)
    tmpl = [x for x in rs if x['base'] == 'LOP3' and ', 0x96,' in x['text']][0]['word']
    seq = []        # dicts: word, ctrl fields, kind, info
    pend_wait = 0
    started = {TA: False, TB: False}
    for i, x in enumerate(rs):
        f = dict(x['f'])
        if i in roset and i != fin:
            # dropped: its stall goes to the instruction before, its wait to the next kept non-chain instruction
            seq[-1]['f']['stall'] = min(11, seq[-1]['f']['stall'] + f['stall'])
            pend_wait |= f['wait']
            continue
        if i == fin:
            # the final xor of the two chains, in ptxas's slot for the readout value
            ff = dict(stall=f['stall'], y=f['y'], w=7, r=7, wait=f['wait'] | pend_wait, reuse=0)
            pend_wait = 0
            seq.append(dict(kind='fin', f=ff, rd=FINAL, ra=TA, rb=TB, rc=255, i=i))
            continue
        word = x['word']
        if x['base'] == 'PRMT' and ', 0x3210, ' in x['text']:
            w = bytearray(word)
            w[8] = 255                  # the gate's tie to the fold: RZ
            word = bytes(w)
        f['wait'] |= pend_wait
        pend_wait = 0
        seq.append(dict(kind='keep', f=f, word=word, i=i, base=x['base']))
        for it in after.get(i, []):
            if it['cached'] is not None:
                seq[-1]['f']['reuse'] |= 4
            s = seq[-1]['f']['stall']
            s1 = max(1, s // 2)
            seq[-1]['f']['stall'] = s1
            ls = max(1, s - s1)
            ch = it['chain']
            a = ch if started[ch] else 255
            started[ch] = True
            rf = it['rf']
            if it['cached'] is not None:
                ra_, rb_, rc_ = a, rf[0], it['cached']
            else:
                rr = rf + [255] * (2 - len(rf))
                ra_, rb_, rc_ = a, rr[0], rr[1]
            seq.append(dict(kind='chain', f=dict(stall=ls, y=YB, w=7, r=7, wait=0, reuse=0),
                            rd=ch, ra=ra_, rb=rb_, rc=rc_, i=i, item=it))
    assert pend_wait == 0
    while len(seq) < len(rs):
        # NOPs right after the final xor
        k = [j for j, e in enumerate(seq) if e['kind'] == 'fin'][0]
        seq.insert(k + 1, dict(kind='nop', f=dict(stall=1, y=1, w=7, r=7, wait=0, reuse=0)))
    assert len(seq) == len(rs), (len(seq), len(rs))
    words = []
    for e in seq:
        c = sm75.mk(e['f'])
        if e['kind'] == 'keep':
            words.append(sm75.setc(e['word'], c))
        elif e['kind'] in ('chain', 'fin'):
            words.append(lop_word(tmpl, e['rd'], e['ra'], e['rb'], e['rc'], c))
        else:
            words.append(nop_word(c))
    return seq, words, items, sums


def check(rs, seq, sums):
    """Value check. A non-IMMA read sees an IMMA's write only if the IMMA is at least 4 instructions back (closer
    is the stale-read case and must be the IMMA right before). The chains must fold exactly ptxas's 128 values."""
    n = len(rs)
    # value of each register: (writer position in seq) -- walk seq twice (cyclic), tracking defs
    texts = []
    vals = {}
    want = {(r, rs[w]['a']) for r, w, _ in sums}     # (reg, address of the IMMA writing it)
    got = []
    hist = {}       # reg -> list of (seq index, writer address, is_imma)
    pos_addr = []
    for e in seq:
        pos_addr.append(rs[e['i']]['a'] if e['kind'] == 'keep' else None)
    for rnd in range(2):
        for j, e in enumerate(seq):
            t = rnd * n + j
            if e['kind'] == 'chain' and rnd == 1:
                for k, r in enumerate((e['ra'], e['rb'], e['rc'])):
                    if r in (255, TA, TB):
                        continue
                    h = hist[r]
                    # latest write visible to this read
                    cand = h[-1]
                    if cand[2] and t - cand[0] < 4:
                        assert t - cand[0] == 1, ('stale read not right after its IMMA', e, cand)
                        cand = h[-2]
                    got.append((r, cand[1]))
            if e['kind'] == 'keep':
                x = rs[e['i']]
                for r in defs_n(x):
                    hist.setdefault(r, []).append((t, x['a'], x['base'] == 'IMMA'))
            elif e['kind'] in ('chain', 'fin'):
                hist.setdefault(e['rd'], []).append((t, 'lop', False))
    assert sorted(got) == sorted(want), (len(got), len(want), sorted(set(got) ^ set(want))[:6])
    # chain latency: cycles between dependent chain LOP3s, and chains -> final -> SHFL
    cyc = 0
    last = {}
    worst = 99
    for j, e in enumerate(seq):
        if e['kind'] in ('chain', 'fin'):
            for r in (e['ra'], e['rb']):
                if r in last:
                    worst = min(worst, cyc - last[r])
            last[e['rd']] = cyc
        if e['kind'] == 'keep' and rs[e['i']]['base'] == 'SHFL':
            worst = min(worst, cyc - last[FINAL])
        cyc += e['f']['stall']
    return len(got), worst


def md5(b):
    return hashlib.md5(b).hexdigest()[:12]


def main():
    cub, out = sys.argv[1], sys.argv[2]
    global YB
    cache = '--nocache' not in sys.argv
    YB = 0 if '--y0' in sys.argv else 1
    kind = ('hr' if cache else 'hrn') + ('0' if YB == 0 else '')
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_%s.cubin' % kind)
    shutil.copy(cub, tmp)
    for pre, sec in (('rg', 'pearl_t75_tile_fold_wmma_rg'), ('d', 'pearl_t75_tile_fold_wmma')):
        data, off, rs = loop(cub, sec)
        seq, words, items, sums = build(rs, cache)
        nread, worst = check(rs, seq, sums)
        ncached = sum(1 for it in items if it['cached'] is not None)
        secs = sm75.sections(data)
        o, nn = secs['.text.' + sec]
        body = bytearray(data[o:o + nn])
        before = md5(body)
        wl = []
        for x, w in zip(rs, words):
            if w != x['word']:
                body[x['a']:x['a'] + 16] = w
                wl.append(dict(addr='0x%x' % x['a'], old=x['word'].hex(), new=w.hex()))
        after = md5(body)
        desc = 'goal-nv20n hand readout (%s): two xor chains placed after the IMMAs that read each pair as C' % (
            'one half from the IMMA C reuse cache' if cache else 'no reuse cache')
        spec = dict(description=desc, section=sec, before=before, after=after, words=wl)
        fn = os.path.join(out, '%s_%s.json' % (pre, kind))
        with open(fn, 'w', newline='\n') as f:
            f.write(json.dumps(spec, indent=0) + '\n')
        r = subprocess.run([PY, '-I', os.path.join(TUR, 'sm75patch.py'), tmp, fn], capture_output=True, text=True)
        print('%s: %d chain LOP3 (%d cached), %d sum reads checked, min dependent gap %d cycles, %d words; %s' % (
            sec, len(items), ncached, nread, worst, len(wl), (r.stdout + r.stderr).strip()))
    r = subprocess.run([sm75.DIS, '-c', tmp], capture_output=True, text=True)
    print('nvdisasm rc', r.returncode)
    with open(tmp + '.sass', 'w') as f:
        f.write(r.stdout)


main()
