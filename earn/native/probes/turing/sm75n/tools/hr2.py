"""hr2.py <cubin> <outdir>: goal-nv20n hand readout v2 for the t75 chunk loop (both fold sections), with no stale
register reads.

ptxas's readout is 68 LOP3 (a 3-input xor tree over the 128 sums), each reading two registers in one bank: two
register-file cycles. Priced on a 2060: the reads are the cost (rz +1.7%), one read fewer a LOP3 (c0) is half.

hr (v1) read one half of each pair right after the IMMA that overwrites it in place, before the result lands. That
is a race (the other warp can hold this one back): 7.5% of hashes failed. v2 folds the 128 sums in two xor chains
(R246, even bank; R1, odd bank; both free across the chains). The pairs of a chain are taken in the order of the
IMMA that next reads each as C (the next chunk's IMMA, I_P). The chain LOP3 for pair P sits right after I_P: I_P
gets a reuse flag on C, so the LOP3 takes one half of P from slot c (the cache holds what I_P read), and from the
register file the chain register and one half of the chain's NEXT pair, which its own IMMA has not reached yet.
Chain R246 caches the even half and reads the odd one, R1 the reverse, so each LOP3 reads two registers in
different banks: one cycle. Two LOP3s before the first I_P read the first pair's register-file half of each chain;
a final LOP3 writes R246 ^ R1 into R224 where ptxas wrote the readout; a NOP fills the slot left. The 8 PRMT gates'
tie to the fold becomes RZ (their value never depended on it). Every other instruction keeps its order, registers
and control bits, except stalls (a dropped LOP3's stall goes to the instruction before it; an IMMA's stall is split
with the LOP3 after it) and waits (a dropped LOP3's wait moves to the next kept instruction).

Correct only if the IMMA C reuse cache holds the value I_P read until this warp's next IMMA (the cr and cb probes
passed verify-hits; ptxas relies on the same for A and B across the other warp's issue). verify-hits decides.
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

TA, TB = 246, 1
FINAL = 224
BEFORE = False      # --before: each chain LOP3 right before its I_P, both halves from the register file
PY = sys.executable


def imma_dc(x):
    ops = x['text'].split(None, 1)[1].split(',')
    return int(re.search(r'R(\d+)', ops[0]).group(1)), int(re.search(r'R(\d+)', ops[3]).group(1))


def defs_n(x):
    return {int(r[1:]) for r in x['defs'] if re.fullmatch(r'R\d+', r)}


def uses_n(x):
    return {int(r[1:]) for r in x['uses'] if re.fullmatch(r'R\d+', r)}


def lop_word(tmpl, rd, ra, rb, rc, ctrl):
    w = bytearray(tmpl)
    w[2], w[3], w[4], w[8] = rd, ra, rb, rc
    return sm75.setc(bytes(w), ctrl)


def nop_word(ctrl):
    return sm75.setc((0x7918).to_bytes(8, 'little') + bytes(8), ctrl)


def plan(rs):
    n = len(rs)
    ro = [i for i, x in enumerate(rs) if x['base'] == 'LOP3' and not x['text'].startswith('@')]
    roset = set(ro)
    fin = [i for i in ro if FINAL in defs_n(rs[i])][-1]
    sums = []
    for p in ro:
        for r in reuse2.slots(rs[p]):
            if r is None or r == 255:
                continue
            w = next((p - k) % n for k in range(1, n + 1) if r in defs_n(rs[(p - k) % n]))
            if rs[w]['base'] == 'IMMA':
                assert w < p
                sums.append((r, w))
    assert len(sums) == 128 and len(set(sums)) == 128
    nxt = {}
    for r, w in sums:
        hit = None
        for q in range(w + 1, fin):
            x = rs[q]
            if x['base'] == 'IMMA' and r in (imma_dc(x)[1], imma_dc(x)[1] + 1):
                hit = q
                break
            if r in defs_n(x) and q not in roset:
                break
        nxt[(r, w)] = hit
    by_i = {}
    for (r, w), q in nxt.items():
        assert q is not None, ('a sum with no next C reader before the readout slot', r, w)
        by_i.setdefault(q, []).append(r)
    pairs = []
    for q, rr in sorted(by_i.items()):
        d, c = imma_dc(rs[q])
        assert sorted(rr) == [c, c + 1], (q, rr)
        pairs.append(dict(i=q, lo=c, hi=c + 1))
    # two chains, alternating in I_P order; chain TA caches lo and reads hi, TB caches hi and reads lo
    chains = {TA: [], TB: []}
    for k, p in enumerate(pairs):
        ch = TA if k % 2 == 0 else TB
        p['chain'] = ch
        p['cached'] = p['lo'] if ch == TA else p['hi']
        p['rf'] = p['hi'] if ch == TA else p['lo']
        chains[ch].append(p)
    return pairs, chains, ro, fin, sums


def build(rs):
    pairs, chains, ro, fin, sums = plan(rs)
    roset = set(ro)
    # what goes after each I_P: chain LOP3 = t ^ next_pair.rf (slot b) ^ cached half (slot c)
    after = {}
    for ch, ps in chains.items():
        for k, p in enumerate(ps):
            nx = ps[k + 1]['rf'] if k + 1 < len(ps) else 255
            after[p['i']] = dict(chain=ch, rb=nx, rc=p['cached'], pair=p, ra=ch if k else 255)
    first_i = min(p['i'] for p in pairs)
    tmpl = [x for x in rs if x['base'] == 'LOP3' and ', 0x96,' in x['text']][0]['word']
    seq = []
    pend_wait = 0
    for i, x in enumerate(rs):
        f = dict(x['f'])
        if BEFORE and i in after:
            it = after[i]
            p = it['pair']
            started = it.get('started')
            if seq and seq[-1]['f']['stall'] >= 2:
                seq[-1]['f']['stall'] -= 1
            seq.append(dict(kind='chain', f=dict(stall=1, y=1, w=7, r=7, wait=pend_wait, reuse=0),
                            rd=it['chain'], ra=it['ra'], rb=p['lo'], rc=p['hi'], i=i, pair=p))
            pend_wait = 0
        if i == first_i and not BEFORE:
            # the first pair's register-file half of each chain, before any I_P
            for ch in (TA, TB):
                seq.append(dict(kind='init', f=dict(stall=3, y=1, w=7, r=7, wait=pend_wait, reuse=0),
                                rd=ch, ra=255, rb=255, rc=chains[ch][0]['rf'], i=i))
                pend_wait = 0
            # the instruction before (if any) loses nothing; the init LOP3s add 2 cycles
        if i in roset and i != fin:
            seq[-1]['f']['stall'] = min(11, seq[-1]['f']['stall'] + f['stall'])
            pend_wait |= f['wait']
            continue
        if i == fin:
            seq.append(dict(kind='fin', f=dict(stall=f['stall'], y=f['y'], w=7, r=7, wait=f['wait'] | pend_wait,
                                               reuse=0), rd=FINAL, ra=TA, rb=TB, rc=255, i=i))
            pend_wait = 0
            continue
        word = x['word']
        if x['base'] == 'PRMT' and ', 0x3210, ' in x['text']:
            w = bytearray(word)
            w[8] = 255
            word = bytes(w)
        f['wait'] |= pend_wait
        pend_wait = 0
        seq.append(dict(kind='keep', f=f, word=word, i=i))
        if i in after and not BEFORE:
            it = after[i]
            seq[-1]['f']['reuse'] |= 4
            s = seq[-1]['f']['stall']
            s1 = max(1, s // 2)
            seq[-1]['f']['stall'] = s1
            seq.append(dict(kind='chain', f=dict(stall=max(1, s - s1), y=1, w=7, r=7, wait=0, reuse=0),
                            rd=it['chain'], ra=it['chain'], rb=it['rb'], rc=it['rc'], i=i, pair=it['pair']))
    assert pend_wait == 0
    k = [j for j, e in enumerate(seq) if e['kind'] == 'fin'][0]
    while len(seq) < len(rs):
        seq.insert(k + 1, dict(kind='nop', f=dict(stall=1, y=1, w=7, r=7, wait=0, reuse=0)))
    assert len(seq) == len(rs), (len(seq), len(rs))
    words = []
    for e in seq:
        c = sm75.mk(e['f'])
        if e['kind'] == 'keep':
            words.append(sm75.setc(e['word'], c))
        elif e['kind'] == 'nop':
            words.append(nop_word(c))
        else:
            words.append(lop_word(tmpl, e['rd'], e['ra'], e['rb'], e['rc'], c))
    return seq, words, pairs, sums


def check(rs, seq, sums):
    """Values, over two turns of the loop. A register-file read must see a write that has landed: an IMMA's write
    counts only 4 or more instructions back, and no IMMA may write a register-file-read register in the 3 before.
    A cached read (slot c of a chain LOP3) gets what the IMMA right before read as C. The chains together must fold
    exactly ptxas's 128 values. Every kept instruction must reach the same writer as in ptxas's loop."""
    n = len(rs)
    want = sorted((r, rs[w]['a']) for r, w in sums)
    got = []
    hist = {}

    def landed(r, t):
        h = [e for e in hist[r] if e[0] < t]
        for tt, who, imma in reversed(h):
            if imma and t - tt < 4:
                raise AssertionError('register-file read of R%d %d after an IMMA wrote it' % (r, t - tt))
            return who
    for rnd in range(2):
        for j, e in enumerate(seq):
            t = rnd * n + j
            if e['kind'] in ('chain', 'init') and rnd == 1:
                for slot, r in (('a', e['ra']), ('b', e['rb']), ('c', e['rc'])):
                    if r in (255, TA, TB):
                        continue
                    if e['kind'] == 'chain' and slot == 'c' and not BEFORE:
                        prev = seq[j - 1]
                        assert prev['kind'] == 'keep' and rs[prev['i']]['base'] == 'IMMA'
                        d, c = imma_dc(rs[prev['i']])
                        assert r in (c, c + 1) and prev['f']['reuse'] & 4
                        got.append((r, landed(r, t - 1)))
                    else:
                        got.append((r, landed(r, t)))
            if e['kind'] == 'keep':
                x = rs[e['i']]
                for r in defs_n(x):
                    hist.setdefault(r, []).append((t, x['a'], x['base'] == 'IMMA'))
            elif e['kind'] in ('chain', 'init', 'fin'):
                hist.setdefault(e['rd'], []).append((t, 'lop', False))
    assert sorted(got) == want, sorted(set(got) ^ set(want))[:6]
    # kept instructions reach the same writer as before
    ro = {i for i, x in enumerate(rs) if x['base'] == 'LOP3' and not x['text'].startswith('@')}
    old = [(('ro' if i in ro else i), defs_n(x)) for i, x in enumerate(rs)]
    new = [((e['i'] if e['kind'] == 'keep' else e['kind']),
            (defs_n(rs[e['i']]) if e['kind'] == 'keep' else ({e['rd']} if 'rd' in e else set()))) for e in seq]

    def reach(entries, j, r):
        for k in range(1, n + 1):
            if r in entries[(j - k) % n][1]:
                return entries[(j - k) % n][0]
    oldpos = {i: j for j, (i, _) in enumerate(old)}
    for j, e in enumerate(seq):
        if e['kind'] != 'keep':
            continue
        x = rs[e['i']]
        for r in uses_n(x):
            a, b = reach(old, oldpos[e['i']], r), reach(new, j, r)
            if a != b and not (x['base'] == 'PRMT' and a == 'ro') and not (r == FINAL and b == 'fin'):
                raise AssertionError('kept %s reads R%d from %s, was %s' % (x['text'], r, b, a))
    # the chain registers must hold nothing else from the first init to the final xor
    j0 = min(j for j, e in enumerate(seq) if e['kind'] in ('init', 'chain'))
    j1 = [j for j, e in enumerate(seq) if e['kind'] == 'fin'][0]
    for j in range(n):
        e = seq[j]
        if e['kind'] == 'keep':
            touched = defs_n(rs[e['i']]) | uses_n(rs[e['i']])
            assert not (touched & {TA, TB}) or not (j0 <= j <= j1), ('chain register touched', rs[e['i']]['text'])
    # cycles between dependent LOP3s
    cyc, last, worst = 0, {}, 99
    for e in seq:
        if e['kind'] in ('chain', 'init', 'fin'):
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
    global BEFORE
    BEFORE = '--before' in sys.argv
    kind = 'hb' if BEFORE else 'hs'
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_%s.cubin' % kind)
    shutil.copy(cub, tmp)
    for pre, sec in (('rg', 'pearl_t75_tile_fold_wmma_rg'), ('d', 'pearl_t75_tile_fold_wmma')):
        data, off, rs = loop(cub, sec)
        seq, words, pairs, sums = build(rs)
        nread, worst = check(rs, seq, sums)
        o, nn = sm75.sections(data)['.text.' + sec]
        body = bytearray(data[o:o + nn])
        before = md5(body)
        wl = []
        for x, w in zip(rs, words):
            if w != x['word']:
                body[x['a']:x['a'] + 16] = w
                wl.append(dict(addr='0x%x' % x['a'], old=x['word'].hex(), new=w.hex()))
        spec = dict(description='goal-nv20n hand readout v2: two xor chains, one half of each pair from the IMMA C '
                    'reuse cache, the other read before its IMMA', section=sec, before=before, after=md5(body),
                    words=wl)
        fn = os.path.join(out, '%s_%s.json' % (pre, kind))
        with open(fn, 'w', newline='\n') as f:
            f.write(json.dumps(spec, indent=0) + '\n')
        r = subprocess.run([PY, '-I', os.path.join(TUR, 'sm75patch.py'), tmp, fn], capture_output=True, text=True)
        print('%s: %d pairs, %d values checked, min dependent gap %d cycles, %d words; %s' % (
            sec, len(pairs), nread, worst, len(wl), (r.stdout + r.stderr).strip()[-60:]))
    r = subprocess.run([sm75.DIS, '-c', tmp], capture_output=True, text=True)
    print('nvdisasm rc', r.returncode)
    with open(tmp + '.sass', 'w') as f:
        f.write(r.stdout)


if __name__ == '__main__':
    main()
