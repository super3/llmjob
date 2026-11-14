"""mklist.py <cubin> <kind> <outdir>: goal-nv20n patch lists for the shipped t75 chunk loop, one per fold section
(rg_<kind>.json, d_<kind>.json), checked by applying them to a copy with sm75patch.py.

kinds:
  rz   TIMING ONLY: every unpredicated LOP3 in the loop reads RZ in all three slots (no register reads, same issue,
       same control bits, same waits).
  nop  TIMING ONLY: those LOP3 become NOPs with the same control bits.
  ru   the 28 IMMAs where ptxas left a reuse flag off that the bank rule (reuse2.py) allows get it.
"""
import hashlib
import json
import os
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

PY = sys.executable


def md5(b):
    return hashlib.md5(b).hexdigest()[:12]


def lop_rz(w):
    lo = bytearray(w)
    lo[3] = 255          # ra bits 24-31
    lo[4] = 255          # rb bits 32-39
    lo[8] = 255          # rc bits 64-71
    return bytes(lo)


def to_nop(w):
    c = sm75.getc(w)
    nop = (0x7918).to_bytes(8, 'little') + (0).to_bytes(8, 'little')
    return sm75.setc(nop, c)


def build(cub, sec, kind):
    data, off, rs = loop(cub, sec)
    words = []
    if kind in ('rz', 'nop'):
        for x in rs:
            if x['base'] == 'LOP3' and not x['text'].startswith('@'):
                new = lop_rz(x['word']) if kind == 'rz' else to_nop(x['word'])
                words.append((x['a'], x['word'], new, x['text']))
    elif kind == 'c0':
        # TIMING ONLY: slot c of every readout LOP3 reads RZ (one read fewer each): the best case for taking one
        # operand of each from a reuse cache
        for x in rs:
            if x['base'] == 'LOP3' and not x['text'].startswith('@') and x['word'][8] != 255:
                w = bytearray(x['word'])
                w[8] = 255
                words.append((x['a'], x['word'], bytes(w), x['text']))
    elif kind in ('rzs', 'rzt', 'rss', 'rsb'):
        n = len(rs)
        for i, x in enumerate(rs):
            if x['base'] != 'LOP3' or x['text'].startswith('@'):
                continue
            w = bytearray(x['word'])
            for byte, reg in zip((3, 4, 8), reuse2.slots(x)):
                if reg is None or reg == 255:
                    continue
                # reaching definition, scanning back around the loop
                src = None
                for k in range(1, n + 1):
                    y = rs[(i - k) % n]
                    if ('R%d' % reg) in y['defs']:
                        src = y['base']
                        break
                is_sum = src == 'IMMA'
                if kind == 'rzs' and is_sum:
                    w[byte] = 255
                elif kind == 'rzt' and not is_sum:
                    w[byte] = 255
                elif kind == 'rss' and is_sum:
                    w[byte] = {3: 226, 4: 228, 8: 230}[byte] + reg % 2
                elif kind == 'rsb' and is_sum:
                    w[byte] = {3: 226, 4: 228, 8: 230}[byte]
            if bytes(w) != x['word']:
                words.append((x['a'], x['word'], bytes(w), x['text']))
    elif kind == 'cb':
        # an IMMA right before a LOP3 that reads the IMMA's C register (renamed accumulator, so the value is still
        # the one the IMMA read): reuse flag on the IMMA's C, and that register moved to the LOP3's slot c
        import re
        for p, x in enumerate(rs[:-1]):
            y = rs[p + 1]
            if x['base'] != 'IMMA' or y['base'] != 'LOP3' or ', 0x96,' not in y['text']:
                continue
            ops = x['text'].split(None, 1)[1].split(',')
            dd = int(re.search(r'R(\d+)', ops[0]).group(1))
            c = int(re.search(r'R(\d+)', ops[3]).group(1))
            if dd == c:
                continue
            sl = reuse2.slots(y)
            hit = [i for i, r in enumerate(sl) if r == c]
            if not hit:
                continue
            f = sm75.fields(sm75.getc(x['word']))
            f['reuse'] |= 4
            words.append((x['a'], x['word'], sm75.setc(x['word'], sm75.mk(f)), x['text']))
            w = bytearray(y['word'])
            pos = (3, 4, 8)
            w[pos[hit[0]]], w[8] = w[8], w[pos[hit[0]]]
            words.append((y['a'], y['word'], bytes(w), y['text']))
    elif kind == 'cr':
        # a LOP3 whose next IMMA reads one of its sources as C: that source goes to slot c with a reuse flag
        import re
        n = len(rs)
        for p, x in enumerate(rs):
            if x['base'] != 'LOP3' or x['text'].startswith('@') or ', 0x96,' not in x['text']:
                continue
            q = p + 1
            while q < n and rs[q]['base'] != 'IMMA':
                q += 1
            if q >= n:
                continue
            ops = rs[q]['text'].split(None, 1)[1].split(',')
            c = int(re.search(r'R(\d+)', ops[3]).group(1))
            sl = reuse2.slots(x)
            hit = [i for i, r in enumerate(sl) if r in (c, c + 1)]
            if not hit:
                continue
            w = bytearray(x['word'])
            pos = (3, 4, 8)
            i = hit[0]
            w[pos[i]], w[8] = w[8], w[pos[i]]
            f = sm75.fields(sm75.getc(bytes(w)))
            f['reuse'] |= 4
            words.append((x['a'], x['word'], sm75.setc(bytes(w), sm75.mk(f)), x['text']))
    elif kind == 'ru':
        new = reuse2.reuse_bank(rs)
        for x, y in zip(rs, new):
            if x['base'] == 'IMMA' and x['f']['reuse'] != y['f']['reuse']:
                f = dict(x['f'])
                f['reuse'] = y['f']['reuse']
                words.append((x['a'], x['word'], sm75.setc(x['word'], sm75.mk(f)), x['text']))
    else:
        raise SystemExit('unknown kind')
    _, _, size = data, off, None
    secs = sm75.sections(data)
    o, n = secs['.text.' + sec]
    body = bytearray(data[o:o + n])
    before = md5(body)
    for a, old, nw, _ in words:
        assert body[a:a + 16] == old
        body[a:a + 16] = nw
    after = md5(body)
    timing = kind not in ('ru', 'cr', 'cb')
    desc = ('TIMING ONLY (wrong results): ' if timing else '') + 'goal-nv20n %s on the shipped t75 loop' % kind
    return dict(description=desc, section=sec, before=before, after=after,
                words=[dict(addr='0x%x' % a, old=old.hex(), new=nw.hex(), note=t[:60]) for a, old, nw, t in words])


def main():
    cub, kind, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, 'check_%s.cubin' % kind)
    shutil.copy(cub, tmp)
    for pre, sec in (('rg', 'pearl_t75_tile_fold_wmma_rg'), ('d', 'pearl_t75_tile_fold_wmma')):
        spec = build(cub, sec, kind)
        fn = os.path.join(out, '%s_%s.json' % (pre, kind))
        with open(fn, 'w', newline='\n') as f:
            f.write(json.dumps(spec, indent=0) + '\n')
        r = subprocess.run([PY, '-I', os.path.join(TUR, 'sm75patch.py'), tmp, fn], capture_output=True, text=True)
        print(fn, len(spec['words']), 'words;', (r.stdout + r.stderr).strip())
    r = subprocess.run([sm75.DIS, '-c', tmp], capture_output=True, text=True)
    print('nvdisasm rc', r.returncode, r.stderr.strip()[:200])


main()
