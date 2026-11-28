"""patch.py: put hand.py's code into a cubin in place of qc_hand_loop's body, and read it back.

  python patch.py <cubin (carryfix applied, built with QC_HAND89)> <out.cubin> [template cubins...]
"""
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enc  # noqa: E402
import hand  # noqa: E402
import iface as IF  # noqa: E402
import ref  # noqa: E402


def ctrl(stall, yld):
    """The control field: stall cycles, yield, no scoreboards, no waits, no reuse."""
    return (stall & 15) | ((1 if yld else 0) << 4) | (7 << 5) | (7 << 8)


def bra(text, here, labels):
    """[@[!]Pn] BRA <label>: the relative branch (ptxas 12's sm_89 word, as probes/hand89/asm89.py has it)."""
    m = re.match(r'(?:@(!?)P(\d) )?BRA (\S+)$', text)
    pred = 7 if m.group(2) is None else int(m.group(2)) | (8 if m.group(1) else 0)
    off = labels[m.group(3)] - (here + 16)
    lo = 0x0947 | ((off & 0xFFFFFFFF) << 32) | (pred << 12)
    hi = 0x3800000 | (0x3ffff if off < 0 else 0)
    return lo, hi


def text_section(b, kernel_sub='qc_scan_kernel'):
    secs = IF.sections(b)
    name = [n for n in secs if n.startswith('.text.') and kernel_sub in n]
    assert len(name) == 1, name
    return secs[name[0]]


def run(cubin, out, tmpl_cubins, iface_json=None, **kw):
    info = IF.find_iface(cubin)
    if iface_json:
        json.dump(info, open(iface_json, 'w'))
    const = IF.const_bank3(cubin)
    # the constants hand.py reads where it expects them
    v = lambda a: const[(3, a)] | const[(3, a + 4)] << 32
    assert all(v(hand.RC_BASE + 8 * i) == ref.RC_P[i] for i in range(21))
    assert all(v(hand.RCT_BASE + 8 * i) == ref.RC_T0[i] for i in range(12))
    tmp = out + '.iface.json'
    json.dump(info, open(tmp, 'w'))
    texts, stalls, bl, t, g, loc = hand.make(tmp, cubin, **kw)
    if not hand.gen.HANDV & 16:
        bad = hand.check(texts, bl, info, const)
        assert bad == 0, 'simulation: %d mismatches' % bad
    entry, ret = info['entry'], info['ret']
    room = (ret - entry) // 16
    code = []
    loop_at = bl.index('loop')
    for i, (x, s) in enumerate(zip(texts, stalls)):
        code.append((x, s))
    code.append(('BRA .RET', 5))
    assert len(code) <= room, (len(code), room)
    T = enc.Templates([IF.listing(c)[0] for c in tmpl_cubins + [cubin]])
    b = bytearray(open(cubin, 'rb').read())
    off, _size = text_section(b)
    rows, labels, _e = IF.listing(cubin)
    base = rows[0]['addr']
    labs = {'.LOOP': entry + 16 * loop_at, '.RET': ret}
    want = {}
    for i in range(room):
        a = entry + 16 * i
        if i < len(code):
            x, s = code[i]
            if 'BRA' in x:
                lo, hi = bra(x, a, labs)
            else:
                lo, hi = T.get(x)
            c = ctrl(min(s, 15), s < 4)
            hi = (hi & ~enc.CTRL_MASK) | (c << 41)
            want[a] = x
        else:
            lo, hi = 0x7918, ctrl(0, False) << 41
        struct.pack_into('<QQ', b, off + a - base, lo, hi)
    open(out, 'wb').write(b)
    rows2, labels2, _ = IF.listing(out)
    got = {r['addr']: r['text'] for r in rows2}
    inv = {}
    for n, a in labels2.items():
        inv[n] = a
    nbad = 0
    for a, x in want.items():
        h = got.get(a, '?').replace('.reuse', '')
        if 'BRA' in x:
            m = re.search(r'\((\.L_x_\d+)\)', h)
            tgt = labs[x.split()[-1]]
            ok = bool(m) and inv.get(m[1]) == tgt and x.split('BRA')[0] == h.split('BRA')[0]
        else:
            ok = h == x
        if not ok:
            nbad += 1
            if nbad < 10:
                print('MISMATCH %05x asked %r got %r' % (a, x, h))
    assert nbad == 0, nbad
    # the patched function, simulated from the cubin itself
    body, labels3, e3, r3, rr3 = IF.function(out)
    print('patched %d of %d instructions; loop %d instructions; read back ok' % (len(code), room, bl.count('loop')))
    return texts, stalls, bl


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3:])
