"""ub75.py: timing kernels for the card, made with full75.py's tools. Not part of any build.

  python ub75.py <in.cubin built with -DQC_HAND75=2> <out dir> [only=name,name] [dump=streams.json]
                 [sets=tag:key=value,key=value;tag2:...] [key=value ...]

Each test replaces qc_hand_loopall's body with a loop of one instruction pattern (or one of the hand body's own
blocks; "base" is six adds, once) and writes <out dir>/<name>.cubin. The kernels compute nothing useful: tools/qcdrv.cu times them
("time" mode, no check), and the rate with the card's clock gives SM-cycles a call. With the "base" test (an empty
loop) taken off, that is the cost of the pattern's instructions as they really issue, six warps a scheduler, with
the control fields full75.py would give them.

A pattern is a list of forms (FORMS below). Six chains each have their own source and destination registers, so
nothing waits on a result; a form's carries stay inside its group of three chains. Prints one line a test:
  ubt <name> K=<instructions a trip> N=<trips a call> <form>=<count a trip> ...
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import full75  # noqa: E402

gen, hand, IF = full75.gen, full75.hand, full75.IF

FORMS = {
    'a1': 'IADD3 {d}, {x}, RZ, RZ',
    'a2': 'IADD3 {d}, {x}, {y}, RZ',
    'a3': 'IADD3 {d}, {x}, {y}, {a}',
    'a2i': 'IADD3 {d}, {x}, 0x5bd1e995, RZ',
    'a3i': 'IADD3 {d}, {x}, 0x5bd1e995, {y}',
    'a2o': 'IADD3 {d}, {p}, {x}, {y}, RZ',                              # a carry out
    'a2x': 'IADD3.X {e}, {y}, {a}, RZ, {p}, !PT',                       # and the add that takes it
    'a3oo': 'IADD3 {d}, {p}, {q}, {x}, {y}, {a}',                       # two carries out
    'a3xx': 'IADD3.X {e}, {p}, {q}, {x}, {y}, {a}, {p}, {q}',           # two in, two out (the fold's high word)
    'a3n': 'IADD3.X {d}, {p}, {q}, {x}, ~{y}, ~{a}, !{p}, !{q}',        # the fold's low word
    'a2xi': 'IADD3.X {e}, {x}, -0x2, RZ, {p}, {q}',                     # the fold's last two
    'a1x': 'IADD3.X {e}, RZ, RZ, {x}, {p}, {q}',                        # a count of two carries
    'w': 'IMAD.WIDE.U32 {w}, {x}, {y}, RZ',
    'wa': 'IMAD.WIDE.U32 {w}, {x}, {y}, {v}',
    'wi': 'IMAD.WIDE.U32 {w}, {x}, 0x1e3779b9, RZ',
    'wai': 'IMAD.WIDE.U32 {w}, {x}, 0x1e3779b9, {v}',
    'wo': 'IMAD.WIDE.U32 {w}, {p}, {x}, {y}, {v}',                      # addend and carry out
    'mx0': 'IMAD.X {d}, RZ, RZ, {x}, {p}',
    'mx1': 'IMAD.X {d}, {x}, 0x1, {y}, {p}',
    'mm': 'IMAD.MOV.U32 {d}, RZ, RZ, {x}',
    'ld': 'LDC.64 {w}, c[0x3][{b}]',                                    # a constant load, as the layer's
}

# name: (pattern, order). Orders: g groups of three chains, form by form (the default); c chain by chain.
TESTS = [
    ('base', 'a1', 'g'),
    ('a1', 'a1', 'g'), ('a2', 'a2', 'g'), ('a3', 'a3', 'g'), ('a2i', 'a2i', 'g'), ('a3i', 'a3i', 'g'),
    ('c64', 'a2o a2x', 'g'), ('c96', 'a3oo a3xx', 'g'), ('fold', 'a3n a3xx a2xi a2xi', 'g'), ('a1x', 'a2o a1x', 'g'),
    ('w', 'w', 'g'), ('wa', 'wa', 'g'), ('wi', 'wi', 'g'), ('wai', 'wai', 'g'), ('wo', 'wo a1x', 'g'),
    ('mx0', 'mx0', 'g'), ('mx1', 'mx1', 'g'), ('mm', 'mm', 'g'),
    ('w_a2', 'w a2', 'g'), ('w_2a2', 'w a2 a2', 'g'), ('w_3a2', 'w a2 a2 a2', 'g'), ('w_4a2', 'w a2 a2 a2 a2', 'g'),
    ('w_6a2', 'w a2 a2 a2 a2 a2 a2', 'g'),
    ('wa_a2', 'wa a2', 'g'), ('wa_2a2', 'wa a2 a2', 'g'), ('wa_3a2', 'wa a2 a2 a2', 'g'), ('wa_4a2', 'wa a2 a2 a2 a2', 'g'),
    ('w_a3', 'w a3', 'g'), ('w_2a3', 'w a3 a3', 'g'), ('w_3a3', 'w a3 a3 a3', 'g'),
    ('wa_a3', 'wa a3', 'g'), ('wa_2a3', 'wa a3 a3', 'g'),
    ('w_mx0', 'w mx0', 'g'), ('w_2mx0', 'w mx0 mx0', 'g'), ('w_mx1', 'w mx1', 'g'), ('w_mm', 'w mm', 'g'),
    ('w_2mm', 'w mm mm', 'g'),
    ('a2_mx0', 'a2 mx0', 'g'), ('a2_mx1', 'a2 mx1', 'g'), ('a2_mm', 'a2 mm', 'g'), ('a3_mx0', 'a3 mx0', 'g'),
    ('2a2_mx0', 'a2 a2 mx0', 'g'), ('a2_2mx0', 'a2 mx0 mx0', 'g'),
    ('w_2a2_mx0', 'w a2 a2 mx0', 'g'), ('w_2a2_2mx0', 'w a2 a2 mx0 mx0', 'g'), ('w_4a2_2mx0', 'w a2 a2 a2 a2 mx0 mx0', 'g'),
    ('wa_2a2_mm', 'wa a2 a2 mm', 'g'), ('w_3a2_c', 'w a2 a2 a2', 'c'), ('wa_3a2_c', 'wa a2 a2 a2', 'c'),
    ('mulA', 'w wo w wa mm a3n a3xx a2xi a2xi', 'g'),       # the shipped multiply's mix
    ('mulB', 'w wo w wa mm a3n a3xx mx0 mx0', 'g'),         # its last two on the multiplier
    ('mulC', 'w wo w w a2o a3n a3xx a2xi a2xi', 'g'),       # no anchor
    ('dm', 'wa wo wa wo a3n a3xx a2xi a2xi', 'g'),          # the diagonal multiply's mix
    ('wa_3a3', 'wa a3 a3 a3', 'g'), ('wa_a2_a3', 'wa a2 a3', 'g'), ('2wa_2a3', 'wa wa a3 a3', 'g'),
    ('2wa_2a2', 'wa wa a2 a2', 'g'), ('2w_2a3', 'w w a3 a3', 'g'), ('wa_c96', 'wa a3oo a3xx', 'g'),
    ('wa_c64', 'wa a2o a2x', 'g'), ('wo_2a3', 'wo a3 a3', 'g'), ('wai_2a3', 'wai a3 a3', 'g'),
    ('ld', 'ld', 'g'), ('ld_a2', 'ld a2', 'g'), ('ld_3a2', 'ld a2 a2 a2', 'g'), ('ld_mx0', 'ld mx0', 'g'),
]


def chains(avail, n=6):
    pool = sorted(avail)
    pairs = [r for r in pool if r % 2 == 0 and r + 1 in pool]
    used, ch = set(), []

    def pair():
        r = next(r for r in pairs if r not in used and r + 1 not in used)
        used.update((r, r + 1))
        return r

    def single():
        r = next(r for r in sorted(pool, key=lambda r: ((r ^ 1) in pool and (r ^ 1) not in used, r)) if r not in used)
        used.add(r)
        return r

    cnt = single()
    for c in range(n):
        d = dict(w='R%d' % pair(), v='R%d' % pair())
        for k in 'xyade':
            d[k] = 'R%d' % single()
        d['p'], d['q'] = 'P%d' % (2 * (c % 3)), 'P%d' % (2 * (c % 3) + 1)
        ch.append(d)
    return cnt, ch


def pattern(pat, order, ch, room):
    forms = pat.split()
    step = []
    if order == 'c':
        for d in ch:
            step += [FORMS[f].format(**d) for f in forms]
    else:
        for g in (ch[:3], ch[3:]):
            for f in forms:
                step += [FORMS[f].format(**d) for d in g]
    reps = max(1, min(480, room) // max(1, len(step)))
    names = []
    if order == 'c':
        for _d in ch:
            names += forms
    else:
        for g in (ch[:3], ch[3:]):
            for f in forms:
                names += [f] * len(g)
    return step * reps, {f: forms.count(f) * len(ch) * reps for f in set(forms)}, names * reps


def block_code(code, names):
    """The hand body's own blocks as one loop body: the lines of these blocks without their loop bookkeeping."""
    out = []
    for text, unit, b in code:
        if b not in names or 'BRA' in text or text.startswith(('UIADD3', 'ISETP')):
            continue
        m = re.fullmatch(r'IADD3 (R\d+), (R\d+), 0x1, RZ', text)
        if m and m[1] == m[2]:
            continue
        if text.startswith('UMOV'):
            continue
        out.append([text, unit, 'S'])
    return out


def main():
    cubin, outdir = sys.argv[1], sys.argv[2]
    kv = dict(a.split('=', 1) for a in sys.argv[3:])
    only = kv.pop('only', '')
    only = only.split(',') if only else None
    calls = int(kv.pop('calls', 16000))
    dump = kv.pop('dump', '')
    # sets=tag:key=value,key=value;tag2:... : the body's blocks again with these options over the given ones, as
    # tests S.tag, E.tag, P.tag
    sets = [s.split(':', 1) for s in kv.pop('sets', '').split(';') if s]
    streams = {}
    base_opt = dict(full75.OPT)
    base_opt.update({k: int(v) for k, v in kv.items()})
    saved = (full75.ORDER, full75.LABEL, full75.PATH)
    os.makedirs(outdir, exist_ok=True)

    def body(extra):
        full75.ORDER, full75.LABEL, full75.PATH = saved
        full75.OPT.clear()
        full75.OPT.update(base_opt)
        full75.OPT.update(extra)
        r = full75.body_code(cubin)
        full75.ORDER = ['ent', 'S', 'Sx', 'exit']
        full75.LABEL = {'.S': 'S'}
        full75.PATH = ['ent'] + ['S', 'Sx'] * 3 + ['exit']
        return r

    def emit(name, ent, loop, fnames, c, counts, info):
        n = 1 if name == 'base' else max(1, round(calls / len(loop)))
        prog = ent + loop + [['IADD3 R%d, R%d, 0x1, RZ' % (c, c), 'A', 'Sx'],
                             ['ISETP.NE.AND P6, PT, R%d, 0x%x, PT' % (c, n), 'A', 'Sx'],
                             ['@P6 BRA .S', 'B', 'Sx']]
        for x in info['retfix']:
            prog.append([x, gen.unit_of(x), 'exit'])
        prog.append(['BRA .RET', 'B', 'exit'])
        try:
            stalls, per, _rows, _labels = full75.finish(cubin, os.path.join(outdir, name + '.cubin'), info, prog)
        except KeyError as e:
            print('ubt %s SKIP %s' % (name, e))
            return
        k0 = len(ent)
        streams[name] = dict(n=n, loop=[[f, stalls[k0 + j]] for j, f in enumerate(fnames)],
                             tail=[stalls[k0 + len(loop) + j] for j in range(3)])
        print('ubt %s K=%d N=%d cyc1=%d %s' % (name, len(loop), n, per['S'][1],
                                              ' '.join('%s=%d' % kv_ for kv_ in sorted(counts.items()))))

    def blocks(tag, code, info):
        ent0 = [x for x in code if x[2] == 'ent' and 'BRA' not in x[0]] +                [[x[0], x[1], 'ent'] for x in code if x[0].startswith('UMOV')]
        c = int(re.search(r'IADD3 R(\d+), RZ, 0x0, RZ', ' ; '.join(x[0] for x in ent0))[1])
        for order in ('S', 'E', 'P', 'SE', 'SEi'):
            name = order + tag
            if only and name not in only and order not in only:
                continue
            names = {'S': ('S',), 'E': ('E',), 'P': ('Ploop',), 'SE': ('S', 'E'), 'SEi': ('S', 'E')}[order]
            loop = block_code(code, names)
            if order == 'SEi':
                # the layer's instructions spread evenly among the S-boxes'. The values are wrong (the layer reads
                # S-boxes that have not run); only the time means anything: what would merging the two blocks buy?
                s_, e_ = block_code(code, ('S',)), block_code(code, ('E',))
                loop, k = [], 0
                for j, x in enumerate(s_):
                    loop.append(x)
                    while k < len(e_) and (k + 1) * len(s_) <= (j + 1) * len(e_):
                        loop.append(e_[k])
                        k += 1
                loop += e_[k:]
            fnames = [full75.form(x[0]) for x in loop]
            counts = {}
            for f in fnames:
                counts[f] = counts.get(f, 0) + 1
            emit(name, list(ent0), loop, fnames, c, counts, info)

    info, const, code, _home, avail = body({})
    room = (info['ret'] - info['entry']) // 16
    cnt, ch = chains(avail)
    used = set(int(v[1:]) for d in ch for k, v in d.items() if k not in 'pq') | {cnt}
    used |= set(int(d[k][1:]) + 1 for d in ch for k in 'wv')
    cb = next(r for r in sorted(avail) if r not in used)       # the loads' base register
    for d in ch:
        d['b'] = 'R%d' % cb
    for name, pat, order in TESTS:
        if pat is None or (only and name not in only):
            continue
        loop, counts, fnames = pattern(pat, order, ch, room - 40)
        if name == 'base':
            loop, counts, fnames = loop[:6], {'a1': 6}, fnames[:6]
        loop = [[t_, 'A' if t_.startswith('LDC') else gen.unit_of(t_), 'S'] for t_ in loop]
        ent = [['IADD3 R%d, RZ, 0x0, RZ' % cnt, 'A', 'ent'],
               ['IADD3 R%d, RZ, 0x%x, RZ' % (cb, full75.CONSTS[0]['seq']), 'A', 'ent']]
        emit(name, ent, loop, fnames, cnt, counts, info)
    blocks('', code, info)
    for tag, opts in sets:
        try:
            info, const, code, _home, avail = body({k: int(v) for k, v in (x.split('=') for x in opts.split(','))})
        except Exception as e:
            print('ubt S.%s SKIP %s' % (tag, e))
            continue
        blocks('.' + tag, code, info)
    if dump:
        import json
        json.dump(streams, open(dump, 'w'))


if __name__ == '__main__':
    main()
