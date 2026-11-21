"""iface.py: the compiled qc_hand_loop's register interface, found by simulating it.

  python iface.py <cubin (carryfix applied)> [--nvdisasm PATH]

Prints a JSON line: {"in": [[lo, hi] x 12], "out": [[lo, hi] x 12], "entry": addr, "ret": addr, "retreg": n,
"clob": [...], "cpreds": [...]} and checks the simulated function against the reference on random states.
"""
import json
import os
import random
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sass  # noqa: E402
import ref  # noqa: E402

NVDISASM = [os.environ.get('NVDISASM', 'nvdisasm')]
LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\* 0x([0-9a-f]{16}) \*/")
HIGH = re.compile(r"^\s+/\* 0x([0-9a-f]{16}) \*/")


def sections(b):
    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    entsize, num, strndx = struct.unpack_from("<HHH", b, 0x3A)
    raw = [struct.unpack_from("<IIQQQQ", b, shoff + i * entsize) for i in range(num)]
    stro = raw[strndx][4]
    out = {}
    for name, _t, _f, _a, off, size in raw:
        end = b.index(b"\0", stro + name)
        out[bytes(b[stro + name:end]).decode()] = (off, size)
    return out


def const_bank3(cubin):
    b = open(cubin, 'rb').read()
    secs = sections(b)
    name = [n for n in secs if n.startswith('.nv.constant3')]
    assert len(name) == 1, name
    off, size = secs[name[0]]
    return {(3, i): struct.unpack_from('<I', b, off + i)[0] for i in range(0, size, 4)}


def listing(cubin, nvdisasm=None):
    r = subprocess.run([nvdisasm or NVDISASM[0], '-hex', '-c', cubin], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    txt = r.stdout.decode('utf-8', 'replace')
    L = txt.splitlines()
    rows, labels, pend, fn_entry, fn = [], {}, [], None, False
    for i, l in enumerate(L):
        s = l.strip()
        if re.match(r'\$\S*qc_hand_loop\S*:', s):
            fn = True
            continue
        m = re.match(r'(\.L_x_\d+):', s)
        if m:
            pend.append(m[1])
            continue
        m = LINE.match(l)
        if m:
            a = int(m[1], 16)
            rows.append(dict(addr=a, text=re.sub(r'\s+', ' ', m[2]), lo=int(m[3], 16),
                             hi=int(HIGH.match(L[i + 1])[1], 16)))
            for p in pend:
                labels[p] = a
            pend = []
            if fn and fn_entry is None:
                fn_entry = a
    return rows, labels, fn_entry


def function(cubin):
    rows, labels, entry = listing(cubin)
    body = [r for r in rows if r['addr'] >= entry]
    ret = next(r for r in body if r['text'].startswith('RET.REL'))
    body = [r for r in body if r['addr'] <= ret['addr']]
    retreg = int(re.match(r'RET\.REL\.NODEC R(\d+)', ret['text'])[1])
    return body, labels, entry, ret['addr'], retreg


def simulate(body, labels, const, regs):
    m = sass.Machine(const)
    for k, v in regs.items():
        m.r[k] = v
    ops = [sass.Op(r['text']) for r in body]
    idx = {r['addr']: i for i, r in enumerate(body)}
    i, steps = 0, 0
    while True:
        op = ops[i]
        if op.op.startswith('RET'):
            return m
        if m.run(op) == 'BRA':
            tgt = re.search(r'\((\.L_x_\d+)\)', op.target)[1]
            i = idx[labels[tgt]]
        else:
            i += 1
        steps += 1
        assert steps < 100000


def live_in(body):
    """Registers read before written in the first pass over the body (straight-line order, which over-reads
    nothing here: the loop's carried registers are written before the loop)."""
    written, li = set(), []
    for r in body:
        op = sass.Op(r['text'])
        for u in op.uregs:
            if u not in written and u not in li:
                li.append(u)
        written.update(op.dregs)
    return li


def find_iface(cubin):
    body, labels, entry, ret, retreg = function(cubin)
    const = const_bank3(cubin)
    li = live_in(body)
    rnd = random.Random(5)
    vals = {r: rnd.getrandbits(32) for r in li}
    # Where does each input value go: run up to the loop's first instruction and record each WIDE by an
    # immediate (the diagonal multiplies' a0 * d0 and a1 * dp1 products) in the first trip.
    D, DP = ref.D, ref.DP
    imm_lo = {}
    for i in range(12):
        imm_lo[D[i] & 0xFFFFFFFF] = ('lo', i)
        imm_lo[DP[i] >> 32] = ('hi', i)
    m = sass.Machine(const)
    for k, v in vals.items():
        m.r[k] = v
    inv = {v: k for k, v in vals.items()}
    found = {}
    first_loop = labels[sorted(labels, key=lambda n: labels[n])[0]] if labels else None
    ops = [sass.Op(r['text']) for r in body]
    for r, op in zip(body, ops):
        if op.op.startswith('IMAD.WIDE') and len(op.srcs) >= 2 and op.srcs[1].kind == 'i':
            key = op.srcs[1].n
            if key in imm_lo and imm_lo[key] not in found:
                v = m.get(op.srcs[0])
                if v in inv:
                    found[imm_lo[key]] = inv[v]
        if op.op.startswith(('BRA', 'RET')):
            break
        m.run(op)
    x0 = ops[0]
    assert x0.op == 'IMAD.WIDE.U32', x0
    found[('lo', 0)] = x0.srcs[0].n
    found[('hi', 0)] = x0.srcs[1].n
    missing = [(h, i) for i in range(12) for h in ('lo', 'hi') if (h, i) not in found]

    free = [r for r in li if r not in found.values()]
    if missing:
        import itertools
        for perm in itertools.permutations(free, len(missing)):
            f2 = dict(found)
            f2.update(zip(missing, perm))
            s = [rnd.randrange(ref.P) for _ in range(12)]
            regs = {}
            for i in range(12):
                regs[f2[('lo', i)]] = s[i] & 0xFFFFFFFF
                regs[f2[('hi', i)]] = s[i] >> 32
            mm = simulate(body, labels, const, regs)
            want = ref.hand_loop(s)
            vals = set((mm.r[a] | mm.r[b] << 32) % ref.P for a in range(80) for b in range(80))
            if all(w in vals for w in want):
                found = f2
                break
    inp = [[found[('lo', i)], found[('hi', i)]] for i in range(12)]
    # Outputs: simulate random states, match register pairs against the reference.
    outs = None
    for trial in range(3):
        s = [rnd.randrange(ref.P) for _ in range(12)]
        regs = {}
        for i in range(12):
            regs[inp[i][0]] = s[i] & 0xFFFFFFFF
            regs[inp[i][1]] = s[i] >> 32
        mm = simulate(body, labels, const, regs)
        want = ref.hand_loop(s)
        cand = []
        for i in range(12):
            c = set()
            for a in range(256):
                for b in range(256):
                    if a != b and (mm.r[a] | mm.r[b] << 32) % ref.P == want[i] % ref.P:
                        c.add((a, b))
            cand.append(c)
        outs = cand if outs is None else [o & c for o, c in zip(outs, cand)]
    regs_w, preds_w = set(), set()
    for op in ops:
        regs_w.update(op.dregs)
        preds_w.update(op.dpreds)
    # A value may sit in more than one register (copies): the replacement writes every one of them.
    out = [sorted([list(x) for x in o if x[0] in regs_w and x[1] in regs_w]) for o in outs]
    assert all(out), outs
    return dict(inp=inp, out=out, entry=entry, ret=ret, retreg=retreg, clob=sorted(regs_w), cpreds=sorted(preds_w),
                live_in=li, n=len(body))


if __name__ == '__main__':
    print(json.dumps(find_iface(sys.argv[1])))
