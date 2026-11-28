"""patch75.py: put the hand-written partial rounds (gen75.py) into an sm_75 cubin built with -DQC_HAND75=1.

  python patch75.py <in.cubin> <out.cubin> [key=value ...]

It drives sm89/hand.py and sm89/patch.py (builder, scheduler, register allocator, encoder by template, read-back
through nvdisasm, simulation against ref.py) with this folder's arithmetic and control-field rules:

  lat=N     every register and carry dependency waits at least N cycles (default 0: sm89/hand.py's table)
  plat=N    a carry-in waits N cycles for its predicate (default 5, or lat when that is more)
  stall=N   every instruction's stall count is at least N (default 1)
  limit=N   the scheduler's cap on live temporary registers (default 24)
  ybit=N    the yield bit is set for stall counts up to N (default 2; ptxas writes it for 1 and 2 only)
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'sm89'))
sys.path.insert(0, HERE)
import enc  # noqa: E402
import gen  # noqa: E402
import hand  # noqa: E402
import iface as IF  # noqa: E402
import patch as P89  # noqa: E402
import ref  # noqa: E402
import gen75  # noqa: E402

OPT = dict(lat=0, plat=0, stall=1, ybit=2, limit=24)


def find_u64(const, vals):
    """Offset in constant bank 3 where the u64 run vals starts."""
    n = max(a for (_b, a) in const) + 4
    for off in range(0, n - 8 * len(vals) + 1, 8):
        if all(const.get((3, off + 8 * i), -1) | const.get((3, off + 8 * i + 4), 0) << 32 == v
               for i, v in enumerate(vals)):
            return off
    raise SystemExit('constants not found in bank 3')


class Templates(enc.Templates):
    """A shape ptxas did not write with these carry-outs is made from the same instruction with fewer of them
    (the predicate-destination fields of an instruction without one hold PT); patch.py reads every word back."""
    def get(self, text):
        sh = enc.shape(text)
        if sh in self.t:
            return super().get(text)
        op, _, cls = sh.rpartition(' ')
        c = cls.split(',')
        for drop in (1, 2):
            if len(c) > 1 + drop and all(x == 'P' for x in c[1:1 + drop]):
                alt = op + ' ' + ','.join(c[:1] + c[1 + drop:])
                if alt in self.t:
                    lo, hi, _ = self.t[alt]
                    return enc.encode(text, (lo, hi))
        if op.endswith('IMAD.WIDE.U32.X'):
            # with a carry-in: the same instruction without one, plus bit 74 (what ptxas's register form differs by)
            alt = op[:-2] + ' ' + ','.join(c[:-1])
            for cand in (alt, op[:-2] + ' ' + ','.join(c[:1] + [x for x in c[1:-1] if x != 'P'][:0] + c[2:-1])):
                if cand in self.t:
                    lo, hi, _ = self.t[cand]
                    return enc.encode(text, (lo, hi | 1 << 10))
        raise KeyError('no template for %r (%s)' % (text, sh))


def ctrl(stall, _yld):
    stall = max(stall, OPT['stall'])
    return (stall & 15) | ((1 if stall <= OPT['ybit'] else 0) << 4) | (7 << 5) | (7 << 8)


_gen_lat = hand.gen_lat


def gen_lat(pu, cu, role_c):
    return max(_gen_lat(pu, cu, role_c), OPT['lat'])


def enforce(texts, stalls, bl):
    """Raises stall counts until every register and predicate read is far enough behind its write along the path
    prologue, loop, loop, epilogue, exit. sm89/hand.py's timing pass forgets what is in flight at a block's edge
    (it waits one cycle there), so a value written at the end of the prologue could be read too early in the loop.
    Distances: hand.py's table or lat for registers, 5 or plat for carry-ins, 13 or lat for a branch's predicate."""
    import re
    import sass
    n = len(texts)
    ops = [None if 'BRA' in x else sass.Op(x) for x in texts]
    units = [gen.unit_of(x) for x in texts]
    idx = {b: [i for i in range(n) if bl[i] == b] for b in ('pro', 'loop', 'epi', 'exit')}
    path = idx['pro'] + idx['loop'] + idx['loop'] + idx['epi'] + idx['exit']
    plat = OPT.get('plat') or max(5, OPT['lat'])
    for _round in range(100000):
        t, wr, wp, short = 0, {}, {}, None
        for pos, i in enumerate(path):
            need = t
            if ops[i] is None:
                m = re.match(r'@!?P(\d) BRA', texts[i])
                if m and int(m.group(1)) in wp:
                    need = max(need, wp[int(m.group(1))] + max(13, OPT['lat']))
            else:
                op = ops[i]
                addend = set()
                if op.op.startswith('IMAD.WIDE') and op.srcs[2].kind == 'r' and op.srcs[2].n != 255:
                    addend = {op.srcs[2].n, op.srcs[2].n + 1}
                for r in op.uregs:
                    if r in wr:
                        need = max(need, wr[r][0] + gen_lat(wr[r][1], units[i], r in addend))
                for p in op.upreds:
                    if p in wp:
                        need = max(need, wp[p] + plat)
            if need > t:
                short = (pos, need - t)
                break
            if ops[i] is not None:
                for r in ops[i].dregs:
                    wr[r] = (t, units[i])
                for p in ops[i].dpreds:
                    wp[p] = t
            t += stalls[i]
        if short is None:
            return stalls
        pos, add = short
        k = pos - 1
        while add > 0:
            assert k >= 0, 'cannot place the wait'
            room = 15 - stalls[path[k]]
            take = min(room, add)
            stalls[path[k]] += take
            add -= take
            k -= 1
    raise RuntimeError('enforce did not settle')


_make = hand.make


def make(*a, **kw):
    texts, stalls, bl, t, g, loc = _make(*a, **kw)
    stalls = enforce(texts, list(stalls), bl)
    return texts, stalls, bl, t, g, loc


def run(cubin, out, **opt):
    OPT.update(opt)
    os.environ.setdefault('NVDISASM', 'nvdisasm')
    IF.NVDISASM[0] = os.environ['NVDISASM']
    const = IF.const_bank3(cubin)
    hand.RC_BASE = find_u64(const, ref.RC_P)
    hand.RCT_BASE = find_u64(const, ref.RC_T0)
    assert hand.RCT_BASE == hand.RC_BASE + 8 * 21, (hand.RC_BASE, hand.RCT_BASE)
    gen.Prog = gen75.Prog75
    enc.Templates = Templates
    P89.ctrl = ctrl
    hand.gen_lat = gen_lat
    hand.make = make
    hand.LIMIT[0] = OPT['limit']
    return P89.run(cubin, out, [])


if __name__ == '__main__':
    kv = dict(a.split('=') for a in sys.argv[3:])
    texts, stalls, bl = run(sys.argv[1], sys.argv[2], **{k: int(v) for k, v in kv.items()})
    from collections import Counter
    for b in ('pro', 'loop', 'epi'):
        u = Counter(gen.unit_of(x) for x, y in zip(texts, bl) if y == b)
        cyc = sum(s for s, y in zip(stalls, bl) if y == b)
        print(b, dict(u), 'cycles one warp', cyc)
