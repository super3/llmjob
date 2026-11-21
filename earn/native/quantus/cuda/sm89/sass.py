"""sass.py: parse nvdisasm text of the sm_89 integer subset, simulate it, and list defs/uses.

An instruction text is what nvdisasm prints (without the trailing ';'). parse() gives an Op with:
  pred (None or (neg, n)), op (e.g. 'IADD3.X'), dregs (registers written, pairs expanded), dpreds (predicates
  written), uregs (registers read), upreds (predicates read), and the operand list for the simulator.
"""
import re

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1


class Opd:
    """One operand: kind 'r' (register, n=255 for RZ), 'i' (immediate), 'c' (constant bank), 'p' (predicate,
    n=7 for PT), 'u' (uniform register, 63 for URZ); neg '-' or '~' or '!' or ''."""
    def __init__(self, kind, n, neg=''):
        self.kind, self.n, self.neg = kind, n, neg

    def __repr__(self):
        return '%s%s%s' % (self.neg, self.kind, self.n)


def popd(s):
    s = s.strip().replace('.reuse', '')
    neg = ''
    if s[:1] in '-~!' and not s.startswith('-0x'):
        neg, s = s[0], s[1:]
    if s == 'RZ':
        return Opd('r', 255, neg)
    if s == 'PT':
        return Opd('p', 7, neg)
    if s == 'URZ':
        return Opd('u', 63, neg)
    m = re.fullmatch(r'R(\d+)', s)
    if m:
        return Opd('r', int(m[1]), neg)
    m = re.fullmatch(r'P(\d)', s)
    if m:
        return Opd('p', int(m[1]), neg)
    m = re.fullmatch(r'UR(\d+)', s)
    if m:
        return Opd('u', int(m[1]), neg)
    m = re.fullmatch(r'(-?)0x([0-9a-f]+)', s)
    if m:
        v = int(m[2], 16)
        return Opd('i', (-v if m[1] else v) & M32, neg)
    m = re.fullmatch(r'c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]', s)
    if m:
        return Opd('c', (int(m[1], 16), int(m[2], 16)), neg)
    m = re.fullmatch(r'c\[0x([0-9a-f]+)\]\[UR(\d+)(?:\+0x([0-9a-f]+))?\]', s)
    if m:
        return Opd('cu', (int(m[1], 16), int(m[2]), int(m[3] or '0', 16)), neg)
    raise ValueError('operand ' + s)


def npreds(op, a):
    """How many predicate destinations follow the register destination."""
    if op in ('IADD3', 'IMAD.WIDE.U32', 'IMAD.WIDE'):
        return len(a) - 4
    if op == 'IADD3.X':
        return len(a) - 6
    if op == 'IMAD.WIDE.U32.X':
        return len(a) - 5
    return 0


class Op:
    def __init__(self, text):
        self.text = re.sub(r'\s+', ' ', text.strip().rstrip(';').strip())
        t = self.text
        self.pred = None
        m = re.match(r'@(!?)P(\d) (.*)$', t)
        if m:
            self.pred = (m[1] == '!', int(m[2]))
            t = m[3]
        op, _, rest = t.partition(' ')
        self.op = op
        self.target = None
        if '`' in rest:
            rest, _, tg = rest.partition('`')
            self.target = '`' + tg
            rest = rest.strip()
        self.args = [popd(a) for a in rest.split(',')] if rest else []
        self._defs_uses()

    def _defs_uses(self):
        op, a = self.op, self.args
        d, dp, u, up = [], [], [], []
        if self.pred and self.pred[1] != 7:
            up.append(self.pred[1])
        base = op.split('.')[0]
        wide = 'WIDE' in op
        if base in ('IMAD', 'IADD3', 'SEL', 'MOV', 'LOP3', 'SHF', 'LEA', 'PRMT', 'LDC', 'ISETP', 'S2R', 'CS2R'):
            i = 0
            if base == 'ISETP':
                dp.append(a[0].n)
                i = 2  # P, PT
                srcs = a[i:]
            else:
                if a[0].kind == 'r' and a[0].n != 255:
                    d.append(a[0].n)
                    if wide or '.64' in op:
                        d.append(a[0].n + 1)
                i = 1
                for _ in range(npreds(op, a)):
                    dp.append(a[i].n)
                    i += 1
                srcs = a[i:]
            for j, s in enumerate(srcs):
                if s.kind == 'r' and s.n != 255:
                    u.append(s.n)
                    # the third source of a WIDE is a 64-bit pair
                    if wide and j == 2:
                        u.append(s.n + 1)
                elif s.kind == 'p' and s.n != 7:
                    up.append(s.n)
            self.srcs = srcs
        else:
            self.srcs = a
            for s in a:
                if s.kind == 'r' and s.n != 255:
                    u.append(s.n)
                elif s.kind == 'p' and s.n != 7:
                    up.append(s.n)
        self.dregs, self.dpreds = d, [x for x in dp if x != 7]
        self.uregs, self.upreds = u, up

    def __repr__(self):
        return self.text


def unit(op):
    """'W' IMAD.WIDE, 'M' other IMAD (FMA pipe), 'A' ALU, 'B' branch, 'U' uniform, 'L' memory/constant."""
    b = op.op
    if b.startswith('IMAD.WIDE'):
        return 'W'
    if b.startswith('IMAD') or b == 'MOV' and False:
        return 'M'
    if b.startswith(('IADD3', 'SEL', 'LOP3', 'ISETP', 'SHF', 'LEA', 'MOV', 'PRMT', 'PLOP3')):
        return 'A'
    if b.startswith(('BRA', 'RET', 'CALL', 'EXIT')):
        return 'B'
    if b.startswith(('U', 'ULDC')):
        return 'U'
    return 'L'


class Machine:
    """Registers, predicates and a constant bank (dict (bank, offset) -> 32-bit word)."""
    def __init__(self, const=None):
        self.r = [0] * 256
        self.p = [False] * 8
        self.p[7] = True
        self.ur = [0] * 64
        self.const = const or {}

    def get(self, o):
        if o.kind == 'r':
            v = 0 if o.n == 255 else self.r[o.n]
        elif o.kind == 'i':
            v = o.n
        elif o.kind == 'c':
            v = self.const[o.n]
        elif o.kind == 'u':
            v = 0 if o.n == 63 else self.ur[o.n]
        else:
            raise ValueError(o)
        if o.neg == '~':
            v = ~v & M32
        return v

    def get64(self, o):
        if o.kind == 'r':
            return 0 if o.n == 255 else self.r[o.n] | self.r[o.n + 1] << 32
        raise ValueError(o)

    def pv(self, o):
        v = self.p[o.n]
        return (not v) if o.neg == '!' else v

    def setr(self, n, v):
        if n != 255:
            self.r[n] = v & M32

    def setp(self, n, v):
        if n != 7:
            self.p[n] = bool(v)

    def run(self, op):
        """Executes one Op; returns 'BRA' with the target when a branch is taken, else None."""
        if op.pred:
            v = self.p[op.pred[1]]
            if op.pred[0]:
                v = not v
            if not v:
                return None
        o, a = op.op, op.args
        if o.startswith('IMAD.WIDE.U32'):
            i = 1
            ps = []
            for _ in range(npreds(o, a)):
                ps.append(a[i].n)
                i += 1
            ra, rb, rc = a[i], a[i + 1], a[i + 2]
            x = self.get(ra) * self.get(rb) + (0 if rc.n == 255 and rc.kind == 'r' else self.get64(rc))
            if o.endswith('.X'):
                x += 1 if self.pv(a[i + 3]) else 0
            self.setr(a[0].n, x)
            self.setr(a[0].n + 1, x >> 32)
            if ps:
                self.setp(ps[0], x >> 64)
            return None
        if o in ('IADD3', 'IADD3.X'):
            i = 1
            ps = []
            for _ in range(npreds(o, a)):
                ps.append(a[i].n)
                i += 1
            s = 0
            for x in a[i:i + 3]:
                v = self.get(x)
                if x.neg == '-':
                    v = (1 << 32) - v   # two's complement as a 33-bit quantity: a - b carries when a >= b
                s += v
            if o == 'IADD3.X':
                s += (1 if self.pv(a[i + 3]) else 0) + (1 if self.pv(a[i + 4]) else 0)
            self.setr(a[0].n, s)
            cnt = s >> 32
            if len(ps) == 1:
                assert cnt <= 1 or True
                self.setp(ps[0], cnt & 1 if cnt <= 1 else 1)
                if cnt > 1:
                    self.overflow = getattr(self, 'overflow', 0) + 1
            elif len(ps) == 2:
                self.setp(ps[0], cnt >= 1)
                self.setp(ps[1], cnt >= 2)
            return None
        if o == 'IMAD.X':
            v = self.get(a[1]) * self.get(a[2]) + self.get(a[3]) + (1 if self.pv(a[4]) else 0)
            self.setr(a[0].n, v)
            return None
        if o in ('IMAD.MOV.U32', 'IMAD.MOV'):
            self.setr(a[0].n, self.get(a[3]))
            return None
        if o in ('IMAD', 'IMAD.IADD'):
            self.setr(a[0].n, self.get(a[1]) * self.get(a[2]) + self.get(a[3]))
            return None
        if o == 'MOV':
            self.setr(a[0].n, self.get(a[1]))
            return None
        if o == 'SEL':
            self.setr(a[0].n, self.get(a[1]) if self.pv(a[3]) else self.get(a[2]))
            return None
        if o == 'ISETP.NE.AND':
            self.setp(a[0].n, (self.get(a[2]) != self.get(a[3])) and self.pv(a[4]))
            return None
        if o == 'UMOV':
            self.ur[a[0].n] = self.get(a[1])
            return None
        if o == 'UIADD3':
            self.ur[a[0].n] = (self.get(a[1]) + self.get(a[2]) + self.get(a[3])) & M32
            return None
        if o == 'ULDC.64':
            bank, ur, k = a[1].n
            off = self.ur[ur] + k
            self.ur[a[0].n] = self.const[(bank, off)]
            self.ur[a[0].n + 1] = self.const[(bank, off + 4)]
            return None
        if o == 'LDC.64':
            raise NotImplementedError(op.text)
        if o == 'NOP':
            return None
        if o == 'BRA' or o == 'CALL.REL.NOINC':
            return 'BRA'
        raise NotImplementedError(op.text)
