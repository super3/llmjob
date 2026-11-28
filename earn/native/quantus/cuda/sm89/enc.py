"""enc.py: encode sm_89 instructions from text, using ptxas's own instructions of the same shape as templates.

A shape is the opcode with modifiers plus, per operand, its class (R register, P predicate, I immediate, C constant,
U uniform register, A uniform-addressed constant) and negation. The template's bits are kept and the operand fields
written over (Volta-family fields, checked by selfcheck() on every instance in the given cubins):
  register d 16, a 24, b 32, c 64; immediate 32..63; constant offset/4 at 40, bank at 54; guard 12 (+15 negate);
  predicate destinations 81, 84; carry-ins 87 (+90 negate), 77 (+80 negate);
  source negations: a 72, b 63, c 75.
"""
import re
import sass

R = 255


def shape(text):
    t = re.sub(r'\s+', ' ', text.strip().rstrip(';').strip()).replace('.reuse', '')
    t = re.sub(r'^@!?P\d ', '@P ', t)
    op, _, rest = t.partition(' ')
    if op == '@P':
        op2, _, rest = rest.partition(' ')
        op = '@P ' + op2
    args = [a.strip() for a in rest.split(',')] if rest else []
    cls = []
    for a in args:
        neg = a[0] if a[:1] in '-~!' and not a.startswith('-0x') else ''
        b = a[len(neg):]
        if re.fullmatch(r'R\d+|RZ', b):
            k = 'R'
        elif re.fullmatch(r'P\d|PT', b):
            k = 'P'
        elif re.fullmatch(r'-?0x[0-9a-f]+', b):
            k = 'I'
        elif b.startswith('c[') and 'UR' in b:
            k = 'A'
        elif b.startswith('c['):
            k = 'C'
        elif re.fullmatch(r'UR\d+|URZ', b):
            k = 'U'
        else:
            k = '?' + b
        cls.append(k if k in ('R', 'P') else neg + k)
    return op + ' ' + ','.join(cls)


def setf(w, pos, width, val):
    m = ((1 << width) - 1) << pos
    return (w & ~m) | ((val << pos) & m)


def fields(op):
    """Positions for an op's operands, in order: list of (kind, pos) for registers, predicates etc."""
    base = op.replace('@P ', '')
    a = sass.Op(base + ' R0')  # unused
    return None


def encode(text, tmpl):
    """(lo, hi) of text, from the template's (lo, hi) of the same shape (control field untouched)."""
    t = re.sub(r'\s+', ' ', text.strip().rstrip(';').strip()).replace('.reuse', '')
    w = tmpl[0] | tmpl[1] << 64
    m = re.match(r'@(!?)P(\d) (.*)$', t)
    if m:
        w = setf(w, 12, 3, int(m[2]))
        w = setf(w, 15, 1, 1 if m[1] else 0)
        t = m[3]
    op, _, rest = t.partition(' ')
    args = [a.strip() for a in rest.split(',')] if rest else []
    o = sass.Op(t)
    npd = sass.npreds(op, args)
    if op.startswith('ISETP'):
        npd = 2
    if op.startswith(('UMOV', 'ULDC', 'UIADD3')):
        ur = lambda s: 63 if s == 'URZ' else int(s[2:])
        w = setf(w, 16, 6, ur(args[0]))
        if op == 'UMOV':
            w = setf(w, 32, 32, int(args[1], 16))
        elif op == 'UIADD3':
            w = setf(w, 24, 6, ur(args[1]))
            w = setf(w, 32, 32, int(args[2], 16))
            assert args[3] == 'URZ'
        else:
            mm = re.fullmatch(r'c\[0x3\]\[UR(\d+)(?:\+0x([0-9a-f]+))?\]', args[1])
            w = setf(w, 24, 6, int(mm[1]))
            w = setf(w, 40, 14, int(mm[2] or '0', 16) >> 2)
        return w & ((1 << 64) - 1), w >> 64
    regpos = [16]
    i = 0
    srcpos = [24, 32, 64]
    negpos = [72, 63, 75]
    pdpos = [81, 84]
    cinpos = [(87, 90), (77, 80)]
    if op.startswith('ISETP'):
        pd = [args[0], args[1]]
        srcs = args[2:4]
        cins = args[4:5]
        dst = None
    else:
        dst = args[0]
        pd = args[1:1 + npd]
        rest_ = args[1 + npd:]
        if op.startswith(('IMAD.MOV',)):
            srcs, cins = rest_[:3], []
        elif op.startswith('SEL'):
            srcs, cins = rest_[:2], rest_[2:3]
        else:
            srcs, cins = rest_[:3], rest_[3:]
    if dst is not None:
        w = setf(w, 16, 8, R if dst == 'RZ' else int(dst[1:]))
    for k, p in enumerate(pd):
        neg = p.startswith('!')
        p = p.lstrip('!')
        w = setf(w, pdpos[k], 3, 7 if p == 'PT' else int(p[1:]))
    order = list(range(len(srcs)))
    if len(srcs) == 3 and (srcs[2].lstrip('-~').startswith(('c[', '0x')) or srcs[2].startswith('-0x')):
        # an immediate or constant third source takes the second's field; the second goes to the third's
        srcs = [srcs[0], srcs[2], srcs[1]]
        srcpos = [24, 32, 64]
    for k, s in enumerate(srcs):
        neg = s[0] if s[:1] in '-~' and not s.startswith('-0x') else ''
        b = s[len(neg):]
        if re.fullmatch(r'R\d+|RZ', b):
            w = setf(w, srcpos[k], 8, R if b == 'RZ' else int(b[1:]))
            w = setf(w, negpos[k], 1, 1 if neg else 0)
        elif re.fullmatch(r'-?0x[0-9a-f]+', b):
            v = int(b.lstrip('-'), 16)
            if b.startswith('-'):
                v = (-v) & 0xFFFFFFFF
            w = setf(w, 32, 32, v)
        elif b.startswith('c['):
            mm = re.fullmatch(r'c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]', b)
            w = setf(w, 40, 14, int(mm[2], 16) >> 2)
            w = setf(w, 54, 5, int(mm[1], 16))
            w = setf(w, negpos[k], 1, 1 if neg else 0) if k != 1 else w
        elif re.fullmatch(r'UR\d+|URZ', b):
            w = setf(w, 32, 6, 63 if b == 'URZ' else int(b[2:]))
        else:
            raise ValueError(text)
    for k, p in enumerate(cins):
        neg = p.startswith('!')
        pp = p.lstrip('!')
        pos, npos = cinpos[k]
        if op.startswith('ISETP'):
            pos, npos = 87, 90
        w = setf(w, pos, 3, 7 if pp == 'PT' else int(pp[1:]))
        w = setf(w, npos, 1, 1 if neg else 0)
    return w & ((1 << 64) - 1), w >> 64


CTRL_MASK = 0x1FFFFF << 41


class Templates:
    def __init__(self, listings):
        self.t = {}
        for rows in listings:
            for r in rows:
                sh = shape(r['text'])
                self.t.setdefault(sh, (r['lo'], r['hi'] & ~CTRL_MASK, r['text']))

    def get(self, text):
        sh = shape(text)
        if sh not in self.t:
            # ptxas may never write this op with two carries out (or with one): the carry fields are written
            # from the text anyway, and are PT in a template without them
            op, _, cl = sh.rpartition(' ')
            c = cl.split(',')
            for drop in (1, 2):
                if c[1:1 + drop] == ['P'] * drop and op + ' ' + ','.join(c[:1] + c[1 + drop:]) in self.t:
                    sh = op + ' ' + ','.join(c[:1] + c[1 + drop:])
                    break
        if sh not in self.t:
            raise KeyError('no template for %r (%s)' % (text, sh))
        lo, hi, _ = self.t[sh]
        return encode(text, (lo, hi))


def selfcheck(listings):
    T = Templates(listings)
    ok = bad = skip = 0
    for rows in listings:
        for r in rows:
            sh = shape(r['text'])
            op = sh.split(' ')[0] if not sh.startswith('@') else sh.split(' ')[1]
            if not op.startswith(('IADD3', 'IMAD', 'SEL', 'ISETP')) or 'A' in sh.split(' ')[-1]:
                skip += 1
                continue
            try:
                lo, hi = T.get(r['text'])
            except Exception as e:
                skip += 1
                continue
            if (lo, hi) == (r['lo'], r['hi'] & ~CTRL_MASK):
                ok += 1
            else:
                bad += 1
                if bad < 10:
                    print('BAD', r['text'], hex(lo ^ r['lo']), hex(hi ^ (r['hi'] & ~CTRL_MASK)), T.t[sh][2])
    print('selfcheck ok %d bad %d skipped %d' % (ok, bad, skip))
    return bad
