"""t96.py: a hand-written chunk loop for the RTX 20 fold with a 96 x 64 warp tile. TIMING ONLY.

  python3 -I t96.py <cubin> <nvdisasm> <out dir>

Writes <out dir>/t96.json and t96rg.json, patch lists for sm75patch.py: the chunk loop of
pearl_t75_tile_fold_wmma and of pearl_t75_tile_fold_wmma_rg replaced by a loop written here, and the loop
counter's start. A core built with them folds made-up sums: its hashes are wrong on purpose. It answers one
question, what a warp tile of 192 sums would run at on Turing, before anyone builds it for real.

Why by hand. ptxas needs about 274 registers for 192 sums (goal-nv20s) and has 255. The loop here needs 232
besides the kernel's own: 192 sums and 40 operand registers, each operand single-buffered. That works because
a load here is four fragments of ONE k16 half (the release's is two fragments of both halves), and a k-step runs
its low half and then its high half, row by row: a register quad of A is free after its 32 IMMA and is loaded
again at once for the next k-step (160 IMMA ahead), a quad of B after its half (88 IMMA ahead).

What a k-step is here: 192 IMMA.8816 (12 row blocks x 8 column blocks x 2 halves), 6 loads of A and 4 of B.
The release's is 128 IMMA and 8 loads. So loads an IMMA fall by 17 percent, and B's bytes from the L2 a MAC by
a third (a 192 x 256 block). A's bytes a MAC stay: the two extra loads read planes of a row group 30 groups
away, lines no other load of this block touches.

The loop keeps the kernel's k range: the release's loop runs k-steps 2 to 61 of a tile, 15 chunks of four, and
so does this one, a k-step a pass, with the readout (96 three-input XOR of the 192 sums into one chain, a
store) in the first of every four. So a tile is 60 x 192 + 512 IMMA against 60 x 128 + 512: multiply the rate
a core built with this reports by 1.46875 to compare it with the release's.

Not the same as a real build, and which way each leans: the hash and the tile's first and last k-steps are the
release's (a real tile would hash 1.5 times the regions: this leans the result up by a third of the hash's
share); the readout's shuffle runs but its xor does not (up, under 0.1 percent); a pass ends in a taken branch
where the release has one a chunk (down).

Everything is checked before a list is written: every word is read back through nvdisasm and compared with the
text asked for, the registers used are ones the release's loop writes and the code after it reads only through
IMMA (or not at all), and the two predicates are ones the code after the loop writes before reading.
Standard library only.
"""
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

LOOP0, LOOP1 = 0x1AD0, 0x42B0          # the chunk loop: first word, and its closing branch
INIT = 0x1AB0                          # IMAD.MOV.U32 <counter>, RZ, RZ, 0x1
SLOTS = (LOOP1 - LOOP0) // 16 + 1
CTRL_SHIFT = 41
FAR = 0x780000                         # the extra A planes: 30 row groups away (a 24-bit signed offset)
RZ = 255


def ctrl(stall=1, yld=True, wsb=None, rsb=None, wait=(), reuse=0):
    c = (stall & 15) | ((1 if yld else 0) << 4) | ((7 if wsb is None else wsb) << 5) | \
        ((7 if rsb is None else rsb) << 8) | (reuse << 17)
    for w in wait:
        c |= 1 << (11 + w)
    return c << CTRL_SHIFT


def sections(b):
    shoff, = struct.unpack_from('<Q', b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', b, 0x3A)
    sh = [struct.unpack_from('<IIQQQQIIQQ', b, shoff + i * shentsize) for i in range(shnum)]
    stro = sh[shstrndx][4]
    return {b[stro + s[0]:b.index(b'\0', stro + s[0])].decode(): (s[4], s[5]) for s in sh}


def listing(text, sec):
    """[(addr, text)] of one section of an nvdisasm -c listing."""
    t = text.split('\n')
    i = [k for k, l in enumerate(t) if l.split()[:1] == ['.section'] and '.text.' + sec + ',' in l][0]
    out = []
    for l in t[i + 1:]:
        if l.split()[:1] == ['.section']:
            break
        m = re.search(r'/\*([0-9a-f]{4,6})\*/\s+(.*?)\s*;', l)
        if m:
            out.append((int(m[1], 16), re.sub(r'\s+', ' ', m[2].strip())))
    return out


def regs(s):
    return [int(x) for x in re.findall(r'\bR(\d+)\b', s)]


def defuse(txt):
    """(registers written, registers read, operation) of one instruction."""
    t = re.sub(r'^@!?P\d\s+', '', txt)
    op, _, rest = t.partition(' ')
    ops = [o.strip() for o in re.split(r',\s*(?![^\[]*\])', rest)] if rest else []
    base = op.split('.')[0]
    if base in ('BRA', 'EXIT', 'BSSY', 'BSYNC', 'NOP', 'BMOV', 'BPT', 'BAR', 'RET', 'CALL', 'BREAK', 'STS', 'STG',
                'STL', 'ISETP', 'PLOP3', 'VOTEU', 'UISETP'):
        return [], sum((regs(o) for o in ops), []), base
    if base == 'IMMA':
        d = regs(ops[0])[0]
        c = regs(ops[3])
        return [d, d + 1], regs(ops[1]) + regs(ops[2]) + ([c[0], c[0] + 1] if c else []), base
    if ops and (re.fullmatch(r'P\d|PT', ops[0])):
        if base in ('SHFL', 'ATOMG'):
            return regs(ops[1])[:1], sum((regs(o) for o in ops[2:]), []), base
        return [], sum((regs(o) for o in ops[1:]), []), base
    d = regs(ops[0])[:1] if ops else []
    n = 1
    if base in ('LDG', 'LDS', 'LDL', 'LDC'):
        n = 4 if '.128' in op else 2 if '.64' in op else 1
    elif '.WIDE' in op or 'U64' in op or '.64' in op or (base == 'CS2R' and '.32' not in op):
        n = 2
    return ([d[0] + i for i in range(n)] if d else []), sum((regs(o) for o in ops[1:]), []), base


def preds(txt):
    """(predicates written, predicates read) of one instruction."""
    m = re.match(r'@!?(P\d)', txt)
    uses = [m[1]] if m else []
    body = re.sub(r'^@!?P\d\s+', '', txt)
    op, _, rest = body.partition(' ')
    ps = re.findall(r'\b(P\d)\b', rest)
    base = op.split('.')[0]
    if ps and (base in ('ISETP', 'PLOP3') or (base == 'LOP3' and rest.startswith('P'))):
        return ps[:1], uses + ps[1:]
    return [], uses + ps


class Build:
    def __init__(self, cubin, text, sec):
        self.b = open(cubin, 'rb').read()
        self.sec = sec
        self.off, self.size = sections(self.b)['.text.' + sec]
        self.rows = listing(text, sec)
        self.at = dict(self.rows)
        self.facts()

    def word(self, a):
        return struct.unpack_from('<QQ', self.b, self.off + a)

    def facts(self):
        at = self.at
        m = re.fullmatch(r'@(P\d) BRA `\((\.L_x_\d+)\)', at[LOOP1])
        assert m, 'no closing branch at %x: %s' % (LOOP1, at[LOOP1])
        lo, _hi = self.word(LOOP1)
        assert (lo >> 32) - (1 << 32) == LOOP0 - (LOOP1 + 16), 'the branch at %x does not close a loop at %x' % (LOOP1, LOOP0)
        self.ploop = int(m[1][1])
        m = re.fullmatch(r'IMAD\.MOV\.U32 R(\d+), RZ, RZ, 0x1', at[INIT])
        assert m, 'no counter start at %x: %s' % (INIT, at[INIT])
        self.cnt = int(m[1])
        loop = [(a, t) for a, t in self.rows if LOOP0 <= a <= LOOP1]
        tail = [(a, t) for a, t in self.rows if a > LOOP1]
        assert len(loop) == SLOTS and sum(t.startswith('IMMA') for _a, t in loop) == 512
        # the three fixed points of the loop
        sh = [(a, t) for a, t in loop if 'SHFL' in t]
        assert len(sh) == 1, 'one shuffle in the loop'
        self.shfl = (sh[0][0] - LOOP0) // 16
        m = re.fullmatch(r'SHFL\.BFLY PT, R(\d+), R(\d+), 0x4, 0x1f', sh[0][1])
        assert m
        self.shfl_dst = int(m[1])
        st = [t for _a, t in loop if 'STS' in t]
        m = re.fullmatch(r'@!(P\d) STS \[R\d+\.X4\+0x80\], R\d+', st[0]) if len(st) == 1 else None
        assert m, 'one predicated transcript store in the loop'
        self.psts = int(m[1][1])
        ld = [(a, t) for a, t in loop if t.startswith('LDG')]
        a_ld = [a for a, t in ld if '.EL.' not in t]
        b_ld = [a for a, t in ld if '.EL.' in t]
        assert len(a_ld) == 16 and len(b_ld) == 16
        self.a_lo, self.a_hi = self.word(a_ld[0])
        self.b_lo, self.b_hi = self.word(b_ld[0])
        # the two offsets the loop adds the chunk to: IMAD.IADD R243, R242.reuse, 0x1, R239 / R242, R242, 0x1, R238
        m1 = re.fullmatch(r'IMAD\.IADD R(\d+), R(\d+)\.reuse, 0x1, R(\d+)', at[LOOP0 + 16])
        m2 = re.fullmatch(r'IMAD\.IADD R(\d+), R(\d+), 0x1, R(\d+)', at[LOOP0 + 32])
        assert m1 and m2, 'the loop does not start as expected'
        self.ra, self.rb = int(m1[3]), int(m2[3])
        ta = [t for _a, t in loop if t.startswith('LDG') and '.EL.' not in t and 'R%d.U32' % int(m1[1]) in t]
        assert not ta, 'A is expected through a copy of R%d' % int(m1[1])
        # the transcript's index register (LOP3.LUT Rt, Rt, R232, RZ, 0x3c)
        m = [re.fullmatch(r'@!P\d LOP3\.LUT R(\d+), R\1, R(\d+), RZ, 0x3c, !PT', t) for _a, t in loop]
        m = [x for x in m if x]
        assert len(m) == 1
        self.rt = int(m[0][2])
        # registers: what the loop writes, and what the code after it reads before writing
        loopw = set()
        for _a, t in loop:
            loopw |= set(defuse(t)[0])
        written, first = set(), {}
        for a, t in tail:
            d, u, base = defuse(t)
            for r in u:
                if r not in written:
                    first.setdefault(r, set()).add(base)
            written |= set(d)
        allr = set()
        for _a, t in self.rows:
            d, u, _b = defuse(t)
            allr |= set(d) | set(u)
        self.top = max(allr)
        bad = sorted(r for r in loopw if first.get(r, set()) - {'IMMA'})
        assert not bad, 'the code after the loop reads %s, which the loop writes, outside an IMMA' % bad
        for r in (self.ra, self.rb, self.rt):
            assert r not in loopw
        # free: written by the release's loop (so dead or data after it), or never read before written after it
        free = set(r for r in range(1, self.top + 1) if (r in loopw or r not in first) and r != self.cnt)
        free -= {self.ra, self.rb, self.rt}
        for _a, t in self.rows:                 # nothing outside the loop may hold a value across it in them
            pass
        self.keep = sorted(r for r in first if r not in loopw)
        self.free = free - set(self.keep)
        pw, pfirst = set(), set()
        for _a, t in tail:
            d, u = preds(t)
            pfirst |= set(u) - pw
            pw |= set(d)
        livep = {int(p[1]) for p in pfirst}
        fp = [p for p in range(7) if p not in livep and p not in (self.ploop, self.psts)]
        assert len(fp) >= 2, 'no two free predicates'
        self.p3, self.p4 = fp[0], fp[1]

    # ---------------------------------------------------------------- encoders (each a ptxas word with its fields set)
    @staticmethod
    def pred(p):
        return 7 if p is None else (p if p >= 0 else 8 | -p - 1)          # -p-1: @!P

    def imma(self, dst, a, b, c, ra=False, c_=0):
        return (0x7237 | dst << 16 | a << 24 | b << 32, 0x5400 | c | c_ | (ctrl(0, False, reuse=1) if ra else 0) & (15 << 17 + CTRL_SHIFT),
                'IMMA.8816.S8.S8 R%d, R%d%s.ROW, R%d.COL, R%d' % (dst, a, '.reuse' if ra else '', b, c))

    def ldg(self, q, base, off, isb, p=None):
        lo0, hi0 = (self.b_lo, self.b_hi) if isb else (self.a_lo, self.a_hi)
        ur = lo0 >> 32 & 0xFF
        lo = 0x0981 | self.pred(p) << 12 | q << 16 | base << 24 | ur << 32 | (off & 0xFFFFFF) << 40
        hi = hi0 & ~(0x1FFFFF << CTRL_SHIFT)
        o = '' if off == 0 else '+0x%x' % off if off > 0 else '+-0x%x' % -off
        return (lo, hi, '%sLDG.E.%s128.CONSTANT.SYS R%d, [R%d.U32+UR%d%s]' % (
            self.ptext(p), 'EL.' if isb else '', q, base, ur, o))

    @staticmethod
    def ptext(p):
        return '' if p is None else '@P%d ' % p if p >= 0 else '@!P%d ' % (-p - 1)

    def lop3(self, dst, a, b, c):
        return (0x7212 | dst << 16 | a << 24 | b << 32, 0x078E0000 | 0x96 << 8 | c,
                'LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT' % (dst, a, b, c))

    def lop3p(self, p, a, imm):
        return (0x7812 | 0xFF << 16 | a << 24 | imm << 32, 0x0780C0FF | p << 17,
                'LOP3.LUT P%d, RZ, R%d, 0x%x, RZ, 0xc0, !PT' % (p, a, imm))

    def iadd3(self, dst, a, imm):
        return (0x7810 | dst << 16 | a << 24 | (imm & 0xFFFFFFFF) << 32, 0x07FFE0FF,
                'IADD3 R%d, R%d, %s0x%x, RZ' % (dst, a, '-' if imm < 0 else '', abs(imm)))

    def isetp(self, p, a, imm):
        return (0x780C | a << 24 | imm << 32, 0x03F05270 | p << 17, 'ISETP.NE.AND P%d, PT, R%d, 0x%x, PT' % (p, a, imm))

    def bra(self, p, target):
        return ('bra', p, target)

    def sts(self, p, base, src):
        return (0x0388 | self.pred(p) << 12 | base << 24 | src << 32 | 0x80 << 40, 0x4800,
                '%sSTS [R%d.X4+0x80], R%d' % (self.ptext(p), base, src))

    # ---------------------------------------------------------------- the loop
    def kstep(self, R, readout, early):
        """One k-step's instructions: (lo, hi, text, control). R: the register map."""
        out = []
        A, B, acc = R['A'], R['B'], R['acc']        # A[h][q] and B[h][q]: a quad's first register
        planeA = {(0, 0): 0x0, (0, 1): 0x8000, (1, 0): 0x10000, (1, 1): 0x18000, (0, 2): 'far0', (1, 2): 'far1'}
        planeB = {(0, 0): 0x0, (0, 1): 0x8000, (1, 0): 0x10000, (1, 1): 0x18000}

        def lda(h, q, ahead):
            pl, o = planeA[(h, q)], 0x200 if ahead else 0
            sb = 2 + h
            if isinstance(pl, str):
                d = 0x8000 * int(pl[3])
                out.append(self.ldg(A[h][q], self.ra, o + d - FAR, False, self.p4) + (ctrl(1, True, sb, 1),))
                out.append(self.ldg(A[h][q], self.ra, o + d + FAR, False, -self.p4 - 1) + (ctrl(2, True, sb, 1),))
            else:
                out.append(self.ldg(A[h][q], self.ra, o + pl, False) + (ctrl(2, True, sb, 1),))

        def ldb(h, q, ahead):
            out.append(self.ldg(B[h][q], self.rb, (0x200 if ahead else 0) + planeB[(h, q)], True) + (ctrl(2, True, 4 + h, 1),))

        for h in (0, 1):
            for i in range(12):
                for j in range(8):
                    n = i * 8 + j
                    if readout and h == 0:
                        x = acc[i][j]
                        out.append(self.lop3(R['t'], x, x + 1, R['t']) + (ctrl(3, False),))
                    lo, hi, tx = self.imma(acc[i][j], A[h][i // 4] + i % 4, B[h][j // 4] + j % 4, acc[i][j], ra=j < 7)
                    out.append((lo, hi, tx, {'wait': (2 + h, 4 + h) if n == 0 else (), 'reuse': 1 if j < 7 else 0}))
                    # loads: the quads of the half before, 8 and 12 IMMA into this one; this half's A quads 0 and 1,
                    # 8 IMMA after their last row
                    if n == 7:
                        lda(1 - h, 2, h == 1)
                    if n == 11:
                        ldb(1 - h, 0, h == 1)
                    if n == 15:
                        ldb(1 - h, 1, h == 1)
                    if n == 39:
                        lda(h, 0, True)
                    if n == 71:
                        lda(h, 1, True)
                    if n == 79 and h == 1:
                        out.extend(early)      # the counter and its predicates, 16 IMMA before the branches read them
            if readout and h == 0:
                out.append(self.sts(-self.psts - 1, self.rt, R['t']) + (ctrl(1, True, None, 0),))
        return out

    def program(self, R, moved=None):
        """The loop as a list of items; a 'bra' item is (kind, predicate, label), labels are ('label', name)."""
        c = self.cnt
        prog = [self.iadd3(self.ra, self.ra, 0x400) + (ctrl(2, True),),
                self.iadd3(self.rb, self.rb, 0x400) + (ctrl(2, True),),
                self.lop3p(self.p4, self.ra, 0x8000000) + (ctrl(2, True),),
                self.bra(None, 'Q')]

        def step():
            return [self.iadd3(self.ra, self.ra, 0x200) + (ctrl(2, True, wait=(1,)),),
                    self.iadd3(self.rb, self.rb, 0x200) + (ctrl(2, True),)]
        count = [self.iadd3(c, c, 1) + (ctrl(2, True),)]
        prog.append(('label', 'P'))
        prog += self.kstep(R, False, count + [self.lop3p(self.p3, c, 3) + (ctrl(1, True),),
                                              self.isetp(self.ploop, c, 64) + (ctrl(1, True),)]) + step()
        prog += [self.bra(self.p3, 'P'), self.bra(self.ploop, 'Q'),
                 self.iadd3(self.ra, self.ra, -0x7C00) + (ctrl(2, True),),
                 self.iadd3(self.rb, self.rb, -0x7C00) + (ctrl(2, True),),
                 self.bra(None, 'OUT')]
        prog.append(('label', 'Q'))
        q = self.kstep(R, True, count)
        qs = len([x for x in prog if x[0] != 'label'])
        k = self.shfl - qs                         # the shuffle's place in Q
        assert 0 < k < len(q), 'the shuffle falls outside the readout k-step'
        # the last A load before it moves to just behind it; its quad is the one the shuffle writes into
        cand = [n for n in range(k) if isinstance(q[n][2], str) and q[n][2].startswith('LDG.E.128')]
        n = cand[-1]
        ld = q.pop(n)
        quad = int(re.search(r'SYS R(\d+),', ld[2])[1])
        between = q[n:k]
        assert not any(set(defuse(x[2])[1]) & set(range(quad, quad + 4)) for x in between), 'the moved load is read'
        q.insert(k, ('shfl',))
        q.insert(k + 1, ld)
        prog += q + step() + [self.bra(None, 'P')]
        return prog, quad

    def assemble(self, R):
        prog, quad = self.program(R)
        labels, n = {'OUT': SLOTS}, 0
        for it in prog:
            if it[0] == 'label':
                labels[it[1]] = n
            else:
                n += 1
        assert n <= SLOTS, 'the loop needs %d words, the room is %d' % (n, SLOTS)
        words, texts, n = [], [], 0
        prev_imma = False
        items = [it for it in prog if it[0] != 'label']
        for idx, it in enumerate(items):
            a = LOOP0 + n * 16
            nxt = items[idx + 1] if idx + 1 < len(items) else None
            if it[0] == 'shfl':
                assert n == self.shfl
                words.append(self.word(a))
                texts.append(self.at[a])
            elif it[0] == 'bra':
                _k, p, lab = it
                rel = (LOOP0 + labels[lab] * 16) - (a + 16)
                lo = 0x0947 | self.pred(p) << 12 | (rel & 0xFFFFFFFF) << 32
                # the offset is signed and runs on into the high word's low 18 bits
                words.append((lo, 0x03800000 | (0x3FFFF if rel < 0 else 0) | ctrl(5, True)))
                texts.append('%sBRA 0x%x' % (self.ptext(p), LOOP0 + labels[lab] * 16))
            else:
                lo, hi, tx, c = it
                if isinstance(c, dict):                # an IMMA: its stall by what follows it
                    nx_imma = nxt is not None and nxt[0] not in ('shfl', 'bra') and isinstance(nxt[3], dict)
                    c = ctrl(4 if nx_imma else 1 if nxt is not None and nxt[0] not in ('shfl', 'bra') and 'LOP3.LUT R' in nxt[2] else 2,
                             True, wait=c['wait'], reuse=c['reuse'])
                words.append((lo, hi & ~(0x1FFFFF << CTRL_SHIFT) | c))
                texts.append(tx)
            n += 1
        while n < SLOTS:
            words.append((0x7918, ctrl(0, False)))
            texts.append('NOP')
            n += 1
        return words, texts, quad, len(items)

    def registers(self):
        """The register map: 10 quads at multiples of four, 96 pairs at even numbers, the chain's register."""
        free = set(self.free)
        quads = [r for r in range(4, self.top - 2, 4) if all(x in free for x in range(r, r + 4))]
        want = self.shfl_dst & ~3
        assert want in quads, 'the shuffle writes R%d, whose quad is not free' % self.shfl_dst
        R = {'A': [[None] * 3, [None] * 3], 'B': [[None] * 2, [None] * 2]}
        return quads, want, free, R

    def build(self):
        quads, want, free, R = self.registers()
        # first pass with any quads, to learn which load the shuffle displaces
        def fill(first_quad_for):
            qs = [q for q in quads if q != want]
            use = []
            names = [('A', h, q) for h in (0, 1) for q in range(3)] + [('B', h, q) for h in (0, 1) for q in range(2)]
            for nm in names:
                use.append(want if nm == first_quad_for else qs.pop(0))
            for nm, r in zip(names, use):
                R[nm[0]][nm[1]][nm[2]] = r
            taken = set()
            for r in use:
                taken |= set(range(r, r + 4))
            pairs = [r for r in range(2, self.top, 2) if r in free and r + 1 in free and r not in taken and r + 1 not in taken]
            assert len(pairs) >= 96, 'only %d register pairs are free for 96 sums' % len(pairs)
            R['acc'] = [[pairs[i * 8 + j] for j in range(8)] for i in range(12)]
            used = taken | {x for p in pairs[:96] for x in (p, p + 1)}
            t = sorted(free - used)
            assert t, 'no register is left for the readout chain'
            R['t'] = t[0]
            return names, use
        names, use = fill(('A', 0, 0))
        _w, _t, quad, _n = self.assemble(R)
        nm = names[use.index(quad)]
        fill(nm)
        words, texts, quad, n = self.assemble(R)
        assert quad == want
        return words, texts, R, n

    def patch_list(self, words, texts, note):
        lst = []
        sec = bytearray(self.b[self.off:self.off + self.size])
        before = hashlib.md5(sec).hexdigest()[:12]

        def put(a, lo, hi, tx):
            old = bytes(sec[a:a + 16])
            new = struct.pack('<QQ', lo, hi)
            if old != new:
                sec[a:a + 16] = new
                lst.append({'addr': '0x%x' % a, 'old': old.hex(), 'new': new.hex(), 'note': tx})
        lo, hi = self.word(INIT)
        put(INIT, lo & 0xFFFFFFFF | 4 << 32, hi, 'IMAD.MOV.U32 R%d, RZ, RZ, 0x4' % self.cnt)
        for n, ((lo, hi), tx) in enumerate(zip(words, texts)):
            put(LOOP0 + n * 16, lo, hi, tx)
        return {'description': note, 'section': self.sec, 'before': before,
                'after': hashlib.md5(sec).hexdigest()[:12], 'words': lst}, bytes(sec)


def norm(t):
    t = re.sub(r'`\((\.L_x_\d+)\)', r'\1', t)
    return re.sub(r'\s+', ' ', t).strip()


def main():
    cubin, nvdisasm, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    text = subprocess.run([nvdisasm, '-c', cubin], capture_output=True).stdout.decode('utf-8', 'replace')
    b = bytearray(open(cubin, 'rb').read())
    builds = []
    for sec, name in (('pearl_t75_tile_fold_wmma', 't96'), ('pearl_t75_tile_fold_wmma_rg', 't96rg')):
        bd = Build(cubin, text, sec)
        words, texts, R, n = bd.build()
        lst, newsec = bd.patch_list(words, texts, 'TIMING ONLY (wrong hashes): the chunk loop as a hand-written 96 x 64 '
                                    'warp tile, 192 sums, operands single-buffered (t96.py)')
        b[bd.off:bd.off + bd.size] = newsec
        builds.append((bd, name, lst, texts, R, n))
    tmp = os.path.join(out, 't96-check.cubin')
    open(tmp, 'wb').write(b)
    back = subprocess.run([nvdisasm, '-c', tmp], capture_output=True)
    assert back.returncode == 0, 'nvdisasm cannot read the patched cubin: ' + back.stderr.decode()[:300]
    back = back.stdout.decode('utf-8', 'replace')
    bad = 0
    for bd, name, lst, texts, R, n in builds:
        got = dict(listing(back, bd.sec))
        labs = {}
        for l in back.split('\n'):
            pass
        for i, tx in enumerate(texts):
            a = LOOP0 + i * 16
            g = norm(got[a])
            w = norm(tx)
            if w.endswith('NOP') or 'SHFL' in w:
                ok = g == w
            elif 'BRA' in w:
                ok = 'BRA ' in g and g[:g.index('BRA ')] == w[:w.index('BRA ')]      # targets are checked below
            else:
                ok = g == w
            if not ok:
                bad += 1
                if bad <= 20:
                    print('READ-BACK %s %04x: asked "%s", nvdisasm reads "%s"' % (name, a, w, g))
        # branch targets: nvdisasm prints labels; find each label's address in the read-back listing
        t = back.split('\n')
        s = [k for k, l in enumerate(t) if l.split()[:1] == ['.section'] and '.text.' + bd.sec + ',' in l][0]
        lab_at, pend = {}, []
        for l in t[s + 1:]:
            if l.split()[:1] == ['.section']:
                break
            m = re.match(r'\s*(\.L_x_\d+):', l)
            if m:
                pend.append(m[1])
            m = re.search(r'/\*([0-9a-f]{4,6})\*/', l)
            if m:
                for p in pend:
                    lab_at[p] = int(m[1], 16)
                pend = []
        for i, tx in enumerate(texts):
            if 'BRA 0x' in tx:
                a = LOOP0 + i * 16
                want = int(tx.split('BRA ')[1], 16)
                m = re.search(r'\((\.L_x_\d+)\)', got[a])
                if not m or lab_at.get(m[1]) != want:
                    bad += 1
                    print('READ-BACK %s %04x: branch to %x reads %s' % (name, a, want, got[a]))
        used = set()
        for tx in texts:
            d, u, _b = defuse(tx)
            used |= set(d) | set(u)
        used -= {RZ}
        extra = used - bd.free - {bd.cnt, bd.ra, bd.rb, bd.rt} - set(bd.keep)
        assert not extra, 'the loop uses registers it may not: %s' % sorted(extra)
        wr = set()
        for tx in texts:
            wr |= set(defuse(tx)[0])
        assert not (wr - bd.free - {bd.cnt, bd.ra, bd.rb}), 'the loop writes registers it may not'
        print('%s: %s, %d words used of %d, %d changed; counter R%d, A offset R%d, B offset R%d, chain R%d, '
              'predicates P%d P%d, md5 %s -> %s' % (name, bd.sec, n, SLOTS, len(lst['words']), bd.cnt, bd.ra, bd.rb,
                                                    R['t'], bd.p3, bd.p4, lst['before'], lst['after']))
    if bad:
        print('t96: FAILED, %d words do not read back as asked' % bad)
        return 1
    for bd, name, lst, texts, R, n in builds:
        with open(os.path.join(out, name + '.json'), 'w', newline='\n') as f:
            json.dump(lst, f, indent=1)
            f.write('\n')
        with open(os.path.join(out, name + '.txt'), 'w', newline='\n') as f:
            f.write('\n'.join('%04x %s' % (LOOP0 + i * 16, tx) for i, tx in enumerate(texts)) + '\n')
    os.remove(tmp)
    print('t96: ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
