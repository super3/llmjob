"""full120.py: the RTX 50 (sm_120) kernel's body from the first round's layer to the last S-boxes, written by hand.

  python full120.py <in.cubin built with -DQC_COMPACT=2 -DQC_HAND75=2> <out.cubin> [key=value ...]

It is sm75/full75.py's body (the same blocks, arithmetic, scheduler, allocator and simulation) with what sm_120
needs done differently:

  * No uniform registers and no IMAD.X or IMAD.MOV: ptxas writes none of them for sm_120's adds. The partial
    rounds' constant comes through an LDC.64 by a pointer register, the last round's constants are immediates,
    and copies are MOVs.
  * The interface: ptxas loads the launch's zero from constant memory inside the function and makes small
    constants with HFMA2, so the entry and exit are read by pattern here (find_iface).
  * Branches are encoded from one of the cubin's own (sm120/fuse.py's bra_words).
  * .nv.capmerc.text is rebuilt for the new code (sm120/capmerc.py): which instructions have records, and the
    records of the constant loads.

On an RTX 50 a nonce's time is its instruction count, an IMAD.WIDE counting about 3.5 and anything else 1
(quantus-ledger.md, 28 Nov), so the options that move work to the multiplier pipe (imx, anch, sq2) stay off.
Latencies: ptxas's own shortest distances in sm_120 code are 4 (add to add), 3 (WIDE to add), 5 (add to WIDE),
4 (WIDE to WIDE), 6 after a three-input add; lat=6 keeps every one of them, and with six warps a scheduler a
longer wait costs nothing.
"""
import os
import random
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("sm89", "sm75", "sm86"):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), d))
sys.path.insert(0, HERE)
import enc  # noqa: E402
import gen  # noqa: E402
import hand  # noqa: E402
import iface as IF  # noqa: E402
import patch as P89  # noqa: E402
import ref  # noqa: E402
import patch75  # noqa: E402
import full75 as F  # noqa: E402
import capmerc  # noqa: E402
import fuse  # noqa: E402
try:
    import templates120 as TB  # noqa: E402
except ImportError:             # before the first dump (QC_DUMP_TEMPLATES)
    TB = None

hx = gen.hx
OPT = F.OPT
# ldcs=4: a load's scoreboard has to be up before anything that waits on it is issued. With the pointer's step one
# cycle behind the first round's load, some warps of a block still stepped it too early (RTX 5070, 28 Nov).
DEFAULTS = dict(lat=6, plat=6, imx=0, anch=0, sq2=0, dmx=0, sqmul=0, ldcs=4)
P = ref.P


class WriteAfterWrite(dict):
    """hand.LATW for sm_120. full75.timing lets a second write to a register or predicate follow the first by
    LATW[first's unit] - LATW.get(second's unit) + 1 cycles, with Ada's figures (a WIDE lands two cycles after an
    add). On sm_120 ptxas reads a WIDE's result after 3 cycles and an add's after 4, and never puts two writes to
    one register closer than 4 or 5 whatever their units, so Ada's rule let an older write land last: the first
    hand kernels gave wrong hits at every latency (28 Nov). Here every second write waits waw cycles."""
    def __getitem__(self, k):
        return OPT.get("waw", 6)

    def get(self, k, d=None):
        return 1


def clean(text):
    t = re.sub(r"\s*[&?]\S+", "", text).replace(".reuse", "")
    return re.sub(r"\s+", " ", t).strip().rstrip(";").strip()


def canon(t):
    """An IADD3 or IADD3.X as nvdisasm prints it for sm_120: always with both carry-out fields."""
    m = re.match(r"((?:@!?P\d )?)(IADD3(?:\.X)?) (.*)$", t)
    if not m:
        return t
    a = [x.strip() for x in m[3].split(",")]
    npo = len(a) - 1 - (5 if m[2].endswith(".X") else 3)
    a = a[:1 + npo] + ["PT"] * (2 - npo) + a[1 + npo:]
    if m[2] == "IADD3" and OPT.get("x3", 1):
        # x3 (default): no plain IADD3. On sm_120 only IADD3.X has been seen to give two carry outs right
        # (sm120/fuse.py's fused adds); a plain one is written as IADD3.X with no carry in.
        # (-R is ~R + 1: the 1 comes in as a carry)
        neg = [i for i in range(3, 6) if a[i].startswith("-R")]
        assert len(neg) <= 1, t
        for i in neg:
            a[i] = "~" + a[i][1:]
        return m[1] + "IADD3.X " + ", ".join(a + ["PT" if neg else "!PT", "!PT"])
    return m[1] + m[2] + " " + ", ".join(a)


def dest_regs(t):
    """Registers an instruction writes, for the ones find_iface has to follow."""
    t = re.sub(r"^@!?P\d ", "", t)
    op, _, rest = t.partition(" ")
    a = [x.strip() for x in rest.split(",")] if rest else []
    if not a or not re.fullmatch(r"R\d+", a[0]) or op.startswith(("ST", "BRA", "RET", "CALL", "ISETP", "ATOM")):
        return []
    d = int(a[0][1:])
    return [d, d + 1] if ("WIDE" in op or ".64" in op) else [d]


def find_iface(cubin):
    body, labels, entry, ret, retreg = IF.function(cubin)
    texts = [clean(r["text"]) for r in body]
    mark = re.compile(r"IMAD R(\d+), R(\d+), 0x([0-9a-f]+), R(\d+)$")
    inp = [[None, None] for _ in range(12)]
    origin = {}
    lab_addrs = set(labels.values())
    for r, t in zip(body, texts):
        if (r["addr"] in lab_addrs and r["addr"] != entry) or "BRA" in t:
            break
        m = mark.match(t)
        if m and 0x1001 <= int(m[3], 16) < 0x1200:
            k = int(m[3], 16) - 0x1001
            inp[(k % 0x100) // 2][k // 0x100] = origin.get(int(m[4]), int(m[4]))
            origin[int(m[1])] = int(m[1])
            continue
        m2 = re.match(r"MOV R(\d+), R(\d+)$", t)
        if m2:
            origin[int(m2[1])] = origin.get(int(m2[2]), int(m2[2]))
            continue
        for d in dest_regs(t):
            origin[d] = d
    assert all(x is not None for p in inp for x in p), inp
    first = next(i for i, t in enumerate(texts) if mark.match(t) and 0x2001 <= int(mark.match(t)[3], 16) < 0x2200)
    # the exit begins where the straight-line run that holds the output markers begins
    start = first
    while start > 0 and body[start]["addr"] not in lab_addrs and "BRA" not in texts[start - 1]:
        start -= 1
    tok = {}
    for t in texts[start:]:
        if t.startswith("RET"):
            break
        assert not t.startswith("@") and "BRA" not in t, "the exit is not straight-line: " + t
        m = mark.match(t)
        m2 = re.match(r"MOV R(\d+), R(\d+)$", t)
        if m and 0x2001 <= int(m[3], 16) < 0x2200:
            k = int(m[3], 16) - 0x2001
            tok[int(m[1])] = ((k % 0x100) // 2, k // 0x100)
        elif m2:
            if int(m2[2]) in tok:
                tok[int(m2[1])] = tok[int(m2[2])]
            else:
                tok.pop(int(m2[1]), None)
        else:
            for d in dest_regs(t):
                tok.pop(d, None)
    out = [[[], []] for _ in range(12)]
    for r, (i, h) in tok.items():
        out[i][h].append(r)
    assert all(o[0] and o[1] for o in out), out
    # the return address: the RET's pair as the compiled exit leaves it (an immediate and a zero)
    retfix = []
    for rr in (retreg, retreg + 1):
        defs = [t for t in texts if rr in dest_regs(t)]
        assert defs, "R%d, which the return needs, is not set in the function" % rr
        t = defs[-1]
        m = re.match(r"MOV R%d, (0x[0-9a-f]+)$" % rr, t)
        if m:
            retfix.append("MOV R%d, %s" % (rr, m[1]))
            ret_mov = [r["addr"] for r, x in zip(body, texts) if x == t][-1]
        elif re.match(r"HFMA2 R%d, -RZ, RZ, 0, 0$" % rr, t):
            retfix.append("MOV R%d, 0x0" % rr)
        else:
            raise AssertionError("the return register R%d is set by %r" % (rr, t))
    regs_w, preds_w = set(), set()
    for t in texts:
        regs_w.update(dest_regs(t))
        tt = re.sub(r"^@!?P\d ", "", t)
        op, _, rest = tt.partition(" ")
        a = [x.strip() for x in rest.split(",")]
        if op.startswith(("IADD3", "IMAD.WIDE")):
            preds_w.update(int(x[1]) for x in a[1:3] if re.fullmatch(r"P\d", x))
        if op.startswith("ISETP") and re.fullmatch(r"P\d", a[0]):
            preds_w.add(int(a[0][1]))
    return dict(inp=inp, out=out, entry=entry, ret=ret, retreg=retreg, retfix=retfix, keep=[],
                clob=sorted(regs_w), cpreds=sorted(preds_w), n=len(body), exit_at=body[start]["addr"],
                ret_mov=ret_mov)


GLOBAL1 = ["cnt", "rcp", "k", "pp", "zr"]
BASE = gen.Prog      # sm89/gen.py's own class, before run() points gen.Prog at this file's


class Prog120(F.ProgAll):
    """full75's program with switches that put one kind of instruction back to a form ptxas itself writes for
    sm_120, to find on a card which hand form the card computes differently (each costs instructions):
      arith=0   products and squares as sm89/gen.py has them (five WIDE, single carries) in place of gen75's folds
      e2=1      the external layer's sums two values at a time with single carries
      s2=1      the partial rounds' sums one term at a time with single carries"""

    def fold(self, L0, L1, M0, u, T, pa, pb, out=None, plus=None):
        # zt=1: where a fold has no T it takes ~%zr (a register that holds zero) in place of the immediate -1, so
        # its first add is the all-register form the S-boxes use
        # plus: sm75/full75.py's dl=1 puts an operand in that slot instead (its dmul passes plus=None without dl)
        if T is None and plus is None and OPT.get("zt", 0):
            T = "%zr"
        return super().fold(L0, L1, M0, u, T, pa, pb, out=out, plus=plus)

    def _out(self, v, out):
        if out is None:
            return v
        self.emit("MOV %s, %s" % (out[0], v[0]))
        self.emit("MOV %s, %s" % (out[1], v[1]))
        return out

    def mul(self, a, b, out=None):
        if OPT.get("arith", 1):
            return super().mul(a, b, out=out)
        return self._out(BASE.mul(self, a, b), out)

    def sqr(self, x):
        if OPT.get("arith", 1):
            return super().sqr(x)
        return BASE.sqr(self, x)

    def dmul(self, a, i, L, H, out=None):
        if OPT.get("arith", 1):
            return super().dmul(a, i, L, H, out=out)
        return BASE.dmul(self, a, i, L, H, out=out)

    def step(self, *ins, out3=None):
        if not OPT.get("e2", 0):
            return super().step(*ins, out3=out3)
        if len(ins) == 3:
            return self.step(self.step(ins[0], ins[1]), ins[2], out3=out3)
        w = out3 or tuple("%" + self.tmp("q") for _ in range(3))
        a, b = ins
        p1, q1 = self.pred(), self.pred()
        top = [v[3] for v in ins if len(v) > 3]
        self.emit("IADD3 %s, %s, %s, %s, RZ" % (w[0], p1, a[1], b[1]))
        self.emit("IADD3.X %s, %s, %s, %s, RZ, %s, !PT" % (w[1], q1, a[2], b[2], p1))
        self.emit("IADD3.X %s, %s, %s, !PT" % (w[2], ", ".join((top + ["RZ", "RZ", "RZ"])[:3]), q1))
        return ("q",) + tuple(w)

    def sum12(self, terms, S=None):
        if not OPT.get("s2", 0):
            return super().sum12(terms, S)
        S = S or self.tmp("S")
        self.pairs.add(S)
        p = self.pred()
        self.emit("IADD3 %%%s.0, %s, %s, %s, RZ" % (S, p, terms[0], terms[1]))
        self.emit("IADD3.X %%%s.1, RZ, RZ, RZ, %s, !PT" % (S, p))
        for t in terms[2:]:
            p = self.pred()
            self.emit("IADD3 %%%s.0, %s, %%%s.0, %s, RZ" % (S, p, S, t))
            self.emit("IADD3.X %%%s.1, RZ, RZ, %%%s.1, %s, !PT" % (S, S, p))
        return S


def build(c):
    """full75.build with the partial rounds' constants through LDC.64 [%pp] and the last round's as immediates."""
    g = Prog120()
    S = [("%%s%dl" % i, "%%s%dh" % i) for i in range(12)]
    g.pairs.update(F.GLOBAL2)

    def sums(y):
        if OPT["sum2"]:
            g.sums2([y[0]] + [S[i][0] for i in range(1, 12)], [y[1]] + [S[i][1] for i in range(1, 12)])
        else:
            g.sum12([y[0]] + [S[i][0] for i in range(1, 12)], "L")
            g.sum12([y[1]] + [S[i][1] for i in range(1, 12)], "H")
        rc = g.tmp("rc")
        g.pairs.add(rc)
        g.emit("LDC.64 %%%s, c[0x3][%%pp]" % rc)
        return g.add64("L", "%%%s.0" % rc), g.add64("H", "%%%s.1" % rc)

    g.block = "ent"
    # probe=N (tools/qcdrv.cu probe, sm120/probe_ref.py): a short path through the same code, to find on a card
    # which block computes something else than the simulator: 1 nothing (in and out), 2 the S-boxes once,
    # 3 one external layer (row 2's constants) then the S-boxes, 4 the 22 partial rounds then the S-boxes.
    probe = OPT.get("probe", 0)
    g.emit("IADD3 %%cnt, RZ, 0x%x, RZ" % ({2: 0xf, 3: 0xe, 4: 0xf}.get(probe, 0)))
    g.emit("IADD3 %%rcp, RZ, 0x%x, RZ" % (c["seq"] + 2 * 0x60))
    g.emit("IADD3 %zr, RZ, 0x0, RZ")
    if OPT["sum2"]:
        g.emit("IADD3 %L.1, RZ, RZ, RZ")      # sums2 never writes the low sum's high word
    g.block = "Ppro"
    g.emit("IADD3 %%pp, RZ, 0x%x, RZ" % c["p22"])
    y = g.sbox(S[0])
    L0, H0 = sums(y)
    g.dmul(y, 0, L0, H0, out=S[0])
    g.emit("IADD3 %k, RZ, 0x1, RZ")
    g.emit("IADD3 %pp, %pp, 0x8, RZ")
    g.block = "Ploop"
    for i in range(1, 12):
        g.dmul(S[i], i, "L", "H", out=S[i])
    y = g.sbox(S[0])
    L0, H0 = sums(y)
    g.dmul(y, 0, L0, H0, out=S[0])
    g.emit("IADD3 %pp, %pp, 0x8, RZ")
    g.emit("IADD3 %k, %k, 0x1, RZ")
    g.emit("ISETP.NE.AND $q, PT, %k, 0x16, PT")
    g.emit("@$q BRA .LOOP")
    g.block = "Pepi"
    for i in range(1, 12):
        g.sub = (i - 1) // 4
        Li = g.add64("L", hx(ref.RC_T0[i] & 0xFFFFFFFF))
        Hi = g.add64("H", hx(ref.RC_T0[i] >> 32))
        g.dmul(S[i], i, Li, Hi, out=S[i])
    g.sub = 0
    g.block = "S"
    for i in range(12):
        g.sub = i // OPT["sbatch"]
        g.sbox(S[i], out=S[i])
    g.sub = 0
    g.block = "Sx"
    g.emit("ISETP.NE.AND $pe, PT, %cnt, 0xf, PT")
    g.emit("@!$pe BRA .EXIT")
    g.block = "E"
    g.ext(S)
    g.sub = 0
    g.block = "Ex"
    g.emit("IADD3 %cnt, %cnt, 0x1, RZ")
    g.emit("IADD3 %rcp, %rcp, 0x60, RZ")
    g.emit("ISETP.NE.AND $xa, PT, %cnt, 0x3, PT")
    g.emit("ISETP.NE.AND $xb, PT, %cnt, 0xc, PT")
    g.emit("ISETP.NE.AND $xc, PT, %cnt, 0x7, PT")
    g.emit("@!$xa BRA .P")
    g.emit("@!$xb BRA .P")
    g.emit("@$xc BRA .S")
    g.emit("IADD3 %%rcp, RZ, 0x%x, RZ" % c["seq"])
    g.emit("BRA .E")
    return g, S


def probe_ref(probe, s):
    """What the function returns for the state s: the whole body, or a probe's short path."""
    s = [v % P for v in s]
    if not probe:
        return F.ref_all(s)
    if probe == 3:
        s = F.ext(s, F.RC_SEQ[2])
    if probe == 4:
        x = s[0]
        for r in range(22):
            x = ref.partial(s, ref.sbox(x), F.RC_P22[r])
        s[0] = x
        for i in range(1, 12):
            s[i] = (s[i] + ref.RC_T0[i]) % P
    if probe >= 2:
        s = [ref.sbox(v) for v in s]
    return s


def bra_words(text, here, labs, tmpl):
    """[@[!]Pn] BRA <label> from one of the cubin's own branches: the guard at bit 12 (15 = not), the distance
    as fuse.bra_words writes it."""
    m = re.match(r"(?:@(!?)P(\d) )?BRA (\S+)$", text)
    lo, hi = fuse.bra_words(tmpl, labs[m[3]] - (here + 16))
    lo = enc.setf(lo, 12, 4, 7 if m[2] is None else int(m[2]) | (8 if m[1] else 0))
    return lo, hi


class Templates(patch75.Templates):
    """Encodings by shape. templates120.py holds one instruction of every shape this file writes, taken from a
    CUDA 12.8 cubin, so a kernel compiled by another toolkit (CUDA 13 writes its adds as other instructions) gets
    the same body. A shape not listed there is looked for in the cubin, as sm75/patch75.py does; QC_DUMP_TEMPLATES
    =<file> writes the table again with whatever this run added."""
    used = {}

    def get(self, text):
        sh = enc.shape(text)
        if TB is not None and sh in TB.TEMPLATES:
            Templates.used[sh] = TB.TEMPLATES[sh]
            return enc.encode(text, TB.TEMPLATES[sh])
        real = enc.encode

        def spy(t, tmpl):
            Templates.used[sh] = (tmpl[0], tmpl[1])
            return real(t, tmpl)
        enc.encode = spy
        try:
            return super().get(text)
        finally:
            enc.encode = real

    def raw(self, sh):
        if TB is not None and sh in TB.TEMPLATES:
            Templates.used[sh] = TB.TEMPLATES[sh]
            return TB.TEMPLATES[sh]
        lo, hi, _ = self.t[sh]
        Templates.used[sh] = (lo, hi)
        return lo, hi


def dump_templates(path):
    t = dict(TB.TEMPLATES) if TB is not None else {}
    t.update(Templates.used)
    nl = chr(10)
    lines = ['"""templates120.py: one sm_120 instruction of each shape sm120/full120.py writes, as (low word, high word',
             'without its control field). Made by full120.py with QC_DUMP_TEMPLATES from a CUDA 12.8 cubin. Do not edit."""',
             "TEMPLATES = {"]
    for k in sorted(t):
        lines.append("    %r: (0x%016x, 0x%016x)," % (k, t[k][0], t[k][1] & ~enc.CTRL_MASK))
    lines.append("}")
    with open(path, "w", newline=nl) as f:
        f.write(nl.join(lines) + nl)


def mov_words(text, T):
    """MOV Rd, Rs or MOV Rd, <immediate>: the source sits in the second source's field (bit 32), which
    sm89/enc.py's encoder does not know for a MOV."""
    m = re.match(r"MOV R(\d+), (?:R(\d+)|(-?0x[0-9a-f]+))$", text)
    lo, hi = T.raw("MOV R,R" if m[2] else "MOV R,I")
    w = enc.setf(lo | hi << 64, 16, 8, int(m[1]))
    if m[2]:
        w = enc.setf(w, 32, 8, int(m[2]))
    else:
        w = enc.setf(w, 32, 32, int(m[3], 16) & 0xFFFFFFFF)
    return w & ((1 << 64) - 1), w >> 64


def run(cubin, out, **opt):
    OPT.update(DEFAULTS)
    OPT.update(opt)
    IF.NVDISASM[0] = os.environ.get("NVDISASM", "nvdisasm")
    F.GLOBAL1[:] = GLOBAL1
    info = find_iface(cubin)
    const = IF.const_bank3(cubin)
    c = dict(seq=patch75.find_u64(const, [v for row in F.RC_SEQ for v in row]), p22=patch75.find_u64(const, F.RC_P22))
    gen.Prog = Prog120
    hand.gen_lat = patch75.gen_lat
    hand.cells = F.cells
    hand.LIMIT[0] = OPT["limit"]
    hand.PLIMIT[0] = OPT["plimit"]
    hand.LATW = WriteAfterWrite()
    g, S = build(c)
    F.g_pairs[0] = g.pairs
    avail = [r for r in info["clob"] if r != 1]
    top = sorted(avail)[-24:]
    home = [(top[2 * i], top[2 * i + 1]) for i in range(12)]
    pre = {}
    for i in range(12):
        pre[S[i][0][1:]], pre[S[i][1][1:]] = home[i]
    mv = [(home[i][h], info["inp"][i][h]) for i in range(12) for h in range(2) if home[i][h] != info["inp"][i][h]]
    busy_mv = set(d for d, _ in mv) | set(s_ for _, s_ in mv)
    ent = [gen.V("MOV R%d, R%d" % m, "ent") for m in hand.parallel_moves(mv, [r for r in avail if r not in busy_mv])]
    for o in ent:
        o.sub = 0
    g.ops = ent + g.ops
    carried = set(pre) | set(GLOBAL1) | set(F.GLOBAL2)
    blocks = {}
    for b in F.ORDER[:-1]:
        ops = [o for o in g.ops if o.block == b]
        if b in F.FIXED:
            blocks[b] = ops
        else:
            blocks[b] = []
            for sub in sorted(set(o.sub for o in ops)):
                blocks[b] += hand.schedule_block([o for o in ops if o.sub == sub], carried)
    loc = F.allocate(blocks, pre, avail, npred=len(info["cpreds"]))
    assert info["cpreds"] == list(range(len(info["cpreds"]))), info["cpreds"]
    blocks["ent"].append(gen.V("BRA " + {1: ".EXIT", 3: ".E", 4: ".P"}.get(OPT.get("probe", 0), ".S"), "ent"))
    code = [[canon(hand.lower(o.text, loc)), o.unit, b] for b in F.ORDER[:-1] for o in blocks[b]]
    moves = []
    for i in range(12):
        for h in range(2):
            for dst in info["out"][i][h]:
                if dst != home[i][h]:
                    moves.append((dst, home[i][h]))
    used = set(d for d, _ in moves) | set(s for _, s in moves)
    for d, s in hand.parallel_moves(moves, [r for r in avail if r not in used]):
        code.append(["MOV R%d, R%d" % (d, s), "A", "exit"])
    # The return address. ptxas keeps it as an immediate in the function (MOV R4, <the call's next instruction>),
    # and the cubin's Mercury half has a relocation on that MOV, placed through .nv.capmerc.text. So the MOV stays
    # where ptxas put it, bit for bit, and the section stays as compiled; the hand code branches to it. (With the
    # MOV written elsewhere and the section rebuilt, every hand kernel gave wrong hits on an RTX 5070, 28 Nov.)
    code.append(["BRA .RA", "B", "exit"])
    idx = {b: [i for i, x in enumerate(code) if x[2] == b] for b in F.ORDER}
    stalls, wait, wsb = F.timing(code, idx)
    stalls[-1] = max(stalls[-1], 9)
    # A constant load reads its pointer when it runs, not when it is issued: the partial rounds' pointer was
    # stepped a few instructions after its LDC.64, and on an RTX 5070 some warps then loaded a later round's
    # constant (the probes: S-boxes and layers right, partial rounds wrong, differently in a block's two halves).
    # So whatever writes a load's pointer register waits on that load's scoreboard, as its readers do.
    for name, ix in idx.items():
        pend = {}
        for i in ix:
            x = code[i][0]
            if x.startswith("LDC.64"):
                pend[wsb[i]] = F.Ldc(x).b
            elif "BRA" not in x:
                for sb, base_reg in pend.items():
                    if base_reg in dest_regs(x):
                        wait[i] |= 1 << sb
    entry, ret = info["entry"], info["ret"]
    room = (ret - entry) // 16
    xi = (info["ret_mov"] - entry) // 16
    tail = {xi + 1: (info["retfix"][1], 9), xi + 2: ("BRA .RET", 5)}
    assert xi + 2 < room and info["retfix"][0].startswith("MOV R%d," % info["retreg"])
    assert len(code) <= xi, "the hand code has %d instructions, the function has room for %d" % (len(code), xi)
    labs = {k: entry + 16 * idx[v][0] for k, v in F.LABEL.items()}
    labs[".RET"] = ret
    labs[".RA"] = info["ret_mov"]
    # nvdisasm prints sm_120's control fields after each instruction: taken off before anything reads the text
    rows = [dict(r, text=clean(r["text"])) for r in IF.listing(cubin)[0]]
    extra = [[dict(r, text=clean(r["text"])) for r in IF.listing(p)[0]]
             for p in os.environ.get("QC_TEMPLATES", "").split(os.pathsep) if p]
    T = Templates([rows] + extra)
    for r in rows:      # the two shapes the encoders below take whole
        if re.match(r"LDC\.64 R\d+, c\[0x3\]\[R\d+", r["text"]):
            T.t.setdefault("LDC.64 R,[R]", (r["lo"], r["hi"] & ~enc.CTRL_MASK, r["text"]))
        if re.match(r"(@!?P\d )?BRA ", r["text"]):
            T.t.setdefault("BRA", (r["lo"], r["hi"] & ~enc.CTRL_MASK, r["text"]))
    ldc_t = T.raw("LDC.64 R,[R]")
    bra_t = dict(zip(("lo", "hi"), T.raw("BRA")))
    b = bytearray(open(cubin, "rb").read())
    off, _size = P89.text_section(b)
    base = rows[0]["addr"]
    want, special = {}, {}
    for i in range(room):
        a = entry + 16 * i
        if i < len(code):
            x = code[i][0]
            if "BRA" in x:
                lo, hi = bra_words(x, a, labs, bra_t)
            elif x.startswith("LDC.64"):
                lo, hi = F.encode_ldc(x, ldc_t)
                special[a] = x
            elif x.startswith("MOV "):
                lo, hi = mov_words(x, T)
            else:
                lo, hi = T.get(x)
                if x.startswith("IMAD.WIDE.U32.X"):
                    special[a] = x
            s = max(stalls[i], OPT["stall"], OPT["ldcs"] if x.startswith("LDC") else 0)
            bn = code[i][2]
            if OPT["slow"] & ((1 if bn[0] == "P" else 2 if bn == "S" else 4 if bn == "E" else 8)
                              | {"Ppro": 16, "Ploop": 32, "Pepi": 64}.get(bn, 0)):
                s = 15
            ct = (s & 15) | ((1 if s <= OPT["ybit"] else 0) << 4) | (wsb[i] << 5) | (7 << 8) | (wait[i] << 11)
            hi = (hi & ~enc.CTRL_MASK) | (ct << 41)
            want[a] = x
        elif i == xi:
            continue
        elif i in tail:
            x, st = tail[i]
            lo, hi = bra_words(x, a, labs, bra_t) if "BRA" in x else mov_words(x, T)
            hi = (hi & ~enc.CTRL_MASK) | (((st & 15) | (7 << 5) | (7 << 8)) << 41)
            want[a] = x
        else:
            lo, hi = fuse.NOP[0], ((7 << 5) | (7 << 8)) << 41
        struct.pack_into("<QQ", b, off + a - base, lo, hi)
    # cap=1 (a trial): .nv.capmerc.text rebuilt for the new code, where its layout is the one capmerc.py knows
    # (CUDA 12.8's). Off by default: see the return address above.
    try:
        if not OPT.get("cap", 0):
            raise ValueError("left as compiled")
        cm = capmerc.parse(b)
        recs = capmerc.records_of(cm)
        i0 = (entry - base) // 16
        ldc_r = next(recs[(r["addr"] - base) // 16] for r in rows
                     if entry <= r["addr"] < ret and re.match(r"LDC\.64 R\d+, c\[0x3\]\[R\d+", r["text"]))
        for i in range(room):
            a = entry + 16 * i
            x = special.get(a)
            if x is None:
                recs[i0 + i] = None
            elif x.startswith("LDC.64"):
                op = F.Ldc(x)
                recs[i0 + i] = capmerc.ldc64(ldc_r, op.d, op.b, op.off)
            else:
                recs[i0 + i] = capmerc.WIDE_X_IMM if re.search(r", -?0x[0-9a-f]+, R\d+, P\d$", x) else capmerc.WIDE_X_REG
        cm["bits"] = [1 if r is None else 0 for r in recs]
        cm["recs"] = [r for r in recs if r is not None]
        b = capmerc.replace(b, cm)
    except (ValueError, StopIteration):
        pass
    open(out, "wb").write(b)
    if os.environ.get("QC_DUMP_TEMPLATES"):
        dump_templates(os.environ["QC_DUMP_TEMPLATES"])
    # read back
    rows2, labels2, _ = IF.listing(out)
    got = {r["addr"]: clean(r["text"]) for r in rows2}
    bad = 0
    for a, x in want.items():
        h = got.get(a, "?")
        if "BRA" in x:
            m = re.search(r"\((\.L_x_\d+)\)", h)
            ok = bool(m) and labels2.get(m[1]) == labs[x.split()[-1]] and x.split("BRA")[0] == h.split("BRA")[0]
        else:
            ok = h == x
        if not ok:
            bad += 1
            if bad < 8:
                print("MISMATCH %05x asked %r got %r" % (a, x, h))
    assert bad == 0, "%d instructions read back differently" % bad
    # simulate the patched function from the cubin
    body = [r for r in rows2 if entry <= r["addr"] <= ret]
    sops = [F.mkop(clean(r["text"])) for r in body]
    at = {r["addr"]: i for i, r in enumerate(body)}
    rnd = random.Random(11)
    steps = 0
    for _trial in range(3):
        s = [rnd.randrange(P) for _ in range(12)]
        m = F.Machine(const)
        for r in range(256):
            m.r[r] = 0xA000 + r
        for i in range(12):
            m.r[info["inp"][i][0]], m.r[info["inp"][i][1]] = s[i] & 0xFFFFFFFF, s[i] >> 32
        i, steps = 0, 0
        while not sops[i].op.startswith("RET"):
            if m.run(sops[i]) == "BRA":
                i = at[labels2[re.search(r"\((\.L_x_\d+)\)", sops[i].target)[1]]]
            else:
                i += 1
            steps += 1
            assert steps < 400000
        w = probe_ref(OPT.get("probe", 0), s)
        for i in range(12):
            for lo in info["out"][i][0]:
                for hi in info["out"][i][1]:
                    assert (m.r[lo] | m.r[hi] << 32) % P == w[i], "simulation: element %d is wrong" % i
    per = {bn: (len(idx[bn]), sum(stalls[i] for i in idx[bn])) for bn in F.ORDER}
    return code, stalls, per, steps


if __name__ == "__main__":
    kv = dict(a.split("=") for a in sys.argv[3:])
    code, stalls, per, steps = run(sys.argv[1], sys.argv[2], **{k: int(v) for k, v in kv.items()})
    from collections import Counter
    print("instructions %d; a call runs %d" % (len(code), steps))
    for bn in F.ORDER:
        u = Counter(x[1] for x in code if x[2] == bn)
        print("%-6s %4d instructions, %5d cycles for one warp, %s" % (bn, per[bn][0], per[bn][1], dict(u)))
