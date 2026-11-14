"""asm89.py: a small assembler for the sm_89 instructions the hand-written fold loop needs, and the tool that puts
such a loop into a compiled cubin in place of the loop function ptxas made (probes/hand89/fold_h.cuh).

  python asm89.py check <cubin> <sass>     every instruction of the listing that the assembler knows is assembled
                                           from its text and compared with the cubin's bytes
  python asm89.py hash <cubin>             the cubin's sha256 with its string table zeroed (fold_hash)

It assembles from the text nvdisasm prints, one instruction form at a time, each form copied from what ptxas 12.6
wrote for sm_89 (the bits outside the operand fields are an example's). A patched cubin is disassembled again and
every instruction's text compared with what was asked for (verify), so a wrong field cannot go unseen.

The 128-bit instruction: bits 0-11 operation and form, 12-14 predicate, 15 its negation, 16-23 destination,
24-31 first source, 32-39 second source or 32-63 an immediate (40-63 a memory offset), 64-71 third source,
105-125 the control field: stall 4 bits, yield 1, write scoreboard 3, read scoreboard 3, wait mask 6, reuse 4.
"""
import os
import re
import struct
import subprocess
import sys

CTRL_SHIFT = 41                     # in the high 64-bit word
CTRL_MASK = 0x1FFFFF << CTRL_SHIFT
RZ = 255


def ctrl(stall=1, yld=True, wsb=None, rsb=None, wait=(), reuse=0):
    c = (stall & 15) | ((1 if yld else 0) << 4) | ((7 if wsb is None else wsb) << 5) | \
        ((7 if rsb is None else rsb) << 8) | (reuse << 17)
    for w in wait:
        c |= 1 << (11 + w)
    return c


def ctrl_text(c):
    wait = (c >> 11) & 63
    b = "".join(str(i) if wait >> i & 1 else "-" for i in range(6))
    r, w = (c >> 8) & 7, (c >> 5) & 7
    return "[B%s:R%s:W%s:%s:S%02d]" % (b, "-" if r == 7 else r, "-" if w == 7 else w, "Y" if c >> 4 & 1 else "-",
                                      c & 15)


def _reg(s):
    return RZ if s == "RZ" else int(s[1:])


def _num(s):
    s = s.strip()
    return -int(s[1:], 16) if s.startswith("-") else int(s, 16)


R = r"(R\d+|RZ)"
I = r"(-?0x[0-9a-f]+)"
MEM = r"\[" + R + r"(?:\+" + I + r")?\]"
MEM64 = r"\[" + R + r"\.64(?:\+" + I + r")?\]"
CB = r"c\[0x0\]\[0x([0-9a-f]+)\]"


def _mem(lo, m, i):
    """Base register and 24-bit offset of a memory operand matched at groups i, i + 1."""
    off = _num(m.group(i + 1)) if m.group(i + 1) else 0
    assert -(1 << 23) <= off < 1 << 24
    return lo | (_reg(m.group(i)) << 24) | ((off & 0xFFFFFF) << 40)


# A shared-memory address with a uniform register in it, [Rn+URm+imm] (ptxas 12.6's encodings, goal-nvk19). The
# second word's bit 27 says a uniform register is there; a load names it in bits 32-37 of the first word, a store
# (whose data register is in those bits) in bits 0-5 of the second.
MEMU = r"\[" + R + r"\+UR(\d+)(?:\+" + I + r")?\]"
UMEM = 0x8000000


def _memu(lo, m, i):
    """Base register, uniform register and 24-bit offset of a MEMU operand matched at groups i to i + 2."""
    off = _num(m.group(i + 2)) if m.group(i + 2) else 0
    assert 0 <= off < 1 << 24 and int(m.group(i + 1)) < 63
    return lo | (_reg(m.group(i)) << 24) | ((off & 0xFFFFFF) << 40)


def _f(lo, hi, **k):
    for name, val in k.items():
        if name == "rd":
            lo |= val << 16
        elif name == "ra":
            lo |= val << 24
        elif name == "rb":
            lo |= val << 32
        elif name == "imm":
            lo |= (val & 0xFFFFFFFF) << 32
        elif name == "rc":
            hi |= val
        elif name == "cb":
            lo |= (val >> 2) << 40
        elif name == "pd":
            hi |= val << 17
    return lo, hi


LDGSTS_UR = [10]


def _ldgsts(m, cached=0):
    dst, src = _num(m[2]) if m[2] else 0, _num(m[4]) if m[4] else 0
    assert 0 <= dst < 1 << 20 and -2048 <= src < 2048, (dst, src)
    lo = 0x7fae | _reg(m[1]) << 16 | _reg(m[3]) << 24 | (src & 0xFFF) << 32 | dst << 44
    return lo, 0x0b901c40 | cached | LDGSTS_UR[0]


# (pattern, builder). The builder gets the match and returns (lo, hi) with predicate 7 (always) and no control.
FORMS = [
    (r"IMMA\.16832\.S8\.S8 " + R + ", " + R + r"\.ROW, " + R + r"\.COL, " + R,
     lambda m: _f(0x7237, 0x405c00, rd=_reg(m[1]), ra=_reg(m[2]), rb=_reg(m[3]), rc=_reg(m[4]))),
    # U8.U8 (goal-nvk11): the same datapath reading its operands as unsigned. The fold's operand bytes are 0 to
    # 126, so the sums are the same either way.
    (r"IMMA\.16832\.U8\.U8 " + R + ", " + R + r"\.ROW, " + R + r"\.COL, " + R,
     lambda m: _f(0x7237, 0x400c00, rd=_reg(m[1]), ra=_reg(m[2]), rb=_reg(m[3]), rc=_reg(m[4]))),
    # Mixed (goal-nvk17, ptxas 12.6's encodings): S8.U8 reads A signed and B unsigned, U8.S8 the other way.
    (r"IMMA\.16832\.S8\.U8 " + R + ", " + R + r"\.ROW, " + R + r"\.COL, " + R,
     lambda m: _f(0x7237, 0x401c00, rd=_reg(m[1]), ra=_reg(m[2]), rb=_reg(m[3]), rc=_reg(m[4]))),
    (r"IMMA\.16832\.U8\.S8 " + R + ", " + R + r"\.ROW, " + R + r"\.COL, " + R,
     lambda m: _f(0x7237, 0x404c00, rd=_reg(m[1]), ra=_reg(m[2]), rb=_reg(m[3]), rc=_reg(m[4]))),
    (r"LDSM\.16\.MT88\.4 " + R + ", " + MEM, lambda m: (_mem(0x783b | _reg(m[1]) << 16, m, 2), 0x4200)),
    # The same three with a uniform register in the address (rfold.py sr: the per-warp bases are there).
    (r"LDSM\.16\.MT88\.4 " + R + ", " + MEMU,
     lambda m: (_memu(0x783b | _reg(m[1]) << 16 | int(m[3]) << 32, m, 2), 0x4200 | UMEM)),
    (r"LDS\.128 " + R + ", " + MEMU, lambda m: (_memu(0x7984 | _reg(m[1]) << 16 | int(m[3]) << 32, m, 2), 0xc00 | UMEM)),
    (r"LDS\.64 " + R + ", " + MEMU, lambda m: (_memu(0x7984 | _reg(m[1]) << 16 | int(m[3]) << 32, m, 2), 0xa00 | UMEM)),
    (r"STS\.128 " + MEMU + ", " + R, lambda m: (_memu(0x7988 | _reg(m[4]) << 32, m, 1), 0xc00 | UMEM | int(m[2]))),
    # An add of a uniform register (ptxas 12.6's IADD3 R4, R0, UR4, RZ), and a register moved into a uniform one
    # (R2UR, a load-like step that sets a scoreboard).
    (r"IADD3 " + R + ", " + R + r", UR(\d+), " + R,
     lambda m: _f(0x7c10 | int(m[3]) << 32, 0xfffe000, rd=_reg(m[1]), ra=_reg(m[2]), rc=_reg(m[4]))),
    (r"R2UR UR(\d+), " + R, lambda m: (0x73c2 | int(m[1]) << 16 | _reg(m[2]) << 24, 0xe0000)),
    # sm_120a's byte transpose (goal-nv50d, ptxas 13.3's word: ldmatrix.m16n16.x2.trans.b8): two 16 x 16 byte
    # matrices, four registers. Only sm_120 has it (rfold.py b8).
    (r"LDSM\.8\.MT1616\.2 " + R + ", " + MEM, lambda m: (_mem(0x783b | _reg(m[1]) << 16, m, 2), 0x13900)),
    # ... with a uniform register in the address (b8 with sr, goal-nv50g; the same fields as LDSM.16's).
    (r"LDSM\.8\.MT1616\.2 " + R + ", " + MEMU,
     lambda m: (_memu(0x783b | _reg(m[1]) << 16 | int(m[3]) << 32, m, 2), 0x13900 | UMEM)),
    (r"LDS\.128 " + R + ", " + MEM, lambda m: (_mem(0x7984 | _reg(m[1]) << 16, m, 2), 0xc00)),
    (r"LDS\.64 " + R + ", " + MEM, lambda m: (_mem(0x7984 | _reg(m[1]) << 16, m, 2), 0xa00)),
    (r"LDS " + R + ", " + MEM, lambda m: (_mem(0x7984 | _reg(m[1]) << 16, m, 2), 0x800)),
    (r"STS\.128 " + MEM + ", " + R, lambda m: (_mem(0x7388 | _reg(m[3]) << 32, m, 1), 0xc00)),
    (r"STS\.64 " + MEM + ", " + R, lambda m: (_mem(0x7388 | _reg(m[3]) << 32, m, 1), 0xa00)),
    (r"STS " + MEM + ", " + R, lambda m: (_mem(0x7388 | _reg(m[3]) << 32, m, 1), 0x800)),
    # 4-byte local load and store (ptxas 12.6's encodings, goal-nvk13: the thread's transcript in the kernel's frame).
    (r"LDL " + R + ", " + MEM, lambda m: (_mem(0x7983 | _reg(m[1]) << 16, m, 2), 0x100800)),
    (r"STL " + MEM + ", " + R, lambda m: (_mem(0x7387 | _reg(m[3]) << 32, m, 1), 0x100800)),
    (r"LDG\.E\.128 " + R + ", " + MEM64, lambda m: (_mem(0x7981 | _reg(m[1]) << 16 | 6 << 32, m, 2), 0x0c1e1d00)),
    (r"LDG\.E\.64 " + R + ", " + MEM64, lambda m: (_mem(0x7981 | _reg(m[1]) << 16 | 6 << 32, m, 2), 0x0c1e1b00)),
    (r"ATOMS\.XOR RZ, " + MEM + ", " + R, lambda m: (_mem(0x738c | RZ << 16 | _reg(m[3]) << 32, m, 1), 0x3800000)),
    (r"IADD3 " + R + ", " + R + ", " + I + ", " + R,
     lambda m: _f(0x7810, 0x7ffe000, rd=_reg(m[1]), ra=_reg(m[2]), imm=_num(m[3]), rc=_reg(m[4]))),
    (r"IADD3 " + R + ", -" + R + ", " + I + ", " + R,
     lambda m: _f(0x7810, 0x7ffe100, rd=_reg(m[1]), ra=_reg(m[2]), imm=_num(m[3]), rc=_reg(m[4]))),
    # SEL d, a, b, [!]Pp: d = Pp ? a : b (ptxas 13.3's sm_120 words, goal-nv50q: the predicate in bits 87-89, its
    # negation in bit 90).
    (r"SEL " + R + ", " + R + ", " + R + r", (!?)P(\d)",
     lambda m: (0x7207 | _reg(m[1]) << 16 | _reg(m[2]) << 24 | _reg(m[3]) << 32,
                int(m[5]) << 23 | (1 << 26 if m[4] else 0))),
    (r"PRMT " + R + ", " + R + ", " + I + ", " + R,
     lambda m: _f(0x7816, 0, rd=_reg(m[1]), ra=_reg(m[2]), imm=_num(m[3]), rc=_reg(m[4]))),
    # LOP3 into a predicate (goal-nv50r trh): Pd = (Ra & imm) != 0 with lut 0xc0; the predicate in bits 81-83.
    (r"LOP3\.LUT P(\d), RZ, " + R + ", " + I + ", RZ, " + I + ", !PT",
     lambda m: _f(0x7812, (0x78e0000 & ~(7 << 17)) | _num(m[4]) << 8, pd=int(m[1]), rd=RZ, ra=_reg(m[2]),
                  imm=_num(m[3]), rc=RZ)),
    # 8-byte local load and store (goal-nv50r): the size field as LDS.64's.
    (r"LDL\.64 " + R + ", " + MEM, lambda m: (_mem(0x7983 | _reg(m[1]) << 16, m, 2), 0x100a00)),
    (r"STL\.64 " + MEM + ", " + R, lambda m: (_mem(0x7387 | _reg(m[3]) << 32, m, 1), 0x100a00)),
    (r"LOP3\.LUT " + R + ", " + R + ", " + R + ", " + R + ", " + I + ", !PT",
     lambda m: _f(0x7212, 0x78e0000 | _num(m[5]) << 8, rd=_reg(m[1]), ra=_reg(m[2]), rb=_reg(m[3]), rc=_reg(m[4]))),
    (r"LOP3\.LUT " + R + ", " + R + ", " + I + ", " + R + ", " + I + ", !PT",
     lambda m: _f(0x7812, 0x78e0000 | _num(m[5]) << 8, rd=_reg(m[1]), ra=_reg(m[2]), imm=_num(m[3]), rc=_reg(m[4]))),
    (r"IMAD(?:\.IADD)? " + R + ", " + R + ", " + I + ", " + R,
     lambda m: _f(0x7824, 0x78e0200, rd=_reg(m[1]), ra=_reg(m[2]), imm=_num(m[3]), rc=_reg(m[4]))),
    (r"IMAD\.SHL\.U32 " + R + ", " + R + ", " + I + ", RZ",
     lambda m: _f(0x7824, 0x78e0000, rd=_reg(m[1]), ra=_reg(m[2]), imm=_num(m[3]), rc=RZ)),
    (r"IMAD\.MOV\.U32 " + R + ", RZ, RZ, " + R,
     lambda m: _f(0x7224, 0x78e0000, rd=_reg(m[1]), ra=RZ, rb=RZ, rc=_reg(m[2]))),
    (r"IMAD\.MOV\.U32 " + R + ", RZ, RZ, " + I,
     lambda m: _f(0x7424, 0x78e0000, rd=_reg(m[1]), ra=RZ, imm=_num(m[2]), rc=RZ)),
    (r"IMAD\.MOV\.U32 " + R + ", RZ, RZ, " + CB,
     lambda m: _f(0x7624, 0x78e0000, rd=_reg(m[1]), ra=RZ, cb=int(m[2], 16), rc=RZ)),
    (r"MOV " + R + ", " + I, lambda m: _f(0x7802, 0xf00, rd=_reg(m[1]), imm=_num(m[2]))),
    (r"IMAD\.WIDE\.U32 " + R + ", " + R + ", " + R + ", " + CB,
     lambda m: _f(0x7625, 0x78e0000, rd=_reg(m[1]), ra=_reg(m[2]), cb=int(m[4], 16), rc=_reg(m[3]))),
    (r"SHF\.L\.W\.U32\.HI " + R + ", " + R + ", " + R + ", " + R,
     lambda m: _f(0x7219, 0x10e00, rd=_reg(m[1]), ra=_reg(m[2]), rb=_reg(m[3]), rc=_reg(m[4]))),
    (r"SHF\.L\.W\.U32\.HI " + R + ", " + R + ", " + I + ", " + R,
     lambda m: _f(0x7819, 0x10e00, rd=_reg(m[1]), ra=_reg(m[2]), imm=_num(m[3]), rc=_reg(m[4]))),
    (r"SHF\.R\.U32\.HI " + R + ", RZ, " + I + ", " + R,
     lambda m: _f(0x7819, 0x11600, rd=_reg(m[1]), ra=RZ, imm=_num(m[2]), rc=_reg(m[3]))),
    (r"ISETP\.GE\.U32\.AND P(\d), PT, " + R + ", " + CB + ", PT",
     lambda m: _f(0x7a0c, 0x3f06070, pd=int(m[1]), ra=_reg(m[2]), cb=int(m[3], 16))),
    (r"ISETP\.GE\.U32\.AND P(\d), PT, " + R + ", " + I + ", PT",
     lambda m: _f(0x780c, 0x3f06070, pd=int(m[1]), ra=_reg(m[2]), imm=_num(m[3]))),
    (r"ISETP\.GE\.U32\.AND P(\d), PT, " + R + ", " + R + ", PT",
     lambda m: _f(0x720c, 0x3f06070, pd=int(m[1]), ra=_reg(m[2]), rb=_reg(m[3]))),
    (r"ISETP\.GT\.U32\.AND P(\d), PT, " + R + ", " + I + ", PT",
     lambda m: _f(0x780c, 0x3f04070, pd=int(m[1]), ra=_reg(m[2]), imm=_num(m[3]))),
    (r"ISETP\.NE\.AND P(\d), PT, " + R + ", " + I + ", PT",
     lambda m: _f(0x780c, 0x3f05270, pd=int(m[1]), ra=_reg(m[2]), imm=_num(m[3]))),
    (r"ISETP\.NE\.AND P(\d), PT, " + R + ", " + R + ", PT",
     lambda m: _f(0x720c, 0x3f05270, pd=int(m[1]), ra=_reg(m[2]), rb=_reg(m[3]))),
    (r"CS2R " + R + ", SR_CLOCKLO", lambda m: _f(0x7805, 0x15000, rd=_reg(m[1]))),
    (r"CS2R " + R + ", SRZ", lambda m: _f(0x7805, 0x1ff00, rd=_reg(m[1]))),
    (r"S2R " + R + r", SR_TID\.X", lambda m: _f(0x7919, 0x2100, rd=_reg(m[1]))),
    (r"BAR\.SYNC\.DEFER_BLOCKING 0x0", lambda m: (0x7b1d, 0x10000)),
    # A barrier of its own number for a part of the block: the number, then how many threads it waits for
    # (bar.sync a, b). Taken from ptxas's own encodings of bar.sync 1, 128; 2, 128; 3, 64.
    (r"BAR\.SYNC\.DEFER_BLOCKING " + I + ", " + I,
     lambda m: (0x7b1d | _num(m[1]) << 54 | _num(m[2]) << 42, 0x10000)),
    # The same with the number in a register (bar.sync r, b; ptxas 12.6's encoding). The card takes the number
    # from the register's low four bits: 16 c + 1 and 16 c + 2 meet as barriers 1 and 2 (checked on rig01's
    # RTX 4090 with a barrier test, goal-nvk12, and against a run with no barrier, which read stale data).
    (r"BAR\.SYNC\.DEFER_BLOCKING " + R + ", " + I,
     lambda m: (0x751d | _reg(m[1]) << 32 | _num(m[2]) << 42, 0x10000)),
    # bar.arrive r, b (ptxas 12.6's encoding, goal-nvk13): the same word as bar.sync r, b with 0x2000 in the
    # second where bar.sync has 0x10000.
    (r"BAR\.ARV " + I + ", " + I,
     lambda m: (0x7b1d | _num(m[1]) << 54 | _num(m[2]) << 42, 0x2000)),
    (r"BAR\.ARV " + R + ", " + I,
     lambda m: (0x751d | _reg(m[1]) << 32 | _num(m[2]) << 42, 0x2000)),
    (r"ISETP\.GT\.U32\.AND P(\d), PT, " + R + ", " + I + ", PT",
     lambda m: _f(0x780c, 0x3f04070, pd=int(m[1]), ra=_reg(m[2]), imm=_num(m[3]))),
    # 4-byte global load and store (the uniform register pair goes in as for the 16-byte load: gfold.ldg).
    (r"LDG\.E " + R + ", " + MEM64, lambda m: (_mem(0x7981 | _reg(m[1]) << 16 | 6 << 32, m, 2), 0x0c1e1900)),
    (r"STG\.E " + MEM64 + ", " + R, lambda m: (_mem(0x7986 | _reg(m[3]) << 32, m, 1), 0x0c101900 | 6)),
    # A 16-byte copy from global to shared memory that touches no register (cp.async): the shared address takes a
    # 20-bit offset, the global one a signed 12-bit offset. Bits 0-5 of the high word name the uniform register
    # pair with the address space's base, as LDG's bits 32-39 do (LDGSTS_UR; nvdisasm does not print it).
    (r"LDGSTS\.E\.BYPASS\.128 " + MEM + ", " + MEM64, lambda m: _ldgsts(m)),
    (r"LDGSTS\.E\.128 " + MEM + ", " + MEM64, lambda m: _ldgsts(m, 0x20000)),      # through the SM's cache
    (r"LDGDEPBAR", lambda m: (0x79af, 0)),
    # The scoreboard's number is in bits 44-46 (checked against nvdisasm for sm_120: SB1, SB2, SB5).
    (r"DEPBAR\.LE SB([0-5]), " + I, lambda m: (0x000080000000791a | int(m[1]) << 44 | (_num(m[2]) & 63) << 38, 0)),
    (r"NOP", lambda m: (0x7918, 0)),
    (r"YIELD", lambda m: (0x7946, 0x3800000)),
    (r"ULDC\.64 UR6, c\[0x0\]\[0x118\]", lambda m: (0x0000460000067ab9, 0x0000000000000a00)),
]
FORMS = [(re.compile(p + r"$"), f) for p, f in FORMS]
BRA_RE = re.compile(r"BRA (\S+)$")


def assemble(text, here=None, labels=None):
    """(lo, hi) of one instruction's text, without the control field. A BRA's operand is a label of `labels`
    (name -> address) and `here` the instruction's own address."""
    t = re.sub(r"\s+", " ", text.strip()).replace(".reuse", "")
    pred = 7
    m = re.match(r"@(!?)P(\d) (.*)$", t)
    if m:
        pred = int(m[2]) | (8 if m[1] else 0)
        t = m[3]
    m = BRA_RE.match(t)
    if m:
        off = labels[m[1]] - (here + 16)
        lo = 0x0947 | ((off & 0xFFFFFFFF) << 32)
        hi = 0x3800000 | (0x3ffff if off < 0 else 0)
    else:
        for rx, f in FORMS:
            m = rx.match(t)
            if m:
                lo, hi = f(m)
                break
        else:
            raise ValueError("no form for: " + text)
        assert lo & 0xF000 == 0x7000, text
        lo &= ~0xF000
    return lo | (pred << 12), hi


class Ins:
    """One instruction of a hand-written stream: its text, its control field's parts, and a label on it."""

    def __init__(self, text, stall=1, yld=True, wsb=None, rsb=None, wait=(), reuse=0, label=None, raw=None):
        self.text, self.stall, self.yld, self.wsb, self.rsb = text, stall, yld, wsb, rsb
        self.wait, self.reuse, self.label, self.raw = tuple(wait), reuse, label, raw

    def ctrl(self):
        return ctrl(self.stall, self.yld, self.wsb, self.rsb, self.wait, self.reuse)

    def __repr__(self):
        return "%s %s" % (ctrl_text(self.ctrl()), self.text)


# ---------------------------------------------------------------- the cubin
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


def set_barriers(path, n):
    """Writes into the cubin how many barriers its kernel uses (numbers 0 .. n - 1). ptxas keeps the count in
    the code section's flags, from bit 20 (SHF_BARRIERS in nvdisasm's listing): a loop that uses bar.sync 1
    and 2 in a kernel compiled with bar.sync 0 alone must raise it to 3."""
    b = bytearray(open(path, "rb").read())
    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    entsize, num, strndx = struct.unpack_from("<HHH", b, 0x3A)
    stro = struct.unpack_from("<IIQQQQ", b, shoff + strndx * entsize)[4]
    done = 0
    for i in range(num):
        at = shoff + i * entsize
        name = struct.unpack_from("<I", b, at)[0]
        end = b.index(bytes(1), stro + name)
        if bytes(b[stro + name:end]).decode().startswith(".text.") and b"pearl" in b[stro + name:end]:
            flags = struct.unpack_from("<Q", b, at + 8)[0]
            struct.pack_into("<Q", b, at + 8, (flags & 0xFFFFF) | (n << 20))
            done += 1
    assert done == 1
    open(path, "wb").write(b)


def read_text(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


LOOP_END = [None]       # the address just past the loop function, when another local function follows it


def listing(sass_text, fn_sub="pearl"):
    """The kernel's instructions as (address, text), the address of each label, and the loop function's entry.
    LOOP_END[0] is set to the first instruction of the local function that follows the loop function, if any
    (cfold_h.cuh's kernel has three local functions; the older probes have one)."""
    rows, labels, pending, inside, entry, want_entry = [], {}, [], False, None, False
    want_end = False
    LOOP_END[0] = None
    for line in sass_text.splitlines():
        if re.match(r"\s*\.section\s", line):
            inside = ".text." in line and fn_sub in line
            continue
        if not inside:
            continue
        m = re.match(r"\s*(\.L_x_\d+):", line)
        if m:
            pending.append(m[1])
            continue
        if re.match(r"\$\S*_loop\S*:", line.strip()):
            want_entry = True
            continue
        if re.match(r"\$\S+:", line.strip()) and entry is not None and LOOP_END[0] is None:
            want_end = True
            continue
        m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
        if m:
            a = int(m[1], 16)
            rows.append((a, re.sub(r"\s+", " ", m[2].strip())))
            for p in pending:
                labels[p] = a
            pending = []
            if want_entry:
                entry, want_entry = a, False
            if want_end:
                LOOP_END[0], want_end = a, False
    return rows, labels, entry


def _in_loop(a, entry):
    return a >= entry and (LOOP_END[0] is None or a < LOOP_END[0])


def disassemble(cubin, nvdisasm):
    r = subprocess.run([nvdisasm, "-c", cubin], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("nvdisasm: " + r.stderr.decode(errors="replace")[:400])
    return r.stdout.decode("utf-8", errors="replace")


RET_REG = [2]           # the register pair the loop function returns through (region() finds it)


def region(sass_text):
    """What fold.py and sched.py may overwrite in a build of fold_h.cuh: from the loop function's entry up to its RET (which
    stays where it is: it is relative to its own address). Returns (entry, ret address, return address value)."""
    rows, _labels, entry = listing(sass_text)
    # ptxas keeps the return address where it likes: R2 in the draw and gather folds, R4 in the direct fold. The
    # function's last unconditional RET is the one kept.
    m = [re.match(r"RET\.REL\.NODEC R(\d+) ", t) for a, t in rows if entry is not None and _in_loop(a, entry)]
    m = [x for x in m if x]
    assert m, "the loop function has no RET"
    RET_REG[0] = int(m[-1][1])
    ret = [a for a, t in rows if _in_loop(a, entry) and re.match(r"RET\.REL\.NODEC R%d " % RET_REG[0], t)]
    # A function with two exits (gfold_h.cuh's two loop copies) has two RET: the last one is kept, and the
    # replacement may overwrite the other.
    assert entry is not None and len(ret) >= 1, (entry, ret)
    ret = ret[-1:]
    back = [(a, t) for a, t in rows if entry <= a < ret[0] and re.match(r"MOV R\d+, 0x[0-9a-f]+$", t)]
    # The function returns to the instruction after its one CALL: ptxas writes that address as a MOV.
    call = [a for a, t in rows if t.startswith("CALL.REL.NOINC `($")]
    if len(call) > 1:
        call = [a for a, t in rows if t.startswith("CALL.REL.NOINC `($") and "_loop" in t]
    # (In the direct fold the caller writes it, into R0, just ahead of the CALL.)
    back += [(a, t) for a, t in rows if a < entry and re.match(r"MOV R[0-9]+, 0x[0-9a-f]+$", t)]
    assert len(call) == 1 and any(int(t.split("0x")[1], 16) == call[0] + 16 for _a, t in back), (call, back)
    return entry, ret[0], call[0] + 16


def clobbers(sass_text):
    """Registers and predicates the compiled loop function writes. The kernel was compiled knowing them, so a
    replacement may write these and no others."""
    rows, _labels, entry = listing(sass_text)
    regs, preds = set(), set()
    for a, t in rows:
        if not _in_loop(a, entry):
            continue
        t = re.sub(r"^@!?P\d ", "", t).replace(".reuse", "")
        op, _, rest = t.partition(" ")
        if op.startswith(("STS", "STG", "STL", "ATOMS", "RED", "BAR", "BRA", "BSSY", "BSYNC", "RET", "CALL", "NOP",
                          "YIELD", "WARPSYNC", "EXIT")):
            continue
        first = rest.split(",")[0].strip()
        m = re.match(r"R(\d+)$", first)
        if m:
            n, wide = int(m[1]), 1
            if op.startswith("IMMA") or op.startswith("LDSM") or ".128" in op:
                wide = 4
            elif op.startswith("CS2R") or ".64" in op or "WIDE" in op:
                wide = 2
            regs.update(range(n, n + wide))
        m = re.match(r"P(\d)$", first)
        if m:
            preds.add(int(m[1]))
        if op.startswith("LEA") or op.startswith("IADD3.X") or op.startswith("IMAD.X"):
            m = re.search(r", P(\d),", rest)
            if m:
                preds.add(int(m[1]))
    return regs, preds


def loop_facts(sass_text):
    """What a hand loop takes from the compiled loop function besides its registers (clobbers): the uniform
    register pairs it loads the global address base into (ULDC.64 URn, c[0x0][0x118]), the constant-bank
    offsets it reads (the kernel's parameters are there), and how many instructions it has room for."""
    rows, _labels, entry = listing(sass_text)
    if entry is None:
        raise RuntimeError("the listing has no loop function (a label with _loop in its name)")
    urs, consts = set(), set()
    for a, t in rows:
        if not _in_loop(a, entry):
            continue
        m = re.search(r"ULDC\.64 UR(\d+), c\[0x0\]\[0x118\]", t)
        if m:
            urs.add(int(m[1]))
        consts.update(int(x, 16) for x in re.findall(r"c\[0x0\]\[0x([0-9a-f]+)\]", t))
    entry, ret, _back = region(sass_text)
    return dict(urs=urs, consts=consts, room=(ret - entry) // 16)


def barriers(sass_text, fn_sub="pearl"):
    """The barrier count nvdisasm prints for the kernel's code section (SHF_BARRIERS), 0 when it prints none."""
    inside, n = False, 0
    for line in sass_text.splitlines():
        if re.match(r"\s*\.section\s", line):
            inside = ".text." in line and fn_sub in line
        elif inside:
            m = re.search(r"SHF_BARRIERS=(\d+)", line)
            if m:
                n = int(m[1])
    return n


def fold_hash(path):
    """sha256 of a cubin with its .strtab zeroed. The string table holds the loop function's name, and nvcc puts
    a hash of the source file's path into the name of a static function, so the whole file's hash differs from
    one checkout folder to the next while every other byte (code, section headers, symbols, parameter info) is
    the same. This is the hash src/pearl_rot_sm89.ref lists."""
    import hashlib
    b = bytearray(open(path, "rb").read())
    off, size = sections(b)[".strtab"]
    b[off:off + size] = bytes(size)
    return hashlib.sha256(bytes(b)).hexdigest()


def patch(cubin_in, sass_text, cubin_out, code, nvdisasm):
    """Writes cubin_out: cubin_in with the loop function's code replaced by `code` (a list of Ins), which must end
    by falling off its end: a move of the return address, a branch to the old RET and NOPs are added. Then
    disassembles the result and compares every instruction's text with what was asked for."""
    b = bytearray(open(cubin_in, "rb").read())
    secs = sections(b)
    name = next(n for n in secs if n.startswith(".text.") and "pearl" in n)
    off, _size = secs[name]
    entry, ret, back = region(sass_text)
    code = list(code) + [Ins("MOV R%d, 0x%x" % (RET_REG[0], back)),
                         Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % (RET_REG[0] + 1)), Ins("BRA .RET", stall=5)]
    room = (ret - entry) // 16
    if len(code) > room:
        raise ValueError("the loop is %d instructions and the function has room for %d" % (len(code), room))
    labels = {".RET": ret}
    for i, ins in enumerate(code):
        if ins.label:
            assert ins.label not in labels, ins.label
            labels[ins.label] = entry + 16 * i
    for i in range(room):
        a = entry + 16 * i
        if i < len(code):
            ins = code[i]
            lo, hi = ins.raw if ins.raw else assemble(ins.text, a, labels)
            hi = (hi & ~CTRL_MASK) | (ins.ctrl() << CTRL_SHIFT)
        else:
            lo, hi = 0x7918, ctrl(0, False) << CTRL_SHIFT
        struct.pack_into("<QQ", b, off + a, lo, hi)
    open(cubin_out, "wb").write(b)
    # Read it back.
    rows, lab2, _e = listing(disassemble(cubin_out, nvdisasm))
    got = dict(rows)
    back_lab = {}
    for n, a in lab2.items():
        back_lab.setdefault(a, n)
    bad = 0
    for i, ins in enumerate(code):
        a = entry + 16 * i
        want = re.sub(r"\s+", " ", ins.text.strip())
        have = got.get(a, "?").replace(".reuse", "")
        m = BRA_RE.search(want)
        if m:
            m2 = re.search(r"BRA `\((\S+)\)", have)
            ok = bool(m2) and lab2.get(m2[1]) == labels[m[1]] and want.split("BRA")[0] == have.split("BRA")[0]
        else:
            ok = want.replace(".reuse", "") == have
        if not ok:
            bad += 1
            if bad <= 8:
                print("MISMATCH at %05x: asked %r, the cubin has %r" % (a, want, have))
    if bad:
        raise RuntimeError("%d instructions did not come back as asked" % bad)
    return len(code), room


def check(cubin, sass):
    b = bytearray(open(cubin, "rb").read())
    secs = sections(b)
    name = next(n for n in secs if n.startswith(".text.") and "pearl" in n)
    off, _size = secs[name]
    rows, labels, _entry = listing(read_text(sass))
    ok, bad, skipped = {}, 0, {}
    for a, t in rows:
        op = re.sub(r"^@!?P\d ", "", t).split(" ")[0]
        t2 = re.sub(r"`\((\S+)\)", r"\1", t)
        try:
            lo, hi = assemble(t2, a, labels)
        except (ValueError, KeyError):
            skipped[op] = skipped.get(op, 0) + 1
            continue
        lo0, hi0 = struct.unpack_from("<QQ", b, off + a)
        if op.startswith("LDG"):
            # Bits 32-39 name the uniform register pair that holds the address space's base (UR6 in the
            # loop function, UR8 in the kernel). nvdisasm does not print it.
            lo, lo0 = lo & ~(0xFF << 32), lo0 & ~(0xFF << 32)
        if op.startswith("STG"):
            hi, hi0 = hi & ~0x3F, hi0 & ~0x3F          # the uniform register pair again, in the high word
        if op.startswith("LDGSTS"):
            hi, hi0 = hi & ~0x3F, hi0 & ~0x3F          # the uniform register pair again
        if (lo, hi) != (lo0, hi0 & ~CTRL_MASK):
            bad += 1
            if bad <= 12:
                print("WRONG %05x %s: made %016x %016x, ptxas %016x %016x" % (a, t, lo, hi, lo0, hi0 & ~CTRL_MASK))
        else:
            ok[op] = ok.get(op, 0) + 1
    print("assembled the same as ptxas: %d instructions (%s)" % (sum(ok.values()), ", ".join(
        "%s %d" % kv for kv in sorted(ok.items(), key=lambda kv: -kv[1]))))
    print("wrong: %d" % bad)
    print("no form (not needed by the hand loop): " + ", ".join("%s %d" % kv for kv in sorted(skipped.items())))
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "check":
        sys.exit(1 if check(sys.argv[2], sys.argv[3]) else 0)
    if len(sys.argv) == 3 and sys.argv[1] == "hash":
        print(fold_hash(sys.argv[2]))
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
