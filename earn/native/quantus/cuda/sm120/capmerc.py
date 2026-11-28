"""capmerc.py: read and rewrite the .nv.capmerc.text section of an sm_120 cubin made by CUDA 12.8's ptxas.

Why it matters: the driver does not take an sm_120 kernel's code from .text alone. .nv.capmerc.text.<kernel> says
which instructions need more than their bits, by position, and holds a record for each of those in code order.
Code that is moved, deleted or written new without the section being rewritten to match runs records against the
wrong instructions (an illegal instruction on the card: quantus-ledger.md, 28 Nov).

Layout, read from this kernel's cubins:
    word 0      the kernel's symbol index
    word 1      0x40000000 | the count N of instructions covered (the code up to its last branch)
    N bits      low bit first; 1: the instruction is taken from .text as it is; 0: it has a record
    pad to 4
    records     one for each 0 bit, in code order; the first byte gives the size:
                0x41, 0x42   4 bytes   (IMAD.WIDE.U32.X, IADD.64, 64-bit compares and moves, CALL, RET)
                0x01        16 bytes   (some constant loads)
                0x02        32 bytes   (LDC, LDCU, LDL, STL, atomics, stores: registers, bank and offset inside)
                the last one, for the branch-to-self that ends the code, runs to the end of the section
Plain (bit 1): IADD3, IADD3.X, IMAD.WIDE.U32 with or without a carry out, IMAD, MOV, SEL, ISETP on 32 bits, NOP,
BRA with or without a predicate, EXIT.

A record of a 4-byte kind depends only on the instruction's form (every IMAD.WIDE.U32.X with a register
multiplicand has 42203014, with an immediate 42203406), so new instructions of those forms take a copy. An
LDC.64 Rd, c[0x3][Rb+off] record carries Rd, Rb and off (ldc64 below); the others are only ever moved.
"""
import struct

SIZES = {0x41: 4, 0x42: 4, 0x01: 16, 0x02: 32}
WIDE_X_REG = bytes.fromhex("42203014")
WIDE_X_IMM = bytes.fromhex("42203406")


def _sections(b):
    shoff, = struct.unpack_from("<Q", b, 0x28)
    es, n, sx = struct.unpack_from("<HHH", b, 0x3A)
    sh = [list(struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * es)) for i in range(n)]
    so = sh[sx][4]
    names = [bytes(b[so + s[0]:b.index(b"\0", so + s[0])]).decode() for s in sh]
    return shoff, es, sh, names


def find(b, kernel=None):
    """(section index, name) of the kernel's capmerc section, or None."""
    _shoff, _es, _sh, names = _sections(b)
    hits = [(i, n) for i, n in enumerate(names) if n.startswith(".nv.capmerc.text.") and (kernel is None or n.endswith(kernel))]
    if not hits:
        return None
    if len(hits) != 1:
        raise ValueError("more than one capmerc section: %r" % (hits,))
    return hits[0]


def parse(b, kernel=None):
    """{'index', 'sym', 'bits': [0/1 an instruction], 'recs': [bytes for each 0 bit], 'padbits', 'pad'}."""
    hit = find(b, kernel)
    if hit is None:
        raise ValueError("no .nv.capmerc.text section")
    _shoff, _es, sh, _names = _sections(b)
    off, size = sh[hit[0]][4], sh[hit[0]][5]
    sym, w1 = struct.unpack_from("<II", b, off)
    if w1 >> 24 != 0x40:
        raise ValueError(".nv.capmerc.text's second word is %#x: not the CUDA 12.8 layout this tool knows" % w1)
    n = w1 & 0xFFFFFF
    nb = (n + 7) // 8
    bm = bytes(b[off + 8:off + 8 + nb])
    bits = [(bm[k >> 3] >> (k & 7)) & 1 for k in range(n)]
    padbits = bm[-1] >> (n & 7) if n & 7 else 0
    p = off + ((8 + nb + 3) & ~3)
    pad = bytes(b[off + 8 + nb:p])
    recs = []
    # the last marked instruction is the branch-to-self that ends the code: its record is whatever is left
    if not bits or bits[-1]:
        raise ValueError("capmerc: the last instruction is not marked")
    for _ in range(bits.count(0) - 1):
        k = b[p]
        if k not in SIZES or p + SIZES[k] > off + size:
            raise ValueError("capmerc record kind %#x at %#x is not known" % (k, p - off))
        recs.append(bytes(b[p:p + SIZES[k]]))
        p += SIZES[k]
    if p >= off + size:
        raise ValueError("capmerc: no record left for the last instruction")
    recs.append(bytes(b[p:off + size]))
    return dict(index=hit[0], sym=sym, bits=bits, recs=recs, padbits=padbits, pad=pad)


def build(c):
    bits, n = c["bits"], len(c["bits"])
    if bits.count(0) != len(c["recs"]):
        raise ValueError("capmerc: %d records for %d marked instructions" % (len(c["recs"]), bits.count(0)))
    bm = bytearray((n + 7) // 8)
    for k, v in enumerate(bits):
        if v:
            bm[k >> 3] |= 1 << (k & 7)
    if n & 7:
        bm[-1] |= (c["padbits"] << (n & 7)) & 0xFF
    out = struct.pack("<II", c["sym"], 0x40000000 | n) + bytes(bm)
    pad = (-len(out)) & 3
    out += (c["pad"] + b"\0\0\0")[:pad]
    return out + b"".join(c["recs"])


def records_of(c):
    """For each instruction index: its record, or None."""
    it = iter(c["recs"])
    return [None if v else next(it) for v in c["bits"]]


def ldc64(tmpl, d, base, off):
    """The record of LDC.64 Rd, c[0x3][Rbase+off], from the record of another such load."""
    r = bytearray(tmpl)
    w, = struct.unpack_from("<I", r, 8)
    struct.pack_into("<I", r, 8, (w & 0x003FFFFF) | (d << 22))
    w, = struct.unpack_from("<I", r, 12)
    struct.pack_into("<I", r, 12, (w & 0x003FFFFF) | (base << 22))
    struct.pack_into("<i", r, 28, off)
    return bytes(r)


def replace(b, c):
    """The cubin with its capmerc section rebuilt from c. When the size changes, everything after the section in
    the file moves by the difference rounded up to 8 bytes, and every offset that points past it follows."""
    b = bytearray(b)
    new = build(c)
    shoff, es, sh, _names = _sections(b)
    off, size = sh[c["index"]][4], sh[c["index"]][5]
    delta = len(new) - size
    if delta == 0:
        b[off:off + size] = new
        return b
    shift = ((delta + 7) & ~7) if delta > 0 else -((-delta) & ~7)
    end = off + size
    if shift >= 0:
        body = new + bytes(shift - delta)
    else:
        body = new + bytes(size + shift - len(new))
    out = bytearray(b[:off] + body + b[end:])
    assert len(out) == len(b) + shift

    def mv(x):
        return x + shift if x >= end else x

    phoff, = struct.unpack_from("<Q", b, 0x20)
    phes, phnum = struct.unpack_from("<HH", b, 0x36)
    struct.pack_into("<Q", out, 0x20, mv(phoff))
    struct.pack_into("<Q", out, 0x28, mv(shoff))
    for i, s in enumerate(sh):
        at = mv(shoff) + i * es
        if i == c["index"]:
            struct.pack_into("<Q", out, at + 0x20, len(new))
        elif s[4] >= end and s[1] != 0:
            struct.pack_into("<Q", out, at + 0x18, s[4] + shift)
    for i in range(phnum):
        at = mv(phoff) + i * phes
        p_off, = struct.unpack_from("<Q", out, at + 8)
        if p_off >= end:
            struct.pack_into("<Q", out, at + 8, p_off + shift)
    return out
