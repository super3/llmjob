"""cubinstat.py <base.cubin> <arm.cubin>... : the tall fold's machine code in each cubin, without a disassembler.

For each cubin and each of pearl_tile_fold_tall and pearl_tile_fold_tall_pinb with a body: instructions (16 bytes
each), the count of the IMMA opcode, the opcode sequence after the last IMMA (the tile's hash and hand-off) as a
count and an md5, and the opcodes whose counts differ from the first cubin's.

The IMMA opcode is found from the first cubin: the 12-bit opcode (the low 12 bits of an instruction's first word)
that occurs exactly 192 times in the fold, the mma a chunk loop holds (2 stages x 2 k-steps x 6 x 8).
"""
import hashlib
import struct
import sys
from collections import Counter

FOLDS = ("pearl_tile_fold_tall", "pearl_tile_fold_tall_pinb")


def texts(path):
    b = open(path, "rb").read()
    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    ent, num, stri = struct.unpack_from("<HHH", b, 0x3A)
    hs = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * ent) for i in range(num)]
    names = b[hs[stri][4]:hs[stri][4] + hs[stri][5]]
    out = {}
    for h in hs:
        n = names[h[0]:names.index(b"\0", h[0])].decode()
        if n.startswith(".text."):
            out[n[6:]] = b[h[4]:h[4] + h[5]]
    return out


def ops(code):
    return [struct.unpack_from("<Q", code, i)[0] & 0xFFF for i in range(0, len(code), 16)]


def main():
    base = None
    imma = {}
    for path in sys.argv[1:]:
        t = texts(path)
        for f in FOLDS:
            if f not in t or len(t[f]) < 16 * 500:
                continue
            o = ops(t[f])
            c = Counter(o)
            if f not in imma:
                cands = [op for op, n in c.items() if n == 192]
                imma[f] = cands[0] if len(cands) == 1 else None
                if imma[f] is None:
                    print(f"  {f}: no single opcode occurs 192 times: {cands}")
            op = imma.get(f)
            n_imma = c.get(op, 0) if op is not None else -1
            last = max((i for i, x in enumerate(o) if x == op), default=-1)
            first = min((i for i, x in enumerate(o) if x == op), default=-1)
            tail = o[last + 1:]
            loop = o[first:last + 1]
            th = hashlib.md5(struct.pack(f"<{len(tail)}H", *tail)).hexdigest()[:10]
            line = (f"{path.replace(chr(92), '/').split('/')[-2]:>12} {f[16:] or 'tall':>10}: instr {len(o):5d}  imma {n_imma:4d}  "
                    f"first-to-last-imma {len(loop):5d}  after-last-imma {len(tail):5d} md5 {th}")
            if base is None or f not in base:
                base = base or {}
                base[f] = c
            else:
                d = {k: c.get(k, 0) - base[f].get(k, 0) for k in set(c) | set(base[f])}
                d = {k: v for k, v in d.items() if v}
                line += "  diff " + " ".join(f"{k:03x}:{v:+d}" for k, v in sorted(d.items()))
            print(line)


main()
