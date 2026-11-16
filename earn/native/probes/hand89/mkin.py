"""mkin.py <pattern> <out> [rowsA colsB k]: a PD_INPUT file for gprobe (hand89/gprobe.cu): 'GPIN', rows of A, columns of B,
k; the rows' dense factors plus 32 (128 bytes each, 0..63), the columns', then A plus, A minus, B plus, B minus
(k bytes each). Random as gprobe's own draw, then shaped by the pattern, to price what operand toggling costs in
the tensor unit. Every pattern is real input (the CPU check still holds); only how alike the operands are changes.
  rand    nothing shaped
  bcst    every column's factors 32: B's noise 0, B' = 63 everywhere
  acst    the same for A's rows
  b8      columns equal in groups of 8 (c -> c & ~7): an n8 fragment's eight columns are one column
  b2      columns equal in pairs (c -> c & ~1)
  bt      a column slot's eight n8 tiles equal (c -> (c & ~63) | (c & 7)): consecutive IMMA see the same B
  a8      rows equal in groups of 8 (r -> r & ~7): an m16 fragment's two halves each one row
  at      a row band's six m16 tiles equal (r -> 96 (r // 96) + r % 16): consecutive A tiles the same
  kc      every k has the same rank indices (B' and A' constant along k, random across rows and columns)
  k2      k in pairs share their indices (kk -> kk & ~1)
  bhalf   B's factors drawn from 0..31 + 16 (noise -31..31, half the spread)
"""
import random
import struct
import sys

pat, out = sys.argv[1], sys.argv[2]
rowsA = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
colsB = int(sys.argv[4]) if len(sys.argv) > 4 else 262144
k = int(sys.argv[5]) if len(sys.argv) > 5 else 65536
rng = random.Random(12345)
DA = bytearray(rng.getrandbits(8) & 63 for _ in range(rowsA * 128))
DB = bytearray(rng.randbytes(colsB * 128).translate(bytes(b & 63 for b in range(256))))
pm = []
for op in range(2):
    p0s, p1s = bytearray(k), bytearray(k)
    for kk in range(k):
        u = rng.getrandbits(24)
        p0 = u & 127
        p1 = p0 ^ (1 + ((u >> 8) % 127))
        p0s[kk], p1s[kk] = p0, p1
    pm.append((p0s, p1s))


def remap(D, n, f):
    E = bytearray(len(D))
    for c in range(n):
        s = f(c)
        E[c * 128:(c + 1) * 128] = D[s * 128:(s + 1) * 128]
    return E


if pat == "bcst":
    DB = bytearray([32]) * len(DB)
elif pat == "acst":
    DA = bytearray([32]) * len(DA)
elif pat == "b8":
    DB = remap(DB, colsB, lambda c: c & ~7)
elif pat == "b2":
    DB = remap(DB, colsB, lambda c: c & ~1)
elif pat == "bt":
    DB = remap(DB, colsB, lambda c: (c & ~63) | (c & 7))
elif pat == "a8":
    DA = remap(DA, rowsA, lambda r: r & ~7)
elif pat == "at":
    DA = remap(DA, rowsA, lambda r: 96 * (r // 96) + r % 16)
elif pat == "kc":
    for op in range(2):
        p0s, p1s = pm[op]
        for kk in range(k):
            p0s[kk], p1s[kk] = p0s[0], p1s[0]
elif pat == "k2":
    for op in range(2):
        p0s, p1s = pm[op]
        for kk in range(k):
            p0s[kk], p1s[kk] = p0s[kk & ~1], p1s[kk & ~1]
elif pat == "bhalf":
    DB = bytearray(bytes(DB).translate(bytes(16 + (b & 31) for b in range(256))))
elif pat != "rand":
    sys.exit("unknown pattern " + pat)
with open(out, "wb") as f:
    f.write(struct.pack("<4I", 0x4E495047, rowsA, colsB, k))
    f.write(DA)
    f.write(DB)
    for op in range(2):
        f.write(pm[op][0])
        f.write(pm[op][1])
print(out, rowsA, colsB, k, pat)
