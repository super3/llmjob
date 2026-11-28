"""probe_ref.py [probe ...]: what "qcdrv <cubin> probe" must print for a kernel whose hand function is full120.py's
probe N (0, the default: the whole body, which is also what the compiled kernel prints).

The arithmetic of the scan kernel around the function, for qcdrv's fixed line: the first round's two S-boxes and
layer (qc_first_round), the function, element 0 of the last layer (qc_ext0), and the hit test on the hash's first
eight bytes. 256 nonces, about half of them hits, printed as 64 hex digits, nonce 0 first.
"""
import sys

import full120
import ref

P = ref.P


def bitmap(probe):
    c = [(i + 1) * 0x0123456789abcdef % P for i in range(12)]
    x0, x4, target = 0x1111111111111111, 0x2222222222222222, 1 << 63
    bits = bytearray(32)
    n = 0
    for i in range(256):
        d = 35 * i
        a, b = ref.sbox((x0 + d) % P), ref.sbox((x4 - d) % P)
        s = []
        for g in range(3):
            v = a * (2 if g == 0 else 1) + b * (2 if g == 1 else 1)
            s += [(c[4 * g + k] + m * v) % P for k, m in enumerate((2, 1, 1, 3))]
        s = full120.probe_ref(probe, s)
        m = [2 * s[4 * g] + 3 * s[4 * g + 1] + s[4 * g + 2] + s[4 * g + 3] for g in range(3)]
        h = (2 * m[0] + m[1] + m[2]) % P
        if int.from_bytes(h.to_bytes(8, "little"), "big") <= target:
            bits[i >> 3] |= 0x80 >> (i & 7)
            n += 1
    return "probe %d hits %s" % (n, bits.hex())


if __name__ == "__main__":
    for a in sys.argv[1:] or ["0"]:
        print(a, bitmap(int(a)))
