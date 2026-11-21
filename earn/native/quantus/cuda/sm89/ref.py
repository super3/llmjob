"""ref.py: the partial rounds qc_hand_loop computes, in Python (mod p), with constants read from qc_kernel.cuh."""
import os
import re

P = (1 << 64) - (1 << 32) + 1
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qc_kernel.cuh')
_t = open(SRC, encoding='utf-8').read()
D = [int(re.search(r'#define QC_D%d (0x[0-9a-f]+)ull' % i, _t)[1], 16) for i in range(12)]
DP = [int(re.search(r'#define QC_DP%d (0x[0-9a-f]+)ull' % i, _t)[1], 16) for i in range(12)]
assert all(DP[i] == D[i] * (1 << 32) % P for i in range(12))


def _arr(name):
    m = re.search(r'__constant__ u64 ' + re.escape(name) + r'\[[^=]*= \{(.*?)\};', _t, re.S)
    body = re.sub(r'ull', '', m[1])
    return [int(v, 0) for v in re.findall(r'0x[0-9a-f]+|\d+', body)]


RC_P = _arr('QC_RC_P')
assert len(RC_P) == 21, len(RC_P)
RC_T = _arr('QC_RC_T')
RC_T0 = RC_T[:12]


def sbox(x):
    return pow(x, 7, P)


def partial(s, x, rc0):
    tot = (x + sum(s[1:])) % P
    out0 = (D[0] * x + tot + rc0) % P
    for i in range(1, 12):
        s[i] = (D[i] * s[i] + tot) % P
    return out0


def hand_loop(s):
    s = list(s)
    x = sbox(s[0] % P)
    for r in range(21):
        x = sbox(partial(s, x, RC_P[r]))
    tot = (x + sum(s[1:])) % P
    out = [(D[0] * x + tot + RC_T0[0]) % P]
    for i in range(1, 12):
        out.append((D[i] * s[i] + tot + RC_T0[i]) % P)
    return out
