"""bank-aware operand reuse for sm_75: the reuse cache holds one entry per (operand slot, register bank), bank = reg % 2.
An IMMA's flag on slot s (a=0, b=1) is set when the next instruction that reads slot s with a register of the same
bank reads the same register, and nothing in between writes it. Only IMMAs evict (ptxas keeps flags across
non-IMMA reads of the slot and bank)."""
import re


def slots(x):
    """register in each source slot (0, 1, 2) of x, or None."""
    t = re.sub(r'^@!?P\w+\s+', '', x['text'].strip())
    parts = t.split(None, 1)
    if len(parts) < 2:
        return [None, None, None]
    ops = [o.strip() for o in parts[1].split(',')]
    src = ops
    if ops and re.match(r'^(R\d+|P\d|PT|RZ)\b', ops[0]) and x['base'] not in ('STS', 'STG', 'ST', 'BRA', 'RED', 'ATOM'):
        src = ops[1:]
        # predicate-writing compares: ISETP P0, PT, R1, R2 -> skip the second predicate too
        while src and re.match(r'^!?(P\d|PT)$', src[0]):
            src = src[1:]
    out = []
    for o in src:
        m = re.search(r'\bR(\d+)\b', o)
        out.append(int(m.group(1)) if m else None)
    return (out + [None, None, None])[:3]


def regnums(s):
    return {int(r[1:]) for r in s if re.fullmatch(r'R\d+', r)}


def reuse_bank(seq, imma_only=True):
    n = len(seq)
    sl = [slots(x) for x in seq]
    out = []
    for p, x in enumerate(seq):
        f = dict(x['f'])
        keep = 0
        if x['base'] == 'IMMA' or not imma_only:
            for s in (0, 1):
                r = sl[p][s]
                if r is None or r == 255 or r in regnums(x['defs']):
                    continue
                written = False
                for q in range(p + 1, n):
                    if r in regnums(seq[q - 1]['defs']) and q - 1 > p:
                        written = True
                        break
                    rq = sl[q][s] if seq[q]['base'] == 'IMMA' else None
                    if rq is not None and rq % 2 == r % 2:
                        if rq == r and not (q - 1 > p and r in regnums(seq[q - 1]['defs'])):
                            keep |= 1 << s
                        break
                    if r in regnums(seq[q]['defs']):
                        written = True
                        break
        f['reuse'] = keep
        out.append(dict(x, f=f))
    return out
