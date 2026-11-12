"""sm75.py: shared helpers for goal-nv20g (sm_75 control words, instruction parsing, dependences)."""
import re
import struct
import subprocess

S = r"C:\Users\template\AppData\Local\Temp\claude\18634d96a113c206\5675ce09-95f3-4f49-ac47-74a8f5730983\scratchpad"
DIS = S + r"\gate\dl\nvd\cuda_nvdisasm-windows-x86_64-12.8.90-archive\bin\nvdisasm.exe"


def sections(data):
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3a)
    hdrs = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize) for i in range(shnum)]
    stro = hdrs[shstrndx][4]
    return {data[stro + h[0]:data.index(b'\0', stro + h[0])].decode(): (h[4], h[5]) for h in hdrs}


def getc(w):
    return (int.from_bytes(w[8:16], 'little') >> 41) & ((1 << 23) - 1)


def setc(w, c):
    hi = int.from_bytes(w[8:16], 'little')
    hi = (hi & ((1 << 41) - 1)) | (c << 41)
    return bytes(w[:8]) + hi.to_bytes(8, 'little')


def fields(c):
    return dict(stall=c & 15, y=(c >> 4) & 1, w=(c >> 5) & 7, r=(c >> 8) & 7, wait=(c >> 11) & 63,
                reuse=(c >> 17) & 15)


def mk(f):
    return f['stall'] | f['y'] << 4 | f['w'] << 5 | f['r'] << 8 | f['wait'] << 11 | f['reuse'] << 17


REG = re.compile(r'(?<![\w.])(U?R\d+|U?P\d)(?![\w])')


def regs_of(tok, width=1):
    out = []
    for r in REG.findall(tok):
        if r[0] in 'RU' and 'P' not in r[:2]:
            pre = 'UR' if r.startswith('UR') else 'R'
            n = int(r[len(pre):])
            out += [pre + str(n + k) for k in range(width)]
        else:
            out.append(r)
    return out


VAR = ('LDG', 'LDS', 'SHFL', 'STS', 'STG', 'IMMA', 'MUFU', 'I2F', 'F2I', 'S2R', 'LDSM', 'BAR', 'ATOMG', 'ATOMS',
       'RED', 'LDL', 'STL', 'POPC', 'FLO', 'BREV', 'I2F', 'CS2R', 'S2UR')


def parse(text):
    """op, defs, uses for one instruction's text (nvdisasm form)."""
    t = text.strip().rstrip(';').strip()
    guard = []
    m = re.match(r'@(!?)(U?P\d|PT)\s+', t)
    if m:
        if m.group(2) != 'PT':
            guard = [m.group(2)]
        t = t[m.end():]
    op, _, rest = t.partition(' ')
    ops = [o.strip() for o in rest.split(',')] if rest else []
    base = op.split('.')[0]
    defs, uses = [], list(guard)
    wid = lambda o: 4 if '.128' in op else (2 if '.64' in op else 1)
    if base in ('STS', 'STG', 'BRA', 'BAR', 'EXIT', 'NOP', 'BSSY', 'BSYNC', 'RED'):
        for o in ops:
            uses += regs_of(o, 1 if o.startswith('[') else wid(o))
    elif base == 'IMMA':
        defs = regs_of(ops[0], 2)
        uses += regs_of(ops[1]) + regs_of(ops[2]) + regs_of(ops[3], 2)
    elif base in ('LDG', 'LDS', 'LDSM'):
        defs = regs_of(ops[0], wid(ops[0]))
        uses += regs_of(ops[1])
    elif base == 'ULDC':
        defs = regs_of(ops[0], 2 if '.64' in op else 1)
    elif base in ('SHFL', 'ISETP') or (base in ('LOP3', 'IADD3') and ops and re.match(r'!?U?P', ops[0])):
        # leading predicate destinations
        k = 0
        while k < len(ops) and re.match(r'^!?(U?P\d|PT)$', ops[k]):
            defs += regs_of(ops[k]); k += 1
        if base == 'SHFL':
            defs += regs_of(ops[k]); k += 1
        elif base in ('LOP3', 'IADD3') and k < len(ops) and re.match(r'^U?R', ops[k]):
            pass
        for o in ops[k:]:
            uses += regs_of(o)
    elif base == 'ATOMG':
        k = 0
        while k < len(ops) and re.match(r'^!?(U?P\d|PT)$', ops[k]):
            defs += regs_of(ops[k]); k += 1
        defs += regs_of(ops[k]); k += 1
        for o in ops[k:]:
            uses += regs_of(o)
    else:
        if ops:
            k = 0
            # destinations: a register, then optional predicate destinations (e.g. IADD3 R1, P0, ...)
            defs = regs_of(ops[0]); k = 1
            while k < len(ops) and re.match(r'^(U?P\d|PT)$', ops[k]) and base in ('IADD3', 'LEA', 'IMAD', 'LOP3'):
                defs += regs_of(ops[k]); k += 1
            for o in ops[k:]:
                uses += regs_of(o)
    if '.WIDE' in op and defs and defs[0][0] == 'R':
        n = int(defs[0][1:]); defs = ['R%d' % n, 'R%d' % (n + 1)] + defs[1:]
    defs = [d for d in defs if d not in ('RZ', 'PT', 'URZ', 'UPT')]
    uses = [u for u in uses if u not in ('RZ', 'PT', 'URZ', 'UPT')]
    if guard:
        uses += defs
    return op, base, set(defs), set(uses), base in VAR


def disasm(cub, sec):
    data = open(cub, 'rb').read()
    off, size = sections(data)['.text.' + sec]
    code = data[off:off + size]
    full = subprocess.run([DIS, '-c', cub], capture_output=True, text=True).stdout
    txt = full.split('.text.' + sec + ':', 1)[1].split('.section', 1)[0]
    texts, labels = {}, {}
    pend = []
    for line in txt.splitlines():
        m = re.match(r'^(\.L_x_\d+):', line.strip())
        if m:
            pend.append(m.group(1)); continue
        m = re.match(r'\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;', line)
        if m:
            a = int(m.group(1), 16)
            texts[a] = m.group(2)
            for p in pend:
                labels[p] = a
            pend = []
    recs = []
    for a in range(0, size, 16):
        w = code[a:a + 16]
        t = texts.get(a, 'NOP')
        op, base, d, u, var = parse(t)
        recs.append(dict(a=a, word=bytes(w), f=fields(getc(w)), text=t, op=op, base=base, defs=d, uses=u, var=var))
    return data, off, size, recs, labels
