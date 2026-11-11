"""loopops.py <fn> <a.sass> [<b.sass> ...] : mnemonic counts inside the chunk loop of kernel <fn>, a line a file.

The loop is sassstat.py's: from the label the first backward branch after the last IMMA jumps to, through that
branch. Counts are static (instructions in the span, taken or not). Also: local-stack loads and stores (spill) in
the loop and outside it, and BAR instructions in the whole kernel.
"""
import collections
import re
import sys

LINE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?);")
LABEL = re.compile(r"^(\.L_x_\d+):")


def kernel(path, fn):
    out, labels, on = [], {}, False
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("//---") and ".text." in line:
            on = (".text." + fn + " ") in line
            continue
        if not on:
            continue
        m = LABEL.match(line)
        if m:
            labels[m.group(1)] = len(out)
            continue
        m = LINE.match(line)
        if m:
            t = m.group(2).strip()
            if t.startswith("@"):
                t = t.split(None, 1)[1]
            out.append((t.split()[0], t))
    return out, labels


fn = sys.argv[1]
keys = ["IMMA", "LDSM", "LOP3", "ATOMS", "SHF", "LDS", "STS", "LDL", "STL", "BAR", "MEMBAR", "BRA", "MOV", "IADD3", "IADD",
        "ISETP", "LDG", "STG", "NOP"]
for path in sys.argv[2:]:
    k, labels = kernel(path, fn)
    imma = [i for i, x in enumerate(k) if x[0].startswith("IMMA")]
    first, last = imma[0], imma[-1]
    close = head = None
    for i in range(last, len(k)):
        if k[i][0].startswith("BRA"):
            m = re.search(r"\.L_x_\d+", k[i][1])
            t = labels.get(m.group(0)) if m else None
            if t is not None and t <= first:
                close, head = i, t
                break
    loop = k[head:close + 1]
    c = collections.Counter(x[0].split(".")[0] for x in loop)
    allc = collections.Counter(x[0].split(".")[0] for x in k)
    name = path.replace("\\", "/").split("/")[-2]
    other = sum(v for kk, v in c.items() if kk not in keys)
    print(f"{name:>7} loop {len(loop):4d}: " + " ".join(f"{x}={c[x]}" for x in keys if c[x]) + f" other={other}"
          + f" | whole kernel {len(k)}: LDL={allc['LDL']} STL={allc['STL']} BAR={allc['BAR']} IMMA={allc['IMMA']}")
