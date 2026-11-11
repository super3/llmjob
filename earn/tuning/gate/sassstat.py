"""sassstat.py <fn> <base.sass> <arm.sass>... : one line an arm about kernel <fn> in nvdisasm -c output.

instr      instructions in the kernel
imma       IMMA instructions (the chunk loop holds 192: 2 stages x 2 k-steps x 6 m16 x 8 n8)
loop       instructions in the chunk loop: from the label its closing branch jumps to (the first branch after the
           last IMMA to a label before the first IMMA) through that branch
copies     LDGSTS (cp.async) in that loop, and how many of them are predicated
lds, sts   plain shared loads (LDS, not LDSM) and stores in that loop
ldsm       LDSM (ldmatrix) in that loop
hash       instructions from the tile hand-off (the BAR.SYNC after the chunk loop) to the tile loop's closing
           branch, and the md5 of their mnemonics in order: the transcript hash and the hit record, which no arm
           may change
"""
import hashlib
import re
import sys

LINE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?);")
LABEL = re.compile(r"^(\.L_x_\d+):")


def kernel(path, fn):
    """(instructions, labels): instructions as (pred, mnemonic, text); labels as name -> index of next instruction."""
    out, labels, on, regs = [], {}, False, "?"
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("//---") and ".text." in line:
            on = (".text." + fn + " ") in line
            continue
        if not on:
            continue
        m = re.search(r"SHI_REGISTERS=(\d+)", line)
        if m:
            regs = m.group(1)
        m = LABEL.match(line)
        if m:
            labels[m.group(1)] = len(out)
            continue
        m = LINE.match(line)
        if m:
            text = m.group(2).strip()
            pred = ""
            if text.startswith("@"):
                pred, text = text.split(None, 1)
            out.append((pred, text.split()[0], text))
    return out, labels, regs


def target(text):
    m = re.search(r"\.L_x_\d+", text)
    return m.group(0) if m else None


def main():
    fn = sys.argv[1]
    print(f"{'arm':>8} {'regs':>4} {'instr':>6} {'imma':>5} {'loop':>5} {'copies':>9} {'lds':>4} {'sts':>4} {'ldsm':>4} "
          f"{'hash':>5} {'hash md5':>10}")
    for path in sys.argv[2:]:
        k, labels, regs = kernel(path, fn)
        name = path.replace("\\", "/").split("/")[-2]
        if not k:
            print(f"{name:>8} no {fn}")
            continue
        imma = [i for i, x in enumerate(k) if x[1].startswith("IMMA")]
        if not imma:
            print(f"{name:>8} {regs:>4} {len(k):6d} no IMMA")
            continue
        first, last = imma[0], imma[-1]
        close = head = None
        for i in range(last, len(k)):
            if k[i][1].startswith("BRA"):
                t = labels.get(target(k[i][2]) or "")
                if t is not None and t <= first:
                    close, head = i, t
                    break
        loop = k[head:close + 1] if close is not None else k[first:last + 1]
        hand = next((i for i in range(close or last, len(k)) if k[i][1].startswith("BAR.SYNC")), None)
        tend = None
        if hand is not None:
            for i in range(hand, len(k)):
                if k[i][1].startswith("BRA") and not k[i][0]:
                    t = labels.get(target(k[i][2]) or "")
                    if t is not None and t <= (head or first):
                        tend = i
                        break
        tail = k[hand:tend + 1] if hand is not None and tend is not None else []
        seq = " ".join(x[0] + x[1] for x in tail)
        cps = [x for x in loop if x[1].startswith("LDGSTS")]
        print(f"{name:>8} {regs:>4} {len(k):6d} {len(imma):5d} {len(loop):5d} "
              f"{len(cps):4d}/{sum(1 for x in cps if x[0]):<4d} "
              f"{sum(1 for x in loop if x[1] == 'LDS' or x[1].startswith('LDS.')):4d} "
              f"{sum(1 for x in loop if x[1].startswith('STS')):4d} "
              f"{sum(1 for x in loop if x[1].startswith('LDSM')):4d} "
              f"{len(tail):5d} {hashlib.md5(seq.encode()).hexdigest()[:10]:>10}")


main()
