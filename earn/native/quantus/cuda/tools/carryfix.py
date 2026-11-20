"""carryfix.py: make QC_RED 3's two-instruction reduction tail right in an sm_86 cubin by inverting one carry-in.

  python carryfix.py <in.cubin> <out.cubin> [--nvdisasm PATH] [--kernel NAME]

The cubin is one built with -DQC_RED=3 (and, on sm_86, -DQC_KZ=1); tools/qcdrv.cu checks its hits against the
kernel as compiled and times it. Not used by the build: kept as the tool behind the 20 Nov RTX 3070 entry.

QC_RED 3 compiles to
    IMAD.WIDE.U32 Re, Pc, Rr2, -0x1, Rlo      E = r2 * EPS + (r1:r0), carry c
    IADD3.X Rl, Pb, ~Rr3, Re, RZ, Pc, !PT     lo = E0 + ~r3 + c       (wanted: E0 + ~r3 + !c = E0 - r3 - c)
    IADD3.X Rh, RZ, -0x1, Re+1, Pb, Pc        hi = E1 - 1 + b + c
PTX cannot say "carry-in inverted", so ptxas's lo is off by one whenever c differs from what the borrow needs.
Here every IADD3.X of the second shape whose first carry-in is the carry-out of the nearest earlier IMAD.WIDE.U32 by
-0x1 (EPS), with that predicate not written in between, gets that carry-in inverted (instruction bit 90). The
result is disassembled again and each changed line must read "!Pc" and differ in nothing else.
"""
import argparse
import re
import struct
import subprocess
import sys

LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\* 0x([0-9a-f]{16}) \*/")
HIGH = re.compile(r"^\s+/\* 0x([0-9a-f]{16}) \*/")
NEG1 = 1 << (90 - 64)


def listing(nvdisasm, cubin, kernel):
    out = subprocess.run([nvdisasm, "-hex", "-c", cubin], capture_output=True, text=True).stdout.splitlines()
    on, ins = False, []
    for i, ln in enumerate(out):
        if ln.startswith("//----") and ".text." in ln:
            on = (".text." + kernel) in ln
            continue
        m = LINE.match(ln)
        if on and m:
            h = HIGH.match(out[i + 1])
            ins.append({"addr": int(m[1], 16), "text": re.sub(r"\s+", " ", m[2].strip()), "lo": int(m[3], 16),
                        "hi": int(h[1], 16)})
    return ins


def text_section(b, kernel):
    shoff, = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    def sh(i):
        return struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize)
    stroff = sh(shstrndx)[4]
    for i in range(shnum):
        s = sh(i)
        name = b[stroff + s[0]:b.index(b"\0", stroff + s[0])].decode()
        if name == ".text." + kernel:
            return s[4], s[5]
    sys.exit("no .text." + kernel)


TAIL = re.compile(r"(@!?P\d )?IADD3\.X R\d+, P\d, (?:~R\d+(?:\.reuse)?, R\d+(?:\.reuse)?|R\d+(?:\.reuse)?, ~R\d+(?:\.reuse)?), RZ, (P[0-6]), !PT")
WIDE = re.compile(r"(@!?P\d )?IMAD\.WIDE\.U32 R\d+, (P[0-6]), R\d+(\.reuse)?, -0x1, R\d+")


def writes_pred(text, p):
    """True when the instruction may write predicate p (any P operand before the first source register)."""
    t = re.sub(r"^@!?P\d ", "", text)
    op, _, rest = t.partition(" ")
    if op.startswith(("ISETP", "PLOP3", "FSETP", "R2P", "LOP3", "PSETP")) and p in rest:
        return True
    args = [x.strip() for x in rest.split(",")]
    # carry-outs come right after the destination register
    for a in args[1:3]:
        if a == p:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--nvdisasm", default="nvdisasm")
    ap.add_argument("--kernel", default="_Z14qc_scan_kernel8QcLaunchPyS0_")
    a = ap.parse_args()
    ins = listing(a.nvdisasm, a.inp, a.kernel)
    if not ins:
        sys.exit("kernel not found in the listing")
    b = bytearray(open(a.inp, "rb").read())
    off, size = text_section(b, a.kernel)
    base = ins[0]["addr"]
    fixes = {}
    for i, x in enumerate(ins):
        m = TAIL.fullmatch(x["text"])
        if not m:
            continue
        p = m[2]
        ok = False
        for j in range(i - 1, max(-1, i - 200), -1):
            w = WIDE.fullmatch(ins[j]["text"])
            if w and w[2] == p:
                ok = True
                break
            if writes_pred(ins[j]["text"], p) or ins[j]["text"].startswith(("BRA", "CALL", "BSYNC", "EXIT")):
                break
        if not ok:
            continue
        hi = x["hi"] ^ NEG1
        struct.pack_into("<QQ", b, off + x["addr"] - base, x["lo"], hi)
        fixes[x["addr"]] = x["text"].replace(", " + p + ", !PT", ", !" + p + ", !PT")
    open(a.out, "wb").write(b)
    after = {x["addr"]: x["text"] for x in listing(a.nvdisasm, a.out, a.kernel)}
    before = {x["addr"]: x["text"] for x in ins}
    for k in before:
        want = fixes.get(k, before[k])
        if after.get(k) != want:
            sys.exit("mismatch at %04x: %r, wanted %r" % (k, after.get(k), want))
    print("%d tails fixed, %d instructions checked" % (len(fixes), len(before)))


main()
