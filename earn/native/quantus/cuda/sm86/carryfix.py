"""carryfix.py: make QC_RED 3's reduction tail right in an sm_86 cubin by inverting one carry-in per tail.

  python carryfix.py <in.cubin> <out.cubin> [--nvdisasm PATH]

The cubin is the scan kernel built for sm_86 with -DQC_RED=3 (build.py in this folder does that and pins the result's
hash). Every tail it changes is checked again through nvdisasm; any doubt and it stops without writing.

What the tail must compute. After the 128-bit product (r3 r2 r1 r0), qc_reduce (qc_field.cuh) forms
    E = r2 * EPS + (r1:r0)  (mod 2^64)   with c its carry out of 2^64, and then
    lo = E0 - r3 - c        (mod 2^32)   with nb = [E0 >= r3 + c] (no borrow),
    hi = E1 + c - 1 + nb    (mod 2^32),
that is (hi:lo) = E + c * 2^32 - (r3 + c) (mod 2^64), which is E + c * EPS - r3, the reduction. QC_RED 2 does it in
four instructions: IMAD.WIDE, an IMAD.X for r3 + c, an IADD3 and an IADD3.X. QC_RED 3 writes it so that ptxas makes
three, but PTX's subc reads the carry flag as a borrow (SASS adds the carry-in to a + ~b), so what compiles is
    IMAD.WIDE.U32 Re, Pc, Rr2, -0x1, Rlo       E, c = Pc
    IADD3.X Rl, Pb, Re, ~Rr3, RZ, Pc, !PT      lo' = E0 + ~r3 + c = E0 - r3 - 1 + c, Pb its carry out
    IADD3.X Rh, RZ, -0x1, Re+1, Pb, Pc         hi = E1 - 1 + Pb + c
(the two operands of the first IADD3.X come in either order). Here the first IADD3.X's carry-in becomes !Pc, bit 90
of the instruction. Why that is exact for every input:
  - The values: E0 and r3 are any 32-bit words, c is 0 or 1. ~r3 = 2^32 - 1 - r3 and !c = 1 - c, so the
    instruction adds E0 + (2^32 - 1 - r3) + (1 - c) = E0 - r3 - c + 2^32. That sum lies in [0, 2^33 - 2]: one carry
    bit holds all of it, so lo = (E0 - r3 - c) mod 2^32 and Pb = [E0 - r3 - c >= 0] = nb, for every E0, r3 and c.
  - hi = E1 - 1 + nb + c is then exactly the line above; it reads Pc unchanged.
  - Nothing else reads the changed instruction differently: Pc is still written only by the IMAD.WIDE and still read
    as it was by the second IADD3.X; the first IADD3.X writes Rl and Pb as before, now with the right values.
So the patched tail gives (hi:lo) = E + c * EPS - r3 (mod 2^64) for all inputs. QC_RED 2 gives the same except when
r3 = 2^32 - 1 and c = 1, where its r3 + c wraps to 0 and it is 2^32 off; the patched tail is right there too. The
64-bit wrap of E + c * EPS - r3 itself is the same slip as QC_RED 2's (qc_field.cuh). The diagonal multiplies' tails
(qc_dmul with QC_RED 3) have the same shape: with cy, a 0/1 carry, in r3's place, and where no cy can arise, with
r3 = 0, which ptxas writes as the immediate -0x1 (~0 = 2^32 - 1) in place of ~Rr3:
    IADD3.X Rl, Pb, Re, -0x1, RZ, Pc, !PT
The argument holds as it is for r3 = 0. test/quantusCarryfix.test.js models the tails word for word over the edge
cases and random inputs, and pins the PTX they are compiled from.

Which instructions change: an IADD3.X of the shapes above whose first carry-in Pc is the carry out of the nearest
earlier IMAD.WIDE.U32 by -0x1 (EPS) in the same basic block (no label, branch or other writer of Pc in between),
and whose next reader of Pc is the tail's high word, IADD3.X Rh, RZ, -0x1, R, Pb, Pc. Every IMAD.WIDE by EPS whose
carry's first reader has a tail's shape must be one of them, or the script stops.
"""
import argparse
import re
import struct
import subprocess
import sys

LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\* 0x([0-9a-f]{16}) \*/")
HIGH = re.compile(r"^\s+/\* 0x([0-9a-f]{16}) \*/")
LABEL = re.compile(r"^\.L_x_\d+:")
NEG1 = 1 << (90 - 64)   # the first carry-in's negation, in the instruction's high word

R = r"R\d+(?:\.reuse)?"
TAIL = re.compile(r"(?:@!?P\d )?IADD3\.X R\d+, (P[0-6]), (?:~" + R + ", " + R + "|" + R + ", ~" + R + "|" + R
                  + r", -0x1), RZ, (P[0-6]), !PT")
HIGHW = re.compile(r"(?:@!?P\d )?IADD3\.X R\d+, RZ, -0x1, " + R + r", (P[0-6]), (P[0-6])")
WIDE = re.compile(r"(?:@!?P\d )?IMAD\.WIDE\.U32 R\d+, (P[0-6]), " + R + r", -0x1, R\d+")
STOP = ("BRA", "CALL", "BSYNC", "BSSY", "EXIT", "RET", "JMP", "WARPSYNC", "BREAK")


def kernel_name(cubin):
    b = open(cubin, "rb").read()
    names = sorted(set(re.findall(rb"\.text\.(_Z14qc_scan_kernel[A-Za-z0-9_]+)", b)))
    if len(names) != 1:
        sys.exit("expected one scan kernel in %s, found %r" % (cubin, names))
    return names[0].decode()


def listing(nvdisasm, cubin, kernel):
    r = subprocess.run([nvdisasm, "-hex", "-c", cubin], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("nvdisasm failed: " + r.stderr.strip()[:300])
    out = r.stdout.splitlines()
    on, ins, label = False, [], False
    for i, ln in enumerate(out):
        if ln.startswith("//----") and ".text." in ln:
            on = (".text." + kernel) in ln
            continue
        if on and LABEL.match(ln.strip()):
            label = True
            continue
        m = LINE.match(ln)
        if on and m:
            h = HIGH.match(out[i + 1])
            ins.append({"addr": int(m[1], 16), "text": re.sub(r"\s+", " ", m[2].strip()), "lo": int(m[3], 16),
                        "hi": int(h[1], 16), "label": label})
            label = False
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


def reads(text, p):
    """True when predicate p appears anywhere in the instruction (as a guard, carry-in or source)."""
    return re.search(r"(?<![A-Z0-9])!?" + p + r"(?![0-9])", text) is not None


def writes(text, p):
    """True when the instruction may write predicate p: a carry out right after the destination, or a setp."""
    t = re.sub(r"^@!?P\d ", "", text)
    op, _, rest = t.partition(" ")
    if op.split(".")[0] in ("ISETP", "PLOP3", "FSETP", "DSETP", "R2P", "PSETP", "LOP3", "VOTE") and p in rest:
        return True
    args = [x.strip() for x in rest.split(",")]
    return p in args[1:3]


def fix(ins):
    """{address: new text} for every tail, or exits."""
    fixes = {}
    for i, x in enumerate(ins):
        m = TAIL.fullmatch(x["text"])
        if not m:
            continue
        pb, pc = m[1], m[2]
        src = None
        for j in range(i, 0, -1):
            if ins[j]["label"]:
                break
            t = ins[j - 1]["text"]
            w = WIDE.fullmatch(t)
            if w and w[1] == pc:
                src = j - 1
                break
            if writes(t, pc) or t.split(" ")[0] in STOP:
                break
        if src is None:
            continue
        nxt = None
        for k in range(i + 1, len(ins)):
            if ins[k]["label"] or ins[k]["text"].split(" ")[0] in STOP:
                break
            if reads(ins[k]["text"], pc):
                nxt = k
                break
        h = HIGHW.fullmatch(ins[nxt]["text"]) if nxt is not None else None
        if not h or h[1] != pb or h[2] != pc:
            sys.exit("tail at %04x: the next reader of %s is not the high word: %r"
                     % (x["addr"], pc, ins[nxt]["text"] if nxt is not None else None))
        fixes[x["addr"]] = x["text"].replace(", " + pc + ", !PT", ", !" + pc + ", !PT")
    for i, x in enumerate(ins):
        w = WIDE.fullmatch(x["text"])
        if not w:
            continue
        for k in range(i + 1, len(ins)):
            t = ins[k]["text"]
            if reads(t, w[1]):
                if TAIL.fullmatch(t) and ins[k]["addr"] not in fixes:
                    sys.exit("an IMAD.WIDE by EPS at %04x feeds a tail at %04x that was not matched"
                             % (x["addr"], ins[k]["addr"]))
                break
            if writes(t, w[1]) or ins[k]["label"]:
                break
    return fixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--nvdisasm", default="nvdisasm")
    a = ap.parse_args()
    kernel = kernel_name(a.inp)
    ins = listing(a.nvdisasm, a.inp, kernel)
    if not ins:
        sys.exit("nvdisasm gave no listing of " + kernel)
    fixes = fix(ins)
    if not fixes:
        sys.exit("no QC_RED 3 tail found: was the cubin built with -DQC_RED=3?")
    b = bytearray(open(a.inp, "rb").read())
    off, size = text_section(b, kernel)
    base = ins[0]["addr"]
    for x in ins:
        if x["addr"] in fixes:
            struct.pack_into("<QQ", b, off + x["addr"] - base, x["lo"], x["hi"] ^ NEG1)
    open(a.out, "wb").write(b)
    after = {x["addr"]: x["text"] for x in listing(a.nvdisasm, a.out, kernel)}
    for x in ins:
        want = fixes.get(x["addr"], x["text"])
        if after.get(x["addr"]) != want:
            sys.exit("mismatch at %04x: %r, wanted %r" % (x["addr"], after.get(x["addr"]), want))
    print("%d tails fixed in %s, %d instructions read back" % (len(fixes), kernel, len(ins)))
    return len(fixes)


if __name__ == "__main__":
    main()
