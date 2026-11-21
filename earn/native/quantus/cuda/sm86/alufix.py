"""alufix.py: move plain moves and carry adds off the multiplier pipe in an sm_86 cubin.

  python alufix.py <in.cubin> <out.cubin> <mask> [--nvdisasm PATH]

ptxas writes many moves and carry adds as IMAD forms that multiply by zero (IMAD.MOV.U32 Rd, RZ, RZ, Rc and
IMAD.X Rd, RZ, RZ, Rc, Pc). They run on the multiplier pipe (fmaheavy), which on RTX 30 also runs every IMAD.WIDE
of the field multiplies. Each has an ALU form that computes the same thing bit for bit:

  mask 1   IMAD.MOV.U32 Rd, RZ, RZ, Rc    -> MOV Rd, Rc                       (Rd = Rc)
  mask 2   IMAD.X Rd, RZ, RZ, Rc, [!]Pc   -> IADD3.X Rd, RZ, RZ, Rc, [!]Pc, !PT  (Rd = Rc + carry, no carry out)

The guard predicate and the control field (stalls, barriers) are kept; the operand reuse bits are cleared (a reuse
bit only lets the next instruction read a cached operand, so clearing it changes timing, never values). An ALU result
is ready no later than an IMAD's, so the stalls ptxas set still cover every reader. Every changed instruction is read
back through nvdisasm and compared with the text asked for; any mismatch and nothing is written.
"""
import argparse
import re
import struct
import sys

import carryfix

CTRL_SHIFT = 41                       # the control field, in the instruction's high word
REUSE = 0xF << (CTRL_SHIFT + 17)      # its four operand reuse bits
CTRL = 0x1FFFFF << CTRL_SHIFT
GUARD = 0xF000                        # predicate and its negation, in the low word

R = r"(R\d+|RZ)(?:\.reuse)?"
MOVF = re.compile(r"(@!?P\d )?IMAD\.MOV\.U32 " + R + r", RZ, RZ, " + R)
XF = re.compile(r"(@!?P\d )?IMAD\.X " + R + r", RZ, RZ, " + R + r", (!?)P([0-6])")


def reg(s):
    return 255 if s == "RZ" else int(s[1:])


def plan(ins, mask):
    """{address: (lo, hi, text)} for every instruction to change."""
    out = {}
    for x in ins:
        t, lo, hi = x["text"], x["lo"], x["hi"]
        ctrl = hi & CTRL & ~REUSE
        g = lo & GUARD
        if mask & 1 and (lo & 0xFFF) == 0x224:
            m = MOVF.fullmatch(t)
            if m:
                rd, rc = reg(m[2]), reg(m[3])
                out[x["addr"]] = (0x202 | g | rd << 16 | rc << 32, ctrl | 0xF00,
                                  "%sMOV %s, %s" % (m[1] or "", m[2], m[3]))
                continue
        if mask & 2 and (lo & 0xFFF) == 0x224 and (hi >> 17) & 7 == 7:
            m = XF.fullmatch(t)
            if m:
                rd, rc, neg, p = reg(m[2]), reg(m[3]), 1 if m[4] else 0, int(m[5])
                nhi = rc | 0x400 | 7 << 13 | 1 << 16 | 7 << 17 | 7 << 20 | p << 23 | neg << 26
                out[x["addr"]] = (0x210 | g | rd << 16 | 0xFF << 24 | 0xFF << 32, ctrl | nhi,
                                  "%sIADD3.X %s, RZ, RZ, %s, %sP%d, !PT" % (m[1] or "", m[2], m[3], m[4], p))
    return out


def apply(nvdisasm, inp, outp, mask):
    kernel = carryfix.kernel_name(inp)
    ins = carryfix.listing(nvdisasm, inp, kernel)
    if not ins:
        sys.exit("nvdisasm gave no listing of " + kernel)
    ch = plan(ins, mask)
    b = bytearray(open(inp, "rb").read())
    off, _ = carryfix.text_section(b, kernel)
    base = ins[0]["addr"]
    for a, (lo, hi, _) in ch.items():
        struct.pack_into("<QQ", b, off + a - base, lo, hi)
    open(outp, "wb").write(b)
    after = {x["addr"]: x["text"] for x in carryfix.listing(nvdisasm, outp, kernel)}
    for x in ins:
        want = ch[x["addr"]][2] if x["addr"] in ch else x["text"]
        if after.get(x["addr"]) != want:
            import os
            os.remove(outp)
            sys.exit("alufix read back: %04x is %r, wanted %r" % (x["addr"], after.get(x["addr"]), want))
    return len(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("mask", type=int)
    ap.add_argument("--nvdisasm", default="nvdisasm")
    a = ap.parse_args()
    n = apply(a.nvdisasm, a.inp, a.out, a.mask)
    print("alufix: %d instructions moved to the ALU" % n)


if __name__ == "__main__":
    main()
