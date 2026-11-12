"""rfold120.py: the rotation fold's hand-written chunk loop (rfold.py, written for sm_89) put into an sm_120 (RTX 50)
build of the probe, in place of the loop function ptxas compiled there.

  python rfold120.py <base.cubin> <nvdisasm> <out.cubin> [rfold key=value ...]

Why. On an RTX 5090 a stream of nothing but mma holds 99.99% of the tensor pipe (mmabench.cu), and the probe's
loop with every part but the mma taken out 99.6%, but the rotation fold as ptxas compiles it from C only 91 to 92%:
the loss is in where ptxas puts the gathers, the stage loads and the barriers. rfold.py's loop holds 99.0% on an
RTX 4090 with all of that in it. This takes that loop's instructions and control fields as rfold.py makes them and
encodes them for sm_120.

The base is gprobe.cu built for sm_120 with -DGP_STAGE=24576 -DGP_ROTC=<form> and no GP_INLINE: the C fold as an
outlined function, which writes nearly every register, so the kernel keeps little across the call. Its records are
gprobe's `rot` 2 (8 bytes a lane a k-step), as rfold.py's co=1 reads. Run the probe with rot 2.

What differs from sm_89, and what is done about it:
  - the dynamic shared memory starts at 0x400 on sm_120 (no cluster: the CTA's rank is 0): every shared address of
    the loop gets 0x400 in its offset. The probe's CPU check judges this.
  - kernel parameters sit at c[0x0][0x380 + ...]: the offsets pointer at 0x390 and the chunk count at 0x398, where
    sm_89 has 0x170 and 0x178; the global address base pair is loaded with LDCU.64 from c[0x0][0x358].
  - LDG names its descriptor (desc[URn]); a branch's offset counts words of 4 bytes.
  - IADD3, MOV, IMAD print otherwise (IADD3 with its carry predicates PT, PT) but encode as on sm_89.
Every instruction is read back through nvdisasm and compared with what was asked for, and the loop may write only
the registers and predicates the compiled function writes.

Latency: rfold.py's own check uses sm_89's floors (ALU 5 cycles, IMMA 28, compare 13); ptxas for sm_120 goes as
low as 4 and 26 (sched120.py's calibration, kA), so sm_89's are kept.
"""
import os
import re
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402
import fold  # noqa: E402
import gfold  # noqa: E402
import rfold  # noqa: E402

SMEM = 0x400          # where sm_120's dynamic shared memory starts
UR_G = 8              # the uniform pair the loop loads the global address base into (the compiled function's)
SB_C = 5              # the scoreboard the prologue's constant loads use
LDCU64 = (0x0000000000000000 | 0x7ac | 7 << 12, 0x8000a00)   # LDCU.64 UR?, c[0x0][0x358] (ptxas 13.3's word)
DBG = [0]
LDG_HI = 0x0c1e9b00   # LDG.E.64.CONSTANT desc[URn][Rn.64+imm] (ptxas 13.3's second word)
MEM_OPS = ("LDSM", "LDS", "STS")


UR_S = [4, 7]         # uniform registers for the shared base (set in main to two the compiled function writes)


def shared(ins):
    """The instruction with the shared base UR_S[0] in its address, as ptxas 13.3 writes it: [Rn+URm+imm]. A load
    takes the uniform register in bits 32-39 and bit 27 of the second word; a store has another operation (0x988)
    and takes it, with the size, in the second word."""
    t = ins.text
    m = re.search(r"\[(R\d+|RZ)(?:\+(-?0x[0-9a-f]+))?\]", t)
    lo, hi = asm89.assemble(t)
    u = UR_S[0]
    if re.sub(r"^@!?P\d ", "", t).startswith("STS"):
        lo, hi = (lo & ~0xFFF) | 0x988, (hi & 0xF00) | u | 0x8000000
    else:
        lo, hi = lo | u << 32, hi | 0x8000000
    ins.text = t[:m.start()] + "[%s+UR%d%s]" % (m[1], u, "+" + m[2] if m[2] else "") + t[m.end():]
    ins.raw = (lo, hi)
    return ins


def base_ins():
    """The shared base, as ptxas computes it: 0x400 + (the CTA's rank in its cluster << 24)."""
    u, t = UR_S
    return [Ins("S2UR UR%d, SR_CgaCtaId" % u, wsb=SB_C, raw=(0x79c3 | 7 << 12 | u << 16, 0x8800)),
            Ins("UMOV UR%d, 0x400" % t, raw=(0x7882 | 7 << 12 | t << 16 | 0x400 << 32, 0)),
            Ins("ULEA UR%d, UR%d, UR%d, 0x18" % (u, u, t), wait=(SB_C,), stall=2,
                raw=(0x7291 | 7 << 12 | u << 16 | u << 24 | t << 32, 0x0f8ec0ff))]


def port(code):
    """sm_89 Ins list -> sm_120 Ins list (texts in sm_89's spelling where the encoding is the same)."""
    out = base_ins()
    for ins in code:
        t = ins.text
        op = re.sub(r"^@!?P\d ", "", t).split(" ")[0]
        if op.split(".")[0] in MEM_OPS:
            ins = shared(ins)
        elif t.startswith("ULDC.64"):
            m = re.match(r"ULDC\.64 UR(\d+), c\[0x0\]\[0x118\]", t)
            assert m, t
            lo = LDCU64[0] | UR_G << 16 | 0xff << 24 | (0x358 // 8) << 40
            ins = Ins("LDCU.64 UR%d, c[0x0][0x358]" % UR_G, stall=1, wsb=SB_C, raw=(lo, LDCU64[1]))
            out.append(ins)
            continue
        elif op.startswith("LDG"):
            m = re.match(r"LDG\.E\.64 R(\d+), \[R(\d+)\.64(?:\+(0x[0-9a-f]+))?\]$", t)
            assert m, t
            off = int(m[3], 16) if m[3] else 0
            lo = 0x7981 | int(m[1]) << 16 | int(m[2]) << 24 | UR_G << 32 | off << 40
            ins.text = "LDG.E.64.CONSTANT R%s, desc[UR%d][R%s.64%s]" % (m[1], UR_G, m[2], "+0x%x" % off if off else "")
            if DBG[0] & 4:
                lo = (lo & ~(0xFF << 32)) | 16 << 32
                ins.text = ins.text.replace("desc[UR%d]" % UR_G, "desc[UR16]")
            if DBG[0] & 1:
                ins.text = ins.text.replace(".CONSTANT", "")
            ins.raw = (lo & ~0xF000 | 7 << 12, 0x0c1e1b00 if DBG[0] & 1 else LDG_HI)
            # The loop's first record load waits for the address base.
            if not any(x.text.startswith("LDG") for x in out):
                ins.wait = tuple(sorted(set(ins.wait) | {SB_C}))
        elif "c[0x0][0x178]" in t:
            # sm_120 has no IMAD with a constant-bank operand: LDC (a load with a scoreboard), then a wait.
            m = re.match(r"IMAD\.MOV\.U32 R(\d+), RZ, RZ, c\[0x0\]\[0x178\]$", t)
            assert m, t
            d = int(m[1])
            out.append(Ins("LDC R%d, c[0x0][0x398]" % d, wsb=SB_C,
                           raw=(0x7b82 | 7 << 12 | d << 16 | 0xff << 24 | (0x398 // 4) << 40, 0x800)))
            out.append(Ins("NOP", wait=(SB_C,)))
            if DBG[0] & 32:     # debugging: chunks as k 2048's, 16
                out.append(Ins("MOV R%d, 0x10" % d))
            continue
        elif "c[0x0][0x170]" in t:
            m = re.match(r"IMAD\.WIDE\.U32 R(\d+), R(\d+), R(\d+), c\[0x0\]\[0x170\]$", t)
            assert m, t
            d, a, b2 = int(m[1]), int(m[2]), int(m[3])
            assert d not in (a, b2) and d + 1 not in (a, b2), t
            # ptxas for sm_120 never adds a 64-bit register in IMAD.WIDE (a first try that did read wrong
            # addresses): the product with RZ, then IADD.64 of the pointer, loaded with LDC.64 into the pair the
            # loop's first record load writes next (free until then).
            tp = int(re.match(r"LDG\.E\.64 R(\d+)", next(x.text for x in code if x.text.startswith("LDG")))[1])
            out.append(Ins("LDC.64 R%d, c[0x0][0x390]" % tp, wsb=SB_C,
                           raw=(0x7b82 | 7 << 12 | tp << 16 | 0xff << 24 | (0x390 // 4) << 40, 0xa00)))
            # The product is a warp's span of records, under 2^32: a 32-bit IMAD and a zero high word. (IMAD.WIDE.U32
            # with a register factor gave a wrong address on the card, dbg=2 against dbg=0 on 12 November.)
            if DBG[0] & 8:      # debugging: the span as if chunks were 0
                out.append(Ins("IMAD.SHL.U32 R%d, R%d, 0x100, RZ" % (d, a)))
            elif DBG[0] & 16:   # debugging: the span as if the warp were 1
                out.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, R%d" % (d, b2)))
            else:
                out.append(Ins("IMAD R%d, R%d, R%d, RZ" % (d, a, b2),
                               raw=(0x7224 | 7 << 12 | d << 16 | a << 24 | b2 << 32, 0x78e02ff)))
            out.append(Ins("MOV R%d, RZ" % (d + 1), raw=(0x7202 | 7 << 12 | (d + 1) << 16 | 0xff << 32, 0xf00)))
            out.append(Ins("NOP", wait=(SB_C,), stall=5))
            out.append(Ins("IADD.64 R%d, R%d, R%d" % (d, d, tp), stall=ins.stall, yld=ins.yld,
                           raw=(0x7235 | 7 << 12 | d << 16 | d << 24 | tp << 32, 0x78e0200)))
            if DBG[0] & 2:
                out[-1] = Ins("IADD.64 R%d, RZ, R%d" % (d, tp), stall=ins.stall, yld=ins.yld,
                              raw=(0x7235 | 7 << 12 | d << 16 | 0xff << 24 | tp << 32, 0x78e0200))
            continue
        out.append(ins)
    return out


BRA_RE = re.compile(r"BRA (\S+)$")


def assemble(text, here, labels):
    t = re.sub(r"\s+", " ", text.strip()).replace(".reuse", "")
    pred = 7
    m = re.match(r"@(!?)P(\d) (.*)$", t)
    if m:
        pred, t = int(m[2]) | (8 if m[1] else 0), m[3]
    m = BRA_RE.match(t)
    if m:
        # The offset in words of 4 bytes, v: its low byte in bits 16-23, the rest (v >> 8, times 4) in bits 32-63
        # (read from ptxas 13.3's branches: +0xd0 bytes is 0x...00340947, -0x1d0 is 0xfffffffc008c8947).
        v = (labels[m[1]] - (here + 16)) // 4
        lo = 0x0947 | pred << 12 | (v & 0xFF) << 16 | (((v >> 8) * 4) & 0xFFFFFFFF) << 32
        return lo, 0x3800000 | (0x3ffff if v < 0 else 0)
    return asm89.assemble(text, here, labels)


def spell120(t):
    """How nvdisasm 13.3 prints an instruction asked for in sm_89's spelling."""
    t = re.sub(r"\s+", " ", t.strip()).replace(".reuse", "")
    g = ""
    m = re.match(r"(@!?P\d )(.*)$", t)
    if m:
        g, t = m[1], m[2]
    m = re.match(r"IADD3 (R\d+), (.*)$", t)
    if m and "PT" not in t:
        a, b, c = [x.strip() for x in m[2].split(",")]
        t = "IADD3 %s, PT, PT, %s, %s, %s" % (m[1], a, b, c)
    return g + t


def patch(cubin_in, sass_text, cubin_out, code, nvdisasm):
    b = bytearray(open(cubin_in, "rb").read())
    secs = asm89.sections(b)
    name = next(n for n in secs if n.startswith(".text.") and "pearl" in n)
    off, _size = secs[name]
    entry, ret, back = asm89.region(sass_text)
    rr = asm89.RET_REG[0]
    code = list(code) + [Ins("MOV R%d, 0x%x" % (rr, back)), Ins("IMAD.MOV.U32 R%d, RZ, RZ, RZ" % (rr + 1)),
                         Ins("BRA .RET", stall=5)]
    room = (ret - entry) // 16
    if len(code) > room:
        raise ValueError("the loop is %d instructions and the function has room for %d" % (len(code), room))
    labels = {".RET": ret}
    for i, ins in enumerate(code):
        if ins.label:
            labels[ins.label] = entry + 16 * i
    for i in range(room):
        a = entry + 16 * i
        if i < len(code):
            ins = code[i]
            lo, hi = ins.raw if ins.raw else assemble(ins.text, a, labels)
            hi = (hi & ~asm89.CTRL_MASK) | (ins.ctrl() << asm89.CTRL_SHIFT)
        else:
            lo, hi = 0x7918, asm89.ctrl(0, False) << asm89.CTRL_SHIFT
        struct.pack_into("<QQ", b, off + a, lo, hi)
    open(cubin_out, "wb").write(b)
    rows, lab2, _e = asm89.listing(asm89.disassemble(cubin_out, nvdisasm))
    got = dict(rows)
    bad = 0
    for i, ins in enumerate(code):
        a = entry + 16 * i
        want = spell120(ins.text)
        have = got.get(a, "?").replace(".reuse", "")
        m = BRA_RE.search(want)
        if m:
            m2 = re.search(r"BRA `\((\S+)\)", have)
            ok = bool(m2) and lab2.get(m2[1]) == labels[m[1]] and want.split("BRA")[0] == have.split("BRA")[0]
        elif want.startswith("MOV ") or "IMAD.MOV.U32" in want:
            ok = have.split(",")[0].split(" ")[-1] == want.split(",")[0].split(" ")[-1]
        else:
            ok = want == have
        if not ok:
            bad += 1
            if bad <= 12:
                print("MISMATCH at %05x: asked %r, the cubin has %r" % (a, want, have))
    if bad:
        os.remove(cubin_out)
        raise RuntimeError("%d instructions did not come back as asked" % bad)
    return len(code), room


def p2_free(sass_text):
    """True when the kernel never reads P2 between the loop function's return and its next write of P2: walking
    on from the CALL (and round from the kernel's start, its tile loop's top), the first instruction that names P2
    sets it."""
    rows, _labels, entry = asm89.listing(sass_text)
    kern = [(a, t) for a, t in rows if a < entry]
    call = [i for i, (a, t) in enumerate(kern) if t.startswith("CALL.REL")]
    if len(call) != 1:
        return False
    for a, t in kern[call[0] + 1:] + kern:
        if re.search(r"\bP2\b", t):
            return bool(re.match(r"I?SETP\.\S+ P2,", t))
    return True


def restore_code(sass_text, want):
    """Instructions that put back, before the return, registers the kernel keeps across the call and the loop
    overwrites. Each must be written once in the kernel, by a form that can be redone: S2R of SR_TID.X, LDC of a
    constant, LOP3 of an immediate with one that is put back, or IADD of an immediate to a register the kernel also
    stored to its frame (STL [R1+o]) and does not store over. Returns a list of Ins, or None if one cannot be."""
    rows, _labels, entry = asm89.listing(sass_text)
    kern = [t for a, t in rows if a < entry]

    def writes(r):
        return [t for t in kern if re.match(r"(@!?P\d )?\S+ R%d(,| )" % r, t) and not t.startswith(("STS", "STL", "STG"))]

    first, alu, done = [], [], set()
    for r in sorted(want):
        w = writes(r)
        if len(w) != 1:
            return None
        t = w[0]
        if t == "S2R R%d, SR_TID.X" % r:
            first.append(Ins(t, wsb=SB_C))
        elif re.match(r"LDC R%d, c\[0x0\]\[0x([0-9a-f]+)\]$" % r, t):
            c = int(re.match(r"LDC R\d+, c\[0x0\]\[0x([0-9a-f]+)\]$", t)[1], 16)
            first.append(Ins(t, wsb=SB_C, raw=(0x7b82 | 7 << 12 | r << 16 | 0xff << 24 | (c // 4) << 40, 0x800)))
        elif re.match(r"IADD R%d, R(\d+), (0x[0-9a-f]+)$" % r, t):
            m = re.match(r"IADD R%d, R(\d+), (0x[0-9a-f]+)$" % r, t)
            st = [x for x in kern if re.match(r"STL \[R1(\+0x[0-9a-f]+)?\], R%s$" % m[1], x)]
            slot = re.match(r"STL \[R1(?:\+(0x[0-9a-f]+))?\]", st[0])[1] if len(st) == 1 else None
            if len(st) != 1 or any(x.startswith("STL [R1%s]" % ("+" + slot if slot else "")) for x in kern if x != st[0]):
                return None
            first.append(Ins("LDL R%d, [R1%s]" % (r, "+" + slot if slot else ""), wsb=SB_C))
            alu.append((r, Ins("IADD3 R%d, R%d, %s, RZ" % (r, r, m[2]))))
        elif re.match(r"LOP3\.LUT R%d, R(\d+), (0x[0-9a-f]+), RZ, (0x[0-9a-f]+), !PT$" % r, t):
            m = re.match(r"LOP3\.LUT R%d, R(\d+), (0x[0-9a-f]+), RZ, (0x[0-9a-f]+), !PT$" % r, t)
            if int(m[1]) not in want:
                return None
            alu.append((r, Ins(t)))
        else:
            return None
        done.add(r)
    out = first + [Ins("NOP", wait=(SB_C,), stall=2)]
    for _r, ins in alu:
        out += [ins, Ins("NOP", stall=5)]
    return out


def rename(code, ren):
    for ins in code:
        new = re.sub(r"\bR(\d+)\b", lambda m: "R%d" % ren.get(int(m[1]), int(m[1])), ins.text)
        if new != ins.text:
            ins.text, ins.raw = new, None


def main():
    cubin, nvd, out = sys.argv[1:4]
    cfg = dict(rfold.CFG)
    cut = None
    for kv in sys.argv[4:]:
        k, v = kv.split("=")
        if k == "dbg":
            DBG[0] = int(v, 0)
            continue
        if k == "cut":
            # TIMING ONLY / debugging: only the first `cut` instructions of the ported loop (no back branch), then a
            # wait for every load and the return. Wrong results; finds where a fault is.
            cut = int(v, 0)
            continue
        assert k in cfg, k
        cfg[k] = int(v, 0)
    text = asm89.disassemble(cubin, nvd)
    # The global address base goes into the uniform pair the compiled function loads it into, so the kernel keeps
    # nothing there across the call.
    global UR_G
    rows, _l, entry = asm89.listing(text)
    urs = {int(m[1]) for a, t in rows if a >= entry
           for m in [re.match(r"LDCU\.64 UR(\d+), c\[0x0\]\[0x358\]", t)] if m}
    if len(urs) != 1:
        sys.exit("the compiled function loads the global address base into %s, not one pair" % sorted(urs))
    UR_G = urs.pop()
    cfg["ur"] = UR_G
    # Two more uniform registers the compiled function writes, for the shared base.
    uw = sorted({int(m[1]) for a, t in rows if a >= entry
                 for m in [re.match(r"(?:@!?U?P\d )?U?\S+ UR(\d+),", t)] if m and not t.startswith(("STG", "STS"))}
                - {UR_G, UR_G + 1})
    if len(uw) < 2:
        sys.exit("the compiled function writes too few uniform registers: %s" % uw)
    UR_S[:] = uw[:2]
    code, bad, most, G = rfold.build(cfg)
    if bad:
        sys.exit("%d latency problems: not written" % bad)
    regs, preds = asm89.clobbers(text)
    # A register the loop writes and the compiled function does not (the kernel may keep a value there across
    # the call) is renamed to one the function writes and the loop never names. Renamed in sm_89's spelling,
    # before the port, so that no word assembled ahead of time keeps the old number.
    used, named = set(), set()
    for ins in code:
        named |= {int(x) for x in re.findall(r"\bR(\d+)\b", ins.text)}
        # A 64-bit operand (Rn.64, IMAD.WIDE) names Rn + 1 too.
        named |= {int(x) + 1 for x in re.findall(r"\bR(\d+)\.64\b", ins.text)}
        if "WIDE" in ins.text or ".64" in ins.text.split(" ")[0]:
            named |= {int(x) + 1 for x in re.findall(r"\bR(\d+)\b", ins.text)}
        if "BRA" in ins.text or ins.text.startswith("ULDC"):
            continue
        r_, w, _k = fold.rw(ins.text)
        used |= {x for x in w if not isinstance(x, str)}
        named |= {x for x in list(r_) + list(w) if not isinstance(x, str)}
    extra = sorted(used - regs)
    # The register the caller passes the return address in is dead once the loop returns (it returns through
    # RET_REG, set at the end).
    _e, _r, back = asm89.region(text)
    kern = [t for a, t in rows if a < entry]
    extra = [r for r in extra if [t for t in kern if re.match(r"\S+ R%d," % r, t)] != ["MOV R%d, 0x%x" % (r, back)]]
    # A rename target is a register the compiled function writes, or one the kernel never names at all.
    in_kern = {int(x) for t in kern for x in re.findall(r"\bR(\d+)\b", t)}
    in_kern |= {int(x) + 1 for t in kern for x in re.findall(r"\bR(\d+)\.64\b", t)}
    free = set(regs) | (set(range(4, 255)) - in_kern)
    spare = sorted(r for r in free if r not in named and r not in (0, 1, 2, 3, 255))
    rest = []
    if len(spare) < len(extra):
        rest = restore_code(text, set(extra))
        if rest is None:
            sys.exit("the loop writes registers the compiled function does not: R%s" % ", R".join(map(str, extra)))
        print("put back before the return: R%s" % ", R".join(map(str, extra)))
    else:
        ren = dict(zip(extra, spare))
        if ren:
            rename(code, ren)
            print("renamed: " + ", ".join("R%d -> R%d" % kv for kv in ren.items()))
    code = port(code)
    if cut is not None:
        code = [x for x in code[:cut] if "BRA" not in x.text] + [Ins("NOP", wait=(0, 1, 2, 3, 4, 5), stall=8)]
    code += rest
    # The compiled function writes P1 alone. P2 is free across the call as well: gprobe's kernel writes P2 (and P3)
    # at the top of its tile loop before it reads them, and reads neither between the call and there.
    if p2_free(text):
        preds = preds | {2}
    bp = 2 if 2 in preds else min(preds - {1})
    lp = 1 if 1 in preds else min(preds - {bp})
    for ins in code:
        t = ins.text
        if lp != 1:
            t = re.sub(r"^(ISETP\.\S+ )P1,", r"\1P%d," % lp, t)
            t = t.replace("@P1 BRA", "@P%d BRA" % lp)
        if bp != 2:
            t = re.sub(r"^(ISETP\.\S+ )P2,", r"\1P%d," % bp, t)
            t = t.replace("P2 BAR", "P%d BAR" % bp).replace("@P2 BRA", "@P%d BRA" % bp)
        ins.text = t
    n, room = patch(cubin, text, out, code, nvd)
    asm89.set_barriers(out, 3)
    print("rfold120: %s, %d instructions of %d, predicates P%d (loop) P%d (band), global base UR%d; read back "
          "through nvdisasm as asked; %d at most in a gap" % (os.path.basename(out), n, room, lp, bp, UR_G, most))


if __name__ == "__main__":
    main()
