"""rebalance.py: move helper instructions from the multiplier pipe to the ALU in a compiled sm_86/sm_89 cubin.

  python rebalance.py <in.cubin> <out.cubin> <kernel> [--every N] [--kinds x,x0,xa,mov,neg,iadd] [--nvdisasm PATH]

On Ampere and Ada an IMAD.WIDE takes two slots of the FMA-heavy pipe, and ptxas, which seems to count it as one,
puts carries and moves on that pipe as IMAD.X, IMAD.MOV and IMAD.IADD. In the Quantus kernel that pipe runs 81%
busy and the ALU 60% (Nsight Compute, RTX 4090). Each rewrite here is the same operation on the ALU:

  x     IMAD.X Rd, RZ, RZ, Rc, Pp        ->  IADD3.X Rd, RZ, Rc, RZ, Pp, !PT
  x0    IMAD.X Rd, RZ, RZ, RZ, Pp        ->  IADD3.X Rd, RZ, RZ, RZ, Pp, !PT
  xa    IMAD.X Rd, Ra, 0x1, Rc, Pp       ->  IADD3.X Rd, Ra, Rc, RZ, Pp, !PT
  mov   IMAD.MOV.U32 Rd, RZ, RZ, Rs      ->  MOV Rd, Rs
  neg   IMAD.MOV Rd, RZ, RZ, -Rs         ->  IADD3 Rd, RZ, -Rs, RZ
  iadd  IMAD.IADD Rd, Ra, 0x1, -Rb       ->  IADD3 Rd, Ra, -Rb, RZ

Result on the RTX 4090 (13 Nov, qcdrv.cu, the build-g kernel): with --window 6 all 509 eligible rewrites keep the
hits exact, and the kernel got slower: 870 MH/s against 950 and 943 for the untouched cubin (--window 10: 890;
--window 6 --every 2: 905). Without the window the results were wrong: a value passed from an IMAD.WIDE to an ALU
instruction (or back) needs more cycles than ptxas leaves between two instructions on the same pipe. The slowdown
fits ubench.cu: an IMAD.WIDE also takes an ALU slot (IMAD.WIDE + IADD3 at 1:1 and at 1:2 both run at 2
instructions an SM a clock), so the ALU, not the multiplier pipe, is what binds, and ptxas had balanced it.
On the RTX 5080 (sm_120), --reverse (ALU to multiplier pipe) kept the hits exact with or without a window and was
6% slower too (597-600 against 637). Kept as a tool, not used by the build.

--every N rewrites one candidate in N (in program order), to balance the two pipes rather than swap the load.
Each rewritten instruction keeps its predicate and its control bits (stall, yield, barriers) and drops its operand
reuse bits, and so does the instruction before it (a reuse hint would name the old operand slots). The result is
disassembled again and every instruction compared with what was asked for; any mismatch aborts.
"""
import argparse
import re
import struct
import subprocess
import sys

LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\* 0x([0-9a-f]{16}) \*/")
HIGH = re.compile(r"^\s+/\* 0x([0-9a-f]{16}) \*/")
REUSE = 0xF << 58          # in the high word: instruction bits 122-125
REG = r"R(\d+)(?:\.reuse)?"


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
            ins.append({"addr": int(m[1], 16), "text": m[2].strip(), "lo": int(m[3], 16), "hi": int(h[1], 16)})
    return ins


def text_section(b, kernel):
    """File offset of the kernel's .text section in an ELF64 cubin."""
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


def pred_bits(lo):
    return lo & 0xF000


def carry(pred):
    """Carry-in predicate field (instruction bits 87-90) for P0..P6."""
    n = int(pred[1:])
    return ((n & 1) << 23) | ((n >> 1) << 24)


def ctrl(hi):
    return hi & (0x1FFFFF << 41) & ~REUSE


def r(n):
    return 255 if n == "Z" else int(n)


def rewrite(ins):
    """(kind, new lo, new hi, expected text) or None."""
    t, lo, hi = ins["text"], ins["lo"], ins["hi"]
    guard = ""
    m = re.match(r"(@!?P\d) (.*)", t)
    if m:
        guard, t = m[1] + " ", m[2]
    p, c = pred_bits(lo), ctrl(hi)
    m = re.fullmatch(r"IMAD\.X R(\d+), RZ, RZ, R(\d+|Z)(?:\.reuse)?, (P[0-6])", t)
    if m:
        d, s, pr = int(m[1]), m[2], m[3]
        if s == "Z":
            return ("x0", 0x210 | p | d << 16 | 255 << 24 | 255 << 32,
                    c | 0xFF | 0xE4 << 8 | 0x7F << 16 | carry(pr), guard + "IADD3.X R%d, RZ, RZ, RZ, %s, !PT" % (d, pr))
        return ("x", 0x210 | p | d << 16 | 255 << 24 | int(s) << 32,
                c | 0xFF | 0xE4 << 8 | 0x7F << 16 | carry(pr), guard + "IADD3.X R%d, RZ, R%s, RZ, %s, !PT" % (d, s, pr))
    m = re.fullmatch(r"IMAD\.X R(\d+), " + REG + r", 0x1, " + REG + r", (P[0-6])", t)
    if m:
        d, a, s, pr = int(m[1]), int(m[2]), int(m[3]), m[4]
        return ("xa", 0x210 | p | d << 16 | a << 24 | s << 32,
                c | 0xFF | 0xE4 << 8 | 0x7F << 16 | carry(pr), guard + "IADD3.X R%d, R%d, R%d, RZ, %s, !PT" % (d, a, s, pr))
    m = re.fullmatch(r"IMAD\.MOV\.U32 R(\d+), RZ, RZ, " + REG, t)
    if m:
        d, s = int(m[1]), int(m[2])
        return ("mov", 0x202 | p | d << 16 | s << 32, c | 0x0F00, guard + "MOV R%d, R%d" % (d, s))
    m = re.fullmatch(r"IMAD\.MOV R(\d+), RZ, RZ, -" + REG, t)
    if m:
        d, s = int(m[1]), int(m[2])
        return ("neg", 0x210 | p | d << 16 | 255 << 24 | s << 32 | 1 << 63, c | 0xFF | 0xE0 << 8 | 0xFF << 16 | 0x07 << 24,
                guard + "IADD3 R%d, RZ, -R%d, RZ" % (d, s))
    m = re.fullmatch(r"IMAD\.IADD R(\d+), " + REG + r", 0x1, -" + REG, t)
    if m:
        d, a, s = int(m[1]), int(m[2]), int(m[3])
        return ("iadd", 0x210 | p | d << 16 | a << 24 | s << 32 | 1 << 63, c | 0xFF | 0xE0 << 8 | 0xFF << 16 | 0x07 << 24,
                guard + "IADD3 R%d, R%d, -R%d, RZ" % (d, a, s))
    return None


def norm(t):
    return re.sub(r"\.reuse", "", re.sub(r"\s+", " ", t)).strip()


def regs(text):
    """(writes, reads) of an instruction as sets of 'R12' / 'P3' names, roughly but on the safe side."""
    t = re.sub(r"^@!?P\d ", "", text)
    op, _, rest = t.partition(" ")
    args = [x.strip() for x in rest.split(",")] if rest else []
    w, r = set(), set()
    names = [re.sub(r"[-|!]|\.reuse|\.H1|\.B\d", "", x) for x in args]
    i = 0
    if names and re.fullmatch(r"R\d+", names[0]):
        w.add(names[0])
        if ".WIDE" in op:
            w.add("R%d" % (int(names[0][1:]) + 1))
        i = 1
    while i < len(names) and re.fullmatch(r"P[0-6]|PT", names[i]) and op.split(".")[0] in ("IADD3", "IMAD", "LEA"):
        w.add(names[i])
        i += 1
    if op.startswith(("ISETP", "PLOP3", "FSETP")):
        for n in names[:2]:
            w.add(n)
        i = 2
    for n in names[i:]:
        for m in re.findall(r"R\d+|P[0-6]", n):
            r.add(m)
    if ".WIDE" in op and len(names) >= 4 and re.fullmatch(r"R\d+", names[-1]):
        r.add("R%d" % (int(names[-1][1:]) + 1))
    w.discard("PT")
    return w, r


ALU_OPS = ("IADD3", "LOP3", "SHF", "MOV", "SEL", "ISETP", "PRMT", "LEA", "PLOP3", "P2R", "R2P", "IABS", "IMNMX")


def opcode(text):
    return re.sub(r"^@!?P\d ", "", text).split(" ")[0]


def is_fma(text):
    return opcode(text).startswith("IMAD")


def is_alu(text):
    return opcode(text).split(".")[0] in ALU_OPS


def near_pipe(ins, i, window, other):
    """Whether an instruction on the other pipe (other(text) true) within `window` stall cycles before i writes what i
    reads, or one within that distance after i reads what i writes. A branch ends the search."""
    w, r = regs(ins[i]["text"])
    cyc = 0
    for j in range(i - 1, max(i - 40, -1), -1):
        cyc += (ins[j]["hi"] >> 41) & 15
        if cyc >= window or re.search(r"\b(BRA|CALL|EXIT|BSYNC|BSSY)\b", ins[j]["text"]):
            break
        if other(ins[j]["text"]) and regs(ins[j]["text"])[0] & r:
            return True
    cyc = (ins[i]["hi"] >> 41) & 15
    for k in range(i + 1, min(i + 40, len(ins))):
        if cyc >= window or re.search(r"\b(BRA|CALL|EXIT|BSYNC|BSSY)\b", ins[k]["text"]):
            break
        if other(ins[k]["text"]) and regs(ins[k]["text"])[1] & w:
            return True
        cyc += (ins[k]["hi"] >> 41) & 15
    return False


def near_fma(ins, i, window):
    return near_pipe(ins, i, window, is_fma)


def imadx(guard, p, c, d, a, s, pr, imm_a):
    """IMAD.X Rd, Ra|RZ, 0x1|RZ, Rc|RZ, Pp. s is a register number or 255 (RZ)."""
    if imm_a is None:
        lo = 0x224 | p | d << 16 | 255 << 24 | 255 << 32
        text = "IMAD.X R%d, RZ, RZ, %s, %s" % (d, "RZ" if s == 255 else "R%d" % s, pr)
    else:
        lo = 0x824 | p | d << 16 | imm_a << 24 | 1 << 32
        text = "IMAD.X R%d, R%d, 0x1, R%d, %s" % (d, imm_a, s, pr)
    return lo, c | s | 0x06 << 8 | 0x0E << 16 | carry(pr), guard + text


def rewrite_rev(ins):
    """ALU -> multiplier pipe, for sm_120 (ptxas there puts every carry and move on the ALU). Same encodings as sm_89.
    (kind, new lo, new hi, expected text) or None."""
    t, lo, hi = ins["text"], ins["lo"], ins["hi"]
    guard = ""
    m = re.match(r"(@!?P\d) (.*)", t)
    if m:
        guard, t = m[1] + " ", m[2]
    p, c = pred_bits(lo), ctrl(hi)
    m = re.fullmatch(r"IADD3\.X R(\d+), PT, PT, RZ, (RZ|" + REG + r"), RZ, (P[0-6]), !PT", t)
    if m:
        s = 255 if m[2] == "RZ" else int(m[3])
        lo2, hi2, text = imadx(guard, p, c, int(m[1]), None, s, m[4], None)
        return ("rx0" if s == 255 else "rx", lo2, hi2, text)
    m = re.fullmatch(r"IADD3\.X R(\d+), PT, PT, " + REG + r", " + REG + r", RZ, (P[0-6]), !PT", t)
    if m:
        lo2, hi2, text = imadx(guard, p, c, int(m[1]), None, int(m[3]), m[4], int(m[2]))
        return ("rxa", lo2, hi2, text)
    m = re.fullmatch(r"MOV R(\d+), " + REG, t)
    if m:
        d, s = int(m[1]), int(m[2])
        return ("rmov", 0x224 | p | d << 16 | 255 << 24 | 255 << 32, c | s | 0x8E << 16 | 0x07 << 24,
                guard + "IMAD.MOV.U32 R%d, RZ, RZ, R%d" % (d, s))
    m = re.fullmatch(r"IADD3 R(\d+), PT, PT, -" + REG + r", RZ, RZ", t)
    if m:
        d, s = int(m[1]), int(m[2])
        return ("rneg", 0x224 | p | d << 16 | 255 << 24 | 255 << 32, c | s | 0x0A << 8 | 0x8E << 16 | 0x07 << 24,
                guard + "IMAD.MOV R%d, RZ, RZ, -R%d" % (d, s))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cubin_in")
    ap.add_argument("cubin_out")
    ap.add_argument("kernel")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--kinds", default="x,x0,xa,mov,neg,iadd")
    ap.add_argument("--window", type=int, default=0, help="skip candidates this close (stall cycles) to an IMAD they "
                    "feed or are fed by: a value crossing between the two pipes seems to take longer than ptxas allowed for")
    ap.add_argument("--reverse", action="store_true", help="ALU -> multiplier pipe (for sm_120): kinds rx,rx0,rxa,rmov,rneg")
    ap.add_argument("--nvdisasm", default="nvdisasm")
    a = ap.parse_args()
    if a.reverse and a.kinds == "x,x0,xa,mov,neg,iadd":
        a.kinds = "rx,rx0,rxa,rmov,rneg"
    kinds = set(a.kinds.split(","))
    rw = rewrite_rev if a.reverse else rewrite
    other = is_alu if a.reverse else is_fma
    b = bytearray(open(a.cubin_in, "rb").read())
    off, size = text_section(b, a.kernel)
    ins = listing(a.nvdisasm, a.cubin_in, a.kernel)
    want = {}
    seen = 0
    counts = {}
    for i, x in enumerate(ins):
        w = rw(x)
        if not w or w[0] not in kinds:
            continue
        if a.window and near_pipe(ins, i, a.window, other):
            continue
        seen += 1
        if (seen - 1) % a.every:
            continue
        kind, lo, hi, text = w
        struct.pack_into("<QQ", b, off + x["addr"], lo, hi)
        want[x["addr"]] = text
        counts[kind] = counts.get(kind, 0) + 1
        if i > 0:  # the instruction before: drop its reuse hints
            prev = ins[i - 1]
            plo, phi = struct.unpack_from("<QQ", b, off + prev["addr"])
            struct.pack_into("<QQ", b, off + prev["addr"], plo, phi & ~REUSE)
    open(a.cubin_out, "wb").write(b)
    after = listing(a.nvdisasm, a.cubin_out, a.kernel)
    if len(after) != len(ins):
        sys.exit("instruction count changed")
    bad = 0
    for x, y in zip(ins, after):
        exp = want.get(x["addr"], x["text"])
        if norm(exp) != norm(y["text"]):
            bad += 1
            if bad < 6:
                print("MISMATCH at %04x: want %r, got %r" % (x["addr"], exp, y["text"]))
    if bad:
        sys.exit("%d mismatches; %s not usable" % (bad, a.cubin_out))
    print("rewrote %d of %d candidates (%s) in %d instructions" % (len(want), seen, counts, len(ins)))


if __name__ == "__main__":
    main()
