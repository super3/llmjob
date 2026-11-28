"""fuse.py: fuse pairs of two-input IADD3s in the sm_120 scan kernel into three-input ones, and close the gaps.

  python fuse.py <nvdisasm> <in.cubin> <out.cubin> [--kernel SUBSTR] [--max N] [--list]

Why: on an RTX 50 the kernel's time is its instruction count, an IMAD.WIDE counting about 3.5 and everything else 1
(quantus-ledger.md, 28 Nov): a three-input add costs one slot, as a two-input one does. PTX has no add of three
values with a two-bit carry, so ptxas writes every multiword sum as chains of two-input adds. IADD3 itself takes
three inputs, two carry ins and two carry outs. This tool rewrites

    i:  IADD3[.X] t, Pa, PT, x, y, RZ[, ca, !PT]
    j:  IADD3[.X] d, Pb, PT, t, z, RZ[, cb, !PT]        (t dead after j)
  into, at j's place,
        IADD3.X   d, Pa, Pb, x, y, z, ca, cb

and deletes i. The two carry outs then hold "at least one carry" and "two carries" where they held i's carry and
j's carry: the same count. So a pair is only fused when both carries are read once each, as the two carry ins of one
add (or of two adds that are fused too), with the same sign. A "-R" source of a plain IADD3 (two's complement) becomes
"~R" with a true carry in, so it needs a free carry-in slot.

Layout: every instruction after a deleted one moves up. The moved range is one loop of the kernel (the region from
the loop's head to its back branch), so every branch inside is rewritten for the new distances and nothing outside
moves; the freed slots at the region's end become one branch over NOPs. Timing: ptxas guarantees each value is ready
by giving every instruction a stall count. A deleted instruction's stall goes to the instruction before it, so no
two remaining instructions come closer than ptxas put them; the fused add gets at least i's and j's stall.

Checked here: the result is read back through nvdisasm and every instruction compared with what was meant. Checked
on a card: the core's startup check against the compiled kernel, then verify-core.js (sm86/build.py --defs
-DQC_FUSE=1 makes the file to try).
"""
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm86"))
import carryfix  # noqa: E402

LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\* 0x([0-9a-f]{16}) \*/")
HIGH = re.compile(r"^\s+/\* 0x([0-9a-f]{16}) \*/")
M64 = (1 << 64) - 1
NOP = (0x0000000000007918, 0x000fc00000000000)
EXTRA = int(os.environ.get("QC_FUSE_STALL", "2"))


def listing(nvdisasm, cubin, kernel):
    r = carryfix.run([nvdisasm, "-hex", "-c", cubin])
    if r.returncode != 0:
        sys.exit("nvdisasm failed: " + r.stderr.strip()[:300])
    out = r.stdout.splitlines()
    on, ins, labels, pend = False, [], {}, []
    for i, ln in enumerate(out):
        if ln.startswith("//----") and ".text." in ln:
            on = (".text." + kernel) in ln
            continue
        if not on:
            continue
        m = re.match(r"^(\.L_x_\d+):", ln)
        if m:
            pend.append(m[1])
            continue
        m = LINE.match(ln)
        if m:
            text = re.sub(r"\s+", " ", m[2].strip())
            text = re.sub(r"\s*\?\w+$", "", text)
            text = re.sub(r"\s*&\w+=(\{[^}]*\}|\S+)", "", text).strip()
            for l in pend:
                labels[l] = len(ins)
            ins.append({"addr": int(m[1], 16), "text": text, "lo": int(m[3], 16),
                        "hi": int(HIGH.match(out[i + 1])[1], 16), "labels": pend})
            pend = []
    return ins, labels


def stall(x):
    return (x["hi"] >> 41) & 15


def set_stall(hi, n):
    return (hi & ~(15 << 41)) | (n << 41)


class Add:
    """An IADD3 or IADD3.X with no guard: d, pouts (two, 7 = PT), srcs (three of ('r', n, neg) / ('i', v, '') /
    ('z',)), cins (two of (n, neg), (7, True) = !PT = no carry)."""

    def __init__(self, text):
        self.ok = False
        m = re.fullmatch(r"(IADD3(?:\.X)?) (.*)", text)
        if not m:
            return
        a = [s.strip().replace(".reuse", "") for s in m[2].split(",")]
        x = m[1].endswith(".X")
        if len(a) == 4:
            a = [a[0], "PT", "PT"] + a[1:]
        if x and len(a) == 6:
            a = [a[0], "PT", "PT"] + a[1:]
        if len(a) != (8 if x else 6) or not re.fullmatch(r"R\d+", a[0]):
            return
        self.d = int(a[0][1:])
        self.pouts = []
        for p in a[1:3]:
            if not re.fullmatch(r"P[0-6T]", p):
                return
            self.pouts.append(7 if p == "PT" else int(p[1]))
        self.srcs = []
        for s in a[3:6]:
            neg = ""
            if s[:1] in "-~" and not s.startswith("-0x"):
                neg, s = s[0], s[1:]
            if s == "RZ" and not neg:
                self.srcs.append(("z",))
            elif re.fullmatch(r"R\d+", s):
                self.srcs.append(("r", int(s[1:]), neg))
            elif re.fullmatch(r"-?0x[0-9a-f]+", s) and not neg:
                v = int(s.lstrip("-"), 16)
                self.srcs.append(("i", (-v if s.startswith("-") else v) & 0xFFFFFFFF, ""))
            else:
                return
        self.cins = []
        for p in (a[6:8] if x else ["!PT", "!PT"]):
            m2 = re.fullmatch(r"(!?)P([0-6T])", p)
            if not m2:
                return
            self.cins.append((7 if m2[2] == "T" else int(m2[2]), m2[1] == "!"))
        self.x = x
        self.ok = True

    def real_srcs(self):
        return [s for s in self.srcs if s[0] != "z"]

    def real_cins(self):
        return [c for c in self.cins if c != (7, True)]


def defs_uses(text):
    """(registers written, registers read, predicates written, predicates read), or None for an instruction this
    tool does not model (a pair is never fused across one)."""
    t = text
    up = set()
    m = re.match(r"@!?P(\d) (.*)$", t)
    if m:
        up.add(int(m[1]))
        t = m[2]
    op, _, rest = t.partition(" ")
    a = [s.strip().replace(".reuse", "") for s in rest.split(",")] if rest else []

    def regs(xs, wide=False):
        out = set()
        for s in xs:
            m3 = re.fullmatch(r"[-~]?R(\d+)", s)
            if m3:
                out.add(int(m3[1]))
                if wide:
                    out.add(int(m3[1]) + 1)
            elif not re.fullmatch(r"RZ|-?0x[0-9a-f]+|!?P[0-6T]|UR\d+|c\[0x[0-9a-f]+\]\[0x[0-9a-f]+\]", s):
                return None
        return out

    def preds(xs):
        return {int(s[-1]) for s in xs if re.fullmatch(r"!?P[0-6]", s)}

    if op in ("IADD3", "IADD3.X"):
        ad = Add(t)
        if not ad.ok and regs(a) is None:
            return None
        n = len(a) - (5 if op == "IADD3.X" else 3)   # destination + carry outs
        d = regs(a[:1])
        u = regs(a[n:])
        if d is None or u is None:
            return None
        return d, u, preds(a[1:n]), up | preds(a[n:])
    if op in ("IMAD.WIDE.U32", "IMAD.WIDE.U32.X"):
        x = op.endswith(".X")
        n = len(a) - (4 if x else 3)
        d = regs(a[:1], True)
        u1 = regs(a[n:n + 2])
        u2 = regs(a[n + 2:n + 3], True)
        if d is None or u1 is None or u2 is None:
            return None
        return d, u1 | u2, preds(a[1:n]), up | preds(a[n + 3:])
    if op in ("MOV", "SEL"):
        d = regs(a[:1])
        u = regs(a[1:])
        if d is None or u is None:
            return None
        return d, u, set(), up | preds(a[1:])
    return None


def find(ins, lo, hi):
    """Pairs (i, j, fused Add fields) to fuse inside ins[lo..hi], by index."""
    du = [defs_uses(x["text"]) for x in ins]
    adds = [Add(x["text"]) for x in ins]
    block_end = set()
    for k in range(lo, hi + 1):
        if du[k] is None or ins[k]["labels"]:
            block_end.add(k)
    cands = {}
    used = set()
    for i in range(lo, hi + 1):
        A = adds[i]
        if not A.ok or i in used or len(A.real_srcs()) > 2 or len(A.real_cins()) > 1:
            continue
        # find j: the only reader of A.d before A.d is written again, in the same block
        j, dead = None, False
        for k in range(i + 1, min(hi, i + 400) + 1):
            if k in block_end:
                break
            d, u, dp, upd = du[k]
            if A.d in u:
                if j is None:
                    j = k
                    if A.d in d:
                        dead = True
                        break
                    continue
                break   # a second reader
            if A.d in d:
                dead = True
                break
        if j is None or not dead or j in used:
            continue
        B = adds[j]
        if not B.ok or len(B.real_srcs()) > 2 or len(B.real_cins()) > 1:
            continue
        tsrc = [s for s in B.srcs if s[0] == "r" and s[1] == A.d]
        if len(tsrc) != 1 or tsrc[0][2]:
            continue
        srcs = A.real_srcs() + [s for s in B.real_srcs() if not (s[0] == "r" and s[1] == A.d)]
        cins = A.real_cins() + B.real_cins()
        # a "-R" of a plain IADD3 is "~R" and a true carry in
        for S in (A, B):
            if not S.x:
                for s in S.real_srcs():
                    if s[0] == "r" and s[2] == "-":
                        cins.append((7, False))
        srcs = [("r", s[1], "~" if s[2] else "") if s[0] == "r" else s for s in srcs]
        if len(srcs) > 3 or len(cins) > 2 or any(s[0] == "i" for s in srcs):   # no immediates yet
            continue
        pa, pb = A.pouts[0], B.pouts[0]
        if A.pouts[1] != 7 or B.pouts[1] != 7 or (pa == 7) != (pb == 7) or (pa != 7 and pa == pb):
            continue
        # between i and j: nobody writes i's sources or carry in, nobody touches i's carry out
        ok = True
        need_r = {s[1] for s in A.real_srcs() if s[0] == "r"}
        need_p = {c[0] for c in A.real_cins()}
        for k in range(i + 1, j):
            d, u, dp, upd = du[k]
            if (d & need_r) or (dp & need_p) or (pa != 7 and (pa in dp or pa in upd)):
                ok = False
                break
        if not ok:
            continue
        cands[i] = {"i": i, "j": j, "d": B.d, "pouts": (pa, pb), "srcs": srcs, "cins": cins}
        used.add(i)
        used.add(j)
    # carries: each of the two carry outs read exactly once before it is written again, both as carry ins of the
    # same sign, by one add or by the two adds of another pair
    byj = {c["j"]: c for c in cands.values()}
    member = {}
    for c in cands.values():
        member[c["i"]] = c
        member[c["j"]] = c

    def reader(k0, p):
        r = None
        for k in range(k0 + 1, hi + 1):
            if k in block_end:
                return None
            d, u, dp, upd = du[k]
            if p in upd:
                if r is not None:
                    return None
                r = k
            if p in dp:
                return r
        return None

    changed = True
    while changed:
        changed = False
        for c in list(cands.values()):
            pa, pb = c["pouts"]
            if pa == 7:
                continue
            ra, rb = reader(c["i"], pa), reader(c["j"], pb)
            good = False
            if ra is not None and rb is not None and ra > c["j"]:
                Ra, Rb = adds[ra], adds[rb]
                if Ra.ok and Rb.ok:
                    sa = [x for x in Ra.cins if x[0] == pa]
                    sb = [x for x in Rb.cins if x[0] == pb]
                    if len(sa) == 1 and len(sb) == 1 and sa[0][1] == sb[0][1]:
                        if ra == rb:
                            good = True
                        elif member.get(ra) is not None and member.get(ra) is member.get(rb) and ra != rb:
                            good = True
            if not good:
                del cands[c["i"]]
                member.pop(c["i"], None)
                member.pop(c["j"], None)
                changed = True
    return sorted(cands.values(), key=lambda c: c["j"])


def encode_add(c, tmpl):
    """The fused IADD3.X's words from a three-register IADD3.X of the same cubin (fields as sm89/enc.py has them)."""
    w = tmpl["lo"] | (tmpl["hi"] << 64)

    def setf(w, pos, width, val):
        m = ((1 << width) - 1) << pos
        return (w & ~m) | ((val << pos) & m)

    w = setf(w, 12, 4, 7)            # no guard
    w = setf(w, 16, 8, c["d"])
    w = setf(w, 81, 3, c["pouts"][0])
    w = setf(w, 84, 3, c["pouts"][1])
    srcs = list(c["srcs"]) + [("z",)] * (3 - len(c["srcs"]))
    pos, neg = [24, 32, 64], [72, 63, 75]
    for k, s in enumerate(srcs):
        w = setf(w, pos[k], 8, 255 if s[0] == "z" else s[1])
        w = setf(w, neg[k], 1, 1 if (s[0] == "r" and s[2]) else 0)
    cins = list(c["cins"]) + [(7, True)] * (2 - len(c["cins"]))
    for (p, n), (ppos, npos) in zip(cins, [(87, 90), (77, 80)]):
        w = setf(w, ppos, 3, p)
        w = setf(w, npos, 1, 1 if n else 0)
    return w & M64, w >> 64


def text_of(c):
    def src(s):
        if s[0] == "z":
            return "RZ"
        if s[0] == "i":
            return "0x%x" % s[1]
        return s[2] + "R%d" % s[1]
    srcs = [src(s) for s in c["srcs"]] + ["RZ"] * (3 - len(c["srcs"]))
    cins = list(c["cins"]) + [(7, True)] * (2 - len(c["cins"]))
    P = lambda p: "PT" if p == 7 else "P%d" % p
    return "IADD3.X R%d, %s, %s, %s, %s, %s" % (c["d"], P(c["pouts"][0]), P(c["pouts"][1]), ", ".join(srcs),
                                               ("!" if cins[0][1] else "") + P(cins[0][0]),
                                               ("!" if cins[1][1] else "") + P(cins[1][0]))


def bra_words(x, off):
    """A BRA's words with a new distance (bytes from the next instruction). The distance in words of four bytes
    sits in two parts: its low eight bits at bit 16, the rest from bit 34 up."""
    w = x["lo"] | (x["hi"] << 64)
    q = off >> 2
    w &= ~(0xFF << 16)
    w |= (q & 0xFF) << 16
    m = ((1 << 48) - 1) << 34
    w = (w & ~m) | (((q >> 8) << 34) & m)
    return w & M64, w >> 64


def bra_off(x):
    w = x["lo"] | (x["hi"] << 64)
    q = ((w >> 34) & ((1 << 48) - 1))
    if q >> 47:
        q -= 1 << 48
    return ((q << 8) | ((w >> 16) & 0xFF)) << 2


def bssy_words(x, off):
    w = x["lo"] | (x["hi"] << 64)
    m = ((1 << 50) - 1) << 32
    w = (w & ~m) | ((off << 32) & m)
    return w & M64, w >> 64


def apply(nvdisasm, src, dst, kernel=None, limit=None, verbose=False):
    kernel = kernel or carryfix.kernel_name(src)
    ins, labels = listing(nvdisasm, src, kernel)
    tgt = {}
    for k, x in enumerate(ins):
        m = re.search(r"`\((\.L_x_\d+)\)", x["text"])
        if m:
            tgt[k] = labels[m[1]]
            if x["text"].split(" ")[0].startswith("@") or x["text"].startswith("BRA"):
                op = re.sub(r"^@!?P\d ", "", x["text"]).split(" ")[0]
                if op == "BRA" and bra_off(x) != ins[tgt[k]]["addr"] - (x["addr"] + 16):
                    sys.exit("fuse: cannot read the branch at %x" % x["addr"])
    # The region: the widest loop that no branch from outside enters except at its head.
    loops = sorted((tgt[k], k) for k in tgt if tgt[k] <= k and "BRA" in ins[k]["text"] and tgt[k] != k)
    region = None
    for head, tail in sorted(loops, key=lambda l: l[0] - l[1]):
        if all(head <= t <= tail or k == tail for k, t in tgt.items() if head <= k <= tail) and \
           all(not (head < t <= tail) for k, t in tgt.items() if not head <= k <= tail):
            # warp-wide and special instructions keep their offsets: none may be inside
            if not any(re.search(r"\b(VOTE|SHFL|S2R|EXIT|RET|CALL|BSSY|BSYNC|ATOM|BAR)", ins[k]["text"])
                       for k in range(head, tail + 1)):
                region = (head, tail)
                break
    if region is None:
        sys.exit("fuse: no loop to compact")
    head, tail = region
    pairs = find(ins, head, tail - 1)
    if limit is not None:
        pairs = pairs[:limit]
    tmpl = None
    for x in ins:
        if re.fullmatch(r"IADD3\.X R\d+, P\d, PT, R\d+, R\d+, R\d+, P\d, P\d", x["text"].replace(".reuse", "")) or \
           re.fullmatch(r"IADD3\.X R\d+, PT, PT, R\d+, R\d+, R\d+, P\d, P\d", x["text"].replace(".reuse", "")):
            tmpl = x
            break
    if tmpl is None:
        sys.exit("fuse: no three-register IADD3.X to copy the opcode from")
    if not pairs:
        sys.exit("fuse: nothing to fuse")
    dele = {c["i"] for c in pairs}
    byj = {c["j"]: c for c in pairs}
    out = []       # (old index or None, lo, hi, text)
    carry = 0
    newidx = {}
    for k in range(head, tail + 1):
        x = ins[k]
        if k in dele:
            carry += stall(x)
            continue
        newidx[k] = len(out)
        lo, hi, text = x["lo"], x["hi"], x["text"]
        if k in byj:
            c = byj[k]
            lo, hi = encode_add(c, tmpl)
            st = min(15, max(stall(x), stall(ins[c["i"]])) + EXTRA)   # a three-input add may take longer
            hi = (hi & ~(0x1FFFFF << 41)) | (x["hi"] & (0x1FFFFF << 41) & ~(0xF << 58))   # j's control, no reuse
            hi = set_stall(hi, st)
            text = text_of(c)
        if carry:
            # the stall of deleted instructions goes to the instruction before this one
            if not out:
                sys.exit("fuse: the loop's first instruction cannot be deleted")
            p = out[-1]
            s = ((p[2] >> 41) & 15) + carry
            if s > 15:
                s = 15
            out[-1] = (p[0], p[1], set_stall(p[2], s), p[3])
            carry = 0
        out.append((k, lo, hi, text))
    # deleted instructions' indices map to the next kept one
    nxt = None
    for k in range(tail, head - 1, -1):
        if k in newidx:
            nxt = newidx[k]
        else:
            newidx[k] = nxt
    n_free = (tail - head + 1) - len(out)
    base = ins[head]["addr"]
    words = []
    for pos, (k, lo, hi, text) in enumerate(out):
        if k in tgt and k not in byj:
            t = tgt[k]
            if not head <= t <= tail:
                sys.exit("fuse: a branch leaves the region")
            off = (newidx[t] - (pos + 1)) * 16
            lo, hi = bra_words({"lo": lo, "hi": hi}, off)
        words.append((lo, hi, text))
    # the freed slots: a branch over the rest to the instruction after the region, then NOPs
    if n_free:
        skip = dict(ins[tail])
        lo, hi = bra_words(skip, (n_free - 1) * 16)
        lo = (lo & ~(0xF << 12)) | (7 << 12)            # unconditional
        hi = set_stall(hi & ~(0x3F << 52), 5)           # no scoreboard wait
        words.append((lo, hi, None))
        for _ in range(n_free - 1):
            words.append((NOP[0], NOP[1], "NOP"))
    b = bytearray(open(src, "rb").read())
    off, size = carryfix.text_section(b, kernel)
    for pos, (lo, hi, text) in enumerate(words):
        struct.pack_into("<QQ", b, off + base - ins[0]["addr"] + pos * 16, lo, hi)
    open(dst, "wb").write(b)
    # read back
    after, _ = listing(nvdisasm, dst, kernel)
    for pos, (lo, hi, text) in enumerate(words):
        got = after[head + pos]["text"].replace(".reuse", "")
        if text is None:
            continue
        want = re.sub(r"`\(\.L_x_\d+\)", "`", text.replace(".reuse", ""))
        if re.sub(r"`\(\.L_x_\d+\)", "`", got) != want:
            sys.exit("fuse: read back at %x: %r, meant %r" % (after[head + pos]["addr"], got, want))
    if verbose:
        for c in pairs[:40]:
            print("  %s + %s -> %s" % (ins[c["i"]]["text"], ins[c["j"]]["text"], text_of(c)))
    return len(pairs), tail - head + 1


def main():
    a = sys.argv[1:]
    kernel = a[a.index("--kernel") + 1] if "--kernel" in a else None
    limit = int(a[a.index("--max") + 1]) if "--max" in a else None
    n, m = apply(a[0], a[1], a[2], kernel, limit, "--list" in a)
    print("fuse: %d pairs fused, loop of %d instructions now %d" % (n, m, m - n))


if __name__ == "__main__":
    main()
