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


WTAIL = re.compile(r"IMAD\.WIDE\.U32 R(\d+), P(\d), R(\d+), -0x1, R(\d+)")


def find_tails(ins, lo, hi):
    """The reductions (qc_reduce as QC_RED 2 compiles: an IMAD.WIDE by 2^32 - 1 and three adds) as four adds each,
    with no multiply: q5-nv40's fold (quantus-ledger.md, 28 Nov 02:20), reached by rewriting the compiled tail.

    Compiled, for the product's words r0..r3 (r3 a register, or nothing where it is zero):
        W: IMAD.WIDE.U32 Re, Pc, r2, -0x1, Rlo      E = r2 (2^32 - 1) + (r1:r0), Pc its carry
        A: IADD3.X Rc, RZ, r3, RZ, Pc, !PT          c = r3 + Pc           (SEL Rc, RZ, 0x1, !Pc where r3 is zero)
        B: IADD3 Rl, Pb, Re, -Rc, RZ                lo = E0 - c, Pb = no borrow
        C: IADD3.X Rh, RZ, -0x1, Re+1, Pb, Pc       hi = E1 - 1 + Pb + Pc
    With f = 2^32 and f^2 = f - 1 (mod p) the product is (r0 - r2 - r3) + (r1 + r2) f. In the same four places,
    with the same two predicates over the same ranges (L and H are Re and Re+1, or swapped when r2 is Re):
        W: IADD3.X L, Pc, r0, ~r2, RZ, PT, !PT       l = r0 - r2 + f Pc             (Pc = no borrow)
        A: IADD3.X H, Pc, r1, r2, RZ, Pc, !PT        h = r1 + r2 + Pc - f Pc'       (h is the high digit + 1)
        B: IADD3.X Rl, Pb, L, ~r3, RZ, !Pc, !PT      lo = l - r3 - Pc' + f (1 - Pb')... Pb = no borrow
        C: IADD3.X Rh, H, -0x2, RZ, Pc, Pb           hi = h - 2 + Pc' + Pb
    A's carry Pc' is f^2 = f - 1: it comes off the low digit at B and goes on the high one at C. B's borrow comes
    off the high digit at C. (r3 zero: B adds -0x1 in ~r3's place.) What is not carried: C going below zero
    (h + Pc' + Pb < 2: h is 0 or 1 without a carry) and A's sum reaching 2^33; each about once in 2^31 reductions
    or less, the order of qc_reduce's own slip (the host checks every hit).
    Returns ({index: fields} for the four places of every tail that passes the checks, the count seen)."""
    du = [defs_uses(x["text"]) for x in ins]
    out, seen = {}, 0

    def dead_after(k0, reg):
        for k in range(k0 + 1, min(hi, k0 + 600) + 1):
            if du[k] is None or ins[k]["labels"]:
                return False
            d, u, dp, upd = du[k]
            if reg in u:
                return False
            if reg in d:
                return True
        return False

    for i in range(lo, hi + 1):
        m = WTAIL.fullmatch(ins[i]["text"].replace(".reuse", ""))
        if not m:
            continue
        seen += 1
        Re, Pc, r2, Rlo = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        r0, r1 = Rlo, Rlo + 1
        L, H = (Re + 1, Re) if r2 == Re else (Re, Re + 1)
        A = C = None
        end = None
        bad = False
        for k in range(i + 1, min(hi, i + 600) + 1):
            if du[k] is None or ins[k]["labels"]:
                bad = True
                break
            d, u, dp, upd = du[k]
            if Pc in upd:
                if A is None:
                    A = k
                elif C is None:
                    C = k
                else:
                    bad = True
                    break
            if Pc in dp:
                end = k
                break
        if bad or A is None or C is None or (end is not None and end <= C):
            continue
        ta = ins[A]["text"].replace(".reuse", "")
        r3 = None
        ma = re.fullmatch(r"SEL R(\d+), RZ, 0x1, !P%d" % Pc, ta)
        if ma:
            Rc = int(ma[1])
        else:
            ad = Add(ta)
            if not (ad.ok and ad.x and ad.pouts == [7, 7] and ad.real_cins() == [(Pc, False)]
                    and len(ad.real_srcs()) == 1 and ad.real_srcs()[0][0] == "r" and not ad.real_srcs()[0][2]):
                continue
            Rc, r3 = ad.d, ad.real_srcs()[0][1]
        cd = Add(ins[C]["text"])
        if not (cd.ok and cd.x and cd.pouts == [7, 7] and len(cd.real_cins()) == 2
                and sorted(s[:2] for s in cd.real_srcs()) == sorted([("i", 0xFFFFFFFF), ("r", Re + 1)])
                and all(not s[2] for s in cd.real_srcs())):
            continue
        cc = dict(cd.real_cins())
        if Pc not in cc or cc[Pc] or len(cc) != 2:
            continue
        Pb = [p for p in cc if p != Pc][0]
        if cc[Pb]:
            continue
        B = None
        for k in range(C - 1, A, -1):
            if Pb in du[k][2]:
                B = k
                break
        if B is None:
            continue
        bd = Add(ins[B]["text"])
        if not (bd.ok and not bd.x and bd.pouts == [Pb, 7] and len(bd.real_srcs()) == 2
                and ("r", Re, "") in bd.real_srcs() and ("r", Rc, "-") in bd.real_srcs()):
            continue
        # W writes L while A still reads r1 and r2 and B reads r3; A writes H while B reads L and r3
        ok = L not in (r1, r2, r3) and H != r3 and (H == r1 or H not in (r0,)) and bd.d != H
        for k in range(i + 1, C):
            if k in (A, B):
                continue
            d, u, dp, upd = du[k]
            # nobody else reads E, c or the borrow, or writes what the new adds read or hold
            if (u & {Re, Re + 1}) or (d & {Re, Re + 1}) or (k > A and Rc in u) or (Pb in upd and k > B):
                ok = False
            if k < A and (d & {r2, r1}):
                ok = False
            if k < B and r3 is not None and r3 in d:
                ok = False
            if A < k < B and Rc in d:
                ok = False
        if not ok or not dead_after(B, Rc) or any(k in out for k in (i, A, B, C)):
            continue
        R = lambda n, neg="": ("r", n, neg)
        n1 = {"d": L, "pouts": (Pc, 7), "srcs": [R(r0), R(r2, "~")], "cins": [(7, False)]}
        n2 = {"d": H, "pouts": (Pc, 7), "srcs": [R(r1), R(r2)], "cins": [(Pc, False)]}
        n3 = {"d": bd.d, "pouts": (Pb, 7), "srcs": [R(L), R(r3, "~") if r3 is not None else ("i", 0xFFFFFFFF, "")],
              "cins": [(Pc, True)]}
        n4 = {"d": cd.d, "pouts": (7, 7), "srcs": [R(H), ("i", 0xFFFFFFFE, "")], "cins": [(Pc, False), (Pb, False)]}
        out[i], out[A], out[B], out[C] = n1, n2, n3, n4
    return out, seen


def find_red4(ins):
    """QC_RED 4's tails (qc_field.cuh): the indices of the third add of each, whose carry-in must be inverted.

    As compiled, for the product's words r0..r3 (ptxas keeps PTX's carry flag as SASS's carry, so a borrow reads
    as "no borrow"):
        1: IADD3 Rl, Pc, x0, -r2, RZ                 l = r0 - r2, Pc = no borrow
        2: IADD3.X Rh, Pc, r1, r2, RZ, Pc, !PT       h = r1 + r2 + Pc, Pc = its carry c
        3: IADD3.X Ro, Pb, ~r3, Rl, RZ, Pc, !PT      l - r3 - 1 + c: wrong; with !Pc it is l - r3 - c, Pb = no borrow
        4: IADD3.X Rq, RZ, -0x2, Rh, Pb, Pc          h - 2 + Pb + c
    (r3 zero: 3 has -0x1 in ~r3's place.) A tail is taken only when all four are there: 3's carry-in Pc was last
    written by a 2 that both reads and writes Pc, whose Pc came from a 1 subtracting one of 2's own addends and
    writing 3's plain source, and the next reader of Pc after 3 is a 4 reading 3's Pb and 2's sum. Every 4-shaped
    add in the kernel must belong to one, or this stops."""
    du = [defs_uses(x["text"]) for x in ins]
    adds = [Add(x["text"]) for x in ins]
    hits, fours = [], set()

    def prev_writer(k0, pred):
        for k in range(k0 - 1, max(-1, k0 - 600), -1):
            t = ins[k]["text"]
            if du[k] is None:
                # an instruction this tool does not model (a constant load, say): it ends the search only if it
                # is a branch or may write the predicate
                if re.sub(r"^@!?P\d ", "", t).split(" ")[0] in carryfix.STOP or carryfix.writes(t, "P%d" % pred):
                    return None
            elif pred in du[k][2]:
                return k
            if ins[k]["labels"]:
                return None
        return None

    for k3, T in enumerate(adds):
        if not (T.ok and T.x and T.pouts[0] != 7 and T.pouts[1] == 7 and len(T.real_cins()) == 1
                and not T.real_cins()[0][1] and T.cins[0] == T.real_cins()[0] and len(T.real_srcs()) == 2):
            continue
        Pc, Pb = T.real_cins()[0][0], T.pouts[0]
        plain = [s for s in T.real_srcs() if s[0] == "r" and not s[2]]
        other = [s for s in T.real_srcs() if not (s[0] == "r" and not s[2])]
        if len(plain) != 1 or len(other) != 1 or not (other[0] == ("i", 0xFFFFFFFF, "") or
                                                      (other[0][0] == "r" and other[0][2] == "~")):
            continue
        Rl = plain[0][1]
        k2 = prev_writer(k3, Pc)
        if k2 is None:
            continue
        S = adds[k2]
        if not (S.ok and S.x and S.pouts == [Pc, 7] and len(S.real_cins()) == 1 and not S.real_cins()[0][1]
                and len(S.real_srcs()) == 2
                and all(s[0] == "r" and not s[2] for s in S.real_srcs())):
            continue
        P1 = S.real_cins()[0][0]   # 1's carry: often the same predicate as 2's
        k1 = prev_writer(k2, P1)
        if k1 is None:
            continue
        F = adds[k1]
        if not (F.ok and not F.x and F.pouts == [P1, 7] and F.d == Rl and len(F.real_srcs()) == 2):
            continue
        neg = [s for s in F.real_srcs() if s[0] == "r" and s[2] == "-"]
        if len(neg) != 1 or neg[0][1] not in [s[1] for s in S.real_srcs()]:
            continue
        # the next reader of Pc after 3
        k4 = None
        for k in range(k3 + 1, min(len(ins), k3 + 600)):
            t = ins[k]["text"]
            if ins[k]["labels"]:
                break
            if du[k] is None:
                if re.sub(r"^@!?P\d ", "", t).split(" ")[0] in carryfix.STOP or carryfix.reads(t, "P%d" % Pc):
                    break
                continue
            if Pc in du[k][3]:
                k4 = k
                break
            if Pc in du[k][2]:
                break
        if k4 is None:
            continue
        G = adds[k4]
        if not (G.ok and G.x and G.pouts == [7, 7] and sorted(G.real_cins()) == sorted([(Pb, False), (Pc, False)])
                and sorted(s[:2] for s in G.real_srcs()) == sorted([("i", 0xFFFFFFFE), ("r", S.d)])):
            continue
        hits.append(k3)
        fours.add(k4)
    for k, G in enumerate(adds):
        if G.ok and G.x and ("i", 0xFFFFFFFE, "") in G.srcs and len(G.real_cins()) == 2 and k not in fours:
            sys.exit("fuse: the add at %x looks like a QC_RED 4 tail's last, but its tail was not matched"
                     % ins[k]["addr"])
    return hits


def encode_add(c, tmpl, tmpl_imm=None):
    """The IADD3.X's words from an IADD3.X of the same cubin (fields as sm89/enc.py has them): a three-register
    one, or for an immediate source one of the shape register, immediate, register."""
    srcs = list(c["srcs"]) + [("z",)] * (3 - len(c["srcs"]))
    imm = [s for s in srcs if s[0] == "i"]
    if imm:
        if tmpl_imm is None or len(imm) > 1:
            raise ValueError("no template for an immediate source")
        regs = [s for s in srcs if s[0] != "i"]
        srcs = [regs[0], imm[0], regs[1]]
        tmpl = tmpl_imm
    w = tmpl["lo"] | (tmpl["hi"] << 64)

    def setf(w, pos, width, val):
        m = ((1 << width) - 1) << pos
        return (w & ~m) | ((val << pos) & m)

    w = setf(w, 12, 4, 7)            # no guard
    w = setf(w, 16, 8, c["d"])
    w = setf(w, 81, 3, c["pouts"][0])
    w = setf(w, 84, 3, c["pouts"][1])
    pos, neg = [24, 32, 64], [72, 63, 75]
    for k, s in enumerate(srcs):
        if s[0] == "i":
            w = setf(w, 32, 32, s[1])
            continue
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
            return "-0x%x" % ((1 << 32) - s[1]) if s[1] >> 31 else "0x%x" % s[1]
        return s[2] + "R%d" % s[1]
    srcs = list(c["srcs"]) + [("z",)] * (3 - len(c["srcs"]))
    if any(s[0] == "i" for s in srcs):
        regs = [s for s in srcs if s[0] != "i"]
        srcs = [regs[0], [s for s in srcs if s[0] == "i"][0], regs[1]]
    cins = list(c["cins"]) + [(7, True)] * (2 - len(c["cins"]))
    P = lambda p: "PT" if p == 7 else "P%d" % p
    return "IADD3.X R%d, %s, %s, %s, %s, %s" % (c["d"], P(c["pouts"][0]), P(c["pouts"][1]),
                                               ", ".join(src(x) for x in srcs),
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
    """Rewrites the kernel's loop in place: reductions as four adds (find_tails), pairs of adds as one (find), a
    NOP where a pair's first add was. QC_FUSE_MODE (a mask, default 3) picks which: 1 reductions, 2 pairs.
    Returns (reductions rewritten, reductions seen, pairs fused)."""
    kernel = kernel or carryfix.kernel_name(src)
    ins, labels = listing(nvdisasm, src, kernel)
    tgt = {}
    for k, x in enumerate(ins):
        m = re.search(r"`\((\.L_x_\d+)\)", x["text"])
        if m:
            tgt[k] = labels[m[1]]
    loops = sorted((tgt[k], k) for k in tgt if tgt[k] < k and "BRA" in ins[k]["text"])
    region = None
    for head, tail in sorted(loops, key=lambda l: l[0] - l[1]):
        if not any(re.search(r"\b(VOTE|SHFL|S2R|EXIT|RET|CALL|BSSY|BSYNC|ATOM|BAR)", ins[k]["text"])
                   for k in range(head, tail + 1)):
            region = (head, tail)
            break
    if region is None:
        sys.exit("fuse: no loop to rewrite")
    head, tail = region
    mode = int(os.environ.get("QC_FUSE_MODE", "3"))
    tmpl = tmpl_imm = None
    for x in ins:
        t = x["text"].replace(".reuse", "")
        if tmpl is None and re.fullmatch(r"IADD3\.X R\d+, (P\d|PT), PT, R\d+, R\d+, R\d+, P\d, P\d", t):
            tmpl = x
        if tmpl_imm is None and re.fullmatch(r"IADD3\.X R\d+, PT, PT, RZ, -0x[0-9a-f]+, R\d+, P\d, P\d", t):
            tmpl_imm = x
    if tmpl is None:
        sys.exit("fuse: no IADD3.X to copy the opcode from")
    # QC_RED 4's tails: one carry-in inverted each (bit 90, as carryfix.py does for QC_RED 3). Always done.
    red4 = find_red4(ins)
    flip = {}
    for k in red4:
        t = ins[k]["text"]
        m4 = re.search(r", (P\d), !PT$", t)
        if not m4:
            sys.exit("fuse: cannot read the carry-in at %x" % ins[k]["addr"])
        flip[k] = t[:m4.start()] + ", !" + m4[1] + ", !PT"
    ins = [dict(x, text=flip[k], hi=x["hi"] ^ (1 << (90 - 64))) if k in flip else x for k, x in enumerate(ins)]
    repl, seen = {}, 0
    if mode & 1:
        repl, seen = find_tails(ins, head, tail - 1)
    ntails = sum(1 for k in repl if WTAIL.fullmatch(ins[k]["text"].replace(".reuse", "")))
    ins2 = [dict(x, text=text_of(repl[k])) if k in repl else x for k, x in enumerate(ins)]
    pairs = []
    if mode & 2:
        pairs = [c for c in find(ins2, head, tail - 1) if c["i"] not in repl and c["j"] not in repl]
    dele = {c["i"] for c in pairs}
    for c in pairs:
        repl[c["j"]] = c
    b = bytearray(open(src, "rb").read())
    off, size = carryfix.text_section(b, kernel)
    want = {}
    for k in flip:
        struct.pack_into("<QQ", b, off + ins[k]["addr"] - ins[0]["addr"], ins[k]["lo"], ins[k]["hi"])
    for k in range(head, tail):
        x = ins[k]
        if k in dele:
            lo, hi = NOP[0], set_stall(NOP[1], stall(x))
            want[k] = "NOP"
        elif k in repl:
            c = repl[k]
            lo, hi = encode_add(c, tmpl, tmpl_imm)
            st = stall(x)
            if "i" in c:
                st = max(st, stall(ins[c["i"]]))
            if len([s for s in c["srcs"] if s[0] != "z"]) == 3:
                st = min(15, st + EXTRA)       # a three-input add may take longer
            hi = (hi & ~(0x1FFFFF << 41)) | (x["hi"] & (0x1FFFFF << 41) & ~(0xF << 58))   # its control, no reuse
            hi = set_stall(hi, st)
            want[k] = text_of(c)
        else:
            continue
        struct.pack_into("<QQ", b, off + x["addr"] - ins[0]["addr"], lo, hi)
    open(dst, "wb").write(b)
    after, _ = listing(nvdisasm, dst, kernel)
    for k, x in enumerate(ins):
        w = want.get(k, x["text"]).replace(".reuse", "")
        if after[k]["text"].replace(".reuse", "") != w:
            sys.exit("fuse: read back at %x: %r, meant %r" % (x["addr"], after[k]["text"], w))
    if verbose:
        for k in sorted(repl)[:40]:
            print("  %s -> %s" % (ins[k]["text"], want[k]))
    return ntails + len(red4), seen + len(red4), len(pairs)


def main():
    a = sys.argv[1:]
    kernel = a[a.index("--kernel") + 1] if "--kernel" in a else None
    n, seen, pairs = apply(a[0], a[1], a[2], kernel, None, "--list" in a)
    print("fuse: %d of %d reductions rewritten, %d pairs of adds fused" % (n, seen, pairs))


if __name__ == "__main__":
    main()
