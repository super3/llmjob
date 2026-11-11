"""spread.py: moves the IMMA of a compiled draw fold so that no warp is away from the tensor pipe for long.

ptxas orders a k-step for one warp alone: a run of IMMA, then a clump of twenty or thirty draw or readout steps.
With a barrier every k-step the two warps of a scheduler run the same clump at the same moment and the pipe idles.
This tool takes the cubin and its listing and, inside every straight-line stretch that has IMMA in it, moves IMMA
LATER, in among the other steps that follow them, so the stretch reads "IMMA, a few steps, IMMA, a few steps".

What it changes: the place of IMMA instructions, stall counts, and operand-reuse flags. Nothing is added or
removed, every other instruction keeps its order and its scoreboard fields, and IMMA keep their own order.

The rules it keeps (verify() checks them again on the finished order, in code of its own):
  1. An IMMA only moves later, so every scoreboard wait that came before it still does. An IMMA that itself waits
     on a scoreboard, or sets one, does not move (a later step may rely on its wait).
  2. An IMMA never passes a step it shares a register with, so every value still flows from the same writer to
     the same readers (verify() compares the two orders' writer-reader pairs).
  3. Stall distance. For a writer and its reader, a reader and the next writer of that register, and two writers
     in a row, the cycles of stall between them never fall under the least ptxas left between two instructions of
     those two kinds in that relation anywhere in this kernel (and never under 32 from an IMMA to a reader of its
     sums, or 15 for any other pair, when ptxas always left more). ptxas's own least is at or above the latency
     it honours, so the new order honours it too. Shortfalls are made up by raising stalls.
  4. Two IMMA are at least 8 cycles of stall apart, as ptxas has them.
  5. A reuse flag is cleared on every instruction whose successor changed.
  6. Stretch ends, since values cross from one stretch to the next: an instruction that reads or overwrites a
     register nothing earlier in the stretch names is no nearer the start than min(what it was, 32 if the
     register can hold an IMMA's sum, else 15); an IMMA is no nearer the end than min(what it was, 32), and any
     other instruction whose register nothing later names, 15.

  python spread.py IN.cubin IN.sass OUT.cubin MODE [fn-substring]
MODE: stat (print each stretch, write nothing), even (spread evenly by stall time), or even,<n> (at most n cycles
of other steps between two IMMA where the rules allow).
"""
import collections
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctl import ctrl, listing, sections

CTL = ("BRA", "BRX", "JMP", "WARPSYNC", "YIELD", "BSSY", "BSYNC", "BAR", "NANOSLEEP", "CALL", "RET", "BREAK",
       "BMOV", "MEMBAR", "EXIT", "KILL", "TRAP", "JMX", "JCAL", "VOTE", "S2R", "CCTL", "BPT", "RTT")
NO_DEST = ("STS", "ST", "STG", "STL", "RED", "NOP")
FIRST_DEST = ("IADD3", "IADD", "IMAD", "PRMT", "LOP3", "LOP", "PLOP3", "SHF", "SHL", "SHR", "LEA", "MOV", "SEL",
              "CSEL", "IMNMX", "ISETP", "LDS", "LD", "LDG", "LDL", "LDSM", "IMMA", "CS2R", "ATOMS", "ATOM",
              "POPC", "FLO", "BMSK", "SGXT", "IABS", "ISUB")
REG = re.compile(r"(?<![A-Za-z0-9_])(UR|R|UP|P)(\d+)(?![A-Za-z0-9_])")
CAP_PLAIN, CAP_IMMA, IMMA_GAP = 15, 32, 8


def split_guard(text):
    t = text.split(None, 1)
    if t[0].startswith("@"):
        return t[0], t[1]
    return "", text


def operand_regs(part, wide):
    """Registers named in a piece of operand text; a data operand of a wide instruction covers `wide` more."""
    out = set()
    inside = [(m.start(), m.end()) for m in re.finditer(r"\[[^\]]*\]", part)]
    for m in REG.finditer(part):
        cls, num = m.group(1), int(m.group(2))
        in_addr = any(a <= m.start() < e for a, e in inside)
        w = wide if cls == "R" and not in_addr else (1 if in_addr and part[m.end():m.end() + 3] == ".64" else 0)
        for q in range(num, num + w + 1):
            out.add(cls + str(q))
    return out


def analyse(text):
    """(op, full mnemonic, registers written, registers read). Wide operands are widened generously, and an
    instruction of a kind not listed in FIRST_DEST reads its first operand as well as writing it."""
    guard, body = split_guard(text)
    mnem = body.split()[0]
    op = mnem.split(".")[0]
    rest = body[len(mnem):].strip()
    wide = 3 if op in ("IMMA", "LDSM") or "128" in mnem else 1 if ("64" in mnem or "WIDE" in mnem or op == "CS2R") else 0
    if op in ("LDS", "LD", "LDL", "LDG", "STS", "ST", "STL", "STG", "ATOMS", "ATOM", "ATOMG", "RED"):
        wide = 3
    parts = rest.split(",")
    uses = operand_regs(guard, 0)
    defs = set()
    if op in NO_DEST or op in CTL:
        uses |= operand_regs(rest, wide)
    else:
        first = parts[0]
        if "[" in first:
            uses |= operand_regs(first, wide)
        else:
            defs |= operand_regs(first, wide)
            if op not in FIRST_DEST:
                uses |= defs
        for x, p in enumerate(parts[1:]):
            r = operand_regs(p, 1 if (op == "IMMA" and x == 1) else wide)
            uses |= r
            if re.fullmatch(r"\s*P\d+\s*", p):   # a bare predicate operand may be a carry out
                defs |= r
    return op, mnem, defs, uses


class Ins:
    __slots__ = ("addr", "text", "label", "op", "mnem", "c", "regs", "defs", "uses", "lo", "hi")


def read(cubin, sass, fn_sub):
    b = bytearray(open(cubin, "rb").read())
    secs = sections(b)
    name = next(n for n in secs if n.startswith(".text.") and fn_sub in n)
    off, size = secs[name]
    out = []
    for a, t, lab in listing(sass, fn_sub):
        i = Ins()
        i.addr, i.text, i.label = a, t, lab
        i.op, i.mnem, i.defs, i.uses = analyse(t)
        i.regs = i.defs | i.uses
        i.lo, i.hi = struct.unpack_from("<QQ", b, off + a)
        i.c = ctrl(i.hi)
        out.append(i)
    for x in range(len(out) - 1):
        if out[x + 1].addr - out[x].addr != 16:
            sys.exit("instruction at %x is not 16 bytes" % out[x].addr)
    info = secs.get(".nv.info." + name[len(".text."):])
    return b, off, size, out, info


def stretches(ins, least=0):
    """(first, end) index ranges with no control instruction or label inside and at least `least` IMMA."""
    out, start = [], 0
    for x, i in enumerate(ins + [None]):
        if i is None or i.op in CTL or (i.label and x > start):
            if x > start and sum(1 for j in ins[start:x] if j.op == "IMMA") >= least:
                out.append((start, x))
            start = x + 1 if (i is not None and i.op in CTL) else x
    return out


def pairs(ins, order):
    """The (kind, earlier, later) relations of a stretch in the given order: DU a writer and a reader of what it
    wrote, UD a reader and the next writer of that register, DD two writers in a row."""
    out = set()
    last_def, readers = {}, collections.defaultdict(list)
    for x in order:
        i = ins[x]
        for r in i.uses:
            if r in last_def:
                out.add(("DU", last_def[r], x))
        for r in i.defs:
            for y in readers[r]:
                if y != x:
                    out.add(("UD", y, x))
            if r in last_def:
                out.add(("DD", last_def[r], x))
        for r in i.uses:
            readers[r].append(x)
        for r in i.defs:
            last_def[r] = x
            readers[r] = []
    return out


def times(order, stall):
    t, out = 0, {}
    for x in order:
        out[x] = t
        t += stall[x]
    return out, t


def learn(ins):
    """The least stall distance ptxas left for each (kind of relation, earlier mnemonic, later mnemonic)."""
    least = {}
    for lo, hi in stretches(ins):
        order = list(range(lo, hi))
        t, _ = times(order, {x: ins[x].c["stall"] for x in order})
        for kind, y, x in pairs(ins, order):
            key = (kind, ins[y].mnem, ins[x].mnem)
            d = t[x] - t[y]
            if d < least.get(key, 10 ** 9):
                least[key] = d
    return least


def need_of(ins, least, kind, y, x):
    cap = CAP_IMMA if (ins[y].op == "IMMA" and kind == "DU") else CAP_PLAIN
    return min(least[(kind, ins[y].mnem, ins[x].mnem)], cap)


def raise_before(new, ns, p, short):
    """Adds `short` cycles of stall just ahead of position p (15 at most on one instruction). False if no room."""
    q = p - 1
    while short > 0 and q >= 0:
        add = min(short, 15 - ns[new[q]])
        ns[new[q]] += add
        short -= add
        q -= 1
    return short <= 0


def edge_caps(ins, idx, acc):
    """Rule 6's caps: for each instruction, (start cap, end cap)."""
    seen, start_cap = set(), {}
    for x in idx:
        fresh = ins[x].regs - seen
        start_cap[x] = CAP_IMMA if fresh & acc else CAP_PLAIN if fresh else 0
        seen |= ins[x].regs
    seen, end_cap = set(), {}
    for x in reversed(idx):
        end_cap[x] = CAP_IMMA if ins[x].op == "IMMA" else CAP_PLAIN if ins[x].regs - seen else 0
        seen |= ins[x].regs
    return start_cap, end_cap


def plan(ins, lo, hi, mode, least, acc):
    """The new order of ins[lo:hi] (a list of indices), the new stall of each, and an error or None.

    A merge in time of two queues, the IMMA and the other steps, each in its own order. Every instruction gets an
    issue time no earlier than its rules allow; an IMMA is taken when its turn has come (one every T cycles, T the
    stretch's stall time over its IMMA count, 8 at least), another step whenever one fits before that."""
    idx = list(range(lo, hi))
    imma = [x for x in idx if ins[x].op == "IMMA"]
    other = [x for x in idx if ins[x].op != "IMMA"]
    stall = {x: ins[x].c["stall"] for x in idx}
    if mode == "stat" or not imma:
        return idx, stall, None
    ot, oend = times(idx, stall)
    preds = collections.defaultdict(list)
    for kind, y, x in pairs(ins, idx):
        preds[x].append((y, need_of(ins, least, kind, y, x)))
    start_cap, end_cap = edge_caps(ins, idx, acc)
    at, n = {}, 0   # an IMMA sits before other[at]
    for x in idx:
        if ins[x].op == "IMMA":
            at[x] = n
        else:
            n += 1
    cum = [0]
    for x in other:
        cum.append(cum[-1] + stall[x])
    total = cum[-1]
    limit = {}
    for x in imma:
        c = ins[x].c
        if c["wait"] or c["wsb"] != 7 or c["rsb"] != 7:
            limit[x] = at[x]
            continue
        limit[x] = next((p for p in range(at[x], len(other)) if ins[other[p]].regs & ins[x].regs), len(other))
        need = min(oend - ot[x], CAP_IMMA)   # rule 6, the end, counted in the other steps' stall behind it
        while limit[x] > at[x] and total - cum[limit[x]] < need:
            limit[x] -= 1
    for j in range(len(imma) - 2, -1, -1):   # IMMA keep their order
        limit[imma[j]] = min(limit[imma[j]], limit[imma[j + 1]])
    period = float(mode.split(",")[1]) if mode.startswith("even,") else max(IMMA_GAP, (total + len(imma)) / len(imma))
    new, issue, ns = [], {}, dict(stall)
    io = ii = 0
    last_imma = None
    while io < len(other) or ii < len(imma):
        prev = new[-1] if new else None

        def earliest(x):
            e = 0 if prev is None else issue[prev] + (1 if ins[prev].op == "IMMA" else stall[prev])
            for y, need in preds[x]:
                e = max(e, issue[y] + need)
            e = max(e, min(ot[x], start_cap[x]))
            if ins[x].op == "IMMA" and last_imma is not None:
                e = max(e, issue[last_imma] + IMMA_GAP)
            return e

        o = other[io] if io < len(other) else None
        im = imma[ii] if ii < len(imma) else None
        can_i = im is not None and io >= at[im]
        must_i = im is not None and io >= limit[im]
        can_o = o is not None and not must_i
        if not can_i and not can_o:
            return None, None, "stuck at %x" % ins[im if im is not None else o].addr
        ei = earliest(im) if can_i else None
        eo = earliest(o) if can_o else None
        if can_i and can_o:
            due = ii * period
            pick = o if (eo < due or eo + stall[o] <= ei) else im
        else:
            pick = im if can_i else o
        e = ei if pick == im else eo
        if prev is not None and e - issue[prev] > 15:
            alt = o if pick == im else im
            ea = eo if pick == im else ei
            if ea is not None and ea - issue[prev] <= 15:
                pick, e = alt, ea
            else:
                return None, None, "a wait of %d cycles ahead of %x" % (e - issue[prev], ins[pick].addr)
        if prev is not None:
            ns[prev] = e - issue[prev]
        issue[pick] = e
        new.append(pick)
        if pick == im:
            ii += 1
            last_imma = pick
        else:
            io += 1
    lastx = new[-1]
    ns[lastx] = stall[lastx] if ins[lastx].op != "IMMA" else max(1, stall[lastx] if lastx == idx[-1] else 1)
    # Rule 6, the end.
    nt, nend = times(new, ns)
    short = max([min(oend - ot[x], end_cap[x]) - (nend - nt[x]) for x in idx] + [0])
    if short > 0:
        if ns[lastx] + short > 15:
            return None, None, "no room at the end"
        ns[lastx] += short
    return new, ns, None


def verify(ins, lo, hi, new, ns, least, acc):
    """The rules, checked on the finished order by plain counting. Returns a list of faults."""
    bad = []
    idx = list(range(lo, hi))
    if sorted(new) != idx:
        return ["not a permutation"]
    stall = {x: ins[x].c["stall"] for x in idx}
    npos = {x: p for p, x in enumerate(new)}
    if [x for x in new if ins[x].op != "IMMA"] != [x for x in idx if ins[x].op != "IMMA"]:
        bad.append("other steps changed order")
    if [x for x in new if ins[x].op == "IMMA"] != [x for x in idx if ins[x].op == "IMMA"]:
        bad.append("IMMA changed order")
    ot, oend = times(idx, stall)
    nt, nend = times(new, ns)
    rel_old, rel_new = pairs(ins, idx), pairs(ins, new)
    if rel_old != rel_new:
        bad.append("%d writer-reader relations differ" % len(rel_old ^ rel_new))
    for kind, y, x in rel_new:
        need = need_of(ins, least, kind, y, x)
        if nt[x] - nt[y] < need:
            bad.append("%s %x to %x: %d cycles, needs %d" % (kind, ins[y].addr, ins[x].addr, nt[x] - nt[y], need))
    for y in idx:
        for x in idx:
            if y < x and ins[y].regs & ins[x].regs and npos[y] > npos[x]:
                bad.append("%x and %x share a register and changed order" % (ins[y].addr, ins[x].addr))
    start_cap, end_cap = edge_caps(ins, idx, acc)
    for x in idx:
        i = ins[x]
        if ns[x] < 1 or ns[x] > 15:
            bad.append("stall %d at %x" % (ns[x], i.addr))
        if i.op != "IMMA" and ns[x] < stall[x]:
            bad.append("stall lowered at %x" % i.addr)
        if nt[x] < min(ot[x], start_cap[x]):
            bad.append("%x nearer the start" % i.addr)
        if nend - nt[x] < min(oend - ot[x], end_cap[x]):
            bad.append("%x nearer the end" % i.addr)
        if i.op == "IMMA":
            before_old = sum(1 for y in idx[:x - lo] if ins[y].op != "IMMA")
            before_new = sum(1 for y in new[:npos[x]] if ins[y].op != "IMMA")
            if before_new < before_old:
                bad.append("IMMA %x moved earlier" % i.addr)
            if before_new != before_old and (i.c["wait"] or i.c["wsb"] != 7 or i.c["rsb"] != 7):
                bad.append("IMMA %x with a scoreboard moved" % i.addr)
    im = [x for x in new if ins[x].op == "IMMA"]
    for a, b2 in zip(im, im[1:]):
        if nt[b2] - nt[a] < IMMA_GAP:
            bad.append("IMMA %x and %x %d apart" % (ins[a].addr, ins[b2].addr, nt[b2] - nt[a]))
    return bad


def gaps(ins, order, stall):
    """Cycles of other steps' stall between consecutive IMMA of a stretch."""
    out, run, seen = [], 0, False
    for x in order:
        if ins[x].op == "IMMA":
            if seen:
                out.append(run)
            run, seen = 0, True
        else:
            run += stall[x]
    return out


def main():
    cubin, sass, dst, mode = sys.argv[1:5]
    fn_sub = sys.argv[5] if len(sys.argv) > 5 else "pearl"
    b, off, size, ins, info = read(cubin, sass, fn_sub)
    least = learn(ins)
    acc = set()
    for i in ins:
        if i.op == "IMMA":
            acc |= i.defs
    moved = raised = cleared = 0
    moved_addr = set()
    for lo, hi in stretches(ins, 4):
        new, ns, err = plan(ins, lo, hi, mode, least, acc)
        if not err and mode != "stat":
            faults = verify(ins, lo, hi, new, ns, least, acc)
            if faults:
                err = "verify: " + "; ".join(faults[:4])
        if err:
            print("stretch %x-%x left alone: %s" % (ins[lo].addr, ins[hi - 1].addr, err))
            continue
        old = list(range(lo, hi))
        g0 = gaps(ins, old, {x: ins[x].c["stall"] for x in old})
        g1 = gaps(ins, new, ns)
        n_imma = sum(1 for x in old if ins[x].op == "IMMA")
        print("stretch %05x-%05x: %3d IMMA %3d others, stall %4d -> %4d, longest gap %3d -> %3d, over 16: %2d -> %2d"
              % (ins[lo].addr, ins[hi - 1].addr, n_imma, hi - lo - n_imma, sum(ins[x].c["stall"] for x in old),
                 sum(ns[x] for x in new), max(g0 or [0]), max(g1 or [0]), sum(1 for g in g0 if g > 16),
                 sum(1 for g in g1 if g > 16)))
        if mode == "stat":
            continue
        for p, x in enumerate(new):
            c = (ins[x].hi >> 41) & 0x1FFFFF
            c = (c & ~15) | ns[x]
            nxt_old = x + 1 if x + 1 < hi else None
            nxt_new = new[p + 1] if p + 1 < len(new) else None
            if nxt_new != nxt_old and c >> 17:
                c &= 0x1FFFF
                cleared += 1
            hi_w = (ins[x].hi & ~(0x1FFFFF << 41)) | (c << 41)
            struct.pack_into("<QQ", b, off + ins[lo].addr + 16 * p, ins[x].lo, hi_w)
            if x != lo + p:
                moved += 1
                moved_addr.add(ins[x].addr)
                moved_addr.add(ins[lo].addr + 16 * p)
            raised += ns[x] > ins[x].c["stall"] and ins[x].op != "IMMA"
    if mode != "stat":
        warn = 0
        if info:
            io, isz = info
            for q in range(io, io + isz - 3, 4):
                if struct.unpack_from("<I", b, q)[0] in moved_addr:
                    warn += 1
        open(dst, "wb").write(b)
        print("%s: %d instructions moved, %d plain stalls raised, %d reuse flags cleared%s" % (
            mode, moved, raised, cleared, ("; WARNING: .nv.info names %d moved addresses" % warn) if warn else ""))


if __name__ == "__main__":
    main()
