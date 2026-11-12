"""sched120.py: the rotation fold's chunk loop for sm_120 (RTX 50), scheduled again by a rule of our own.

  python sched120.py <in.cubin> <nvdisasm> <out.cubin> [key=value ...]

ptxas compiles the C fold (src/pearl_rot_c.inc) for sm_120 into a loop that holds 84 to 92% of the tensor pipe on
an RTX 5090. On sm_120 a plain step costs the pipe when several sit in one gap between two IMMA, and ptxas packs
them (probes/README.md and the nvx2 notes: one step a gap is free, two cost a little, seven or eight cost much).
This keeps every instruction ptxas wrote, with its registers, and only moves instructions inside the loop and
writes their control fields again (stall counts, yield, scoreboards and waits; operand reuse flags are cleared).
No instruction is assembled, so nothing here depends on sm_120's encodings beyond the control field, whose layout
asm89.py check found to be sm_89's (bits 105 to 125).

How the loop is cut. The loop is the stretch from the backward branch's target to the branch. Its barriers (the
band's bar.sync after each k-step) split it into regions; nothing moves across a barrier or the branch. Inside a
region the IMMA keep their order and every other step goes into the gaps between them: a step is given the
latest gap its first IMMA reader allows (going back through the steps it feeds, a gap a hop, two for a load),
and the gaps are filled in order, the most urgent first, `per` steps a gap at most unless something is due.

What keeps the result right, checked again on the final code (a walk over the prologue, two passes of the loop,
and the code after it):
  - registers: every register an instruction names is read; the first operand of most is written (both, when the
    instruction is predicated). Width from the form (IMMA 4/4/2/4, .128 4, .64 and CS2R 2, desc[URn] 2).
  - memory: a shared store keeps its order with every shared access, local ones among themselves, a barrier with
    everything. Global loads only read.
  - fixed latency (an instruction ptxas gave no write scoreboard): a reader issues LAT cycles or more behind its
    writer (sums of stall counts; the floors are what ptxas itself never goes below on sm_120, plus margin), and
    an ALU step that overwrites an IMMA's operand issues WAR_IMMA cycles or more behind the IMMA.
  - variable latency (loads, S2R): the writer sets a scoreboard; a reader, an overwriter of its result, and an
    overwriter of an address or a stored value of a load or store still under way wait on it. At the branch every
    scoreboard is waited for, so nothing is under way across the loop's edge.
"""
import collections
import re
import struct
import subprocess
import sys

CFG = dict(per=2, gap=12, ist=1, step=2, lat_alu=6, lat_imma=30, lat_pred=13, lat_u=13, war_imma=10,
           ld=32, yld=1, sbs=6, block=1, slack=4)
LAT_LOAD_HOP = {"LDSM": 2, "LDS": 2, "LDG": 6, "LDL": 2, "S2R": 2}
REG = re.compile(r"(?<![A-Za-z0-9_])(-|!|~|\|)?(UR|R|UP|P)(\d+)((?:\.[A-Z0-9]+)*)")
CTRL_SHIFT = 41


def sections(b):
    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    entsize, num, strndx = struct.unpack_from("<HHH", b, 0x3A)
    raw = [struct.unpack_from("<IIQQQQ", b, shoff + i * entsize) for i in range(num)]
    stro = raw[strndx][4]
    out = {}
    for name, _type, _flags, _addr, off, size in raw:
        end = b.index(b"\0", stro + name)
        out[bytes(b[stro + name:end]).decode()] = (off, size)
    return out


class Ins:
    def __init__(self, sect, addr, text, lo, hi):
        self.sect, self.addr, self.text, self.lo, self.hi = sect, addr, text, lo, hi
        c = (hi >> CTRL_SHIFT) & 0x1FFFFF
        self.stall, self.yld, self.wsb, self.rsb, self.wait = c & 15, (c >> 4) & 1, (c >> 5) & 7, (c >> 8) & 7, (c >> 11) & 63
        g = re.match(r"^@(!?)(U?P[T\d]+)\s+(.*)$", text)
        self.guard = g.group(2) if g and g.group(2) not in ("PT", "UPT") else None
        body = g.group(3) if g else text
        parts = body.split(None, 1)
        self.mnem = parts[0]
        self.op = self.mnem.split(".")[0]
        self.ops = parts[1] if len(parts) > 1 else ""
        self.parse()

    def parse(self):
        m, op = self.mnem, self.op
        toks = []
        # split operands at top-level commas
        depth, cur = 0, ""
        for ch in self.ops:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            if ch == "," and depth == 0:
                toks.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            toks.append(cur.strip())
        self.toks = toks
        dst, src = set(), set()

        def regs_of(tok, width_default=1, pos=None):
            out = set()
            inside = "[" in tok
            for mm in REG.finditer(tok):
                cls, n, suf = mm.group(2), int(mm.group(3)), mm.group(4) or ""
                if cls == "R" and n == 255 or cls == "UR" and n == 63 or cls in ("P", "UP") and n == 7:
                    continue
                w = 1
                if ".64" in suf:
                    w = 2
                elif not inside or cls == "UR":
                    w = width_default
                if "desc[" in tok and cls == "UR":
                    w = 2
                for q in range(n, n + w):
                    out.add((cls, q))
            return out

        wide = 4 if (op in ("LDSM",) or ".128" in m) else 2 if (".64" in m or op == "CS2R" or "WIDE" in m) else 1
        if op == "IMMA":
            ws = [4, 4, 2, 4]
            for i, t in enumerate(toks):
                s = regs_of(t, ws[i] if i < 4 else 1)
                (dst if i == 0 else src).update(s)
        elif op in ("ST", "STS", "STG", "STL", "BAR", "BRA", "RET", "EXIT", "NOP", "BSYNC", "WARPSYNC", "RED", "REDG"):
            for i, t in enumerate(toks):
                w = wide if ("[" not in t and op.startswith("ST")) else 1
                src |= regs_of(t, w)
        else:
            for i, t in enumerate(toks):
                if i == 0:
                    w = wide
                    if op in ("IMAD",) and "WIDE" in m:
                        w = 2
                    dst |= regs_of(t, w)
                elif i in (1, 2) and op in ("IADD3", "LEA", "ISETP", "UISETP", "IADD", "UIADD3", "LOP3", "ULOP3",
                                            "PLOP3", "IMAD", "FSETP", "DSETP") and re.fullmatch(r"!?U?P\d", t):
                    dst |= regs_of(t)
                    src |= regs_of(t)
                else:
                    w = 2 if (op in ("IMAD",) and "WIDE" in m and i == 3) or (".64" in m and op in ("IADD", "MOV", "UMOV")) else 1
                    src |= regs_of(t, w)
        if self.guard:
            src.add((("UP" if self.guard.startswith("UP") else "P"), int(re.sub(r"\D", "", self.guard))))
            src |= dst              # a predicated write may leave the old value
        self.dst, self.src = dst, src
        if op in ("LDS", "LDSM"):
            self.mem = "sld"
        elif op == "STS":
            self.mem = "sst"
        elif op in ("LDL", "STL"):
            self.mem = "loc"
        elif op in ("LDG",):
            self.mem = "gld"
        elif op in ("BAR",):
            self.mem = "bar"
        else:
            self.mem = None
        self.var = op in ("LDS", "LDSM", "LDG", "LDL", "S2R", "LDC", "LDCU", "S2UR", "CS2R") and op != "CS2R"
        if op == "CS2R" and "SR_CLOCK" in self.ops:
            self.var = False
        self.holds = op in ("LDS", "LDSM", "LDG", "LDL", "STS", "STL", "STG")   # reads its registers late

    def lat(self, cfg):
        if self.op == "IMMA":
            return cfg["lat_imma"]
        if any(c in ("P", "UP") for c, _ in self.dst):
            return cfg["lat_pred"]
        if self.op.startswith("U") or any(c == "UR" for c, _ in self.dst):
            return cfg["lat_u"]
        return cfg["lat_alu"]

    def ctrl(self):
        return (self.stall & 15) | (self.yld << 4) | (self.wsb << 5) | (self.rsb << 8) | (self.wait << 11)

    def encode(self):
        hi = self.hi & ~(0x1FFFFF << CTRL_SHIFT)
        return struct.pack("<QQ", self.lo, hi | (self.ctrl() << CTRL_SHIFT))

    def __repr__(self):
        return "%05x %s" % (self.addr, self.text)


def read(cubin, nvd):
    out = subprocess.run([nvd, "-hex", "-c", cubin], capture_output=True, text=True).stdout.splitlines()
    ins, labels, sect = [], {}, None
    pat = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;?\s*/\* (0x[0-9a-f]{16}) \*/")
    pend = []
    for i, ln in enumerate(out):
        ms = re.match(r"\s*\.section\s+(\.text\.[^,\s]+)", ln)
        if ms:
            sect = ms.group(1)
        ml = re.match(r"\s*(\.L_x_\d+):", ln)
        if ml:
            pend.append(ml.group(1))
        m = pat.search(ln)
        if m and sect:
            hi = int(re.search(r"/\* (0x[0-9a-f]{16}) \*/", out[i + 1]).group(1), 16)
            x = Ins(sect, int(m.group(1), 16), m.group(2).strip().rstrip(";").strip(), int(m.group(3), 16), hi)
            for p in pend:
                labels[(sect, p)] = len(ins)
            pend = []
            ins.append(x)
    return ins, labels


def find_loop(ins, labels):
    best = None
    for i, x in enumerate(ins):
        if x.op != "BRA":
            continue
        m = re.search(r"`\((\.L_x_\d+)\)", x.text)
        if not m or (x.sect, m.group(1)) not in labels:
            continue
        t = labels[(x.sect, m.group(1))]
        if t < i:
            n = sum(1 for y in ins[t:i] if y.op == "IMMA")
            if n and (best is None or n > best[2]):
                best = (t, i, n)
    return best


def deps(seq):
    """pred[j] = set of i < j that j must follow (registers, memory, barriers)."""
    pred = [set() for _ in seq]
    lastw, reads, mem_last_st, mem_all, loc = {}, collections.defaultdict(list), None, [], []
    for j, x in enumerate(seq):
        for r in x.src:
            if r in lastw:
                pred[j].add(lastw[r])
        for r in x.dst:
            if r in lastw:
                pred[j].add(lastw[r])
            for i in reads[r]:
                pred[j].add(i)
        if x.mem in ("sld", "sst"):
            if x.mem == "sst":
                pred[j].update(mem_all)
            elif mem_last_st is not None:
                pred[j].add(mem_last_st)
            mem_all.append(j)
            if x.mem == "sst":
                mem_last_st = j
        if x.mem == "loc":
            pred[j].update(loc)
            loc.append(j)
        for r in x.src:
            reads[r].append(j)
        for r in x.dst:
            lastw[r] = j
            reads[r] = []
        pred[j].discard(j)
    return pred


def schedule_region(seq, cfg):
    """Order one region (no barrier inside). IMMA keep their order; steps go into the gaps between them. A rough
    clock of the warp's own issue (an IMMA's gap is cfg gap cycles, a step cfg step) says when a step's inputs
    are there: a fixed-latency input LAT cycles behind its writer, a load's ld cycles. Of the steps whose inputs
    are placed, those ready by the clock go first, the most urgent (by `late`) first."""
    n = len(seq)
    pred = deps(seq)
    succ = [set() for _ in seq]
    for j in range(n):
        for i in pred[j]:
            succ[i].add(j)
    imma = [j for j in range(n) if seq[j].op == "IMMA"]
    if not imma:
        return list(range(n))
    gi = {j: k for k, j in enumerate(imma)}
    INF = 10 ** 6
    # latest gap: a step must be in a gap before its IMMA reader's index (gap k is after IMMA k; -1 before all)
    late = {}
    for j in reversed(range(n)):
        if j in gi:
            continue
        x = seq[j]
        best = INF
        for s in succ[j]:
            if s in gi:
                hop = (LAT_LOAD_HOP.get(x.op, 2) if x.var else 1) if j in pred[s] else 0
                best = min(best, gi[s] - hop)
            else:
                hop = (LAT_LOAD_HOP.get(x.op, 2) if x.var else 0)
                best = min(best, late.get(s, INF) - hop)
        late[j] = best
    done, out = set(), []
    tm = {}
    now = 0

    def ready(j):
        return all(i in done for i in pred[j])

    def when(j):
        r = 0
        for i in pred[j]:
            x = seq[i]
            if x.var:
                r = max(r, tm[i] + (cfg["ld"] if j in succ[i] and (x.dst & (seq[j].src | seq[j].dst)) else 0))
            elif x.dst & seq[j].src:
                r = max(r, tm[i] + x.lat(cfg))
        return r

    def place(j, cost):
        nonlocal now
        tm[j] = max(now, when(j))
        now = tm[j] + cost
        out.append(j)
        done.add(j)

    nimma = len(imma)
    remaining = [j for j in range(n) if j not in gi]
    last_imma = -INF
    for g in range(-1, nimma):
        if g >= 0:
            j = imma[g]
            # whatever the IMMA needs and is not placed yet goes first, with what it needs (in order)
            need, stack = set(), [i for i in pred[j] if i not in done]
            while stack:
                i = stack.pop()
                if i not in need:
                    need.add(i)
                    stack += [q for q in pred[i] if q not in done]
            for i in sorted(need):
                place(i, cfg["step"])
                remaining.remove(i)
            now = max(now, last_imma + cfg["gap"])
            place(j, cfg["ist"])
            last_imma = tm[j]
        gaps_left = nimma - g
        quota = cfg["per"] if g >= 0 else 0
        last_gap = g == nimma - 1
        took = 0
        while True:
            cand = [j for j in remaining if ready(j)]
            if not cand:
                break
            if last_gap:
                cand.sort(key=lambda j: (max(when(j), now), late[j], j))
            else:
                cand.sort(key=lambda j: (when(j) > now, late[j], j))
            j = cand[0]
            forced = late[j] <= g
            behind = len(remaining) > cfg["per"] * (gaps_left - 1)
            if not (took < quota or forced or behind or last_gap):
                break
            if not forced and not last_gap and when(j) > now + cfg["slack"]:
                break
            place(j, cfg["step"])
            remaining.remove(j)
            took += 1
    assert len(out) == n and sorted(out) == list(range(n)), (len(out), n)
    return out


def assign(code, cfg, entry_pending, block_rule=True):
    """Control fields for the loop's code in its new order: stalls, yields, scoreboards, waits."""
    INF = 10 ** 9
    # scoreboards: give each variable-latency instruction the one with the fewest others still under way
    for x in code:
        x.wsb = x.rsb = 7
        x.wait = 0
        x.stall = 0
        x.yld = cfg["yld"]
        x.lo = x.lo
        x.hi &= ~(0xF << (122 - 64))        # operand reuse flags off
    for k, x in enumerate(code):
        if x.op == "IMMA":
            x.stall = cfg["ist"]
        elif x.op == "BAR":
            x.stall = 1
        else:
            x.stall = cfg["step"]
    # IMMA's stall fills its gap to cfg["gap"] cycles
    k = 0
    while k < len(code):
        if code[k].op == "IMMA":
            m = k + 1
            while m < len(code) and code[m].op not in ("IMMA", "BAR", "BRA"):
                m += 1
            used = sum(code[q].stall for q in range(k + 1, m))
            code[k].stall = max(cfg["ist"], min(15, cfg["gap"] - used))
            k = m
        else:
            k += 1
    # variable-latency writers and holders
    nsb = cfg["sbs"]
    out_sb = collections.defaultdict(list)    # sb -> list of instruction indices still possibly under way
    pend_w = dict(entry_pending)               # register -> sb of a load still writing it
    pend_r = collections.defaultdict(set)      # register -> sbs of loads/stores still reading it
    rr = 0
    for k, x in enumerate(code):
        waits = set()
        for r in x.src:
            if r in pend_w:
                waits.add(pend_w[r])
        for r in x.dst:
            if r in pend_w:
                waits.add(pend_w[r])
            waits |= pend_r.get(r, set())
        if x.op == "BRA":
            waits = {s for s in range(nsb) if out_sb[s]} | set(pend_w.values())
            for rs in pend_r.values():
                waits |= rs
        for s in waits:
            out_sb[s] = []
        if waits:
            x.wait = sum(1 << s for s in waits)
            for r in list(pend_w):
                if pend_w[r] in waits:
                    del pend_w[r]
            for r in list(pend_r):
                pend_r[r] -= waits
                if not pend_r[r]:
                    del pend_r[r]
        if x.var or x.holds:
            # the scoreboard with nothing under way, else the one whose newest is oldest
            free = [s for s in range(nsb) if not out_sb[s]]
            if free:
                s = free[rr % len(free)]
                rr += 1
            else:
                s = min(range(nsb), key=lambda q: out_sb[q][-1])
            out_sb[s].append(k)
            if x.var and x.dst:
                x.wsb = s
                for r in x.dst:
                    pend_w[r] = s
                if x.holds:
                    for r in x.src:
                        pend_r[r].add(s)
            else:
                x.rsb = s
                for r in x.src:
                    pend_r[r].add(s)
    return code


def check_fix(stream, cfg, fixable, fix=True):
    """Walk a straight stream of Ins (prologue tail, loop twice, epilogue head) summing stalls. A reader too early
    for a fixed-latency writer gets the stalls of the instructions from the writer to it raised (latest first,
    each up to 15) where they are in `fixable` (ids); what cannot be fixed is returned."""
    last = {}            # reg -> (time, writer, index)
    imma_reads = {}      # reg -> (time, index) an IMMA read it
    bad = []
    t = 0
    for k, x in enumerate(stream):
        need, lo = 0, k
        for r in x.src:
            if r in last and not last[r][1].var:
                tw, w, iw = last[r]
                if id(w) not in fixable and id(x) not in fixable:
                    continue            # ptxas's own code before or after the loop
                d = w.lat(cfg) - (t - tw)
                if d > 0:
                    need = max(need, d)
                    lo = min(lo, iw)
        if x.op != "IMMA" and not x.var:
            for r in x.dst:
                if r in imma_reads:
                    tr, ir = imma_reads[r]
                    d = cfg["war_imma"] - (t - tr)
                    if d > 0:
                        need = max(need, d)
                        lo = min(lo, ir)
        if need > 0:
            q = k - 1
            while fix and need > 0 and q >= lo:
                if id(stream[q]) in fixable and stream[q].stall < 15:
                    add = min(need, 15 - stream[q].stall)
                    stream[q].stall += add
                    t += add
                    need -= add
                q -= 1
            if need > 0:
                bad.append((k, x, need))
        if x.op == "IMMA":
            for r in x.src:
                imma_reads[r] = (t, k)
        for r in x.dst:
            last[r] = (t, x, k)
            imma_reads.pop(r, None) if x.op != "IMMA" else None
        t += x.stall
    return bad


def main():
    src, nvd, dst = sys.argv[1:4]
    cfg = dict(CFG)
    for kv in sys.argv[4:]:
        k, v = kv.split("=", 1)
        if k not in cfg:
            sys.exit("unknown key " + k)
        cfg[k] = int(v)
    b = bytearray(open(src, "rb").read())
    secs = sections(b)
    ins, labels = read(src, nvd)
    lp = find_loop(ins, labels)
    if not lp:
        sys.exit("no loop with IMMA")
    t0, tb, nimma = lp
    sect = ins[tb].sect
    body = ins[t0:tb]          # the branch stays where it is, last
    bra = ins[tb]
    nops = [x for x in body if x.op == "NOP"]
    body = [x for x in body if x.op != "NOP"]
    # regions split at barriers
    regions, cur = [], []
    for x in body:
        if x.op == "BAR":
            regions.append(("code", cur))
            regions.append(("bar", [x]))
            cur = []
        else:
            cur.append(x)
    regions.append(("code", cur))
    new = []
    for kind, seq in regions:
        if kind == "bar" or not seq:
            new += seq
            continue
        order = schedule_region(seq, cfg)
        new += [seq[i] for i in order]
    # the BLOCK rule: the step right after an IMMA names no register in the 8-register blocks of its sums
    swaps = 0
    if cfg["block"]:
        for k in range(len(new) - 1):
            a = new[k]
            if a.op != "IMMA":
                continue
            blk = {n // 8 for c, n in a.dst if c == "R"}
            nx = new[k + 1]
            if nx.op in ("IMMA", "BAR") or not any(c == "R" and n // 8 in blk for c, n in nx.src | nx.dst):
                continue
            # swap with a later step of the gap that is free of it and of what lies between
            for m in range(k + 2, len(new)):
                y = new[m]
                if y.op in ("IMMA", "BAR"):
                    break
                if any(c == "R" and n // 8 in blk for c, n in y.src | y.dst):
                    continue
                between = new[k + 1:m]
                if any(deps([z, y])[1] for z in between):
                    continue
                new.insert(k + 1, new.pop(m))
                swaps += 1
                break
    # the code must fill exactly the loop's old length: what is left is NOPs, put behind IMMA (a NOP takes a cycle
    # of the IMMA's stall)
    nop_proto = nops[0] if nops else None
    room = (tb - t0) - len(new)
    if room and not nop_proto:
        sys.exit("internal: room without a NOP")
    if room:
        immas = [k for k, x in enumerate(new) if x.op == "IMMA"]
        step = max(1, len(immas) // room)
        at = sorted(immas[::step][:room], reverse=True)
        for k in at:
            nx = Ins(nop_proto.sect, 0, "NOP", nop_proto.lo, nop_proto.hi)
            new.insert(k + 1, nx)
        while len(new) < tb - t0:
            new.append(Ins(nop_proto.sect, 0, "NOP", nop_proto.lo, nop_proto.hi))
    assert len(new) == tb - t0
    code = new + [bra]
    # pending loads at the loop's entry: the prologue's (from its own scoreboards), as a reader would see them
    pro = ins[:t0]
    entry = {}
    for x in pro:
        if x.sect != sect:
            continue
        for s in range(6):
            if x.wait >> s & 1:
                entry = {r: v for r, v in entry.items() if v != s}
        if x.wsb != 7:
            for r in x.dst:
                entry[r] = x.wsb
    assign(code, cfg, {})
    # the prologue's loads and stores: the loop's first instruction waits for every scoreboard (a later pass finds
    # them all idle: the branch waited for every one)
    code[0].wait = 63
    for x in code:
        if x.op == "NOP":
            x.stall = 1
    # NOPs take their cycle out of the IMMA's stall before them
    for k, x in enumerate(code):
        if x.op == "NOP" and k and code[k - 1].op == "IMMA" and code[k - 1].stall > cfg["ist"]:
            code[k - 1].stall -= 1
    # entry waits: a body reader of a register the prologue's load writes waits on it (assign did); the first pass
    # also waits there, which is all a later pass's wait costs.
    # fixed latency over prologue -> loop -> loop -> epilogue
    fixable = {id(x) for x in code}
    pro_sect = [x for x in pro if x.sect == sect][-40:]
    epi = [x for x in ins[tb + 1:tb + 40] if x.sect == sect]
    for it in range(6):
        stream = pro_sect + code + code + epi
        bad = check_fix(stream, cfg, fixable, fix=True)
        if not bad:
            break
    bad = check_fix(pro_sect + code + code + epi, cfg, fixable, fix=False)
    if bad:
        stream = pro_sect + code + code + epi
        for k, x, need in bad[:4]:
            print("TOO EARLY by %d: %r" % (need, x))
            for q in range(max(0, k - 12), k + 1):
                print("    S%02d %r" % (stream[q].stall, stream[q]))
        sys.exit("latency problems left: %d" % len(bad))
    # write
    off, size = secs[sect]
    base = off + ins[t0].addr
    blob = b"".join(x.encode() for x in code[:-1])
    assert len(blob) == 16 * (tb - t0)
    b[base:base + len(blob)] = blob
    # the branch: wait mask and stall
    bo = off + bra.addr
    b[bo:bo + 16] = bra.encode()
    open(dst, "wb").write(bytes(b))
    gaps, run = [], 0
    for x in code[:-1]:
        if x.op == "IMMA":
            gaps.append(run)
            run = 0
        elif x.op not in ("NOP",):
            run += 1
    gaps = gaps[1:] + [run]
    hist = ",".join("%d:%d" % kv for kv in sorted(collections.Counter(gaps).items()))
    cyc = sum(x.stall for x in code)
    print("sched120 %s: loop of %d (IMMA %d, nops %d), %d BLOCK swaps, gaps %s, %d cycles of stall a pass (%.2f an IMMA)" % (
        " ".join("%s=%d" % kv for kv in cfg.items()), tb - t0, nimma, room, swaps, hist, cyc, cyc / nimma))


if __name__ == "__main__":
    main()
