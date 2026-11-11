"""perm.py: puts the steps of a pipeprobe2 loop where a rule says, in the compiled kernel itself.

ptxas orders a loop by its own model of the card, so the source cannot say "one step after every IMMA" or "all
96 steps in one run". This takes a cubin of the probe, moves whole 16-byte instructions inside the loop's
straight-line stretches, and writes a cubin that pipeprobe2 runs when PP_CUBIN names it. Nothing is added or
removed, branches and branch targets stay where they are, and the steps keep their own order, so only their
place among the IMMA changes. The streams compute nothing real, so a step that now reads a sum before its IMMA
wrote it does no harm; say so when it matters to the timing.

  python3 perm.py IN.cubin IN.sass OUT.cubin RULES

IN.sass is `nvdisasm -c IN.cubin`. RULES is KIND=PLACE joined by "/". KIND is X (LOP3), M (ldmatrix), L (other
loads), S (stores and shared atomics), A (any other plain step) or * (every kind without a rule of its own).
PLACE, with IMMA numbered from 0 in each stretch:
  orig            where the compiler put them
  even,N[,K]      N after every IMMA from IMMA K on (default 0), round the loop, until none is left
  every,E,N[,K]   N after IMMA K, K+E, K+2E ... (default K = E-1); N = 0 shares them out evenly
  all,K           all of them after IMMA K
  list,K,G0.G1... G0 after IMMA K, G1 after K+1, ... and round again
A kind without a rule stays where the compiler put it.
"""
import collections
import re
import struct
import sys

CTL = ("BRA", "BRX", "JMP", "WARPSYNC", "YIELD", "BSSY", "BSYNC", "BAR", "NANOSLEEP", "CALL", "RET", "BREAK",
       "BMOV", "MEMBAR", "EXIT", "KILL", "TRAP", "JMX", "JCAL", "VOTE", "S2R")   # and steps .nv.info names


def kind(o):
    if o == "IMMA":
        return "I"
    if o in CTL:
        return "C"
    if o in ("LOP3", "PLOP3", "LOP"):
        return "X"
    if o == "LDSM":
        return "M"
    if o in ("LDS", "LD", "LDG", "LDL"):
        return "L"
    if o in ("STS", "ST", "STG", "STL", "ATOMS", "ATOM", "ATOMG"):
        return "S"
    return "A"


def sections(b):
    """name -> (file offset, size) of an ELF64 little-endian file's sections."""
    if b[:4] != b"\x7fELF" or b[4] != 2 or b[5] != 1:
        sys.exit("not a 64-bit little-endian ELF")
    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    entsize, num, strndx = struct.unpack_from("<HHH", b, 0x3A)
    raw = [struct.unpack_from("<IIQQQQ", b, shoff + i * entsize) for i in range(num)]
    stro = raw[strndx][4]
    out = {}
    for name, _type, _flags, _addr, off, size in raw:
        end = b.index(b"\0", stro + name)
        out[bytes(b[stro + name:end]).decode()] = (off, size)
    return out


odd = []   # listing lines with an address that did not read as an instruction


def listing(path, fn):
    """The kernel's instructions in order as (address, name), and the addresses that carry a label."""
    ins, labels, inside, pending = [], set(), False, False
    del odd[:]
    for line in open(path, errors="replace"):
        if re.match(r"\s*\.section\s", line):
            inside = (".text." + fn + ",") in line
            continue
        if not inside:
            continue
        if re.match(r"\s*\.L_x_\d+:", line):
            pending = True
            continue
        m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(@\S+\s+)?([A-Za-z_][A-Za-z0-9_.]*)", line)
        if m:
            addr = int(m.group(1), 16)
            ins.append((addr, m.group(3).split(".")[0]))
            if pending:
                labels.add(addr)
                pending = False
        elif re.search(r"/\*[0-9a-f]{4,}\*/", line) and len(odd) < 3:
            odd.append(line.strip()[:90])
    return ins, labels


def slots(place, n_imma, count):
    """The IMMA after which each of `count` steps goes, in order, for one PLACE."""
    p = place.split(",")
    out = []
    if p[0] == "even":
        n, k = int(p[1]), int(p[2]) if len(p) > 2 else 0
        while len(out) < count:
            out += [k % n_imma] * n
            k += 1
    elif p[0] == "every":
        e, n = int(p[1]), int(p[2])
        k = int(p[3]) if len(p) > 3 else e - 1
        if n == 0:
            n = -(-count // max(1, n_imma // e))
        while len(out) < count:
            out += [k % n_imma] * n
            k += e
    elif p[0] == "all":
        out = [int(p[1]) % n_imma] * count
    elif p[0] == "list":
        k, g, i = int(p[1]), [int(x) for x in p[2].split(".")], 0
        if sum(g) <= 0:
            sys.exit("list needs a gap above 0")
        while len(out) < count:
            out += [k % n_imma] * g[i % len(g)]
            k += 1
            i += 1
    else:
        sys.exit("unknown place " + place)
    return out[:count]


def main():
    cubin, sass, dst, spec = sys.argv[1:5]
    fn = sys.argv[5] if len(sys.argv) > 5 else "pipeprobe2"
    b = bytearray(open(cubin, "rb").read())
    secs = sections(b)
    off, size = secs[".text." + fn]
    ins, labels = listing(sass, fn)
    # An instruction's bytes run to the next one's address. Most are 16 bytes; whatever the sizes, a stretch keeps
    # its length when its instructions change places, so branches and their targets stay where they are.
    addrs = [a for a, _ in ins]
    lens = [(addrs[i + 1] if i + 1 < len(ins) else size) - a for i, a in enumerate(addrs)]
    if not ins or addrs[0] != 0 or min(lens) < 16 or any(n % 16 for n in lens[:-1]):
        sys.exit("the listing does not match the kernel's text: %d instructions, %d bytes, sizes %s, unread %s" % (
            len(ins), size, sorted(collections.Counter(lens).items())[:6], odd))
    lens[-1] = 16 if lens[-1] % 16 or lens[-1] > 64 else lens[-1]
    ops = [o for _, o in ins]
    im = [i for i, o in enumerate(ops) if o == "IMMA"]
    if not im:
        sys.exit("no IMMA")
    end = next((i for i in range(im[-1], len(ops)) if ops[i] == "BRA"), len(ops))
    rules = dict(r.split("=", 1) for r in spec.split("/") if r)
    # Stretches: runs of the body with no control instruction and no branch target inside.
    order, start, moved, stretches = list(range(len(ops))), im[0], 0, 0
    bounds = [i for i in range(im[0], end + 1) if i == end or kind(ops[i]) == "C" or (16 * i in labels and i > im[0])]
    for bd in bounds:
        seg = list(range(start, bd))
        start = bd + 1 if bd < len(ops) and (bd == end or kind(ops[bd]) == "C") else bd
        imma = [i for i in seg if ops[i] == "IMMA"]
        if len(imma) < 2:
            continue
        stretches += 1
        n = len(imma)
        anchor = {}
        seen = -1
        for i in seg:   # where the compiler put each step: after IMMA `seen` of the stretch (-1: before the first)
            if ops[i] == "IMMA":
                seen += 1
            else:
                anchor[i] = seen
        by_kind = collections.defaultdict(list)
        for i in seg:
            if ops[i] != "IMMA":
                k = kind(ops[i])
                by_kind[k if k in rules else "*"].append(i)
        for k, members in by_kind.items():
            place = rules.get(k, "orig")
            if place == "orig":
                continue
            for i, a in zip(members, slots(place, n, len(members))):
                anchor[i] = a
        new = [i for i in seg if ops[i] != "IMMA" and anchor[i] < 0]
        for q, i in enumerate(imma):
            new.append(i)
            new += [j for j in seg if ops[j] != "IMMA" and anchor[j] == q]
        assert sorted(new) == seg
        for pos, src in zip(seg, new):
            order[pos] = src
            moved += pos != src
    text = bytes(b[off:off + size])
    out = b"".join(text[addrs[src]:addrs[src] + lens[src]] for src in order)
    assert len(out) == sum(lens)
    b[off:off + len(out)] = out
    open(dst, "wb").write(b)
    # Tables in .nv.info that name an instruction's address would now point at another instruction.
    warn = []
    if ".nv.info." + fn in secs:
        io, isz = secs[".nv.info." + fn]
        was = {addrs[src]: ops[src] for pos, src in enumerate(order) if pos != src}
        for q in range(io, io + isz - 3, 4):
            v = struct.unpack_from("<I", b, q)[0]
            if v in was:
                warn.append("0x%x %s" % (v, was[v]))
    new_ops = [ops[src] for src in order]
    gaps, run = [], 0
    for o in new_ops[im[0]:end]:
        if o == "IMMA":
            gaps.append(run)
            run = 0
        else:
            run += 1
    gaps = gaps[1:] + [run]
    hist = ",".join("%d:%d" % kv for kv in sorted(collections.Counter(gaps).items()))
    sizes = ",".join("%d:%d" % kv for kv in sorted(collections.Counter(lens).items()))
    print("perm %s: %d stretches, %d of %d instructions moved, sizes %s, gaps=%s%s%s" % (
        spec, stretches, moved, end - im[0], sizes, hist,
        (" WARN nv.info names moved: " + " ".join(warn[:6])) if warn else "", (" unread: %s" % odd) if odd else ""))


if __name__ == "__main__":
    main()
