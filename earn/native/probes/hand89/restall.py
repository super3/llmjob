"""restall.py <in.cubin> <in.sass> <out.cubin> [behind=5] [give=0|1] [kinds=alu|all]

Changes stall counts only, in a kernel as ptxas compiled it: no instruction moves and no register changes.

Why: on an RTX 4090 a plain instruction that reads two or more registers costs the tensor pipe up to 2 cycles
an IMMA when it issues 1 to 4 cycles behind its warp's IMMA, and nothing when it issues 5 behind (measured with
probes/hand89/sched.py; one-register steps never cost). ptxas writes "IMMA stall 1, step, step ..., pad", which
puts such steps right behind the IMMA. This raises the IMMA's stall so that the first such step behind it
issues `behind` cycles after it.

Raising a stall is always safe for the values computed: every instruction issues no earlier than before, in the
same order. With give=1 the cycles added are taken back from the stall of the last instruction ahead of the next
IMMA where that is padding (its stall stays 1 or more, and no later reader of what it writes comes closer than
ptxas's own least distances: 5 cycles after an ALU step, 28 after an IMMA, 13 after a compare); a load or store
never gives cycles back.

It prints how many IMMA it changed. The result must still pass the kernel's own check (hprobe's CPU check for
the draw fold, verify-hits.js for a miner's kernel) before any figure from it counts.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
import fold  # noqa: E402


def main():
    cubin, sass, out = sys.argv[1:4]
    opt = dict(behind=5, give=0, kinds="alu")
    for kv in sys.argv[4:]:
        k, v = kv.split("=")
        opt[k] = v if k == "kinds" else int(v)
    b = bytearray(open(cubin, "rb").read())
    secs = asm89.sections(b)
    name = next(n for n in secs if n.startswith(".text.") and "pearl" in n)
    off, _size = secs[name]
    rows, labels, _entry = asm89.listing(asm89.read_text(sass))
    targets = set(labels.values())

    def ctl(a):
        _lo, hi = struct.unpack_from("<QQ", b, off + a)
        return (hi >> asm89.CTRL_SHIFT) & 0x1FFFFF

    def set_stall(a, st):
        lo, hi = struct.unpack_from("<QQ", b, off + a)
        c = (hi >> asm89.CTRL_SHIFT) & 0x1FFFFF
        c = (c & ~15) | st
        hi = (hi & ~asm89.CTRL_MASK) | (c << asm89.CTRL_SHIFT)
        struct.pack_into("<QQ", b, off + a, lo, hi)

    def costly(text):
        """A plain step that reads two or more registers."""
        op = text.split(" ")[1] if text.startswith("@") else text.split(" ")[0]
        if op.startswith(("IMMA", "BRA", "BAR", "NOP", "CALL", "RET", "EXIT", "BSSY", "BSYNC", "WARPSYNC", "YIELD",
                          "CS2R", "S2R", "ULDC", "MOV")):
            return False
        try:
            reads, _w, kind = fold.rw(text)
        except (AssertionError, ValueError, IndexError):
            return False
        if kind not in ("alu", "setp") and opt["kinds"] == "alu":
            return False
        return len([r for r in reads if r != "P0"]) >= 2

    stalls = {a: ctl(a) & 15 for a, _t in rows}
    changed = added = given = 0
    n = len(rows)
    for i, (a, t) in enumerate(rows):
        if "IMMA" not in t:
            continue
        # The steps behind this IMMA up to the next IMMA, a label or a branch: where does the first costly one issue?
        d, j, first, last = stalls[a], i + 1, None, None
        while j < n and "IMMA" not in rows[j][1] and rows[j][0] not in targets and "BRA" not in rows[j][1] \
                and "CALL" not in rows[j][1] and "RET" not in rows[j][1]:
            if first is None and costly(rows[j][1]):
                first = d
            last = j
            d += stalls[rows[j][0]]
            j += 1
        if first is None or first >= opt["behind"]:
            continue
        add = opt["behind"] - first
        if stalls[a] + add > 11:
            continue
        stalls[a] += add
        changed += 1
        added += add
        if opt["give"] and last is not None and j < n and "IMMA" in rows[j][1]:
            la, lt = rows[last]
            try:
                _r, writes, kind = fold.rw(lt)
            except (AssertionError, ValueError, IndexError):
                continue
            if kind not in ("alu", "setp") or stalls[la] - add < 1:
                continue
            # Later readers of what it writes, with the distance they would have.
            need, dist, ok = fold.LAT.get(kind, 5), stalls[la] - add, True
            for k2 in range(last + 1, min(n, last + 40)):
                if rows[k2][0] in targets or dist >= need:
                    break
                try:
                    r2, w2, _k = fold.rw(rows[k2][1])
                except (AssertionError, ValueError, IndexError):
                    ok = False
                    break
                if r2 & writes:
                    ok = False
                    break
                dist += stalls[rows[k2][0]]
            if ok:
                stalls[la] -= add
                given += add
    for a, _t in rows:
        if stalls[a] != ctl(a) & 15:
            set_stall(a, stalls[a])
    open(out, "wb").write(b)
    immas = sum(1 for _a, t in rows if "IMMA" in t)
    print("%s: %d of %d IMMA given a longer stall (%d cycles added, %d taken back from padding)" % (
        os.path.basename(out), changed, immas, added, given))


if __name__ == "__main__":
    main()
