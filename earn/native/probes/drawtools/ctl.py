"""ctl.py <cubin> <sass> [from-address-hex] [to-address-hex]: the listing with each instruction's control field
decoded from the cubin: [B wait mask : R read scoreboard : W write scoreboard : Y yield : S stall] and reuse flags.
The control field is bits 105 to 125 of the 128-bit instruction (stall 4 bits, yield 1, write scoreboard 3, read
scoreboard 3, wait mask 6, reuse 4)."""
import re
import struct
import sys


def sections(b):
    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    entsize, num, strndx = struct.unpack_from("<HHH", b, 0x3A)
    raw = [struct.unpack_from("<IIQQQQ", b, shoff + i * entsize) for i in range(num)]
    stro = raw[strndx][4]
    out = {}
    for name, _t, _f, _a, off, size in raw:
        end = b.index(b"\0", stro + name)
        out[bytes(b[stro + name:end]).decode()] = (off, size)
    return out


def read_text(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def listing(path, fn_sub):
    """(address, text, label-before) of the kernel's instructions."""
    out, inside, pending = [], False, False
    for line in read_text(path).splitlines():
        if re.match(r"\s*\.section\s", line):
            inside = ".text." in line and fn_sub in line
            continue
        if not inside:
            continue
        if re.match(r"\s*\.L_x_\d+:", line):
            pending = True
            continue
        m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
        if m:
            out.append((int(m.group(1), 16), re.sub(r"\s+", " ", m.group(2).strip()), pending))
            pending = False
    return out


def ctrl(word_hi):
    c = (word_hi >> 41) & 0x1FFFFF
    return dict(stall=c & 15, yld=(c >> 4) & 1, wsb=(c >> 5) & 7, rsb=(c >> 8) & 7, wait=(c >> 11) & 63,
                reuse=(c >> 17) & 15)


def fmt(c):
    b = "".join(str(i) if c["wait"] >> i & 1 else "-" for i in range(6))
    return "[B%s:R%s:W%s:%s:S%02d]" % (b, "-" if c["rsb"] == 7 else c["rsb"], "-" if c["wsb"] == 7 else c["wsb"],
                                      "Y" if c["yld"] else "-", c["stall"])


def load(cubin, sass, fn_sub="pearl"):
    b = bytearray(open(cubin, "rb").read())
    secs = sections(b)
    name = next(n for n in secs if n.startswith(".text.") and fn_sub in n)
    off, size = secs[name]
    ins = listing(sass, fn_sub)
    rows = []
    for a, t, lab in ins:
        lo, hi = struct.unpack_from("<QQ", b, off + a)
        rows.append((a, t, lab, ctrl(hi)))
    return b, off, size, rows


def main():
    b, off, size, rows = load(sys.argv[1], sys.argv[2])
    lo = int(sys.argv[3], 16) if len(sys.argv) > 3 else 0
    hi = int(sys.argv[4], 16) if len(sys.argv) > 4 else 1 << 30
    for a, t, lab, c in rows:
        if lo <= a <= hi:
            print("%s%05x %s %s" % ("L " if lab else "  ", a, fmt(c), t))


if __name__ == "__main__":
    main()
