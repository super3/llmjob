"""build.py: the RTX 30 (sm_86) or RTX 40 (sm_89) scan kernel with carryfix.py's tails, as a C header the CUDA core
embeds.

  python build.py <nvcc> <nvdisasm> <out dir> [--arch sm_86|sm_89] [--ref FILE] [--defs "-D..."]

  1. nvcc -cubin -arch=sm_86 -DQC_RED=3 qc_host.cu  -> <out dir>/qc_sm86_red3.cubin (its tails are wrong as compiled)
     (sm_89: -arch=sm_89 -DQC_RED=3 -DQC_KZ=1, into qc_sm89_red3.cubin)
  2. carryfix.py                                     -> <out dir>/qc_sm86.cubin (every changed tail read back)
  3. the kernel's hash (below) must be listed in qc_sm86.ref (qc_sm89.ref) for this nvcc's version: the files that
     have run verify-core.js on an RTX 30 (RTX 40) card. If it is not, the file is left as qc_sm86.cubin.unverified
     and the exit code is 1, so a toolkit that compiles the kernel another way ships no patched kernel. --defs adds
     flags for a trial build; its hash is then not listed, and its .unverified file is what to try on a card.
  4. <out dir>/qc_sm86_cubin.h (qc_sm89_cubin.h): the cubin as bytes, its kernel's name and hash.
build.sh and build.ps1 embed the header when they are given its folder (QC_SM86_DIR, -Sm86Dir; QC_SM89_DIR, -Sm89Dir);
qc_host.cu then loads it on compute 8.6 (8.9) cards, checks it against the compiled kernel at startup and falls back to that kernel on any
failure (QUANTUS_CUDA_CUBIN=0 turns it off).

The hash: sha256 over every ELF section but the three that hold names (.shstrtab, .strtab, .symtab), each as its
name with the kernel's mangled name taken out, then its bytes. The mangled name differs between Linux and Windows
(u64 is unsigned long on one and unsigned long long on the other) while the code does not, so one line serves both.
On Windows run it where MSVC is on PATH: nvcc needs cl.exe even for a cubin.
"""
import argparse
import hashlib
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import carryfix  # noqa: E402
import alufix  # noqa: E402

FLAGS = ["-cubin", "-O3", "-std=c++17", "-arch=sm_86", "-DQC_RED=3"]
# RTX 40: QC_KZ 1 also makes the squares' carry on the multiplier pipe, which balances sm_89's two pipes once the
# tails' IMAD.X are gone (quantus-benchmark.md, 20 Nov).
FLAGS_SM89 = ["-cubin", "-O3", "-std=c++17", "-arch=sm_89", "-DQC_RED=3", "-DQC_KZ=1"]
# RTX 50: QC_RED 3 alone (21 Nov); sm_120's own path (QC_B120) picks the rest.
FLAGS_SM120 = ["-cubin", "-O3", "-std=c++17", "-arch=sm_120", "-DQC_RED=3"]
ARCHS = {"sm_86": (FLAGS, "RTX 30"), "sm_89": (FLAGS_SM89, "RTX 40"), "sm_120": (FLAGS_SM120, "RTX 50")}


def sections(b):
    shoff, = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    def sh(i):
        return struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize)
    stroff = sh(shstrndx)[4]
    out = []
    for i in range(shnum):
        s = sh(i)
        name = b[stroff + s[0]:b.index(b"\0", stroff + s[0])].decode()
        data = b[s[4]:s[4] + s[5]] if s[1] != 8 else b""   # SHT_NOBITS has no bytes in the file
        out.append((name, data))
    return out


def kernel_hash(path, kernel):
    h = hashlib.sha256()
    for name, data in sections(open(path, "rb").read()):
        if name in (".shstrtab", ".strtab", ".symtab"):
            continue
        h.update(name.replace(kernel, "K").encode() + b"\0")
        h.update(struct.pack("<Q", len(data)) + data)
    return h.hexdigest()


def listed(ref, version):
    hashes = set()
    for ln in open(ref, encoding="utf-8"):
        ln = ln.split("#", 1)[0].split()
        if len(ln) >= 2 and ln[0] == version:
            hashes.add(ln[1])
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nvcc")
    ap.add_argument("nvdisasm")
    ap.add_argument("out")
    ap.add_argument("--arch", default="sm_86", choices=sorted(ARCHS))
    ap.add_argument("--ref", default=None)
    ap.add_argument("--defs", default="")
    a = ap.parse_args()
    flags, cards = ARCHS[a.arch]
    tag = a.arch.replace("_", "")
    ref = a.ref or os.path.join(HERE, "qc_%s.ref" % tag)
    os.makedirs(a.out, exist_ok=True)
    v = carryfix.run([a.nvcc, "--version"]).stdout
    m = re.search(r"\bV(\d+\.\d+\.\d+)", v)
    if not m:
        sys.exit("could not read nvcc's version")
    version = "V" + m[1]
    raw = os.path.join(a.out, "qc_%s_red3.cubin" % tag)
    r = carryfix.run([a.nvcc] + flags + a.defs.split() + ["-I", CUDA_DIR, "-o", raw, os.path.join(CUDA_DIR, "qc_host.cu")])
    if r.returncode != 0:
        sys.exit("nvcc failed:\n" + (r.stderr or r.stdout)[-2000:])
    kernel = carryfix.kernel_name(raw)
    ins = carryfix.listing(a.nvdisasm, raw, kernel)
    fixes = carryfix.fix(ins)
    if not fixes:
        sys.exit("no QC_RED 3 tail in the compiled kernel")
    b = bytearray(open(raw, "rb").read())
    off, _ = carryfix.text_section(b, kernel)
    base = ins[0]["addr"]
    for x in ins:
        if x["addr"] in fixes:
            struct.pack_into("<QQ", b, off + x["addr"] - base, x["lo"], x["hi"] ^ carryfix.NEG1)
    out = os.path.join(a.out, "qc_%s.cubin" % tag)
    header = os.path.join(a.out, "qc_%s_cubin.h" % tag)
    for f in (out, out + ".unverified", header):
        if os.path.exists(f):
            os.remove(f)
    open(out + ".unverified", "wb").write(b)
    after = {x["addr"]: x["text"] for x in carryfix.listing(a.nvdisasm, out + ".unverified", kernel)}
    for x in ins:
        if after.get(x["addr"]) != fixes.get(x["addr"], x["text"]):
            sys.exit("read back: %04x is %r" % (x["addr"], after.get(x["addr"])))
    m = re.search(r"-DQC_ALUFIX=(\d+)", a.defs)
    if m and int(m[1]):
        moved = alufix.apply(a.nvdisasm, out + ".unverified", out + ".unverified", int(m[1]))
        print("alufix %s: %d instructions moved to the ALU" % (m[1], moved))
        b = bytearray(open(out + ".unverified", "rb").read())
    if a.arch == "sm_89" and "-DQC_HAND89=1" in flags + a.defs.split():
        # The hand-written partial rounds (sm89/patch.py): qc_hand_loop's body replaced, simulated against the
        # reference and read back there.
        sys.path.insert(0, os.path.join(CUDA_DIR, "sm89"))
        os.environ["NVDISASM"] = a.nvdisasm
        v = [d.split("=", 1)[1] for d in a.defs.split() if d.startswith("-DQC_HANDV=")]
        if v:
            os.environ["QC_HANDV"] = v[0]
        v = [d.split("=", 1)[1] for d in a.defs.split() if d.startswith("-DQC_HANDS=")]
        if v:
            os.environ["QC_HANDS"] = v[0]
        import patch as hand_patch  # noqa: E402
        hand_patch.run(out + ".unverified", out + ".unverified", [])
        os.remove(out + ".unverified.iface.json")
        b = bytearray(open(out + ".unverified", "rb").read())
    h = kernel_hash(out + ".unverified", kernel)
    toff, tsize = carryfix.text_section(b, kernel)
    th = hashlib.sha256(bytes(b[toff:toff + tsize])).hexdigest()[:16]
    print("%s, %d tails fixed, kernel %s, hash %s (code alone %s)" % (version, len(fixes), kernel, h, th))
    if h not in listed(ref, version):
        sys.exit("%s is not in %s for %s: left as %s. Run it on an %s card first (QUANTUS_CUDA_CUBIN_FILE, "
                 "verify-core.js), then add the line." % (h, os.path.basename(ref), version, out + ".unverified", cards))
    os.replace(out + ".unverified", out)
    with open(header, "w", newline="\n") as f:
        f.write("// Made by earn/native/quantus/cuda/sm86/build.py (%s, %s, %d tails fixed). Do not edit.\n"
                % (a.arch, version, len(fixes)))
        f.write('static const char qc_%s_kernel[] = "%s";\n' % (tag, kernel))
        f.write('static const char qc_%s_hash[] = "%s";\n' % (tag, h))
        f.write("static const unsigned char qc_%s_image[] = {\n" % tag)
        for i in range(0, len(b), 24):
            f.write("    " + ", ".join("0x%02x" % c for c in b[i:i + 24]) + ",\n")
        f.write("};\n")
    print("wrote " + header)


if __name__ == "__main__":
    main()
