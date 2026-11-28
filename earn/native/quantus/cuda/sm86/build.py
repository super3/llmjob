"""build.py: the RTX 30 (sm_86) or RTX 40 (sm_89) scan kernel with carryfix.py's tails, as a C header the CUDA core
embeds. With --arch sm_75 (RTX 20) the kernel has no such tails: it gets sm75/full75.py's hand-written body instead,
and the rest (the hash list qc_sm75.ref, the header) is the same.

  python build.py <nvcc> <nvdisasm> <out dir> [--arch sm_75|sm_86|sm_89|sm_120] [--ref FILE] [--defs "-D..."]

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

# RTX 30: since 28 Nov 2026 the hand-written body too (see HAND), in the compact kernel. It ran 6 to 7% faster on an
# RTX 3070 than the QC_RED 3 kernel with carryfix.py's tails, which this replaced; nothing compiled is left that has
# such a tail.
FLAGS = ["-cubin", "-O3", "-std=c++17", "-arch=sm_86", "-DQC_HAND75=2", "-DQC_COMPACT=2"]
# RTX 40: since 28 Nov 2026 the hand-written body too, in the compact kernel, as on RTX 30: on an RTX 4070 Ti it ran 9%
# faster than the QC_RED 3 + QC_KZ 1 kernel with carryfix.py's tails that this replaced.
FLAGS_SM89 = ["-cubin", "-O3", "-std=c++17", "-arch=sm_89", "-DQC_HAND75=2", "-DQC_COMPACT=2"]
# RTX 50: QC_RED 4, the reduction as four adds with no multiply, its carry-in made right and pairs of adds fused
# by sm120/fuse.py (28 Nov: +7.7% on an RTX 5070 and an RTX 5060 over QC_RED 3 with CUDA 13). main() keeps QC_RED 3 for
# a 12.8 toolkit and for --defs -DQC_FUSE=0.
FLAGS_SM120 = ["-cubin", "-O3", "-std=c++17", "-arch=sm_120", "-DQC_RED=4"]
# RTX 20: no QC_RED 3 tails either. The kernel is built with QC_HAND75 2 and sm75/full75.py replaces
# qc_hand_loopall's body (every step of both permutations after the first round) with the hand-written one, with
# HAND's options, simulated against a Python reference and read back through nvdisasm there.
FLAGS_SM75 = ["-cubin", "-O3", "-std=c++17", "-arch=sm_75", "-DQC_HAND75=2"]
# Since 28 Nov 2026 with the round constants folded away (sum2, kf, kt: the partial rounds' constant is a thirteenth
# value of the sum and the terminal constants ride the layer before, 3% fewer instructions a nonce), the S-box
# output added to the sum last (ylast), the sum's low word in the diagonal product's fold (dl), sum2's and the
# layer's inner counts on the multiplier (imx 8 and 32) and the yield bit up to stall 3: +4% on an RTX 3070 and an
# RTX 2060 over the first hand body (quantus-benchmark.md).
# RTX 40 takes the 18-op sum (sum2) and no anchors: of four option sets on an RTX 4070 Ti these ran the most a clock.
# Since then also q5-nv2030b's kf and kt (the partial rounds' and the terminal constants folded into the tables), sm
# (the order that keeps both pipes fed), every carry count of the layer and the sum on the multiplier (imx=63), six
# S-boxes scheduled together, and sq3 (full-round S-box outputs handed to the layer as three words): of nine option
# sets these ran fastest on an RTX 4070 Ti and an RTX 4080, 2.5 to 2.8% over the set before.
HAND = {"sm_75": "imx=47 dmx=1 sq2=1 anch=6 sum2=1 kf=1 ylast=1 kt=1 dl=1 ybit=3", "sm_86": "imx=47 dmx=1 sq2=1 anch=6 sum2=1 kf=1 ylast=1 kt=1 dl=1 ybit=3", "sm_89": "imx=63 dmx=1 sq2=1 sum2=1 kf=1 kt=1 sq3=1 sm=1 sbatch=6"}
# RTX 50's body: the diagonal products' carry on IMAD.WIDE.U32.X (dmx) and q5-nv40's 18-instruction sum (sum2).
HAND120 = "dmx:1+sum2:1"
ARCHS = {"sm_75": (FLAGS_SM75, "RTX 20"), "sm_86": (FLAGS, "RTX 30"), "sm_89": (FLAGS_SM89, "RTX 40"), "sm_120": (FLAGS_SM120, "RTX 50")}


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
    # RTX 50: QC_RED 4 with CUDA 13 and newer, QC_RED 3 with 12.8. On an RTX 5070 (driver 621) QC_RED 4 ran 7.9% faster
    # with 12.8 too, but on an RTX 5060 (driver 616) the 12.8 build of it ran 1.3% slower than QC_RED 3 while the
    # CUDA 13 build ran 7.7% faster (28 Nov). The 12.8 core is what drivers below 580 get, and none was tried.
    cu13 = int(version[1:].split(".")[0]) >= 13
    fusing = (a.arch == "sm_120" and cu13 and "-DQC_FUSE=0" not in a.defs)
    fusing = fusing or re.search(r"-DQC_FUSE=[1-9]", a.defs) is not None
    # RTX 50, since 28 Nov 2026: the hand-written body (sm120/full120.py, options HAND120) in the compact kernel, built
    # with CUDA 13 and newer. On an RTX 5070 it ran 502 MH/s where the QC_RED 4 kernel above ran 421 and the compiled
    # one 405. The rest of the kernel keeps QC_RED 2, right as compiled. --defs -DQC_HAND120=0, or any -DQC_FUSE=, builds
    # the kernels above instead; -DQC_HANDO=key:value+key:value gives full120.py other options; -DQC_HAND120=2 leaves the
    # function as compiled (a control).
    # Not with CUDA 12.8 unless asked (-DQC_HAND120=1): a 12.8 cubin's .nv.capmerc.text describes its code position by
    # position, and an RTX 5060's newer driver read it: the 12.8 hand kernel was an illegal instruction there, which
    # also takes the card's CUDA context down, while it ran right on an RTX 5070 with driver 591.86. CUDA 13's
    # section is of another kind and its hand kernel ran right on both.
    explicit = re.search(r"-DQC_HAND120=[1-9]", a.defs) is not None
    hand120 = a.arch == "sm_120" and (explicit or (cu13 and "-DQC_HAND120=0" not in a.defs
                                                   and re.search(r"-DQC_FUSE=\d", a.defs) is None))
    if hand120:
        fusing = False
        flags = [f for f in flags if not f.startswith("-DQC_RED=")] + ["-DQC_HAND75=2", "-DQC_COMPACT=2"]
    if fusing and a.arch != "sm_120":
        sys.exit("-DQC_FUSE is for --arch sm_120 only (sm120/fuse.py knows sm_120's forms)")
    if not fusing and "-DQC_RED=4" in a.defs:
        sys.exit("-DQC_RED=4 is wrong as compiled; it needs sm120/fuse.py (leave -DQC_FUSE=0 out)")
    if a.arch == "sm_120" and not fusing and not hand120:
        flags = [f for f in flags if f != "-DQC_RED=4"] + ["-DQC_RED=3"]
    if fusing:
        # sm120/fuse.py rewrites QC_RED 2's reductions itself: the kernel is compiled right, no tails to fix
        flags = [f for f in flags if not f.startswith("-DQC_RED=")] + ([] if "-DQC_RED=" in a.defs else ["-DQC_RED=4"])
    raw = os.path.join(a.out, "qc_%s_red3.cubin" % tag)
    r = carryfix.run([a.nvcc] + flags + a.defs.split() + ["-I", CUDA_DIR, "-o", raw, os.path.join(CUDA_DIR, "qc_host.cu")])
    if r.returncode != 0:
        sys.exit("nvcc failed:\n" + (r.stderr or r.stdout)[-2000:])
    kernel = carryfix.kernel_name(raw)
    ins = carryfix.listing(a.nvdisasm, raw, kernel)
    fixes = carryfix.fix(ins)
    if not fixes and not fusing and a.arch not in HAND and not hand120:
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
    if a.arch in HAND:
        sys.path.insert(0, os.path.join(CUDA_DIR, "sm75"))
        os.environ["NVDISASM"] = a.nvdisasm
        import full75  # noqa: E402
        opts = dict(x.split("=") for x in HAND[a.arch].split())
        code, _stalls, _per, steps = full75.run(out + ".unverified", out + ".unverified",
                                                **{k: int(v) for k, v in opts.items()})
        print("hand body (%s): %d instructions, %d run a nonce" % (HAND[a.arch], len(code), steps))
    if hand120 and "-DQC_HAND120=2" not in a.defs:      # 2: the function as compiled, for a control run
        sys.path.insert(0, os.path.join(CUDA_DIR, "sm120"))
        os.environ["NVDISASM"] = a.nvdisasm
        import full120  # noqa: E402
        mo = re.search(r"-DQC_HANDO=(\S+)", a.defs)
        # options joined by "+" (nvcc sees the flag too, and splits a -D at commas)
        opts = dict(x.split(":") for x in re.split("[+,]", mo[1] if mo else HAND120))
        try:
            code, _stalls, per, steps = full120.run(out + ".unverified", out + ".unverified",
                                                    **{k: int(v) for k, v in opts.items()})
        except Exception as e:      # an assertion in full120.py: a toolkit that compiles the function another way
            if explicit:
                raise
            # Not the plain compiled kernel (405 MH/s on an RTX 5070 against 480): this script again for the QC_RED 4
            # kernel with fuse.py, which is what RTX 50 ran before the hand body.
            print("hand body (sm_120) not built: %s: %s. Building the QC_RED 4 kernel instead." % (type(e).__name__, e))
            sys.stdout.flush()
            import subprocess
            sys.exit(subprocess.call([sys.executable, os.path.abspath(__file__), a.nvcc, a.nvdisasm, a.out, "--arch",
                                      a.arch, "--ref", ref, "--defs=" + (a.defs + " -DQC_HAND120=0").strip()]))
        print("hand body (sm_120, %s): %d instructions, %d run a nonce; S %d, E %d, loop %d cycles for one warp"
              % (mo[1] if mo else HAND120, len(code), steps, per["S"][1], per["E"][1], per["Ploop"][1]))
        b = bytearray(open(out + ".unverified", "rb").read())
    if fusing:
        # RTX 50 trial: pairs of two-input adds fused into three-input ones and the loop closed up
        # (sm120/fuse.py). QC_FUSE's value is the most pairs to fuse (1 = all).
        sys.path.insert(0, os.path.join(CUDA_DIR, "sm120"))
        import fuse  # noqa: E402
        mm = re.search(r"-DQC_FUSEM=(\d+)", a.defs)
        if mm:
            os.environ["QC_FUSE_MODE"] = mm[1]
        mm = re.search(r"-DQC_FUSEL=(\d+)", a.defs)
        if mm:
            os.environ["QC_FUSE_LIMIT"] = mm[1]
        n, seen, pairs = fuse.apply(a.nvdisasm, out + ".unverified", out + ".unverified", kernel)
        print("fuse: %d of %d reductions rewritten, %d pairs of adds fused" % (n, seen, pairs))
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
    # The image is the file that is hashed, read again here: a step above that rewrites the file must not leave `b`
    # behind. After a merge the hand body's step did, and the header held the kernel as compiled under the hand
    # body's hash: right, so the core's check passed, and no faster.
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
