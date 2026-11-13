"""build_rot.py: builds the rotation fold's cubin, the file the CUDA host loads when the rotation fold is asked
for (PEARL_FOLD_ROT=1, or --rotation-fold on the CLI; src/pearl_host_rot.inc; off by default).

  python build_rot.py <nvcc> <nvdisasm> <out dir> [--arch sm_89|sm_86|sm_120] [--ref <file>]

--arch sm_89 (the default) builds pearl_rot_sm89.cubin for RTX 40 cards against src/pearl_rot_sm89.ref;
--arch sm_86 builds pearl_rot_sm86.cubin for RTX 30 cards against src/pearl_rot_sm86.ref. ptxas gives the two
architectures the same code for this kernel, so the same hand loop and the same checks serve both.

--arch sm_120 builds pearl_rot_sm120.cubin for RTX 50 cards against src/pearl_rot_sm120.ref, with the same hand
loop: pearl_rot.cu is compiled with its C fold outlined and given room (-DROT_INLINE=0 -DROT_C=16387), and
rfold120.py puts rfold.py's loop, encoded for sm_120, in that function's place. Step 2 (asm89's check of its
own encodings) does not apply there: rfold120.py reads every word it writes back through nvdisasm.

The fold's chunk loop is written by hand and put into a kernel the compiler made. So every step is checked, and
the file is only written when all of them pass:
  1. nvcc -arch=<arch> -cubin src/pearl_rot.cu -> <out dir>/rot_base.cubin. As compiled, its chunk loop is the
     gather fold: the room, the registers and the reference for the hand loop. The host never loads this file.
  2. nvdisasm -c -> <out dir>/rot_base.sass, and asm89's check of itself against it: every instruction of the
     listing whose form the assembler knows is assembled from its text and must give the compiler's own bytes.
  3. rfold.py builds the hand loop and runs its latency checker over it (every register is read late enough
     behind the instruction that wrote it).
  4. rfold.py compares what the loop takes for granted with what this compiler made: the registers and
     predicates the compiled function writes, the uniform register it keeps the address base in, the kernel
     parameters it reads, where the transcripts are in shared memory, and the room.
  5. asm89 writes the loop into a copy of the cubin, disassembles the result with nvdisasm and compares every
     instruction's text with what was asked for; the barrier count is read back too.
  6. The result's hash (asm89.fold_hash: sha256 with the string table zeroed) must be one of those
     src/pearl_rot_<arch>.ref lists for this toolkit: the hashes of the files that have mined on a card with
     every hit verified. A toolkit with no line there, or a hash that is not listed, fails: the file is left
     as pearl_rot_<arch>.cubin.unverified, to be run on a card of that kind (PEARL_ROT_CUBIN names it; then
     probes/verify-hits.js) before its hash is added.
On any failure the exit code is 1 and <out dir> has no pearl_rot_<arch>.cubin.
On Windows run it from a shell that has MSVC's environment (vcvars64.bat): nvcc needs cl.exe.
Put pearl_rot_<arch>.cubin beside pearl_core.node, or name it in PEARL_ROT_CUBIN.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
import asm89  # noqa: E402

ARCHS = ("sm_89", "sm_86", "sm_120")
# rfold.py keys an architecture takes beyond the defaults. On an RTX 3090 the IMMA without their yield flag
# (iy=0) mined 141.6 TH/s against 141.0 (two races on one Salad PC, 12 November). On rig01's RTX 4090 the
# readout reading each IMMA's sums as (0, 2) and (1, 3) on the chain of the other register parity, so that no
# step reads two registers of one bank (rpr=2), with U8 IMMA (u8=1), mined 336.92, 336.92 and 336.91 TH/s
# against 335.92, 336.10 and 336.07 for the loop before (hashrate.js, 12 November); on a Salad RTX 3090 the same
# two keys with iy=0 mined 124.35, 124.46 and 124.42 against 123.62, 123.91 and 123.81 for iy=0 alone. On the
# 4090 the readout's steps first in their gap and allowed into more of the chunk's end (rgf=1 rb0=3 rb1=0: the
# probe's pipe 99.54% against 99.18%, the level of a loop with no readout) mined 338.00, 338.04 and 337.93
# TH/s against 336.93, 337.07 and 337.10 for the loop before (hashrate.js, 12 November); on the Salad RTX 3090,
# with iy=0, 121.16, 121.17 and 121.25 against 120.74, 120.81 and 120.87. sm_120 (rfold120.py): the same four
# readout keys and the IMMA's stall 7 (ist=7, not 5). The probe's pipe at k 65536: 97.60% against 96.02% on an RTX
# 5090, 96.53% against 94.97% on a 5080 (ist 6, 8, 9, 10: 95.4, 97.4, 93.6, 94.8%; ist=7 without the readout keys
# 96.3%); hashrate.js 434.44 and 434.46 TH/s against 430.30, 430.22 and 430.35 for the loop before on one 5090 PC,
# 237.53, 237.49 and 237.54 against 233.75 and 233.69 on one 5080 PC (12 November). And the IMMA without their yield flag
# (iy=0, as on sm_86): on a 488.75 W 5090 PC 402.11 and 402.17 TH/s against 399.18 and 399.45 without it (389.44
# and 389.46 for the file before both). And no barrier after a chunk's second turn (bt=13: no fragment is stored in
# the second or third turn, so it orders nothing; every load keeps a barrier between it and the stores around it):
# the probe's pipe 99.03% against 98.95% on a 5080 and 98.26% against 98.18% on a 5090, CPU check 16,384 of 16,384
# at k 2048 and 8192 (12 November). And no over-waits on the gathers' scoreboards (dpb=7 dpm=1 dpg=3: a subtract
# that waited on a scoreboard whose newest load went out a gap before, while its own went out 9 or more gaps before,
# has that wait moved onto the first newer load, where its own load has long landed): the probe's pipe 99.04%
# against 98.59% on a clock-bound 5080 (count +0.46%), 98.03% against 97.53% on a capped 5090 (12 November).
# sm_89 (goal-nvk19): B drawn once for the two warps of a column slot (bx, with its exchange keys: bsym=4 bsg=8
# bsg3=24 bss=2 bord=2 barv=1 bxt=1, readout window rb0=6 rb1=3) and records the eight warps share (sr). The
# miner's kernel is built for both (ROT_BX, ROT_SR below). sm_86 (goal-nv30b): sr too, on the iy=0 loop. Salad RTX
# 3060 (170 W): 53.52, 53.55, 53.53 TH/s against 53.33, 53.32, 53.32 (hashrate.js, 12 November); Salad RTX 3090
# (280 W): 108.09, 107.95 against 107.76, 107.34. B drawn once lost there (52.1 and 105.3 with iy=0).
ARCH_KEYS = {"sm_89": ["rpr=2", "u8=1", "rgf=1", "rb0=6", "rb1=3", "bsym=4", "bsg=8", "bsg3=24", "bss=2", "bord=2", "bx=1",
                       "barv=1", "bxt=1", "sr=1"], "sm_86": ["iy=0", "rpr=2", "u8=1", "rgf=1", "rb0=3", "rb1=0",
                                                                                  "sr=1"],
             "sm_120": ["rpr=2", "u8=1", "rgf=1", "rb0=3", "rb1=0", "ist=7", "iy=0", "b8=1", "bt=13",
                        "dpb=7", "dpm=1", "dpg=3", "sr=1", "srp=1"]}


def toolkit(nvcc):
    """The toolkit's release as nvcc states it: '12.8' from 'Cuda compilation tools, release 12.8, V12.8.93'."""
    r = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    m = re.search(r"release (\d+\.\d+)", r.stdout + r.stderr)
    return m[1] if m else None


def read_ref(path):
    """{toolkit release: set of hashes} from the reference file: lines of '<release> <sha256>', # comments."""
    out = {}
    for line in open(path, encoding="utf-8"):
        f = line.split("#")[0].split()
        if len(f) == 2 and re.fullmatch(r"[0-9a-f]{64}", f[1]):
            out.setdefault(f[0], set()).add(f[1])
    return out


def fail(msg):
    print("build_rot: FAILED: " + msg)
    return 1


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = sys.argv[1:]
    arch = "sm_89"
    if "--arch" in args:
        i = args.index("--arch")
        arch = args[i + 1]
        del args[i:i + 2]
    if arch not in ARCHS:
        return fail("--arch is one of %s, not %s" % (", ".join(ARCHS), arch))
    NAME = "pearl_rot_%s.cubin" % arch.replace("_", "")
    ref = os.path.join(SRC, "pearl_rot_%s.ref" % arch.replace("_", ""))
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        del args[i:i + 2]
    if len(args) != 3:
        print(__doc__)
        return 2
    nvcc, nvdisasm, out = args
    os.makedirs(out, exist_ok=True)
    base, sass, final = (os.path.join(out, n) for n in ("rot_base.cubin", "rot_base.sass", NAME))
    tmp, unverified = final + ".tmp", final + ".unverified"
    for f in (final, tmp, unverified):
        if os.path.exists(f):
            os.remove(f)
    release = toolkit(nvcc)
    if not release:
        return fail("nvcc --version does not state a release")
    print("build_rot: step 1, nvcc %s compiles src/pearl_rot.cu for %s" % (release, arch))
    defs = ["-DROT_INLINE=0", "-DROT_C=16387"] if arch == "sm_120" else []
    if "b8=1" in ARCH_KEYS[arch]:
        defs.append("-DROT_TAB8=1")         # the b8 gathers read tables in plain row order (rot_aux_tab)
    if "sr=1" in ARCH_KEYS[arch]:
        defs.append("-DROT_SR=1")           # sr: records the eight warps share (rot_aux_offs)
    if "bx=1" in ARCH_KEYS[arch]:
        defs.append("-DROT_BX=1")           # bx: the transcript in the kernel's frame, the 16 KB for the exchange
    if "w8=1" in ARCH_KEYS[arch]:
        defs.append("-DROT_W8=1")           # w8: the 96 x 512 tile, the transcript in the kernel's frame
    if "w8m=1" in ARCH_KEYS[arch]:
        # w8m: the stage's mbarriers, made by the kernel before each tile (w8m1: counting warps, 2).
        defs.append("-DROT_W8M=%d" % (3 if "w8mc=1" in ARCH_KEYS[arch] else 2 if "w8m1=1" in ARCH_KEYS[arch] else 1))
    r = subprocess.run([nvcc, "-arch=" + arch, "-std=c++17", "-O3", "-cubin"] + defs + ["-I", SRC,
                        os.path.join(SRC, "pearl_rot.cu"), "-o", base], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        return fail("nvcc")
    if arch == "sm_120":
        print("build_rot: steps 3 to 5 for sm_120: rfold120.py, the hand loop encoded for sm_120 and read back")
        r = subprocess.run([nvdisasm, "-c", base], capture_output=True, text=True)
        if r.returncode != 0 or "error" in r.stderr:
            return fail("nvdisasm cannot read the compiled kernel")
        open(sass, "w", encoding="utf-8", newline="\n").write(r.stdout)
        r = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.join(HERE, "rfold120.py"), base, nvdisasm, tmp]
                           + ARCH_KEYS[arch])
        if r.returncode != 0 or not os.path.exists(tmp):
            if os.path.exists(tmp):
                os.remove(tmp)
            return fail("rfold120.py (exit %d)" % r.returncode)
        return finish(tmp, ref, release, arch, NAME, final, unverified)
    print("build_rot: step 2, the assembler against this compiler's own bytes")
    text = asm89.disassemble(base, nvdisasm)
    open(sass, "w", encoding="utf-8", newline="\n").write(text)
    if asm89.check(base, sass):
        return fail("asm89 does not reproduce this compiler's encodings")
    print("build_rot: steps 3 to 5, the hand loop: latency, what it expects of the compiled kernel, read-back")
    r = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.join(HERE, "rfold.py"), base, sass, nvdisasm, tmp]
                       + ARCH_KEYS[arch])
    if r.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        return fail("rfold.py (exit %d)" % r.returncode)
    return finish(tmp, ref, release, arch, NAME, final, unverified)


def finish(tmp, ref, release, arch, NAME, final, unverified):
    have = asm89.fold_hash(tmp)
    known = read_ref(ref).get(release, set())
    print("build_rot: step 6, the reference hash for toolkit %s" % release)
    print("  built:  %s" % have)
    for h in sorted(known):
        print("  listed: %s" % h)
    if have not in known:
        os.replace(tmp, unverified)
        return fail("this cubin's hash is not listed for toolkit %s in %s. It has not mined on a card. It is "
                    "kept as %s: run it on an RTX %s card (PEARL_ROT_CUBIN, probes/verify-hits.js), then add the "
                    "line '%s %s'." % (release, os.path.basename(ref), os.path.basename(unverified),
                                                   {"sm_89": "40", "sm_86": "30"}.get(arch, "50"), release, have))
    os.replace(tmp, final)
    print("build_rot: ok, %s (%d bytes) is a file that has mined with every hit verified" % (
        NAME, os.path.getsize(final)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
