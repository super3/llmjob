"""build_rot.py: builds the rotation fold's cubin, the file the CUDA host loads when the rotation fold is asked
for (PEARL_FOLD_ROT=1, or --rotation-fold on the CLI; src/pearl_host_rot.inc; off by default).

  python build_rot.py <nvcc> <nvdisasm> <out dir> [--ref <file>]

The fold's chunk loop is written by hand and put into a kernel the compiler made. So every step is checked, and
the file is only written when all of them pass:
  1. nvcc -arch=sm_89 -cubin src/pearl_rot.cu -> <out dir>/rot_base.cubin. As compiled, its chunk loop is the
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
     src/pearl_rot_sm89.ref lists for this toolkit: the hashes of the files that have mined on a card with
     every hit verified. A toolkit with no line there, or a hash that is not listed, fails: the file is left
     as pearl_rot_sm89.cubin.unverified, to be run on an RTX 40 card (PEARL_ROT_CUBIN names it; then
     probes/verify-hits.js) before its hash is added.
On any failure the exit code is 1 and <out dir> has no pearl_rot_sm89.cubin.
On Windows run it from a shell that has MSVC's environment (vcvars64.bat): nvcc needs cl.exe.
Put pearl_rot_sm89.cubin beside pearl_core.node, or name it in PEARL_ROT_CUBIN.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
import asm89  # noqa: E402

NAME = "pearl_rot_sm89.cubin"


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
    ref = os.path.join(SRC, "pearl_rot_sm89.ref")
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
    print("build_rot: step 1, nvcc %s compiles src/pearl_rot.cu for sm_89" % release)
    r = subprocess.run([nvcc, "-arch=sm_89", "-std=c++17", "-O3", "-cubin", "-I", SRC, os.path.join(SRC, "pearl_rot.cu"),
                        "-o", base], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        return fail("nvcc")
    print("build_rot: step 2, the assembler against this compiler's own bytes")
    text = asm89.disassemble(base, nvdisasm)
    open(sass, "w", encoding="utf-8", newline="\n").write(text)
    if asm89.check(base, sass):
        return fail("asm89 does not reproduce this compiler's encodings")
    print("build_rot: steps 3 to 5, the hand loop: latency, what it expects of the compiled kernel, read-back")
    r = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.join(HERE, "rfold.py"), base, sass, nvdisasm, tmp])
    if r.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        return fail("rfold.py (exit %d)" % r.returncode)
    have = asm89.fold_hash(tmp)
    known = read_ref(ref).get(release, set())
    print("build_rot: step 6, the reference hash for toolkit %s" % release)
    print("  built:  %s" % have)
    for h in sorted(known):
        print("  listed: %s" % h)
    if have not in known:
        os.replace(tmp, unverified)
        return fail("this cubin's hash is not listed for toolkit %s in %s. It has not mined on a card. It is "
                    "kept as %s: run it on an RTX 40 card (PEARL_ROT_CUBIN, probes/verify-hits.js), then add the "
                    "line '%s %s'." % (release, os.path.basename(ref), os.path.basename(unverified), release, have))
    os.replace(tmp, final)
    print("build_rot: ok, %s (%d bytes) is a file that has mined with every hit verified" % (
        NAME, os.path.getsize(final)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
