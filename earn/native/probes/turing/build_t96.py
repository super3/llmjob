"""build_t96.py: builds the RTX 20 wide-tile fold's cubin, the file the CUDA host loads on compute 7.5 cards
(src/pearl_host_t96.inc; PEARL_FOLD_T96=0 turns the fold off).

  python build_t96.py <nvcc> <nvdisasm> <out dir> [--ref <file>] [key=value ...]

The fold's k loop is written by hand and put into a kernel the compiler made, so every step is checked and the
file is only written when all of them pass:
  1. nvcc -arch=sm_75 -cubin src/pearl_t96.cu -> <out dir>/t96_base.cubin. Its function pearl_t96_loop, as
     compiled, is the room and the register set for the hand loop. The host never loads this file.
  2. t96gen.py checks what the hand loop takes for granted of that kernel (where the arguments are, the
     registers and predicates the compiled function writes, the room, no word the driver may rewrite, the
     constants), runs the loop on a model of a warp against the sums worked out directly (when numpy is
     installed), writes the loop into a copy of the cubin and reads every word back through nvdisasm.
  3. The result's hash (sha256 with the string table zeroed, as the rotation fold's) must be one of those
     src/pearl_t96_sm75.ref lists for this toolkit: the files that have mined on an RTX 20 card with every hit
     verified. A hash that is not listed fails: the file is left as pearl_t96_sm75.cubin.unverified, to be run
     on an RTX 20 card (PEARL_T96_CUBIN names it; then probes/verify-hits.js) before its line is added, there
     and in kT96Cubins (src/pearl_host_t96.inc).
On any failure the exit code is 1 and <out dir> has no pearl_t96_sm75.cubin.
key=value arguments are t96gen.py's tuning keys and -DNAME=value ones are defines for the kernel, both for
races; a release build has none.
On Windows run it from a shell that has MSVC's environment (vcvars64.bat): nvcc needs cl.exe.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "hand89"))
import asm89  # noqa: E402

NAME = "pearl_t96_sm75.cubin"


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
    print("build_t96: FAILED: " + msg)
    return 1


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = sys.argv[1:]
    ref = os.path.join(SRC, "pearl_t96_sm75.ref")
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        del args[i:i + 2]
    defs = [a for a in args if a.startswith("-D")]   # timing-only races of the kernel's C part
    args = [a for a in args if not a.startswith("-D")]
    keys = [a for a in args if "=" in a]
    args = [a for a in args if "=" not in a]
    if len(args) != 3:
        print(__doc__)
        return 2
    nvcc, nvdisasm, out = args
    os.makedirs(out, exist_ok=True)
    base, final = os.path.join(out, "t96_base.cubin"), os.path.join(out, NAME)
    tmp, unverified = final + ".tmp", final + ".unverified"
    for f in (final, tmp, unverified):
        if os.path.exists(f):
            os.remove(f)
    release = toolkit(nvcc)
    if not release:
        return fail("nvcc --version does not state a release")
    print("build_t96: step 1, nvcc %s compiles src/pearl_t96.cu for sm_75" % release)
    r = subprocess.run([nvcc, "-arch=sm_75", "-std=c++17", "-O3", "-cubin"] + defs + ["-I", SRC,
                        os.path.join(SRC, "pearl_t96.cu"), "-o", base], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        return fail("nvcc")
    print("build_t96: step 2, the hand loop: what it expects of the compiled kernel, the model, read-back")
    r = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.join(HERE, "t96gen.py"), base, nvdisasm, tmp] + keys)
    if r.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        return fail("t96gen.py (exit %d)" % r.returncode)
    have = asm89.fold_hash(tmp)
    known = read_ref(ref).get(release, set())
    print("build_t96: step 3, the reference hash for toolkit %s" % release)
    print("  built:  %s" % have)
    for h in sorted(known):
        print("  listed: %s" % h)
    if have not in known:
        os.replace(tmp, unverified)
        return fail("this cubin's hash is not listed for toolkit %s in %s. It has not mined on a card. It is kept "
                    "as %s: run it on an RTX 20 card (PEARL_T96_CUBIN, probes/verify-hits.js), then add the line "
                    "'%s %s'." % (release, os.path.basename(ref), os.path.basename(unverified), release, have))
    os.replace(tmp, final)
    print("build_t96: ok, %s (%d bytes) is a file that has mined with every hit verified" % (
        NAME, os.path.getsize(final)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
