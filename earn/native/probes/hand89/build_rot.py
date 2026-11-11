"""build_rot.py: builds the rotation fold's cubin, the file the CUDA host loads when PEARL_FOLD_ROT=1
(src/pearl_host_rot.inc; off by default, and no release build makes or ships this file).

  python build_rot.py <nvcc> <nvdisasm> <out dir>

Three steps, each checked:
  1. nvcc -arch=sm_89 -cubin src/pearl_rot.cu -> <out dir>/rot_base.cubin. As compiled, its chunk loop is the
     gather fold: the room, the registers and the reference for the hand loop. The host never loads this file.
  2. nvdisasm -c -> <out dir>/rot_base.sass, and asm89's check of itself against it: every instruction of the
     listing whose form the assembler knows is assembled from its text and must give the cubin's own bytes.
  3. rfold.py: the hand-written loop in place of the compiled one -> <out dir>/pearl_rot_sm89.cubin. asm89
     disassembles the result and compares every instruction's text with what was asked for.
On Windows run it from a shell that has MSVC's environment (vcvars64.bat): nvcc needs cl.exe.
Put pearl_rot_sm89.cubin beside pearl_core.node, or name it in PEARL_ROT_CUBIN.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
import asm89  # noqa: E402


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    nvcc, nvdisasm, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    base, sass, final = (os.path.join(out, n) for n in ("rot_base.cubin", "rot_base.sass", "pearl_rot_sm89.cubin"))
    r = subprocess.run([nvcc, "-arch=sm_89", "-std=c++17", "-O3", "-cubin", "-I", SRC, os.path.join(SRC, "pearl_rot.cu"),
                        "-o", base], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        return 1
    text = asm89.disassemble(base, nvdisasm)
    open(sass, "w", encoding="utf-8", newline="\n").write(text)
    if asm89.check(base, sass):
        print("asm89 does not reproduce this compiler's encodings: not patched")
        return 1
    r = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.join(HERE, "rfold.py"), base, sass, nvdisasm, final])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
