"""inside.py: is the patched kernel that build.py wrote really inside the addon?

  python inside.py <addon.node> <out dir>:<arch> [<out dir>:<arch> ...]      (arch: sm_75, sm_86, sm_89, sm_120)

For each <out dir> (what build.py was given), three checks on bytes, not on a hash string:
  1. the image in qc_<arch>_cubin.h is the patched file qc_<arch>.cubin, byte for byte, and its kernel hash is the
     one the header names;
  2. its code is not the code of the kernel as compiled (qc_<arch>_red3.cubin): a build that embedded the compiled
     kernel under the patched kernel's hash passed every other check in November 2026 and ran at the compiled speed;
  3. the addon holds that image, whole.
One line for each architecture: "the patched <arch> (<cards>) kernel is in it: ..." or "MISSING patched <arch>
kernel in <addon>: <why>", and the exit status is 1 if any is missing. Python 3.6 and newer.
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build  # noqa: E402
import carryfix  # noqa: E402


def code_alone(b, kernel):
    off, size = carryfix.text_section(bytearray(b), kernel)
    return bytes(b[off:off + size])


def check(addon, out, arch):
    """None when the patched kernel of that directory is inside the addon, or why not."""
    tag = arch.replace("_", "")
    header = os.path.join(out, "qc_%s_cubin.h" % tag)
    patched = os.path.join(out, "qc_%s.cubin" % tag)
    raw = os.path.join(out, "qc_%s_red3.cubin" % tag)
    for f in (header, patched, raw):
        if not os.path.exists(f):
            return "no %s (build.py did not finish there)" % os.path.basename(f), None
    h = open(header).read()
    kernel = re.search(r'_kernel\[\] = "([^"]+)"', h)
    named = re.search(r'_hash\[\] = "([0-9a-f]{64})"', h)
    if not kernel or not named or "_image[] = {" not in h:
        return "%s is not a header build.py wrote" % os.path.basename(header), None
    kernel, named = kernel.group(1), named.group(1)
    image = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-f]{2})", h.split("_image[] = {", 1)[1]))
    if image != open(patched, "rb").read():
        return "the header's image is not %s" % os.path.basename(patched), None
    if build.kernel_hash(patched, kernel) != named:
        return "the image's hash is not the one the header names (%s)" % named, None
    code = code_alone(image, kernel)
    compiled = code_alone(open(raw, "rb").read(), kernel)
    if code == compiled:
        return "the header's image has the code of the kernel as compiled, not the patched code", None
    if image not in addon:
        return "the addon does not hold the header's image (%d bytes)" % len(image), None
    sha = lambda b: hashlib.sha256(b).hexdigest()[:16]  # noqa: E731
    return None, ("hash %s, code alone %s (%d bytes; as compiled %s), image of %d bytes found whole in the addon"
                  % (named, sha(code), len(code), sha(compiled), len(image)))


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    addon = open(argv[1], "rb").read()
    missing = 0
    for spec in argv[2:]:
        out, arch = spec.rsplit(":", 1)
        if arch not in build.ARCHS:
            sys.exit("unknown architecture %r" % arch)
        why, found = check(addon, out, arch)
        if why:
            missing += 1
            print("MISSING patched %s kernel in %s: %s" % (arch, os.path.basename(argv[1]), why))
        else:
            print("the patched %s (%s) kernel is in it: %s" % (arch, build.ARCHS[arch][1], found))
    sys.stdout.flush()
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
