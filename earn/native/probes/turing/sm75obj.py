# sm75obj.py <object> <patch list> -- apply an sm75patch.py list to the sm_75 cubin inside a host object.
#
#   python3 -I sm75obj.py cuda-build/pearl_kernel_t75.o sm75g/<list>.json
#
# nvcc -c puts the device code into the host object (ELF on Linux, COFF on Windows) as a fatbinary
# that holds the sm_75 cubin byte for byte. This finds the one embedded CUDA ELF that has the list's
# section, patches a copy of it with sm75patch.py (every check of that script applies: the fold's md5
# before and after, every old word, no word the driver rewrites at load time), and writes it back in
# place. The cubin keeps its size, so nothing else in the object moves. Refuses, and leaves the file
# alone, unless exactly one embedded cubin has the section.
#
# Standard library only. Errors are one line on stderr, and any refusal exits 1.
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sm75patch  # noqa: E402


def cubins(b):
    """(offset, size) of each embedded little-endian 64-bit CUDA ELF in b."""
    out = []
    p = b.find(b'\x7fELF')
    while p >= 0:
        if p + 64 <= len(b) and b[p + 4] == 2 and b[p + 5] == 1 and \
                struct.unpack_from('<H', b, p + 0x12)[0] == sm75patch.EM_CUDA:
            shoff = struct.unpack_from('<Q', b, p + 0x28)[0]
            ent, num = struct.unpack_from('<HH', b, p + 0x3A)
            size = shoff + ent * num
            if ent == 64 and num and p + size <= len(b):
                out.append((p, size))
        p = b.find(b'\x7fELF', p + 1)
    return out


def main(argv):
    if len(argv) != 3:
        print('usage: python3 -I sm75obj.py <object> <patch list>', file=sys.stderr)
        return 2
    obj, plist = argv[1], argv[2]
    try:
        before, after, words = sm75patch.load_list(plist)
        with open(obj, 'rb') as f:
            b = bytearray(f.read())
        hits = []
        for off, size in cubins(b):
            try:
                sm75patch.fold_section(bytes(b[off:off + size]))
                hits.append((off, size))
            except sm75patch.Refuse:
                pass
        if len(hits) != 1:
            raise sm75patch.Refuse('%d embedded cubins have %s, not 1' % (len(hits), sm75patch.SECTION.decode()))
        off, size = hits[0]
        fd, tmp = tempfile.mkstemp(suffix='.cubin')
        os.close(fd)
        try:
            with open(tmp, 'wb') as f:
                f.write(b[off:off + size])
            n = sm75patch.patch_cubin(tmp, before, after, words)[0]
            with open(tmp, 'rb') as f:
                new = f.read()
        finally:
            os.remove(tmp)
        if len(new) != size:
            raise sm75patch.Refuse('the patched cubin changed size')
        b[off:off + size] = new
        with open(obj + '.sm75obj', 'wb') as f:
            f.write(b)
        os.replace(obj + '.sm75obj', obj)
    except (sm75patch.Refuse, OSError, struct.error) as e:
        print('sm75obj: %s: %s' % (os.path.basename(obj), str(e).replace('\n', ' ')), file=sys.stderr)
        return 1
    print('sm75obj: %s: %d words patched with %s in the cubin at byte %d, fold %s -> %s'
          % (os.path.basename(obj), n, os.path.basename(plist), off, before, after))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
