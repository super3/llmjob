# sm75patch.py <cubin> <patch list> -- patch the RTX 20 fold's machine code after ptxas.
#
#   python3 -I sm75patch.py pearl_kernel.cubin sm75/sbfix1.json
#
# Rewrites 16-byte instruction words of the .text.pearl_tile_fold_wmma section of one
# sm_75 cubin, in place. A patch list is only valid for the exact code it was made from,
# so it refuses, and leaves the file alone, unless the section's md5 (first 12 hex, as
# compilers.sh prints it) is the list's "before" and every word it changes still holds
# the list's "old" bytes; the md5 after the edit must then be the list's "after".
#
# A patch list is JSON:
#   {"description": "one line", "before": "<md5[:12]>", "after": "<md5[:12]>",
#    "words": [{"addr": "0x2b30", "old": "<32 hex>", "new": "<32 hex>", "note": "..."}]}
# addr is the word's offset in the section (the /*2b30*/ of the disassembly); old and new
# are its 16 bytes in file order. note is optional and only for people reading the list.
#
# The md5 covers the bytes in the file, not what the card runs: the driver rewrites some
# instructions when it loads a cubin. So it also refuses a word at an offset the fold's
# .nv.info lists (CODE_OFFSETS) or under a relocation into the fold.
#
# Standard library only. Errors are one line on stderr, and any refusal exits 1.
import hashlib
import json
import os
import struct
import sys

SECTION = b'.text.pearl_tile_fold_wmma'
EM_CUDA = 190
SHT_RELA, SHT_REL, SHT_CUDA_INFO = 4, 9, 0x70000000
# The .nv.info attributes that list places in the code, by id, with the names cuobjdump
# 12.8 prints for them (checked by giving it a cubin with each id). The driver can
# rewrite the instructions these name at load time: an EXIT, a cooperative-groups or
# warp-wide instruction, a load whose cache hint it sets. The 5c97fbadf64a fold has the
# EXIT, COOP_GROUP and INT_WARP_WIDE lists, and no relocation into its code.
CODE_OFFSETS = {
    0x14: 'BINDLESS_IMAGE_OFFSETS', 0x1c: 'EXIT_INSTR_OFFSETS', 0x1d: 'S2RCTAID_INSTR_OFFSETS',
    0x25: 'LD_CACHEMOD_INSTR_OFFSETS', 0x27: 'ATOM_SYS_INSTR_OFFSETS', 0x28: 'COOP_GROUP_INSTR_OFFSETS',
    0x2d: 'ATOMF16_EMUL_INSTR_OFFSETS', 0x2e: 'ATOM16_EMUL_INSTR_REG_MAP',
    0x31: 'INT_WARP_WIDE_INSTR_OFFSETS', 0x39: 'MBARRIER_INSTR_OFFSETS', 0x3a: 'COROUTINE_RESUME_ID_OFFSETS',
    0x40: 'INSTR_REG_MAP', 0x46: 'SYSCALL_OFFSETS', 0x47: 'SW_WAR_MEMBAR_SYS_INSTR_OFFSETS',
    0x57: 'STACK_CANARY_TRAP_OFFSETS', 0x59: 'LOCAL_CTA_ASYNC_STORE_OFFSETS',
}


class Refuse(Exception):
    pass


def md5(data):
    return hashlib.md5(data).hexdigest()[:12]


def is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in '0123456789abcdef' for c in s)


def load_list(path):
    with open(path) as f:
        try:
            spec = json.load(f)
        except ValueError as e:
            raise Refuse('the patch list is not valid JSON: %s' % e) from None
    if not isinstance(spec, dict):
        raise Refuse('the patch list is not a JSON object')
    extra = set(spec) - {'description', 'before', 'after', 'words'}
    if extra:
        raise Refuse('the patch list has unknown keys: ' + ', '.join(sorted(extra)))
    for k in ('before', 'after'):
        if not is_hex(spec.get(k), 12):
            raise Refuse('the patch list\'s "%s" is not 12 lowercase hex digits' % k)
    words = spec.get('words')
    if not isinstance(words, list) or not words:
        raise Refuse('the patch list has no words')
    seen = set()
    out = []
    for i, w in enumerate(words):
        if not isinstance(w, dict) or set(w) - {'addr', 'old', 'new', 'note'}:
            raise Refuse('word %d is not {addr, old, new[, note]}' % i)
        try:
            addr = int(w.get('addr'), 16)
        except (TypeError, ValueError):
            raise Refuse('word %d has no hex addr' % i) from None
        if addr < 0 or addr % 16 or addr in seen:
            why = 'repeated' if addr in seen else 'not a 16-byte aligned offset'
            raise Refuse('word %d: addr %s is %s' % (i, w['addr'], why))
        seen.add(addr)
        if not is_hex(w.get('old'), 32) or not is_hex(w.get('new'), 32):
            raise Refuse('word %d (%s): old and new must be 32 lowercase hex digits' % (i, w['addr']))
        out.append((addr, bytes.fromhex(w['old']), bytes.fromhex(w['new'])))
    return spec['before'], spec['after'], out


def fold_section(b):
    # The section headers of a 64-bit little-endian ELF, as compilers.sh's texthash.py reads them.
    if len(b) < 64 or b[:4] != b'\x7fELF' or b[4] != 2 or b[5] != 1:
        raise Refuse('not a 64-bit little-endian ELF file')
    machine = struct.unpack_from('<H', b, 0x12)[0]
    if machine != EM_CUDA:
        raise Refuse('ELF machine %d is not a CUDA cubin (%d)' % (machine, EM_CUDA))
    shoff = struct.unpack_from('<Q', b, 0x28)[0]
    ent, num, stri = struct.unpack_from('<HHH', b, 0x3A)
    if ent != 64 or not num or stri >= num or shoff + num * ent > len(b):
        raise Refuse('the ELF section header table is not readable')
    hs = [struct.unpack_from('<IIQQQQIIQQ', b, shoff + i * ent) for i in range(num)]
    names = bytes(b[hs[stri][4]:hs[stri][4] + hs[stri][5]])
    found = [i for i, h in enumerate(hs) if h[0] < len(names) and names[h[0]:names.find(b'\0', h[0])] == SECTION]
    if len(found) != 1:
        raise Refuse('%d %s sections, not 1' % (len(found), SECTION.decode()))
    off, size = hs[found[0]][4], hs[found[0]][5]
    if off + size > len(b):
        raise Refuse('%s runs past the end of the file' % SECTION.decode())
    return off, size, found[0], hs


def load_time_rewrites(b, hs, fold):
    # (offset, what) for each place in the fold the driver may change at load time: the
    # offsets in the fold's .nv.info code lists, and the targets of relocations into it.
    out = []
    for h in hs:
        if h[7] != fold or h[1] not in (SHT_CUDA_INFO, SHT_REL, SHT_RELA):
            continue
        if h[4] + h[5] > len(b):
            raise Refuse('a section linked to the fold runs past the end of the file')
        d = bytes(b[h[4]:h[4] + h[5]])
        if h[1] != SHT_CUDA_INFO:
            # An Elf64_Rel is 16 bytes and an Elf64_Rela 24; r_offset comes first.
            es = 16 if h[1] == SHT_REL else 24
            if len(d) % es:
                raise Refuse('a relocation section of the fold is not readable')
            out += [(struct.unpack_from('<Q', d, p)[0], 'a relocation') for p in range(0, len(d), es)]
            continue
        # .nv.info entries: format, attribute, then a 2-byte value, or for format 4
        # (EIFMT_SVAL) a 2-byte size and that many bytes.
        p = 0
        while p < len(d):
            if p + 4 > len(d) or d[p] not in (1, 2, 3, 4):
                raise Refuse('the fold\'s .nv.info is not readable at byte %d' % p)
            n = struct.unpack_from('<H', d, p + 2)[0] if d[p] == 4 else 0
            if p + 4 + n > len(d):
                raise Refuse('the fold\'s .nv.info is not readable at byte %d' % p)
            if d[p + 1] in CODE_OFFSETS:
                what = 'EIATTR_' + CODE_OFFSETS[d[p + 1]]
                out += [(v, what) for v in struct.unpack_from('<%dI' % (n // 4), d, p + 4)]
            p += 4 + n
    return out


def patch(cubin, plist):
    try:
        before, after, words = load_list(plist)
    except (Refuse, OSError) as e:
        raise Refuse('%s: %s' % (os.path.basename(plist), e)) from None
    try:
        return patch_cubin(cubin, before, after, words)
    except (Refuse, OSError, struct.error) as e:
        raise Refuse('%s: %s' % (os.path.basename(cubin), e)) from None


def patch_cubin(cubin, before, after, words):
    with open(cubin, 'rb') as f:
        b = bytearray(f.read())
    off, size, fold, hs = fold_section(b)
    got = md5(b[off:off + size])
    if got != before:
        why = ' (already patched?)' if got == after else ''
        raise Refuse('fold md5 %s, the list is for %s%s: not patched' % (got, before, why))
    for addr, old, _ in words:
        if addr + 16 > size:
            raise Refuse('word 0x%x is past the section\'s %d bytes: not patched' % (addr, size))
        cur = bytes(b[off + addr:off + addr + 16])
        if cur != old:
            raise Refuse('word 0x%x is %s, the list expects %s: not patched'
                         % (addr, cur.hex(), old.hex()))
    # A place the driver rewrites is taken as the 16 bytes from its offset, so a field a
    # relocation points into the middle of still counts.
    rewrites = load_time_rewrites(b, hs, fold)
    for addr, _, _ in words:
        for at, what in rewrites:
            if addr - 16 < at < addr + 16:
                raise Refuse('word 0x%x: the driver may rewrite 0x%x (%s) when it loads the cubin: not patched'
                             % (addr, at, what))
    for addr, _, new in words:
        b[off + addr:off + addr + 16] = new
    got = md5(b[off:off + size])
    if got != after:
        raise Refuse('patched fold md5 would be %s, the list says %s: not written' % (got, after))
    # A whole new file, then a rename, so a failed write cannot leave half a patch.
    tmp = cubin + '.sm75patch'
    with open(tmp, 'wb') as f:
        f.write(b)
    os.replace(tmp, cubin)
    with open(cubin, 'rb') as f:
        b = f.read()
    off, size = fold_section(b)[:2]
    got = md5(b[off:off + size])
    if got != after:
        raise Refuse('fold md5 %s after writing, the list says %s' % (got, after))
    return len(words), before, after


def main(argv):
    if len(argv) != 3:
        print('usage: python3 -I sm75patch.py <cubin> <patch list>', file=sys.stderr)
        return 2
    cubin, plist = argv[1], argv[2]
    try:
        n, before, after = patch(cubin, plist)
    except Refuse as e:
        print('sm75patch: %s' % str(e).replace('\n', ' '), file=sys.stderr)
        return 1
    print('sm75patch: %s: %d words patched with %s, fold %s -> %s'
          % (os.path.basename(cubin), n, os.path.basename(plist), before, after))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
