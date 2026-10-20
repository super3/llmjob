'use strict';

const fs = require('fs');
const path = require('path');

// The RTX 20 fold's post-ptxas patches: probes/turing/sm75/*.json, which sm75patch.py
// writes into the sm_75 cubin when a compilers.sh variant names one. ptxas sets the
// control bits itself, so a list is the only way to race SBFIX's scoreboard repair,
// or HASH_RG's role swap and AEF's load hint without the reschedule their source
// builds bring. Nothing here has the cubin, so each listed word is decoded with the
// sm_75 layout, and the checks are that a list changes only the fields it says it
// does, where it says, and that its notes say what its bits say. The lists' "before"
// is pinned to the default fold's hash in pearl_config.h, and sm75patch.py's guards
// and compilers.sh's parsing of the new field are pinned to their text. Whether a
// patch is safe in the rest of the loop (hazard.py's check) is not covered here.

const NATIVE = path.join(__dirname, '..', 'native');
const TURING = path.join(NATIVE, 'probes', 'turing');
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
// Line endings normalised: a Windows checkout has CRLF.
const read = (...parts) => fs.readFileSync(path.join(...parts), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const HEADER = read(NATIVE, 'src', 'pearl_config.h');
const README = read(NATIVE, 'probes', 'README.md');
const PATCHER = read(TURING, 'sm75patch.py');
const COMPILERS = read(TURING, 'compilers.sh');
const NAMES = ['sbfix1', 'sbfix2', 'hashrg', 'rg_aef'];
const LISTS = Object.fromEntries(NAMES.map((n) => [n, JSON.parse(read(TURING, 'sm75', n + '.json'))]));
const AFTER = { sbfix1: '8c23128cae97', sbfix2: '19e018bf281a', hashrg: '6d9fcd3f8d0e', rg_aef: 'd09b308bae13' };
const COUNT = { sbfix1: 15, sbfix2: 18, hashrg: 3, rg_aef: 39 };

// The default fold's chunk loop, from 0x19e0 to its back branch (/*4320*/ @P1 BRA to
// 0x19e0 in nvdisasm's listing of the 5c97fbadf64a cubin).
const LOOP = [0x19e0, 0x4320];
// The opcodes (bits 0-8) of the instructions the lists touch.
const OP = { IMMA: 0x037, LOP3: 0x012, SEL: 0x007, LDG: 0x181 };
const MNEMONIC = { 'IMMA.8816.S8.S8': OP.IMMA, 'LOP3.LUT': OP.LOP3, SEL: OP.SEL };

// A word's 16 bytes, in file order, as one little-endian 128-bit number.
function word(hex) {
  expect(hex).toMatch(/^[0-9a-f]{32}$/);
  let w = 0n;
  for (let i = 15; i >= 0; i--) w = (w << 8n) | BigInt(parseInt(hex.slice(2 * i, 2 * i + 2), 16));
  return w;
}
const bits = (w, lo, n) => Number((w >> BigInt(lo)) & ((1n << BigInt(n)) - 1n));

// The sm_75 fields: opcode 0-8, operand form 9-11 (1 registers, 4 an immediate),
// guard predicate 12-15 (7 is none), registers at 16-23 (destination), 24-31, 32-39
// and 64-71; and the control bits: stall 105-108, yield 109, write SB 110-112, read SB
// 113-115 (7 is none), wait mask 116-121, reuse 122-125.
function decode(hex) {
  const w = word(hex);
  return {
    op: bits(w, 0, 9), form: bits(w, 9, 3), guard: bits(w, 12, 4),
    dst: bits(w, 16, 8), a: bits(w, 24, 8), b: bits(w, 32, 8), c: bits(w, 64, 8),
    stall: bits(w, 105, 4), yld: bits(w, 109, 1), wsb: bits(w, 110, 3), rsb: bits(w, 113, 3),
    wait: bits(w, 116, 6), reuse: bits(w, 122, 4),
  };
}

// The bit numbers where a word's old and new bytes differ.
function changed(entry) {
  const x = word(entry.old) ^ word(entry.new);
  const out = [];
  for (let i = 0; i < 128; i++) if ((x >> BigInt(i)) & 1n) out.push(i);
  return out;
}

const addr = (entry) => parseInt(entry.addr, 16);
const byAddr = (list) => new Map(list.words.map((e) => [e.addr, e]));
// A wait mask as the notes write it: none, {SB0} or {SB0,SB1}.
const waitText = (m) => (m ? '{' + [0, 1, 2, 3, 4, 5].filter((i) => (m >> i) & 1).map((i) => 'SB' + i).join(',') + '}' : 'none');

// The text of the role words (SEL and LOP3.LUT with an immediate, as nvdisasm prints
// them), from their bits: the predicate operand is 87-89 with 90 negating it, the
// immediate 32-63 and LOP3's lookup table 72-79.
const reg = (r) => (r === 255 ? 'RZ' : 'R' + r);
const pred = (p, neg) => (neg ? '!' : '') + (p === 7 ? 'PT' : 'P' + p);
function roleText(hex) {
  const w = word(hex);
  const f = decode(hex);
  const guard = f.guard === 7 ? '' : '@' + pred(f.guard & 7, f.guard >> 3) + ' ';
  const imm = '0x' + bits(w, 32, 32).toString(16);
  const p = pred(bits(w, 87, 3), bits(w, 90, 1));
  if (f.op === OP.SEL) return guard + ['SEL ' + reg(f.dst), reg(f.a), f.form === 4 ? imm : reg(f.b), p].join(', ');
  expect(f.op).toBe(OP.LOP3);
  expect(f.form).toBe(4);
  return guard + ['LOP3.LUT ' + reg(f.dst), reg(f.a), imm, reg(f.c), '0x' + bits(w, 72, 8).toString(16), p].join(', ');
}

// bash's IFS='|' read -r with n names: each name but the last takes one field, the
// last takes the rest of the line, less one trailing '|' when that is the rest's only
// '|'. Checked below against what bash 5.2 gives.
function readFields(line, n) {
  const out = [];
  let rest = line;
  for (let i = 0; i < n - 1; i++) {
    const at = rest.indexOf('|');
    out.push(at < 0 ? rest : rest.slice(0, at));
    rest = at < 0 ? '' : rest.slice(at + 1);
  }
  out.push(rest.endsWith('|') && rest.indexOf('|') === rest.length - 1 ? rest.slice(0, -1) : rest);
  return out;
}

describe('RTX 20 fold patches (probes/turing/sm75)', () => {
  test('every list is for the default fold pearl_config.h names, and well formed', () => {
    const m = HEADER.match(/The fold's machine code under CUDA 12\.8: [^]*?\(with HIT_COORDS ([0-9a-f]{12}), the default\)/);
    expect(m).not.toBeNull();
    expect(m[1]).toBe('5c97fbadf64a');
    const seen = new Map();
    for (const n of NAMES) {
      const L = LISTS[n];
      expect(Object.keys(L).sort()).toEqual(['after', 'before', 'description', 'words']);
      expect(L.before).toBe(m[1]);
      expect(L.after).toBe(AFTER[n]);
      expect(L.description).not.toContain(NL);
      expect(L.words).toHaveLength(COUNT[n]);
      let last = -1;
      for (const e of L.words) {
        expect(Object.keys(e).sort()).toEqual(['addr', 'new', 'note', 'old']);
        expect(e.addr).toMatch(/^0x[0-9a-f]+$/);
        expect(addr(e) % 16).toBe(0);
        expect(addr(e)).toBeGreaterThan(last);
        last = addr(e);
        expect(e.new).not.toBe(e.old);
        expect(e.note.length).toBeGreaterThan(0);
        // All four were made from one cubin, so a word two lists touch starts the same.
        if (seen.has(e.addr)) expect(e.old).toBe(seen.get(e.addr));
        seen.set(e.addr, e.old);
      }
    }
    expect(new Set([m[1], ...Object.values(AFTER)]).size).toBe(5);
  });

  test('SBFIX changes only read scoreboards and wait masks, inside the chunk loop', () => {
    const NOTE = /^(IMMA\.8816\.S8\.S8|LOP3\.LUT): (?:read SB(\d) -> SB(\d)|wait (none|\{[SB0-9,]+\}) -> (none|\{[SB0-9,]+\}))$/;
    for (const n of ['sbfix1', 'sbfix2']) {
      for (const e of LISTS[n].words) {
        const where = n + ' ' + e.addr;
        expect([where, addr(e) >= LOOP[0] && addr(e) <= LOOP[1]]).toEqual([where, true]);
        // Bits 113-121: read SB and wait mask. Opcode, registers, stall, yield, write SB
        // and reuse stay.
        expect([where, changed(e).filter((i) => i < 113 || i > 121)]).toEqual([where, []]);
        const o = decode(e.old);
        const v = decode(e.new);
        const m = e.note.match(NOTE);
        expect([where, m !== null]).toEqual([where, true]);
        expect(o.op).toBe(MNEMONIC[m[1]]);
        if (m[2] !== undefined) {
          expect([o.rsb, v.rsb]).toEqual([Number(m[2]), Number(m[3])]);
          expect(v.wait).toBe(o.wait);
        } else {
          expect([waitText(o.wait), waitText(v.wait)]).toEqual([m[4], m[5]]);
          expect(v.rsb).toBe(o.rsb);
        }
      }
    }
    // SBFIX=1 as its description counts it: two IMMA waits moved (SB0 to SB1), five
    // IMMA operand reads moved to SB5 with their LOP3's wait, and three repair waits.
    const kinds = {};
    for (const e of LISTS.sbfix1.words) {
      const o = decode(e.old);
      const v = decode(e.new);
      let k;
      if (o.rsb !== v.rsb) k = 'read to SB5';
      else if (v.wait === 1 << 5) k = 'LOP3 waits on SB5';
      else if (o.wait === 1 && v.wait === 2) k = 'wait moved to SB1';
      else if ((v.wait & o.wait) === o.wait) k = 'wait added';
      kinds[k] = (kinds[k] || 0) + 1;
    }
    expect(kinds).toEqual({ 'read to SB5': 5, 'LOP3 waits on SB5': 5, 'wait moved to SB1': 2, 'wait added': 3 });
    expect(LISTS.sbfix1.description).toContain('two chunk-loop IMMA waits moved off the IMMA just before them (SB0 to SB1), five IMMA operand reads moved to SB5, three repair waits');
  });

  test('each IMMA read moved to SB5 is the one its LOP3 waited for', () => {
    const words = LISTS.sbfix1.words;
    const moved = words.filter((e) => decode(e.old).rsb !== decode(e.new).rsb);
    expect(moved.map((e) => e.addr)).toEqual(['0x2b30', '0x2ba0', '0x2c10', '0x2da0', '0x30d0']);
    // SB5 was not in any of these words before.
    for (const e of words) {
      const o = decode(e.old);
      expect([e.addr, o.rsb !== 5 && o.wsb !== 5 && !((o.wait >> 5) & 1)]).toEqual([e.addr, true]);
    }
    for (const e of moved) {
      const i = words.indexOf(e);
      const im = decode(e.old);
      const lop = words[i + 1];
      const lo = decode(lop.old);
      expect(im.op).toBe(OP.IMMA);
      expect(decode(e.new).rsb).toBe(5);
      // The next listed word is the LOP3 a few instructions on that waited only on the
      // IMMA's old read SB, and now waits only on SB5.
      expect(lo.op).toBe(OP.LOP3);
      expect(addr(lop) - addr(e)).toBeLessThanOrEqual(0x30);
      expect([lop.addr, lo.wait, decode(lop.new).wait]).toEqual([lop.addr, 1 << im.rsb, 1 << 5]);
      // It overwrites one of the IMMA's operands (A, B, or the accumulator pair C), so
      // it has to wait for the IMMA to read them; it reads none of the IMMA's results.
      expect([e.addr, [im.a, im.b, im.c, im.c + 1].includes(lo.dst)]).toEqual([e.addr, true]);
      expect([e.addr, [lo.a, lo.b, lo.c].filter((r) => r === im.dst || r === im.dst + 1)]).toEqual([e.addr, []]);
      // After the patch no IMMA has its read and its result on one scoreboard: 2c10 and
      // 2da0 had SB1 for both, and 30d0 SB0.
      expect(decode(e.new).rsb).not.toBe(im.wsb);
    }
    expect(moved.filter((e) => decode(e.old).rsb === decode(e.old).wsb).map((e) => e.addr))
      .toEqual(['0x2c10', '0x2da0', '0x30d0']);
    // The LOP3 at 2c40 waited on SB1, which also held the 2c10 IMMA's result R100-R101.
    // It now waits only on SB5, so the repair wait at 2e70, which reads R101, adds SB1.
    const at = byAddr(LISTS.sbfix1);
    const imma = decode(at.get('0x2c10').old);
    const repair = decode(at.get('0x2e70').old);
    expect([imma.dst, imma.wsb]).toEqual([100, 1]);
    expect(decode(at.get('0x2c40').old).wait).toBe(1 << 1);
    expect([repair.op, repair.a]).toEqual([OP.LOP3, 101]);
    expect(waitText(decode(at.get('0x2e70').new).wait)).toBe('{SB0,SB1}');
  });

  test('SBFIX=2 is SBFIX=1 and three IMMA waits dropped', () => {
    const two = byAddr(LISTS.sbfix2);
    for (const e of LISTS.sbfix1.words) {
      expect(two.get(e.addr)).toEqual(e);
    }
    const one = byAddr(LISTS.sbfix1);
    const extra = LISTS.sbfix2.words.filter((e) => !one.has(e.addr));
    expect(extra.map((e) => e.addr)).toEqual(['0x2030', '0x3560', '0x4240']);
    for (const e of extra) {
      expect(decode(e.old).op).toBe(OP.IMMA);
      expect([decode(e.old).wait, decode(e.new).wait]).toEqual([1, 0]);
      expect(changed(e)).toEqual([116]);
    }
    expect(LISTS.sbfix2.description).toContain('PEARL_BD_SBFIX=2: SBFIX=1, plus no wait on the IMMA at 2030, 3560 and 4240');
  });

  test('HASH_RG changes only the three role words, and none of their control bits', () => {
    const words = LISTS.hashrg.words;
    expect(words.map((e) => e.addr)).toEqual(['0x5e90', '0x5ed0', '0x5f40']);
    for (const e of words) {
      // After the chunk loop, at the hand-off; opcode, guard, destination and control
      // bits stay, only the operands change.
      expect(addr(e)).toBeGreaterThan(LOOP[1]);
      expect([e.addr, changed(e).filter((i) => i < 9 || (i >= 12 && i < 24) || i >= 105)]).toEqual([e.addr, []]);
      // The note is old -> new as nvdisasm prints them, and the bits say the same.
      expect(e.note).toBe(roleText(e.old) + ' -> ' + roleText(e.new));
    }
    expect(words.map((e) => e.note)).toEqual([
      'SEL R2, RZ, 0xffffffff, !P3 -> SEL R2, RZ, 0xffffffff, !P0',
      '@!P0 SEL R2, RZ, 0xffffffff, P4 -> @!P0 SEL R2, RZ, RZ, P4',
      'LOP3.LUT R3, R3, 0xffffffc0, R222, 0xe2, !PT -> LOP3.LUT R3, R222, 0x7f, RZ, 0xc0, !PT',
    ]);
  });

  test('rg_aef is hashrg and the evict-first bit of the 36 A loads', () => {
    const rg = byAddr(LISTS.hashrg);
    const loads = LISTS.rg_aef.words.filter((e) => !rg.has(e.addr));
    for (const e of LISTS.hashrg.words) expect(byAddr(LISTS.rg_aef).get(e.addr)).toEqual(e);
    expect(loads).toHaveLength(36);
    const LDG = /^LDG\.E(\.EF)?(\.128(?:\.CONSTANT)?\.SYS) R(\d+), \[R(\d+)\.U32\+UR(\d+)(?:\+0x([0-9a-f]+))?\]$/;
    for (const e of loads) {
      const w = word(e.old);
      const f = decode(e.old);
      expect(f.op).toBe(OP.LDG);
      // Only bit 84: bits 84-86 are the eviction priority, 1 (none shown) before and
      // 0 (.EF) after.
      expect([e.addr, changed(e)]).toEqual([e.addr, [84]]);
      expect([bits(w, 84, 3), bits(word(e.new), 84, 3)]).toEqual([1, 0]);
      // The note's old and new differ only by .EF, and name the load's registers and
      // offset: destination 16-23, address 24-31, uniform base 32-39, offset 40-63. A's
      // loads are the ones off UR6.
      const [before, after] = e.note.split(' -> ');
      const o = before.match(LDG);
      const v = after.match(LDG);
      expect([e.addr, o !== null && v !== null]).toEqual([e.addr, true]);
      expect([o[1], v[1]]).toEqual([undefined, '.EF']);
      expect(v.slice(2)).toEqual(o.slice(2));
      expect([f.dst, f.a, f.b, bits(w, 40, 24)]).toEqual([Number(o[3]), Number(o[4]), Number(o[5]), parseInt(o[6] || '0', 16)]);
      expect(f.b).toBe(6);
    }
    // SBFIX touches no load and no role word, so the lists could be combined.
    const fix = byAddr(LISTS.sbfix2);
    expect(LISTS.rg_aef.words.filter((e) => fix.has(e.addr))).toEqual([]);
  });

  test('no list changes an instruction the driver rewrites when it loads the fold', () => {
    // The 5c97fbadf64a fold's .nv.info code lists (cuobjdump -elf of a CUDA 12.8.93
    // build): the driver may rewrite these instructions, so a patch there might not run.
    // The fold has no relocation section. sm75patch.py checks this on the cubin itself.
    const rewritten = {
      EXIT: [0x90, 0x2d0, 0x9800],
      COOP_GROUP: [0x3c10, 0x3c30, 0x3c60, 0x3c80, 0x3d20, 0x3d40, 0x5b00, 0x5b20, 0x5b50, 0x5bd0, 0x5c00, 0x5c20],
      INT_WARP_WIDE: [0x9260, 0x92e0],
    };
    const all = new Set(Object.values(rewritten).flat());
    for (const n of NAMES) {
      expect(LISTS[n].words.filter((e) => all.has(addr(e))).map((e) => e.addr)).toEqual([]);
    }
  });

  test('the README lists each patch as it is', () => {
    expect(README).toContain('### Racing a patched fold: compilers.sh\'s patch field');
    expect(README).toContain('The lists in `probes/turing/sm75/`, all for the default fold (`5c97fbadf64a`)');
    for (const n of NAMES) {
      expect(README).toContain('| `' + n + '.json` | ' + LISTS[n].words.length + ' | `' + LISTS[n].after + '` |');
    }
    // sbfix2 leaves IMMA results to distance alone, and a race cannot show that is safe.
    expect(README).toContain('relies on an IMMA latency nobody has measured (below) |');
    expect(README).toContain('`sbfix2` is not safe just because it passes a race.');
    // The README's example spec names sbfix1's own after hash, and a list that exists.
    const spec = 'sbfix1||8c23128cae97|sm75/sbfix1.json';
    expect(README).toContain('(`' + spec + '`)');
    const [, , expectHash, patch] = readFields(spec, 4);
    expect(expectHash).toBe(LISTS.sbfix1.after);
    expect(fs.existsSync(path.join(TURING, patch))).toBe(true);
  });

  test('sm75patch.py refuses unless the fold and every old word match, and checks after before it writes', () => {
    // The fold section and the ELF reading are compilers.sh's texthash.py's.
    expect(PATCHER).toContain("SECTION = b'.text.pearl_tile_fold_wmma'");
    expect(COMPILERS).toContain('if n.startswith(".text."):');
    expect(COMPILERS).toContain('print(n[6:], hashlib.md5(b[h[4]:h[4] + h[5]]).hexdigest()[:12])');
    expect(COMPILERS).toContain('awk \'$1 == "pearl_tile_fold_wmma" {print $2; exit}\'');
    expect(PATCHER).toContain("shoff = struct.unpack_from('<Q', b, 0x28)[0]");
    expect(PATCHER).toContain("ent, num, stri = struct.unpack_from('<HHH', b, 0x3A)");
    expect(PATCHER).toContain("struct.unpack_from('<IIQQQQIIQQ', b, shoff + i * ent)");
    expect(PATCHER).toContain('return hashlib.md5(data).hexdigest()[:12]');
    expect(PATCHER).toContain('EM_CUDA = 190');
    // Standard library only.
    const imports = PATCHER.split(NL).filter((l) => /^\s*(import|from)\s/.test(l));
    expect(imports).toEqual(['import hashlib', 'import json', 'import os', 'import struct', 'import sys']);
    // The refusals: a list that is not exactly the format, a file that is not one
    // sm_75 cubin with one fold, a fold that is not the list's before, a word that is
    // not the list's old, and an after that does not come out.
    for (const s of [
      "raise Refuse('the patch list is not valid JSON: %s' % e) from None",
      "raise Refuse('the patch list is not a JSON object')",
      "extra = set(spec) - {'description', 'before', 'after', 'words'}",
      "raise Refuse('the patch list has unknown keys: ' + ', '.join(sorted(extra)))",
      'is not 12 lowercase hex digits',
      "raise Refuse('the patch list has no words')",
      "if not isinstance(w, dict) or set(w) - {'addr', 'old', 'new', 'note'}:",
      "raise Refuse('word %d has no hex addr' % i) from None",
      'if addr < 0 or addr % 16 or addr in seen:',
      "why = 'repeated' if addr in seen else 'not a 16-byte aligned offset'",
      'old and new must be 32 lowercase hex digits',
      "raise Refuse('not a 64-bit little-endian ELF file')",
      "raise Refuse('ELF machine %d is not a CUDA cubin (%d)' % (machine, EM_CUDA))",
      "raise Refuse('the ELF section header table is not readable')",
      "raise Refuse('%d %s sections, not 1' % (len(found), SECTION.decode()))",
      'runs past the end of the file',
      "why = ' (already patched?)' if got == after else ''",
      "raise Refuse('fold md5 %s, the list is for %s%s: not patched' % (got, before, why))",
      'is past the section' + BS + "'s %d bytes: not patched' % (addr, size))",
      "raise Refuse('word 0x%x is %s, the list expects %s: not patched'",
      "raise Refuse('patched fold md5 would be %s, the list says %s: not written' % (got, after))",
      "raise Refuse('fold md5 %s after writing, the list says %s' % (got, after))",
      // A word the driver may rewrite at load time: one at an offset in the fold's
      // .nv.info code lists, or within 16 bytes of a relocation's target.
      'SHT_RELA, SHT_REL, SHT_CUDA_INFO = 4, 9, 0x70000000',
      "0x1c: 'EXIT_INSTR_OFFSETS'",
      "0x25: 'LD_CACHEMOD_INSTR_OFFSETS'",
      "0x28: 'COOP_GROUP_INSTR_OFFSETS'",
      "0x31: 'INT_WARP_WIDE_INSTR_OFFSETS'",
      'if h[7] != fold or h[1] not in (SHT_CUDA_INFO, SHT_REL, SHT_RELA):',
      'es = 16 if h[1] == SHT_REL else 24',
      "raise Refuse('the fold" + BS + "'s .nv.info is not readable at byte %d' % p)",
      'if addr - 16 < at < addr + 16:',
      "raise Refuse('word 0x%x: the driver may rewrite 0x%x (%s) when it loads the cubin: not patched'",
    ]) {
      expect(PATCHER).toContain(s);
    }
    // In order: the before hash, every old word, the places the driver rewrites, then the
    // new words in memory, the after hash, and only then the file, replaced whole, and
    // read back.
    const at = PATCHER.indexOf('def patch_cubin(');
    const body = PATCHER.slice(at, PATCHER.indexOf(NL + 'def main(', at));
    const steps = [
      'if got != before:',
      'for addr, old, _ in words:',
      'if cur != old:',
      'rewrites = load_time_rewrites(b, hs, fold)',
      'the driver may rewrite',
      'for addr, _, new in words:',
      'b[off + addr:off + addr + 16] = new',
      'if got != after:',
      "tmp = cubin + '.sm75patch'",
      "with open(tmp, 'wb') as f:",
      'os.replace(tmp, cubin)',
      'after writing',
    ].map((s) => body.indexOf(s));
    expect(steps.every((p) => p > 0)).toBe(true);
    expect([...steps].sort((x, y) => x - y)).toEqual(steps);
    // That is the only write: the cubin is otherwise opened to read.
    expect(PATCHER.match(/open\([^)]*\)/g)).toEqual(['open(path)', "open(cubin, 'rb')", "open(tmp, 'wb')", "open(cubin, 'rb')"]);
    expect(PATCHER.split('os.replace(').length - 1).toBe(1);
    // One-line errors and the exit codes compilers.sh relies on: 1 refused, 2 usage.
    const main = PATCHER.slice(PATCHER.indexOf('def main('));
    expect(main).toContain("print('usage: python3 -I sm75patch.py <cubin> <patch list>', file=sys.stderr)" + NL + '        return 2');
    expect(main).toContain("print('sm75patch: %s' % str(e).replace('" + BS + "n', ' '), file=sys.stderr)" + NL + '        return 1');
    expect(main).toContain("print('sm75patch: %s: %d words patched with %s, fold %s -> %s'");
    expect(PATCHER).toContain('sys.exit(main(sys.argv))');
  });

  test('compilers.sh reads a 4th field, a patch list, and builds 2- and 3-field entries as before', () => {
    // readFields against bash 5.2's IFS='|' read -r name flags expect patch.
    for (const [line, want] of [
      ['base|', ['base', '', '', '']],
      ['base', ['base', '', '', '']],
      ['base||ac306adc8702', ['base', '', 'ac306adc8702', '']],
      ['a|b|c|', ['a', 'b', 'c', '']],
      ['a|b|c||', ['a', 'b', 'c', '']],
      ['a|b|c|d', ['a', 'b', 'c', 'd']],
      ['a|b|c|d|', ['a', 'b', 'c', 'd']],
      ['a|b|c|d|e', ['a', 'b', 'c', 'd|e']],
      ['a|||', ['a', '', '', '']],
    ]) {
      expect(readFields(line, 4)).toEqual(want);
    }
    expect(readFields('a|b|c||', 3)).toEqual(['a', 'b', 'c||']);
    expect(readFields('a|b|c|d|e', 3)).toEqual(['a', 'b', 'c|d|e']);

    // The read, and what build_variant does with an empty 4th field.
    expect(COMPILERS).toContain("if [ -n \"${VARIANTS_SPEC:-}\" ]; then IFS=';' read -r -a VARIANTS <<< \"$VARIANTS_SPEC\"; fi");
    expect(COMPILERS).toContain("IFS='|' read -r name flags expect patch <<< \"$v\"" + NL
      + '    build_variant "$name" "$flags" "$expect" "$patch"');
    expect(COMPILERS).toContain('build_variant() {   # name flags expect [patch list]' + NL
      + '  local name=$1 flags=$2 expect=$3 patch=${4:-} K=$OUT/v/$1 list="" after="" keep=()');
    // Without a patch, keep stays empty, so this is the old nvcc line, and the build
    // line has no patch=[...].
    expect(COMPILERS).toContain('if ! nvcc $COMMON $flags "${keep[@]}" -Xptxas -v -c "$K/src/pearl_kernel.cu" -o "$K/cuda-build/pearl_kernel.o" > "$K/ptxas.txt" 2>&1; then');
    expect(COMPILERS).toContain('fold=$h $([ -z "$expect" ] && echo "expect=-" || { [ "$h" = "$expect" ] && echo "expect=$expect match" || echo "expect=$expect MISMATCH"; })'
      + '$([ -n "$patch" ] && echo " patch=[$patch, $(grep -oE \'[0-9]+ words\' "$K/patch.txt")]") flags=[$flags]"');
    // Everything else the patch adds is behind [ -n "$patch" ].
    const at = COMPILERS.indexOf('build_variant() {');
    const bv = COMPILERS.slice(at, COMPILERS.indexOf(NL + '}', at));
    expect(bv.split('[ -n "$patch" ]').length - 1).toBe(4);
    expect(bv).toContain('  if [ -n "$patch" ]; then' + NL + '    list=$NATIVE/probes/turing/$patch' + NL);
    expect(bv).toContain('    keep=(--keep --keep-dir "$K/keep"); mkdir -p "$K/keep"' + NL + '  fi');
    expect(bv.split('keep=(--keep').length - 1).toBe(1);
    expect(bv).toContain('  if [ -n "$patch" ]; then patch_fold "$K" "$flags" "$list" "$name" || return 1; fi');
    expect(bv).toContain('  if [ -n "$patch" ] && { [ "$h" != "$after" ] || [ "$(fold_hash "$K/bench")" != "$after" ]; }; then');
    // Other readers of an entry take its name the same way with four fields.
    expect(COMPILERS).toContain("IFS='|' read -r name _ _ <<< \"$v\"");
    expect(COMPILERS).toContain('*" ${v%%|*} "*) keep+=("$v") ;;');

    // The default builds have three fields, so they read as before, with no patch.
    const list = COMPILERS.slice(COMPILERS.indexOf('VARIANTS=(' + NL));
    const entries = list.slice(0, list.indexOf(NL + ')' + NL)).split(NL).slice(1).map((l) => l.trim().replace(/^"|"$/g, ''));
    expect(entries.map((v) => v.split('|')[0])).toEqual(['base', 'rul8', 'cmr255', 'cmr255rul8']);
    for (const v of [...entries, 'base|', 'plain|-DPEARL_BD_LEAN=0 -DPEARL_BD_HIT_COORDS=0', 'base||5c97fbadf64a']) {
      const now = readFields(v, 4);
      expect(now.slice(0, 3)).toEqual(readFields(v, 3));
      expect(now[3]).toBe('');
      expect(now[0]).toBe(v.split('|')[0]);
    }
    // A 4th field names a list relative to probes/turing, with or without the hash.
    for (const n of NAMES) {
      for (const v of [n + '||' + LISTS[n].after + '|sm75/' + n + '.json', n + '|||sm75/' + n + '.json']) {
        const [name, flags, , patch] = readFields(v, 4);
        expect([name, flags, patch]).toEqual([n, '', 'sm75/' + n + '.json']);
        expect(fs.existsSync(path.join(TURING, patch))).toBe(true);
      }
    }
    expect(COMPILERS).toContain('python3 -I "$NATIVE/probes/turing/sm75patch.py" "$cubin" "$list" > "$K/patch.txt" 2>&1 ||');

    // The replay reads nvcc's steps exactly as the driver probe does (there from $P, and
    // the cubin named orig).
    for (const s of [
      "fb=$(grep '^#" + BS + "$ fatbinary ' \"$K/dryrun.txt\" | sed 's/^#" + BS + "$ //')",
      "host=$(tail -n 1 \"$K/dryrun.txt\" | sed 's/^#" + BS + "$ //; s/[[:space:]]*$//')",
      "cubin=$(printf '%s" + BS + "n' \"$fb\" | grep -oE 'sm=75,file=[^\"]+' | sed 's/^sm=75,file=//')",
    ]) {
      expect(COMPILERS).toContain(s);
      expect(COMPILERS).toContain(s.split('$K/').join('$P/').replace(/^cubin=/, 'orig='));
    }
    expect(COMPILERS).toContain('(eval "$fb" && eval "$host") > "$K/replay.txt" 2>&1 ||');
    expect(COMPILERS).toContain('(eval "$fb" && eval "$host") > "$D/swap.log" 2>&1 || ok=0');
  });
});
