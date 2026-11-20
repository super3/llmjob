'use strict';

// handfold.js's ops for RDNA 3's wave32 fold (pearl_amd_fold_wmma11), run on the chunk loop as ROCm 7.2's clang
// writes it for gfx1100 (fixtures/amdFold11Loop.s: the loads before the loop, the loop less most of its readout,
// and the kernel's register count). No compiler runs here. A small model walks the code into the loop and twice
// round it, and says what each WMMA adds to which sums and whether its operands' loads are known to be back.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const HAND = path.join(__dirname, '..', 'native', 'amd', 'hand');
const { opOf } = require(path.join(HAND, 'handfold.js'));
const BUILD = fs.readFileSync(path.join(HAND, '..', 'build-amd.sh'), 'utf8').replace(/\r\n/g, '\n');
const FIXTURE = fs.readFileSync(path.join(__dirname, 'fixtures', 'amdFold11Loop.s'), 'utf8').replace(/\r\n/g, '\n');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

const WMMA = /v_wmma\S+\s+v\[(\d+):\d+\],\s*v\[(\d+):\d+\],\s*v\[(\d+):\d+\]/;
const DEST = /^\s+\S+\s+v\[(\d+):\d+\],\s*(.*)$/;

// The WMMAs of the way into the loop and two turns of it, as "slab: sums += A' load * B' load", each slab's sorted.
// Loads of a kind come back in the order they were issued, and a wait for N lets the wave on once at most N are
// out. A WMMA whose operand may still be out, or was never loaded, throws.
function meaning(text) {
  const lines = text.split('\n');
  const label = new Map();
  lines.forEach((l, i) => { const m = /^(\.LBB[0-9_]+):/.exec(l); if (m) label.set(m[1], i); });
  const holds = new Map();   // register quad -> the load that filled it
  let vm = [];
  let lds = [];
  const slabs = [[]];
  let turns = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].split(';')[0];
    const op = opOf(line);
    if (op === 'buffer_load_b128' || op === 'ds_load_b128') {
      const m = DEST.exec(line);
      const load = { what: `${op} ${m[2].trim()} (turn ${turns})` };
      holds.set(Number(m[1]), load);
      (op === 'ds_load_b128' ? lds : vm).push(load);
    } else if (op === 'ds_store_b128') {
      lds.push({ what: 'store' });
    } else if (op === 's_waitcnt') {
      const v = /vmcnt\((\d+)\)/.exec(line);
      const g = /lgkmcnt\((\d+)\)/.exec(line);
      if (v) vm = vm.slice(vm.length - Math.min(vm.length, Number(v[1])));
      if (g) lds = lds.slice(lds.length - Math.min(lds.length, Number(g[1])));
    } else if (op === 's_barrier') {
      slabs.push([]);
    } else if (WMMA.test(line)) {
      const m = WMMA.exec(line);
      const a = holds.get(Number(m[2]));
      const b = holds.get(Number(m[3]));
      if (!a || !b) throw new Error(`line ${i + 1}: an operand was never loaded`);
      if (!a.what.startsWith('ds_load') || !b.what.startsWith('buffer_load')) throw new Error(`line ${i + 1}: A' is not an LDS read or B' not a buffer load`);
      if (lds.includes(a) || vm.includes(b)) throw new Error(`line ${i + 1}: an operand's load may still be out`);
      slabs[slabs.length - 1].push(`v${m[1]} += ${a.what} * ${b.what}`);
    } else if (/^s_branch$/.test(op) || op === 's_cbranch_vccnz') {
      const to = label.get(/(\.LBB[0-9_]+)/.exec(line)[1]);
      if (op === 's_cbranch_vccnz' && ++turns === 2) break;
      i = to;
    }
  }
  return slabs.filter((s) => s.length).map((s) => s.sort());
}

// The instructions between a slab's first and last WMMA, other than WMMAs and waits.
function breaks(text) {
  const lines = text.split('\n').filter((l) => opOf(l));
  const w = lines.map((l, i) => (WMMA.test(l) ? i : -1)).filter((i) => i >= 0);
  const out = [];
  for (const [a, b] of [[w[0], w[31]], [w[32], w[63]]]) {
    out.push(lines.slice(a, b + 1).filter((l) => !WMMA.test(l) && opOf(l) !== 's_waitcnt').map(opOf));
  }
  return out;
}

describe('handfold.js on RDNA 3\'s loop', () => {
  let tmp;
  let src;
  beforeAll(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-hand11-'));
    src = path.join(tmp, 'in.s');
    fs.writeFileSync(src, FIXTURE);
  });
  afterAll(() => { fs.rmSync(tmp, { recursive: true, force: true }); });
  const edit = (ops, input = src) => {
    const out = path.join(tmp, `out-${ops.replace(/\W/g, '_')}.s`);
    fs.rmSync(out, { force: true });
    const r = spawnSync(process.execPath, [path.join(HAND, 'handfold.js'), input, out, ops, 'pearl_amd_fold_wmma11'], { encoding: 'utf8' });
    return { status: r.status, stdout: r.stdout, stderr: r.stderr, text: fs.existsSync(out) ? fs.readFileSync(out, 'utf8') : null };
  };

  it('reads the compiler\'s loop as four slabs of 32 WMMAs with one break of three reads in each', () => {
    const said = meaning(FIXTURE);
    expect(said.map((s) => s.length)).toEqual([32, 32, 32, 32]);
    expect(breaks(FIXTURE)).toEqual([Array(3).fill('ds_load_b128'), Array(3).fill('ds_load_b128')]);
  });

  it('counts the waits as the compiler does (w11rw gives its loop back)', () => {
    const r = edit('w11rw');
    expect([r.status, r.stderr]).toEqual([0, '']);
    const code = (t) => t.split('\n').filter((l) => opOf(l));
    expect(code(r.text)).toEqual(code(FIXTURE));
  });

  it('a8 reads all eight A\' tiles before a slab\'s first WMMA, in 12 new registers, and adds the same products', () => {
    const r = edit('a8');
    expect([r.status, r.stderr]).toEqual([0, '']);
    expect(r.stdout).toBe('handfold: pearl_amd_fold_wmma11 loop of 64 WMMAs, 153 instructions -> 153 (a8)\n');
    expect(breaks(r.text)).toEqual([[], []]);
    expect(r.text).toMatch(/\n\t\t\.amdhsa_next_free_vgpr 229\n/);
    expect(r.text).toMatch(/ds_load_b128 v\[217:220\], v191 offset:784\n\tds_load_b128 v\[221:224\], v191 offset:1552\n\tds_load_b128 v\[225:228\], v191 offset:2320\n/);
    expect(meaning(r.text)).toEqual(meaning(FIXTURE));
    // the B' registers the compiler reused for A' hold only B' now
    const wm = r.text.split('\n').filter((l) => WMMA.test(l)).map((l) => WMMA.exec(l));
    expect(wm.filter((m) => Number(m[2]) < 192)).toEqual([]);
  });

  it.each(['a8am', 'a8as', 'a8bm', 'a8bs'])('%s reorders a slab\'s WMMAs and adds the same products, each after its loads', (op) => {
    const r = edit(op);
    expect([r.status, r.stderr]).toEqual([0, '']);
    expect(breaks(r.text)).toEqual([[], []]);
    expect(meaning(r.text)).toEqual(meaning(FIXTURE));
    // the order: the held operand changes every fourth WMMA, and in the back-and-forth orders only one changes a WMMA
    const wm = r.text.split('\n').filter((l) => WMMA.test(l)).map((l) => WMMA.exec(l)).slice(0, 16);
    const held = op[2] === 'a' ? 2 : 3;
    const other = op[2] === 'a' ? 3 : 2;
    for (let i = 1; i < 16; i++) {
      expect(wm[i][held] === wm[i - 1][held]).toBe(i % 4 !== 0);
      if (op[3] === 's' && i % 4 === 0) expect(wm[i][other]).toBe(wm[i - 1][other]);
    }
    expect(wm.map((m) => m[0])).not.toEqual(edit('a8').text.split('\n').filter((l) => WMMA.test(l)).map((l) => WMMA.exec(l)[0]).slice(0, 16));
  });

  it('the model refuses a WMMA that runs before its read is back', () => {
    const early = edit('a8').text.replace('s_waitcnt vmcnt(8) lgkmcnt(7)', 's_waitcnt vmcnt(8) lgkmcnt(8)');
    expect(() => meaning(early)).toThrow(/may still be out/);
  });

  it('prices the loop\'s parts by cutting them (timing only)', () => {
    const count = (t, re) => t.split('\n').filter((l) => re.test(opOf(l))).length;
    expect(count(FIXTURE, /^buffer_load_b128$/)).toBe(27);
    expect(count(edit('nob11').text, /^buffer_load_b128$/)).toBe(11);   // the 8 before the loop, the staging loads
    expect(count(edit('nost11').text, /^buffer_load_b128$/)).toBe(25);
    expect(count(edit('nost11').text, /^ds_store_b128$/)).toBe(1);      // the one after the loop's branch
    expect(count(edit('noa').text, /^ds_load_b128$/)).toBe(0);
    expect(count(edit('nobar11').text, /^s_barrier$/)).toBe(0);
    expect(count(edit('noro11').text, /^v_xor/)).toBe(0);
    expect(count(edit('nob11').text, /^s_clause$/)).toBe(0);
  });

  it('writes nothing and exits 3 for a loop it does not know', () => {
    const other = path.join(tmp, 'other.s');
    fs.writeFileSync(other, FIXTURE.replace(/\tds_load_b128 v\[208:211\], v191 offset:16\n/, ''));
    const r = edit('a8', other);
    expect(r.status).toBe(3);
    expect(r.text).toBe(null);
    expect(r.stderr).toMatch(/^handfold: /);
    const nocount = path.join(tmp, 'nocount.s');
    fs.writeFileSync(nocount, FIXTURE.replace(/.*amdhsa_next_free_vgpr.*\n/, ''));
    expect(edit('a8', nocount).stderr).toBe('handfold: no register count found for the kernel\n');
  });

  bashIt('build-amd.sh takes the route for gfx1100 and gfx1101 under PEARL_AMD_HAND11 only', () => {
    const shFn = (name) => {
      const at = BUILD.indexOf('\n' + name + '() {\n');
      return BUILD.slice(at + 1, BUILD.indexOf('\n}\n', at) + 3);
    };
    const hand = (arch, flags) => spawnSync('bash', ['-c', shFn('arch_flags') + shFn('wave64_of') + shFn('hand_of') + 'hand_of ' + arch],
      { env: { PATH: process.env.PATH, PEARL_AMD_CFLAGS: flags }, encoding: 'utf8' }).stdout;
    expect(hand('gfx1100', '-DPEARL_AMD_HAND11=a8')).toBe('a8\n');
    expect(hand('gfx1101', '-O1 -DPEARL_AMD_HAND11=a8as+nodly')).toBe('a8as+nodly\n');
    expect(hand('gfx1100', '-DPEARL_AMD_HAND=g8')).toBe('');
    expect(hand('gfx1201', '-DPEARL_AMD_HAND11=a8')).toBe('');
    expect(hand('gfx1102', '-DPEARL_AMD_HAND11=a8')).toBe('');
    expect(hand('gfx1100', '')).toBe('');
    expect(shFn('hand_co')).toMatch(/case "\$a" in gfx11\*\) fn=pearl_amd_fold_wmma11 ;; esac/);
  });
});
