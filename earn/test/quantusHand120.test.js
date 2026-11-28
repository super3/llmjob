// The hand-written kernel body for RTX 50 (native/quantus/cuda/sm120/full120.py): sm75/full75.py's body with what
// sm_120 needs done differently, put into the cubin sm86/build.py builds for either toolkit. The card checks it
// against the compiled kernel when it opens. These tests pin what was learned on the card (28 Nov) and what the
// tools rely on; the arithmetic itself is modelled in quantusHand75.test.js.
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const cuda = path.join(__dirname, '..', 'native', 'quantus', 'cuda');
const read = (...p) => fs.readFileSync(path.join(cuda, ...p), 'utf8');
const build = read('sm86', 'build.py');
const full = read('sm120', 'full120.py');

describe('RTX 50 hand body: the build', () => {
  test('build.py builds it for sm_120 with CUDA 13 by default, and keeps the older kernels behind flags', () => {
    expect(build).toContain('HAND120 = "dmx:1+sum2:1"');
    expect(build).toContain('hand120 = a.arch == "sm_120" and (explicit or (cu13 and "-DQC_HAND120=0" not in a.defs');
    expect(build).toContain('flags = [f for f in flags if not f.startswith("-DQC_RED=")] + ["-DQC_HAND75=2", "-DQC_COMPACT=2"]');
    expect(build).toContain('code, _stalls, per, steps = full120.run(out + ".unverified", out + ".unverified",');
  });

  test('the kernel is listed in qc_sm120.ref only after it ran on a card, as the other patched kernels are', () => {
    expect(build).toContain('if h not in listed(ref, version):');
  });
});

describe('RTX 50 hand body: what the card needed', () => {
  test('a write to a constant load\'s pointer register waits on the load\'s scoreboard', () => {
    expect(full).toContain('if base_reg in dest_regs(x):\n                        wait[i] |= 1 << sb');
  });

  test('nothing is issued sooner than four cycles after a constant load', () => {
    expect(full).toContain('DEFAULTS = dict(lat=6, plat=6, imx=0, anch=0, sq2=0, dmx=0, sqmul=0, ldcs=4)');
    expect(full).toContain('s = max(stalls[i], OPT["stall"], OPT["ldcs"] if x.startswith("LDC") else 0)');
  });

  test('a second write to a register waits as long as ptxas makes it wait on sm_120', () => {
    expect(full).toContain('hand.LATW = WriteAfterWrite()');
    expect(full).toContain('return OPT.get("waw", 6)');
  });

  test('the return address stays in the instruction ptxas wrote, where ptxas put it', () => {
    expect(full).toContain('labs[".RA"] = info["ret_mov"]');
    expect(full).toContain('elif i == xi:\n            continue');
  });

  test('every instruction is read back through nvdisasm and the function simulated against the reference', () => {
    expect(full).toContain('assert bad == 0, "%d instructions read back differently" % bad');
    expect(full).toContain('assert (m.r[lo] | m.r[hi] << 32) % P == w[i], "simulation: element %d is wrong" % i');
  });

  test('no uniform registers and no multiplier-pipe adds: sm_120 gains nothing from them', () => {
    expect(full).not.toMatch(/emit\("U(MOV|LDC|IADD3)/);
  });
});

// The probes' reference (sm120/probe_ref.py) is the arithmetic of the kernel around the function. Its line for the
// whole body is what the compiled kernel printed on an RTX 5070 through "qcdrv - probe", so it is pinned here.
describe('RTX 50 hand body: the probes', () => {
  const py = ['python3', 'python'].find((c) => {
    const r = spawnSync(c, ['-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)']);
    return r.status === 0;
  });
  (py ? test : test.skip)('probe_ref.py gives the compiled kernel\'s hits for the whole body', () => {
    const r = spawnSync(py, [path.join(cuda, 'sm120', 'probe_ref.py'), '0'], { encoding: 'utf8' });
    expect(r.stdout.trim()).toBe('0 probe 126 hits e64d11b21ac3fcdc2fed13dcd59874b960810874088dd00fd762fb569bf90d90');
  });
});
