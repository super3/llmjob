'use strict';

// An AMD card whose fold has not run on its kind of card in a release (gfx9, gfx103x, or an architecture the
// host could not read) mines when the opener checks every hit before sending it (createCore's checkHits), and
// the core then reports core.untested; PearlMiner checks those hits (pearlMiner.test.js has that half). No HIP
// or CUDA compiler runs here, so the native half is read out of the sources, and the AMD host's gate is run in
// JS. The probe's half runs verify-hits-profile.js against a stand-in core whose hits are built in JS.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const CORE = read('src', 'pearl_core.cc');
const CUDA = read('src', 'pearl_host.cu');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const HELPER = path.join(__dirname, 'helpers', 'pearlHit.js');

function slice(text, from, to) {
  const i = text.indexOf(from);
  if (i < 0) throw new Error('not found: ' + from);
  const j = text.indexOf(to, i + from.length);
  return text.slice(i, j < 0 ? undefined : j + to.length);
}
// A body's statements, trimmed, without comments or blank lines.
const statements = (body) => body.split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('//'));

describe('the binding (pearl_core.cc)', () => {
  const ctor = slice(CORE, 'PearlCore::PearlCore(const Napi::CallbackInfo &info)', '\n}\n');

  it('reads checkHits as it reads cardK, and opens the card with it', () => {
    expect(ctor).toContain('Napi::Value ch = o.Get("checkHits");');
    expect(ctor).toContain('check_hits = ch.IsBoolean() && ch.As<Napi::Boolean>().Value();');
    expect(ctor).toContain('ctx_ = pearl_host_create_checked(&profile, check_hits, &untested_, err, sizeof(err));');
    // The only create it calls: the plain one would open no untested card for the app.
    expect(ctor).not.toMatch(/pearl_host_create\(/);
  });

  it('reports core.untested, false unless the host set it', () => {
    expect(CORE).toContain('bool untested_ = false;');
    expect(CORE).toContain('InstanceAccessor("untested", &PearlCore::GetUntested, nullptr)');
    expect(slice(CORE, 'Napi::Value PearlCore::GetUntested(', '\n}\n')).toContain('Napi::Boolean::New(info.Env(), untested_)');
    expect(CORE).toContain('createCore(profile, { deviceIndex, saltBase, saltStride, cardK, checkHits })');
  });
});

describe('the CUDA host (pearl_host.cu)', () => {
  it('has no untested card: the checked create is pearl_host_create, and says so', () => {
    const body = slice(CUDA, 'extern "C" void *pearl_host_create_checked(', '\n}\n');
    expect(statements(body).slice(2)).toEqual([
      'if (untested) *untested = false;', 'return pearl_host_create(profile, err, err_len);', '}',
    ]);
  });
});

describe('the AMD host (pearl_host_hip.cpp)', () => {
  const create = slice(HOST, 'extern "C" void *pearl_host_create_checked(', '\n}\n');
  const resolve = slice(HOST, 'void resolve_fold(Ctx *ctx) {', '\n}\n');

  it('tells resolve_fold whether the opener checks hits, and the opener whether the card is untested', () => {
    const lines = statements(create);
    const at = (s) => {
      const i = lines.indexOf(s);
      expect([s, i >= 0]).toEqual([s, true]);
      return i;
    };
    expect(at('if (untested) *untested = false;')).toBe(2);
    const set = at('ctx->checkHits = check_hits;');
    const fold = at('resolve_fold(ctx);');
    const told = at('if (untested) *untested = ctx->untested;');
    expect([at('Ctx *ctx = new Ctx();') < set, set < fold, fold < told]).toEqual([true, true, true]);
    expect(lines.slice(-2)).toEqual(['return ctx;', '}']);
    expect(HOST).toContain('  bool checkHits = false;');
    expect(HOST).toContain('  bool untested = false;');
    expect(HOST.match(/ctx->untested = /g)).toHaveLength(1);
  });

  it('keeps pearl_host_create for an opener that does not check, refusing untested cards for it', () => {
    const body = slice(HOST, 'extern "C" void *pearl_host_create(', '\n}\n');
    expect(statements(body).slice(1)).toEqual(['return pearl_host_create_checked(profile, false, nullptr, err, err_len);', '}']);
  });

  // The gate, from `const char *unchecked` to `ctx->fold = want;`, run in JS: what a card gets.
  const gate = slice(resolve, '  const char *unchecked = getenv("PEARL_AMD_UNTESTED");', '  ctx->fold = want;\n');
  const js = gate
    .replace('const char *unchecked = getenv("PEARL_AMD_UNTESTED");', 'const unchecked = env;')
    .replace(/ctx->/g, 'ctx.')
    .replace(/snprintf\(err, err_len,[\s\S]*?\);\n\s*return;/, "return 'refused';")
    .replace('ctx.fold = want;', "return ctx.untested ? 'checked' : 'mines';");
  const strncmp = (a, b, n) => (a.slice(0, n) === b.slice(0, n) ? 0 : 1);
  const strcmp = (a, b) => (a === b ? 0 : 1);
  const run = new Function('ctx', 'env', 'strncmp', 'strcmp', js);
  const card = (arch, { checkHits = false, env, emulated = false } = {}) =>
    run({ arch, checkHits, emulated, untested: false }, env, strncmp, strcmp);

  it('the gate reads as JS', () => {
    expect(js).not.toMatch(/->|snprintf|const char/);
    expect(gate).toContain('"PEARL_AMD_UNTESTED=1 (earn/native/amd/README.md, \\"Testing on real hardware\\")",');
  });

  it('an untested card is refused unless the opener checks hits, and checked when it does', () => {
    for (const arch of ['gfx1030', 'gfx1031', 'gfx1032', 'gfx908', 'gfx90a:sramecc+:xnack-', 'gfx942:sramecc+:xnack-', 'gfx950', '']) {
      expect([arch, card(arch)]).toEqual([arch, 'refused']);
      expect([arch, card(arch, { checkHits: true })]).toEqual([arch, 'checked']);
      // PEARL_AMD_UNTESTED=1, and only that, opens it for an opener that checks nothing (hashrate.js). It never
      // turns the check off: v0.6.8 and v0.6.9 told RX 6000 owners to set it for the reference fold, and those
      // cards now run dot4.
      for (const env of ['0', 'yes', '', undefined]) {
        expect([arch, env, card(arch, { env }), card(arch, { env, checkHits: true })]).toEqual([arch, env, 'refused', 'checked']);
      }
      expect([arch, card(arch, { env: '1' }), card(arch, { env: '1', checkHits: true })]).toEqual([arch, 'mines', 'checked']);
    }
  });

  it('a card that has mined in a release, and the CPU build, are never checked', () => {
    for (const arch of ['gfx1100', 'gfx1101', 'gfx1102', 'gfx1150', 'gfx1151', 'gfx1200', 'gfx1201']) {
      for (const checkHits of [false, true]) expect([arch, checkHits, card(arch, { checkHits })]).toEqual([arch, checkHits, 'mines']);
    }
    expect([card('', { emulated: true }), card('', { emulated: true, checkHits: true })]).toEqual(['mines', 'mines']);
    expect([card('gfx1100', { env: '1', checkHits: true }), card('gfx1201', { env: '1' })]).toEqual(['mines', 'mines']);
  });
});

// A stand-in addon: it prints the options it was opened with, and on setJob reports STUB_SALTS genuine hits,
// each with one transcript word changed when STUB_WRONG is set.
const STUB = `
const { EventEmitter } = require('events');
const { hitFor } = require(${JSON.stringify(HELPER)});
module.exports = { createCore(profile, opts = {}) {
  console.log('OPTS ' + JSON.stringify(opts));
  const c = new EventEmitter();
  c.k = profile.k;
  c.device = { index: 0, name: 'stand-in', fold: 'js' };
  const tamper = process.env.STUB_WRONG ? (jp) => { jp[9] ^= 1; } : undefined;
  c.setJob = (job) => setImmediate(() => {
    const h = hitFor(profile, job.header, { bound: job.target, jobId: job.jobId, tamper });
    for (let s = 0; s < Number(process.env.STUB_SALTS || 1); s++) c.emit('hit', { ...h, salt: s });
  });
  c.stop = () => {};
  return c;
} };
`;

describe('verify-hits-profile.js, with the check the app runs', () => {
  let tmp;
  beforeAll(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-checkhits-'));
    fs.writeFileSync(path.join(tmp, 'stub.js'), STUB);
  });
  afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));
  const run = (env) => {
    const e = { ...process.env, STUB_SALTS: '20', ...env };
    for (const k of ['PEARL_CARD_K', 'PEARL_SHARE_DUMP', 'STUB_WRONG']) if (!(k in env)) delete e[k];
    const r = spawnSync(process.execPath, [path.join(NATIVE, 'amd', 'test', 'verify-hits-profile.js'), path.join(tmp, 'stub.js'),
      '1', '256', '64', '64', '1'], { env: e, encoding: 'utf8' });
    const out = r.stdout.trim().split('\n');
    return { status: r.status, opts: JSON.parse(out.find((l) => l.startsWith('OPTS ')).slice(5)), json: JSON.parse(out.pop()) };
  };

  it('opens the core with checkHits, passes right hits and fails wrong ones with the check\'s reason', () => {
    const ok = run({});
    expect(ok.opts).toEqual({ checkHits: true });
    expect(ok.json).toMatchObject({ verified: 20, failed: 0, PASS: true });
    expect(ok.status).toBe(0);
    const bad = run({ STUB_WRONG: '1', PEARL_CARD_K: '1' });
    expect(bad.opts).toEqual({ cardK: true, checkHits: true });
    expect(bad.json).toMatchObject({ verified: 0, failed: 20, PASS: false });
    expect(bad.json.fails[0].why).toBe('jackpot hash mismatch (fold/transcript)');
    expect(bad.status).toBe(1);
  }, 60000);
});
