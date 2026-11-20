'use strict';

// RTX 20's wide-tile fold as something a release carries (pearl_t96_sm75.cubin): how its cubin is built and
// refused, where it rides in each package, and what the host does before it mines on it. No flag asks for
// the fold: an RTX 20 card uses it when the file is beside the core, and PEARL_FOLD_T96=0 turns it off.
// These read the sources: the runners have no GPU, and the loop in that file is written by hand, so the
// checks around it are pinned here. rotFoldShip.test.js does the same for the rotation fold.

const fs = require('fs');
const path = require('path');

const EARN = path.join(__dirname, '..');
const read = (...p) => fs.readFileSync(path.join(EARN, ...p), 'utf8').replace(/\r\n/g, '\n');
const NATIVE_CORE = read('..', '.github', 'workflows', 'native-core.yml');
const MINER_BUILD = read('..', '.github', 'workflows', 'miner-build.yml');
const BUILD_T96 = read('native', 'probes', 'turing', 'build_t96.py');
const REF = read('native', 'src', 'pearl_t96_sm75.ref');
const HOST = read('native', 'src', 'pearl_host.cu');
const T96 = read('native', 'src', 'pearl_host_t96.inc');
const FILE = 'pearl_t96_sm75.cubin';

// A workflow step, from its "- name:" line to the next step.
function step(workflow, name) {
  const at = workflow.indexOf('      - name: ' + name);
  expect([name, at >= 0]).toEqual([name, true]);
  const next = workflow.indexOf('\n      - ', at + 1);
  return workflow.slice(at, next < 0 ? undefined : next);
}

// A function of the host, from its first line to the next line that starts a new one.
function fn(source, head) {
  const at = source.indexOf('\n' + head);
  expect([head, at >= 0]).toEqual([head, true]);
  const end = source.indexOf('\n}\n', at);
  return source.slice(at + 1, end + 2);
}

// The reference's lines: [toolkit, hash] for each one that is not a comment.
const refLines = REF.split('\n').filter((l) => l.trim() && !l.startsWith('#'));
// The strings in the host's list, comments left out.
function hostList() {
  const head = 'const char *const kT96Cubins[] = {';
  const at = T96.indexOf(head);
  expect(at).toBeGreaterThan(0);
  const body = T96.slice(at + head.length, T96.indexOf('};', at)).replace(/\/\/[^\n]*/g, '');
  return [...body.matchAll(/"([^"]*)"/g)].map((m) => m[1]);
}

describe('the cubin is only written when every check passes (build_t96.py)', () => {
  test('the steps run in order, and the file gets its name last', () => {
    const order = [
      '[nvcc, "-arch=sm_75", "-std=c++17", "-O3", "-cubin"]',
      'os.path.join(HERE, "t96gen.py"), base, nvdisasm, tmp]',
      'have = asm89.fold_hash(tmp)',
      'if have not in known:',
      'os.replace(tmp, unverified)',
      'os.replace(tmp, final)',
    ].map((text) => [text, BUILD_T96.indexOf(text)]);
    for (const [text, at] of order) expect([text, at > 0]).toEqual([text, true]);
    expect(order.map((o) => o[1])).toEqual(order.map((o) => o[1]).slice().sort((a, b) => a - b));
    // Nothing else writes the final name, and a build starts by removing the last one.
    expect(BUILD_T96.split('os.replace(tmp, final)')).toHaveLength(2);
    expect(BUILD_T96).toContain('NAME = "' + FILE + '"');
    expect(BUILD_T96).toContain('base, final = os.path.join(out, "t96_base.cubin"), os.path.join(out, NAME)');
    expect(BUILD_T96).toContain('for f in (final, tmp, unverified):\n        if os.path.exists(f):\n            os.remove(f)');
    // A failed hand-loop step leaves no file behind.
    expect(BUILD_T96).toContain('if r.returncode != 0 or not os.path.exists(tmp):\n        if os.path.exists(tmp):\n            os.remove(tmp)\n'
      + '        return fail("t96gen.py (exit %d)" % r.returncode)');
  });

  test('a hash that is not listed leaves .unverified and exits 1', () => {
    expect(BUILD_T96).toContain('tmp, unverified = final + ".tmp", final + ".unverified"');
    expect(BUILD_T96).toContain('if have not in known:\n        os.replace(tmp, unverified)\n        return fail(');
    // Every failure is exit 1.
    expect(BUILD_T96).toContain('def fail(msg):\n    print("build_t96: FAILED: " + msg)\n    return 1');
    expect(BUILD_T96).toContain('sys.exit(main())');
    // The list is the reference's, for the toolkit that built the file.
    expect(BUILD_T96).toContain('ref = os.path.join(SRC, "pearl_t96_sm75.ref")');
    expect(BUILD_T96).toContain('known = read_ref(ref).get(release, set())');
  });
});

describe('the reference and the host list the same files', () => {
  test('every line of the reference is a CUDA 12.8 hash', () => {
    for (const l of refLines) expect(l).toMatch(/^12\.8 [0-9a-f]{64}$/);
  });

  test('kT96Cubins holds the reference\'s hashes, and nothing else', () => {
    const ref = refLines.map((l) => l.split(' ')[1]);
    const host = hostList();
    if (ref.length === 0) {
      // No file has mined yet. C++ has no empty array, so the list holds one empty string, which no hash equals.
      expect(host).toEqual(['']);
      return;
    }
    for (const h of host) expect(h).toMatch(/^[0-9a-f]{64}$/);
    expect(host.slice().sort()).toEqual([...new Set(ref)].sort());
  });
});

describe('CI builds it with the release toolkit (native-core.yml)', () => {
  const name = 'Build RTX 20\'s wide-tile fold\'s cubin (not blocking)';

  test('in the two CUDA 12.8 jobs, after the addon, without blocking them', () => {
    const build = step(NATIVE_CORE, name);
    expect(build).toContain("        id: t96\n        if: matrix.variant == ''\n        continue-on-error: true\n");
    expect(build).toContain('"$py" -X utf8 probes/turing/build_t96.py nvcc nvdisasm rot-build/t96 || rc=$?');
    expect(build).toContain('if [ "$rc" != 0 ] || [ ! -f rot-build/t96/' + FILE + ' ]; then');
    expect(build).toContain('echo "::error::$msg"');
    expect(build).toContain('cp rot-build/t96/' + FILE + ' build/Release/' + FILE);
    expect(build).toContain('dnf install -y --setopt=install_weak_deps=False cuda-nvdisasm-12-8');
    // node-gyp rebuild empties build/, so the cubin is built after it.
    expect(NATIVE_CORE.indexOf('- name: ' + name)).toBeGreaterThan(NATIVE_CORE.indexOf('- name: Confirm the addon was produced'));
  });

  test('the cubin rides in the core\'s artifact, and what the build made is kept either way', () => {
    const at = NATIVE_CORE.indexOf('          name: pearl-core${{ matrix.variant }}-${{ matrix.os }}\n');
    expect(at).toBeGreaterThan(0);
    const upload = NATIVE_CORE.slice(at, NATIVE_CORE.indexOf('if-no-files-found:', at));
    expect(upload).toContain('            earn/native/build/Release/' + FILE + '\n');
    // The addon is in the same list, so a build with no cubin still has a file to upload.
    expect(upload).toContain('            earn/native/build/Release/${{ matrix.file }}\n');
    // rot-build/t96 is inside the folder the 12.8 jobs keep, with the .unverified file when the hash is not listed.
    expect(NATIVE_CORE).toContain("        if: always() && matrix.variant == ''\n        with:\n"
      + '          name: pearl-rot-build-${{ matrix.os }}\n          path: earn/native/rot-build\n');
  });

  test('the host\'s own hash of the file is checked against the build\'s', () => {
    const check = step(NATIVE_CORE, 'Check where the rotation fold\'s cubin is looked for (Linux, not blocking)');
    expect(check).toMatch(new RegExp('\\n +for f in [^\\n]* rot-build/t96/' + FILE.replace('.', '\\.') + '[^\\n]*; do\\n'));
    expect(check).toContain('[ -f "$f" ] || continue');
    // That step runs after the build, or it would find no file.
    expect(NATIVE_CORE.indexOf('- name: Check where the rotation fold\'s cubin is looked for')).toBeGreaterThan(NATIVE_CORE.indexOf('- name: ' + name));
  });
});

describe('every package carries it beside the core', () => {
  test('the installer and the AppImage, in resources/native', () => {
    const pkg = JSON.parse(read('package.json'));
    const native = Object.values(pkg.build)
      .flatMap((v) => (v && Array.isArray(v.extraResources) ? v.extraResources : []))
      .filter((x) => x.to === 'native');
    expect(native).toHaveLength(2);
    for (const r of native) expect(r.filter).toContain(FILE);
  });

  test('the Linux CLI and the HiveOS package, beside the binary', () => {
    const cli = read('scripts', 'build-cli.mjs');
    expect(cli).toContain("const t96 = join(root, 'vendor', 'native', '" + FILE + "');");
    // A cubin left in dist/ by an earlier build must not ship beside a core it was not built with.
    expect(cli).toContain("rmSync(join(dist, '" + FILE + "'), { force: true });\nif (existsSync(t96)) {\n"
      + "  copyFileSync(t96, join(dist, '" + FILE + "'));\n"
      + "  chmodSync(join(dist, '" + FILE + "'), 0o644);");
    expect(cli).toContain('warning: no vendor/native/' + FILE + ' — RTX 20 rigs will mine on the normal kernel');
    const hive = read('scripts', 'build-hiveos.mjs');
    expect(hive).toContain("const t96 = join(dist, '" + FILE + "');\nif (existsSync(t96)) {\n"
      + "  copyFileSync(t96, join(pkgDir, '" + FILE + "'));\n"
      + "  chmodSync(join(pkgDir, '" + FILE + "'), 0o644);");
    expect(hive).toContain('warning: no ' + FILE + ' — RTX 20 rigs will mine on the normal kernel');
    // The HiveOS smoke test: optional, and mode 644 when it is there.
    const smoke = read('..', '.github', 'scripts', 'hiveos-smoke.sh');
    expect(smoke).toMatch(/\nfor f in pearl_core_cu13\.node [^\n]* pearl_t96_sm75\.cubin [^\n]*; do\n {2}\[ -f "\$f" \] \|\| echo "::warning::/);
    expect(smoke).toMatch(/for f in h-manifest\.conf pearl_core\.node [^\n]* pearl_t96_sm75\.cubin [^\n]*; do mode_is "\$f" 644; done/);
  });

  test('the self-updater keeps it with its release\'s core', () => {
    const { OPTIONAL_ASSETS } = require('../src/shared/selfUpdate');
    expect(OPTIONAL_ASSETS).toContainEqual({ key: 't96CubinUrl', file: FILE });
  });

  test('the installer build says when it has none, and a release publishes it with the cores', () => {
    expect(MINER_BUILD).toContain('if [ -f vendor/native/' + FILE + ' ]; then');
    expect(MINER_BUILD).toContain('msg="no ' + FILE + ' staged — RTX 20 rigs will mine on the normal kernel.');
    // The warning names the step as native-core.yml does.
    expect(MINER_BUILD).toContain("Check the native-core run's 'Build RTX 20's wide-tile fold's cubin' step.");
    const publish = step(MINER_BUILD, 'Publish CLI + HiveOS package to release');
    const up = publish.indexOf('gh release upload "v$VERSION" dist/' + FILE + ' --clobber');
    expect(up).toBeGreaterThan(0);
    // Before the binary, as the cores are: a rig that updates mid-upload must not pair a new binary with an old fold.
    expect(up).toBeLessThan(publish.indexOf('gh release upload "v$VERSION" dist/llmjob-earn-cli-linux'));
    // A failed upload must not stop the binary's.
    expect(publish).toContain('|| optional_failed ' + FILE);
    // A release without one drops a stale copy, since the self-updater removes a rig's copy then.
    expect(publish).toContain('gh release delete-asset "v$VERSION" ' + FILE + ' --yes');
  });
});

describe('the host: RTX 20 only, checked before it mines, and the normal kernel when it cannot run', () => {
  test('only compute 7.5 cards', () => {
    const archOk = fn(T96, 'bool t96_arch_ok(int device) {');
    expect(archOk).toMatch(/\n {2}return major == 7 && minor == 5\b[^\n]*;\n\}$/);
    // A card whose compute level cannot be read is not one.
    expect(archOk).toMatch(/!= cudaSuccess\) \{\n {4}\(void\)cudaGetLastError\(\);\n {4}return false;\n {2}\}/);
    expect(fn(T96, 'bool t96_card_ok(const PearlProfile *p, int device, char *why, size_t why_len) {'))
      .toContain('if (!t96_arch_ok(device)) return no("it is for RTX 20 cards");');
  });

  test('on by default, and PEARL_FOLD_T96=0 turns it off', () => {
    expect(T96).toContain('bool t96_off() {\n  const char *v = getenv("PEARL_FOLD_T96");\n  return v && v[0] == \'0\' && v[1] == 0;\n}');
    // The only switch in front of it: no flag and no other variable asks for the fold.
    expect(HOST).toContain('if (profile && !t96_off() && t96_arch_ok(device)) {');
    expect(HOST).toContain('if (ctx->t96) return t96_submit(ctx, slot, st, nonce_base, regions_out, err, err_len);');
  });

  test('pearl_host_card_k keeps the profile\'s k, and marks the card only when its file passes', () => {
    const at = HOST.indexOf('extern "C" uint32_t pearl_host_card_k(const PearlProfile *profile, int device, char *note,');
    expect(at).toBeGreaterThan(0);
    const cardK = HOST.slice(at, HOST.indexOf('if (profile && rot_env() && rot_arch_ok(device)) {', at));
    expect(cardK).toContain('  if (profile && !t96_off() && t96_arch_ok(device)) {\n'
      + '    char why[128] = {0};\n'
      + '    if (t96_card_ok(profile, device, why, sizeof why)) {\n'
      + '      if (note && note_len) snprintf(note, note_len, "wide-tile fold");\n'
      + '      tT96Device = device;\n'
      + '    }\n'
      + '    return profile->k;\n'
      + '  }\n');
    // The mark is cleared first, so one card's answer is never another's.
    expect(cardK.indexOf('tT96Device = -1;')).toBeLessThan(cardK.indexOf('if (profile && !t96_off()'));
    expect(cardK.indexOf('tT96Device = -1;')).toBeGreaterThan(0);
  });

  test('a file that is not listed is never handed to the driver', () => {
    const listed = fn(T96, 'bool t96_listed(const std::vector<char> &image) {');
    expect(listed).toContain('if (!rot_fold_hash(copy.data(), copy.size(), hex)) return false;\n'
      + '  for (const char *h : kT96Cubins)\n    if (strcmp(h, hex) == 0) return true;\n  return false;\n}');
    // Only a developer's own file, named by PEARL_T96_CUBIN, is not held to the list.
    expect(listed).toContain('if (t96_named()) return true;');
    expect(T96).toContain('bool t96_named() {\n  const char *e = getenv("PEARL_T96_CUBIN");\n  return e && *e;\n}');
    const create = fn(T96, 'bool t96_create(Ctx *ctx, char *why, size_t why_len) {');
    const check = create.indexOf('!t96_listed(image))');
    expect(check).toBeGreaterThan(0);
    expect(create.indexOf('load(&r->mod, image.data())')).toBeGreaterThan(check);
    expect(create.split('load(&r->mod').length).toBe(2);
    // And the card is not marked for the fold at all with such a file.
    expect(fn(T96, 'bool t96_card_ok(const PearlProfile *p, int device, char *why, size_t why_len) {'))
      .toContain('if (!t96_listed(image)) return no("its file is not the one this core was built with");');
  });

  test('the startup check runs before the fold is used, and a failure leaves the release fold', () => {
    const create = fn(T96, 'bool t96_create(Ctx *ctx, char *why, size_t why_len) {');
    const check = create.indexOf('if (!t96_startup_check(ctx, cw, sizeof cw)) {\n'
      + '    if (why && why_len) snprintf(why, why_len, "its startup check failed: %s", cw);\n'
      + '    return no("", 0);\n  }');
    expect(check).toBeGreaterThan(create.indexOf('load(&r->mod, image.data())'));
    // Nothing of the context is changed for the fold until the check has passed.
    expect(create.indexOf('ctx->batch = r->tilesPerBatch * PEARL_ROT_UNITS;')).toBeGreaterThan(check);
    expect(create.indexOf('startup check passed')).toBeGreaterThan(check);
    // Every refusal takes the fold's state off the context, which is what every use of the fold tests.
    expect(create).toMatch(/auto no = \[&\]\(const char \*msg, int code\) \{\n[^}]*\n {4}t96_destroy\(ctx\);\n {4}return false;\n {2}\};/);
    expect(fn(T96, 'void t96_destroy(Ctx *ctx) {')).toContain('  delete r;\n  ctx->t96 = nullptr;\n}');
    // The check compares the hand-written kernel with the compiled one, word for word.
    const startup = fn(T96, 'bool t96_startup_check(Ctx *ctx, char *why, size_t why_len) {');
    expect(startup).toContain('t96_launch_fold(r, r->fFold, nullptr, ctx->dAp, ctx->dBp, dCtr, dDump, 0u, tiles, test, hits, &code)');
    expect(startup).toContain('t96_launch_fold(r, r->fRef, nullptr, ctx->dAp, ctx->dBp, dCtr, dDump + words, 0u, tiles, test, hits, &code)');
    expect(startup).toContain('if (got[i] != got[words + i]) wrong++;');
    expect(startup).toContain('if (wrong) return no("the hand-written loop and the compiled one disagree on this card, words wrong", (int)wrong);');
    // The create goes on without the fold, and says so in one line.
    expect(HOST).toContain('    if (!t96_create(ctx, why, sizeof why)) {\n'
      + '      (void)cudaGetLastError();\n'
      + '      fprintf(stderr, "[pearl] GPU %d: wide-tile fold not used: %s. Mining on the normal kernel\\n", ctx->device,\n'
      + '              why);\n'
      + '    }\n'
      + '  }\n\n  return ctx;\n}');
  });
});
