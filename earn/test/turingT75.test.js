// The RTX 20 folds at ptxas register-usage level 8 (PEARL_T75_RUL8, src/pearl_kernel_t75.cu).
//
// pearl_kernel_t75.cu is pearl_kernel.cu included whole with every linked name renamed, so CI
// can build it for sm_75 alone with -Xptxas --register-usage-level=8 and link it beside
// pearl_kernel.o, which stays byte for byte what it was. What this checks without a GPU:
// - every name with linkage in pearl_kernel.cu (and the fold headers it includes) is renamed,
//   so the two objects link (a missed one is a duplicate symbol);
// - CI builds the object for sm_75 alone at level 8 on the CUDA 12.8 entries only, tells the
//   host, archives it, and fails on a spill;
// - the host takes the level-8 kernel only when its sm_75 body loaded, and launches it with the
//   same arguments as the kernel it stands in for.
const fs = require('fs');
const path = require('path');

const NL = String.fromCharCode(10);
const SRC = path.join(__dirname, '..', 'native', 'src');
const read = (file) => fs.readFileSync(file, 'utf8').split(String.fromCharCode(13) + NL).join(NL);
const T75 = read(path.join(SRC, 'pearl_kernel_t75.cu'));
const KERNEL = read(path.join(SRC, 'pearl_kernel.cu'));
const HOST = read(path.join(SRC, 'pearl_host.cu'));
const CONFIG = read(path.join(SRC, 'pearl_config.h'));
const CI = read(path.join(__dirname, '..', '..', '.github', 'workflows', 'native-core.yml'));
const BOX = read(path.join(__dirname, '..', 'tuning', 'salad', 'nv20box.sh'));

// The renames, in file order: "#define <name> <new name>".
const renames = [...T75.matchAll(/^#define (\w+) (\w+)$/gm)].map((m) => [m[1], m[2]]);

// The names with linkage in the kernel's translation unit: pearl_kernel.cu and the local
// headers it includes (each once; their own includes are pearl_config.h and the fold headers).
function unitText() {
  const seen = new Set();
  const take = (name) => {
    if (seen.has(name)) return '';
    seen.add(name);
    const text = read(path.join(SRC, name));
    return text.replace(/^#include "([\w.]+)".*$/gm, (_, inc) => take(inc));
  };
  return take('pearl_kernel.cu');
}
const UNIT = unitText();

function linkedNames(text) {
  const names = new Set();
  // Kernels: extern "C" __global__ [__launch_bounds__(...)] void <name>(, the name on the same
  // line or the next.
  // Hopper's kernel has its launch bound under an #if, between the void and the name.
  const kernel = /extern "C"\s+__global__(?:\s+__launch_bounds__\([^)]*\))?\s+void\s+(?:#if \w+\s+__launch_bounds__\([^)]*\)\s+#endif\s+)?(\w+)\s*\(/g;
  for (const m of text.matchAll(kernel)) {
    // A name in capitals is a macro (PEARL_TALL_FN, below).
    if (!/^[A-Z_]+$/.test(m[1])) names.add(m[1]);
  }
  // Kernels named through a macro (the tall fold's two builds).
  for (const m of text.matchAll(/^#define PEARL_TALL_FN (\w+)$/gm)) names.add(m[1]);
  // Host functions with C linkage.
  for (const m of text.matchAll(/^extern "C" (?:void|int|bool|uint32_t) (\w+)\(/gm)) names.add(m[1]);
  // __device__ and __constant__ variables at file scope.
  for (const m of text.matchAll(/^(?:__device__|__constant__) [\w ]+? (\w+)\[[^\]]*\](?: =|;)/gm)) names.add(m[1]);
  // __device__ functions that are neither inline nor in the anonymous namespace (blake3's
  // chunk hash): the host side gets a symbol for each.
  const anonEnd = text.indexOf('}  // namespace');
  for (const m of text.slice(anonEnd).matchAll(/^__device__ (?:void|uint32_t|bool) (\w+)\(/gm)) names.add(m[1]);
  return names;
}

describe('the level-8 RTX 20 folds (PEARL_T75_RUL8)', () => {
  test('pearl_kernel_t75.cu renames every name with linkage, then includes the kernel file', () => {
    expect(T75.trimEnd().endsWith('#include "pearl_kernel.cu"')).toBe(true);
    const from = renames.map(([a]) => a);
    expect(new Set(from).size).toBe(from.length);
    expect(new Set(from)).toEqual(linkedNames(UNIT));
    // The two folds the host launches have the names it declares.
    expect(renames).toContainEqual(['pearl_tile_fold_wmma', 'pearl_t75_tile_fold_wmma']);
    expect(renames).toContainEqual(['pearl_tile_fold_wmma_rg', 'pearl_t75_tile_fold_wmma_rg']);
    // Every new name is new.
    for (const [, b] of renames) expect(UNIT.includes(b)).toBe(false);
    // The kernel file itself does not know about the second build.
    expect(KERNEL).not.toMatch(/T75/);
  });

  test('the scan finds what it should', () => {
    const names = linkedNames(UNIT);
    for (const n of ['pearl_tile_fold_wmma', 'pearl_tile_fold_hopper', 'pearl_tile_fold_tall_pinb',
      'pearl_noise_materialize_kblocked', 'pearl_tall_cyc_take', 'BLAKE3_IV', 'pearl_e4_flag', 'blake3_chunk_cv']) {
      expect([n, names.has(n)]).toEqual([n, true]);
    }
    expect(names.has('blake3_compress')).toBe(false);
  });

  test('off unless the build says so', () => {
    expect(CONFIG).toContain('#ifndef PEARL_T75_RUL8' + NL + '#define PEARL_T75_RUL8 0' + NL + '#endif');
  });

  test('CI builds it for sm_75 alone at level 8 on the CUDA 12.8 entries, links it, and checks it', () => {
    const entries = CI.split(/\n {10}- os: /).slice(1);
    expect(entries.map((e) => [/variant: '([^']*)'/.exec(e)[1], /^ {12}t75: 'yes'$/m.test(e)]))
      .toEqual([['', true], ['', true], ['-cu13', false], ['-cu13', false]]);
    const compile = CI.slice(CI.indexOf('      - name: Compile the CUDA kernels'));
    const step = compile.slice(0, compile.indexOf('\n      - name: ', 10));
    expect(step).toContain('if [ "${{ matrix.t75 }}" = "yes" ]; then');
    expect(step).toContain('-gencode arch=compute_75,code=sm_75 \\' + NL
      + '              -Xptxas -v -Xptxas --register-usage-level=8 \\' + NL
      + '              -c src/pearl_kernel_t75.cu -o cuda-build/pearl_kernel_t75.o');
    expect(step).toContain('T75="-DPEARL_T75_RUL8=1"');
    expect(step).toContain('${{ matrix.gencode }} $T75 \\' + NL + '            -c src/pearl_host.cu');
    // pearl_kernel.cu's own line has no level option.
    expect(step.match(/register-usage-level/g)).toHaveLength(1);
    expect(CI).toContain('if [ -f cuda-build/pearl_kernel_t75.o ]; then objs="$objs cuda-build/pearl_kernel_t75.o"; fi');
    expect(CI).toContain('nvcc -lib -cudart static $objs -o cuda-build/pearl_cuda.lib');
    expect(CI).toContain('ar rcs cuda-build/libpearl_cuda.a $objs');
    const check = CI.slice(CI.indexOf('      - name: Check the level-8 RTX 20 folds'));
    expect(check).toMatch(/^ {6}- name: Check the level-8 RTX 20 folds\n {8}if: matrix.t75 == 'yes'\n/);
    expect(check).toContain('for f in pearl_t75_tile_fold_wmma pearl_t75_tile_fold_wmma_rg; do');
    expect(check).toContain('[ "$spill" = 0 ] ||');
  });

  test('the host takes it only where its sm_75 body loaded, and launches it as the kernel it replaces', () => {
    expect(HOST).toContain('  bool bdT75 = false;' + NL);
    expect(HOST.match(/bdT75 = .*/g)).toEqual(['bdT75 = false;', 'bdT75 = true;', 'bdT75 = false;']);
    const at = HOST.indexOf('    s.hashRg = ctx->bdHashRg;' + NL + '#if PEARL_T75_RUL8');
    expect(at).toBeGreaterThan(0);
    const check = HOST.slice(at, HOST.indexOf('#endif', at));
    expect(check).toContain('if (cudaFuncGetAttributes(&ft, pearl_wmma_fold_fn(ctx)) != cudaSuccess' + NL
      + '          || ft.binaryVersion != 75) {');
    // Declared with the arguments of the kernels they stand in for.
    const decl = (k) => {
      const i = HOST.indexOf(`extern "C" __global__ void ${k}(`);
      return HOST.slice(HOST.indexOf('(', i), HOST.indexOf(');', i));
    };
    expect(decl('pearl_t75_tile_fold_wmma')).toBe(decl('pearl_tile_fold_wmma'));
    expect(decl('pearl_t75_tile_fold_wmma_rg')).toBe(decl('pearl_tile_fold_wmma_rg'));
    const launch = (k) => {
      const i = HOST.indexOf(`    ${k}<<<blocks, threads, smem, st>>>(`);
      expect([k, i > 0]).toEqual([k, true]);
      return HOST.slice(HOST.indexOf('(', HOST.indexOf('>>>', i)), HOST.indexOf(';', i));
    };
    expect(launch('pearl_t75_tile_fold_wmma')).toBe(launch('pearl_tile_fold_wmma'));
    expect(launch('pearl_t75_tile_fold_wmma_rg')).toBe(launch('pearl_tile_fold_wmma_rg'));
    expect(HOST).toContain('  else if (ctx->bdT75 && ctx->bdHashRg)' + NL + '    pearl_t75_tile_fold_wmma_rg<<<');
    expect(HOST).toContain('  else if (ctx->bdT75 && !ctx->bdHashRg)' + NL + '    pearl_t75_tile_fold_wmma<<<');
    expect(HOST).toContain('if (c->bdT75) snprintf(c->foldName + n, sizeof c->foldName - n, " RUL8");');
  });

  test('a Salad box can build the same core (nv20box.sh %t75)', () => {
    expect(BOX).toContain('%t75) t75="--register-usage-level=8" ;;');
    expect(BOX).toContain('-c "$K/src/pearl_kernel_t75.cu" -o "$K/cuda-build/pearl_kernel_t75.o"');
    expect(BOX).toContain('C="$C -DPEARL_T75_RUL8=1"');
  });
});
