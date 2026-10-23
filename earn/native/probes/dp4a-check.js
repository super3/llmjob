// node dp4a-check.js [samples=400]
// Checks Pascal's dp4a fold (pearl_tile_fold_dp4a) without a GPU. Needs a C++ compiler
// (c++, g++ or clang++ on PATH, or CXX) and nothing from CUDA.
//
// For two small geometries it writes random noised operands -- every int8 value, not
// just the int7 the miner draws, so a sign slip in the dp4a byte order shows -- and runs
// dp4a-check.cc on them. That program does one launch of the fold on the CPU with the
// kernel's own index and inner-loop functions (src/pearl_fold_dp4a.h), checks every
// region against a plain fold of its own, and prints the transcripts. This script then
// recomputes a random sample of them with the JS reference itself (reference.js
// foldTranscript, with regionToTile picking the rows and columns, as verify-hits.js
// does), so the fold is checked against the same oracle the pool's verifier mirrors.
//
// The geometries: 2 row groups by 4 column groups from column offset 32 (a band
// shorter than PEARL_DP4A_BAND), and 20 row groups by 3 column groups from offset 8
// (two whole bands and a short one). PASS is every region matching both ways.
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { execFileSync, spawnSync } = require('child_process');

const R = path.join(__dirname, '..', '..', 'src', 'shared', 'miner') + path.sep;
const { PROFILE, regionToTile } = require(R + 'pearlhash');
const { foldTranscript } = require(R + 'reference');

const samples = Number(process.argv[2] || 400);
const { k, rank } = PROFILE;
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dp4a-check-'));
const bin = path.join(dir, 'dp4a-check');

function compile() {
  const tried = [process.env.CXX, 'c++', 'g++', 'clang++'].filter(Boolean);
  for (const cxx of tried) {
    try {
      execFileSync(cxx, ['-O2', '-std=c++17', '-I', path.join(__dirname, '..', 'src'),
        path.join(__dirname, 'dp4a-check.cc'), '-o', bin], { stdio: 'inherit' });
      return cxx;
    } catch (e) { /* the next compiler */ }
  }
  throw new Error('no C++ compiler worked (tried ' + tried.join(', ') + ')');
}

// A deterministic stream of bytes, so a failure can be run again exactly.
function bytes(seed, len) {
  const out = Buffer.alloc(len);
  let block = 0;
  for (let off = 0; off < len; off += 32, block++) {
    crypto.createHash('sha256').update(seed + ':' + block).digest().copy(out, off);
  }
  return out;
}

function run(name, m, n, colOff, colGroups) {
  const a = bytes(name + ':A', m * k), b = bytes(name + ':B', n * k);
  const fa = path.join(dir, name + '.A'), fb = path.join(dir, name + '.B');
  fs.writeFileSync(fa, a);
  fs.writeFileSync(fb, b);
  // Its own failures (a mismatch, or a layout check) exit 1 with the reason on stderr,
  // which goes straight to ours; the transcripts come back either way.
  const res = spawnSync(bin, [fa, fb, m, n, colOff, colGroups].map(String),
    { maxBuffer: 1 << 28, stdio: ['ignore', 'pipe', 'inherit'] });
  const lines = String(res.stdout || '').trim().split('\n');
  const tail = lines.pop() || '';
  const cpp = /checked (\d+) regions, (\d+) mismatches/.exec(tail);
  if (!cpp) return { name, regions: 0, cppMismatches: -1, jsChecked: 0, jsMismatches: -1 };
  const rowsValid = m / 16;
  const A = new Int8Array(a.buffer, a.byteOffset, a.length);
  const Bt = new Int8Array(b.buffer, b.byteOffset, b.length);
  const zero = Array.from({ length: 16 }, () => new Int8Array(k));
  const profile = { ...PROFILE, m, n };
  // A random sample of the batch, plus its first and last region.
  const pick = new Set([0, lines.length - 1]);
  const rnd = bytes(name + ':pick', samples * 4);
  for (let i = 0; i < samples && pick.size < lines.length; i++) {
    pick.add(rnd.readUInt32LE(i * 4) % lines.length);
  }
  let bad = 0;
  for (const i of pick) {
    const parts = lines[i].split(' ');
    const idx = Number(parts[0]);
    const got = parts.slice(1).map((h) => parseInt(h, 16) >>> 0);
    // The batch-local index, as the host makes it global: nonce_base is col_off rows of
    // regions in.
    const { rows, cols } = regionToTile(colOff * rowsValid + idx, profile);
    const jp = foldTranscript({ A, Bt, noiseA: zero, noiseB: zero, rows, cols, k, rank });
    if (Array.from(jp).some((w, j) => w !== got[j])) {
      bad++;
      if (bad <= 5) console.error(name + ': region ' + idx + ' differs from reference.js');
    }
  }
  return { name, regions: lines.length, cppMismatches: Number(cpp[2]),
    jsChecked: pick.size, jsMismatches: bad };
}

const cxx = compile();
const results = [
  run('short-band', 256, 1024, 32, 32),
  run('three-bands', 2560, 512, 8, 24),
];
fs.rmSync(dir, { recursive: true, force: true });
const PASS = results.every((r) => r.cppMismatches === 0 && r.jsMismatches === 0);
console.log(JSON.stringify({ compiler: cxx, results, PASS }));
process.exit(PASS ? 0 : 1);
