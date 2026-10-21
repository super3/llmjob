'use strict';

const fs = require('fs');
const path = require('path');
const { keyedHash, compress, MSG_PERMUTATION } = require('../src/shared/miner/blake3');

// Turing's split hash (PEARL_BD_HASH_SPLIT=N, off by default). At hand-off h the
// partner warp of each scheduler (X: warp 4, 5, 2 or 3) runs the first N of the 14
// half-rounds of the transcript compression for tile h - 1's regions and leaves the
// state in shared. After the barrier the hasher (Y: warp 0, 1, 6 or 7) loads it, runs
// the other 14 - N and tests the region. The block's last tile Y hashes whole after
// the loop. A flag a region (sHread) stops X overwriting a state Y has not loaded.
//
// Nothing here can run the device code, so this copies it to the CPU and pins the
// copy to the source, line for line. It checks:
// - each half, for every N, against BLAKE3's keyed hash, through the shared slot, for
//   2000 random keys and transcripts: all 8 words, and the top word in both byte
//   orders;
// - that the two roles take hash_tile's regions, each region once, with one flag and
//   one state slot of its own;
// - the hand-off, run as 256 threads at random speeds over 1 to 200 tiles: every
//   region of every tile hashed once, with its own words and coordinates, the same
//   hits as the shipped hash, nothing at the first hand-off, the last tile after the
//   loop, and no wait that nothing will end;
// - the guards, through a small preprocessor, and the host's shared footprint.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
// Line endings normalised: a Windows checkout has CRLF.
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const FOLD = read('pearl_fold_bd.cuh');
const HEADER = read('pearl_config.h');
const HOST = read('pearl_host.cu');
const KERNEL = read('pearl_kernel.cu');

// Source without comments: comment lines dropped, trailing ones cut. The pins below
// hold the code, so a reworded comment does not break them.
function code(text) {
  return text.split(NL).filter((l) => !l.trim().startsWith('//')).map((l) => {
    const at = l.indexOf('//');
    return at > 0 && l.slice(0, at).trim() ? l.slice(0, at).trimEnd() : l;
  }).join(NL);
}
const FOLD_CODE = code(FOLD);
const lines = (...l) => l.join(NL);

// The body of the function or lambda whose head is head, braces and all.
function body(src, head) {
  const at = src.indexOf(head);
  expect([head, at >= 0]).toEqual([head, true]);
  const open = src.indexOf('{', at + head.length - 1);
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    if (src[i] === '{') depth++;
    if (src[i] === '}' && --depth === 0) return src.slice(open, i + 1);
  }
  throw new Error('unbalanced ' + head);
}

// The parts come in this order in src, each of them once.
function inOrder(src, ...parts) {
  let at = -1;
  for (const p of parts) {
    const next = src.indexOf(p);
    expect([p, next > at, src.indexOf(p, next + 1)]).toEqual([p, true, -1]);
    at = next;
  }
}

function xorshift(seed) {
  let x = (seed >>> 0) || 1;
  return () => {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    return x;
  };
}
const unit = (rand) => rand() / 4294967296;

// ---------------------------------------------------------------------------
// The compression, as the fold has it (pinned in 'the copy matches the source').

const rotr = (x, n) => ((x >>> n) | (x << (32 - n))) >>> 0;
// pearl_kernel.cu's g().
function g(s, a, b, c, d, mx, my) {
  s[a] = (s[a] + s[b] + mx) >>> 0;
  s[d] = rotr((s[d] ^ s[a]) >>> 0, 16);
  s[c] = (s[c] + s[d]) >>> 0;
  s[b] = rotr((s[b] ^ s[c]) >>> 0, 12);
  s[a] = (s[a] + s[b] + my) >>> 0;
  s[d] = rotr((s[d] ^ s[a]) >>> 0, 8);
  s[c] = (s[c] + s[d]) >>> 0;
  s[b] = rotr((s[b] ^ s[c]) >>> 0, 7);
}

// pearl_bd_half_rounds<H0, H1>: half-round 2r is round r's column step, 2r + 1 its
// diagonal step, and the message is permuted after every round, run or not.
function halfRounds(s, m0, h0, h1) {
  const m = m0.slice();
  for (let r = 0; r < 7; r++) {
    if (2 * r >= h0 && 2 * r < h1) {
      g(s, 0, 4, 8, 12, m[0], m[1]);
      g(s, 1, 5, 9, 13, m[2], m[3]);
      g(s, 2, 6, 10, 14, m[4], m[5]);
      g(s, 3, 7, 11, 15, m[6], m[7]);
    }
    if (2 * r + 1 >= h0 && 2 * r + 1 < h1) {
      g(s, 0, 5, 10, 15, m[8], m[9]);
      g(s, 1, 6, 11, 12, m[10], m[11]);
      g(s, 2, 7, 8, 13, m[12], m[13]);
      g(s, 3, 4, 9, 14, m[14], m[15]);
    }
    if (r < 6) {
      const p = [m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13],
        m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]];
      for (let i = 0; i < 16; i++) m[i] = p[i];
    }
  }
}

const IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a];
const FLAGS = 1 | 2 | 8 | 16;  // PEARL_TRANSCRIPT_FLAGS: CHUNK_START | CHUNK_END | ROOT | KEYED_HASH

// split_first: X's starting state, then half-rounds [0, N).
function splitFirst(key, tm, N) {
  const s = new Array(16);
  for (let i = 0; i < 8; i++) s[i] = key[i];
  s[8] = IV[0]; s[9] = IV[1]; s[10] = IV[2]; s[11] = IV[3];
  s[12] = 0; s[13] = 0; s[14] = 64; s[15] = FLAGS;
  halfRounds(s, tm, 0, N);
  return s;
}

// The pair's slot: X stores word 4q + i of lane j's state as uint4 q * 32 + j, word i
// of it, and Y loads it from there (split_first, split_rest). Word offsets in the slot.
const slotWord = (lane, q, i) => (q * 32 + lane) * 4 + i;
function xStore(slot, lane, s) {
  for (let q = 0; q < 4; q++) for (let i = 0; i < 4; i++) slot[slotWord(lane, q, i)] = s[4 * q + i];
}
function yLoad(slot, lane) {
  const s = new Array(16);
  for (let q = 0; q < 4; q++) for (let i = 0; i < 4; i++) s[4 * q + i] = slot[slotWord(lane, q, i)];
  return s;
}

const bswap = (x) => (((x & 0xff) << 24) | ((x & 0xff00) << 8) | ((x >>> 8) & 0xff00) | (x >>> 24)) >>> 0;
// pearl_hash_word_msf: word i of the hash counting from the most significant.
const msf = (h, i, hbe) => (hbe ? bswap(h[i]) : h[7 - i]);
// split_rest's test word: output words 0 and 7 only.
function topWord(s, hbe) {
  const out = [(s[0] ^ s[8]) >>> 0, 0, 0, 0, 0, 0, 0, (s[7] ^ s[15]) >>> 0];
  return msf(out, 0, hbe);
}
// pearl_hash_meets_words.
function meets(h, target, hbe) {
  for (let i = 0; i < 8; i++) {
    const hw = msf(h, i, hbe);
    if (hw !== target[i]) return hw < target[i];
  }
  return true;
}

// BLAKE3's rounds as the spec states them, a half-round at a time, for the other side
// of each check: round r reads the message permuted r times (blake3.js's
// MSG_PERMUTATION); the column step mixes column i with words 2i and 2i + 1, the
// diagonal step diagonal i with words 8 + 2i and 9 + 2i.
const SCHEDULE = [Array.from({ length: 16 }, (_, i) => i)];
for (let r = 1; r < 7; r++) SCHEDULE.push(MSG_PERMUTATION.map((p) => SCHEDULE[r - 1][p]));
function specHalfRounds(s, m, h0, h1) {
  for (let hr = h0; hr < h1; hr++) {
    const w = SCHEDULE[hr >> 1];
    for (let i = 0; i < 4; i++) {
      if (hr % 2 === 0) g(s, i, 4 + i, 8 + i, 12 + i, m[w[2 * i]], m[w[2 * i + 1]]);
      else g(s, i, 4 + ((i + 1) & 3), 8 + ((i + 2) & 3), 12 + ((i + 3) & 3), m[w[8 + 2 * i]], m[w[9 + 2 * i]]);
    }
  }
}
const specStart = (key) => [...key, ...IV, 0, 0, 64, FLAGS];
const hashWords = (s) => Array.from({ length: 8 }, (_, i) => (s[i] ^ s[i + 8]) >>> 0);

// ---------------------------------------------------------------------------
// The block's layout (pinned below): 8 warps, 128 regions, 16 words a region.

const TWORDS = 2048;
// Word c of region L in a buffer (readout_st, load_tm, hash_tile).
const wordOf = (L, c) => L * 16 + (c ^ ((L >> 1) & 15));
// readout_st: region rl of a lane, written by lanes 0-3 and 16-19 only (-1 for the rest).
const writerRegion = (warp, lane, rl) => (((lane >> 2) & 3) !== 0 ? -1
  : warp * 16 + (lane & 3) * 4 + 2 * rl + ((lane >> 4) & 1));
// hs_ids: scheduler, lane, region and role; the slot's first word, after both buffers.
function hsIds(t) {
  const s = (t >> 5) & 3;
  const lane = t & 31;
  return { s, lane, L: s * 32 + lane, Y: ((t >> 7) & 1) === (s >= 2 ? 1 : 0), sH: 2 * TWORDS + s * 512 };
}
// hash_tile's hasher and region (PEARL_BD_GROUPS 2, persistent).
function hasherOf(t) {
  const wr = (t >> 5) >> 2;
  const sid = t & 127;
  return { hasher: wr ? sid >= 64 : sid < 64, L: wr * 64 + (sid & 63) };
}

// tile_coords (PEARL_BD_WALK 1, column bands) and region_of, for a hit's record.
function tileCoords(v, geo) {
  const rowBlockGroups = geo.rowsValid / 4 / 2;
  const colBlockGroups = geo.tiles / rowBlockGroups;
  const bandTiles = geo.bandDepth * rowBlockGroups;
  const band = Math.floor(v / bandTiles);
  const inBand = v % bandTiles;
  const bandFirst = band * geo.bandDepth;
  const bandCols = colBlockGroups - bandFirst < geo.bandDepth ? colBlockGroups - bandFirst : geo.bandDepth;
  return [Math.floor(inBand / bandCols), bandFirst + (inBand % bandCols)];
}
function regionOf(L, rbg, cbg, geo) {
  const ow = Math.floor(L / 16);
  const ocb = Math.floor((L % 16) / 4);
  const oreg = L % 4;
  const orb = rbg * 2 + Math.floor(ow / 4);
  const ocgb = cbg * 4 + (ow % 4);
  return (ocgb * 4 + ocb) * geo.rowsValid + orb * 4 + oreg;
}

// ---------------------------------------------------------------------------
// One block of the persistent fold with the split, as 256 threads that each run at
// their own speed: the prologue, then for each tile its 16 chunks of readout stores
// (next_srcs before the last), the hand-off as the source has it, and after the loop
// the last tile. Shared is modelled word by word with a tag on each word, and every
// access is checked: a word is read only while it holds the tile the reader expects,
// and overwritten only once both warps that read it have; a state is loaded only
// once X stored that tile's, and overwritten only once Y has loaded it. A spin is a
// wait: the thread sleeps until its flag is written with the value it waits for. A
// barrier opens when all 256 threads are at it. If no thread can move and some are
// not done, that is a deadlock. At the end every (tile, region) must have been
// hashed once, where and with what the protocol says, and the hits must be the ones
// the shipped hash finds. Throws on anything else.
const STEP = 0;
const BAR = 1;
const WAIT = 2;
const GARBAGE = 0x5eed;  // whatever shared holds at launch

// A transcript word for (tile, region, chunk): Thomas Wang's integer hash, three
// times. Shifts and adds only: Math.imul and Math.log are slow inside jest's sandbox.
function wang(x) {
  x = (~x + (x << 15)) | 0;
  x ^= x >>> 12;
  x = (x + (x << 2)) | 0;
  x ^= x >>> 4;
  x = (x + (x << 3) + (x << 11)) | 0;
  return x ^ (x >>> 16);
}
const mix = (a, b, c, salt) => wang(wang(wang(a ^ salt) ^ b) ^ c) >>> 0;
// Gaps between a thread's steps: exponential, or heavy-tailed (Pareto, alpha 1.1),
// from tables made once.
const EXP_GAPS = Float64Array.from({ length: 4096 }, (_, i) => -Math.log(1 - (i + 0.5) / 4096));
const PARETO_GAPS = Float64Array.from({ length: 4096 }, (_, i) => (4096 / (4096 - i - 0.5)) ** (1 / 1.1));

function simulate(o) {
  const { seq, T, geo, N, key, hbe, target, model, mutate = '' } = o;
  const tiles = geo.tiles;
  const rand = xorshift(o.seed);
  const speed = model(rand);
  const fail = (m) => { throw new Error(m); };
  // The words region L of tile v holds, for chunk c.
  const salt = o.salt | 0;
  const wordVal = (v, L, c) => mix(v, L, c, salt);

  const tag = new Int32Array(2 * TWORDS).fill(-1);  // tile index * 2048 + L * 16 + c
  const val = new Uint32Array(2 * TWORDS);
  const readBy = new Uint8Array(2 * TWORDS);  // bit 1 X, bit 2 Y
  const slots = new Uint32Array(4 * 512).fill(GARBAGE);  // the 8 KB after the buffers
  const slotTile = new Int32Array(128).fill(-1);  // the tile region L's state is for
  const slotLoaded = new Uint8Array(128).fill(1);
  const sHread = new Int32Array(128).fill(GARBAGE);  // sHread[s][j] as word 32 s + j
  const sVprev = new Int32Array(128).fill(GARBAGE);
  const sNext = [GARBAGE, GARBAGE];
  const waiter = new Int32Array(128).fill(-1);
  const waitCond = new Array(256).fill(null);
  const rateOf = Float64Array.from({ length: 256 }, (_, t) => speed.rate(t, 0));
  const hashed = new Int32Array(T * 128);
  const hashedAt = new Int32Array(T * 128).fill(-2);  // hand-off number, or -1 after the loop
  const xParts = new Int32Array(T * 128);
  const hashedMsw = new Float64Array(T * 128);
  const atHandoff = new Int32Array(T + 1);
  const hits = [];
  let now = 0;

  const store = (tb, R, c, it) => {
    const a = tb + wordOf(R, c);
    if (tag[a] >= 0 && readBy[a] !== 3) {
      fail(`tile ${it} overwrote region ${R} chunk ${c} of tile ${tag[a] >> 11} before both warps read it`);
    }
    tag[a] = it * 2048 + R * 16 + c;
    val[a] = wordVal(seq[it], R, c);
    readBy[a] = 0;
  };
  const load = (b, L, c, want, Y) => {
    const a = b + wordOf(L, c);
    if (tag[a] !== want * 2048 + L * 16 + c) {
      fail(`${Y ? 'Y' : 'X'} read ${tag[a] < 0 ? 'nothing' : `tile ${tag[a] >> 11} region ${(tag[a] >> 4) & 127}`}`
        + ` for tile ${want} region ${L} chunk ${c}`);
    }
    readBy[a] |= Y ? 2 : 1;
    return val[a];
  };
  const setFlag = (L, value) => {
    sHread[L] = value;
    const x = waiter[L];
    if (x >= 0 && state[x] === 2 && waitCond[x]()) {
      state[x] = 0;
      push(now + delay(x), x);
    }
  };
  // hash_tile's test, or split_rest's after its rounds: the region's top word, and on a
  // hit the full hash and the region's number from the tile coordinates it was given.
  const hashRegion = (i, L, tm, msw, vCoords, when) => {
    if (vCoords !== seq[i]) {
      fail(`region ${L} of tile ${i} hashed with ${vCoords === GARBAGE ? 'no' : `tile ${vCoords}'s`} coordinates`);
    }
    hashed[i * 128 + L]++;
    hashedAt[i * 128 + L] = when;
    hashedMsw[i * 128 + L] = msw;
    if (when >= 0) atHandoff[when]++;
    if (msw > target[0]) return;
    const hh = compress(key, tm, 0, 64, FLAGS).slice(0, 8);  // pearl_transcript_hash_again
    if (!meets(hh, target, hbe)) return;
    const [rbg, cbg] = tileCoords(vCoords, geo);
    hits.push({ index: regionOf(L, rbg, cbg, geo), hash: hh, transcript: tm.slice() });
  };

  function* thread(t) {
    const warp = t >> 5;
    const lane = t & 31;
    if (t === 0) sNext[0] = seq[1];
    if (t < 128 && mutate !== 'no-init') sHread[(t >> 5) * 32 + (t & 31)] = 0;
    yield BAR;
    let it = 0;
    for (let v = seq[0]; v < tiles; it++) {
      rateOf[t] = speed.rate(t, it);
      const tbuf = (it & 1) * TWORDS;
      const vgrab = t === 0 ? seq[it + 2] : 0;
      let vnext = 0;
      // tile(): chunk c's words, chunks 0-15; next_srcs before the last chunk's.
      for (let c = 0; c < 16; c++) {
        if (c === 15) vnext = sNext[it & 1];
        for (let rl = 0; rl < 2; rl++) {
          const R = writerRegion(warp, lane, rl);
          if (R >= 0) store(tbuf, R, c, it);
        }
        yield STEP;
      }
      const hseq = it;
      const h = hsIds(t);
      let tm = null;
      const readTm = function* (buf) {
        tm = [];
        for (let c = 0; c < 16; c++) {
          tm.push(load(buf, h.L, c, it - 1, h.Y));
          if (c % 4 === 3) yield STEP;
        }
      };
      if (hseq !== 0 || mutate === 'hash-at-0') {
        if (!(h.Y && mutate === 'y-reads-late')) yield* readTm(mutate === 'same-buffer' ? tbuf : tbuf ^ TWORDS);
        if (!h.Y) {
          // split_first: do { done = sHread[s][lane] } while (done + 1 != hseq).
          const flagSet = () => (mutate === 'wait-own' ? sHread[h.L] === hseq : (sHread[h.L] + 1) >>> 0 === hseq);
          if (mutate !== 'no-wait') {
            while (!flagSet()) {
              waitCond[t] = flagSet;
              waiter[h.L] = t;
              yield WAIT;
            }
          }
          const s = splitFirst(key, tm, N);
          yield STEP;
          if (!slotLoaded[h.L]) fail(`X overwrote region ${h.L}'s state for tile ${slotTile[h.L]} before Y loaded it`);
          xStore(slots.subarray(h.sH - 2 * TWORDS, h.sH - 2 * TWORDS + 512), h.lane, s);
          slotTile[h.L] = it - 1;
          slotLoaded[h.L] = 0;
          xParts[(it - 1) * 128 + h.L]++;
          yield STEP;
        }
      }
      if (t === 0) sNext[(it + 1) & 1] = vgrab;
      yield BAR;
      if (h.Y) {
        if (mutate === 'vprev-early') sVprev[h.L] = v;
        if (hseq !== 0) {
          if (mutate === 'y-reads-late') yield* readTm(tbuf ^ TWORDS);
          // split_rest: the state into registers, then the flag, then the rounds.
          if (mutate === 'flag-early') {
            setFlag(h.L, hseq);
            yield STEP;
          }
          if (slotLoaded[h.L] || slotTile[h.L] !== it - 1) {
            fail(`Y at hand-off ${hseq} found ${slotLoaded[h.L] ? 'no new state' : `tile ${slotTile[h.L]}'s state`}`
              + ` for region ${h.L}`);
          }
          const s = yLoad(slots.subarray(h.sH - 2 * TWORDS, h.sH - 2 * TWORDS + 512), h.lane);
          slotLoaded[h.L] = 1;
          yield STEP;
          if (mutate !== 'flag-early') {
            setFlag(h.L, hseq);
            yield STEP;
          }
          halfRounds(s, tm, N, 14);
          hashRegion(it - 1, h.L, tm, topWord(s, hbe), sVprev[h.L], hseq);
        }
        sVprev[h.L] = v;
        yield STEP;
      }
      v = vnext;
    }
    if (it !== T) fail(`thread ${t} ran ${it} tiles, not ${T}`);
    // After the loop: Y hashes the last tile whole from its own buffer (hash_tile).
    const h = hsIds(t);
    if (h.Y && mutate !== 'no-epilogue') {
      const buf = ((mutate === 'epilogue-buf' ? it : it - 1) & 1) * TWORDS;
      const tm = [];
      for (let c = 0; c < 16; c++) tm.push(load(buf, h.L, c, it - 1, true));
      yield STEP;
      hashRegion(it - 1, h.L, tm, msf(compress(key, tm, 0, 64, FLAGS), 0, hbe), sVprev[h.L], -1);
    }
  }

  // The scheduler: each runnable thread's next step comes at a random time at its own
  // rate (a binary heap of times). Threads at a barrier or a wait are off the heap.
  const state = new Uint8Array(256);  // 0 runnable, 1 at the barrier, 2 waiting, 3 done
  const hTime = new Float64Array(256);
  const hId = new Int32Array(256);
  let hLen = 0;
  const swap = (i, j) => {
    const tt = hTime[i]; hTime[i] = hTime[j]; hTime[j] = tt;
    const ti = hId[i]; hId[i] = hId[j]; hId[j] = ti;
  };
  function push(time, t) {
    let i = hLen++;
    hTime[i] = time;
    hId[i] = t;
    while (i > 0 && hTime[(i - 1) >> 1] > hTime[i]) {
      swap(i, (i - 1) >> 1);
      i = (i - 1) >> 1;
    }
  }
  const pop = () => {
    const top = hId[0];
    hLen--;
    hTime[0] = hTime[hLen];
    hId[0] = hId[hLen];
    for (let i = 0; ;) {
      const l = 2 * i + 1;
      let m = i;
      if (l < hLen && hTime[l] < hTime[m]) m = l;
      if (l + 1 < hLen && hTime[l + 1] < hTime[m]) m = l + 1;
      if (m === i) break;
      swap(i, m);
      i = m;
    }
    return top;
  };
  const gaps = speed.pareto ? PARETO_GAPS : EXP_GAPS;
  function delay(t) {
    return gaps[rand() & 4095] / rateOf[t];
  }
  const gens = Array.from({ length: 256 }, (_, t) => thread(t));
  for (let t = 0; t < 256; t++) push(delay(t), t);
  let atBar = 0;
  let done = 0;
  let steps = 0;
  while (done < 256) {
    if (hLen === 0) {
      const waiting = state.filter((x) => x === 2).length;
      fail(`deadlock: ${atBar} threads at the barrier and ${waiting} waiting on a flag no thread will set`);
    }
    now = hTime[0];
    const t = pop();
    const r = gens[t].next();
    steps++;
    if (r.done) {
      state[t] = 3;
      done++;
      if (atBar) fail(`thread ${t} left while ${atBar} threads wait at a barrier`);
    } else if (r.value === STEP) {
      push(now + delay(t), t);
    } else if (r.value === BAR) {
      state[t] = 1;
      if (++atBar === 256) {
        atBar = 0;
        for (let y = 0; y < 256; y++) {
          state[y] = 0;
          push(now + delay(y), y);
        }
      }
    } else if (waitCond[t]()) {
      push(now + delay(t), t);
    } else {
      state[t] = 2;
    }
  }

  // Every (tile, region) once: tile i < T - 1 at hand-off i + 1, X's part included;
  // the last tile after the loop, whole. Nothing at hand-off 0.
  for (let i = 0; i < T; i++) {
    for (let L = 0; L < 128; L++) {
      const k = i * 128 + L;
      if (hashed[k] !== 1) fail(`tile ${i} region ${L} hashed ${hashed[k]} times`);
      const last = i === T - 1;
      if (hashedAt[k] !== (last ? -1 : i + 1) || xParts[k] !== (last ? 0 : 1)) {
        fail(`tile ${i} region ${L} hashed at ${hashedAt[k]} with ${xParts[k]} first halves`);
      }
    }
  }
  if (atHandoff[0] !== 0) fail(`${atHandoff[0]} regions hashed at the first hand-off`);
  // The top words and hits the shipped hash gives: each tile's regions hashed whole,
  // with that tile's words and coordinates.
  const want = [];
  for (let i = 0; i < T; i++) {
    const [rbg, cbg] = tileCoords(seq[i], geo);
    for (let L = 0; L < 128; L++) {
      const tm = Array.from({ length: 16 }, (_, c) => wordVal(seq[i], L, c));
      const hh = compress(key, tm, 0, 64, FLAGS).slice(0, 8);
      if (hashedMsw[i * 128 + L] !== msf(hh, 0, hbe)) fail(`tile ${i} region ${L}: top word differs`);
      if (msf(hh, 0, hbe) <= target[0] && meets(hh, target, hbe)) {
        want.push({ index: regionOf(L, rbg, cbg, geo), hash: hh, transcript: tm });
      }
    }
  }
  // Hits in region order, as the host reports them (pearl_host_collect).
  const flat = (list) => list.map((x) => [x.index, ...x.hash, ...x.transcript].join(' '))
    .sort((a, b) => Number(a.split(' ')[0]) - Number(b.split(' ')[0]));
  const got = flat(hits);
  const exp = flat(want);
  if (got.length !== exp.length || got.some((x, i) => x !== exp[i])) {
    fail(`hits differ from the shipped hash's: ${got.length} against ${exp.length},`
      + ` first ${got.find((x, i) => x !== exp[i])}`);
  }
  return { hits: hits.length, steps };
}

// Warp speeds: a rate per thread (higher is faster), maybe changing by tile, and
// exponential gaps between steps, or heavy-tailed ones with long stalls.
const logUniform = (rand, decades) => 10 ** (-decades * unit(rand));
const MODELS = {
  even: () => ({ rate: () => 1 }),
  threads: (rand) => {
    const r = Float64Array.from({ length: 256 }, () => logUniform(rand, 3));
    return { rate: (t) => r[t] };
  },
  warps: (rand) => {
    const r = Float64Array.from({ length: 8 }, () => logUniform(rand, 2));
    return { rate: (t) => r[t >> 5] };
  },
  hashersBehind: () => ({ rate: (t) => (hsIds(t).Y ? 0.02 : 1) }),
  partnersBehind: () => ({ rate: (t) => (hsIds(t).Y ? 1 : 0.02) }),
  oneLane: () => ({ rate: (t) => ((t & 31) === 7 ? 0.01 : 1) }),
  changing: (rand) => {
    const byTile = new Map();
    return {
      rate: (t, it) => {
        if (!byTile.has(it)) byTile.set(it, Float64Array.from({ length: 8 }, () => logUniform(rand, 2)));
        return byTile.get(it)[t >> 5];
      },
    };
  },
  bursty: (rand) => {
    const r = Float64Array.from({ length: 256 }, () => logUniform(rand, 1));
    return { rate: (t) => r[t], pareto: true };
  },
};
const MODEL_NAMES = Object.keys(MODELS);

// 64 valid rows (8 row groups) by 300 column groups, bands 7 column groups deep, so
// the last band is short.
const GEO = { rowsValid: 64, tiles: 8 * 300, bandDepth: 7 };

// A block's tiles: blockIdx.x, then later ones from the counter, then the one past
// the end that stops the loop and the one thread 0 grabbed with it.
function run(T, seed, modelName, extra = {}) {
  const rand = xorshift(seed);
  const seq = [rand() % 50];
  while (seq.length < T) seq.push(seq[seq.length - 1] + 1 + (rand() % 10));
  seq.push(GEO.tiles + (rand() % 20));
  seq.push(seq[T] + 1 + (rand() % 20));
  const key = Array.from({ length: 8 }, () => rand());
  // About one region in 16 gets past the top word.
  const target = [0x0fffffff, ...Array.from({ length: 7 }, () => rand())];
  return simulate({
    seq, T, geo: GEO, N: 1 + (seed % 13), key, hbe: seed & 1, target, seed: seed * 7 + 1,
    salt: rand(), model: MODELS[modelName], ...extra,
  });
}

// ---------------------------------------------------------------------------
// A small C preprocessor for the guards, as turingSnake.test.js has it: object-like
// #define, #undef, #if, #ifdef, #ifndef, #elif, #else, #endif and #error, with -D
// flags. Returns the #error lines it reaches and the definitions.
function tokens(expr) {
  const out = [];
  const word = (c) => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c === '_';
  let i = 0;
  while (i < expr.length) {
    const c = expr[i];
    if (c === ' ' || c === String.fromCharCode(9)) {
      i++;
    } else if (word(c)) {
      let j = i;
      while (j < expr.length && word(expr[j])) j++;
      out.push(expr.slice(i, j));
      i = j;
    } else if (['&&', '||', '==', '!=', '<=', '>='].includes(expr.slice(i, i + 2))) {
      out.push(expr.slice(i, i + 2));
      i += 2;
    } else if ('()!<>+-*/%?:'.includes(c)) {
      out.push(c);
      i++;
    } else {
      throw new Error(`no token at ${expr.slice(i)}`);
    }
  }
  return out;
}

function evaluate(expr, defs, seen = new Set()) {
  const t = tokens(expr);
  let p = 0;
  const take = (s) => (t[p] === s ? (p++, true) : false);
  const unary = () => {
    const tk = t[p++];
    if (tk === undefined) throw new Error(`short expression: ${expr}`);
    if (tk === '(') {
      const v = ternary();
      if (!take(')')) throw new Error(`no ) in ${expr}`);
      return v;
    }
    if (tk === '!') return unary() === 0 ? 1 : 0;
    if (tk === '-') return -unary();
    if (tk === 'defined') {
      const paren = take('(');
      const name = t[p++];
      if (paren && !take(')')) throw new Error(`no ) in ${expr}`);
      return defs.has(name) ? 1 : 0;
    }
    if (tk[0] >= '0' && tk[0] <= '9') {
      let s = tk;
      while ('uUlL'.includes(s[s.length - 1])) s = s.slice(0, -1);
      const v = Number(s);
      if (!Number.isFinite(v)) throw new Error(`not a number: ${tk}`);
      return v;
    }
    if (')!<>+*/%?:&|='.includes(tk[0])) throw new Error(`unexpected ${tk} in ${expr}`);
    if (!defs.has(tk) || seen.has(tk)) return 0;
    if (defs.get(tk) === null) throw new Error(`function-like ${tk} in ${expr}`);
    return evaluate(defs.get(tk), defs, new Set([...seen, tk]));
  };
  const binary = (next, ops, f) => () => {
    let v = next();
    for (;;) {
      const op = ops.find((x) => t[p] === x);
      if (!op) return v;
      p++;
      v = f(op, v, next());
    }
  };
  const num = (b) => (b ? 1 : 0);
  const mul = binary(unary, ['*', '/', '%'],
    (op, a, b) => (op === '*' ? a * b : op === '/' ? Math.trunc(a / b) : a % b));
  const add = binary(mul, ['+', '-'], (op, a, b) => (op === '+' ? a + b : a - b));
  const rel = binary(add, ['<', '>', '<=', '>='],
    (op, a, b) => num(op === '<' ? a < b : op === '>' ? a > b : op === '<=' ? a <= b : a >= b));
  const eq = binary(rel, ['==', '!='], (op, a, b) => num(op === '==' ? a === b : a !== b));
  const and = binary(eq, ['&&'], (op, a, b) => num(a !== 0 && b !== 0));
  const or = binary(and, ['||'], (op, a, b) => num(a !== 0 || b !== 0));
  function ternary() {
    const c = or();
    if (!take('?')) return c;
    const x = ternary();
    if (!take(':')) throw new Error(`no : in ${expr}`);
    const y = ternary();
    return c !== 0 ? x : y;
  }
  const v = ternary();
  if (p !== t.length) throw new Error(`left over in ${expr}`);
  return v;
}

function preprocess(text, flags) {
  const defs = new Map(Object.entries(flags).map(([k, v]) => [k, String(v)]));
  const errors = [];
  const stack = [];
  for (const raw of text.split(BS + NL).join(' ').split(NL)) {
    const line = raw.trim();
    if (!line.startsWith('#')) continue;
    const sp = line.indexOf(' ');
    const dir = sp < 0 ? line.slice(1) : line.slice(1, sp);
    let rest = sp < 0 ? '' : line.slice(sp + 1).trim();
    if (dir !== 'error' && rest.includes('//')) rest = rest.slice(0, rest.indexOf('//')).trim();
    const live = stack.every((f) => f.on);
    if (dir === 'if' || dir === 'ifdef' || dir === 'ifndef') {
      const on = live && (dir === 'if' ? evaluate(rest, defs) !== 0 : defs.has(rest) === (dir === 'ifdef'));
      stack.push({ on, done: on, outer: live });
    } else if (dir === 'elif') {
      const f = stack[stack.length - 1];
      f.on = f.outer && !f.done && evaluate(rest, defs) !== 0;
      f.done = f.done || f.on;
    } else if (dir === 'else') {
      const f = stack[stack.length - 1];
      f.on = f.outer && !f.done;
      f.done = true;
    } else if (dir === 'endif') {
      expect(stack.length).toBeGreaterThan(0);
      stack.pop();
    } else if (!live) {
      continue;
    } else if (dir === 'define') {
      let end = 0;
      while (end < rest.length && rest[end] !== ' ' && rest[end] !== '(') end++;
      defs.set(rest.slice(0, end), rest[end] === '(' ? null : rest.slice(end).trim());
    } else if (dir === 'undef') {
      defs.delete(rest);
    } else if (dir === 'error') {
      errors.push(rest);
    }
  }
  expect(stack).toEqual([]);
  return { errors, defs };
}

// What a build of the fold goes through: the header's Turing block, from its first
// switch to PEARL_BD_SPLIT_SMEM, and the fold's own #error lines.
function guards() {
  const slice = (src, from, to) => {
    const at = src.indexOf(from);
    const end = src.indexOf(to, at);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(at);
    return src.slice(at, end);
  };
  return slice(HEADER, '#ifndef PEARL_TURING_WIDE' + NL, NL + '// A block size forced from the command line') + NL
    + slice(FOLD, '#if PEARL_BD_A_DIRECT && (defined(PEARL_ABLATE_BARRIER)', '#if PEARL_BD_GROUPS == 2');
}

describe('Turing split hash (PEARL_BD_HASH_SPLIT)', () => {
  test('is off by default, builds only with the load parts, and is refused where it does not fit', () => {
    const at = HEADER.indexOf('#ifndef PEARL_BD_HASH_SPLIT' + NL);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(HEADER.slice(at).split(NL)[1]).toBe('#define PEARL_BD_HASH_SPLIT 0');
    const GUARDS = guards();
    expect(GUARDS).toContain('#define PEARL_BD_SPLIT_SMEM (PEARL_BD_HASH_SPLIT ? 4u * 32u * 16u * 4u : 0u)');
    const errorsFor = (flags) => preprocess(GUARDS, flags).errors;
    const value = (name, flags) => evaluate(name, preprocess(GUARDS, flags).defs);
    expect(value('PEARL_BD_HASH_SPLIT', {})).toBe(0);
    expect(value('PEARL_BD_SPLIT_SMEM', {})).toBe(0);
    // -D flags: PEARL_BD_ is left off the switch names below.
    const D = (o) => Object.fromEntries(Object.entries(o)
      .map(([k, v]) => [k.startsWith('PEARL_') ? k : 'PEARL_BD_' + k, v]));
    const LOADS = { RUNBASE: 1, TAIL_LDG: 1 };
    // The split turns LEAN off, its alternative, and keeps HIT_COORDS on, as the default
    // fold has it. With no flags LEAN is on.
    expect([value('PEARL_BD_LEAN', {}), value('PEARL_BD_HIT_COORDS', {})]).toEqual([1, 1]);
    expect(['LEAN', 'HIT_COORDS', 'GATE'].map((s) => value('PEARL_BD_' + s, D({ HASH_SPLIT: 4, ...LOADS }))))
      .toEqual([0, 1, 2]);
    // Every N from 1 to 13 with RUNBASE and TAIL_LDG, and without HIT_COORDS too.
    const allowed = [{}];
    for (let n = 1; n <= 13; n++) allowed.push({ HASH_SPLIT: n, ...LOADS });
    allowed.push({ HASH_SPLIT: 4, ...LOADS, HIT_COORDS: 0 });
    for (const f of allowed) {
      expect([f, errorsFor(D(f))]).toEqual([f, []]);
      if (f.HASH_SPLIT) expect(value('PEARL_BD_SPLIT_SMEM', D(f))).toBe(8192);
    }
    // Refused by the split's own guards, each fine without the split.
    const own = [
      [{ HASH_SPLIT: 14, ...LOADS }, LOADS], [{ HASH_SPLIT: -1, ...LOADS }, LOADS], [{ HASH_SPLIT: 4 }, {}],
      [{ HASH_SPLIT: 4, RUNBASE: 1 }, { RUNBASE: 1 }], [{ HASH_SPLIT: 4, TAIL_LDG: 1 }, { TAIL_LDG: 1 }],
      [{ HASH_SPLIT: 4, RED: 1 }, { RED: 1 }], [{ HASH_SPLIT: 4, ...LOADS, TADDR: 1 }, { ...LOADS, TADDR: 1 }],
      [{ HASH_SPLIT: 4, ...LOADS, SNAKE: 1 }, { ...LOADS, SNAKE: 1 }],
      [{ HASH_SPLIT: 4, ...LOADS, GATE: 1 }, { ...LOADS, GATE: 1 }],
      [{ HASH_SPLIT: 4, LEAN: 1 }, { LEAN: 1 }], [{ HASH_SPLIT: 4, ...LOADS, GROUPS: 1 }, null],
      [{ HASH_SPLIT: 4, ...LOADS, PEARL_ABLATE_TRANSCRIPT_HASH: 1 }, { ...LOADS, PEARL_ABLATE_TRANSCRIPT_HASH: 1 }],
    ];
    for (const [f, without] of own) {
      const errors = errorsFor(D(f));
      expect([f, errors.length > 0 && errors.some((e) => e.includes('PEARL_BD_HASH_SPLIT'))]).toEqual([f, true]);
      if (without) expect([without, errorsFor(D(without))]).toEqual([without, []]);
    }
    // The fold it is written for, through RUNBASE's guard: persistent, two buffers,
    // A-direct, k-lo first.
    for (const f of [{ PERSIST: 0 }, { A_REALIGN: 1 }, { KORDER: 0 }, { A_DIRECT: 0 }]) {
      expect([f, errorsFor(D({ HASH_SPLIT: 4, ...LOADS, ...f })).length > 0]).toEqual([f, true]);
    }
    // And in the source: the fold's own guard and static_assert, and the default
    // hand-off under the #else.
    expect(FOLD).toContain('#if PEARL_BD_HASH_SPLIT && defined(PEARL_ABLATE_TRANSCRIPT_HASH)' + NL + '#error ');
    expect(FOLD_CODE).toContain(lines(
      '  static_assert(PEARL_BD_PERSIST && PEARL_BD_T_BUFS == 2 && PEARL_BD_GROUPS == 2',
      '                    && PEARL_BD_HASH_SPLIT > 0 && PEARL_BD_HASH_SPLIT < 14',
      '                    && 4u * 512u * 4u == PEARL_BD_SPLIT_SMEM,'));
    expect(FOLD_CODE).toContain(lines(
      '#else',
      '    if (threadIdx.x == 0u) sNext[(it + 1u) & 1u] = vgrab;',
      '    __syncthreads();',
      '#if PEARL_BD_HIT_COORDS',
      '    hash_tile(v);',
      '#else',
      '    {',
      '      uint32_t rbg, cbg;',
      '      tile_coords(v, rbg, cbg);',
      '      hash_tile(rbg, cbg);',
      '    }',
      '#endif',
      '#endif'));
  });

  test('the copy matches the source, line for line', () => {
    // pearl_bd_half_rounds: the bounds, its eight G calls and the permutation, and
    // nothing else that mixes.
    const hr = body(FOLD, 'template <int H0, int H1>' + NL
      + '__device__ __forceinline__ void pearl_bd_half_rounds(uint32_t s[16], const uint32_t m0[16]) {');
    for (const l of ['for (int i = 0; i < 16; i++) m[i] = m0[i];', 'for (int r = 0; r < 7; r++) {',
      'if (2 * r >= H0 && 2 * r < H1) {', 'if (2 * r + 1 >= H0 && 2 * r + 1 < H1) {', 'if (r < 6) {',
      'g(s, 0, 4, 8, 12, m[0], m[1]);', 'g(s, 1, 5, 9, 13, m[2], m[3]);', 'g(s, 2, 6, 10, 14, m[4], m[5]);',
      'g(s, 3, 7, 11, 15, m[6], m[7]);', 'g(s, 0, 5, 10, 15, m[8], m[9]);', 'g(s, 1, 6, 11, 12, m[10], m[11]);',
      'g(s, 2, 7, 8, 13, m[12], m[13]);', 'g(s, 3, 4, 9, 14, m[14], m[15]);',
      'const uint32_t p0 = m[2], p1 = m[6], p2 = m[3], p3 = m[10];',
      'const uint32_t p4 = m[7], p5 = m[0], p6 = m[4], p7 = m[13];',
      'const uint32_t p8 = m[1], p9 = m[11], p10 = m[12], p11 = m[5];',
      'const uint32_t p12 = m[9], p13 = m[14], p14 = m[15], p15 = m[8];',
      'm[0] = p0; m[1] = p1; m[2] = p2; m[3] = p3;', 'm[4] = p4; m[5] = p5; m[6] = p6; m[7] = p7;',
      'm[8] = p8; m[9] = p9; m[10] = p10; m[11] = p11;', 'm[12] = p12; m[13] = p13; m[14] = p14; m[15] = p15;']) {
      expect([l, hr.includes(l)]).toEqual([l, true]);
    }
    expect(hr.split('g(s, ').length - 1).toBe(8);
    inOrder(hr, 'g(s, 3, 7, 11, 15, m[6], m[7]);', 'if (2 * r + 1 >= H0', 'g(s, 0, 5, 10, 15, m[8], m[9]);',
      'if (r < 6) {');
    // pearl_kernel.cu's g, which the template calls.
    expect(KERNEL).toContain(lines(
      '  s[a] = s[a] + s[b] + mx;', '  s[d] = rotr32(s[d] ^ s[a], 16);', '  s[c] = s[c] + s[d];',
      '  s[b] = rotr32(s[b] ^ s[c], 12);', '  s[a] = s[a] + s[b] + my;', '  s[d] = rotr32(s[d] ^ s[a], 8);',
      '  s[c] = s[c] + s[d];', '  s[b] = rotr32(s[b] ^ s[c], 7);'));
    expect(KERNEL).toContain('  return (x >> n) | (x << (32 - n));');
    expect(KERNEL).toContain('#define PEARL_TRANSCRIPT_FLAGS (CHUNK_START | CHUNK_END | ROOT | KEYED_HASH)');
    for (const [name, bit] of [['CHUNK_START', 0], ['CHUNK_END', 1], ['ROOT', 3], ['KEYED_HASH', 4]]) {
      expect(KERNEL).toContain(`#define ${name} (1u << ${bit})`);
    }
    expect(KERNEL).toContain('__constant__ uint32_t BLAKE3_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u,'
      + NL + '                                      0xA54FF53Au,');
    expect(KERNEL).toContain('return hash_big_endian ? pearl_bswap32(h[i]) : h[7 - i];');
    expect(KERNEL).toContain('return __byte_perm(x, 0u, 0x0123u);');
    expect(KERNEL).toContain('if (hw != target_w[i]) return hw < target_w[i];');
    // The default's test word: output word 7, or word 0 byte-reversed.
    expect(KERNEL).toContain('return hash_big_endian ? pearl_bswap32(out16[0]) : out16[7];');
    expect(KERNEL).toContain('out[i] = s[i] ^ s[i + 8];');

    // X's half, and the state into the pair's slot.
    const first = code(body(FOLD, 'auto split_first = [&](const PearlBdHs &h, const uint32_t (&tm)[16], uint32_t hseq) {'));
    expect(first).toContain(lines(
      '    const uint32_t fa =',
      '        (uint32_t)__cvta_generic_to_shared(&sHread[0][0]) + 4u * (threadIdx.x & 127u);',
      '    uint32_t done;',
      '    do {',
      '      asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(done) : "r"(fa) : "memory");',
      '    } while (done + 1u != hseq);',
      '    uint32_t s[16];',
      '#pragma unroll',
      '    for (int i = 0; i < 8; i++) s[i] = test.key[i];',
      '    s[8] = BLAKE3_IV[0]; s[9] = BLAKE3_IV[1]; s[10] = BLAKE3_IV[2]; s[11] = BLAKE3_IV[3];',
      '    s[12] = 0u; s[13] = 0u; s[14] = 64u; s[15] = PEARL_TRANSCRIPT_FLAGS;',
      '    pearl_bd_half_rounds<0, PEARL_BD_HASH_SPLIT>(s, tm);',
      '#pragma unroll',
      '    for (uint32_t q = 0; q < 4u; q++)',
      '      reinterpret_cast<uint4 *>(h.sH)[q * 32u + h.lane] =',
      '          make_uint4(s[4 * q], s[4 * q + 1], s[4 * q + 2], s[4 * q + 3]);'));
    // Y's half: the state into registers, the flag (a release), then the rounds, the
    // test word and the hit, with the region's number from the tile it kept.
    const rest = code(body(FOLD, 'auto split_rest = [&](const PearlBdHs &h, const uint32_t (&tm)[16], uint32_t hseq) {'));
    expect(rest).toContain(lines(
      '    uint32_t s[16];',
      '#pragma unroll',
      '    for (uint32_t q = 0; q < 4u; q++) {',
      '      const uint4 u = reinterpret_cast<const uint4 *>(h.sH)[q * 32u + h.lane];',
      '      s[4 * q] = u.x; s[4 * q + 1] = u.y; s[4 * q + 2] = u.z; s[4 * q + 3] = u.w;',
      '    }',
      '    asm volatile("fence.acq_rel.cta;" ::: "memory");',
      '    *reinterpret_cast<volatile uint32_t *>(&sHread[h.s][h.lane]) = hseq;',
      '    pearl_bd_half_rounds<PEARL_BD_HASH_SPLIT, 14>(s, tm);',
      '    const uint32_t out[8] = {s[0] ^ s[8], 0u, 0u, 0u, 0u, 0u, 0u, s[7] ^ s[15]};',
      '    const uint32_t msw = pearl_hash_word_msf(out, 0, test.hash_big_endian);',
      '    if (msw > test.target_w[0]) return;',
      '    uint32_t hh[8];',
      '    pearl_transcript_hash_again(test.key, tm, hh);',
      '    if (!pearl_hash_meets_words(hh, test.target_w, test.hash_big_endian)) return;',
      '    const uint32_t slot = atomicAdd(hits.count, 1u);',
      '    if (slot >= PEARL_MAX_HITS) return;',
      '    uint32_t rbg, cbg;',
      '    tile_coords(sVprev[h.s][h.lane], rbg, cbg);',
      '    const uint32_t L = h.L;',
      '    const uint32_t ow = L / warp_regions;',
      '    const uint32_t ocb = (L % warp_regions) / regions_per_warp;',
      '    const uint32_t oreg = L % regions_per_warp;',
      '    const uint32_t orb = rbg * 2u + ow / 4u;',
      '    const uint32_t ocgb = cbg * 4u + ow % 4u;',
      '    hits.index[slot] = (ocgb * 4u + ocb) * rows_valid + orb * regions_per_warp + oreg;',
      '#pragma unroll',
      '    for (int i = 0; i < 8; i++) hits.hash[slot * 8u + i] = hh[i];',
      '#pragma unroll',
      '    for (int i = 0; i < 16; i++) hits.transcript[slot * 16u + i] = tm[i];'));
    // Each step of the protocol once, in this order: X waits, runs its rounds, stores;
    // Y loads, releases the slot, runs its rounds. And these are the only places the
    // fold touches the flags, the kept tiles and the slots.
    inOrder(first, 'ld.acquire.cta.shared.u32', 'while (done + 1u != hseq)', 'pearl_bd_half_rounds<',
      'reinterpret_cast<uint4 *>(h.sH)');
    inOrder(rest, 'reinterpret_cast<const uint4 *>(h.sH)', 'fence.acq_rel.cta', 'sHread', '= hseq;',
      'pearl_bd_half_rounds<', 'sVprev');
    const count = (s, x) => s.split(x).length - 1;
    expect([count(FOLD_CODE, 'sHread'), count(FOLD_CODE, 'sVprev'), count(FOLD_CODE, 'h.sH')]).toEqual([4, 5, 3]);
    expect(FOLD_CODE).toContain('    h.sH = smem_u32 + PEARL_BD_T_BUFS * TWORDS + h.s * 512u;');
    expect(count(FOLD_CODE, 'sVprev[h.s][h.lane] = v;')).toBe(1);
    expect(KERNEL).toContain('  pearl_transcript_hash(k2, m2, h);');
    expect(KERNEL).toContain('  blake3_compress(key, m, 0, 64, PEARL_TRANSCRIPT_FLAGS, out16);');

    // The ids, the reads, hash_tile's hasher and region (the default's: the split refuses
    // PEARL_BD_HASH_RG), and the readout's store.
    expect(code(body(FOLD, 'auto hs_ids = [&]() -> PearlBdHs {'))).toContain(lines(
      '    uint32_t t = threadIdx.x;', '    asm volatile("" : "+r"(t));', '    PearlBdHs h;',
      '    h.s = (t >> 5) & 3u;', '    h.lane = t & 31u;', '    h.L = h.s * 32u + h.lane;',
      '    h.Y = ((t >> 7) & 1u) == (h.s >= 2u ? 1u : 0u);',
      '    h.sH = smem_u32 + PEARL_BD_T_BUFS * TWORDS + h.s * 512u;', '    return h;'));
    expect(body(FOLD, 'auto load_tm = [&](const PearlBdHs &h, uint32_t buf, uint32_t (&tm)[16]) {'))
      .toContain('for (uint32_t c = 0; c < 16u; c++) tm[c] = sT[buf + h.L * 16u + (c ^ ((h.L >> 1) & 15u))];');
    expect(FOLD_CODE).toContain('  __shared__ uint32_t sHread[4][32], sVprev[4][32];');
    expect(FOLD_CODE).toContain(lines(
      '#elif PEARL_BD_GROUPS == 2 && PEARL_BD_PERSIST',
      '    const bool hasher = wr ? sid >= 64u : sid < 64u;',
      '    const uint32_t L = wr * 64u + (sid & 63u);'));
    expect(FOLD_CODE).toContain('  const uint32_t sid = threadIdx.x & 127u;');
    expect(FOLD_CODE).toContain('  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;');
    expect(FOLD_CODE).toContain('  const uint32_t wr = warp >> 2, wc = warp & 3u;');
    expect(FOLD_CODE).toContain('  constexpr uint32_t regions_per_warp = 4u, warp_regions = 16u;');
    expect(FOLD_CODE).toContain(
      '      for (uint32_t c = 0; c < 16u; c++) tm[c] = sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))];');
    expect(FOLD_CODE).toContain(
      '        if (pearl_transcript_msw(test.key, tm, test.hash_big_endian) > test.target_w[0]) return;');
    expect(FOLD_CODE).toContain(lines(
      '        const uint32_t ow = L / warp_regions;',
      '        const uint32_t ocb = (L % warp_regions) / regions_per_warp;',
      '        const uint32_t oreg = L % regions_per_warp;',
      '        const uint32_t orb = rbg_ * 2u + ow / 4u;',
      '        const uint32_t ocgb = cbg_ * 4u + ow % 4u;',
      '        return (ocgb * 4u + ocb) * rows_valid + orb * regions_per_warp + oreg;'));
    expect(FOLD_CODE).toContain(lines(
      '#elif PEARL_BD_HASH_SPLIT',
      '  auto readout_st = [&](uint32_t rl, uint32_t c, uint32_t x) {',
      '    const uint32_t L = warp * warp_regions + (lane & 3u) * regions_per_warp + 2u * rl',
      '                       + ((lane >> 4) & 1u);',
      '    const uint32_t a =',
      '        (uint32_t)__cvta_generic_to_shared(sT + tbuf + L * 16u + (c ^ ((L >> 1) & 15u)));',
      '    asm volatile("{ .reg .pred p; setp.eq.u32 p, %2, 0; @p st.shared.u32 [%0], %1; }"',
      '                 ::"r"(a), "r"(x), "r"((lane >> 2) & 3u)',
      '                 : "memory");',
      '  };',
      '#else',
      '  auto readout_st = [&](uint32_t rl, uint32_t c, uint32_t x) {',
      '    if (((lane >> 2) & 3u) == 0u) {',
      '      const uint32_t L = warp * warp_regions + (lane & 3u) * regions_per_warp + 2u * rl',
      '                         + ((lane >> 4) & 1u);',
      '      sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))] = x;'));
    // The tile writes chunks 0-14 in its loop, then the last; next_srcs reads the next
    // tile in between.
    expect(FOLD_CODE).toContain('    constexpr uint32_t first_chunk = 0u;');
    expect(FOLD_CODE).toContain('    for (uint32_t chunk = first_chunk; chunk + 1u < chunks; chunk++) {');
    expect(FOLD_CODE).toContain('      for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, chunk, rx[rl]);');
    const tile = body(FOLD_CODE, '  auto tile = [&](auto next_srcs) {');
    inOrder(tile, '    next_srcs();', '    for (uint32_t rl = 0; rl < RPL; rl++) readout(rl, chunks - 1u);');
    // Within a tile the warps meet nowhere: the one group barrier is in A_REALIGN's
    // peeled chunk, which the split does not build with, and the k-step has none.
    expect(tile.split('PEARL_BD_BAR()').length - 1).toBe(1);
    inOrder(tile, '#if PEARL_BD_A_REALIGN', 'PEARL_BD_BAR();', '    constexpr uint32_t first_chunk = 1u;');
    expect(tile).not.toContain('__syncthreads');
    const kstep = body(FOLD_CODE, '  auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last, uint32_t dch = 0u) {');
    expect([kstep.includes('PEARL_BD_BAR'), kstep.includes('__syncthreads'), kstep.includes('bar_sync')])
      .toEqual([false, false, false]);
    // tile_coords, column bands (the default walk).
    expect(HEADER).toContain('#ifndef PEARL_BD_WALK' + NL + '#define PEARL_BD_WALK 1');
    expect(FOLD_CODE).toContain(lines(
      '  const uint32_t row_blocks = rows_valid / regions_per_warp;',
      '  const uint32_t row_block_groups = row_blocks / 2u;',
      '  const uint32_t col_block_groups = tiles / row_block_groups;'));
    expect(FOLD_CODE).toContain(lines(
      '#if PEARL_BD_WALK == 1',
      '  const uint32_t band_tiles = band_depth * row_block_groups;',
      '  auto tile_coords = [&](uint32_t v, uint32_t &rbg_, uint32_t &cbg_) {',
      '    const uint32_t band = v / band_tiles;',
      '    const uint32_t in_band = v % band_tiles;',
      '    const uint32_t band_first = band * band_depth;',
      '    const uint32_t band_cols = col_block_groups - band_first < band_depth',
      '                                   ? col_block_groups - band_first',
      '                                   : band_depth;',
      '    cbg_ = band_first + in_band % band_cols;',
      '    rbg_ = in_band / band_cols;'));

    // The block's walk and the hand-off, as simulate runs them.
    const from = FOLD_CODE.indexOf('  if (blockIdx.x >= tiles) return;');
    const persist = FOLD_CODE.slice(from, FOLD_CODE.indexOf('  if (blockIdx.x >= tiles) return;', from + 1));
    // One __syncthreads a tile, with the split or without, after the prologue's.
    expect(persist.split('__syncthreads();').length - 1).toBe(3);
    inOrder(persist,
      '  if (threadIdx.x == 0u) sNext[0] = gridDim.x + atomicAdd(tile_ctr, 1u);',
      lines('#if PEARL_BD_HASH_SPLIT', '  if (threadIdx.x < 128u) sHread[threadIdx.x >> 5][threadIdx.x & 31u] = 0u;'),
      lines('  __syncthreads();', '  uint32_t it = 0u;', '  for (uint32_t v = blockIdx.x; v < tiles; it++) {',
        '#if PEARL_BD_T_BUFS == 2', '    tbuf = (it & 1u) * TWORDS;'),
      lines('    uint32_t vgrab = 0u;', '    if (threadIdx.x == 0u) vgrab = gridDim.x + atomicAdd(tile_ctr, 1u);',
        '    uint32_t vnext = 0u;', '    tile([&]() {', '      vnext = sNext[it & 1u];'),
      lines(
        '#if PEARL_BD_HASH_SPLIT',
        '    uint32_t hseq = it;',
        '    asm volatile("" : "+r"(hseq));',
        '    const PearlBdHs h = hs_ids();',
        '    uint32_t tm[16];',
        '    if (hseq != 0u) {',
        '      load_tm(h, tbuf ^ TWORDS, tm);',
        '      if (!h.Y) split_first(h, tm, hseq);',
        '    }',
        '    if (threadIdx.x == 0u) sNext[(it + 1u) & 1u] = vgrab;',
        '    __syncthreads();',
        '    if (h.Y) {',
        '      if (hseq != 0u) split_rest(h, tm, hseq);',
        '      sVprev[h.s][h.lane] = v;',
        '    }',
        '#else'),
      lines(
        '#endif',
        '    aoff = aoffn;',
        '    boff = boffn;',
        '    v = vnext;',
        '  }',
        '#if PEARL_BD_HASH_SPLIT',
        '  {',
        '    const PearlBdHs h = hs_ids();',
        '    if (h.Y) {',
        '      tbuf = ((it - 1u) & 1u) * TWORDS;',
        '#if PEARL_BD_HIT_COORDS',
        '      hash_tile(sVprev[h.s][h.lane]);',
        '#else',
        '      uint32_t rbg, cbg;',
        '      tile_coords(sVprev[h.s][h.lane], rbg, cbg);',
        '      hash_tile(rbg, cbg);',
        '#endif',
        '    }',
        '  }',
        '#endif'));
    // The fold is the kernel's whole body, called once a launch, so sHread starts at
    // 0 in every launch and nothing of an earlier call is in flight.
    expect(KERNEL.split('pearl_fold_bd(Aprime, Bprime, col_off, rows_valid, tiles, test, hits);').length).toBe(2);
  });

  test('each half, for every N, gives BLAKE3\'s keyed hash, through the shared slot', () => {
    const rand = xorshift(41);
    const cases = [
      [new Array(8).fill(0), new Array(16).fill(0)],
      [new Array(8).fill(0xffffffff), new Array(16).fill(0xffffffff)],
      [[...IV, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19], Array.from({ length: 16 }, (_, i) => i)],
    ];
    while (cases.length < 2003) {
      cases.push([Array.from({ length: 8 }, () => rand()), Array.from({ length: 16 }, () => rand())]);
    }
    const slot = new Uint32Array(512).fill(GARBAGE);
    const bad = [];
    let checked = 0;
    for (let n = 0; n < cases.length; n++) {
      const [kw, m] = cases[n];
      const key = Buffer.alloc(32);
      const msg = Buffer.alloc(64);
      kw.forEach((w, i) => key.writeUInt32LE(w >>> 0, 4 * i));
      m.forEach((w, i) => msg.writeUInt32LE(w >>> 0, 4 * i));
      const out = keyedHash(key, msg);
      const h = Array.from({ length: 8 }, (_, i) => out.readUInt32LE(4 * i));
      // The default's compression is this keyed hash, and so is the spec's, whole.
      if (n < 50) {
        expect(compress(kw, m, 0, 64, FLAGS).slice(0, 8)).toEqual(h);
        const s = specStart(kw);
        specHalfRounds(s, m, 0, 14);
        expect(hashWords(s)).toEqual(h);
      }
      const lane = n & 31;
      for (let N = 0; N <= 14; N++) {
        const note = (what) => bad.length < 8 && bad.push(`case ${n} N ${N}: ${what}`);
        // X's half as built, then the spec's rest: X's state after N half-rounds.
        const a = splitFirst(kw, m, N);
        specHalfRounds(a, m, N, 14);
        if (hashWords(a).some((w, i) => w !== h[i])) note('first half');
        // The spec's first N, then Y's half as built.
        const b = specStart(kw);
        specHalfRounds(b, m, 0, N);
        halfRounds(b, m, N, 14);
        if (hashWords(b).some((w, i) => w !== h[i])) note('second half');
        // Both as built, X's state through the slot: all 8 words, and Y's test word,
        // in both byte orders, the full compare's first word.
        xStore(slot, lane, splitFirst(kw, m, N));
        const c = yLoad(slot, lane);
        halfRounds(c, m, N, 14);
        if (hashWords(c).some((w, i) => w !== h[i])) note('both halves');
        for (const hbe of [0, 1]) if (topWord(c, hbe) !== msf(h, 0, hbe)) note(`top word, hbe ${hbe}`);
        checked++;
      }
    }
    expect(bad).toEqual([]);
    expect(checked).toBe(2003 * 15);
    // With N = 0 or 14 one side would do nothing; the switch takes 1 to 13 only.
    expect(HEADER).toContain('#if PEARL_BD_HASH_SPLIT < 0 || PEARL_BD_HASH_SPLIT > 13');
  });

  test('the pair takes hash_tile\'s regions, each once, with a flag and a state slot of its own', () => {
    const X = new Map();
    const Y = new Map();
    for (let t = 0; t < 256; t++) {
      const r = hsIds(t);
      const hz = hasherOf(t);
      // Y is hash_tile's hasher, on the same region.
      expect([t, r.Y]).toEqual([t, hz.hasher]);
      if (r.Y) expect([t, r.L]).toEqual([t, hz.L]);
      expect(r.s).toBe((t >> 5) & 3);  // the scheduler, warp & 3
      // X waits on word threadIdx.x & 127 of sHread, which is [s][lane]; Y stores there.
      expect((t & 127)).toBe(r.s * 32 + r.lane);
      (r.Y ? Y : X).set(r.L, t);
    }
    // Each region once in each role: X and Y of it on one scheduler, one lane, two warps.
    expect([X.size, Y.size]).toEqual([128, 128]);
    for (let L = 0; L < 128; L++) {
      const x = X.get(L);
      const y = Y.get(L);
      expect([L, (x >> 5) & 3, x & 31]).toEqual([L, (y >> 5) & 3, y & 31]);
      expect(x >> 5).not.toBe(y >> 5);
    }
    expect([0, 1, 2, 3].map((s) => Y.get(32 * s) >> 5)).toEqual([0, 1, 6, 7]);
    expect([0, 1, 2, 3].map((s) => X.get(32 * s) >> 5)).toEqual([4, 5, 2, 3]);
    // The prologue's threads 0-127 set each flag once.
    const set = new Uint8Array(128);
    for (let t = 0; t < 128; t++) set[(t >> 5) * 32 + (t & 31)]++;
    expect(set.every((n) => n === 1)).toBe(true);
    // The readout writes every word of a buffer once a tile, region L's from warp L >> 4,
    // and the word for (L, c) is where X, Y and hash_tile read (L, c). The split's
    // predicated store takes the default's lanes and words: lanes 0-3 and 16-19.
    const owner = new Int32Array(2 * TWORDS).fill(-1);
    for (const tbuf of [0, TWORDS]) {
      for (let warp = 0; warp < 8; warp++) {
        for (let lane = 0; lane < 32; lane++) {
          for (let rl = 0; rl < 2; rl++) {
            const L = warp * 16 + (lane & 3) * 4 + 2 * rl + ((lane >> 4) & 1);
            const p = ((lane >> 2) & 3) === 0;  // setp.eq.u32 p, (lane >> 2) & 3, 0
            expect(writerRegion(warp, lane, rl)).toBe(p ? L : -1);
            if (!p) continue;
            expect(L >> 4).toBe(warp);
            for (let c = 0; c < 16; c++) {
              const a = tbuf + wordOf(L, c);
              expect(owner[a]).toBe(-1);
              owner[a] = L * 16 + c;
            }
          }
        }
      }
    }
    for (const tbuf of [0, TWORDS]) {
      for (let L = 0; L < 128; L++) {
        for (let c = 0; c < 16; c++) expect(owner[tbuf + wordOf(L, c)]).toBe(L * 16 + c);
      }
    }
    // The state slots: each lane's 16 words once, the 2048 words right after both
    // transcript buffers, inside the 24 KB the host gives.
    const smem = new Int32Array(24576 / 4).fill(-1);
    for (let t = 0; t < 256; t++) {
      const r = hsIds(t);
      if (r.Y) continue;
      for (let q = 0; q < 4; q++) {
        for (let i = 0; i < 4; i++) {
          const a = r.sH + slotWord(r.lane, q, i);
          expect(smem[a]).toBe(-1);
          smem[a] = r.L * 16 + 4 * q + i;
        }
      }
    }
    expect(smem.slice(0, 2 * TWORDS).every((x) => x === -1)).toBe(true);
    expect(smem.slice(2 * TWORDS).every((x) => x >= 0)).toBe(true);
    // Y loads the words X stored, in order; and uint4 q * 32 + lane is 16-byte aligned.
    for (let L = 0; L < 128; L++) {
      const r = hsIds(Y.get(L));
      const s = Array.from({ length: 16 }, (_, i) => smem[r.sH + slotWord(r.lane, i >> 2, i & 3)]);
      expect(s).toEqual(Array.from({ length: 16 }, (_, i) => L * 16 + i));
      expect(((r.sH + slotWord(r.lane, 0, 0)) * 4) % 16).toBe(0);
    }
    expect(FOLD).toContain('extern __shared__ __align__(128) uint32_t smem_u32[];');
    expect(FOLD).toContain('  constexpr uint32_t TWORDS = (PEARL_FOLD_THREADS / 32u) * warp_regions * PEARL_JACKPOT_BUCKETS;');
    expect((256 / 32) * 16 * 16).toBe(TWORDS);
  });

  test('the hand-off, at random warp speeds over 1 to 200 tiles, hashes each tile\'s regions once, as the shipped hash does', () => {
    let runs = 0;
    let hits = 0;
    // Every speed model on 1 to 12 tiles a block, at every N, both byte orders.
    for (let T = 1; T <= 12; T++) {
      for (let k = 0; k < 4; k++) {
        const seed = 1000 + T * 8 + k;
        hits += run(T, seed, MODEL_NAMES[(T + 2 * k) % MODEL_NAMES.length]).hits;
        runs++;
      }
    }
    // Longer blocks, up to 200 tiles.
    for (const [T, name] of [[16, 'bursty'], [33, 'changing'], [64, 'threads'], [127, 'warps'],
      [200, 'changing'], [200, 'hashersBehind']]) {
      hits += run(T, 5000 + T, name).hits;
      runs++;
    }
    expect(runs).toBe(54);
    // The hit path ran: about one region in 16 gets past the top word.
    expect(hits).toBeGreaterThan(1000);
  });

  test('the simulation catches a broken hand-off', () => {
    const broken = [
      // X does not wait for Y's flag: it overwrites a state Y has not loaded.
      ['no-wait', 'hashersBehind', /X overwrote region \d+'s state/],
      // Y sets its flag before it loads the state: the same.
      ['flag-early', 'hashersBehind', /X overwrote region \d+'s state|found tile \d+'s state/],
      // X waits for this hand-off's flag, which Y sets only past this barrier.
      ['wait-own', 'even', /deadlock/],
      // The flags are never set to 0: X waits at hand-off 1 for good.
      ['no-init', 'even', /deadlock/],
      // Hashing at hand-off 0, before there is a tile to hash.
      ['hash-at-0', 'even', /read nothing for tile -1|deadlock/],
      // The current tile's buffer instead of the last one's.
      ['same-buffer', 'even', /read .* for tile/],
      // Y reads its words after the barrier: the next tile may already write them.
      ['y-reads-late', 'hashersBehind', /overwrote region \d+ chunk \d+ of tile|read tile/],
      // Y keeps the tile it is in now, not the one it hashes.
      ['vprev-early', 'even', /coordinates/],
      // No hash after the loop, or one from the wrong buffer.
      ['no-epilogue', 'even', /hashed 0 times/],
      ['epilogue-buf', 'even', /read .* for tile/],
    ];
    for (const [mutate, model, error] of broken) {
      let got = '';
      try {
        run(6, 77, model, { mutate });
      } catch (e) {
        got = e.message;
      }
      expect([mutate, got]).toEqual([mutate, expect.stringMatching(error)]);
    }
  });

  test('the host gives the state its 8 KB, inside the shared split it asks for', () => {
    // The host's footprint for Turing's B-direct fold, and its geometry.
    expect(HOST).toContain(lines(
      '      : ctx->foldBDirect',
      '          ? (size_t)PEARL_BD_SMEM_A_STAGES * warpRows * regionsPerWarp * PEARL_ROWS_COUNT',
      '                    * PEARL_SB_STRIDE',
      '                + (size_t)PEARL_BD_T_BUFS * warpsPerBlock * regionsPerWarp * PEARL_WMMA_COL_BLK',
      '                      * PEARL_JACKPOT_BUCKETS * 4u',
      '                + (size_t)PEARL_BD_SPLIT_SMEM'));
    expect(HOST).toContain('  const uint32_t warpsPerBlock = threads / 32;');
    expect(HOST).toContain('  const uint32_t regionsPerWarp = rowTiles * (PEARL_WMMA_ROWS / PEARL_ROWS_COUNT);');
    expect(HOST).toContain('  const uint32_t rowTiles = wideWarps ? PEARL_FOLD_WIDE_ROW_TILES : PEARL_WMMA_ROW_TILES;');
    expect(HOST).toContain('  const bool wideWarps = ctx->foldWide || (ctx->foldTuring && PEARL_TURING_WIDE);');
    expect(HOST).toContain('                           : ctx->foldTuring ? PEARL_FOLD_TURING_THREADS');
    for (const d of ['#define PEARL_FOLD_TURING_THREADS 256u', '#define PEARL_FOLD_WIDE_ROW_TILES 4',
      '#define PEARL_WMMA_ROWS 16', '#define PEARL_ROWS_COUNT 16', '#define PEARL_WMMA_COL_BLK 4',
      '#define PEARL_JACKPOT_BUCKETS 16']) {
      expect(HEADER).toContain(d);
    }
    const GUARDS = guards();
    const value = (name, flags) => evaluate(name, preprocess(GUARDS, flags).defs);
    const smem = (flags) => value('PEARL_BD_SMEM_A_STAGES', flags) * 2 * 4 * 16 * 128
      + value('PEARL_BD_T_BUFS', flags) * (256 / 32) * (4 * (16 / 16)) * 4 * 16 * 4
      + value('PEARL_BD_SPLIT_SMEM', flags);
    const split = { PEARL_BD_HASH_SPLIT: 4, PEARL_BD_RUNBASE: 1, PEARL_BD_TAIL_LDG: 1 };
    expect([value('PEARL_BD_SMEM_A_STAGES', split), value('PEARL_BD_T_BUFS', split)]).toEqual([0, 2]);
    expect(smem({})).toBe(16384);
    expect(smem(split)).toBe(24576);
    for (let n = 1; n <= 13; n++) expect(smem({ ...split, PEARL_BD_HASH_SPLIT: n })).toBe(24576);
    // The fold's own sizes agree: two buffers of TWORDS words, then 4 slots of 512.
    expect(2 * TWORDS * 4).toBe(16384);
    expect(2 * TWORDS * 4 + 4 * 512 * 4).toBe(smem(split));
    // Turing has 64 KB of shared a multiprocessor, and the A-direct fold asks for half
    // as shared. With sHread and sVprev (1 KB) on the default's 128 bytes of static
    // shared, the split fits.
    expect(value('PEARL_BD_A_DIRECT_CARVEOUT', {})).toBe(50);
    const carve = (65536 * value('PEARL_BD_A_DIRECT_CARVEOUT', {})) / 100;
    expect(smem(split) + 128 + 2 * 4 * 32 * 4).toBe(24576 + 1152);
    expect(smem(split) + 1152).toBeLessThanOrEqual(carve);
    expect(smem({}) + 128).toBeLessThanOrEqual(carve);
    // A host built with the split refuses a footprint past that split. The default
    // host does not have the check: for it the carveout stays a hint.
    expect(HOST).toContain(lines(
      '    // so a refusal is dropped rather than failing the search.',
      '#if PEARL_BD_HASH_SPLIT'));
    expect(HOST).toContain(lines(
      '      cudaDeviceGetAttribute(&smPerSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, ctx->device);',
      '      const size_t split = (size_t)smPerSm * PEARL_BD_A_DIRECT_CARVEOUT / 100u;',
      '      if (smPerSm > 0 && smem + fs.sharedSizeBytes > split) {'));
    expect(HOST).toContain(lines(
      '        return false;',
      '      }',
      '    }',
      '#endif',
      '    if (ctx->foldBDirect',
      '        && cudaFuncSetAttribute(foldFn, cudaFuncAttributePreferredSharedMemoryCarveout,'));
    expect(HOST.split('cudaDevAttrMaxSharedMemoryPerMultiprocessor').length).toBe(2);
  });
});
