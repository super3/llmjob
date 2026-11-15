'use strict';

// The app's log on disk. Until v0.7.2 the log lived only in the window, so when
// the app closed itself (a crash in the miner's native code takes the whole
// process with it) there was nothing left to read. Every line the window shows
// is also appended here, as it happens, so the last lines before a crash are on
// disk. A user's Windows PC closed the app at START (14 November) and there was
// no log to ask for.
//
// Files, in `dir` (the app's userData/logs):
//   earn.log     this run and the ones before it, up to maxBytes
//   earn.1.log   the file before that, kept when earn.log fills
//   mining.json  written while a mining session runs and removed when it ends,
//                so a launch that finds it knows the last run closed without
//                stopping (takeCrash)
//
// Nothing here may break the app: a disk that is full, read-only or gone makes
// each call a no-op. Writes are synchronous so a line is on disk before the
// next instruction, which is the point when the next instruction crashes.

const path = require('path');

const MAX_BYTES = 1024 * 1024;

function makeLogFile({ fs, dir, maxBytes = MAX_BYTES, now = () => new Date() }) {
  const file = path.join(dir, 'earn.log');
  const old = path.join(dir, 'earn.1.log');
  const marker = path.join(dir, 'mining.json');
  let size = null;

  // Make the folder and learn the file's size, once.
  function open() {
    if (size !== null) return;
    fs.mkdirSync(dir, { recursive: true });
    try {
      size = fs.statSync(file).size;
    } catch (e) {
      size = 0;
    }
  }

  // One line: "2026-11-14T13:05:01.123Z INFO  pearl: ..." Levels are padded so the
  // text lines up.
  function write(level, line) {
    try {
      open();
      const text = now().toISOString() + ' ' + String(level || 'info').toUpperCase().padEnd(5) + ' '
        + String(line) + '\n';
      const bytes = Buffer.byteLength(text);
      if (size > 0 && size + bytes > maxBytes) {
        fs.renameSync(file, old);
        size = 0;
      }
      fs.appendFileSync(file, text);
      size += bytes;
    } catch (e) {
      /* logging must never take the app down */
    }
  }

  // A mining session started (START) or ended (STOP, quit).
  function markMining(info) {
    try {
      open();
      fs.writeFileSync(marker, JSON.stringify({ startedAt: now().toISOString(), ...info }));
    } catch (e) {
      /* see write */
    }
  }
  function clearMining() {
    try {
      fs.unlinkSync(marker);
    } catch (e) {
      /* already gone */
    }
  }

  // What the last run left if it closed while mining, or null. Read once: the
  // marker goes with it, so the next launch doesn't report the same crash.
  function takeCrash() {
    let text;
    try {
      text = fs.readFileSync(marker, 'utf8');
    } catch (e) {
      return null;
    }
    clearMining();
    try {
      const m = JSON.parse(text);
      return m && typeof m === 'object' ? m : {};
    } catch (e) {
      return {};
    }
  }

  return { dir, file, write, markMining, clearMining, takeCrash };
}

module.exports = { makeLogFile, MAX_BYTES };
