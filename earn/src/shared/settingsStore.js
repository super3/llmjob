'use strict';

// Read/write the desktop app's JSON settings file. The filesystem and a logger
// are injected, so the tricky paths — a missing file on first run, a corrupt
// file, a failed write — are unit-tested here rather than living untested inside
// the Electron main-process shell. main.js binds the real fs, the userData path,
// and console.error.

// Return the parsed settings object, or {} when the file is absent (first run)
// or unreadable. A corrupt/unreadable file is logged rather than swallowed
// silently, so one bad write doesn't wipe the user's saved address/region with
// no signal. So is a file that parses to something other than an object (a
// bare `null`): every caller reads keys off the answer.
function readSettings(file, { fs, log = () => {} }) {
  if (!fs.existsSync(file)) return {};
  try {
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (data && typeof data === 'object' && !Array.isArray(data)) return data;
    throw new Error('not a JSON object');
  } catch (e) {
    log('Could not read settings (' + file + '): ' + e.message);
    return {};
  }
}

// Replace a file whole, so a crash or a power cut mid-write leaves the old
// file or the new one, never a torn one: write a temp file beside it, then
// rename it over. A power cut once left app-update.yml full of NUL bytes (see
// shared/updateConfig). The temp name carries the process id because the desktop
// app and the CLI share node.json, and each must only ever rename its own.
//
// Windows can refuse the rename (EPERM, EACCES, EBUSY) while something else
// holds the file open, an antivirus scan for one. Then the file is written in
// place, as it always used to be, rather than not saved at all.
// `mode` applies to a file this creates, as with fs.writeFileSync. Throws when
// nothing could be written.
const RENAME_REFUSED = new Set(['EPERM', 'EACCES', 'EBUSY']);
function writeFileAtomic(file, text, { fs, mode }) {
  const opts = mode == null ? undefined : { mode };
  const tmp = file + '.' + process.pid + '.tmp';
  try {
    fs.writeFileSync(tmp, text, opts);
    fs.renameSync(tmp, file);
    return;
  } catch (e) {
    try { fs.unlinkSync(tmp); } catch (_) { /* never written, or already gone */ }
    if (!RENAME_REFUSED.has(e.code)) throw e;
  }
  fs.writeFileSync(file, text, opts);
}

// Persist settings as pretty JSON. Best-effort, but a write failure is logged
// (it means the next launch loses these settings) instead of being dropped,
// naming the file: main keeps preferences and lost cards this way too.
// Returns the same object for chaining.
function writeSettings(file, data, { fs, log = () => {} }) {
  try {
    writeFileAtomic(file, JSON.stringify(data, null, 2), { fs });
  } catch (e) {
    log('Could not save ' + file + ': ' + e.message);
  }
  return data;
}

module.exports = { readSettings, writeSettings, writeFileAtomic };
