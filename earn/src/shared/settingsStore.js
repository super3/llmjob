'use strict';

// Read/write the desktop app's JSON settings file. The filesystem and a logger
// are injected, so the tricky paths — a missing file on first run, a corrupt
// file, a failed write — are unit-tested here rather than living untested inside
// the Electron main-process shell. main.js binds the real fs, the userData path,
// and console.error.

// Return the parsed settings object, or {} when the file is absent (first run)
// or unreadable. A corrupt/unreadable file is logged rather than swallowed
// silently, so one bad write doesn't wipe the user's saved address/region with
// no signal.
function readSettings(file, { fs, log = () => {} }) {
  if (!fs.existsSync(file)) return {};
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (e) {
    log('Could not read settings (' + file + '): ' + e.message);
    return {};
  }
}

// Persist settings as pretty JSON. Best-effort, but a write failure is logged
// (it means the next launch loses these settings) instead of being dropped.
// Returns the same object for chaining.
function writeSettings(file, data, { fs, log = () => {} }) {
  try {
    fs.writeFileSync(file, JSON.stringify(data, null, 2));
  } catch (e) {
    log('Could not save settings: ' + e.message);
  }
  return data;
}

module.exports = { readSettings, writeSettings };
