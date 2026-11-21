'use strict';

// How many libuv pool threads this process really runs, started the way the
// shells start (shared/threadPool.js). Run it on Windows both ways:
//
//   node earn/scripts/check-thread-pool.js
//   earn/node_modules/electron/dist/electron.exe earn/scripts/check-thread-pool.js
//
// It runs 12 pbkdf2 jobs at once, each about the same length, and counts how
// many finished in the first round. "threads 4" is libuv's default; the shells
// should print 12 (every job at once, POOL_THREADS being 32). --inproc only sets
// the variable in this process, the way the shells did before, to show it does
// nothing on Windows.

const threadPool = require('../src/shared/threadPool');
const { spawn } = require('child_process');

const JOBS = 12;
const inproc = process.argv.includes('--inproc');
const electron = !!process.versions.electron;

function measure(done) {
  const crypto = require('crypto');
  const t0 = Date.now();
  const ends = [];
  for (let i = 0; i < JOBS; i++) {
    crypto.pbkdf2('p', 's', 200000, 64, 'sha512', () => {
      ends.push(Date.now() - t0);
      if (ends.length < JOBS) return;
      // A job that ran in the first round ends within half a job of the first.
      const first = Math.round(ends[0] * 1.5);
      const round = ends.filter((t) => t <= first).length;
      process.stdout.write((electron ? 'electron' : 'node') + (inproc ? ' --inproc' : '')
        + ': UV_THREADPOOL_SIZE=' + process.env.UV_THREADPOOL_SIZE + ', threads ' + round
        + ' (finish ms ' + ends.join(',') + ')\n');
      done();
    });
  }
}

if (inproc) {
  threadPool.widenThreadPool(process.env);
  measure(() => process.exit(0));
} else if (electron) {
  // Exits here on Windows; the copy it starts measures. Its output goes
  // nowhere (the app's copy is detached), so it also goes to a file in the
  // temp folder: check-thread-pool.out.
  if (!threadPool.relaunchApp(process, spawn)) {
    const fs = require('fs');
    const path = require('path');
    const os = require('os');
    const write = process.stdout.write.bind(process.stdout);
    process.stdout.write = (s) => {
      fs.appendFileSync(path.join(os.tmpdir(), 'check-thread-pool.out'), s);
      return write(s);
    };
    measure(() => require('electron').app.exit(0));
  }
} else if (threadPool.needsRelaunch(process)) {
  threadPool.relaunchCli(process, spawn).then((code) => process.exit(code == null ? 1 : code));
} else {
  measure(() => process.exit(0));
}
