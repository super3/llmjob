#!/usr/bin/env node
'use strict';

// Say which model the pool is on, while another miner runs: a race compares rates, and a NEB rate means nothing
// without its model. Logs in as a worker of its own, mines nothing, and prints one line a model:
//   model 73cfcbb5 slot 0 · 5230 of 13404 live · 2.21% with 2 or fewer live · 5.37% with 3
// and one a new block:
//   block 14226 slot 0 model 73cfcbb5
//
//   node earn/native/neb/tools/nebwatch.js --dataset neb-datasets.bin --pool host:port --address neb1… [--secs 300]
//
// Counting a model's live samples is the reference on the CPU: about a second a model.

const net = require('net');
const nh = require('../../../src/shared/neb/nebhash2');
const stratum = require('../../../src/shared/neb/stratum');
const { loadDatasets } = require('./pool');

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const say = (line) => console.log(new Date().toISOString().slice(11, 19) + ' ' + line);
const pct = (x) => (x * 100).toFixed(2) + '%';

function main() {
  const { datasets } = loadDatasets(arg('--dataset', 'neb-datasets.bin'));
  const [host, port] = String(arg('--pool', 'neb.rabidmining.com:3470')).split(':');
  const secs = Number(arg('--secs', 300));
  const seen = new Map();
  const sock = net.connect({ host, port: Number(port) });
  let buf = '';
  let block = null;
  const stop = (code) => {
    sock.destroy();
    process.exit(code);
  };
  setTimeout(() => stop(0), secs * 1000);
  sock.on('connect', () => sock.write(stratum.encode(stratum.buildSubscribe('llmjob-earn-tools'))));
  sock.on('error', (e) => {
    say('watch: ' + e.message);
    stop(1);
  });
  sock.on('close', () => stop(0));
  sock.on('data', (chunk) => {
    buf += String(chunk);
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const m = stratum.parseMessage(buf.slice(0, i));
      buf = buf.slice(i + 1);
      if (m.kind === 'subscribed') sock.write(stratum.encode(stratum.buildAuthorize(stratum.loginOf(arg('--address', ''), 'watch'))));
      if (m.kind === 'login-fail' || m.kind === 'subscribe-fail') {
        say('watch: ' + stratum.errorText(m.error));
        stop(1);
      }
      if (m.kind === 'model' && datasets[m.slot] && !seen.has(m.weightsHash)) {
        const ds = datasets[m.slot];
        const live = nh.liveSamples(nh.modelFromPacked(ds.arch, m.weights), ds).reduce((a, b) => a + b, 0);
        const p = live / ds.n;
        seen.set(m.weightsHash, true);
        say('model ' + m.weightsHash.slice(0, 8) + ' slot ' + m.slot + ' · ' + live + ' of ' + ds.n + ' live · '
          + pct(nh.candidateShare(p, 2)) + ' with 2 or fewer live · ' + pct(nh.exactShare(p, 3)) + ' with 3');
      }
      if (m.kind === 'job' && m.parent.toString('hex') !== block) {
        block = m.parent.toString('hex');
        say('block ' + m.height + ' slot ' + m.slot + ' model ' + m.weightsHash.slice(0, 8));
      }
    }
  });
}

main();
