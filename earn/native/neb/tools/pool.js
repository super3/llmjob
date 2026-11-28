'use strict';

// For the NEB tools: the network's dataset file, and a model from a pool.

const fs = require('fs');
const net = require('net');
const blake3 = require('../../../src/shared/miner/blake3');
const nh = require('../../../src/shared/neb/nebhash2');
const stratum = require('../../../src/shared/neb/stratum');

// The main network's dataset file (docs/miner-integration.md of github.com/SilasEigen/neb).
const MAINNET_DATASET_HASH = '790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72';

function loadDatasets(file) {
  const buf = fs.readFileSync(file);
  return { datasets: nh.parseDatasets(buf), hash: blake3.hash(buf).toString('hex') };
}

// The first model the pool sends for each slot in `slots`, within `ms`: { slot: packed weights }.
function fetchModels(endpoint, address, slots, ms) {
  return new Promise((resolve, reject) => {
    const [host, port] = endpoint.split(':');
    const got = {};
    const sock = net.connect({ host, port: Number(port) });
    let buf = '';
    const done = (err) => {
      clearTimeout(timer);
      sock.destroy();
      if (Object.keys(got).length) resolve(got);
      else reject(err || new Error('the pool sent no model in ' + Math.round(ms / 1000) + ' s'));
    };
    const timer = setTimeout(() => done(null), ms);
    sock.on('connect', () => sock.write(stratum.encode(stratum.buildSubscribe('llmjob-earn-tools'))));
    sock.on('error', done);
    sock.on('data', (chunk) => {
      buf += String(chunk);
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const m = stratum.parseMessage(buf.slice(0, i));
        buf = buf.slice(i + 1);
        if (m.kind === 'subscribed') sock.write(stratum.encode(stratum.buildAuthorize(stratum.loginOf(address, 'tools'))));
        if (m.kind === 'login-fail' || m.kind === 'subscribe-fail') return done(new Error(stratum.errorText(m.error)));
        if (m.kind === 'model' && slots.includes(m.slot) && !got[m.slot]) {
          got[m.slot] = m.weights;
          if (slots.every((s) => got[s])) return done(null);
        }
      }
      return undefined;
    });
  });
}

module.exports = { MAINNET_DATASET_HASH, loadDatasets, fetchModels };
