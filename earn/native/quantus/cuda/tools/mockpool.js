#!/usr/bin/env node
'use strict';

// A local Quantus pool for timing a miner without a wallet: it speaks the
// LuckyPool stratum (login, job, submit, keepalived; src/shared/quantus/stratum.js
// is the client side), hands out a job at a fixed share difficulty, checks every
// submitted nonce on the reference hash, and prints a line every 10 s:
//
//   t <s> shares <valid>/<all> rate <MH/s from valid shares x difficulty> mean <MH/s since the first share>
//
// Nothing leaves this machine; no address is used for anything. A miner's own
// hashrate readout and this share count are two measures of the same run.
//
//   node mockpool.js [--port 3334] [--difficulty 2000000000] [--secs 0]

const net = require('net');
const crypto = require('crypto');
const path = require('path');
const q = require(path.join(__dirname, '..', '..', '..', '..', 'src', 'shared', 'quantus', 'qpow'));

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const port = Number(arg('--port', 3334));
const difficulty = BigInt(arg('--difficulty', '2000000000'));
const secs = Number(arg('--secs', 0));
const target = q.targetFromDifficulty(difficulty);
const t0 = Date.now();
let seq = 1;
let job = null;
let valid = 0;
let all = 0;
let firstAt = 0;
let sinceFirst = 0;
let mark = 0;
let markT = t0;
const seen = new Set();
const clients = new Set();

function newJob() {
  seq += 1;
  job = {
    algo: 'qpow-poseidon2',
    difficulty: Number(difficulty),
    extranonce: crypto.randomBytes(4).toString('hex'),
    job_id: seq + '-' + difficulty,
    mining_hash: crypto.randomBytes(32).toString('hex'),
    seq,
    target: target.toString(16).padStart(128, '0'),
  };
  for (const c of clients) send(c, { jsonrpc: '2.0', method: 'job', params: { clean_jobs: true, job } });
}

function send(sock, msg) {
  if (!sock.destroyed) sock.write(JSON.stringify(msg) + '\n');
}

function check(nonceHex) {
  if (typeof nonceHex !== 'string' || !/^[0-9a-f]{128}$/i.test(nonceHex)) return 'bad nonce';
  const nonce = Buffer.from(nonceHex, 'hex');
  if (!nonce.subarray(0, 4).equals(Buffer.from(job.extranonce, 'hex'))) return 'wrong extranonce';
  const key = job.job_id + ':' + nonceHex.toLowerCase();
  if (seen.has(key)) return 'duplicate';
  seen.add(key);
  const hash = q.nonceHash(Buffer.from(job.mining_hash, 'hex'), nonce);
  return q.meetsTarget(hash, target) ? null : 'low difficulty';
}

const server = net.createServer((sock) => {
  clients.add(sock);
  let buf = '';
  sock.setEncoding('utf8');
  sock.on('data', (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line) continue;
      let m;
      try {
        m = JSON.parse(line);
      } catch (e) {
        console.log('not JSON: ' + line.slice(0, 120));
        continue;
      }
      if (m.method === 'login') {
        console.log('login from ' + String((m.params && m.params.agent) || '?').slice(0, 60));
        send(sock, { id: m.id, jsonrpc: '2.0', result: { id: 'mock', status: 'OK', extensions: ['keepalive'], job } });
      } else if (m.method === 'submit') {
        all += 1;
        const why = m.params && m.params.job_id === job.job_id ? check(m.params.nonce) : 'stale';
        if (!why) {
          valid += 1;
          if (!firstAt) firstAt = Date.now();
          else sinceFirst += 1;
          send(sock, { id: m.id, jsonrpc: '2.0', result: { status: 'OK' } });
        } else {
          console.log('rejected: ' + why);
          send(sock, { id: m.id, jsonrpc: '2.0', error: { code: -1, message: why } });
        }
      } else if (m.method === 'keepalived') {
        send(sock, { id: m.id, jsonrpc: '2.0', result: { status: 'KEEPALIVED' } });
      } else {
        console.log('method ' + String(m.method).slice(0, 40));
        send(sock, { id: m.id, jsonrpc: '2.0', result: { status: 'OK' } });
      }
    }
  });
  sock.on('close', () => clients.delete(sock));
  sock.on('error', () => clients.delete(sock));
});

newJob();
server.listen(port, '127.0.0.1', () => console.log('mock pool on 127.0.0.1:' + port + ', difficulty ' + difficulty));
setInterval(newJob, 60000);
setInterval(() => {
  const now = Date.now();
  const d = Number(difficulty);
  const rate = ((valid - mark) * d) / ((now - markT) / 1000) / 1e6;
  const mean = firstAt && now > firstAt ? (sinceFirst * d) / ((now - firstAt) / 1000) / 1e6 : 0;
  console.log('t ' + Math.round((now - t0) / 1000) + ' shares ' + valid + '/' + all + ' rate ' + rate.toFixed(1)
    + ' mean ' + mean.toFixed(1));
  mark = valid;
  markT = now;
  if (secs && now - t0 >= secs * 1000) process.exit(0);
}, 10000);
