'use strict';
// stubpool.js <port> [difficulty]: a local Quantus pool in LuckyPool's stratum shape (login, job, submit,
// keepalived), so a miner can be timed on a card with no wallet and no real pool. It hands out one random
// job (a new one every 120 s), checks every share against our reference (earn/src/shared/quantus/qpow.js),
// answers OK and logs it with the rate the good shares stand for. Listens on 127.0.0.1 only. Timing only.
const net = require('net');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');
const here = path.join(__dirname, '..', '..', '..', '..', 'src', 'shared', 'quantus', 'qpow.js');
const q = require(fs.existsSync(here) ? here : path.join(process.env.HOME, 'qtc/src/earn/src/shared/quantus/qpow'));
const port = Number(process.argv[2] || 3333);
const diff = BigInt(process.argv[3] || '10000000000');
const targetN = ((1n << 512n) - 1n) / diff;
const target = targetN.toString(16).padStart(128, '0');
const t0 = Date.now();
const jobs = new Map();
let good = 0, bad = 0;
let seq = 1, mining = crypto.randomBytes(32).toString('hex');
jobs.set(seq + '-' + diff, mining);
const subs = new Set();
const job = (en) => ({ algo: 'qpow-poseidon2', difficulty: Number(diff), extranonce: en, job_id: seq + '-' + diff,
  mining_hash: mining, seq, target });
setInterval(() => {
  seq++; mining = crypto.randomBytes(32).toString('hex'); jobs.set(seq + '-' + diff, mining);
  for (const s of subs) s.write(JSON.stringify({ jsonrpc: '2.0', method: 'job', params: { clean_jobs: true, job: job(s.en) } }) + '\n');
}, 120000).unref();
let next = 1;
net.createServer((sock) => {
  sock.en = (next++).toString(16).padStart(8, '0');
  let buf = '';
  sock.on('data', (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line) continue;
      let m; try { m = JSON.parse(line); } catch (e) { continue; }
      if (m.method === 'login') {
        subs.add(sock);
        console.log(new Date().toISOString(), 'login', String((m.params || {}).agent || ''));
        sock.write(JSON.stringify({ id: m.id, result: { id: sock.en, status: 'OK', extensions: ['keepalive'], job: job(sock.en) } }) + '\n');
      } else if (m.method === 'submit') {
        const p = m.params || {}, hdr = jobs.get(p.job_id);
        let ok = false;
        try { ok = !!hdr && q.meetsTarget(q.nonceHash(Buffer.from(hdr, 'hex'), Buffer.from(String(p.nonce), 'hex')), targetN); } catch (e) { ok = false; }
        if (ok) good++; else bad++;
        const secs = (Date.now() - t0) / 1000;
        console.log(new Date().toISOString(), ok ? 'good' : 'BAD', 'share', good, bad, 'effective', (Number(diff) * good / secs / 1e6).toFixed(2), 'MH/s over', secs.toFixed(0), 's');
        sock.write(JSON.stringify({ id: m.id, result: { status: 'OK' } }) + '\n');
      } else if (m.method === 'keepalived') {
        sock.write(JSON.stringify({ id: m.id, result: { status: 'KEEPALIVED' } }) + '\n');
      } else {
        sock.write(JSON.stringify({ id: m.id, result: null, error: { code: -1, message: 'unknown method' } }) + '\n');
      }
    }
  });
  sock.on('close', () => subs.delete(sock));
  sock.on('error', () => subs.delete(sock));
}).listen(port, '127.0.0.1', () => console.log('stub pool on 127.0.0.1:' + port + ', difficulty ' + diff));
