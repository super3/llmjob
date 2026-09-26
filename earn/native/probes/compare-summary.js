// node compare-summary.js <compare dir> -- one table: each miner's OWN displayed hashrate,
// averaged over minutes 1-5 and at the end, with clock/power/shares alongside.
const fs = require('fs'); const D = process.argv[2];
const read = (f) => { try { return fs.readFileSync(D + '/' + f, 'utf8'); } catch (e) { return ''; } };
const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
function gpu(name) {
  const rows = read(name + '.gpu.csv').trim().split('\n').filter(Boolean).map((l) => l.split(',').map(Number)).filter((r) => r[0] >= 60);
  return { mhz: mean(rows.map((r) => r[1])), w: mean(rows.map((r) => r[2])), c: mean(rows.map((r) => r[3])) };
}
const out = [];
{ // ours: the CLI's status line, one per second
  const lines = read('ours.log').split('\n').filter((l) => l.includes('TH/s'));
  const pts = lines.map((l) => { const th = +l.match(/([\d.]+) TH\/s/)[1]; const m = l.match(/up (\d+)m (\d+)s/); return { t: +m[1] * 60 + +m[2], th, acc: +(l.match(/(\d+) accepted/) || [0, 0])[1], rej: +(l.match(/(\d+) rejected/) || [0, 0])[1] }; });
  const steady = pts.filter((p) => p.t >= 60); const last = pts[pts.length - 1] || {};
  out.push({ miner: 'LLMJob Earn (this core)', avg: mean(steady.map((p) => p.th)), end: last.th, acc: last.acc, rej: last.rej, ...gpu('ours') });
}
function api(file, pick) {
  return read(file).split('\n').filter((l) => l.includes('{')).map((l) => { const i = l.indexOf(' '); try { return { t: +l.slice(0, i), j: JSON.parse(l.slice(i + 1)) }; } catch (e) { return null; } }).filter(Boolean).map((x) => ({ t: x.t, ...pick(x.j) }));
}
{
  const pts = api('srb.api', (j) => { const a = j.algorithms[0]; return { th: a.hashrate.gpu.total / 1e12, acc: a.shares.accepted, rej: a.shares.rejected }; });
  const steady = pts.filter((p) => p.t >= 60); const last = pts[pts.length - 1] || {};
  out.push({ miner: 'SRBMiner 3.6.9 (2% fee)', avg: mean(steady.map((p) => p.th)), end: last.th, acc: last.acc, rej: last.rej, ...gpu('srb') });
}
{
  const pts = api('peak.api', (j) => {
    // PeakMiner /summary: find the largest plausible hashrate field (H/s) and share counters
    let hs = 0, acc = 0, rej = 0; const walk = (o) => { for (const k in o) { const v = o[k]; if (v && typeof v === 'object') walk(v); else if (typeof v === 'number') { if (/hash/i.test(k) && v > hs) hs = v; if (/^(accepted|ok|shares_ok|accepted_shares)$/i.test(k)) acc = Math.max(acc, v); if (/^(rejected|invalid|inv|rejected_shares)$/i.test(k)) rej = Math.max(rej, v); } } };
    walk(j); return { th: hs > 1e9 ? hs / 1e12 : hs, acc, rej };
  });
  const steady = pts.filter((p) => p.t >= 60); const last = pts[pts.length - 1] || {};
  out.push({ miner: 'PeakMiner 2.17.1 (2% fee)', avg: mean(steady.map((p) => p.th)), end: last.th, acc: last.acc, rej: last.rej, ...gpu('peak') });
}
const top = Math.max(...out.map((r) => r.avg || 0));
console.log('| Miner | Displayed TH/s (avg min 1-5) | At 5:00 | vs top | Shares ok/rej | SM clock | Power |');
console.log('|---|---|---|---|---|---|---|');
for (const r of out) console.log(`| ${r.miner} | ${r.avg.toFixed(1)} | ${(+r.end).toFixed(1)} | ${((r.avg / top - 1) * 100).toFixed(1)}% | ${r.acc}/${r.rej} | ${r.mhz.toFixed(0)} MHz | ${r.w.toFixed(0)} W |`);
