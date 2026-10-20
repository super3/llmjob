// Helper for launch.sh: medians, the probes' output, and the sweep's summaries.
//
//   node launch.js med <nvidia-smi samples> <skip>   -> "mhz watts tempC", medians of the
//                                                      samples after the first <skip>
//   node launch.js hr <hashrate.js stdout>          -> "th sd n"
//   node launch.js bench <bench stdout>             -> "th"
//   node launch.js verify <verify-hits.js stdout>   -> "hits verified failed salts PASS"
//   node launch.js summary <results.tsv> <phase> <control> [points...]
//       One line a point: "pt runs th mhz watts thpg d wins". th, mhz, watts and thpg
//       (TH/s per GHz of SM clock) are medians over the phase's runs; d is the change of
//       the median th against the control's, in percent; wins is how many rounds the point
//       beat the control run of the same round, out of the rounds both ran. Then
//       "BEST <pt> <d>": the other point with the highest median th.
//
// results.tsv: one run a line, "phase round point th mhz watts tempC". Prints "-" for a
// value it could not read, so the shell always gets the same number of fields.
'use strict';
const fs = require('fs');

const median = (xs) => {
  const s = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (!s.length) return NaN;
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
};
const fmt = (x, d) => (Number.isFinite(x) ? x.toFixed(d) : '-');
const read = (f) => { try { return fs.readFileSync(f, 'utf8'); } catch (e) { return ''; } };
const lastJson = (text) => {
  const lines = text.split('\n').filter((l) => l.startsWith('{'));
  try { return JSON.parse(lines[lines.length - 1]); } catch (e) { return {}; }
};

const [cmd, file, ...rest] = process.argv.slice(2);

if (cmd === 'med') {
  const skip = Number(rest[0] || 0);
  const rows = read(file).split('\n').filter((l) => /\d/.test(l)).slice(skip)
    .map((l) => l.split(',').map((v) => parseFloat(v)));
  console.log([0, 1, 2].map((i) => fmt(median(rows.map((r) => r[i])), i === 1 ? 1 : 0)).join(' '));
} else if (cmd === 'hr') {
  const j = lastJson(read(file));
  console.log([fmt(j.th, 2), fmt(j.sd, 2), Number.isFinite(j.n) ? j.n : '-'].join(' '));
} else if (cmd === 'bench') {
  const m = read(file).match(/->\s+([0-9.]+) TH\/s/g);
  console.log(m ? m[m.length - 1].replace(/[^0-9.]/g, '') : '-');
} else if (cmd === 'verify') {
  const j = lastJson(read(file));
  console.log(['hits', 'verified', 'failed', 'salts', 'PASS'].map((k) => (k in j ? String(j[k]) : '-')).join(' '));
} else if (cmd === 'summary') {
  const [phase, control, ...order] = rest;
  const runs = read(file).split('\n').map((l) => l.trim().split(/\s+/))
    .filter((f) => f.length >= 7 && f[0] === phase)
    .map((f) => ({ round: f[1], pt: f[2], th: +f[3], mhz: +f[4], w: +f[5] }))
    .filter((r) => Number.isFinite(r.th));
  const points = order.length ? order : [...new Set(runs.map((r) => r.pt))];
  const of = (pt) => runs.filter((r) => r.pt === pt);
  const ctlTh = median(of(control).map((r) => r.th));
  const ctlByRound = new Map(of(control).map((r) => [r.round, r.th]));
  let best = null;
  for (const pt of points) {
    const rs = of(pt);
    if (!rs.length) continue;
    const th = median(rs.map((r) => r.th));
    const d = (th / ctlTh - 1) * 100;
    const paired = rs.filter((r) => ctlByRound.has(r.round));
    const wins = paired.filter((r) => r.th > ctlByRound.get(r.round)).length;
    const thpg = median(rs.map((r) => (r.mhz > 0 ? r.th / (r.mhz / 1000) : NaN)));
    console.log([pt, rs.length, fmt(th, 2), fmt(median(rs.map((r) => r.mhz)), 0),
      fmt(median(rs.map((r) => r.w)), 1), fmt(thpg, 2),
      pt === control ? '+0.00' : (d >= 0 ? '+' : '') + fmt(d, 2),
      pt === control ? '-' : wins + '/' + paired.length].join(' '));
    if (pt !== control && Number.isFinite(d) && (!best || d > best.d)) best = { pt, d };
  }
  console.log(best ? 'BEST ' + best.pt + ' ' + fmt(best.d, 2) : 'BEST - -');
} else {
  console.error('usage: see the top of launch.js');
  process.exit(2);
}
