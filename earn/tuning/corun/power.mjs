// Usage: node power.mjs <smi.csv> <startIso> <endIso>
// smi.csv from: nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu --format=csv,noheader,nounits -lms 500
import fs from 'fs';
const [file, a, b] = process.argv.slice(2);
const s = Date.parse(a), e = Date.parse(b);
const rows = fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map((l) => l.split(',').map((x) => x.trim()))
  .map(([ts, p, sm, mem, u]) => ({ t: new Date(ts.replace(/\//g, '-').replace(' ', 'T')).getTime(), p: +p, sm: +sm, mem: +mem, u: +u }))
  .filter((r) => r.t >= s && r.t <= e && Number.isFinite(r.p));
const m = (k) => rows.length ? +(rows.reduce((x, r) => x + r[k], 0) / rows.length).toFixed(0) : null;
console.log(JSON.stringify({ n: rows.length, watts: m('p'), smMhz: m('sm'), memMhz: m('mem'), util: m('u') }));
