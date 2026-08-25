'use strict';

const { EventEmitter } = require('events');
const { PearlEngine } = require('../src/main/pearlEngine');

// PearlEngine's whole job is to make our own miner indistinguishable from
// alpha-miner as far as main.js and the renderer are concerned. The UI is
// driven by two parsed events and nothing else:
//
//   { type: 'status',    gpuIndex, hashrate, accepted, rejected, power, temp, gpu }
//   { type: 'connected', gpuIndex, endpoint, gpu }
//
// alpha-miner produces those by having its stdout scraped. PearlMiner produces
// share/rejected/hashrate/job events instead, because it IS the miner. If this
// translation is wrong the miner still mines perfectly and the UI shows a dead
// card, which is the kind of failure nobody reports as a bug.

function fakeSocket() {
  const s = new EventEmitter();
  s.written = [];
  s.write = (d) => { s.written.push(d); return true; };
  s.destroy = () => { s.emit('close'); };
  return s;
}

function fakeCore() {
  const c = new EventEmitter();
  c.setJob = jest.fn();
  c.stop = jest.fn();
  return c;
}

const ADDR = 'prl1px5ervx6ftaegmdhqa5ajemh20j2uw7l9jt5j5s97rljp72yt3s8qncrxud';
const HEADER = '000000203a49fea8b6d42c60c543fe0f029749787679372495e5d2d1007e29e3'
  + '25e1c08065a0cedc057ba091aabd10476017b3d3f4e38eafb59707ded57f71e1feb7b5d155918b6a0dea0018';
const TARGET = '00000000000007fff80000000000000000000000000000000000000000000000';

function boot() {
  const sock = fakeSocket();
  const core = fakeCore();
  const e = new PearlEngine({ connect: () => sock, createCore: () => core });
  const events = { log: [], event: [], error: [], stopped: [] };
  for (const k of Object.keys(events)) e.on(k, (x) => events[k].push(x));
  return { e, sock, core, events };
}

function started() {
  const b = boot();
  b.e.start({ address: ADDR, worker: 'rig01', endpoint: 'us.pearl.herominers.com:1200', gpu: 'RTX 4090' });
  b.sock.emit('connect');
  return b;
}

function jobLine() {
  return JSON.stringify({
    id: null,
    method: 'mining.notify',
    params: { job_id: 'j1', header: HEADER, target: TARGET, height: 1, cert_version: 3 },
  }) + '\n';
}

describe('PearlEngine — the MinerManager surface', () => {
  test('exposes exactly what main.js drives', () => {
    const { e } = boot();
    expect(typeof e.start).toBe('function');
    expect(typeof e.stop).toBe('function');
    expect(typeof e.isRunning).toBe('function');
    expect(e.isRunning()).toBe(false);
  });

  test('runs once started and stops on stop', () => {
    const { e, core } = started();
    expect(e.isRunning()).toBe(true);
    e.stop();
    expect(core.stop).toHaveBeenCalled();
    expect(e.isRunning()).toBe(false);
  });

  // Without a built addon there is nothing to mine with. That is a clean stop
  // with an explanation, not a crash and not a silently idle card.
  test('an unbuilt core stops cleanly and says why', () => {
    const sock = fakeSocket();
    const e = new PearlEngine({ connect: () => sock, createCore: null });
    const logs = [];
    e.on('log', (l) => logs.push(l));
    e.start({ address: ADDR, worker: 'rig01', endpoint: 'h:1' });
    expect(e.isRunning()).toBe(false);
    expect(logs.some((l) => /not built/i.test(l.line))).toBe(true);
  });
});

describe('PearlEngine — the events the UI actually reads', () => {
  test('announces the card once work arrives, not merely on connect', () => {
    const { sock, events } = started();
    expect(events.event.filter((e) => e.type === 'connected')).toHaveLength(0);
    sock.emit('data', jobLine());
    const c = events.event.filter((e) => e.type === 'connected');
    expect(c).toHaveLength(1);
    expect(c[0]).toEqual({
      type: 'connected', gpuIndex: 0,
      endpoint: 'us.pearl.herominers.com:1200', gpu: 'RTX 4090',
    });
  });

  // A job per block would otherwise re-announce the card every few seconds.
  test('the card is announced only once', () => {
    const { sock, events } = started();
    sock.emit('data', jobLine());
    sock.emit('data', jobLine());
    expect(events.event.filter((e) => e.type === 'connected')).toHaveLength(1);
  });

  test('a hashrate tick becomes a status the sparkline can read', () => {
    const { core, events } = started();
    core.emit('hashrate', 101.5);
    const s = events.event.filter((e) => e.type === 'status');
    expect(s).toHaveLength(1);
    expect(s[0].hashrate).toBe(101.5);
    expect(s[0].gpuIndex).toBe(0);
  });

  // The UI's counters are cumulative for the session; PearlMiner reports one
  // verdict per share as it lands.
  test('share verdicts accumulate rather than replace', () => {
    const { e, core, events } = started();
    e.miner.emit('share', { jobId: 'j1', accepted: true });
    e.miner.emit('share', { jobId: 'j1', accepted: true });
    e.miner.emit('rejected', { jobId: 'j1', reason: 'nope' });
    const last = events.event.filter((x) => x.type === 'status').pop();
    expect(last.accepted).toBe(2);
    expect(last.rejected).toBe(1);
  });

  test('a restart clears the counters', () => {
    const b = started();
    b.e.miner.emit('share', {});
    b.e.stop();
    b.e.start({ address: ADDR, worker: 'rig01', endpoint: 'h:1' });
    expect(b.e.accepted).toBe(0);
  });

  // We do not read NVML. Reporting a made-up number would be worse than the
  // blank the UI already renders for a card that does not report one.
  test('power and temperature are absent, not invented', () => {
    const { core, events } = started();
    core.emit('hashrate', 50);
    const s = events.event.find((e) => e.type === 'status');
    expect(s.power).toBeNull();
    expect(s.temp).toBeNull();
  });

  test('miner logs and errors pass straight through', () => {
    const { e, events } = started();
    e.miner.emit('log', { level: 'info', line: 'hello' });
    e.miner.emit('error', new Error('boom'));
    expect(events.log.some((l) => l.line === 'hello')).toBe(true);
    expect(events.error[0].message).toBe('boom');
  });
});
