'use strict';

// PeakEngine watches a separate proprietary process, so every seam it has --
// spawn, the summary API, signals -- is injected and exercised here with no
// binary, no GPU and no sockets.

const { EventEmitter } = require('events');
const { PeakEngine, httpSummaryFetcher, API_PORT } = require('../src/main/peakEngine');

const ADDR = 'prl1pqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq';

function fakeChild(pid = 4242) {
  const c = new EventEmitter();
  c.pid = pid;
  c.stdout = new EventEmitter();
  c.stderr = new EventEmitter();
  return c;
}

function summary(over = {}) {
  return Object.assign({
    hashrate: 143.6e12,
    accepted_shares: 7,
    invalid_shares: 1,
    pool: { connected: true, url: 'us.pearl.herominers.com:1200' },
    gpus: [{ id: 0, name: 'RTX 5090', temperature_c: 53, power_w: 599 }],
  }, over);
}

function make(over = {}) {
  const child = over.child || fakeChild();
  const spawn = jest.fn(() => child);
  const fetchSummary = over.fetchSummary || jest.fn(async () => summary());
  const kill = jest.fn();
  const eng = new PeakEngine(Object.assign({
    binPath: '/usr/bin/peakminer', spawn, fetchSummary, kill, pollMs: 1000,
  }, over.opts));
  const logs = []; const events = []; const errors = []; const stops = [];
  eng.on('log', (l) => logs.push(l));
  eng.on('event', (e) => events.push(e));
  eng.on('error', (e) => errors.push(e));
  eng.on('stopped', (c) => stops.push(c));
  return {
    eng, child, spawn, fetchSummary, kill, logs, events, errors, stops,
  };
}

const settings = { address: ADDR, worker: 'rig01', endpoint: 'us.pearl.herominers.com:1200' };
const flush = () => new Promise((r) => setImmediate(r));

describe('start', () => {
  test('spawns with the pool, wallet and worker the settings carry', () => {
    const h = make();
    expect(h.eng.start(settings)).toBe(true);
    const [bin, args, opts] = h.spawn.mock.calls[0];
    expect(bin).toBe('/usr/bin/peakminer');
    expect(args).toEqual(expect.arrayContaining([
      '--coin', 'pearl',
      '-o', 'stratum+tcp://us.pearl.herominers.com:1200',
      '-u', ADDR,
      '-w', 'rig01',
    ]));
    expect(opts.detached).toBe(true);
  });

  // The miner forks. A plain kill of the pid we hold leaves the child mining --
  // that is how a "stopped" miner once held the card through an entire
  // benchmark window -- so it must lead its own process group for stop() to be
  // able to signal the group.
  test('is detached so the whole process group can be signalled later', () => {
    const h = make();
    h.eng.start(settings);
    expect(h.spawn.mock.calls[0][2].detached).toBe(true);
  });

  test('defaults the worker name when the settings have none', () => {
    const h = make();
    h.eng.start({ address: ADDR, endpoint: 'p:1' });
    expect(h.spawn.mock.calls[0][1]).toEqual(expect.arrayContaining(['-w', 'rig01']));
  });

  // Our own core takes nothing. Someone who switched engines is now paying a
  // fee they were not paying before, and should not have to read the vendor's
  // site to discover it.
  test('discloses the dev fee on every start, not once at install', () => {
    const h = make();
    h.eng.start(settings);
    expect(h.logs.some((l) => /2% dev fee/.test(l.line))).toBe(true);
    expect(h.logs.some((l) => /--miner native/.test(l.line))).toBe(true);
    h.eng.stop();
    h.child.emit('exit', 0);
    h.eng.start(settings);
    expect(h.logs.filter((l) => /2% dev fee/.test(l.line))).toHaveLength(2);
  });

  test('no binary is a clean refusal, not a crash', () => {
    const h = make({ opts: { binPath: null } });
    expect(h.eng.start(settings)).toBe(false);
    expect(h.logs[0].level).toBe('error');
    expect(h.logs[0].line).toMatch(/not found/);
    expect(h.stops).toEqual([0]);
    expect(h.eng.isRunning()).toBe(false);
  });

  test('a spawn that throws is reported and stops cleanly', () => {
    const boom = new Error('EACCES');
    const h = make();
    h.eng.spawn = jest.fn(() => { throw boom; });
    expect(h.eng.start(settings)).toBe(false);
    expect(h.errors).toEqual([boom]);
    expect(h.stops).toEqual([0]);
    expect(h.eng.isRunning()).toBe(false);
  });

  test('starting twice is refused', () => {
    const h = make();
    expect(h.eng.start(settings)).toBe(true);
    expect(h.eng.start(settings)).toBe(false);
    expect(h.spawn).toHaveBeenCalledTimes(1);
  });

  test('counters reset on each start', async () => {
    const h = make();
    h.eng.start(settings);
    await flush();
    expect(h.eng.accepted).toBe(7);
    h.child.emit('exit', 0);
    h.eng.start(settings);
    expect(h.eng.accepted).toBe(0);
    expect(h.eng.hashrate).toBe(0);
  });

  test('a child that has no pipes does not crash the wiring', () => {
    const c = fakeChild();
    c.stdout = null;
    c.stderr = null;
    const h = make({ child: c });
    expect(h.eng.start(settings)).toBe(true);
  });
});

describe('console output', () => {
  test('splits lines and strips the ANSI the miner colours with', () => {
    const h = make();
    h.eng.start(settings);
    h.child.stdout.emit('data', '\x1B[32mconnected\x1B[0m\npartial');
    const lines = h.logs.filter((l) => l.level === 'info').map((l) => l.line);
    expect(lines).toContain('connected');
    // the partial line is held until its newline arrives
    expect(lines).not.toContain('partial');
    h.child.stdout.emit('data', ' rest\n');
    expect(h.logs.map((l) => l.line)).toContain('partial rest');
  });

  test('stderr is logged at error level', () => {
    const h = make();
    h.eng.start(settings);
    h.child.stderr.emit('data', 'pool unreachable\n');
    expect(h.logs.some((l) => l.level === 'error' && /pool unreachable/.test(l.line))).toBe(true);
  });

  test('blank lines are dropped', () => {
    const h = make();
    h.eng.start(settings);
    const before = h.logs.length;
    h.child.stdout.emit('data', '\n   \n');
    expect(h.logs).toHaveLength(before);
  });
});

describe('telemetry', () => {
  test('translates one summary into a status event in TH/s', async () => {
    const h = make();
    h.eng.start(settings);
    await flush();
    const s = h.events.filter((e) => e.type === 'status').pop();
    expect(s.hashrate).toBeCloseTo(143.6, 3);
    expect(s.accepted).toBe(7);
    expect(s.rejected).toBe(1);
    expect(s.temp).toBe(53);
    expect(s.gpuIndex).toBe(0);
  });

  // Matching PearlEngine: the per-card figure goes to the network board, and
  // starting to send it because the engine changed would be a behaviour change
  // wearing a display change's clothes. PeakMiner does report it.
  test('power is deliberately not reported even though the API has it', async () => {
    const h = make();
    h.eng.start(settings);
    await flush();
    expect(h.events.filter((e) => e.type === 'status').pop().power).toBe(null);
  });

  test('announces connected exactly once, when the pool has us', async () => {
    const fetchSummary = jest.fn()
      .mockResolvedValueOnce(summary({ pool: { connected: false } }))
      .mockResolvedValue(summary());
    const h = make({ fetchSummary });
    h.eng.start(settings);
    await flush();
    expect(h.events.filter((e) => e.type === 'connected')).toHaveLength(0);
    await h.eng._applySummary(summary());
    await h.eng._applySummary(summary());
    const conn = h.events.filter((e) => e.type === 'connected');
    expect(conn).toHaveLength(1);
    expect(conn[0].endpoint).toBe('us.pearl.herominers.com:1200');
  });

  test('takes the card name from the API when the settings had none', async () => {
    const h = make();
    h.eng.start({ address: ADDR, endpoint: 'p:1' });
    await flush();
    expect(h.eng.gpu).toBe('RTX 5090');
  });

  test('an explicit gpu setting is not overwritten by the API', async () => {
    const h = make();
    h.eng.start(Object.assign({ gpu: 'My Card' }, settings));
    await flush();
    expect(h.eng.gpu).toBe('My Card');
  });

  test('a nonsense temperature reads as absent rather than zero', () => {
    const h = make();
    h.eng.start(settings);
    h.eng._applySummary(summary({ gpus: [{ temperature_c: 0 }] }));
    expect(h.eng.temp).toBe(null);
    h.eng._applySummary(summary({ gpus: [{ temperature_c: 'n/a' }] }));
    expect(h.eng.temp).toBe(null);
  });

  test('missing or malformed fields leave the previous reading alone', async () => {
    const h = make();
    h.eng.start(settings);
    await flush();
    h.eng._applySummary({ pool: { connected: true } });
    expect(h.eng.hashrate).toBeCloseTo(143.6, 3);
    expect(h.eng.accepted).toBe(7);
  });

  test('an empty or non-object summary is ignored', () => {
    const h = make();
    h.eng.start(settings);
    const before = h.events.length;
    h.eng._applySummary(null);
    h.eng._applySummary('nope');
    h.eng._applySummary(undefined);
    expect(h.events).toHaveLength(before);
  });

  test('a summary with no gpus array still produces a status', () => {
    const h = make();
    h.eng.start(settings);
    h.eng._applySummary(summary({ gpus: [] }));
    expect(h.events.filter((e) => e.type === 'status').length).toBeGreaterThan(0);
  });

  // The miner takes a few seconds to bind its API. A failed poll during that
  // window is normal, not an error worth showing anyone.
  test('a failing poll is swallowed while the miner is still starting', async () => {
    const fetchSummary = jest.fn(() => Promise.reject(new Error('ECONNREFUSED')));
    const h = make({ fetchSummary });
    h.eng.start(settings);
    await flush();
    expect(h.errors).toHaveLength(0);
  });

  test('a fetcher that throws synchronously does not take the engine down', async () => {
    const fetchSummary = jest.fn(() => { throw new Error('sync boom'); });
    const h = make({ fetchSummary });
    expect(h.eng.start(settings)).toBe(true);
    await flush();
    expect(h.errors).toHaveLength(0);
  });

  test('polls on a timer and stops polling once the child exits', async () => {
    jest.useFakeTimers();
    try {
      const h = make();
      h.eng.start(settings);
      expect(h.fetchSummary).toHaveBeenCalledTimes(1);
      jest.advanceTimersByTime(3000);
      expect(h.fetchSummary).toHaveBeenCalledTimes(4);
      h.child.emit('exit', 0);
      jest.advanceTimersByTime(5000);
      expect(h.fetchSummary).toHaveBeenCalledTimes(4);
    } finally {
      jest.useRealTimers();
    }
  });
});

describe('lifecycle', () => {
  test('a child error is surfaced', () => {
    const h = make();
    h.eng.start(settings);
    const e = new Error('spawn failed later');
    h.child.emit('error', e);
    expect(h.errors).toEqual([e]);
  });

  test('exit reports the code and clears running state', () => {
    const h = make();
    h.eng.start(settings);
    expect(h.eng.isRunning()).toBe(true);
    h.child.emit('exit', 3);
    expect(h.stops).toEqual([3]);
    expect(h.eng.isRunning()).toBe(false);
  });

  test('a signal-killed child reports 0 rather than null', () => {
    const h = make();
    h.eng.start(settings);
    h.child.emit('exit', null);
    expect(h.stops).toEqual([0]);
  });

  test('stop signals the process GROUP, not just the leader', () => {
    const h = make();
    h.eng.start(settings);
    h.eng.stop();
    expect(h.kill).toHaveBeenCalledWith(-4242, 'SIGTERM');
  });

  test('a miner that ignores SIGTERM is SIGKILLed after the grace period', () => {
    jest.useFakeTimers();
    try {
      const h = make({ opts: { killGraceMs: 500 } });
      h.eng.start(settings);
      h.eng.stop();
      jest.advanceTimersByTime(600);
      expect(h.kill).toHaveBeenCalledWith(-4242, 'SIGKILL');
    } finally {
      jest.useRealTimers();
    }
  });

  test('a child that exits within the grace period is not SIGKILLed', () => {
    jest.useFakeTimers();
    try {
      const h = make({ opts: { killGraceMs: 500 } });
      h.eng.start(settings);
      h.eng.stop();
      h.child.emit('exit', 0);
      jest.advanceTimersByTime(600);
      expect(h.kill).not.toHaveBeenCalledWith(-4242, 'SIGKILL');
    } finally {
      jest.useRealTimers();
    }
  });

  // The normal race between our grace timer and the miner's own exit.
  test('signalling a process that has already gone is not an error', () => {
    const h = make();
    h.eng.start(settings);
    h.eng.kill = jest.fn(() => { throw new Error('ESRCH'); });
    expect(() => h.eng.stop()).not.toThrow();
    expect(h.errors).toHaveLength(0);
  });

  test('stop before start does nothing', () => {
    const h = make();
    expect(() => h.eng.stop()).not.toThrow();
    expect(h.kill).not.toHaveBeenCalled();
  });

  test('gpuIndex is the single card', () => {
    expect(make().eng.gpuIndex()).toBe(0);
  });

  test('defaults fill in when no tuning is passed', () => {
    const e = new PeakEngine({});
    expect(e.apiPort).toBe(API_PORT);
    expect(e.pollMs).toBeGreaterThan(0);
    expect(e.statusIntervalS).toBeGreaterThan(0);
    expect(e.killGraceMs).toBeGreaterThan(0);
    expect(e.isRunning()).toBe(false);
  });

  test('the default kill reaches process.kill', () => {
    const e = new PeakEngine({});
    const spy = jest.spyOn(process, 'kill').mockImplementation(() => {});
    e.kill(process.pid, 0);
    expect(spy).toHaveBeenCalledWith(process.pid, 0);
    spy.mockRestore();
  });

  test('constructed with no arguments at all', () => {
    expect(() => new PeakEngine()).not.toThrow();
  });

  test('start with no settings at all does not throw', () => {
    const h = make();
    expect(h.eng.start()).toBe(true);
    expect(h.eng.endpoint).toBe(null);
  });

  test('settings with no endpoint leave it null rather than "undefined"', () => {
    const h = make();
    h.eng.start({ address: ADDR });
    expect(h.eng.endpoint).toBe(null);
  });

  // The fake-timer test above covers the handle that has no unref; this one runs
  // on real timers, where it does, and also exercises the SIGKILL escalation
  // end to end rather than by clock-advance.
  test('escalates to SIGKILL on a real timer', async () => {
    const h = make({ opts: { killGraceMs: 10 } });
    h.eng.start(settings);
    h.eng.stop();
    await new Promise((r) => setTimeout(r, 40));
    expect(h.kill).toHaveBeenCalledWith(-4242, 'SIGKILL');
  });

  // Timer handles are not guaranteed to carry unref (they do not under some
  // fake-timer implementations), and assuming it would crash the stop path.
  test('a kill timer handle with no unref still escalates', () => {
    const h = make();
    h.eng.start(settings);
    const st = jest.spyOn(global, 'setTimeout').mockImplementation((fn) => { fn(); return {}; });
    try {
      h.eng.stop();
      expect(h.kill).toHaveBeenCalledWith(-4242, 'SIGKILL');
    } finally { st.mockRestore(); }
  });

  test('the escalation is skipped when the child has already gone', () => {
    const h = make();
    h.eng.start(settings);
    const st = jest.spyOn(global, 'setTimeout').mockImplementation((fn) => {
      h.eng.child = null;
      fn();
      return {};
    });
    try {
      h.eng.stop();
      expect(h.kill).not.toHaveBeenCalledWith(-4242, 'SIGKILL');
    } finally { st.mockRestore(); }
  });

  test('a poll timer handle with no unref is tolerated', () => {
    const si = jest.spyOn(global, 'setInterval').mockImplementation(() => ({}));
    try {
      const h = make();
      expect(h.eng.start(settings)).toBe(true);
    } finally { si.mockRestore(); }
  });
});

describe('httpSummaryFetcher', () => {
  function fakeHttp(behaviour) {
    return {
      get: jest.fn((opts, cb) => {
        const req = new EventEmitter();
        req.destroy = jest.fn((e) => req.emit('error', e));
        behaviour(req, cb, opts);
        return req;
      }),
    };
  }

  test('parses a JSON body', async () => {
    const http = fakeHttp((req, cb) => {
      const res = new EventEmitter();
      process.nextTick(() => {
        cb(res);
        res.emit('data', '{"hashrate":');
        res.emit('data', '1}');
        res.emit('end');
      });
    });
    await expect(httpSummaryFetcher(http)(4068)).resolves.toEqual({ hashrate: 1 });
    expect(http.get.mock.calls[0][0]).toMatchObject({ host: '127.0.0.1', port: 4068, path: '/summary' });
  });

  test('rejects on a body that is not JSON', async () => {
    const http = fakeHttp((req, cb) => {
      const res = new EventEmitter();
      process.nextTick(() => { cb(res); res.emit('data', '<html>'); res.emit('end'); });
    });
    await expect(httpSummaryFetcher(http)(4068)).rejects.toThrow();
  });

  test('rejects on a request error', async () => {
    const http = fakeHttp((req) => {
      process.nextTick(() => req.emit('error', new Error('ECONNREFUSED')));
    });
    await expect(httpSummaryFetcher(http)(4068)).rejects.toThrow('ECONNREFUSED');
  });

  // A wedged miner must not stall the poll loop forever.
  test('a timeout destroys the request and rejects', async () => {
    const http = fakeHttp((req) => {
      process.nextTick(() => req.emit('timeout'));
    });
    await expect(httpSummaryFetcher(http)(4068)).rejects.toThrow('summary timeout');
  });
});
