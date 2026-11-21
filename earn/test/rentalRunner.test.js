'use strict';

// The rented-container runner against a fake docker, a fake server and
// hand-driven timers.

const { RentalRunner, LLM_WAIT_MS, LLM_WAIT_STEP_MS, NEXT_POLL_AFTER_JOB_MS } = require('../src/main/rentalRunner');
const R = require('../src/shared/rental');

const IMAGE = 'ghcr.io/super3/llmjob-bench@sha256:' + 'd'.repeat(64);
const OFFER = { runId: 'cr_1', jobId: 'cj_1', name: 'bench', image: IMAGE, env: { MODE: 'x' }, card: 0, durationSec: 600 };
const CARDS = [{ index: 0, gpu: 'RTX 4090', vramMb: 24564, vendor: 'nvidia' }];

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

function make(over = {}) {
  const timers = [];
  const intervals = [];
  const logs = [];
  let clock = 1000;
  const waitD = deferred();
  const follower = { stop: jest.fn(), onLine: null };
  const d = {
    api: {
      poll: jest.fn(async () => ({ job: null, pollMs: 30000 })),
      heartbeat: jest.fn(async () => ({ action: 'continue' })),
      end: jest.fn(async () => ({ success: true })),
    },
    docker: {
      available: jest.fn(async () => true),
      cleanup: jest.fn(async () => 0),
      run: jest.fn(async () => ({ code: 0, stdout: '', stderr: '' })),
      logs: jest.fn((name, onLine) => { follower.onLine = onLine; return follower; }),
      wait: jest.fn(() => waitD.promise),
    },
    cosign: { available: jest.fn(async () => false), verify: jest.fn(async () => ({ code: 0 })) },
    cards: jest.fn(async () => CARDS),
    driver: jest.fn(async () => '580.65'),
    pauseCard: jest.fn(),
    resumeCard: jest.fn(),
    llmBusy: jest.fn(() => false),
    temps: jest.fn(async () => ({ 0: { core: 60, hotspot: 70, fan: 50, util: 99 } })),
    takeBackRequested: jest.fn(() => false),
    clearTakeBack: jest.fn(),
    log: (level, line) => logs.push(level + ' ' + line),
    platform: 'linux',
    worker: 'rig01',
    caps: { cpus: 4, memoryMb: 8192 },
    now: () => clock,
    setTimer: jest.fn((fn, ms) => { const t = { fn, ms, cleared: false }; timers.push(t); return t; }),
    clearTimer: jest.fn((t) => { t.cleared = true; }),
    every: jest.fn((fn, ms) => { const t = { fn, ms, cleared: false }; intervals.push(t); return t; }),
    stopEvery: jest.fn((t) => { t.cleared = true; }),
    sleep: jest.fn(async () => {}),
  };
  for (const k of Object.keys(over)) {
    if (d[k] && typeof d[k] === 'object' && !Array.isArray(d[k]) && typeof over[k] === 'object') Object.assign(d[k], over[k]);
    else d[k] = over[k];
  }
  const runner = new RentalRunner(d);
  const live = (list, ms) => list.filter((t) => !t.cleared && (ms == null || t.ms === ms));
  return {
    runner, d, logs, follower, waitD,
    advance: (ms) => { clock += ms; },
    timers: (ms) => live(timers, ms),
    intervals: (ms) => live(intervals, ms),
  };
}

// Start the runner and run its first poll, handing it `offer`.
async function startWithJob(h, offer = OFFER) {
  h.d.api.poll.mockResolvedValueOnce({ job: offer, pollMs: 30000 });
  await h.runner.start();
  await h.runner.tick();
  await flush();
}

describe('RentalRunner', () => {
  test('default timers and clock come from the platform', () => {
    jest.useFakeTimers();
    try {
      const r = new RentalRunner({ log: () => {} });
      expect(r.d.now).toBe(Date.now);
      const fn = jest.fn();
      r.d.clearTimer(r.d.setTimer(fn, 10));
      r.d.stopEvery(r.d.every(fn, 10));
      jest.advanceTimersByTime(50);
      expect(fn).not.toHaveBeenCalled();
      const done = jest.fn();
      r.d.sleep(10).then(done);
      jest.advanceTimersByTime(10);
    } finally {
      jest.useRealTimers();
    }
  });

  test('on Windows it says the runner is not there yet and does nothing', async () => {
    const h = make({ platform: 'win32' });
    expect(await h.runner.start()).toBe(false);
    expect(h.logs).toEqual(['info Rented containers: not available on Windows yet']);
    expect(h.d.docker.available).not.toHaveBeenCalled();
    expect(h.runner.status()).toBeNull();
    expect(h.runner.label()).toBeNull();
  });

  test('without docker it still checks in, offering no card', async () => {
    const h = make({ docker: { available: jest.fn(async () => false) } });
    expect(await h.runner.start()).toBe(true);
    expect(await h.runner.start()).toBe(true);   // a second start is a no-op
    expect(h.logs[0]).toMatch(/docker is not installed/);
    expect(h.d.docker.cleanup).not.toHaveBeenCalled();
    expect(h.timers(0)).toHaveLength(1);
    h.timers(0)[0].fn();
    await flush();
    expect(h.d.api.poll).toHaveBeenCalledWith({
      worker: 'rig01', platform: 'linux', docker: false, driver: '580.65',
      cards: [Object.assign({}, CARDS[0], { free: false })],
    });
    // The server's poll interval is used for the next one.
    expect(h.timers(30000)).toHaveLength(1);
  });

  test('a stop while docker is being checked leaves it stopped', async () => {
    const avail = deferred();
    const h = make({ docker: { available: jest.fn(() => avail.promise) } });
    const started = h.runner.start();
    expect(await h.runner.stop()).toBe(false);
    avail.resolve(true);
    expect(await started).toBe(false);
    h.runner._schedule(0);
    expect(h.timers()).toHaveLength(0);
    await h.runner.tick();
    expect(h.d.api.poll).not.toHaveBeenCalled();
  });

  test('a failed poll is logged and retried after the default interval', async () => {
    const h = make();
    h.d.api.poll.mockRejectedValueOnce(new Error('HTTP 500'));
    await h.runner.start();
    await h.runner.tick();
    expect(h.logs.pop()).toMatch(/could not ask the server for work: HTTP 500/);
    expect(h.timers(R.POLL_MS)).toHaveLength(1);
    // An answer with no pollMs keeps the default too, and a new schedule
    // replaces the old timer.
    h.d.api.poll.mockResolvedValueOnce(null);
    await h.runner.tick();
    expect(h.timers(R.POLL_MS)).toHaveLength(1);
  });

  test('no card is offered while the LLM is answering', async () => {
    const h = make({ llmBusy: jest.fn(() => true) });
    await h.runner.start();
    await h.runner.tick();
    expect(h.d.api.poll.mock.calls[0][0].cards[0].free).toBe(false);
  });

  test('a job from start to finish: pause, pull, run, logs, exit, resume, report', async () => {
    const h = make();
    await startWithJob(h);
    expect(h.d.docker.cleanup).toHaveBeenCalled();
    expect(h.d.pauseCard).toHaveBeenCalledWith(0);
    expect(h.d.docker.run.mock.calls.map((c) => c[0][0])).toEqual(['pull', 'rm', 'run']);
    const [args, env] = h.d.docker.run.mock.calls[2];
    expect(args).toEqual(expect.arrayContaining(['--gpus', 'device=0', '-e', 'MODE', IMAGE]));
    expect(env).toEqual({ MODE: 'x' });
    expect(h.runner.status()).toEqual({ card: 0, state: 'running', endsAt: 1000 + 600000, jobId: 'cj_1', name: 'llmjob-run-cr_1' });
    expect(h.runner.label()).toBe('Rented: running a container · 10m left');
    expect(h.logs.some((l) => /GPU 0 is rented for 10m; mining paused/.test(l))).toBe(true);
    expect(h.logs.some((l) => /GPU 0 · Rented: running a container · 10m left/.test(l))).toBe(true);
    // The first heartbeat went out at once.
    expect(h.d.api.heartbeat).toHaveBeenCalledWith('cr_1', { state: 'running', logs: [] });
    // Unsigned images are warned about once.
    expect(h.logs.filter((l) => /signatures are not checked/.test(l))).toHaveLength(1);

    h.follower.onLine('hello');
    h.follower.onLine('world');
    // Heartbeats carry the lines; every fourth logs the status.
    for (let i = 0; i < 3; i++) await h.intervals(R.HEARTBEAT_MS)[0].fn();
    expect(h.d.api.heartbeat.mock.calls[1][1]).toEqual({ state: 'running', logs: ['hello', 'world'] });
    expect(h.logs.filter((l) => /GPU 0 · Rented/.test(l))).toHaveLength(2);

    h.follower.onLine('bye');
    h.waitD.resolve(0);
    await flush();
    await flush();
    expect(h.d.resumeCard).toHaveBeenCalledWith(0);
    // An exited container needs no docker stop, only removing.
    expect(h.d.docker.run.mock.calls.slice(3).map((c) => c[0].slice(0, 2))).toEqual([['rm', '-f']]);
    expect(h.follower.stop).toHaveBeenCalled();
    expect(h.d.api.end).toHaveBeenCalledWith('cr_1', { reason: 'exited', exitCode: 0, detail: null, logs: ['bye'] });
    expect(h.logs.some((l) => /ended \(exited, exit 0\); mining resumed/.test(l))).toBe(true);
    expect(h.runner.status()).toBeNull();
    expect(h.timers(NEXT_POLL_AFTER_JOB_MS)).toHaveLength(1);
    expect(h.intervals(R.HEARTBEAT_MS)).toHaveLength(0);
    // A second poll while a job runs does nothing.
  });

  test('an AMD card gets /dev/kfd and /dev/dri', async () => {
    const h = make({ cards: jest.fn(async () => [{ index: 0, gpu: 'RX 6800', vramMb: 16368, vendor: 'amd' }]) });
    await startWithJob(h);
    expect(h.d.docker.run.mock.calls[2][0]).toEqual(expect.arrayContaining(['/dev/kfd', '/dev/dri']));
  });

  test('a card the rig did not list runs as NVIDIA', async () => {
    const h = make({ cards: jest.fn(async () => []) });
    await startWithJob(h);
    expect(h.d.docker.run.mock.calls[2][0]).toEqual(expect.arrayContaining(['--gpus', 'device=0']));
  });

  test('an image from elsewhere, or by tag, is refused before anything stops', async () => {
    const h = make();
    await startWithJob(h, Object.assign({}, OFFER, { image: 'docker.io/evil@sha256:' + 'e'.repeat(64) }));
    expect(h.d.pauseCard).not.toHaveBeenCalled();
    expect(h.d.docker.run).not.toHaveBeenCalled();
    expect(h.d.api.end).toHaveBeenCalledWith('cr_1', expect.objectContaining({ reason: 'refused', detail: expect.stringMatching(/not from/) }));
    expect(h.logs.some((l) => /ended \(refused: image is not from ghcr.io\/super3\/\)$/.test(l))).toBe(true);
    const h2 = make();
    await startWithJob(h2, Object.assign({}, OFFER, { env: [] }));
    expect(h2.d.api.end.mock.calls[0][1].reason).toBe('refused');
  });

  test('cosign: a failed check refuses, a passed one runs', async () => {
    const bad = make({ cosign: { available: jest.fn(async () => true), verify: jest.fn(async () => ({ code: 1, stderr: 'no signatures found' })) } });
    await startWithJob(bad);
    expect(bad.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'refused', detail: 'signature check failed: no signatures found' });
    expect(bad.d.resumeCard).toHaveBeenCalledWith(0);
    const good = make({ cosign: { available: jest.fn(async () => true) } });
    await startWithJob(good);
    expect(good.d.cosign.verify).toHaveBeenCalledWith(IMAGE);
    expect(good.runner.status().state).toBe('running');
    expect(good.logs.some((l) => /signatures are not checked/.test(l))).toBe(false);
  });

  test('with signatures required and no cosign, every image is refused', async () => {
    const h = make({ requireSignature: true });
    await startWithJob(h);
    expect(h.d.api.end.mock.calls[0][1].detail).toMatch(/LLMJOB_RENT_REQUIRE_SIGNATURE=1/);
  });

  test('a failed pull or run fails the job with docker\'s words', async () => {
    const pull = make();
    pull.d.docker.run.mockResolvedValueOnce({ code: 1, stderr: 'line1\nmanifest unknown' });
    await startWithJob(pull);
    expect(pull.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'failed', detail: 'docker pull failed: line1 | manifest unknown' });
    const run = make();
    run.d.docker.run
      .mockResolvedValueOnce({ code: 0 })
      .mockResolvedValueOnce({ code: 0 })
      .mockResolvedValueOnce({ code: 125 });
    await startWithJob(run);
    expect(run.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'failed', detail: 'docker run failed: ' });
    // It never started, so nothing to stop or remove after.
    expect(run.d.docker.run).toHaveBeenCalledTimes(3);
  });

  test('a docker error that throws fails the job', async () => {
    const h = make();
    h.d.docker.run.mockRejectedValueOnce(new Error('spawn EACCES'));
    await startWithJob(h);
    expect(h.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'failed', detail: 'spawn EACCES' });
  });

  test('a container is never started while the LLM answers: it waits, then gives up', async () => {
    const waits = make();
    waits.d.llmBusy.mockReturnValueOnce(false).mockReturnValueOnce(true).mockReturnValueOnce(false);
    await startWithJob(waits);
    expect(waits.d.sleep).toHaveBeenCalledWith(LLM_WAIT_STEP_MS);
    expect(waits.runner.status().state).toBe('running');

    const busy = make();
    busy.d.llmBusy.mockReturnValueOnce(false).mockReturnValue(true);
    await startWithJob(busy);
    expect(busy.d.sleep).toHaveBeenCalledTimes(LLM_WAIT_MS / LLM_WAIT_STEP_MS + 1);
    expect(busy.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'refused', detail: "the owner's LLM stayed busy" });
  });

  test('the server says stop: docker stop, then remove', async () => {
    const h = make();
    await startWithJob(h);
    h.d.api.heartbeat.mockResolvedValueOnce({ action: 'stop', reason: 'job deleted' });
    await h.intervals(R.HEARTBEAT_MS)[0].fn();
    const after = h.d.docker.run.mock.calls.slice(3).map((c) => c[0].slice(0, 3));
    expect(after).toEqual([['stop', '-t', '10'], ['rm', '-f', 'llmjob-run-cr_1']]);
    expect(h.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'stopped', detail: 'job deleted' });
    // A stop with no reason gets one.
    const h2 = make();
    await startWithJob(h2);
    h2.d.api.heartbeat.mockResolvedValueOnce({ action: 'stop' });
    await h2.intervals(R.HEARTBEAT_MS)[0].fn();
    expect(h2.d.api.end.mock.calls[0][1].detail).toBe('the server said stop');
  });

  test('a failed heartbeat keeps its lines for the next one', async () => {
    const h = make();
    await startWithJob(h);
    h.follower.onLine('a');
    h.d.api.heartbeat.mockRejectedValueOnce(new Error('offline'));
    await h.intervals(R.HEARTBEAT_MS)[0].fn();
    await h.intervals(R.HEARTBEAT_MS)[0].fn();
    expect(h.d.api.heartbeat.mock.calls.pop()[1].logs).toEqual(['a']);
  });

  test('the log buffer keeps only the newest lines', async () => {
    const h = make();
    await startWithJob(h);
    for (let i = 0; i < R.MAX_LOG_BUFFER + 5; i++) h.follower.onLine('l' + i);
    expect(h.runner.job.logs).toHaveLength(R.MAX_LOG_BUFFER);
    expect(h.runner.job.logs[0]).toBe('l5');
  });

  test('the temperature guard stops a hot card', async () => {
    const h = make();
    await startWithJob(h);
    await h.intervals(R.TEMP_CHECK_MS)[0].fn();
    expect(h.d.api.end).not.toHaveBeenCalled();
    h.d.temps.mockResolvedValueOnce({ 0: { core: 88, hotspot: null, fan: 60, util: 99 } });
    await h.intervals(R.TEMP_CHECK_MS)[0].fn();
    expect(h.logs.some((l) => /stopping the container on GPU 0: core at 88/.test(l))).toBe(true);
    expect(h.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'temperature', detail: 'core at 88 °C (limit 85)' });
  });

  test('the job time running out stops it as expired', async () => {
    const h = make();
    await startWithJob(h);
    const end = h.timers(600000)[0];
    await end.fn();
    expect(h.d.api.end.mock.calls[0][1].reason).toBe('expired');
    expect(h.logs.some((l) => /^info .*ended \(expired\)/.test(l))).toBe(true);
  });

  test('docker wait failing fails the job', async () => {
    const h = make();
    await startWithJob(h);
    h.waitD.reject(new Error('no such container'));
    await flush();
    await flush();
    expect(h.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'failed', detail: 'docker wait failed: no such container' });
  });

  test('the owner takes the PC back: the job ends, and none starts for an hour', async () => {
    const h = make();
    await startWithJob(h);
    h.d.takeBackRequested.mockReturnValue(true);
    h.intervals(R.TAKEBACK_CHECK_MS)[0].fn();
    expect(h.d.clearTakeBack).toHaveBeenCalled();
    await flush();
    await flush();
    expect(h.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'taken_back', detail: 'the owner took the PC back' });
    expect(h.logs.some((l) => /the PC is yours again/.test(l))).toBe(true);
    h.d.takeBackRequested.mockReturnValue(false);
    h.intervals(R.TAKEBACK_CHECK_MS)[0].fn();
    await h.runner.tick();
    expect(h.d.api.poll.mock.calls.pop()[0].cards[0].free).toBe(false);
    h.advance(R.HOLD_AFTER_TAKEBACK_MS);
    await h.runner.tick();
    expect(h.d.api.poll.mock.calls.pop()[0].cards[0].free).toBe(true);
    // With no job it only starts the hour.
    expect(await h.runner.takeBack()).toBe(false);
    expect(h.logs.pop()).toMatch(/no container jobs for the next hour/);
  });

  test('stopping the miner ends a job at once, with no grace period', async () => {
    const h = make();
    await startWithJob(h);
    expect(await h.runner.stop()).toBe(true);
    expect(h.d.docker.run.mock.calls.slice(3).map((c) => c[0][0])).toEqual(['rm']);
    expect(h.d.api.end.mock.calls[0][1]).toMatchObject({ reason: 'taken_back', detail: 'the owner stopped the miner' });
    // Stopped: no new poll is scheduled.
    expect(h.timers(NEXT_POLL_AFTER_JOB_MS)).toHaveLength(0);
    expect(h.intervals(R.TAKEBACK_CHECK_MS)).toHaveLength(0);
    // Stopping again is harmless.
    expect(await h.runner.stop()).toBe(false);
  });

  test('reassert pauses the rented card again, and only while a job runs', async () => {
    const h = make();
    h.runner.reassert();
    expect(h.d.pauseCard).not.toHaveBeenCalled();
    await startWithJob(h);
    h.runner.reassert();
    expect(h.d.pauseCard).toHaveBeenCalledTimes(2);
  });

  test('a failed end report is logged', async () => {
    const h = make();
    h.d.api.end.mockRejectedValueOnce(new Error('HTTP 502'));
    await startWithJob(h, Object.assign({}, OFFER, { image: 'bad' }));
    await flush();
    expect(h.logs.pop()).toMatch(/could not report the end of the job: HTTP 502/);
  });

  test('ending while the signature is checked, the image pulls, or the LLM is waited on', async () => {
    // During the signature check.
    const sig = deferred();
    const a = make({ cosign: { available: jest.fn(() => sig.promise) } });
    await startWithJob(a);
    a.runner.takeBack();
    sig.resolve(false);
    await flush();
    expect(a.d.docker.run).not.toHaveBeenCalled();

    // During the pull.
    const pull = deferred();
    const b = make();
    b.d.docker.run.mockReturnValueOnce(pull.promise);
    await startWithJob(b);
    b.runner.takeBack();
    pull.resolve({ code: 0 });
    await flush();
    expect(b.d.docker.run).toHaveBeenCalledTimes(1);

    // While waiting on the LLM.
    const c = make();
    c.d.llmBusy.mockReturnValueOnce(false).mockReturnValue(true);
    c.d.sleep.mockImplementationOnce(async () => { c.runner.takeBack(); });
    await startWithJob(c);
    await flush();
    expect(c.d.api.end.mock.calls[0][1].reason).toBe('taken_back');
    expect(c.d.docker.run).toHaveBeenCalledTimes(1);
  });

  test('ending while the container starts removes it once it exists', async () => {
    const run = deferred();
    const h = make();
    h.d.docker.run
      .mockResolvedValueOnce({ code: 0 })
      .mockResolvedValueOnce({ code: 0 })
      .mockReturnValueOnce(run.promise);
    await startWithJob(h);
    h.runner.takeBack();
    run.resolve({ code: 0 });
    await flush();
    expect(h.d.docker.run.mock.calls.pop()[0]).toEqual(['rm', '-f', 'llmjob-run-cr_1']);
    expect(h.d.docker.logs).not.toHaveBeenCalled();

    // And one that failed to start while ending needs nothing more.
    const run2 = deferred();
    const h2 = make();
    h2.d.docker.run
      .mockResolvedValueOnce({ code: 0 })
      .mockResolvedValueOnce({ code: 0 })
      .mockReturnValueOnce(run2.promise);
    await startWithJob(h2);
    h2.runner.takeBack();
    run2.resolve({ code: 1 });
    await flush();
    expect(h2.d.docker.run).toHaveBeenCalledTimes(3);
    expect(h2.d.api.end.mock.calls[0][1].reason).toBe('taken_back');
  });

  test('timers that fire after the end do nothing', async () => {
    const h = make();
    await startWithJob(h);
    const hb = h.intervals(R.HEARTBEAT_MS)[0];
    const temp = h.intervals(R.TEMP_CHECK_MS)[0];
    await h.runner.stop();
    const beats = h.d.api.heartbeat.mock.calls.length;
    await hb.fn();
    await temp.fn();
    expect(h.d.api.heartbeat).toHaveBeenCalledTimes(beats);
    expect(h.runner.status()).toBeNull();
    // A tick during a job does nothing either.
    const h2 = make();
    await startWithJob(h2);
    await h2.runner.tick();
    expect(h2.d.api.poll).toHaveBeenCalledTimes(1);
    expect(h2.runner.status()).not.toBeNull();
    // A job that is ending reads as no job.
    h2.runner.job.ending = true;
    expect(h2.runner.status()).toBeNull();
  });

  test('the starting label shows while the image pulls', async () => {
    const pull = deferred();
    const h = make();
    h.d.docker.run.mockReturnValueOnce(pull.promise);
    await startWithJob(h);
    expect(h.runner.label()).toBe('Rented: starting a container');
    pull.resolve({ code: 0 });
  });
  test('a second end while the first is under way waits on it, and the unsigned warning is said once per run', async () => {
    const h = make();
    await startWithJob(h);
    const stopD = deferred();
    h.d.docker.run.mockReturnValueOnce(stopD.promise);
    const first = h.runner._finish(h.runner.job, 'stopped');
    const again = h.runner.stop();
    stopD.resolve({ code: 0 });
    await first;
    expect(await again).toBe(true);
    expect(h.d.api.end).toHaveBeenCalledTimes(1);
    // A second job on the same runner does not warn again.
    h.runner.stopped = false;
    h.d.api.poll.mockResolvedValueOnce({ job: Object.assign({}, OFFER, { runId: 'cr_2' }) });
    await h.runner.tick();
    await flush();
    expect(h.logs.filter((l) => /signatures are not checked/.test(l))).toHaveLength(1);
  });

});
