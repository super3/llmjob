'use strict';

// sea-entry.js is what the packaged binary runs, on HiveOS and under systemd.
// It runs on require (it's a bootstrap), so each case loads it in isolation,
// with the real earn-cli behind it: the exit it must get is earn-cli's main().
// Timers are fake so the forced exit can be seen without waiting 5 s, and
// process.exit is stubbed so it doesn't end the test run.
describe('sea-entry', () => {
  let savedExitCode;
  let savedArgv;

  beforeEach(() => {
    savedExitCode = process.exitCode;
    savedArgv = process.argv;
    // setImmediate stays real: it is how a case waits for the run to settle.
    jest.useFakeTimers({ doNotFake: ['setImmediate'] });
  });
  afterEach(() => {
    jest.useRealTimers();
    jest.restoreAllMocks();
    process.exitCode = savedExitCode;
    process.argv = savedArgv;
  });

  // Run the binary with `args` after [execPath, invokedPath]. `cliArgs`, when
  // given, replaces the argument parser, which is the first thing run() calls.
  const load = async (args, cliArgs) => {
    const out = [];
    const errs = [];
    jest.spyOn(process.stdout, 'write').mockImplementation((s) => { out.push(String(s)); return true; });
    jest.spyOn(process.stderr, 'write').mockImplementation((s) => { errs.push(String(s)); return true; });
    const exit = jest.spyOn(process, 'exit').mockImplementation(() => {});
    process.exitCode = undefined;
    process.argv = ['/opt/llmjob/llmjob-earn-cli-linux', '/opt/llmjob/llmjob-earn-cli-linux'].concat(args);
    await jest.isolateModulesAsync(async () => {
      if (cliArgs) jest.doMock('../src/shared/cliArgs', () => cliArgs);
      require('../src/cli/sea-entry');
      await new Promise((r) => setImmediate(r)); // let the run settle
    });
    return { out: out.join(''), errs: errs.join(''), exit };
  };

  it('passes the user args, after the binary path, to the CLI', async () => {
    const { out } = await load(['--version']);
    expect(out).toBe(require('../package.json').version + '\n');
    expect(process.exitCode).toBe(0);
  });

  // The bug: a run that failed set exitCode and nothing more. A llama-server
  // child that outlived its stop kept the process alive, systemd saw it as
  // active, and Restart=on-failure never fired.
  it('forces the exit 5 s after a failed run, in case something holds the process open', async () => {
    const { errs, exit } = await load(['--no-such-flag']);
    expect(errs).toContain('unknown option: --no-such-flag');
    expect(process.exitCode).toBe(1);
    jest.advanceTimersByTime(4999);
    expect(exit).not.toHaveBeenCalled();
    jest.advanceTimersByTime(1);
    expect(exit).toHaveBeenCalledWith(1);
  });

  it('forces the exit after a clean run too', async () => {
    const { exit } = await load(['--version']);
    jest.advanceTimersByTime(5000);
    expect(exit).toHaveBeenCalledWith(0);
  });

  it('reports an Error thrown by the run and exits 1', async () => {
    const { errs, exit } = await load([], { parseCliArgs: () => { throw new Error('boom'); } });
    expect(errs).toContain('fatal: boom');
    expect(process.exitCode).toBe(1);
    jest.advanceTimersByTime(5000);
    expect(exit).toHaveBeenCalledWith(1);
  });

  it('reports a non-Error rejection and exits 1', async () => {
    const { errs, exit } = await load([], { parseCliArgs: () => { throw 'weird failure'; } });
    expect(errs).toContain('fatal: weird failure');
    expect(process.exitCode).toBe(1);
    jest.advanceTimersByTime(5000);
    expect(exit).toHaveBeenCalledWith(1);
  });
});
