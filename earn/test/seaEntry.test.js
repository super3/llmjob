'use strict';

// sea-entry.js runs on require (it's a bootstrap), so each case loads it in
// isolation with a mocked earn-cli run() and a saved/restored process.exitCode.
describe('sea-entry', () => {
  let savedExitCode;

  beforeEach(() => { savedExitCode = process.exitCode; });
  afterEach(() => { process.exitCode = savedExitCode; });

  const load = async (runImpl) => {
    const run = jest.fn(runImpl);
    const writes = [];
    const stderr = jest.spyOn(process.stderr, 'write').mockImplementation((s) => { writes.push(s); return true; });
    process.exitCode = undefined;
    await jest.isolateModulesAsync(async () => {
      jest.doMock('../src/cli/earn-cli', () => ({ run }));
      require('../src/cli/sea-entry');
      await new Promise((r) => setImmediate(r)); // let the promise chain settle
    });
    stderr.mockRestore();
    return { run, writes };
  };

  it('calls run with the user args and sets exitCode from its result', async () => {
    const { run } = await load(() => Promise.resolve(0));
    expect(run).toHaveBeenCalledWith(process.argv.slice(2));
    expect(process.exitCode).toBe(0);
  });

  it('propagates a non-zero exit code', async () => {
    await load(() => Promise.resolve(7));
    expect(process.exitCode).toBe(7);
  });

  it('reports an Error and exits 1', async () => {
    const { writes } = await load(() => Promise.reject(new Error('boom')));
    expect(process.exitCode).toBe(1);
    expect(writes.join('')).toContain('fatal: boom');
  });

  it('reports a non-Error rejection and exits 1', async () => {
    const { writes } = await load(() => Promise.reject('weird failure'));
    expect(process.exitCode).toBe(1);
    expect(writes.join('')).toContain('fatal: weird failure');
  });
});
