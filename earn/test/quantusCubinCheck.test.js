'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const cc = require('../src/main/quantusCubinCheck');

// A spawnSync stand-in: each call answers with the next of `runs` ({ stdout, status, signal, error }), and
// records the cards and arguments it was asked for.
function fakeSpawn(runs) {
  const calls = [];
  const fn = jest.fn((exe, args, opts) => {
    calls.push({ exe, args, cards: opts.env[cc.CHECK_CARDS], opts });
    return Object.assign({ stdout: '', status: 0, signal: null }, runs.shift());
  });
  fn.calls = calls;
  return fn;
}
const line = (r) => JSON.stringify(r) + '\n';
const addon = (wanted = () => true) => ({ cubinWanted: jest.fn(wanted), cubinCheck: jest.fn() });

describe('the child', () => {
  test('checks each card in order, one JSON line each, and stops after a card that killed its context', () => {
    const out = [];
    const a = { cubinCheck: jest.fn((d) => ({ device: d, wanted: true, ok: d === 0, lost: d === 1, why: '' })) };
    const req = jest.fn(() => a);
    cc.runChild({ env: { [cc.CHECK_FILE]: '/x/q.node', [cc.CHECK_CARDS]: '0,1,2' }, stdout: { write: (s) => out.push(s) } }, req);
    expect(req).toHaveBeenCalledWith('/x/q.node');
    expect(a.cubinCheck.mock.calls.map((c) => c[0])).toEqual([0, 1]);
    expect(cc.parseLines(out.join(''))).toEqual([
      { device: 0, wanted: true, ok: true, lost: false, why: '' },
      { device: 1, wanted: true, ok: false, lost: true, why: '' },
    ]);
  });

  test('no cards: nothing checked; it loads the addon the way a single-file binary can by default', () => {
    const a = { cubinCheck: jest.fn() };
    cc.runChild({ env: { [cc.CHECK_FILE]: 'x' }, stdout: { write: jest.fn() } }, () => a);
    expect(a.cubinCheck).not.toHaveBeenCalled();
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'qcc-'));
    const file = path.join(dir, 'fake.js');
    fs.writeFileSync(file, 'module.exports = { cubinCheck: (d) => ({ device: d, ok: true }) };');
    const out = [];
    cc.runChild({ execPath: process.execPath, env: { [cc.CHECK_FILE]: file, [cc.CHECK_CARDS]: '3' }, stdout: { write: (s) => out.push(s) } });
    expect(out).toEqual([line({ device: 3, ok: true })]);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test('only result lines count', () => {
    expect(cc.parseLines('noise\n' + line({ device: 1, ok: true }) + 'null\n"x"\n{"device":"0"}\n')).toEqual([{ device: 1, ok: true }]);
    expect(cc.parseLines(undefined)).toEqual([]);
  });

  test('whether this is a single-file binary', () => {
    expect(cc.seaBinary()).toBe(false);
    jest.isolateModules(() => {
      jest.doMock('node:sea', () => { throw new Error('no such module'); }, { virtual: true });
      expect(require('../src/main/quantusCubinCheck').seaBinary()).toBe(false);
    });
  });
});

describe('the gate', () => {
  test('an addon without the calls, or no file: nothing to check', () => {
    const spawnSync = fakeSpawn([]);
    const none = cc.cubinGate({}, '/x/q.node', { spawnSync });
    none.prepare([0]);
    expect(none.options(0)).toEqual({});
    expect(cc.cubinGate(addon(), null, { spawnSync }).options(0)).toEqual({});
    expect(spawnSync).not.toHaveBeenCalled();
  });

  test('a card with no patched kernel, or one whose question throws: no child, and createCore as before', () => {
    const spawnSync = fakeSpawn([]);
    const g = cc.cubinGate(addon((i) => { if (i) throw new Error('boom'); return false; }), '/x/q.node', { spawnSync });
    g.prepare([0, 1]);
    g.prepare('not a list');
    expect(g.options(0)).toEqual({});
    expect(g.options(1)).toEqual({});
    expect(spawnSync).not.toHaveBeenCalled();
  });

  test('prepare checks every card in one child; createCore then passes each answer, and nothing is checked twice', () => {
    const spawnSync = fakeSpawn([{
      stdout: line({ device: 0, ok: true }) + line({ device: 2, ok: false, why: 'its hits differ from the compiled kernel\'s' }),
      stderr: 'quantus_cuda: card 0 (sm_89) runs the patched kernel (x)\n\n',
    }]);
    const log = jest.fn();
    const g = cc.cubinGate(addon(), '/x/q.node', { spawnSync, log, isSea: () => false, execPath: '/app/earn', env: { A: '1' } });
    g.prepare([0, 2]);
    g.prepare([0, 2]);
    expect(spawnSync).toHaveBeenCalledTimes(1);
    const { exe, args, cards, opts } = spawnSync.calls[0];
    expect(exe).toBe('/app/earn');
    expect(args[0]).toBe('-e');
    expect(args[1]).toContain('runChild(process)');
    expect(cards).toBe('0,2');
    expect(opts.env).toMatchObject({ A: '1', ELECTRON_RUN_AS_NODE: '1', [cc.CHECK_FILE]: '/x/q.node' });
    expect(opts).toMatchObject({ timeout: cc.CHILD_MS, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    expect(log).toHaveBeenCalledWith('info', 'Quantus core: check process: quantus_cuda: card 0 (sm_89) runs the patched kernel (x)');
    expect(g.options(0)).toEqual({ cubin: true });
    expect(g.options(2)).toEqual({ cubin: false });
    expect(log).toHaveBeenCalledWith('info', 'Quantus core: card 0: the patched kernel passed its check in a separate process');
    expect(log).toHaveBeenCalledWith('warn', 'Quantus core: card 2: the patched kernel is not used (its hits differ from the '
      + 'compiled kernel\'s); the compiled kernel runs');
  });

  test('a card prepare did not see is checked on its own when it opens; a single-file binary is run with no -e', () => {
    const spawnSync = fakeSpawn([{ stdout: line({ device: 1, ok: true }) }]);
    const g = cc.cubinGate(addon(), '/x/q.node', { spawnSync, isSea: () => true });
    expect(g.options(1)).toEqual({ cubin: true });
    expect(spawnSync.calls[0].args).toEqual([]);
    expect(spawnSync.calls[0].cards).toBe('1');
  });

  test('a card that killed the child\'s context: the cards after it get a new child', () => {
    const spawnSync = fakeSpawn([
      { stdout: line({ device: 0, ok: true }) + line({ device: 1, ok: false, lost: true }) },
      { stdout: line({ device: 2, ok: true }) + line({ device: 9, ok: true }) + line({ device: 0, ok: false }) },
    ]);
    const log = jest.fn();
    const g = cc.cubinGate(addon(), '/x/q.node', { spawnSync, log, isSea: () => false });
    g.prepare([0, 1, 2]);
    expect(spawnSync.calls.map((c) => c.cards)).toEqual(['0,1,2', '2']);
    expect([0, 1, 2].map((i) => g.options(i))).toEqual([{ cubin: true }, { cubin: false }, { cubin: true }]);
    expect(log).toHaveBeenCalledWith('warn', 'Quantus core: card 1: the patched kernel is not used (it faulted); the compiled kernel runs');
  });

  test('a child that ends without an answer: that card runs the compiled kernel, and the rest are checked again', () => {
    const spawnSync = fakeSpawn([
      { stdout: line({ device: 0, ok: false }), status: null, signal: 'SIGSEGV' },
      { status: 3 },
      { error: new Error('spawnSync /app/earn ETIMEDOUT'), status: null },
    ]);
    const log = jest.fn();
    const g = cc.cubinGate(addon(), '/x/q.node', { spawnSync, log, isSea: () => false });
    g.prepare([0, 1, 2, 3]);
    expect(spawnSync.calls.map((c) => c.cards)).toEqual(['0,1,2,3', '2,3', '3']);
    expect([0, 1, 2, 3].map((i) => g.options(i).cubin)).toEqual([false, false, false, false]);
    expect(log).toHaveBeenCalledWith('warn', 'Quantus core: card 0: the patched kernel is not used (its check failed); the compiled kernel runs');
    expect(log).toHaveBeenCalledWith('warn', expect.stringContaining('card 1: the patched kernel is not used (the check process ended (signal SIGSEGV))'));
    expect(log).toHaveBeenCalledWith('warn', expect.stringContaining('card 2: the patched kernel is not used (the check process ended (exit code 3))'));
    expect(log).toHaveBeenCalledWith('warn', expect.stringContaining('card 3: the patched kernel is not used (the check process ended (spawnSync /app/earn ETIMEDOUT))'));
  });

  test('a real child: this node, -e, and an addon that faults on its second card', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'qcc-'));
    const file = path.join(dir, 'fake.js');
    fs.writeFileSync(file, [
      'module.exports = {',
      '  cubinWanted: () => true,',
      '  cubinCheck: (d) => {',
      '    if (process.env.ELECTRON_RUN_AS_NODE !== "1") throw new Error("not run as node");',
      '    if (d === 1) process.exit(7);',
      '    return { device: d, wanted: true, ok: d !== 2, lost: false, why: d === 2 ? "its hits differ" : "" };',
      '  },',
      '};',
    ].join('\n'));
    const log = jest.fn();
    const g = cc.cubinGate(require(file), file, { log });
    g.prepare([0, 1, 2]);
    expect([0, 1, 2].map((i) => g.options(i))).toEqual([{ cubin: true }, { cubin: false }, { cubin: false }]);
    expect(log).toHaveBeenCalledWith('warn', expect.stringContaining('card 1: the patched kernel is not used (the check process ended (exit code 7))'));
    expect(log).toHaveBeenCalledWith('warn', expect.stringContaining('card 2: the patched kernel is not used (its hits differ)'));
    fs.rmSync(dir, { recursive: true, force: true });
  });
});
