'use strict';

const { createPromptPause, cardForPrompt } = require('../src/shared/promptPause');

function make(over = {}) {
  const calls = [];
  const logs = [];
  const p = createPromptPause(Object.assign({
    pause: (i) => { calls.push(['pause', i]); return true; },
    resume: (i) => { calls.push(['resume', i]); return true; },
    log: (level, line) => logs.push([level, line]),
    resumeAfterMs: 10000,
    maxPauseMs: 600000,
  }, over));
  return { p, calls, logs };
}

describe('promptPause', () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => { jest.clearAllTimers(); jest.useRealTimers(); });

  test('pauses a card at once and resumes it a while after the prompt ends', () => {
    const { p, calls, logs } = make();
    p.setBusy(0, true);
    expect(calls).toEqual([['pause', [0]]]);
    expect(p.isPaused(0)).toBe(true);
    expect(p.isBusy(0)).toBe(true);
    p.setBusy(0, false);
    expect(p.isBusy(0)).toBe(false);
    jest.advanceTimersByTime(9999);
    expect(calls).toHaveLength(1);
    jest.advanceTimersByTime(1);
    expect(calls).toEqual([['pause', [0]], ['resume', [0]]]);
    expect(p.isPaused(0)).toBe(false);
    expect(logs).toEqual([
      ['info', 'mining paused on GPU 0 while the local LLM answers a prompt'],
      ['info', 'mining resumed on GPU 0'],
    ]);
  });

  test('a prompt that follows soon after keeps the card paused, without a stop and start', () => {
    const { p, calls } = make();
    p.setBusy(0, true);
    p.setBusy(0, false);
    jest.advanceTimersByTime(5000);
    p.setBusy(0, true);
    jest.advanceTimersByTime(20000);
    expect(calls).toEqual([['pause', [0]]]);
    p.setBusy(0, false);
    jest.advanceTimersByTime(10000);
    expect(calls).toEqual([['pause', [0]], ['resume', [0]]]);
  });

  test('each card on its own', () => {
    const { p, calls } = make();
    p.setBusy(0, true);
    p.setBusy(1, true);
    p.setBusy(0, false);
    jest.advanceTimersByTime(10000);
    expect(calls).toEqual([['pause', [0]], ['pause', [1]], ['resume', [0]]]);
    expect(p.isPaused(1)).toBe(true);
  });

  test('an instance with no known card pauses every card', () => {
    const { p, calls, logs } = make();
    p.setBusy(null, true);
    p.setBusy(undefined, false);
    jest.advanceTimersByTime(10000);
    expect(calls).toEqual([['pause', null], ['resume', null]]);
    expect(logs[0][1]).toBe('mining paused while the local LLM answers a prompt');
    expect(logs[1][1]).toBe('mining resumed');
  });

  test('an idle on a card that was never paused does nothing, and a second idle does not double the timer', () => {
    const { p, calls } = make();
    p.setBusy(2, false);
    p.setBusy(0, true);
    p.setBusy(0, false);
    p.setBusy(0, false);
    jest.advanceTimersByTime(10000);
    expect(calls).toEqual([['pause', [0]], ['resume', [0]]]);
  });

  test('a card busy without a break for maxPauseMs mines again, and says why', () => {
    const { p, calls, logs } = make();
    p.setBusy(0, true);
    jest.advanceTimersByTime(600000);
    expect(calls).toEqual([['pause', [0]], ['resume', [0]]]);
    expect(p.isBusy(0)).toBe(false);
    expect(logs.pop()).toEqual(['warn', 'mining resumed on GPU 0: the local LLM has been busy for 10 minutes without a break']);
    // The line that was missed arriving late does nothing more.
    p.setBusy(0, false);
    jest.advanceTimersByTime(10000);
    expect(calls).toHaveLength(2);
  });

  test('the safety net counts from the current stretch of work, not from the pause', () => {
    const { p, calls } = make();
    p.setBusy(0, true);
    jest.advanceTimersByTime(400000);
    p.setBusy(0, false);
    jest.advanceTimersByTime(5000);
    p.setBusy(0, true);
    jest.advanceTimersByTime(400000);
    expect(calls).toEqual([['pause', [0]]]);
  });

  test('reapply pauses on a new miner whatever is paused now', () => {
    const { p, calls } = make();
    p.setBusy(0, true);
    p.setBusy(null, true);
    p.reapply();
    expect(calls).toEqual([['pause', [0]], ['pause', null], ['pause', [0]], ['pause', null]]);
  });

  test('reset resumes everything at once and forgets the busy cards', () => {
    const { p, calls } = make();
    p.setBusy(0, true);
    p.setBusy(1, true);
    p.setBusy(1, false);
    p.reset();
    expect(calls).toEqual([['pause', [0]], ['pause', [1]], ['resume', [0]], ['resume', [1]]]);
    expect(p.isBusy(0)).toBe(false);
    jest.advanceTimersByTime(600000);
    expect(calls).toHaveLength(4);
  });

  test('stop forgets everything and touches nothing', () => {
    const { p, calls } = make();
    p.setBusy(0, true);
    p.setBusy(1, true);
    p.setBusy(1, false);
    p.stop();
    jest.advanceTimersByTime(600000);
    expect(calls).toEqual([['pause', [0]], ['pause', [1]]]);
    expect(p.isPaused(0)).toBe(false);
  });

  test('says nothing about a resume that started nothing, as when STOP took the miner away', () => {
    const { p, logs } = make({ resume: () => false });
    p.setBusy(0, true);
    p.reset();
    expect(logs).toEqual([['info', 'mining paused on GPU 0 while the local LLM answers a prompt']]);
  });

  test('a new miner announces its own pause, even when the last one already had', () => {
    const { p, logs } = make();
    p.setBusy(0, true);
    p.reapply();
    expect(logs).toEqual([
      ['info', 'mining paused on GPU 0 while the local LLM answers a prompt'],
      ['info', 'mining paused on GPU 0 while the local LLM answers a prompt'],
    ]);
  });

  test('says nothing when there was no mining to pause, but still remembers the pause', () => {
    let mining = false;
    const { p, calls, logs } = make({ pause: (i) => { calls.push(['pause', i]); return mining; } });
    p.setBusy(0, true);
    expect(logs).toEqual([]);
    expect(p.isPaused(0)).toBe(true);
    // A miner that starts during the prompt is paused, and that is said once.
    mining = true;
    p.reapply();
    expect(logs).toEqual([['info', 'mining paused on GPU 0 while the local LLM answers a prompt']]);
    p.setBusy(0, false);
    jest.advanceTimersByTime(10000);
    expect(logs.pop()).toEqual(['info', 'mining resumed on GPU 0']);
    // And a resume of a pause that stopped nothing is quiet too.
    mining = false;
    p.setBusy(1, true);
    p.setBusy(1, false);
    jest.advanceTimersByTime(10000);
    expect(logs).toHaveLength(1);
    expect(calls.pop()).toEqual(['resume', [1]]);
  });

  test('defaults: no log needed, and works with timers that cannot be unref\'d', () => {
    const timers = [];
    const p = createPromptPause({
      pause: () => true, resume: () => {}, resumeAfterMs: 1, maxPauseMs: 2,
      setTimer: (fn) => { timers.push(fn); return 1; }, clearTimer: () => {},
    });
    p.setBusy(0, true);
    p.setBusy(0, false);
    timers.forEach((fn) => fn());
    expect(p.isPaused(0)).toBe(false);
  });
});

describe('cardForPrompt', () => {
  const TWO = { 0: '00000000:01:00.0', 1: '00000000:0A:00.0' };

  test('the card the server says it loaded on, whatever it was planned for', () => {
    // Planned for card 0, loaded on card 1: Vulkan counted the cards its own way.
    expect(cardForPrompt({ index: 0, busId: '0000:0a:00.0' }, TWO)).toBe(1);
    expect(cardForPrompt({ index: 1, busId: '0000:01:00.0' }, TWO)).toBe(0);
  });

  test('a bus id nvidia-smi does not list is a card that does not mine: leave mining alone', () => {
    expect(cardForPrompt({ index: 0, busId: '0000:05:00.0' }, TWO)).toBeUndefined();
  });

  test('without a bus id, several cards are left mining rather than guessed at', () => {
    expect(cardForPrompt({ index: 0, busId: null }, TWO)).toBeUndefined();
    expect(cardForPrompt({ index: 0, busId: 'unknown id' }, TWO)).toBeUndefined();
  });

  test('without a bus id, a one-card rig has only one card it can be', () => {
    expect(cardForPrompt({ index: 0, busId: null }, { 0: '00000000:01:00.0' })).toBe(0);
    expect(cardForPrompt({ index: null, busId: null }, { 0: '00000000:01:00.0' })).toBeNull();
  });

  test('with no bus ids from nvidia-smi, the planned card (or every card) is all there is to go on', () => {
    expect(cardForPrompt({ index: 2, busId: '0000:0a:00.0' }, {})).toBe(2);
    expect(cardForPrompt({ index: undefined, busId: null }, null)).toBeNull();
    expect(cardForPrompt(null, {})).toBeNull();
  });
});
