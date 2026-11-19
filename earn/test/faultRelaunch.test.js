'use strict';

const { FAULT_RELAUNCH_LIMIT, FAULT_RELAUNCH_WINDOW_MS, nextRelaunches } = require('../src/shared/faultRelaunch');

describe('nextRelaunches', () => {
  const HOUR = FAULT_RELAUNCH_WINDOW_MS;

  test('allows a first start, and adds it to the list', () => {
    expect(nextRelaunches(undefined, 1000)).toEqual([1000]);
    expect(nextRelaunches([], 1000)).toEqual([1000]);
  });

  test('allows up to the limit within the hour, then says no', () => {
    let times = [];
    for (let i = 0; i < FAULT_RELAUNCH_LIMIT; i++) times = nextRelaunches(times, HOUR + i * 1000);
    expect(times).toHaveLength(FAULT_RELAUNCH_LIMIT);
    expect(nextRelaunches(times, HOUR + 10000)).toBeNull();
  });

  test('forgets starts older than the hour', () => {
    const times = [0, 1000, 2000];
    expect(nextRelaunches(times, HOUR + 1000)).toEqual([2000, HOUR + 1000]);
  });

  // A damaged file must not block every start: what isn't a past time is dropped.
  test('drops entries that are not past times', () => {
    expect(nextRelaunches(['x', null, 5000, 999999], 6000)).toEqual([5000, 6000]);
    expect(nextRelaunches({ times: 1 }, 6000)).toEqual([6000]);
  });
});
