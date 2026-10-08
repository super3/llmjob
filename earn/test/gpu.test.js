'use strict';

const {
  pickGpu, countGpus, alignCudaDeviceOrder, clearCudaVisibleDevices, describeClearedCuda,
  parseGpuIndexList, parseGpuIndexEnv, planMinerGpus, parseGpuStats, parsePciBusIds, parseMacGpu,
} = require('../src/shared/gpu');

describe('pickGpu', () => {
  test('picks the real GPU and skips the basic display adapter', () => {
    expect(pickGpu(['Microsoft Basic Display Adapter', 'NVIDIA GeForce RTX 4090'])).toBe('NVIDIA GeForce RTX 4090');
  });

  test('returns the first real adapter when several discrete cards are present', () => {
    expect(pickGpu(['NVIDIA GeForce RTX 4090', 'AMD Radeon RX 7900'])).toBe('NVIDIA GeForce RTX 4090');
  });

  test('prefers a discrete GPU over an integrated one listed first', () => {
    // The reported field case: an AMD APU enumerates before the RTX 4090.
    expect(pickGpu(['AMD Radeon(TM) Graphics', 'NVIDIA GeForce RTX 4090'])).toBe('NVIDIA GeForce RTX 4090');
    expect(pickGpu(['Intel(R) UHD Graphics 630', 'NVIDIA GeForce RTX 3080'])).toBe('NVIDIA GeForce RTX 3080');
    expect(pickGpu(['AMD Radeon(TM) Vega 8 Graphics', 'AMD Radeon RX 6800 XT'])).toBe('AMD Radeon RX 6800 XT');
  });

  test('falls back to an integrated GPU when it is the only real adapter', () => {
    expect(pickGpu(['Microsoft Basic Display Adapter', 'AMD Radeon(TM) Graphics'])).toBe('AMD Radeon(TM) Graphics');
    expect(pickGpu(['Intel(R) Iris(R) Xe Graphics'])).toBe('Intel(R) Iris(R) Xe Graphics');
  });

  test('trims whitespace and ignores blank/nullish entries', () => {
    expect(pickGpu([null, '', undefined, '   ', '  NVIDIA GeForce RTX 4090  '])).toBe('NVIDIA GeForce RTX 4090');
  });

  test('returns null when only virtual adapters are present', () => {
    expect(pickGpu(['Microsoft Basic Display Adapter', 'VMware SVGA 3D'])).toBeNull();
  });

  test('returns null for non-arrays', () => {
    expect(pickGpu(null)).toBeNull();
    expect(pickGpu(undefined)).toBeNull();
    expect(pickGpu('NVIDIA')).toBeNull();
  });
});

describe('countGpus', () => {
  test('counts discrete GPUs, ignoring an iGPU riding alongside', () => {
    expect(countGpus(['Intel UHD Graphics 770', 'NVIDIA GeForce RTX 3070', 'NVIDIA GeForce RTX 3070'])).toBe(2);
    expect(countGpus(Array(8).fill('NVIDIA GeForce RTX 3070'))).toBe(8);
  });

  test('an integrated-only machine counts as one miner', () => {
    expect(countGpus(['Intel UHD Graphics 770'])).toBe(1);
  });

  test('virtual adapters and junk count as zero', () => {
    expect(countGpus(['Microsoft Basic Display Adapter', ''])).toBe(0);
    expect(countGpus([])).toBe(0);
    expect(countGpus(null)).toBe(0);
    expect(countGpus([null])).toBe(0);
  });
});

// What HiveOS's h-stats.sh turns into bus_numbers, so each card's hashrate lands
// on its own dashboard row.
describe('parsePciBusIds', () => {
  test('maps each card index to its PCI bus id', () => {
    const out = '0, 00000000:01:00.0\n1, 00000000:0A:00.0\n';
    expect(parsePciBusIds(out)).toEqual({ 0: '00000000:01:00.0', 1: '00000000:0A:00.0' });
  });

  test('accepts a 4-digit domain and skips rows that do not parse', () => {
    const out = '\n0, 0000:81:00.0\nNo devices were found\n1, [N/A]\nx, 00000000:02:00.0\n';
    expect(parsePciBusIds(out)).toEqual({ 0: '0000:81:00.0' });
  });

  test('returns an empty map for empty or nullish input', () => {
    expect(parsePciBusIds('')).toEqual({});
    expect(parsePciBusIds(null)).toEqual({});
    expect(parsePciBusIds(undefined)).toEqual({});
  });
});

describe('parseGpuStats', () => {
  test('parses one entry per card from the nvidia-smi CSV', () => {
    const out = '0, NVIDIA GeForce RTX 4090, 4096, 24564\n1, NVIDIA GeForce RTX 4060 Ti, 2000, 16380\n';
    expect(parseGpuStats(out)).toEqual([
      { index: 0, name: 'NVIDIA GeForce RTX 4090', usedMb: 4096, totalMb: 24564 },
      { index: 1, name: 'NVIDIA GeForce RTX 4060 Ti', usedMb: 2000, totalMb: 16380 },
    ]);
  });

  test('reads the numbers positionally so a name that holds extra commas can\'t misalign them', () => {
    // used/total are always the last two fields; the name is everything between.
    expect(parseGpuStats('2, GPU, X, 100, 8192')).toEqual([
      { index: 2, name: 'GPU,X', usedMb: 100, totalMb: 8192 },
    ]);
  });

  test('a blank name field becomes null', () => {
    expect(parseGpuStats('0, , 100, 8192')).toEqual([
      { index: 0, name: null, usedMb: 100, totalMb: 8192 },
    ]);
  });

  test('skips blank lines and rows that do not parse', () => {
    expect(parseGpuStats('\n0, RTX 4090, 4096, 24564\n\ngarbage\n1, RTX 4090, notanumber, 24564\n')).toEqual([
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
    ]);
  });

  test('returns an empty list for empty or nullish input', () => {
    expect(parseGpuStats('')).toEqual([]);
    expect(parseGpuStats(null)).toEqual([]);
    expect(parseGpuStats(undefined)).toEqual([]);
  });
});

describe('parseMacGpu', () => {
  const wrap = (entries) => JSON.stringify({ SPDisplaysDataType: entries });

  test('reads the chip name off an Apple silicon Mac', () => {
    expect(parseMacGpu(wrap([{ _name: 'Apple M3 Max', sppci_model: 'Apple M3 Max', sppci_cores: '40' }])))
      .toEqual({ name: 'Apple M3 Max', count: 1 });
  });

  // The two keys have swapped roles between macOS versions, so both are read —
  // a missed rename would silently cost the label.
  test('accepts an entry carrying only the older _name key', () => {
    expect(parseMacGpu(wrap([{ _name: 'Apple M1 Pro' }]))).toEqual({ name: 'Apple M1 Pro', count: 1 });
  });

  // Same rule as Windows: an Intel Mac listing its iGPU first must still report
  // the discrete card, which is the one that would run anything.
  test('prefers the discrete card on a dual-GPU Intel Mac', () => {
    expect(parseMacGpu(wrap([
      { sppci_model: 'Intel UHD Graphics 630' },
      { sppci_model: 'AMD Radeon Pro 5500M' },
    ]))).toEqual({ name: 'AMD Radeon Pro 5500M', count: 1 });
  });

  test('returns null for anything unusable', () => {
    expect(parseMacGpu('not json')).toBeNull();
    expect(parseMacGpu('')).toBeNull();
    expect(parseMacGpu(null)).toBeNull();
    expect(parseMacGpu(undefined)).toBeNull();
    expect(parseMacGpu(wrap([]))).toBeNull();
    expect(parseMacGpu(wrap([{}]))).toBeNull();
    expect(parseMacGpu(JSON.stringify({ SPDisplaysDataType: 'nope' }))).toBeNull();
    expect(parseMacGpu(JSON.stringify({}))).toBeNull();
    expect(parseMacGpu('null')).toBeNull();
  });
});

// The one line that makes "GPU 1" mean the same card to nvidia-smi as it does to
// our mining core. Without it the CUDA runtime numbers cards by its own "fastest
// first" heuristic, which is how a rig came to show a 32 GB RTX PRO 4500 on the
// device label while an RTX 4070 did the mining (issue #226).
describe('alignCudaDeviceOrder', () => {
  test('pins CUDA to nvidia-smi ordering when nothing has set it', () => {
    const env = {};
    expect(alignCudaDeviceOrder(env)).toBe('PCI_BUS_ID');
    expect(env.CUDA_DEVICE_ORDER).toBe('PCI_BUS_ID');
  });

  // An operator who set it meant it. Overriding their choice would be the same
  // class of bug as the one this fixes: the machine doing something other than
  // what it was told.
  test('leaves an operator\'s own ordering alone', () => {
    const env = { CUDA_DEVICE_ORDER: 'FASTEST_FIRST' };
    expect(alignCudaDeviceOrder(env)).toBe('FASTEST_FIRST');
    expect(env.CUDA_DEVICE_ORDER).toBe('FASTEST_FIRST');
  });

  // Both shells call it at load, and it has to reach the REAL environment: the
  // mining core initialises CUDA inside this process and reads it from there.
  test('defaults to the real process environment', () => {
    const had = Object.prototype.hasOwnProperty.call(process.env, 'CUDA_DEVICE_ORDER');
    const before = process.env.CUDA_DEVICE_ORDER;
    delete process.env.CUDA_DEVICE_ORDER;
    try {
      expect(alignCudaDeviceOrder()).toBe('PCI_BUS_ID');
      expect(process.env.CUDA_DEVICE_ORDER).toBe('PCI_BUS_ID');
    } finally {
      if (had) process.env.CUDA_DEVICE_ORDER = before;
      else delete process.env.CUDA_DEVICE_ORDER;
    }
  });
});

// CUDA_VISIBLE_DEVICES hides cards from CUDA but not from nvidia-smi, which the
// mining fleet is planned from. A rig with it set to 0 asked the core for GPU 1,
// got "this machine has 1", and its second card never mined.
describe('clearCudaVisibleDevices', () => {
  test('removes it and returns what it was', () => {
    const env = { CUDA_VISIBLE_DEVICES: '0' };
    expect(clearCudaVisibleDevices(env)).toBe('0');
    expect('CUDA_VISIBLE_DEVICES' in env).toBe(false);
  });

  // An empty value hides every card, so it goes too.
  test('removes an empty value as well', () => {
    const env = { CUDA_VISIBLE_DEVICES: '' };
    expect(clearCudaVisibleDevices(env)).toBe('');
    expect('CUDA_VISIBLE_DEVICES' in env).toBe(false);
  });

  test('returns null when it was not set', () => {
    const env = { CUDA_DEVICE_ORDER: 'PCI_BUS_ID' };
    expect(clearCudaVisibleDevices(env)).toBeNull();
    expect(env).toEqual({ CUDA_DEVICE_ORDER: 'PCI_BUS_ID' });
  });

  // Like alignCudaDeviceOrder, it has to reach the real environment: the core
  // reads it when CUDA initialises inside this process.
  test('defaults to the real process environment', () => {
    const had = Object.prototype.hasOwnProperty.call(process.env, 'CUDA_VISIBLE_DEVICES');
    const before = process.env.CUDA_VISIBLE_DEVICES;
    process.env.CUDA_VISIBLE_DEVICES = '1';
    try {
      expect(clearCudaVisibleDevices()).toBe('1');
      expect(process.env.CUDA_VISIBLE_DEVICES).toBeUndefined();
    } finally {
      if (had) process.env.CUDA_VISIBLE_DEVICES = before;
      else delete process.env.CUDA_VISIBLE_DEVICES;
    }
  });
});

describe('describeClearedCuda', () => {
  test('names the value and the variable that replaces it', () => {
    expect(describeClearedCuda('0')).toBe('ignoring CUDA_VISIBLE_DEVICES=0 so every GPU can mine '
      + '(set PEARL_GPU_INDEX to mine on one card)');
  });

  // The CLI has a flag for choosing cards, so it names that instead.
  test('takes the hint the shell passes', () => {
    expect(describeClearedCuda('GPU-1a2b', 'use --gpu-index to choose the cards'))
      .toBe('ignoring CUDA_VISIBLE_DEVICES=GPU-1a2b so every GPU can mine (use --gpu-index to choose the cards)');
  });

  test('says nothing when nothing was removed', () => {
    expect(describeClearedCuda(null)).toBeNull();
  });
});

// The one way to say which cards mine: the CLI's --gpu-index and
// PEARL_GPU_INDEX both read it.
describe('parseGpuIndexList', () => {
  test('takes one card or a comma list, sorted, duplicates dropped', () => {
    expect(parseGpuIndexList('0')).toEqual({ indices: [0] });
    expect(parseGpuIndexList('0,2')).toEqual({ indices: [0, 2] });
    expect(parseGpuIndexList('2,0,2,0')).toEqual({ indices: [0, 2] });
    expect(parseGpuIndexList(' 3 , 1 ')).toEqual({ indices: [1, 3] });
    expect(parseGpuIndexList(12)).toEqual({ indices: [12] });
  });

  test('"none" is an empty list', () => {
    expect(parseGpuIndexList('none')).toEqual({ indices: [] });
    expect(parseGpuIndexList(' NONE ')).toEqual({ indices: [] });
  });

  test('blank is no choice at all', () => {
    expect(parseGpuIndexList(undefined)).toBeNull();
    expect(parseGpuIndexList(null)).toBeNull();
    expect(parseGpuIndexList('')).toBeNull();
    expect(parseGpuIndexList('   ')).toBeNull();
  });

  // A typo must not turn into an instruction, and -1 must not become a card:
  // a negative index is how the core is told "choose for me".
  test('anything that is not a list of whole numbers is an error', () => {
    for (const bad of ['first', '1.5', '-1', '0,', ',0', '0,,2', '0 2', '0;2', '0x1', '1e3',
      'GPU-1a2b3c4d', 'MIG-1a2b', '99999999999999999999']) {
      expect(parseGpuIndexList(bad)).toEqual({ error: expect.stringContaining('such as 0,2, or none') });
    }
  });
});

// PEARL_GPU_INDEX. The same list, but not "none": the desktop app has no way
// to mine on nothing, and an empty list would read as "let the core choose".
describe('parseGpuIndexEnv', () => {
  test('takes a card or a list', () => {
    expect(parseGpuIndexEnv({ PEARL_GPU_INDEX: '0' })).toEqual([0]);
    expect(parseGpuIndexEnv({ PEARL_GPU_INDEX: ' 2 ' })).toEqual([2]);
    expect(parseGpuIndexEnv({ PEARL_GPU_INDEX: '3,1' })).toEqual([1, 3]);
  });

  test('ignores anything else', () => {
    expect(parseGpuIndexEnv({})).toBeNull();
    for (const v of ['', '  ', 'none', 'first', '1.5', '-1', '0,x']) {
      expect(parseGpuIndexEnv({ PEARL_GPU_INDEX: v })).toBeNull();
    }
  });

  test('defaults to the process environment', () => {
    const had = Object.prototype.hasOwnProperty.call(process.env, 'PEARL_GPU_INDEX');
    const before = process.env.PEARL_GPU_INDEX;
    process.env.PEARL_GPU_INDEX = '1';
    try {
      expect(parseGpuIndexEnv()).toEqual([1]);
    } finally {
      if (had) process.env.PEARL_GPU_INDEX = before;
      else delete process.env.PEARL_GPU_INDEX;
    }
  });
});

// Which cards mine. One core per card, so this list is the mining fleet.
describe('planMinerGpus', () => {
  const CARDS = [
    { index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell', usedMb: 4360, totalMb: 32623 },
    { index: 1, name: 'NVIDIA GeForce RTX 4070', usedMb: 6694, totalMb: 12282 },
  ];
  const THREE = [
    { index: 0, name: 'RTX 4090 A' }, { index: 1, name: 'RTX 4090 B' }, { index: 2, name: 'RTX 4090 C' },
  ];

  test('mines on every card', () => {
    expect(planMinerGpus(CARDS, null)).toEqual([
      { index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell' },
      { index: 1, name: 'NVIDIA GeForce RTX 4070' },
    ]);
  });

  // Index order, not nvidia-smi's print order, because the salt slice each card
  // gets is its position in this list -- a list that reordered between runs
  // would move cards onto each other's slice mid-rig.
  test('is in index order', () => {
    expect(planMinerGpus([CARDS[1], CARDS[0]], null).map((g) => g.index)).toEqual([0, 1]);
  });

  test('narrows to the cards chosen', () => {
    expect(planMinerGpus(CARDS, [1])).toEqual([{ index: 1, name: 'NVIDIA GeForce RTX 4070' }]);
    expect(planMinerGpus(THREE, [0, 2])).toEqual([
      { index: 0, name: 'RTX 4090 A' }, { index: 2, name: 'RTX 4090 C' },
    ]);
  });

  // HiveOS listed a card nvidia-smi doesn't have. Mine on the rest; the CLI
  // logs the one it dropped.
  test('drops a chosen card nvidia-smi does not list', () => {
    expect(planMinerGpus(THREE, [1, 5])).toEqual([{ index: 1, name: 'RTX 4090 B' }]);
  });

  // When none of them is listed, or nvidia-smi listed nothing, they go on to the
  // core: it checks them against the real device count and says so, which
  // beats mining on cards nobody chose.
  test('passes the choice on when it cannot name any of it', () => {
    expect(planMinerGpus(CARDS, [7])).toEqual([{ index: 7, name: null }]);
    expect(planMinerGpus(CARDS, [9, 7])).toEqual([{ index: 7, name: null }, { index: 9, name: null }]);
    expect(planMinerGpus([], [0])).toEqual([{ index: 0, name: null }]);
    expect(planMinerGpus(null, [0, 2])).toEqual([{ index: 0, name: null }, { index: 2, name: null }]);
    expect(planMinerGpus([{ index: 0 }], [0])).toEqual([{ index: 0, name: null }]);
  });

  // Nothing from nvidia-smi: the caller starts one core and lets it choose.
  test('is empty when there are no cards to list', () => {
    expect(planMinerGpus([], null)).toEqual([]);
    expect(planMinerGpus(null, null)).toEqual([]);
    expect(planMinerGpus(undefined, null)).toEqual([]);
  });

  test('skips entries with no usable index', () => {
    expect(planMinerGpus([null, {}, { index: -1 }, { index: 2 }], null))
      .toEqual([{ index: 2, name: null }]);
  });
});
