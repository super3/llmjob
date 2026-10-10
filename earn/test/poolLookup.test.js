'use strict';

const path = require('path');
const {
  makePoolLookup, fileCache, usableAddress, PUBLIC_DNS, QUERY_TIMEOUT_MS,
} = require('../src/shared/poolLookup');

const HOST = 'us2.pearl.herominers.com';
const ENOENT = Object.assign(new Error('getaddrinfo ENOENT ' + HOST), { code: 'ENOENT' });

// A dns module double. `system` answers dns.lookup; `direct` and `public` answer
// the resolvers (a list of IPs, or an Error to reject with).
function fakeDns({ system, direct = new Error('ETIMEOUT'), publicDns = new Error('ETIMEOUT') }) {
  const resolvers = [];
  const dns = {
    lookup: jest.fn((host, opts, cb) => (system instanceof Error ? cb(system) : cb(null, system))),
    promises: {
      Resolver: jest.fn(function Resolver(opts) {
        const r = {
          opts,
          servers: null,
          setServers: jest.fn((s) => { r.servers = s; }),
          resolve4: jest.fn(() => {
            const result = r.servers ? publicDns : direct;
            return result instanceof Error ? Promise.reject(result) : Promise.resolve(result);
          }),
        };
        resolvers.push(r);
        return r;
      }),
    },
  };
  return { dns, resolvers };
}

function memCache(initial = {}) {
  const entries = Object.assign({}, initial);
  return { get: jest.fn((h) => entries[h] || null), set: jest.fn((h, a) => { entries[h] = a; }), entries };
}

// Call the lookup the way net.connect does, and collect what it answers.
function ask(lookup, opts) {
  return new Promise((resolve) => {
    const cb = (err, address, family) => resolve({ err, address, family });
    if (opts === undefined) lookup(HOST, cb);
    else lookup(HOST, opts, cb);
  });
}

describe('makePoolLookup', () => {
  test('uses the system lookup when it works, and remembers the answer', async () => {
    const addrs = [{ address: '51.81.1.1', family: 4 }];
    const { dns } = fakeDns({ system: addrs });
    const cache = memCache();
    const log = jest.fn();
    const lookup = makePoolLookup({ dns, cache, log });
    expect(await ask(lookup, { family: 0 })).toEqual({ err: null, address: '51.81.1.1', family: 4 });
    // It always asks the system for every address, keeping net's own options.
    expect(dns.lookup).toHaveBeenCalledWith(HOST, { family: 0, all: true }, expect.any(Function));
    expect(cache.set).toHaveBeenCalledWith(HOST, addrs);
    expect(dns.promises.Resolver).not.toHaveBeenCalled();
    expect(log).not.toHaveBeenCalled();
  });

  // Node asks for every address when it races IPv4 and IPv6.
  test('answers with the whole list when net asks for all of them', async () => {
    const addrs = [{ address: '51.81.1.1', family: 4 }, { address: '2001:db8::1', family: 6 }];
    const { dns } = fakeDns({ system: addrs });
    const lookup = makePoolLookup({ dns, cache: memCache() });
    expect(await ask(lookup, { all: true })).toEqual({ err: null, address: addrs, family: undefined });
  });

  test('takes the callback in place of options, and no options at all', async () => {
    const { dns } = fakeDns({ system: [{ address: '51.81.1.1', family: 4 }] });
    const lookup = makePoolLookup({ dns, cache: memCache() });
    expect((await ask(lookup)).address).toBe('51.81.1.1');
    expect((await ask(lookup, null)).address).toBe('51.81.1.1');
  });

  test('when the system lookup fails, asks the DNS servers directly, briefly', async () => {
    const { dns, resolvers } = fakeDns({ system: ENOENT, direct: ['51.81.2.2'] });
    const cache = memCache();
    const log = jest.fn();
    const lookup = makePoolLookup({ dns, cache, log });
    expect(await ask(lookup, {})).toEqual({ err: null, address: '51.81.2.2', family: 4 });
    expect(resolvers[0].opts).toEqual({ timeout: QUERY_TIMEOUT_MS, tries: 1 });
    expect(resolvers[0].setServers).not.toHaveBeenCalled();
    expect(cache.set).toHaveBeenCalledWith(HOST, [{ address: '51.81.2.2', family: 4 }]);
    expect(log).toHaveBeenCalledWith('could not look up ' + HOST + ' (ENOENT); using 51.81.2.2 from DNS');
  });

  test('then public DNS servers', async () => {
    const { dns, resolvers } = fakeDns({ system: ENOENT, direct: [], publicDns: ['51.81.3.3'] });
    const log = jest.fn();
    const lookup = makePoolLookup({ dns, cache: memCache(), log });
    expect((await ask(lookup, {})).address).toBe('51.81.3.3');
    expect(resolvers[1].servers).toEqual(PUBLIC_DNS);
    expect(log).toHaveBeenCalledWith('could not look up ' + HOST + ' (ENOENT); using 51.81.3.3 from public DNS');
  });

  // The case the user's log needed most: nothing answers at all, including the
  // first minutes after a restart.
  test('then the last address that worked', async () => {
    const saved = [{ address: '51.81.4.4', family: 4 }];
    const { dns } = fakeDns({ system: ENOENT });
    const cache = memCache({ [HOST]: saved });
    const log = jest.fn();
    const lookup = makePoolLookup({ dns, cache, log });
    expect(await ask(lookup, {})).toEqual({ err: null, address: '51.81.4.4', family: 4 });
    expect(cache.set).not.toHaveBeenCalled();
    expect(log).toHaveBeenCalledWith('could not look up ' + HOST + ' (ENOENT); using 51.81.4.4 from the last address that worked');
  });

  test('with nothing to fall back on, fails with the system error as before', async () => {
    const { dns } = fakeDns({ system: ENOENT });
    const log = jest.fn();
    const lookup = makePoolLookup({ dns, cache: memCache(), log });
    expect((await ask(lookup, {})).err).toBe(ENOENT);
    expect(log).not.toHaveBeenCalled();
  });

  test('names the failure by its message when it has no code', async () => {
    const { dns } = fakeDns({ system: new Error('lookup broke'), direct: ['51.81.2.2'] });
    const log = jest.fn();
    const lookup = makePoolLookup({ dns, cache: memCache(), log });
    await ask(lookup, {});
    expect(log).toHaveBeenCalledWith('could not look up ' + HOST + ' (lookup broke); using 51.81.2.2 from DNS');
  });

  test('works without a log', async () => {
    const { dns } = fakeDns({ system: ENOENT, direct: ['51.81.2.2'] });
    const lookup = makePoolLookup({ dns, cache: memCache() });
    expect((await ask(lookup, {})).address).toBe('51.81.2.2');
  });
});

// A DNS filter (the user's router forwarded to Cloudflare's 1.1.1.2, which
// blocks crypto-mining names) often answers a blocked name with 0.0.0.0 rather
// than failing. That answer leads nowhere, so it counts as no answer at every
// step and is never saved over the last address that worked.
describe('blocked answers', () => {
  const BLOCKED_REASON = 'DNS answered 0.0.0.0, the address a DNS filter gives a name it blocks';

  test('only real addresses are usable', () => {
    for (const a of ['0.0.0.0', '0.1.2.3', '::', '0:0:0:0:0:0:0:0', '::0000', '::ffff:0.0.0.0', 'nonsense', undefined]) {
      expect([a, usableAddress(a)]).toEqual([a, false]);
    }
    for (const a of ['51.81.1.1', '2001:db8::1', '::ffff:51.81.1.1', '1::', '::2']) {
      expect([a, usableAddress(a)]).toEqual([a, true]);
    }
  });

  // A local stratum proxy reached by name (localhost:3333) must keep working.
  test('loopback is a real answer, not a blocked one', () => {
    for (const a of ['127.0.0.1', '::1', '::ffff:127.0.0.1']) {
      expect([a, usableAddress(a)]).toEqual([a, true]);
    }
  });

  test('a system answer of 0.0.0.0 falls through, past a filtered DNS server too, to public DNS', async () => {
    const { dns } = fakeDns({
      system: [{ address: '0.0.0.0', family: 4 }, { address: '::', family: 6 }],
      direct: ['0.0.0.0'],
      publicDns: ['51.81.3.3'],
    });
    const cache = memCache();
    const log = jest.fn();
    const lookup = makePoolLookup({ dns, cache, log });
    expect((await ask(lookup, {})).address).toBe('51.81.3.3');
    expect(cache.set.mock.calls).toEqual([[HOST, [{ address: '51.81.3.3', family: 4 }]]]);
    expect(log).toHaveBeenCalledWith('could not look up ' + HOST + ' (' + BLOCKED_REASON + '); using 51.81.3.3 from public DNS');
  });

  test('a mixed answer keeps only the real addresses, and saves only those', async () => {
    const { dns } = fakeDns({ system: [{ address: '0.0.0.0', family: 4 }, { address: '51.81.1.1', family: 4 }] });
    const cache = memCache();
    const lookup = makePoolLookup({ dns, cache });
    expect(await ask(lookup, { all: true })).toEqual({ err: null, address: [{ address: '51.81.1.1', family: 4 }], family: undefined });
    expect(cache.set).toHaveBeenCalledWith(HOST, [{ address: '51.81.1.1', family: 4 }]);
  });

  test('the last address that worked still serves when every lookup is filtered', async () => {
    const { dns } = fakeDns({ system: [{ address: '0.0.0.0', family: 4 }], direct: ['0.0.0.0'], publicDns: [] });
    const cache = memCache({ [HOST]: [{ address: '0.0.0.0', family: 4 }, { address: '51.81.4.4', family: 4 }] });
    const log = jest.fn();
    const lookup = makePoolLookup({ dns, cache, log });
    expect((await ask(lookup, {})).address).toBe('51.81.4.4');
    expect(cache.set).not.toHaveBeenCalled();
    expect(log).toHaveBeenCalledWith('could not look up ' + HOST + ' (' + BLOCKED_REASON + '); using 51.81.4.4 from the last address that worked');
  });

  // It reads like the system's own lookup errors ("getaddrinfo ..."), so the
  // app's "could not resolve ... check DNS" hint still shows.
  test('with nothing else, fails with an error that says the name was blocked', async () => {
    const { dns } = fakeDns({ system: [{ address: '0.0.0.0', family: 4 }] });
    const lookup = makePoolLookup({ dns, cache: memCache({ [HOST]: [{ address: '0.0.0.0', family: 4 }] }) });
    const { err } = await ask(lookup, {});
    expect(err.code).toBe('EBLOCKED');
    expect(err.message).toBe('getaddrinfo EBLOCKED ' + HOST + ' (' + BLOCKED_REASON + ')');
  });
});

describe('fileCache', () => {
  const FILE = path.join('/store', 'pool-addresses.json');
  function fakeFs(content) {
    return {
      readFileSync: jest.fn(() => {
        if (content instanceof Error) throw content;
        return content;
      }),
      mkdirSync: jest.fn(),
      writeFileSync: jest.fn(),
    };
  }

  test('reads saved addresses once, and keeps only well-formed ones', () => {
    const fs = fakeFs(JSON.stringify({
      [HOST]: [{ address: '51.81.1.1', family: 4 }, { family: 4 }, null],
      'empty.example': [],
      'odd.example': 'nope',
    }));
    const cache = fileCache(FILE, fs, path);
    expect(cache.get(HOST)).toEqual([{ address: '51.81.1.1', family: 4 }]);
    expect(cache.get('empty.example')).toBeNull();
    expect(cache.get('odd.example')).toBeNull();
    expect(cache.get('unknown.example')).toBeNull();
    expect(fs.readFileSync).toHaveBeenCalledTimes(1);
  });

  test('a missing or broken file is an empty cache', () => {
    expect(fileCache(FILE, fakeFs(new Error('ENOENT')), path).get(HOST)).toBeNull();
    expect(fileCache(FILE, fakeFs('null'), path).get(HOST)).toBeNull();
  });

  test('saves a new address, only the fields it needs, and skips an unchanged one', () => {
    const fs = fakeFs(new Error('ENOENT'));
    const cache = fileCache(FILE, fs, path);
    cache.set(HOST, [{ address: '51.81.1.1', family: 4, ttl: 300 }]);
    // path.dirname, not '/store': on Windows the folder is '\store'.
    expect(fs.mkdirSync).toHaveBeenCalledWith(path.dirname(FILE), { recursive: true });
    expect(JSON.parse(fs.writeFileSync.mock.calls[0][1])).toEqual({ [HOST]: [{ address: '51.81.1.1', family: 4 }] });
    cache.set(HOST, [{ address: '51.81.1.1', family: 4 }]);
    expect(fs.writeFileSync).toHaveBeenCalledTimes(1);
    expect(cache.get(HOST)).toEqual([{ address: '51.81.1.1', family: 4 }]);
  });

  test('a failed write still serves this run from memory', () => {
    const fs = fakeFs(new Error('ENOENT'));
    fs.writeFileSync.mockImplementation(() => { throw new Error('EACCES'); });
    const cache = fileCache(FILE, fs, path);
    expect(() => cache.set(HOST, [{ address: '51.81.1.1', family: 4 }])).not.toThrow();
    expect(cache.get(HOST)).toEqual([{ address: '51.81.1.1', family: 4 }]);
  });
});
