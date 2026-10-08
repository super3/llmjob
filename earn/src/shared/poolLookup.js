'use strict';

// Finding the pool's address when the computer's own lookup fails.
//
// A user's log (Windows, RTX 5070 Ti, October 2026) shows `getaddrinfo ENOENT`
// for us2.pearl.herominers.com, retried every 5 seconds and failing for 2 to 30
// minutes at a time: 3.9 of 7.4 hours, including the first 5 minutes after every
// start. The network itself was up the whole time. The rig kept reporting to the
// LLMJob network board, and between the failures the same name resolved and
// connected within a second. Retrying the same system lookup is what kept
// failing.
//
// So the pool connection resolves through a chain, and stops at the first that
// answers:
//   1. the system lookup, exactly as before;
//   2. a direct DNS query to the computer's own DNS servers, which skips the
//      operating system's resolver and its cache;
//   3. the same query to public DNS servers;
//   4. the last address that worked for this name, saved on disk, which also
//      covers the minutes right after a restart.
// Stratum is plain TCP and never sends the name, so an address from any step
// connects to the same pool.
//
// The result is shaped for net.connect's `lookup` option: Node asks for every
// address (`all: true`) when it races IPv4 and IPv6, and for one otherwise.

const PUBLIC_DNS = ['1.1.1.1', '8.8.8.8'];
// One short try per server set: a lookup that will not answer should fail over
// in seconds, not hold the reconnect for the resolver's default ~20 s.
const QUERY_TIMEOUT_MS = 2000;

// `dns` is node's dns module (injected for tests). `cache` is { get(host),
// set(host, addresses) }, see fileCache. `log(line)` hears each fallback used.
function makePoolLookup({ dns, cache, log = () => {}, publicServers = PUBLIC_DNS }) {
  // Steps 2 and 3. A server list of null means the computer's own.
  async function queryDns(hostname) {
    const sets = [['DNS', null], ['public DNS', publicServers]];
    for (const [source, servers] of sets) {
      const resolver = new dns.promises.Resolver({ timeout: QUERY_TIMEOUT_MS, tries: 1 });
      if (servers) resolver.setServers(servers);
      try {
        const ips = await resolver.resolve4(hostname);
        if (ips.length) return { source, addresses: ips.map((address) => ({ address, family: 4 })) };
      } catch (e) {
        // On to the next.
      }
    }
    return null;
  }

  return function poolLookup(hostname, options, callback) {
    if (typeof options === 'function') {
      callback = options;
      options = {};
    }
    const opts = options || {};
    const answer = (addresses) => (opts.all
      ? callback(null, addresses)
      : callback(null, addresses[0].address, addresses[0].family));

    dns.lookup(hostname, Object.assign({}, opts, { all: true }), (err, addresses) => {
      if (!err) {
        cache.set(hostname, addresses);
        answer(addresses);
        return;
      }
      queryDns(hostname).then((found) => {
        const saved = found ? null : cache.get(hostname);
        if (!found && !saved) {
          callback(err);
          return;
        }
        const addresses = found ? found.addresses : saved;
        if (found) cache.set(hostname, addresses);
        log('could not look up ' + hostname + ' (' + (err.code || err.message) + '); using '
          + addresses[0].address + ' from ' + (found ? found.source : 'the last address that worked'));
        answer(addresses);
      });
    });
  };
}

// The last addresses that worked, per name, in one small JSON file shared by
// both shells. Read once. Written only when an address changes, so a steady
// pool does not rewrite it on every reconnect. Any read or write failure leaves
// the cache empty or unsaved: it is a fallback, never a reason to fail.
function fileCache(file, fs, path) {
  let entries = null;
  const load = () => {
    if (entries) return entries;
    try {
      entries = JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch (e) {
      entries = {};
    }
    if (!entries || typeof entries !== 'object') entries = {};
    return entries;
  };
  return {
    get(host) {
      const saved = load()[host];
      const valid = Array.isArray(saved) ? saved.filter((a) => a && typeof a.address === 'string') : [];
      return valid.length ? valid : null;
    },
    set(host, addresses) {
      const all = load();
      const next = addresses.map((a) => ({ address: a.address, family: a.family }));
      if (JSON.stringify(all[host]) === JSON.stringify(next)) return;
      all[host] = next;
      try {
        fs.mkdirSync(path.dirname(file), { recursive: true });
        fs.writeFileSync(file, JSON.stringify(all));
      } catch (e) {
        // Unsaved this time; the in-memory copy still serves this run.
      }
    },
  };
}

module.exports = { makePoolLookup, fileCache, PUBLIC_DNS, QUERY_TIMEOUT_MS };
