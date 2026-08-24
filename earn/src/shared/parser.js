'use strict';

// Parse a line of PeakMiner output into a structured event.
//
// Every shape below was captured from a real 2.11.0 run against
// us.pearl.herominers.com:1200 on an RTX 4090, not read off a docs page. That
// distinction has cost us before: the alpha-miner 1.9.4 parser was written to a
// format the binary had already stopped emitting, and the dashboard showed zeros
// while the rig mined perfectly.
//
// The lines that matter:
//
//   2026-08-23 23:38:40  INFO connected us.pearl.herominers.com:1200  diff —  ping 1059ms
//   2026-08-23 23:30:25 accepted   GPU 0  lat 82ms  diff 9.01 PH  effort 50%
//   2026-08-23 23:39:42 ERROR failed to connect to no-such-pool.invalid:1200: No such host is known. (os error 11001)
//
// plus the periodic status table, one row per card:
//
//     0  RTX 4090  296.5 TH/s       3 / 0      78°C   48%   449W  660.4 GH/W  10251MHz   2340MHz
//
// (index, name, hashrate, accepted/invalid, temp, fan, power, efficiency, then
// the two clocks). The `Total` row carries no index and is skipped, so a
// multi-GPU rig still accumulates per card.
//
// Anything unrecognised returns null so callers pass it through as raw log text.

// The ESC is optional because the two halves of a colour sequence can be split
// across stdio chunks, and because the previous engine's parser matched only the
// bracket form — leaving a bare 0x1b at the head of the line, which then defeats
// the timestamp match and silently drops the event. Consume both.
// Built with RegExp rather than a regex literal so the escape byte is spelled
// out as an escape sequence instead of embedded raw: a literal 0x1b in source
// is invisible in a diff and trivially lost in an edit.
const ANSI = new RegExp('\\u001b?\\[[0-9;]*m', 'g'); // eslint-disable-line no-control-regex

// A leading `YYYY-MM-DD HH:MM:SS` stamp, which every log line carries and no
// status-table row does. Stripping it first means the row matcher never has to
// worry about a date that happens to start with digits.
const STAMP = /^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+/;

const HASHRATE = /([\d.]+)\s*([KMGTPE]?)H\/s/i;
const UNIT_TH = { '': 1e-12, K: 1e-9, M: 1e-6, G: 1e-3, T: 1, P: 1e3, E: 1e6 };

// One row of the status table. Anchored on the hashrate token and read outwards
// rather than matched as a fixed column layout: a rig that reports no fan or no
// clocks would otherwise fail the whole row and silently show zero — the exact
// failure this style of parsing replaced.
function parseStatsRow(s) {
  const head = s.match(/^(\d+)\s+(\S.*)$/);
  if (!head) return null;
  const rest = head[2];
  const hr = rest.match(HASHRATE);
  if (!hr) return null;

  const tail = rest.slice(hr.index + hr[0].length);
  // Shares are printed as `ok / inv`, so take the first such pair rather than
  // the trailing bare integers — the clock columns are bare integers too.
  const shares = tail.match(/(\d+)\s*\/\s*(\d+)/);
  const temp = tail.match(/(\d+)\s*°?C\b/);
  const power = tail.match(/(\d+)\s*W\b/);
  const name = rest.slice(0, hr.index).trim();

  return {
    type: 'status',
    gpuIndex: Number(head[1]),
    hashrate: Number(hr[1]) * UNIT_TH[hr[2].toUpperCase()],
    accepted: shares ? Number(shares[1]) : null,
    rejected: shares ? Number(shares[2]) : null,
    power: power ? Number(power[1]) : null,
    temp: temp ? Number(temp[1]) : null,
    gpu: name || null,
  };
}

function parseLine(line) {
  const raw = String(line == null ? '' : line).replace(ANSI, '').trim();
  if (!raw) return null;
  const s = raw.replace(STAMP, '');

  // Pool connection. The trailing `diff — ping 1059ms` is deliberately not read
  // for difficulty: at connect time the pool has not assigned one yet and the
  // field is a literal em dash, so there is nothing there to believe.
  const conn = s.match(/^INFO\s+connected\s+(\S+?):(\d+)/i);
  if (conn) {
    return { type: 'connected', gpuIndex: null, endpoint: conn[1] + ':' + conn[2], gpu: null };
  }

  // A single accepted share, which is what makes the UI's counter move between
  // status tables. PeakMiner prints this per share with no `INFO` prefix.
  const share = s.match(/^accepted\s+GPU\s+(\d+)/i);
  if (share) {
    return { type: 'share', gpuIndex: Number(share[1]), accepted: true };
  }

  // A failed connection attempt, and whether the name even resolved. The miner
  // retries with a backoff and prints an identical line each time, which is how
  // a rig produces eight lines of "No such host is known" that never say WHICH
  // host or that name resolution is the actual problem. Reported from the field
  // in exactly that shape, so the `dns` flag exists to let the UI say so.
  // Greedy on the endpoint, not lazy: the endpoint is `host:port` and the reason
  // follows a second colon, so a lazy match stops at the first one and reports
  // the host as `no-such-pool.invalid` with the port glued onto the front of the
  // reason ("1200: No such host is known").
  const failed = s.match(/^ERROR\s+failed to connect to\s+(\S+):\s*(.+)$/i);
  if (failed) {
    const reason = failed[2].trim();
    return {
      type: 'connect-failed',
      endpoint: failed[1],
      reason,
      dns: /no such host|dns|name (or service )?not known|getaddrinfo|os error 11001/i.test(reason),
    };
  }

  // The status table. Checked last because its rows are the least distinctive
  // thing here — a bare index followed by free text.
  return parseStatsRow(s);
}

module.exports = { parseStatsRow, parseLine };
