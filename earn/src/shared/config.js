'use strict';

// Static configuration for the LLMJob Earn desktop wrapper.
//
// The app mines Pearl (PRL) with our own CUDA core — see src/main/pearlMiner.js
// and earn/native. There is no external engine to download, no vendored binary
// and no dev fee.
//
// The pool is HeroMiners. AlphaPool was dropped along with alpha-miner: it gates
// its stratum behind a GPU-solved challenge and then DICTATES the mining
// geometry (M/N=131072, K=4096, a 2x64 tile), which is shaped for Hopper's
// wgmma and awkward on everything else. HeroMiners sends a header and a target
// and lets the miner choose its own geometry, which is what the protocol
// actually allows — the verifier reconstructs the tile pattern from the row
// indices in the submitted proof.

// Stratum pool endpoints (host:port). Pick the closest for lowest share latency.
// Every one of these was checked to resolve and accept a connection on 1200.
const REGIONS = {
  us: { label: 'us', flag: '🇺🇸', name: 'N. America · East', endpoint: 'us.pearl.herominers.com:1200' },
  us2: { label: 'us2', flag: '🇺🇸', name: 'N. America · West', endpoint: 'us2.pearl.herominers.com:1200' },
  ca: { label: 'ca', flag: '🇨🇦', name: 'N. America · Canada', endpoint: 'ca.pearl.herominers.com:1200' },
  br: { label: 'br', flag: '🇧🇷', name: 'S. America · Brazil', endpoint: 'br.pearl.herominers.com:1200' },
  de: { label: 'de', flag: '🇩🇪', name: 'Europe · Germany', endpoint: 'de.pearl.herominers.com:1200' },
  fi: { label: 'fi', flag: '🇫🇮', name: 'Europe · Finland', endpoint: 'fi.pearl.herominers.com:1200' },
  fr: { label: 'fr', flag: '🇫🇷', name: 'Europe · France', endpoint: 'fr.pearl.herominers.com:1200' },
  tr: { label: 'tr', flag: '🇹🇷', name: 'Europe · Turkey', endpoint: 'tr.pearl.herominers.com:1200' },
  sg: { label: 'sg', flag: '🇸🇬', name: 'Asia · Singapore', endpoint: 'sg.pearl.herominers.com:1200' },
  hk: { label: 'hk', flag: '🇭🇰', name: 'Asia · Hong Kong', endpoint: 'hk.pearl.herominers.com:1200' },
  kr: { label: 'kr', flag: '🇰🇷', name: 'Asia · Korea', endpoint: 'kr.pearl.herominers.com:1200' },
  au: { label: 'au', flag: '🇦🇺', name: 'Oceania · Australia', endpoint: 'au.pearl.herominers.com:1200' },
};

const DEFAULTS = {
  region: 'us',
  worker: 'rig01',
  algo: 'pearlhash',
  powerLimit: 318,
};

// Engine / pool metadata.
//
// There is nothing to download any more: the engine is our own CUDA core,
// linked into this process as an N-API addon (earn/native), so no URL, no zip
// and no Docker image. The dev fee is zero and there is no dev-address code
// path — this is our own implementation written against the ISC-licensed
// reference, not a derivative of any fee-bearing miner.
//
// The pool terms are HeroMiners' own, read from
// https://pearl.herominers.com/api/stats rather than transcribed from a setup
// page: fee 0, paymentsInterval 3600s, minPaymentThreshold 1e8 against
// coinUnits 1e8 (so 1 PRL), rewardScheme "prop".
const MINER = {
  engine: 'llmjob-pearl',
  pool: 'HeroMiners',
  pow: 'pearlhash',
  devFeePct: 0,
  poolFeePct: 0,
  payoutScheme: 'PROP',
  payoutIntervalHours: 1,
  minPayoutPrl: 1,
};

// Network economics used to estimate earnings. Mirrors the design mock.
// LLMJob network page: where the app reports its live mining status so it shows
// up on the public "who's mining now" board, and how often.
const NETWORK = {
  reportUrl: 'https://llmjob-production.up.railway.app/api/miners/ping',
  reportIntervalMs: 60000, // report once a minute while mining
  // How often to re-check for an app update after the one at startup. That
  // startup check used to be the only one, so a rig that launched while the
  // GitHub releases feed was unavailable — a real 503 window was observed in
  // the wild — never looked again until someone restarted it, and would sit on
  // a broken build indefinitely. Six hours is frequent enough that a hotfix
  // lands the same day without polling GitHub for no reason.
  updateCheckIntervalMs: 6 * 60 * 60 * 1000,
  // Where a user goes to update by hand. Only macOS needs it — the Mac build is
  // ad-hoc signed, so Squirrel.Mac cannot install an update over it and the app
  // does not wire the updater there at all (see shared/platform.autoUpdateSupported).
  // Sending them to the releases page beats a "Check for updates" button that
  // can only ever fail.
  releasesUrl: 'https://github.com/super3/llmjob/releases/latest',
};

// Network economics for earnings estimates. The app live-refreshes these from
// the prlscan API at runtime (see shared/economics.js + main.js); the constants
// here are only the offline fallback, so keep them roughly current — a stale
// fallback silently overstates earnings (a network that doubled makes every
// estimate ~2x too high). Snapshot: 2026-07 (prlscan).
const ECON = {
  NET_TH: 61e6, // network hashrate in TH/s (~61 EH/s) — prlscan
  DAILY_NET_PRL: 1.62e6, // ~2,489 PRL/block × ~650 blocks/day
  FEE: 0.99, // share kept after the 1% pool fee
  PRL_USD: 0.30, // PRL price in USD — prlscan (SafeTrade-sourced)
};

// prlscan API endpoints the app live-refreshes economics from (CORS-open; the
// explorer's own backend). Mirrors the site calculator's sources.
const ECON_API = {
  price: 'https://api.prlscan.com/v1/market/prl',
  metrics: 'https://api.prlscan.com/v1/analytics/block-metrics',
  blocks: 'https://api.prlscan.com/v1/blocks?limit=1',
};

function regionFor(region) {
  return REGIONS[region] || REGIONS[DEFAULTS.region];
}

function endpointFor(region) {
  return regionFor(region).endpoint;
}

// Normalise a hand-supplied endpoint override into the bare `host:port` the
// engine's --host wants.
//
// alpha-miner 1.9.4 takes a bare endpoint, but every older doc — and our own
// pre-1.9.4 argument vector — wrote it as `stratum+tcp://host:port`, because
// that is what the old --pool flag took. Pasting that form into an endpoint
// override used to be harmless; now it is handed to --host verbatim, the engine
// tries to resolve the literal string `stratum+tcp://host`, and the rig loops on
// "DNS lookup failed: No such host is known" with nothing naming the cause.
// Reported from the field, which is the only reason we know the shape of it.
//
// Deliberately forgiving rather than clever: strip any scheme and surrounding
// whitespace, and hand back null for something empty so the caller falls back to
// the region default instead of spawning a miner pointed at nothing.
function normalizeEndpoint(endpoint) {
  const raw = String(endpoint == null ? '' : endpoint).trim();
  const bare = raw.replace(/^[a-z][a-z0-9+.-]*:\/\//i, '').replace(/\/+$/, '').trim();
  return bare || null;
}

// Split a `host:port` endpoint into its parts. Returns port null when there
// is none, so callers can decide the default rather than inventing one here.
// Rightmost colon wins, which keeps a bracketed IPv6 literal intact.
function splitEndpoint(endpoint) {
  const bare = normalizeEndpoint(endpoint);
  if (!bare) return { host: null, port: null };
  const m = bare.match(/^(.*):(\d+)$/);
  return m ? { host: m[1], port: Number(m[2]) } : { host: bare, port: null };
}

// The endpoint to mine against: a normalised override if there is one, else the
// region's. One function so the argument vector and the "connecting to …" log
// line can never disagree about where the rig is actually pointed.
function resolveEndpoint(settings) {
  const s = settings || {};
  return normalizeEndpoint(s.endpoint) || endpointFor(s.region || DEFAULTS.region);
}

// Where a saved AlphaPool region should land now that the pool is HeroMiners.
//
// Every 0.3.x install has one of these in its settings file, and none of them
// exist any more. Without a mapping the Settings dropdown is handed a value
// with no matching <option>, which leaves a <select> BLANK rather than
// erroring — and the renderer's own fallback then quietly rewrote the choice to
// whatever the default was. An upgrading rig would move continent without being
// told.
//
// Mapped to the nearest live endpoint rather than all to the default: someone
// who picked Singapore should stay in Singapore. 'us2' is deliberately absent
// because HeroMiners has a us2 as well, so it needs no translation.
const LEGACY_REGIONS = {
  us1: 'us',   // N. America East
  eu1: 'de',   // Europe -> Germany
  eu2: 'fi',   // the second European choice -> Finland
  ru1: 'fi',   // Eurasia: HeroMiners' ru resolves but refused connections
  sg1: 'sg',
  hk1: 'hk',
  in1: 'sg',   // India -> Singapore, the closest that answers
};

// A live region id for whatever was saved: unchanged when it still exists,
// translated when it is a known AlphaPool id, else the default.
function migrateRegion(region) {
  const id = String(region == null ? '' : region).trim();
  if (Object.prototype.hasOwnProperty.call(REGIONS, id)) return id;
  if (Object.prototype.hasOwnProperty.call(LEGACY_REGIONS, id)) return LEGACY_REGIONS[id];
  return DEFAULTS.region;
}

function regionLabel(region) {
  const r = regionFor(region);
  return r.flag + ' ' + r.label;
}


module.exports = {
  REGIONS, LEGACY_REGIONS, DEFAULTS, MINER, NETWORK, ECON, ECON_API,
  regionFor, endpointFor, normalizeEndpoint, resolveEndpoint, splitEndpoint, regionLabel,
  migrateRegion,
};
