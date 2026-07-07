'use strict';

// Static configuration for the LLMJob Earn desktop wrapper.
//
// The app wraps the AlphaPool `alpha-miner` engine for Pearl (PRL). All values
// below come from the AlphaPool setup page (pearl.alphapool.tech/#setup):
// Stratum endpoints, the per-card static-difficulty table, the engine binaries,
// and the network economics used for earnings estimates. The binary path is
// configurable so a downloaded alpha-miner build drops in without code changes.

// Stratum pool endpoints (host:port). Pick the closest for lowest share latency.
const REGIONS = {
  us1: { label: 'us1', flag: '🇺🇸', name: 'N. America · East', endpoint: 'us1.alphapool.tech:5566' },
  us2: { label: 'us2', flag: '🇺🇸', name: 'N. America · West', endpoint: 'us2.alphapool.tech:5566' },
  eu1: { label: 'eu1', flag: '🇪🇺', name: 'Europe', endpoint: 'eu1.alphapool.tech:5566' },
  eu2: { label: 'eu2', flag: '🇪🇺', name: 'Europe', endpoint: 'eu2.alphapool.tech:5566' },
  ru1: { label: 'ru1', flag: '🇷🇺', name: 'Russia · Eurasia', endpoint: 'ru1.alphapool.tech:5566' },
  sg1: { label: 'sg1', flag: '🇸🇬', name: 'Asia · Singapore', endpoint: 'sg1.alphapool.tech:5566' },
  hk1: { label: 'hk1', flag: '🇭🇰', name: 'Asia · Hong Kong', endpoint: 'hk1.alphapool.tech:5566' },
  in1: { label: 'in1', flag: '🇮🇳', name: 'India', endpoint: 'in1.alphapool.tech:5566' },
};

const DEFAULTS = {
  region: 'us2',
  worker: 'rig01',
  difficulty: 524288, // RTX 4090 / 5080 class — a safe general default
  algo: 'pearlhash',
  powerLimit: 318,
};

// Engine / pool metadata.
const MINER = {
  engine: 'alpha-miner',
  downloadUrl: 'https://pearl.alphapool.tech/downloads/alpha-miner',
  windowsZipNvidia: 'AlphaMiner-Pearl-Windows.zip', // contains alpha-miner-windows.exe
  windowsZipAmd: 'AlphaMiner-Pearl-AMD.zip', // contains alpha-miner-amd-windows-fixed.exe
  dockerImage: 'alphaminetech/pearl-miner:1.7.9',
  pow: 'pearlhash',
  devFeePct: 0,
  poolFeePct: 1,
  payoutScheme: 'PPLNS',
  payoutIntervalHours: 4,
  minPayoutPrl: 1,
};

// Network economics used to estimate earnings. Mirrors the design mock.
// LLMJob network page: where the app reports its live mining status so it shows
// up on the public "who's mining now" board, and how often.
const NETWORK = {
  reportUrl: 'https://llmjob-production.up.railway.app/api/miners/ping',
  reportIntervalMs: 60000, // report once a minute while mining
};

// Network economics for earnings estimates. Mirrors the earn.html calculator
// (which live-refreshes from prlscan / SafeTrade and falls back to these).
const ECON = {
  NET_TH: 30.79e6, // network hashrate in TH/s (~30.79 EH/s) — prlscan
  DAILY_NET_PRL: 1.2e6, // ~2,600 PRL/block × ~480 blocks/day
  FEE: 0.99, // share kept after the 1% pool fee
  PRL_USD: 0.47, // PRL price in USD (PRL/USDT — SafeTrade)
};

// Recommended static difficulty per card class, from the pool's table. Order
// matters: more specific patterns first.
const DIFFICULTY_BY_CARD = [
  { match: /5090|h100|h200|b100|b200|pro 6000/i, difficulty: 1048576 }, // incl. RTX PRO 6000 (Blackwell)
  { match: /4090|5080/i, difficulty: 524288 },
  { match: /4070|4080/i, difficulty: 262144 },
  { match: /3080|3090|70hx|90hx/i, difficulty: 262144 },
  { match: /3060 ti|3070/i, difficulty: 131072 },
  { match: /4060|5060/i, difficulty: 131072 },
  { match: /a100/i, difficulty: 131072 },
  { match: /2070|2080|rtx 20|\bt4\b/i, difficulty: 16384 },
  { match: /v100|titan v|cmp [12]\d\d/i, difficulty: 4096 },
];

// HeroMiners' Pearl (PRL) + modelOS (MDL) merged-mining pool. All endpoints are
// port 1200 (plain TCP and TLS on the same port), per their beginner's guides
// (herominers.medium.com). Region keys are theirs; 'ww' is the geo-routed host.
const HERO_REGIONS = {
  ww: { label: 'auto', flag: '🌍', name: 'Worldwide', endpoint: 'pearl.herominers.com:1200' },
  us: { label: 'us', flag: '🇺🇸', name: 'N. America', endpoint: 'us.pearl.herominers.com:1200' },
  de: { label: 'de', flag: '🇩🇪', name: 'Europe · Germany', endpoint: 'de.pearl.herominers.com:1200' },
  fr: { label: 'fr', flag: '🇫🇷', name: 'Europe · France', endpoint: 'fr.pearl.herominers.com:1200' },
  sg: { label: 'sg', flag: '🇸🇬', name: 'Asia · Singapore', endpoint: 'sg.pearl.herominers.com:1200' },
};

// Supported upstream pools. Both have in-app balance lookups (balance.js talks
// to each pool's API). The two differ in how the engine identifies the rig:
// AlphaPool takes a separate --worker flag and pins static difficulty via the
// stratum password (`x;d=N`); HeroMiners uses the classic `wallet.worker`
// login suffix and vardiff only.
const DEFAULT_POOL = 'alphapool';
const POOLS = {
  alphapool: {
    label: 'AlphaPool',
    site: 'https://pearl.alphapool.tech',
    regions: REGIONS,
    defaultRegion: 'us2',
    workerStyle: 'flag',
    staticDifficulty: true,
    balances: true,
  },
  herominers: {
    label: 'HeroMiners',
    site: 'https://pearl.herominers.com',
    regions: HERO_REGIONS,
    defaultRegion: 'ww',
    workerStyle: 'suffix',
    staticDifficulty: false,
    balances: true,
  },
};

function poolFor(pool) {
  return POOLS[pool] || POOLS[DEFAULT_POOL];
}

function regionsFor(pool) {
  return poolFor(pool).regions;
}

// Endpoint for a pool + region, falling back to that pool's default region.
function poolEndpointFor(pool, region) {
  const p = poolFor(pool);
  return (p.regions[region] || p.regions[p.defaultRegion]).endpoint;
}

function regionFor(region) {
  return REGIONS[region] || REGIONS[DEFAULTS.region];
}

function endpointFor(region) {
  return regionFor(region).endpoint;
}

function regionLabel(region) {
  const r = regionFor(region);
  return r.flag + ' ' + r.label;
}

// Suggested static difficulty for a GPU name; falls back to the default.
function difficultyForCard(name) {
  const s = String(name == null ? '' : name);
  for (const row of DIFFICULTY_BY_CARD) {
    if (row.match.test(s)) return row.difficulty;
  }
  return DEFAULTS.difficulty;
}

module.exports = {
  REGIONS, HERO_REGIONS, POOLS, DEFAULT_POOL, DEFAULTS, MINER, NETWORK, ECON, DIFFICULTY_BY_CARD,
  regionFor, endpointFor, regionLabel, difficultyForCard,
  poolFor, regionsFor, poolEndpointFor,
};
