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
const ECON = {
  NET_TH: 18400, // network hashrate (TH/s)
  DAILY_NET_PRL: 7200, // PRL minted across the network per day
  FEE: 0.99, // share kept after the 1% pool fee
  PRL_USD: 0.082, // PRL price in USD
};

// Recommended static difficulty per card class, from the pool's table. Order
// matters: more specific patterns first.
const DIFFICULTY_BY_CARD = [
  { match: /5090|h100|h200|b100|b200/i, difficulty: 1048576 },
  { match: /4090|5080/i, difficulty: 524288 },
  { match: /4070|4080/i, difficulty: 262144 },
  { match: /3080|3090|70hx|90hx/i, difficulty: 262144 },
  { match: /3060 ti|3070/i, difficulty: 131072 },
  { match: /a100/i, difficulty: 131072 },
  { match: /2070|2080|rtx 20|\bt4\b/i, difficulty: 16384 },
  { match: /v100|titan v|cmp [12]\d\d/i, difficulty: 4096 },
];

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
  REGIONS, DEFAULTS, MINER, ECON, DIFFICULTY_BY_CARD,
  regionFor, endpointFor, regionLabel, difficultyForCard,
};
