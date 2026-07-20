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

// Local LLM inference (llama.cpp `llama-server`), run alongside the miner. It's
// an OpenAI-compatible HTTP server we spawn like the mining engine; the same
// endpoint powers both the local API and (later) the job queue.
const LLM = {
  host: '127.0.0.1',
  port: 8080,
  ctxSize: 4096,
  parallel: 1,
  // Keep this much VRAM free for the miner when co-running (the budgeter caps
  // GPU layers so the model fits in whatever's left).
  miningReserveMb: 2048,
  // llama-server binary per platform (bundled/downloaded like the miner engine).
  serverBin: { win32: 'llama-server.exe', linux: 'llama-server', darwin: 'llama-server' },
  // Where to fetch the llama-server build if it isn't bundled. llama.cpp embeds
  // the build number in the asset name, so a fixed `latest/download/<name>` 404s
  // — these are pinned to a specific build. Windows uses the **Vulkan** archive:
  // a single self-contained zip (no separate CUDA `cudart` package) that runs
  // GPU-accelerated on NVIDIA/AMD/Intel. (Linux/macOS still point at the old
  // fixed-name assets and need the same treatment — tracked in #88.)
  serverUrl: {
    win32: 'https://github.com/ggml-org/llama.cpp/releases/download/b9902/llama-b9902-bin-win-vulkan-x64.zip',
    linux: 'https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-bin-ubuntu-x64.zip',
    darwin: 'https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-bin-macos-arm64.zip',
  },
  // A small, capable model to start with: Google Gemma 4 E4B Instruct, Q4_K_M
  // GGUF (~5 GB). "E4B" = ~4.5B *effective* params via Per-Layer Embeddings, so
  // it keeps a low VRAM footprint (runs in ~5 GB at 4-bit) while adding 128K
  // context, tool-calling, and 140+ languages — a good default that co-runs with
  // mining without hogging the GPU. `layers` is the text model's transformer-layer
  // count (for --n-gpu-layers; llama.cpp clamps a larger value to what's present)
  // and `vramFullMb` the approximate VRAM for a full GPU offload at ctxSize
  // (weights + KV cache). `minVramMb` is the hard floor of free VRAM we require
  // before starting the model on the GPU — a little above the ~5.8 GB full-offload
  // estimate so we never spawn llama-server right at the edge and OOM.
  model: {
    name: 'Gemma-4-E4B-it-Q4_K_M',
    file: 'gemma-4-E4B-it-Q4_K_M.gguf',
    url: 'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf',
    layers: 42,
    vramFullMb: 5800,
    minVramMb: 6144, // ~6 GB free required before we put it on the GPU
    quant: 'Q4_K_M',
  },
};

// Linking this machine to an LLMJob account ("Connect with LLMJob"). The node
// self-registers with a pairing/join token (only its public key leaves the box),
// then pings on an interval so it shows online in the user's cluster. Mirrors the
// server's /api/nodes/join + /api/nodes/ping contract.
const NODE = {
  serverUrl: 'https://llmjob-production.up.railway.app',
  // Where the user copies their pairing token (sign in, Dashboard → Add node).
  dashboardUrl: 'https://llmjob-production.up.railway.app/dashboard.html',
  pingIntervalMs: 5 * 60 * 1000,
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
  REGIONS, DEFAULTS, MINER, NETWORK, ECON, LLM, NODE, DIFFICULTY_BY_CARD,
  regionFor, endpointFor, regionLabel, difficultyForCard,
};
