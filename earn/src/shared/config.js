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
// explorer's own backend). Mirrors the earn.html calculator's sources.
const ECON_API = {
  price: 'https://api.prlscan.com/v1/market/prl',
  metrics: 'https://api.prlscan.com/v1/analytics/block-metrics',
  blocks: 'https://api.prlscan.com/v1/blocks?limit=1',
};

// Local LLM inference (llama.cpp `llama-server`), run alongside the miner. It's
// an OpenAI-compatible HTTP server we spawn like the mining engine; the same
// endpoint powers both the local API and (later) the job queue.
const LLM = {
  host: '127.0.0.1',
  port: 8080,
  // Bounds prompt + generation together, so this — not the server's max_tokens —
  // is what actually caps a reply. At 4096 a reasoning model ran out of context
  // mid-thought on hard prompts and returned nothing; the model itself supports
  // 128K, so this was our own conservative floor rather than a limit of Gemma.
  //
  // 6400 turned out to be the same floor one step up. On AIME 2025 it cut 26% of
  // samples off mid-working — six problems truncated on more than half their
  // samples and scored 1/45 between them, against 52/87 on the problems that
  // always fit. Reasoning benchmarks are normally run at 32768, and asking the
  // model to recover a cut-off answer does not work: it fabricates one (returned
  // 0 against a gold of 62), so the window is the only lever.
  //
  // The 280s serving budget still caps what any single request can actually
  // produce — at fleet speeds that is roughly 8-14k tokens — so this is headroom
  // rather than a licence to generate 32k.
  ctxSize: 32768,
  parallel: 1,
  // Self-heal a llama-server that exits before ready because it couldn't bind the
  // port yet — the previous server (e.g. from an update relaunch) is still
  // releasing 8080. Retry the spawn a few times, spaced out, so the LLM comes up
  // on its own once the port frees instead of silently staying down.
  startAttempts: 5,
  startRetryMs: 2000,
  // Keep this much VRAM free for the miner when co-running (the budgeter caps
  // GPU layers so the model fits in whatever's left).
  miningReserveMb: 2048,
  // llama-server binary per platform (bundled/downloaded like the miner engine).
  serverBin: { win32: 'llama-server.exe', linux: 'llama-server', darwin: 'llama-server' },
  // Where to fetch the llama-server build if it isn't bundled. llama.cpp embeds
  // the build number in the asset name, so a fixed `latest/download/<name>` 404s
  // — these are pinned to a specific build (old releases keep their assets, so
  // the pins stay resolvable). Windows/Linux use the **Vulkan** archive: a single
  // self-contained bundle (no separate CUDA `cudart` package) that runs
  // GPU-accelerated on NVIDIA/AMD/Intel. macOS uses the Metal builds, which are
  // built per architecture. Windows ships a .zip; Linux/macOS ship .tar.gz — the
  // extractor (io.extractLlamaZip) sniffs the archive type and flattens either
  // into place, which is exactly what the macOS archives need: one top folder
  // holding llama-server beside its dylibs, and the binary's only LC_RPATH is
  // `@loader_path`, so the flattened layout is the layout it expects.
  //
  // Keys are `<platform>` with an optional `<platform>-<arch>` override that
  // wins when present (see llama.resolveServerUrl). Only macOS needs one today:
  // Apple silicon and Intel Macs get different archives, and handing an Intel
  // Mac the arm64 build yields a binary the kernel refuses to exec — the LLM's
  // one job on the platform where mining can't run at all.
  serverUrl: {
    win32: 'https://github.com/ggml-org/llama.cpp/releases/download/b9902/llama-b9902-bin-win-vulkan-x64.zip',
    linux: 'https://github.com/ggml-org/llama.cpp/releases/download/b9902/llama-b9902-bin-ubuntu-vulkan-x64.tar.gz',
    darwin: 'https://github.com/ggml-org/llama.cpp/releases/download/b9902/llama-b9902-bin-macos-arm64.tar.gz',
    'darwin-x64': 'https://github.com/ggml-org/llama.cpp/releases/download/b9902/llama-b9902-bin-macos-x64.tar.gz',
  },
  // A small, capable model to start with: Google Gemma 4 E4B Instruct, Q4_K_M
  // GGUF (~5 GB). "E4B" = ~4.5B *effective* params via Per-Layer Embeddings, so
  // it keeps a low VRAM footprint (runs in ~5 GB at 4-bit) while adding 128K
  // context, tool-calling, and 140+ languages — a good default that co-runs with
  // mining without hogging the GPU. `layers` is the text model's transformer-layer
  // count (for --n-gpu-layers; llama.cpp clamps a larger value to what's present)
  // and `vramFullMb` the VRAM for a full GPU offload at ctxSize (weights + KV
  // cache). `minVramMb` is the hard floor of free VRAM we require before
  // starting the model on the GPU — above the full-offload figure so we never
  // spawn llama-server right at the edge and OOM.
  model: {
    name: 'Gemma-4-E4B-it-Q4_K_M',
    file: 'gemma-4-E4B-it-Q4_K_M.gguf',
    url: 'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf',
    layers: 42,
    // MEASURED, not estimated. On a 4090 running the shipped command line
    // (--n-gpu-layers 999 --ctx-size 6400 --split-mode none) llama-server holds
    // 3308 MB of dedicated VRAM, flat — sampled through a 700-token generation,
    // because llama.cpp allocates the KV cache up front, so steady state is also
    // peak. Nothing lands in shared/host memory.
    //
    // The old 6300/6656 were guessed from the 4.6 GiB file size and were nearly
    // 2x reality: E4B keeps its Per-Layer Embeddings in host RAM, so VRAM sits
    // well under the weight file. That over-booking locked out a 16 GB card with
    // 6.7 GB free — the model would have fitted twice over.
    //
    // ESTIMATED for ctxSize 32768, and that is the weak part of this change —
    // the 3308 MB above was measured on a 4090 at ctxSize 6400 and nobody has
    // yet run llama-server at 32768 to read the real figure.
    //
    // The estimate: llama.cpp allocates the KV cache up front and it scales with
    // context, so 5.12x the window grows only the cache, not the weights. Taking
    // the pessimistic split of that 3308 (weights ~2300, cache ~1000) puts the
    // cache near 5100 MB and the total near 7400. 7680 is that plus headroom.
    //
    // Deliberately pessimistic, because the two failure modes are not
    // symmetric: over-booking keeps a node out of the pool, while under-booking
    // OOMs llama-server and crash-loops it. Gemma's alternating local/global
    // attention may well make the real cache far cheaper than linear — which is
    // exactly why this needs measuring rather than trusting. Re-measure at
    // 32768 and bring these numbers down if it comes in under.
    vramFullMb: 7680,
    minVramMb: 8704, // ~8.5 GB free required before we put it on the GPU
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
  REGIONS, DEFAULTS, MINER, NETWORK, ECON, ECON_API, LLM, NODE, DIFFICULTY_BY_CARD,
  regionFor, endpointFor, normalizeEndpoint, resolveEndpoint, splitEndpoint, regionLabel, difficultyForCard,
};
