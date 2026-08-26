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
  // Models the fleet can serve, biggest first. A node runs exactly ONE of these:
  // shared/models.pickModel walks this list and takes the first whose VRAM floor
  // the chosen card clears, so a 32 GB card serves the large vision model and a
  // 12 GB card keeps serving Gemma. Nothing is split across cards and nothing is
  // partially offloaded — see vram.js for why.
  //
  // Order is capability order, not preference-by-size for its own sake: the list
  // is walked top-down and the first fit wins.
  tiers: [
    // Qwen3.8-27B (Alibaba, Apache-2.0, released 2026-08-14): 27B dense with a
    // vision tower and a 262,144-token native context. Vision needs the separate
    // mmproj projector alongside the weights — llama-server loads it with
    // --mmproj and without it the model runs text-only.
    //
    // File sizes are READ FROM THE HUB, not guessed: the Q4_K_XL weights are
    // 17.56 GB and mmproj-F16 is 0.93 GB, so ~18.5 GB is resident before a single
    // token of KV cache. Both URLs were checked to resolve 200.
    //
    // vramFullMb/minVramMb are NOT measured and are the weak part of this entry —
    // nobody has run llama-server at ctxSize 262144 to read the real figure, and
    // the KV cache is the whole unknown (48 of its 64 layers use linear attention,
    // whose state does not grow with context, so only 16 layers hold full-context
    // KV — which is why 256K is plausible at all on one card).
    //
    // Rather than guess that number and have it be wrong in the direction that
    // OOMs a node, the floor below is set high enough that only a large card even
    // attempts it, and llmManager walks `ctxLadder` downward when llama-server
    // fails to start. A card that cannot hold 262144 lands on a window it can
    // hold instead of crash-looping. Re-measure and tighten both figures once a
    // 5090 has actually run it.
    {
      key: 'qwen3.8-27b',
      name: 'Qwen3.8-27B-UD-Q4_K_XL',
      file: 'Qwen3.8-27B-UD-Q4_K_XL.gguf',
      url: 'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf',
      // The vision projector. Separate download, separate file, loaded with
      // --mmproj; the weights alone are a text-only model.
      mmproj: {
        file: 'Qwen3.8-27B-mmproj-F16.gguf',
        url: 'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-F16.gguf',
      },
      vision: true,
      layers: 64,
      ctxSize: 262144,
      // Tried in order when the one above fails to start. Each step roughly
      // halves the KV cache, so a card that is short by a little gets a working
      // server rather than a restart loop.
      ctxLadder: [262144, 131072, 65536, 32768],
      vramFullMb: 28672,  // ESTIMATE — 18.5 GB resident + KV headroom. Measure me.
      minVramMb: 30720,   // ESTIMATE — deliberately high so only a 32 GB card tries.
      quant: 'Q4_K_XL',
    },
  ],
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
  //
  // Still the fallback every node lands on when it cannot host a tier above, and
  // still what `LLM.model` means everywhere that reads it.
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
    // MEASURED at ctxSize 32768, on a 5090 running the shipped command line
    // (--n-gpu-layers 999 --ctx-size 32768 --split-mode none): llama-server
    // holds 3719 MB of dedicated VRAM. Confirmed GPU-resident rather than
    // silently offloaded — 123 tok/s, and the server log reports
    // n_ctx_slot = 32768. llama.cpp allocates the KV cache up front, so steady
    // state is also peak.
    //
    // The same card at ctxSize 6400 holds 3227 MB, which is within 2.5% of the
    // 3308 measured on a 4090 — so the two rigs agree and the delta below is a
    // like-for-like reading of the context change, not a hardware difference.
    //
    // THE WHOLE 5.12x CONTEXT INCREASE COSTS 492 MB.
    //
    // This replaces an estimate that assumed the cache scales linearly and put
    // the total near 7400. It does not scale linearly: Gemma's alternating
    // local/global attention gives most layers a small sliding window, so only a
    // few hold full-context KV. The estimate was deliberately pessimistic on the
    // grounds that over-booking merely idles a node while under-booking
    // crash-loops it — sound reasoning, wrong by ~2x, and it would have put the
    // floor at 9728 MB of free VRAM for a model that needs 3.7 GB, excluding a
    // 5060 Ti and making a 4070 borderline for no reason.
    //
    // 4096 is the measured figure plus ~10% headroom; 4608 keeps roughly the old
    // ratio between the floor and the full-offload figure. Note the pre-32768
    // value (3800) already covered the real 32768 usage, with 81 MB to spare —
    // the context window barely moves VRAM at all. Re-measure if ctxSize or the
    // model changes: both move the KV cache.
    vramFullMb: 4096,
    minVramMb: 4608, // ~4.5 GB free required before we put it on the GPU
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
  REGIONS, LEGACY_REGIONS, DEFAULTS, MINER, NETWORK, ECON, ECON_API, LLM, NODE,
  regionFor, endpointFor, normalizeEndpoint, resolveEndpoint, splitEndpoint, regionLabel,
  migrateRegion,
};
