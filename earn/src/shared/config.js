'use strict';

// Static configuration for the LLMJob Earn desktop wrapper.
//
// The app mines Pearl (PRL) with our own CUDA core — see src/main/pearlMiner.js
// and earn/native. There is no external engine to download, no vendored binary
// and no dev fee.
//
// The default pool is HeroMiners, and Kryptex is the other choice (POOLS
// below). AlphaPool was dropped along with alpha-miner: it gates
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

// Kryptex's servers, one per continent: prl-<id>.kryptex.network for Pearl and
// xmr-<id> for Monero (KRYPTEX below). Its Middle East name pointed at the eu
// servers on 2026-10-20, and its Russian one refused connections from the US.
const KRYPTEX_REGIONS = {
  us: { flag: '🇺🇸', name: 'N. America' },
  eu: { flag: '🇪🇺', name: 'Europe' },
  br: { flag: '🇧🇷', name: 'S. America · Brazil' },
  sg: { flag: '🇸🇬', name: 'Asia · Singapore' },
  hk: { flag: '🇭🇰', name: 'Asia · Hong Kong' },
};

// The pools the app can mine on, each with its servers. Settings → Connection
// picks one pool and server for both coins: Pearl on the GPU and Monero on the
// CPU. HeroMiners is the default. Fees are each pool's own: HeroMiners'
// /api/stats (Pearl 0, Monero 0.9) and Kryptex's /{prl,xmr}/api/v1/pool/info
// (Pearl 3, Monero 1, both pay-per-share), read on 2026-10-20.
const POOLS = {
  herominers: { name: 'HeroMiners', prlFeePct: 0, xmrFeePct: 0.9, regions: REGIONS },
  kryptex: { name: 'Kryptex', prlFeePct: 3, xmrFeePct: 1, regions: KRYPTEX_REGIONS },
};
const DEFAULT_POOL = 'herominers';

// Kryptex's ports and its balance API (shared/balance). The same stratum
// protocol as HeroMiners for both coins; the login differences are in the two
// stratum clients (isKryptexHost).
const KRYPTEX = {
  prlPort: 7048,
  xmrPort: 7029,
  apiBase: 'https://pool.kryptex.com',
};

const DEFAULTS = {
  region: 'us',
  worker: 'rig01',
};

// The fees that come off every earnings estimate (ECON.FEE below).
//
// The dev fee is zero and there is no dev-address code path: the engine is our
// own CUDA core (earn/native), written against the ISC-licensed reference, not a
// derivative of any fee-bearing miner. The pool fee is the default pool's; a
// rig on another pool gets that pool's (feeShare below).
const MINER = {
  devFeePct: 0,
  poolFeePct: POOLS[DEFAULT_POOL].prlFeePct,
};

// Monero on the CPU, opt-in and off by default (main/xmrEngine.js drives the
// worker built from earn/xmr-worker), on HeroMiners unless Kryptex is picked.
// From HeroMiners'
// https://monero.herominers.com/api/stats on 2026-10-11: coinUnits 1e12, which
// turns the balance API's atomic units into XMR. Port 1111 starts at difficulty
// 480,000; 1112 is its NiceHash port. The host follows the Pearl region
// (xmrEndpointFor below).
const XMR = {
  port: 1111,
  apiBase: 'https://monero.herominers.com',
  coinUnits: 1e12,
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

// Network economics for earnings estimates (see shared/earnings.js). The app
// live-refreshes these from the prlscan API at runtime (see shared/economics.js
// + main.js); the constants here are only the offline fallback, so keep them
// roughly current — a stale fallback silently overstates earnings (a network
// whose difficulty doubled makes every estimate ~2x too high).
// Snapshot: 2026-11-11 (prlscan, block 141674). The one before, of 2026-10-06, read 34.45e6,
// 2278 and 1.14: in five weeks difficulty rose 66%, and the fallback overstated a day's
// earnings by about 40%.
const ECON = {
  DIFFICULTY: 57.26e6, // network difficulty of the latest block
  BLOCK_REWARD: 2177, // PRL per block
  // The share of the work that pays you: what is left after the dev fee and the
  // pool fee, so a fee change in MINER reaches every estimate.
  FEE: 1 - (MINER.devFeePct + MINER.poolFeePct) / 100,
  PRL_USD: 1.42, // PRL price in USD — prlscan (CoinMarketCap-sourced)
};

// prlscan API endpoints the app live-refreshes economics from (CORS-open; the
// explorer's own backend). Mirrors the site calculator's sources (site/pages/index.html).
const ECON_API = {
  price: 'https://api.prlscan.com/v1/market/prl',
  blocks: 'https://api.prlscan.com/v1/blocks?limit=1',
};

// Local LLM inference (llama.cpp `llama-server`), run alongside the miner. It's
// an OpenAI-compatible HTTP server we spawn like the mining engine; the same
// endpoint powers both the local API and (later) the job queue.
//
// The VRAM kept free for the miner (see LLM.miningReserveMb for how it was
// measured). Named so a tier meant to run beside the miner can use it as its
// own minReserveMb.
const MINING_RESERVE_MB = 2600;

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
  // Demand-driven auto mode (see shared/llmGate). The gate owns this port and
  // keeps llama-server behind it, so callers have one stable address whether the
  // card is currently mining or serving.
  //
  // 8000 rather than the LLM's own 8080: the gate is the public endpoint and the
  // model stays where it already listens, so nothing that talks to llama-server
  // directly has to move.
  //
  // 60s of quiet before handing the card back. Long enough that a conversational
  // caller does not pay the ~4s reload between turns, short enough that a node
  // nobody is calling is mining within a minute. The reload cost is real: a
  // restart drops the prompt cache, so a follow-up in a long conversation
  // re-prefills.
  //
  // "quiet", not "idle": the node is never idle. It is running the card flat out
  // either way -- the axis is whether anything is ASKING for tokens, not whether
  // the GPU has work.
  gate: { port: 8000, quietMs: 60000 },
  // Pausing mining while a co-running LLM answers a prompt (shared/promptPause).
  // A card stops mining as soon as its llama-server starts a request, and mines
  // again `resumeAfterMs` after the last one ends: long enough that a chat turn
  // or the next cluster job does not stop and start the miner every few
  // seconds, short enough that a node nobody is calling is mining again almost
  // at once. A resume reopens the card's core, so it is not free.
  //
  // `maxPauseMs` is a safety net. A card paused this long mines again whatever
  // the log says, so a missed "stop processing" line cannot stop mining for
  // good. Longer than any real request: the gateway gives up at 280 s, and the
  // longest generation seen was ~454 s.
  promptPause: { resumeAfterMs: 10000, maxPauseMs: 10 * 60 * 1000 },
  // What to keep free for the miner when deciding whether a model can co-run.
  //
  // Measured, not guessed: the rank-128 profile asks for 2,081 MiB and the CUDA
  // context it runs in costs a further ~500, so ~2,581 in practice. 2,048 was
  // optimistic by exactly that ~500, and on a card near the boundary it decides
  // the wrong way -- a 5090 with 32,589 MiB free computes Qwen3.8 as co-runnable
  // (30,150 + 2,048 = 32,198) when the real total is 32,731 and does not fit. The
  // node then starts both and one of them loses the allocation.
  //
  // Rounded up to 2,600 for a little headroom. Raising it only ever makes the
  // co-run decision more conservative, which is the safe direction: the failure
  // it prevents is an OOM, the failure it can cause is idling a little VRAM.
  //
  // An AMD core needs ~1,137 MiB at k 2048, plus its HIP context. On Linux an
  // RDNA 4 card mines k 4096 (~1,969 MiB), and a gfx1201 card with 56+ CUs or
  // a gfx1200 card with 12 GB or more k 8192 (~3,633 MiB: RX 9070, RX 9070 XT,
  // R9700, the 16 GB RX 9060 XT). The Windows AMD core mines
  // k 2048 on every card: its build leaves out the wrap readout that larger k
  // needs (build-amd.sh, KWRAP_OS). This reserve never sizes an LLM on those
  // cards: VRAM is read through nvidia-smi only, so on an AMD card the model is
  // the smallest one (~4 GB), which fits beside k 8192 on 16 GB and up. Short
  // of VRAM, the Linux core steps down to k 4096, then k 2048, and the "mining
  // on GPU" line says why (pearl_host_card_k).
  miningReserveMb: MINING_RESERVE_MB,
  // llama-server binary per platform (bundled/downloaded like the miner engine).
  serverBin: { win32: 'llama-server.exe', linux: 'llama-server', darwin: 'llama-server' },
  // Where to fetch the llama-server build if it isn't bundled. llama.cpp embeds
  // the build number in the asset name, so a fixed `latest/download/<name>` 404s
  // — these are pinned to a specific build (old releases keep their assets, so
  // the pins stay resolvable). Windows/Linux use the **Vulkan** archive: a single
  // self-contained bundle (no separate CUDA `cudart` package) that runs
  // GPU-accelerated on NVIDIA/AMD/Intel. On NVIDIA cards it is the fallback: the
  // CUDA build below is tried first. macOS uses the Metal builds, which are
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
  // The CUDA build of the same llama.cpp tag. It is the first choice on NVIDIA
  // cards, on Windows and Linux; the Vulkan build above is the fallback when
  // the CUDA build can't be fetched or lists no NVIDIA card (a driver too old
  // for it, say). The founder's call on 24 Nov 2026, from a user's RTX PRO 4500
  // Blackwell: Qwen3.8-27B decoded at ~42 tok/s on our Vulkan build and ~55 on
  // CUDA. CUDA also prefills 7x faster on the 27B (see the tier below), and it
  // runs on Linux containers that have no NVIDIA Vulkan driver, such as
  // Salad's (WSL2), where the Vulkan build ran Gemma on the CPU.
  //
  // Linux: llama.cpp publishes no CUDA build, so we build one
  // (scripts/build-llama-cuda.sh, CUDA 12.8) and attach it to each release as
  // `asset`.
  //
  // Windows: llama.cpp's own CUDA 13.3 build, plus its separate CUDA runtime
  // zip (cudart, cuBLAS), which is unpacked first. 13.3 has native code for
  // RTX 30, 40 and 50 cards; the 12.4 build would leave Blackwell to compile
  // PTX at first start. It needs a driver new enough for CUDA 13; an older
  // one lists no card and the Vulkan build runs instead.
  //
  // `dir` is where it is unpacked under the LLM folder, named for the tag, so a
  // new tag downloads afresh instead of reusing the old binary.
  cudaServer: {
    tag: 'b9902',
    asset: 'llama-b9902-bin-ubuntu-cuda-12.8-x64.tar.gz',
    dir: 'cuda-b9902',
    win32: {
      url: 'https://github.com/ggml-org/llama.cpp/releases/download/b9902/llama-b9902-bin-win-cuda-13.3-x64.zip',
      // Unpacked before the server, so a server on disk means its runtime is
      // there too. `file` is how an install is checked for it.
      runtime: {
        url: 'https://github.com/ggml-org/llama.cpp/releases/download/b9902/cudart-llama-bin-win-cuda-13.3-x64.zip',
        file: 'cudart64_13.dll',
      },
    },
  },
  // Models the fleet can serve, biggest first. A node runs exactly ONE of these:
  // shared/models.pickModel walks this list and takes the first whose VRAM floor
  // the chosen card clears. Nothing is split across cards and nothing is
  // partially offloaded — see vram.js for why. The founder's call of
  // 24 Nov 2026, which replaced the IQ4_XS tier of #327:
  //
  //   32 GB, GPU to itself   Qwen3.8-27B Q5_K_XL, 131K, vision, MTP
  //   32 GB, mining          Qwen3.8-27B Q4_K_XL, 131K, vision, MTP
  //   smaller                Gemma
  //
  // Order is capability order, not preference-by-size for its own sake: the list
  // is walked top-down and the first fit wins.
  //
  // A tier meant to run BESIDE the miner carries `minReserveMb:
  // MINING_RESERVE_MB`. vram.requiredFreeMb then offers it only where it fits
  // next to the miner, even when no miner runs, so the cards that get it are
  // the same in every mode.
  tiers: [
    // Qwen3.8-27B (Alibaba, Apache-2.0, released 2026-08-14): 27B dense with a
    // vision tower and a 262,144-token native context, at Unsloth's UD-Q5_K_XL
    // quant. Vision needs the separate mmproj projector alongside the weights —
    // llama-server loads it with --mmproj and without it the model runs
    // text-only.
    //
    // These are a user's settings, copied whole on the founder's call of
    // 24 Nov 2026. Aushyro runs this on an RTX PRO 4500 Blackwell (32 GB):
    // UD-Q5_K_XL, the projector on, 131,072 context, MTP on (70 of 159 drafted
    // tokens accepted, 44%), and the CUDA build of llama-server, which took
    // decode from ~42 tok/s on our Vulkan build to ~55. It replaced the
    // Q4_K_XL tier at 262,144 context.
    //
    // File sizes are READ FROM THE HUB: the weights are 20,976,962,464 bytes
    // (20,005 MiB) and mmproj-F16 927,607,488 (885 MiB).
    {
      key: 'qwen3.8-27b',
      name: 'Qwen3.8-27B-UD-Q5_K_XL',
      file: 'Qwen3.8-27B-UD-Q5_K_XL.gguf',
      url: 'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q5_K_XL.gguf',
      // From the Hub's file listing (LFS oid). Not checked at download time; kept
      // so a node's copy can be checked by hand.
      bytes: 20976962464,
      sha256: 'a9c4a1d9f7298dfb00c95dd83241530bc07b141ee01ec6962cd31b39d746837c',
      // The vision projector. Separate download, separate file, loaded with
      // --mmproj; the weights alone are a text-only model.
      mmproj: {
        file: 'Qwen3.8-27B-mmproj-F16.gguf',
        url: 'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-F16.gguf',
      },
      vision: true,
      // The smallest completion budget this model can actually answer within.
      //
      // A reasoning model spends its budget on the <think> block BEFORE it writes
      // a word of the answer, and both halves bill against the same max_tokens.
      // Measured on a 5090: max_tokens 60 returned content "" -- the thinking used
      // all 60 -- while the identical request at 600 answered correctly. An empty
      // string is strictly worse than a truncated answer, because there is nothing
      // in it to salvage or even to tell the caller what went wrong.
      //
      // 512 is ~3x the 166 completion tokens a one-line answer actually cost end
      // to end, so the thinking half can run long and still leave the answer room.
      // Deliberately absent from Gemma, which does not reason: flooring it there
      // would buy nothing and would make `max_tokens: 20` generate 512.
      minCompletionTokens: 512,
      ctxSize: 131072,
      // Tried in order when the one above fails to start. Each step roughly
      // halves the KV cache, so a card that is short by a little gets a working
      // server rather than a restart loop. Only the top rung has a measured VRAM
      // figure; the rest are the fallback path, not a promise.
      ctxLadder: [131072, 65536, 32768],
      // Carried per-model rather than globally because they are specific to it:
      //
      //   --cache-type-k/v q8_0  quantises the KV cache, halving it. The VRAM
      //                          figure below is only valid with it.
      //   -fa 1                  flash attention; required for the quantised
      //                          cache to be worth anything at this context.
      //   --kv-unified           one shared cache rather than per-slot.
      //   --jinja                use the model's own chat template. Without it
      //                          the served output does not match what the model
      //                          was tuned for.
      //   --spec-type draft-mtp  self-speculative decoding via the MTP head that
      //   --spec-draft-n-max 3   ships INSIDE this GGUF, so there is no draft
      //                          model to download. The Q5_K_XL header (read from
      //                          the Hub) has qwen35.nextn_predict_layers and the
      //                          blk.64.nextn tensors. The repo's separate
      //                          MTP/mtp-Qwen3.8-27B-Q4_0.gguf repeats that layer
      //                          and is not downloaded.
      //
      // Deliberately NOT set: `-n`, which caps a reply. Capping generation
      // silently truncates reasoning, and our jobs carry an explicit max_tokens.
      extraArgs: [
        '-fa', '1',
        '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
        '--kv-unified',
        '--jinja',
        '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
      ],
      // MEASURED by the user above on the CUDA build: 29.1 GB of his 32 GB card
      // in use at 131,072 context with the projector and MTP on. 29.1 GiB is
      // 29,798 MiB. That is the whole card, so it may hold a little besides
      // llama-server; it is used as is, which errs towards gating a card rather
      // than OOMing one.
      //
      // For comparison, the old Q4_K_XL tier's five-point sweep on a 5090
      // (fixed 18,967 MiB plus 0.042659 MiB a token of q8_0 KV), with the
      // weights swapped for these, predicts ~27,700 MiB for llama-server alone.
      //
      // It does NOT fit beside the miner: 29,800 plus the 2,600 mining reserve
      // is 32,400 of the 32,607 MiB a 5090 reports, and CUDA sees only ~32,149
      // of those. So a 32 GB card serves this with the GPU to itself (LLM-only
      // mode, CLI auto's demand mode) and the Q4_K_XL tier below while it mines.
      // A card with room for both, like a 96 GB one, co-runs it.
      vramFullMb: 29800,
      minVramMb: 30370,   // measurement + ~570 MiB so we never spawn at the edge.
      quant: 'Q5_K_XL',
    },
    // The same Qwen3.8-27B at UD-Q4_K_XL, with the same settings as the tier
    // above, for a 32 GB card that mines (founder, 24 Nov 2026). Q5_K_XL plus
    // the miner does not fit 32 GB; this does.
    //
    // VRAM: the old Q4_K_XL tier's sweep on a 5090 measured 24,518 MiB at
    // 131,072 with the projector and MTP on (CUDA b10453, no miner). Not yet
    // measured BESIDE the miner, nor on the pinned b9902 build. 25,100 is that
    // plus ~580 MiB. With the 2,600 reserve it needs 27,700 free, which a
    // 32 GB card has and a 24 GB card does not.
    {
      key: 'qwen3.8-27b-corun',
      name: 'Qwen3.8-27B-UD-Q4_K_XL',
      file: 'Qwen3.8-27B-UD-Q4_K_XL.gguf',
      url: 'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf',
      bytes: 17659407264,
      sha256: '2a5f3565f39cf885103da591182c1266da64745737b42b452fd3e675369f1211',
      // The same projector file as the tier above, so a card that has one has both.
      mmproj: {
        file: 'Qwen3.8-27B-mmproj-F16.gguf',
        url: 'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-F16.gguf',
      },
      vision: true,
      minCompletionTokens: 512,
      ctxSize: 131072,
      ctxLadder: [131072, 65536, 32768],
      // The flags of the tier above. This GGUF carries the MTP layer too.
      extraArgs: [
        '-fa', '1',
        '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
        '--kv-unified',
        '--jinja',
        '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
      ],
      vramFullMb: 25100,
      minVramMb: 25670,
      // Offered only where it fits beside the miner, even with no miner (see
      // the note above the list). A card with the GPU to itself gets Q5_K_XL.
      minReserveMb: MINING_RESERVE_MB,
      quant: 'Q4_K_XL',
    },
  ],
  // A small, capable model to start with: Google Gemma 4 E4B Instruct, Q4_K_M
  // GGUF (~5 GB). "E4B" = ~4.5B *effective* params via Per-Layer Embeddings, so
  // it keeps a low VRAM footprint (runs in ~5 GB at 4-bit) while adding 128K
  // context, tool-calling, and 140+ languages — a good default that co-runs with
  // mining without hogging the GPU. `vramFullMb` is the VRAM for a full GPU
  // offload at ctxSize (weights + KV cache). `minVramMb` is the hard floor of free VRAM we require before
  // starting the model on the GPU — above the full-offload figure so we never
  // spawn llama-server right at the edge and OOM.
  //
  // Still the fallback every node lands on when it cannot host a tier above, and
  // still what `LLM.model` means everywhere that reads it.
  model: {
    name: 'Gemma-4-E4B-it-Q4_K_M',
    file: 'gemma-4-E4B-it-Q4_K_M.gguf',
    url: 'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf',
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
  dashboardUrl: 'https://llmjob-production.up.railway.app/dashboard',
  pingIntervalMs: 5 * 60 * 1000,
};


function regionFor(region) {
  return REGIONS[region] || REGIONS[DEFAULTS.region];
}

// The pool id for whatever was saved: a known one, else the default. A
// settings file from before the choice has none.
function poolId(pool) {
  return Object.prototype.hasOwnProperty.call(POOLS, pool) ? pool : DEFAULT_POOL;
}

// A server the pool has, for whatever was saved: on HeroMiners through
// migrateRegion, on another pool as saved, else that pool's first.
function liveRegion(pool, region) {
  const id = poolId(pool);
  if (id === DEFAULT_POOL) return migrateRegion(region);
  const regions = POOLS[id].regions;
  return Object.prototype.hasOwnProperty.call(regions, region) ? region : Object.keys(regions)[0];
}

// The Pearl endpoint for a server on a pool.
function endpointFor(region, pool) {
  if (poolId(pool) === 'kryptex') return 'prl-' + liveRegion(pool, region) + '.kryptex.network:' + KRYPTEX.prlPort;
  return regionFor(region).endpoint;
}

// The share of the Pearl a rig mines that it is paid, after the dev fee and
// the pool's fee: ECON.FEE for the default pool.
function feeShare(pool) {
  return 1 - (MINER.devFeePct + POOLS[poolId(pool)].prlFeePct) / 100;
}

// The Monero pool host for a Pearl region, so the CPU miner picks its pool the
// way the GPU miner does: us2 mines on us2.monero.herominers.com. HeroMiners
// has a Monero name for each of the regions above. On 2026-10-12 they all
// pointed at one server in Frankfurt, so today this changes nothing, but a
// region the pool adds later is used without a release. Kryptex's Monero
// servers have the same names as its Pearl ones.
function xmrEndpointFor(region, pool) {
  if (poolId(pool) === 'kryptex') return 'xmr-' + liveRegion(pool, region) + '.kryptex.network:' + KRYPTEX.xmrPort;
  return (REGIONS[region] ? region : DEFAULTS.region) + '.monero.herominers.com:' + XMR.port;
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
// Deliberately forgiving rather than clever: strip a plain-TCP scheme and
// surrounding whitespace, and hand back null for something empty so the caller
// falls back to the region default instead of spawning a miner pointed at nothing.
function normalizeEndpoint(endpoint) {
  const p = parseEndpoint(endpoint);
  return p ? p.bare : null;
}

// The schemes a pool address may be typed with. Plain TCP ones are dropped. A
// TLS one (ssl://, stratum+ssl://, tls://, stratum+tls://) names a pool's TLS
// port, and the address keeps it as ssl://host:port, so the miner speaks TLS
// there (shared/minerDeps): plain TCP to a TLS port only ever times out. A
// network that blocks a pool's plain port often lets its TLS port out (issue
// #323). Any other scheme (http://) is no address the miner can use.
const PLAIN_SCHEME = /^(stratum\+tcp|stratum|tcp):\/\//i;
const TLS_SCHEME = /^(stratum\+ssl|stratum\+tls|ssl|tls):\/\//i;
const TLS_PREFIX = 'ssl://';
const ANY_SCHEME = /^[a-z][a-z0-9+.-]*:\/\//i;

// Read a pool endpoint as it was typed: --pool, --xmr-pool, the HiveOS flight
// sheet's Pool URL, or a settings file's endpoint. Returns { host, port, bare,
// tls } (`bare` is the text without a plain scheme and trailing slash, and
// with ssl:// in front when `tls`; port is null when there is none, so callers
// decide the default rather than inventing one here), or null for nothing
// usable: empty, or a scheme that is neither plain TCP nor TLS.
// Rightmost colon wins, which keeps a bracketed IPv6 literal intact.
function parseEndpoint(raw) {
  const typed = String(raw == null ? '' : raw).trim();
  const tls = TLS_SCHEME.test(typed);
  const text = typed.replace(tls ? TLS_SCHEME : PLAIN_SCHEME, '');
  if (ANY_SCHEME.test(text)) return null;
  const rest = text.replace(/\/+$/, '').trim();
  if (!rest) return null;
  const bare = (tls ? TLS_PREFIX : '') + rest;
  const m = /^(.*):(\d+)$/.exec(rest);
  return m ? { host: m[1], port: Number(m[2]), bare, tls } : { host: rest, port: null, bare, tls };
}

// How an engine reads the endpoint it is started with (`host:port`, or
// `ssl://host:port`): the host, the port as text, and whether to speak TLS.
function wireEndpoint(endpoint) {
  const text = String(endpoint || '');
  const tls = text.startsWith(TLS_PREFIX);
  const [host, port] = (tls ? text.slice(TLS_PREFIX.length) : text).split(':');
  return { host, port, tls };
}

// Split a `host:port` endpoint into its parts, { host: null, port: null } for
// nothing usable (parseEndpoint).
function splitEndpoint(endpoint) {
  const p = parseEndpoint(endpoint);
  return p ? { host: p.host, port: p.port } : { host: null, port: null };
}

// A pool endpoint that names both a host and a port in range, as `host:port`
// (`ssl://host:port` for a TLS port), or null. A pool has no default port, so
// one without a port is not guessed at.
function hostPort(raw) {
  const p = parseEndpoint(raw);
  if (!p || !p.host || !Number.isInteger(p.port) || p.port < 1 || p.port > 65535) return null;
  return (p.tls ? TLS_PREFIX : '') + p.host + ':' + p.port;
}

// The endpoint to mine against: a normalised override if there is one, else the
// region's on the chosen pool. One function so the argument vector and the
// "connecting to …" log line can never disagree about where the rig is
// actually pointed.
function resolveEndpoint(settings) {
  const s = settings || {};
  return normalizeEndpoint(s.endpoint) || endpointFor(s.region || DEFAULTS.region, s.pool);
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
  REGIONS, KRYPTEX_REGIONS, LEGACY_REGIONS, DEFAULTS, POOLS, DEFAULT_POOL, KRYPTEX, MINER, XMR, NETWORK, ECON, ECON_API,
  LLM, NODE, regionFor, endpointFor, xmrEndpointFor, poolId, liveRegion, feeShare, normalizeEndpoint,
  resolveEndpoint, splitEndpoint, parseEndpoint, wireEndpoint, hostPort, regionLabel, migrateRegion,
};
