'use strict';

const { LLM } = require('./config');
const { ALL_LAYERS } = require('./vram');
const { normalizePciBusId } = require('./gpu');

// Pure helpers for the local llama.cpp `llama-server` (OpenAI-compatible). The
// process spawn/supervision lives in main/llmManager.js; this module just builds
// the command line, resolves paths/URLs, and parses the server's output — all
// unit-testable without a GPU or a running server.

// Resolve the llama-server binary: a configured path wins, else the per-platform
// name (callers pass the install dir; here we only decide the executable name).
function resolveServerBinary(binaryPath, platform) {
  if (binaryPath) return binaryPath;
  return LLM.serverBin[platform] || LLM.serverBin.linux;
}

// Which llama-server archive this machine downloads. Most platforms publish a
// single build, but macOS ships separate Apple-silicon (arm64) and Intel (x64)
// archives, so an arch-qualified key ("darwin-x64") wins over the bare platform
// key when config has one. Handing an Intel Mac the arm64 build would install a
// binary the kernel refuses to exec, and the LLM is the whole point of the Mac
// build. Unknown platforms fall back to Linux, mirroring resolveServerBinary.
function resolveServerUrl(platform, arch) {
  return LLM.serverUrl[platform + '-' + arch] || LLM.serverUrl[platform] || LLM.serverUrl.linux;
}

function serverBaseUrl(opts = {}) {
  return 'http://' + (opts.host || LLM.host) + ':' + (opts.port || LLM.port);
}

// Build the llama-server argument vector. `modelPath` and `nGpuLayers` are the
// per-run bits (the VRAM budgeter picks nGpuLayers, which is all-or-nothing —
// see shared/vram.computeGpuLayers); host/port/ctx default from config.
//
// The default is ALL_LAYERS rather than a layer count of our own: llama.cpp
// clamps an over-large value to what the model really has, whereas a hardcoded
// guess that's too low silently leaves the remainder on the CPU, where the
// weights sit in host RAM and OOM a small-RAM rig.
function buildServerArgs(opts = {}) {
  const ngl = opts.nGpuLayers != null ? opts.nGpuLayers : ALL_LAYERS;
  const args = [
    '--model', opts.modelPath || '',
    '--host', opts.host || LLM.host,
    '--port', String(opts.port || LLM.port),
    '--ctx-size', String(opts.ctxSize || LLM.ctxSize),
    '--n-gpu-layers', String(ngl),
    '--parallel', String(LLM.parallel),
    // Keep the model on ONE GPU. On multi-GPU rigs the Vulkan backend tries to
    // split the graph across every device and can trip
    // GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) in ggml-backend.cpp,
    // crash-looping the server before it ever serves (seen in the field on a
    // 3060 Ti + 5060 Ti box). Our model is ~5 GB — a single card is plenty, and
    // single-GPU machines are unaffected by the flag.
    '--split-mode', 'none',
  ];
  // Pin the model to a specific GPU when the caller picked one. With
  // --split-mode none the model loads on a single device; without a choice
  // that's always device 0, which on a mining rig is busy and may lack the
  // headroom the model needs. The caller (VRAM budgeter) chooses the card with
  // the most free VRAM so the server lands there instead.
  //
  // `device` is llama-server's own name for that card ("Vulkan1"), from
  // placeLlmPlan below, and it wins. --main-gpu counts cards in llama-server's
  // order, which need not be nvidia-smi's, so an index can name another card.
  // With one device named there is only one to pick, and --main-gpu is left
  // out: any index but 0 would then be out of range. Without a name, a
  // non-negative integer `mainGpu` is passed as before; anything else falls
  // back to device 0.
  if (opts.device) {
    args.push('--device', String(opts.device));
  } else if (Number.isInteger(opts.mainGpu) && opts.mainGpu >= 0) {
    args.push('--main-gpu', String(opts.mainGpu));
  }
  // The vision projector, for a multimodal model that ships one. The weights
  // alone load and serve happily as a text-only model, so a missing projector is
  // a silent capability loss rather than a startup failure — which is exactly why
  // the caller passes the path explicitly instead of us inferring it.
  // What the server calls itself. Without it llama-server falls back to the model
  // PATH, so /v1/models and the `model` field of every completion come back as
  // an absolute local filesystem path -- which leaks the node's directory layout
  // to anything consuming the endpoint and does not match the model name the
  // fleet reports elsewhere. Verified on a 5090: without this the id is
  // "/home/.../llm/Qwen3.8-27B-UD-Q4_K_XL.gguf".
  if (opts.alias) args.push('--alias', String(opts.alias));
  if (opts.mmprojPath) args.push('--mmproj', String(opts.mmprojPath));
  // Per-model flags, appended last so a model can override an earlier default.
  // A large model is not simply a bigger version of a small one: Qwen3.8 needs a
  // quantised KV cache to fit its context at all, its own chat template to
  // produce output matching what it was tuned for, and can self-speculate from a
  // head inside its own GGUF. None of that belongs in the shared arg list, and
  // none of it can be inferred — it comes from the model entry in config.
  if (Array.isArray(opts.extraArgs)) {
    for (const a of opts.extraArgs) if (a != null && a !== '') args.push(String(a));
  }
  return args;
}

// The environment for a llama-server, or null to inherit ours unchanged.
//
// The vision projector does not follow --device. mtmd loads it on the card
// named by MTMD_BACKEND_DEVICE, or else on the first GPU in llama-server's list
// (b9902, tools/mtmd/clip.cpp). On a rig whose first card is not the model's,
// that put ~1 GB of projector on another card. So an instance with a device and
// a projector gets MTMD_BACKEND_DEVICE set to the same device. A name mtmd
// doesn't know only logs a warning and falls back to the first GPU. A value
// already set is the user's, and is kept.
function buildServerEnv(opts = {}, env = {}) {
  if (!opts.device || !opts.mmprojPath || env.MTMD_BACKEND_DEVICE != null) return null;
  return Object.assign({}, env, { MTMD_BACKEND_DEVICE: String(opts.device) });
}

// Flip the manager to "ready" only on lines llama-server prints AFTER the model
// has loaded ("model loaded", "server is listening on … - starting the main
// loop", "all slots are idle"). The earlier "main: HTTP server is listening"
// line appears BEFORE the multi-GB model loads — while /v1/chat/completions
// still returns 503 — so it must NOT count as ready.
function isServerReady(line) {
  return /model loaded|starting the main loop|all slots are idle/i.test(String(line == null ? '' : line));
}

// llama-server prints TWO timing lines per request:
//
//   prompt eval time = ...  1840.00 tokens per second   <- prefill
//          eval time = ...   162.02 tokens per second   <- generation
//
// A single /tokens per second/ regex matched both, so whichever printed last
// won and the two were reported under one name. They measure different things
// and routinely differ by an order of magnitude, which made the number we
// showed -- and sent to the network board -- meaningless.
//
// `tokens` is how many tokens the rate was measured over ("/ 270 tokens"), or
// null when the line doesn't say. The in-app chat picks a card by its measured
// rate (LlmFleet.chatUrl), and a 4-token warm-up reply or a progress line
// printed mid-prefill is no measure of a card.
//
// Returns { kind: 'prompt'|'gen', tokensPerSec, tokens } or null.
function parseTiming(line) {
  const s = String(line == null ? '' : line);
  const m = s.match(/([\d.]+)\s*tokens per second/i);
  if (!m) return null;
  const n = s.match(/\/\s*(\d+)\s+tokens\b/i);
  return {
    kind: /prompt eval/i.test(s) ? 'prompt' : 'gen',
    tokensPerSec: Number(m[1]),
    tokens: n ? Number(n[1]) : null,
  };
}

// A slot starting or finishing a request, from llama-server's own log. Every
// prompt shows up here, whichever way it arrived: the API tab's address, the
// built-in web UI, cluster jobs, the LAN share, or another app on the machine.
// Most of those never pass through our code, so this is the one place that sees
// them all. The lines (b9902, tools/server/server-context.cpp) are:
//
//   slot launch_slot_: id  0 | task 12 | processing task, is_child = 0
//   slot      release: id  0 | task 12 | stop processing: n_tokens = 250, truncated = 0
//
// Every "processing task" has exactly one "stop processing": release() is the
// only place a slot goes idle, cancelled requests included. Both print at the
// default INFO level. Pinned by a test, so a llama.cpp upgrade that rewords them
// fails CI instead of leaving mining paused.
//
// Returns { slot, busy } or null.
function parseSlotEvent(line) {
  const m = String(line == null ? '' : line)
    .match(/\bid\s+(\d+)\s*\|\s*task\s+-?\d+\s*\|\s*(processing task|stop processing)\b/);
  if (!m) return null;
  return { slot: Number(m[1]), busy: m[2] === 'processing task' };
}

// The GPU dropped out from under a running llama-server: the driver reset the
// card, or lost it. The Vulkan build (what the app ships on Windows and Linux)
// throws vk::DeviceLostError, and the server catches it per request, logs it,
// fails the request (a 500, or an error at the end of a reply that was already
// streaming) and carries on. It never recovers in-process (ggml-vulkan keeps
// one device for the life of the process), so from then on every request fails
// while /health still says ok. Only a new process gets the card back. The
// macOS Metal build reports a lost GPU differently and isn't covered here. The
// lines (b9902, tools/server/server-context.cpp, after the log's timestamp) are:
//
//   E srv  update_slots: decode() failed: vk::Device::waitForFences: ErrorDeviceLost
//   E slot      iterate: id  0 | task 14 | got exception: vk::Queue::submit: ErrorDeviceLost
//
// The first is the request that hit it; the second is each one after, which
// fails on submit. Anchored to those two shapes, so the follow-up "send_error"
// line (it repeats the message) and anything that merely quotes it, such as a
// request echoed into the log, do not count. Pinned by a test, like
// parseSlotEvent.
function isDeviceLost(line) {
  return /^(?:\d+\.\d{2}\.\d{3}\.\d{3} )?(?:E )?(?:srv +update_slots: (?:pre_|post_)?decode\(\) failed|slot +iterate: id +\d+ \| task -?\d+ \| got exception): vk::\w+::\w+: ErrorDeviceLost\s*$/
    .test(String(line == null ? '' : line));
}

// A card llama-server loads on, and its PCI bus id, from the line it prints
// for each one while loading a model (b9902, src/llama.cpp):
//
//   llama_prepare_model_devices: using device Vulkan0 (NVIDIA GeForce RTX 3090) (0000:02:00.0) - 23893 MiB free
//
// With --split-mode none there is one such line. b9902 prints it only at -lv 4,
// since it counts llama.cpp's own info lines as trace, so the servers the app
// runs never show it. probe.detectLlmDevices runs a short one that does.
// Returns { device, busId }, or null when the backend did not know the bus id
// ("unknown id") or the line is anything else.
function parseUsingDeviceLine(line) {
  const m = String(line == null ? '' : line)
    .match(/using device (\S+) \(.*\) \(([0-9a-fA-F]+:[0-9a-fA-F]+:[0-9a-fA-F]+\.[0-9a-fA-F])\) - \d+ MiB free/);
  return m ? { device: m[1], busId: m[2] } : null;
}

// The bus id alone (parseUsingDeviceLine), for a server that prints one.
function parseUsingDevice(line) {
  const e = parseUsingDeviceLine(line);
  return e ? e.busId : null;
}

// The cards llama-server can load on, from `llama-server --list-devices`
// (b9902, common/arg.cpp), as [{ device, name }] in its order:
//
//   Available devices:
//     Vulkan0: NVIDIA GeForce RTX 4070 (11874 MiB, 11200 MiB free)
//     Vulkan1: NVIDIA RTX PRO 4500 Blackwell (32111 MiB, 31500 MiB free)
//
// `device` is what --device takes. Any other line is skipped.
function parseDeviceList(out) {
  const list = [];
  for (const line of String(out == null ? '' : out).split(/\r?\n/)) {
    const m = line.match(/^\s*(\S+): (.*) \(\d+ MiB, \d+ MiB free\)\s*$/);
    if (m) list.push({ device: m[1], name: m[2].trim() });
  }
  return list;
}

// Which of llama-server's cards is each nvidia-smi card, as { index: device }.
// `devices` is probe.detectLlmDevices's list, `busIds` nvidia-smi's
// { index: pciBusId } (probe.detectPciBusIds).
//
// llama-server ships as a Vulkan build, and Vulkan numbers cards its own way:
// on Windows the card driving the desktop comes first. So --main-gpu with
// nvidia-smi's index could load the model on another card. On a rig with a
// 32 GB RTX PRO 4500 (GPU 0) and a 12 GB RTX 4070 (Vulkan0), the 27B model
// planned for the PRO 4500 went to the 4070, ran out of memory, and never
// started.
//
// Cards are matched by PCI bus id first, where both sides have one: that is
// exact, cards of the same name included. A device carries a bus id when the
// probe read one (more than one card listed, and a model on disk to load).
//
// The rest are matched by name, ignoring case and punctuation. A card and a
// device that both have a bus id, and differ, are never paired. Cards of the
// same name are paired in order, which is only a guess: nothing in the list
// tells them apart (its free-memory figure leaves out other programs on
// Windows), so placeLlmPlan warns when it matters.
//
// When one NVIDIA card is left over on each side, and both sides list as many
// NVIDIA cards, those two are paired, so a name spelled differently still
// matches. Only NVIDIA devices count there: llama-server also lists
// integrated GPUs and other makers' cards, which nvidia-smi does not, and
// pairing a card nvidia-smi lists but Vulkan can't see (a card in TCC mode,
// say) with an iGPU would put the model in shared system memory. A card with
// no match is left out of the map.
function matchLlmDevices(cards, devices, busIds) {
  const gpus = (Array.isArray(cards) ? cards : [])
    .filter((c) => c && Number.isInteger(c.index))
    .sort((a, b) => a.index - b.index);
  const devs = (Array.isArray(devices) ? devices : []).filter((d) => d && d.device);
  const cardBus = (c) => normalizePciBusId(busIds && busIds[c.index]);
  const devBus = (d) => normalizePciBusId(d.busId);
  const open = (c, d) => !(cardBus(c) && devBus(d));
  const map = {};
  const taken = new Set();
  for (const c of gpus) {
    const d = cardBus(c) && devs.find((x) => !taken.has(x.device) && devBus(x) === cardBus(c));
    if (d) { map[c.index] = d.device; taken.add(d.device); }
  }
  const rest = gpus.filter((c) => map[c.index] === undefined);
  for (const name of new Set(rest.map((c) => nameKey(c.name)).filter(Boolean))) {
    const mine = rest.filter((c) => nameKey(c.name) === name);
    const theirs = devs.filter((d) => !taken.has(d.device) && nameKey(d.name) === name);
    if (mine.length !== theirs.length || !mine.every((c, i) => open(c, theirs[i]))) continue;
    mine.forEach((c, i) => { map[c.index] = theirs[i].device; taken.add(theirs[i].device); });
  }
  const nvidia = devs.filter((d) => nameKey(d.name).includes('nvidia'));
  const leftCards = gpus.filter((c) => map[c.index] === undefined);
  const leftDevs = nvidia.filter((d) => !taken.has(d.device));
  if (gpus.length === nvidia.length && leftCards.length === 1 && leftDevs.length === 1
    && open(leftCards[0], leftDevs[0])) {
    map[leftCards[0].index] = leftDevs[0].device;
  }
  return map;
}

// A card's name as matchLlmDevices compares it: lower case, letters and digits.
function nameKey(s) {
  return String(s == null ? '' : s).toLowerCase().replace(/[^a-z0-9]/g, '');
}

// The plan with each card's llama-server device set (matchLlmDevices), and
// the warnings that go with it. A card with no device keeps --main-gpu, as
// before.
//
// A placed entry also gets `busId`, nvidia-smi's bus id for its card, when the
// card it loads on is certain: matched by bus id, or by a name no other card
// here has. Prompt pause needs it to know which card to stop mining on, since
// the server's own "using device" line is not printed at the log level the
// app runs it at (parseUsingDeviceLine).
// Twins paired by name get none, and a prompt on a rig of several cards then
// pauses nothing (promptPause.cardForPrompt).
//
// The warnings:
//
//   * llama-server listed no cards (it would not run, say). One line, so a
//     failed listing shows even on a one-card rig.
//   * a planned card had no match. On a one-card rig too: a card Vulkan can't
//     see (one in TCC mode, say) beside an iGPU or another maker's card sends
//     --main-gpu 0 there, since llama.cpp uses an iGPU when no discrete card
//     is listed.
//   * a planned card, paired by name, shares it with a card the plan left out
//     that has less free memory. Same-name cards are paired in order, so it
//     may be on that one, such as the card driving the desktop. A twin left out
//     with as much room (by the CLI's --llm-max-instances) holds the model just
//     as well. A card matched by bus id is never in doubt.
function placeLlmPlan(plan, cards, devices, busIds) {
  const list = (Array.isArray(cards) ? cards : []).filter(Boolean);
  const entries = Array.isArray(plan) ? plan : [];
  const listed = Array.isArray(devices) && devices.length > 0;
  const map = matchLlmDevices(list, devices, busIds);
  const planned = new Set(entries.map((e) => e.index));
  const warnings = [];
  if (!listed && entries.some((e) => e.index != null)) {
    warnings.push('llama-server did not list its cards, so the local LLM is placed by GPU number'
      + ' (--main-gpu), which may be a different card');
  }
  const describe = (card, index) => 'GPU ' + index + (card && card.name ? ' (' + card.name + ')' : '');
  // Unmeasured reads as no room, so the warning errs on the side of showing.
  const roomy = (c, e) => Number(c.totalMb) - Number(c.usedMb) >= Number(e.freeMb);
  const placed = entries.map((e) => {
    if (e.index == null) return e;
    const card = list.find((c) => c.index === e.index);
    if (map[e.index]) {
      const bus = busIds && busIds[e.index];
      const dev = devices.find((d) => d && d.device === map[e.index]);
      const byBus = !!normalizePciBusId(bus) && normalizePciBusId(bus) === normalizePciBusId(dev.busId);
      const twin = (c) => c.index !== e.index && nameKey(c.name) === nameKey(card.name);
      const twins = !byBus && !!card && !!nameKey(card.name) && list.some(twin);
      if (twins && list.some((c) => twin(c) && !planned.has(c.index) && !roomy(c, e))) {
        warnings.push(describe(card, e.index) + ' has the same name as another card here, so llama-server'
          + ' may load the local LLM on that one instead');
      }
      const out = Object.assign({}, e, { device: map[e.index] });
      if (!twins && normalizePciBusId(bus)) out.busId = bus;
      return out;
    }
    if (listed) {
      warnings.push('could not tell which of llama-server\'s cards is ' + describe(card, e.index)
        + ', so the local LLM may load on a different card');
    }
    return e;
  });
  return { plan: placed, warnings };
}

// The log line for a fleet start: "local LLM starting on 1 GPU [0]". When
// llama-server numbers a card differently, its names follow, so a log shows
// which card the model went to: "[0] (llama-server calls it Vulkan1)".
function describeLlmStart(plan) {
  const list = Array.isArray(plan) ? plan : [];
  const gpus = list.map((p) => (p.index == null ? 'auto' : p.index)).join(', ');
  let line = 'local LLM starting on ' + list.length + ' GPU' + (list.length === 1 ? '' : 's') + ' [' + gpus + ']';
  if (list.some((p) => p.device && String(p.device).replace(/^\D+/, '') !== String(p.index))) {
    line += ' (llama-server calls ' + (list.length === 1 ? 'it ' : 'them ')
      + list.map((p) => p.device || '?').join(', ') + ')';
  }
  return line;
}

// What a running llama-server says it serves, from its /v1/models answer:
// { name, ctxSize }. The name is its --alias, which we start it with, so a
// server of ours names its catalogue model. ctxSize is its real window
// (meta.n_ctx), or null if it does not say. Null for anything else.
function parseServedModel(body) {
  const m = body && Array.isArray(body.data) ? body.data[0] : null;
  if (!m || typeof m.id !== 'string') return null;
  const n = Number(m.meta && m.meta.n_ctx);
  return { name: m.id, ctxSize: n > 0 ? n : null };
}

module.exports = {
  resolveServerBinary, resolveServerUrl, serverBaseUrl, buildServerArgs, buildServerEnv, isServerReady,
  parseTiming, parseSlotEvent, isDeviceLost, parseUsingDevice, parseUsingDeviceLine,
  parseDeviceList, matchLlmDevices, placeLlmPlan, describeLlmStart, parseServedModel,
};
