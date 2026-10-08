'use strict';

// Pure helper for GPU auto-detection. main.js runs the system query
// (Win32_VideoController) and feeds the raw adapter-name list here; keeping the
// selection logic here makes it unit-testable without touching the OS.

// Virtual / basic display adapters that aren't real mining GPUs.
const IGNORE = /basic display|microsoft basic|remote|rdp|virtual|meta|parsec|citrix|vmware|oray/i;

// Integrated GPUs (iGPU / APU). They enumerate alongside a discrete card — often
// first — but mine at a fraction of its rate, so they're a last resort. Matches
// Intel iGPUs (UHD/HD/Iris) and AMD APUs, whose adapter name is a bare "Radeon
// Graphics" / "…Vega… Graphics" with no discrete model (RX/Pro/Instinct).
const INTEGRATED = /\bintel\b|\buhd\b|\bhd graphics\b|\biris\b|radeon(\(tm\))? graphics|vega.*graphics|integrated/i;

// Pick the best real GPU from a list of adapter names, or null if none. Prefers
// a discrete card over an integrated one (Win32_VideoController may list the
// iGPU before the discrete GPU that actually mines), keeping list order within a
// tier and falling back to an integrated GPU only when it's all that's present.
function pickGpu(names) {
  if (!Array.isArray(names)) return null;
  const real = [];
  for (const raw of names) {
    const name = String(raw == null ? '' : raw).trim();
    if (name && !IGNORE.test(name)) real.push(name);
  }
  if (!real.length) return null;
  return real.find((n) => !INTEGRATED.test(n)) || real[0];
}

// Count the GPUs that actually mine: discrete cards when any are present (an
// iGPU alongside them contributes nothing worth counting), else 1 if only an
// integrated GPU exists, else 0. Multi-GPU rigs report the count on their
// board row, so one entry represents the whole rig rather than a single card.
function countGpus(names) {
  if (!Array.isArray(names)) return 0;
  let real = 0;
  let discrete = 0;
  for (const raw of names) {
    const name = String(raw == null ? '' : raw).trim();
    if (!name || IGNORE.test(name)) continue;
    real++;
    if (!INTEGRATED.test(name)) discrete++;
  }
  return discrete > 0 ? discrete : (real > 0 ? 1 : 0);
}

// Make CUDA number the cards the way nvidia-smi does, by setting
// CUDA_DEVICE_ORDER=PCI_BUS_ID before anything touches the driver.
//
// Everything this app knows about GPUs comes from nvidia-smi, which numbers
// cards by PCI bus: the name on the device label, the per-card VRAM the LLM
// planner budgets against, the temperature the UI shows, the rows on the
// network board. The CUDA runtime does NOT: left alone it orders devices by its
// own "fastest first" heuristic, so its device 1 can be nvidia-smi's device 0.
//
// On a single-card rig the two agree and nothing shows. On a multi-card rig they
// need not, and then an index means different cards on either side of that line:
// the mining core mined on one GPU while the app named another — a 32 GB RTX PRO
// 4500 on screen, an RTX 4070 doing the work (issue #226).
//
// The variable is read when the CUDA driver initialises, which for our core is
// inside this process on the first Start, so it has to be in place before then:
// both shells set it at load. (It does NOT reach the local LLM: llama-server
// ships as a Vulkan build, whose --main-gpu indices come from Vulkan's own
// device enumeration and are a separate question.)
//
// An operator who has already set CUDA_DEVICE_ORDER means it — leave it alone.
// Returns the value now in effect, for the caller to log.
function alignCudaDeviceOrder(env) {
  const e = env || process.env;
  if (!e.CUDA_DEVICE_ORDER) e.CUDA_DEVICE_ORDER = 'PCI_BUS_ID';
  return e.CUDA_DEVICE_ORDER;
}

// Let CUDA see every card nvidia-smi lists, by removing CUDA_VISIBLE_DEVICES
// before anything touches the driver.
//
// That variable hides cards from CUDA and renumbers the ones left. nvidia-smi
// ignores it, and the mining fleet is planned from nvidia-smi's list, so with it
// set the two sides disagree. A two-card rig with CUDA_VISIBLE_DEVICES=0 asked
// the core for GPU 1, CUDA answered that the machine has one card, and the
// second card never mined. Set to 1 instead, it would be worse: the core's GPU 0
// would be nvidia-smi's GPU 1, the wrong-card bug from issue #226 again.
//
// It is usually left over from other software or old troubleshooting, not a
// choice about this app. Choosing cards is what PEARL_GPU_INDEX (and the CLI's
// --gpu-index) is for, and those use the same numbers as everything else here.
//
// Same timing as alignCudaDeviceOrder: both shells call it at load. Returns the
// value it removed, or null when it wasn't set, so the caller can log it.
function clearCudaVisibleDevices(env) {
  const e = env || process.env;
  const was = e.CUDA_VISIBLE_DEVICES;
  if (was == null) return null;
  delete e.CUDA_VISIBLE_DEVICES;
  return was;
}

// The log line for a CUDA_VISIBLE_DEVICES that clearCudaVisibleDevices removed,
// or null when there was nothing to remove. One wording for both shells; the
// CLI passes its own `hint`, since it has a flag for choosing cards.
function describeClearedCuda(was, hint) {
  if (was == null) return null;
  return 'ignoring CUDA_VISIBLE_DEVICES=' + was + ' so every GPU can mine '
    + '(' + (hint || 'set PEARL_GPU_INDEX to mine on one card') + ')';
}

// Parse a list of cards to mine on: nvidia-smi indices separated by commas,
// such as "0,2", or "none" for no card at all. The CLI's --gpu-index and
// PEARL_GPU_INDEX both use this, so "which cards" means one thing.
//
// Returns null for a blank value (no choice made), { indices } for a valid one,
// and { error } otherwise. `indices` is sorted with duplicates dropped, and is
// empty for "none". Spaces around an entry are fine; anything else that isn't
// a whole number from 0 up is an error, so a typo can't read as an instruction.
function parseGpuIndexList(raw) {
  if (raw == null) return null;
  const text = String(raw).trim();
  if (text === '') return null;
  if (text.toLowerCase() === 'none') return { indices: [] };
  const indices = [];
  for (const part of text.split(',')) {
    const entry = part.trim();
    const n = Number(entry);
    if (!/^[0-9]+$/.test(entry) || !Number.isSafeInteger(n)) {
      return { error: 'give GPU numbers from nvidia-smi separated by commas, such as 0,2, or none' };
    }
    if (!indices.includes(n)) indices.push(n);
  }
  indices.sort((a, b) => a - b);
  return { indices };
}

// PEARL_GPU_INDEX as the desktop app reads it: { indices, warning }. `indices`
// is the cards chosen, or null for every card. `warning` is the log line for a
// value that is set but not used, or null.
//
// It is an escape hatch, not the mechanism: normally every card mines. But "it
// picked the wrong card" is the report we cannot reproduce from here, and a rig
// that can pin cards in one env var can answer it in one run. Same idea as
// PEARL_CORE_PATH, and the same place to look for it.
//
// Takes the same list as the CLI's --gpu-index, except "none": the desktop app
// cannot run with no mining card (its LLM compute mode is how it doesn't mine),
// and an empty list means "let the core choose" to planMinerGpus. So none is
// ignored, and so is anything that doesn't parse, with a warning either way.
// The CLI reads the variable itself (earn-cli gpuChoice) and honours none.
function readGpuIndexEnv(env) {
  const raw = (env || process.env).PEARL_GPU_INDEX;
  const parsed = parseGpuIndexList(raw);
  if (!parsed) return { indices: null, warning: null };
  const said = 'PEARL_GPU_INDEX=' + String(raw).trim();
  if (parsed.error) {
    return { indices: null, warning: said + ' ignored (' + parsed.error + '); every GPU mines' };
  }
  if (!parsed.indices.length) {
    return {
      indices: null,
      warning: said + ' ignored: the app cannot mine on no GPU (set Compute Mode to LLM for that); every GPU mines',
    };
  }
  return { indices: parsed.indices, warning: null };
}

// The log lines for chosen cards that planMinerGpus dropped because nvidia-smi
// doesn't list them, one per card, in the same words in both shells. `from`
// names what chose them. Empty when nothing was chosen or nothing was dropped.
function describeSkippedGpus(chosen, gpus, from) {
  if (!Array.isArray(chosen)) return [];
  const mining = (Array.isArray(gpus) ? gpus : []).map((g) => g.index);
  return chosen.filter((i) => !mining.includes(i))
    .map((i) => 'skipping GPU ' + i + ' from ' + from + ': nvidia-smi does not list it');
}

// Which cards mine. One core per card, so this list IS the mining fleet.
//
// `cards` is nvidia-smi's list (parseGpuStats). Every card mines: the miner is
// one CUDA context and a search thread per card, and the LLM planner already
// sets the mining reserve aside on every card, so a rig that mined on one card
// was leaving that reserve unused everywhere else.
//
// `chosen` (PEARL_GPU_INDEX, or the CLI's --gpu-index) narrows it to those
// cards: a non-empty list of indices, or null for every card. An index
// nvidia-smi doesn't list is dropped, as long as another chosen card is listed,
// and both shells log it (describeSkippedGpus). When none of them is, or nvidia-smi listed nothing at
// all, the choice is passed on as it is: the core checks each index against the
// real device count and says so, which beats silently mining on cards nobody
// asked for.
//
// An empty list means nvidia-smi told us nothing (not installed, not NVIDIA)
// and nothing was chosen. The caller then starts a single core with no index
// and lets it choose, which is what a single-card rig did before any of this
// existed.
function planMinerGpus(cards, chosen) {
  const list = [];
  for (const c of (Array.isArray(cards) ? cards : [])) {
    if (!c) continue;
    const index = Math.floor(Number(c.index));
    if (!Number.isFinite(index) || index < 0) continue;
    list.push({ index, name: c.name || null });
  }
  list.sort((a, b) => a.index - b.index);
  if (!Array.isArray(chosen) || !chosen.length) return list;
  const listed = list.filter((g) => chosen.includes(g.index));
  if (listed.length) return listed;
  return chosen.slice().sort((a, b) => a - b).map((index) => ({ index, name: null }));
}

// Parse `nvidia-smi --query-gpu=index,name,memory.used,memory.total
// --format=csv,noheader,nounits` into one entry per card:
//   [{ index, name, usedMb, totalMb }, ...]
// The network board uses this to report each GPU's own VRAM (the limiting
// factor for co-running an LLM) instead of the rig's summed total. Rows that
// don't parse cleanly are skipped; index/used/total are read positionally (the
// name is the middle field and never contains a comma) so a stray column can't
// misalign the numbers.
function parseGpuStats(out) {
  const list = [];
  for (const row of String(out == null ? '' : out).split(/\r?\n/)) {
    const line = row.trim();
    if (!line) continue;
    const parts = line.split(',').map((x) => x.trim());
    if (parts.length < 4) continue;
    const index = parseInt(parts[0], 10);
    const usedMb = parseInt(parts[parts.length - 2], 10);
    const totalMb = parseInt(parts[parts.length - 1], 10);
    if (!Number.isFinite(index) || !Number.isFinite(usedMb) || !Number.isFinite(totalMb)) continue;
    const name = parts.slice(1, parts.length - 2).join(',').trim() || null;
    list.push({ index, name, usedMb, totalMb });
  }
  return list;
}

// Parse `nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader` into a
// map of card index → PCI bus id, e.g. { 0: '00000000:01:00.0' }.
//
// The stats file carries it beside each card's numbers so HiveOS's h-stats.sh
// can tell the dashboard which GPU row each hashrate belongs to. The index is
// the core's CUDA device, which is nvidia-smi's index because both shells pin
// CUDA_DEVICE_ORDER=PCI_BUS_ID (see alignCudaDeviceOrder), so pairing the two
// here is the same pairing the board rows already rely on. Rows that don't
// parse are skipped.
const PCI_BUS_ID_RE = /^[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$/i;

function parsePciBusIds(out) {
  const map = {};
  for (const row of String(out == null ? '' : out).split(/\r?\n/)) {
    const parts = row.split(',').map((x) => x.trim());
    if (parts.length < 2) continue;
    const index = parseInt(parts[0], 10);
    if (!Number.isInteger(index) || !PCI_BUS_ID_RE.test(parts[1])) continue;
    map[index] = parts[1];
  }
  return map;
}

// Parse `system_profiler SPDisplaysDataType -json` into { name, count }, or
// null when nothing usable is in there.
//
// This is the Mac's answer to nvidia-smi. macOS has neither that nor WMI, so
// without it the device label sat at "GPU · auto-detect" on the one platform
// where the GPU is the only thing the app uses at all.
//
// Each entry carries the GPU under `sppci_model` ("Apple M3 Max"), with `_name`
// as the older/alternate key — both are read because the pairing has moved
// between macOS versions and a missed rename would silently cost the label.
// The names then go through the same pickGpu/countGpus the other platforms use,
// which matters on an Intel Mac with both an iGPU and a discrete card: the
// discrete one wins there exactly as it does on Windows.
function parseMacGpu(out) {
  let json;
  try {
    json = JSON.parse(String(out == null ? '' : out));
  } catch (e) {
    return null; // not JSON (an error string, an empty read, a future format)
  }
  const list = json && Array.isArray(json.SPDisplaysDataType) ? json.SPDisplaysDataType : [];
  const names = list.map((d) => (d && (d.sppci_model || d._name)) || '');
  const name = pickGpu(names);
  return name ? { name, count: countGpus(names) } : null;
}

module.exports = {
  IGNORE, INTEGRATED, pickGpu, countGpus, alignCudaDeviceOrder,
  clearCudaVisibleDevices, describeClearedCuda, parseGpuIndexList, readGpuIndexEnv,
  describeSkippedGpus,
  planMinerGpus, parseGpuStats, parsePciBusIds, parseMacGpu,
};
