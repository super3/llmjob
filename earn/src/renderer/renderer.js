'use strict';

// Renderer UI for the LLMJob Earn window. Pure display + IPC glue — all
// computation happens in the main process (../shared modules). Kept thin and
// out of the coverage gate; the logic it leans on is unit-tested separately.
(function () {
  const $ = (id) => document.getElementById(id);
  const api = window.llmjob || {};

  const el = {
    addrInput: $('addr-input'),
    addrStatic: $('addr-static'),
    balanceMeta: $('balance-meta'),
    balance: $('balance'),
    balanceUsd: $('balance-usd'),
    getWallet: $('get-wallet'),
    hashrate: $('hashrate'),
    accepted: $('accepted'),
    uptime: $('uptime'),
    estday: $('estday'),
    line: $('mk-line'),
    area: $('mk-area'),
    btnStart: $('btn-start'),
    btnStop: $('btn-stop'),
    btnSettings: $('btn-settings'),
    btnLogs: $('btn-logs'),
    viewMiner: $('view-miner'),
    viewSettings: $('view-settings'),
    viewLogs: $('view-logs'),
    deviceLabel: $('device-label'),
    setWorker: $('set-worker'),
    setPool: $('set-pool'),
    poolNote: $('pool-note'),
    setRegion: $('set-region'),
    setDifficulty: $('set-difficulty'),
    setMdl: $('set-mdl'),
    mdlNote: $('mdl-note'),
    mdlBalanceMeta: $('mdl-balance-meta'),
    mdlBalance: $('mdl-balance'),
    logTerm: $('log-term'),
    engineStatus: $('engine-status'),
    appVersion: $('app-version'),
    btnCheckUpdate: $('btn-check-update'),
    updateStatus: $('update-status'),
  };

  const state = { mining: false, view: 'miner', address: '', gpu: '' };

  // Pool metadata from config:get. Falls back to an AlphaPool-only shape when
  // the bridge is unavailable, which preserves the pre-multi-pool behavior.
  let poolsCfg = null;
  let defaultPool = 'alphapool';

  const currentPool = () => (el.setPool && el.setPool.value) || defaultPool;
  // In-app balance lookups only exist for pools that support them (AlphaPool).
  const poolBalancesOn = () => {
    const p = poolsCfg && poolsCfg[currentPool()];
    return p ? !!p.balances : true;
  };

  const BAL_REFRESH_MS = 60000; // re-poll the pool balance once a minute
  let balDebounce = null;
  let mdlBalDebounce = null;
  let updateDismiss = null; // timer to auto-hide a transient update message
  let updateReady = false;  // an update is downloaded — the button installs + restarts

  const ADDR_RE = /^prl1p[0-9a-z]{20,80}$/i;
  const MDL_RE = /^mdl1p[0-9a-z]{20,80}$/i;
  const isValid = (a) => ADDR_RE.test(String(a || '').trim());
  const isValidMdl = (a) => MDL_RE.test(String(a || '').trim());

  const MDL_NOTE = {
    empty: 'Earn <b>MDL</b> on the exact hashrate already mining Pearl — same shares, no extra power or hardware. Leave blank to mine Pearl only.',
    on: '<b class="ok">✓ Merge-mining MDL</b> — your Pearl hashrate now also earns MDL, credited by the pool.',
    bad: '<b class="warn">That doesn\'t look like an mdl1… address.</b> Double-check it, or clear the field to mine Pearl only.',
  };

  function renderMdlNote() {
    const v = String(el.setMdl.value || '').trim();
    const key = !v ? 'empty' : isValidMdl(v) ? 'on' : 'bad';
    el.mdlNote.innerHTML = MDL_NOTE[key];
  }

  // The MDL balance line only makes sense for a well-formed mdl1… address on a
  // pool whose balances we can query; hide it otherwise.
  function renderMdlBalanceMeta() {
    el.mdlBalanceMeta.hidden = !(isValidMdl(String(el.setMdl.value || '').trim()) && poolBalancesOn());
  }

  // Note under the Connection rows for pools without in-app balance lookups.
  function renderPoolNote() {
    const p = poolsCfg && poolsCfg[currentPool()];
    if (!el.poolNote) return;
    if (!p || p.balances) { el.poolNote.hidden = true; el.poolNote.textContent = ''; return; }
    el.poolNote.hidden = false;
    el.poolNote.textContent = p.label + ' balances aren’t shown in-app yet — track your earnings on the pool’s site. ' +
      'This pool manages share difficulty automatically (the static difficulty setting doesn’t apply).';
  }

  // Rebuild the region dropdown for a pool and select its default region.
  function renderRegionOptions(poolKey) {
    const p = (poolsCfg && poolsCfg[poolKey]) || null;
    const regions = p ? p.regions : {};
    el.setRegion.innerHTML = '';
    Object.keys(regions).forEach((key) => {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = regions[key].flag + ' ' + regions[key].label + ' · ' + regions[key].name;
      el.setRegion.appendChild(opt);
    });
    if (p) el.setRegion.value = p.defaultRegion;
  }

  function resetMdlBalance() {
    el.mdlBalance.textContent = '0.000';
  }

  // Pull the merge-mined MDL total and show it. The pool keys the merge-mining
  // record by the PRL payout address (its miner endpoint rejects mdl1… lookups)
  // and echoes back which mdl1… address it pays, so this queries with the
  // payout address and only shows a balance when the pool's pairing matches the
  // address in the field — a mismatched entry earns nothing and stays 0.000.
  // Best-effort like the PRL lookup; guards against a stale response landing
  // after either field changed. There's no MDL/USD price, so no dollar figure.
  async function refreshMdlBalance() {
    const mdl = String(el.setMdl.value || '').trim();
    const addr = state.address.trim();
    if (!isValidMdl(mdl) || !isValid(addr) || !api.getMdlBalance || !poolBalancesOn()) return;
    const b = await api.getMdlBalance(addr);
    if (!b || mdl !== String(el.setMdl.value || '').trim() || addr !== state.address.trim()) return;
    if (b.mdlAddress && b.mdlAddress.toLowerCase() !== mdl.toLowerCase()) return;
    el.mdlBalance.textContent = fmt3(b.earned);
  }

  const FLAT_LINE = 'M0 55 L480 55';
  const FLAT_AREA = 'M0 56 L0 55 L480 55 L480 56 Z';

  function chartPaths(points) {
    const W = 480, H = 56;
    if (!points || !points.length) return { line: FLAT_LINE, area: FLAT_AREA };
    // Auto-scale to the data's own range (the hashrate varies by GPU), with a
    // little headroom; a floor keeps flat/tiny series centered rather than glued
    // to an edge.
    let lo = Math.min.apply(null, points);
    let hi = Math.max.apply(null, points);
    const pad = (hi - lo) * 0.2 || Math.max(1, hi * 0.1);
    lo -= pad; hi += pad;
    const span = (hi - lo) || 1;
    const stepX = W / Math.max(1, points.length - 1);
    const toY = (v) => +(H - ((v - lo) / span) * H).toFixed(1);
    const xy = points.map((v, i) => [+(i * stepX).toFixed(1), toY(v)]);
    const line = xy.map((p, i) => (i ? 'L' : 'M') + p[0] + ' ' + p[1]).join(' ');
    const area = 'M0 ' + H + ' ' + xy.map((p) => 'L' + p[0] + ' ' + p[1]).join(' ') + ' L' + W + ' ' + H + ' Z';
    return { line, area };
  }

  function renderView() {
    el.viewMiner.hidden = state.view !== 'miner';
    el.viewSettings.hidden = state.view !== 'settings';
    el.viewLogs.hidden = state.view !== 'logs';
    el.btnSettings.classList.toggle('active', state.view === 'settings');
  }

  function renderMiningState() {
    el.addrInput.hidden = state.mining;
    el.addrStatic.hidden = !state.mining;
    el.btnStart.hidden = state.mining;
    el.btnStop.hidden = !state.mining;
    if (state.mining) {
      el.addrStatic.textContent = state.address;
    } else {
      el.hashrate.textContent = '0.0';
      el.accepted.textContent = '0';
      el.uptime.textContent = '0m 00s';
      el.estday.textContent = '$0.00';
      el.deviceLabel.textContent = state.gpu || 'GPU · auto-detect';
      el.line.setAttribute('d', FLAT_LINE);
      el.area.setAttribute('d', FLAT_AREA);
      el.engineStatus.hidden = true;
    }
    el.btnStart.disabled = !isValid(state.address);
  }

  // With no valid payout address, show a "Get Wallet Address" link instead of
  // the (meaningless) balance line.
  function renderBalanceMeta() {
    const has = isValid(state.address);
    el.balanceMeta.hidden = !(has && poolBalancesOn());
    el.getWallet.hidden = has;
  }

  const fmt3 = (n) => Number(n).toLocaleString('en-US', { minimumFractionDigits: 3, maximumFractionDigits: 3 });

  function resetBalance() {
    el.balance.textContent = '0.000';
    el.balanceUsd.textContent = '≈ $0.00';
  }

  // Pull the pool balance for the current payout address and show it: total
  // earned = pending payout + lifetime paid. Best-effort — a null result
  // (offline / unknown address / pool hiccup) simply leaves the last shown value
  // in place. Guards against a late response landing after the address changed.
  async function refreshBalance() {
    const addr = state.address.trim();
    if (!isValid(addr) || !api.getBalance || !poolBalancesOn()) return;
    const b = await api.getBalance(addr);
    if (!b || addr !== state.address.trim()) return;
    el.balance.textContent = fmt3(b.earned);
    el.balanceUsd.textContent = b.usd != null ? '≈ $' + b.usd.toFixed(2) : '';
  }

  function applyStats(s) {
    if (!state.mining) return;
    el.hashrate.textContent = s.total;
    el.accepted.textContent = s.acceptedLabel;
    el.uptime.textContent = s.uptime;
    el.estday.textContent = s.estDay;
    if (s.gpu) el.deviceLabel.textContent = s.gpu;
    const p = chartPaths(s.points);
    el.line.setAttribute('d', p.line);
    el.area.setAttribute('d', p.area);
  }

  function appendLog(l) {
    const div = document.createElement('div');
    div.className = 'ln ' + (l.level || 'info');
    const t = document.createElement('span');
    t.className = 't';
    t.textContent = new Date().toLocaleTimeString('en-GB') + ' ';
    const m = document.createElement('span');
    m.className = 'm';
    m.textContent = l.line;
    div.appendChild(t);
    div.appendChild(m);
    el.logTerm.appendChild(div);
    el.logTerm.scrollTop = el.logTerm.scrollHeight;
  }

  function currentSettings() {
    const mdl = String(el.setMdl.value || '').trim();
    return {
      address: state.address.trim(),
      mdlAddress: isValidMdl(mdl) ? mdl : '',
      worker: el.setWorker.value.trim() || 'rig01',
      pool: currentPool(),
      region: el.setRegion.value || 'us2',
      difficulty: Number(el.setDifficulty.value) || 524288,
    };
  }

  function start() {
    if (!isValid(state.address)) return;
    state.mining = true;
    renderMiningState();
    appendLog({ level: 'info', line: 'starting LLMJob Earn…' });
    if (api.startMiner) api.startMiner(currentSettings());
  }

  function stop() {
    state.mining = false;
    renderMiningState();
    if (api.stopMiner) api.stopMiner();
  }

  function wire() {
    el.addrInput.addEventListener('input', (e) => {
      state.address = e.target.value;
      el.btnStart.disabled = !isValid(state.address);
      renderBalanceMeta();
      // Debounce the balance lookup so we don't hit the pool on every keystroke;
      // clear a stale balance the moment the address stops being valid. The MDL
      // lookup is keyed by this payout address too, so refresh it alongside.
      if (balDebounce) clearTimeout(balDebounce);
      if (mdlBalDebounce) clearTimeout(mdlBalDebounce);
      if (isValid(state.address)) {
        balDebounce = setTimeout(refreshBalance, 600);
        mdlBalDebounce = setTimeout(refreshMdlBalance, 600);
      } else {
        resetBalance();
        resetMdlBalance();
      }
    });
    el.setPool.addEventListener('change', () => {
      const pool = currentPool();
      renderRegionOptions(pool);
      renderPoolNote();
      // Balance lookups are pool-specific — hide/clear them off AlphaPool, and
      // refresh them (plus the best region) for the newly selected pool.
      renderBalanceMeta();
      renderMdlBalanceMeta();
      resetBalance();
      resetMdlBalance();
      if (poolBalancesOn()) { refreshBalance(); refreshMdlBalance(); }
      if (api.detectRegion && !state.mining) {
        api.detectRegion(pool).then((region) => {
          if (region && pool === currentPool()) el.setRegion.value = region;
        });
      }
    });
    el.setMdl.addEventListener('input', () => {
      renderMdlNote();
      renderMdlBalanceMeta();
      // Debounce the pool lookup so we don't hit it on every keystroke; clear a
      // stale balance the moment the address stops being valid.
      if (mdlBalDebounce) clearTimeout(mdlBalDebounce);
      if (isValidMdl(String(el.setMdl.value || '').trim())) mdlBalDebounce = setTimeout(refreshMdlBalance, 600);
      else resetMdlBalance();
    });
    el.btnStart.addEventListener('click', start);
    el.btnStop.addEventListener('click', stop);
    el.btnCheckUpdate.addEventListener('click', () => {
      if (updateReady) { if (api.installUpdate) api.installUpdate(); return; }
      if (!api.checkForUpdate) return;
      el.btnCheckUpdate.disabled = true;
      el.btnCheckUpdate.textContent = 'Checking…';
      api.checkForUpdate();
    });
    el.btnSettings.addEventListener('click', () => {
      state.view = state.view === 'settings' ? 'miner' : 'settings';
      renderView();
    });
    el.btnLogs.addEventListener('click', () => {
      state.view = state.view === 'logs' ? 'miner' : 'logs';
      renderView();
      // Jump to the latest line whenever the logs view is opened.
      if (state.view === 'logs') el.logTerm.scrollTop = el.logTerm.scrollHeight;
    });
    document.querySelectorAll('[data-back]').forEach((b) =>
      b.addEventListener('click', () => { state.view = 'miner'; renderView(); }));
    document.querySelectorAll('[data-ext]').forEach((a) =>
      a.addEventListener('click', (e) => {
        e.preventDefault();
        if (api.openExternal) api.openExternal(a.getAttribute('data-ext'));
      }));
  }

  async function init() {
    wire();
    if (api.getConfig) {
      const config = await api.getConfig();
      // Older main processes expose only `regions`; synthesize a one-pool table
      // so the rest of the renderer has a single shape to work with.
      poolsCfg = (config && config.pools) || {
        alphapool: { label: 'AlphaPool', regions: (config && config.regions) || {}, defaultRegion: 'us2', balances: true },
      };
      defaultPool = (config && config.defaultPool) || 'alphapool';
      el.setPool.innerHTML = '';
      Object.keys(poolsCfg).forEach((key) => {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = poolsCfg[key].label;
        el.setPool.appendChild(opt);
      });
      el.setPool.value = defaultPool;
      renderRegionOptions(defaultPool);
    }
    let resumeMining = false;
    if (api.getSettings) {
      const s = await api.getSettings();
      state.address = s.address || '';
      el.addrInput.value = state.address;
      el.setWorker.value = s.worker || 'rig01';
      if (poolsCfg && poolsCfg[s.pool]) {
        el.setPool.value = s.pool;
        renderRegionOptions(s.pool);
      }
      if (s.region) el.setRegion.value = s.region;
      renderPoolNote();
      el.setDifficulty.value = s.difficulty || 524288;
      el.setMdl.value = s.mdlAddress || '';
      // Set when "Update & restart" was clicked while mining — resume after launch.
      resumeMining = !!(s.resumeMining && isValid(state.address));
    }
    renderMdlNote();
    renderMdlBalanceMeta();
    if (api.detectGpu) {
      const gpu = await api.detectGpu();
      if (gpu) {
        state.gpu = gpu;
        if (!state.mining) el.deviceLabel.textContent = gpu; // shown on the main screen
        // Auto-match the recommended static difficulty to the detected card,
        // unless the user has saved a non-default value.
        if (api.difficultyForCard && Number(el.setDifficulty.value) === 524288) {
          const d = await api.difficultyForCard(gpu);
          if (d) el.setDifficulty.value = d;
        }
      }
    }
    // Auto-pick the lowest-latency pool region (unless the user is already
    // mining or has picked a non-default region this session).
    if (api.detectRegion && !state.mining) {
      const region = await api.detectRegion(currentPool());
      if (region) el.setRegion.value = region;
    }
    if (api.onStats) api.onStats(applyStats);
    if (api.onLog) api.onLog(appendLog);
    if (api.onStopped) api.onStopped(() => { state.mining = false; renderMiningState(); });
    if (api.onEngine) api.onEngine((e) => {
      if (!e) return;
      if (e.phase === 'downloading') {
        el.engineStatus.hidden = false;
        el.engineStatus.classList.remove('err');
        el.engineStatus.textContent = 'Downloading & setting up the mining engine…';
      } else if (e.phase === 'ready') {
        el.engineStatus.hidden = true;
        el.engineStatus.textContent = '';
      } else if (e.phase === 'error') {
        el.engineStatus.hidden = false;
        el.engineStatus.classList.add('err');
        el.engineStatus.textContent = e.message || 'Engine setup failed — see Logs.';
      }
    });
    if (api.onUpdate) api.onUpdate((s) => {
      if (!s) return;
      // Everything lives in the Software Update section (no top bar). The inline
      // line shows the result/progress; during 'checking' the button reads
      // "Checking…" so the inline line stays hidden.
      el.updateStatus.hidden = !s.show || s.phase === 'checking';
      el.updateStatus.textContent = s.text;
      el.updateStatus.classList.toggle('err', !!s.error);
      // When an update is downloaded, the button becomes the "Update & restart"
      // action; otherwise (once a check resolves) it's back to "Check for updates".
      if (s.ready) {
        updateReady = true;
        el.btnCheckUpdate.disabled = false;
        el.btnCheckUpdate.textContent = 'Update & restart';
        el.btnCheckUpdate.classList.add('ready');
      } else if (s.phase !== 'checking') {
        updateReady = false;
        el.btnCheckUpdate.disabled = false;
        el.btnCheckUpdate.textContent = 'Check for updates';
        el.btnCheckUpdate.classList.remove('ready');
      }
      // Auto-dismiss the transient "you're up to date" result after a few seconds.
      if (updateDismiss) { clearTimeout(updateDismiss); updateDismiss = null; }
      if (s.transient) updateDismiss = setTimeout(() => { el.updateStatus.hidden = true; }, 5000);
    });
    if (api.getVersion) api.getVersion().then((v) => { if (v) el.appVersion.textContent = 'v' + v; });
    renderView();
    renderMiningState();
    renderBalanceMeta();

    // Show the pending pool balance for a saved address right away, then keep it
    // fresh (shares get credited as mining continues).
    refreshBalance();
    setInterval(refreshBalance, BAL_REFRESH_MS);

    // Same for the merge-mined MDL balance, when an MDL address is configured.
    refreshMdlBalance();
    setInterval(refreshMdlBalance, BAL_REFRESH_MS);

    // Resume mining automatically if we restarted to install an update mid-mine.
    // start() persists fresh settings (without the flag), so it self-clears.
    if (resumeMining) start();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
