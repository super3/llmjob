'use strict';

// Renderer UI for the LLMJob Earn window. Pure display + IPC glue — all
// computation (mining, stats, the balance) happens in the main process
// (../shared + ../main). Kept thin and out of the coverage gate; the
// logic it leans on is unit-tested separately.
(function () {
  const $ = (id) => document.getElementById(id);
  const api = window.llmjob || {};

  const el = {
    // views
    tabMine: $('tab-mine'),
    btnSettings: $('btn-settings'), btnLogs: $('btn-logs'),
    viewMine: $('view-mine'), viewSettings: $('view-settings'), viewLogs: $('view-logs'),
    // mine
    addrInput: $('addr-input'), addrStatic: $('addr-static'),
    balanceMeta: $('balance-meta'), balance: $('balance'), balanceUsd: $('balance-usd'),
    getWallet: $('get-wallet'),
    hashrate: $('hashrate'), accepted: $('accepted'), rejected: $('rejected'), uptime: $('uptime'), estday: $('estday'),
    line: $('mk-line'), area: $('mk-area'),
    deviceLabel: $('device-label'),
    mineDot: $('mine-dot'),
    btnStart: $('btn-start'), btnStop: $('btn-stop'), engineStatus: $('engine-status'),
    updateBar: $('update-bar'), updateBarText: $('update-bar-text'), updateBarBtn: $('update-bar-btn'),
    // settings
    setWorker: $('set-worker'), setRegion: $('set-region'),
    appVersion: $('app-version'), btnCheckUpdate: $('btn-check-update'), updateStatus: $('update-status'),
    logTerm: $('log-term'),
  };

  const state = {
    // Overwritten from config.defaults at boot. Only a bare literal because the
    // renderer has no access to shared/config across the preload bridge.
    defaultRegion: 'us',
    mining: false,       // the miner is running
    view: 'mine',        // mine | settings | logs
    address: '', gpu: '', mdlAddress: '',
    // The card the ENGINE says it is mining on, once it says so. `gpu` above is
    // only the startup auto-detect, and on a multi-GPU rig the two can be
    // different cards (issue #226) — this one is the one doing the work.
    engineGpu: '',
    temp: 0,             // last core temperature reported, so a frame without one keeps it
    canMine: true,       // false on macOS — the Pearl core is CUDA, and Macs have no NVIDIA GPU
  };

  const BAL_REFRESH_MS = 60000; // re-poll the pool balance once a minute
  let balDebounce = null;
  let updateDismiss = null; // timer to auto-hide a transient update message
  let updateReady = false;  // an update is downloaded — the button installs + restarts

  const ADDR_RE = /^prl1p[0-9a-z]{20,80}$/i;
  const isValid = (a) => ADDR_RE.test(String(a || '').trim());

  // Shown in place of START's usual job on a platform with no miner at all.
  const NO_MINER_NOTE = 'Mining needs an NVIDIA GPU on Windows or Linux.';

  // ── Navigation ─────────────────────────────────────────────────────────────
  function renderView() {
    const v = state.view;
    el.viewMine.hidden = v !== 'mine';
    el.viewSettings.hidden = v !== 'settings';
    el.viewLogs.hidden = v !== 'logs';
    el.tabMine.classList.toggle('active', v === 'mine');
    el.btnSettings.classList.toggle('active', v === 'settings');
    // The footer link is the way out of the logs view as well as the way in, so
    // it has to say which. Driven from renderView, not the click handler, so
    // ← Back and the brand relabel it too.
    el.btnLogs.textContent = v === 'logs' ? 'CLOSE LOGS' : 'VIEW LOGS';
    if (v === 'logs') el.logTerm.scrollTop = el.logTerm.scrollHeight;
  }

  function goView(view) {
    state.view = view;
    renderView();
  }

  // Apply what the main process says this OS can do: on a platform with no
  // mining engine, START can never work, so say why instead of arming it.
  function applyPlatform(p) {
    state.canMine = !(p && p.minerSupported === false);
  }

  // START needs a valid payout address and a platform that can mine.
  function canStart() {
    return state.canMine && isValid(state.address);
  }

  // ── Charts / mining stats ──────────────────────────────────────────────────
  const FLAT_LINE = 'M0 55 L480 55';
  const FLAT_AREA = 'M0 56 L0 55 L480 55 L480 56 Z';

  // Points per side of the sparkline's smoothing window (so w = 2*SMOOTH_HALF+1).
  // 2 gives a five-point window, which is where the benefit stops: measured on a
  // real 60-point steady-state series off the live app, a centred mean took the
  // rendered spread from 7.2px of 56 to 2.2px, and widening to seven or nine
  // points gave 2.2 and 2.3 — nothing more, for strictly more lag.
  const SMOOTH_HALF = 2;

  // A centred mean of the visible points, edges clamped so the line keeps its
  // full width instead of shortening or hooking at the ends.
  //
  // Smoothing the DATA rather than just the zoom is the honest move here because
  // the jitter is a measurement artifact, not the card changing speed. The rate
  // is work-over-elapsed on a window that closes at >= 0.5s (pearl_core.cc), so a
  // window catches 12 batch completions or 13, and 3 operand reseeds or 4. The
  // give-away is that the series is ANTI-correlated — lag-1 autocorrelation
  // measured -0.345 on the live app, with the direction reversing at 67% of
  // points — because a window that grabs a 13th batch borrows it from the next
  // one. Work is conserved; the wobble is where the boundary fell, not the GPU.
  //
  // Centred rather than trailing or an EMA: this is a fixed history buffer, not a
  // live stream, so there is no causality to respect and no reason to accept lag
  // or an EMA's infinite memory (which would drag the startup ramp forward into
  // the steady-state window).
  //
  // What it costs: a one-point excursion — a single half-second window — is cut
  // to a fifth. That is the intended trade. Everything the chart exists to show
  // (a card dropping out, mining stopping, a thermal sag) persists for tens of
  // points and survives at full amplitude; verified against a 226 -> 120 step,
  // which comes through this at its exact original height.
  // The window SHRINKS at the two ends rather than clamping to the endpoint.
  // Both of the usual alternatives are wrong here. Dropping the ends shortens the
  // line by the most recent second — the second the miner is actually looking at.
  // Repeating the endpoint gives the newest sample 3/5 of its own window, so the
  // tip stays nearly as noisy as raw while the rest of the line is smooth, and
  // the chart wiggles precisely where the eye rests: measured on the live series,
  // the last five points span 1.52px clamped against 0.71px this way.
  //
  // The half-width is also held below half the array so a window can never cover
  // all of it — a three-point series must not flatten to a horizontal line, or
  // the ramp up from zero disappears during the first seconds of mining.
  function smooth(points) {
    const n = points.length;
    const k = Math.min(SMOOTH_HALF, (n - 1) >> 1);
    if (k < 1) return points;
    return points.map((_, i) => {
      const a = Math.max(0, i - k);
      const b = Math.min(n - 1, i + k);
      let sum = 0;
      for (let j = a; j <= b; j++) sum += points[j];
      return sum / (b - a + 1);
    });
  }

  function chartPaths(raw) {
    const W = 480, H = 56;
    if (!raw || !raw.length) return { line: FLAT_LINE, area: FLAT_AREA };
    const points = smooth(raw);
    let lo = Math.min.apply(null, points);
    let hi = Math.max.apply(null, points);
    // Don't let the axis zoom into noise. The range is taken from the visible
    // points alone, so once a rig has been up long enough that the window no
    // longer holds the ramp from zero, the only thing left to scale against is
    // the wiggle — and a steady card drawing a flat 226 TH/s rendered as a
    // violent sawtooth, because a ~1% variation was being stretched over the
    // full 56px. It reads as an unstable rig when nothing is wrong.
    //
    // A floor of 10% of the mean is what separates the two cases: normal
    // variation (a 0.5s window catching 12 batches or 13, and 3 operand reseeds
    // or 4) stays visibly small, while anything worth noticing — a card
    // dropping out, a job stall, mining stopping — still fills the chart.
    const mean = points.reduce((a, b) => a + b, 0) / points.length;
    const minSpan = Math.abs(mean) * 0.1;
    if (hi - lo < minSpan) {
      const mid = (hi + lo) / 2;
      lo = mid - minSpan / 2;
      hi = mid + minSpan / 2;
    }
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

  function renderMiningState() {
    el.addrInput.hidden = state.mining;
    el.addrStatic.hidden = !state.mining;
    el.btnStart.hidden = state.mining;
    el.btnStop.hidden = !state.mining;
    el.mineDot.className = 'dot2' + (state.mining ? ' on' : '');
    if (state.mining) {
      el.addrStatic.textContent = state.address;
    } else {
      el.hashrate.textContent = '0.0';
      el.accepted.textContent = '0';
      el.rejected.textContent = '0';
      el.uptime.textContent = '0m 00s';
      el.estday.textContent = '$0.00';
      el.deviceLabel.textContent = state.gpu || 'GPU · auto-detect';
      el.line.setAttribute('d', FLAT_LINE);
      el.area.setAttribute('d', FLAT_AREA);
      // On a Mac nothing will ever mine, so the status line says so up front
      // rather than leaving a START that stays grey for no stated reason.
      el.engineStatus.hidden = state.canMine;
      el.engineStatus.classList.remove('err');
      el.engineStatus.textContent = state.canMine ? '' : NO_MINER_NOTE;
    }
    el.btnStart.disabled = !canStart();
  }

  function renderBalanceMeta() {
    const has = isValid(state.address);
    el.balanceMeta.hidden = !has;
    el.getWallet.hidden = has;
  }

  const fmt3 = (n) => Number(n).toLocaleString('en-US', { minimumFractionDigits: 3, maximumFractionDigits: 3 });

  function resetBalance() {
    el.balance.textContent = '0.000';
    el.balanceUsd.textContent = '≈ $0.00';
  }

  async function refreshBalance() {
    const addr = state.address.trim();
    if (!isValid(addr) || !api.getBalance) return;
    const b = await api.getBalance(addr);
    if (!b || addr !== state.address.trim()) return;
    el.balance.textContent = fmt3(b.earned);
    el.balanceUsd.textContent = b.usd != null ? '≈ $' + b.usd.toFixed(2) : '';
  }


  // "NVIDIA GeForce RTX 4090 (86°C)". The engine reports a core temperature on
  // every status line, so it rides next to the GPU name while mining — a rig that
  // keeps crashing can be checked for heat without leaving the app for
  // nvidia-smi. On a multi-GPU rig this is the HOTTEST card (see miningStats),
  // since an average would hide one cooking card behind several cool ones.
  // Falls back to the bare name before the engine has reported a temperature.
  function deviceText(gpu, temp) {
    const t = Math.round(Number(temp) || 0);
    return t > 0 ? gpu + ' (' + t + '°C)' : gpu;
  }

  function applyStats(s) {
    if (!state.mining) return;
    el.hashrate.textContent = s.total;
    el.accepted.textContent = s.acceptedLabel;
    el.rejected.textContent = s.rejectedLabel;
    el.uptime.textContent = s.uptime;
    el.estday.textContent = s.estDay;
    // The ENGINE's name wins while mining, and the detected one is the fallback.
    //
    // It used to be the other way round, for a reason that has since expired:
    // alpha-miner 1.9.4's stats table abbreviated the card ("RTX 5090" against
    // nvidia-smi's "NVIDIA GeForce RTX 5090"), so trusting the engine made the
    // app contradict the board about the same GPU. Our own core doesn't
    // abbreviate — it reports the driver's own name for the device it opened, the
    // same string nvidia-smi prints.
    //
    // And the two are not always the same CARD. `state.gpu` is a guess made at
    // startup: nvidia-smi's first card, which is only the mining card while CUDA
    // and nvidia-smi agree on the ordering. On the rig in issue #226 they did
    // not, and this line is where that became a wrong name on screen — an idle
    // 32 GB RTX PRO 4500 labelled as the miner while an RTX 4070 did the work.
    // The core says which card it opened; nothing here knows better.
    //
    // Gated on having a NAME to show, from either source — not on the engine
    // being the one to supply it. It used to require `s.gpu`, which quietly
    // meant the temperature could never appear: currentSettings() sends no
    // `gpu`, so PearlEngine reports `gpu: null` on every status and the whole
    // branch was dead. alpha-miner scraped a name out of its own stdout, which
    // is what made the old gate work.
    //
    // The last reading is remembered rather than read straight off the frame, so
    // a status that carries no temperature leaves the label as it was instead of
    // dropping the degrees back off it.
    if (Number(s.temp) > 0) state.temp = Number(s.temp);
    // Remembered for the same reason as the temperature: a status frame that
    // carries no name must leave the label alone, not flip it back to the
    // startup guess for one frame and then back again.
    if (s.gpu) state.engineGpu = s.gpu;
    const deviceName = state.engineGpu || state.gpu;
    if (deviceName) el.deviceLabel.textContent = deviceText(deviceName, state.temp);
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

  // mdlAddress is carried through untouched, never shown. Merge mining is
  // retired from the UI, but an address someone already configured keeps
  // earning — and settings are persisted FROM this object, so omitting the key
  // would silently erase their address on the next Start and end the earnings
  // we are deliberately preserving. There is no way to set one any more; this
  // only round-trips what is already there.
  function currentSettings() {
    return {
      address: state.address.trim(),
      worker: el.setWorker.value.trim() || 'rig01',
      region: el.setRegion.value || state.defaultRegion,
      mdlAddress: state.mdlAddress || '',
    };
  }

  function start() {
    if (!canStart()) return;
    state.mining = true;
    // A fresh run chooses its card again — the last run's answer is not this
    // run's, and a stale one would outrank the new detection.
    state.engineGpu = '';
    renderMiningState();
    appendLog({ level: 'info', line: 'starting LLMJob Earn…' });
    if (api.startMiner) api.startMiner(currentSettings());
  }

  function stop() {
    state.mining = false;
    renderMiningState();
    if (api.stopMiner) api.stopMiner();
  }

  // Keep the OS window sized to the content: a view switch or a state change
  // (mining start/stop, an update bar appearing) changes the app's height, and
  // without this the frame keeps its old size — leaving a gap under the footer
  // or clipping a taller view. The log scroller is bounded, so this fires on
  // discrete layout changes, not per log line.
  //
  // Fires on HEIGHT changes only. A ResizeObserver reports width too, and the
  // fit never touches width — so reacting to it made this self-sustaining:
  // resizing the frame nudges .app's width (a scrollbar appearing, or DIP
  // rounding on a scaled display), which read as a fresh layout change and
  // asked for another fit, which nudged it again. The window resized
  // continuously on startup until a minimize/restore settled it, and any manual
  // resize started it off once more.
  function watchWindowFit() {
    if (!api.fitWindow || typeof ResizeObserver === 'undefined') return;
    const appEl = document.querySelector('.app');
    if (!appEl) return;
    let t = null;
    let lastH = null;
    const ro = new ResizeObserver(() => {
      const h = Math.ceil(appEl.getBoundingClientRect().height);
      if (h === lastH) return;
      lastH = h;
      if (t) clearTimeout(t);
      t = setTimeout(() => api.fitWindow(), 80);
    });
    ro.observe(appEl);
  }

  function wire() {
    el.addrInput.addEventListener('input', (e) => {
      state.address = e.target.value;
      el.btnStart.disabled = !canStart();
      renderBalanceMeta();
      if (balDebounce) clearTimeout(balDebounce);
      if (isValid(state.address)) balDebounce = setTimeout(refreshBalance, 600);
      else resetBalance();
    });
    el.btnStart.addEventListener('click', start);
    el.btnStop.addEventListener('click', stop);
    el.updateBarBtn.addEventListener('click', () => { if (api.installUpdate) api.installUpdate(); });
    el.btnCheckUpdate.addEventListener('click', () => {
      if (updateReady) { if (api.installUpdate) api.installUpdate(); return; }
      if (!api.checkForUpdate) return;
      el.btnCheckUpdate.disabled = true;
      el.btnCheckUpdate.textContent = 'Checking…';
      api.checkForUpdate();
    });

    // The brand returns to the Mine view.
    document.querySelectorAll('[data-tab]').forEach((t) =>
      t.addEventListener('click', () => goView(t.getAttribute('data-tab'))));
    // Settings gear + Logs toggle back to Mine
    el.btnSettings.addEventListener('click', () => goView(state.view === 'settings' ? 'mine' : 'settings'));
    el.btnLogs.addEventListener('click', () => goView(state.view === 'logs' ? 'mine' : 'logs'));
    document.querySelectorAll('[data-back]').forEach((b) =>
      b.addEventListener('click', () => goView('mine')));

    document.querySelectorAll('[data-ext]').forEach((a) =>
      a.addEventListener('click', (e) => {
        e.preventDefault();
        if (api.openExternal) api.openExternal(a.getAttribute('data-ext'));
      }));
  }

  async function init() {
    wire();
    watchWindowFit();
    if (api.getConfig) {
      const config = await api.getConfig();
      applyPlatform(config && config.platform);
      const regions = (config && config.regions) || {};
      const cfgDefaults = (config && config.defaults) || {};
      state.defaultRegion = cfgDefaults.region || state.defaultRegion;
      el.setRegion.innerHTML = '';
      Object.keys(regions).forEach((key) => {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = regions[key].flag + ' ' + regions[key].label + ' · ' + regions[key].name;
        el.setRegion.appendChild(opt);
      });
    }
    let resumeMining = false;
    if (api.getSettings) {
      const s = await api.getSettings();
      state.address = s.address || '';
      el.addrInput.value = state.address;
      state.mdlAddress = s.mdlAddress || '';
      el.setWorker.value = s.worker || 'rig01';
      // main migrates a stale AlphaPool id before we see it, so this always
      // names an option that exists.
      el.setRegion.value = s.region || state.defaultRegion;
      resumeMining = !!(s.resumeMining && isValid(state.address));
    }
    if (api.detectGpu) {
      const gpu = await api.detectGpu();
      if (gpu) {
        state.gpu = gpu;
        if (!state.mining) el.deviceLabel.textContent = gpu;
      }
    }
    if (api.detectRegion && !state.mining) {
      const region = await api.detectRegion();
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
      el.updateStatus.hidden = !s.show || s.phase === 'checking';
      el.updateStatus.textContent = s.text;
      el.updateStatus.classList.toggle('err', !!s.error);
      if (s.ready) {
        updateReady = true;
        el.btnCheckUpdate.disabled = false;
        el.btnCheckUpdate.textContent = 'Update & restart';
        el.btnCheckUpdate.classList.add('ready');
        // Announce it on the Mine view too: Settings is the one screen nobody
        // opens, so a downloaded update could sit there unnoticed.
        el.updateBarText.textContent = 'Update downloaded'
          + (s.version ? ' (v' + s.version + ')' : '') + ' — restart to apply.';
        el.updateBar.hidden = false;
      } else if (s.phase !== 'checking') {
        updateReady = false;
        el.btnCheckUpdate.disabled = false;
        el.btnCheckUpdate.textContent = 'Check for updates';
        el.btnCheckUpdate.classList.remove('ready');
        el.updateBar.hidden = true;
      }
      if (updateDismiss) { clearTimeout(updateDismiss); updateDismiss = null; }
      if (s.transient) updateDismiss = setTimeout(() => { el.updateStatus.hidden = true; }, 5000);
    });
    if (api.getVersion) api.getVersion().then((v) => { if (v) el.appVersion.textContent = 'v' + v; });
    renderView();
    renderMiningState();
    renderBalanceMeta();
    refreshBalance();
    setInterval(refreshBalance, BAL_REFRESH_MS);
    if (resumeMining) start();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
