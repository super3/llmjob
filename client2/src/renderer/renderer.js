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
    setDevice: $('set-device'),
    setWorker: $('set-worker'),
    setRegion: $('set-region'),
    setDifficulty: $('set-difficulty'),
    logTerm: $('log-term'),
    engineStatus: $('engine-status'),
    updateBar: $('update-bar'),
    updateText: $('update-text'),
    updateInstall: $('update-install'),
  };

  const state = { mining: false, view: 'miner', address: '' };

  const ADDR_RE = /^prl1p[0-9a-z]{20,80}$/i;
  const isValid = (a) => ADDR_RE.test(String(a || '').trim());

  const FLAT_LINE = 'M0 55 L480 55';
  const FLAT_AREA = 'M0 56 L0 55 L480 55 L480 56 Z';

  function chartPaths(points) {
    const W = 480, H = 56, MIN = 326, MAX = 372;
    if (!points || !points.length) return { line: FLAT_LINE, area: FLAT_AREA };
    const stepX = W / Math.max(1, points.length - 1);
    const toY = (v) => +(H - ((v - MIN) / (MAX - MIN)) * H).toFixed(1);
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
      el.line.setAttribute('d', FLAT_LINE);
      el.area.setAttribute('d', FLAT_AREA);
      el.engineStatus.hidden = true;
    }
    el.btnStart.disabled = !isValid(state.address);
  }

  function applyStats(s) {
    if (!state.mining) return;
    el.hashrate.textContent = s.total;
    el.accepted.textContent = s.acceptedLabel;
    el.uptime.textContent = s.uptime;
    el.estday.textContent = s.estDay;
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
    return {
      address: state.address.trim(),
      worker: el.setWorker.value.trim() || 'rig01',
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
    });
    el.btnStart.addEventListener('click', start);
    el.btnStop.addEventListener('click', stop);
    el.updateInstall.addEventListener('click', () => { if (api.installUpdate) api.installUpdate(); });
    el.btnSettings.addEventListener('click', () => {
      state.view = state.view === 'settings' ? 'miner' : 'settings';
      renderView();
    });
    el.btnLogs.addEventListener('click', () => {
      state.view = state.view === 'logs' ? 'miner' : 'logs';
      renderView();
    });
    document.querySelectorAll('[data-back]').forEach((b) =>
      b.addEventListener('click', () => { state.view = 'miner'; renderView(); }));
    document.querySelectorAll('[data-ext]').forEach((a) =>
      a.addEventListener('click', (e) => {
        e.preventDefault();
        if (api.openExternal) api.openExternal(a.getAttribute('data-ext'));
      }));
    el.setDevice.addEventListener('change', async () => {
      if (api.difficultyForCard) {
        const d = await api.difficultyForCard(el.setDevice.value);
        if (d) el.setDifficulty.value = d;
      }
    });
  }

  async function init() {
    wire();
    if (api.getConfig) {
      const config = await api.getConfig();
      const regions = (config && config.regions) || {};
      el.setRegion.innerHTML = '';
      Object.keys(regions).forEach((key) => {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = regions[key].flag + ' ' + regions[key].label + ' · ' + regions[key].name;
        el.setRegion.appendChild(opt);
      });
    }
    if (api.getSettings) {
      const s = await api.getSettings();
      state.address = s.address || '';
      el.addrInput.value = state.address;
      el.setWorker.value = s.worker || 'rig01';
      el.setRegion.value = s.region || 'us2';
      el.setDifficulty.value = s.difficulty || 524288;
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
        el.engineStatus.textContent = 'Engine setup failed — see Logs.';
      }
    });
    if (api.onUpdate) api.onUpdate((s) => {
      if (!s) return;
      el.updateBar.hidden = !s.show;
      el.updateText.textContent = s.text;
      el.updateBar.classList.toggle('err', !!s.error);
      el.updateInstall.hidden = !s.ready;
    });
    renderView();
    renderMiningState();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
