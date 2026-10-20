'use strict';

// Renderer UI for the LLMJob Earn window. Pure display + IPC glue — all
// computation (mining, the LLM engine, chat streaming) happens in the main
// process (../shared + ../main). test/renderer.test.js drives it in jsdom.
(function () {
  const $ = (id) => document.getElementById(id);
  const api = window.llmjob || {};

  const el = {
    // tabs / views
    tabMine: $('tab-mine'), tabChat: $('tab-chat'), tabApi: $('tab-api'),
    btnSettings: $('btn-settings'), btnLogs: $('btn-logs'),
    viewMine: $('view-mine'), viewChat: $('view-chat'), viewApi: $('view-api'),
    viewSettings: $('view-settings'), viewLogs: $('view-logs'),
    // mine
    addrInput: $('addr-input'), addrStatic: $('addr-static'),
    balanceMeta: $('balance-meta'), balance: $('balance'), balanceUsd: $('balance-usd'),
    getWallet: $('get-wallet'),
    hashrate: $('hashrate'), accepted: $('accepted'), uptime: $('uptime'), estday: $('estday'),
    chart: $('mk-chart'), line: $('mk-line'), area: $('mk-area'),
    deviceLabel: $('device-label'),
    mineDot: $('mine-dot'), llmHeroTps: $('llm-hero-tps'), llmHeroDot: $('llm-hero-dot'), llmHeroDetail: $('llm-hero-detail'),
    btnStart: $('btn-start'), btnStop: $('btn-stop'), engineStatus: $('engine-status'),
    // chat
    chatRunning: $('chat-running'), chatStopped: $('chat-stopped'), chatStoppedModel: $('chat-stopped-model'),
    chatHead: $('chat-head'), chatModel: $('chat-model'), chatNew: $('chat-new'),
    updateBar: $('update-bar'), updateBarText: $('update-bar-text'), updateBarBtn: $('update-bar-btn'),
    chatList: $('chat-list'), chatEmpty: $('chat-empty'), chatSuggestions: $('chat-suggestions'),
    chatMessages: $('chat-messages'), chatForm: $('chat-form'), chatInput: $('chat-input'), chatSend: $('chat-send'),
    // api
    apiRunning: $('api-running'), apiStopped: $('api-stopped'),
    apiEndpointUrl: $('api-endpoint-url'), apiCopy: $('api-copy'), apiOpen: $('api-open'),
    apiModel: $('api-model'), apiModelMeta: $('api-model-meta'), apiLan: $('api-lan'), apiLanUrl: $('api-lan-url'),
    // connect
    connectHint: $('connect-hint'), connectForm: $('connect-form'), connectToken: $('connect-token'),
    connectError: $('connect-error'), connectLink: $('connect-link'), connectDashboard: $('connect-dashboard'),
    connectPairToggle: $('connect-pair-toggle'), connectPair: $('connect-pair'),
    connectDone: $('connect-done'), connectedTitle: $('connected-title'), connectedName: $('connected-name'),
    connectedAvatar: $('connected-avatar'), connectedRename: $('connected-rename'), connectDisconnect: $('connect-disconnect'),
    connectReach: $('connect-reach'),
    // settings
    modeSeg: $('mode-seg'), modeHint: $('mode-hint'),
    setWorker: $('set-worker'), setRegion: $('set-region'),
    setShare: $('set-share'), shareNote: $('share-note'),
    appVersion: $('app-version'), btnCheckUpdate: $('btn-check-update'), updateCheckLabel: $('update-check-label'), updateStatus: $('update-status'),
    prefTray: $('pref-tray'), prefStartup: $('pref-startup'), prefIdle: $('pref-idle'),
    idleStatus: $('idle-status'),
    // CPU mining (Monero)
    cpuNote: $('cpu-note'), setXmr: $('set-xmr'),
    xmrBalanceMeta: $('xmr-balance-meta'), xmrBalance: $('xmr-balance'), xmrBalanceUsd: $('xmr-balance-usd'),
    setCpuThreads: $('set-cpu-threads'),
    chartCard: $('chart-card'), activityLabel: $('activity-label'), mineLabel: $('mine-label'), llmCol: $('llm-col'),
    cpuRow: $('cpu-row'), cpuDot: $('cpu-dot'), cpuRate: $('cpu-rate'), cpuMeta: $('cpu-meta'),
    cpuShares: $('cpu-shares'), cpuAccepted: $('cpu-accepted'),
    logTerm: $('log-term'),
  };

  const state = {
    // Overwritten from config.defaults at boot. Only a bare literal because the
    // renderer has no access to shared/config across the preload bridge.
    defaultRegion: 'us',
    // This machine's worker name, from config.defaults too: main's
    // defaultWorker(), the hostname. Not a shared literal like "rig01", which
    // makes two rigs on one address collide on the board (shared/worker).
    defaultWorker: '',
    mining: false,       // master process running (miner and/or LLM per mode)
    // START stands but the work is paused, because "only mine when idle" is on
    // and someone is using the computer. STOP still shows.
    paused: false,
    idleAfterSec: 300,
    view: 'mine',        // mine | chat | api | settings | logs
    returnTab: 'mine',   // where settings/logs return to
    address: '', gpu: '', mode: 'auto', mdlAddress: '',
    shareLlm: false,     // API tab: share the local LLM on port 8000
    // The card the ENGINE says it is mining on, once it says so. `gpu` above is
    // only the startup auto-detect, and on a multi-GPU rig the two can be
    // different cards (issue #226) — this one is the one doing the work.
    engineGpu: '',
    canMine: true,       // false on macOS — the miner needs an NVIDIA GPU
    llm: { ready: false, endpoint: null, webUrl: null, tps: 0, model: null, vramMb: null, ctxSize: null, error: null, note: null, lan: null, held: null },
    chat: { messages: [], streaming: false, streamText: '', bubble: null },
    node: { connected: false, nodeId: null, name: null },
    // Monero on the CPU (Settings → CPU Mining). On while a Monero address is
    // filled in, the way merge mining was. `cpu` is what main says this
    // computer can do; `cpuView` the CPU line main last sent.
    cpuMining: false, xmrAddress: '', xmrOk: false, xmrChecking: false, xmrError: null, xmrBalance: null, xmrUsd: null, cpuThreads: 0,
    cpu: null, cpuView: null,
  };

  const BAL_REFRESH_MS = 60000; // re-poll the pool balance once a minute
  // The Logs view keeps this many lines. The window lives on in the tray for
  // weeks, with a line for every accepted share, and it used to keep them all.
  const LOG_MAX_LINES = 2000;
  let balDebounce = null;
  let updateDismiss = null; // timer to auto-hide a transient update message
  let updateReady = false;  // an update is downloaded — the button installs + restarts

  const ADDR_RE = /^prl1p[0-9a-z]{20,80}$/i;
  const isValid = (a) => ADDR_RE.test(String(a || '').trim());

  const MODE_HINTS = {
    auto: 'Balances mining and the local LLM from free VRAM.',
    mining: 'Pearl mining only — the local LLM stays off.',
    llm: 'Local model only — no mining, no payout address needed.',
  };

  // The modes that need a mining engine. On a platform without one (macOS,
  // where there is no NVIDIA GPU for the CUDA miner) they are removed
  // from the segmented control rather than left to disappoint: picking "Mining"
  // there would arm a START that runs nothing at all. 'auto' stays, because it
  // degrades correctly on its own — main.js refuses the miner and the local LLM
  // still comes up — and it is the default a fresh install lands on.
  const MINING_MODES = ['mining'];
  const NO_MINER_HINT = 'Runs the local LLM on this Mac. Mining needs an NVIDIA GPU on Windows or Linux.';

  // Prompt chips shown in the empty chat — the real model answers them.
  const SUGGESTIONS = [
    { title: 'What is LLMJob?', prompt: 'What is LLMJob?' },
    { title: 'What is PPLNS?', prompt: 'What is PPLNS?' },
    { title: 'Help me write an email', prompt: 'Help me write a short email asking my landlord to fix the heater.' },
  ];


  // ── Navigation ─────────────────────────────────────────────────────────────
  function renderView() {
    const v = state.view;
    el.viewMine.hidden = v !== 'mine';
    el.viewChat.hidden = v !== 'chat';
    el.viewApi.hidden = v !== 'api';
    el.viewSettings.hidden = v !== 'settings';
    el.viewLogs.hidden = v !== 'logs';
    el.tabMine.classList.toggle('active', v === 'mine');
    el.tabChat.classList.toggle('active', v === 'chat');
    el.tabApi.classList.toggle('active', v === 'api');
    el.btnSettings.classList.toggle('active', v === 'settings');
    // The footer link is the way out of the logs view as well as the way in, so
    // it has to say which. Driven from renderView, not the click handler, so
    // ← Back and the tabs relabel it too.
    el.btnLogs.textContent = v === 'logs' ? 'CLOSE LOGS' : 'VIEW LOGS';
    if (v === 'chat' && state.llm.ready) setTimeout(() => el.chatInput.focus(), 0);
    if (v === 'logs') el.logTerm.scrollTop = el.logTerm.scrollHeight;
  }

  function goTab(tab) {
    state.view = tab;
    if (tab === 'mine' || tab === 'chat' || tab === 'api') state.returnTab = tab;
    renderView();
  }

  // ── Compute mode (segmented) ───────────────────────────────────────────────
  function renderMode() {
    const btns = el.modeSeg.querySelectorAll('[data-mode]');
    btns.forEach((b) => b.classList.toggle('active', b.getAttribute('data-mode') === state.mode));
    el.modeHint.textContent = !state.canMine && state.mode === 'auto'
      ? NO_MINER_HINT
      : (MODE_HINTS[state.mode] || MODE_HINTS.auto);
  }

  // Apply what the main process says this OS can do: on a platform with no
  // mining engine, drop the mining modes from the segmented control.
  function applyPlatform(p) {
    state.canMine = !(p && p.minerSupported === false);
    if (state.canMine) return;
    el.modeSeg.querySelectorAll('[data-mode]').forEach((b) => {
      if (MINING_MODES.indexOf(b.getAttribute('data-mode')) !== -1) b.hidden = true;
    });
  }

  // The saved mode, corrected for this platform. A settings file carrying
  // 'mining' or 'both' — written on another machine, or by a build from before
  // this one — must not select a button that is no longer on screen, leaving the
  // segment with nothing lit and START arming a run that does nothing.
  // 'both' ("Mining+LLM") was retired: it resolved to exactly what 'auto' does.
  // Mapping it keeps a rig that stored it on the same plan — an unrecognised
  // mode would fall through to mining-only and silently drop the LLM.
  function usableMode(mode) {
    if (mode === 'both') mode = 'auto';
    return !state.canMine && MINING_MODES.indexOf(mode) !== -1 ? 'auto' : mode;
  }

  function setMode(m) {
    state.mode = m;
    renderMode();
    el.btnStart.disabled = !canStart();
  }

  // START is allowed when we can mine (valid address) or the mode will run the
  // LLM anyway (llm/auto co-run the model even without a payout address). A
  // Monero address that checked out is enough on its own: the CPU miner runs for
  // any START, with or without Pearl.
  function canStart() {
    return isValid(state.address) || state.mode !== 'mining' || cpuReady();
  }

  function cpuReady() {
    return state.cpuMining && state.xmrOk;
  }

  // ── Local LLM (GPU Activity hero + chat/api gating) ────────────────────────
  // Detail line priority: a real error, then the transient stage note
  // ("downloading model … 42%", "Starting…"), then the model name. The note is
  // what tells a first-run user that a multi-GB download is moving rather than
  // stuck — without it the hero shows the model name and a grey dot the whole
  // time, which is indistinguishable from nothing happening.
  function renderLlmHero() {
    const err = state.llm.error;
    const ready = state.llm.ready;
    const note = !ready && !err ? state.llm.note : null;
    el.llmHeroTps.textContent = ready ? Number(state.llm.tps || 0).toFixed(1) : '0.0';
    el.llmHeroDot.className = 'dot2' + (err ? ' err' : ready ? ' on' : note ? ' busy' : '');
    el.llmHeroDetail.textContent = err || note || state.llm.model || 'gemma-4-E4B-it';
    el.llmHeroDetail.classList.toggle('err', !!err);
    el.llmHeroDetail.classList.toggle('quiet', !err && !note);
    // Why a card is not running a model, when main says, since the line above
    // has room for only a few words of it.
    el.llmCol.title = state.llm.held || state.llm.model || '';
    el.chatStoppedModel.textContent = state.llm.model || 'the local model';
    el.apiModel.textContent = state.llm.model || '—';
    el.apiModelMeta.textContent = modelMeta(state.llm.vramMb, state.llm.ctxSize);
  }

  // "~24 GB VRAM · 128K ctx" for the model being served. Was hard-coded to
  // "~5 GB VRAM · 128K ctx", which matched no model the app runs. Either half
  // is left out when main has not said.
  function modelMeta(vramMb, ctxSize) {
    const parts = [];
    if (vramMb > 0) parts.push('~' + Math.round(vramMb / 1024) + ' GB VRAM');
    if (ctxSize > 0) parts.push(Math.round(ctxSize / 1024) + 'K ctx');
    return parts.join(' · ');
  }

  function renderChatGate() {
    const up = state.llm.ready;
    el.chatRunning.hidden = !up;
    el.chatStopped.hidden = up;
    updateSendEnabled();
  }

  // The share switch on the API tab, and the network address above it. main
  // reports lan only while the network gate runs (or failed to
  // bind). Each note is one line inside the switch's row, so it stays short; the
  // full text is also its tooltip in case a long address cuts it off.
  function renderShare() {
    el.setShare.checked = state.shareLlm;
    const lan = state.shareLlm ? state.llm.lan : null;
    const url = lan && lan.urls[0];
    let note;
    if (!state.shareLlm) {
      note = 'Anyone on your network can use it on port 8000, with no key.';
    } else if (lan && lan.error) {
      note = 'Could not share it: ' + lan.error + '.';
    } else if (url) {
      // The address is on the line below, where a click copies it.
      note = 'Other devices on your network can use it, with no key.';
    } else if (lan) {
      note = 'Shared on port 8000, but this computer has no network address right now.';
    } else {
      note = 'Other devices can reach it on port 8000 while the local LLM runs.';
    }
    el.shareNote.textContent = note;
    el.shareNote.title = note;
    el.shareNote.classList.toggle('err', !!(lan && lan.error));
    el.apiLan.hidden = !url;
    if (url) el.apiLanUrl.textContent = url;
    el.connectReach.textContent = url ? 'this computer and your network' : 'this computer';
  }

  function renderApiGate() {
    const up = state.llm.ready;
    el.apiRunning.hidden = !up;
    el.apiStopped.hidden = up;
    if (up && state.llm.endpoint) el.apiEndpointUrl.textContent = state.llm.endpoint;
  }

  function renderLlm(s) {
    state.llm = {
      ready: !!(s && s.ready),
      endpoint: (s && s.endpoint) || null,
      webUrl: (s && s.webUrl) || null,
      tps: (s && s.tokensPerSec) || 0,
      model: (s && s.model) || state.llm.model,
      vramMb: (s && s.vramMb) || state.llm.vramMb,
      ctxSize: (s && s.ctxSize) || state.llm.ctxSize,
      error: (s && s.error) || null,
      note: (s && s.note) || null,
      lan: (s && s.lan) || null,
      held: (s && s.held) || null,
    };
    renderLlmHero();
    renderChatGate();
    renderApiGate();
    renderShare();
    // If the model went away mid-reply, unbrick the composer even if the
    // main-process chat-error event was lost in the shuffle.
    if (!state.llm.ready && state.chat.streaming) onChatError({ message: 'the local LLM stopped' });
  }

  // ── Chat ───────────────────────────────────────────────────────────────────
  function initSuggestions() {
    el.chatSuggestions.innerHTML = '';
    SUGGESTIONS.forEach((sg) => {
      const chip = document.createElement('span');
      chip.className = 'suggest';
      chip.textContent = sg.title;
      chip.addEventListener('click', () => sendChat(sg.prompt));
      el.chatSuggestions.appendChild(chip);
    });
  }

  function bolt() {
    const span = document.createElement('span');
    span.className = 'msg-bolt';
    span.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="#fff"><path d="M13 2 L4 14 h6 l-1 8 10-13 h-7 l1-7 z"></path></svg>';
    return span;
  }

  function addMsg(role, text) {
    const row = document.createElement('div');
    row.className = 'chat-msg ' + role;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    if (role === 'assistant') row.appendChild(bolt());
    row.appendChild(bubble);
    el.chatMessages.appendChild(row);
    scrollChat();
    return bubble;
  }

  // rAF-throttled: at 30–60 tok/s an unthrottled scrollHeight read would force a
  // synchronous layout pass per token while the GPU is already busy generating.
  let chatScrollPending = false;
  function scrollChat() {
    if (chatScrollPending) return;
    chatScrollPending = true;
    requestAnimationFrame(() => {
      chatScrollPending = false;
      el.chatList.scrollTop = el.chatList.scrollHeight;
    });
  }

  function updateSendEnabled() {
    const ok = !!el.chatInput.value.trim() && !state.chat.streaming && state.llm.ready;
    el.chatSend.disabled = !ok;
  }

  // The model + "New chat" header shows once a conversation has started.
  function newChat() {
    if (state.chat.streaming) return; // don't wipe a reply mid-stream
    state.chat.messages = [];
    state.chat.streamText = '';
    el.chatMessages.innerHTML = '';
    el.chatEmpty.hidden = false;
    el.chatHead.hidden = true;
    updateSendEnabled();
    if (state.view === 'chat') el.chatInput.focus();
  }

  function sendChat(text) {
    const t = String(text || '').trim();
    if (!t || state.chat.streaming || !state.llm.ready || !api.sendChat) return;
    el.chatEmpty.hidden = true;
    el.chatModel.textContent = state.llm.model || 'gemma-4-E4B-it';
    el.chatHead.hidden = false;
    addMsg('user', t);
    state.chat.messages.push({ role: 'user', text: t });
    state.chat.bubble = addMsg('assistant', '');
    state.chat.bubble.classList.add('streaming');
    state.chat.streaming = true;
    state.chat.streamText = '';
    el.chatInput.value = '';
    updateSendEnabled();
    api.sendChat(state.chat.messages.map((m) => ({ role: m.role, content: m.text })));
  }

  function onChatDelta(d) {
    if (!state.chat.streaming || !state.chat.bubble) return;
    const text = (d && d.text) || '';
    if (!text) return;
    state.chat.streamText += text;
    // Append-only (O(1) per delta) — rewriting textContent with the whole
    // accumulated reply is O(n) per token and stutters on long answers.
    state.chat.bubble.appendChild(document.createTextNode(text));
    scrollChat();
  }

  function endStream() {
    if (state.chat.bubble) state.chat.bubble.classList.remove('streaming');
    state.chat.streaming = false;
    state.chat.bubble = null;
    updateSendEnabled();
    if (state.view === 'chat') el.chatInput.focus();
  }

  function onChatDone() {
    if (!state.chat.streaming) return;
    state.chat.messages.push({ role: 'assistant', text: state.chat.streamText || '' });
    endStream();
  }

  function onChatError(d) {
    if (state.chat.bubble) {
      const msg = (d && d.message) || 'the chat request failed';
      state.chat.bubble.textContent = (state.chat.streamText ? state.chat.streamText + '\n\n' : '') + '⚠ ' + msg;
      state.chat.bubble.classList.add('err');
    }
    endStream();
  }

  // ── Connect with LLMJob (link this node to an account) ─────────────────────
  function renderNode(s) {
    state.node = {
      connected: !!(s && s.connected),
      nodeId: (s && s.nodeId) || null,
      name: (s && s.name) || null,
      user: (s && s.user) || null,
    };
    const on = state.node.connected;
    // Title/avatar use the account handle when the server resolved one; the
    // "live as …" line always shows the worker (rig) name.
    const title = state.node.user || state.node.name || 'Connected';
    const rig = state.node.name || state.node.nodeId || 'this rig';
    el.connectForm.hidden = on;
    el.connectDone.hidden = !on;
    el.connectHint.textContent = on ? '' : 'Not linked to an account';
    if (on) {
      el.connectedAvatar.textContent = title.charAt(0).toUpperCase();
      el.connectedTitle.textContent = title;
      el.connectedName.textContent = rig;
    }
  }

  function showConnectError(msg) {
    el.connectError.textContent = msg;
    el.connectError.hidden = false;
  }

  async function doConnect() {
    const token = String(el.connectToken.value || '').trim();
    el.connectError.hidden = true;
    if (!token) { showConnectError('Enter your pairing token first.'); return; }
    if (!api.connectNode) return;
    el.connectLink.disabled = true;
    el.connectLink.textContent = 'Linking…';
    const res = await api.connectNode({ token, name: el.setWorker.value.trim() || undefined });
    el.connectLink.disabled = false;
    el.connectLink.textContent = 'Link';
    if (res && res.success) {
      el.connectToken.value = '';
      renderNode({ connected: true, nodeId: res.nodeId, name: res.name, user: res.user });
    } else {
      showConnectError((res && res.error) || 'Connection failed.');
    }
  }

  async function doDisconnect() {
    if (!api.disconnectNode) return;
    await api.disconnectNode();
    renderNode({ connected: false });
  }

  // ── Charts / mining stats ──────────────────────────────────────────────────
  // The chart's height on screen (.chart-card svg in styles.css) and its
  // viewBox, which is 480 by 56 at that height. A window made taller than the
  // content gives the chart the spare height (see watchWindowFit), and the
  // viewBox grows by the same ratio, so the line keeps its scale and thickness
  // and only the chart gets taller.
  const CHART_PX = 84;
  const CHART_W = 480;
  const CHART_H = 56;
  // The spare height, in px, and the points on the chart (null while stopped).
  let fillPx = 0;
  let chartPoints = null;

  function chartHeight() {
    return +(CHART_H * (CHART_PX + fillPx) / CHART_PX).toFixed(1);
  }

  // A flat line along the bottom, which is all a stopped rig has.
  function flatPaths(H) {
    const y = +(H - 1).toFixed(1);
    return { line: 'M0 ' + y + ' L' + CHART_W + ' ' + y, area: 'M0 ' + H + ' L0 ' + y + ' L' + CHART_W + ' ' + y + ' L' + CHART_W + ' ' + H + ' Z' };
  }

  function drawChart() {
    const H = chartHeight();
    const p = chartPaths(chartPoints, H);
    el.chart.setAttribute('viewBox', '0 0 ' + CHART_W + ' ' + H);
    el.line.setAttribute('d', p.line);
    el.area.setAttribute('d', p.area);
  }

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

  function chartPaths(raw, H) {
    const W = CHART_W;
    if (!raw || !raw.length) return flatPaths(H);
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

  // ── CPU mining (Monero) ────────────────────────────────────────────────────
  // The settings ride on START like the others, so they are locked while
  // mining and apply at the next START. The address is checked in main: the
  // renderer has no Keccak for Monero's checksum.
  const CPU_STATES = {
    starting: 'starting…',
    dataset: 'building its dataset…',
    restarting: 'restarting…',
  };
  let xmrCheckTimer = null;

  // The thread count that will run: the one picked, capped to leave a CPU
  // free, or the default.
  function cpuThreadsNow() {
    const c = state.cpu;
    return state.cpuThreads > 0 ? Math.min(state.cpuThreads, c.available - 1) : c.auto;
  }

  // A build without the worker (macOS has none, and CI builds it as an
  // optional step) can't mine on the CPU at all.
  function cpuNoWorker() {
    return !!state.cpu && state.cpu.worker === false;
  }

  // The line under the address shows only when something stops CPU mining:
  // a bad address, no worker, or no threads. One line keeps Settings in its
  // 470px, and the title has all of it.
  function cpuNote() {
    if (state.xmrError) return 'This address won\'t work: ' + state.xmrError + '.';
    if (cpuNoWorker()) return 'This build has no CPU miner (llmjob-xmr-worker is not beside the app).';
    const c = state.cpu;
    if (!state.cpuMining || !c) return '';
    if (!c.canMine) return 'Can\'t run here: ' + c.reason + '.';
    if (!cpuThreadsNow()) return 'Auto: ' + c.reason + '.';
    return '';
  }

  const threadsText = (n) => n + (n === 1 ? ' thread' : ' threads');

  function renderCpuSettings() {
    const c = state.cpu;
    el.cpuNote.textContent = cpuNote();
    el.cpuNote.title = el.cpuNote.textContent;
    el.cpuNote.hidden = !el.cpuNote.textContent;
    el.cpuNote.classList.toggle('err', !!state.xmrError);
    el.setCpuThreads.innerHTML = '';
    const max = c ? c.available - 1 : 0;
    const auto = document.createElement('option');
    auto.value = '0';
    auto.textContent = c && c.auto ? 'Auto (' + threadsText(c.auto) + ')' : 'Auto (off)';
    el.setCpuThreads.appendChild(auto);
    for (let n = 1; n <= max; n++) {
      const opt = document.createElement('option');
      opt.value = String(n);
      opt.textContent = threadsText(n);
      el.setCpuThreads.appendChild(opt);
    }
    el.setCpuThreads.value = String(Math.min(state.cpuThreads, max));
    for (const f of [el.setXmr, el.setCpuThreads]) f.disabled = state.mining;
    // Without the worker it can't be turned on, but a saved address can be
    // cleared.
    if (cpuNoWorker() && !state.cpuMining) el.setXmr.disabled = true;
  }

  const fmt4 = (n) => Number(n).toLocaleString('en-US', { minimumFractionDigits: 4, maximumFractionDigits: 4 });

  // It runs for any START, with or without the GPU miner, and waits out an
  // idle-only pause like it.
  function cpuMetaText(v) {
    if (!state.mining) return cpuIdleText();
    if (state.paused) return 'paused while in use';
    if (!v) return 'starting…';
    if (v.error) return 'stopped: ' + v.error;
    if (v.pool) return v.pool;
    if (v.state === 'mining') {
      return v.accepted + ' accepted' + (v.rejected ? ' · ' + v.rejected + ' rejected' : '') + ' · ' + v.threads + ' threads';
    }
    return CPU_STATES[v.state] || v.state;
  }

  // Stopped: what START will do. main forgets why the last run failed on STOP,
  // so this works it out from what it knows of this computer.
  function cpuIdleText() {
    // The line shows only with an address, so one that isn't OK is wrong.
    if (state.xmrChecking) return 'checking address…';
    if (!state.xmrOk) return 'check the address';
    if (cpuNoWorker()) return "no CPU miner here";
    if (state.cpu && !state.cpu.canMine) return "can't run here";
    if (state.cpu && !cpuThreadsNow()) return "off: set CPU threads";
    return 'ready to start';
  }

  function renderCpuRow() {
    el.cpuRow.hidden = !state.cpuMining;
    el.cpuShares.hidden = !state.cpuMining;
    el.chartCard.classList.toggle('with-cpu', state.cpuMining);
    el.activityLabel.textContent = state.cpuMining ? 'GPU & CPU Activity' : 'GPU Activity';
    // Next to Monero, "Mining" doesn't say which; the coin does.
    el.mineLabel.textContent = state.cpuMining ? 'Pearl' : 'Mining';
    const v = state.mining ? state.cpuView : null;
    el.cpuRate.textContent = v ? v.hashrate : '0.00';
    el.cpuAccepted.textContent = v ? String(v.accepted || 0) : '0';
    const bad = !!(v && (v.error || v.pool));
    el.cpuDot.className = 'dot2' + (!v ? '' : bad ? ' err' : v.state === 'mining' ? ' on' : CPU_STATES[v.state] ? ' busy' : '');
    el.cpuMeta.textContent = cpuMetaText(v);
    el.cpuMeta.title = el.cpuMeta.textContent;
    el.cpuMeta.classList.toggle('err', bad);
    // The share count is in the stats and "ready to start" says nothing the
    // grey dot doesn't, so those two stay off the card.
    el.cpuMeta.classList.toggle('quiet', !bad && (v ? v.state === 'mining' : el.cpuMeta.textContent === 'ready to start'));
    const bal = state.xmrBalance != null && state.xmrOk ? fmt4(state.xmrBalance) + ' XMR earned' : '';
    el.cpuRow.title = [el.cpuMeta.textContent, bal].filter(Boolean).join(' · ');
    renderXmrBalance();
  }

  // Settings → CPU Mining: the Monero balance above the address, as the Pearl
  // balance is above the payout address. It shows once the address checks
  // out, at 0 until the pool answers, and without dollars when the pool has
  // no price.
  function renderXmrBalance() {
    const has = state.cpuMining && state.xmrOk;
    el.xmrBalanceMeta.hidden = !has;
    const known = has && state.xmrBalance != null;
    el.xmrBalance.textContent = fmt4(known ? state.xmrBalance : 0);
    el.xmrBalanceUsd.textContent = !known ? '≈ $0.00' : state.xmrUsd != null ? '≈ ' + fmtUsd(state.xmrUsd) : '';
  }

  async function refreshXmrBalance() {
    const a = state.xmrAddress.trim();
    if (!state.cpuMining || !state.xmrOk || !api.getXmrBalance) return;
    const b = await api.getXmrBalance(a);
    if (!b || a !== state.xmrAddress.trim()) return;
    state.xmrBalance = b.earned;
    state.xmrUsd = b.usd;
    renderCpuRow();
  }

  async function checkXmr() {
    const a = state.xmrAddress.trim();
    state.xmrBalance = null;
    state.xmrUsd = null;
    if (!a || !api.checkXmrAddress) {
      state.xmrOk = false;
      state.xmrChecking = false;
      state.xmrError = null;
    } else {
      const r = await api.checkXmrAddress(a);
      if (a !== state.xmrAddress.trim()) return;
      state.xmrOk = !!(r && r.ok);
      state.xmrChecking = false;
      state.xmrError = r && !r.ok ? r.error : null;
    }
    renderCpuSettings();
    renderCpuRow();
    el.btnStart.disabled = !canStart();
    refreshXmrBalance();
  }

  // Whether this run mines Pearl on the GPU, by main's rule (planFor): a
  // valid address, a mode that mines, and a platform with a miner. A Monero-
  // or LLM-only run leaves the GPU's dot grey.
  function gpuMinerOn() {
    return state.mining && !state.paused && state.mode !== 'llm' && state.canMine !== false && isValid(state.address);
  }

  function renderMiningState() {
    el.addrInput.hidden = state.mining;
    el.addrStatic.hidden = !state.mining;
    el.btnStart.hidden = state.mining;
    el.btnStop.hidden = !state.mining;
    el.mineDot.className = 'dot2' + (gpuMinerOn() ? ' on' : '');
    el.idleStatus.hidden = !(state.mining && state.paused);
    el.idleStatus.textContent = 'Waiting for your computer to be idle. Mining starts after '
      + Math.round(state.idleAfterSec / 60) + ' minutes without keyboard or mouse use.';
    if (state.mining) {
      el.addrStatic.textContent = state.address;
    } else {
      el.hashrate.textContent = '0.0';
      el.accepted.textContent = '0';
      el.uptime.textContent = '0m 00s';
      el.estday.textContent = '$0.00';
      el.deviceLabel.textContent = state.gpu || 'GPU · auto-detect';
      chartPoints = null;
      drawChart();
      el.engineStatus.hidden = true;
      state.cpuView = null;
    }
    el.btnStart.disabled = !canStart();
    renderCpuSettings();
    renderCpuRow();
  }

  function renderBalanceMeta() {
    const has = isValid(state.address);
    el.balanceMeta.hidden = !has;
    el.getWallet.hidden = has;
  }

  const fmt3 = (n) => Number(n).toLocaleString('en-US', { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  // Dollars with a thousands separator, as the PRL balance has: $1,110.69.
  const fmtUsd = (n) => '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

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
    el.balanceUsd.textContent = b.usd != null ? '≈ ' + fmtUsd(b.usd) : '';
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
    // The temperature is read straight off the frame. main's accumulator already
    // keeps the last reading between frames, and sends 0 exactly when there is
    // no current one: an idle pause, a switch to LLM-only, a new START. Keeping
    // the last reading here as well showed a paused card at its mining
    // temperature for the whole pause.
    //
    // The name is remembered: a status frame that carries no name must leave the
    // label alone, not flip it back to the startup guess for one frame and then
    // back again.
    if (s.gpu) state.engineGpu = s.gpu;
    const deviceName = state.engineGpu || state.gpu;
    if (deviceName) el.deviceLabel.textContent = deviceText(deviceName, s.temp);
    chartPoints = s.points;
    drawChart();
  }

  function applyCpuStats(v) {
    state.cpuView = v || null;
    renderCpuRow();
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
    // The oldest line goes once the view is full (see LOG_MAX_LINES).
    if (el.logTerm.children.length > LOG_MAX_LINES) el.logTerm.firstElementChild.remove();
    el.logTerm.scrollTop = el.logTerm.scrollHeight;
  }

  // ── Background (Settings): tray, start with the computer, idle-only ──────
  function renderPrefs(p) {
    if (!p) return;
    el.prefTray.checked = !!p.closeToTray;
    el.prefStartup.checked = !!p.runAtStartup;
    el.prefIdle.checked = !!p.mineWhenIdle;
  }

  // Main answers with what now holds, which can differ from the click: a build
  // that cannot start with the computer turns that switch back off.
  function savePref(change) {
    if (api.setPrefs) api.setPrefs(change).then(renderPrefs);
  }

  // The main process paused or resumed the work for idle-only mining.
  function onIdle(d) {
    state.paused = !!(d && d.paused);
    if (d && d.idleAfterSec) state.idleAfterSec = d.idleAfterSec;
    renderMiningState();
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
      worker: el.setWorker.value.trim() || state.defaultWorker,
      region: el.setRegion.value || state.defaultRegion,
      mode: state.mode || 'mining',
      mdlAddress: state.mdlAddress || '',
      // Carried so a START, which rewrites settings.json from this object, keeps
      // the switch. main also saves it the moment it is flipped.
      shareLlm: state.shareLlm,
      cpuMining: state.cpuMining,
      xmrAddress: state.xmrAddress.trim(),
      cpuThreads: state.cpuThreads,
    };
  }

  function start() {
    if (!canStart()) return;
    state.mining = true;
    state.paused = false;
    // A fresh run chooses its card again — the last run's answer is not this
    // run's, and a stale one would outrank the new detection.
    state.engineGpu = '';
    renderMiningState();
    appendLog({ level: 'info', line: 'starting LLMJob Earn…' });
    if (api.startMiner) api.startMiner(currentSettings());
  }

  function stop() {
    state.mining = false;
    state.paused = false;
    renderMiningState();
    if (api.stopMiner) api.stopMiner();
  }

  // START LLM (from the Chat/API tabs): make sure the compute mode actually runs
  // the model, then start. Mining-only becomes Auto (or LLM-only with no
  // payout address); llm/auto start as-is.
  //
  // It also gives a card that kept losing its GPU another try. START and a
  // launch at login leave that card alone.
  function startLlmIntent() {
    if (state.llm.ready) return;
    if (api.retryLlm) api.retryLlm();
    if (state.mode === 'mining') setMode(isValid(state.address) ? 'auto' : 'llm');
    start();
  }

  // Keep the OS window sized to the content: a tab switch or a state change
  // (mining start/stop, LLM coming up) changes the app's height, and without
  // this the frame keeps its old size — leaving a gap under the footer or
  // clipping a taller view. Inner scrollers (chat, logs) are bounded, so this
  // fires on discrete layout changes, not per streamed token.
  //
  // Fires on HEIGHT changes only. A ResizeObserver reports width too, and the
  // fit never touches width — so reacting to it made this self-sustaining:
  // resizing the frame nudges .app's width (a scrollbar appearing, or DIP
  // rounding on a scaled display), which read as a fresh layout change and
  // asked for another fit, which nudged it again. The window resized
  // continuously on startup until a minimize/restore settled it, and any manual
  // resize started it off once more.
  //
  // A window made taller by hand (snapped to half the screen, maximised) is
  // filled instead of left with a gap under the footer: --fill, the spare
  // height, is added to every view (styles.css), and on Mine it goes to the
  // chart, so START/STOP and the footer sit at the bottom of the window. Every
  // view grows by exactly --fill, so .app minus the fill is the content's own
  // height, and that is what a fit is about. At the fitted size the fill is 0
  // and nothing changes.
  //
  // A fitted window still follows its content, as before. A taller one keeps
  // its size when the content changes, and the fill takes up the difference,
  // until the content no longer fits it.
  function watchWindowFit() {
    if (!api.fitWindow || typeof ResizeObserver === 'undefined') return;
    const appEl = document.querySelector('.app');
    if (!appEl) return;
    const root = document.documentElement;
    const content = () => appEl.getBoundingClientRect().height - fillPx;
    // Spare height under a fit's tolerance (FIT_TOLERANCE_PX in main) is
    // ignored, so a fitted window that is a pixel or two off is left as it was.
    const spare = () => {
      const px = Math.floor(window.innerHeight - content());
      return px > 2 ? px : 0;
    };
    const setFill = (px) => {
      if (px === fillPx) return;
      fillPx = px;
      if (px) root.style.setProperty('--fill', px + 'px');
      else root.style.removeProperty('--fill');
      drawChart();
    };
    let t = null;
    let lastH = null;
    // Both the window (the root fills it) and the content: either changes the fill.
    const ro = new ResizeObserver(() => {
      const h = Math.ceil(content());
      if (h === lastH) {
        setFill(spare());
        return;
      }
      lastH = h;
      if (fillPx) {
        setFill(spare());
        if (fillPx) return;
      }
      if (t) clearTimeout(t);
      t = setTimeout(() => api.fitWindow(), 80);
    });
    ro.observe(root);
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
    el.prefTray.addEventListener('change', () => savePref({ closeToTray: el.prefTray.checked }));
    el.prefStartup.addEventListener('change', () => savePref({ runAtStartup: el.prefStartup.checked }));
    el.prefIdle.addEventListener('change', () => savePref({ mineWhenIdle: el.prefIdle.checked }));
    el.updateBarBtn.addEventListener('click', () => { if (api.installUpdate) api.installUpdate(); });
    el.btnCheckUpdate.addEventListener('click', () => {
      if (updateReady) { if (api.installUpdate) api.installUpdate(); return; }
      if (!api.checkForUpdate) return;
      el.btnCheckUpdate.disabled = true;
      el.btnCheckUpdate.classList.add('checking');
      el.updateCheckLabel.textContent = 'Checking…';
      api.checkForUpdate();
    });

    // Tabs (brand=Mine, Chat, API, and the Local-LLM row's Open Chat / API chips)
    document.querySelectorAll('[data-tab]').forEach((t) =>
      t.addEventListener('click', () => goTab(t.getAttribute('data-tab'))));
    // Settings gear + Logs toggle back to the last tab
    el.btnSettings.addEventListener('click', () => {
      state.view = state.view === 'settings' ? state.returnTab : 'settings';
      renderView();
    });
    el.btnLogs.addEventListener('click', () => {
      state.view = state.view === 'logs' ? state.returnTab : 'logs';
      renderView();
    });
    document.querySelectorAll('[data-back]').forEach((b) =>
      b.addEventListener('click', () => { state.view = state.returnTab; renderView(); }));

    // Compute-mode segmented control
    el.modeSeg.querySelectorAll('[data-mode]').forEach((b) =>
      b.addEventListener('click', () => setMode(b.getAttribute('data-mode'))));

    // START LLM buttons (chat/api stopped states)
    document.querySelectorAll('[data-start-llm]').forEach((b) =>
      b.addEventListener('click', startLlmIntent));

    // Chat composer
    el.chatInput.addEventListener('input', updateSendEnabled);
    el.chatForm.addEventListener('submit', (e) => { e.preventDefault(); sendChat(el.chatInput.value); });
    el.chatNew.addEventListener('click', newChat);

    // API endpoint: copy the /v1 URL, or open the llama-server web UI.
    const copyEndpoint = () => {
      if (!state.llm.endpoint) return;
      if (api.copyText) api.copyText(state.llm.endpoint);
      const prev = el.apiCopy.textContent;
      el.apiCopy.textContent = 'Copied';
      setTimeout(() => { el.apiCopy.textContent = prev; }, 1200);
    };
    el.apiCopy.addEventListener('click', copyEndpoint);
    el.apiEndpointUrl.addEventListener('click', copyEndpoint);
    el.apiOpen.addEventListener('click', () => {
      const url = state.llm.webUrl || state.llm.endpoint;
      if (url && api.openExternal) api.openExternal(url);
    });

    // Connect with LLMJob (pairing token → link this node; Disconnect to unlink)
    el.connectLink.addEventListener('click', doConnect);
    el.connectToken.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); doConnect(); } });
    el.connectDisconnect.addEventListener('click', doDisconnect);
    // Primary "Connect with LLMJob" opens the dashboard sign-in; "Use a pairing
    // token" reveals the manual token field (collapsed by default).
    el.connectDashboard.addEventListener('click', () => { if (api.openNodeDashboard) api.openNodeDashboard(); });
    el.connectPairToggle.addEventListener('click', () => {
      el.connectPair.hidden = !el.connectPair.hidden;
      if (!el.connectPair.hidden) el.connectToken.focus();
    });
    el.connectedRename.addEventListener('click', () => { state.view = 'settings'; renderView(); });

    // Share the local LLM on the network. Applied at once by main, not at START.
    el.setShare.addEventListener('change', () => {
      state.shareLlm = el.setShare.checked;
      if (api.setShareLlm) api.setShareLlm(state.shareLlm);
      renderShare();
    });
    el.setXmr.addEventListener('input', (e) => {
      state.xmrAddress = e.target.value;
      // An address turns it on and a blank box turns it off, at once; the
      // check follows.
      const on = !!state.xmrAddress.trim();
      if (on !== state.cpuMining) {
        state.cpuMining = on;
        renderCpuSettings();
      }
      // Not OK until the check answers.
      state.xmrOk = false;
      state.xmrChecking = on;
      renderCpuRow();
      el.btnStart.disabled = !canStart();
      if (xmrCheckTimer) clearTimeout(xmrCheckTimer);
      xmrCheckTimer = setTimeout(checkXmr, 400);
    });
    el.setCpuThreads.addEventListener('change', () => {
      state.cpuThreads = Number(el.setCpuThreads.value) || 0;
      renderCpuSettings();
      // Auto can mean off, which the Mine tab says.
      renderCpuRow();
    });
    el.apiLanUrl.addEventListener('click', () => {
      if (api.copyText) api.copyText(el.apiLanUrl.textContent);
    });

    document.querySelectorAll('[data-ext]').forEach((a) =>
      a.addEventListener('click', (e) => {
        e.preventDefault();
        if (api.openExternal) api.openExternal(a.getAttribute('data-ext'));
      }));
  }

  async function init() {
    wire();
    watchWindowFit();
    initSuggestions();
    // Before anything slow below (GPU and region detection can take seconds),
    // so Settings never shows the switches in their unchecked HTML state.
    if (api.onIdle) api.onIdle(onIdle);
    if (api.getPrefs) api.getPrefs().then(renderPrefs);
    if (api.getConfig) {
      const config = await api.getConfig();
      // Before the saved settings are applied — usableMode() below depends on it.
      applyPlatform(config && config.platform);
      state.cpu = (config && config.cpu) || null;
      const regions = (config && config.regions) || {};
      const cfgDefaults = (config && config.defaults) || {};
      state.defaultRegion = cfgDefaults.region || state.defaultRegion;
      state.defaultWorker = cfgDefaults.worker || state.defaultWorker;
      // A cleared box shows the name a START will use.
      el.setWorker.placeholder = state.defaultWorker;
      el.setRegion.innerHTML = '';
      Object.keys(regions).forEach((key) => {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = regions[key].flag + ' ' + regions[key].label + ' · ' + regions[key].name;
        el.setRegion.appendChild(opt);
      });
    }
    let resumeMining = false;
    let regionSaved = false;
    if (api.getSettings) {
      const s = await api.getSettings();
      state.address = s.address || '';
      el.addrInput.value = state.address;
      state.mdlAddress = s.mdlAddress || '';
      state.shareLlm = s.shareLlm === true;
      state.xmrAddress = s.xmrAddress || '';
      state.cpuMining = !!state.xmrAddress.trim();
      state.xmrChecking = state.cpuMining;
      el.setXmr.value = state.xmrAddress;
      state.cpuThreads = Number(s.cpuThreads) || 0;
      el.setWorker.value = s.worker || state.defaultWorker;
      // main migrates a stale AlphaPool id before we see it, so this always
      // names an option that exists.
      el.setRegion.value = s.region || state.defaultRegion;
      regionSaved = !!s.regionSaved;
      // 'auto' to match shared/llmMode's DEFAULT_MODE, which is what main sends
      // for a fresh install. This fallback only fires on a FALSY stored mode —
      // a hand-edited or half-written settings.json holding null or "" — and it
      // used to land on 'mining', which switches the LLM off with no error
      // anywhere. That is indistinguishable from "the LLM is broken", so it must
      // not be the quiet default. A missing key already falls through to main's
      // default; this covers the value being present but empty.
      state.mode = usableMode(s.mode || 'auto');
      resumeMining = !!s.resumeMining;
    }
    renderMode();
    renderShare();
    if (api.onLlm) api.onLlm(renderLlm);
    if (api.getLlmStatus) api.getLlmStatus().then(renderLlm);
    if (api.onChatDelta) api.onChatDelta(onChatDelta);
    if (api.onChatDone) api.onChatDone(onChatDone);
    if (api.onChatError) api.onChatError(onChatError);
    if (api.onNodeStatus) api.onNodeStatus(renderNode);
    if (api.getNodeStatus) api.getNodeStatus().then(renderNode);
    if (api.detectGpu) {
      const gpu = await api.detectGpu();
      if (gpu) {
        state.gpu = gpu;
        if (!state.mining) el.deviceLabel.textContent = gpu;
      }
    }
    // Only a fresh install picks the nearest region. A saved one is the user's
    // choice: detection used to replace it on every launch, and the next START
    // saved the replacement. Offline at login, that was the US fallback.
    if (api.detectRegion && !state.mining && !regionSaved) {
      const region = await api.detectRegion();
      if (region) el.setRegion.value = region;
    }
    if (api.onStats) api.onStats(applyStats);
    if (api.onCpuStats) api.onCpuStats(applyCpuStats);
    if (api.onLog) api.onLog(appendLog);
    if (api.onStopped) api.onStopped(() => { state.mining = false; state.paused = false; renderMiningState(); });
    if (api.onEngine) api.onEngine((e) => {
      // Main sends only 'error': the miner could not start.
      if (!e || e.phase !== 'error') return;
      el.engineStatus.hidden = false;
      el.engineStatus.classList.add('err');
      el.engineStatus.textContent = e.message || 'Engine setup failed — see Logs.';
    });
    if (api.onUpdate) api.onUpdate((s) => {
      if (!s) return;
      el.updateStatus.hidden = !s.show || s.phase === 'checking';
      el.updateStatus.textContent = s.text;
      el.updateStatus.classList.toggle('err', !!s.error);
      if (s.phase !== 'checking') el.btnCheckUpdate.classList.remove('checking');
      if (s.ready) {
        updateReady = true;
        el.btnCheckUpdate.disabled = false;
        el.updateCheckLabel.textContent = 'Update & restart';
        el.btnCheckUpdate.classList.add('ready');
        // Announce it on the Mine view too: Settings is the one screen nobody
        // opens, so a downloaded update could sit there unnoticed.
        el.updateBarText.textContent = 'Update downloaded'
          + (s.version ? ' (v' + s.version + ')' : '') + ' — restart to apply.';
        el.updateBar.hidden = false;
      } else if (s.phase !== 'checking') {
        updateReady = false;
        el.btnCheckUpdate.disabled = false;
        el.updateCheckLabel.textContent = 'Check for updates';
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
    // Awaited so a rig that mines only Monero can resume too.
    await checkXmr();
    setInterval(() => { refreshBalance(); refreshXmrBalance(); }, BAL_REFRESH_MS);
    // start() resumes whatever START could start, an LLM-only or Auto run with
    // no payout address too. Main saves resumeMining for any running START.
    if (resumeMining) start();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
