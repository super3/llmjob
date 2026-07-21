'use strict';

// Renderer UI for the LLMJob Earn window. Pure display + IPC glue — all
// computation (mining, the LLM engine, chat streaming) happens in the main
// process (../shared + ../main). Kept thin and out of the coverage gate; the
// logic it leans on is unit-tested separately.
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
    line: $('mk-line'), area: $('mk-area'),
    deviceLabel: $('device-label'),
    mineDot: $('mine-dot'), llmHeroTps: $('llm-hero-tps'), llmHeroDot: $('llm-hero-dot'), llmHeroDetail: $('llm-hero-detail'),
    btnStart: $('btn-start'), btnStop: $('btn-stop'), engineStatus: $('engine-status'),
    // chat
    chatRunning: $('chat-running'), chatStopped: $('chat-stopped'), chatStoppedModel: $('chat-stopped-model'),
    chatHead: $('chat-head'), chatModel: $('chat-model'), chatNew: $('chat-new'),
    chatList: $('chat-list'), chatEmpty: $('chat-empty'), chatSuggestions: $('chat-suggestions'),
    chatMessages: $('chat-messages'), chatForm: $('chat-form'), chatInput: $('chat-input'), chatSend: $('chat-send'),
    // api
    apiRunning: $('api-running'), apiStopped: $('api-stopped'),
    apiEndpointUrl: $('api-endpoint-url'), apiCopy: $('api-copy'), apiOpen: $('api-open'),
    apiModel: $('api-model'),
    // connect
    connectHint: $('connect-hint'), connectForm: $('connect-form'), connectToken: $('connect-token'),
    connectError: $('connect-error'), connectLink: $('connect-link'), connectDashboard: $('connect-dashboard'),
    connectPairToggle: $('connect-pair-toggle'), connectPair: $('connect-pair'),
    connectDone: $('connect-done'), connectedTitle: $('connected-title'), connectedName: $('connected-name'),
    connectedAvatar: $('connected-avatar'), connectedRename: $('connected-rename'), connectDisconnect: $('connect-disconnect'),
    // settings
    modeSeg: $('mode-seg'), modeHint: $('mode-hint'),
    setWorker: $('set-worker'), setRegion: $('set-region'), setDifficulty: $('set-difficulty'),
    setMdl: $('set-mdl'), mdlNote: $('mdl-note'), mdlBalanceMeta: $('mdl-balance-meta'), mdlBalance: $('mdl-balance'),
    appVersion: $('app-version'), btnCheckUpdate: $('btn-check-update'), updateStatus: $('update-status'),
    logTerm: $('log-term'),
  };

  const state = {
    mining: false,       // master process running (miner and/or LLM per mode)
    view: 'mine',        // mine | chat | api | settings | logs
    returnTab: 'mine',   // where settings/logs return to
    address: '', gpu: '', mode: 'auto',
    llm: { ready: false, endpoint: null, webUrl: null, tps: 0, model: null, error: null },
    chat: { messages: [], streaming: false, streamText: '', bubble: null },
    node: { connected: false, nodeId: null, name: null },
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

  const MODE_HINTS = {
    auto: 'Balances mining and the local LLM from free VRAM.',
    mining: 'Pearl mining only — the local LLM stays off.',
    llm: 'Local model only — no mining, no payout address needed.',
    both: 'Mine and serve — the model takes ~5 GB VRAM, mining keeps the rest.',
  };

  // Prompt chips shown in the empty chat — the real model answers them.
  const SUGGESTIONS = [
    { title: 'What is LLMJob?', prompt: 'What is LLMJob?' },
    { title: 'What is PPLNS?', prompt: 'What is PPLNS?' },
    { title: 'Help me write an email', prompt: 'Help me write a short email asking my landlord to fix the heater.' },
  ];

  const MDL_NOTE = {
    empty: 'Earn <b>MDL</b> on the exact hashrate already mining Pearl — same shares, no extra power or hardware. Leave blank to mine Pearl only.',
    on: '<b class="ok">✓ Merge-mining MDL</b> — your Pearl hashrate now also earns MDL, credited by the pool.',
    bad: '<b class="warn">That doesn\'t look like an mdl1… address.</b> Double-check it, or clear the field to mine Pearl only.',
  };

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
    el.modeHint.textContent = MODE_HINTS[state.mode] || MODE_HINTS.auto;
  }

  function setMode(m) {
    state.mode = m;
    renderMode();
    el.btnStart.disabled = !canStart();
  }

  // START is allowed when we can mine (valid address) or the mode will run the
  // LLM anyway (llm/both/auto co-run the model even without a payout address).
  function canStart() {
    return isValid(state.address) || state.mode !== 'mining';
  }

  // ── Local LLM (GPU Activity hero + chat/api gating) ────────────────────────
  function renderLlmHero() {
    const err = state.llm.error;
    const ready = state.llm.ready;
    el.llmHeroTps.textContent = ready ? Number(state.llm.tps || 0).toFixed(1) : '0.0';
    el.llmHeroDot.className = 'dot2' + (err ? ' err' : ready ? ' on' : '');
    el.llmHeroDetail.textContent = err || state.llm.model || 'gemma-4-E4B-it';
    el.llmHeroDetail.classList.toggle('err', !!err);
    el.chatStoppedModel.textContent = state.llm.model || 'the local model';
    el.apiModel.textContent = state.llm.model || '—';
  }

  function renderChatGate() {
    const up = state.llm.ready;
    el.chatRunning.hidden = !up;
    el.chatStopped.hidden = up;
    updateSendEnabled();
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
      error: (s && s.error) || null,
    };
    renderLlmHero();
    renderChatGate();
    renderApiGate();
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
  const FLAT_LINE = 'M0 55 L480 55';
  const FLAT_AREA = 'M0 56 L0 55 L480 55 L480 56 Z';

  function chartPaths(points) {
    const W = 480, H = 56;
    if (!points || !points.length) return { line: FLAT_LINE, area: FLAT_AREA };
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
      el.uptime.textContent = '0m 00s';
      el.estday.textContent = '$0.00';
      el.deviceLabel.textContent = state.gpu || 'GPU · auto-detect';
      el.line.setAttribute('d', FLAT_LINE);
      el.area.setAttribute('d', FLAT_AREA);
      el.engineStatus.hidden = true;
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

  function renderMdlNote() {
    const v = String(el.setMdl.value || '').trim();
    const key = !v ? 'empty' : isValidMdl(v) ? 'on' : 'bad';
    el.mdlNote.innerHTML = MDL_NOTE[key];
  }

  function renderMdlBalanceMeta() {
    el.mdlBalanceMeta.hidden = !isValidMdl(String(el.setMdl.value || '').trim());
  }

  function resetMdlBalance() { el.mdlBalance.textContent = '0.000'; }

  async function refreshMdlBalance() {
    const mdl = String(el.setMdl.value || '').trim();
    const addr = state.address.trim();
    if (!isValidMdl(mdl) || !isValid(addr) || !api.getMdlBalance) return;
    const b = await api.getMdlBalance(addr);
    if (!b || mdl !== String(el.setMdl.value || '').trim() || addr !== state.address.trim()) return;
    if (b.mdlAddress && b.mdlAddress.toLowerCase() !== mdl.toLowerCase()) return;
    el.mdlBalance.textContent = fmt3(b.earned);
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
      region: el.setRegion.value || 'us2',
      difficulty: Number(el.setDifficulty.value) || 524288,
      mode: state.mode || 'mining',
    };
  }

  function start() {
    if (!canStart()) return;
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

  // START LLM (from the Chat/API tabs): make sure the compute mode actually runs
  // the model, then start. Mining-only becomes Mining+LLM (or LLM-only with no
  // payout address); llm/both/auto start as-is.
  function startLlmIntent() {
    if (state.llm.ready) return;
    if (state.mode === 'mining') setMode(isValid(state.address) ? 'both' : 'llm');
    start();
  }

  // Keep the OS window sized to the content: a tab switch or a state change
  // (mining start/stop, LLM coming up) changes the app's height, and without
  // this the frame keeps its old size — leaving a gap under the footer or
  // clipping a taller view. Inner scrollers (chat, logs) are bounded, so this
  // fires on discrete layout changes, not per streamed token.
  function watchWindowFit() {
    if (!api.fitWindow || typeof ResizeObserver === 'undefined') return;
    const appEl = document.querySelector('.app');
    if (!appEl) return;
    let t = null;
    const ro = new ResizeObserver(() => {
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
      if (mdlBalDebounce) clearTimeout(mdlBalDebounce);
      if (isValid(state.address)) {
        balDebounce = setTimeout(refreshBalance, 600);
        mdlBalDebounce = setTimeout(refreshMdlBalance, 600);
      } else {
        resetBalance();
        resetMdlBalance();
      }
    });
    el.setMdl.addEventListener('input', () => {
      renderMdlNote();
      renderMdlBalanceMeta();
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
    let resumeMining = false;
    if (api.getSettings) {
      const s = await api.getSettings();
      state.address = s.address || '';
      el.addrInput.value = state.address;
      el.setWorker.value = s.worker || 'rig01';
      el.setRegion.value = s.region || 'us2';
      el.setDifficulty.value = s.difficulty || 524288;
      el.setMdl.value = s.mdlAddress || '';
      state.mode = s.mode || 'mining';
      resumeMining = !!(s.resumeMining && isValid(state.address));
    }
    renderMode();
    renderMdlNote();
    renderMdlBalanceMeta();
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
        if (api.difficultyForCard && Number(el.setDifficulty.value) === 524288) {
          const d = await api.difficultyForCard(gpu);
          if (d) el.setDifficulty.value = d;
        }
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
      } else if (s.phase !== 'checking') {
        updateReady = false;
        el.btnCheckUpdate.disabled = false;
        el.btnCheckUpdate.textContent = 'Check for updates';
        el.btnCheckUpdate.classList.remove('ready');
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
    refreshMdlBalance();
    setInterval(refreshMdlBalance, BAL_REFRESH_MS);
    if (resumeMining) start();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
