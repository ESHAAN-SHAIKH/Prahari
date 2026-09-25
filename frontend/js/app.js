/* Console bootstrap: fetch the system description, open the frame stream, wire controls. */

import { decodeFrame } from './wire.js';
import { Viewport } from './viewport.js';
import { Telemetry } from './telemetry.js';
import { renderEvidence, renderLive } from './evidence.js';

const $ = (id) => document.getElementById(id);

const MODE_NOTES = {
  semantic: 'Terrain, static structure and moving objects, coloured by class.',
  elevation: 'Mean elevation per cell — the 2.5D payload a flat occupancy grid throws away.',
  confidence: 'How sure the perception stage is. Red marks degraded sensing, which is what triggers refinement.',
  resolution: 'Cell size. Fine near the vehicle, coarse at range — and fine again wherever risk says so.',
};

let system = null;
let viewport = null;
let telemetry = null;
let socket = null;
let streaming = true;
let lastLive = null;

// ────────────────────────────────────────────────────────────────── startup ───

async function boot() {
  try {
    system = await (await fetch('/api/system')).json();
  } catch (err) {
    notice(`Cannot reach the server: ${err.message}`);
    return;
  }

  document.title = `PRAHARI — ${system.subtitle}`;
  renderSource(system.perception);
  renderLevels(system.level_table);
  if (system.perception.fallback_note) notice(system.perception.fallback_note);

  telemetry = new Telemetry(system);
  viewport = new Viewport($('map'), { onCursor: showCursor });

  window.addEventListener('resize', () => viewport.resize());
  wireControls();
  connect();
}

function notice(text) {
  const el = $('notice');
  el.textContent = text;
  el.hidden = false;
}

function renderSource(p) {
  const el = $('source-badge');
  const live = p.source === 'checkpoint';
  el.className = `source ${live ? 'is-live' : 'is-synthetic'}`;
  el.querySelector('.source-text').textContent =
    live ? p.label : 'Synthetic sensor — not measured data';
  el.title = p.notes || '';
}

function renderLevels(rows) {
  $('levels').innerHTML = (rows || []).map(r => `
    <span class="level-item">
      <span>level ${r.level}</span>
      <span class="num">${(r.size_m * 100).toFixed(0)} cm</span>
      <span>${r.range_from_m != null ? `${r.range_from_m}–${r.range_to_m} m` : 'unused'}</span>
    </span>`).join('');
}

// ─────────────────────────────────────────────────────────────────── stream ───

let pollTimer = null;
let wsAttempts = 0;
const MAX_WS_ATTEMPTS = 2;
const streamParams = {
  risk_adaptive: true,
  compress: true,
  conf_threshold: 0.60,
  link_tier: null,
  preserve_risk_on_link: true,
};

function startHttpPolling() {
  if (pollTimer) return;
  if (socket) {
    socket.onclose = null;
    socket.onerror = null;
    socket.close();
    socket = null;
  }
  setConn('live (http)', true);

  async function poll() {
    if (!streaming) {
      pollTimer = setTimeout(poll, 400);
      return;
    }
    try {
      const q = new URLSearchParams();
      q.set('compress', String(streamParams.compress));
      q.set('risk_adaptive', String(streamParams.risk_adaptive));
      q.set('conf_threshold', String(streamParams.conf_threshold));
      if (streamParams.link_tier !== null) q.set('link_tier', String(streamParams.link_tier));
      q.set('preserve_risk_on_link', String(streamParams.preserve_risk_on_link));

      const res = await fetch(`/api/frame?${q.toString()}`);
      if (res.ok) {
        const buf = await res.arrayBuffer();
        const frame = await decodeFrame(buf);
        onFrame(frame);
        setConn('live', true);
      } else {
        setConn('error', false);
      }
    } catch (err) {
      setConn('reconnecting', false);
    }
    const delay = Math.max(50, Math.floor(1000 / (system?.target_hz || 10)));
    pollTimer = setTimeout(poll, delay);
  }

  poll();
}

function connect() {
  if (pollTimer) return;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  try {
    socket = new WebSocket(`${proto}://${location.host}/ws/stream`);
  } catch (err) {
    startHttpPolling();
    return;
  }
  socket.binaryType = 'arraybuffer';

  socket.onopen = () => {
    wsAttempts = 0;
    setConn('live', true);
  };
  socket.onclose = () => {
    wsAttempts++;
    if (wsAttempts >= MAX_WS_ATTEMPTS) {
      startHttpPolling();
    } else {
      setConn('reconnecting', false);
      setTimeout(connect, 1000);
    }
  };
  socket.onerror = () => {
    wsAttempts++;
    if (wsAttempts >= MAX_WS_ATTEMPTS) {
      startHttpPolling();
    }
  };
  socket.onmessage = async (ev) => {
    try {
      const frame = await decodeFrame(ev.data);
      onFrame(frame);
    } catch (err) {
      notice(`Frame decode failed: ${err.message}`);
    }
  };
}

function setConn(text, up) {
  const el = $('conn');
  el.textContent = text;
  el.className = `conn ${up ? 'is-up' : 'is-down'}`;
}

function send(patch) {
  Object.assign(streamParams, patch);
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ cmd: 'set', ...patch }));
  }
}

function onFrame(frame) {
  $('plot-empty').hidden = true;
  $('frame-id').textContent = `frame ${frame.frame}`;

  // Link cost is computed here from the bytes that actually arrived.
  const roll = frame.tail?.rolling || {};
  const hz = roll.fps || system.target_hz;
  frame.link = {
    mbps: (frame.wireBytes * 8 * hz) / 1e6,
    compression_ratio: frame.tail?.stats
      ? +(estimateRaw(frame) / frame.wireBytes).toFixed(2) : null,
  };

  viewport.setFrame(frame);
  telemetry.update(frame);
  telemetry.renderLegend($('legend'), viewport.legend, viewport.opts.mode);
  lastLive = telemetry.lastLive;

  const bar = viewport.scaleBar();
  $('scale-label').textContent = `${bar.metres} m`;
  document.querySelector('.scalebar-rule').style.width = `${bar.px.toFixed(0)}px`;

  renderSource({ ...system.perception, source: frame.tail?.source,
                 label: frame.tail?.source_label, notes: frame.tail?.notes });
}

/** Bytes this frame would have taken uncompressed, for the compression ratio. */
function estimateRaw(frame) {
  const tailBytes = JSON.stringify(frame.tail).length;
  return 26 + frame.n * 9 + tailBytes;
}

function showCursor(w) {
  $('cursor-readout').textContent = w
    ? `x ${w.x.toFixed(1)}  y ${w.y.toFixed(1)}  range ${Math.hypot(w.x, w.y).toFixed(1)} m`
    : '—';
}

// ─────────────────────────────────────────────────────────────── controls ─────

function wireControls() {
  document.querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.seg-btn').forEach(b => {
        b.classList.toggle('is-on', b === btn);
        b.setAttribute('aria-checked', String(b === btn));
      });
      viewport.setOptions({ mode: btn.dataset.mode });
      telemetry.renderLegend($('legend'), viewport.legend, btn.dataset.mode);
      $('mode-note').textContent = MODE_NOTES[btn.dataset.mode];
    });
  });

  const overlay = (id, key) => $(id).addEventListener('change', (e) =>
    viewport.setOptions({ [key]: e.target.checked }));
  overlay('opt-quadtree', 'quadtree');
  overlay('opt-rings', 'rings');
  overlay('opt-detections', 'detections');
  overlay('opt-hazards', 'hazards');

  $('opt-risk').addEventListener('change', e =>
    send({ risk_adaptive: e.target.checked }));

  $('opt-conf').addEventListener('input', e => {
    $('opt-conf-out').textContent = (+e.target.value).toFixed(2);
    send({ conf_threshold: +e.target.value });
  });

  $('opt-tier').addEventListener('change', e =>
    send({ link_tier: e.target.value === '' ? null : +e.target.value }));

  $('opt-preserve').addEventListener('change', e =>
    send({ preserve_risk_on_link: e.target.checked }));

  $('stream-toggle').addEventListener('click', () => {
    streaming = !streaming;
    $('stream-toggle').textContent = streaming ? 'Pause' : 'Resume';
    $('stream-toggle').setAttribute('aria-pressed', String(streaming));
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ cmd: streaming ? 'resume' : 'pause' }));
    }
  });

  document.querySelectorAll('.view-tab').forEach(tab => {
    tab.addEventListener('click', () => switchView(tab.dataset.view));
  });

  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.key === 'r' || e.key === 'R') viewport.reset();
  });
}

async function switchView(name) {
  document.querySelectorAll('.view-tab').forEach(t => {
    const on = t.dataset.view === name;
    t.classList.toggle('is-active', on);
    t.setAttribute('aria-selected', String(on));
  });
  $('view-live').classList.toggle('is-active', name === 'live');
  $('view-evidence').classList.toggle('is-active', name === 'evidence');

  if (name === 'live') {
    requestAnimationFrame(() => viewport.resize());
    return;
  }
  try {
    const data = await (await fetch('/api/metrics')).json();
    renderEvidence(data, lastLive);
  } catch (err) {
    notice(`Could not load evidence: ${err.message}`);
    renderLive(lastLive);
  }
}

boot();
