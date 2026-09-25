/* The right-hand column. Everything here is measured by the server this session —
   no figure on this panel comes from a paper or a previous run. */

import { CLASS_COLORS, DRIVER_COLORS, HAZARD_IDS } from './palette.js';

const $ = (id) => document.getElementById(id);

const CLASS_NAMES = {
  0: 'unknown', 1: 'drivable', 2: 'rough ground', 3: 'vegetation', 4: 'wall',
  5: 'pole', 6: 'barrier', 7: 'pedestrian', 8: 'vehicle', 9: 'two-wheeler',
  40: 'pothole', 41: 'ditch', 42: 'loose rock', 43: 'water crossing', 44: 'snow drift',
};

export function bytes(n) {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} kB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

const compact = (n) => (n == null ? '—' : n.toLocaleString('en-US'));

export class Telemetry {
  constructor(system) {
    this.system = system;
    this.spark = $('spark');
    this.sparkCtx = this.spark.getContext('2d');
    this.targetHz = system?.target_hz || 10;
    this.lastLive = null;
    this._sizeSpark();
    window.addEventListener('resize', () => this._sizeSpark());
  }

  _sizeSpark() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.spark.clientWidth || 240;
    this.spark.width = Math.round(w * dpr);
    this.spark.height = Math.round(34 * dpr);
    this.sparkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._sparkW = w;
  }

  update(frame) {
    const t = frame.tail || {};
    const st = t.stats || {};
    const roll = t.rolling || {};
    const timing = t.timing || {};

    this.lastLive = { roll, st, timing, frame };

    // ------------------------------------------------------------- rate ------
    $('t-fps').textContent = roll.fps != null ? roll.fps.toFixed(1) : '—';
    $('t-frame-ms').textContent = roll.frame_ms_mean != null ? roll.frame_ms_mean.toFixed(1) : '—';
    $('t-frame-p95').textContent = roll.frame_ms_p95 != null ? roll.frame_ms_p95.toFixed(1) : '—';

    const tgt = $('t-target');
    if (roll.fps != null) {
      const met = roll.fps >= this.targetHz;
      tgt.textContent = met
        ? `Clearing the ${this.targetHz} Hz requirement with ${(roll.fps - this.targetHz).toFixed(1)} Hz to spare.`
        : `Below the ${this.targetHz} Hz requirement by ${(this.targetHz - roll.fps).toFixed(1)} Hz.`;
      tgt.className = `rate-target ${met ? 'is-met' : 'is-missed'}`;
    }
    this._drawSpark(roll.history_ms || []);

    // ------------------------------------------------------------ stages -----
    const stages = [
      ['Perception', roll.perception_ms ?? timing.perception_ms],
      ['Grid build', roll.grid_ms ?? timing.grid_ms],
      ['Transmit tier', timing.decimate_ms],
      ['Encode', roll.codec_ms],
    ].filter(([, v]) => v != null && v > 0.001);
    const max = Math.max(...stages.map(([, v]) => v), 0.001);
    $('stages').innerHTML = stages.map(([name, v]) => `
      <div class="stage-row">
        <span>${name}</span>
        <span class="stage-bar"><span class="stage-fill" style="width:${(v / max * 100).toFixed(1)}%"></span></span>
        <span class="num">${v.toFixed(1)}</span>
      </div>`).join('');

    // --------------------------------------------------------------- map -----
    $('t-points').textContent = compact(st.n_points);
    $('t-cells').textContent = compact(st.n_cells);
    $('t-cellmin').textContent = st.cell_size_min_m != null ? `${(st.cell_size_min_m * 100).toFixed(0)} cm` : '—';
    $('t-cellmax').textContent = st.cell_size_max_m != null ? `${(st.cell_size_max_m * 100).toFixed(0)} cm` : '—';
    $('t-elev').textContent = st.elevation_range_m
      ? `${st.elevation_range_m[0].toFixed(1)} to ${st.elevation_range_m[1].toFixed(1)} m` : '—';

    // ------------------------------------------------------------ memory -----
    const m = st.memory || {};
    const rows = [
      { label: 'This map', v: m.adaptive_bytes, ours: true },
      { label: 'Raw point cloud', v: m.raw_cloud_bytes },
      { label: 'Uniform 5 cm 2.5D grid', v: m.dense_uniform_bytes },
      { label: 'Uniform 5 cm 3D voxels', v: m.dense_3d_voxel_bytes },
    ].filter(r => r.v);
    if (rows.length) {
      const lmax = Math.log10(Math.max(...rows.map(r => r.v)));
      const lmin = Math.log10(Math.min(...rows.map(r => r.v)));
      $('mem-bars').innerHTML = rows.map(r => {
        const f = lmax === lmin ? 1 : (Math.log10(r.v) - lmin) / (lmax - lmin);
        return `<div class="bar-row ${r.ours ? 'is-ours' : ''}">
          <div class="bar-head"><span>${r.label}</span><span class="num">${bytes(r.v)}</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${(4 + f * 96).toFixed(1)}%"></div></div>
        </div>`;
      }).join('');
      $('mem-note').textContent =
        `${m.reduction_vs_dense}× smaller than the uniform 5 cm grid, ` +
        `${m.reduction_vs_dense_3d}× smaller than 5 cm voxels, ` +
        `${m.reduction_vs_cloud}× smaller than the cloud it came from. ` +
        `Bars use a log scale.`;
    }

    // -------------------------------------------------------------- link -----
    const link = frame.link || {};
    $('t-wire').textContent = bytes(frame.wireBytes);
    $('t-ratio').textContent = link.compression_ratio ? `${link.compression_ratio}×` : '—';
    $('t-mbps').textContent = link.mbps != null ? `${link.mbps.toFixed(2)} Mbit/s` : '—';
    if (link.mbps != null) {
      const fits1 = link.mbps <= 1.0, fits256 = link.mbps <= 0.256;
      $('link-fit').innerHTML =
        `<span class="${fits1 ? 'fit' : 'nofit'}">${fits1 ? 'Fits' : 'Exceeds'} a 1 Mbit/s link</span> · ` +
        `<span class="${fits256 ? 'fit' : 'nofit'}">${fits256 ? 'fits' : 'exceeds'} 256 kbit/s</span>`;
    }

    // -------------------------------------------------------------- risk -----
    $('t-haz').textContent = compact(st.hazard_cells);
    $('t-hazfar').textContent = compact(st.hazard_cells_beyond_fovea);
    $('t-degraded').textContent = compact(st.degraded_cells);
    $('t-risk').textContent = compact(st.risk_driven_cells);

    const drv = st.cells_by_driver || {};
    const dmax = Math.max(...Object.values(drv), 1);
    const order = ['range', 'hazard', 'low confidence', 'class boundary'];
    const dcolor = { range: DRIVER_COLORS[0], hazard: DRIVER_COLORS[1],
                     'low confidence': DRIVER_COLORS[2], 'class boundary': DRIVER_COLORS[3] };
    $('drivers').innerHTML = order.filter(k => drv[k]).map(k => `
      <div class="driver-row">
        <span>${k}</span>
        <span class="driver-bar"><span class="driver-fill" style="width:${(drv[k] / dmax * 100).toFixed(1)}%;background:${dcolor[k]}"></span></span>
        <span class="num">${compact(drv[k])}</span>
      </div>`).join('');

    // ----------------------------------------------------------- objects -----
    const dets = t.detections || [];
    $('objects').innerHTML = dets.length
      ? dets.slice(0, 10).map(d => `
          <div class="obj-row">
            <span style="color:${CLASS_COLORS[d.cls] || '#ccc'}">${CLASS_NAMES[d.cls] || d.cls}</span>
            <span class="obj-r">${d.r.toFixed(1)} m</span>
            <span class="obj-s">${d.score.toFixed(2)}</span>
          </div>`).join('')
      : '<p class="tel-note">None in range.</p>';
  }

  _drawSpark(history) {
    const ctx = this.sparkCtx;
    const w = this._sparkW, h = 34;
    ctx.clearRect(0, 0, w, h);
    if (!history.length) return;

    const budget = 1000 / this.targetHz;
    const max = Math.max(budget * 1.25, ...history);

    // The frame budget, drawn as the line everything must stay under.
    const by = h - (budget / max) * h;
    ctx.strokeStyle = '#be3a21';
    ctx.globalAlpha = 0.55;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(0, by); ctx.lineTo(w, by);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    ctx.strokeStyle = '#59615a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    history.forEach((v, i) => {
      const x = (i / Math.max(history.length - 1, 1)) * w;
      const y = h - (v / max) * h;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  renderLegend(el, legend, mode) {
    if (!legend) { el.innerHTML = ''; return; }
    if (legend.kind === 'ramp') {
      el.innerHTML = `
        <div class="legend-ramp" style="background:linear-gradient(90deg,${legend.stops.join(',')})"></div>
        <div class="legend-ends"><span class="num">${legend.lo}</span><span class="num">${legend.hi}</span></div>`;
      return;
    }
    if (legend.kind === 'list') {
      el.innerHTML = legend.rows.map(r => `
        <div class="legend-row">
          <span class="legend-sw" style="background:${r.color}"></span>
          <span>${r.label}</span>
          <span class="legend-count num">${compact(r.count)}</span>
        </div>`).join('');
      return;
    }
    el.innerHTML = legend.ids.map((id, i) => `
      <div class="legend-row ${HAZARD_IDS.has(id) ? 'is-hazard' : ''}">
        <span class="legend-sw" style="background:${legend.colors[i]}"></span>
        <span>${CLASS_NAMES[id] || id}</span>
        <span class="legend-count num">${compact(legend.counts[i])}</span>
      </div>`).join('');
  }
}
