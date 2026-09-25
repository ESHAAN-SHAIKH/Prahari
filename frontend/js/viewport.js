/* The plotting surface.

   Top-down, sensor at the origin, +x forward and +y to the left — the vehicle frame,
   not a screen frame, so what you read off the map is what the autonomy stack sees.

   Cells are drawn batched by colour: a frame is tens of thousands of rectangles, and
   changing fillStyle is the expensive part, not the fill itself. */

import {
  CLASS_COLORS, ELEVATION_RAMP, CONFIDENCE_RAMP, LEVEL_RAMP,
  HAZARD_IDS, PLOT_BG, RING, EGO, RED_PENCIL, CELL_EDGE,
} from './palette.js';

const RANGE_RINGS = [10, 30, 50, 100];

export class Viewport {
  constructor(canvas, { onCursor } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.onCursor = onCursor;

    this.frame = null;
    this.opts = { mode: 'semantic', quadtree: true, rings: true,
                  detections: true, hazards: true };

    this.centre = { x: 26, y: 0 };      // look up the road, not at your own wheels
    this.scale = 8;                      // px per metre, set properly on first resize
    this._fitted = false;
    this._drag = null;
    this._colorIndex = null;
    this._palette = null;

    this._bindEvents();
    this.resize();
  }

  // ------------------------------------------------------------- interaction --
  _bindEvents() {
    const c = this.canvas;

    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = c.getBoundingClientRect();
      const before = this.toWorld(e.clientX - rect.left, e.clientY - rect.top);
      const k = Math.exp(-e.deltaY * 0.0015);
      this.scale = Math.min(220, Math.max(1.2, this.scale * k));
      const after = this.toWorld(e.clientX - rect.left, e.clientY - rect.top);
      this.centre.x += before.x - after.x;
      this.centre.y += before.y - after.y;
      this.render();
    }, { passive: false });

    c.addEventListener('pointerdown', (e) => {
      this._drag = { x: e.clientX, y: e.clientY };
      c.setPointerCapture(e.pointerId);
    });

    c.addEventListener('pointermove', (e) => {
      const rect = c.getBoundingClientRect();
      if (this._drag) {
        this.centre.x -= (e.clientY - this._drag.y) / this.scale * -1;
        this.centre.y += (e.clientX - this._drag.x) / this.scale;
        this._drag = { x: e.clientX, y: e.clientY };
        this.render();
      }
      if (this.onCursor) {
        this.onCursor(this.toWorld(e.clientX - rect.left, e.clientY - rect.top));
      }
    });

    const end = (e) => {
      this._drag = null;
      if (e.pointerId !== undefined && this.canvas.hasPointerCapture?.(e.pointerId)) {
        this.canvas.releasePointerCapture(e.pointerId);
      }
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('pointerleave', () => { if (this.onCursor) this.onCursor(null); });
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.w = Math.max(1, Math.round(r.width));
    this.h = Math.max(1, Math.round(r.height));
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!this._fitted && this.w > 1) {
      this.scale = Math.min(this.w, this.h) / 95;
      this._fitted = true;
    }
    this.render();
  }

  /** Fit the whole mapped area back in view. */
  reset() {
    this.centre = { x: 26, y: 0 };
    this.scale = Math.min(this.w, this.h) / 95;
    this.render();
  }

  // -------------------------------------------------------------- projection --
  toScreen(x, y) {
    return {
      sx: this.w / 2 - (y - this.centre.y) * this.scale,
      sy: this.h / 2 - (x - this.centre.x) * this.scale,
    };
  }

  toWorld(sx, sy) {
    return {
      x: this.centre.x - (sy - this.h / 2) / this.scale,
      y: this.centre.y - (sx - this.w / 2) / this.scale,
    };
  }

  // ------------------------------------------------------------------- state --
  setFrame(frame) {
    this.frame = frame;
    this._colorIndex = null;
    this.render();
  }

  setOptions(patch) {
    const modeChanged = patch.mode && patch.mode !== this.opts.mode;
    Object.assign(this.opts, patch);
    if (modeChanged) this._colorIndex = null;
    this.render();
  }

  /** Map every cell to an index into a small palette, so drawing can batch by colour. */
  _buildColorIndex() {
    const f = this.frame;
    const mode = this.opts.mode;
    const idx = new Uint8Array(f.n);
    let palette;

    if (mode === 'elevation') {
      palette = ELEVATION_RAMP;
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < f.n; i++) { if (f.z[i] < lo) lo = f.z[i]; if (f.z[i] > hi) hi = f.z[i]; }
      const span = Math.max(hi - lo, 0.01);
      for (let i = 0; i < f.n; i++) {
        idx[i] = Math.min(palette.length - 1,
                          Math.floor(((f.z[i] - lo) / span) * palette.length));
      }
      this.legend = { kind: 'ramp', stops: palette,
                      lo: `${lo.toFixed(2)} m`, hi: `${hi.toFixed(2)} m` };

    } else if (mode === 'confidence') {
      palette = CONFIDENCE_RAMP;
      for (let i = 0; i < f.n; i++) {
        idx[i] = Math.min(palette.length - 1, Math.floor(f.conf[i] * palette.length));
      }
      this.legend = { kind: 'ramp', stops: palette, lo: '0.00', hi: '1.00' };

    } else if (mode === 'resolution') {
      const levels = [...new Set(Array.from(f.level))].sort((a, b) => a - b);
      palette = levels.map((_, i) =>
        LEVEL_RAMP[Math.round((i / Math.max(levels.length - 1, 1)) * (LEVEL_RAMP.length - 1))]);
      const pos = new Map(levels.map((l, i) => [l, i]));
      for (let i = 0; i < f.n; i++) idx[i] = pos.get(f.level[i]);
      this.legend = {
        kind: 'list',
        rows: levels.map((l, i) => ({
          color: palette[i],
          label: `${(f.root / (1 << l)).toFixed(2)} m`,
          count: countWhere(f.level, l),
        })),
      };

    } else {
      const ids = [...new Set(Array.from(f.cls))].sort((a, b) => a - b);
      palette = ids.map(id => CLASS_COLORS[id] || '#4a544d');
      const pos = new Map(ids.map((id, i) => [id, i]));
      for (let i = 0; i < f.n; i++) idx[i] = pos.get(f.cls[i]);
      this.legend = { kind: 'classes', ids, colors: palette,
                      counts: ids.map(id => countWhere(f.cls, id)) };
    }

    this._colorIndex = idx;
    this._palette = palette;
  }

  // ------------------------------------------------------------------ render --
  render() {
    const ctx = this.ctx;
    ctx.fillStyle = PLOT_BG;
    ctx.fillRect(0, 0, this.w, this.h);

    if (this.opts.rings) this._drawRings(ctx);
    if (!this.frame || this.frame.n === 0) return;
    if (!this._colorIndex) this._buildColorIndex();

    this._drawCells(ctx);
    if (this.opts.hazards) this._drawHazardMarks(ctx);
    if (this.opts.detections) this._drawDetections(ctx);
    this._drawEgo(ctx);
  }

  _drawRings(ctx) {
    const o = this.toScreen(0, 0);
    ctx.save();
    ctx.strokeStyle = RING;
    ctx.lineWidth = 1;
    for (const r of RANGE_RINGS) {
      const rp = r * this.scale;
      if (rp < 14 || rp > Math.hypot(this.w, this.h)) continue;
      ctx.globalAlpha = r === 10 ? 0.55 : 0.22;
      ctx.setLineDash(r === 10 ? [5, 4] : []);
      ctx.beginPath();
      ctx.arc(o.sx, o.sy, rp, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = RING;
      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.fillText(`${r} m`, o.sx + 4, o.sy - rp - 3);
    }
    ctx.restore();
  }

  _drawCells(ctx) {
    const f = this.frame;
    const idx = this._colorIndex;
    const pal = this._palette;
    const s = this.scale;
    const cx = this.w / 2, cy = this.h / 2;
    const ox = this.centre.x, oy = this.centre.y;

    // One pass per colour keeps fillStyle changes down to the size of the palette.
    const buckets = Array.from({ length: pal.length }, () => []);
    for (let i = 0; i < f.n; i++) {
      const half = f.size[i] * 0.5;
      const sx = cx - (f.y[i] + half - oy) * s;
      const sy = cy - (f.x[i] + half - ox) * s;
      const px = f.size[i] * s;
      if (sx > this.w || sy > this.h || sx + px < 0 || sy + px < 0) continue;
      buckets[idx[i]].push(sx, sy, px);
    }

    for (let b = 0; b < buckets.length; b++) {
      const list = buckets[b];
      if (!list.length) continue;
      ctx.fillStyle = pal[b];
      for (let k = 0; k < list.length; k += 3) {
        const px = Math.max(list[k + 2], 1);
        ctx.fillRect(list[k], list[k + 1], px, px);
      }
    }

    // Cell boundaries, only where they are actually readable.
    if (this.opts.quadtree) {
      ctx.save();
      ctx.strokeStyle = CELL_EDGE;
      ctx.lineWidth = 1;
      ctx.beginPath();
      let drawn = 0;
      for (let i = 0; i < f.n && drawn < 30000; i++) {
        const px = f.size[i] * s;
        if (px < 4) continue;
        const half = f.size[i] * 0.5;
        const sx = cx - (f.y[i] + half - oy) * s;
        const sy = cy - (f.x[i] + half - ox) * s;
        if (sx > this.w || sy > this.h || sx + px < 0 || sy + px < 0) continue;
        ctx.rect(Math.round(sx) + 0.5, Math.round(sy) + 0.5, Math.round(px), Math.round(px));
        drawn++;
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  _drawHazardMarks(ctx) {
    const f = this.frame;
    // Cluster hazard cells into 3 m bins so one pothole gets one mark, not forty.
    const bins = new Map();
    for (let i = 0; i < f.n; i++) {
      if (!HAZARD_IDS.has(f.cls[i])) continue;
      const key = `${Math.round(f.x[i] / 3)}:${Math.round(f.y[i] / 3)}`;
      const b = bins.get(key) || { x: 0, y: 0, n: 0, cls: f.cls[i] };
      b.x += f.x[i]; b.y += f.y[i]; b.n++;
      bins.set(key, b);
    }
    if (!bins.size) return;

    ctx.save();
    ctx.strokeStyle = RED_PENCIL;
    ctx.lineWidth = 1.25;
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.fillStyle = RED_PENCIL;
    for (const b of bins.values()) {
      const x = b.x / b.n, y = b.y / b.n;
      const { sx, sy } = this.toScreen(x, y);
      const r = Math.max(7, Math.min(26, 1.6 * this.scale));
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(sx - r - 4, sy); ctx.lineTo(sx - r + 2, sy);
      ctx.moveTo(sx + r - 2, sy); ctx.lineTo(sx + r + 4, sy);
      ctx.stroke();
      const rng = Math.hypot(x, y);
      if (this.scale > 3) ctx.fillText(`${rng.toFixed(0)} m`, sx + r + 6, sy + 3);
    }
    ctx.restore();
  }

  _drawDetections(ctx) {
    const dets = this.frame.tail?.detections || [];
    if (!dets.length) return;
    ctx.save();
    ctx.lineWidth = 1.25;
    ctx.font = '10px "IBM Plex Mono", monospace';
    for (const d of dets) {
      const color = CLASS_COLORS[d.cls] || EGO;
      const { sx, sy } = this.toScreen(d.x, d.y);
      const wpx = Math.max(6, (d.w || 1) * this.scale);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.strokeRect(sx - wpx / 2, sy - wpx / 2, wpx, wpx);
      if (this.scale > 2.5) {
        ctx.globalAlpha = 0.75;
        ctx.fillStyle = color;
        ctx.fillText(`${d.r.toFixed(0)} m  ${d.score.toFixed(2)}`, sx + wpx / 2 + 4, sy + 3);
      }
    }
    ctx.restore();
  }

  _drawEgo(ctx) {
    const { sx, sy } = this.toScreen(0, 0);
    const r = 7;
    ctx.save();
    ctx.fillStyle = EGO;
    ctx.beginPath();
    ctx.moveTo(sx, sy - r);
    ctx.lineTo(sx + r * 0.62, sy + r * 0.72);
    ctx.lineTo(sx, sy + r * 0.34);
    ctx.lineTo(sx - r * 0.62, sy + r * 0.72);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  /** A round number of metres that fits in roughly 80 px, for the scale bar. */
  scaleBar() {
    const target = 80 / this.scale;
    const pow = Math.pow(10, Math.floor(Math.log10(target)));
    const metres = [1, 2, 5, 10].map(m => m * pow).find(m => m >= target) || pow * 10;
    return { metres, px: metres * this.scale };
  }
}

function countWhere(arr, v) {
  let c = 0;
  for (let i = 0; i < arr.length; i++) if (arr[i] === v) c++;
  return c;
}
