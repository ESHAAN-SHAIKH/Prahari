/* Colour on the plotting surface.

   Semantic colours are cartographic rather than signal-coloured: sage for drivable
   ground, ochre for what you can cross but shouldn't, slate for structure, brass for
   posts, chalk for people. Red is spent entirely on hazard and degraded sensing, so a
   red mark on this map always means the same thing. */

export const CLASS_COLORS = {
  0:  '#39423d',   // unknown
  1:  '#6f8f74',   // drivable
  2:  '#97824f',   // rough ground
  3:  '#46603f',   // vegetation
  4:  '#7b8794',   // wall
  5:  '#c9ac63',   // pole
  6:  '#94806a',   // barrier
  7:  '#eef1e8',   // pedestrian — the brightest thing on the map, deliberately
  8:  '#9fbcd6',   // vehicle
  9:  '#c2d4e4',   // two-wheeler
  40: '#c84024',   // pothole
  41: '#a8301c',   // ditch
  42: '#d4674a',   // loose rock
  43: '#b0452f',   // water crossing
  44: '#d98a72',   // snow drift
};

export const DRIVER_COLORS = {
  0: '#6c7a72',    // range
  1: '#c84024',    // hazard
  2: '#d4a24a',    // low confidence
  3: '#7fa9c4',    // class boundary
};

export const HAZARD_IDS = new Set([40, 41, 42, 43, 44]);

function lerp(a, b, t) { return a + (b - a) * t; }

function hexToRgb(h) {
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function rgbToHex([r, g, b]) {
  return '#' + [r, g, b].map(v => Math.round(v).toString(16).padStart(2, '0')).join('');
}

/** Sample a list of hex stops into `steps` colours. */
export function ramp(stops, steps) {
  const rgb = stops.map(hexToRgb);
  const out = [];
  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? 0 : (i / (steps - 1)) * (rgb.length - 1);
    const k = Math.min(Math.floor(t), rgb.length - 2);
    const f = t - k;
    out.push(rgbToHex([
      lerp(rgb[k][0], rgb[k + 1][0], f),
      lerp(rgb[k][1], rgb[k + 1][1], f),
      lerp(rgb[k][2], rgb[k + 1][2], f),
    ]));
  }
  return out;
}

// Hypsometric tint: water-slate low ground through sage to bone-white high ground.
export const ELEVATION_STOPS = ['#2b3f52', '#3f6b5a', '#6f8f74', '#a9a878', '#ded9c3'];
export const ELEVATION_RAMP = ramp(ELEVATION_STOPS, 24);

// Confidence: the red pencil at the bottom, quiet chalk at the top.
export const CONFIDENCE_STOPS = ['#be3a21', '#b5713a', '#8c9a84', '#dfe4da'];
export const CONFIDENCE_RAMP = ramp(CONFIDENCE_STOPS, 16);

// Cell size: finer cells read brighter, so foveation is legible at a glance.
export const LEVEL_STOPS = ['#24485e', '#3c6480', '#6d94b0', '#9dbdd4', '#d5e6f1'];
export const LEVEL_RAMP = ramp(LEVEL_STOPS, 8);

export const PLOT_BG = '#0e1311';
export const RING = '#4a7fa5';
export const EGO = '#eef1e8';
export const RED_PENCIL = '#be3a21';
export const CELL_EDGE = 'rgba(232, 238, 232, 0.16)';
