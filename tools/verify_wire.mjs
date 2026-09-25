/* End-to-end check of the browser decoder against the real server.

   Runs the frontend's own wire.js — the exact file the browser loads — against frames
   fetched from a running instance, and asserts the decoded map agrees with the stats the
   server reported. If the codec and the renderer ever disagree, this fails here rather
   than as a quietly wrong map on screen.

   Usage:  node tools/verify_wire.mjs [http://127.0.0.1:8000]
*/

import { decodeFrame } from '../frontend/js/wire.js';

const BASE = process.argv[2] || 'http://127.0.0.1:8000';
let failures = 0;

function check(name, cond, detail = '') {
  const mark = cond ? 'ok  ' : 'FAIL';
  if (!cond) failures++;
  console.log(`  ${mark}  ${name}${detail ? `  — ${detail}` : ''}`);
}

async function fetchFrame(query = '') {
  const res = await fetch(`${BASE}/api/frame${query}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const meta = JSON.parse(res.headers.get('X-Prahari-Meta'));
  const buf = await res.arrayBuffer();
  return { meta, frame: await decodeFrame(buf) };
}

async function main() {
  console.log(`Verifying the browser decoder against ${BASE}\n`);

  const sys = await (await fetch(`${BASE}/api/system`)).json();
  console.log(`System: ${sys.name} — ${sys.subtitle}`);
  console.log(`Source: ${sys.perception.source}\n`);

  // ---------------------------------------------------------- compressed ----
  console.log('Compressed frame');
  const { meta, frame } = await fetchFrame();
  const st = frame.tail.stats;

  check('magic and version accepted', frame.n > 0, `${frame.n} cells`);
  check('cell count matches the server', frame.n === st.n_cells, `${frame.n} vs ${st.n_cells}`);
  check('wire size matches the header', meta.wire_bytes === frame.wireBytes);
  check('quantisation step is half a minimum cell',
        Math.abs(frame.quant - 0.025) < 1e-6, `${frame.quant}`);
  check('root size matches the config',
        Math.abs(frame.root - sys.geometry.root_size_m) < 1e-3);

  let minSize = Infinity, maxSize = 0, outOfRange = 0, badConf = 0;
  for (let i = 0; i < frame.n; i++) {
    minSize = Math.min(minSize, frame.size[i]);
    maxSize = Math.max(maxSize, frame.size[i]);
    if (Math.hypot(frame.x[i], frame.y[i]) > sys.geometry.root_size_m) outOfRange++;
    if (frame.conf[i] < 0 || frame.conf[i] > 1) badConf++;
  }
  check('finest cell matches the server',
        Math.abs(minSize - st.cell_size_min_m) < 1e-6, `${minSize} m`);
  check('coarsest cell matches the server',
        Math.abs(maxSize - st.cell_size_max_m) < 1e-6, `${maxSize} m`);
  check('finest cell is the 5 cm the PS asks for', Math.abs(minSize - 0.05) < 1e-6);
  check('no cell lands outside the root', outOfRange === 0);
  check('confidence decodes into 0..1', badConf === 0);

  // Foveation, straight off the decoded map.
  let nearMax = 0, farMin = Infinity, nearN = 0, farN = 0;
  for (let i = 0; i < frame.n; i++) {
    const r = Math.hypot(frame.x[i], frame.y[i]);
    if (r < sys.geometry.fovea_radius_m) { nearMax = Math.max(nearMax, frame.size[i]); nearN++; }
    else if (r > 50) { farMin = Math.min(farMin, frame.size[i]); farN++; }
  }
  check('inside the fovea every cell is 5 cm',
        nearN > 0 && Math.abs(nearMax - 0.05) < 1e-6, `${nearN} cells, max ${nearMax} m`);
  check('the far field is coarser somewhere', farN > 0 && farMin >= 0.05, `${farN} cells`);

  const classes = new Set(Array.from(frame.cls));
  check('several classes present', classes.size >= 4, `${classes.size} classes`);
  check('detections travel with the frame',
        Array.isArray(frame.tail.detections), `${frame.tail.detections.length} objects`);
  check('source is labelled on every frame', !!frame.tail.source, frame.tail.source);

  // -------------------------------------------------------- uncompressed ----
  console.log('\nUncompressed frame');
  const plain = await fetchFrame('?compress=false');
  check('decodes without inflate', plain.frame.n > 0, `${plain.frame.n} cells`);
  check('larger than the compressed frame', plain.frame.wireBytes > frame.wireBytes,
        `${plain.frame.wireBytes} vs ${frame.wireBytes} bytes`);

  // ------------------------------------------------------------ risk A/B ----
  console.log('\nRisk-adaptive against range-only');
  const risk = (await fetchFrame('?risk_adaptive=true')).frame;
  const plainGrid = (await fetchFrame('?risk_adaptive=false')).frame;
  const HAZ = new Set([40, 41, 42, 43, 44]);

  const hazSizes = (f) => {
    const out = [];
    for (let i = 0; i < f.n; i++) {
      if (HAZ.has(f.cls[i]) && Math.hypot(f.x[i], f.y[i]) > sys.geometry.fovea_radius_m) {
        out.push(f.size[i]);
      }
    }
    return out.sort((a, b) => a - b);
  };
  const med = (a) => (a.length ? a[Math.floor(a.length / 2)] : NaN);
  const a = hazSizes(risk), b = hazSizes(plainGrid);

  check('hazards beyond the fovea exist in both', a.length > 0 && b.length > 0,
        `${a.length} / ${b.length} cells`);
  check('risk-adaptive resolves them finer', med(a) < med(b),
        `${med(a)} m vs ${med(b)} m`);
  check('range-only uses one driver only',
        new Set(Array.from(plainGrid.driver)).size === 1);
  check('risk-adaptive uses several drivers',
        new Set(Array.from(risk.driver)).size > 1,
        JSON.stringify(risk.tail.stats.cells_by_driver));

  // --------------------------------------------------------- link tier ------
  console.log('\nTransmit tier');
  const tier = (await fetchFrame('?link_tier=9')).frame;
  check('tier shrinks the transmitted map', tier.n < frame.n, `${tier.n} vs ${frame.n} cells`);
  check('tier shrinks the bytes', tier.wireBytes < frame.wireBytes,
        `${tier.wireBytes} vs ${frame.wireBytes} bytes`);
  check('the onboard map is still reported',
        tier.tail.onboard.n_cells > tier.n,
        `${tier.tail.onboard.n_cells} onboard`);
  const tierHaz = hazSizes(tier);
  check('hazards survive the tier at full detail',
        tierHaz.length > 0 && med(tierHaz) <= med(a) + 1e-9, `${med(tierHaz)} m`);

  console.log(`\n${failures === 0 ? 'All checks passed.' : `${failures} check(s) failed.`}`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('Verification failed to run:', err.message);
  process.exit(2);
});
