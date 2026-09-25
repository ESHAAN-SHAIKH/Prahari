/* The evidence view.

   One rule governs this whole page: a measured number and an unmeasured one must never
   look alike. Anything the notebook has not produced renders as an explicit gap with the
   command that would fill it — never as a plausible-looking placeholder. */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function chip(measured) {
  return measured
    ? '<span class="chip is-measured">Measured</span>'
    : '<span class="chip is-absent">Not measured</span>';
}

function absentBlock(block) {
  return `<div class="absent">
    <p>${esc(block.what)} has not been produced yet.</p>
    <p>Run: <span class="cmd">${esc(block.how)}</span></p>
    <p>The server reads <span class="cmd">${esc(block.evidence_path)}</span>; nothing is
       shown here until that file contains the number.</p>
  </div>`;
}

function section(id, block, renderer) {
  const el = $(id);
  const h2 = el.querySelector('h2');
  h2.querySelector('.chip')?.remove();
  h2.insertAdjacentHTML('beforeend', chip(block.status === 'measured'));
  el.querySelector('.ev-body').innerHTML =
    block.status === 'measured' ? renderer(block) : absentBlock(block);
}

function caveatList(caveats) {
  if (!caveats?.length) return '';
  return `<div class="caveats"><p>Carried on every number above:</p><ul>${
    caveats.map(c => `<li>${esc(c)}</li>`).join('')}</ul></div>`;
}

export function renderEvidence(data, live) {
  // ---------------------------------------------------------------- header ---
  $('ev-meta').innerHTML = data.available
    ? `Export generated <span class="num">${esc(data.generated || 'unknown')}</span> ·
       board <span class="num">${esc(data.board)}</span> ·
       read from <span class="cmd">${esc(data.evidence_path)}</span>`
    : `No evidence export found at <span class="cmd">${esc(data.evidence_path)}</span>.
       Run the training notebook through Phase 7 and copy
       <span class="cmd">trinetra_evidence_export.json</span> there.`;

  // ------------------------------------------------------------- per class ---
  section('ev-perclass', data.per_class, (b) => b.runs.map(run => `
    <h3 class="tel-note">${esc(run.label)}${run.aggregation ? ` · aggregated by ${esc(run.aggregation)}` : ''}</h3>
    <table class="ev">
      <thead><tr><th>Category</th><th>Segmentation mIoU</th><th>Detection AP</th><th>Constituent classes</th></tr></thead>
      <tbody>${run.rows.map(r => `<tr>
        <td>${esc(r.ps_category)}</td>
        <td class="n">${r.seg_mIoU ?? '—'}</td>
        <td class="n">${r.det_AP ?? '—'}</td>
        <td class="muted">${esc(Object.entries({ ...(r.seg_constituents || {}), ...(r.det_constituents || {}) })
          .map(([k, v]) => `${k} ${v}`).join(', ') || '—')}</td>
      </tr>`).join('')}</tbody>
    </table>`).join('') + caveatList(b.caveats));

  // -------------------------------------------------------------- distance ---
  section('ev-distance', data.distance, (b) => b.runs.map(run => {
    const bands = Object.entries(run.summary || {});
    const max = Math.max(...bands.map(([, v]) => v.mIoU_dataset || 0), 1);
    return `
      <h3 class="tel-note">${esc(run.label)}${run.n_scans ? ` · ${run.n_scans} scans` : ''}</h3>
      <div class="bars">${bands.map(([band, v]) => `
        <div class="bar-row is-ours">
          <div class="bar-head"><span>${esc(band)}</span><span class="num">${
            v.mIoU_dataset != null ? v.mIoU_dataset.toFixed(2) + '%' : '—'}</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${
            ((v.mIoU_dataset || 0) / max * 100).toFixed(1)}%"></div></div>
        </div>`).join('')}</div>
      ${run.command ? `<p class="tel-note">Produced by <span class="cmd">${esc(run.command)}</span></p>` : ''}`;
  }).join('') + caveatList(b.caveats));

  // ------------------------------------------------------------------ edge ---
  section('ev-edge', data.edge, (b) => `
    <table class="ev">
      <thead><tr><th>Model</th><th>Latency</th><th>Rate</th><th>Peak memory</th><th>Hardware</th></tr></thead>
      <tbody>${b.models.map(m => `<tr>
        <td>${esc(m.model)}</td>
        <td class="n">${m.latency_ms != null ? m.latency_ms + ' ms' : '—'}</td>
        <td class="n">${m.fps != null ? m.fps + ' Hz' : '—'}</td>
        <td class="n">${m.peak_mem_mb != null ? m.peak_mem_mb + ' MB' : '—'}</td>
        <td class="muted">${esc(m.hardware || '—')}</td>
      </tr>`).join('')}</tbody>
    </table>
    <p class="tel-note">Measured on the board with trtexec and tegrastats, not estimated.</p>`);

  // ------------------------------------------------------------- baselines ---
  section('ev-baselines', data.baselines, (b) => `
    <table class="ev">
      <thead><tr><th>Baseline</th><th>Axis</th><th>Checkpoint or config</th></tr></thead>
      <tbody>${Object.entries(b.baselines).map(([k, v]) => `<tr>
        <td>${esc(v.name)}</td>
        <td class="muted">${esc(v.axis || '—')}</td>
        <td class="muted"><span class="cmd">${esc(v.checkpoint || v.config || '—')}</span></td>
      </tr>`).join('')}</tbody>
    </table>
    ${b.comparisons?.length ? `<p class="tel-note">${b.comparisons.map(c => esc(c.claim)).join('<br>')}</p>` : ''}`);

  // ------------------------------------------------------------- checklist ---
  const cl = data.checklist;
  section('ev-checklist',
    cl.status === 'measured' ? { status: 'measured', ...cl }
      : { status: 'not_measured', what: 'The notebook verification checklist',
          how: 'Run the training notebook — CHECK.show()', evidence_path: data.evidence_path },
    (b) => `
      <p class="tel-note"><span class="num">${b.done}</span> of
         <span class="num">${b.total}</span> checks complete.</p>
      <div class="checks">${Object.entries(b.phases).map(([p, ph]) =>
        ph.items.map(i => `<div class="check-item ${i.done ? 'is-done' : ''}">
          <span class="check-box">[${i.done ? '×' : ' '}]</span>
          <span>${esc(i.id)} ${esc(i.text)}</span></div>`).join('')).join('')}</div>`);

  // ------------------------------------------------ measured in this session --
  renderLive(live);
}

export function renderLive(live) {
  const el = $('ev-live');
  const h2 = el.querySelector('h2');
  h2.querySelector('.chip')?.remove();
  h2.insertAdjacentHTML('beforeend', chip(!!live));
  if (!live) {
    el.querySelector('.ev-body').innerHTML =
      '<div class="absent"><p>Open the live map to measure throughput and memory.</p></div>';
    return;
  }
  const { roll, st } = live;
  const m = st.memory || {};
  el.querySelector('.ev-body').innerHTML = `
    <table class="ev">
      <thead><tr><th>Quantity</th><th>Value</th><th>How</th></tr></thead>
      <tbody>
        <tr><td>End-to-end rate</td><td class="n">${roll.fps?.toFixed(2) ?? '—'} Hz</td>
            <td class="muted">Wall clock over the last ${roll.samples ?? 0} frames, this machine</td></tr>
        <tr><td>Frame time, mean</td><td class="n">${roll.frame_ms_mean?.toFixed(2) ?? '—'} ms</td>
            <td class="muted">perception + grid + encode</td></tr>
        <tr><td>Frame time, p95</td><td class="n">${roll.frame_ms_p95?.toFixed(2) ?? '—'} ms</td>
            <td class="muted">same window</td></tr>
        <tr><td>Cells per frame</td><td class="n">${(st.n_cells ?? 0).toLocaleString('en-US')}</td>
            <td class="muted">from ${(st.n_points ?? 0).toLocaleString('en-US')} points</td></tr>
        <tr><td>Map size</td><td class="n">${(m.adaptive_bytes / 1024).toFixed(1)} kB</td>
            <td class="muted">${m.bytes_per_cell} bytes per cell</td></tr>
        <tr><td>Against a uniform 5 cm grid</td><td class="n">${m.reduction_vs_dense}×</td>
            <td class="muted">${(m.dense_uniform_cells ?? 0).toLocaleString('en-US')} cells, same bytes each</td></tr>
        <tr><td>Against 5 cm 3D voxels</td><td class="n">${m.reduction_vs_dense_3d}×</td>
            <td class="muted">10 m vertical extent, 1 byte occupancy</td></tr>
      </tbody>
    </table>
    <p class="tel-note">These are this machine's numbers, not the Jetson's. Edge figures
       belong in the section above and come from the board.</p>`;
}
