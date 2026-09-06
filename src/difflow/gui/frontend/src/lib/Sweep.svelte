<!--
  The sliders, the chart and the readout of a precomputed sweep.

  It draws; `model/sweep.js` computes. That split is what lets the
  arithmetic a published page depends on -- interpolation between solved
  operating points, and the exact derivatives recorded alongside them --
  be tested under bare node instead of living as a string of JavaScript
  inside a Python template, which is where it used to live.
-->
<script>
  import { untrack } from 'svelte'

  import { chart, coordsAt, fmt, middle, readout } from './model/sweep.js'

  let { sweep, version = '', points = 0 } = $props()

  // `$derived`, not `const`: `sweep` is a prop, and reading a prop's
  // field once at setup captures it rather than tracking it. The data on
  // a published page never changes, so nothing would break -- but the
  // component would be wrong for any other caller, and the compiler is
  // right to say so.
  let axes = $derived(sweep.axes)
  let outputs = $derived(sweep.outputs)

  // Where the sliders start. Read once on purpose -- after that the
  // reader owns them -- and `untrack` says so, since a bare read of a
  // prop at setup is the mistake the compiler is looking for.
  let state = $state(untrack(() => middle(sweep.axes)))
  let pick = $state(0)

  let here = $derived(coordsAt(axes, state))
  let plot = $derived(chart(axes, outputs[pick], here))
  let rows = $derived(readout(axes, outputs, here))
</script>

<div class="layout">
  <div class="panel" id="controls">
    <h2>Parameters</h2>
    <div id="sliders">
      {#each axes as axis, i}
        <div class="control">
          <label for="ax{i}">
            <span>{axis.label}</span>
            <span class="value">{fmt(here[i])}{axis.units ? ` ${axis.units}` : ''}</span>
          </label>
          <input type="range" id="ax{i}" min="0" max={axis.n - 1} step="1"
                 value={state[i]}
                 oninput={(e) => (state[i] = +e.currentTarget.value)} />
        </div>
      {/each}
    </div>
  </div>

  <div>
    <figure class="panel">
      <h2>Response</h2>
      <div class="control">
        <label for="output-pick"><span>Plotted quantity</span></label>
        <select id="output-pick" bind:value={pick}>
          {#each outputs as out, i}
            <option value={i}>{out.name}{out.units ? ` (${out.units})` : ''}</option>
          {/each}
        </select>
      </div>
      <svg id="chart" viewBox="0 0 {plot.W} {plot.H}" width="100%"
           role="img" aria-label="Model response">
        {#each plot.yTicks as tick}
          <line x1={plot.L} y1={tick.y} x2={plot.W - plot.R} y2={tick.y}
                stroke="var(--grid)" stroke-width="1" />
          <text x={plot.L - 8} y={tick.y + 4} text-anchor="end"
                font-size="11" fill="var(--ink-soft)">{tick.label}</text>
        {/each}
        {#each plot.xTicks as tick}
          <text x={tick.x} y={plot.H - plot.B + 20} text-anchor="middle"
                font-size="11" fill="var(--ink-soft)">{tick.label}</text>
        {/each}
        <text x={(plot.L + plot.W - plot.R) / 2} y={plot.H - 8}
              text-anchor="middle" font-size="12" fill="var(--ink-soft)"
        >{plot.xLabel}</text>
        <path d={plot.path} fill="none" stroke="var(--series)"
              stroke-width="2" stroke-linejoin="round" />
        <!-- Where the sliders currently sit. -->
        <line x1={plot.cursor.x} y1={plot.T} x2={plot.cursor.x}
              y2={plot.H - plot.B} stroke="var(--line)" stroke-width="1"
              stroke-dasharray="4 3" />
        <circle cx={plot.cursor.x} cy={plot.cursor.y} r="5"
                fill="var(--series)" stroke="var(--surface)" stroke-width="2" />
      </svg>
    </figure>

    <div class="panel">
      <h2>Values at this point</h2>
      <table>
        <thead>
          <tr><th>Quantity</th><th class="r">Value</th><th class="r">Sensitivity</th></tr>
        </thead>
        <tbody id="readout">
          {#each rows as row}
            <tr>
              <td>{row.name}</td>
              <td class="num">{fmt(row.value)}
                {#if row.units}<span class="units">{row.units}</span>{/if}
              </td>
              <td class="sens">{row.sensitivity === null ? '--' : fmt(row.sensitivity)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>
</div>

<footer>
  Precomputed with difflow {version} over {points} solved operating points;
  values between grid points are interpolated. Sensitivities are exact
  derivatives from automatic differentiation, reported per unit of
  {axes[0].label}.
</footer>

<style>
  .layout {
    display: grid; grid-template-columns: 17rem 1fr; gap: 1.5rem;
    align-items: start;
  }
  @media (max-width: 46rem) { .layout { grid-template-columns: 1fr; } }
  .panel {
    background: var(--panel); border: 1px solid var(--grid);
    border-radius: 10px; padding: 1rem 1.1rem;
  }
  figure.panel { margin: 0 0 1.5rem; }
  h2 {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--ink-soft); margin: 0 0 0.9rem; font-weight: 600;
  }
  .control { margin-bottom: 1.1rem; }
  .control:last-child { margin-bottom: 0; }
  .control label {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 0.85rem; margin-bottom: 0.35rem; gap: 0.5rem;
  }
  .value { font-variant-numeric: tabular-nums; color: var(--ink); font-weight: 600; }
  input[type='range'] { width: 100%; accent-color: var(--series); }
  select {
    width: 100%; padding: 0.35rem 0.5rem; border: 1px solid var(--line);
    border-radius: 6px; background: var(--surface); color: var(--ink);
    font: inherit; font-size: 0.85rem;
  }
  table { width: 100%; border-collapse: collapse; margin-top: 0.25rem; }
  th, td {
    text-align: left; padding: 0.4rem 0.3rem; font-size: 0.85rem;
    border-bottom: 1px solid var(--grid);
  }
  th {
    color: var(--ink-soft); font-weight: 600; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  th.r { text-align: right; }
  td.num, td.sens { text-align: right; font-variant-numeric: tabular-nums; }
  td.sens { color: var(--ink-soft); }
  .units { color: var(--ink-soft); }
  footer {
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--grid);
    color: var(--ink-soft); font-size: 0.8rem;
  }
</style>
