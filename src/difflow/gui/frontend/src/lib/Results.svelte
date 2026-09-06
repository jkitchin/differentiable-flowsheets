<!--
  What the solve said, and what the derivatives say.

  Three tabs, because they answer three different questions. The stream
  table is the flowsheet's state. The solve tab is whether to believe it
  -- a recycle that stopped at max_iter returns numbers that look exactly
  like an answer, and `converged` is the only thing that says otherwise.
  The sensitivity tab is the one no other flowsheet editor has: the model
  is differentiable, so "what happens if I change this" is one pass, not
  a parameter sweep.
-->
<script>
  import { fmt, streamTable, tornado } from './model/results.js'

  let {
    solve = null,
    levers = null,
    sensitivity = null,
    busy = false,
    onsensitivity = () => {},
    onclose = () => {},
  } = $props()

  let tab = $state('streams')
  let showFractions = $state(false)
  let lever = $state('')
  let target = $state('')

  let table = $derived(streamTable(solve))
  let chart = $derived(
    sensitivity?.mode === 'reverse' ? tornado(sensitivity.levers) : null,
  )

  // Forward mode answers for every quantity of every stream. Ranked by
  // relative response and cut at the ones that actually moved, since a
  // flowsheet of twenty streams otherwise buries the three that did.
  let responses = $derived.by(() => {
    if (sensitivity?.mode !== 'forward') return []
    const rows = []
    for (const [name, row] of Object.entries(sensitivity.streams)) {
      for (const [quantity, cell] of Object.entries(row)) {
        if (!cell.d) continue
        rows.push({ name, quantity, ...cell })
      }
    }
    rows.sort((a, b) => Math.abs(b.rel ?? 0) - Math.abs(a.rel ?? 0))
    return rows
  })

  const run = () => {
    if (lever) onsensitivity({ lever })
    else if (target) onsensitivity({ target })
  }

  /** One dropdown at a time: the two directions are two questions. */
  function pickLever(value) {
    lever = value
    if (value) target = ''
  }
  function pickTarget(value) {
    target = value
    if (value) lever = ''
  }
</script>

<section class="results">
  <header>
    <nav>
      <button class:on={tab === 'streams'}
              onclick={() => (tab = 'streams')}>Streams</button>
      <button class:on={tab === 'solve'}
              onclick={() => (tab = 'solve')}>Solve</button>
      <button class:on={tab === 'sensitivity'}
              onclick={() => (tab = 'sensitivity')}>Sensitivity</button>
    </nav>
    <span class="spacer"></span>
    {#if tab === 'streams' && table.rows.length}
      <label class="toggle">
        <input type="checkbox" bind:checked={showFractions} />
        mole fractions
      </label>
    {/if}
    <button onclick={onclose}>Close</button>
  </header>

  <div class="body">
    {#if !solve}
      <p class="hint">Nothing solved yet. Press Solve.</p>

    {:else if tab === 'streams'}
      <table>
        <thead>
          <tr>
            <th class="name">stream</th>
            <th>T (K)</th>
            <th>P (Pa)</th>
            <th>F (mol/s)</th>
            {#each table.species as s (s)}
              <th>{showFractions ? `x_${s}` : `F_${s}`}</th>
            {/each}
            <th class="name">phase</th>
          </tr>
        </thead>
        <tbody>
          {#each table.rows as row (row.name)}
            <tr>
              <td class="name">{row.name}</td>
              <td>{fmt(row.T)}</td>
              <td>{fmt(row.P)}</td>
              <td>{fmt(row.total)}</td>
              {#each table.species as s, i (s)}
                <td>{fmt(showFractions ? row.fractions[i] : row.flows[i])}</td>
              {/each}
              <td class="name soft">{row.phase ?? ''}</td>
            </tr>
          {/each}
        </tbody>
      </table>

    {:else if tab === 'solve'}
      <dl class="diagnostics">
        <dt>converged</dt>
        <dd class:bad={solve.converged === false}>
          {solve.converged === null ? 'not reported' : String(solve.converged)}
        </dd>
        <dt>method</dt><dd>{solve.method ?? '—'}</dd>
        <dt>iterations</dt><dd>{solve.iterations ?? '—'}</dd>
        <dt>residual</dt>
        <dd class:bad={solve.residual !== null && solve.tol !== null
                       && solve.residual > solve.tol}>
          {fmt(solve.residual)}
        </dd>
        <dt>tolerance</dt><dd>{fmt(solve.tol)}</dd>
        <dt>tear streams</dt>
        <dd>{solve.tear_streams?.length
             ? solve.tear_streams.join(', ')
             : 'none — solved in sequence'}</dd>
      </dl>
      {#if solve.converged === false}
        <p class="warn">
          The tear residual never reached the tolerance. The streams above
          are the last iterate, not a solution.
        </p>
      {/if}

    {:else}
      <div class="ask">
        <label>
          <span>one lever, every stream</span>
          <select value={lever} disabled={busy}
                  onchange={(e) => pickLever(e.currentTarget.value)}>
            <option value="">—</option>
            {#each levers?.levers ?? [] as l (l.key)}
              <option value={l.key}>
                {l.key}{l.units ? ` (${l.units})` : ''} = {fmt(l.value)}
              </option>
            {/each}
          </select>
        </label>
        <label>
          <span>one output, every lever</span>
          <select value={target} disabled={busy || !levers?.solved}
                  onchange={(e) => pickTarget(e.currentTarget.value)}>
            <option value="">—</option>
            {#each levers?.outputs ?? [] as o (o.key)}
              <option value={o.key}>{o.key} = {fmt(o.value)}</option>
            {/each}
          </select>
        </label>
        <button class="primary" disabled={busy || (!lever && !target)}
                onclick={run}>Differentiate</button>
      </div>

      {#if sensitivity?.mode === 'forward'}
        <p class="hint">
          One forward pass through the solve, including the recycle tear:
          d(everything)/d(<code>{sensitivity.lever}</code>) at
          {fmt(sensitivity.u0)}. Edges on the canvas are tinted by it.
        </p>
        {#if responses.length}
          <table>
            <thead>
              <tr><th class="name">stream</th><th class="name">quantity</th>
                  <th>value</th><th>d/du</th><th>d ln y / d ln u</th></tr>
            </thead>
            <tbody>
              {#each responses as r (r.name + r.quantity)}
                <tr>
                  <td class="name">{r.name}</td>
                  <td class="name soft">{r.quantity}</td>
                  <td>{fmt(r.value)}</td>
                  <td>{fmt(r.d)}</td>
                  <td>{r.rel === null ? '—' : fmt(r.rel)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        {:else}
          <p class="hint">Nothing in the flowsheet responds to this lever.</p>
        {/if}

      {:else if sensitivity?.mode === 'reverse'}
        <p class="hint">
          One reverse pass: d(<code>{sensitivity.target}</code>)/d(every
          lever) at {fmt(sensitivity.y0)}. Bars are dimensionless
          (d ln y / d ln u), so levers in different units compare.
        </p>
        {#if chart.bars.length}
          <div class="tornado">
            <svg width={chart.width} height={chart.height}
                 role="img" aria-label="lever sensitivities">
              <line x1={chart.mid} x2={chart.mid} y1="0" y2={chart.height}
                    stroke="var(--line)" />
              {#each chart.bars as b (b.key)}
                <rect x={b.x} y={b.y} width={b.width} height={b.height}
                      fill={b.negative ? 'var(--accent)' : 'var(--series)'} />
              {/each}
            </svg>
            <ul>
              {#each chart.bars as b (b.key)}
                <li style="height: {b.height}px">
                  <code>{b.key}</code>
                  <span class="soft">{fmt(b.rel)}</span>
                </li>
              {/each}
            </ul>
          </div>
          {#if chart.unscaled.length}
            <p class="hint">
              No relative sensitivity for {chart.unscaled.join(', ')} — the
              base value or the output is zero.
            </p>
          {/if}
        {:else}
          <p class="hint">No lever moves this output.</p>
        {/if}
      {/if}
    {/if}
  </div>
</section>

<style>
  .results {
    display: flex;
    flex-direction: column;
    height: 17rem;
    flex: none;
    border-top: 1px solid var(--grid);
    background: var(--panel);
  }
  header { display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0.8rem; }
  nav { display: flex; gap: 0.3rem; }
  nav button { border-color: transparent; background: transparent; }
  nav button.on { background: var(--surface); border-color: var(--line); }
  .spacer { flex: 1; }
  .toggle { font-size: 0.78rem; color: var(--ink-soft); }
  .body { flex: 1; min-height: 0; overflow: auto; padding: 0 0.8rem 0.8rem; }
  .hint { color: var(--ink-soft); font-size: 0.78rem; }
  .warn { color: var(--bad); font-size: 0.78rem; }

  table { border-collapse: collapse; font-size: 0.78rem; }
  th, td {
    padding: 0.2rem 0.55rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid var(--grid);
    white-space: nowrap;
  }
  th { color: var(--ink-soft); font-weight: 600; position: sticky; top: 0;
       background: var(--panel); }
  .name { text-align: left; font-family: ui-monospace, SFMono-Regular, Menlo,
          monospace; }
  .soft { color: var(--ink-soft); }

  dl.diagnostics {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.15rem 0.9rem;
    font-size: 0.8rem;
    margin: 0.4rem 0;
    max-width: 34rem;
  }
  dt { color: var(--ink-soft); }
  dd { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  dd.bad { color: var(--bad); }

  .ask { display: flex; align-items: flex-end; gap: 0.8rem; margin: 0.5rem 0; }
  .ask label { display: flex; flex-direction: column; gap: 0.15rem;
               font-size: 0.75rem; color: var(--ink-soft); }
  select { font: inherit; font-size: 0.78rem; max-width: 22rem;
           padding: 0.25rem; border: 1px solid var(--line);
           border-radius: 5px; background: var(--surface); color: var(--ink); }

  .tornado { display: flex; gap: 0.6rem; align-items: flex-start; }
  .tornado ul { list-style: none; margin: 0; padding: 0; font-size: 0.75rem; }
  .tornado li { display: flex; gap: 0.5rem; align-items: center; margin-bottom: 4px; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         font-size: 0.95em; }
</style>
