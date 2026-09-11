<!--
  Turning the flowsheet into delta vectors for an LP planner.

  A planning model wants a Jacobian around a base case, plus the bounds
  and the trust radius that say where it is valid. difflow can compute
  that; what it never had was a way to say *which* levers, by pointing at
  them. This panel is that way.

  Two things it deliberately does not do. It does not run a planner, so
  there are no prices, no shadow prices and no `.lp`/`.mps` -- those are
  renderings of an LP that does not exist until someone poses one, and
  `difflow plan-export` is where that lives. And it does not check the
  Jacobian against finite differences unless asked: the check costs 2n
  extra solves, and a button that says so is more honest than a wait
  nobody chose.
-->
<script>
  import { untrack } from 'svelte'

  import { post } from './api.js'
  import { offer } from './export.js'
  import {
    DEFAULT_RADIUS, FORMATS, blocked, cell, cleanBounds, jacobian,
    leverGroups, outputGroups, ranked, toggle, verdict,
  } from './model/planning.js'
  import { exportName } from './model/download.js'

  let { path = '', document: doc = null, levers = null, solved = false,
        onclose = () => {} } = $props()

  let u = $state([])
  let y = $state([])
  let bounds = $state({})
  let radius = $state(DEFAULT_RADIUS)
  let check = $state(false)
  let running = $state(false)
  let saving = $state('')
  let error = $state('')
  let result = $state(null)

  // The last selection travels in the document, so reopening a saved
  // flowsheet reopens the panel on the same levers. Read once, on the
  // way in: after that the panel owns the selection, and a reload that
  // brought the old one back would undo whatever the user just picked.
  let loaded = false
  $effect(() => {
    const saved = doc?.view?.planning
    if (loaded || !saved) return
    loaded = true
    untrack(() => {
      u = saved.u ?? []
      y = saved.y ?? []
      radius = saved.radius ?? DEFAULT_RADIUS
      bounds = Object.fromEntries(
        Object.entries(saved.bounds ?? {}).map(([k, w]) => [
          k, { lb: w.lb ?? '', ub: w.ub ?? '' }]))
    })
  })

  let leverList = $derived(leverGroups(levers?.levers))
  let outputList = $derived(outputGroups(levers?.outputs))
  let why = $derived(blocked({ solved, u, y, bounds }))
  let table = $derived(jacobian(result?.delta_vectors?.vectors?.[0]))
  let findings = $derived(ranked(result?.health))

  /** Bounds are edited as strings so a half-typed number is not a zero. */
  function bound(key, side, value) {
    bounds = { ...bounds, [key]: { ...(bounds[key] ?? { lb: '', ub: '' }), [side]: value } }
  }

  async function linearize() {
    if (why || running) return
    running = true
    error = ''
    try {
      const answer = await post('/api/linearize', {
        u, y, bounds: cleanBounds(bounds, u), radius: Number(radius), check,
      })
      if (!answer.ok) { error = answer.error; result = null }
      else result = answer
    } catch (e) {
      error = String(e.message ?? e)
      result = null
    } finally {
      running = false
    }
  }

  /**
   * Download, rendered by the server's own writers.
   *
   * Not built here from `result.delta_vectors`: the bytes a planning
   * system reads have to be the bytes `difflow plan-export` writes, and
   * the only way to guarantee that is to let the same code write them.
   */
  async function download(fmt) {
    saving = fmt
    error = ''
    try {
      const answer = await post('/api/linearize/files', { format: fmt })
      if (!answer.ok) throw new Error(answer.error)
      const type = fmt === 'json' ? 'application/json' : 'text/csv'
      for (const file of answer.files) offer(new Blob([file.text], { type }), file.name)
    } catch (e) {
      error = `download failed: ${e.message ?? e}`
    } finally {
      saving = ''
    }
  }
</script>

<section class="planning">
  <header>
    <h2>Planning</h2>
    <span class="hint">
      Delta vectors: d(output)/d(lever) around the solved base case.
    </span>
    <span class="spacer"></span>
    <label class="toggle" title="AD against central differences: 2n extra solves">
      <input type="checkbox" bind:checked={check} />
      verify
    </label>
    <label class="radius">
      radius
      <input type="number" min="0.01" max="1" step="0.05" bind:value={radius} />
    </label>
    <button class="primary" onclick={linearize} disabled={!!why || running}>
      {running ? 'Linearizing...' : 'Linearize'}
    </button>
    <button onclick={onclose}>Close</button>
  </header>

  {#if why}
    <p class="why">{why}</p>
  {/if}
  {#if error}
    <p class="error">{error}</p>
  {/if}

  <div class="body">
    <div class="picker">
      <h3>Levers <small>{u.length} picked</small></h3>
      {#each leverList as group (group.name)}
        <p class="group">{group.name}</p>
        {#each group.items as item (item.key)}
          <div class="row" class:on={u.includes(item.key)}>
            <label>
              <input type="checkbox" checked={u.includes(item.key)}
                     onchange={() => (u = toggle(u, item.key))} />
              {item.field}
              {#if item.units}<small>{item.units}</small>{/if}
            </label>
            {#if u.includes(item.key)}
              <input class="num" type="text" placeholder="lb"
                     value={bounds[item.key]?.lb ?? ''}
                     oninput={(e) => bound(item.key, 'lb', e.currentTarget.value)} />
              <input class="num" type="text" placeholder="ub"
                     value={bounds[item.key]?.ub ?? ''}
                     oninput={(e) => bound(item.key, 'ub', e.currentTarget.value)} />
            {/if}
          </div>
        {/each}
      {:else}
        <p class="hint">No levers. Load a flowsheet.</p>
      {/each}
      <p class="note">
        A lever left blank gets a window around its own value, not an
        infinity -- an unbounded lever makes the trust region meaningless.
      </p>
    </div>

    <div class="picker">
      <h3>Outputs <small>{y.length} picked</small></h3>
      {#each outputList as group (group.name)}
        <p class="group">{group.name}</p>
        {#each group.items as item (item.key)}
          <div class="row" class:on={y.includes(item.key)}>
            <label>
              <input type="checkbox" checked={y.includes(item.key)}
                     onchange={() => (y = toggle(y, item.key))} />
              {item.quantity}
              {#if item.units}<small>{item.units}</small>{/if}
            </label>
          </div>
        {/each}
      {:else}
        <p class="hint">Solve first -- outputs are named from the streams.</p>
      {/each}
    </div>

    <div class="result">
      {#if !result}
        <p class="hint">
          Pick levers and outputs, then Linearize. The Jacobian, its health
          and the downloads appear here.
        </p>
      {:else}
        {#if result.check}
          <p class="verdict" class:bad={!result.check.passed}>
            {verdict(result.check)}
          </p>
        {/if}

        <div class="scroll">
          <table>
            <thead>
              <tr>
                <th class="name">output</th>
                <th class="base">base</th>
                {#each table.levers as lever (lever.name)}
                  <th title="{lever.name} = {lever.u0} in [{lever.lb}, {lever.ub}]">
                    {lever.name}{#if lever.units}<small> {lever.units}</small>{/if}
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each table.rows as row (row.name)}
                <tr>
                  <td class="name">
                    {row.name}{#if row.units}<small> {row.units}</small>{/if}
                  </td>
                  <td class="base">{cell(row.y0)}</td>
                  {#each row.cells as value, j (j)}
                    <td class:zero={cell(value) === '0'}>{cell(value)}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>

        <p class="meta">
          {table.mode} mode, valid within {table.radius} of each lever's range.
        </p>

        {#each findings as finding (finding.variable + finding.kind)}
          <p class="finding" class:err={finding.severity === 'error'}>
            <b>{finding.kind}</b>
            {#if finding.variable}<code>{finding.variable}</code>{/if}
            {finding.detail}
          </p>
        {/each}

        <div class="downloads">
          {#each FORMATS as f (f.id)}
            <button onclick={() => download(f.id)} disabled={!!saving}>
              {saving === f.id ? 'Saving...' : f.label}
              <small>{f.note}</small>
            </button>
          {/each}
          <span class="hint">
            as <code>{exportName(path, 'json')}</code> and friends. For
            <code>.lp</code>/<code>.mps</code> run a planner --
            <code>difflow plan-export</code> -- since those are renderings
            of an LP, and this panel poses none.
          </span>
        </div>
      {/if}
    </div>
  </div>
</section>

<style>
  .planning {
    display: flex;
    flex-direction: column;
    height: 26rem;
    flex: none;
    border-top: 1px solid var(--grid);
    background: var(--panel);
  }
  header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px dashed var(--grid);
  }
  h2 { font-size: 0.85rem; margin: 0; }
  h3 { font-size: 0.78rem; margin: 0 0 0.4rem; }
  h3 small, .row small { color: var(--ink-soft); font-weight: 400; }
  .spacer { flex: 1; }
  .hint, .note { color: var(--ink-soft); font-size: 0.76rem; margin: 0; }
  .note { margin-top: 0.5rem; line-height: 1.5; }
  .why { color: var(--ink-soft); font-size: 0.78rem; margin: 0; padding: 0.4rem 0.8rem; }
  .error { color: var(--bad); font-size: 0.78rem; margin: 0; padding: 0.4rem 0.8rem; }
  .toggle, .radius { font-size: 0.76rem; color: var(--ink-soft); }
  .radius input { width: 4rem; }
  input[type="text"], input[type="number"] {
    font: inherit;
    font-size: 0.76rem;
    padding: 0.15rem 0.3rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface);
    color: inherit;
  }

  .body { display: flex; flex: 1; min-height: 0; gap: 0.8rem; padding: 0.5rem 0.8rem; }
  .picker {
    width: 15rem;
    flex: none;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
    padding: 0.5rem 0.6rem;
  }
  .result {
    flex: 1;
    min-width: 0;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
    padding: 0.5rem 0.6rem;
  }
  .group {
    margin: 0.5rem 0 0.15rem;
    font-size: 0.72rem;
    color: var(--ink-soft);
    text-transform: lowercase;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.1rem 0;
    font-size: 0.78rem;
  }
  .row label { display: flex; align-items: center; gap: 0.3rem; flex: 1; min-width: 0; }
  .row.on { color: var(--accent); }
  .num { width: 3.4rem; }

  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; font-size: 0.76rem; }
  th, td {
    padding: 0.2rem 0.5rem;
    text-align: right;
    border-bottom: 1px solid var(--grid);
    font-variant-numeric: tabular-nums;
  }
  th { color: var(--ink-soft); font-weight: 500; white-space: nowrap; }
  .name { text-align: left; }
  .base { color: var(--ink-soft); }
  /* A structurally zero column is the finding a reader must not miss. */
  .zero { color: var(--ink-soft); }
  .meta { color: var(--ink-soft); font-size: 0.74rem; margin: 0.4rem 0 0; }
  .verdict { font-size: 0.78rem; margin: 0 0 0.5rem; color: var(--ink-soft); }
  .verdict.bad { color: var(--bad); }
  .finding {
    font-size: 0.74rem;
    line-height: 1.5;
    margin: 0.5rem 0 0;
    color: var(--ink-soft);
  }
  .finding.err { color: var(--bad); }
  .finding code { font-size: 0.72rem; }

  .downloads {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
  }
  .downloads button { display: flex; align-items: baseline; gap: 0.4rem; }
  .downloads small { color: var(--ink-soft); font-size: 0.7rem; }
</style>
