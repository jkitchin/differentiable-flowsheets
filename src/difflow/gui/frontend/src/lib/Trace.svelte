<!--
  A recorded solve, to scrub through.

  "Record" solves the flowsheet once under the debugger with nobody at
  the prompt, keeping every checkpoint (`difflow.debug.record`); the
  slider then walks it: unit by unit, iteration by iteration, or straight
  to the events (the first NaN, a unit that raised, the residual going
  up). The unit the solve is at is outlined on the canvas, and the tables
  say what that step changed. Nothing here re-solves: the trace is the
  record of one solve, read on this side of the wire.

  The live debugger (`difflow debug`) is the same checkpoints with a
  person at each one; this is its first form in the editor.
-->
<script>
  import { post } from './api.js'
  import {
    changesAt, describeFrame, eventFrames, formatNumber, iterationAt,
    residualPath, residualSeries, stateAt, STEPS, stepFrom, unitAt,
  } from './model/trace.js'

  let { onfocus = () => {}, onclose = () => {} } = $props()

  let trace = $state(null)
  let at = $state(0)
  let step = $state('unit')
  let busy = $state(false)
  let error = $state('')
  let warnings = $state([])
  let playing = $state(null)
  let showAll = $state(false)

  const W = 320
  const H = 90

  let frames = $derived(trace?.frames ?? [])
  let frame = $derived(frames[at] ?? null)
  let changes = $derived(trace ? changesAt(trace, at) : [])
  let state = $derived(trace && showAll ? stateAt(trace, at) : {})
  let series = $derived(trace ? residualSeries(trace) : [])
  let plot = $derived(residualPath(series, W, H))
  let events = $derived(trace ? eventFrames(trace) : [])
  let iteration = $derived(trace ? iterationAt(trace, at) : null)

  // The canvas follows the scrubber.
  $effect(() => onfocus(trace ? unitAt(trace, at) : null))
  // And lets go when the panel closes.
  $effect(() => () => onfocus(null))

  async function recordSolve() {
    stop()
    busy = true
    error = ''
    try {
      const answer = await post('/api/debug/record', {})
      if (!answer.ok) {
        error = answer.error ?? 'the solve could not be recorded'
        return
      }
      trace = answer.trace
      warnings = answer.warnings ?? []
      at = 0
    } catch (e) {
      error = String(e)
    } finally {
      busy = false
    }
  }

  const go = (dir) => (at = stepFrom(trace, at, dir, step))

  function play() {
    if (playing) return stop()
    playing = setInterval(() => {
      const next = stepFrom(trace, at, 1, step)
      if (next === at) stop()
      else at = next
    }, 350)
  }

  function stop() {
    if (playing) clearInterval(playing)
    playing = null
  }

  function keydown(event) {
    if (!trace || event.target.tagName === 'SELECT') return
    if (event.key === 'ArrowRight') { event.preventDefault(); go(1) }
    if (event.key === 'ArrowLeft') { event.preventDefault(); go(-1) }
  }

  $effect(() => stop)
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
<section class="trace" aria-label="Solve trace" tabindex="0" onkeydown={keydown}>
  <header>
    <h2>Trace</h2>
    <button onclick={recordSolve} disabled={busy}>{busy ? 'Recording…' : trace ? 'Record again' : 'Record solve'}</button>
    {#if trace}
      <span class="summary">
        {frames.length} frames ·
        {#if trace.result.ok}
          {trace.result.converged ? 'converged' : 'NOT converged'} in {trace.result.iterations} iterations ({trace.result.method})
        {:else}
          <span class="bad">raised: {trace.result.error}</span>
        {/if}
        {#if trace.truncated}<span class="bad">· recording stopped at {frames.length} frames</span>{/if}
      </span>
    {:else}
      <span class="hint">solve once and keep every unit call and tear iteration, then scrub through it</span>
    {/if}
    <span class="spacer"></span>
    <button onclick={onclose}>Close</button>
  </header>
  {#if error}<p class="error">{error}</p>{/if}

  {#if trace}
    <div class="controls">
      <label>step by
        <select bind:value={step}>
          {#each Object.entries(STEPS) as [key, s] (key)}
            <option value={key}>{s.label}</option>
          {/each}
        </select>
      </label>
      <button onclick={() => (at = 0)} title="first frame">⏮</button>
      <button onclick={() => go(-1)} title="previous (←)">◀</button>
      <button onclick={play} title="play">{playing ? '⏸' : '▶︎'}</button>
      <button onclick={() => go(1)} title="next (→)">▶</button>
      <button onclick={() => (at = frames.length - 1)} title="last frame">⏭</button>
      <input type="range" min="0" max={Math.max(frames.length - 1, 0)} bind:value={at}
             aria-label="frame" />
      <span class="where">{at + 1}/{frames.length} · {describeFrame(frame)}</span>
    </div>

    <div class="body">
      <div class="left">
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img"
             aria-label="tear residual by iteration, log scale">
          <rect x="0" y="0" width={W} height={H} class="frame" />
          {#if plot.points.length > 1}
            <polyline class="line" points={plot.points.filter((p) => p.ok).map((p) => `${p.x},${p.y}`).join(' ')} />
          {/if}
          {#each plot.points as p (p.frame)}
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <circle cx={p.x} cy={p.y} r={p.iteration === iteration ? 4 : 2.5}
                    class:bad={!p.ok} class:here={p.iteration === iteration}
                    role="button" tabindex="-1" onclick={() => (at = p.frame)}>
              <title>iteration {p.iteration}: {formatNumber(p.residual)}</title>
            </circle>
          {/each}
        </svg>
        <p class="axis">
          {#if plot.hi !== null}residual 10<sup>{Math.floor(plot.lo) + 0}</sup>–10<sup>{Math.ceil(plot.hi) + 0}</sup> by iteration, log scale{:else}no tear iteration recorded{/if}
        </p>
        {#if events.length}
          <h3>Events</h3>
          <ul class="events">
            {#each events as e (e.frame)}
              <li><button class="link" class:here={e.frame === at} onclick={() => (at = e.frame)}>{e.label}</button></li>
            {/each}
          </ul>
        {/if}
      </div>

      <div class="right">
        {#if frame?.error}<p class="error">{frame.error}</p>{/if}
        {#if frame?.tear?.length}
          <h3>Tear: what the residual is made of</h3>
          <table>
            <thead><tr><th>stream</th><th>field</th><th>x</th><th>g(x)</th><th>|g − x|</th></tr></thead>
            <tbody>
              {#each frame.tear.slice(0, 8) as r, i (r.stream + r.field)}
                <tr class:top={i === 0}><td>{r.stream}</td><td>{r.field}</td><td>{formatNumber(r.x)}</td><td>{formatNumber(r.g)}</td><td>{formatNumber(r.delta)}</td></tr>
              {/each}
            </tbody>
          </table>
        {/if}
        {#if changes.length}
          <h3>Changed at this step</h3>
          <table>
            <thead><tr><th>stream</th><th>field</th><th>before</th><th>after</th></tr></thead>
            <tbody>
              {#each changes as c (c.stream + c.field)}
                <tr class:bad={typeof c.after === 'number' && !Number.isFinite(c.after)}>
                  <td>{c.stream}</td><td>{c.field}</td><td>{formatNumber(c.before)}</td><td>{formatNumber(c.after)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        {:else if !frame?.tear?.length && !frame?.error}
          <p class="hint">nothing written at this step</p>
        {/if}
        {#if warnings.length}
          <h3>Warnings during the solve</h3>
          <ul class="warnings">{#each warnings as w, i (i)}<li>{w.message ?? w}</li>{/each}</ul>
        {/if}
        <label class="all"><input type="checkbox" bind:checked={showAll} /> every stream at this step</label>
        {#if showAll}
          <table>
            <tbody>
              {#each Object.entries(state) as [name, values] (name)}
                <tr><th colspan="2">{name}</th></tr>
                {#each Object.entries(values) as [field, value] (field)}
                  <tr><td>{field}</td><td>{formatNumber(value)}</td></tr>
                {/each}
              {/each}
            </tbody>
          </table>
        {/if}
      </div>
    </div>
  {/if}
</section>

<style>
  .trace {
    display: flex;
    flex-direction: column;
    height: 22rem;
    flex: none;
    border-top: 1px solid var(--grid);
    background: var(--panel);
    outline: none;
  }
  header, .controls {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px dashed var(--grid);
  }
  .controls { padding: 0.35rem 0.8rem; }
  .controls input[type='range'] { flex: 1; min-width: 6rem; }
  .controls label { font-size: 0.76rem; color: var(--ink-soft); }
  h2 { font-size: 0.85rem; margin: 0; }
  h3 { font-size: 0.74rem; margin: 0.6rem 0 0.25rem; color: var(--ink-soft); font-weight: 600; }
  .spacer { flex: 1; }
  .hint, .summary, .where, .axis { color: var(--ink-soft); font-size: 0.76rem; margin: 0; }
  .where { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 40%; }
  .error { color: var(--bad); font-size: 0.78rem; margin: 0; padding: 0.3rem 0; }
  .trace > .error { padding: 0.4rem 0.8rem; }
  .bad { color: var(--bad); }
  .body {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 22rem 1fr;
    gap: 1rem;
    padding: 0.5rem 0.8rem;
    overflow: auto;
    background: var(--surface);
  }
  svg .frame { fill: none; stroke: var(--grid); }
  svg .line { fill: none; stroke: var(--accent); stroke-width: 1.5; }
  svg circle { fill: var(--accent); cursor: pointer; }
  svg circle.here { fill: var(--ink); }
  svg circle.bad { fill: var(--bad); }
  .events, .warnings { list-style: none; margin: 0; padding: 0; font-size: 0.76rem; }
  .warnings li { color: var(--ink-soft); }
  .link { background: none; border: none; padding: 0.1rem 0; color: var(--accent); cursor: pointer; text-align: left; font-size: 0.76rem; }
  .link.here { font-weight: 650; }
  table { border-collapse: collapse; font-size: 0.76rem; font-variant-numeric: tabular-nums; }
  th, td { padding: 0.1rem 0.6rem 0.1rem 0; text-align: left; }
  thead th { color: var(--ink-soft); font-weight: 600; border-bottom: 1px solid var(--grid); }
  tr.top td { font-weight: 650; }
  tr.bad td { color: var(--bad); }
  .all { display: block; font-size: 0.76rem; color: var(--ink-soft); margin-top: 0.6rem; }
</style>
