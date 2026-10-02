<!--
  The key to the solution views, in the canvas's corner: what the wire
  colour means and over what range, and what the width is scaled to.

  The colour-by choice lives here rather than in the View menu because
  the list depends on the flowsheet --- one row per species --- and a menu
  that grows a row for every species of a large flowsheet is a menu no
  one can find anything in. The menu turns the view on; this says what
  it is showing and lets you change it.
-->
<script>
  import { fmt } from './model/results.js'
  import { RAMP } from './model/solution.js'

  let {
    options = [],
    colorBy = '',
    lo = null,
    hi = null,
    uniform = false,
    peak = null,
    showWidth = false,
    theme = 'light',
    onchange = () => {},
    onclose = () => {},
  } = $props()

  let option = $derived(options.find((o) => o.key === colorBy) ?? null)
  let gradient = $derived(`linear-gradient(to right, ${(RAMP[theme] ?? RAMP.light).join(', ')})`)
</script>

<div class="legend">
  {#if colorBy}
    <div class="row">
      <label>
        <span class="sr">colour wires by</span>
        <select value={colorBy} onchange={(e) => onchange(e.currentTarget.value)}>
          {#each options as o (o.key)}
            <option value={o.key}>{o.label}</option>
          {/each}
        </select>
      </label>
      <button type="button" class="close" aria-label="stop colouring wires" onclick={onclose}>&times;</button>
    </div>
    {#if lo === null}
      <p class="note">no stream has a value for this</p>
    {:else}
      <div class="ramp" style:background={uniform ? (RAMP[theme] ?? RAMP.light)[2] : gradient}></div>
      <div class="ends">
        <span>{fmt(lo, 4)}</span>
        <span class="u">{option?.units ?? ''}</span>
        {#if !uniform}<span>{fmt(hi, 4)}</span>{/if}
      </div>
      {#if uniform}<p class="note">the same on every stream</p>{/if}
    {/if}
  {/if}
  {#if showWidth}
    <p class="width" class:rule={colorBy}>
      <svg width="34" height="10" aria-hidden="true">
        <path d="M1 5 H12" stroke="var(--wire)" stroke-width="1.2" />
        <path d="M17 5 H33" stroke="var(--wire)" stroke-width="7" />
      </svg>
      <span>width scales with flow{#if peak}; widest {fmt(peak, 3)}&nbsp;<span class="u">mol/s</span>{/if}</span>
    </p>
  {/if}
</div>

<style>
  .legend {
    position: absolute;
    /* Bottom right: the zoom controls have the other corner, and the
       library's attribution sits under this one. */
    right: 0.6rem;
    bottom: 1.6rem;
    z-index: 5;
    width: 15rem;
    padding: 0.45rem 0.55rem;
    border: 1px solid var(--node-line);
    border-radius: 7px;
    background: var(--node-fill);
    box-shadow: 0 2px 8px var(--node-shadow);
    color: var(--ink);
    font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
  }
  .row { display: flex; align-items: center; gap: 0.3rem; }
  label { flex: 1; }
  select {
    width: 100%;
    font: inherit;
    padding: 0.1rem 0.2rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface);
    color: var(--ink);
  }
  .close {
    padding: 0 0.3rem;
    border: none;
    background: none;
    color: var(--ink-soft);
    font-size: 1rem;
    line-height: 1;
  }
  .ramp { height: 8px; margin-top: 0.4rem; border-radius: 4px; }
  .ends { display: flex; justify-content: space-between; margin-top: 0.15rem; }
  .u, .note { color: var(--ink-soft); }
  .note { margin: 0.25rem 0 0; }
  .width { display: flex; align-items: center; gap: 0.35rem; margin: 0; color: var(--ink-soft); }
  .width.rule { margin-top: 0.4rem; padding-top: 0.35rem; border-top: 1px solid var(--grid); }
  .sr {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
  }
</style>
