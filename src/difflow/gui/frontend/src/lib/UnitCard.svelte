<!--
  What a unit did, read from the streams around it: the card shown while
  a unit is hovered after a solve.

  Three short tables rather than one wide one: in, out (with each
  outlet's share of what leaves, which is a flash's vapor fraction or a
  splitter's split), and the change in each species on the way through,
  which is where a reactor's product or a separator's cut shows up. Then
  the unit's own numbers, so the card answers "and what is it set to?"
  without a trip to the inspector.
-->
<script>
  import HoverCard from './HoverCard.svelte'
  import { fmt } from './model/results.js'

  let { name = '', operation = '', summary = null, x = 0, y = 0 } = $props()

  const pct = (x) => (x === null || x === undefined ? '' : `${(100 * x).toFixed(1)}%`)
  // Signed, so a species a unit consumes reads as a loss at a glance.
  const signed = (v) => {
    if (!Number.isFinite(v)) return ''
    if (Math.abs(v) < 1e-12) return '0'
    return (v > 0 ? '+' : '−') + fmt(Math.abs(v), 4)
  }
  const value = (v) => (Array.isArray(v) ? v.map((n) => fmt(n, 4)).join(', ') : fmt(v, 5))
</script>

<HoverCard {x} {y} label="unit {name}">
  <header>
    <strong>{name}</strong>
    <span class="soft">{operation}</span>
  </header>

  {#if !summary}
    <p class="hint">not in the last solve</p>
  {:else}
    <h4>in</h4>
    <table>
      <thead><tr><th></th><th>T <span class="soft">K</span></th><th>F <span class="soft">mol/s</span></th></tr></thead>
      <tbody>
        {#each summary.inlets as r (r.stream)}
          <tr class:zero={!r.total}><td>{r.stream}</td><td>{fmt(r.T, 5)}</td><td>{fmt(r.total, 4)}</td></tr>
        {/each}
      </tbody>
    </table>

    <h4>out</h4>
    <table>
      <thead><tr><th></th><th>T <span class="soft">K</span></th><th>F <span class="soft">mol/s</span></th><th>share</th></tr></thead>
      <tbody>
        {#each summary.outlets as r (r.stream)}
          <tr class:zero={!r.total}>
            <td>{r.stream}</td><td>{fmt(r.T, 5)}</td><td>{fmt(r.total, 4)}</td><td>{pct(r.share)}</td>
          </tr>
        {/each}
      </tbody>
    </table>

    {#if summary.change.length}
      <h4>change, out &minus; in</h4>
      <table>
        <thead><tr><th></th><th>in</th><th>out</th><th>&Delta; <span class="soft">mol/s</span></th></tr></thead>
        <tbody>
          {#each summary.change as c (c.species)}
            <tr class:zero={Math.abs(c.delta) < 1e-12}>
              <td><i class="swatch" style:background={c.color}></i>{c.species}</td>
              <td>{fmt(c.in, 4)}</td><td>{fmt(c.out, 4)}</td><td>{signed(c.delta)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}

    {#if summary.params.length}
      <h4>parameters</h4>
      <dl>
        {#each summary.params as p (p.key)}
          <dt>{p.key}</dt><dd>{value(p.value)}</dd>
        {/each}
      </dl>
    {/if}
  {/if}
</HoverCard>

<style>
  dl { display: grid; grid-template-columns: auto 1fr; gap: 0 0.6rem; margin: 0; }
  dt { color: var(--ink-soft); }
  dd { margin: 0; text-align: right; }
</style>
