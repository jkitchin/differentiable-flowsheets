<!--
  What a stream carries, where it is drawn: the card shown while a wire,
  its label or one of its ports is hovered, or pinned by clicking a wire.

  The numbers are the stream table's row for this stream, drawn where the
  eye already is. The composition bar is a picture of the rows under it,
  never a replacement for them: every segment has a labelled row with its
  number, so no species is told apart by colour alone (three of the eight
  hues are under 3:1 on the light surface, which is legal only because of
  exactly that).
-->
<script>
  import HoverCard from './HoverCard.svelte'
  import { fmt } from './model/results.js'

  let {
    name = '',
    dest = '',
    summary = null,
    x = 0,
    y = 0,
    pinned = false,
    hint = 'click the wire to pin',
    onclose = () => {},
  } = $props()

  const pct = (x) => (x === null ? '' : `${(100 * x).toFixed(x < 0.001 && x > 0 ? 3 : 1)}%`)
</script>

<HoverCard {x} {y} {pinned} label="stream {name}">
  <header>
    <strong>{name}</strong>
    {#if dest && dest !== name}<span class="soft">&rarr; {dest}</span>{/if}
    {#if summary?.phase}<span class="tag">{summary.phase}</span>{/if}
    {#if pinned}
      <button type="button" class="close" aria-label="close" onclick={onclose}>&times;</button>
    {/if}
  </header>

  {#if !summary}
    <p class="hint">not in the last solve</p>
  {:else}
    <dl>
      <dt>T</dt><dd>{fmt(summary.T, 5)} <span class="soft">K</span></dd>
      <dt>P</dt><dd>{summary.P === null ? '' : fmt(summary.P / 1000, 5)} <span class="soft">kPa</span></dd>
      <dt>F</dt><dd>{fmt(summary.total, 4)} <span class="soft">mol/s</span></dd>
    </dl>

    {#if summary.parts.length}
      <!-- A 2px surface gap between segments, from the flex gap, so two
           neighbouring hues never share an edge. -->
      <div class="bar" role="img"
           aria-label="composition: {summary.parts.map((p) => `${p.species} ${pct(p.x)}`).join(', ')}">
        {#each summary.parts as part (part.key)}
          <span style:flex-grow={part.x} style:background={part.color}
                title="{part.species} {pct(part.x)}"></span>
        {/each}
      </div>
    {/if}

    <table>
      <thead><tr><th></th><th>x</th><th>F <span class="soft">mol/s</span></th></tr></thead>
      <tbody>
        {#each summary.rows as row (row.species)}
          <tr class:zero={!row.flow}>
            <td><i class="swatch" style:background={row.color}></i>{row.species}</td>
            <td>{pct(row.x)}</td>
            <td>{fmt(row.flow, 4)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    {#if !pinned && hint}<p class="hint">{hint}</p>{/if}
  {/if}
</HoverCard>

<style>
  dl { display: grid; grid-template-columns: auto 1fr; gap: 0 0.6rem; margin: 0 0 0.4rem; }
  dt { color: var(--ink-soft); }
  dd { margin: 0; text-align: right; }

  .bar {
    display: flex;
    gap: 2px;
    height: 8px;
    margin: 0.2rem 0 0.35rem;
  }
  .bar span { min-width: 2px; flex-basis: 0; }
  .bar span:first-child { border-radius: 4px 0 0 4px; }
  .bar span:last-child { border-radius: 0 4px 4px 0; }
  .bar span:only-child { border-radius: 4px; }
</style>
