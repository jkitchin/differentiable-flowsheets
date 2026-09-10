<!--
  What a published flowsheet says about itself.

  The editor's inspector asks the server what a unit is; a published page
  has no server, so `difflow.publish` bakes the catalog entries for the
  operations this flowsheet actually uses into the file. Same source --
  `difflow.catalog` reading the classes -- so a published page cannot
  describe a unit differently from the editor, and cannot go stale
  against the code it was published from.

  Read-only throughout: there is nothing here to edit, and pretending
  otherwise on a frozen page would be a lie about what the reader can do.
-->
<script>
  import { fmt } from './model/sweep.js'

  let { node = null, topology = null, catalog = {} } = $props()

  let units = $derived(topology?.units ?? [])
  let unit = $derived(
    node?.type === 'unit' ? units.find((u) => u.name === node.id) ?? null : null,
  )
  let spec = $derived(unit ? catalog[unit.operation] ?? null : null)
  let specs = $derived(
    Object.fromEntries((spec?.parameters ?? []).map((p) => [p.name, p])),
  )
  let feed = $derived(
    node?.type === 'stream' && node.data?.kind === 'feed'
      ? topology?.feeds?.[node.data.label] ?? null
      : null,
  )

  /** A parameter value as one line, whatever it turned out to be. */
  function show(value) {
    if (value === null || value === undefined) return '--'
    if (typeof value === 'number') return fmt(value)
    if (typeof value === 'string') return value
    if (Array.isArray(value))
      return value.length > 6
        ? `[${value.length} values]`
        : `[${value.map(show).join(', ')}]`
    return typeof value === 'object' ? '{...}' : String(value)
  }
</script>

<aside class="details">
  {#if unit}
    <h3>{unit.name}<span class="op">{unit.operation}</span></h3>
    {#if spec?.description}<p class="desc">{spec.description}</p>{/if}

    {#if Object.keys(unit.params ?? {}).length}
      <h4>Parameters</h4>
      <table>
      <tbody>
        {#each Object.entries(unit.params) as [name, value]}
          <tr>
            <td class="name">{name}</td>
            <td class="val">{show(value)}</td>
            <td class="units">{specs[name]?.units ?? ''}</td>
          </tr>
        {/each}
      </tbody>
      </table>
    {/if}

    <h4>Ports</h4>
    <p class="ports">
      in <code>{unit.inlets.join(', ') || '--'}</code><br />
      out <code>{unit.outlets.join(', ') || '--'}</code>
    </p>

    {#if spec?.equations?.length}
      <h4>Equations</h4>
      <!-- Shown as their LaTeX source. Rendering them would mean putting
           KaTeX in a file that travels with a paper, and a quarter of a
           megabyte of typesetting is a poor trade for a page whose job is
           to still open in ten years. -->
      <ul class="eqs">{#each spec.equations as eq}<li><code>{eq}</code></li>{/each}</ul>
    {/if}
    {#if spec?.assumptions?.length}
      <h4>Assumptions</h4>
      <ul>{#each spec.assumptions as a}<li>{a}</li>{/each}</ul>
    {/if}
    {#if spec?.references?.length}
      <h4>References</h4>
      <ul class="refs">{#each spec.references as r}<li>{r}</li>{/each}</ul>
    {/if}
  {:else if feed}
    <h3>{node.data.label}<span class="op">feed</span></h3>
    <table>
    <tbody>
      {#each Object.entries(feed) as [name, value]}
        <tr><td class="name">{name}</td><td class="val">{show(value)}</td></tr>
      {/each}
    </tbody>
    </table>
  {:else}
    <h3>{units.length} units<span class="op">flowsheet</span></h3>
    <p class="desc">Click a unit to see what it is and what it was solved with.</p>
    <table>
    <tbody>
      {#each units as u}
        <tr><td class="name">{u.name}</td><td class="val">{u.operation}</td></tr>
      {/each}
    </tbody>
    </table>
    {#if Object.keys(topology?.recycles ?? {}).length}
      <h4>Recycles</h4>
      <table>
      <tbody>
        {#each Object.entries(topology.recycles) as [source, dest]}
          <tr><td class="name">{source}</td><td class="val">&rarr; {dest}</td></tr>
        {/each}
      </tbody>
      </table>
    {/if}
  {/if}
</aside>

<style>
  .details {
    width: 19rem; flex: none; overflow: auto; padding: 0 0.2rem 0 1.1rem;
    border-left: 1px solid var(--grid);
  }
  h3 {
    margin: 0 0 0.5rem; font-size: 0.95rem;
    display: flex; align-items: baseline; gap: 0.5rem;
  }
  .op { margin-left: auto; color: var(--ink-soft); font-size: 0.72rem;
        font-weight: 400; }
  h4 {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--ink-soft); margin: 1rem 0 0.4rem; font-weight: 600;
  }
  .desc { margin: 0 0 0.5rem; font-size: 0.8rem; color: var(--ink-soft); }
  table { width: 100%; border-collapse: collapse; }
  td {
    padding: 0.25rem 0.2rem; font-size: 0.78rem;
    border-bottom: 1px solid var(--grid); vertical-align: baseline;
  }
  td.name { color: var(--ink-soft); word-break: break-word; }
  td.val { text-align: right; font-variant-numeric: tabular-nums; }
  /* The units are the column that must not be cut: a volume in `m` is
     a different number from one in `m^3`. */
  td.units { width: 4.4rem; color: var(--ink-soft); font-size: 0.7rem;
             text-align: right; white-space: nowrap; padding-left: 0.45rem; }
  .ports { margin: 0; font-size: 0.78rem; color: var(--ink-soft); }
  code { font-size: 0.95em; color: var(--ink); }
  ul { margin: 0; padding-left: 1.1rem; font-size: 0.78rem; }
  li { margin-bottom: 0.25rem; }
  .eqs { list-style: none; padding-left: 0; }
  .refs { color: var(--ink-soft); }
</style>
