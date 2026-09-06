<!--
  The operations the catalog knows, grouped and searchable. Drag one onto
  the canvas, or click it to drop it in the middle.

  Unbuildable operations are shown and flagged rather than hidden: 34 of
  the 87 need a `thermo` or a rate law, and a palette without a CSTR in
  it is not a palette. Dragging one is allowed; the server refuses with
  the message that names what is missing, which is more use than a
  greyed-out row that says nothing.
-->
<script>
  import { paletteGroups } from './model/edit.js'

  let { catalog = {}, ondrop = () => {} } = $props()

  let query = $state('')
  let groups = $derived(paletteGroups(catalog, query))

  function dragstart(event, name) {
    event.dataTransfer.setData('application/difflow-operation', name)
    event.dataTransfer.effectAllowed = 'copy'
  }
</script>

<aside class="palette">
  <input
    class="search"
    type="search"
    placeholder="search {Object.keys(catalog).length} operations"
    bind:value={query}
  />

  <div class="groups">
    {#each groups as group (group.category)}
      <section>
        <h2>{group.category}</h2>
        {#each group.ops as op (op.name)}
          <button
            class="op"
            class:unbuildable={!op.buildable}
            draggable="true"
            title={op.buildable
              ? op.description
              : `needs ${op.needs.join(', ') || 'a constructor object'} — ${op.description}`}
            ondragstart={(e) => dragstart(e, op.name)}
            onclick={() => ondrop(op.name, null)}
          >
            <span class="op-name">{op.name}</span>
            {#if !op.buildable}<span class="needs">{op.needs.join(', ') || 'code'}</span>{/if}
          </button>
        {/each}
      </section>
    {:else}
      <p class="empty">nothing matches</p>
    {/each}
  </div>
</aside>

<style>
  .palette {
    width: 15rem;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--grid);
    background: var(--panel);
    min-height: 0;
  }
  .search {
    margin: 0.55rem;
    padding: 0.35rem 0.5rem;
    font: inherit;
    font-size: 0.8rem;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
    color: inherit;
  }
  .groups { overflow-y: auto; padding: 0 0.55rem 0.8rem; }
  h2 {
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-soft);
    margin: 0.7rem 0 0.3rem;
    font-weight: 650;
  }
  .op {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.4rem;
    width: 100%;
    text-align: left;
    padding: 0.28rem 0.45rem;
    margin-bottom: 2px;
    font: inherit;
    font-size: 0.79rem;
    border: 1px solid transparent;
    border-radius: 5px;
    background: none;
    color: inherit;
    cursor: grab;
  }
  .op:hover { background: var(--surface); border-color: var(--line); }
  .op-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .unbuildable .op-name { color: var(--ink-soft); }
  .needs {
    flex: none;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--accent);
  }
  .empty { color: var(--ink-soft); font-size: 0.8rem; padding: 0.5rem; }
</style>
