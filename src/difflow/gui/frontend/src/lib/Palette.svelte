<!--
  The operations the catalog knows, grouped and searchable. Drag one onto
  the canvas, or click it to drop it in the middle.

  Blocked operations are shown and flagged rather than hidden: over half
  the catalog needs a `thermo` or a rate law, and a palette without a
  CSTR in it is not a palette. What each one is waiting for is named on
  the row -- `rate_fn`, `thermo`, `solvent` -- because "unbuildable" tells
  you nothing you can act on, and the name is the thing you go and write.
  `needs` is answered against the code context as it stands, so a row
  un-dims when its binding appears rather than at the next reload.

  Dragging a blocked one is still allowed. The refusal names the same
  fields the row does, and a palette that silently swallowed the gesture
  would leave the reader with no idea why nothing happened. For anyone
  who would rather not see them at all, the filter hides them.

  Each row carries the same symbol the node will be drawn with, which is
  the only way the picture on the canvas can be recognised before it is
  there -- and the reason the drag has something to show.
-->
<script>
  import { paletteGroups } from './model/edit.js'
  import UnitSymbol from './nodes/UnitSymbol.svelte'

  let { catalog = {}, ondrop = () => {} } = $props()

  let query = $state('')
  let hideBlocked = $state(false)
  let groups = $derived(paletteGroups(catalog, query, hideBlocked))
  let blocked = $derived(
    Object.values(catalog).filter((s) => (s.needs ?? []).length).length
  )

  /**
   * Hand the operation name to the drag.
   *
   * Two types, on purpose. `application/difflow-operation` is the one the
   * canvas reads; `text/plain` is there because a drag carrying no
   * standard type is treated as carrying nothing by some browsers, and
   * the drop then never fires -- the gesture looks like it works right up
   * until nothing appears.
   */
  function dragstart(event, name) {
    event.dataTransfer.setData('application/difflow-operation', name)
    event.dataTransfer.setData('text/plain', name)
    event.dataTransfer.effectAllowed = 'copy'
  }

  // Enter and Space drop in the middle, the same as a click. A row that
  // can only be reached by mouse is a row that cannot be reached at all
  // by anyone driving this from the keyboard.
  function keydown(event, name) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      ondrop(name, null)
    }
  }
</script>

<aside class="palette">
  <input
    class="search"
    type="search"
    placeholder="search {Object.keys(catalog).length} operations"
    bind:value={query}
  />

  {#if blocked}
    <label class="filter">
      <input type="checkbox" bind:checked={hideBlocked} />
      hide the {blocked} that need something first
    </label>
  {/if}

  <div class="groups">
    {#each groups as group (group.category)}
      <section>
        <h2>{group.category}</h2>
        {#each group.ops as op (op.name)}
          <!-- A div rather than a <button>: a button is not a reliable
               drag source (Firefox and Safari will not start a drag from
               one), and "drag this onto the canvas" is the primary
               gesture here. The role, tabindex and keydown put back
               everything the button was giving us. -->
          <div
            class="op"
            class:unbuildable={!op.buildable}
            role="button"
            tabindex="0"
            draggable="true"
            title={op.buildable
              ? op.description
              : `needs ${op.needs.join(', ') || 'a constructor object'}, defined in the code context — ${op.description}`}
            ondragstart={(e) => dragstart(e, op.name)}
            onclick={() => ondrop(op.name, null)}
            onkeydown={(e) => keydown(e, op.name)}
          >
            <span class="glyph">
              <UnitSymbol operation={op.name} category={op.category} size={20} />
            </span>
            <span class="op-name">{op.name}</span>
            {#if !op.buildable}
              <span class="needs" title="define these in the code context">
                needs {op.needs.join(', ') || 'code'}
              </span>
            {/if}
          </div>
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
  .filter {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin: 0.4rem 0.15rem 0;
    font-size: 0.7rem;
    color: var(--ink-soft);
    cursor: pointer;
  }
  .filter input { margin: 0; }
  .op {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    width: 100%;
    text-align: left;
    padding: 0.28rem 0.45rem;
    margin-bottom: 2px;
    font-size: 0.79rem;
    border: 1px solid transparent;
    border-radius: 5px;
    color: inherit;
    cursor: grab;
    user-select: none;
  }
  .op:hover { background: var(--surface); border-color: var(--line); }
  .op:focus-visible { outline: 2px solid var(--series); outline-offset: 1px; }
  /* The symbol is the same one the node draws with, at row size. Fixed
     width so the names line up whatever the symbol's aspect. */
  .glyph {
    flex: 0 0 22px;
    display: flex;
    justify-content: center;
    color: var(--node-accent);
  }
  .op-name { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  /* Dimmed, but still legible: this row is an instruction, not a
     tombstone. The `needs` chip is what the reader acts on, so it keeps
     full contrast while the name softens. */
  .unbuildable .op-name { color: var(--ink-soft); }
  .unbuildable .glyph { opacity: 0.5; }
  .needs {
    flex: 0 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 0.6rem;
    letter-spacing: 0.04em;
    color: var(--accent);
  }
  .empty { color: var(--ink-soft); font-size: 0.8rem; padding: 0.5rem; }
</style>
