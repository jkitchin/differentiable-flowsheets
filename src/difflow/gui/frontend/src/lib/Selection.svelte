<!--
  What is selected, and the few things step 3 can do to it: rename it,
  read its ports, delete it. Parameter fields, the docstring and the
  equations belong to the inspector and arrive with it.
-->
<script>
  let { node = null, onrename = () => {}, ondelete = () => {} } = $props()

  let draft = $state('')
  let unit = $derived(node?.type === 'unit')

  // Reset whenever the selection changes, so the box never shows the
  // name of a node that is no longer selected.
  $effect(() => { draft = node?.data?.label ?? '' })

  function commit() {
    const name = draft.trim()
    if (name && name !== node.data.label) onrename(node.id, name)
  }
</script>

<aside class="selection">
  {#if !node}
    <p class="hint">
      Drag an operation onto the canvas. Wire a unit's outlet to another
      unit's inlet. Select and press Delete to remove.
    </p>
  {:else if !unit}
    <h2>{node.data.label}</h2>
    <p class="kind">{node.data.kind}</p>
    <p class="hint">
      A stream, not an object the flowsheet holds: it follows from the
      units that make and read it.
    </p>
  {:else}
    <h2>{node.data.operation}</h2>
    <label>
      name
      <input value={draft} oninput={(e) => (draft = e.currentTarget.value)}
             onblur={commit} onkeydown={(e) => e.key === 'Enter' && commit()} />
    </label>
    <h3>inlets</h3>
    <ul>{#each node.data.inlets as s (s)}<li>{s}</li>{/each}</ul>
    <h3>outlets</h3>
    <ul>{#each node.data.outlets as s (s)}<li>{s}</li>{/each}</ul>
    <button class="danger" onclick={() => ondelete(node.id)}>Delete unit</button>
  {/if}
</aside>

<style>
  .selection {
    width: 14rem;
    flex: none;
    padding: 0.8rem;
    border-left: 1px solid var(--grid);
    background: var(--panel);
    overflow-y: auto;
    font-size: 0.8rem;
  }
  h2 { font-size: 0.9rem; margin: 0 0 0.5rem; }
  h3 {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-soft);
    margin: 0.9rem 0 0.25rem;
  }
  .kind, .hint { color: var(--ink-soft); line-height: 1.45; }
  label { display: block; color: var(--ink-soft); }
  input {
    display: block;
    width: 100%;
    margin-top: 0.2rem;
    padding: 0.3rem 0.4rem;
    font: inherit;
    border: 1px solid var(--line);
    border-radius: 5px;
    background: var(--surface);
    color: inherit;
  }
  ul { margin: 0; padding-left: 1.1rem; }
  li { font-family: var(--mono, ui-monospace, monospace); font-size: 0.76rem; }
  .danger {
    margin-top: 1rem;
    width: 100%;
    padding: 0.35rem;
    font: inherit;
    font-size: 0.79rem;
    border: 1px solid var(--line);
    border-radius: 5px;
    background: var(--surface);
    color: var(--bad);
    cursor: pointer;
  }
  .danger:hover { border-color: var(--bad); }
</style>
