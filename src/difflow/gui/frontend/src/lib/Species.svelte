<!--
  The species the flowsheet is written in, in the header, editable while
  it is still empty.

  It belongs here rather than in a dialog because it is the first thing an
  empty editor needs and nothing else can be done until it exists: every
  stream is an array indexed by this list, so a unit cannot be built
  before it is named. The palette says "needs species" on the rows that
  are waiting; this is where that is answered.

  Which is also why it stops being editable once the flowsheet holds a
  unit. Re-ordering or renaming the list under existing units would
  re-index arrays that are already full of numbers -- silently turning a
  water flow into an ethanol flow. The server refuses it for the same
  reason; the field greys itself out so the refusal is not a surprise.
-->
<script>
  let {
    species = [],
    editable = true,
    busy = false,
    onapply = () => {},
  } = $props()

  // `null` means "show what the server said". Anything else is an unsaved
  // draft. Keeping the two apart is what lets a reload refresh the field
  // without yanking the text out from under someone mid-word.
  let draft = $state(null)
  let served = $derived(species.join(', '))
  let text = $derived(draft ?? served)
  let dirty = $derived(draft !== null && draft.trim() !== served)

  /** `water, ethanol` -> `['water', 'ethanol']`, comma or space. */
  function parse(value) {
    return String(value).split(/[,\s]+/).map((s) => s.trim()).filter(Boolean)
  }

  async function apply() {
    if (!dirty) { draft = null; return }
    const answer = await onapply(parse(draft))
    // Held on a refusal: the text is what the reader has to fix, and
    // resetting it to the old list would hide what they typed.
    if (answer?.ok !== false) draft = null
  }

  function keydown(event) {
    if (event.key === 'Enter') { event.preventDefault(); apply() }
    else if (event.key === 'Escape') { draft = null; event.currentTarget.blur() }
    // The canvas listens for single letters; a species name must not
    // toggle the theme on its way in.
    event.stopPropagation()
  }
</script>

<label class="species" class:empty={!species.length}>
  <span class="tag">species</span>
  {#if editable}
    <input
      type="text"
      value={text}
      placeholder="water, ethanol"
      disabled={busy}
      size="18"
      title="the species every stream is written in, in order"
      oninput={(e) => (draft = e.currentTarget.value)}
      onkeydown={keydown}
      onblur={apply}
    />
    {#if dirty}<span class="hint">↵</span>{/if}
  {:else}
    <span
      class="fixed"
      title="the flowsheet's units index their streams by this list; delete them, or edit the file, to change it"
    >{served}</span>
  {/if}
</label>

<style>
  .species { display: flex; align-items: center; gap: 0.3rem; }
  .tag {
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--ink-soft);
  }
  input {
    font: inherit;
    font-size: 0.78rem;
    padding: 0.2rem 0.4rem;
    border: 1px solid var(--line);
    border-radius: 5px;
    background: var(--surface);
    color: inherit;
  }
  /* An empty list is the one thing standing between an empty canvas and a
     flowsheet, so the field says so rather than sitting there quietly. */
  .empty input { border-color: var(--accent); }
  .hint { color: var(--accent); font-size: 0.7rem; }
  .fixed {
    font-size: 0.78rem;
    color: var(--ink-soft);
    max-width: 14rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
