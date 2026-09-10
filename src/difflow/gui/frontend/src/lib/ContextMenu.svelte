<!--
  A right-click menu, positioned where the click was.

  It holds no knowledge of what the items do: the caller passes them,
  already decided for whatever was clicked, and this draws them and
  reports the choice. That keeps the flowsheet's vocabulary in `App`
  with every other gesture, and leaves this file about the one thing a
  menu is hard at -- staying on screen, closing when it should, and
  being usable from the keyboard.

  Closing is deliberately generous: Escape, a click anywhere else, a
  scroll, a wheel, the window losing focus. A context menu left behind
  after the thing it was about has moved is worse than one that closes
  too eagerly, because it still looks live and its items now lie.
-->
<script>
  let { x = 0, y = 0, title = '', items = [], onclose = () => {} } = $props()

  // Enough to place the menu on the very first frame, before it has been
  // measured. Only this is a guess --- the real box is measured below,
  // and that is what keeps a menu opened near the bottom-right corner of
  // the window on screen.
  const GUESS = { width: 210, height: 240 }
  const MARGIN = 8

  let menu = $state(null)
  // The measured box, once there is one. State rather than a local, so
  // the placement below is a derivation of it and not a second copy.
  let size = $state(null)

  /**
   * Where the menu goes: at the click, flipped rather than slid.
   *
   * Sliding a menu along an edge until it fits puts its first item
   * somewhere the cursor is not, and the first item is the one about to
   * be clicked. Flipping keeps a corner of the menu under the cursor.
   */
  function place(x, y, box) {
    const w = box?.width || GUESS.width
    const h = box?.height || GUESS.height
    return {
      x: x + w + MARGIN > window.innerWidth ? Math.max(MARGIN, x - w) : x,
      y: y + h + MARGIN > window.innerHeight ? Math.max(MARGIN, y - h) : y,
    }
  }

  let at = $derived(place(x, y, size))

  // Measured after every render that could have changed the box, which
  // is any change to the items. Reading the DOM is not a reactive read,
  // so writing `size` here cannot re-trigger this.
  $effect(() => {
    void items
    const box = menu?.getBoundingClientRect()
    if (box) size = { width: box.width, height: box.height }
  })

  // The first enabled item takes focus, so the menu is immediately
  // arrow-navigable. Deferred a frame: the item is in the DOM by now but
  // a flip has not been painted, and focusing a box that is about to
  // move scrolls the page to where it used to be.
  $effect(() => {
    void x, y, items
    const id = requestAnimationFrame(() => {
      menu?.querySelector('button:not([disabled])')?.focus()
    })
    return () => cancelAnimationFrame(id)
  })

  function choose(item) {
    if (item.disabled) return
    onclose()
    // After the close, so an action that opens a panel or moves focus
    // is not immediately undone by this menu tearing itself down.
    item.run?.()
  }

  /** Arrow keys walk the enabled items, wrapping; Escape gives up. */
  function keydown(event) {
    if (event.key === 'Escape') {
      event.stopPropagation()
      onclose()
      return
    }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const all = [...(menu?.querySelectorAll('button:not([disabled])') ?? [])]
    if (!all.length) return
    const last = all.length - 1
    if (event.key === 'Home') return all[0].focus()
    if (event.key === 'End') return all[last].focus()
    const here = all.indexOf(document.activeElement)
    // Nothing in the menu has focus yet -- the frame that gives it to the
    // first item has not run, or a click landed outside it. Start at the
    // end the key is coming from rather than wherever the arithmetic
    // below would land from -1, which is one short of the right answer.
    if (here < 0) return (event.key === 'ArrowUp' ? all[last] : all[0]).focus()
    const next = event.key === 'ArrowDown'
      ? (here === last ? 0 : here + 1)
      : (here === 0 ? last : here - 1)
    all[next].focus()
  }
</script>

<svelte:window
  onkeydown={keydown}
  onresize={onclose}
  onblur={onclose}
  onwheel={onclose}
  onscrollcapture={onclose}
/>

<!-- A transparent sheet over everything, so the next click anywhere
     closes the menu and does nothing else. Without it the click both
     dismisses the menu and lands on whatever was underneath, which on
     this canvas means dropping the selection or starting a pan. -->
<div
  class="sheet"
  role="presentation"
  onpointerdown={onclose}
  oncontextmenu={(e) => { e.preventDefault(); onclose() }}
></div>

<menu
  bind:this={menu}
  class="menu"
  style="left: {at.x}px; top: {at.y}px"
  aria-label={title || 'actions'}
>
  {#if title}
    <li class="title" aria-hidden="true">{title}</li>
  {/if}
  <!-- Unkeyed on purpose. The list is built once when the menu opens
       and thrown away when it closes, so there is nothing to keep
       identity for -- and a key expression here is worse than none: the
       index is not in scope in one, so `label + i` silently keys both
       separators the same and Svelte refuses to render the menu at all. -->
  {#each items as item}
    {#if item.separator}
      <li class="rule" role="separator"></li>
    {:else}
      <li>
        <button
          type="button"
          class:danger={item.danger}
          disabled={!!item.disabled}
          title={item.hint || ''}
          onclick={() => choose(item)}
        >
          <span class="label">{item.label}</span>
          {#if item.note}<span class="note">{item.note}</span>{/if}
        </button>
      </li>
    {/if}
  {/each}
</menu>

<style>
  /* Above the drawers, which are the tallest thing on the page. */
  .sheet {
    position: fixed;
    inset: 0;
    z-index: 60;
  }

  .menu {
    position: fixed;
    z-index: 61;
    margin: 0;
    padding: 0.25rem;
    /* `max-content` rather than `auto`: a fixed box shrinks to the space
       left of the viewport edge, so a menu measured near the right edge
       would report a squeezed width and then flip by too little. */
    width: max-content;
    min-width: 11rem;
    max-width: 17rem;
    list-style: none;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    box-shadow: 0 6px 20px var(--node-shadow);
    font-size: 0.82rem;
  }

  .title {
    padding: 0.25rem 0.5rem 0.3rem;
    color: var(--ink-soft);
    font-weight: 600;
    /* A name can be long and a menu should not stretch to fit one. */
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .rule {
    height: 1px;
    margin: 0.25rem 0.3rem;
    background: var(--line);
  }

  .menu button {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    width: 100%;
    padding: 0.3rem 0.5rem;
    background: none;
    border: 0;
    border-radius: 4px;
    color: var(--ink);
    font: inherit;
    text-align: left;
    cursor: pointer;
  }

  .menu button:hover:not(:disabled),
  .menu button:focus-visible {
    background: var(--grid);
    outline: none;
  }

  .menu button:disabled {
    color: var(--ink-soft);
    cursor: default;
    opacity: 0.6;
  }

  .menu button.danger:hover:not(:disabled),
  .menu button.danger:focus-visible {
    background: var(--bad);
    color: var(--node-fill);
  }

  .label { flex: 1; }

  /* The keyboard shortcut, or "not here and why". Never the point of
     the row, so it stays quiet and does not widen the menu. */
  .note {
    color: var(--ink-soft);
    font-size: 0.75rem;
    white-space: nowrap;
  }

  .menu button.danger:hover .note,
  .menu button.danger:focus-visible .note { color: inherit; }
</style>
