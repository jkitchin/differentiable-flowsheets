<!--
  The header's menus.

  A bar of words rather than a row of buttons, for the ordinary reason:
  the editor has eleven panels, preferences and exits, and a toolbar that
  shows all of them at once shows none of them. What goes in the menus is
  `model/menubar.js`; this is the opening and closing of them, and it
  hands the drawing to `ContextMenu` so that a menu here and a menu on a
  right-clicked node are the same object with the same keyboard.
-->
<script>
  import ContextMenu from './ContextMenu.svelte'

  let { menus = [] } = $props()

  // Which menu is open, and the box of the button it hangs from. The
  // button element is kept too, so closing can give the focus back to
  // where the keyboard user left it.
  let open = $state(null)
  let current = $derived(menus.find((m) => m.id === open?.id) ?? null)

  function show(menu, button) {
    const box = button.getBoundingClientRect()
    open = { id: menu.id, button, x: box.left, right: box.right, y: box.bottom + 4 }
  }

  function toggle(menu, event) {
    if (open?.id === menu.id) close()
    else show(menu, event.currentTarget)
  }

  function close() {
    const button = open?.button
    open = null
    // Only when the focus is in the menu that is going away, which is
    // to say only when it would otherwise be dropped on the floor: a
    // menu dismissed by a click somewhere else must not pull the focus
    // back out of wherever that click put it.
    if (button && document.activeElement?.closest?.('menu.menu')) button.focus()
  }

  /** The button for the nth menu, from any button in the bar. */
  const sibling = (button, index) => button.parentElement.children[index]

  const wrap = (index, step) => (index + step + menus.length) % menus.length

  const STEP = { ArrowRight: 1, ArrowLeft: -1 }

  /**
   * Walking the bar with nothing open: the arrows move the focus, and
   * down opens what the focus is on.
   */
  function keydown(event, index) {
    // With a menu open the focus is inside it, not here, and the window
    // below is what is listening. This still fires for the frame before
    // the menu takes the focus, and must not act twice.
    if (open) return
    const step = STEP[event.key]
    if (step) {
      event.preventDefault()
      sibling(event.currentTarget, wrap(index, step)).focus()
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      show(menus[index], event.currentTarget)
    }
  }

  /**
   * Walking the bar with a menu open, which is the thing that makes a
   * menu bar a bar: left and right carry the open menu along it, rather
   * than leaving you to close one popup and go and find the next.
   *
   * On the window because the focus is in the menu by then, and the
   * menu is not a child of this component.
   */
  function walk(event) {
    const step = open && STEP[event.key]
    if (!step) return
    event.preventDefault()
    const next = wrap(menus.findIndex((m) => m.id === open.id), step)
    show(menus[next], sibling(open.button, next))
  }
</script>

<svelte:window onkeydown={walk} />

<nav class="bar" aria-label="editor menus">
  {#each menus as menu, index (menu.id)}
    <button
      type="button"
      class:open={open?.id === menu.id}
      aria-haspopup="menu"
      aria-expanded={open?.id === menu.id}
      onclick={(event) => toggle(menu, event)}
      onkeydown={(event) => keydown(event, index)}
    >{menu.label}</button>
  {/each}
</nav>

{#if current}
  <ContextMenu
    x={open.x}
    right={open.right}
    y={open.y}
    name={current.label}
    items={current.items}
    onclose={close}
  />
{/if}

<style>
  /* Above the menu's own sheet, which covers the page to catch the
     click that dismisses it. Without this, clicking File while View is
     open would close View and stop there, and every switch between two
     menus would cost two clicks. */
  .bar {
    position: relative;
    z-index: 62;
    display: flex;
    gap: 0.1rem;
  }

  button {
    padding: 0.25rem 0.55rem;
    border: 1px solid transparent;
    border-radius: 5px;
    background: none;
    color: var(--ink-soft);
    font-size: 0.82rem;
  }

  button:hover { border-color: transparent; background: var(--grid); color: var(--ink); }

  /* The open one stays lit while its menu is down, which is the only
     thing on screen saying where the menu came from. */
  button.open {
    background: var(--grid);
    color: var(--ink);
  }
</style>
