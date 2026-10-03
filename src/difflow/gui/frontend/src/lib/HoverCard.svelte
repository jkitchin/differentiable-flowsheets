<!--
  The frame the canvas's hover cards share: placed beside the pointer and
  kept inside the canvas, flipped left of the cursor near the right edge
  and above it near the bottom, so a card never opens off screen.

  A card that is only hovered lets the mouse through --- it must not
  steal the hover from the wire under it, or it would flicker as the
  cursor crossed its own edge. A pinned card is there to be read and
  closed, so it takes the mouse.
-->
<script>
  let { x = 0, y = 0, pinned = false, label = '', children } = $props()

  let card = $state(null)
  let left = $derived.by(() => {
    const width = card?.offsetWidth ?? 240
    const room = card?.parentElement?.clientWidth ?? Infinity
    return x + 14 + width > room ? Math.max(4, x - 14 - width) : x + 14
  })
  let top = $derived.by(() => {
    const height = card?.offsetHeight ?? 160
    const room = card?.parentElement?.clientHeight ?? Infinity
    return y + 14 + height > room ? Math.max(4, y - 14 - height) : y + 14
  })
</script>

<div
  class="card"
  class:pinned
  bind:this={card}
  style:left="{left}px"
  style:top="{top}px"
  role="dialog"
  aria-label={label}
>
  {@render children?.()}
</div>

<style>
  .card {
    position: absolute;
    z-index: 20;
    min-width: 13rem;
    max-width: 22rem;
    padding: 0.5rem 0.65rem 0.45rem;
    border: 1px solid var(--node-line);
    border-radius: 7px;
    background: var(--node-fill);
    box-shadow: 0 4px 14px var(--node-shadow);
    color: var(--ink);
    font-size: 0.75rem;
    font-variant-numeric: tabular-nums;
    pointer-events: none;
  }
  .card.pinned { pointer-events: auto; border-color: var(--ink-soft); }

  /* The pieces both cards are made of. Global to the card, since the
     markup inside comes from whichever card is rendering. */
  .card :global(header) { display: flex; align-items: baseline; gap: 0.4rem; margin-bottom: 0.3rem; }
  .card :global(.soft) { color: var(--ink-soft); }
  .card :global(.tag) {
    padding: 0 0.3rem;
    border: 1px solid var(--line);
    border-radius: 3px;
    color: var(--ink-soft);
    font-size: 0.68rem;
  }
  .card :global(.close) {
    margin-left: auto;
    padding: 0 0.3rem;
    border: none;
    background: none;
    color: var(--ink-soft);
    font-size: 1rem;
    line-height: 1;
  }
  .card :global(table) { width: 100%; border-collapse: collapse; }
  .card :global(th) { color: var(--ink-soft); font-weight: normal; text-align: right; }
  .card :global(td) { text-align: right; padding: 0 0 0 0.6rem; }
  .card :global(td:first-child), .card :global(th:first-child) { text-align: left; padding: 0; }
  .card :global(tr.zero td) { color: var(--ink-soft); }
  .card :global(i.swatch) {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: 0.35rem;
    border-radius: 2px;
  }
  .card :global(.hint) { margin: 0.3rem 0 0; font-size: 0.68rem; color: var(--ink-soft); }
  .card :global(h4) {
    margin: 0.4rem 0 0.1rem;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--ink-soft);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
</style>
