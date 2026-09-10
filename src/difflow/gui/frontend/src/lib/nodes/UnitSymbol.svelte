<!--
  One PFD symbol, drawn from the primitive list in `symbols.js`.

  Stroked in `currentColor` so the symbol takes the colour of whatever
  draws it -- the node, the palette row, the inspector heading -- and
  needs no second copy for the dark theme.

  The primitives are spelled out rather than dispatched through
  `<svelte:element>`: SVG children need the SVG namespace, and getting
  that wrong renders nothing at all with no error anywhere.
-->
<script>
  import { symbol } from './symbols.js'

  let { operation = '', category = '', size = 34 } = $props()

  let drawing = $derived(symbol(operation, category))
</script>

<svg
  class="symbol"
  viewBox="0 0 48 40"
  width={size}
  height={(size * 40) / 48}
  role="img"
  aria-label={drawing.label}
>
  {#each drawing.shapes as [kind, attrs], i (i)}
    {#if kind === 'rect'}
      <rect {...attrs} />
    {:else if kind === 'line'}
      <line {...attrs} />
    {:else if kind === 'circle'}
      <circle {...attrs} />
    {:else if kind === 'ellipse'}
      <ellipse {...attrs} />
    {:else if kind === 'path'}
      <path {...attrs} />
    {:else if kind === 'polyline'}
      <polyline {...attrs} />
    {/if}
  {/each}
</svg>

<style>
  /* One place the stroke is set, so a symbol is a shape and not a
     collection of per-shape styling decisions. Shapes that want a fill
     say so in the data and override this. */
  .symbol {
    display: block;
    overflow: visible;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
    vector-effect: non-scaling-stroke;
  }
</style>
