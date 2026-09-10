<!--
  One unit on the canvas: its PFD symbol, its name, and one handle per
  port. The handles are named `in:<stream>` / `out:<stream>` so an edge
  lands on the port that actually carries that stream rather than on the
  middle of the box, which is the whole reason to draw ports at all.

  The symbol is the point of the node. A flowsheet of identical labelled
  rectangles has to be read name by name; a flowsheet of columns, drums
  and compressors can be taken in at a glance, which is why process
  engineers have drawn them that way for a century. `symbols.js` holds
  the vocabulary and the operation -> symbol map.

  Port labels are off by default and toggled for the whole canvas at
  once. On a wired flowsheet the edge already carries the stream name,
  so labelling both ends of every arc triples the text on screen to say
  what it already said; while wiring, they are exactly what you need.
-->
<script>
  import { Handle, Position } from '@xyflow/svelte'

  import UnitSymbol from './UnitSymbol.svelte'

  let { data } = $props()

  // Ports are spread over the side of the node, so the geometry has to
  // agree with the CSS: `--port-span` is the fraction of the height they
  // are allowed to use, and the same expression places both sides.
  const at = (i, n) => `${((i + 1) * 100) / (n + 1)}%`
</script>

<div class="unit">
  <div class="art">
    <UnitSymbol operation={data.operation} category={data.category} size={38} />
  </div>
  <div class="text">
    <div class="name" title={data.label}>{data.label}</div>
    <div class="op" title={data.operation ?? ''}>{data.operation ?? ''}</div>
  </div>

  {#each data.inlets as stream, i (stream)}
    <Handle
      type="target"
      position={Position.Left}
      id={`in:${stream}`}
      style={`top:${at(i, data.inlets.length)}`}
    />
    {#if data.portLabels}
      <span class="port in" style={`top:${at(i, data.inlets.length)}`}>{stream}</span>
    {/if}
  {/each}
  {#each data.outlets as stream, i (stream)}
    <Handle
      type="source"
      position={Position.Right}
      id={`out:${stream}`}
      style={`top:${at(i, data.outlets.length)}`}
    />
    {#if data.portLabels}
      <span class="port out" style={`top:${at(i, data.outlets.length)}`}>{stream}</span>
    {/if}
  {/each}
</div>

<style>
  .unit {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-width: 9.5rem;
    max-width: 13rem;
    padding: 0.45rem 0.7rem 0.45rem 0.5rem;
    border: 1px solid var(--node-line);
    border-radius: 10px;
    background: var(--node-fill);
    box-shadow: 0 1px 2px var(--node-shadow);
  }
  /* The symbol carries the accent; the text stays in the reading colour,
     because a node whose label is also coloured reads as a warning. */
  .art { color: var(--node-accent); flex: 0 0 auto; }
  .text { min-width: 0; text-align: left; }
  .name {
    font-weight: 650;
    letter-spacing: -0.01em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .op {
    font-size: 0.7rem;
    color: var(--ink-soft);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Port names, set just outside the node so they do not eat into it.
     `pointer-events: none` matters -- a label over a handle would take
     the mousedown that starts a wire. */
  .port {
    position: absolute;
    transform: translateY(-50%);
    font-size: 0.6rem;
    line-height: 1;
    letter-spacing: 0.02em;
    color: var(--ink-soft);
    background: var(--node-fill);
    padding: 1px 3px;
    border-radius: 3px;
    white-space: nowrap;
    pointer-events: none;
  }
  .port.in { right: 100%; margin-right: 6px; }
  .port.out { left: 100%; margin-left: 6px; }
</style>
