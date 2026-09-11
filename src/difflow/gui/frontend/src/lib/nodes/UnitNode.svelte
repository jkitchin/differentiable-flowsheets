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

  A node may also be PENDING: dropped, and waiting on something that has
  to exist before it can be built -- a `thermo`, a rate law, the species
  order. It is drawn in red and dashed. It keeps its ports and can be
  wired like any other node, because wiring first and writing the rate
  law afterwards is the order people work in; what it will not do is
  solve. It is on the canvas rather than in a message because a message
  scrolls away and the drop does not: the red box is both the record
  that you asked for a Flash and the place to find out what it is
  waiting for.

  A port with nothing attached is drawn as a RED DOT. Nothing else says
  so: the canvas draws a box only for a feed someone declared, so an
  unwired port is otherwise just an unremarkable handle at the edge of a
  node, and the flowsheet looks finished when it is not.
-->
<script>
  import { Handle, Position } from '@xyflow/svelte'

  import UnitSymbol from './UnitSymbol.svelte'

  let { data } = $props()

  // Ports are spread over the side of the node, so the geometry has to
  // agree with the CSS: `--port-span` is the fraction of the height they
  // are allowed to use, and the same expression places both sides.
  const at = (i, n) => `${((i + 1) * 100) / (n + 1)}%`

  // The class @xyflow puts on the handle. Missing lists mean a caller
  // that does not compute them, and the honest answer then is to say
  // nothing rather than to mark every port as open -- or, now, to call
  // every port satisfied, which would be the same lie the other way up.
  //
  // `open` and `wired` are the two halves of a question that has been
  // answered: a port is open when nothing is joined to it and wired when
  // something is, and a port whose state is unknown gets neither.
  const state = (which, stream) =>
    !which ? '' : which.includes(stream) ? 'open' : 'wired'
</script>

<div class="unit" class:pending={!!data.pending} title={data.pending?.hint ?? ''}>
  <div class="art">
    <UnitSymbol operation={data.operation} category={data.category} size={38} />
  </div>
  <div class="text">
    <div class="name" title={data.label}>{data.label}</div>
    <div class="op" title={data.operation ?? ''}>{data.operation ?? ''}</div>
    {#if data.pending}
      <!-- What it is waiting for, on the node itself. The same list the
           palette row carries, and the inspector says what to do about
           it -- but a canvas of red boxes with no reason on them is a
           canvas you have to click through one at a time. -->
      <div class="needs">needs {data.pending.needs.join(', ') || 'code'}</div>
    {/if}
  </div>

  {#each data.inlets as stream, i (stream)}
    <Handle
      type="target"
      position={Position.Left}
      id={`in:${stream}`}
      class={state(data.openInlets, stream)}
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
      class={state(data.openOutlets, stream)}
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
  /* Red, dashed and slightly faded: it is a unit that is not there yet.
     Dashed rather than merely red because the difference that matters is
     "not built", and a solid red box reads as a unit that failed. */
  .unit.pending {
    border-style: dashed;
    border-color: var(--bad);
    background: var(--node-fill);
    box-shadow: none;
  }
  .unit.pending .art { color: var(--bad); opacity: 0.75; }
  .unit.pending .name { color: var(--bad); }
  .needs {
    font-size: 0.62rem;
    line-height: 1.2;
    margin-top: 0.15rem;
    color: var(--bad);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
