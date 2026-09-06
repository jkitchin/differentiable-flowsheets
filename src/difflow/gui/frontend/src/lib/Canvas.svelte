<!--
  The canvas: pan, zoom, drag, wire, delete, and drop from the palette.

  It draws and it reports gestures; it does not decide what they mean.
  Every gesture goes up to `App`, which asks the server and redraws from
  the answer. That is one round trip per gesture on the loopback and it
  buys the only property worth having here -- there is exactly one place
  a topology is described, so the picture cannot drift from the model.
-->
<script>
  import {
    Background,
    Controls,
    MiniMap,
    SvelteFlow,
  } from '@xyflow/svelte'
  import '@xyflow/svelte/dist/style.css'

  import StreamNode from './nodes/StreamNode.svelte'
  import UnitNode from './nodes/UnitNode.svelte'
  import { connectionWire, deleteRequests, dropPosition } from './model/edit.js'
  import { toGraph, toPositions } from './model/graph.js'
  import { decorate } from './model/results.js'

  let {
    document: doc = null,
    positions = null,
    flows = null,
    tints = null,
    onconnect = () => {},
    ondeletions = () => {},
    onmove = () => {},
    onadd = () => {},
    onselect = () => {},
    onrefuse = () => {},
  } = $props()

  const nodeTypes = { unit: UnitNode, stream: StreamNode }

  let nodes = $state.raw([])
  let edges = $state.raw([])
  let viewport = $state.raw({ x: 0, y: 0, zoom: 1 })
  let surface = $state(null)

  // Rebuilt whenever the served document changes. Positions the user has
  // dragged live on the node objects, so this deliberately re-reads them
  // from `positions` -- the caller decides what the truth is.
  $effect(() => {
    const graph = toGraph(doc, positions)
    nodes = graph.nodes
    edges = decorate(graph.edges, { flows, tints })
  })

  // The results drawer opens underneath, taking height off the bottom of
  // the canvas, and the flowsheet would simply go behind it. Re-centre
  // rather than re-fit: whoever zoomed in on a corner meant it, and
  // opening a panel is not a reason to throw that away.
  $effect(() => {
    if (!surface) return
    let last = surface.clientHeight
    const observer = new ResizeObserver(() => {
      const height = surface.clientHeight
      if (height && last) viewport = { ...viewport, y: viewport.y + (height - last) / 2 }
      last = height
    })
    observer.observe(surface)
    return () => observer.disconnect()
  })

  function connected({ connection }) {
    const answer = connectionWire(connection)
    if (answer.wire) onconnect(answer.wire)
    else onrefuse(answer.error)
  }

  // @xyflow has already taken them off the canvas by the time this runs.
  // Whatever the server does with the request, the reload that follows
  // puts back anything it kept -- an edge into a product node, say, which
  // is a picture of a dangling stream rather than a wire to cut.
  function deleted({ nodes: gone = [], edges: cut = [] }) {
    ondeletions(deleteRequests({ nodes: gone, edges: cut }))
  }

  function dropped(event) {
    event.preventDefault()
    const operation = event.dataTransfer.getData('application/difflow-operation')
    if (!operation) return
    onadd(operation, dropPosition(
      surface.getBoundingClientRect(), viewport, event.clientX, event.clientY,
    ))
  }
</script>

<div
  class="canvas"
  bind:this={surface}
  role="application"
  ondragover={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }}
  ondrop={dropped}
>
  <SvelteFlow
    bind:nodes
    bind:edges
    bind:viewport
    {nodeTypes}
    fitView
    deleteKey={['Backspace', 'Delete']}
    onconnect={connected}
    ondelete={deleted}
    onnodedragstop={() => onmove(toPositions(nodes))}
    onnodeclick={({ node }) => onselect(node)}
    onpaneclick={() => onselect(null)}
  >
    <Background />
    <Controls />
    <MiniMap />
  </SvelteFlow>
</div>

<style>
  .canvas { width: 100%; height: 100%; }

  /* A recycle is not an ordinary arrow and must not read as one. */
  .canvas :global(.svelte-flow__edge.recycle .svelte-flow__edge-path) {
    stroke: var(--accent);
    stroke-dasharray: 6 4;
  }
  /* Edge labels are portaled out of the edge group, so they cannot be
     styled through it -- one rule for all of them. They carry the stream
     name and, once solved, its flow, so they sit over the lines and need
     a ground of their own to stay readable. */
  .canvas :global(.svelte-flow__edge-label) {
    background: var(--surface);
    color: var(--ink-soft);
    font-size: 0.7rem;
    padding: 0 3px;
    border-radius: 3px;
    font-variant-numeric: tabular-nums;
  }

  /* Sensitivity tinting. The colour is chosen by sign and the weight by
     magnitude, both from the stylesheet: the model layer sets a class and
     a `--tint` and knows nothing about the palette. Placed after the
     recycle rules so a tinted recycle reads as tinted. */
  .canvas :global(.svelte-flow__edge.tinted .svelte-flow__edge-path) {
    stroke-width: calc(1px + 3.5px * var(--tint, 0));
    stroke-dasharray: none;
  }
  .canvas :global(.svelte-flow__edge.tinted.up .svelte-flow__edge-path) {
    stroke: var(--series);
  }
  .canvas :global(.svelte-flow__edge.tinted.down .svelte-flow__edge-path) {
    stroke: var(--accent);
  }
</style>
