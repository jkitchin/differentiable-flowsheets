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

  let {
    document: doc = null,
    positions = null,
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
  let surface

  // Rebuilt whenever the served document changes. Positions the user has
  // dragged live on the node objects, so this deliberately re-reads them
  // from `positions` -- the caller decides what the truth is.
  $effect(() => {
    const graph = toGraph(doc, positions)
    nodes = graph.nodes
    edges = graph.edges
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
  .canvas :global(.svelte-flow__edge.recycle .svelte-flow__edge-text) {
    fill: var(--accent);
  }
  .canvas :global(.svelte-flow__edge-textbg) { fill: var(--surface); }
</style>
