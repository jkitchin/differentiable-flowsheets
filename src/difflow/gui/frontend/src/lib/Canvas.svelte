<!--
  The canvas. Read-only for now: pan, zoom and drag a node, but no
  wiring and no palette -- those arrive with the editing routes. What it
  does establish is that the topology the server serves is the topology
  drawn, ports and recycles included.
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
  import { toGraph } from './model/graph.js'

  let { document: doc = null, positions = null } = $props()

  const nodeTypes = { unit: UnitNode, stream: StreamNode }

  let nodes = $state.raw([])
  let edges = $state.raw([])

  // Rebuilt whenever the served document changes. Positions the user has
  // dragged live on the node objects, so this deliberately re-reads them
  // from `positions` -- the caller decides what the truth is.
  $effect(() => {
    const graph = toGraph(doc, positions)
    nodes = graph.nodes
    edges = graph.edges
  })
</script>

<div class="canvas">
  <SvelteFlow bind:nodes bind:edges {nodeTypes} fitView>
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
