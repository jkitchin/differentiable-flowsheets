<!--
  The canvas proper: pan, zoom, drag, wire, delete, and drop from the
  palette. `Canvas.svelte` wraps this in a provider; everything that needs
  the live flow instance lives here, because `useSvelteFlow` reads a
  context the provider sets and a component cannot read its own.

  It draws and it reports gestures; it does not decide what they mean.
  Every gesture goes up to `App`, which asks the server and redraws from
  the answer. That is one round trip per gesture on the loopback and it
  buys the only property worth having here -- there is exactly one place
  a topology is described, so the picture cannot drift from the model.
-->
<script>
  import {
    Background,
    BackgroundVariant,
    Controls,
    SvelteFlow,
    useSvelteFlow,
  } from '@xyflow/svelte'
  import '@xyflow/svelte/dist/style.css'

  import StreamCard from './StreamCard.svelte'
  import UnitCard from './UnitCard.svelte'
  import WireLegend from './WireLegend.svelte'
  import StreamNode from './nodes/StreamNode.svelte'
  import UnitNode from './nodes/UnitNode.svelte'
  import { connectionWire, deleteRequests, dropPosition } from './model/edit.js'
  import { toGraph, toPositions } from './model/graph.js'
  import { decorate } from './model/results.js'
  import { colorOptions, colorScale, flowWidths, streamSummary, unitSummary } from './model/solution.js'

  let {
    document: doc = null,
    positions = null,
    pending = [],
    flows = null,
    tints = null,
    // The last good solve, for the hover card and the solution views.
    solve = null,
    widthByFlow = false,
    colorBy = '',
    oncolorby = () => {},
    catalog = {},
    portLabels = false,
    dark = false,
    onconnect = () => {},
    ondeletions = () => {},
    onmove = () => {},
    onadd = () => {},
    onselect = () => {},
    onmenu = () => {},
    onrefuse = () => {},
    readonly = false,
  } = $props()

  const nodeTypes = { unit: UnitNode, stream: StreamNode }

  // Orthogonal, like every process drawing ever printed. Bezier arcs read
  // as a wiring diagram; right angles read as pipe, and they make it
  // possible to see which of two nearly-parallel lines goes where.
  const defaultEdgeOptions = { type: 'smoothstep' }

  const flow = useSvelteFlow()

  let nodes = $state.raw([])
  let edges = $state.raw([])
  let surface = $state(null)

  // Rebuilt whenever the served document changes. Positions the user has
  // dragged live on the node objects, so this deliberately re-reads them
  // from `positions` -- the caller decides what the truth is.
  let theme = $derived(dark ? 'dark' : 'light')
  let sized = $derived(widthByFlow && solve ? flowWidths(solve) : null)
  let scale = $derived(colorBy && solve ? colorScale(solve, colorBy, theme) : null)

  $effect(() => {
    const graph = toGraph(doc, positions, { catalog, portLabels, pending })
    // Solved, a port's native tooltip ("leaves the flowsheet as a
    // product") would open on top of the stream card, which says more.
    // `flows` too, for the label beside each product's ring.
    nodes = solve
      ? graph.nodes.map((n) => ({ ...n, data: { ...n.data, solved: true, flows } }))
      : graph.nodes
    edges = decorate(graph.edges, {
      flows, tints, widths: sized?.widths ?? null, colors: scale?.colors ?? null,
    })
  })

  // What is being hovered --- a stream or a unit --- or the stream pinned
  // by a click, and where the pointer was relative to the canvas. A pinned
  // card outlives the hover; hovering something else while one is pinned
  // shows nothing new --- the pin is what the user asked to keep reading.
  let hover = $state(null)
  let pinned = $state(null)
  let dragging = $state(false)
  let shown = $derived(pinned ?? hover)

  /**
   * The wire a label belongs to. On a short wire the label covers most of
   * it, so the label has to answer a hover as the wire would. The library
   * portals labels out of their edge and gives them no id, but every
   * label's text is one `decorate` wrote, and one stream feeds one wire.
   */
  function labelled(label) {
    const text = label.textContent.trim()
    return edges.find((e) => String(e.label ?? '').trim() === text) ?? null
  }

  const fromEdge = (edge) =>
    edge ? { kind: 'stream', stream: edge.data?.stream, dest: edge.data?.destStream } : null

  /**
   * What the element under the pointer stands for, most specific first.
   *
   * A port before its unit: a handle is inside its node, and the stream
   * at a port is what someone pointing at a port wants. That is also the
   * only way to read a PRODUCT, which has no wire --- the last unit's
   * outlets are rings on its edge and nothing else. Then a wire's label,
   * the wire, a feed's box, and last the unit itself.
   */
  function target(el) {
    if (!el?.closest || !surface?.contains(el)) return null
    const handle = el.closest('.svelte-flow__handle')
    const id = handle?.getAttribute('data-handleid') ?? ''
    if (/^(in|out):/.test(id)) return { kind: 'stream', stream: id.slice(id.indexOf(':') + 1) }
    const label = el.closest('.svelte-flow__edge-label')
    if (label) return fromEdge(labelled(label))
    const wire = el.closest('.svelte-flow__edge')
    if (wire) return fromEdge(edges.find((e) => e.id === wire.getAttribute('data-id')))
    const node = el.closest('.svelte-flow__node')
    const nodeId = node?.getAttribute('data-id') ?? ''
    if (nodeId.startsWith('feed:')) return { kind: 'stream', stream: nodeId.slice(5) }
    const unit = (doc?.units ?? []).find((u) => u.name === nodeId)
    if (unit) return { kind: 'unit', unit }
    return null
  }

  function place(event, what) {
    const box = surface.getBoundingClientRect()
    return { ...what, x: event.clientX - box.left, y: event.clientY - box.top }
  }

  function over(event) {
    if (!solve || dragging) return
    const what = target(event.target)
    const same = what && hover && what.kind === hover.kind &&
      (what.kind === 'unit' ? what.unit.name === hover.unit?.name : what.stream === hover.stream)
    // Moving between the pieces of one thing (a node's label and its
    // symbol) keeps the card where it opened rather than chasing the cursor.
    if (what && !same) hover = place(event, what)
  }

  function out(event) {
    if (!target(event.relatedTarget)) hover = null
  }

  /** Clicking a wire or its label pins that stream's card. */
  function click(event) {
    if (!solve) return
    const el = event.target
    if (!el?.closest?.('.svelte-flow__edge, .svelte-flow__edge-label')) return
    const what = target(el)
    if (what?.kind !== 'stream') return
    pinned = pinned?.stream === what.stream ? null : place(event, what)
  }

  // A solve that has gone (an edit) takes the cards with it: they would
  // be describing a flowsheet that no longer exists.
  $effect(() => {
    if (!solve) { hover = null; pinned = null }
  })

  // The results drawer opens underneath, taking height off the bottom of
  // the canvas, and the flowsheet would simply go behind it. Re-centre
  // rather than re-fit: whoever zoomed in on a corner meant it, and
  // opening a panel is not a reason to throw that away.
  //
  // Read and written through the flow instance rather than through a
  // bound `viewport` prop: binding it makes the viewport *controlled*,
  // and a controlled viewport is one this component then owns entirely --
  // `fitView` included, which is why a flowsheet opened from a file used
  // to sit off screen until it was panned to.
  $effect(() => {
    if (!surface) return
    let last = surface.clientHeight
    const observer = new ResizeObserver(() => {
      const height = surface.clientHeight
      if (height && last) {
        const now = flow.getViewport()
        flow.setViewport({ ...now, y: now.y + (height - last) / 2 })
      }
      last = height
    })
    observer.observe(surface)
    return () => observer.disconnect()
  })

  // @xyflow calls this with the `Connection` ITSELF -- `{source, target,
  // sourceHandle, targetHandle}` -- not with an event object wrapping
  // one. Destructuring `{ connection }` off it read `undefined`, and the
  // failure was silent in the worst way: the library adds the edge to the
  // canvas before calling this, so the wire appeared, the server never
  // heard about it, and it vanished at the next redraw.
  function connected(connection) {
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

  /**
   * Right-click on a node: our menu instead of the browser's.
   *
   * The node is selected on the way past, as every context menu on
   * anything does --- a menu about a box that is not the highlighted one
   * invites reading its items against the highlighted one instead. The
   * screen coordinate goes up as-is: the menu is positioned in the
   * viewport, not on the canvas, so it must not scale with the zoom or
   * slide out from under the cursor when the flowsheet is panned.
   */
  function contextMenu({ event, node }) {
    event.preventDefault()
    onselect(node)
    onmenu({ node, x: event.clientX, y: event.clientY })
  }

  function dropped(event) {
    if (readonly) return
    event.preventDefault()
    // The palette sets both types; `text/plain` is the one some browsers
    // will carry, so a drag with only the custom type never arrives.
    const operation =
      event.dataTransfer.getData('application/difflow-operation') ||
      event.dataTransfer.getData('text/plain')
    if (!operation) return
    onadd(operation, dropPosition(
      flow.screenToFlowPosition({ x: event.clientX, y: event.clientY }),
    ))
  }
</script>

<div
  class="canvas"
  bind:this={surface}
  role="application"
  onpointerover={over}
  onpointerout={out}
  onclick={click}
  ondragover={(e) => {
    if (readonly) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }}
  ondrop={dropped}
>
  <SvelteFlow
    bind:nodes
    bind:edges
    {nodeTypes}
    {defaultEdgeOptions}
    colorMode={dark ? 'dark' : 'light'}
    connectionLineType="smoothstep"
    fitView
    fitViewOptions={{ padding: 0.25, maxZoom: 1.2 }}
    nodesDraggable={!readonly}
    nodesConnectable={!readonly}
    deleteKey={readonly ? [] : ['Backspace', 'Delete']}
    onconnect={connected}
    ondelete={deleted}
    onnodedragstart={() => { dragging = true; hover = null }}
    onnodedragstop={() => { dragging = false; onmove(toPositions(nodes)) }}
    onnodeclick={({ node }) => onselect(node)}
    onpaneclick={() => { pinned = null; onselect(null) }}

    onnodecontextmenu={contextMenu}
  >
    <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
    <!-- The lock toggles dragging, which a frozen canvas does not do. No
         minimap: it is an overview of a figure the reader can already see
         whole, and it covers the corner a product node lands in. -->
    <Controls showInteractive={!readonly} />
  </SvelteFlow>

  {#if shown?.kind === 'unit' && solve}
    <UnitCard
      name={shown.unit.name}
      operation={shown.unit.operation}
      summary={unitSummary(shown.unit, solve, theme)}
      x={shown.x}
      y={shown.y}
    />
  {:else if shown && solve}
    <StreamCard
      name={shown.stream}
      dest={shown.dest}
      summary={streamSummary(solve.streams?.[shown.stream], solve.species, theme)}
      x={shown.x}
      y={shown.y}
      pinned={!!pinned}
      hint={shown.dest !== undefined ? 'click the wire to pin' : ''}
      onclose={() => (pinned = null)}
    />
  {/if}

  {#if solve && (colorBy || widthByFlow)}
    <WireLegend
      options={colorOptions(solve)}
      {colorBy}
      lo={scale?.lo ?? null}
      hi={scale?.hi ?? null}
      uniform={scale?.uniform ?? false}
      peak={sized?.peak ?? null}
      showWidth={widthByFlow}
      {theme}
      onchange={oncolorby}
      onclose={() => oncolorby('')}
    />
  {/if}
</div>

<style>
  .canvas { position: relative; width: 100%; height: 100%; }

  /* Ports. Small, grey and square-ish rather than the library's blue
     circles: on a drawing whose accent means "this is a recycle" and
     whose blue means "this is a feed", a blue dot on every node is a
     third meaning for a colour that already has two. */
  .canvas :global(.svelte-flow__handle) {
    width: 7px;
    height: 7px;
    border-radius: 2px;
    border: 1px solid var(--node-fill);
    background: var(--port);
  }
  .canvas :global(.svelte-flow__handle:hover),
  .canvas :global(.svelte-flow__handle.connectingto) {
    background: var(--series);
  }

  /* An inlet with nothing arriving: a red dot, and round, so that it
     does not read as one of the square grey ports that happens to be
     tinted. This is the only mark on the canvas that says the flowsheet
     cannot be solved -- `solve` refuses this stream by name -- and it is
     an INLET mark only. An unconnected port used to be shown by giving
     it a stream box, which drew boxes for streams nobody had declared
     and made every fresh node look already connected. */
  .canvas :global(.svelte-flow__handle.open) {
    background: var(--bad);
    border-radius: 50%;
    border-color: var(--node-fill);
  }

  /* And its answer: a port something is joined to is a green dot. Round,
     like the red one, because the two are one state with two values and
     changing the shape as well would read as two unrelated marks. The
     edge already says a wire exists; what the dot adds is the same
     sentence at the port that was red a moment ago, which is where the
     user is looking after making the connection. A port with no lists
     computed stays a plain grey square and claims nothing. */
  .canvas :global(.svelte-flow__handle.wired) {
    background: var(--good);
    border-radius: 50%;
    border-color: var(--node-fill);
  }

  /* The third state, and the one the other two were being asked to cover
     between them: an outlet nothing reads. That is a PRODUCT -- the
     stream leaves, `solve` returns it with the rest, and there is
     nothing further to do about it. Drawn in red it read as an unfinished
     flowsheet, so the last unit of a perfectly complete one always
     looked broken.

     A ring rather than a colour: it is the wire's own grey, so it claims
     neither trouble nor success, and hollow, which is what an open end
     looks like. Slightly larger than the other two so the hole survives
     the border at this size. */
  .canvas :global(.svelte-flow__handle.product) {
    width: 9px;
    height: 9px;
    background: var(--node-fill);
    border: 2px solid var(--wire);
    border-radius: 50%;
  }

  /* Wires, and the line that follows the cursor while one is being made. */
  .canvas :global(.svelte-flow__edge-path),
  .canvas :global(.svelte-flow__connectionline path) {
    stroke: var(--wire);
    stroke-width: 1.4;
  }
  .canvas :global(.svelte-flow__edge.selected .svelte-flow__edge-path) {
    stroke: var(--series);
    stroke-width: 2;
  }

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

  /* The solution views. After the tint rules, so a wire sized or
     coloured by the solve reads as that; a recycle keeps its dashes, which
     say what kind of stream it is and not what is in it. The hit area is
     the library's own invisible wide path, so a thin wire is still easy
     to hover. */
  .canvas :global(.svelte-flow__edge.sized .svelte-flow__edge-path) {
    stroke-width: var(--w);
  }
  /* A recycle's dashes scale with its width. At a fixed 6-4 a wide
     dashed wire is a row of squares, which reads as a different line
     style rather than as the same one drawn heavier. */
  .canvas :global(.svelte-flow__edge.recycle.sized .svelte-flow__edge-path) {
    stroke-dasharray: calc(var(--w) * 2.5 + 4px) calc(var(--w) * 1.5 + 3px);
  }
  .canvas :global(.svelte-flow__edge.colored .svelte-flow__edge-path) {
    stroke: var(--c);
  }
  .canvas :global(.svelte-flow__edge:hover .svelte-flow__edge-path) {
    filter: drop-shadow(0 0 2px var(--series));
  }

  /* The library's own furniture, in difflow's palette. */
  .canvas :global(.svelte-flow__controls-button) {
    background: var(--node-fill);
    border-bottom: 1px solid var(--grid);
    fill: var(--ink-soft);
  }
  .canvas :global(.svelte-flow__controls-button:hover) { background: var(--panel); }
</style>
