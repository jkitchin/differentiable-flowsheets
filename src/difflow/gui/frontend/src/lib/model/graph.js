/**
 * The flowsheet document, as a graph the canvas can draw.
 *
 * Pure functions over plain data: no Svelte, no DOM, no fetch. That is
 * deliberate. Everything here is a place where the wiring can silently
 * go wrong -- an inlet attached to the producer of the wrong stream, a
 * recycle drawn as an ordinary arrow, a node left without a position --
 * and none of it is visible in a screenshot. Keeping it out of the
 * components is what lets `node --test` check it.
 *
 * The vocabulary matches the Python side exactly: a unit node is keyed
 * by its bare name, a feed by `feed:<stream>`, a dangling product by
 * `product:<stream>`. Those are the keys `difflow.gui.layout.auto_layout`
 * returns and the keys the `view.nodes` block is written with.
 */

export const FEED = 'feed:'
export const PRODUCT = 'product:'

/** Fallback grid spacing, for a node the layout did not place. */
const COL_W = 220
const ROW_H = 110

/**
 * Which unit, if any, produces each stream.
 * @param {Array} units
 * @returns {Object<string,string>} stream name -> unit name
 */
export function producers(units) {
  const out = {}
  for (const unit of units) for (const s of unit.outlets) out[s] = unit.name
  return out
}

/**
 * Streams that something reads.
 * @param {Array} units
 * @returns {Set<string>}
 */
export function consumed(units) {
  const out = new Set()
  for (const unit of units) for (const s of unit.inlets) out.add(s)
  return out
}

/**
 * Feed nodes: the streams the flowsheet actually declares as feeds.
 *
 * Only those. A dangling inlet used to get one too, and the effect on
 * the canvas was that every unit arrived already surrounded by boxes it
 * was joined to -- boxes standing for streams nobody had asked for,
 * attached to ports that were in fact free. What a free port needs is
 * to look free, which is what `openPorts` is for; a box is for a feed
 * someone declared, with a composition behind it.
 */
export function feedStreams(doc) {
  return Object.keys(doc.feeds || {})
}

/**
 * The ports nothing is joined to, as sets of stream names.
 *
 * An inlet is open unless a unit produces that stream, a declared feed
 * supplies it, or a recycle lands on it. An outlet is open unless a unit
 * reads it or a recycle carries it back. The canvas marks these, and
 * marking them is the whole of what tells the user where the wiring is
 * still missing --- on a flowsheet drawn without stream boxes there is
 * otherwise nothing to see at a port that has nothing attached.
 */
export function openPorts(doc) {
  const units = doc.units || []
  const made = producers(units)
  const eaten = consumed(units)
  const feeds = new Set(Object.keys(doc.feeds || {}))
  const recycles = doc.recycles || {}
  const carriedBack = new Set(Object.keys(recycles))
  const fedByRecycle = new Set(Object.values(recycles))
  const inlets = new Set()
  const outlets = new Set()
  for (const unit of units) {
    for (const s of unit.inlets) {
      if (!made[s] && !feeds.has(s) && !fedByRecycle.has(s)) inlets.add(s)
    }
    for (const s of unit.outlets) {
      if (!eaten.has(s) && !carriedBack.has(s)) outlets.add(s)
    }
  }
  return { inlets, outlets }
}

/**
 * Recycle destinations no unit reads.
 *
 * A real state while editing: `add_recycle(source, dest)` names a
 * destination stream, and nothing has to consume it yet. The arc has to
 * land on a node -- @xyflow drops an edge whose target does not exist,
 * without a word -- so the destination gets one of its own.
 */
export function danglingDestinations(doc) {
  const units = doc.units || []
  const eaten = consumed(units)
  return Object.values(doc.recycles || {}).filter((s) => !eaten.has(s))
}

/**
 * Every arc in the flowsheet, as {from, to, stream, destStream, recycle}.
 *
 * `from` and `to` are node keys. A recycle appears twice in the
 * document -- once as the source unit's outlet, once as the
 * destination's inlet, under two different stream names -- and is drawn
 * as the single arc it is, from the producer of the source stream to
 * whatever reads the destination stream.
 *
 * Which is why an arc carries TWO stream names. On an ordinary arc they
 * are the same name and the distinction costs nothing; on a recycle they
 * differ (`vap` leaves the flash, `recycle` enters the mixer), and using
 * one name at both ends would aim the arrow at a port that does not
 * exist -- an edge that silently detaches rather than one that is
 * visibly wrong.
 */
export function arcs(doc) {
  const units = doc.units || []
  const made = producers(units)
  const recycles = doc.recycles || {}
  // Declared feeds. An inlet nothing produces is not a feed, it is an
  // open port, and it is drawn as one rather than as an arc arriving
  // from a box that stands for nothing.
  const feeds = new Set(feedStreams(doc))
  const out = []

  for (const unit of units) {
    for (const inlet of unit.inlets) {
      const src = made[inlet]
      if (src)
        out.push({ from: src, to: unit.name, stream: inlet,
                   destStream: inlet, recycle: false })
      else if (feeds.has(inlet))
        out.push({ from: FEED + inlet, to: unit.name, stream: inlet,
                   destStream: inlet, recycle: false })
    }
    for (const outlet of unit.outlets) {
      const dest = recycles[outlet]
      if (dest === undefined) continue
      // The recycle's destination is a stream name, not a unit: find who
      // reads it. An unread destination is a dangling recycle, which the
      // canvas should show as going nowhere rather than not at all.
      const reader = units.find((u) => u.inlets.includes(dest))
      out.push({
        from: unit.name,
        to: reader ? reader.name : PRODUCT + dest,
        stream: outlet,
        destStream: dest,
        recycle: true,
      })
    }
  }
  return out
}

/**
 * Build the node and edge arrays @xyflow/svelte wants.
 *
 * @param {Object} doc  a `serialize.to_dict` document
 * @param {Object} [positions]  `view.nodes`, `{key: {x, y}}`. Missing keys
 *   fall back to a grid, so a node added since the last layout still lands
 *   somewhere reachable rather than on top of the origin.
 * @param {Object} [options]
 * @param {Object} [options.catalog]  the served catalog, read only for each
 *   operation's `category`. That is what picks a symbol for an operation
 *   this front end has never heard of -- a plugin's own units -- so a node
 *   drawn without it falls back to a plain block.
 * @param {boolean} [options.portLabels]  name each port beside its handle.
 * @param {Array} [options.pending]  units on the canvas that cannot be
 *   built yet, as the server reports them: `{name, operation, needs,
 *   hint}`. They ARE in the document, with real ports, so that they can
 *   be wired before they are finished -- this list only says which of
 *   those nodes to draw in red, and what each is waiting for.
 */
export function toGraph(doc, positions, options = {}) {
  if (!doc) return { nodes: [], edges: [] }
  const { catalog = {}, portLabels = false, pending = [] } = options
  const place = positions || (doc.view && doc.view.nodes) || {}
  const unfinished = new Map(
    (pending || []).filter((u) => u && u.name).map((u) => [u.name, u])
  )
  const open = openPorts(doc)
  const nodes = []
  let spare = 0
  const at = (key) => {
    const p = place[key]
    if (p && Number.isFinite(p.x) && Number.isFinite(p.y)) return { x: p.x, y: p.y }
    spare += 1
    return { x: COL_W * spare, y: ROW_H * 4 }
  }

  for (const name of feedStreams(doc)) {
    nodes.push({
      id: FEED + name,
      type: 'stream',
      position: at(FEED + name),
      data: { label: name, kind: 'feed', portLabels },
    })
  }
  for (const unit of doc.units || []) {
    // An unfinished unit is one of these, not a node of its own. It has
    // to be: it is in the document with real ports, and minting a second
    // node under the same id would give @xyflow two nodes with one key
    // and leave the edges landing on whichever it saw last.
    const waiting = unfinished.get(unit.name)
    nodes.push({
      id: unit.name,
      type: 'unit',
      position: at(unit.name),
      data: {
        label: unit.name,
        operation: unit.operation,
        category: (catalog[unit.operation] || {}).category || '',
        inlets: unit.inlets,
        outlets: unit.outlets,
        openInlets: unit.inlets.filter((s) => open.inlets.has(s)),
        openOutlets: unit.outlets.filter((s) => open.outlets.has(s)),
        portLabels,
        ...(waiting
          ? { pending: { needs: waiting.needs || [], hint: waiting.hint || '' } }
          : {}),
      },
    })
  }
  // A recycle destination nothing reads still needs somewhere for its
  // arc to land -- @xyflow drops an edge whose target does not exist,
  // without a word. Products do not: an outlet nobody reads is an open
  // port, and it is marked on the unit rather than given a box.
  for (const name of danglingDestinations(doc)) {
    nodes.push({
      id: PRODUCT + name,
      type: 'stream',
      position: at(PRODUCT + name),
      data: { label: name, kind: 'product', portLabels },
    })
  }

  const edges = arcs(doc).map((a) => ({
    id: `${a.from}->${a.to}:${a.stream}->${a.destStream}`,
    source: a.from,
    target: a.to,
    sourceHandle: `out:${a.stream}`,
    targetHandle: `in:${a.destStream}`,
    // A recycle joins two differently named streams, and saying so on the
    // edge is the only place the reader can see which inlet it lands on.
    label: a.recycle ? `${a.stream} \u2192 ${a.destStream}` : a.stream,
    animated: false,
    class: a.recycle ? 'recycle' : '',
    data: { stream: a.stream, destStream: a.destStream, recycle: a.recycle },
  }))
  return { nodes, edges }
}

/** `view.nodes` shaped from the canvas's current node array. */
export function toPositions(nodes) {
  const out = {}
  for (const n of nodes) out[n.id] = { x: n.position.x, y: n.position.y }
  return out
}
