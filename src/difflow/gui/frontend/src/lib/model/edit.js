/**
 * Turning canvas gestures into requests the server understands.
 *
 * Pure functions again, and for the same reason as `graph.js`: every one
 * of these is a place where a gesture can be sent to the wrong endpoint,
 * or sent with the two ends of a wire swapped, and nothing about that is
 * visible on screen -- the node moves, the edge draws, and the model
 * quietly disagrees with the picture.
 *
 * The server is the source of truth. A gesture becomes a request, the
 * request is followed by a reload, and the canvas is redrawn from what
 * came back. It costs a loopback round trip per gesture and it is worth
 * it: there is exactly one place a topology can be described, and no
 * chance of the picture and the model drifting apart.
 */

import { FEED, PRODUCT } from './graph.js'

/** What a node id refers to: `{kind, name}`. */
export function parseNodeId(id) {
  if (typeof id !== 'string') return null
  if (id.startsWith(FEED)) return { kind: 'feed', name: id.slice(FEED.length) }
  if (id.startsWith(PRODUCT)) return { kind: 'product', name: id.slice(PRODUCT.length) }
  return { kind: 'unit', name: id }
}

/** The stream a handle carries: `out:rx` -> `rx`. */
export function parseHandle(handle) {
  if (typeof handle !== 'string') return null
  const cut = handle.indexOf(':')
  return cut < 0 ? null : handle.slice(cut + 1)
}

/**
 * The wire a connection drag describes, or a reason it is not one.
 *
 * Only unit-to-unit wires are edits. A feed node and a product node are
 * drawn *from* the topology -- they are stream names with nothing on one
 * end, not objects the flowsheet holds -- so dragging one would have to
 * mean something else, and quietly doing nothing is the worst of the
 * available answers.
 */
export function connectionWire(connection) {
  const source = parseNodeId(connection?.source)
  const target = parseNodeId(connection?.target)
  if (!source || !target) return { error: 'that is not a connection' }
  if (source.kind !== 'unit' || target.kind !== 'unit') {
    return { error: 'wire one unit to another; feeds and products follow from the topology' }
  }
  const outlet = parseHandle(connection.sourceHandle)
  const inlet = parseHandle(connection.targetHandle)
  if (!outlet || !inlet) return { error: 'drag from a port to a port' }
  return { wire: { source: source.name, outlet, target: target.name, inlet } }
}

/**
 * The wire an existing edge stands for, or `null` if it is not one.
 *
 * A recycle's two ends carry different stream names, which is why the
 * edge keeps both: `disconnect` matches a recycle on the pair, and
 * sending the source name at both ends would find nothing and fall
 * through to renaming an inlet that was never wired.
 */
export function edgeWire(edge) {
  const source = parseNodeId(edge?.source)
  const target = parseNodeId(edge?.target)
  if (source?.kind !== 'unit' || target?.kind !== 'unit') return null
  const outlet = edge.data?.stream ?? parseHandle(edge.sourceHandle)
  const inlet = edge.data?.destStream ?? parseHandle(edge.targetHandle)
  if (!outlet || !inlet) return null
  return { source: source.name, outlet, target: target.name, inlet }
}

/**
 * The requests a delete gesture becomes, as `{method, path, body}`.
 *
 * Anything the server has no verb for is left out rather than sent and
 * refused: a feed node is not an object to delete, and an edge into one
 * is a picture of a dangling stream, not a wire.
 *
 * Deleting a unit deletes its edges too --- @xyflow reports both in one
 * gesture --- and those edge requests are dropped rather than sent.
 * Unwiring first would rename the doomed unit's ports on the way past,
 * and unwiring *after* would ask the server to disconnect a unit that is
 * no longer there. Removing the unit already drops what depended on it.
 */
export function deleteRequests({ nodes = [], edges = [] } = {}) {
  const units = []
  for (const node of nodes) {
    const parsed = parseNodeId(node.id)
    if (parsed?.kind === 'unit') units.push(parsed.name)
  }
  const going = new Set(units)
  const out = []
  for (const edge of edges) {
    const wire = edgeWire(edge)
    if (wire && !going.has(wire.source) && !going.has(wire.target)) {
      out.push({ method: 'DELETE', path: '/api/connect', body: wire })
    }
  }
  for (const name of units) {
    out.push({ method: 'DELETE', path: `/api/unit/${encodeURIComponent(name)}` })
  }
  return out
}

/**
 * Where a palette drop lands, in flow coordinates.
 *
 * @param {DOMRect} rect  the canvas element's bounding box
 * @param {{x: number, y: number, zoom: number}} viewport
 * @param {number} clientX
 * @param {number} clientY
 *
 * The node is drawn from its top-left corner, so the drop point is
 * nudged up and left by about half a node; dropping on a spot and
 * watching the box appear below and to the right of the cursor reads as
 * a bug even though the coordinate is right.
 */
export function dropPosition(rect, viewport, clientX, clientY, offset = { x: 70, y: 24 }) {
  const zoom = viewport?.zoom || 1
  return {
    x: (clientX - rect.left - (viewport?.x || 0)) / zoom - offset.x,
    y: (clientY - rect.top - (viewport?.y || 0)) / zoom - offset.y,
  }
}

/**
 * The palette, grouped by catalog category and filtered by a query.
 *
 * Unbuildable operations are kept and flagged rather than hidden: 34 of
 * the 87 need a `thermo` or a rate law, and a palette that cannot show a
 * CSTR is not a palette. What they need is on the entry, so the UI can
 * say why rather than merely refusing.
 */
export function paletteGroups(catalog, query = '') {
  const needle = query.trim().toLowerCase()
  const groups = new Map()
  for (const [name, spec] of Object.entries(catalog || {})) {
    const haystack = `${name} ${spec.category || ''} ${spec.description || ''}`
    if (needle && !haystack.toLowerCase().includes(needle)) continue
    const category = spec.category || 'other'
    if (!groups.has(category)) groups.set(category, [])
    groups.get(category).push({
      name,
      description: spec.description || '',
      buildable: spec.buildable !== false,
      needs: spec.constructor_extras || [],
      ports: spec.ports || {},
    })
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([category, ops]) => ({
      category,
      ops: ops.sort((a, b) => a.name.localeCompare(b.name)),
    }))
}

/**
 * Positions that differ from the ones already on the server.
 *
 * Dragging a node fires a stop event for that node, but the array
 * carries every node; sending all of them on every drag would rewrite
 * the whole `view` block to say one box moved.
 */
export function movedPositions(current, known, tolerance = 0.5) {
  const out = {}
  for (const [key, p] of Object.entries(current || {})) {
    const was = (known || {})[key]
    if (!was || Math.abs(was.x - p.x) > tolerance || Math.abs(was.y - p.y) > tolerance) {
      out[key] = p
    }
  }
  return out
}
