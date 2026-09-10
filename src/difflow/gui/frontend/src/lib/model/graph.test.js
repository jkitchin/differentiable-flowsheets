/**
 * `node --test src/lib/model/*.test.js`
 *
 * The wiring is the part of the canvas that can be wrong without
 * looking wrong, so it is the part that gets tested. node is not a
 * difflow dependency; the Python suite skips these when it is absent.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  FEED,
  PRODUCT,
  arcs,
  feedStreams,
  pendingPositions,
  productStreams,
  toGraph,
  toPositions,
} from './graph.js'

/** feed -> mix -> split -> (product, purge -> recycle back to mix). */
const recycleDoc = {
  units: [
    { name: 'mix', operation: 'Mixer', inlets: ['feed', 'recycle'], outlets: ['mixed'] },
    { name: 'split', operation: 'Splitter', inlets: ['mixed'], outlets: ['product', 'purge'] },
  ],
  feeds: { feed: {} },
  recycles: { purge: 'recycle' },
}

const chainDoc = {
  units: [
    { name: 'mix', operation: 'Mixer', inlets: ['feed'], outlets: ['mixed'] },
    { name: 'split', operation: 'Splitter', inlets: ['mixed'], outlets: ['product', 'purge'] },
  ],
  feeds: { feed: {} },
  recycles: {},
}

test('an empty document draws nothing', () => {
  assert.deepEqual(toGraph(null), { nodes: [], edges: [] })
  assert.deepEqual(toGraph({}), { nodes: [], edges: [] })
})

test('every unit, feed and dangling product is a node', () => {
  const ids = toGraph(chainDoc, {}).nodes.map((n) => n.id)
  assert.deepEqual(ids.sort(), [
    'feed:feed', 'mix', 'product:product', 'product:purge', 'split',
  ])
})

test('a recycled stream is not a product', () => {
  assert.deepEqual(productStreams(recycleDoc), ['product'])
})

test('a recycle destination is not a feed', () => {
  assert.deepEqual(feedStreams(recycleDoc), ['feed'])
})

test('a dangling inlet still gets a node', () => {
  // A unit dropped on the canvas has one until it is wired. Dropping it
  // on the floor would make the unwired inlet invisible.
  const doc = { units: [{ name: 'u', inlets: ['nowhere'], outlets: ['out'] }], feeds: {} }
  assert.deepEqual(feedStreams(doc), ['nowhere'])
})

test('a recycle is one arc, drawn back to its reader', () => {
  const back = arcs(recycleDoc).filter((a) => a.recycle)
  assert.equal(back.length, 1)
  assert.deepEqual(
    { from: back[0].from, to: back[0].to, stream: back[0].stream },
    { from: 'split', to: 'mix', stream: 'purge' },
  )
})

test('a recycle whose destination nothing reads still draws', () => {
  const doc = {
    units: [{ name: 'u', inlets: ['in'], outlets: ['out'] }],
    feeds: { in: {} },
    recycles: { out: 'unread' },
  }
  const back = arcs(doc).filter((a) => a.recycle)
  assert.equal(back.length, 1, 'a dangling recycle goes nowhere, not missing')
  assert.equal(back[0].to, PRODUCT + 'unread')

  // And it lands on a node. An edge pointing at an id that is not in the
  // node array is not drawn at all, which is the failure this guards.
  const g = toGraph(doc, {})
  const ids = new Set(g.nodes.map((n) => n.id))
  for (const e of g.edges) {
    assert.ok(ids.has(e.source) && ids.has(e.target), `${e.id} dangles`)
  }
})

test('a recycle attaches to the inlet it actually feeds', () => {
  // The one thing about a recycle that is easy to get wrong and
  // impossible to see: it leaves `purge` and arrives as `recycle`, so
  // the two ends carry different names. Aiming both at the source name
  // points the arrow at a port `mix` does not have, and @xyflow drops
  // the edge onto a default handle instead of saying anything.
  const edge = toGraph(recycleDoc, {}).edges.find((e) => e.class === 'recycle')
  assert.equal(edge.source, 'split')
  assert.equal(edge.target, 'mix')
  assert.equal(edge.sourceHandle, 'out:purge')
  assert.equal(edge.targetHandle, 'in:recycle')
  const mix = toGraph(recycleDoc, {}).nodes.find((n) => n.id === 'mix')
  assert.ok(mix.data.inlets.includes('recycle'), 'and that port exists')
})

test('an ordinary arc names the same stream at both ends', () => {
  for (const a of arcs(chainDoc)) assert.equal(a.destStream, a.stream)
})

test('edge ids are unique', () => {
  const ids = toGraph(recycleDoc, {}).edges.map((e) => e.id)
  assert.equal(new Set(ids).size, ids.length)
})

test('positions come from the view block', () => {
  const g = toGraph(chainDoc, { mix: { x: 12, y: 34 } })
  assert.deepEqual(g.nodes.find((n) => n.id === 'mix').position, { x: 12, y: 34 })
})

test('an unplaced node lands somewhere reachable', () => {
  const g = toGraph(chainDoc, {})
  for (const n of g.nodes) {
    assert.ok(Number.isFinite(n.position.x) && Number.isFinite(n.position.y), n.id)
  }
  const seen = new Set(g.nodes.map((n) => `${n.position.x},${n.position.y}`))
  assert.equal(seen.size, g.nodes.length, 'and not all on top of each other')
})

test('a nonsense position falls back rather than breaking the canvas', () => {
  const g = toGraph(chainDoc, { mix: { x: null, y: 'up' } })
  const mix = g.nodes.find((n) => n.id === 'mix')
  assert.ok(Number.isFinite(mix.position.x) && Number.isFinite(mix.position.y))
})

test('positions round-trip back out of the canvas', () => {
  const g = toGraph(chainDoc, { mix: { x: 12, y: 34 } })
  assert.deepEqual(toPositions(g.nodes).mix, { x: 12, y: 34 })
})

test('the node keys match the Python vocabulary', () => {
  // difflow.gui.layout writes view.nodes with exactly these keys.
  const ids = new Set(toGraph(chainDoc, {}).nodes.map((n) => n.id))
  assert.ok(ids.has('mix'), 'a unit is its bare name')
  assert.ok(ids.has(FEED + 'feed'))
  assert.ok(ids.has(PRODUCT + 'product'))
})

test('no edge points at a node that is not there', () => {
  // @xyflow drops such an edge silently, so the flowsheet would draw with
  // a connection simply missing and nothing to say it had been.
  for (const doc of [chainDoc, recycleDoc]) {
    const g = toGraph(doc, {})
    const ids = new Set(g.nodes.map((n) => n.id))
    for (const e of g.edges) {
      assert.ok(ids.has(e.source), `${e.id}: no source node`)
      assert.ok(ids.has(e.target), `${e.id}: no target node`)
    }
  }
})

test('every edge lands on a port its node declares', () => {
  const g = toGraph(recycleDoc, {})
  const byId = new Map(g.nodes.map((n) => [n.id, n]))
  const ports = (node, side) =>
    node.type === 'stream'
      ? [`${side}:${node.data.label}`]
      : (side === 'out' ? node.data.outlets : node.data.inlets).map((s) => `${side}:${s}`)
  for (const e of g.edges) {
    assert.ok(ports(byId.get(e.source), 'out').includes(e.sourceHandle), e.id)
    assert.ok(ports(byId.get(e.target), 'in').includes(e.targetHandle), e.id)
  }
})

// -- pending nodes ---------------------------------------------------
//
// A unit that could not be built lands on the canvas anyway, in red.
// It is not in the document -- a flowsheet holds units that exist --
// so everything about it comes in beside the document, and the thing
// that can go wrong is that it is drawn as if it were real: with
// ports, in the node positions, or wired.

const parked = [
  { name: 'flash', operation: 'Flash', needs: ['thermo'],
    hint: 'Flash needs thermo, which is code rather than data.',
    position: { x: 10, y: 20 } },
]

test('a pending unit is drawn although the document has never heard of it', () => {
  const g = toGraph(chainDoc, {}, { pending: parked })
  const node = g.nodes.find((n) => n.id === 'flash')
  assert.ok(node, 'the drop has to be visible')
  assert.equal(node.type, 'unit')
  assert.equal(node.data.operation, 'Flash')
  assert.deepEqual(node.position, { x: 10, y: 20 })
  assert.deepEqual(node.data.pending, {
    needs: ['thermo'], hint: parked[0].hint,
  })
})

test('a pending unit has no ports', () => {
  // It was never built, so it has none -- and a handle would invite a
  // wire to a unit that cannot receive one.
  const g = toGraph(chainDoc, {}, { pending: parked })
  const node = g.nodes.find((n) => n.id === 'flash')
  assert.deepEqual(node.data.inlets, [])
  assert.deepEqual(node.data.outlets, [])
  assert.ok(!g.edges.some((e) => e.source === 'flash' || e.target === 'flash'))
})

test('a pending unit with no position still lands somewhere', () => {
  const g = toGraph(chainDoc, {}, { pending: [{ name: 'x', operation: 'Flash' }] })
  const node = g.nodes.find((n) => n.id === 'x')
  assert.ok(Number.isFinite(node.position.x) && Number.isFinite(node.position.y))
  assert.deepEqual(node.data.pending, { needs: [], hint: '' })
})

test('no pending list draws no pending nodes', () => {
  assert.deepEqual(toGraph(chainDoc, {}).nodes, toGraph(chainDoc, {}, {}).nodes)
})

test('a pending node draws over a stream node, not under one', () => {
  // Later in the array wins in @xyflow, and the red box is the thing
  // that has to be noticed when two nodes land on the same spot.
  const g = toGraph(chainDoc, {}, { pending: parked })
  const ids = g.nodes.map((n) => n.id)
  assert.ok(ids.indexOf('flash') > ids.indexOf('mix'))
  assert.ok(ids.indexOf('flash') < ids.indexOf(PRODUCT + 'product'))
})

test('pending positions are keyed like view.nodes', () => {
  // `movedPositions` treats a key it does not know as moved, so these
  // have to merge into the known set or every drag would re-save a
  // layout for nodes nobody touched.
  assert.deepEqual(pendingPositions(parked), { flash: { x: 10, y: 20 } })
  assert.deepEqual(pendingPositions([]), {})
  assert.deepEqual(pendingPositions(undefined), {})
  assert.deepEqual(pendingPositions([{ name: 'x' }]), {},
    'no position is not a position of NaN')
})
