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
  openPorts,
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

test('the units and the declared feeds are the nodes', () => {
  // And nothing else. A product used to get a box of its own, and so did
  // every unwired inlet, with the result that a unit dropped on the
  // canvas arrived already surrounded by boxes standing for streams
  // nobody had declared -- looking wired when it was not.
  const ids = toGraph(chainDoc, {}).nodes.map((n) => n.id)
  assert.deepEqual(ids.sort(), ['feed:feed', 'mix', 'split'])
})

test('only a declared feed gets a box', () => {
  assert.deepEqual(feedStreams(recycleDoc), ['feed'])
  const doc = { units: [{ name: 'u', inlets: ['nowhere'], outlets: ['out'] }], feeds: {} }
  assert.deepEqual(feedStreams(doc), [], 'a dangling inlet is not a feed')
})

test('an unwired port is open at both ends', () => {
  const doc = { units: [{ name: 'u', inlets: ['nowhere'], outlets: ['out'] }], feeds: {} }
  const open = openPorts(doc)
  assert.deepEqual([...open.inlets], ['nowhere'])
  assert.deepEqual([...open.outlets], ['out'])
})

test('a joined port is not open', () => {
  // Three ways to be joined, and every one of them has to count: a unit
  // upstream, a declared feed, and a recycle landing on the inlet.
  const open = openPorts(recycleDoc)
  assert.deepEqual([...open.inlets], [], 'feed, recycle and mixed are all fed')
  assert.deepEqual([...open.outlets].sort(), ['product'],
    'purge leaves by the recycle, so only the product hangs')
})

test('the open ports reach the node that has them', () => {
  // And the two ends arrive under different names, because they mean
  // different things: `openInlets` is what blocks a solve, `products` is
  // what the flowsheet produces. Drawing both as open ports said a
  // finished flowsheet was half-wired.
  const doc = {
    units: [
      { name: 'a', inlets: ['loose'], outlets: ['mid'] },
      { name: 'b', inlets: ['mid'], outlets: ['out'] },
    ],
    feeds: {},
  }
  const byId = new Map(toGraph(doc, {}).nodes.map((n) => [n.id, n]))
  assert.deepEqual(byId.get('a').data.openInlets, ['loose'])
  assert.deepEqual(byId.get('a').data.products, [], 'b reads mid')
  assert.deepEqual(byId.get('b').data.openInlets, [])
  assert.deepEqual(byId.get('b').data.products, ['out'])
})

test('a fed inlet is not open, and a read outlet is not a product', () => {
  // The other half of the same sentence: both ends can be answered
  // without a wire. A declared feed answers an inlet; a recycle carrying
  // an outlet back answers an outlet.
  const doc = {
    units: [
      { name: 'a', inlets: ['feedstock'], outlets: ['vap'] },
      { name: 'b', inlets: ['recycle'], outlets: ['prod'] },
    ],
    feeds: { feedstock: {} },
    recycles: { vap: 'recycle' },
  }
  const byId = new Map(toGraph(doc, {}).nodes.map((n) => [n.id, n]))
  assert.deepEqual(byId.get('a').data.openInlets, [], 'a feed supplies it')
  assert.deepEqual(byId.get('a').data.products, [], 'the recycle carries it')
  assert.deepEqual(byId.get('b').data.openInlets, [], 'the recycle lands here')
  assert.deepEqual(byId.get('b').data.products, ['prod'], 'nothing reads it')
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
  const dangling = toGraph(
    { units: [{ name: 'u', inlets: ['in'], outlets: ['out'] }],
      feeds: { in: {} }, recycles: { out: 'unread' } },
    {},
  )
  assert.ok(new Set(dangling.nodes.map((n) => n.id)).has(PRODUCT + 'unread'),
    'the one product key left is a recycle with nowhere to land')
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

// -- unfinished units ------------------------------------------------
//
// A unit that cannot be built yet is on the flowsheet all the same,
// with its real ports, so that it can be wired now and finished later.
// The pending list beside the document says only which nodes to draw in
// red and what each is waiting for. What can go wrong is that it is
// drawn twice -- once from the document and once from the list -- which
// gives @xyflow two nodes under one id and aims the edges at whichever
// it happened to see last.

const unfinishedDoc = {
  units: [
    { name: 'mix', operation: 'Mixer', inlets: ['feed'], outlets: ['mixed'] },
    { name: 'flash', operation: 'Flash', inlets: ['mixed'], outlets: ['liq', 'vap'] },
  ],
  feeds: { feed: {} },
  recycles: {},
}

const parked = [
  { name: 'flash', operation: 'Flash', needs: ['thermo'],
    hint: 'Flash needs thermo, which is code rather than data.' },
]

test('an unfinished unit is one node, not two', () => {
  const g = toGraph(unfinishedDoc, {}, { pending: parked })
  assert.deepEqual(g.nodes.filter((n) => n.id === 'flash').length, 1)
  const node = g.nodes.find((n) => n.id === 'flash')
  assert.deepEqual(node.data.pending, { needs: ['thermo'], hint: parked[0].hint })
})

test('an unfinished unit keeps its ports and its wires', () => {
  // The whole point of putting it on the flowsheet. Held off it, the
  // node had no ports, and a compressor upstream of a reactor had
  // nothing to connect to until the rate law existed -- which is to say
  // until after the wiring was done.
  const g = toGraph(unfinishedDoc, {}, { pending: parked })
  const node = g.nodes.find((n) => n.id === 'flash')
  assert.deepEqual(node.data.inlets, ['mixed'])
  assert.deepEqual(node.data.outlets, ['liq', 'vap'])
  assert.ok(g.edges.some((e) => e.target === 'flash' && e.targetHandle === 'in:mixed'))
})

test('a unit nobody is waiting on is not drawn as pending', () => {
  const g = toGraph(unfinishedDoc, {}, { pending: parked })
  assert.equal(g.nodes.find((n) => n.id === 'mix').data.pending, undefined)
})

test('a pending entry for a unit that is gone draws nothing', () => {
  // The two arrive in separate fetches, so the list can name a unit the
  // document no longer has. Inventing a node for it would put back the
  // portless red box this replaced.
  const g = toGraph(chainDoc, {}, { pending: [{ name: 'ghost', operation: 'Flash' }] })
  assert.ok(!g.nodes.some((n) => n.id === 'ghost'))
})

test('no pending list draws no pending nodes', () => {
  assert.deepEqual(toGraph(chainDoc, {}).nodes, toGraph(chainDoc, {}, {}).nodes)
})

test('an unfinished unit is placed like every other node', () => {
  // Its coordinate is in view.nodes now, because it is in the document.
  const g = toGraph(unfinishedDoc, { flash: { x: 10, y: 20 } }, { pending: parked })
  assert.deepEqual(g.nodes.find((n) => n.id === 'flash').position, { x: 10, y: 20 })
})
