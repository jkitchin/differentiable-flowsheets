/**
 * `node --test src/lib/model/*.test.js`
 *
 * A gesture sent to the wrong endpoint, or with the two ends of a wire
 * swapped, looks exactly like a gesture that worked until the next
 * reload. So the translation gets tested and the components stay thin.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  connectionWire,
  deleteRequests,
  dropPosition,
  edgeWire,
  movedPositions,
  paletteGroups,
  parseHandle,
  parseNodeId,
} from './edit.js'

test('a node id says what it refers to', () => {
  assert.deepEqual(parseNodeId('reactor'), { kind: 'unit', name: 'reactor' })
  assert.deepEqual(parseNodeId('feed:F1'), { kind: 'feed', name: 'F1' })
  assert.deepEqual(parseNodeId('product:liq'), { kind: 'product', name: 'liq' })
})

test('a handle names its stream', () => {
  assert.equal(parseHandle('out:rx'), 'rx')
  assert.equal(parseHandle('in:feed'), 'feed')
  assert.equal(parseHandle('nonsense'), null)
})

test('a stream name containing a colon survives', () => {
  // `feed:` is a node prefix, not a stream one, and only the FIRST colon
  // separates a handle from its stream.
  assert.equal(parseHandle('out:a:b'), 'a:b')
})

test('a drag between two units is a wire', () => {
  const { wire } = connectionWire({
    source: 'flash', sourceHandle: 'out:liq',
    target: 'mixer', targetHandle: 'in:recycle',
  })
  assert.deepEqual(wire, {
    source: 'flash', outlet: 'liq', target: 'mixer', inlet: 'recycle',
  })
})

test('a drag from a feed says why it is not', () => {
  const answer = connectionWire({
    source: 'feed:F1', sourceHandle: 'out:F1',
    target: 'mixer', targetHandle: 'in:feed',
  })
  assert.ok(!answer.wire)
  assert.match(answer.error, /feeds and products/)
})

test('a drag onto the body of a node is not a wire', () => {
  const answer = connectionWire({ source: 'a', target: 'b' })
  assert.ok(!answer.wire && answer.error)
})

test('an edge carries both ends of a recycle', () => {
  // The one that matters: `vap` leaves the flash and arrives as
  // `recycle`. Sending `vap` at both ends matches no recycle, and
  // `disconnect` would fall through to renaming an inlet nothing wired.
  assert.deepEqual(
    edgeWire({
      source: 'flash', target: 'mixer',
      sourceHandle: 'out:vap', targetHandle: 'in:recycle',
      data: { stream: 'vap', destStream: 'recycle', recycle: true },
    }),
    { source: 'flash', outlet: 'vap', target: 'mixer', inlet: 'recycle' },
  )
})

test('an edge into a product node is not a wire', () => {
  assert.equal(edgeWire({ source: 'flash', target: 'product:liq' }), null)
})

test('a delete gesture becomes one request per thing the server owns', () => {
  const requests = deleteRequests({
    nodes: [{ id: 'reactor' }, { id: 'feed:F1' }],
    edges: [
      { source: 'a', target: 'b', data: { stream: 's', destStream: 's' } },
      { source: 'b', target: 'product:out' },
    ],
  })
  assert.deepEqual(requests.map((r) => r.path), ['/api/connect', '/api/unit/reactor'])
  assert.equal(requests[1].method, 'DELETE')
})

test('deleting a unit does not also try to unwire it', () => {
  // @xyflow reports the node and its edges in one gesture. Unwiring first
  // renames the doomed unit's ports on the way past; unwiring after asks
  // the server about a unit that is gone. Removing it covers both.
  const requests = deleteRequests({
    nodes: [{ id: 'reactor' }],
    edges: [
      { source: 'mixer', target: 'reactor', data: { stream: 'm', destStream: 'm' } },
      { source: 'reactor', target: 'flash', data: { stream: 'rx', destStream: 'rx' } },
      { source: 'flash', target: 'mixer', data: { stream: 'v', destStream: 'r' } },
    ],
  })
  assert.deepEqual(requests.map((r) => r.path), ['/api/connect', '/api/unit/reactor'])
  assert.deepEqual(requests[0].body.source, 'flash', 'the untouched wire still goes')
})

test('a unit name with a space is escaped into the path', () => {
  const [request] = deleteRequests({ nodes: [{ id: 'hot reactor' }] })
  assert.equal(request.path, '/api/unit/hot%20reactor')
})

test('deleting nothing asks for nothing', () => {
  assert.deepEqual(deleteRequests(), [])
  assert.deepEqual(deleteRequests({ nodes: [], edges: [] }), [])
})

test('a drop lands under the cursor whatever the viewport', () => {
  const rect = { left: 100, top: 50 }
  const at = (viewport) =>
    dropPosition(rect, viewport, 300, 250, { x: 0, y: 0 })
  assert.deepEqual(at({ x: 0, y: 0, zoom: 1 }), { x: 200, y: 200 })
  assert.deepEqual(at({ x: 40, y: 20, zoom: 1 }), { x: 160, y: 180 })
  assert.deepEqual(at({ x: 0, y: 0, zoom: 2 }), { x: 100, y: 100 })
})

test('a drop with no viewport still lands somewhere finite', () => {
  const p = dropPosition({ left: 0, top: 0 }, undefined, 10, 10)
  assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y))
})

test('the palette groups by category and sorts', () => {
  const groups = paletteGroups({
    Splitter: { category: 'mixing', buildable: true },
    Mixer: { category: 'mixing', buildable: true },
    CSTR: { category: 'reactors', buildable: false, constructor_extras: [] },
  })
  assert.deepEqual(groups.map((g) => g.category), ['mixing', 'reactors'])
  assert.deepEqual(groups[0].ops.map((o) => o.name), ['Mixer', 'Splitter'])
})

test('an unbuildable operation is kept and flagged', () => {
  // 34 of 87 need a thermo or a rate law. A palette that cannot show a
  // CSTR is not a palette; it has to say why instead of hiding it.
  const [group] = paletteGroups({
    Flash: { category: 'separation', buildable: false, constructor_extras: ['thermo'] },
  })
  assert.equal(group.ops[0].buildable, false)
  assert.deepEqual(group.ops[0].needs, ['thermo'])
})

test('the served needs list beats the class-level buildable flag', () => {
  // The catalog's own `buildable` asks whether a form could construct
  // the class. The server answers a narrower question -- whether a drop
  // would succeed here, now -- and where they disagree the server wins,
  // because it is the one the adder also consults. AbsorberParams.solvent
  // is a required str: no callable, no constructor object, so the class
  // reads as buildable and the drop still fails.
  const [group] = paletteGroups({
    AmineAbsorber: { category: 'capture', buildable: true, needs: ['solvent'] },
  })
  assert.equal(group.ops[0].buildable, false)
  assert.deepEqual(group.ops[0].needs, ['solvent'])
})

test('an empty needs list un-blocks an operation the class calls unbuildable', () => {
  // What the code context defines changes the answer: once a thermo
  // exists, a Flash is droppable even though its class still requires
  // one. The row has to un-dim, or defining the binding looks like it
  // did nothing.
  const [group] = paletteGroups({
    Flash: { category: 'separation', buildable: false, needs: [] },
  })
  assert.equal(group.ops[0].buildable, true)
})

test('a catalog with no needs field falls back to constructor_extras', () => {
  // An older difflow serving this page knows about `thermo` and cannot
  // know about a missing `rate_fn`. Reading the old field keeps the
  // flagging it could do rather than dropping it entirely.
  const [group] = paletteGroups({
    Flash: { category: 'separation', buildable: false, constructor_extras: ['thermo'] },
  })
  assert.equal(group.ops[0].buildable, false)
  assert.deepEqual(group.ops[0].needs, ['thermo'])
})

test('hideBlocked drops the ones waiting on something, and only those', () => {
  const catalog = {
    Mixer: { category: 'mixing', needs: [] },
    Flash: { category: 'separation', needs: ['thermo'] },
  }
  const names = (hide) =>
    paletteGroups(catalog, '', hide).flatMap((g) => g.ops.map((o) => o.name))
  assert.deepEqual(names(false), ['Mixer', 'Flash'])
  assert.deepEqual(names(true), ['Mixer'])
})

test('hiding every operation in a category drops the category too', () => {
  // An empty heading is worse than no heading: it reads as a loading bug.
  const groups = paletteGroups(
    { Flash: { category: 'separation', needs: ['thermo'] } }, '', true,
  )
  assert.deepEqual(groups, [])
})

test('the search reads names, categories and descriptions', () => {
  const catalog = {
    CSTR: { category: 'reactors', description: 'A stirred tank.' },
    Heater: { category: 'heat transfer', description: 'Raises temperature.' },
  }
  const names = (q) => paletteGroups(catalog, q).flatMap((g) => g.ops.map((o) => o.name))
  assert.deepEqual(names('cstr'), ['CSTR'])
  assert.deepEqual(names('heat'), ['Heater'])
  assert.deepEqual(names('temperature'), ['Heater'])
  // grouped by category, so 'heat transfer' sorts before 'reactors'
  assert.deepEqual(names('  '), ['Heater', 'CSTR'], 'blank shows everything')
  assert.deepEqual(names('zzz'), [])
})

test('only the nodes that actually moved are sent', () => {
  const known = { a: { x: 0, y: 0 }, b: { x: 10, y: 10 } }
  const now = { a: { x: 0, y: 0 }, b: { x: 10, y: 40 }, c: { x: 1, y: 1 } }
  assert.deepEqual(movedPositions(now, known), {
    b: { x: 10, y: 40 },
    c: { x: 1, y: 1 },
  })
})

test('a node the server has never placed counts as moved', () => {
  assert.deepEqual(movedPositions({ a: { x: 1, y: 2 } }, {}), { a: { x: 1, y: 2 } })
  assert.deepEqual(movedPositions({ a: { x: 1, y: 2 } }, null), { a: { x: 1, y: 2 } })
})
