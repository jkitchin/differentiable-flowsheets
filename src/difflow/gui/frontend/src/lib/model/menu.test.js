/**
 * `node --test src/lib/model/*.test.js`
 *
 * A menu is a list of promises about what clicking will do, and a wrong
 * one is invisible: the row reads the same whether it names the node
 * under the cursor or the last one. So the list is built by a pure
 * function and the promises are checked here.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { nodeMenu } from './menu.js'

const CATALOG = {
  Flash: { docs_url: 'https://example.test/unit-operations.html#flash' },
  Mystery: {},
}

const unit = (name, operation) => ({ id: name, type: 'unit', data: { operation } })

/** Every item that is a row, by label. */
const rows = (items) => items.filter((i) => !i.separator).map((i) => i.label)
const row = (items, label) => items.find((i) => i.label === label)

test('a unit offers its documentation, and says so when there is none', () => {
  const open = []
  const { title, items } = nodeMenu({
    node: unit('flash1', 'Flash'),
    catalog: CATALOG,
    actions: { docs: (url) => open.push(url) },
  })
  assert.equal(title, 'flash1')

  const docs = row(items, 'Documentation')
  assert.equal(docs.disabled, false)
  docs.run()
  assert.deepEqual(open, [CATALOG.Flash.docs_url])

  // The row stays, disabled, for an operation with no page: one that
  // vanished would move every item below it up under the cursor.
  const bare = nodeMenu({ node: unit('m', 'Mystery'), catalog: CATALOG }).items
  assert.equal(row(bare, 'Documentation').disabled, true)
  assert.equal(row(bare, 'Documentation').note, 'none yet')
})

test('the ask row names the operation it will ask about', () => {
  const { items } = nodeMenu({ node: unit('flash1', 'Flash'), catalog: CATALOG })
  assert.ok(row(items, 'Ask about Flash'))
})

test('delete carries the unit name, not the node id', () => {
  const gone = []
  const { items } = nodeMenu({
    node: unit('flash1', 'Flash'),
    catalog: CATALOG,
    actions: { remove: (name) => gone.push(name) },
  })
  const kill = row(items, 'Delete')
  assert.equal(kill.danger, true)
  kill.run()
  assert.deepEqual(gone, ['flash1'])
})

test('copy sends the bare name, so a feed does not paste its prefix', () => {
  const copied = []
  const actions = { copy: (name) => copied.push(name) }
  nodeMenu({ node: unit('flash1', 'Flash'), actions }).items
    .find((i) => i.label === 'Copy name').run()
  nodeMenu({ node: { id: 'feed:water', type: 'stream' }, actions }).items
    .find((i) => i.label === 'Copy name').run()
  assert.deepEqual(copied, ['flash1', 'water'])
})

test('a feed is offered only what a feed has', () => {
  // It is drawn from the topology rather than held by the flowsheet, so
  // there is no operation to document and no object to delete. Offering
  // either would be a row that does nothing when clicked.
  const { title, items } = nodeMenu({
    node: { id: 'feed:water', type: 'stream', data: { kind: 'feed' } },
    catalog: CATALOG,
  })
  assert.equal(title, 'feed water')
  assert.deepEqual(rows(items), ['Code context', 'Copy name'])
})

test('a product says which it is', () => {
  const { title } = nodeMenu({ node: { id: 'product:liq', type: 'stream' } })
  assert.equal(title, 'product liq')
})

test('an unfinished unit gets the whole menu', () => {
  // It is a real unit on the flowsheet with an `Incomplete` stand-in for
  // its operation, and the two rows most likely to finish it --- the
  // documentation and the code context --- are the point of the menu.
  const { items } = nodeMenu({ node: unit('flash2', 'Flash'), catalog: CATALOG })
  assert.deepEqual(rows(items), [
    'Documentation', 'Ask about Flash', 'Code context', 'Copy name', 'Delete',
  ])
})

test('nothing to show beats an empty box', () => {
  const { items } = nodeMenu({ node: null })
  assert.deepEqual(items, [])
  assert.deepEqual(nodeMenu({}).items, [])
  assert.deepEqual(nodeMenu().items, [])
})

test('a missing action is not a crash', () => {
  // The caller wires what it has. A row whose action was never passed
  // must do nothing rather than take the page down with it.
  const { items } = nodeMenu({ node: unit('flash1', 'Flash'), catalog: CATALOG })
  for (const item of items.filter((i) => !i.separator && !i.disabled)) {
    assert.doesNotThrow(() => item.run?.())
  }
})
