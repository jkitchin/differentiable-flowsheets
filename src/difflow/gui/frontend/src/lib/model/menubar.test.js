import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import { menuBar, shortPath } from './menubar.js'

/** One menu by id, and one row in it by label. */
const menu = (bar, id) => bar.find((m) => m.id === id)
const row = (bar, id, label) =>
  menu(bar, id).items.find((i) => i.label === label)
const labels = (bar, id) =>
  menu(bar, id).items.filter((i) => !i.separator).map((i) => i.label)

const full = (over = {}) =>
  menuBar({ path: 'plant.json', doc: { units: [] }, links: {
    documentation: 'https://difflow.example/book',
    repository: 'https://github.com/example/difflow',
  }, ...over })

test('the bar is File, Edit, Examples, View and Help, in that order', () => {
  assert.deepEqual(full().map((m) => m.id), ['file', 'edit', 'examples', 'view', 'help'])
  assert.deepEqual(full().map((m) => m.label), ['File', 'Edit', 'Examples', 'View', 'Help'])
})

test('Undo and Redo are offered only when there is a step to take', () => {
  const { actions, seen } = recorder()
  assert.ok(row(full(), 'edit', 'Undo').disabled)
  assert.ok(row(full(), 'edit', 'Redo').disabled)
  const bar = full({ actions, canUndo: true, canRedo: true })
  row(bar, 'edit', 'Undo').run()
  row(bar, 'edit', 'Redo').run()
  assert.deepEqual(seen, [['undo'], ['redo']])
  assert.ok(row(full({ canUndo: true, busy: true }), 'edit', 'Undo').disabled)
})

const EXAMPLES = [
  { key: '01_flash', title: 'Flash drum', description: 'a feed, split' },
  { key: '02_rx', title: 'Reactor', description: '' },
]

test('the examples menu lists each example and opens it by key', () => {
  const { actions, seen } = recorder()
  const bar = full({ actions, examples: EXAMPLES })
  assert.deepEqual(labels(bar, 'examples'), ['Flash drum', 'Reactor'])
  row(bar, 'examples', 'Reactor').run()
  assert.deepEqual(seen, [['example', '02_rx']])
  // It replaces the canvas, and says so before the click.
  assert.match(row(bar, 'examples', 'Flash drum').hint, /a feed, split.*replaces/)
})

test('the examples menu is disabled while busy, and says when it is empty', () => {
  const bar = full({ busy: true, examples: EXAMPLES })
  assert.ok(menu(bar, 'examples').items.every((i) => i.disabled))
  const empty = menu(full(), 'examples').items
  assert.equal(empty.length, 1)
  assert.ok(empty[0].disabled)
})

/** Every action the bar knows how to ask for, recording what it was asked. */
function recorder() {
  const seen = []
  const names = ['undo', 'redo', 'save', 'saveAs', 'newFile', 'openFile', 'reload', 'export', 'quit', 'results', 'context',
                 'console', 'planning', 'assistant', 'portLabels', 'dark',
                 'open', 'classic', 'example', 'widthByFlow', 'colorBy', 'script']
  const actions = Object.fromEntries(
    names.map((n) => [n, (...args) => seen.push([n, ...args])]))
  return { actions, seen }
}

test('every row either separates or runs something', () => {
  const { actions } = recorder()
  for (const m of full({ actions })) {
    for (const item of m.items) {
      assert.ok(item.separator || (item.label && typeof item.run === 'function'),
                `${m.id}: ${JSON.stringify(item)}`)
    }
  }
})

test('each row runs the action it is named for', () => {
  const { actions, seen } = recorder()
  const bar = full({ actions })
  for (const [id, label, name] of [
    ['file', 'Save', 'save'], ['file', 'Save as…', 'saveAs'],
    ['file', 'New', 'newFile'], ['file', 'Open…', 'openFile'], ['file', 'Reload', 'reload'], ['file', 'Quit', 'quit'],
    ['view', 'Results', 'results'], ['view', 'Code context', 'context'],
    ['view', 'Script', 'script'], ['view', 'Console', 'console'], ['view', 'Planning', 'planning'],
    ['view', 'Ask difflow', 'assistant'], ['view', 'Port names', 'portLabels'],
    ['view', 'Dark theme', 'dark'], ['help', 'Classic editor', 'classic'],
  ]) {
    seen.length = 0
    row(bar, id, label).run()
    assert.equal(seen[0]?.[0], name, label)
  }
})

test('Save with no file to save to asks where, rather than greying out', () => {
  const { actions, seen } = recorder()
  const bare = full({ path: '', actions })
  assert.equal(row(bare, 'file', 'Save').disabled, false)
  assert.match(row(bare, 'file', 'Save').hint, /no file yet/)
  row(bare, 'file', 'Save').run()
  assert.deepEqual(seen, [['saveAs']])
})

test('busy disables everything that would edit or write', () => {
  const bar = full({ busy: true })
  for (const label of ['New', 'Open…', 'Save', 'Save as…', 'Reload',
                       'Python script', 'Diagram PNG']) {
    assert.equal(row(bar, 'file', label).disabled, true, label)
  }
  // Opening a drawer is not an edit, and is the one thing still worth
  // doing while a solve is running.
  assert.ok(!row(bar, 'view', 'Results').disabled)
})

test('the four exports are offered, and only with a flowsheet', () => {
  assert.deepEqual(labels(full(), 'file'),
    ['New', 'Open…', 'Save', 'Save as…', 'Reload', 'Python script', 'Flowsheet JSON', 'Diagram SVG',
     'Diagram PNG', 'Quit'])
  const empty = full({ doc: null })
  for (const label of ['Python script', 'Flowsheet JSON', 'Diagram SVG', 'Diagram PNG']) {
    assert.equal(row(empty, 'file', label).disabled, true, label)
  }
})

test('an export in flight marks its own row and bars the others', () => {
  const bar = full({ exporting: 'png' })
  assert.equal(row(bar, 'file', 'Diagram PNG').note, '…')
  assert.equal(row(bar, 'file', 'Diagram PNG').disabled, true)
  assert.equal(row(bar, 'file', 'Python script').disabled, true)
})

test('an export row asks for its own kind', () => {
  const asked = []
  const bar = full({ actions: { export: (kind) => asked.push(kind) } })
  for (const label of ['Python script', 'Flowsheet JSON', 'Diagram SVG', 'Diagram PNG']) {
    row(bar, 'file', label).run()
  }
  assert.deepEqual(asked, ['py', 'json', 'svg', 'png'])
})

test('Quit is the only dangerous row', () => {
  const danger = full().flatMap((m) => m.items.filter((i) => i.danger).map((i) => i.label))
  assert.deepEqual(danger, ['Quit'])
})

test('View ticks the drawers that are open and the preferences that are on', () => {
  const bar = full({ panels: { results: true, console: true }, dark: true })
  assert.equal(row(bar, 'view', 'Results').on, true)
  assert.equal(row(bar, 'view', 'Console').on, true)
  assert.equal(row(bar, 'view', 'Planning').on, false)
  assert.equal(row(bar, 'view', 'Dark theme').on, true)
  assert.equal(row(bar, 'view', 'Port names').on, false)
})

test('a code context that did not run is flagged on its row', () => {
  assert.equal(row(full(), 'view', 'Code context').note, '')
  assert.equal(row(full({ contextError: true }), 'view', 'Code context').note, '!')
  assert.match(row(full({ contextError: true }), 'view', 'Code context').hint,
               /did not run/)
})

test('Help links are disabled when the package named none', () => {
  const bar = menuBar({})
  assert.equal(row(bar, 'help', 'Documentation').disabled, true)
  assert.equal(row(bar, 'help', 'Source on GitHub').disabled, true)
  // The classic editor is served by this same process, so it is never
  // a link that might not exist.
  assert.ok(!row(bar, 'help', 'Classic editor').disabled)
})

test('a Help link opens the URL the server reported', () => {
  const opened = []
  const bar = full({ actions: { open: (url) => opened.push(url) } })
  row(bar, 'help', 'Documentation').run()
  row(bar, 'help', 'Source on GitHub').run()
  assert.deepEqual(opened,
    ['https://difflow.example/book', 'https://github.com/example/difflow'])
})

test('a missing action does not throw', () => {
  const bar = full({ actions: {} })
  for (const m of bar) {
    for (const item of m.items) {
      if (!item.separator && item.run) item.run()
    }
  }
})

test('menuBar() with nothing at all still describes a bar', () => {
  const bar = menuBar()
  assert.equal(bar.length, 5)
  assert.ok(bar.every((m) => m.items.length))
})

test('a long path keeps its tail, which is the half that names the file', () => {
  assert.equal(shortPath('/Users/x/Dropbox/projects/flow/plant.json'),
               '…/flow/plant.json')
  assert.equal(shortPath('plant.json'), 'plant.json')
  assert.equal(shortPath('flow/plant.json'), 'flow/plant.json')
  // Short enough to say in full is said in full, separator and all.
  assert.equal(shortPath('/plant.json'), '/plant.json')
  assert.equal(shortPath('C:\\Users\\x\\flow\\plant.json'), '…/flow/plant.json')
  assert.equal(shortPath('/a/b/c/d.json', 3), '…/b/c/d.json')
})

test('there being no path at all is not a crash', () => {
  assert.equal(shortPath(''), '')
  assert.equal(shortPath(null), '')
  assert.equal(shortPath(undefined), '')
})

test('the solution views wait for a solve, then run and tick', () => {
  const { actions, seen } = recorder()
  const before = full({ actions })
  assert.ok(row(before, 'view', 'Wire width by flow').disabled)
  assert.ok(row(before, 'view', 'Colour wires').disabled)
  assert.equal(row(before, 'view', 'Colour wires').hint, 'solve first')

  const after = full({ actions, solved: true, widthByFlow: true, colorBy: 'T' })
  assert.ok(!row(after, 'view', 'Wire width by flow').disabled)
  assert.ok(row(after, 'view', 'Wire width by flow').on)
  assert.ok(row(after, 'view', 'Colour wires').on)
  row(after, 'view', 'Colour wires').run()
  row(after, 'view', 'Wire width by flow').run()
  assert.deepEqual(seen.map((s) => s[0]), ['colorBy', 'widthByFlow'])
})
