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

test('the bar is File, View and Help, in that order', () => {
  assert.deepEqual(full().map((m) => m.id), ['file', 'view', 'help'])
  assert.deepEqual(full().map((m) => m.label), ['File', 'View', 'Help'])
})

/** Every action the bar knows how to ask for, recording what it was asked. */
function recorder() {
  const seen = []
  const names = ['save', 'reload', 'export', 'quit', 'results', 'context',
                 'console', 'planning', 'assistant', 'portLabels', 'dark',
                 'open', 'classic']
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
    ['file', 'Save', 'save'], ['file', 'Reload', 'reload'], ['file', 'Quit', 'quit'],
    ['view', 'Results', 'results'], ['view', 'Code context', 'context'],
    ['view', 'Console', 'console'], ['view', 'Planning', 'planning'],
    ['view', 'Ask difflow', 'assistant'], ['view', 'Port names', 'portLabels'],
    ['view', 'Dark theme', 'dark'], ['help', 'Classic editor', 'classic'],
  ]) {
    seen.length = 0
    row(bar, id, label).run()
    assert.equal(seen[0]?.[0], name, label)
  }
})

test('Save is disabled when there is no file to save to', () => {
  assert.equal(row(full(), 'file', 'Save').disabled, false)
  assert.equal(row(full({ path: '' }), 'file', 'Save').disabled, true)
  // ...and it says so where the hint goes, rather than greying out mutely.
  assert.match(row(full({ path: '' }), 'file', 'Save').hint, /without a file/)
})

test('busy disables everything that would edit or write', () => {
  const bar = full({ busy: true })
  for (const label of ['Save', 'Reload', 'Python script', 'Diagram PNG']) {
    assert.equal(row(bar, 'file', label).disabled, true, label)
  }
  // Opening a drawer is not an edit, and is the one thing still worth
  // doing while a solve is running.
  assert.ok(!row(bar, 'view', 'Results').disabled)
})

test('the four exports are offered, and only with a flowsheet', () => {
  assert.deepEqual(labels(full(), 'file'),
    ['Save', 'Reload', 'Python script', 'Flowsheet JSON', 'Diagram SVG',
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
  assert.equal(bar.length, 3)
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
