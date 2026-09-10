/**
 * `node --test src/lib/model/*.test.js`
 *
 * The console's rules, tested where they are: recalling a line must not
 * eat what was half-typed, and an output this page cannot draw must say
 * so rather than render as nothing.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  append, describe, entry, isBlank, LIMIT, prompt, recall, remember, touched,
} from './console.js'

test('a blank cell is not worth a round trip', () => {
  assert.ok(isBlank(''))
  assert.ok(isBlank('  \n\t '))
  assert.ok(isBlank(undefined))
  assert.ok(!isBlank('1'))
})

test('the echo carries a prompt, continuation lines included', () => {
  assert.equal(prompt('x = 1'), '>>> x = 1')
  assert.equal(prompt('def f():\n    return 1'),
               '>>> def f():\n...     return 1')
  // a trailing blank line keeps its prompt but gains no trailing space
  assert.equal(prompt('a\n'), '>>> a\n...')
})

test('a cell that printed and then failed keeps both halves', () => {
  const item = entry('print(1)\nboom', {
    outputs: [{ kind: 'text', text: '1\n' }],
    error: 'NameError: name \'boom\' is not defined',
  })
  assert.equal(item.outputs.length, 1)
  assert.match(item.error, /NameError/)
  assert.equal(item.changed, false)
})

test('the transcript drops the oldest rather than growing forever', () => {
  let log = []
  for (let i = 0; i < LIMIT + 10; i++) log = append(log, entry(`${i}`, {}))
  assert.equal(log.length, LIMIT)
  assert.equal(log[log.length - 1].source, String(LIMIT + 9))
})

test('history skips blanks and an immediate repeat', () => {
  let h = remember([], 'a')
  h = remember(h, 'a')
  h = remember(h, '   ')
  h = remember(h, 'b')
  assert.deepEqual(h, ['a', 'b'])
  // the same line again later is worth keeping: it is a re-run
  assert.deepEqual(remember(h, 'a'), ['a', 'b', 'a'])
})

test('arrowing up and back down returns the half-typed line', () => {
  const h = ['first', 'second']
  let s = { index: null, draft: 'half typ', stash: '' }
  s = { ...recall(h, s, 'up') }
  assert.equal(s.source, 'second')
  assert.equal(s.stash, 'half typ')
  s = { ...recall(h, { ...s, draft: s.source }, 'up') }
  assert.equal(s.source, 'first')
  s = { ...recall(h, { ...s, draft: s.source }, 'down') }
  assert.equal(s.source, 'second')
  s = { ...recall(h, { ...s, draft: s.source }, 'down') }
  assert.equal(s.index, null)
  assert.equal(s.source, 'half typ', 'the draft came back')
})

test('up stops at the oldest, down at the live line', () => {
  const h = ['only']
  let s = recall(h, { index: 0, draft: 'only', stash: '' }, 'up')
  assert.equal(s.index, 0)
  s = recall(h, { index: null, draft: 'live', stash: '' }, 'down')
  assert.equal(s.index, null)
  assert.equal(s.source, 'live')
})

test('with no history the line is left alone', () => {
  const s = recall([], { index: null, draft: 'typing', stash: '' }, 'up')
  assert.equal(s.source, 'typing')
  assert.equal(s.index, null)
})

test('text and stderr are told apart', () => {
  assert.deepEqual(describe({ kind: 'text', text: 'hi\n', stream: 'stdout' }),
                   { kind: 'text', stream: 'stdout', text: 'hi\n' })
  assert.equal(describe({ kind: 'text', text: 'bad', stream: 'stderr' }).stream,
               'stderr')
  assert.equal(describe({ kind: 'value', text: '42' }).kind, 'value')
})

test('an image is a data URI, so plots need no new wire format', () => {
  const drawn = describe({ kind: 'image', mime: 'image/png', data: 'AAAA' })
  assert.equal(drawn.kind, 'image')
  assert.equal(drawn.src, 'data:image/png;base64,AAAA')
  // png is the assumed default, as it is on the Python side
  assert.match(describe({ kind: 'image', data: 'AAAA' }).src, /^data:image\/png;/)
})

test('an output this page cannot draw says so rather than vanishing', () => {
  const unknown = describe({ kind: 'plotly', spec: {} })
  assert.equal(unknown.kind, 'unknown')
  assert.match(unknown.text, /plotly/)
  assert.match(describe({}).text, /cannot render/)
})

test('a cell that moved the model marks the transcript', () => {
  assert.ok(!touched([entry('1', { changed: false })]))
  assert.ok(touched([entry('1', {}), entry('fs...', { changed: true })]))
})
