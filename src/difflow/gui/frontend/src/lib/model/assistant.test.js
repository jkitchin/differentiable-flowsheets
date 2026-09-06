/**
 * `node --test src/lib/model/*.test.js`
 *
 * The panel is allowed to be wrong about a lot of things; it is not
 * allowed to be wrong about *where the question went*. So the provider
 * table, the destination sentence and the request the panel builds are
 * tested here, and the component stays a view of them.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  DEFAULTS, KINDS, PROVIDERS, contextPath, destination, inferKind,
  messages, provider, readSettings, writeSettings,
} from './assistant.js'

test('the kinds offered are the ones the server can assemble', () => {
  assert.deepEqual(KINDS.map((k) => k.id),
                   ['flowsheet', 'block', 'solve', 'planning'])
  for (const kind of KINDS) assert.ok(kind.label && kind.hint, kind.id)
})

test('every provider says whether it leaves the machine', () => {
  for (const p of PROVIDERS) {
    assert.equal(typeof p.local, 'boolean', p.id)
    assert.ok(p.hint.length > 20, p.id)
  }
  assert.deepEqual(PROVIDERS.filter((p) => !p.local).map((p) => p.id), ['server'])
})

test('an unknown provider falls back to the one that sends nothing', () => {
  assert.equal(provider('nonsense').id, 'none')
  assert.equal(provider(undefined).id, 'none')
  assert.equal(DEFAULTS.provider, 'none')
})

test('a selected unit is what the question is about', () => {
  assert.equal(inferKind({ unit: { name: 'reactor' } }), 'block')
  // even when the solve also failed: the click is the more recent signal
  assert.equal(inferKind({ unit: { name: 'r' }, solve: { ok: false } }), 'block')
})

test('with nothing selected, a failed solve is the likely question', () => {
  assert.equal(inferKind({ solve: { ok: false, error: 'boom' } }), 'solve')
  assert.equal(inferKind({ solve: { ok: true } }), 'flowsheet')
  assert.equal(inferKind({}), 'flowsheet')
  assert.equal(inferKind(), 'flowsheet')
})

test('the context request carries only what was asked for', () => {
  assert.equal(contextPath({ kind: 'flowsheet' }), '/api/context?kind=flowsheet')
  assert.equal(
    contextPath({ kind: 'block', name: 'rx 1', question: 'what is V?' }),
    '/api/context?kind=block&q=what+is+V%3F&name=rx+1',
  )
  assert.equal(contextPath({ kind: 'block', operation: 'CSTR' }),
               '/api/context?kind=block&operation=CSTR')
})

test('the system prompt is the server\'s, not the page\'s', () => {
  const pack = { system: 'answer only from the brief', prompt: '## Unit\n...' }
  assert.deepEqual(messages(pack), [
    { role: 'system', content: 'answer only from the brief' },
    { role: 'user', content: '## Unit\n...' },
  ])
})

test('the destination sentence names the actual endpoint', () => {
  assert.match(destination({ ...DEFAULTS }), /Nothing is sent/)
  assert.match(destination({ ...DEFAULTS, provider: 'webllm' }),
               /this browser by Llama-3\.2-3B/)
  assert.match(
    destination({ ...DEFAULTS, provider: 'openai', baseUrl: 'http://x/v1',
                  openaiModel: 'qwen' }),
    /Sent to http:\/\/x\/v1 as qwen/,
  )
  assert.match(destination({ ...DEFAULTS, provider: 'server' }), /Anthropic API/)
})

test('settings round-trip, and a broken store is not fatal', () => {
  const store = new Map()
  const storage = {
    getItem: (k) => store.get(k) ?? null,
    setItem: (k, v) => store.set(k, v),
  }
  writeSettings(storage, { ...DEFAULTS, provider: 'openai' })
  assert.equal(readSettings(storage).provider, 'openai')
  // a setting added since the last write still arrives
  assert.equal(readSettings(storage).temperature, DEFAULTS.temperature)

  const broken = {
    getItem: () => { throw new Error('denied') },
    setItem: () => { throw new Error('denied') },
  }
  assert.deepEqual(readSettings(broken), DEFAULTS)
  writeSettings(broken, DEFAULTS)          // must not throw
  assert.deepEqual(readSettings(null), DEFAULTS)

  store.set('difflow.assistant', '{not json')
  assert.deepEqual(readSettings(storage), DEFAULTS)
})
