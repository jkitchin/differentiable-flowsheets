import assert from 'node:assert/strict'
import { test } from 'node:test'

import { clientId, keepAlive } from './lifetime.js'

/** A `setInterval`/`clearInterval` pair whose clock we advance by hand. */
function fakeTimers() {
  let next = 1
  const live = new Map()
  return {
    setInterval: (fn, ms) => (live.set(next, { fn, ms }), next++),
    clearInterval: (id) => live.delete(id),
    fire: () => [...live.values()].forEach(({ fn }) => fn()),
    get size() { return live.size },
    get every() { return [...live.values()][0]?.ms },
  }
}

/** A stand-in for `api.post` that records what it was asked to send. */
function recorder(answer = () => Promise.resolve({ ok: true })) {
  const calls = []
  const post = (...args) => (calls.push(args), answer())
  return { post, calls }
}

test('two pages get two different client ids', () => {
  assert.notEqual(clientId(), clientId())
})

test('a ping goes out at once rather than after the first interval', () => {
  const { post, calls } = recorder()
  keepAlive({ post, timers: fakeTimers() }).start()
  assert.equal(calls.length, 1)
  assert.equal(calls[0][0], '/api/ping')
})

test('every ping carries the same client id', () => {
  const { post, calls } = recorder()
  const timers = fakeTimers()
  keepAlive({ post, interval: 15, client: 'tab-a', timers }).start()
  timers.fire()
  timers.fire()
  assert.equal(calls.length, 3)
  for (const [path, payload] of calls) {
    assert.equal(path, '/api/ping')
    assert.deepEqual(payload, { client: 'tab-a' })
  }
  assert.equal(timers.every, 15000)
})

test('starting twice starts one interval, and stopping ends it', () => {
  const timers = fakeTimers()
  const alive = keepAlive({ post: () => Promise.resolve({}), timers })
  alive.start()
  alive.start()
  assert.equal(timers.size, 1)
  alive.stop()
  assert.equal(timers.size, 0)
})

test('the farewell survives the page, because it is sent with keepalive', () => {
  const { post, calls } = recorder()
  const alive = keepAlive({ post, client: 'tab-a', timers: fakeTimers() })
  assert.equal(alive.farewell({ persisted: false }), true)
  assert.deepEqual(calls[0], ['/api/bye', { client: 'tab-a' }, { keepalive: true }])
})

test('a page going into the back/forward cache says nothing', () => {
  const { post, calls } = recorder()
  const alive = keepAlive({ post, timers: fakeTimers() })
  assert.equal(alive.farewell({ persisted: true }), false)
  assert.equal(calls.length, 0)
})

test('a failed ping is swallowed rather than left as a loose rejection', async () => {
  const alive = keepAlive({
    post: () => Promise.reject(new Error('server gone')),
    timers: fakeTimers(),
  })
  assert.equal(await alive.ping(), undefined)
})

test('the default timers survive a browser that checks its receiver', () => {
  // `setInterval` is defined on Window and a real browser enforces it:
  // called with anything else as `this` it throws "Illegal invocation".
  // Node's is a plain function and jsdom's is too, so this is the only
  // place below an actual browser where the difference shows up --- and
  // it is worth a test, because the throw lands in App's mount effect,
  // where Svelte answers it by rendering nothing whatsoever.
  const real = { set: globalThis.setInterval, clear: globalThis.clearInterval }
  const brandCheck = (fn) =>
    function (...args) {
      if (this !== globalThis) throw new TypeError('Illegal invocation')
      return fn.apply(globalThis, args)
    }
  globalThis.setInterval = brandCheck(real.set)
  globalThis.clearInterval = brandCheck(real.clear)
  try {
    const { post } = recorder()
    const alive = keepAlive({ post })     // no timers: the browser's own
    alive.start()                          // this is what went blank
    alive.stop()
  } finally {
    globalThis.setInterval = real.set
    globalThis.clearInterval = real.clear
  }
})
