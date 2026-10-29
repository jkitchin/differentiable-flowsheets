import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  absorb, banner, describeFinished, resumes, resuming, eventBreak, focusOf, initial, keepWaiting,
  parseBreakpoints, unitBreak,
} from './live.js'

const PAUSE = { event: 'pause', checkpoint: 'after_unit', unit: 'R1', iteration: 2,
                phase: 'solve', residual: null, reason: 'step' }

test('a start lands paused at its first event', () => {
  const live = absorb(initial(), { ok: true, state: 'paused',
    event: { ...PAUSE, checkpoint: 'solve_start', unit: null, iteration: null, reason: 'start' } })
  assert.equal(live.status, 'paused')
  assert.equal(live.pause.checkpoint, 'solve_start')
  assert.equal(focusOf(live), null)
})

test('a resume carries the next pause under next', () => {
  const reply = { ok: true, state: 'paused', event: {
    event: 'result', ok: true, output: ['stepping to the next unit'], resumed: true, next: PAUSE } }
  const live = absorb(initial(), reply, 'next')
  assert.equal(live.status, 'paused')
  assert.equal(focusOf(live), 'R1')
  assert.deepEqual(live.log.map((e) => e.kind), ['command', 'output', 'pause'])
})

test('running means wait again, and clears the pause', () => {
  const reply = { ok: true, state: 'running', event: {
    event: 'result', ok: true, output: ['continuing'], resumed: true,
    next: { event: 'running' } } }
  assert.ok(keepWaiting(reply))
  const live = absorb({ ...initial(), status: 'paused', pause: PAUSE }, reply, 'continue')
  assert.equal(live.status, 'running')
  assert.equal(live.pause, null)
  assert.equal(focusOf(live), null)
  assert.ok(!keepWaiting({ ok: true, state: 'paused' }))
  assert.ok(!keepWaiting({ ok: false, error: 'x' }))
})

test('the end and a failure are said', () => {
  let live = absorb(initial(), { ok: true, state: 'finished',
    event: { event: 'finished', ok: true, converged: true, iterations: 4 } })
  assert.equal(live.status, 'finished')
  assert.match(live.log.at(-1).text, /converged in 4/)
  live = absorb(live, { ok: false, error: 'no live debugger: start one' }, 'step')
  assert.equal(live.log.at(-1).kind, 'error')
  assert.equal(describeFinished({ ok: false, stopped: true }), 'stopped from the debugger')
})

test('banners', () => {
  assert.equal(banner(PAUSE), '@after_unit · R1 · iter 2  ↳ step')
  assert.match(banner({ checkpoint: 'terminated', converged: false, iteration: 9 }), /NOT converged/)
  assert.match(banner({ checkpoint: 'unit_error', unit: 'R1', error: 'RuntimeError: x' }), /✗ RuntimeError/)
})

test('breakpoints read back from the server, and the commands that set them', () => {
  assert.deepEqual(parseBreakpoints(['unit R1', 'on nonfinite', 'iteration 3']),
                   { units: ['R1'], events: ['nonfinite'], other: ['iteration 3'] })
  assert.deepEqual(parseBreakpoints(['no breakpoints']), { units: [], events: [], other: [] })
  assert.equal(unitBreak('R1', true), 'break unit R1')
  assert.equal(unitBreak('R1', false), 'break del R1')
  assert.equal(eventBreak('nonfinite', true), 'break on nonfinite')
})

test('the transcript is bounded', () => {
  let live = initial()
  for (let i = 0; i < 500; i++) live = absorb(live, { ok: true, state: 'paused', event: PAUSE })
  assert.equal(live.log.length, 400)
})

test('a resume shows running at once, before its reply', () => {
  assert.ok(resumes('continue') && resumes(' n ') && resumes('run 5') && resumes('Q'))
  assert.ok(!resumes('p tear') && !resumes('break R1') && !resumes('set tear r.T 1'))
  const live = resuming({ ...initial(), status: 'paused', pause: PAUSE })
  assert.equal(live.status, 'running')
  assert.equal(focusOf(live), null)
})
