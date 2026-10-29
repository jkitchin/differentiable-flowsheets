import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  changesAt, describeFrame, describeWhatIf, plotScale, whatIfFields, eventFrames, formatNumber, iterationAt,
  residualPath, residualSeries, stateAt, stepFrom, unitAt,
} from './trace.js'

const TRACE = {
  protocol: 'difflow-trace/1',
  units: ['M1', 'R1'],
  tears: ['recycle'],
  frames: [
    { checkpoint: 'solve_start', phase: 'start', iteration: null, unit: null,
      residual: null, set: { feed: { F_A: 1, T: 300 } } },
    { checkpoint: 'after_unit', phase: 'solve', iteration: 0, unit: 'M1',
      residual: null, set: { mixed: { F_A: 1.5, T: 300 } } },
    { checkpoint: 'iteration', phase: 'solve', iteration: 0, unit: null,
      residual: 0.5, set: { recycle: { F_A: 0.5 } } },
    { checkpoint: 'after_unit', phase: 'solve', iteration: 1, unit: 'M1',
      residual: null, set: { mixed: { F_A: 1.75, T: 300 } } },
    { checkpoint: 'iteration', phase: 'solve', iteration: 1, unit: null,
      residual: NaN, trigger: 'nonfinite', set: {} },
  ],
}

test('state replays the frames up to and including i', () => {
  assert.deepEqual(stateAt(TRACE, 1), { feed: { F_A: 1, T: 300 }, mixed: { F_A: 1.5, T: 300 } })
  assert.equal(stateAt(TRACE, 3).mixed.F_A, 1.75)
  assert.deepEqual(stateAt(null, 3), {})
})

test('changes leave out fields that did not move', () => {
  assert.deepEqual(changesAt(TRACE, 3), [{ stream: 'mixed', field: 'F_A', before: 1.5, after: 1.75 }])
  assert.equal(changesAt(TRACE, 1).length, 2)          // a new stream: every field
})

test('stepping by kind skips the frames it does not keep', () => {
  assert.equal(stepFrom(TRACE, 0, 1, 'iteration'), 2)
  assert.equal(stepFrom(TRACE, 2, 1, 'iteration'), 4)
  assert.equal(stepFrom(TRACE, 4, 1, 'iteration'), 4)    // none further: stay
  assert.equal(stepFrom(TRACE, 4, -1, 'unit'), 3)
  assert.equal(stepFrom(TRACE, 0, 1, 'event'), 4)
})

test('where the solve is', () => {
  assert.equal(unitAt(TRACE, 1), 'M1')
  assert.equal(unitAt(TRACE, 2), null)
  assert.equal(iterationAt(TRACE, 0), null)
  assert.equal(iterationAt(TRACE, 3), 1)
  assert.match(describeFrame(TRACE.frames[1]), /after M1 · iter 0/)
  assert.match(describeFrame(TRACE.frames[4]), /nonfinite/)
  assert.deepEqual(eventFrames(TRACE).map((e) => e.frame), [4])
})

test('the residual plot keeps a NaN, on the top edge', () => {
  const series = residualSeries(TRACE)
  assert.equal(series.length, 2)
  const { points } = residualPath(series, 100, 50)
  assert.equal(points[1].ok, false)
  assert.equal(points[1].y, 4)
  assert.deepEqual(residualPath([], 100, 50).points, [])
  const exact = residualPath([{ frame: 0, iteration: 0, residual: 1 },
                              { frame: 1, iteration: 1, residual: 0 }], 100, 50).points
  assert.equal(exact[1].ok, true)
  assert.equal(exact[1].y, 46)                         // zero: the bottom edge
})

test('numbers', () => {
  assert.equal(formatNumber(null), '–')
  assert.equal(formatNumber(0.5), '0.5')
  assert.equal(formatNumber(1e-8), '1.000e-8')
  assert.equal(formatNumber(NaN), 'NaN')
})

test('two series share one scale', () => {
  const a = [{ residual: 1 }, { residual: 1e-2 }]
  const b = [{ residual: 10 }, { residual: 1 }, { residual: 1e-4 }]
  assert.deepEqual(plotScale(a, b), { lo: -4, hi: 1, n: 2 })
  const s = plotScale(a, b)
  // the same residual lands at the same height in either series
  assert.equal(residualPath(a, 100, 50, 4, s).points[0].y,
               residualPath(b, 100, 50, 4, s).points[1].y)
})

test('a what-if starts at a tear iteration or an accelerated guess', () => {
  assert.deepEqual(whatIfFields(TRACE, 1), null)                      // after a unit
  const withTear = { frames: [{ checkpoint: 'iteration',
    tear: [{ stream: 'recycle', field: 'F_A', x: 0.4, g: 0.5, delta: 0.1 }] }] }
  assert.deepEqual(whatIfFields(withTear, 0), [{ stream: 'recycle', field: 'F_A', value: 0.5 }])
  const accel = { frames: [{ checkpoint: 'after_accel', set: { recycle: { F_A: 0.7 } } }] }
  assert.deepEqual(whatIfFields(accel, 0), [{ stream: 'recycle', field: 'F_A', value: 0.7 }])
})

test('a what-if says what it changed', () => {
  const t = { result: { edits: [{ stream: 'recycle', field: 'F_A', was: 0.5, value: 3,
                                  iteration: 2, checkpoint: 'iteration' }] } }
  assert.equal(describeWhatIf(t), 'recycle.F_A: 0.5 → 3 at iter 2 (tear iteration)')
  assert.equal(describeWhatIf({ result: {} }), '')
})
