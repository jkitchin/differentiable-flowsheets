import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import { SOLVER_DEFAULTS, solverOptions, solverValue } from './solver.js'

test('the defaults are filled in under what the file stores', () => {
  assert.deepEqual(solverOptions(null), SOLVER_DEFAULTS)
  assert.deepEqual(solverOptions({ solver: { max_iter: 7 } }),
                   { ...SOLVER_DEFAULTS, max_iter: 7 })
})

test('a blank field is sent as null, so the default comes back', () => {
  assert.equal(solverValue('tol', '  '), null)
  assert.equal(solverValue('tol', '1e-6'), 1e-6)
  assert.equal(solverValue('max_iter', '250'), 250)
})

test('text that is not a number is sent as text, for the server to refuse', () => {
  assert.equal(solverValue('tol', 'tight'), 'tight')
})

test('the choices pass through', () => {
  assert.equal(solverValue('acceleration', 'wegstein'), 'wegstein')
  assert.equal(solverValue('clip_negative_flows', false), false)
})
