/**
 * `node --test src/lib/model/*.test.js`
 *
 * A parameter field that quietly turns `1.0` into `"1.0"`, or that
 * offers an input for a JAX array and posts `[object Object]` back,
 * breaks the model without looking broken. So the classification and
 * the parsing are tested here and the component stays thin.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { classify, describe, label, parse, shape, tagOf } from './params.js'

test('a plain value is editable, by its own type', () => {
  assert.deepEqual(classify(1.0), { kind: 'number', editable: true, text: '1' })
  assert.equal(classify('vapor').kind, 'text')
  assert.equal(classify(true).kind, 'boolean')
})

test('a tagged value is not something an input can edit', () => {
  for (const value of [
    { $array: [1, 2] },
    { $callable: { factory: 'mass_action_kinetics' } },
    { $ref: 'thermo' },
    { $thermo: 'ideal' },
    { $dataclass: 'Antoine' },
  ]) {
    assert.equal(classify(value).editable, false, JSON.stringify(value))
  }
})

test('a tagged value says where it comes from', () => {
  assert.equal(describe({ $ref: 'thermo' }), 'thermo (code context)')
  assert.equal(
    describe({ $callable: { factory: 'mass_action_kinetics' } }),
    'mass_action_kinetics(...)',
  )
  assert.equal(describe({ $array: [[1, 2], [3, 4]] }), 'array 2×2')
  assert.equal(describe({ $array: 3 }), 'array scalar')
})

test('the tag is found wherever it sits in the object', () => {
  assert.equal(tagOf({ $ref: 'x' }), '$ref')
  assert.equal(tagOf({ V: 1 }), null)
  assert.equal(tagOf([1, 2]), null, 'a list is not a tagged object')
  assert.equal(tagOf(null), null)
})

test('an unset field is typed from its declaration, not from its value', () => {
  assert.equal(classify(null, { type: 'float | None' }).kind, 'number')
  assert.equal(classify(null, { type: 'str | None' }).kind, 'text')
  assert.equal(classify(null, { type: 'jax.Array | None' }).kind, 'number')
  const fn = classify(null, { type: 'Callable | None', is_callable: true })
  assert.equal(fn.editable, false, 'a rate law is not typed into a box')
})

test('a flat list is editable as text, a nested one is not', () => {
  assert.deepEqual(classify(['water', 'ethanol']), {
    kind: 'list', editable: true, text: 'water, ethanol',
  })
  assert.equal(classify([[1, 2], [3, 4]]).editable, false)
})

test('a list of numbers does not come back as a list of strings', () => {
  assert.deepEqual(parse('1, 2.5, 3', 'list'), [1, 2.5, 3])
  assert.deepEqual(parse('water, ethanol', 'list'), ['water', 'ethanol'])
  assert.deepEqual(parse('', 'list'), [])
})

test('an emptied field is cleared, not set to zero', () => {
  assert.equal(parse('   ', 'number'), null)
  assert.equal(parse('', 'text'), null)
})

test('a non-finite float survives as the string the server restores', () => {
  // K_eq is `inf` for every irreversible reaction, so this is the
  // common case rather than an exotic one.
  assert.equal(parse('Infinity', 'number'), 'Infinity')
  assert.equal(parse('1e6', 'number'), 1e6)
})

test('a number field that is given a word keeps the word', () => {
  // The server refuses it by name; silently posting NaN would not.
  assert.equal(parse('warm', 'number'), 'warm')
})

test('the shape of a nested array is read outermost first', () => {
  assert.deepEqual(shape([[1, 2, 3], [4, 5, 6]]), [2, 3])
  assert.deepEqual(shape(7), [])
})

test('a label carries the units when the catalog knows them', () => {
  assert.equal(label({ name: 'V', units: 'm^3' }), 'V (m^3)')
  assert.equal(label({ name: 'rate_fn' }), 'rate_fn')
})
