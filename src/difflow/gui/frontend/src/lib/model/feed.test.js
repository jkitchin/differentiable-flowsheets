/**
 * `node --test src/lib/model/*.test.js`
 *
 * A feed is typed, not drawn, so every mistake in it is a number that
 * looks like a number. These are the checks a screenshot cannot make.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { feedEdit, feedFields, feedValues } from './feed.js'

const SPECIES = ['water', 'ethanol']
const DEFAULTS = { flow: 1.0, T: 298.15, P: 101325.0 }

test('a declared feed starts from what it carries', () => {
  const stream = { T: 320.0, P: 2e5, F_water: 3.0, F_ethanol: 0.5 }
  assert.deepEqual(feedValues(stream, SPECIES, DEFAULTS), {
    T: 320.0,
    P: 2e5,
    flows: { water: 3.0, ethanol: 0.5 },
  })
})

test('an undeclared feed starts from the flowsheet defaults', () => {
  assert.deepEqual(feedValues(null, SPECIES, DEFAULTS), {
    T: 298.15,
    P: 101325.0,
    flows: { water: 1.0, ethanol: 1.0 },
  })
})

test('a species added after the feed gets the default, not a hole', () => {
  const stream = { T: 300.0, P: 1e5, F_water: 2.0 }
  const values = feedValues(stream, ['water', 'ethanol'], DEFAULTS)
  assert.equal(values.flows.ethanol, 1.0)
  assert.equal(values.flows.water, 2.0)
})

test('with nothing to go on every field is still a number', () => {
  const values = feedValues(undefined, SPECIES, {})
  for (const field of feedFields(values, SPECIES)) {
    assert.ok(Number.isFinite(field.value), `${field.key} is ${field.value}`)
  }
})

test('the fields are T, P and one flow per species, in order', () => {
  const values = feedValues(null, SPECIES, DEFAULTS)
  assert.deepEqual(
    feedFields(values, SPECIES).map((f) => [f.key, f.units]),
    [['T', 'K'], ['P', 'Pa'], ['F_water', 'mol/s'], ['F_ethanol', 'mol/s']],
  )
})

test('an untouched form is not dirty and sends what is already there', () => {
  const values = feedValues({ T: 320.0, P: 2e5, F_water: 3.0, F_ethanol: 0.5 },
                            SPECIES, DEFAULTS)
  const { spec, dirty, errors } = feedEdit({}, values)
  assert.equal(dirty, false)
  assert.deepEqual(errors, [])
  assert.deepEqual(spec, { T: 320.0, P: 2e5, flows: { water: 3.0, ethanol: 0.5 } })
})

test('one edited field still sends the rest, so nothing is zeroed', () => {
  const values = feedValues({ T: 320.0, P: 2e5, F_water: 3.0, F_ethanol: 0.5 },
                            SPECIES, DEFAULTS)
  const { spec, dirty } = feedEdit({ T: '340' }, values)
  assert.equal(dirty, true)
  assert.deepEqual(spec, { T: 340.0, P: 2e5, flows: { water: 3.0, ethanol: 0.5 } })
})

test('retyping the same number is not a change', () => {
  const values = feedValues(null, SPECIES, DEFAULTS)
  assert.equal(feedEdit({ T: '298.15', F_water: '1' }, values).dirty, false)
})

test('a zero flow is a real flow', () => {
  const values = feedValues(null, SPECIES, DEFAULTS)
  const { spec, dirty, errors } = feedEdit({ F_ethanol: '0' }, values)
  assert.deepEqual(errors, [])
  assert.equal(dirty, true)
  assert.equal(spec.flows.ethanol, 0)
})

test('a blank box is a question, not a zero', () => {
  const values = feedValues(null, SPECIES, DEFAULTS)
  const { spec, dirty, errors } = feedEdit({ F_water: '  ' }, values)
  assert.deepEqual(errors, ['the water flow needs a number'])
  assert.equal(dirty, false)
  assert.equal(spec.flows.water, 1.0)
})

test('what is not a number says which field it is', () => {
  const values = feedValues(null, SPECIES, DEFAULTS)
  assert.deepEqual(feedEdit({ T: 'hot', P: 'x', F_ethanol: '' }, values).errors, [
    'temperature needs a number',
    'pressure needs a number',
    'the ethanol flow needs a number',
  ])
})

test('a draft for a species that is gone is ignored', () => {
  const values = feedValues(null, ['water'], DEFAULTS)
  const { spec, dirty } = feedEdit({ F_ethanol: '9' }, values)
  assert.equal(dirty, false)
  assert.deepEqual(spec.flows, { water: 1.0 })
})
