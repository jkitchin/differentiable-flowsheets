import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  axisFrac, chart, coordsAt, fmt, gridAt, interp, middle, readout,
} from './sweep.js'

const AX = { key: 'reactor.V', label: 'Volume', units: 'm^3',
             lo: 0, hi: 4, n: 5, values: [0, 1, 2, 3, 4] }
const OUT = { name: 'Product', units: 'mol/s',
              values: [0, 10, 20, 30, 40],
              gradients: { 'reactor.V': [10, 10, 10, 10, 10] } }

test('numbers are formatted for a reader', () => {
  assert.equal(fmt(1.5), '1.500')
  assert.equal(fmt(123.456), '123.5')
  assert.equal(fmt(1e-9), '1.000e-9')
  assert.equal(fmt(NaN), '--')
  assert.equal(fmt(Infinity), '--')
})

test('a point between samples splits into index and fraction', () => {
  assert.deepEqual(axisFrac(AX, 2.5), [2, 0.5])
  assert.deepEqual(axisFrac(AX, 2), [2, 0])
})

test('the ends clamp rather than run off the grid', () => {
  // The last index must leave room for i+1, or interpolation reads undefined.
  assert.deepEqual(axisFrac(AX, 4), [3, 1])
  assert.deepEqual(axisFrac(AX, 99), [3, 1])
  assert.deepEqual(axisFrac(AX, -99), [0, 0])
})

test('interpolation is exact at the grid points', () => {
  for (let i = 0; i < AX.n; i++)
    assert.equal(interp([AX], OUT.values, [AX.values[i]]), OUT.values[i])
})

test('and linear between them', () => {
  assert.equal(interp([AX], OUT.values, [2.5]), 25)
})

test('a second axis is interpolated bilinearly', () => {
  const b = { ...AX, key: 'b', lo: 0, hi: 1, n: 2, values: [0, 1] }
  const grid = [[0, 10], [2, 12], [4, 14], [6, 16], [8, 18]]
  assert.equal(interp([AX, b], grid, [0, 0]), 0)
  assert.equal(interp([AX, b], grid, [4, 1]), 18)
  assert.equal(interp([AX, b], grid, [2, 0.5]), 9)
})

test('a cell is reached by its index path', () => {
  assert.equal(gridAt([[1, 2], [3, 4]], [1, 0]), 3)
})

test('the sliders start in the middle and name real values', () => {
  assert.deepEqual(middle([AX]), [2])
  assert.deepEqual(coordsAt([AX], [2]), [2])
})

test('the chart puts the curve inside its own box', () => {
  const c = chart([AX], OUT, [2])
  assert.ok(c.path.startsWith('M64 '))
  assert.equal(c.yTicks.length, 5)
  assert.equal(c.xTicks.length, 3)
  assert.equal(c.xLabel, 'Volume (m^3)')
  // The cursor sits where the slider does: the middle of the x range.
  assert.equal(c.cursor.x, (c.L + c.W - c.R) / 2)
  assert.ok(c.cursor.y > c.T && c.cursor.y < c.H - c.B)
})

test('a flat response still gets a band to draw in', () => {
  const flat = { ...OUT, values: [0, 0, 0, 0, 0], gradients: {} }
  const c = chart([AX], flat, [2])
  assert.ok(Number.isFinite(c.cursor.y))
  assert.ok(c.yTicks.every((t) => Number.isFinite(t.y)))
})

test('the readout carries the exact sensitivity, or says it has none', () => {
  const [row] = readout([AX], [OUT], [2.5])
  assert.equal(row.value, 25)
  assert.equal(row.sensitivity, 10)
  const [bare] = readout([AX], [{ ...OUT, gradients: {} }], [2.5])
  assert.equal(bare.sensitivity, null)
})
