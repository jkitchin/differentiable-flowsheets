import assert from 'node:assert/strict'
import test from 'node:test'

import { decorate, flowLabels, flowTints, fmt, speciesOf, streamTable,
         tornado, total } from './results.js'

const SOLVE = {
  ok: true,
  species: ['water', 'ethanol'],
  streams: {
    feed: { F_water: 1.0, F_ethanol: 0.1, T: 350, P: 101325 },
    liq: { F_water: 0.5, F_ethanol: 0.6, T: 350, P: 101325, phase: 'liquid' },
    vap: { F_water: 0.0, F_ethanol: 0.0, T: 350, P: 101325 },
  },
}

test('species follow the flowsheet order, not the dict order', () => {
  // The stream literal puts ethanol second; `species_order` puts it second
  // too, and a sorted list would put it first. The table must follow the
  // model, since that is the order every other difflow report uses.
  assert.deepEqual(speciesOf(SOLVE.streams.liq, ['water', 'ethanol']),
                   ['F_water', 'F_ethanol'])
  assert.deepEqual(speciesOf(SOLVE.streams.liq, []),
                   ['F_ethanol', 'F_water'])
})

test('a species the order does not mention is still shown', () => {
  const stream = { F_a: 1, F_zz: 2, T: 300 }
  assert.deepEqual(speciesOf(stream, ['zz']), ['F_zz', 'F_a'])
})

test('total is the sum of the species flows and nothing else', () => {
  assert.equal(total(SOLVE.streams.feed), 1.1)
  assert.equal(total({ T: 350, P: 101325 }), 0)
})

test('the table has a column per species across every stream', () => {
  const table = streamTable({
    species: ['a'],
    streams: { one: { F_a: 1 }, two: { F_a: 1, F_b: 2 } },
  })
  assert.deepEqual(table.species, ['a', 'b'])
  assert.deepEqual(table.rows[0].flows, [1, null])
})

test('fractions of a zero-flow stream are absent, not NaN', () => {
  const table = streamTable(SOLVE)
  const vap = table.rows.find((r) => r.name === 'vap')
  assert.deepEqual(vap.fractions, [null, null])
  const liq = table.rows.find((r) => r.name === 'liq')
  assert.deepEqual(liq.fractions.map((x) => Number(x.toFixed(4))),
                   [0.4545, 0.5455])
  assert.equal(liq.phase, 'liquid')
})

test('edge labels carry each stream total', () => {
  assert.deepEqual(flowLabels(SOLVE), { feed: 1.1, liq: 1.1, vap: 0 })
})

test('tints are scaled against the largest response in the flowsheet', () => {
  const tints = flowTints({
    streams: {
      a: { total_flow: { value: 1, d: 0.5, rel: 0.5 } },
      b: { total_flow: { value: 1, d: -1, rel: -1 } },
      c: { total_flow: { value: 0, d: 0, rel: null } },
    },
  })
  assert.deepEqual(tints, { a: 0.5, b: -1 })
})

test('a flowsheet nothing responds in gets no tints at all', () => {
  assert.deepEqual(flowTints({ streams: { a: { total_flow: { rel: 0 } } } }), {})
  assert.deepEqual(flowTints(null), {})
})

test('the tornado puts negative bars left of the centre', () => {
  const chart = tornado(
    [{ key: 'a', rel: 2, d: 1 }, { key: 'b', rel: -1, d: -1 },
     { key: 'c', rel: null, d: 0 }],
    { width: 100, row: 10, gap: 0 },
  )
  assert.equal(chart.mid, 50)
  assert.deepEqual(chart.bars.map((b) => [b.x, b.width, b.negative]),
                   [[50, 50, false], [25, 25, true]])
  assert.deepEqual(chart.unscaled, ['c'])
  assert.equal(chart.height, 20)
})

test('an empty tornado has no height and does not divide by zero', () => {
  const chart = tornado([])
  assert.equal(chart.height, 0)
  assert.equal(chart.peak, 0)
  assert.deepEqual(chart.bars, [])
})

test('numbers are readable rather than complete', () => {
  assert.equal(fmt(101325), '1.013e+5')
  assert.equal(fmt(0.5936974270923778), '0.5937')
  assert.equal(fmt(0), '0')
  assert.equal(fmt(1.0), '1')
  assert.equal(fmt(2.5e-9), '2.500e-9')
  assert.equal(fmt(null), '')
  assert.equal(fmt('liquid'), 'liquid')
  assert.equal(fmt(Infinity), 'Infinity')
})

test('decoration adds the flow to the label and a signed tint class', () => {
  const edges = [
    { id: 'a', label: 'liq', class: '', data: { stream: 'liq' } },
    { id: 'b', label: 'vap → rec', class: 'recycle', data: { stream: 'vap' } },
    { id: 'c', label: 'other', class: '', data: { stream: 'other' } },
  ]
  const out = decorate(edges, {
    flows: { liq: 1.1, vap: 0.25 },
    tints: { liq: -0.5, vap: 1 },
  })
  assert.equal(out[0].label, 'liq  1.1')
  assert.equal(out[0].class, 'tinted down')
  assert.equal(out[0].style, '--tint: 0.500')
  assert.equal(out[1].class, 'recycle tinted up')
  // A stream the run says nothing about keeps exactly what it had.
  assert.deepEqual(out[2], edges[2])
})

test('decoration with nothing to say returns the edges themselves', () => {
  const edges = [{ id: 'a', label: 'liq', class: '', data: { stream: 'liq' } }]
  assert.equal(decorate(edges, {}), edges)
})
