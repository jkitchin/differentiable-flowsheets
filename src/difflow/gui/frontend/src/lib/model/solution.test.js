import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import { decorate, fmt } from './results.js'
import {
  OTHER_COLOR,
  RAMP,
  SPECIES_COLORS,
  colorOptions,
  colorScale,
  flowWidths,
  rampAt,
  speciesColors,
  streamSummary,
  unitSummary,
  validColorBy,
  valueOf,
} from './solution.js'

const close = (a, b, eps = 1e-12) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`)

const SOLVE = {
  species: ['water', 'ethanol'],
  streams: {
    feed: { F_water: 1.0, F_ethanol: 1.0, T: 362, P: 101325 },
    vapor: { F_water: 0.25, F_ethanol: 0.75, T: 362, P: 101325 },
    liquid: { F_water: 0.75, F_ethanol: 0.25, T: 350, P: 101325 },
    empty: { F_water: 0, F_ethanol: 0, T: 300, P: 101325 },
  },
}

test('a summary states T, P and a composition that sums to one', () => {
  const s = streamSummary(SOLVE.streams.vapor, SOLVE.species)
  assert.equal(s.T, 362)
  assert.equal(s.P, 101325)
  close(s.total, 1.0)
  assert.deepEqual(s.rows.map((r) => r.species), ['water', 'ethanol'])
  close(s.rows.reduce((a, r) => a + r.x, 0), 1)
  close(s.parts.reduce((a, p) => a + p.x, 0), 1)
})

test('a stream with no flow has no composition, not NaN', () => {
  const s = streamSummary(SOLVE.streams.empty, SOLVE.species)
  assert.equal(s.total, 0)
  assert.ok(s.rows.every((r) => r.x === null))
  assert.deepEqual(s.parts, [])
})

test('a species absent from a stream is listed, but has no segment', () => {
  const s = streamSummary({ F_water: 1, F_ethanol: 0, T: 300, P: 1e5 }, SOLVE.species)
  assert.deepEqual(s.rows.map((r) => r.x), [1, 0])
  assert.deepEqual(s.parts.map((p) => p.species), ['water'])
})

test('a species keeps its colour whatever its rank in the stream', () => {
  const a = streamSummary(SOLVE.streams.vapor, SOLVE.species)
  const b = streamSummary(SOLVE.streams.liquid, SOLVE.species)
  assert.equal(a.rows[0].color, b.rows[0].color)
  assert.equal(a.rows[0].color, SPECIES_COLORS.light[0])
  assert.equal(streamSummary(SOLVE.streams.vapor, SOLVE.species, 'dark').rows[0].color,
               SPECIES_COLORS.dark[0])
})

test('past eight species the rest fold into one grey "other"', () => {
  const order = Array.from({ length: 10 }, (_, i) => `s${i}`)
  const stream = Object.fromEntries(order.map((s) => [`F_${s}`, 1]))
  const s = streamSummary(stream, order)
  assert.equal(s.parts.length, 9)
  assert.equal(s.parts.at(-1).species, 'other')
  close(s.parts.at(-1).x, 0.2)
  assert.equal(speciesColors(order).s9, OTHER_COLOR.light)
  // The row list still names every species.
  assert.equal(s.rows.length, 10)
})

test('widths are linear in flow, and a dry wire keeps the minimum', () => {
  const { widths, peak } = flowWidths(SOLVE, { min: 1, max: 9 })
  close(peak, 2)
  close(widths.feed, 9)
  close(widths.vapor, 5)
  close(widths.empty, 1)
})

test('with no flow anywhere every wire is the minimum', () => {
  const { widths } = flowWidths({ streams: { a: { F_x: 0 } } }, { min: 2 })
  assert.equal(widths.a, 2)
})

test('colour options are T, P, flow and each species', () => {
  assert.deepEqual(colorOptions(SOLVE).map((o) => o.key),
                   ['T', 'P', 'F', 'x:water', 'x:ethanol'])
})

test('a colour-by key the solve has no row for falls back to temperature', () => {
  assert.equal(validColorBy(SOLVE, 'x:ethanol'), 'x:ethanol')
  assert.equal(validColorBy(SOLVE, 'x:acetone'), 'T')
  assert.equal(validColorBy(SOLVE, ''), '')
  assert.equal(validColorBy(null, 'x:acetone'), 'x:acetone')
})

test('values: pressure in kPa, and a fraction only where there is flow', () => {
  close(valueOf(SOLVE.streams.feed, 'P'), 101.325)
  close(valueOf(SOLVE.streams.vapor, 'x:ethanol'), 0.75)
  assert.equal(valueOf(SOLVE.streams.empty, 'x:water'), null)
  assert.equal(valueOf(SOLVE.streams.feed, 'nonsense'), null)
})

test('the ramp runs end to end, and is monotone in lightness', () => {
  assert.equal(rampAt(0), RAMP.light[0])
  assert.equal(rampAt(1), RAMP.light.at(-1))
  assert.equal(rampAt(-5), RAMP.light[0])
  const lum = (hex) => [1, 3, 5].reduce((a, i) => a + parseInt(hex.slice(i, i + 2), 16), 0)
  const light = [0, 0.25, 0.5, 0.75, 1].map((t) => lum(rampAt(t)))
  assert.ok(light.every((v, i) => i === 0 || v < light[i - 1]), 'light: darker with t')
  const dark = [0, 0.25, 0.5, 0.75, 1].map((t) => lum(rampAt(t, 'dark')))
  assert.ok(dark.every((v, i) => i === 0 || v > dark[i - 1]), 'dark: lighter with t')
})

test('the colour scale spans the streams own range', () => {
  const { colors, lo, hi } = colorScale(SOLVE, 'T')
  assert.equal(lo, 300)
  assert.equal(hi, 362)
  assert.equal(colors.feed, RAMP.light.at(-1))
  assert.equal(colors.empty, RAMP.light[0])
})

test('a variable with one value everywhere takes the middle, not an end', () => {
  const { colors, lo, hi, uniform } = colorScale(SOLVE, 'P')
  assert.equal(lo, hi)
  assert.ok(uniform)
  assert.ok(Object.values(colors).every((c) => c === rampAt(0.5)))
})

test('round-off is not a range', () => {
  const solve = { streams: { a: { F_x: 1, T: 350 }, b: { F_x: 1, T: 350 + 1e-11 } } }
  const { colors, uniform } = colorScale(solve, 'T')
  assert.ok(uniform)
  assert.equal(colors.a, colors.b)
  assert.ok(!colorScale(SOLVE, 'T').uniform)
})

test('a stream with no value for the variable is left uncoloured', () => {
  const { colors } = colorScale(SOLVE, 'x:water')
  assert.ok(!('empty' in colors))
  assert.ok('feed' in colors)
})

test('decorate carries width and colour as custom properties', () => {
  const edges = [{ id: 'e', class: 'recycle', label: 'vapor', data: { stream: 'vapor' } }]
  const [e] = decorate(edges, { widths: { vapor: 4 }, colors: { vapor: '#123456' } })
  assert.equal(e.class, 'recycle sized colored')
  assert.match(e.style, /--w: 4\.00px/)
  assert.match(e.style, /--c: #123456/)
})

test('a coloured wire drops the sensitivity tint: one colour, one meaning', () => {
  const edges = [{ id: 'e', label: 'vapor', data: { stream: 'vapor' } }]
  const [e] = decorate(edges, { tints: { vapor: 0.5 }, colors: { vapor: '#123456' } })
  assert.ok(!e.class.includes('tinted'))
  const [t] = decorate(edges, { tints: { vapor: 0.5 } })
  assert.ok(t.class.includes('tinted'))
})

const FLASH = { name: 'flash', operation: 'Flash', inlets: ['feed'], outlets: ['liquid', 'vapor'],
                params: { species_order: ['water', 'ethanol'], split: [0.2, 0.8], V: 2, fn: { $callable: {} },
                          big: [1, 2, 3, 4, 5], label: 'x', bad: NaN } }

test('a unit summary lists its streams and each outlet share', () => {
  const u = unitSummary({ ...FLASH, outlets: ['vapor', 'liquid'] }, SOLVE)
  assert.deepEqual(u.inlets.map((r) => r.stream), ['feed'])
  close(u.totalIn, 2)
  close(u.totalOut, 2)
  assert.deepEqual(u.outlets.map((r) => r.share), [0.5, 0.5])
})

test('the change is out minus in per species, and a balanced unit nets zero', () => {
  const u = unitSummary(FLASH, SOLVE)
  assert.deepEqual(u.change.map((c) => c.species), ['water', 'ethanol'])
  close(u.change[0].in, 1)
  close(u.change[0].out, 1)
  close(u.change[0].delta, 0)
})

test('a reactor shows what it made', () => {
  const solve = { species: ['A', 'B'], streams: {
    in: { F_A: 1, F_B: 0, T: 300, P: 1e5 }, out: { F_A: 0.4, F_B: 0.6, T: 300, P: 1e5 } } }
  const u = unitSummary({ name: 'r', inlets: ['in'], outlets: ['out'], params: {} }, solve)
  close(u.change[0].delta, -0.6)
  close(u.change[1].delta, 0.6)
})

test('only numbers and short numeric lists count as parameters', () => {
  const u = unitSummary(FLASH, SOLVE)
  assert.deepEqual(u.params.map((p) => p.key), ['split', 'V'])
})

test('per-call arguments count as parameters too', () => {
  const u = unitSummary({ ...FLASH, params: {}, extra_params: { split_frac: 0.7 } }, SOLVE)
  assert.deepEqual(u.params, [{ key: 'split_frac', value: 0.7 }])
})

test('a stream missing from the solve is listed without numbers', () => {
  const u = unitSummary({ name: 'u', inlets: ['nowhere'], outlets: ['feed'], params: {} }, SOLVE)
  assert.equal(u.inlets[0].total, null)
  close(u.totalIn, 0)
  assert.equal(unitSummary(null, SOLVE), null)
  assert.equal(unitSummary(FLASH, null), null)
})

test('a species called "other" does not share a key with the fold', () => {
  const order = ['other', ...Array.from({ length: 9 }, (_, i) => `s${i}`)]
  const stream = Object.fromEntries(order.map((s) => [`F_${s}`, 1]))
  const s = streamSummary(stream, order)
  assert.equal(s.parts.length, 9)
  assert.equal(new Set(s.parts.map((p) => p.key)).size, s.parts.length)
})

test('a stream missing a species does not shift the others\' colours', () => {
  const order = ['A', 'B', 'C']
  const s = streamSummary({ F_B: 1, F_C: 1 }, order)
  assert.equal(s.rows[0].color, SPECIES_COLORS.light[1])
  assert.equal(s.rows[1].color, SPECIES_COLORS.light[2])
})

test('the fold goes by flowsheet order, not by rank in the stream', () => {
  const order = Array.from({ length: 10 }, (_, i) => `s${i}`)
  // s0 absent: s8 is still the ninth species and folds.
  const stream = Object.fromEntries(order.slice(1).map((s) => [`F_${s}`, 1]))
  const s = streamSummary(stream, order)
  assert.ok(!s.parts.some((p) => p.species === 's8'))
  assert.equal(s.parts.at(-1).key, 'other')
})

test('a negative flow leaves the stream without a composition', () => {
  const s = streamSummary({ T: 300, P: 1e5, F_A: 2, F_B: -1 }, ['A', 'B'])
  assert.equal(s.negative, true)
  assert.deepEqual(s.parts, [], 'no bar showing A at 200 %')
  assert.deepEqual(s.rows.map((r) => r.x), [null, null])
  assert.deepEqual(s.rows.map((r) => r.flow), [2, -1], 'the flows are still shown')
  assert.equal(streamSummary({ F_A: 1, F_B: 0 }, ['A', 'B']).negative, false)
  assert.equal(valueOf({ F_A: 2, F_B: -1 }, 'x:A'), null, 'nor a wire colour by it')
})

test('a span past the float limit still spreads the ramp', () => {
  const solve = { streams: { a: { T: -1e308 }, b: { T: 0 }, c: { T: 1e308 } } }
  const scale = colorScale(solve, 'T')
  assert.equal(scale.uniform, false)
  assert.equal(new Set(Object.values(scale.colors)).size, 3)
  assert.equal(scale.colors.b, colorScale({ streams: { a: { T: -1 }, b: { T: 0 }, c: { T: 1 } } }, 'T').colors.b)
})

test('a NaN is shown as NaN on the cards, as in the stream table', () => {
  const s = streamSummary({ T: NaN, P: 1e5, F_A: NaN, F_B: 1 }, ['A', 'B'])
  assert.ok(Number.isNaN(s.T))
  assert.ok(Number.isNaN(s.rows[0].flow), 'not null, which prints as a blank')
  assert.ok(Number.isNaN(s.total))
  assert.equal(fmt(s.rows[0].flow), 'NaN')
})
