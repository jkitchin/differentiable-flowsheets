/**
 * `node --test src/lib/model/*.test.js`
 *
 * A delta vector is read by a planning system that never sees this
 * panel, so the tests here are about the parts that would be wrong
 * silently: a column attributed to the wrong lever, a bound pair that is
 * not a range, and a structurally dead column rendered as a small number
 * instead of a zero.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  DEFAULT_RADIUS, FORMATS, blocked, cell, cleanBounds, jacobian,
  leverGroups, outputGroups, ranked, toggle, verdict,
} from './planning.js'

const LEVERS = [
  { key: 'reactor.V', owner: 'reactor', field: 'V', value: 1, units: 'm^3', kind: 'unit' },
  { key: 'reactor.T_damping', owner: 'reactor', field: 'T_damping', value: 0.3, units: '-', kind: 'unit' },
  { key: 'feed:feed.total_flow', owner: 'feed', field: 'total_flow', value: 1.1, units: 'mol/s', kind: 'feed' },
]

const VECTOR = {
  block: 'flowsheet',
  u_names: ['flowsheet.reactor_V', 'flowsheet.feed_feed_total_flow'],
  y_names: ['flowsheet.vap_total_flow', 'flowsheet.liq_F_ethanol'],
  u_units: ['m^3', 'mol/s'],
  y_units: ['mol/s', 'mol/s'],
  u0: [1, 1.1],
  y0: [0.4, 0.1],
  lb: [0, 0],
  ub: [2, 2.2],
  J: [[6.78e-21, -5.88e-21], [0.25, 0.3125]],
  mode: 'rev',
  radius: 0.3,
}

test('the formats offered are the two the server can write', () => {
  assert.deepEqual(FORMATS.map((f) => f.id), ['json', 'csv'])
  assert.equal(DEFAULT_RADIUS, 0.3)
})

test('feed levers are grouped first, because they are picked first', () => {
  const groups = leverGroups(LEVERS)
  assert.deepEqual(groups.map((g) => g.name), ['feed feed', 'reactor'])
  assert.equal(groups[1].items.length, 2)
  assert.deepEqual(leverGroups(null), [])
})

test('outputs are grouped by stream in the order given', () => {
  const groups = outputGroups([
    { key: 'vap.T', stream: 'vap', quantity: 'T' },
    { key: 'liq.T', stream: 'liq', quantity: 'T' },
    { key: 'vap.P', stream: 'vap', quantity: 'P' },
  ])
  assert.deepEqual(groups.map((g) => g.name), ['vap', 'liq'])
  assert.deepEqual(groups[0].items.map((i) => i.key), ['vap.T', 'vap.P'])
})

test('picking keeps the order picked, since it is the column order', () => {
  let keys = []
  keys = toggle(keys, 'feed:feed.total_flow')
  keys = toggle(keys, 'reactor.V')
  assert.deepEqual(keys, ['feed:feed.total_flow', 'reactor.V'])
  assert.deepEqual(toggle(keys, 'reactor.V'), ['feed:feed.total_flow'])
})

test('a blank bound is absent, not zero', () => {
  const clean = cleanBounds(
    { 'reactor.V': { lb: '', ub: '2' },
      'reactor.T_damping': { lb: '', ub: '' },
      'unpicked.k': { lb: '1', ub: '2' } },
    ['reactor.V', 'reactor.T_damping'])
  assert.deepEqual(clean, { 'reactor.V': { ub: 2 } })
})

test('a lower bound of zero survives -- it is a bound, not a blank', () => {
  assert.deepEqual(cleanBounds({ 'reactor.V': { lb: '0', ub: '2' } }, ['reactor.V']),
                   { 'reactor.V': { lb: 0, ub: 2 } })
})

test('what blocks a linearization is said, in the order it is fixed', () => {
  assert.match(blocked({ solved: false }), /solve the flowsheet first/)
  assert.match(blocked({ solved: true }), /at least one lever/)
  assert.match(blocked({ solved: true, u: ['reactor.V'] }), /at least one output/)
  assert.equal(blocked({ solved: true, u: ['reactor.V'], y: ['vap.T'] }), '')
})

test('bounds that are not a range block the export', () => {
  const ask = { solved: true, u: ['reactor.V'], y: ['vap.T'] }
  assert.match(blocked({ ...ask, bounds: { 'reactor.V': { lb: '3', ub: '1' } } }),
               /reactor\.V are not a range/)
  assert.match(blocked({ ...ask, bounds: { 'reactor.V': { lb: 'x', ub: '1' } } }),
               /not a range/)
  assert.equal(blocked({ ...ask, bounds: { 'reactor.V': { lb: '0', ub: '2' } } }), '')
})

test('the block prefix is trimmed off every name, and only that', () => {
  const j = jacobian(VECTOR)
  assert.deepEqual(j.levers.map((l) => l.name), ['reactor_V', 'feed_feed_total_flow'])
  assert.deepEqual(j.rows.map((r) => r.name), ['vap_total_flow', 'liq_F_ethanol'])
  assert.deepEqual(j.levers[0], { name: 'reactor_V', units: 'm^3', u0: 1, lb: 0, ub: 2 })
  assert.deepEqual(j.rows[1].cells, [0.25, 0.3125])
  assert.equal(jacobian(null), null)
})

test('a dead column reads as a zero, not as a small number', () => {
  assert.equal(cell(6.78e-21), '0')
  assert.equal(cell(0), '0')
  assert.equal(cell(0.3125), '0.3125')
  assert.equal(cell(1234.5678), '1235')
  assert.equal(cell(1.5e-8), '1.500e-8')
  assert.equal(cell(null), '--')
})

test('findings come worst-first', () => {
  const sorted = ranked([{ severity: 'info' }, { severity: 'error' },
                         { severity: 'warning' }])
  assert.deepEqual(sorted.map((f) => f.severity), ['error', 'warning', 'info'])
})

test('the finite-difference check reports its number either way', () => {
  assert.equal(verdict(null), '')
  assert.match(verdict({ passed: true, max_rel_error: 1e-10 }), /1.0e-10 relative/)
  const bad = verdict({ passed: false, max_rel_error: 0.2 })
  assert.match(bad, /2.0e-1/)
  assert.match(bad, /do not\s+export/)
})
