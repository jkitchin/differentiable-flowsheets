/**
 * A solve, drawn onto the canvas: what a wire says when it is hovered,
 * how wide it is, and what colour.
 *
 * The stream table in the results drawer has every number, but it is a
 * table beside the drawing, and reading it means finding the row for the
 * wire you were looking at. These put the same numbers on the wire.
 *
 * Pure functions over the `/api/solve` answer, like `results.js`, and
 * for the same reason: a mole fraction that does not sum to one or a
 * colour scale pinned to the wrong end is invisible in a screenshot and
 * obvious to `node --test`.
 */

import { speciesOf, total } from './results.js'

/**
 * Species colours, in the fixed categorical order the rest of difflow's
 * palette starts from (`--series`, then `--accent`). Validated for colour
 * vision deficiency on both of the editor's surfaces. A ninth species is
 * never a generated hue: it folds into "other".
 */
export const SPECIES_COLORS = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
}

/** The grey "other" takes, when there are more species than colours. */
export const OTHER_COLOR = { light: '#8f8d87', dark: '#7b7a83' }

/**
 * One hue, light to dark, for a wire coloured by magnitude. The light end
 * still clears 2:1 on its surface, because a wire at the bottom of the
 * range must not vanish. On the dark surface the ramp runs the other way,
 * so "more" is always "more contrast".
 */
export const RAMP = {
  light: ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b'],
  dark: ['#184f95', '#256abf', '#3987e5', '#6da7ec', '#b7d3f6'],
}

/**
 * The colour each species wears, fixed by its place in the flowsheet's
 * species order and never by rank in one stream: water is the same blue
 * on every wire whether it is the most of a stream or a trace.
 */
export function speciesColors(order, theme = 'light') {
  const colors = SPECIES_COLORS[theme] ?? SPECIES_COLORS.light
  const out = {}
  ;(order ?? []).forEach((s, i) => {
    out[s] = i < colors.length ? colors[i] : (OTHER_COLOR[theme] ?? OTHER_COLOR.light)
  })
  return out
}

/** Whether any species flow on the stream is negative; card and wire agree. */
function hasNegativeFlow(stream) {
  return Object.keys(stream).some((k) => k.startsWith('F_') && stream[k] < 0)
}

/**
 * Everything the hover card says about one stream.
 *
 * `parts` is the composition bar: one segment per species with a nonzero
 * flow, in species order, and species past the eighth folded into a
 * single "other" so the bar never needs a colour nobody validated.
 * `rows` lists every species, zeros included --- a stream that carries
 * none of something is a fact worth reading, not an absence to hide.
 *
 * A negative flow (a signed tear, or an iteration stopped part way)
 * leaves the stream without a composition: the fractions would run past
 * 100 % and below 0, and a bar drawn from the positive ones would show
 * a mixture that is not there. `negative` says so, and every `x` is null.
 *
 * @returns {null | {T, P, phase, total, negative, rows, parts}}
 */
export function streamSummary(stream, order = [], theme = 'light') {
  if (!stream) return null
  const keys = speciesOf(stream, order)
  const names = keys.map((k) => k.slice(2))
  const F = total(stream)
  // Colour and fold by place in the FLOWSHEET's order, not this stream's:
  // a stream that lacks the first species must not shift every other
  // species' colour down one. Species the order does not name go last.
  const full = [...(order ?? []), ...names.filter((s) => !(order ?? []).includes(s))]
  const rank = Object.fromEntries(full.map((s, i) => [s, i]))
  const colors = speciesColors(full, theme)
  const negative = hasNegativeFlow(stream)
  const rows = names.map((s) => {
    const flow = stream[`F_${s}`]
    return {
      species: s,
      // NaN stays NaN: the card prints it, as the stream table does. A
      // blank would read as a value nobody computed, not as a failure.
      flow: typeof flow === 'number' ? flow : null,
      // No composition for a stream with no flow, rather than NaN.
      x: F > 0 && !negative && Number.isFinite(flow) ? flow / F : null,
      color: colors[s],
    }
  })

  const limit = (SPECIES_COLORS[theme] ?? SPECIES_COLORS.light).length
  const parts = []
  let other = 0
  rows.forEach((r) => {
    if (!(r.x > 0)) return
    if (rank[r.species] < limit) parts.push({ key: `F_${r.species}`, species: r.species, x: r.x, color: r.color })
    else other += r.x
  })
  // Keyed apart from the species: a species may itself be called
  // "other", and two parts under one key throw in a keyed each.
  if (other > 0) {
    parts.push({ key: 'other', species: 'other', x: other,
                 color: OTHER_COLOR[theme] ?? OTHER_COLOR.light })
  }

  return {
    T: typeof stream.T === 'number' ? stream.T : null,
    P: typeof stream.P === 'number' ? stream.P : null,
    phase: typeof stream.phase === 'string' ? stream.phase : null,
    total: F,
    negative,
    rows,
    parts,
  }
}

/**
 * Wire widths proportional to total molar flow, `{stream: px}`.
 *
 * Linear, so a wire twice as wide carries twice as much --- the one
 * reading a width invites. Scaled to the largest flow on the flowsheet,
 * and never thinner than `min`: a stream at zero flow is still a pipe,
 * and a wire that disappears reads as a wire that was deleted.
 */
export function flowWidths(solve, { min = 1.2, max = 9 } = {}) {
  const flows = {}
  for (const [name, stream] of Object.entries(solve?.streams ?? {})) {
    flows[name] = total(stream)
  }
  const peak = Math.max(0, ...Object.values(flows).filter(Number.isFinite))
  const out = {}
  for (const [name, F] of Object.entries(flows)) {
    out[name] = peak > 0 && F > 0 ? min + (max - min) * (F / peak) : min
  }
  return { widths: out, peak }
}

/**
 * What a wire can be coloured by, for this solve: temperature, pressure,
 * total flow, and each species' mole fraction.
 */
export function colorOptions(solve) {
  const order = solve?.species ?? []
  const seen = new Set()
  for (const stream of Object.values(solve?.streams ?? {})) {
    for (const k of speciesOf(stream, order)) seen.add(k.slice(2))
  }
  const species = speciesOf(Object.fromEntries([...seen].map((s) => [`F_${s}`, 0])), order)
    .map((k) => k.slice(2))
  return [
    { key: 'T', label: 'Temperature', units: 'K' },
    { key: 'P', label: 'Pressure', units: 'kPa' },
    { key: 'F', label: 'Total flow', units: 'mol/s' },
    ...species.map((s) => ({ key: `x:${s}`, label: `x ${s}`, units: 'mole fraction' })),
  ]
}

/**
 * The colour-by key to use for `solve`, given the one chosen earlier.
 *
 * A species key outlives the flowsheet it was chosen on: open another
 * file and `x:ethanol` names nothing, the legend's select shows its first
 * row (Temperature) while the wires are coloured by nothing at all. A key
 * the solve has no row for falls back to temperature, which every solve
 * has; no key, or no solve yet, is left as it is.
 */
export function validColorBy(solve, key) {
  if (!key || !solve) return key
  return colorOptions(solve).some((o) => o.key === key) ? key : 'T'
}

/** One stream's value of a colour-by key, or null where it has none. */
export function valueOf(stream, key) {
  if (!stream) return null
  if (key === 'T') return Number.isFinite(stream.T) ? stream.T : null
  if (key === 'P') return Number.isFinite(stream.P) ? stream.P / 1000 : null
  if (key === 'F') return total(stream)
  if (key?.startsWith('x:')) {
    const F = total(stream)
    const flow = stream[`F_${key.slice(2)}`] ?? 0
    // as on the card: a stream with a negative flow has no composition
    return F > 0 && !hasNegativeFlow(stream) && Number.isFinite(flow) ? flow / F : null
  }
  return null
}

/** Linear interpolation between two hex colours, t in [0, 1]. */
function mix(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16))
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16))
  return '#' + pa.map((v, i) =>
    Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, '0')).join('')
}

/** The ramp's colour at t in [0, 1]. */
export function rampAt(t, theme = 'light') {
  const ramp = RAMP[theme] ?? RAMP.light
  const u = Math.min(1, Math.max(0, t)) * (ramp.length - 1)
  const i = Math.min(ramp.length - 2, Math.floor(u))
  return mix(ramp[i], ramp[i + 1], u - i)
}

/**
 * Wire colours for one variable, plus the range the legend states.
 *
 * The scale runs over the streams' own range, not from zero: the point is
 * to see which wire is hotter, and a 300-to-360 K flowsheet drawn on a
 * 0-to-360 scale is five shades of the same dark blue. When every stream
 * has the same value there is nothing to rank, and every wire takes the
 * middle of the ramp rather than one end, which would claim an extreme.
 * "The same" is judged relative to the size of the values: a solve hands
 * back 350 and 350.00000000001 for streams that are both at 350 K, and a
 * scale stretched across that difference would rank round-off.
 * A stream with no value (no flow, for a mole fraction) is left uncoloured.
 */
export function colorScale(solve, key, theme = 'light') {
  const values = {}
  for (const [name, stream] of Object.entries(solve?.streams ?? {})) {
    const v = valueOf(stream, key)
    if (v !== null && Number.isFinite(v)) values[name] = v
  }
  const all = Object.values(values)
  if (!all.length) return { colors: {}, lo: null, hi: null, uniform: false }
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  // Halved before subtracting: two finite values of opposite sign near
  // the float limit have a span of Infinity, and every stream would then
  // sit at the bottom of the ramp.
  const span = hi / 2 - lo / 2
  const uniform = span <= 0.5e-9 * Math.max(Math.abs(lo), Math.abs(hi), 1e-300)
  const colors = {}
  for (const [name, v] of Object.entries(values)) {
    colors[name] = rampAt(uniform ? 0.5 : (v / 2 - lo / 2) / span, theme)
  }
  return { colors, lo, hi, uniform }
}

/** A number worth showing as a unit parameter: finite, or a short list of them. */
function shownParam(value) {
  if (typeof value === 'number') return Number.isFinite(value)
  return Array.isArray(value) && value.length > 0 && value.length <= 4 &&
    value.every((v) => typeof v === 'number' && Number.isFinite(v))
}

/**
 * Everything the hover card says about one unit, from the streams around
 * it: what goes in, what comes out and in what shares, how each species
 * changed on the way through, and the unit's own numeric parameters.
 *
 * Nothing here is the unit's internal result --- a heater's duty, a
 * flash's K values --- because the solve does not report those. What it
 * does report is enough to read most units anyway: a reactor's product
 * shows as a positive change, a separator as its outlet shares.
 *
 * `change` is out minus in per species, in mol/s. It is a molar balance,
 * so for a reaction that changes the number of moles it does not sum to
 * zero, and it is not meant to.
 *
 * @param unit  the unit as the served document has it
 * @returns {null | {inlets, outlets, totalIn, totalOut, change, params}}
 */
export function unitSummary(unit, solve, theme = 'light') {
  if (!unit || !solve?.streams) return null
  const streams = solve.streams
  const order = solve.species ?? []
  const side = (names) => (names ?? []).map((name) => {
    const s = streams[name]
    return {
      stream: name,
      T: typeof s?.T === 'number' ? s.T : null,
      P: typeof s?.P === 'number' ? s.P : null,
      total: s ? total(s) : null,
    }
  })
  const inlets = side(unit.inlets)
  const outlets = side(unit.outlets)
  const sum = (rows) => rows.reduce((a, r) => a + (r.total ?? 0), 0)
  const totalIn = sum(inlets)
  const totalOut = sum(outlets)
  // Each outlet's share of what leaves --- a flash's vapor fraction, a
  // splitter's split --- and none where nothing leaves at all.
  for (const r of outlets) r.share = totalOut > 0 && r.total !== null ? r.total / totalOut : null

  const seen = new Set()
  for (const name of [...(unit.inlets ?? []), ...(unit.outlets ?? [])]) {
    if (streams[name]) for (const k of speciesOf(streams[name], order)) seen.add(k)
  }
  const names = speciesOf(Object.fromEntries([...seen].map((k) => [k, 0])), order)
    .map((k) => k.slice(2))
  const colors = speciesColors(names, theme)
  const flowOf = (list, s) => (list ?? []).reduce((a, n) => {
    const v = streams[n]?.[`F_${s}`]
    return a + (Number.isFinite(v) ? v : 0)
  }, 0)
  const change = names.map((s) => {
    const fin = flowOf(unit.inlets, s)
    const fout = flowOf(unit.outlets, s)
    return { species: s, in: fin, out: fout, delta: fout - fin, color: colors[s] }
  })

  // Both kinds: the Params object's fields, and the per-call arguments a
  // unit takes at solve time (`extra_params`: a splitter's split_frac).
  const params = Object.entries({ ...(unit.params ?? {}), ...(unit.extra_params ?? {}) })
    .filter(([, v]) => shownParam(v))
    .map(([key, value]) => ({ key, value }))

  return { inlets, outlets, totalIn, totalOut, change, params }
}
