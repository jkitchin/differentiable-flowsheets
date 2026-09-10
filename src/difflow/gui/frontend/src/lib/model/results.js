/**
 * A solve, and a sensitivity run, as tables and geometry.
 *
 * Pure functions over the JSON the server sends: no Svelte, no DOM. The
 * arithmetic here is the part that can be quietly wrong -- a mole
 * fraction that does not sum to one, a tornado bar drawn on the wrong
 * side of zero, a stale total left on an edge -- and none of it shows
 * up in a screenshot, so it is kept where `node --test` can reach it.
 */

/** Species keys of a stream, in the flowsheet's own order where given. */
export function speciesOf(stream, order) {
  const present = Object.keys(stream).filter((k) => k.startsWith('F_'))
  if (!order || !order.length) return present.sort()
  const known = order.map((s) => `F_${s}`).filter((k) => k in stream)
  const extra = present.filter((k) => !known.includes(k)).sort()
  return [...known, ...extra]
}

/** Total molar flow of a stream. */
export function total(stream) {
  let sum = 0
  for (const [k, v] of Object.entries(stream)) {
    if (k.startsWith('F_') && Number.isFinite(v)) sum += v
  }
  return sum
}

/**
 * The stream table: one row per stream, columns fixed across all of them.
 *
 * Columns are the union over every stream rather than the first one's,
 * because a flowsheet whose units carry different species -- a separator
 * that drops one, a reactor that makes one -- would otherwise show a
 * table missing the column that matters.
 *
 * @param {Object} solve the `/api/solve` answer
 * @returns {{species: string[], rows: Array}}
 */
export function streamTable(solve) {
  const streams = solve?.streams ?? {}
  const order = solve?.species ?? []
  const seen = new Set()
  for (const stream of Object.values(streams)) {
    for (const k of speciesOf(stream, order)) seen.add(k)
  }
  const species = speciesOf(
    Object.fromEntries([...seen].map((k) => [k, 0])), order,
  ).map((k) => k.slice(2))

  const rows = Object.entries(streams).map(([name, stream]) => {
    const F = total(stream)
    return {
      name,
      T: stream.T ?? null,
      P: stream.P ?? null,
      total: F,
      phase: typeof stream.phase === 'string' ? stream.phase : null,
      flows: species.map((s) => stream[`F_${s}`] ?? null),
      // A zero-flow stream has no composition, and dividing by its total
      // would fill the row with NaN rather than say so.
      fractions: species.map((s) =>
        F ? (stream[`F_${s}`] ?? 0) / F : null),
    }
  })
  return { species, rows }
}

/** Total flow per stream, for badging the canvas edges. */
export function flowLabels(solve) {
  const out = {}
  for (const [name, stream] of Object.entries(solve?.streams ?? {})) {
    out[name] = total(stream)
  }
  return out
}

/**
 * Edge tints from a forward sensitivity run, in [-1, 1].
 *
 * Scaled against the largest response anywhere in the flowsheet, so the
 * colour says "compared with everything else this lever moves", which is
 * the only comparison that means anything across streams in different
 * units. Relative response (`d ln F / d ln u`) is used where it exists;
 * a stream at zero flow has none and is left untinted rather than shown
 * as unresponsive, which is a different claim.
 */
export function flowTints(sens) {
  const raw = {}
  for (const [name, row] of Object.entries(sens?.streams ?? {})) {
    const cell = row.total_flow
    if (cell && Number.isFinite(cell.rel)) raw[name] = cell.rel
  }
  const peak = Math.max(...Object.values(raw).map(Math.abs), 0)
  if (!peak) return {}
  const out = {}
  for (const [name, v] of Object.entries(raw)) out[name] = v / peak
  return out
}

/**
 * A tornado chart of a reverse run: bars from a centre line, ranked.
 *
 * Geometry only -- x, y, width, height and which side of zero -- so the
 * component draws rectangles and decides nothing.
 *
 * @param {Array} levers the `levers` array of a reverse answer
 * @param {{width?: number, row?: number, gap?: number}} [box]
 */
export function tornado(levers, { width = 320, row = 18, gap = 4 } = {}) {
  const usable = (levers ?? []).filter((l) => Number.isFinite(l.rel))
  const peak = Math.max(...usable.map((l) => Math.abs(l.rel)), 0)
  const mid = width / 2
  const bars = usable.map((l, i) => {
    const span = peak ? (Math.abs(l.rel) / peak) * mid : 0
    return {
      key: l.key,
      rel: l.rel,
      d: l.d,
      units: l.units ?? null,
      y: i * (row + gap),
      height: row,
      x: l.rel < 0 ? mid - span : mid,
      width: span,
      negative: l.rel < 0,
    }
  })
  return {
    bars,
    mid,
    width,
    height: bars.length ? bars.length * (row + gap) - gap : 0,
    peak,
    // Levers whose base value or output is zero have no relative
    // sensitivity to plot; they are named rather than dropped.
    unscaled: (levers ?? []).filter((l) => !Number.isFinite(l.rel))
                            .map((l) => l.key),
  }
}

/**
 * A number as a reader wants it: four significant figures, or an
 * exponent once that stops being readable.
 *
 * 101325 becomes 1.013e5 rather than 101325.0000: a pressure is not
 * known to a tenth of a pascal here, and the digits imply it is.
 */
export function fmt(value, digits = 4) {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  if (!Number.isFinite(value)) return String(value)
  if (value === 0) return '0'
  const size = Math.abs(value)
  if (size >= 1e-3 && size < 1e5) {
    return String(Number(value.toPrecision(digits)))
  }
  return value.toExponential(digits - 1)
}

/**
 * Put the solve onto the canvas edges: a flow on the label, a tint on
 * the line.
 *
 * The tint is expressed as a class plus a `--tint` custom property on
 * the path rather than as a colour computed here, so the two palettes --
 * `app.css` and this file -- cannot drift apart. Sign picks the class
 * (a stream that goes *down* when the lever goes up is a different fact
 * from one that goes up), magnitude sets the property, and the
 * stylesheet decides what either looks like.
 *
 * @param {Array} edges  edges from `toGraph`
 * @param {{flows?: Object, tints?: Object}} data
 */
export function decorate(edges, { flows = null, tints = null } = {}) {
  if (!flows && !tints) return edges
  return edges.map((edge) => {
    const stream = edge.data?.stream
    const flow = flows ? flows[stream] : undefined
    const tint = tints ? tints[stream] : undefined
    const next = { ...edge }
    if (flow !== undefined) next.label = `${edge.label}  ${fmt(flow, 3)}`
    if (tint !== undefined && tint !== 0) {
      next.class = `${edge.class} tinted ${tint > 0 ? 'up' : 'down'}`.trim()
      next.style = `--tint: ${Math.abs(tint).toFixed(3)}`
    }
    return next
  })
}
