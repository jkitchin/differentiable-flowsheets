/**
 * A feed stream, as a form and as a request.
 *
 * A feed is the one part of a flowsheet that cannot be built by
 * dropping and wiring: it is data --- a temperature, a pressure and a
 * flow per species --- and it has to be typed. Which makes it the one
 * part where a typo is invisible: `1e5` for a pressure and `1.e5` for a
 * flow both look like numbers, and a field left blank by accident is
 * indistinguishable from one left blank on purpose.
 *
 * So the arithmetic lives here rather than in the component: what the
 * boxes should show, whether anything actually changed, and what is not
 * a number yet. Pure functions over plain data, checked under bare node.
 */

/** The flow field for a species, as both the draft key and the wire name. */
export function flowKey(species) {
  return `F_${species}`
}

/**
 * The numbers a feed form starts from.
 *
 * An undeclared feed --- an inlet with nothing on the other end --- has
 * no stream to read, so it starts from the flowsheet's own defaults:
 * the same numbers `Flowsheet.solve` invents for a tear stream, which
 * makes an untouched feed not a new guess about the model. A species
 * added after the feed was declared is in the same position, and gets
 * the same answer rather than a hole.
 */
export function feedValues(stream, species = [], defaults = {}) {
  const pick = (value, fallback) => (Number.isFinite(value) ? value : fallback)
  const flow = pick(defaults.flow, 1)
  const held = stream && typeof stream === 'object' ? stream : null
  return {
    T: pick(held?.T, pick(defaults.T, 298.15)),
    P: pick(held?.P, pick(defaults.P, 101325)),
    flows: Object.fromEntries(
      species.map((s) => [s, pick(held?.[flowKey(s)], flow)]),
    ),
  }
}

/** The fields a feed form shows, in the order they are read. */
export function feedFields(values, species = []) {
  return [
    { key: 'T', label: 'T', units: 'K', value: values.T },
    { key: 'P', label: 'P', units: 'Pa', value: values.P },
    ...species.map((s) => ({
      key: flowKey(s),
      label: s,
      units: 'mol/s',
      value: values.flows[s],
    })),
  ]
}

/** What a field is called when something is wrong with it. */
function nameOf(key) {
  if (key === 'T') return 'temperature'
  if (key === 'P') return 'pressure'
  return `the ${key.slice(2)} flow`
}

/**
 * The request a part-typed form becomes: `{spec, dirty, errors}`.
 *
 * `spec` always carries every field, because `set_feed` fills a missing
 * one from the *existing* feed and a form that sent only what changed
 * would be indistinguishable from one that sent nothing. `dirty` says
 * whether sending it would change anything, so applying an untouched
 * form is a no-op rather than a round trip.
 *
 * A blank box is an error and not a zero. Zero is a real flow --- a
 * species absent from a feed --- and guessing at which one was meant
 * would put a number into the model that nobody typed.
 */
export function feedEdit(draft, values) {
  const spec = { T: values.T, P: values.P, flows: { ...values.flows } }
  const errors = []
  let dirty = false
  for (const [key, raw] of Object.entries(draft || {})) {
    const text = String(raw).trim()
    const value = text === '' ? NaN : Number(text)
    if (!Number.isFinite(value)) {
      errors.push(`${nameOf(key)} needs a number`)
      continue
    }
    if (key === 'T' || key === 'P') {
      if (value !== spec[key]) dirty = true
      spec[key] = value
    } else {
      const species = key.slice(2)
      if (!(species in spec.flows)) continue
      if (value !== spec.flows[species]) dirty = true
      spec.flows[species] = value
    }
  }
  return { spec, dirty, errors }
}
