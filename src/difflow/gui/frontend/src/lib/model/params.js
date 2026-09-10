/**
 * What kind of thing a parameter holds, and how it goes back over the wire.
 *
 * A `Params` field is not always a number in a box. `serialize` writes
 * whatever JSON has no type for behind a tag -- `$array` for a JAX
 * array, `$callable` for a rate law built by `mass_action_kinetics`,
 * `$ref` for a name in the code context, `$dataclass` and `$thermo` for
 * an object -- and none of those is something a text input can edit.
 * The old editor said "set in code: rate_fn" in a line under the form,
 * which is the right idea and the wrong place: the field is there, it
 * simply is not typed into.
 *
 * So every field is classified, and the classification decides whether
 * it gets an input or a read-only line saying where its value comes
 * from. Pure functions with no imports, tested under bare node.
 */

/** serialize's tags, and the word the inspector uses for each. */
export const TAGGED = {
  $array: 'array',
  $callable: 'callable',
  $ref: 'reference',
  $dataclass: 'object',
  $namedtuple: 'object',
  $thermo: 'thermo',
}

const NUMERIC = /\b(float|int|Array|Scalar)\b/

/** The tag on a value, or null if it is plain. */
export function tagOf(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  for (const tag of Object.keys(TAGGED)) {
    if (Object.prototype.hasOwnProperty.call(value, tag)) return tag
  }
  return null
}

/**
 * Classify one parameter.
 *
 * @param value the value as the document holds it (already `restore`d).
 * @param spec the catalog's `ParameterSpec`, or undefined.
 * @returns `{kind, editable, text}` -- `text` is what to show when it
 *   is not editable, and is never the raw JSON of a whole array.
 */
export function classify(value, spec = {}) {
  const tag = tagOf(value)
  if (tag) return { kind: TAGGED[tag], editable: false, text: describe(value, tag) }

  if (value === null || value === undefined) {
    // Not set. The declared type decides which input it gets, so that
    // an unset float does not come back as the string "3".
    const kind = spec.is_callable ? 'callable'
      : NUMERIC.test(String(spec.type ?? '')) ? 'number' : 'text'
    return { kind, editable: !spec.is_callable, text: 'not set' }
  }
  if (typeof value === 'boolean') return { kind: 'boolean', editable: true, text: String(value) }
  if (typeof value === 'number') return { kind: 'number', editable: true, text: String(value) }
  if (typeof value === 'string') return { kind: 'text', editable: true, text: value }
  if (Array.isArray(value)) {
    const flat = value.every((v) => typeof v === 'number' || typeof v === 'string')
    return flat
      ? { kind: 'list', editable: true, text: value.join(', ') }
      : { kind: 'array', editable: false, text: describe({ $array: value }, '$array') }
  }
  return { kind: 'object', editable: false, text: `${Object.keys(value).length} entries` }
}

/** A one-line account of a tagged value: what it is, not what it contains. */
export function describe(value, tag = tagOf(value)) {
  const payload = value[tag]
  if (tag === '$ref') return `${payload} (code context)`
  if (tag === '$callable') {
    const factory = payload?.factory
    return factory ? `${factory}(...)` : 'a function'
  }
  if (tag === '$thermo') return typeof payload === 'string' ? payload : 'thermo'
  if (tag === '$dataclass' || tag === '$namedtuple') {
    return typeof payload === 'string' ? payload : 'object'
  }
  if (tag === '$array') return `array ${shape(payload).join('×') || 'scalar'}`
  return tag
}

/** The dimensions of a nested array, outermost first. */
export function shape(value) {
  const out = []
  let cursor = value
  while (Array.isArray(cursor)) {
    out.push(cursor.length)
    cursor = cursor[0]
  }
  return out
}

/**
 * Turn what was typed back into a value the server will accept.
 *
 * `isFinite`, not `!isNaN`: a field may legitimately hold `Infinity`
 * (`mass_action_kinetics` writes it into `K_eq` for every irreversible
 * reaction), and it travels as the string the server knows how to
 * restore rather than as a number JSON cannot write.
 */
export function parse(raw, kind) {
  const text = String(raw).trim()
  if (kind === 'boolean') return raw === true || text === 'true'
  if (kind === 'list') {
    if (text === '') return []
    const parts = text.split(',').map((p) => p.trim())
    const numbers = parts.map(Number)
    // a list of numbers must not arrive as a list of strings
    return numbers.every((n) => Number.isFinite(n)) ? numbers : parts
  }
  if (text === '') return null
  if (kind === 'number') {
    const n = Number(text)
    return Number.isFinite(n) ? n : text
  }
  return text
}

/** The label for a field: its name, its symbol and its units. */
export function label(spec) {
  const units = spec?.units ? ` (${spec.units})` : ''
  return `${spec?.name ?? ''}${units}`
}
