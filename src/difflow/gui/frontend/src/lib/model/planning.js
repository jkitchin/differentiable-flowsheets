/**
 * The Planning panel's pure parts.
 *
 * A delta vector is a Jacobian a planning system will price decisions
 * against, so the two things that must not be quietly wrong are which
 * lever a column belongs to and whether the bounds a user typed are a
 * range at all. Both are decided here, where they can be tested without
 * a browser.
 */

/** Trust-region radius the panel starts at, matching the server's. */
export const DEFAULT_RADIUS = 0.3

/** What the panel can hand back as a file. */
export const FORMATS = [
  { id: 'json', label: 'JSON manifest', note: 'names, units, bounds, health' },
  { id: 'csv', label: 'CSV tables', note: 'one matrix per block, plus bases' },
]

/**
 * Levers grouped the way the picker shows them.
 *
 * Feeds first: a planning model's first lever is almost always feed rate,
 * and it is the one the flowsheet's own parameter list buries.
 */
export function leverGroups(levers) {
  const groups = new Map()
  for (const item of levers ?? []) {
    const key = item.kind === 'feed' ? `feed ${item.owner}` : item.owner
    if (!groups.has(key)) groups.set(key, { name: key, kind: item.kind, items: [] })
    groups.get(key).items.push(item)
  }
  const all = [...groups.values()]
  return [...all.filter((g) => g.kind === 'feed'),
          ...all.filter((g) => g.kind !== 'feed')]
}

/** Outputs grouped by their stream, in the order the server listed them. */
export function outputGroups(outputs) {
  const groups = new Map()
  for (const item of outputs ?? []) {
    if (!groups.has(item.stream)) groups.set(item.stream, { name: item.stream, items: [] })
    groups.get(item.stream).items.push(item)
  }
  return [...groups.values()]
}

/**
 * Add or drop one key, keeping the order it was picked in.
 *
 * Order is not cosmetic: it is the column order of the exported
 * Jacobian, and a user who picks feed rate first expects it first.
 */
export function toggle(keys, key) {
  return keys.includes(key) ? keys.filter((k) => k !== key) : [...keys, key]
}

/**
 * The bounds a lever will actually be exported with.
 *
 * A blank field is not zero -- it means "no bound given", and the server
 * fills a symmetric window around the base value rather than an infinity
 * that would make the trust region meaningless. So a blank stays absent
 * here, and only a pair that parses and orders is sent.
 */
export function cleanBounds(bounds, keys) {
  const out = {}
  for (const key of keys) {
    const window = bounds?.[key]
    if (!window) continue
    const lb = window.lb === '' || window.lb == null ? null : Number(window.lb)
    const ub = window.ub === '' || window.ub == null ? null : Number(window.ub)
    const entry = {}
    if (lb != null && Number.isFinite(lb)) entry.lb = lb
    if (ub != null && Number.isFinite(ub)) entry.ub = ub
    if (Object.keys(entry).length) out[key] = entry
  }
  return out
}

/**
 * Why the panel cannot linearize yet, or `''`.
 *
 * Stated as a sentence rather than by disabling a button with no
 * explanation, because "solve first" is the answer three quarters of the
 * time and a dead button does not say it.
 */
export function blocked({ solved = false, u = [], y = [], bounds = {} } = {}) {
  if (!solved) return 'solve the flowsheet first -- the outputs are named from its streams'
  if (!u.length) return 'pick at least one lever'
  if (!y.length) return 'pick at least one output'
  const bad = u.filter((key) => {
    const w = bounds?.[key]
    if (!w) return false
    const lb = w.lb === '' || w.lb == null ? null : Number(w.lb)
    const ub = w.ub === '' || w.ub == null ? null : Number(w.ub)
    if ((w.lb !== '' && w.lb != null && !Number.isFinite(lb))
        || (w.ub !== '' && w.ub != null && !Number.isFinite(ub))) return true
    return lb != null && ub != null && lb >= ub
  })
  if (bad.length) return `bounds on ${bad.join(', ')} are not a range`
  return ''
}

/**
 * The Jacobian as rows the panel can render.
 *
 * Names arrive block-qualified, and the block prefix is on every one of
 * them: it costs a third of the column width and distinguishes nothing.
 */
export function jacobian(vector) {
  if (!vector) return null
  const trim = (name) =>
    name.startsWith(`${vector.block}.`) ? name.slice(vector.block.length + 1) : name
  return {
    block: vector.block,
    mode: vector.mode,
    radius: vector.radius,
    levers: vector.u_names.map((name, j) => ({
      name: trim(name), units: vector.u_units?.[j] ?? null,
      u0: vector.u0[j], lb: vector.lb?.[j], ub: vector.ub?.[j],
    })),
    rows: vector.y_names.map((name, i) => ({
      name: trim(name), units: vector.y_units?.[i] ?? null,
      y0: vector.y0[i], cells: vector.J[i],
    })),
  }
}

/**
 * A number for a table cell.
 *
 * An exact zero is written `0` rather than `0.000e+0`: a structurally
 * dead column is the one thing in this table a reader must be able to
 * spot at a glance, and scientific notation hides it among the small
 * numbers. Anything under 1e-12 is that same zero arrived at by
 * arithmetic.
 */
export function cell(value) {
  if (value == null || value === '') return '--'
  const x = Number(value)
  if (!Number.isFinite(x)) return '--'
  if (Math.abs(x) < 1e-12) return '0'
  return Math.abs(x) >= 1e-4 && Math.abs(x) < 1e6
    ? String(Number(x.toPrecision(4)))
    : x.toExponential(3)
}

/** Findings ordered worst-first, so the one that matters is on top. */
export function ranked(findings) {
  const rank = { error: 0, warning: 1, info: 2 }
  return [...(findings ?? [])].sort(
    (a, b) => (rank[a.severity] ?? 3) - (rank[b.severity] ?? 3))
}

/**
 * What the finite-difference check says, in one sentence.
 *
 * This is the sentence that decides whether a Jacobian should leave the
 * building, so it reports the number rather than only a verdict.
 */
export function verdict(check) {
  if (!check) return ''
  const rel = Number(check.max_rel_error)
  return check.passed
    ? `AD matches central differences to ${rel.toExponential(1)} relative -- safe to export`
    : `AD and central differences disagree by ${rel.toExponential(1)} relative. `
      + 'Something in the flowsheet is not smooth at this base case; do not '
      + 'export this Jacobian until you know what.'
}
