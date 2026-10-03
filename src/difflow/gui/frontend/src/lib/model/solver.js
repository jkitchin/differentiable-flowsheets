// The recycle solver's options, as the Solve tab edits them.
//
// The defaults are `Flowsheet.solve`'s own (and `session.SOLVER_DEFAULTS`);
// the server checks every value, so this only turns form text into the
// request: a blank field asks for the default back, which is `null`.

export const SOLVER_DEFAULTS = Object.freeze({
  tol: 1e-8,
  max_iter: 100,
  acceleration: 'anderson',
  clip_negative_flows: true,
})

export const ACCELERATIONS = ['anderson', 'wegstein', 'none']

/** The options in force: what the file stores over the defaults. */
export function solverOptions(view) {
  return { ...SOLVER_DEFAULTS, ...(view?.solver ?? {}) }
}

/**
 * One field's text as the value to send.
 *
 * A number that does not parse is sent as the text, so the server's
 * refusal names it, rather than quietly becoming the default.
 */
export function solverValue(key, text) {
  if (key === 'acceleration' || key === 'clip_negative_flows') return text
  const trimmed = String(text ?? '').trim()
  if (trimmed === '') return null
  const value = Number(trimmed)
  return Number.isNaN(value) ? trimmed : value
}
