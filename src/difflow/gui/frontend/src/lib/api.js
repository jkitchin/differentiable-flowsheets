/**
 * The one place that talks to the Python server.
 *
 * JSON has no literal for the non-finite floats, and `JSON.parse` rejects
 * the `Infinity` that Python's `json` writes -- which is the common case,
 * not an exotic one: `mass_action_kinetics` puts `inf` in `K_eq` for every
 * irreversible reaction. So they cross the wire as strings and are
 * restored here, matching `_json_safe` / `_json_restore` in server.py.
 */

const NON_FINITE = { Infinity: Infinity, '-Infinity': -Infinity, NaN: NaN }

/**
 * The token the server put in the page it served, sent back on every
 * mutating request.
 *
 * The editor can `exec` Python (the code context), so a request that
 * merely reaches the port is not enough -- it has to come from this
 * page. Another origin can send us a request, but it cannot read this
 * page to learn the token. In `npm run dev` the page comes from vite
 * and carries no tag; the proxy adds the header instead, matching
 * `python -m difflow.gui --token dev`.
 */
const TOKEN =
  globalThis.document?.querySelector('meta[name="difflow-token"]')?.content ?? '' 

/** Undo the server's `_json_safe`. */
export function restore(value) {
  if (typeof value === 'string') {
    return Object.prototype.hasOwnProperty.call(NON_FINITE, value)
      ? NON_FINITE[value]
      : value
  }
  if (Array.isArray(value)) return value.map(restore)
  if (value && typeof value === 'object') {
    const out = {}
    for (const [k, v] of Object.entries(value)) out[k] = restore(v)
    return out
  }
  return value
}

/** Match the server's `_json_restore` on the way back. */
export function safe(value) {
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return 'NaN'
    if (value === Infinity) return 'Infinity'
    if (value === -Infinity) return '-Infinity'
    return value
  }
  if (Array.isArray(value)) return value.map(safe)
  if (value && typeof value === 'object') {
    const out = {}
    for (const [k, v] of Object.entries(value)) out[k] = safe(v)
    return out
  }
  return value
}

async function request(path, init) {
  const response = await fetch(path, init)
  const text = await response.text()
  let body
  try {
    body = text ? JSON.parse(text) : {}
  } catch {
    // A traceback rendered as HTML, or an empty 500: say so rather than
    // failing with a parse error three layers up.
    throw new Error(`${path}: ${response.status} ${text.slice(0, 200)}`)
  }
  // A refused edit comes back as `{ok: false, error: ...}` with a 200 and is
  // the caller's to read. A non-2xx is not: it means the route is wrong or
  // the server broke, and quietly returning its body would hide that.
  if (!response.ok) {
    throw new Error(`${path}: ${response.status} ${body.error ?? text.slice(0, 200)}`)
  }
  return restore(body)
}

export const get = (path) => request(path)

/** POST / PATCH / DELETE, all of which carry a JSON body here. */
export const send = (method, path, payload) =>
  request(path, {
    method,
    headers: { 'Content-Type': 'application/json', 'X-Difflow-Token': TOKEN },
    body: JSON.stringify(safe(payload ?? {})),
  })

export const post = (path, payload) => send('POST', path, payload)
export const patch = (path, payload) => send('PATCH', path, payload)
export const del = (path, payload) => send('DELETE', path, payload)
