/**
 * Reading a recorded solve (`difflow-trace/1`, from /api/debug/record).
 *
 * A trace is every checkpoint of one solve, in order, each holding only
 * what it wrote: the feeds at `solve_start`, a unit's outlets at
 * `after_unit`, the tear iterate at `iteration`, the next guess at
 * `after_accel`. The state at frame `i` is therefore the frames up to `i`
 * replayed, which is all `stateAt` does. Kept free of Svelte so it can be
 * tested on its own, like the rest of `model/`.
 */

/** The ways a scrub can step, and which frames each one stops on. */
export const STEPS = {
  frame: { label: 'Frame', keep: () => true },
  unit: { label: 'Unit', keep: (f) => f.checkpoint === 'after_unit' || f.checkpoint === 'unit_error' },
  iteration: { label: 'Iteration', keep: (f) => f.checkpoint === 'iteration' },
  event: { label: 'Event', keep: (f) => !!f.trigger || !!f.error },
}

/**
 * Every stream's values as of frame `i` (inclusive).
 *
 * @param {Object} trace  a difflow-trace/1 object
 * @param {number} i      frame index
 * @returns {Object} stream name -> {field: value}
 */
export function stateAt(trace, i) {
  const state = {}
  const frames = trace?.frames ?? []
  for (let k = 0; k <= Math.min(i, frames.length - 1); k++) {
    for (const [name, values] of Object.entries(frames[k].set ?? {})) {
      state[name] = { ...(state[name] ?? {}), ...values }
    }
  }
  return state
}

/**
 * What frame `i` changed: one row per field it wrote, with the value
 * before and after. A field that did not move is left out, unless it is
 * new (a stream seen for the first time).
 */
export function changesAt(trace, i) {
  const frame = trace?.frames?.[i]
  if (!frame?.set) return []
  const before = i > 0 ? stateAt(trace, i - 1) : {}
  const rows = []
  for (const [stream, values] of Object.entries(frame.set)) {
    for (const [field, after] of Object.entries(values)) {
      const was = before[stream]?.[field]
      if (was === undefined || !same(was, after)) {
        rows.push({ stream, field, before: was ?? null, after })
      }
    }
  }
  return rows
}

function same(a, b) {
  if (Number.isNaN(a) && Number.isNaN(b)) return true
  return a === b
}

/**
 * The residual at each tear iteration, with the frame it came from, for
 * the plot. Non-finite residuals are kept (the plot marks them).
 */
export function residualSeries(trace) {
  const out = []
  ;(trace?.frames ?? []).forEach((f, frame) => {
    if (f.checkpoint === 'iteration' && f.residual !== null && f.residual !== undefined) {
      out.push({ frame, iteration: f.iteration, residual: f.residual })
    }
  })
  return out
}

/**
 * The next (dir = 1) or previous (dir = -1) frame from `i` that `step`
 * keeps; `i` itself if there is none.
 */
export function stepFrom(trace, i, dir, step = 'frame') {
  const frames = trace?.frames ?? []
  const keep = (STEPS[step] ?? STEPS.frame).keep
  for (let k = i + dir; k >= 0 && k < frames.length; k += dir) {
    if (keep(frames[k])) return k
  }
  return i
}

/** The unit the solve is at, at frame `i` (null between units). */
export function unitAt(trace, i) {
  const f = trace?.frames?.[i]
  return f?.unit ?? null
}

/** The latest iteration number at or before frame `i`. */
export function iterationAt(trace, i) {
  const frames = trace?.frames ?? []
  for (let k = Math.min(i, frames.length - 1); k >= 0; k--) {
    if (frames[k].iteration !== null && frames[k].iteration !== undefined) {
      return frames[k].iteration
    }
  }
  return null
}

/** One line naming a frame, for the scrubber. */
export function describeFrame(frame) {
  if (!frame) return ''
  const parts = []
  const where = {
    solve_start: 'start',
    before_unit: `before ${frame.unit}`,
    after_unit: `after ${frame.unit}`,
    unit_error: `${frame.unit} raised`,
    iteration: 'tear iteration',
    after_accel: 'accelerated guess',
    terminated: 'end',
  }[frame.checkpoint] ?? frame.checkpoint
  parts.push(where)
  if (frame.iteration !== null && frame.iteration !== undefined) parts.push(`iter ${frame.iteration}`)
  if (frame.phase && !['solve', 'start', 'end'].includes(frame.phase)) parts.push(frame.phase)
  if (frame.residual !== null && frame.residual !== undefined) {
    parts.push(`residual ${formatNumber(frame.residual)}`)
  }
  if (frame.trigger) parts.push(`⚑ ${frame.trigger}`)
  return parts.join(' · ')
}

/** The frames that raised an event or an error, for the jump list. */
export function eventFrames(trace) {
  const out = []
  ;(trace?.frames ?? []).forEach((f, frame) => {
    if (f.trigger || f.error) out.push({ frame, label: describeFrame(f), error: f.error ?? null })
  })
  return out
}

/** A compact number: 4 significant figures, exponent when it needs one. */
export function formatNumber(x) {
  if (x === null || x === undefined) return '–'
  if (typeof x !== 'number') return String(x)
  if (!Number.isFinite(x)) return String(x)
  if (x === 0) return '0'
  const a = Math.abs(x)
  return a >= 1e5 || a < 1e-3 ? x.toExponential(3) : String(Number(x.toPrecision(4)))
}

/**
 * Plot coordinates for the residual series on a log axis, in a
 * `width` x `height` box. A residual of exactly zero (a loop that closed
 * exactly) sits on the bottom edge; a non-finite one on the top edge,
 * flagged `ok: false`, so a NaN is visible rather than missing.
 */
export function residualPath(series, width, height, pad = 4, scale = null) {
  if (!series.length) return { points: [], lo: null, hi: null }
  const { lo, hi, n } = scale ?? plotScale(series)
  const span = hi - lo || 1
  const points = series.map((p, k) => {
    const x = pad + (k / n) * (width - 2 * pad)
    const ok = Number.isFinite(p.residual) && p.residual >= 0
    const y = !ok ? pad
      : p.residual === 0 ? height - pad
      : pad + (1 - (Math.log10(p.residual) - lo) / span) * (height - 2 * pad)
    return { x, y, ok, frame: p.frame, iteration: p.iteration, residual: p.residual }
  })
  return { points, lo, hi }
}

/**
 * One scale for several residual series, so a what-if and the solve it
 * replays plot on the same axes: the log range over every finite positive
 * residual, and the longest series' length.
 */
export function plotScale(...many) {
  const logs = many.flat()
    .filter((p) => Number.isFinite(p.residual) && p.residual > 0)
    .map((p) => Math.log10(p.residual))
  return {
    lo: logs.length ? Math.min(...logs) : 0,
    hi: logs.length ? Math.max(...logs) : 1,
    n: Math.max(...many.map((s) => s.length - 1), 1),
  }
}

/**
 * The tear values a what-if can set at frame `i`, or null if it cannot
 * start there. Only a tear iteration or an accelerated guess can be
 * edited (that is where `set tear` works), and each value is offered as
 * the one the next pass would otherwise run on: `g(x)` at an iteration,
 * the accelerated guess at `after_accel`.
 */
export function whatIfFields(trace, i) {
  const frame = trace?.frames?.[i]
  if (!frame) return null
  if (frame.checkpoint === 'iteration' && frame.tear?.length) {
    return frame.tear.map((r) => ({ stream: r.stream, field: r.field, value: r.g }))
  }
  if (frame.checkpoint === 'after_accel' && frame.set) {
    const out = []
    for (const [stream, values] of Object.entries(frame.set)) {
      for (const [field, value] of Object.entries(values)) out.push({ stream, field, value })
    }
    return out
  }
  return null
}

/** One line saying what a what-if changed and whether it is a true replay. */
export function describeWhatIf(trace) {
  const r = trace?.result
  if (!r?.edits?.length) return ''
  const e = r.edits[0]
  const where = `iter ${e.iteration} (${e.checkpoint === 'after_accel' ? 'accelerated guess' : 'tear iteration'})`
  return `${e.stream}.${e.field}: ${formatNumber(e.was)} → ${formatNumber(e.value)} at ${where}`
}
