/**
 * Reading a precomputed sweep: interpolation, formatting, chart geometry.
 *
 * A published page has no solver in it. What it has is a grid of solved
 * operating points, and everything the sliders do is arithmetic on that
 * grid --- which means every way the page can be wrong is a way this
 * file can be wrong, and none of it is visible in a screenshot. So it
 * is plain functions over plain data, tested under `node --test`, and
 * the component does nothing but draw what it is handed.
 *
 * This is the same arithmetic the old `publish.py` template carried as
 * a string of JavaScript inside a Python file, where nothing could test
 * it.
 */

/** A number for a reader: fixed where that is legible, exponential where not. */
export function fmt(v) {
  if (!Number.isFinite(v)) return '--'
  const m = Math.abs(v)
  if (m !== 0 && (m < 1e-3 || m >= 1e5)) return v.toExponential(3)
  return v.toFixed(m >= 100 ? 1 : m >= 1 ? 3 : 4)
}

/**
 * Where `x` falls on one axis: the lower grid index and the fraction past it.
 *
 * Clamped at both ends, and the index stops one short of the last point
 * so `i + 1` is always a real sample.
 */
export function axisFrac(axis, x) {
  const t = ((x - axis.lo) / (axis.hi - axis.lo)) * (axis.n - 1)
  const i = Math.max(0, Math.min(axis.n - 2, Math.floor(t)))
  return [i, Math.max(0, Math.min(1, t - i))]
}

/** One cell of a nested-array grid. */
export const gridAt = (grid, idx) => idx.reduce((g, i) => g[i], grid)

/**
 * Multilinear interpolation over however many axes there are.
 *
 * Exact at the grid points and approximate between them, which is the
 * whole bargain a published page makes: take enough points that the
 * curve is smooth and the approximation is invisible.
 */
export function interp(axes, grid, coords) {
  const parts = coords.map((x, i) => axisFrac(axes[i], x))
  const n = coords.length
  let total = 0
  for (let mask = 0; mask < (1 << n); mask++) {
    let weight = 1
    const idx = []
    for (let ai = 0; ai < n; ai++) {
      const [i, f] = parts[ai]
      const up = (mask >> ai) & 1
      weight *= up ? f : 1 - f
      idx.push(i + up)
    }
    if (weight > 0) total += weight * gridAt(grid, idx)
  }
  return total
}

/** The parameter values the sliders currently name. */
export const coordsAt = (axes, state) => axes.map((a, i) => a.values[state[i]])

/** The default slider position: the middle of every axis. */
export const middle = (axes) => axes.map((a) => Math.floor(a.n / 2))

/**
 * The chart, as numbers.
 *
 * The first axis is swept along the x-axis and the rest are held where
 * the sliders put them, so the curve answers "what does this quantity do
 * as I turn this one knob, here". Returned as data rather than as SVG so
 * the geometry can be checked without a DOM.
 */
export function chart(axes, output, coords, box = {}) {
  const W = box.W ?? 640, H = box.H ?? 300
  const L = box.L ?? 64, R = box.R ?? 16, T = box.T ?? 16, B = box.B ?? 46
  const axis = axes[0]
  const xs = axis.values
  const ys = xs.map((x) => interp(axes, output.values, [x, ...coords.slice(1)]))

  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  // A flat response still needs a band to draw in, and a flat response
  // at exactly zero needs one that does not collapse to nothing.
  const pad = (yMax - yMin) * 0.08 || Math.abs(yMax) * 0.08 || 1
  const lo = yMin - pad, hi = yMax + pad

  const sx = (x) => L + ((x - axis.lo) / (axis.hi - axis.lo)) * (W - L - R)
  const sy = (y) => H - B - ((y - lo) / (hi - lo)) * (H - T - B)

  return {
    W, H, L, R, T, B,
    path: xs.map((x, i) => `${i ? 'L' : 'M'}${sx(x)} ${sy(ys[i])}`).join(' '),
    yTicks: [0, 1, 2, 3, 4].map((k) => {
      const v = lo + ((hi - lo) * k) / 4
      return { y: sy(v), label: fmt(v) }
    }),
    xTicks: [0, 0.5, 1].map((f) => {
      const v = axis.lo + (axis.hi - axis.lo) * f
      return { x: sx(v), label: fmt(v) }
    }),
    xLabel: axis.label + (axis.units ? ` (${axis.units})` : ''),
    cursor: { x: sx(coords[0]), y: sy(interp(axes, output.values, coords)) },
  }
}

/**
 * Every output at the current point, with its sensitivity.
 *
 * The derivative is the exact one difflow recorded at the grid points,
 * interpolated like any other quantity --- reported, never used to
 * interpolate the value itself.
 */
export function readout(axes, outputs, coords) {
  const key = axes[0].key
  return outputs.map((o) => ({
    name: o.name,
    units: o.units,
    value: interp(axes, o.values, coords),
    sensitivity: o.gradients && o.gradients[key]
      ? interp(axes, o.gradients[key], coords)
      : null,
  }))
}
