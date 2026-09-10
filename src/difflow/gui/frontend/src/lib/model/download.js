/**
 * Turning what the server sends into a file the browser will save.
 *
 * The naming and the SVG handling are here, apart from the DOM, because
 * they are the parts that can be quietly wrong: a download called
 * `undefined.py`, or an SVG that opens in the editor and not in Inkscape.
 * The `Blob` and the anchor click are in the component.
 */

/**
 * What to call the file, from the flowsheet's own path.
 *
 * An unsaved flowsheet has no path, and `flowsheet.py` is a better guess
 * than the page title -- it is what the exported script would be called
 * if the user saved it first.
 */
export function exportName(path, ext) {
  const base = String(path || '').split(/[\\/]/).pop() || ''
  const stem = base.replace(/\.[^.]*$/, '') || 'flowsheet'
  return `${stem}.${ext}`
}

/**
 * One `<svg>` element as a standalone document.
 *
 * The inline SVG a report embeds is a fragment: it carries the namespace
 * but no XML declaration, and a file that opens fine in a browser can be
 * refused by a drawing program without one.
 */
export function svgDocument(svg) {
  return `<?xml version="1.0" encoding="UTF-8"?>\n${String(svg || '').trim()}\n`
}

/**
 * The pixel size to raster an SVG at, from its own `viewBox`.
 *
 * `width="100%"` is right for a document that flows and useless for a
 * canvas, which needs a number. `scale` is the supersampling factor: a
 * diagram pasted into a slide at 1x looks like a screenshot of a
 * diagram.
 */
export function svgSize(svg, scale = 2) {
  const box = /viewBox="([\d.\-\s]+)"/.exec(String(svg || ''))
  if (!box) return null
  const [, , w, h] = box[1].trim().split(/\s+/).map(Number)
  if (!(w > 0 && h > 0)) return null
  return { width: Math.round(w * scale), height: Math.round(h * scale) }
}
