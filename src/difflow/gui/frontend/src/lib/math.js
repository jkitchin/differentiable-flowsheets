/**
 * The `equations` a unit declares, rendered.
 *
 * KaTeX is bundled rather than pulled from a CDN: `publish.py` has a
 * `test_is_self_contained` that forbids an external `src=`, and the
 * editor is meant to work on a laptop with no network.
 *
 * It renders to **MathML**, not to KaTeX's own HTML. The HTML output is
 * laid out against KaTeX's fonts and looks wrong without them, which
 * would mean committing twenty `.woff2` files and a stylesheet whose
 * only job is to position glyphs. MathML asks the browser to do that
 * instead, and every browser that can run this editor has had MathML
 * Core for years. Balance equations -- fractions, subscripts, sums --
 * are what it is good at.
 */

import katex from 'katex'

/**
 * One LaTeX expression as MathML.
 *
 * @returns `{html, error}`. A malformed expression is shown as its own
 *   source rather than swallowed: the equation is difflow's to fix, and
 *   an inspector that silently drops it hides that.
 */
export function render(latex, { display = true } = {}) {
  try {
    return {
      html: katex.renderToString(latex, {
        output: 'mathml',
        displayMode: display,
        throwOnError: true,
        strict: false,
      }),
      error: null,
    }
  } catch (e) {
    return { html: null, error: String(e.message ?? e) }
  }
}
