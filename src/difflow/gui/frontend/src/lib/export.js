/**
 * Taking the flowsheet out of the editor.
 *
 * An editor you can only enter is worse than none, so all four exits
 * are in the File menu: the script, the document, the drawing, and a
 * raster of the drawing for people who paste into slides. Three of them
 * are fetched from the server rather than rebuilt here --- the exported
 * script has to be the one `codegen` writes, and the exported diagram
 * the one the reports draw --- so what leaves the editor is what difflow
 * itself would produce.
 *
 * This was a component with its own button and its own popup, back when
 * every panel in the editor had one. The menu made the popup redundant
 * and the button along with it, and what is left is the part that was
 * never about the button: four files and how to hand each one over.
 */
import { get } from './api.js'
import { exportName, svgDocument, svgSize } from './model/download.js'

/** What each kind is called while it is being written. */
export const EXPORTS = {
  py: 'the Python script',
  json: 'the flowsheet',
  svg: 'the diagram',
  png: 'the diagram',
}

/**
 * Hand the browser a named file.
 *
 * Exported because the planning panel writes files of its own, and two
 * copies of the revoke-on-the-next-turn dance is one too many.
 */
export function offer(blob, name) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  anchor.click()
  // Revoked on the next turn: Safari has not necessarily started
  // reading the blob by the time `click` returns.
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

/**
 * Rasterise an SVG through an image and a canvas.
 *
 * The SVG has to go in as a data URL rather than a blob URL: a canvas
 * that has drawn a cross-origin image is tainted and `toBlob` on it
 * throws, and a blob URL counts as cross-origin here.
 */
function png(svg, name) {
  const size = svgSize(svg)
  if (!size) throw new Error('the diagram has no viewBox to size a raster by')
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = size.width
      canvas.height = size.height
      const ctx = canvas.getContext('2d')
      // Diagrams are drawn for a light page and say nothing about
      // their own ground; without this the PNG is dark-on-transparent.
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, size.width, size.height)
      ctx.drawImage(image, 0, 0, size.width, size.height)
      canvas.toBlob((blob) => {
        if (blob) { offer(blob, name); resolve(name) }
        else reject(new Error('the browser would not encode the PNG'))
      }, 'image/png')
    }
    image.onerror = () => reject(new Error('the diagram would not render'))
    image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)
  })
}

/**
 * Write one file out, and answer with what it was called.
 *
 * Throws what went wrong rather than reporting it: the caller has the
 * note line, and this has no way to show anything.
 */
export async function exportFlowsheet(kind, { path = '', doc = null } = {}) {
  if (kind === 'json') {
    if (!doc) throw new Error('no flowsheet loaded')
    const name = exportName(path, 'json')
    offer(new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' }), name)
    return name
  }
  if (kind === 'py') {
    const answer = await get('/api/code')
    if (answer.error) throw new Error(answer.error)
    const name = exportName(path, 'py')
    offer(new Blob([answer.source], { type: 'text/x-python' }), name)
    return name
  }
  const answer = await get('/api/diagram')
  if (!answer.ok) throw new Error(answer.error)
  if (kind === 'svg') {
    const name = exportName(path, 'svg')
    offer(new Blob([svgDocument(answer.svg)], { type: 'image/svg+xml' }), name)
    return name
  }
  return png(answer.svg, exportName(path, 'png'))
}
