<!--
  Taking the flowsheet out of the editor.

  An editor you can only enter is worse than none, so all four exits are
  one menu: the script, the document, the drawing, and a raster of the
  drawing for people who paste into slides. Three of them are fetched
  from the server rather than rebuilt here -- the exported script has to
  be the one `codegen` writes, and the exported diagram the one the
  reports draw -- so what leaves the editor is what difflow itself would
  produce.
-->
<script>
  import { get } from './api.js'
  import { exportName, svgDocument, svgSize } from './model/download.js'

  let { path = '', document: doc = null, disabled = false,
        onerror = () => {} } = $props()

  let open = $state(false)
  let working = $state('')

  /** Hand the browser a named file. */
  function save(blob, name) {
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
          if (blob) { save(blob, name); resolve() }
          else reject(new Error('the browser would not encode the PNG'))
        }, 'image/png')
      }
      image.onerror = () => reject(new Error('the diagram would not render'))
      image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)
    })
  }

  async function run(kind) {
    open = false
    working = kind
    try {
      if (kind === 'json') {
        if (!doc) throw new Error('no flowsheet loaded')
        save(new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' }),
             exportName(path, 'json'))
      } else if (kind === 'py') {
        const answer = await get('/api/code')
        if (answer.error) throw new Error(answer.error)
        save(new Blob([answer.source], { type: 'text/x-python' }),
             exportName(path, 'py'))
      } else {
        const answer = await get('/api/diagram')
        if (!answer.ok) throw new Error(answer.error)
        if (kind === 'svg') {
          save(new Blob([svgDocument(answer.svg)], { type: 'image/svg+xml' }),
               exportName(path, 'svg'))
        } else {
          await png(answer.svg, exportName(path, 'png'))
        }
      }
    } catch (e) {
      onerror(`export failed: ${e.message ?? e}`)
    } finally {
      working = ''
    }
  }
</script>

<div class="export">
  <button onclick={() => (open = !open)} disabled={disabled || !!working}>
    {working ? `Exporting ${working}...` : 'Export'}
  </button>
  {#if open}
    <!-- Anything else the user does means they are done with the menu. -->
    <button class="scrim" aria-label="dismiss" onclick={() => (open = false)}
    ></button>
    <ul>
      <li><button onclick={() => run('py')}>Python script<small>codegen</small></button></li>
      <li><button onclick={() => run('json')}>Flowsheet JSON<small>reloadable</small></button></li>
      <li><button onclick={() => run('svg')}>Diagram SVG<small>your layout</small></button></li>
      <li><button onclick={() => run('png')}>Diagram PNG<small>2x</small></button></li>
    </ul>
  {/if}
</div>

<style>
  .export { position: relative; }
  .scrim {
    position: fixed; inset: 0; z-index: 10;
    background: none; border: 0; padding: 0; cursor: default;
  }
  ul {
    position: absolute; right: 0; top: calc(100% + 4px); z-index: 11;
    margin: 0; padding: 0.25rem; list-style: none; min-width: 12rem;
    background: var(--panel); border: 1px solid var(--grid);
    border-radius: 5px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
  }
  li button {
    display: flex; width: 100%; align-items: baseline; gap: 0.5rem;
    background: none; border: 0; border-radius: 4px; text-align: left;
    padding: 0.35rem 0.5rem; font-size: 0.8rem; color: var(--ink);
  }
  li button:hover { background: var(--grid); }
  small { margin-left: auto; color: var(--ink-soft); font-size: 0.7rem; }
</style>
