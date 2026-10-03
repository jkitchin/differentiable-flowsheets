<!--
  The whole flowsheet as the Python script File > Python script writes,
  highlighted and read-only.

  Read-only on purpose: the script is generated from the canvas, and an
  edit made here would be gone at the next change to the flowsheet. The
  place to write Python that the flowsheet keeps is the code context,
  which is also this script's preamble. Regenerated whenever the
  flowsheet changes, so it is always the script of what is on screen.
-->
<script>
  import CodeEditor from './CodeEditor.svelte'
  import { get } from './api.js'

  let { doc = null, onclose = () => {} } = $props()

  let source = $state('')
  let error = $state('')
  let copied = $state(false)

  // `doc` is replaced on every reload after an edit, so reading it here is
  // what makes the panel follow the flowsheet. A reply that arrives after
  // a newer request was sent is dropped, so a slow one cannot overwrite a
  // fresher script.
  let asked = 0
  $effect(() => {
    doc
    const mine = ++asked
    get('/api/code')
      .then((answer) => {
        if (mine !== asked) return
        source = answer.source ?? ''
        error = answer.error ?? ''
      })
      .catch((e) => { if (mine === asked) error = String(e) })
  })

  async function copy() {
    try {
      await navigator.clipboard.writeText(source)
      copied = true
      setTimeout(() => (copied = false), 1500)
    } catch (e) {
      error = `could not copy: ${e}`
    }
  }
</script>

<section class="script">
  <header>
    <h2>Script</h2>
    <p class="hint">
      The flowsheet as runnable Python, as File &rsaquo; Python script writes it.
      Read-only: edit the canvas, or the code context for the preamble.
    </p>
    <span class="spacer"></span>
    <button onclick={copy} disabled={!source}>{copied ? 'Copied' : 'Copy'}</button>
    <button onclick={onclose}>Close</button>
  </header>

  <div class="body">
    {#if error}
      <p class="error">{error}</p>
    {:else}
      <CodeEditor value={source} readonly label="generated script" />
    {/if}
  </div>
</section>

<style>
  .script {
    display: flex;
    flex-direction: column;
    height: 22rem;
    flex: none;
    border-top: 1px solid var(--grid);
    background: var(--panel);
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    padding: 0.5rem 0.8rem;
  }
  h2 { font-size: 0.85rem; margin: 0; }
  .spacer { flex: 1; }
  .hint { color: var(--ink-soft); font-size: 0.76rem; margin: 0; }
  .body {
    display: flex;
    flex: 1;
    min-height: 0;
    margin: 0 0.8rem 0.7rem;
  }
  .error { margin: 0; color: var(--bad); font-family: ui-monospace, monospace; font-size: 0.76rem; }
</style>
