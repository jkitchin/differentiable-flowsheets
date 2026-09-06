<script>
  import Canvas from './lib/Canvas.svelte'
  import { get } from './lib/api.js'

  let doc = $state(null)
  let path = $state('')
  let error = $state('')

  async function load() {
    error = ''
    try {
      const payload = await get('/api/flowsheet')
      doc = payload.flowsheet
      path = payload.path
    } catch (e) {
      error = String(e)
    }
  }

  $effect(() => { load() })

  let positions = $derived(doc?.view?.nodes ?? null)
  let summary = $derived(
    doc ? `${doc.units?.length ?? 0} units, ${Object.keys(doc.feeds ?? {}).length} feeds` : '',
  )
</script>

<header>
  <h1>difflow</h1>
  <span class="path">{path || 'no file'}</span>
  <span class="summary">{summary}</span>
  <span class="spacer"></span>
  <button onclick={load}>Reload</button>
  <a class="classic" href="/classic">classic editor</a>
</header>

<main>
  {#if error}
    <p class="error">{error}</p>
  {:else if !doc}
    <p class="empty">No flowsheet loaded.</p>
  {:else}
    <Canvas document={doc} {positions} />
  {/if}
</main>

<style>
  header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.7rem 1rem;
    border-bottom: 1px solid var(--grid);
    background: var(--panel);
  }
  h1 { font-size: 0.95rem; margin: 0; font-weight: 650; letter-spacing: -0.01em; }
  .path, .summary { color: var(--ink-soft); font-size: 0.8rem; }
  .spacer { flex: 1; }
  .classic { font-size: 0.78rem; }
  main { height: calc(100vh - 3.1rem); }
  .error, .empty { padding: 1.5rem; color: var(--ink-soft); }
  .error { color: var(--bad); }
</style>
