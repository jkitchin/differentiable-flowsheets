<script>
  import Canvas from './lib/Canvas.svelte'
  import CodeContext from './lib/CodeContext.svelte'
  import Palette from './lib/Palette.svelte'
  import Selection from './lib/Selection.svelte'
  import { del, get, patch, post, send } from './lib/api.js'
  import { movedPositions } from './lib/model/edit.js'

  let doc = $state(null)
  let path = $state('')
  let catalog = $state({})
  let error = $state('')
  let note = $state('')
  let selected = $state(null)
  let busy = $state(false)
  let context = $state({ source: '', names: [], error: null })
  let showContext = $state(false)

  async function load() {
    const payload = await get('/api/flowsheet')
    doc = payload.flowsheet
    path = payload.path
    // The selection is a snapshot of a node; after a reload it may name a
    // unit that no longer exists, or one whose ports have changed.
    const id = selected?.id
    selected = id
      ? (doc?.units ?? []).some((u) => u.name === id) ? selected : null
      : null
  }

  const loadContext = () =>
    get('/api/code-context').then((c) => (context = c))

  // Once, at startup. Deliberately not an `$effect`: `load` both reads and
  // writes `selected`, and an effect that does that re-runs itself forever.
  Promise.all([
    load(),
    loadContext(),
    get('/api/catalog').then((c) => (catalog = c)),
  ]).catch((e) => (error = String(e)))

  /**
   * Run one edit, then redraw from what the server says.
   *
   * A refusal is an answer, not a failure: the server sends
   * `{ok: false, error}` with a 200 for anything it understood and
   * declined, and it is reported without disturbing the canvas.
   */
  async function edit(run, { reload = true } = {}) {
    error = ''
    note = ''
    busy = true
    try {
      const answer = await run()
      if (answer && answer.ok === false) note = answer.error
      if (reload) await load()
      return answer
    } catch (e) {
      error = String(e)
    } finally {
      busy = false
    }
  }

  const connect = (wire) => edit(() => post('/api/connect', wire))

  async function add(operation, position) {
    const answer = await edit(() => post('/api/unit', { operation, position }))
    // A required number with no default gets a placeholder rather than
    // blocking the drop. Saying so is the whole difference between a
    // default and a guess.
    if (answer?.ok && answer.placeholders?.length) {
      note = `${answer.name}: ${answer.placeholders.join(', ')} set to a placeholder`
    }
    return answer
  }

  /** Apply the snippet, then reload: what it defines changes what builds. */
  async function applyContext(source) {
    const answer = await edit(() => post('/api/code-context', { source }))
    await loadContext()
    if (answer?.ok) note = `code context: ${answer.names.length} names defined`
    return answer
  }

  const rename = (name, to) =>
    edit(() => patch(`/api/unit/${encodeURIComponent(name)}`, { name: to }))

  const remove = (name) =>
    edit(() => del(`/api/unit/${encodeURIComponent(name)}`))

  async function applyDeletions(requests) {
    if (!requests.length) return load()   // redraw whatever was taken off
    await edit(async () => {
      for (const r of requests) {
        const answer = await send(r.method, r.path, r.body)
        if (answer && answer.ok === false) return answer
      }
    })
  }

  /** Positions only: no rebuild, no solve, and no reload to fight the drag. */
  function move(positions) {
    const moved = movedPositions(positions, doc?.view?.nodes)
    if (!Object.keys(moved).length) return
    // Adopted locally too, so the next reload does not snap the node back
    // to where the document still says it is.
    doc = { ...doc, view: { ...doc.view, nodes: { ...doc.view?.nodes, ...moved } } }
    edit(() => post('/api/layout', { nodes: moved }), { reload: false })
  }

  const solve = () =>
    edit(async () => {
      const answer = await post('/api/solve')
      note = answer.ok
        ? `solved: ${Object.keys(answer.streams).length} streams`
        : answer.error
      return null
    }, { reload: false })

  const save = () =>
    edit(async () => {
      const answer = await post('/api/save')
      note = answer.ok ? `saved to ${answer.path}` : answer.error
      return null
    }, { reload: false })

  let summary = $derived(
    doc ? `${doc.units?.length ?? 0} units, ${Object.keys(doc.feeds ?? {}).length} feeds` : '',
  )
</script>

<header>
  <h1>difflow</h1>
  <span class="path">{path || 'no file'}</span>
  <span class="summary">{summary}</span>
  <span class="spacer"></span>
  {#if note}<span class="note">{note}</span>{/if}
  <button onclick={() => (showContext = !showContext)}
          class:primary={context.error}>Code context</button>
  <button onclick={solve} disabled={busy}>Solve</button>
  <button onclick={save} disabled={busy || !path}>Save</button>
  <button onclick={() => edit(load)} disabled={busy}>Reload</button>
  <a class="classic" href="/classic">classic editor</a>
</header>

<main>
  <Palette {catalog} ondrop={(op) => add(op, null)} />

  <div class="stage">
    {#if error}
      <p class="error">{error}</p>
    {:else if !doc}
      <p class="empty">No flowsheet loaded.</p>
    {:else}
      <Canvas
        document={doc}
        positions={doc.view?.nodes ?? null}
        onconnect={connect}
        ondeletions={applyDeletions}
        onmove={move}
        onadd={add}
        onselect={(node) => (selected = node)}
        onrefuse={(why) => (note = why)}
      />
    {/if}
  </div>

  <Selection node={selected} onrename={rename} ondelete={remove} />
</main>

{#if showContext}
  <CodeContext
    source={context.source}
    names={context.names}
    error={context.error ?? ''}
    {busy}
    onapply={applyContext}
    onclose={() => (showContext = false)}
  />
{/if}

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
  .note { color: var(--accent); font-size: 0.8rem; }
  .spacer { flex: 1; }
  .classic { font-size: 0.78rem; }
  main { display: flex; flex: 1; min-height: 0; }
  .stage { flex: 1; min-width: 0; }
  .error, .empty { padding: 1.5rem; color: var(--ink-soft); }
  .error { color: var(--bad); }
</style>
