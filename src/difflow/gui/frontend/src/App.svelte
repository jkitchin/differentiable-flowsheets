<script>
  import Assistant from './lib/Assistant.svelte'
  import Canvas from './lib/Canvas.svelte'
  import CodeContext from './lib/CodeContext.svelte'
  import Console from './lib/Console.svelte'
  import Export from './lib/Export.svelte'
  import Inspector from './lib/Inspector.svelte'
  import Palette from './lib/Palette.svelte'
  import Planning from './lib/Planning.svelte'
  import Results from './lib/Results.svelte'
  import Species from './lib/Species.svelte'
  import { del, get, patch, post, send } from './lib/api.js'
  import { inferKind } from './lib/model/assistant.js'
  import { movedPositions } from './lib/model/edit.js'
  import { flowLabels, flowTints } from './lib/model/results.js'

  let doc = $state(null)
  let path = $state('')
  // The species list, and whether it can still be changed -- the server
  // answers both with the document, because "can I edit this" is a fact
  // about the flowsheet (has it any units yet?) and not a preference.
  let species = $state([])
  let speciesEditable = $state(true)
  let catalog = $state({})
  let error = $state('')
  let note = $state('')
  let selected = $state(null)
  let busy = $state(false)
  let context = $state({ source: '', names: [], error: null })
  let showContext = $state(false)
  let result = $state(null)
  let pickers = $state(null)
  let sens = $state(null)
  let showResults = $state(false)
  // What the solver last said, kept whether or not it worked -- `result`
  // is dropped on a failure because nothing can be drawn from it, but a
  // failed solve is the thing a user is most likely to have a question
  // about, so the assistant needs it.
  let lastSolve = $state(null)
  let showAssistant = $state(false)
  let showPlanning = $state(false)
  let showConsole = $state(false)
  // Drawing preferences. Port names are off because on a wired flowsheet
  // the edge already carries the stream name, so labelling both ends of
  // every arc triples the text on screen to repeat itself; while wiring,
  // they are exactly what you want. Both are remembered, because being
  // asked the same question every time you open the editor is worse than
  // either answer.
  let portLabels = $state(remember('difflow:portLabels', false))
  let dark = $state(remember('difflow:dark', false))

  /** A remembered preference, or the default if there is nothing to read. */
  function remember(key, fallback) {
    try {
      const saved = localStorage.getItem(key)
      return saved === null ? fallback : saved === 'true'
    } catch {
      return fallback   // private window, or storage turned off
    }
  }

  // The palette is driven entirely by tokens, so the theme is one
  // attribute on the root element and nothing downstream has to know.
  $effect(() => {
    document.documentElement.dataset.theme = dark ? 'dark' : 'light'
    try {
      localStorage.setItem('difflow:dark', String(dark))
      localStorage.setItem('difflow:portLabels', String(portLabels))
    } catch { /* nothing to do about it, and nothing depends on it */ }
  })

  /**
   * `T` for the theme, `L` for port labels.
   *
   * Guarded on the target: these are single letters, and a flowsheet has
   * text fields in it. Typing "Toluene" into the species box must not
   * flip the theme twice on the way past.
   */
  function hotkey(event) {
    if (event.metaKey || event.ctrlKey || event.altKey) return
    const tag = event.target?.tagName
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
    if (event.target?.isContentEditable) return
    const key = event.key.toLowerCase()
    if (key === 't') dark = !dark
    else if (key === 'l') portLabels = !portLabels
  }

  async function load() {
    const payload = await get('/api/flowsheet')
    doc = payload.flowsheet
    path = payload.path
    species = payload.species ?? []
    speciesEditable = payload.editable !== false
    // The selection is a snapshot of a node; after a reload it may name a
    // unit that no longer exists, or one whose ports have changed.
    const id = selected?.id
    selected = id
      ? (doc?.units ?? []).some((u) => u.name === id) ? selected : null
      : null
  }

  const loadContext = () =>
    get('/api/code-context').then((c) => (context = c))

  /** What a sensitivity can be taken with respect to, and of. */
  const loadPickers = () =>
    get('/api/levers').then((p) => (pickers = p.ok ? p : null))

  // Once, at startup. Deliberately not an `$effect`: `load` both reads and
  // writes `selected`, and an effect that does that re-runs itself forever.
  Promise.all([
    load(),
    loadContext(),
    loadPickers(),
    get('/api/catalog').then((c) => (catalog = c)),
  ]).catch((e) => (error = String(e)))

  /**
   * Run one edit, then redraw from what the server says.
   *
   * A refusal is an answer, not a failure: the server sends
   * `{ok: false, error}` with a 200 for anything it understood and
   * declined, and it is reported without disturbing the canvas.
   */
  async function edit(run, { reload = true, stale = true } = {}) {
    error = ''
    note = ''
    busy = true
    try {
      const answer = await run()
      if (answer && answer.ok === false) note = answer.error
      // The server drops its own copy of the streams on any edit that is
      // not a move; the panel has to drop them too, or it goes on
      // describing a flowsheet that no longer exists. The lever list
      // goes with them: its values are the ones the edit just changed.
      if (stale) { result = null; sens = null; await loadPickers() }
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
    // The palette's flags are answered against these bindings, so a
    // `thermo` defined here un-blocks every unit that wanted one. Refetch
    // rather than reason about which: the server already knows.
    catalog = await get('/api/catalog')
    if (answer?.ok) note = `code context: ${answer.names.length} names defined`
    return answer
  }

  /**
   * Name the species, then reload: it decides what can be built at all.
   *
   * Not `stale: false` -- naming species cannot change a stream, but it
   * changes every palette answer, and the catalog is refetched for the
   * same reason the code context refetches it.
   */
  async function setSpecies(names) {
    const answer = await edit(() => post('/api/species', { species: names }))
    if (answer?.ok) {
      catalog = await get('/api/catalog')
      note = names.length ? `species: ${names.join(', ')}` : 'species cleared'
    }
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
    edit(() => post('/api/layout', { nodes: moved }),
         { reload: false, stale: false })
  }

  const solve = () =>
    edit(async () => {
      const answer = await post('/api/solve')
      lastSolve = answer
      result = answer.ok ? answer : null
      sens = null
      if (answer.ok) {
        showResults = true
        await loadPickers()
        note = answer.converged === false
          ? 'solved, but the tear residual did not reach the tolerance'
          : `solved: ${Object.keys(answer.streams).length} streams`
      } else {
        note = answer.error
      }
      return null
    }, { reload: false, stale: false })

  /** One derivative sweep. Which direction is the server's to decide. */
  const differentiate = (ask) =>
    edit(async () => {
      const answer = await post('/api/sensitivity', ask)
      sens = answer.ok ? answer : null
      if (!answer.ok) note = answer.error
      return null
    }, { reload: false, stale: false })

  const save = () =>
    edit(async () => {
      const answer = await post('/api/save')
      note = answer.ok ? `saved to ${answer.path}` : answer.error
      return null
    }, { reload: false, stale: false })

  // The canvas node says what is selected; the document says what it
  // holds and the catalog says what those parameters mean. The inspector
  // needs all three, and they are joined here rather than inside it so
  // it stays a view of state it does not own.
  let selectedUnit = $derived(
    selected?.type === 'unit'
      ? (doc?.units ?? []).find((u) => u.name === selected.id) ?? null
      : null,
  )
  let selectedSpec = $derived(
    selected?.data?.operation ? catalog[selected.data.operation] ?? null : null,
  )
  // The stream a selected feed node carries, or `null` for an inlet
  // nothing feeds yet -- which is a node the canvas draws either way,
  // so the inspector has to tell the two apart by the document.
  let selectedFeed = $derived(
    selected?.type === 'stream' && selected.data?.kind === 'feed'
      ? doc?.feeds?.[selected.data.label] ?? null
      : null,
  )

  // The solve decorates the canvas: a flow on every edge, and -- once a
  // forward sensitivity has been run -- a tint saying how hard that
  // stream responds to the lever, relative to everything else.
  let flows = $derived(result?.ok ? flowLabels(result) : null)
  let tints = $derived(sens?.mode === 'forward' ? flowTints(sens) : null)

  let summary = $derived(
    doc ? `${doc.units?.length ?? 0} units, ${Object.keys(doc.feeds ?? {}).length} feeds` : '',
  )
</script>

<svelte:window onkeydown={hotkey} />

<header>
  <h1>difflow</h1>
  <span class="path">{path || 'no file'}</span>
  <Species {species} editable={speciesEditable} {busy} onapply={setSpecies} />
  <span class="summary">{summary}</span>
  <span class="spacer"></span>
  {#if note}<span class="note">{note}</span>{/if}
  <button class="toggle" class:on={portLabels} title="name every port (L)"
          onclick={() => (portLabels = !portLabels)}>L</button>
  <button class="toggle" title="light or dark (T)"
          onclick={() => (dark = !dark)}>{dark ? '\u25d1' : '\u25d0'}</button>
  <button onclick={() => (showContext = !showContext)}
          class:primary={context.error}>Code context</button>
  <button onclick={() => (showResults = !showResults)}>Results</button>
  <button onclick={() => (showAssistant = !showAssistant)}>Ask</button>
  <button onclick={() => (showPlanning = !showPlanning)}>Planning</button>
  <button onclick={() => (showConsole = !showConsole)}>Console</button>
  <button onclick={solve} disabled={busy}>Solve</button>
  <button onclick={save} disabled={busy || !path}>Save</button>
  <Export {path} document={doc} disabled={busy || !doc}
          onerror={(why) => (note = why)} />
  <button onclick={() => edit(load)} disabled={busy}>Reload</button>
  <a class="classic" href="/classic">classic editor</a>
</header>

<main>
  <Palette {catalog} ondrop={(op) => add(op, null)} />

  <div class="stage">
    {#if error}
      <p class="error">{error}</p>
    {:else if !doc}
      <!-- Only before the first fetch answers. An editor opened with no
           file gets an empty flowsheet, not no flowsheet. -->
      <p class="empty">Loading&hellip;</p>
    {:else}
      <Canvas
        document={doc}
        positions={doc.view?.nodes ?? null}
        {catalog}
        {portLabels}
        {dark}
        onconnect={connect}
        ondeletions={applyDeletions}
        onmove={move}
        onadd={add}
        {flows}
        {tints}
        onselect={(node) => (selected = node)}
        onrefuse={(why) => (note = why)}
      />
    {/if}
  </div>

  <Inspector
    node={selected}
    unit={selectedUnit}
    spec={selectedSpec}
    feed={selectedFeed}
    {species}
    defaults={doc?.defaults ?? null}
    {busy}
    onrename={rename}
    ondelete={remove}
    onedit={edit}
  />
</main>

{#if showResults}
  <Results
    solve={result}
    levers={pickers}
    sensitivity={sens}
    {busy}
    onsensitivity={differentiate}
    onclose={() => (showResults = false)}
  />
{/if}

{#if showConsole}
  <Console
    onchanged={() => edit(async () => ({ ok: true }))}
    onclose={() => (showConsole = false)}
  />
{/if}

{#if showPlanning}
  <Planning
    {path}
    document={doc}
    levers={pickers}
    solved={!!pickers?.solved}
    onclose={() => (showPlanning = false)}
  />
{/if}

{#if showAssistant}
  <Assistant
    kind={inferKind({ unit: selectedUnit, solve: lastSolve })}
    unit={selectedUnit?.name ?? null}
    operation={selected?.data?.operation ?? null}
    onclose={() => (showAssistant = false)}
  />
{/if}

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
  /* One-letter switches, sized so they do not read as actions. */
  .toggle {
    padding: 0.2rem 0.45rem;
    min-width: 1.7rem;
    font-size: 0.75rem;
    color: var(--ink-soft);
  }
  .toggle.on { color: var(--surface); background: var(--series); border-color: var(--series); }
  main { display: flex; flex: 1; min-height: 0; }
  .stage { flex: 1; min-width: 0; }
  .error, .empty { padding: 1.5rem; color: var(--ink-soft); }
  .error { color: var(--bad); }
</style>
