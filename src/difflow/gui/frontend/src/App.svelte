<script>
  import Assistant from './lib/Assistant.svelte'
  import Canvas from './lib/Canvas.svelte'
  import CodeContext from './lib/CodeContext.svelte'
  import Console from './lib/Console.svelte'
  import ContextMenu from './lib/ContextMenu.svelte'
  import Inspector from './lib/Inspector.svelte'
  import MenuBar from './lib/MenuBar.svelte'
  import Palette from './lib/Palette.svelte'
  import Planning from './lib/Planning.svelte'
  import Results from './lib/Results.svelte'
  import Species from './lib/Species.svelte'
  import { del, get, patch, post, send } from './lib/api.js'
  import { EXPORTS, exportFlowsheet } from './lib/export.js'
  import { inferKind } from './lib/model/assistant.js'
  import { movedPositions } from './lib/model/edit.js'
  import { nodeMenu } from './lib/model/menu.js'
  import { menuBar, shortPath } from './lib/model/menubar.js'
  import { keepAlive } from './lib/model/lifetime.js'
  import { flowLabels, flowTints } from './lib/model/results.js'

  let doc = $state(null)
  let path = $state('')
  // The script the flowsheet was built by running, when it was one.
  // `path` is then the JSON beside it that Save writes, which is not
  // the name the user typed and not the name to show them first.
  let source = $state('')
  // The species list, and whether it can still be changed -- the server
  // answers both with the document, because "can I edit this" is a fact
  // about the flowsheet (has it any units yet?) and not a preference.
  let species = $state([])
  let speciesEditable = $state(true)
  // What each unfinished unit is still waiting for. The units themselves
  // are in the document --- they go on the flowsheet so that they have
  // ports to wire --- and this is the note beside them, exactly as the
  // server sends it: the name, and what would finish it.
  let pending = $state([])
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
  // The right-click menu: `{x, y, title, items}` while it is open. The
  // items are built when it opens rather than derived, because they are
  // about the node that was clicked and that is not state anything else
  // needs --- and because an item's action must not change under it.
  let menu = $state(null)
  // Drawing preferences. Port names are off because on a wired flowsheet
  // the edge already carries the stream name, so labelling both ends of
  // every arc triples the text on screen to repeat itself; while wiring,
  // they are exactly what you want. Both are remembered, because being
  // asked the same question every time you open the editor is worse than
  // either answer.
  let portLabels = $state(remember('difflow:portLabels', false))
  let dark = $state(remember('difflow:dark', false))
  // Where the project lives, asked of the server rather than built into
  // the bundle: the URLs are in `pyproject.toml` and should be in one
  // place. Empty until the answer arrives, which is why the links are
  // rendered conditionally rather than with a placeholder href.
  let about = $state({ version: '', links: {}, heartbeat: 15 })
  // Which file the File menu is writing out, if any. The menu closed
  // behind the click, so this is what stops a second click from asking
  // for the same file twice while the first is still being drawn.
  let exporting = $state('')
  // Quit arms on the first click and fires on the second. A single
  // click that ends the process is the wrong shape for a menu row, and
  // the armed half is a button in the header rather than a second trip
  // into the menu -- a question asked somewhere you are not looking is
  // not asked at all.
  let quitArmed = $state(false)
  let stopped = $state(false)
  let armedTimer = null
  const ASKING = 'click \u201cReally quit?\u201d to stop the editor'

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

  /** Is this keystroke on its way into something that takes text? */
  function typing(event) {
    const tag = event.target?.tagName
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      || !!event.target?.isContentEditable
  }

  /**
   * `T` for the theme, `L` for port labels, and the two accelerators
   * the menus advertise.
   *
   * The single letters are guarded on the target: a flowsheet has text
   * fields in it, and typing "Toluene" into the species box must not
   * flip the theme twice on the way past. Cmd-S is not guarded, because
   * saving the flowsheet is what it means wherever it is pressed --- and
   * because the browser's own answer to it, offering to save the page,
   * has never once been what anyone wanted here. Cmd-Enter is guarded:
   * the console already ends a cell with it.
   */
  function hotkey(event) {
    if (menu) return       // the open menu owns the keyboard
    if (event.metaKey || event.ctrlKey) {
      if (event.altKey || event.shiftKey) return
      if (event.key.toLowerCase() === 's') {
        event.preventDefault()
        if (!busy && path) save()
      } else if (event.key === 'Enter' && !typing(event)) {
        event.preventDefault()
        if (!busy) solve()
      }
      return
    }
    if (event.altKey || typing(event)) return
    const key = event.key.toLowerCase()
    if (key === 't') dark = !dark
    else if (key === 'l') portLabels = !portLabels
  }

  async function load() {
    const payload = await get('/api/flowsheet')
    doc = payload.flowsheet
    path = payload.path
    source = payload.source ?? ''
    species = payload.species ?? []
    speciesEditable = payload.editable !== false
    pending = payload.pending ?? []
    // The selection is a snapshot of a node; after a reload it may name a
    // unit that no longer exists, or one whose ports have changed. A
    // pending node counts as still there -- it is the one selection the
    // user is most likely to be in the middle of answering.
    const id = selected?.id
    const alive = (name) =>
      (doc?.units ?? []).some((u) => u.name === name) ||
      pending.some((p) => p.name === name)
    selected = id ? (alive(id) ? selected : null) : null
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
   * Telling the server this tab is open, so that closing it stops it.
   *
   * Started at the default interval straight away rather than waiting
   * for `/api/about` to say what the interval should be: the window
   * between the two is exactly when a page that failed to load would
   * otherwise look, to the server, like a tab that was never opened.
   */
  let alive = keepAlive({ post })
  alive.start()

  // Not in the `Promise.all` above: a failure here is a header without
  // links, which is a smaller thing than a flowsheet that would not
  // load, and it must not be reported as the latter.
  get('/api/about')
    .then((a) => {
      about = a
      // Re-pitched to whatever the server says its grace is built
      // around. One agreement, and the server is the half that holds it.
      alive.stop()
      alive = keepAlive({ post, client: alive.client, interval: a.heartbeat ?? 15 })
      alive.start()
    })
    .catch(() => {})

  /**
   * Stop the editor.
   *
   * Two clicks, because this ends a process and it sits in a row of
   * buttons that do not. The armed state lapses after a few seconds so
   * a stray first click does not leave a live trigger in the header.
   */
  async function quit() {
    if (!quitArmed) {
      quitArmed = true
      note = ASKING
      clearTimeout(armedTimer)
      armedTimer = setTimeout(() => {
        quitArmed = false
        // Only our own note: six seconds is long enough for something
        // else to have had something to say.
        if (note === ASKING) note = ''
      }, 6000)
      return
    }
    clearTimeout(armedTimer)
    quitArmed = false
    alive.stop()
    // The server answers before it stops, so a thrown request here is
    // a real failure rather than the expected dropped connection. It is
    // still not worth a red banner: the page is about to say it has
    // stopped either way, and if the server is already gone that
    // statement is true.
    try {
      await post('/api/quit')
    } catch { /* already gone, which is the outcome asked for */ }
    stopped = true
  }

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
    // A drop that cannot be built is not a failure: the node is on the
    // canvas in red, and the hint is what it is waiting for. Select it,
    // so the inspector is already showing the way out of it.
    if (answer?.ok && answer.pending) {
      note = answer.hint
      selected = { id: answer.name, type: 'unit',
                   data: { label: answer.name, operation } }
    } else if (answer?.ok && answer.placeholders?.length) {
      // A required number with no default gets a placeholder rather than
      // blocking the drop. Saying so is the whole difference between a
      // default and a guess.
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
    if (answer?.ok) note = built(answer, `${answer.names.length} names defined`)
    return answer
  }

  /**
   * What an edit that can promote a red node has to say about it.
   *
   * The server retries every pending drop after the code context or the
   * species change, and the interesting half of "4 names defined" is
   * which of the red boxes went away because of it.
   */
  function built(answer, fallback) {
    const n = answer.promoted?.length ?? 0
    if (!n) return fallback
    return `${fallback} -- ${answer.promoted.join(', ')} built`
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
      note = built(answer,
                   names.length ? `species: ${names.join(', ')}` : 'species cleared')
    }
    return answer
  }

  const rename = (name, to) =>
    edit(() => patch(`/api/unit/${encodeURIComponent(name)}`, { name: to }))

  const remove = (name) =>
    edit(() => del(`/api/unit/${encodeURIComponent(name)}`))

  /**
   * A node's name on the clipboard.
   *
   * Worth a menu row because the names are typed elsewhere: a lever is
   * `unit.param`, a planning spec names a stream, and the console works
   * in the same namespace. Retyping one from the canvas is how a spec
   * ends up pointing at a unit that does not exist.
   */
  async function copyName(name) {
    try {
      await navigator.clipboard.writeText(name)
      note = `copied \u201c${name}\u201d`
    } catch {
      // Denied, or no clipboard at all over a bare http origin in some
      // browsers. Say so: silence looks like it worked.
      note = `could not copy \u201c${name}\u201d -- the browser refused the clipboard`
    }
  }

  /**
   * Write one file out.
   *
   * The menu is closed by the time this runs, so the note line is the
   * only place left to say that a diagram is being drawn --- and it is
   * where the name of the file that arrived goes too, because a
   * download landing quietly in a folder nobody is looking at reads as
   * a menu row that did nothing.
   */
  async function runExport(kind) {
    error = ''
    exporting = kind
    note = `writing ${EXPORTS[kind]}\u2026`
    try {
      note = `wrote ${await exportFlowsheet(kind, { path, doc })}`
    } catch (e) {
      note = `export failed: ${e.message ?? e}`
    } finally {
      exporting = ''
    }
  }

  /**
   * Open the right-click menu on a node.
   *
   * Every item is a shortcut to a control that is already on screen, so
   * this adds no verbs --- it hands `nodeMenu` the ones it has. The
   * assistant and the inspector both read the selection, which the
   * canvas has already set by the time this runs.
   */
  function openMenu({ node, x, y }) {
    const { title, items } = nodeMenu({
      node,
      catalog,
      actions: {
        docs: (url) => window.open(url, '_blank', 'noopener,noreferrer'),
        ask: () => (showAssistant = true),
        context: () => (showContext = true),
        copy: copyName,
        remove,
      },
    })
    menu = items.length ? { x, y, title, items } : null
  }

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
    // Every node's coordinate is in `view.nodes`, an unfinished unit's
    // included: it is on the flowsheet, so the file has a place for it.
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
        const held = answer.pending?.length
          // What was solved is not what is on the canvas. Said here
          // rather than left to the picture, because the numbers in the
          // results panel look exactly the same either way.
          ? ` (${answer.pending.join(', ')} not built, and not in it)`
          : ''
        note = (answer.converged === false
          ? 'solved, but the tear residual did not reach the tolerance'
          : `solved: ${Object.keys(answer.streams).length} streams`) + held
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
      note = answer.ok ? `saved to ${shortPath(answer.path)}` : answer.error
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
  // A red node: dropped, not built. It has no entry in the document, so
  // it is looked up here and the inspector shows what it is waiting for
  // instead of a parameter form for a unit that does not exist.
  let selectedPending = $derived(
    selected?.type === 'unit'
      ? pending.find((p) => p.name === selected.id) ?? null
      : null,
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

  // The header's menus. Derived rather than built when one is opened,
  // because half of what they report is whether something is already
  // open: a row ticked when the drawer behind it was closed a moment
  // ago is worse than no tick at all. Every action is a toggle or a
  // verb that already existed -- the menus add no capability, they are
  // where the capabilities went.
  let menus = $derived(menuBar({
    path,
    doc,
    busy,
    exporting,
    contextError: !!context.error,
    portLabels,
    dark,
    panels: {
      results: showResults,
      context: showContext,
      console: showConsole,
      planning: showPlanning,
      assistant: showAssistant,
    },
    links: about.links ?? {},
    actions: {
      save,
      reload: () => edit(load),
      export: runExport,
      quit,
      results: () => (showResults = !showResults),
      context: () => (showContext = !showContext),
      console: () => (showConsole = !showConsole),
      planning: () => (showPlanning = !showPlanning),
      assistant: () => (showAssistant = !showAssistant),
      portLabels: () => (portLabels = !portLabels),
      dark: () => (dark = !dark),
      open: (url) => window.open(url, '_blank', 'noopener,noreferrer'),
      // In this tab, as it has always been: the classic editor is the
      // other half of the same session, not a page about difflow.
      classic: () => (window.location.href = '/classic'),
    },
  }))
</script>

<svelte:window
  onkeydown={hotkey}
  onpagehide={(e) => alive.farewell({ persisted: e.persisted })}
  onpageshow={() => { if (!stopped) alive.ping() }}
/>

<header>
  <h1>difflow</h1>
  {#if about.version}<span class="version">{about.version}</span>{/if}
  <MenuBar {menus} />
  <span
    class="path"
    title={source ? `${source}\nsaves to ${path}` : path}
  >{shortPath(source || path) || 'no file'}</span>
  <Species {species} editable={speciesEditable} {busy} onapply={setSpecies} />
  <span class="summary">{summary}</span>
  <span class="spacer"></span>
  {#if note}<span class="note">{note}</span>{/if}
  <!-- Only while it is armed. Quit is a row in the File menu, and the
       menu shuts behind the click; this is the second half of the
       question, asked where the answer can be seen. -->
  {#if quitArmed}
    <button class="quit" onclick={quit}
            title="stop the editor and free the port">Really quit?</button>
  {/if}
  <!-- The one verb the editor exists for, and the only control here
       that is not a menu. -->
  <button class="primary" onclick={solve} disabled={busy}
          title="solve the flowsheet (Cmd-Enter)">Solve</button>
</header>

{#if stopped}
  <!-- The server is gone, so nothing on the page can work any more.
       Said plainly rather than left to fail one request at a time. -->
  <div class="stopped" role="status">
    <h2>The editor has stopped.</h2>
    <p>
      Port is free. Close this tab; run <code>difflow gui{source || path ? ` ${source || path}` : ''}</code>
      to start it again.
    </p>
  </div>
{/if}

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
        {pending}
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
        onmenu={openMenu}
        onrefuse={(why) => (note = why)}
      />
    {/if}
  </div>

  <Inspector
    node={selected}
    unit={selectedUnit}
    spec={selectedSpec}
    feed={selectedFeed}
    pending={selectedPending}
    {species}
    defaults={doc?.defaults ?? null}
    {busy}
    onrename={rename}
    ondelete={remove}
    onedit={edit}
    oncontext={applyContext}
  />
</main>

{#if menu}
  <ContextMenu
    x={menu.x}
    y={menu.y}
    title={menu.title}
    items={menu.items}
    onclose={() => (menu = null)}
  />
{/if}

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
  .path, .summary, .version { color: var(--ink-soft); font-size: 0.8rem; }
  /* Already shortened, and never the reason the header is two lines tall. */
  .path { white-space: nowrap; }
  .version { font-variant-numeric: tabular-nums; opacity: 0.75; }
  .note { color: var(--accent); font-size: 0.8rem; }
  .spacer { flex: 1; }
  /* It is on screen only to be answered, and it ends the process. */
  .quit {
    color: var(--surface);
    background: var(--bad);
    border-color: var(--bad);
  }
  /* Covers the editor rather than sitting above it: every control
     behind this is talking to a server that is no longer there. */
  .stopped {
    position: fixed;
    inset: 0;
    z-index: 50;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    text-align: center;
    background: var(--surface);
    color: var(--ink-soft);
  }
  .stopped h2 { margin: 0; font-size: 1rem; color: var(--ink); }
  .stopped p { margin: 0; font-size: 0.85rem; }
  main { display: flex; flex: 1; min-height: 0; }
  .stage { flex: 1; min-width: 0; }
  .error, .empty { padding: 1.5rem; color: var(--ink-soft); }
  .error { color: var(--bad); }
</style>
