<!--
  What is selected, and what the code already knows about it.

  This replaces the step-3 selection panel. That one could rename a unit,
  list its ports and delete it, which is the part of an inspector that
  needs no knowledge of the unit at all. The rest -- what the parameters
  mean, what they are measured in, what equations the unit solves, what
  it assumes and where it comes from -- is already declared on the unit
  classes for the report writer (`difflow.report.metadata`) and was read
  by nobody. The catalog now carries it and this panel shows it, so the
  documentation cannot drift from the model: there is no second copy.

  Parameters are edited in place. A field that holds an array, a rate
  law or a name from the code context is shown with where its value came
  from instead of an input, since none of those is something that can be
  typed into a box.
-->
<script>
  import { del, get, patch, post } from './api.js'
  import { render } from './math.js'
  import { feedEdit, feedFields, feedValues } from './model/feed.js'
  import { classify, label, parse } from './model/params.js'

  let {
    node = null,
    unit = null,
    spec = null,
    feed = null,
    pending = null,
    species = [],
    defaults = null,
    busy = false,
    onrename = () => {},
    ondelete = () => {},
    onedit = () => {},
    oncontext = () => {},
  } = $props()

  let draft = $state('')
  let docs = $state(null)
  let isUnit = $derived(node?.type === 'unit')
  let isFeed = $derived(node?.type === 'stream' && node.data?.kind === 'feed')

  // Reset whenever the selection changes, so the box never shows the
  // name of a node that is no longer selected.
  $effect(() => { draft = node?.data?.label ?? '' })

  /**
   * The rendered docstring, fetched per operation and kept.
   *
   * Rendering all 87 docstrings to open the palette would be work done
   * for the one the user eventually clicks, so the server renders one at
   * a time and the answers are cached here for the session.
   */
  const cache = new Map()

  $effect(() => {
    const operation = isUnit ? node.data.operation : null
    if (!operation) { docs = null; return }
    if (cache.has(operation)) { docs = cache.get(operation); return }
    docs = null
    get(`/api/docs/${encodeURIComponent(operation)}`)
      .then((answer) => {
        cache.set(operation, answer)
        // The selection may have moved on while this was in flight.
        if (node?.data?.operation === operation) docs = answer
      })
      .catch(() => {})
  })

  /** The catalog's spec for each field, by name. */
  let specs = $derived(
    Object.fromEntries((spec?.parameters ?? []).map((p) => [p.name, p])),
  )

  /**
   * The constructor objects the unit was built with.
   *
   * Half the catalog needs one -- a `thermo`, an `eos` -- and they are
   * not `Params` fields, so a panel that showed only parameters would
   * not say where a Flash got its thermodynamics from. They are always
   * a `$ref` into the code context, which is exactly the thing worth
   * reporting.
   */
  let extras = $derived(
    Object.entries(unit?.constructor ?? {}).map(([name, value]) => ({
      name, ...classify(value, { name }),
    })),
  )

  /**
   * Every parameter the unit holds, in the order the `Params` class
   * declares them, classified into what can be edited and what can only
   * be reported.
   */
  let fields = $derived.by(() => {
    if (!unit) return []
    const params = unit.params ?? {}
    const declared = (spec?.parameters ?? []).map((p) => p.name)
    const extra = Object.keys(params).filter((k) => !declared.includes(k))
    return [...declared, ...extra]
      .filter((name) => name in params || name in specs)
      .map((name) => {
        const value = params[name]
        const field = specs[name] ?? { name }
        return { name, value, spec: field, ...classify(value, field) }
      })
  })

  /**
   * The feed form: what a stream carries, if anything feeds it.
   *
   * The one part of a flowsheet that cannot be built by dropping and
   * wiring. `feed` is the declared stream or `null` for an inlet with
   * nothing on the other end, and both get the same form -- the
   * difference between "no feed" and "a feed at the defaults" is one the
   * user is about to erase anyway, and the button says which it is.
   *
   * The arithmetic is in `model/feed.js`, where node can test it: a
   * blank box must not become a zero, and one edited field must not zero
   * the others.
   */
  let feedDraft = $state({})

  // A new selection, or a reload after an edit, drops the drafts: they
  // describe a form that is no longer on screen.
  $effect(() => {
    void node
    void feed
    feedDraft = {}
  })

  let feedNow = $derived(feedValues(feed, species, defaults ?? {}))
  let feedForm = $derived(feedFields(feedNow, species))
  let pendingFeed = $derived(feedEdit(feedDraft, feedNow))
  let canApplyFeed = $derived(
    !busy && !pendingFeed.errors.length && (pendingFeed.dirty || !feed),
  )

  async function applyFeed() {
    if (!canApplyFeed) return
    const answer = await onedit(() =>
      post('/api/feed', { name: node.data.label, ...pendingFeed.spec }),
    )
    if (answer?.ok !== false) feedDraft = {}
  }

  function removeFeed() {
    onedit(() => del(`/api/feed/${encodeURIComponent(node.data.label)}`))
  }

  function feedKeydown(event) {
    if (event.key === 'Enter') {
      event.preventDefault()
      applyFeed()
    } else if (event.key === 'Escape') {
      feedDraft = {}
      event.currentTarget.blur()
    }
    // A species name is not a hotkey: `t` in a flow box must not flip
    // the theme.
    event.stopPropagation()
  }

  function commitName() {
    const name = draft.trim()
    if (name && name !== node.data.label) onrename(node.id, name)
  }

  /**
   * Renaming a stream is one request and a redraw, like every other
   * edit -- but it is worth knowing that it is not a label change. A
   * stream name *is* the wiring in difflow, so the server moves the
   * feed, both ends of any recycle and every port that reads or writes
   * it together, and refuses a name another stream already has, since
   * that would join the two rather than rename one.
   */
  function commitStreamName() {
    const name = draft.trim()
    if (!name || name === node.data.label) return
    onedit(() =>
      patch(`/api/stream/${encodeURIComponent(node.data.label)}`, { name }),
    )
  }

  // A mixer mixes however many streams it is handed, so its inlet count
  // belongs to the flowsheet and not to the class -- which makes it the
  // canvas's to change. Every other unit's ports are fixed by its
  // equations and there is nothing to offer.
  let variadic = $derived(Boolean(spec?.ports?.variadic))

  function addInlet() {
    onedit(() => post('/api/inlet', { unit: unit.name }))
  }

  /**
   * Declare a feed on an open inlet, at the flowsheet's own defaults.
   *
   * The canvas used to draw a feed box on every unwired inlet, and
   * selecting that box was the only way to reach this. Boxes for streams
   * nobody had declared were exactly the thing that made a fresh node
   * look already connected, so they are gone -- and the way in has to
   * live somewhere. It lives on the port that would carry the feed.
   *
   * The numbers come after: this declares the stream, the box appears,
   * and selecting it opens the form. Setting them here would be a second
   * feed form in a panel that already has one.
   */
  function feedInlet(stream) {
    onedit(() => post('/api/feed', { name: stream }))
  }

  const isOpen = (which, stream) => (which ?? []).includes(stream)

  function removeInlet(stream) {
    onedit(() => del('/api/inlet', { unit: unit.name, stream }))
  }

  function commitParam(field, raw) {
    const value = parse(raw, field.kind)
    if (value === field.value) return
    onedit(() =>
      patch(`/api/unit/${encodeURIComponent(unit.name)}`, {
        params: { [field.name]: value },
      }),
    )
  }

  let equations = $derived(
    (docs?.equations ?? spec?.equations ?? []).map((latex) => ({
      latex,
      ...render(latex),
    })),
  )

  /**
   * The code context this unit is waiting for, written out by the server.
   *
   * Fetched on demand rather than with the node: it is a page of Python
   * per red box, and most red boxes are answered by a `thermo` that is
   * about to be written for the first one. Kept once fetched, because
   * the panel shows it and applying it is a second, separate click --
   * you are meant to read the thing before running it.
   */
  let snippet = $state(null)

  // A new selection is a new question; the old answer would be about a
  // unit that is no longer on screen.
  $effect(() => { void node; snippet = null })

  async function writeBoilerplate() {
    const answer = await post('/api/boilerplate', {
      operation: node.data.operation, name: node.id,
    })
    snippet = answer?.ok ? answer : null
  }

  /** Apply the merged context: the snippet *added* to what is there. */
  function applySnippet() {
    if (!snippet) return
    oncontext(snippet.merged)
  }
</script>

<!--
  The ports, on the panel for the node that has them.

  A port with nothing joined to it is marked here as well as on the
  canvas, and an open INLET carries the way to declare a feed on it. It
  has to: the canvas draws a box only for a feed someone declared, so
  there is no longer a feed node to select for a stream that has never
  been one.
-->
{#snippet ports()}
  <h3>inlets</h3>
  <ul class:ports={variadic}>
    {#each node.data.inlets as s (s)}
      <li class:open={isOpen(node.data.openInlets, s)}>
        {s}
        {#if isOpen(node.data.openInlets, s)}
          <button class="link" disabled={busy || !species.length}
                  title={species.length
                           ? `declare ${s} as a feed, at the flowsheet's defaults`
                           : 'name the species in the header first'}
                  onclick={() => feedInlet(s)}>feed it</button>
        {/if}
        {#if variadic && node.data.inlets.length > 1}
          <button class="drop" title="remove this inlet"
                  disabled={busy} onclick={() => removeInlet(s)}>&minus;</button>
        {/if}
      </li>
    {/each}
  </ul>
  {#if variadic}
    <button class="add-port" disabled={busy} onclick={addInlet}>
      Add inlet
    </button>
  {/if}
  <h3>outlets</h3>
  <ul>
    {#each node.data.outlets as s (s)}
      <li class:open={isOpen(node.data.openOutlets, s)}>{s}</li>
    {/each}
  </ul>
{/snippet}

<aside class="inspector">
  {#if !node}
    <p class="hint">
      Drag an operation onto the canvas. Wire a unit's outlet to another
      unit's inlet. Select and press Delete to remove.
    </p>
  {:else if !isUnit}
    <h2>{node.data.label}</h2>
    <!-- An inlet with nothing on the other end is drawn as a feed node,
         because that is where a feed would go; saying "feed" about it
         would claim the flowsheet has an inlet it does not have yet. -->
    <p class="kind">{isFeed && !feed ? 'inlet, unfed' : node.data.kind}</p>

    <label>
      name
      <input value={draft} disabled={busy}
             oninput={(e) => (draft = e.currentTarget.value)}
             onblur={commitStreamName}
             onkeydown={(e) => e.key === 'Enter' && commitStreamName()} />
    </label>
    <p class="hint">
      The name is the wiring: renaming moves the feed, both ends of a
      recycle and every port that reads or writes it, together.
    </p>

    {#if !isFeed}
      <p class="hint">
        A stream, not an object the flowsheet holds: it follows from the
        units that make and read it.
      </p>
    {:else if !species.length}
      <p class="hint">
        Name the species in the header first: a feed is a flow per
        species, and there are none yet.
      </p>
    {:else}
      {#if !feed}
        <p class="hint">
          Nothing feeds this inlet yet. These are the flowsheet's own
          defaults &mdash; the numbers a tear stream starts from &mdash;
          so setting them unchanged is not a new guess.
        </p>
      {/if}

      <h3>feed</h3>
      <dl class="params">
        {#each feedForm as field (field.key)}
          <dt>{field.label} <span class="units">{field.units}</span></dt>
          <dd>
            <input
              value={feedDraft[field.key] ?? String(field.value)}
              disabled={busy}
              inputmode="decimal"
              oninput={(e) => (feedDraft = { ...feedDraft, [field.key]: e.currentTarget.value })}
              onkeydown={feedKeydown}
            />
          </dd>
        {/each}
      </dl>

      {#each pendingFeed.errors as problem (problem)}
        <p class="problem">{problem}</p>
      {/each}

      <div class="row">
        <button onclick={applyFeed} disabled={!canApplyFeed}>
          {feed ? 'Apply' : 'Set feed'}
        </button>
        {#if feed}
          <button class="danger" onclick={removeFeed} disabled={busy}>
            Remove feed
          </button>
        {/if}
      </div>
    {/if}
  {:else if pending}
    <!-- A red node: on the flowsheet, wired like any other, and with
         nothing behind its ports yet. The panel leads with what would
         finish it, because that is the only thing here that cannot be
         done by dragging. Its ports come after, since they are real and
         wiring them is what you are meant to do meanwhile. Parameters
         are not offered: they belong to a unit that does not exist yet,
         and they arrive with it. -->
    <h2>
      {pending.operation}
      {#if docs?.docs_url}
        <a class="doc-link" href={docs.docs_url} target="_blank"
           rel="noopener noreferrer"
           title="read about {pending.operation} in the documentation">docs &#8599;</a>
      {/if}
    </h2>
    <p class="kind waiting">not built yet &mdash; wire it now, solve later</p>

    <label>
      name
      <input value={draft} oninput={(e) => (draft = e.currentTarget.value)}
             onblur={commitName} onkeydown={(e) => e.key === 'Enter' && commitName()} />
    </label>

    <h3>waiting for</h3>
    <ul class="needs">
      {#each pending.needs as need (need)}<li>{need}</li>{/each}
    </ul>
    <p class="problem">{pending.hint}</p>

    {@render ports()}

    <!-- Saying what is missing answers "what"; this answers "what do I
         write". Two clicks, not one: the snippet invents numbers where
         it has to, marks them, and running it unread would put them in
         the model. -->
    {#if !snippet}
      <button class="wide" disabled={busy} onclick={writeBoilerplate}>
        Write the code for me
      </button>
    {:else}
      <h3>a starting point</h3>
      <pre class="snippet">{snippet.source}</pre>
      <p class="hint">
        Added to the code context, not replacing it. Read every
        <code>INVENTED</code>, <code>GUESSED</code> and
        <code>PLACEHOLDER</code> comment first &mdash; those are the
        numbers this cannot know.
      </p>
      <div class="row">
        <button class="primary" disabled={busy} onclick={applySnippet}>
          Add to the code context
        </button>
        <button disabled={busy} onclick={() => (snippet = null)}>Discard</button>
      </div>
    {/if}

    <button class="danger" onclick={() => ondelete(node.id)}>Delete node</button>
  {:else}
    <h2>
      {node.data.operation}
      {#if docs?.symbol && docs.symbol !== node.data.operation}
        <span class="symbol">{docs.symbol}</span>
      {/if}
      {#if docs?.docs_url}
        <!-- The panel below renders this unit's docstring, which says
             what the arguments are. This is the book, which says what
             the unit is for. -->
        <a class="doc-link" href={docs.docs_url} target="_blank"
           rel="noopener noreferrer"
           title="read about {node.data.operation} in the documentation">docs &#8599;</a>
      {/if}
    </h2>
    {#if spec?.description}<p class="kind">{spec.description}</p>{/if}

    <label>
      name
      <input value={draft} oninput={(e) => (draft = e.currentTarget.value)}
             onblur={commitName} onkeydown={(e) => e.key === 'Enter' && commitName()} />
    </label>

    {@render ports()}

    {#if extras.length}
      <h3>built with</h3>
      <dl class="params">
        {#each extras as extra (extra.name)}
          <dt>{extra.name}</dt>
          <dd><span class="fixed">{extra.text}</span></dd>
        {/each}
      </dl>
    {:else if spec?.constructor_extras?.length}
      <h3>built with</h3>
      <p class="hint">
        Needs {spec.constructor_extras.join(', ')} from the code context.
      </p>
    {/if}

    <h3>parameters</h3>
    {#if !fields.length}
      <p class="hint">This operation takes none.</p>
    {:else}
      <dl class="params">
        {#each fields as field (field.name)}
          <dt title={field.spec.type ?? ''}>
            {label(field.spec)}
            {#if field.spec.required}<span class="req" title="required">*</span>{/if}
          </dt>
          <dd>
            {#if !field.editable}
              <span class="fixed" title="{field.kind}: set in the code context or by a factory">
                {field.text}
              </span>
            {:else if field.kind === 'boolean'}
              <input type="checkbox" checked={field.value === true} disabled={busy}
                     onchange={(e) => commitParam(field, e.currentTarget.checked)} />
            {:else}
              <input value={field.value === null || field.value === undefined
                              ? '' : field.text}
                     placeholder={field.value === null ? 'not set' : ''}
                     disabled={busy}
                     onchange={(e) => commitParam(field, e.currentTarget.value)} />
            {/if}
            {#if field.spec.description}
              <span class="help">{field.spec.description}</span>
            {/if}
          </dd>
        {/each}
      </dl>
    {/if}

    {#if equations.length}
      <h3>equations</h3>
      {#each equations as eq (eq.latex)}
        {#if eq.html}
          <div class="eq">{@html eq.html}</div>
        {:else}
          <pre class="eq bad" title={eq.error}>{eq.latex}</pre>
        {/if}
      {/each}
    {/if}

    {#if docs?.numerical_method}
      <h3>method</h3>
      <p class="prose">{docs.numerical_method}</p>
    {/if}

    {#if docs?.assumptions?.length}
      <details>
        <summary>assumptions ({docs.assumptions.length})</summary>
        <ul class="prose">
          {#each docs.assumptions as a (a)}<li>{a}</li>{/each}
        </ul>
      </details>
    {/if}

    {#if docs?.html}
      <details>
        <summary>documentation</summary>
        <!-- The unit's own docstring, rendered by the server. -->
        <div class="doc">{@html docs.html}</div>
      </details>
    {/if}

    {#if docs?.references?.length}
      <details>
        <summary>references ({docs.references.length})</summary>
        <ul class="prose refs">
          {#each docs.references as r (r)}<li>{r}</li>{/each}
        </ul>
      </details>
    {/if}

    <button class="danger" onclick={() => ondelete(node.id)}>Delete unit</button>
  {/if}
</aside>

<style>
  .inspector {
    width: 19rem;
    flex: none;
    padding: 0.8rem;
    border-left: 1px solid var(--grid);
    background: var(--panel);
    overflow-y: auto;
    font-size: 0.8rem;
  }
  h2 { font-size: 0.9rem; margin: 0 0 0.3rem; }
  .symbol {
    font-weight: 400;
    color: var(--ink-soft);
    font-family: var(--mono, ui-monospace, monospace);
    font-size: 0.76rem;
  }
  h3 {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-soft);
    margin: 0.9rem 0 0.25rem;
  }
  .kind, .hint { color: var(--ink-soft); line-height: 1.45; }
  .kind { margin: 0 0 0.6rem; }
  label { display: block; color: var(--ink-soft); }
  input {
    display: block;
    width: 100%;
    margin-top: 0.2rem;
    padding: 0.3rem 0.4rem;
    font: inherit;
    border: 1px solid var(--line);
    border-radius: 5px;
    background: var(--surface);
    color: inherit;
  }
  input[type='checkbox'] { width: auto; margin-top: 0.3rem; }
  ul { margin: 0; padding-left: 1.1rem; }
  li { font-family: var(--mono, ui-monospace, monospace); font-size: 0.76rem; }

  /* A variadic unit's inlet list carries a remove per row, so the rows
     stop being bullets and become a small table of one column. */
  .ports { list-style: none; padding-left: 0; }
  .ports li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
    padding: 0.1rem 0;
  }
  .drop {
    flex: none;
    width: 1.2rem;
    line-height: 1.05rem;
    font: inherit;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface);
    color: var(--ink-soft);
    cursor: pointer;
    opacity: 0;
  }
  .ports li:hover .drop, .drop:focus-visible { opacity: 1; }
  .drop:hover { color: var(--bad); border-color: var(--bad); }
  .add-port {
    margin-top: 0.35rem;
    padding: 0.2rem 0.5rem;
    font: inherit;
    font-size: 0.75rem;
    border: 1px dashed var(--line);
    border-radius: 5px;
    background: none;
    color: var(--ink-soft);
    cursor: pointer;
  }
  .add-port:hover:not(:disabled) { border-style: solid; color: inherit; }

  /* A port with nothing attached, marked the way the canvas marks it:
     the same red, so the dot on the node and the row in the panel are
     recognisably about the same thing. The marker is a bullet rather
     than red text, because a red stream name reads as a bad name. */
  li.open { list-style: none; position: relative; padding-left: 0.85rem; }
  li.open::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0.42em;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--bad);
  }
  /* The way to declare a feed on an inlet nothing supplies. It sits on
     the port because there is no longer a feed box to select for a
     stream that has never been one. */
  .link {
    font: inherit;
    font-size: 0.72rem;
    padding: 0 0.25rem;
    margin-left: 0.35rem;
    border: 1px solid transparent;
    border-radius: 4px;
    background: none;
    color: var(--series);
    cursor: pointer;
  }
  .link:hover:not(:disabled) { border-color: var(--line); }
  .link:disabled { color: var(--ink-soft); cursor: default; }

  .params { margin: 0; }
  .params dt {
    font-family: var(--mono, ui-monospace, monospace);
    font-size: 0.74rem;
    color: var(--ink-soft);
    margin-top: 0.5rem;
  }
  .params dd { margin: 0.1rem 0 0; }
  .req { color: var(--accent); }
  .fixed {
    display: block;
    padding: 0.3rem 0.4rem;
    border: 1px dashed var(--line);
    border-radius: 5px;
    color: var(--ink-soft);
    font-family: var(--mono, ui-monospace, monospace);
    font-size: 0.74rem;
  }
  .help { display: block; color: var(--ink-soft); font-size: 0.72rem; margin-top: 0.15rem; }

  .eq { margin: 0.45rem 0; overflow-x: auto; font-size: 0.95rem; }
  .eq.bad {
    font-family: var(--mono, ui-monospace, monospace);
    font-size: 0.72rem;
    color: var(--bad);
    white-space: pre-wrap;
  }

  details { margin-top: 0.9rem; }
  summary {
    cursor: pointer;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-soft);
  }
  .prose, .doc { line-height: 1.5; color: var(--ink-soft); }
  .prose li, .refs li { font-family: inherit; font-size: 0.78rem; margin: 0.3rem 0; }
  p.prose { margin: 0.3rem 0 0; }

  .doc-link {
    float: right;
    font-size: 0.72rem;
    font-weight: 400;
    color: var(--ink-soft);
    text-decoration: none;
  }
  .doc-link:hover { color: var(--series); text-decoration: underline; }

  /* The rendered docstring: docutils markup we do not control. */
  .doc :global(p) { margin: 0.5rem 0; }
  .doc :global(pre) {
    overflow-x: auto;
    padding: 0.4rem;
    background: var(--surface);
    border-radius: 5px;
    font-size: 0.72rem;
  }
  .doc :global(dt) { font-weight: 600; margin-top: 0.5rem; }
  .doc :global(dd) { margin: 0.1rem 0 0 0.8rem; }
  .doc :global(ul) { padding-left: 1.1rem; }
  .doc :global(li) { font-family: inherit; font-size: 0.78rem; }
  .doc :global(.literal) {
    font-family: var(--mono, ui-monospace, monospace);
    font-size: 0.74rem;
  }

  .units {
    color: var(--ink-soft);
    font-weight: 400;
    font-size: 0.7rem;
  }
  .problem {
    margin: 0.35rem 0 0;
    color: var(--bad);
    font-size: 0.75rem;
  }

  /* The red-node panel. It says the same thing the node says, at
     length, and then offers the one thing that answers it. */
  .kind.waiting { color: var(--bad); }
  ul.needs { padding-left: 1.1rem; margin: 0.2rem 0; }
  ul.needs li {
    font-family: var(--mono, ui-monospace, monospace);
    font-size: 0.75rem;
    color: var(--bad);
  }
  button.wide { width: 100%; margin-top: 0.8rem; }
  .snippet {
    max-height: 18rem;
    overflow: auto;
    margin: 0.3rem 0 0;
    padding: 0.5rem;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 5px;
    font-family: var(--mono, ui-monospace, monospace);
    font-size: 0.7rem;
    line-height: 1.45;
    white-space: pre;
  }
  .row {
    display: flex;
    gap: 0.4rem;
    margin-top: 0.7rem;
  }
  /* Two buttons side by side: `.danger` is written for the full-width
     Delete at the foot of the panel, and its own width would fight the
     row. */
  .row button { flex: 1 1 auto; width: auto; margin-top: 0; }

  .danger {
    margin-top: 1.2rem;
    width: 100%;
    padding: 0.35rem;
    font: inherit;
    font-size: 0.79rem;
    border: 1px solid var(--line);
    border-radius: 5px;
    background: var(--surface);
    color: var(--bad);
    cursor: pointer;
  }
  .danger:hover { border-color: var(--bad); }
</style>
