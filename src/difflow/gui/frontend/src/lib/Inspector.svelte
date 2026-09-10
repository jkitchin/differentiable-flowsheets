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
    species = [],
    defaults = null,
    busy = false,
    onrename = () => {},
    ondelete = () => {},
    onedit = () => {},
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
</script>

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

    <h3>inlets</h3>
    <ul>{#each node.data.inlets as s (s)}<li>{s}</li>{/each}</ul>
    <h3>outlets</h3>
    <ul>{#each node.data.outlets as s (s)}<li>{s}</li>{/each}</ul>

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
