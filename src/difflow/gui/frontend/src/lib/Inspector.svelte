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
  import { get, patch } from './api.js'
  import { render } from './math.js'
  import { classify, label, parse } from './model/params.js'

  let {
    node = null,
    unit = null,
    spec = null,
    busy = false,
    onrename = () => {},
    ondelete = () => {},
    onedit = () => {},
  } = $props()

  let draft = $state('')
  let docs = $state(null)
  let isUnit = $derived(node?.type === 'unit')

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
    <p class="kind">{node.data.kind}</p>
    <p class="hint">
      A stream, not an object the flowsheet holds: it follows from the
      units that make and read it.
    </p>
  {:else}
    <h2>
      {node.data.operation}
      {#if docs?.symbol && docs.symbol !== node.data.operation}
        <span class="symbol">{docs.symbol}</span>
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
