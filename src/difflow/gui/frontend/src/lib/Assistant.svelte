<!--
  The assistant: ask about the open flowsheet, and see exactly what was
  asked on its behalf.

  The brief is the product here. `/api/context` assembles it out of the
  live model -- the catalog entry for the selected unit, the equations,
  the solve diagnostics, the docs sections that mention them -- and this
  panel shows it whole, next to the answer, always. That is the
  debugging affordance, and it is what makes the panel useful when the
  local model is weak or absent: copy the brief into a better one.

  Nothing here edits the flowsheet. An answer is model output and is
  labelled as such; every fact about a parameter in the brief came out
  of the catalog, not out of a model.
-->
<script>
  import { untrack } from 'svelte'

  import { get } from './api.js'
  import {
    KINDS, PROVIDERS, contextPath, destination, provider,
    readSettings, writeSettings,
  } from './model/assistant.js'

  let {
    kind = 'flowsheet',
    unit = null,
    operation = null,
    onclose = () => {},
  } = $props()

  // Seeded once: the caller's guess at what the question is about, from
  // what is selected. The panel is remounted every time it opens, so the
  // guess is fresh, and after that the choice is the user's.
  let choice = $state(untrack(() => kind))
  let question = $state('')
  let pack = $state(null)
  let error = $state('')
  let loading = $state(false)
  let showSettings = $state(false)
  let showBrief = $state(true)
  let copied = $state('')
  let settings = $state(readSettings(globalThis.localStorage))

  let active = $derived(provider(settings.provider))

  $effect(() => writeSettings(globalThis.localStorage, { ...settings }))

  /**
   * Fetch the brief.
   *
   * Retrieval over the docs is scored against the question, so the
   * brief for a typed question is not the brief for an empty one --
   * asking re-assembles it rather than reusing what is on screen.
   */
  async function brief(q = '') {
    loading = true
    error = ''
    copied = ''
    try {
      const answer = await get(contextPath({
        kind: choice, question: q, name: unit, operation,
      }))
      if (answer.ok === false) { pack = null; error = answer.error }
      else pack = answer
      return pack
    } catch (e) {
      pack = null
      error = String(e)
      return null
    } finally {
      loading = false
    }
  }

  // On open, and whenever the subject changes. Deliberately not keyed on
  // `question`: that would re-assemble the brief on every keystroke.
  $effect(() => {
    choice; unit; operation           // eslint-disable-line no-unused-expressions
    brief()
  })

  async function copy() {
    if (!pack) return
    try {
      await globalThis.navigator?.clipboard?.writeText(pack.prompt)
      copied = 'brief copied'
    } catch {
      copied = 'the browser refused the clipboard — select the text and copy it'
    }
  }

  const ask = (event) => {
    event?.preventDefault()
    brief(question)
  }
</script>

<section class="assistant">
  <header>
    <h2>Assistant</h2>
    <select bind:value={choice} aria-label="what to ask about">
      {#each KINDS as k (k.id)}<option value={k.id}>{k.label}</option>{/each}
    </select>
    <p class="hint">{KINDS.find((k) => k.id === choice)?.hint}</p>
    <span class="spacer"></span>
    <button onclick={copy} disabled={!pack}>Copy brief</button>
    <button onclick={() => (showSettings = !showSettings)}
            class:primary={active.id === 'none'}>Model…</button>
    <button onclick={onclose}>Close</button>
  </header>

  {#if showSettings}
    <div class="settings">
      <label>
        Answered by
        <select bind:value={settings.provider}>
          {#each PROVIDERS as p (p.id)}<option value={p.id}>{p.label}</option>{/each}
        </select>
      </label>
      {#if settings.provider === 'webllm'}
        <label>Model <input bind:value={settings.webllmModel} size="34" /></label>
      {:else if settings.provider === 'openai'}
        <label>Base URL <input bind:value={settings.baseUrl} size="26" /></label>
        <label>Model <input bind:value={settings.openaiModel} size="12" /></label>
      {/if}
      <p class="hint">{active.hint}</p>
    </div>
  {/if}

  <form onsubmit={ask}>
    <input
      bind:value={question}
      placeholder={choice === 'solve'
        ? 'why did the recycle not converge?'
        : 'what does this do, and what would I change?'}
      aria-label="question"
    />
    <button class="primary" type="submit" disabled={loading}>Ask</button>
  </form>

  <div class="panes">
    <div class="answer">
      {#if error}
        <p class="error">{error}</p>
      {:else if active.id === 'none'}
        <p class="hint">
          No model is configured, so no answer is generated here. The brief
          on the right is assembled from your live flowsheet — copy it into
          an assistant of your choosing, or pick one under <b>Model…</b>.
        </p>
      {:else}
        <p class="hint">Ask a question to get an answer.</p>
      {/if}
    </div>

    {#if showBrief}
      <div class="brief">
        <div class="briefhead">
          <b>Context sent</b>
          {#if pack}<span class="hint">{pack.tokens} tokens</span>{/if}
          <span class="spacer"></span>
          <button class="link" onclick={() => (showBrief = false)}>hide</button>
        </div>
        {#if loading && !pack}
          <p class="hint">assembling…</p>
        {:else if pack}
          <pre>{pack.prompt}</pre>
        {/if}
      </div>
    {:else}
      <button class="link show" onclick={() => (showBrief = true)}>
        show the context sent
      </button>
    {/if}
  </div>

  <footer>
    <span class="hint" class:remote={!active.local}>{destination(settings)}</span>
    {#each pack?.notes ?? [] as n (n)}<span class="hint">· {n}</span>{/each}
    {#if copied}<span class="hint">· {copied}</span>{/if}
  </footer>
</section>

<style>
  .assistant {
    display: flex;
    flex-direction: column;
    height: 24rem;
    flex: none;
    border-top: 1px solid var(--grid);
    background: var(--panel);
  }
  header, .settings, form, footer {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.45rem 0.8rem;
  }
  header { padding-top: 0.55rem; }
  footer { padding-bottom: 0.6rem; font-size: 0.76rem; }
  h2 { font-size: 0.85rem; margin: 0; }
  .spacer { flex: 1; }
  .hint { color: var(--ink-soft); font-size: 0.76rem; margin: 0; }
  /* The one sentence a user needs before typing, when it says the brief
     is about to leave the machine. */
  .remote { color: var(--accent); }
  .settings { border-bottom: 1px dashed var(--grid); flex-wrap: wrap; }
  .settings label { font-size: 0.78rem; color: var(--ink-soft); }
  form { padding-bottom: 0.2rem; }
  form input { flex: 1; }
  input, select {
    font: inherit;
    font-size: 0.8rem;
    padding: 0.25rem 0.4rem;
    border: 1px solid var(--line);
    border-radius: 5px;
    background: var(--surface);
    color: inherit;
  }

  .panes { display: flex; flex: 1; min-height: 0; gap: 0.8rem; padding: 0.3rem 0.8rem; }
  .answer, .brief {
    flex: 1;
    min-width: 0;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
    padding: 0.5rem 0.6rem;
  }
  .briefhead { display: flex; align-items: baseline; gap: 0.5rem; font-size: 0.76rem; }
  .brief pre {
    margin: 0.4rem 0 0;
    white-space: pre-wrap;
    word-break: break-word;
    font: 0.73rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--ink-soft);
  }
  .show { align-self: flex-start; }
  .link {
    background: none;
    border: none;
    padding: 0;
    color: var(--ink-soft);
    font-size: 0.76rem;
    text-decoration: underline;
    cursor: pointer;
  }
  .error { margin: 0; color: var(--bad); font-size: 0.78rem; }
</style>
