<!--
  A Python prompt over the objects the editor is already holding.

  The other panels answer questions someone built a button for. This is
  for the rest: `fs` here is the flowsheet on the canvas, not a copy, so
  a gradient taken at this prompt is a gradient of the model on screen.

  Which also means a cell can move it. The server says when one did and
  the page redraws, because a canvas quietly describing a flowsheet that
  no longer exists is worse than no console at all.
-->
<script>
  import { get, post } from './api.js'
  import {
    append, describe, entry, GREETING, isBlank, prompt, recall, remember,
  } from './model/console.js'

  let { onchanged = () => {}, onclose = () => {} } = $props()

  let transcript = $state([])
  let history = $state([])
  let draft = $state('')
  let cursor = $state({ index: null, stash: '' })
  let busy = $state(false)
  let error = $state('')
  let scope = $state(null)
  let log                                     // the scrolling transcript

  // What is in scope before anything is typed, so the empty panel
  // is a list of what can be asked rather than a blank box.
  get('/api/console').then((s) => (scope = s)).catch(() => {})

  async function run() {
    if (busy || isBlank(draft)) return
    const source = draft
    busy = true
    error = ''
    try {
      const answer = await post('/api/console', { source })
      transcript = append(transcript, entry(source, answer))
      history = remember(history, source)
      draft = ''
      cursor = { index: null, stash: '' }
      if (answer.names) scope = { ...(scope ?? {}), defined: answer.names }
      // The canvas is stale the moment a cell touches `fs`.
      if (answer.changed) onchanged()
    } catch (e) {
      error = String(e)
    } finally {
      busy = false
      queueMicrotask(() => log?.scrollTo({ top: log.scrollHeight }))
    }
  }

  async function reset() {
    try {
      const answer = await post('/api/console/reset', {})
      transcript = append(transcript, {
        source: '', outputs: [], error: null, changed: false, note: 'reset',
      })
      scope = { ...(scope ?? {}), defined: answer.names ?? [] }
    } catch (e) {
      error = String(e)
    }
  }

  /**
   * Enter runs; Shift+Enter is a newline. Arrows walk the history only
   * from the edge of the text, so they still move the caret inside a
   * cell someone is editing.
   */
  function keydown(event) {
    const box = event.currentTarget
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      return run()
    }
    if (event.key === 'Tab') {
      event.preventDefault()
      const at = box.selectionStart
      draft = draft.slice(0, at) + '    ' + draft.slice(box.selectionEnd)
      return queueMicrotask(() => box.setSelectionRange(at + 4, at + 4))
    }
    const up = event.key === 'ArrowUp' && box.selectionStart === 0
    const down = event.key === 'ArrowDown' && box.selectionEnd === draft.length
    if (!up && !down) return
    const moved = recall(history, { ...cursor, draft }, up ? 'up' : 'down')
    if (moved.index === cursor.index && moved.source === draft) return
    event.preventDefault()
    draft = moved.source ?? ''
    cursor = { index: moved.index, stash: moved.stash }
  }
</script>

<section class="console">
  <header>
    <h2>Console</h2>
    <p class="hint">
      Python on the server, over the live model. A cell that loops
      forever holds it until the process is killed.
    </p>
    <span class="spacer"></span>
    <button onclick={reset} disabled={busy}>Reset</button>
    <button onclick={onclose}>Close</button>
  </header>

  {#if error}<p class="error">{error}</p>{/if}

  <div class="log" bind:this={log}>
    {#if !transcript.length}
      <p class="greeting">{GREETING}</p>
      {#if scope}
        <dl class="scope">
          {#each Object.entries(scope.live ?? {}) as [name, what] (name)}
            <dt>{name}</dt><dd>{what}</dd>
          {/each}
          {#if scope.bindings?.length}
            <dt>from the code context</dt><dd>{scope.bindings.join(', ')}</dd>
          {/if}
        </dl>
      {/if}
    {/if}

    {#each transcript as cell, i (i)}
      {#if cell.note}
        <p class="note">— {cell.note} —</p>
      {:else}
        <pre class="echo">{prompt(cell.source)}</pre>
        {#each cell.outputs as output, j (j)}
          {@const drawn = describe(output)}
          {#if drawn.kind === 'image'}
            <img alt="figure from the console" src={drawn.src} />
          {:else if drawn.kind === 'unknown'}
            <pre class="unknown">{drawn.text}</pre>
          {:else}
            <pre class={drawn.stream}>{drawn.text}</pre>
          {/if}
        {/each}
        {#if cell.error}<pre class="stderr">{cell.error}</pre>{/if}
        {#if cell.changed}<p class="note">the canvas was redrawn</p>{/if}
      {/if}
    {/each}
  </div>

  <div class="entry">
    <span class="caret" class:busy>{busy ? '···' : '>>>'}</span>
    <textarea
      bind:value={draft}
      onkeydown={keydown}
      disabled={busy}
      spellcheck="false"
      placeholder="fs.units"
      rows="2"
    ></textarea>
    <button onclick={run} disabled={busy || isBlank(draft)}>Run</button>
  </div>

  {#if scope?.defined?.length}
    <p class="defined">defined here: {scope.defined.join(', ')}</p>
  {/if}
</section>

<style>
  .console {
    display: flex;
    flex-direction: column;
    height: 22rem;
    flex: none;
    border-top: 1px solid var(--grid);
    background: var(--panel);
  }
  header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px dashed var(--grid);
  }
  h2 { font-size: 0.85rem; margin: 0; }
  .spacer { flex: 1; }
  .hint { color: var(--ink-soft); font-size: 0.76rem; margin: 0; }
  .error { color: var(--bad); font-size: 0.78rem; margin: 0; padding: 0.4rem 0.8rem; }

  .log {
    flex: 1;
    min-height: 0;
    overflow: auto;
    padding: 0.5rem 0.8rem;
    background: var(--surface);
  }
  .greeting { color: var(--ink-soft); font-size: 0.78rem; margin: 0 0 0.6rem; line-height: 1.5; }
  .scope {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.15rem 0.6rem;
    margin: 0;
    font-size: 0.76rem;
  }
  .scope dt { font-family: var(--mono, ui-monospace, monospace); color: var(--accent); }
  .scope dd { margin: 0; color: var(--ink-soft); }

  pre {
    margin: 0 0 0.3rem;
    font-family: var(--mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 0.76rem;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .echo { color: var(--ink-soft); margin-top: 0.4rem; }
  .stderr { color: var(--bad); }
  .unknown { color: var(--ink-soft); font-style: italic; }
  img { max-width: 100%; display: block; margin: 0.2rem 0 0.4rem; }
  .note { color: var(--ink-soft); font-size: 0.72rem; margin: 0 0 0.4rem; }

  .entry {
    display: flex;
    align-items: flex-start;
    gap: 0.4rem;
    padding: 0.5rem 0.8rem;
    border-top: 1px dashed var(--grid);
  }
  .caret {
    font-family: var(--mono, ui-monospace, monospace);
    font-size: 0.78rem;
    color: var(--accent);
    padding-top: 0.35rem;
  }
  .caret.busy { color: var(--ink-soft); }
  textarea {
    flex: 1;
    font-family: var(--mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 0.78rem;
    line-height: 1.45;
    resize: vertical;
    padding: 0.3rem 0.4rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface);
    color: inherit;
  }
  .defined {
    margin: 0;
    padding: 0 0.8rem 0.5rem;
    font-size: 0.72rem;
    color: var(--ink-soft);
  }
</style>
