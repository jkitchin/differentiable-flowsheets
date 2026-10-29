<!--
  The live debugger: a solve that stops and waits for you.

  The Trace panel reads a solve after it happened; this one is in it.
  Start runs the solve on a copy of the flowsheet and stops before the
  first pass; the buttons step it (to the next tear iteration, the next
  unit, the next checkpoint) or let it run to a breakpoint. The canvas
  outlines the unit it is at. Anything the buttons do not cover goes in
  the command line: it is the same debugger as `difflow debug`, word for
  word (`help` lists it).

  Each request waits until the solve pauses again or for twenty seconds,
  whichever comes first; "still running" just means the page asks again.
  Pause stops a running solve at its next checkpoint --- a unit in the
  middle of its own calculation finishes it first.
-->
<script>
  import { post } from './api.js'
  import {
    absorb, banner, EVENTS, eventBreak, focusOf, initial, keepWaiting,
    parseBreakpoints, RESUME, resumes, resuming, unitBreak,
  } from './model/live.js'
  import { formatNumber } from './model/trace.js'

  let { unit = null, onfocus = () => {}, onclose = () => {} } = $props()

  let live = $state(initial())
  let busy = $state(false)          // a request in flight that is not a wait
  let waiting = $state(false)       // parked on /api/debug/wait
  let breaks = $state({ units: [], events: [], other: [] })
  let tear = $state(null)           // `p tear` at an iteration pause
  let draft = $state('')
  let history = $state([])
  let recall = $state(-1)
  let log                            // the transcript element

  let paused = $derived(live.status === 'paused')
  let running = $derived(live.status === 'running')
  let active = $derived(paused || running)

  $effect(() => onfocus(focusOf(live)))
  // Closing the panel stops the solve: nobody is left to step it.
  $effect(() => () => {
    onfocus(null)
    post('/api/debug/stop', {}).catch(() => {})
  })

  function show(reply, sent) {
    live = absorb(live, reply, sent)
    queueMicrotask(() => log?.scrollTo({ top: log.scrollHeight }))
  }

  /** Follow a reply until the solve pauses or ends: the long poll. */
  async function follow(reply) {
    waiting = true
    try {
      while (keepWaiting(reply) && live.status !== 'idle') {
        reply = await post('/api/debug/wait', {})
        show(reply)
      }
    } finally {
      waiting = false
    }
    if (live.status === 'paused') await afterPause()
  }

  /** What a pause shows without being asked: the tear, at an iteration. */
  async function afterPause() {
    tear = null
    if (live.pause?.checkpoint !== 'iteration') return
    const reply = await post('/api/debug/command', { cmd: 'print tear' })
    if (reply.ok && reply.event?.ok) tear = reply.event.data?.tear ?? null
  }

  // `blocking` marks the buttons busy while the request is out. A resume
  // does not: its reply may take the whole long-poll window, and Pause
  // and Stop have to stay pressable meanwhile.
  async function request(path, body, sent = null, blocking = true) {
    busy = blocking
    try {
      const reply = await post(path, body)
      show(reply, sent)
      return reply
    } catch (e) {
      show({ ok: false, error: String(e) }, sent)
      return null
    } finally {
      busy = false
    }
  }

  async function start(onError = false) {
    tear = null
    live = initial()
    const reply = await request('/api/debug/start', { on_error: onError })
    if (reply?.ok) {
      await refreshBreaks()
      await follow(reply)
    }
  }

  /** One command at the pause; a resume is followed to the next stop. */
  async function send(cmd) {
    if (!paused || !cmd.trim()) return
    tear = null
    const resume = resumes(cmd)
    if (resume) live = resuming(live)
    const reply = await request('/api/debug/command', { cmd }, cmd, !resume)
    if (reply?.ok && reply.event?.resumed) await follow(reply)
    else if (/^(b|break|tb|tbreak)\b/.test(cmd.trim())) await refreshBreaks()
    else if (live.status === 'paused' && live.pause?.checkpoint === 'iteration') await afterPause()
  }

  async function refreshBreaks() {
    const reply = await post('/api/debug/command', { cmd: 'break' })
    if (reply.ok && reply.event?.ok) breaks = parseBreakpoints(reply.event.data?.breakpoints)
  }

  /** Set or clear a breakpoint without writing to the transcript twice. */
  async function toggle(cmd) {
    await request('/api/debug/command', { cmd }, cmd)
    await refreshBreaks()
  }

  const pause = () => request('/api/debug/pause', {}, 'pause')
  async function stop() {
    await request('/api/debug/stop', {}, 'stop')
    live = { ...live, status: 'idle', pause: null }
    tear = null
  }

  function keydown(event) {
    if (event.key === 'Enter') {
      event.preventDefault()
      const cmd = draft
      if (cmd.trim()) history = [...history, cmd].slice(-100)
      draft = ''
      recall = -1
      send(cmd)
    } else if (event.key === 'ArrowUp' && history.length) {
      event.preventDefault()
      recall = recall < 0 ? history.length - 1 : Math.max(recall - 1, 0)
      draft = history[recall]
    } else if (event.key === 'ArrowDown' && recall >= 0) {
      event.preventDefault()
      recall = recall + 1 < history.length ? recall + 1 : -1
      draft = recall < 0 ? '' : history[recall]
    }
  }

  let unitBroken = $derived(!!unit && breaks.units.includes(unit))
</script>

<section class="debugger" aria-label="Live debugger">
  <header>
    <h2>Debug</h2>
    {#if !active}
      <button onclick={() => start(false)} disabled={busy} title="solve, stopping before the first pass">Start</button>
      <button onclick={() => start(true)} disabled={busy} title="solve freely; stop only where something fails">Run to error</button>
    {:else}
      {#each RESUME as r (r.key)}
        <button onclick={() => send(r.command)} disabled={!paused || busy} title={r.hint}>{r.label}</button>
      {/each}
      <button onclick={pause} disabled={!running || busy} title="stop at the next checkpoint">Pause</button>
      <button class="stop" onclick={stop} disabled={busy} title="stop the solve">Stop</button>
    {/if}
    <span class="status" class:run={running} class:pause={paused}>
      {running ? (waiting ? 'running…' : 'running') : live.status}
    </span>
    <span class="spacer"></span>
    <button onclick={onclose}>Close</button>
  </header>

  {#if active}
    <div class="breaks">
      <span class="label">break</span>
      {#if unit}
        <button class:on={unitBroken} onclick={() => toggle(unitBreak(unit, !unitBroken))}
                disabled={!paused || busy}
                title="pause before this unit on every pass">{unitBroken ? '● before' : '○ before'} {unit}</button>
      {:else}
        <span class="hint">select a unit to break before it</span>
      {/if}
      {#each breaks.units.filter((u) => u !== unit) as u (u)}
        <button class="on" onclick={() => toggle(unitBreak(u, false))} disabled={!paused || busy}
                title="remove">● {u} ×</button>
      {/each}
      <span class="sep"></span>
      {#each EVENTS as e (e.key)}
        <label><input type="checkbox" checked={breaks.events.includes(e.key)} disabled={!paused || busy}
                      onchange={(ev) => toggle(eventBreak(e.key, ev.currentTarget.checked))} /> {e.label}</label>
      {/each}
      {#each breaks.other as o (o)}<span class="other">{o}</span>{/each}
    </div>
  {/if}

  <div class="body">
    <div class="state">
      {#if live.pause}
        <p class="banner">{banner(live.pause)}</p>
        {#if tear?.length}
          <table>
            <thead><tr><th>tear</th><th>field</th><th>x</th><th>g(x)</th><th>|g − x|</th></tr></thead>
            <tbody>
              {#each tear.slice(0, 8) as r, i (r.stream + r.field)}
                <tr class:top={i === 0}><td>{r.stream}</td><td>{r.field}</td><td>{formatNumber(r.x)}</td><td>{formatNumber(r.g)}</td><td>{formatNumber(r.delta)}</td></tr>
              {/each}
            </tbody>
          </table>
          <p class="hint">try <code>set tear {tear[0].stream}.{tear[0].field} VALUE</code> to choose the next guess</p>
        {/if}
        {#each live.pause.watches ?? [] as w (w.target)}
          <pre class="watch">[{w.target}] {w.text}</pre>
        {/each}
      {:else if running}
        <p class="hint">the solve is running{waiting ? '; this page will show where it stops' : ''}</p>
      {:else if live.status === 'idle' && !live.log.length}
        <p class="hint">
          Start solves a copy of the flowsheet and stops before the first pass; step it by tear iteration or by
          unit, or set breakpoints and continue. Run to error stops only where a unit raises, a NaN is born, or
          the solve ends unconverged. The command line takes everything <code>difflow debug</code> does.
        </p>
      {/if}
    </div>
    <div class="console">
      <div class="log" bind:this={log}>
        {#each live.log as entry, i (i)}
          <pre class={entry.kind}>{entry.kind === 'command' ? `› ${entry.text}` : entry.text}</pre>
        {/each}
      </div>
      <input class="entry" bind:value={draft} onkeydown={keydown} disabled={!paused}
             placeholder={paused ? 'p STREAM, p UNIT, p history, break if residual<1e-6, set tear …, help' : ''}
             spellcheck="false" aria-label="debugger command" />
    </div>
  </div>
</section>

<style>
  .debugger {
    display: flex;
    flex-direction: column;
    height: 22rem;
    flex: none;
    border-top: 1px solid var(--grid);
    background: var(--panel);
  }
  header, .breaks {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding: 0.45rem 0.8rem;
    border-bottom: 1px dashed var(--grid);
  }
  .breaks { padding: 0.3rem 0.8rem; font-size: 0.76rem; }
  .breaks .label { color: var(--ink-soft); font-weight: 600; }
  .breaks button { font-size: 0.74rem; padding: 0.1rem 0.45rem; }
  .breaks button.on { color: var(--bad); border-color: var(--bad); }
  .breaks .sep { width: 1px; height: 1rem; background: var(--grid); }
  .breaks label { color: var(--ink-soft); display: inline-flex; gap: 0.2rem; align-items: center; }
  .breaks .other { color: var(--ink-soft); font-family: var(--mono, ui-monospace, monospace); }
  h2 { font-size: 0.85rem; margin: 0; }
  .spacer { flex: 1; }
  .stop { color: var(--bad); }
  .status { font-size: 0.74rem; color: var(--ink-soft); padding: 0.1rem 0.4rem; border-radius: 3px; border: 1px solid var(--grid); }
  .status.pause { color: var(--accent); border-color: var(--accent); }
  .status.run { color: var(--good); border-color: var(--good); }
  .hint { color: var(--ink-soft); font-size: 0.76rem; margin: 0.2rem 0; line-height: 1.5; }
  .body {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(18rem, 2fr) 3fr;
    gap: 0.8rem;
    padding: 0.5rem 0.8rem;
    background: var(--surface);
  }
  .state { overflow: auto; }
  .banner { font-weight: 600; font-size: 0.8rem; margin: 0 0 0.4rem; }
  table { border-collapse: collapse; font-size: 0.76rem; font-variant-numeric: tabular-nums; }
  th, td { padding: 0.1rem 0.6rem 0.1rem 0; text-align: left; }
  thead th { color: var(--ink-soft); font-weight: 600; border-bottom: 1px solid var(--grid); }
  tr.top td { font-weight: 650; }
  .console { display: flex; flex-direction: column; min-height: 0; }
  .log { flex: 1; min-height: 0; overflow: auto; }
  pre { margin: 0; font-size: 0.75rem; white-space: pre-wrap; font-family: var(--mono, ui-monospace, monospace); }
  pre.command { color: var(--accent); }
  pre.pause { font-weight: 600; margin-top: 0.3rem; }
  pre.error { color: var(--bad); }
  pre.finished { color: var(--good); font-weight: 600; }
  pre.watch { color: var(--ink-soft); }
  .entry { margin-top: 0.3rem; font-family: var(--mono, ui-monospace, monospace); font-size: 0.78rem; }
</style>
