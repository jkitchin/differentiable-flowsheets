<!--
  The code context: Python that runs on the server, defining the objects
  the flowsheet refers to by name.

  This is the answer to the half of the catalog a palette cannot build.
  A Flash needs a `thermo`, a CSTR needs a rate law; neither is data, so
  neither can come out of a form. One `thermo = IdealThermo(...)` here
  and the palette can place both -- and the same snippet is emitted as
  the preamble of the exported script, so what is exported is what ran.
-->
<script>
  import { untrack } from 'svelte'

  import CodeEditor from './CodeEditor.svelte'

  let {
    source = '',
    names = [],
    error = '',
    busy = false,
    onapply = () => {},
    // Cmd-S with a draft not yet applied: apply it, then save. Left to
    // the window's Cmd-S, the flowsheet was saved WITHOUT the text on
    // screen, and the "saved" note said the opposite. Called with null
    // for a save with nothing to apply.
    onsave = () => {},
    onclose = () => {},
  } = $props()

  function keydown(event) {
    const mod = event.metaKey || event.ctrlKey
    if (!mod || event.altKey || event.shiftKey || event.key.toLowerCase() !== 's') return
    if (!dirty) return       // nothing unapplied: the window's save is right
    event.preventDefault()
    event.stopPropagation()
    // Busy (a solve, a console cell): remember the keystroke and save
    // when it is done. Dropping it said nothing, and the user had every
    // reason to think the file was saved.
    if (busy) { saveQueued = true; saveCancelled = false }
    else onsave(draft)
  }

  // Seeded once, at mount, and deliberately so: the panel exists only
  // while it is open, and the text in it is the user's, not the
  // server's. `dirty` is what tracks the two drifting apart.
  let draft = $state(untrack(() => source))
  let dirty = $derived(draft !== source)

  // The source can change while the panel is open -- Undo, Open, an
  // example, a console cell. An untouched draft follows it; seeding only
  // at mount left the old text there, marked "not applied", one click
  // from writing it back over the change. A draft the user has edited is
  // theirs and stays, but says it was written against an older version.
  let seen = untrack(() => source)
  let behind = $state(false)
  $effect(() => {
    const next = source
    untrack(() => {
      if (next === seen) return
      if (draft === seen) draft = next
      else behind = true
      seen = next
    })
  })
  $effect(() => { if (!dirty) behind = false })

  // The queued Cmd-S (see `keydown`). Declared after the effects that
  // set `behind`, so in a shared flush it reads their answer.
  // When it fires: a draft still unapplied is applied and saved; one
  // applied meanwhile (the Apply request is itself what made the panel
  // busy) or reverted is just a save, `onsave(null)` -- the keystroke was
  // stopped, so the window's save never ran. A draft the run left
  // `behind` (an Open, an Undo) was written against a flowsheet that is
  // gone: saving it would put the old code into the new file, so the
  // save is cancelled, and says so.
  let saveQueued = $state(false)
  let saveCancelled = $state(false)
  $effect(() => {
    if (busy || !saveQueued) return
    untrack(() => {
      saveQueued = false
      if (dirty && behind) saveCancelled = true
      else onsave(dirty ? draft : null)
    })
  })
  $effect(() => { if (!behind) saveCancelled = false })

  const STARTER = `from difflow import IdealThermo, get_species_data, mass_action_kinetics

SPECIES = ["water", "ethanol"]
thermo = IdealThermo({s: get_species_data(s) for s in SPECIES})

# A rate law built from data: rate_fn, stoich and rate_params arrive
# together, and a reactor dropped from the palette picks them up.
kin = mass_action_kinetics([{
    "equation": "water -> ethanol",
    "reactants": {"water": 1.0}, "products": {"ethanol": 1.0},
    "rate_params": {"A": 1.0e3, "Ea": 40_000.0, "n": 0.0},
}], SPECIES)
`

</script>

<section class="context" onkeydown={keydown}>
  <header>
    <h2>Code context</h2>
    <p class="hint">
      Python evaluated on the server. What it defines, the flowsheet can
      use by name — and the exported script carries it verbatim.
    </p>
    <span class="spacer"></span>
    {#if !draft.trim()}
      <button onclick={() => (draft = STARTER)}>Insert an example</button>
    {/if}
    <button class="primary" disabled={busy || !dirty}
            title="apply (Cmd-Enter in the editor)"
            onclick={() => onapply(draft)}>Apply</button>
    <button onclick={onclose}>Close</button>
  </header>

  <div class="body">
    <CodeEditor
      value={draft}
      label="code context"
      placeholder="thermo = IdealThermo(...)"
      onchange={(text) => (draft = text)}
      onsubmit={(text) => { if (!busy && text !== source) onapply(text) }}
    />
  </div>

  <footer>
    {#if saveQueued}
      <p class="names">will apply and save once the current run finishes</p>
    {:else if saveCancelled}
      <p class="stale" role="alert">Not saved: the code context changed while the save waited.</p>
    {/if}
    {#if error}
      <p class="error">{error}</p>
    {:else if names.length}
      <p class="names">
        defines {#each names as n, i (n)}<code>{n}</code>{i < names.length - 1 ? ', ' : ''}{/each}
      </p>
    {:else}
      <p class="hint">Nothing defined yet.</p>
    {/if}
    {#if behind}
      <p class="stale" role="alert">
        The code context changed since this draft was started.
        <button onclick={() => (draft = source)}>Discard the draft</button>
      </p>
    {:else if dirty && !error}<p class="hint">Not applied.</p>{/if}
  </footer>
</section>

<style>
  .context {
    display: flex;
    flex-direction: column;
    height: 16rem;
    flex: none;
    border-top: 1px solid var(--grid);
    background: var(--panel);
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    padding: 0.5rem 0.8rem;
  }
  h2 { font-size: 0.85rem; margin: 0; }
  .spacer { flex: 1; }
  .hint { color: var(--ink-soft); font-size: 0.76rem; margin: 0; }
  .body {
    display: flex;
    flex: 1;
    min-height: 0;
    margin: 0 0.8rem;
  }
  footer {
    display: flex;
    gap: 0.8rem;
    padding: 0.4rem 0.8rem 0.6rem;
    font-size: 0.76rem;
  }
  .names { margin: 0; color: var(--ink-soft); }
  .stale { margin: 0; color: var(--bad); }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .error { margin: 0; color: var(--bad); font-family: ui-monospace, monospace; }
</style>
