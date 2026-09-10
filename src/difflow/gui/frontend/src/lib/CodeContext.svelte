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

  let {
    source = '',
    names = [],
    error = '',
    busy = false,
    onapply = () => {},
    onclose = () => {},
  } = $props()

  // Seeded once, at mount, and deliberately so: the panel exists only
  // while it is open, and the text in it is the user's, not the
  // server's. `dirty` is what tracks the two drifting apart.
  let draft = $state(untrack(() => source))
  let dirty = $derived(draft !== source)

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

  /** Tab indents rather than leaving the box: this is a code editor. */
  function keydown(event) {
    if (event.key === 'Tab') {
      event.preventDefault()
      const box = event.currentTarget
      const at = box.selectionStart
      draft = draft.slice(0, at) + '    ' + draft.slice(box.selectionEnd)
      queueMicrotask(() => box.setSelectionRange(at + 4, at + 4))
    }
  }
</script>

<section class="context">
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
            onclick={() => onapply(draft)}>Apply</button>
    <button onclick={onclose}>Close</button>
  </header>

  <textarea
    spellcheck="false"
    value={draft}
    oninput={(e) => (draft = e.currentTarget.value)}
    onkeydown={keydown}
    placeholder="thermo = IdealThermo(...)"
  ></textarea>

  <footer>
    {#if error}
      <p class="error">{error}</p>
    {:else if names.length}
      <p class="names">
        defines {#each names as n, i (n)}<code>{n}</code>{i < names.length - 1 ? ', ' : ''}{/each}
      </p>
    {:else}
      <p class="hint">Nothing defined yet.</p>
    {/if}
    {#if dirty && !error}<p class="hint">Not applied.</p>{/if}
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
  textarea {
    flex: 1;
    min-height: 0;
    margin: 0 0.8rem;
    padding: 0.5rem 0.6rem;
    resize: none;
    font: 0.79rem/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
    tab-size: 4;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
    color: inherit;
  }
  footer {
    display: flex;
    gap: 0.8rem;
    padding: 0.4rem 0.8rem 0.6rem;
    font-size: 0.76rem;
  }
  .names { margin: 0; color: var(--ink-soft); }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .error { margin: 0; color: var(--bad); font-family: ui-monospace, monospace; }
</style>
