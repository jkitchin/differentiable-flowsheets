<!--
  A Python editor, or with `readonly` a viewer: CodeMirror 6 with Python
  highlighting, line numbers, bracket matching, undo and auto-indent.

  The text is the caller's. `value` seeds the editor and is pushed in
  again only when it changes from outside (an "Insert an example" click,
  a regenerated script); every keystroke goes back out through
  `onchange`, so the caller's `draft` stays the one copy of what is typed.

  CodeMirror itself is loaded on first use (`cm.js`). Until it arrives ---
  a moment, from localhost --- the text is shown plain, so a panel never
  opens empty.
-->
<script>
  import { onMount } from 'svelte'

  let {
    value = '',
    readonly = false,
    placeholder = '',
    label = 'code',
    onchange = () => {},
    // Mod-Enter: the editor's "run this", as the console already has it.
    onsubmit = () => {},
  } = $props()

  let host = $state(null)
  let view = $state(null)
  let cm = null

  onMount(() => {
    let gone = false
    import('./cm.js').then((mod) => {
      if (gone) return
      cm = mod
      view = mod.createEditor(host, {
        doc: value, readonly, placeholder, label, onchange, onsubmit,
      })
    })
    return () => { gone = true; view?.destroy() }
  })

  // Text set from outside replaces the document. Typing does not come back
  // through here: `onchange` has already made `value` equal to what the
  // editor holds, so `setDoc` finds nothing to do.
  $effect(() => {
    const next = value
    if (view && cm) cm.setDoc(view, next)
  })
</script>

<div class="editor" bind:this={host}>
  {#if !view}<pre class="plain">{value}</pre>{/if}
</div>

<style>
  .editor {
    flex: 1;
    min-height: 0;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
  }
  .editor:focus-within { border-color: var(--ink-soft); }
  .plain {
    margin: 0;
    padding: 0.4rem 0.6rem;
    font: 0.79rem/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: pre;
    overflow: auto;
    height: 100%;
  }
</style>
