<!--
  The canvas, wrapped in the flow provider.

  Two components for one canvas because of how `useSvelteFlow` finds the
  flow: through a context that `SvelteFlowProvider` sets. A component
  cannot read a context its own child sets, so the surface -- which needs
  `screenToFlowPosition` to turn a drop into a coordinate, and
  `getViewport`/`setViewport` to re-centre when the drawer opens -- has to
  sit inside the provider rather than be the thing that renders it.

  Everything else stays where it was: props pass straight through, and
  `Surface.svelte` is where the gestures are.
-->
<script>
  import { SvelteFlowProvider } from '@xyflow/svelte'

  import Surface from './Surface.svelte'

  let props = $props()
</script>

<SvelteFlowProvider>
  <Surface {...props} />
</SvelteFlowProvider>
