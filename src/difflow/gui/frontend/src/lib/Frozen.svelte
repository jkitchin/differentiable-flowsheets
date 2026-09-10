<!--
  A published flowsheet: the frozen half of the editor.

  Same canvas, same graph model, same palette -- the difference is that
  nothing can be changed and nothing can be solved. `difflow.publish`
  bakes in the topology, the catalog entries for the operations used, and
  a grid of solved operating points; the page reads them and never
  reaches out. That is the whole bargain: no server, no Python, no JAX,
  and nothing to rot next to a paper.

  Before this, a published page carried the sliders and the chart and
  said nothing at all about the flowsheet they belonged to.
-->
<script>
  import Canvas from './Canvas.svelte'
  import Details from './Details.svelte'
  import Sweep from './Sweep.svelte'

  let { data } = $props()
  let selected = $state(null)
</script>

{#if data.topology}
  <section class="flowsheet panel">
    <h2>Flowsheet</h2>
    <div class="stage">
      <div class="slot">
        <!-- The catalog is passed for the same reason `Details` gets it:
             it is where an operation's category comes from, and the
             category is what picks a symbol for a plugin's own unit. A
             published page without it draws every node as a box. -->
        <Canvas document={data.topology} positions={data.topology.view?.nodes}
                catalog={data.catalog}
                readonly onselect={(node) => (selected = node)} />
      </div>
      <Details node={selected} topology={data.topology} catalog={data.catalog} />
    </div>
  </section>
{/if}

<Sweep sweep={data.sweep} version={data.version} points={data.n_points} />

<style>
  .panel {
    background: var(--panel); border: 1px solid var(--grid);
    border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: 1.5rem;
  }
  h2 {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--ink-soft); margin: 0 0 0.9rem; font-weight: 600;
  }
  /* A figure in a document, so it needs a height of its own: the editor's
     canvas takes one from the window and this one cannot. */
  .stage {
    display: flex; height: 26rem; background: var(--surface);
    border: 1px solid var(--grid); border-radius: 8px; padding: 0.5rem;
  }
  .slot { flex: 1; min-width: 0; }
  @media (max-width: 46rem) { .stage { height: 20rem; } }
</style>
