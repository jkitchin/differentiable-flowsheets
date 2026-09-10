<!--
  A feed on the left or a dangling product on the right. Not a unit:
  difflow has no node for these, they are stream names with nothing on
  one end, and drawing them is how an unconnected inlet stays visible.

  Drawn as a chevron pointing the way the material goes, so a feed and a
  product are told apart by shape rather than only by colour -- and so
  neither can be mistaken for a unit, which is the confusion a second
  rounded rectangle would invite.
-->
<script>
  import { Handle, Position } from '@xyflow/svelte'

  let { data } = $props()
  let feed = $derived(data.kind === 'feed')
</script>

<div class="stream" class:feed class:product={!feed}>
  <svg class="arrow" viewBox="0 0 14 20" width="11" height="16" aria-hidden="true">
    {#if feed}
      <path d="M1 1l11 9-11 9z" />
    {:else}
      <path d="M13 1L2 10l11 9z" />
    {/if}
  </svg>
  <span class="name" title={data.label}>{data.label}</span>

  {#if feed}
    <Handle type="source" position={Position.Right} id={`out:${data.label}`} />
  {:else}
    <Handle type="target" position={Position.Left} id={`in:${data.label}`} />
  {/if}
</div>

<style>
  .stream {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    max-width: 11rem;
    padding: 0.3rem 0.65rem;
    border: 1px solid var(--node-line);
    border-radius: 999px;
    background: var(--node-fill);
    box-shadow: 0 1px 2px var(--node-shadow);
  }
  /* A product's chevron points the same way the material moves, so it
     sits after the name rather than before it. */
  .product { flex-direction: row-reverse; }

  .arrow { flex: 0 0 auto; fill: currentColor; stroke: none; }
  .feed { color: var(--series); }
  .product { color: var(--good); }
  .feed { border-color: color-mix(in oklab, var(--series) 45%, var(--node-line)); }
  .product { border-color: color-mix(in oklab, var(--good) 45%, var(--node-line)); }

  /* The name is the content; the colour belongs to the chevron. */
  .name {
    color: var(--ink);
    font-weight: 600;
    font-size: 0.8rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
