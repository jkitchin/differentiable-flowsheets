<!--
  A feed on the left or a dangling product on the right. Not a unit:
  difflow has no node for these, they are stream names with nothing on
  one end, and drawing them is how an unconnected inlet stays visible.
-->
<script>
  import { Handle, Position } from '@xyflow/svelte'

  let { data } = $props()
  let feed = $derived(data.kind === 'feed')
</script>

<div class="stream" class:feed class:product={!feed}>
  <div class="name">{data.label}</div>
  <div class="kind">{data.kind}</div>
  {#if feed}
    <Handle type="source" position={Position.Right} id={`out:${data.label}`} />
  {:else}
    <Handle type="target" position={Position.Left} id={`in:${data.label}`} />
  {/if}
</div>

<style>
  .stream {
    min-width: 7rem;
    padding: 0.4rem 0.7rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--panel);
    text-align: center;
  }
  .feed { border-color: var(--series); }
  .product { border-color: var(--good); }
  .name { font-weight: 600; font-size: 0.85rem; }
  .kind {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--ink-soft);
  }
</style>
