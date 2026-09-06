<!--
  One unit on the canvas: its name, its operation, and one handle per
  port. The handles are named `in:<stream>` / `out:<stream>` so an edge
  lands on the port that actually carries that stream rather than on the
  middle of the box, which is the whole reason to draw ports at all.
-->
<script>
  import { Handle, Position } from '@xyflow/svelte'

  let { data } = $props()
</script>

<div class="unit">
  <div class="name">{data.label}</div>
  <div class="op">{data.operation ?? ''}</div>

  {#each data.inlets as stream, i (stream)}
    <Handle
      type="target"
      position={Position.Left}
      id={`in:${stream}`}
      style={`top:${((i + 1) * 100) / (data.inlets.length + 1)}%`}
    />
  {/each}
  {#each data.outlets as stream, i (stream)}
    <Handle
      type="source"
      position={Position.Right}
      id={`out:${stream}`}
      style={`top:${((i + 1) * 100) / (data.outlets.length + 1)}%`}
    />
  {/each}
</div>

<style>
  .unit {
    min-width: 9rem;
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--surface);
    text-align: center;
  }
  .name { font-weight: 650; letter-spacing: -0.01em; }
  .op { font-size: 0.72rem; color: var(--ink-soft); }
</style>
