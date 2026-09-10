/**
 * What a right-click on a node offers.
 *
 * Pure, and here rather than inline in the menu component, for the same
 * reason the rest of this directory is: a menu item that is enabled when
 * it should not be, or that names the wrong node, looks perfectly fine on
 * screen. The failure is silent until it is clicked.
 *
 * Every item is a shortcut to something the editor could already do --
 * the palette's documentation link, the code-context drawer, the
 * inspector's Delete. Nothing here is a new capability, and nothing here
 * talks to the server; the actions are handed in by `App`, which is where
 * the flowsheet's verbs live.
 */

import { parseNodeId } from './edit.js'

/** Where a `feed:`/`product:` id came from, for the menu's heading. */
function heading(parsed) {
  if (!parsed) return ''
  if (parsed.kind === 'unit') return parsed.name
  return `${parsed.kind} ${parsed.name}`
}

/**
 * The items for a right-click on `node`, in order.
 *
 * @param {object} args
 * @param {object} args.node      the @xyflow node that was clicked
 * @param {object} [args.catalog] the served catalog, for `docs_url`
 * @param {object} [args.actions] `{docs, ask, context, copy, remove}`
 * @returns {{title: string, items: Array}} heading and items, `items`
 *   empty when the node is not one this knows about --- the caller shows
 *   no menu at all rather than an empty box.
 *
 * `docs` is a URL rather than a callback: the item is a link, the caller
 * decides how to open it, and this stays the one place that knows the
 * catalog carries it. An operation with no documentation page still gets
 * the row, disabled and saying so --- a row that vanishes reads as a
 * menu that moved, and the next item down gets clicked instead.
 */
export function nodeMenu({ node, catalog = {}, actions = {} } = {}) {
  const parsed = parseNodeId(node?.id)
  if (!parsed) return { title: '', items: [] }

  const shared = [
    {
      label: 'Code context',
      hint: 'the Python evaluated before the flowsheet is built',
      run: actions.context,
    },
    {
      label: 'Copy name',
      hint: 'to paste into a lever, a spec or the console',
      run: () => actions.copy?.(parsed.name),
    },
  ]

  // A feed or a product is drawn *from* the topology rather than held by
  // the flowsheet, so the two rows that act on a unit have nothing to
  // act on: there is no operation to document and no object to delete.
  if (parsed.kind !== 'unit') {
    return { title: heading(parsed), items: shared }
  }

  const operation = node?.data?.operation
  const url = operation ? catalog[operation]?.docs_url : null

  return {
    title: heading(parsed),
    items: [
      {
        label: 'Documentation',
        note: url ? '↗' : 'none yet',
        hint: url || 'this operation has no page in the book',
        disabled: !url,
        run: () => actions.docs?.(url),
      },
      {
        label: operation ? `Ask about ${operation}` : 'Ask about this',
        hint: 'open the assistant on this unit',
        run: actions.ask,
      },
      { separator: true },
      ...shared,
      { separator: true },
      {
        label: 'Delete',
        note: 'Del',
        danger: true,
        run: () => actions.remove?.(parsed.name),
      },
    ],
  }
}
