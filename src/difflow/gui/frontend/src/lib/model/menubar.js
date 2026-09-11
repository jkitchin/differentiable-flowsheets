/**
 * What the header has to decide: its menus, and how much of the path
 * there is room to say.
 *
 * ## The menus
 *
 * The header had grown to fifteen controls in a row, which is what a
 * toolbar becomes when every panel earns a button: the two that matter
 * on any given day sit in a hedge of the eleven that do not. Here it is
 * three words and one button, and the arrangement is the whole idea ---
 * **File** writes the flowsheet out, **View** decides what is on the
 * screen, **Help** leaves the editor, and **Solve** stays a button
 * because it is the verb the editor exists for.
 *
 * Nothing here knows how to do anything. Every row's `run` is handed in,
 * which keeps the arrangement testable without a browser and keeps the
 * flowsheet's vocabulary in `App.svelte` beside the rest of it. The item
 * shape is `ContextMenu`'s, so the right-click menu on a node and the
 * menus up here are one component and one keyboard model.
 */

// A row that leaves the editor says so, in the glyph the palette uses.
const AWAY = '↗'

/**
 * A path with its middle taken out: `…/scratchpad/plant.json`.
 *
 * The header names the file it is editing, and an absolute path to a
 * file in a project two directories deep inside a Dropbox folder is
 * longer than the rest of the header put together --- it wrapped the
 * header onto a second line, which is a strange price to pay for a
 * string nobody reads past the end of. The tail is the half that
 * identifies the file, so the tail is what is kept; the whole thing
 * stays in the element's `title`.
 *
 * Both separators, because the editor runs wherever Python does.
 */
export function shortPath(path, keep = 2) {
  const parts = String(path ?? '').split(/[\\/]+/).filter(Boolean)
  if (parts.length <= keep) return String(path ?? '')
  return `…/${parts.slice(-keep).join('/')}`
}

/**
 * @param state.exporting  which file is being written, if any --- the
 *   four export rows are one server round trip and a raster, so a second
 *   click while the first is still working would write the file twice.
 * @param state.panels  which drawers are open, so their rows can be ticked
 * @param state.links  the project's URLs, as `/api/about` reported them
 */
export function menuBar({
  path = '',
  doc = null,
  busy = false,
  exporting = '',
  contextError = false,
  portLabels = false,
  dark = false,
  panels = {},
  links = {},
  actions = {},
} = {}) {
  // Every export reads the flowsheet, three of them through the server.
  const noExport = busy || !doc || !!exporting
  const file = (kind, label, note) => ({
    label,
    note: exporting === kind ? '…' : note,
    disabled: noExport,
    run: () => actions.export?.(kind),
  })

  return [
    {
      id: 'file',
      label: 'File',
      items: [
        {
          label: 'Save',
          note: '⌘S',
          hint: path || 'this editor was opened without a file to save to',
          disabled: busy || !path,
          run: actions.save,
        },
        {
          label: 'Reload',
          hint: 'draw the flowsheet again from what the server holds',
          disabled: busy,
          run: actions.reload,
        },
        { separator: true },
        file('py', 'Python script', 'codegen'),
        file('json', 'Flowsheet JSON', 'reloadable'),
        file('svg', 'Diagram SVG', 'your layout'),
        file('png', 'Diagram PNG', '2x'),
        { separator: true },
        {
          label: 'Quit',
          hint: 'stop the editor and free the port; it asks once more',
          danger: true,
          run: actions.quit,
        },
      ],
    },
    {
      id: 'view',
      label: 'View',
      items: [
        {
          label: 'Results',
          on: !!panels.results,
          hint: 'the solved streams, and derivatives of them',
          run: actions.results,
        },
        {
          label: 'Code context',
          on: !!panels.context,
          // The one row that can be *wrong* rather than merely closed:
          // a snippet that raised leaves every unit that wanted a name
          // from it unbuilt, and the red nodes on the canvas do not say
          // where to go and fix it.
          note: contextError ? '!' : '',
          hint: contextError
            ? 'the snippet did not run --- open it to see why'
            : 'the Python the flowsheet carries',
          run: actions.context,
        },
        {
          label: 'Console',
          on: !!panels.console,
          hint: 'a Python prompt in the editor process',
          run: actions.console,
        },
        {
          label: 'Planning',
          on: !!panels.planning,
          hint: 'delta-base planning over this flowsheet',
          run: actions.planning,
        },
        {
          label: 'Ask difflow',
          on: !!panels.assistant,
          hint: 'the assistant, on whatever is selected',
          run: actions.assistant,
        },
        { separator: true },
        {
          label: 'Port names',
          note: 'L',
          on: !!portLabels,
          hint: 'name every port beside its handle',
          run: actions.portLabels,
        },
        {
          label: 'Dark theme',
          note: 'T',
          on: !!dark,
          hint: 'light or dark palette',
          run: actions.dark,
        },
      ],
    },
    {
      id: 'help',
      label: 'Help',
      items: [
        {
          label: 'Documentation',
          note: AWAY,
          // Read from the installed package's metadata, so an install
          // that carries no URLs gets a row that says so rather than a
          // link to nowhere.
          hint: links.documentation || 'the installed package names no documentation',
          disabled: !links.documentation,
          run: () => actions.open?.(links.documentation),
        },
        {
          label: 'Source on GitHub',
          note: AWAY,
          hint: links.repository || 'the installed package names no repository',
          disabled: !links.repository,
          run: () => actions.open?.(links.repository),
        },
        { separator: true },
        {
          label: 'Classic editor',
          hint: 'the form editor at /classic, in this tab',
          run: actions.classic,
        },
      ],
    },
  ]
}
