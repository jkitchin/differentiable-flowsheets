/**
 * CodeMirror, set up for Python: loaded on demand, not with the page.
 *
 * CodeMirror and its Python parser weigh about as much as the rest of the
 * editor together, and they are wanted only while a panel with code in it
 * is open. So `CodeEditor.svelte` imports this module dynamically, and the
 * build puts it (and everything it pulls in) in its own `codemirror.js`
 * chunk, as it does for `webllm.js`.
 *
 * Colours come from the palette's custom properties rather than from a
 * CodeMirror theme, so the editor follows the light/dark toggle the way
 * everything else does: the tokens change underneath it.
 */

import { EditorState } from '@codemirror/state'
import {
  EditorView,
  drawSelection,
  highlightActiveLine,
  highlightActiveLineGutter,
  keymap,
  lineNumbers,
  placeholder as placeholderText,
} from '@codemirror/view'
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import {
  HighlightStyle,
  bracketMatching,
  indentOnInput,
  indentUnit,
  syntaxHighlighting,
} from '@codemirror/language'
import { python } from '@codemirror/lang-python'
import { tags as t } from '@lezer/highlight'

const highlight = HighlightStyle.define([
  { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword],
    color: 'var(--code-keyword)' },
  { tag: [t.string, t.special(t.string)], color: 'var(--code-string)' },
  { tag: [t.number, t.bool, t.null], color: 'var(--code-number)' },
  { tag: t.comment, color: 'var(--code-comment)', fontStyle: 'italic' },
  { tag: [t.function(t.definition(t.variableName)), t.function(t.variableName)],
    color: 'var(--code-def)' },
  { tag: [t.className, t.definition(t.className)], color: 'var(--code-class)' },
  { tag: t.self, color: 'var(--code-keyword)', fontStyle: 'italic' },
])

// Layout and chrome only; every colour is a palette token.
const theme = EditorView.theme({
  '&': {
    height: '100%',
    backgroundColor: 'var(--surface)',
    color: 'var(--ink)',
    fontSize: '0.79rem',
  },
  '&.cm-focused': { outline: 'none' },
  '.cm-scroller': {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    lineHeight: '1.55',
  },
  '.cm-content': { caretColor: 'var(--ink)', padding: '0.4rem 0' },
  '.cm-cursor': { borderLeftColor: 'var(--ink)' },
  '.cm-gutters': {
    backgroundColor: 'var(--panel)',
    color: 'var(--ink-soft)',
    border: 'none',
    borderRight: '1px solid var(--grid)',
  },
  '.cm-activeLine': { backgroundColor: 'color-mix(in srgb, var(--series) 6%, transparent)' },
  '.cm-activeLineGutter': { backgroundColor: 'transparent', color: 'var(--ink)' },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': {
    backgroundColor: 'color-mix(in srgb, var(--series) 25%, transparent)',
  },
  '&.cm-focused .cm-matchingBracket': {
    backgroundColor: 'color-mix(in srgb, var(--series) 20%, transparent)',
    outline: '1px solid color-mix(in srgb, var(--series) 50%, transparent)',
  },
  '.cm-placeholder': { color: 'var(--ink-soft)' },
})

/**
 * A Python editor in `parent`.
 *
 * @param {HTMLElement} parent
 * @param {object} options
 * @param {string} options.doc  the initial text
 * @param {boolean} [options.readonly]  a viewer: selectable and copyable,
 *   not editable, and no active-line highlight chasing a caret nobody types at
 * @param {(text: string) => void} [options.onchange]  every edit
 * @param {(text: string) => void} [options.onsubmit]  Mod-Enter
 * @returns {EditorView}
 */
export function createEditor(parent, {
  doc = '',
  readonly = false,
  placeholder = '',
  label = 'code',
  onchange = () => {},
  onsubmit = () => {},
} = {}) {
  const view = new EditorView({
    parent,
    state: EditorState.create({
      doc,
      extensions: [
        lineNumbers(),
        drawSelection(),
        python(),
        syntaxHighlighting(highlight),
        theme,
        indentUnit.of('    '),
        EditorState.tabSize.of(4),
        EditorView.contentAttributes.of({ 'aria-label': label, spellcheck: 'false' }),
        ...(readonly
          ? [EditorState.readOnly.of(true), keymap.of(defaultKeymap)]
          : [
              highlightActiveLineGutter(),
              highlightActiveLine(),
              history(),
              indentOnInput(),
              bracketMatching(),
              placeholderText(placeholder),
              // Mod-Enter first, so it is not taken as a newline. Tab
              // indents, because this is a code editor; Escape then Tab
              // still leaves it, which is CodeMirror's answer to the
              // keyboard trap.
              keymap.of([
                { key: 'Mod-Enter', run: (v) => { onsubmit(v.state.doc.toString()); return true } },
                indentWithTab,
                ...historyKeymap,
                ...defaultKeymap,
              ]),
              EditorView.updateListener.of((update) => {
                if (update.docChanged) onchange(update.state.doc.toString())
              }),
            ]),
      ],
    }),
  })
  return view
}

/** Replace the whole document, if it differs. */
export function setDoc(view, text) {
  if (text !== view.state.doc.toString()) {
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } })
  }
}
