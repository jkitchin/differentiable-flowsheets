/**
 * The console's pure half: the transcript, the history, and how one
 * output is rendered.
 *
 * The panel is a view of these. What is worth testing without a browser
 * is the part with rules -- that a recalled line does not lose what was
 * half-typed, that an unknown output kind degrades to something legible
 * rather than an empty box, and that a cell which both printed and
 * failed keeps both halves.
 */

/** How many entries the transcript keeps before dropping the oldest. */
export const LIMIT = 200

/** What the panel says before anything has been typed. */
export const GREETING =
  'Python, on the server, over the live model. `fs` is the flowsheet on ' +
  'the canvas; `streams` is the last solve. Enter runs, Shift+Enter is a ' +
  'newline.'

/** Whether a cell is worth sending at all. */
export const isBlank = (source) => !String(source ?? '').trim()

/**
 * The prompt-prefixed echo of what was typed.
 *
 * `>>>` then `...`, as a Python prompt does -- it is the convention
 * that makes a transcript readable when it is pasted somewhere else.
 */
export function prompt(source) {
  return String(source ?? '')
    .split('\n')
    .map((line, i) => `${i === 0 ? '>>>' : '...'} ${line}`.trimEnd())
    .join('\n')
}

/**
 * One transcript entry from a cell and the server's answer.
 *
 * A failed cell keeps whatever it printed first: the print is often the
 * debugging the user added, and dropping it because the next line threw
 * would throw away the answer.
 */
export function entry(source, answer) {
  return {
    source,
    outputs: answer?.outputs ?? [],
    error: answer?.error ?? null,
    changed: !!answer?.changed,
  }
}

/** Append, oldest dropped past `LIMIT`. */
export const append = (transcript, item) =>
  [...transcript, item].slice(-LIMIT)

/**
 * Move through past inputs, as a shell does.
 *
 * State in, state out: `index` is how far back we are (`null` at the
 * live line) and `stash` is what was half-typed when we left it. Going
 * back up stashes the draft and coming back down returns it, rather
 * than eating a line someone was in the middle of writing.
 */
export function recall(history, state, direction) {
  const { index = null, draft = '', stash = '' } = state ?? {}
  const at = (i, keep) => ({ index: i, source: history[i], stash: keep })
  if (!history.length) return { index, source: draft, stash }
  if (direction === 'up') {
    if (index === null) return at(history.length - 1, draft)
    return at(Math.max(0, index - 1), stash)
  }
  if (index === null) return { index: null, source: draft, stash }
  if (index + 1 >= history.length) return { index: null, source: stash, stash: '' }
  return at(index + 1, stash)
}

/** The inputs worth recalling: non-blank, and not the same line twice. */
export function remember(history, source) {
  if (isBlank(source)) return history
  if (history[history.length - 1] === source) return history
  return [...history, source].slice(-LIMIT)
}

/**
 * How one output should be drawn.
 *
 * The unknown branch is deliberate. Images already round-trip through
 * here so that adding plots is a server-side hook and nothing else, and
 * anything a later server sends that this page does not know must read
 * as an unrendered output rather than as nothing at all.
 */
export function describe(output) {
  const kind = output?.kind
  if (kind === 'image' && output.data) {
    return { kind: 'image', src: `data:${output.mime || 'image/png'};base64,${output.data}` }
  }
  if (kind === 'text' || kind === 'value') {
    return { kind, stream: output.stream === 'stderr' ? 'stderr' : 'stdout',
             text: String(output.text ?? '') }
  }
  return { kind: 'unknown',
           text: `[${kind ?? 'output'} — this editor cannot render it]` }
}

/** Whether anything in the transcript means the canvas is out of date. */
export const touched = (transcript) => transcript.some((e) => e.changed)
