/**
 * What the editor says before a file is allowed to run code.
 *
 * A flowsheet can carry code: its code context, a `.py` script that
 * builds it, and modules of its author's that it names. The server opens
 * none of it unasked: it answers `{ok: false, needs_trust}` with what the
 * file would run, and the page shows that and asks. The question is built
 * here so it can be tested without a browser.
 */

/** How much of the code to show; the rest is summarised, not hidden. */
export const SHOWN = 1200

/**
 * The confirmation text for a `needs_trust` answer.
 *
 * @param {{path: string, kind: 'script' | 'json', code: string, modules: string[]}} info
 * @returns {string}
 */
export function trustQuestion(info) {
  const name = (info.path || 'this flowsheet').split(/[\\/]/).pop()
  const parts = [info.kind === 'script'
    ? `${name} is a Python script. Opening it runs it:`
    : `${name} carries code that runs when it opens.`]
  const code = (info.code || '').trim()
  if (code) {
    if (info.kind !== 'script') parts.push('Its code context:')
    parts.push(code.length > SHOWN
      ? `${code.slice(0, SHOWN)}\n… (${code.length - SHOWN} more characters)`
      : code)
  }
  if (info.modules?.length) {
    parts.push(`It imports ${info.modules.length === 1 ? 'a module' : 'modules'} ` +
      `from outside difflow: ${info.modules.join(', ')}.`)
  }
  parts.push('Open it only if you trust where it came from. Trust it and open?')
  return parts.join('\n\n')
}
