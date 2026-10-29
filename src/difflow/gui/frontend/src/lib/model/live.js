/**
 * The live debugger's client side: what the page knows between requests.
 *
 * The server half is a long poll (/api/debug/start, command, wait, pause,
 * stop; see FlowsheetSession.debug_*): every reply carries the event it
 * got to and the runner's `state`, and a reply whose state is `running`
 * means "ask /api/debug/wait again". The events are the JSON protocol's
 * (`difflow-dbg/1`), so this reads the same objects the CLI prints.
 * Pure functions, so they test under bare node like the rest of `model/`.
 */

import { formatNumber } from './trace.js'

/** The buttons, and the debugger command each one sends. */
export const RESUME = [
  { key: 'step', label: 'Step iteration', command: 'step', hint: 'run to the next tear iteration' },
  { key: 'next', label: 'Next unit', command: 'next', hint: 'run to the next unit (or iteration)' },
  { key: 'stepi', label: 'Step in', command: 'stepi', hint: 'run to the next checkpoint of any kind' },
  { key: 'continue', label: 'Continue', command: 'continue', hint: 'run to the next breakpoint, a unit error, or the end' },
]

/** The events a breakpoint can wait for, as the page offers them. */
export const EVENTS = [
  { key: 'nonfinite', label: 'first NaN' },
  { key: 'unit_error', label: 'unit error' },
  { key: 'residual_rise', label: 'residual rises' },
  { key: 'inner_unconverged', label: 'inner solve fails' },
]

/** A live debugger that has not started. */
export function initial() {
  return { status: 'idle', pause: null, finished: null, log: [] }
}

/** Keep the transcript to its last `limit` entries. */
const trim = (log, limit = 400) => log.slice(-limit)

/**
 * The state after one server reply.
 *
 * @param {Object} live   the current state
 * @param {Object} reply  `{ok, event, state}` from a /api/debug/* route,
 *   or `{ok: false, error}`
 * @param {string} [sent] the command that produced it, for the transcript
 */
export function absorb(live, reply, sent = null) {
  const log = [...live.log]
  if (sent) log.push({ kind: 'command', text: sent })
  if (!reply?.ok) {
    log.push({ kind: 'error', text: reply?.error ?? 'the request failed' })
    return { ...live, log: trim(log) }
  }
  let { pause, finished } = live
  const take = (event) => {
    if (!event) return
    if (event.event === 'result') {
      for (const line of event.output ?? []) log.push({ kind: event.ok ? 'output' : 'error', text: line })
      take(event.next)
    } else if (event.event === 'pause') {
      pause = event
      log.push({ kind: 'pause', text: banner(event) })
    } else if (event.event === 'finished') {
      finished = event
      pause = null
      log.push({ kind: event.ok ? 'finished' : 'error', text: describeFinished(event) })
    }
  }
  take(reply.event)
  const status = reply.state ?? live.status
  return { status, pause: status === 'paused' ? pause : null, finished, log: trim(log) }
}

/** The debugger verbs that resume the solve (engine.py's flow commands). */
const RESUMES = new Set(['step', 's', 'next', 'n', 'stepi', 'si', 'continue', 'c',
                         'run', 'r', 'detach', 'quit', 'q'])

/** Whether a command resumes the solve rather than inspecting it. */
export const resumes = (cmd) => RESUMES.has(String(cmd).trim().split(/\s+/)[0].toLowerCase())

/**
 * The state the moment a resume is sent, before its reply: running.
 * The reply can take the whole long-poll window to come back, and until
 * then the page must say the solve is running -- and offer Pause --
 * rather than show a pause that is already over.
 */
export function resuming(live) {
  return { ...live, status: 'running', pause: null }
}

/** Whether the page should ask /api/debug/wait for the next event. */
export const keepWaiting = (reply) => !!reply?.ok && reply.state === 'running'

/** One line for a pause, like the REPL's banner. */
export function banner(pause) {
  if (!pause) return ''
  const parts = [`@${pause.checkpoint}`]
  if (pause.unit) parts.push(pause.unit)
  if (pause.iteration !== null && pause.iteration !== undefined) parts.push(`iter ${pause.iteration}`)
  if (pause.phase && !['solve', 'start', 'end'].includes(pause.phase)) parts.push(pause.phase)
  if (pause.residual !== null && pause.residual !== undefined) parts.push(`residual ${formatNumber(pause.residual)}`)
  if (pause.checkpoint === 'terminated') parts.push(pause.converged ? 'converged' : 'NOT converged')
  let text = parts.join(' · ')
  if (pause.reason) text += `  ↳ ${pause.reason}`
  if (pause.error) text += `  ✗ ${pause.error}`
  return text
}

/** One line for the end of the solve. */
export function describeFinished(event) {
  if (event.ok) {
    return `finished: ${event.converged ? 'converged' : 'NOT converged'}` +
      (event.iterations !== null && event.iterations !== undefined ? ` in ${event.iterations} iterations` : '')
  }
  return event.stopped ? 'stopped from the debugger' : `the solve raised: ${event.error}`
}

/** The unit to outline on the canvas: where a pause is, if at a unit. */
export const focusOf = (live) => (live.status === 'paused' ? live.pause?.unit ?? null : null)

/**
 * The breakpoints the server lists (`break` with no argument answers
 * lines like "unit R1", "on nonfinite", "iteration 3", "if residual<1").
 */
export function parseBreakpoints(lines) {
  const units = []
  const events = []
  const other = []
  for (const line of lines ?? []) {
    if (line.startsWith('unit ')) units.push(line.slice(5))
    else if (line.startsWith('on ')) events.push(line.slice(3))
    else if (line !== 'no breakpoints') other.push(line)
  }
  return { units, events, other }
}

/** The command that turns a unit breakpoint on or off. */
export const unitBreak = (unit, on) => (on ? `break unit ${unit}` : `break del ${unit}`)

/** The command that turns an event breakpoint on or off. */
export const eventBreak = (event, on) => (on ? `break on ${event}` : `break del ${event}`)
