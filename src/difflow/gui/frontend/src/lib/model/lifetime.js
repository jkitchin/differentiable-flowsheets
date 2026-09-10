/**
 * Telling the server this page is still open, so that closing it isn't.
 *
 * The editor is a Python process and a browser tab, and to the person
 * using it they are one thing. Closing the tab should give port 8756
 * back; it should not leave a server running that refuses to start the
 * next one. Nothing in HTTP says a page has gone, so the page says so:
 * a ping while it is open and a farewell as it unloads.
 *
 * Two channels, because neither alone is enough.
 *
 * - The **farewell** is what makes the port come back in the moment. It
 *   goes out from `pagehide` with `keepalive`, which is what lets a
 *   request outlive the document that started it. `sendBeacon` would
 *   also survive, but it cannot set a header, and the token this server
 *   demands on every mutating request travels in one.
 * - The **ping** is the fallback for every ending that sends nothing --
 *   a crashed browser, a killed tab, a closed lid. It is deliberately
 *   slow, and the server's grace slower still, because a background tab
 *   has its timers throttled to roughly one firing a minute and an
 *   impatient watchdog would read "the user looked at another tab" as
 *   "the user has gone".
 *
 * A page restored from the back/forward cache is a live page again, so
 * `pagehide` with `persisted` set sends no farewell, and `pageshow`
 * pings straight away rather than waiting out an interval.
 */

/** A per-page id. Two tabs on one flowsheet must not be one client. */
export function clientId(random = Math.random) {
  return `${Date.now().toString(36)}-${random().toString(36).slice(2, 10)}`
}

/**
 * Keep the server informed that this page is open.
 *
 * Everything it touches is injected, so the whole thing is testable
 * without a browser and without waiting out a real interval.
 *
 * @param {object} options
 * @param {(path: string, payload: object) => Promise} options.post - the API's POST.
 * @param {number} [options.interval] - seconds between pings.
 * @param {string} [options.client] - this page's id.
 * @param {object} [options.timers] - `setInterval`/`clearInterval` to use.
 * @returns {{start: Function, stop: Function, ping: Function, farewell: Function, client: string}}
 */
export function keepAlive({
  post,
  interval = 15,
  client = clientId(),
  timers = { setInterval, clearInterval },
} = {}) {
  let handle = null
  // A ping that fails is not worth reporting: the server being
  // unreachable is exactly the state where saying so changes nothing,
  // and an unhandled rejection every fifteen seconds is noise in a
  // console the user may be reading for their own flowsheet.
  const ping = () => Promise.resolve(post('/api/ping', { client })).catch(() => {})

  return {
    client,
    ping,
    start() {
      if (handle !== null) return
      ping()
      handle = timers.setInterval(ping, interval * 1000)
    },
    stop() {
      if (handle === null) return
      timers.clearInterval(handle)
      handle = null
    },
    /**
     * Say this page has gone.
     *
     * Returns whether it was actually sent: a `pagehide` into the
     * back/forward cache is not a departure, and a page that may come
     * back must not tell the server to stop.
     */
    farewell({ persisted = false } = {}) {
      if (persisted) return false
      Promise.resolve(post('/api/bye', { client }, { keepalive: true })).catch(() => {})
      return true
    },
  }
}
