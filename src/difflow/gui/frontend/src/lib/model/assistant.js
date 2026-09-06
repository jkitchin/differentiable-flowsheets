/**
 * The assistant's model layer: what to ask for, where to send it, and
 * what the answer is allowed to claim.
 *
 * The panel's value is the *brief*, not the model. `/api/context` builds
 * one from the live flowsheet -- the catalog entry for the selected
 * unit, the solve diagnostics, the docs sections that mention it -- and
 * this file decides which brief a click means and turns it into a
 * request. The provider is deliberately the last thing chosen: with no
 * model at all the panel still assembles the brief and hands it over,
 * which is the honest fallback and often the more useful one.
 */

/** The four briefs `context.py` can assemble, in the order they are offered. */
export const KINDS = [
  { id: 'flowsheet', label: 'This flowsheet',
    hint: 'topology, recycles, tear streams and the generated code' },
  { id: 'block', label: 'The selected unit',
    hint: 'its catalog entry, equations, docstring and current parameters' },
  { id: 'solve', label: 'The last solve',
    hint: 'method, residual, iterations, and a troubleshooting card' },
  { id: 'planning', label: 'The linearization',
    hint: 'delta vectors, prices, shadow prices and health findings' },
]

/**
 * Where a question goes.
 *
 * Named plainly, because "no model" and "off this machine" are the two
 * facts a user actually needs before typing. `local` is what the panel
 * puts in front of them; it is not a security claim about the endpoint.
 */
export const PROVIDERS = [
  { id: 'none', label: 'No model', local: true,
    hint: 'assemble the brief only. Copy it into an assistant of your choosing.' },
  { id: 'webllm', label: 'In this browser (WebLLM)', local: true,
    hint: 'a small model over WebGPU. The weights download once (~1-2 GB) and are '
        + 'cached by the browser; nothing you type leaves the machine.' },
  { id: 'openai', label: 'Local server (OpenAI-compatible)', local: true,
    hint: 'Ollama, llama.cpp or vLLM on a base URL you give. The brief is sent there.' },
  { id: 'server', label: 'Anthropic (key held by the difflow server)', local: false,
    hint: 'the page posts to difflow, which forwards to the API with a key from '
        + 'its own environment. The brief leaves this machine.' },
]

/** Small instruct models that WebLLM publishes, largest last. */
export const WEBLLM_MODELS = [
  'Llama-3.2-1B-Instruct-q4f16_1-MLC',
  'Llama-3.2-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
]

export const DEFAULTS = {
  provider: 'none',
  webllmModel: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',
  baseUrl: 'http://localhost:11434/v1',
  openaiModel: 'llama3.2',
  temperature: 0,
}

const SETTINGS_KEY = 'difflow.assistant'

export const provider = (id) =>
  PROVIDERS.find((p) => p.id === id) ?? PROVIDERS[0]

/**
 * Which brief the current state implies.
 *
 * A selected unit is the strongest signal -- someone who clicked a node
 * and then opened the panel is asking about that node. Failing that, a
 * solve that went wrong is what a user is most likely staring at. The
 * whole flowsheet is the fallback, and every one of these is a default
 * the user can override in the panel.
 */
export function inferKind({ unit = null, solve = null } = {}) {
  if (unit) return 'block'
  if (solve && solve.ok === false) return 'solve'
  return 'flowsheet'
}

/** The `/api/context` request for one question. */
export function contextPath({ kind = 'flowsheet', question = '', name = null,
                              operation = null } = {}) {
  const query = new URLSearchParams({ kind })
  if (question) query.set('q', question)
  if (name) query.set('name', name)
  if (operation) query.set('operation', operation)
  return `/api/context?${query}`
}

/**
 * The chat turns for one brief.
 *
 * The system prompt is the server's (`context.SYSTEM`): it is the half
 * of the prompt that states the model may answer only from the brief,
 * and it belongs next to the code that assembles the brief rather than
 * in the page.
 */
export function messages(pack) {
  return [
    { role: 'system', content: pack.system },
    { role: 'user', content: pack.prompt },
  ]
}

/** One sentence on where the next question goes. Shown, not buried. */
export function destination(settings) {
  const p = provider(settings.provider)
  if (p.id === 'none') return 'Nothing is sent anywhere; the brief is assembled here.'
  if (p.id === 'webllm') return `Answered in this browser by ${settings.webllmModel}.`
  if (p.id === 'openai') return `Sent to ${settings.baseUrl} as ${settings.openaiModel}.`
  return 'Sent to the difflow server, which forwards it to the Anthropic API.'
}

/** Settings survive a reload; nothing else about the panel does. */
export function readSettings(storage) {
  try {
    const raw = storage?.getItem(SETTINGS_KEY)
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : { ...DEFAULTS }
  } catch {
    return { ...DEFAULTS }
  }
}

export function writeSettings(storage, settings) {
  try {
    storage?.setItem(SETTINGS_KEY, JSON.stringify(settings))
  } catch {
    // private browsing, a full quota, a disabled store: the panel works
    // without persistence, so this is not worth reporting.
  }
}
