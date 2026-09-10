/**
 * The three things that can answer a brief.
 *
 * One function, `run`, and three implementations behind it. The panel
 * knows which one is selected and nothing else about them --- it hands
 * over chat turns and receives text back, growing.
 *
 * They differ in where the brief goes, which is the only difference a
 * user has to care about and is stated in the panel:
 *
 *   webllm  the page itself, over WebGPU. Weights come from the MLC
 *           CDN on first use and are cached by the browser.
 *   openai  a base URL the user gives: Ollama, llama.cpp, vLLM.
 *   server  the difflow process, which forwards to the Anthropic API
 *           with a key from its own environment (`gui/assistant.py`).
 *
 * `@mlc-ai/web-llm` is imported dynamically, so it is a chunk the
 * browser fetches only if someone selects it. An editor that never
 * opens this panel never loads a byte of it.
 */

import { post } from './api.js'
import { sseChunk, sseDelta } from './model/assistant.js'

/** Kept across questions: loading a model is the expensive part. */
let engine = null
let engineModel = ''

async function webllm(settings, messages, { onText, onStatus, signal }) {
  if (!globalThis.navigator?.gpu) {
    throw new Error(
      'this browser has no WebGPU, so a model cannot run in the page. '
      + 'Copy the brief instead, or point the panel at a local server.')
  }
  const mlc = await import('@mlc-ai/web-llm')
  if (!engine || engineModel !== settings.webllmModel) {
    onStatus(`loading ${settings.webllmModel} — the first load downloads it`)
    engine = await mlc.CreateMLCEngine(settings.webllmModel, {
      initProgressCallback: (report) => onStatus(report.text),
    })
    engineModel = settings.webllmModel
  }
  onStatus('')

  const stream = await engine.chat.completions.create({
    messages, stream: true, temperature: settings.temperature,
  })
  let text = ''
  for await (const chunk of stream) {
    if (signal?.aborted) {
      await engine.interruptGenerate()
      break
    }
    const token = chunk.choices?.[0]?.delta?.content
    if (token) onText((text += token))
  }
  return text
}

async function openai(settings, messages, { onText, signal }) {
  const url = `${settings.baseUrl.replace(/\/+$/, '')}/chat/completions`
  const response = await fetch(url, {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: settings.openaiModel,
      messages,
      stream: true,
      temperature: settings.temperature,
    }),
  })
  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => '')
    throw new Error(`${url}: ${response.status} ${detail.slice(0, 200)}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let text = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    const split = sseChunk(buffer, decoder.decode(value, { stream: true }))
    buffer = split.rest
    for (const line of split.lines) {
      const token = sseDelta(line)
      if (token) onText((text += token))
    }
  }
  return text
}

/**
 * The server holds the key and does the talking, so this one turn is
 * not streamed --- `gui/assistant.py` says why.
 */
async function server(settings, messages) {
  const answer = await post('/api/assistant', { messages })
  if (answer.ok === false) throw new Error(answer.error)
  return answer.text
}

const RUNNERS = { webllm, openai, server }

/**
 * Answer one brief.
 *
 * @param settings  the panel's persisted settings
 * @param messages  chat turns, `[{role, content}]`
 * @param hooks     `onText(soFar)` on every token, `onStatus(line)` for
 *                  model loading, `signal` to stop
 * @returns the whole answer
 */
export function run(settings, messages,
                    { onText = () => {}, onStatus = () => {}, signal } = {}) {
  const runner = RUNNERS[settings.provider]
  if (!runner) {
    return Promise.reject(new Error(
      'no model is selected, so there is nothing to answer with'))
  }
  return runner(settings, messages, { onText, onStatus, signal })
}
