// Record the debugger in JupyterLab: an interactive session typed into the
// input box under a cell, and a record / what-if notebook run cell by cell.
//
//   (cd demo && JAX_PLATFORMS=cpu jupyter lab --no-browser --port 8888 --ServerApp.token=tok)
//   node record_notebook.js <outdir>
//
// Writes notebook-debug.webm / notebook-whatif.webm, the second at which each
// became watchable (<name>.start, for trimming the page load) and a poster PNG.
const { chromium } = require('playwright')
const fs = require('fs')
const path = require('path')

const BASE = 'http://127.0.0.1:8888/lab/tree/'
const OUT = process.argv[2]
const SIZE = { width: 1200, height: 760 }
// The notebook on screen (JupyterLab keeps other tabs in the DOM, hidden).
const NB = '.jp-NotebookPanel:not(.lm-mod-hidden) .jp-Notebook'

// A cursor the video can see (Playwright's own is invisible).
const CURSOR = `
  window.addEventListener('DOMContentLoaded', () => {
    const c = document.createElement('div')
    c.id = '__cursor'
    c.style.cssText = 'position:fixed;left:600px;top:380px;width:22px;height:22px;z-index:99999;' +
      'pointer-events:none;transition:left .5s ease, top .5s ease;' +
      'background:url("data:image/svg+xml;utf8,<svg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 22 22%27>' +
      '<path d=%27M3 2 L3 18 L7.5 13.5 L10.5 20 L13 19 L10 12.5 L16 12.5 Z%27 fill=%27white%27 stroke=%27black%27 stroke-width=%271.4%27/></svg>") no-repeat'
    document.body.appendChild(c)
  })`

async function point(page, locator) {
  const box = await locator.boundingBox()
  await page.evaluate(([x, y]) => {
    const c = document.getElementById('__cursor'); c.style.left = x + 'px'; c.style.top = y + 'px'
  }, [box.x + Math.min(box.width / 2, 60), box.y + box.height / 2])
  await page.waitForTimeout(550)
}

// A kernel left from an earlier run may still be parked at a debugger prompt;
// a notebook reopened on it would wait on that prompt, not start afresh.
async function freshKernels() {
  const api = (p, init) => fetch('http://127.0.0.1:8888/api/' + p + '?token=tok', init)
  for (const k of await (await api('kernels')).json()) await api('kernels/' + k.id, { method: 'DELETE' })
}

async function open(browser, notebook) {
  await freshKernels()
  const context = await browser.newContext({ viewport: SIZE, recordVideo: { dir: OUT, size: SIZE } })
  await context.addInitScript(CURSOR)
  const t0 = Date.now()
  const page = await context.newPage()
  await page.goto(BASE + notebook + '?reset&token=tok')     // a fresh workspace: one tab
  await page.waitForSelector(NB + ' .jp-Cell', { timeout: 60000 })
  await page.waitForTimeout(2500)
  const no = page.getByRole('button', { name: 'No', exact: true })
  if (await no.count()) await no.click()                    // the "Jupyter news" toast
  await page.waitForFunction(() => /Idle/.test(document.body.innerText), null, { timeout: 60000 })
  const files = page.locator('.jp-SideBar.jp-mod-left .lm-TabBar-tab').first()
  if (await page.locator('.jp-FileBrowser').isVisible()) await files.click()   // fold the file browser away
  await page.waitForTimeout(800)
  return { context, page, start: (Date.now() - t0) / 1000 }
}

const cell = (page, i) => page.locator(NB + ' .jp-Cell').nth(i)

async function run(page, i, settle = 1200) {
  const c = cell(page, i)
  await c.scrollIntoViewIfNeeded()
  await point(page, c.locator('.jp-InputArea-editor'))
  await c.locator('.jp-InputArea-editor').click()
  await page.keyboard.press('Shift+Enter')
  await page.waitForTimeout(settle)
}

// Wait until the cell has finished: its prompt shows a number, not [*].
async function finished(page, i, timeout = 120000) {
  await page.waitForFunction(([nb, i]) => {
    const p = document.querySelectorAll(nb + ' .jp-Cell')[i]?.querySelector('.jp-InputPrompt')
    return p && /\[\d+\]/.test(p.textContent)
  }, [NB, i], { timeout })
}

async function finish({ context, page, start }, name) {
  await page.waitForTimeout(2500)
  await page.screenshot({ path: path.join(OUT, `${name}.png`) })
  const video = page.video()
  await context.close()
  fs.renameSync(await video.path(), path.join(OUT, `${name}.webm`))
  fs.writeFileSync(path.join(OUT, `${name}.start`), String(start))
}

;(async () => {
  const browser = await chromium.launch()
  const errors = []

  // ---- an interactive session, typed under the cell ----------------------
  let s = await open(browser, 'debug-in-a-notebook.ipynb')
  let p = s.page
  p.on('pageerror', (e) => errors.push(String(e)))
  await run(p, 1, 300)
  await finished(p, 1)
  await p.waitForTimeout(600)
  await run(p, 2, 300)
  const commands = ['where', 'next', 'next', 'p reactor_feed', 'step', 'p tear',
                    'break if residual<1e-6', 'continue', 'p history', 'break clear', 'continue',
                    'continue']
  for (const command of commands) {
    const box = p.locator(NB + ' .jp-Stdin-input').last()
    await box.waitFor({ timeout: 90000 })
    await box.scrollIntoViewIfNeeded()
    await p.waitForTimeout(700)
    await point(p, box)
    await box.pressSequentially(command, { delay: 75 })
    await p.waitForTimeout(350)
    await box.press('Enter')
    await p.waitForTimeout(1400)
  }
  await finished(p, 2)
  await cell(p, 2).locator('.jp-OutputArea').last().evaluate((e) => e.scrollIntoView({ block: 'end' }))
  await finish(s, 'notebook-debug')

  // ---- record a solve, then a what-if ------------------------------------
  s = await open(browser, 'record-and-what-if.ipynb')
  p = s.page
  p.on('pageerror', (e) => errors.push(String(e)))
  for (const i of [1, 2, 3]) {
    await run(p, i, 300)
    await finished(p, i)
    await p.waitForTimeout(2600)
  }
  await cell(p, 3).locator('.jp-OutputArea').evaluate((e) => e.scrollIntoView({ block: 'end' }))
  await p.waitForTimeout(1500)
  await finish(s, 'notebook-whatif')

  await browser.close()
  console.log('errors:', JSON.stringify(errors))
})()
