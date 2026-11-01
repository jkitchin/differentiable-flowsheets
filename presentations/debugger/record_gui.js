// Record the editor's Debug and Trace panels, driven like a person would.
//   node record_gui.js <outdir>
// Writes debug.webm / trace.webm plus the second at which each became
// watchable (for trimming the page load), and poster PNGs.
const { chromium } = require('playwright')
const fs = require('fs')
const path = require('path')

const URL = 'http://127.0.0.1:8801/'
const OUT = process.argv[2]
const SIZE = { width: 1280, height: 800 }

// A cursor the video can see: Playwright moves the real one invisibly.
const CURSOR = `
  window.addEventListener('DOMContentLoaded', () => {
    const c = document.createElement('div')
    c.id = '__cursor'
    c.style.cssText = 'position:fixed;left:640px;top:400px;width:22px;height:22px;z-index:99999;' +
      'pointer-events:none;transition:left .55s ease, top .55s ease;' +
      'background:url("data:image/svg+xml;utf8,<svg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 22 22%27>' +
      '<path d=%27M3 2 L3 18 L7.5 13.5 L10.5 20 L13 19 L10 12.5 L16 12.5 Z%27 fill=%27white%27 stroke=%27black%27 stroke-width=%271.4%27/></svg>") no-repeat'
    document.body.appendChild(c)
    const ring = document.createElement('div')
    ring.id = '__ring'
    ring.style.cssText = 'position:fixed;width:34px;height:34px;margin:-14px 0 0 -14px;border-radius:50%;' +
      'border:3px solid #eb6834;opacity:0;z-index:99998;pointer-events:none;transition:opacity .35s, transform .35s;'
    document.body.appendChild(ring)
  })`

async function point(page, locator) {
  const box = await locator.boundingBox()
  const x = box.x + box.width / 2, y = box.y + box.height / 2
  await page.evaluate(([x, y]) => {
    const c = document.getElementById('__cursor'); c.style.left = x + 'px'; c.style.top = y + 'px'
  }, [x, y])
  await page.waitForTimeout(650)
  return [x, y]
}

async function click(page, locator, pause = 900) {
  const [x, y] = await point(page, locator)
  await page.evaluate(([x, y]) => {
    const r = document.getElementById('__ring')
    r.style.left = x + 'px'; r.style.top = y + 'px'; r.style.transform = 'scale(.6)'; r.style.opacity = 1
    setTimeout(() => { r.style.opacity = 0; r.style.transform = 'scale(1.3)' }, 260)
  }, [x, y])
  await locator.click()
  await page.waitForTimeout(pause)
}

async function type(page, locator, text) {
  await click(page, locator, 300)
  await locator.pressSequentially(text, { delay: 70 })
  await page.waitForTimeout(350)
  await locator.press('Enter')
  await page.waitForTimeout(1600)
}

async function open(browser, name) {
  const context = await browser.newContext({ viewport: SIZE, recordVideo: { dir: OUT, size: SIZE } })
  await context.addInitScript(CURSOR)
  const t0 = Date.now()
  const page = await context.newPage()
  await page.goto(URL)
  await page.waitForSelector('.unit', { timeout: 90000 })
  await page.waitForTimeout(800)
  return { context, page, start: (Date.now() - t0) / 1000 }
}

async function finish({ context, page, start }, name) {
  await page.waitForTimeout(2500)
  await page.screenshot({ path: path.join(OUT, `${name}.png`) })
  const video = page.video()
  await context.close()
  fs.renameSync(await video.path(), path.join(OUT, `${name}.webm`))
  fs.writeFileSync(path.join(OUT, `${name}.start`), String(start))
}

const until = (page, js, timeout = 90000) => page.waitForFunction(js, null, { timeout })

;(async () => {
  const browser = await chromium.launch()
  const errors = []

  // ---- the Debug panel: a live solve, stepped ----------------------------
  let s = await open(browser, 'debug')
  let p = s.page
  p.on('pageerror', (e) => errors.push(String(e)))
  await click(p, p.getByRole('button', { name: /^View/ }).first(), 500)
  await click(p, p.getByText('Debug', { exact: true }).first())
  await click(p, p.getByRole('button', { name: 'Start', exact: true }))
  await until(p, () => document.querySelector('.debugger .status')?.textContent.trim() === 'paused')
  await p.waitForTimeout(900)
  await click(p, p.getByRole('button', { name: 'Next unit' }), 1300)
  await click(p, p.getByRole('button', { name: 'Next unit' }), 1300)
  await click(p, p.locator('.svelte-flow__node', { hasText: 'split' }).first(), 700)
  await click(p, p.getByRole('button', { name: /before split/ }), 900)
  await click(p, p.getByRole('button', { name: 'Continue' }), 300)
  await until(p, () => /before_unit · split/.test(document.querySelector('.debugger .banner')?.textContent ?? ''))
  await p.waitForTimeout(1500)
  await click(p, p.getByRole('button', { name: /● before split/ }), 700)          // take the breakpoint off
  await click(p, p.getByRole('button', { name: 'Step iteration' }), 300)
  await p.waitForSelector('.debugger .state table', { timeout: 60000 })
  await p.waitForTimeout(2200)
  const entry = p.locator('.debugger .entry')
  await type(p, entry, 'p reactor_feed')
  await type(p, entry, 'set tear recycle.F_n_hexane 3.0')
  await click(p, p.getByRole('button', { name: 'Continue' }), 300)
  await until(p, () => /terminated/.test(document.querySelector('.debugger .banner')?.textContent ?? ''))
  await p.waitForTimeout(1800)
  await click(p, p.getByRole('button', { name: 'Continue' }), 300)
  await until(p, () => document.querySelector('.debugger .status')?.textContent.trim() === 'finished')
  await finish(s, 'debug')

  // ---- the Trace panel: record, scrub, what-if ---------------------------
  s = await open(browser, 'trace')
  p = s.page
  p.on('pageerror', (e) => errors.push(String(e)))
  await click(p, p.getByRole('button', { name: /^View/ }).first(), 500)
  await click(p, p.getByText('Trace', { exact: true }).first())
  await click(p, p.getByRole('button', { name: 'Record solve' }), 300)
  await p.waitForSelector('.trace input[type=range]', { timeout: 90000 })
  await p.waitForTimeout(1000)
  for (let i = 0; i < 4; i++) await click(p, p.getByTitle('next (→)'), 700)        // unit by unit
  await point(p, p.locator('.trace .controls select'))
  await p.selectOption('.trace .controls select', 'iteration')
  await p.waitForTimeout(700)
  await click(p, p.getByTitle('next (→)'), 1800)                                     // a tear iteration
  const value = p.locator('.whatif-form input')
  await click(p, value, 300)
  await value.fill('')
  await value.pressSequentially('3', { delay: 120 })
  await p.waitForTimeout(600)
  await click(p, p.getByRole('button', { name: 'Replay from here' }), 300)
  await p.waitForSelector('.toggle', { timeout: 90000 })
  await p.waitForTimeout(2500)
  await click(p, p.getByRole('button', { name: 'Recorded', exact: true }), 1800)
  await click(p, p.getByRole('button', { name: 'What if', exact: true }), 1500)
  await click(p, p.getByTitle('play'), 6000)
  await finish(s, 'trace')

  await browser.close()
  console.log('errors:', JSON.stringify(errors))
})()
