import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import { SHOWN, trustQuestion } from './trust.js'

const info = (over = {}) => ({
  path: '/home/me/plant.json', kind: 'json', code: 'thermo = IdealThermo(...)',
  modules: [], ...over,
})

test('it names the file and shows the code it would run', () => {
  const q = trustQuestion(info())
  assert.match(q, /^plant\.json carries code/)
  assert.match(q, /thermo = IdealThermo\(\.\.\.\)/)
  assert.match(q, /Trust it and open\?$/)
})

test('a script is called a script', () => {
  assert.match(trustQuestion(info({ path: 'C:\\lab\\plant.py', kind: 'script' })),
    /^plant\.py is a Python script/)
})

test('the modules it would import are listed', () => {
  assert.match(trustQuestion(info({ code: '', modules: ['mylab', 'lab.k'] })),
    /modules from outside difflow: mylab, lab\.k\./)
})

test('long code is cut, and says how much was left out', () => {
  const q = trustQuestion(info({ code: 'x'.repeat(SHOWN + 50) }))
  assert.match(q, /… \(50 more characters\)/)
})
