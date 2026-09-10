import assert from 'node:assert/strict'
import { test } from 'node:test'

import { exportName, svgDocument, svgSize } from './download.js'

test('a name comes from the flowsheet path', () => {
  assert.equal(exportName('/home/j/plant.json', 'py'), 'plant.py')
  assert.equal(exportName('plant.json', 'json'), 'plant.json')
  assert.equal(exportName('C:\\models\\plant.json', 'svg'), 'plant.svg')
})

test('an unsaved flowsheet still gets one', () => {
  assert.equal(exportName('', 'py'), 'flowsheet.py')
  assert.equal(exportName(null, 'json'), 'flowsheet.json')
})

test('a dotless name keeps its whole self', () => {
  assert.equal(exportName('plant', 'py'), 'plant.py')
})

test('an svg fragment becomes a document', () => {
  const out = svgDocument('  <svg viewBox="0 0 10 5"></svg>  ')
  assert.ok(out.startsWith('<?xml version="1.0" encoding="UTF-8"?>\n<svg'))
  assert.ok(out.endsWith('\n'))
})

test('the raster size comes from the viewBox, not the width', () => {
  assert.deepEqual(svgSize('<svg viewBox="0 0 640 300" width="100%">'),
                   { width: 1280, height: 600 })
  assert.deepEqual(svgSize('<svg viewBox="0 0 640 300">', 1),
                   { width: 640, height: 300 })
})

test('no viewBox means no raster', () => {
  assert.equal(svgSize('<svg width="100%">'), null)
  assert.equal(svgSize(''), null)
  assert.equal(svgSize('<svg viewBox="0 0 0 0">'), null)
})
