import assert from 'node:assert/strict'
import test from 'node:test'

import {
  CATEGORY_SYMBOLS,
  OPERATION_SYMBOLS,
  PRIMITIVES,
  SYMBOLS,
  symbol,
  symbolFor,
} from './symbols.js'

test('every mapping points at a symbol that exists', () => {
  // A typo here is invisible on screen: `symbol()` returns undefined and
  // the component renders an empty <svg>, so the node loses its picture
  // and nothing anywhere says why.
  for (const [op, id] of Object.entries(OPERATION_SYMBOLS)) {
    assert.ok(SYMBOLS[id], `${op} -> ${id}, which is not a symbol`)
  }
  for (const [cat, id] of Object.entries(CATEGORY_SYMBOLS)) {
    assert.ok(SYMBOLS[id], `${cat} -> ${id}, which is not a symbol`)
  }
})

test('every symbol draws with primitives the renderer knows', () => {
  // UnitSymbol.svelte spells the primitives out one by one. Anything not
  // on that list is silently skipped -- so a symbol using `polygon` would
  // come out partly drawn rather than as an error.
  for (const [id, drawing] of Object.entries(SYMBOLS)) {
    assert.ok(drawing.label, `${id} has no label`)
    assert.ok(drawing.shapes.length, `${id} draws nothing`)
    for (const [kind] of drawing.shapes) {
      assert.ok(PRIMITIVES.includes(kind), `${id} uses ${kind}`)
    }
  }
})

test('an operation is matched by name before category', () => {
  // `Compressor` lives in gas_network, whose category symbol is a pipe.
  // The name has to win or every compressor in a gas network draws as
  // a length of pipe.
  assert.equal(symbolFor('Compressor', 'gas_network'), 'compressor')
  assert.equal(symbolFor('GasPipe', 'gas_network'), 'pipe')
})

test('an unknown operation falls back to its category, then to a block', () => {
  assert.equal(symbolFor('SomeNewReactor', 'reactors'), 'stirred_tank')
  assert.equal(symbolFor('SomeNewThing', 'no_such_category'), 'block')
  assert.equal(symbolFor('SomeNewThing'), 'block')
})

test('symbol() always returns something drawable', () => {
  const drawing = symbol('NothingLikeThis', 'nor this')
  assert.equal(drawing.label, 'unit')
  assert.ok(drawing.shapes.length)
})

test('units that share equipment share a symbol', () => {
  // The point of drawing by name: these pairs are the same picture even
  // though they sit in different plugins and categories.
  assert.equal(symbolFor('EOSCompressor'), symbolFor('Compressor'))
  assert.equal(symbolFor('Flash'), symbolFor('EOSFlash'))
  assert.equal(symbolFor('Ultrafiltration'), symbolFor('MembraneSeparator'))
  assert.equal(symbolFor('Turboexpander'), symbolFor('GasTurbine'))
})
