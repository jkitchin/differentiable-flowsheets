/**
 * PFD symbols for the catalog, as data.
 *
 * A flowsheet drawn as labelled rectangles is a graph, not a flowsheet.
 * The whole reason process engineers draw columns as columns and valves
 * as bowties is that the shape carries the operation, so a reader takes
 * in a unit at a glance instead of reading 87 names. This module is that
 * vocabulary: one symbol per kind of thing difflow can do, and a map
 * from every catalog operation onto one of them.
 *
 * Symbols are primitive lists rather than SVG source or a component per
 * symbol, for two reasons. `node --test` can check that every operation
 * resolves and that no symbol names a primitive the renderer cannot
 * draw, which is the failure that would otherwise show up as a blank
 * box on somebody's canvas. And the renderer stays one small component
 * with no `{@html}` in it.
 *
 * Geometry is in a 48x40 viewBox, drawn in `currentColor`, stroked not
 * filled unless a shape asks -- so a symbol inherits the node's colour
 * and works in either theme without a second copy.
 */

/**
 * The primitives `UnitSymbol.svelte` knows how to draw. A symbol that
 * uses anything else is a bug this list is here to make findable.
 */
export const PRIMITIVES = ['rect', 'line', 'circle', 'ellipse', 'path', 'polyline']

/** Fill a shape with the current colour at low opacity, for beds and bands. */
const WASH = { fill: 'currentColor', 'fill-opacity': 0.16, stroke: 'none' }
/** Fill solidly -- crystals, arrowheads, bus bars. */
const SOLID = { fill: 'currentColor', stroke: 'none' }

/**
 * The symbols, keyed by id.
 *
 * Each is `{label, shapes}`; `label` is what the shape is called in
 * prose, used as the accessible name and the palette tooltip, because a
 * symbol nobody can name is decoration.
 */
export const SYMBOLS = {
  /** A trayed tower: distillation, absorption, stripping, any cascade. */
  column: {
    label: 'column',
    shapes: [
      ['rect', { x: 17, y: 3, width: 14, height: 34, rx: 7 }],
      ['line', { x1: 17, y1: 12, x2: 31, y2: 12 }],
      ['line', { x1: 17, y1: 18, x2: 31, y2: 18 }],
      ['line', { x1: 17, y1: 24, x2: 31, y2: 24 }],
      ['line', { x1: 17, y1: 30, x2: 31, y2: 30 }],
    ],
  },

  /** A packed tower: the same vessel, filled rather than trayed. */
  packed_column: {
    label: 'packed column',
    shapes: [
      ['rect', { x: 17, y: 3, width: 14, height: 34, rx: 7 }],
      ['rect', { x: 18, y: 10, width: 12, height: 20, rx: 3, ...WASH }],
      ['line', { x1: 19, y1: 14, x2: 29, y2: 20 }],
      ['line', { x1: 29, y1: 14, x2: 19, y2: 20 }],
      ['line', { x1: 19, y1: 22, x2: 29, y2: 28 }],
      ['line', { x1: 29, y1: 22, x2: 19, y2: 28 }],
    ],
  },

  /** A stirred vessel: CSTR, bioreactor, mixer-settler, any tank. */
  stirred_tank: {
    label: 'stirred tank',
    shapes: [
      ['path', { d: 'M13 6h22v18a11 9 0 0 1-22 0z' }],
      ['line', { x1: 24, y1: 3, x2: 24, y2: 22 }],
      ['line', { x1: 17, y1: 22, x2: 31, y2: 22 }],
      ['line', { x1: 17, y1: 19, x2: 17, y2: 25 }],
      ['line', { x1: 31, y1: 19, x2: 31, y2: 25 }],
    ],
  },

  /** A tube in a shell: plug flow. */
  tubular: {
    label: 'tubular reactor',
    shapes: [
      ['rect', { x: 5, y: 11, width: 38, height: 18, rx: 4 }],
      ['polyline', {
        points: '10,20 14,14 19,26 24,14 29,26 34,14 38,20',
        fill: 'none',
      }],
    ],
  },

  /** Shell and tube, with the U-bend that says which way the tubes run. */
  heat_exchanger: {
    label: 'heat exchanger',
    shapes: [
      ['rect', { x: 6, y: 9, width: 36, height: 22, rx: 5 }],
      ['path', { d: 'M13 14h16a6 6 0 0 1 0 12H13', fill: 'none' }],
      ['line', { x1: 13, y1: 14, x2: 13, y2: 26 }],
      ['line', { x1: 20, y1: 31, x2: 20, y2: 36 }],
      ['line', { x1: 30, y1: 4, x2: 30, y2: 9 }],
    ],
  },

  /** Heat in: the exchanger circle with the arrow pointing at it. */
  heater: {
    label: 'heater',
    shapes: [
      ['circle', { cx: 24, cy: 20, r: 13 }],
      ['line', { x1: 24, y1: 27, x2: 24, y2: 13 }],
      ['path', { d: 'M20 17l4-4 4 4', fill: 'none' }],
      ['line', { x1: 15, y1: 30, x2: 33, y2: 30 }],
    ],
  },

  /** Heat out. */
  cooler: {
    label: 'cooler',
    shapes: [
      ['circle', { cx: 24, cy: 20, r: 13 }],
      ['line', { x1: 24, y1: 13, x2: 24, y2: 27 }],
      ['path', { d: 'M20 23l4 4 4-4', fill: 'none' }],
      ['line', { x1: 15, y1: 10, x2: 33, y2: 10 }],
    ],
  },

  /** A drum with a level in it: vapour off the top, liquid out the bottom. */
  flash_drum: {
    label: 'flash drum',
    shapes: [
      ['path', { d: 'M15 13a9 9 0 0 1 18 0v14a9 9 0 0 1-18 0z' }],
      ['path', { d: 'M15 25h18v2a9 9 0 0 1-18 0z', ...WASH }],
      ['line', { x1: 15, y1: 25, x2: 33, y2: 25 }],
      ['circle', { cx: 21, cy: 17, r: 1.6, ...SOLID }],
      ['circle', { cx: 27, cy: 14, r: 1.2, ...SOLID }],
    ],
  },

  /** A box split down the middle: a separator that is not a real column. */
  separator: {
    label: 'separator',
    shapes: [
      ['rect', { x: 9, y: 8, width: 30, height: 24, rx: 3 }],
      ['line', { x1: 24, y1: 8, x2: 24, y2: 32 }],
      ['line', { x1: 13, y1: 15, x2: 20, y2: 15 }],
      ['line', { x1: 28, y1: 25, x2: 35, y2: 25 }],
    ],
  },

  /** A membrane: the dashed line is the membrane, and it is the point. */
  membrane: {
    label: 'membrane',
    shapes: [
      ['rect', { x: 9, y: 9, width: 30, height: 22, rx: 3 }],
      ['line', { x1: 24, y1: 6, x2: 24, y2: 34, 'stroke-dasharray': '3 3' }],
      ['path', { d: 'M12 15h8m-8 10h8', fill: 'none' }],
      ['path', { d: 'M28 20h8', fill: 'none' }],
    ],
  },

  /** A packed bed you cycle: PSA, TSA, VSA. */
  adsorber: {
    label: 'adsorber',
    shapes: [
      ['rect', { x: 16, y: 4, width: 16, height: 32, rx: 4 }],
      ['rect', { x: 17, y: 11, width: 14, height: 18, ...WASH }],
      ['circle', { cx: 21, cy: 15, r: 1.5, ...SOLID }],
      ['circle', { cx: 27, cy: 15, r: 1.5, ...SOLID }],
      ['circle', { cx: 24, cy: 20, r: 1.5, ...SOLID }],
      ['circle', { cx: 21, cy: 25, r: 1.5, ...SOLID }],
      ['circle', { cx: 27, cy: 25, r: 1.5, ...SOLID }],
    ],
  },

  /** A spinning bowl. */
  centrifuge: {
    label: 'centrifuge',
    shapes: [
      ['path', { d: 'M10 8h28l-9 24h-10z' }],
      ['ellipse', { cx: 24, cy: 8, rx: 14, ry: 3.5 }],
      ['path', { d: 'M19 17a6 4 0 0 1 10 0', fill: 'none' }],
    ],
  },

  /** A column with bands moving down it. */
  chromatography: {
    label: 'chromatography column',
    shapes: [
      ['rect', { x: 18, y: 3, width: 12, height: 34, rx: 3 }],
      ['rect', { x: 19, y: 10, width: 10, height: 5, ...WASH }],
      ['rect', { x: 19, y: 19, width: 10, height: 5, fill: 'currentColor', 'fill-opacity': 0.35, stroke: 'none' }],
      ['rect', { x: 19, y: 28, width: 10, height: 4, ...WASH }],
    ],
  },

  /** A vessel growing solids. */
  precipitator: {
    label: 'precipitator',
    shapes: [
      ['path', { d: 'M13 6h22v16l-11 14L13 22z' }],
      ['path', { d: 'M20 14l2.5 4h-5z', ...SOLID }],
      ['path', { d: 'M28 18l2.5 4h-5z', ...SOLID }],
      ['path', { d: 'M23 22l2.5 4h-5z', ...SOLID }],
    ],
  },

  /** Work in: the trapezoid narrows the way the gas does. */
  compressor: {
    label: 'compressor',
    shapes: [
      ['path', { d: 'M11 7l26 8v10l-26 8z' }],
      ['line', { x1: 24, y1: 4, x2: 24, y2: 11 }],
      ['line', { x1: 18, y1: 4, x2: 30, y2: 4 }],
    ],
  },

  /** Work out. */
  turbine: {
    label: 'turbine',
    shapes: [
      ['path', { d: 'M11 15l26-8v26l-26-8z' }],
      ['line', { x1: 24, y1: 4, x2: 24, y2: 9 }],
      ['line', { x1: 18, y1: 4, x2: 30, y2: 4 }],
    ],
  },

  /** A bowtie, with a stem so it reads as a valve and not an hourglass. */
  valve: {
    label: 'valve',
    shapes: [
      ['path', { d: 'M12 12l12 8-12 8z' }],
      ['path', { d: 'M36 12L24 20l12 8z' }],
      ['line', { x1: 24, y1: 20, x2: 24, y2: 9 }],
      ['line', { x1: 18, y1: 7, x2: 30, y2: 7 }],
    ],
  },

  /** A control valve: the same body under an actuator. */
  control_valve: {
    label: 'control valve',
    shapes: [
      ['path', { d: 'M12 18l12 8-12 8z' }],
      ['path', { d: 'M36 18L24 26l12 8z' }],
      ['line', { x1: 24, y1: 26, x2: 24, y2: 14 }],
      ['ellipse', { cx: 24, cy: 10, rx: 8, ry: 5 }],
    ],
  },

  /** A run of pipe, flanged at both ends. */
  pipe: {
    label: 'pipe',
    shapes: [
      ['line', { x1: 8, y1: 20, x2: 40, y2: 20 }],
      ['line', { x1: 11, y1: 14, x2: 11, y2: 26 }],
      ['line', { x1: 37, y1: 14, x2: 37, y2: 26 }],
      ['path', { d: 'M26 16l5 4-5 4', fill: 'none' }],
    ],
  },

  /** Streams in, one out. */
  mixer: {
    label: 'mixer',
    shapes: [
      ['circle', { cx: 24, cy: 20, r: 9 }],
      ['line', { x1: 8, y1: 11, x2: 16, y2: 16 }],
      ['line', { x1: 8, y1: 29, x2: 16, y2: 24 }],
      ['line', { x1: 33, y1: 20, x2: 42, y2: 20 }],
      ['path', { d: 'M19 23l10-6M19 17l10 6', fill: 'none' }],
    ],
  },

  /** One in, streams out. */
  splitter: {
    label: 'splitter',
    shapes: [
      ['circle', { cx: 24, cy: 20, r: 5 }],
      ['line', { x1: 6, y1: 20, x2: 19, y2: 20 }],
      ['line', { x1: 29, y1: 20, x2: 42, y2: 11 }],
      ['line', { x1: 29, y1: 20, x2: 42, y2: 29 }],
    ],
  },

  /** A flame. */
  combustor: {
    label: 'combustor',
    shapes: [
      ['circle', { cx: 24, cy: 20, r: 13 }],
      ['path', { d: 'M24 28c-4 0-6-3-6-6 0-4 4-5 4-9 3 2 4 4 4 6 1-1 1-2 1-3 2 2 3 4 3 6 0 3-2 6-6 6z', ...WASH }],
      ['path', { d: 'M24 28c-4 0-6-3-6-6 0-4 4-5 4-9 3 2 4 4 4 6 1-1 1-2 1-3 2 2 3 4 3 6 0 3-2 6-6 6z', fill: 'none' }],
    ],
  },

  /** A busbar. */
  bus: {
    label: 'bus',
    shapes: [
      ['rect', { x: 21, y: 4, width: 6, height: 32, rx: 1, ...SOLID }],
      ['line', { x1: 8, y1: 13, x2: 21, y2: 13 }],
      ['line', { x1: 27, y1: 27, x2: 40, y2: 27 }],
    ],
  },

  /** A machine: the circle with a sine in it. */
  generator: {
    label: 'generator',
    shapes: [
      ['circle', { cx: 24, cy: 20, r: 12 }],
      ['path', { d: 'M17 22a3.5 3.5 0 0 1 7-4 3.5 3.5 0 0 1 7-4', fill: 'none' }],
      ['line', { x1: 36, y1: 20, x2: 42, y2: 20 }],
    ],
  },

  /** A draw: the arrow leaves the network. */
  load: {
    label: 'load',
    shapes: [
      ['line', { x1: 24, y1: 5, x2: 24, y2: 26 }],
      ['path', { d: 'M17 22l7 9 7-9z', ...SOLID }],
      ['line', { x1: 14, y1: 36, x2: 34, y2: 36 }],
    ],
  },

  /** Two coupled windings. */
  transformer: {
    label: 'transformer',
    shapes: [
      ['circle', { cx: 19, cy: 20, r: 9 }],
      ['circle', { cx: 29, cy: 20, r: 9 }],
      ['line', { x1: 4, y1: 20, x2: 10, y2: 20 }],
      ['line', { x1: 38, y1: 20, x2: 44, y2: 20 }],
    ],
  },

  /** A series element: the impedance box on a line. */
  branch: {
    label: 'branch',
    shapes: [
      ['line', { x1: 4, y1: 20, x2: 16, y2: 20 }],
      ['rect', { x: 16, y: 14, width: 16, height: 12, rx: 1 }],
      ['line', { x1: 32, y1: 20, x2: 44, y2: 20 }],
    ],
  },

  /** A constraint rather than a machine: two streams held equal. */
  equality: {
    label: 'equality',
    shapes: [
      ['rect', { x: 10, y: 9, width: 28, height: 22, rx: 4, 'stroke-dasharray': '4 3' }],
      ['line', { x1: 18, y1: 17, x2: 30, y2: 17 }],
      ['line', { x1: 18, y1: 23, x2: 30, y2: 23 }],
    ],
  },

  /** A composite: a flowsheet standing in for itself. */
  train: {
    label: 'flowsheet',
    shapes: [
      ['rect', { x: 4, y: 12, width: 13, height: 12, rx: 2 }],
      ['rect', { x: 21, y: 20, width: 13, height: 12, rx: 2 }],
      ['rect', { x: 31, y: 6, width: 13, height: 12, rx: 2 }],
      ['line', { x1: 17, y1: 20, x2: 21, y2: 24 }],
      ['line', { x1: 34, y1: 22, x2: 37, y2: 18 }],
    ],
  },

  /** The fallback, for a plugin whose units this vocabulary has not met. */
  block: {
    label: 'unit',
    shapes: [
      ['rect', { x: 9, y: 10, width: 30, height: 20, rx: 3 }],
      ['line', { x1: 4, y1: 20, x2: 9, y2: 20 }],
      ['line', { x1: 39, y1: 20, x2: 44, y2: 20 }],
    ],
  },
}

/**
 * Operation name -> symbol id. Names, not categories, because the
 * category is about the domain and the symbol is about the equipment: a
 * `Compressor` in `gas_network` and an `EOSCompressor` in
 * `pressure_change` are the same drawing, and `Flash` and `Mixer` share
 * the `separations` category while sharing nothing else.
 */
export const OPERATION_SYMBOLS = {
  // reactors
  CSTR: 'stirred_tank',
  PFR: 'tubular',
  GasPFR: 'tubular',
  SemiBatchReactor: 'stirred_tank',
  FedBatchReactor: 'stirred_tank',

  // bio
  ContinuousBioreactor: 'stirred_tank',
  FedBatchBioreactor: 'stirred_tank',
  Centrifuge: 'centrifuge',
  DiscStackCentrifuge: 'centrifuge',
  Ultrafiltration: 'membrane',
  Diafiltration: 'membrane',
  TFF: 'membrane',
  ProteinAChromatography: 'chromatography',
  IonExchangeChromatography: 'chromatography',
  SizeExclusionChromatography: 'chromatography',

  // separations
  Flash: 'flash_drum',
  EOSFlash: 'flash_drum',
  PHFlash: 'flash_drum',
  Mixer: 'mixer',
  Splitter: 'splitter',
  ComponentSeparator: 'separator',

  // distillation and contacting
  DistillationColumn: 'column',
  ShortcutColumn: 'column',
  MultistageCascade: 'column',
  DifferentialContactor: 'packed_column',
  LLEEquilibrium: 'stirred_tank',

  // heat
  Heater: 'heater',
  Cooler: 'cooler',
  CounterCurrentHX: 'heat_exchanger',
  CoCurrentHX: 'heat_exchanger',
  CrossFlowHX: 'heat_exchanger',
  ShellAndTubeHX: 'heat_exchanger',
  EnthalpyCounterCurrentHX: 'heat_exchanger',

  // pressure
  EOSCompressor: 'compressor',
  Turboexpander: 'turbine',
  JTValve: 'valve',

  // power island
  Combustor: 'combustor',
  GasTurbine: 'turbine',
  GasCompressor: 'compressor',

  // carbon capture
  AmineAbsorber: 'packed_column',
  AmineStripper: 'packed_column',
  MembraneSeparator: 'membrane',
  MultistageMembrane: 'membrane',
  PSAUnit: 'adsorber',
  TSAUnit: 'adsorber',
  VSAUnit: 'adsorber',
  TVSAUnit: 'adsorber',

  // REE
  REEExtractor: 'stirred_tank',
  REEMixerSettler: 'stirred_tank',
  REEScrubber: 'column',
  REEStripper: 'column',
  Saponifier: 'stirred_tank',
  CeriumOxidizer: 'stirred_tank',
  OxalatePrecipitator: 'precipitator',
  CarbonatePrecipitator: 'precipitator',
  HydroxidePrecipitator: 'precipitator',
  ExtractStripCircuit: 'train',
  ExtractScrubStripCircuit: 'train',
  SplitShellCascade: 'train',
  FullSeparationTrain: 'train',
  GroupSeparator: 'train',

  // gas network
  GasPipe: 'pipe',
  BackPipe: 'pipe',
  PipePressure: 'pipe',
  PressureDrivenPipe: 'pipe',
  Compressor: 'compressor',
  CompressorBoost: 'compressor',
  OpenValve: 'valve',
  ControlValveDrop: 'control_valve',
  PressureEqual: 'equality',
  AffineFlow: 'equality',
  SourceHead: 'generator',
  Junction: 'mixer',
  FlowSplit: 'splitter',
  TearSplit: 'splitter',
  FlowMinus: 'load',

  // power network
  BusNode: 'bus',
  SeriesBranch: 'branch',
  BranchFlow: 'branch',
  BranchDrop: 'branch',
  LadderClose: 'branch',
  Transformer: 'transformer',
  GeneratorInject: 'generator',
  SlackSource: 'generator',
  LoadDraw: 'load',
  ShuntDraw: 'load',
  PowerSplit: 'splitter',
}

/**
 * Last resort when an operation is not named above: the category. A
 * plugin can register units this module has never heard of, and a
 * category-shaped guess beats a blank box.
 */
export const CATEGORY_SYMBOLS = {
  reactors: 'stirred_tank',
  bioreactors: 'stirred_tank',
  distillation: 'column',
  extraction: 'column',
  separations: 'separator',
  filtration: 'membrane',
  chromatography: 'chromatography',
  heat_transfer: 'heat_exchanger',
  pressure_change: 'compressor',
  power: 'turbine',
  gas_network: 'pipe',
  power_network: 'branch',
  carbon_capture_amine: 'packed_column',
  carbon_capture_membrane: 'membrane',
  carbon_capture_adsorption: 'adsorber',
  ree_extraction: 'stirred_tank',
  ree_precipitation: 'precipitator',
  ree_flowsheets: 'train',
}

/**
 * The symbol id for an operation.
 *
 * @param {string} operation  a catalog name, e.g. `"CSTR"`
 * @param {string} [category]  the catalog category, used only as a fallback
 * @returns {string} a key of {@link SYMBOLS}; `"block"` if nothing matched
 */
export function symbolFor(operation, category = '') {
  return (
    OPERATION_SYMBOLS[operation] ??
    CATEGORY_SYMBOLS[category] ??
    'block'
  )
}

/** `{label, shapes}` for an operation, ready to draw. */
export function symbol(operation, category = '') {
  return SYMBOLS[symbolFor(operation, category)]
}
