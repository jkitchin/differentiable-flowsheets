"""Gas Transmission Network Plugin for difflow.

This plugin provides everything needed to model, simulate and optimize
steady-state gas transmission networks as *sequential-modular
differentiable flowsheets*:

- Physics: squared-pressure Weymouth pipes, resistors, adiabatic
  compressor power (plain and NLP-smoothed), unit conversions
- A parser-agnostic network data model (:class:`GasNetwork`) with
  pipes, compressor stations, valves, control valves, resistors and
  short pipes
- A topology-driven sequential decomposition (:func:`decompose`):
  spanning tree with non-invertible arcs forced in-tree, the most
  resistive loop arcs as chord/tear streams, leaf-to-root affine mass
  balances, root-to-leaf pressure propagation
- Unit operations for both computed and hand-built decompositions
- :class:`GasNetworkFlowsheet`: signed-flow tear solving (Anderson)
  and a damped, jit- and grad-safe fixed-point solve whose gradients
  come from the implicit function theorem
- A mechanical flowsheet builder (:func:`build_network_flowsheet`)
- Equation-oriented residual verification (:mod:`difflow_gas.verify`)

Quick Start:
    >>> import difflow_gas as dg
    >>>
    >>> # a triangle network: source at n0, sinks at n1 and n2
    >>> net = dg.GasNetwork(
    ...     arcs={
    ...         "p01": ("n0", "n1", "pipe"),
    ...         "p12": ("n1", "n2", "pipe"),
    ...         "p02": ("n0", "n2", "pipe"),
    ...     },
    ...     beta={"p01": 1e8, "p12": 2e8, "p02": 4e8},
    ...     supply_kg_s={"n0": 30.0, "n1": -10.0, "n2": -20.0},
    ... )
    >>> fs, dec = dg.build_network_flowsheet(net, root="n0",
    ...                                      p_slack_pa=50e5)
    >>> streams = fs.solve(tol=1e-8)          # Anderson, signed flows
    >>> dg.verify.residual_report(streams, net, dec).ok
    True

The GasLib benchmark studies that motivated this plugin (GasLib-11 and
GasLib-40 solved sequentially and simultaneously, with timing and
accuracy comparisons) live in the ``gaslib`` repository; see this
package's README for the findings that shaped the defaults here.
"""

__version__ = "0.1.0"

# =============================================================================
# Physics
# =============================================================================

from difflow_gas.physics import (
    BAR_TO_PA,
    DEFAULT_CP,
    DEFAULT_ETA_AD,
    DEFAULT_KAPPA,
    DEFAULT_TEMP_K,
    DEFAULT_Z,
    EPS_FLOW,
    compressor_power,
    kg_s_to_knm3h,
    knm3h_to_kg_s,
    nikuradse_friction,
    resistor_xi,
    smoothed_power_w,
    specific_gas_constant,
    weymouth_beta,
)

# =============================================================================
# Streams
# =============================================================================

from difflow_gas.streams import FLOW_KEY, GAS, gas_stream

# =============================================================================
# Network model and decomposition
# =============================================================================

from difflow_gas.network import (
    ARC_KINDS,
    FORCED_TREE_KINDS,
    RESISTANCE_KINDS,
    Arc,
    BalanceSpec,
    CompressorLimits,
    Decomposition,
    GasNetwork,
    decompose,
    spanning_tree,
)

# =============================================================================
# Unit operations
# =============================================================================

from difflow_gas.units import (
    MIN_P,
    MIN_P_SQUARED,
    AffineFlow,
    AffineFlowParams,
    BackPipe,
    Compressor,
    CompressorBoost,
    CompressorParams,
    ControlValveDrop,
    ControlValveParams,
    FlowMinus,
    FlowSplit,
    FlowSplitParams,
    GasPipe,
    Junction,
    OpenValve,
    PipeParams,
    PipePressure,
    PressureDrivenPipe,
    PressureEqual,
    SourceHead,
    SourceHeadParams,
    TearSplit,
    adiabatic_power_w,
)

# =============================================================================
# Flowsheet and builder
# =============================================================================

from difflow_gas.flowsheet import GasNetworkFlowsheet
from difflow_gas.flowsheets import (
    arc_flow_stream,
    build_network_flowsheet,
    cs_unit_name,
    cv_unit_name,
    node_pressure_stream,
    src_unit_name,
    total_compressor_power_w,
)

# =============================================================================
# Verification
# =============================================================================

from difflow_gas import verify
from difflow_gas.verify import ResidualReport, residual_report, residuals_from_values

# ---------------------------------------------------------------------
# Differentiable residuals and data reconciliation
# ---------------------------------------------------------------------

from difflow_gas.residuals import (
    GasStateLayout,
    gas_state_layout,
    network_residuals,
    residual_names,
)
from difflow_gas.plotting import circular_positions, draw_network
from difflow_gas.reconcile import (
    measurement_sigma,
    monitor_network,
    network_residual_fn,
    perturb,
    reconcile_network,
    reconcile_network_multi,
    reconciled_values,
)

__all__ = [
    "__version__",
    # physics
    "weymouth_beta",
    "resistor_xi",
    "nikuradse_friction",
    "specific_gas_constant",
    "compressor_power",
    "smoothed_power_w",
    "knm3h_to_kg_s",
    "kg_s_to_knm3h",
    "BAR_TO_PA",
    "DEFAULT_Z",
    "DEFAULT_KAPPA",
    "DEFAULT_ETA_AD",
    "DEFAULT_CP",
    "DEFAULT_TEMP_K",
    "EPS_FLOW",
    # streams
    "GAS",
    "FLOW_KEY",
    "gas_stream",
    # network
    "Arc",
    "GasNetwork",
    "CompressorLimits",
    "BalanceSpec",
    "Decomposition",
    "decompose",
    "spanning_tree",
    "ARC_KINDS",
    "RESISTANCE_KINDS",
    "FORCED_TREE_KINDS",
    # units
    "GasPipe",
    "BackPipe",
    "PipePressure",
    "PressureDrivenPipe",
    "PipeParams",
    "MIN_P_SQUARED",
    "Compressor",
    "CompressorBoost",
    "CompressorParams",
    "adiabatic_power_w",
    "OpenValve",
    "PressureEqual",
    "ControlValveDrop",
    "ControlValveParams",
    "MIN_P",
    "SourceHead",
    "SourceHeadParams",
    "AffineFlow",
    "AffineFlowParams",
    "FlowSplit",
    "FlowSplitParams",
    "TearSplit",
    "Junction",
    "FlowMinus",
    # flowsheet / builder
    "GasNetworkFlowsheet",
    "build_network_flowsheet",
    "total_compressor_power_w",
    "cs_unit_name",
    "cv_unit_name",
    "src_unit_name",
    "node_pressure_stream",
    "arc_flow_stream",
    # verification
    "verify",
    "ResidualReport",
    "residual_report",
    "residuals_from_values",
    # differentiable residuals
    "GasStateLayout",
    "gas_state_layout",
    "network_residuals",
    "residual_names",
    # data reconciliation
    "reconcile_network",
    "reconcile_network_multi",
    "monitor_network",
    "network_residual_fn",
    "measurement_sigma",
    "perturb",
    "reconciled_values",
    # plotting
    "draw_network",
    "circular_positions",
]


def register(registry):
    """Register gas network operations with difflow.

    This function is called by ``difflow.plugins.load_plugins()`` when
    the plugin is discovered via entry points.

    Args:
        registry: difflow OperationRegistry instance
    """
    for name, cls, description in [
        ("GasPipe", GasPipe,
         "Weymouth pipe, forward mode (outlet pressure from inlet)"),
        ("BackPipe", BackPipe,
         "Weymouth pipe, backward mode (required source pressure)"),
        ("PipePressure", PipePressure,
         "Weymouth pipe/resistor, tree propagation (either direction)"),
        ("PressureDrivenPipe", PressureDrivenPipe,
         "Weymouth pipe, pressure-driven mode (chord / tear update)"),
        ("Compressor", Compressor,
         "Fixed-ratio compressor station, forward mode"),
        ("CompressorBoost", CompressorBoost,
         "Fixed-ratio compressor station, tree propagation"),
        ("OpenValve", OpenValve,
         "Open valve: pressure equality, forward mode"),
        ("PressureEqual", PressureEqual,
         "Valve / short pipe, tree propagation (pressure equality)"),
        ("ControlValveDrop", ControlValveDrop,
         "Control valve with parametric pressure drop, tree propagation"),
        ("SourceHead", SourceHead,
         "Slack node: pin a feed to a prescribed pressure"),
        ("AffineFlow", AffineFlow,
         "Affine mass balance producing a tree-arc flow"),
        ("FlowSplit", FlowSplit,
         "Split with a fixed draw to the first outlet"),
        ("TearSplit", TearSplit,
         "Split with the draw specified by a tear stream"),
        ("Junction", Junction,
         "Node: sum flows, pressure from the first inlet"),
        ("FlowMinus", FlowMinus,
         "Flow difference of two streams (tear bookkeeping)"),
    ]:
        registry.register(
            name=name,
            cls=cls,
            category="gas_network",
            description=description,
            plugin="difflow_gas",
        )
