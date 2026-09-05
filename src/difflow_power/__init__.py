"""Electrical Grid Plugin for difflow: AC power flow, AC-OPF and control.

This plugin provides everything needed to model, simulate, optimise and
control steady-state electrical transmission and distribution networks
as *differentiable flowsheets*:

- Physics: the pi-model branch that serves as line, transformer and
  phase shifter alike; per-unit conversions; polynomial cost curves
- A parser-agnostic network data model
  (:class:`~difflow_power.network.PowerNetwork`) with buses, branches,
  generators and loads, plus a MATPOWER case importer
- One JAX-traceable definition of the equation set
  (:func:`~difflow_power.residuals.power_flow_residuals`), which the
  power flow, the OPF, the state estimator and the verifier all consume
- :func:`~difflow_power.powerflow.solve_power_flow`: Newton-Raphson via
  ``optimistix``, with implicit-function-theorem gradients
- :func:`~difflow_power.opf.solve_acopf`: the full nonconvex AC optimal
  power flow, solved by a primal-dual interior-point method written in
  JAX (:mod:`difflow_power.ipm`), with locational marginal prices from
  the multipliers and exact sensitivities from the KKT system
- :func:`~difflow_power.dc.solve_dcopf`, :func:`~difflow_power.dc.ptdf`
  and :func:`~difflow_power.dc.lodf`: the linearised model, market
  clearing and contingency screening
- :class:`~difflow_power.flowsheet.RadialFeederFlowsheet`: the
  backward/forward sweep, the sequential-modular method distribution
  feeders actually use
- Unit operations (:mod:`difflow_power.units`) that compose into a
  :class:`difflow.Flowsheet`
- State estimation over :mod:`difflow.reconciliation`
  (:mod:`difflow_power.estimation`), which is the same computation as
  chemical data reconciliation
- Verification against the full equation set and every limit
  (:mod:`difflow_power.verify`)

Quick start:
    >>> import difflow_power as dp
    >>>
    >>> net = dp.cases.case9()                # WSCC 9-bus benchmark
    >>> pf = dp.solve_power_flow(net)
    >>> pf.converged
    True
    >>> opf = dp.solve_acopf(net)
    >>> round(opf.cost, 2)                    # MATPOWER: 5296.69
    5296.69
    >>> sorted(opf.lmp_mw)[:3]                # $/MWh at each bus
    ['1', '2', '3']

Every benchmark result in this package's tests is asserted against
MATPOWER's published answer for the same case: the power flow on
``case9`` and ``case14``, the AC-OPF optimum on ``case3``, ``case5``,
``case9`` and ``case14``, and the DC-OPF on ``case5``. A power-flow
implementation that has not been checked that way is not worth
trusting --- the phase-shift sign, the ``tap = 0`` sentinel and the
half-charging split are each easy to get subtly wrong.

Conventions used throughout:

===============  ==========================================
quantity         unit
===============  ==========================================
voltage          per unit on the bus base kV
angle            radians internally, degrees in reports
impedance        per unit on ``base_mva``
power (state)    per unit; MW / MVAr in reports
cost             $/h, with real power in MW
===============  ==========================================
"""

__version__ = "0.1.0"

# =============================================================================
# Physics
# =============================================================================

from difflow_power.physics import (
    DEFAULT_BASE_MVA,
    DEFAULT_FREQUENCY_HZ,
    EPS_POWER,
    apparent_power_squared,
    base_impedance,
    branch_admittances,
    branch_power_flows,
    build_ybus,
    bus_injections,
    line_charging_pu,
    line_reactance_pu,
    marginal_cost,
    mw_to_pu,
    ohms_to_pu,
    polynomial_cost,
    pu_to_mw,
    pu_to_ohms,
    siemens_to_pu,
    voltage_rectangular,
)

# =============================================================================
# Streams
# =============================================================================

from difflow_power.streams import (
    P_KEY,
    Q_KEY,
    REACTIVE,
    REAL,
    SPECIES,
    apparent_power,
    complex_power,
    complex_voltage,
    current,
    from_complex,
    power_factor,
    power_stream,
)

# =============================================================================
# Network model
# =============================================================================

from difflow_power.network import (
    BUS_KINDS,
    Branch,
    Bus,
    Generator,
    Load,
    PowerNetwork,
)

# =============================================================================
# Equations
# =============================================================================

from difflow_power.residuals import (
    PowerState,
    PowerStateLayout,
    branch_flows,
    bus_injection_arrays,
    power_flow_residuals,
    power_state_layout,
    residual_names,
    state_ybus,
    total_losses,
)

# =============================================================================
# Power flow
# =============================================================================

from difflow_power.powerflow import (
    PowerFlowResult,
    Specification,
    flat_start,
    power_flow_system,
    setpoint_residuals,
    solve_power_flow,
    solve_state,
    specification_from_network,
    specification_names,
)

# =============================================================================
# Optimisation
# =============================================================================

from difflow_power import ipm
from difflow_power.ipm import (
    NLP,
    IPMResult,
    differentiable_solution,
    kkt_residuals,
    solve_nlp,
)
from difflow_power.opf import (
    ACOPFResult,
    OPFStructure,
    acopf_problem,
    acopf_structure,
    generation_cost,
    solve_acopf,
)
from difflow_power.dc import (
    DCMatrices,
    DCOPFResult,
    DCPowerFlowResult,
    contingency_flows,
    dc_matrices,
    lodf,
    ptdf,
    solve_dc_power_flow,
    solve_dcopf,
)

# =============================================================================
# Unit operations and flowsheets
# =============================================================================

from difflow_power.units import (
    BranchDrop,
    BranchFlow,
    BranchParams,
    BusNode,
    GeneratorInject,
    GeneratorParams,
    LadderClose,
    LadderCloseParams,
    LoadDraw,
    LoadParams,
    PowerSplit,
    SeriesBranch,
    ShuntDraw,
    ShuntParams,
    SlackSource,
    SlackSourceParams,
    SplitParams,
    Transformer,
)
from difflow_power.flowsheet import (
    FeederTree,
    RadialFeederFlowsheet,
    branch_stream_name,
    build_ladder_flowsheet,
    bus_stream_name,
    feeder_tree,
)

# =============================================================================
# Verification, sensitivity, estimation, plotting, cases
# =============================================================================

from difflow_power import cases, estimation, sensitivity, verify
from difflow_power.cases import (
    CASES,
    case3,
    case5,
    case9,
    case14,
    from_matpower,
    load_case,
    radial_feeder,
)
from difflow_power.verify import (
    OperatingReport,
    branch_loss_report,
    operating_report,
)
from difflow_power.sensitivity import (
    branch_flow_sensitivity,
    demand_sensitivity,
    loss_sensitivity,
    parameter_sensitivity,
    voltage_stability_margin,
)
from difflow_power.estimation import (
    estimate_state,
    estimate_state_multi,
    estimated_values,
    measurement_sigma,
    monitor_network,
    network_residual_fn,
    perturb,
)
from difflow_power.plotting import (
    circular_positions,
    draw_legend,
    draw_network,
    tree_positions,
)

__all__ = [
    "__version__",
    # physics
    "branch_admittances",
    "build_ybus",
    "bus_injections",
    "branch_power_flows",
    "voltage_rectangular",
    "polynomial_cost",
    "marginal_cost",
    "apparent_power_squared",
    "base_impedance",
    "ohms_to_pu",
    "pu_to_ohms",
    "siemens_to_pu",
    "mw_to_pu",
    "pu_to_mw",
    "line_reactance_pu",
    "line_charging_pu",
    "DEFAULT_BASE_MVA",
    "DEFAULT_FREQUENCY_HZ",
    "EPS_POWER",
    # streams
    "power_stream",
    "complex_power",
    "complex_voltage",
    "from_complex",
    "current",
    "apparent_power",
    "power_factor",
    "REAL",
    "REACTIVE",
    "P_KEY",
    "Q_KEY",
    "SPECIES",
    # network
    "PowerNetwork",
    "Bus",
    "Branch",
    "Generator",
    "Load",
    "BUS_KINDS",
    # equations
    "PowerStateLayout",
    "PowerState",
    "power_state_layout",
    "power_flow_residuals",
    "residual_names",
    "branch_flows",
    "total_losses",
    "state_ybus",
    "bus_injection_arrays",
    # power flow
    "solve_power_flow",
    "solve_state",
    "PowerFlowResult",
    "Specification",
    "specification_from_network",
    "specification_names",
    "setpoint_residuals",
    "power_flow_system",
    "flat_start",
    # optimisation
    "ipm",
    "NLP",
    "IPMResult",
    "solve_nlp",
    "kkt_residuals",
    "differentiable_solution",
    "solve_acopf",
    "ACOPFResult",
    "OPFStructure",
    "acopf_structure",
    "acopf_problem",
    "generation_cost",
    "solve_dcopf",
    "DCOPFResult",
    "solve_dc_power_flow",
    "DCPowerFlowResult",
    "dc_matrices",
    "DCMatrices",
    "ptdf",
    "lodf",
    "contingency_flows",
    # units
    "BranchParams",
    "SeriesBranch",
    "BranchDrop",
    "BranchFlow",
    "Transformer",
    "SlackSource",
    "SlackSourceParams",
    "LoadDraw",
    "LoadParams",
    "ShuntDraw",
    "ShuntParams",
    "GeneratorInject",
    "GeneratorParams",
    "BusNode",
    "PowerSplit",
    "SplitParams",
    "LadderClose",
    "LadderCloseParams",
    # flowsheets
    "RadialFeederFlowsheet",
    "FeederTree",
    "feeder_tree",
    "build_ladder_flowsheet",
    "bus_stream_name",
    "branch_stream_name",
    # verification
    "verify",
    "OperatingReport",
    "operating_report",
    "branch_loss_report",
    # sensitivity
    "sensitivity",
    "demand_sensitivity",
    "branch_flow_sensitivity",
    "loss_sensitivity",
    "parameter_sensitivity",
    "voltage_stability_margin",
    # estimation
    "estimation",
    "estimate_state",
    "estimate_state_multi",
    "monitor_network",
    "measurement_sigma",
    "network_residual_fn",
    "perturb",
    "estimated_values",
    # plotting
    "draw_network",
    "draw_legend",
    "circular_positions",
    "tree_positions",
    # cases
    "cases",
    "load_case",
    "from_matpower",
    "case3",
    "case5",
    "case9",
    "case14",
    "radial_feeder",
    "CASES",
]


def register(registry):
    """Register electrical unit operations with difflow.

    This function is called by ``difflow.plugins.load_plugins()`` when
    the plugin is discovered via entry points.

    Args:
        registry: difflow OperationRegistry instance
    """
    for name, cls, description in [
        ("SeriesBranch", SeriesBranch,
         "Line/transformer, forward mode (to-end voltage and power "
         "from the from end)"),
        ("BranchDrop", BranchDrop,
         "Line/transformer, voltage propagation at a known current"),
        ("BranchFlow", BranchFlow,
         "Line/transformer, equation-oriented (both end powers from "
         "both end voltages)"),
        ("Transformer", Transformer,
         "Tap-changing or phase-shifting transformer"),
        ("SlackSource", SlackSource,
         "Slack bus: pin a feed to a regulated voltage"),
        ("LoadDraw", LoadDraw,
         "Constant-power demand withdrawn from a stream"),
        ("ShuntDraw", ShuntDraw,
         "Fixed shunt (capacitor bank or reactor) at a bus"),
        ("GeneratorInject", GeneratorInject,
         "Generator injection with a polynomial cost curve"),
        ("BusNode", BusNode,
         "Bus: sum powers, voltage from the first inlet"),
        ("PowerSplit", PowerSplit,
         "Split a bus's outgoing power between two branches"),
        ("LadderClose", LadderClose,
         "Correct a feeder's infeed guess by the residual at its open end"),
    ]:
        registry.register(
            name=name,
            cls=cls,
            category="power_network",
            description=description,
            plugin="difflow_power",
        )
