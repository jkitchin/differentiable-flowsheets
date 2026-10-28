"""Base classes and protocols for dynamic unit operations.

This module defines the interface that all dynamic units must implement
to work with the unified integrator framework. Units can operate in
different modes:

- **Pure ODE**: dx/dt = f(x, u, t) - state evolves continuously
- **Pure algebraic**: 0 = g(x, u) - instantaneous equilibrium (steady-state)
- **DAE (mixed)**: dx/dt = f(x, y, u, t), 0 = g(x, y, u, t)

The DynamicUnit protocol is the core abstraction that allows the
integrator to work with any unit type uniformly.
"""

from contextlib import contextmanager
from typing import Protocol, Callable, Any, Literal, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, make_stream
from difflow.init_args import record_init_args
from difflow.dynamic.state import StateSpec, StateVector, StateVar
from difflow.numerics import safe_divide


def _takes_args(fn) -> bool:
    """True for a feed written ``(t, args)``: a second positional parameter
    that is required or is named ``args`` (or a ``*args``)."""
    import inspect
    try:
        prms = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = []
    for prm in prms:
        if prm.kind is prm.VAR_POSITIONAL:
            return len(positional) <= 1
        if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD):
            positional.append(prm)
    if len(positional) < 2:
        return False
    second = positional[1]
    return second.default is second.empty or second.name == "args"


def _value_at(v: Any, t: Array, args: Any, inputs: dict[str, "Stream"]) -> Array:
    """Evaluate a unit input given as a constant, ``f(t)``, ``f(t, args)``
    (``args`` as handed to ``DynamicFlowsheet.simulate``, read the way
    ``add_feed`` reads a feed) or an inlet port name (that stream's ``T``)."""
    if isinstance(v, str):
        if v not in inputs:
            raise KeyError(f"no inlet stream on port {v!r} (ports: {sorted(inputs)})")
        return inputs[v]["T"]
    if callable(v):
        return v(t, args) if _takes_args(v) else v(t)
    return v


# Type aliases
Params = dict[str, Any]
RateFunction = Callable[[dict[str, Array], Array, Params], Array]


@runtime_checkable
class DynamicUnit(Protocol):
    """Protocol defining the interface for dynamic unit operations.

    All dynamic units must implement these methods to work with the
    unified integrator. This enables polymorphic handling of different
    unit types (reactors, separators, tanks, etc.) in a flowsheet.

    The key methods are:
    - state_spec: Declares what state variables the unit tracks
    - derivatives: Computes dx/dt for ODE integration
    - outputs: Maps internal states to outlet stream(s)

    Optional methods for advanced functionality:
    - residual: For steady-state or algebraic constraints
    - events: For discontinuity detection (phase changes, etc.)
    """

    def state_spec(self) -> StateSpec:
        """Return specification of state variables for this unit.

        This defines the "shape" of the unit's dynamic state - what
        variables are tracked, their units, bounds, and scales.

        Returns:
            StateSpec describing all state variables
        """
        ...

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute initial state from inlet streams and parameters.

        Args:
            inputs: Dictionary of inlet streams by name
            params: Optional parameters to override defaults

        Returns:
            Initial state array matching state_spec order
        """
        ...

    def derivatives(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute time derivatives of state variables.

        This is the core ODE function: dx/dt = f(t, x, inputs, params)

        Args:
            t: Current time
            state: Current state array (matches state_spec order)
            inputs: Dictionary of inlet streams by name
            params: Optional parameter overrides

        Returns:
            Array of derivatives dx/dt (same shape as state)
        """
        ...

    def outputs(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute outlet streams from current state.

        Maps the internal state to outlet stream(s) that can be
        connected to downstream units.

        Args:
            t: Current time
            state: Current state array
            inputs: Dictionary of inlet streams
            params: Optional parameter overrides

        Returns:
            Dictionary of outlet streams by name
        """
        ...


class DynamicUnitBase(ABC):
    """Abstract base class for dynamic units with common functionality.

    Provides default implementations and utilities while requiring
    subclasses to implement the core dynamic methods.

    This base class handles:
    - Parameter management
    - State vector wrapping/unwrapping
    - Common validation
    - Steady-state solving via derivatives=0
    - Per-call ``params`` overrides and integrator ``args`` (``wants_args``)

    Subclasses must implement:
    - _build_state_spec(): Define state variables
    - _derivatives(): Compute dx/dt
    - _outputs(): Map states to outlet streams
    """

    #: DynamicFlowsheet hands ``simulate(args=...)`` to ``derivatives`` and
    #: ``outputs`` of a unit that sets this.
    wants_args = True

    def __init_subclass__(cls, **kwargs):
        # Remember each unit's constructor call, so difflow.serialize can
        # rebuild it as built: DynamicCSTR(volume=...) keeps volume only
        # as params["V"], where no reader would look for it.
        super().__init_subclass__(**kwargs)
        record_init_args(cls)

    def __init__(
        self,
        params: Params | None = None,
        name: str | None = None,
    ):
        """Initialize dynamic unit.

        Args:
            params: Unit parameters (kinetics, geometry, etc.)
            name: Optional name for the unit
        """
        self._params = params or {}
        self._args: Any = None
        self.name = name or self.__class__.__name__
        self._state_spec: StateSpec | None = None

    @property
    def params(self) -> Params:
        """Unit parameters."""
        return self._params

    @abstractmethod
    def _build_state_spec(self) -> StateSpec:
        """Build and return the state specification.

        Subclasses must implement this to define their state variables.
        """
        ...

    def state_spec(self) -> StateSpec:
        """Return specification of state variables (cached)."""
        if self._state_spec is None:
            self._state_spec = self._build_state_spec()
        return self._state_spec

    @abstractmethod
    def _derivatives(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """Compute derivatives using StateVector interface.

        Subclasses implement this with convenient named access to states.
        """
        ...

    @abstractmethod
    def _outputs(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> dict[str, Stream]:
        """Compute outputs using StateVector interface."""
        ...

    def derivatives(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
        args: Any = None,
    ) -> Array:
        """Compute time derivatives (public interface).

        Wraps state array in StateVector for convenient access. ``params``
        overrides the unit's parameters for this call only (it is not
        stored, so a traced value under ``jax.grad`` does not leak into the
        unit); ``args`` is the integrator ``args`` of
        ``DynamicFlowsheet.simulate``, read by inputs given as ``f(t, args)``.
        """
        state_vec = StateVector(state, self.state_spec())
        with self._call_context(params, args):
            return self._derivatives(t, state_vec, inputs)

    def outputs(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
        args: Any = None,
    ) -> dict[str, Stream]:
        """Compute outlet streams (public interface); ``params`` and ``args``
        as in :meth:`derivatives`."""
        state_vec = StateVector(state, self.state_spec())
        with self._call_context(params, args):
            return self._outputs(t, state_vec, inputs)

    @contextmanager
    def _call_context(self, params: Params | None, args: Any):
        """Run one call with ``params`` merged over the unit's own and
        ``args`` visible as ``self._args``; restore both afterwards."""
        saved_params, saved_args = self._params, self._args
        if params:
            self._params = {**saved_params, **params}
        self._args = args
        try:
            yield
        finally:
            self._params, self._args = saved_params, saved_args

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Default initial state from spec defaults.

        Override for unit-specific initialization logic.
        """
        return self.state_spec().get_default_initial()

    def residual(
        self,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Residual for steady-state: r(x) = dx/dt.

        Default implementation uses derivatives at t=0.
        Override for algebraic constraints.
        """
        return self.derivatives(jnp.array(0.0), state, inputs, params)

    def __repr__(self) -> str:
        n_states = self.state_spec().n_states
        return f"{self.__class__.__name__}(name='{self.name}', n_states={n_states})"


# =============================================================================
# Common Dynamic Unit Implementations
# =============================================================================

class DynamicCSTR(DynamicUnitBase):
    """Dynamic CSTR with holdup dynamics.

    State variables:
    - n_i: Moles of each species in reactor (mol)
    - T: Reactor temperature (K) [if non-isothermal]

    Material balance:
        dn_i/dt = F_in * x_in_i - F_out * x_out_i + V * sum_j(nu_ij * r_j)

    Energy balance (non-isothermal):
        d(n_total * Cp * T)/dt = F_in*H_in - F_out*H_out + Q + V*sum_j(r_j*(-dH_j))

    with the heat added ``Q`` set by ``mode``:

    - ``"isothermal"``: no energy balance; T = inlet T.
    - ``"adiabatic"``: Q = 0.
    - ``"specified_duty"``: Q = ``duty`` (W, positive heats the reactor).
    - ``"jacket"``: Q = UA * (Tc - T), a cooling/heating jacket with overall
      conductance ``UA`` (W/K) and coolant temperature ``Tc`` (K).

    ``duty``, ``UA`` and ``Tc`` may each be a constant (float or JAX array),
    a function of time ``f(t)``, or ``f(t, args)`` reading the ``args`` passed
    to ``DynamicFlowsheet.simulate`` (the disturbance channel ``add_feed``
    uses). Each may also be the name of an inlet port, whose stream
    temperature ``T`` is then the value: a coolant stream for ``Tc``, or the
    signal of a :class:`~difflow.dynamic.controller.PIDController` for any
    of the three. The outlet does not read those ports, so a controller that
    measures the outlet and drives one of them closes the loop without an
    algebraic loop. They are stored under the
    same keys in ``params``, so ``simulate(params={"UA": ua})`` overrides
    them, differentiably. Cp = 75 J/(mol K) and the liquid molar density
    55 500 mol/m^3 are fixed (water-like).

    For perfectly mixed: x_out = n / sum(n)

    Attributes:
        V: Reactor volume (m^3).
        dH_rxn: Heats of reaction, negative for exothermic (J/mol).
        duty: Heat added in ``"specified_duty"`` mode (W).
        UA: Jacket overall conductance in ``"jacket"`` mode (W/K).
        Tc: Jacket coolant temperature in ``"jacket"`` mode (K).
    """

    symbol = "Dynamic CSTR"
    equations = [
        r"\frac{dn_i}{dt} = F_\mathrm{in}\,x_{i,\mathrm{in}} - F_\mathrm{out}\,x_i + V\sum_j \nu_{ij}\, r_j",
        r"\frac{d(n_\mathrm{tot}\,C_p\,T)}{dt} = F_\mathrm{in} h_\mathrm{in} - F_\mathrm{out} h + Q + V\sum_j r_j(-\Delta H_{r,j})",
        r"x_i = n_i / n_\mathrm{tot}",
    ]
    assumptions = [
        "Perfectly mixed (outlet = reactor contents).",
        "Constant volume unless ``variable_volume=True``.",
        "Integrated as an ODE in the reactor moles and (optional) temperature.",
    ]
    references = ["Fogler, Elements of Chemical Reaction Engineering, 5e, Ch. 13."]
    parameter_symbols = {"V": "V", "dH_rxn": r"\Delta H_{r,j}", "duty": "Q",
                         "UA": "UA", "Tc": "T_c"}
    parameter_units = {"V": "m^3", "dH_rxn": "J/mol", "duty": "W", "UA": "W/K",
                       "Tc": "K"}
    numerical_method = "ODE integration via diffrax (Tsit5/Kvaerno5 depending on stiffness)."

    def __init__(
        self,
        volume: float | Array,
        rate_fn: RateFunction,
        stoich: Array,
        species_order: list[str],
        rate_params: Params | None = None,
        dH_rxn: Array | None = None,
        mode: Literal["isothermal", "adiabatic", "specified_duty", "jacket"] = "isothermal",
        variable_volume: bool = False,
        outlet_flow_fn: Callable[[Array], Array] | None = None,
        name: str | None = None,
        *,
        duty: float | Array | Callable | None = None,
        UA: float | Array | Callable | None = None,
        Tc: float | Array | Callable | str | None = None,
    ):
        """Initialize dynamic CSTR.

        Args:
            volume: Reactor volume (m³) — initial volume if variable_volume=True
            rate_fn: Reaction rate function: rate_fn(C, T, params) -> r
            stoich: Stoichiometry matrix (n_species, n_reactions)
            species_order: List of species names
            rate_params: Parameters for rate function
            dH_rxn: Heats of reaction (J/mol), negative for exothermic
            mode: Energy balance mode: ``"isothermal"``, ``"adiabatic"``,
                ``"specified_duty"`` (needs ``duty``) or ``"jacket"`` (needs
                ``UA`` and ``Tc``).
            variable_volume: If True, volume is a dynamic state variable
                            (dV/dt = Q_in - Q_out)
            outlet_flow_fn: Optional outlet volumetric-flow law ``Q_out(V)``
                (m³/s) as a function of the current liquid volume. When given
                (with ``variable_volume=True``) the outlet draw is set
                independently of the inlet, so the level actually rises or
                falls under feed/draw imbalance (overflow or dry-down) — e.g.
                a constant draw ``lambda V: 1e-4`` or gravity drain
                ``lambda V: k*jnp.sqrt(V)``. When None, the outlet volumetric
                flow matches the inlet (constant volume; backward compatible).
            name: Unit name
            duty: Heat added (W, positive heats) in ``"specified_duty"``
                mode: a constant, ``f(t)``, ``f(t, args)`` or an inlet port
                name whose stream ``T`` carries it (a controller's signal).
            UA: Jacket overall conductance (W/K) in ``"jacket"`` mode: a
                constant, ``f(t)`` or ``f(t, args)`` (fouling, say), or an
                inlet port name whose stream ``T`` carries it.
            Tc: Coolant temperature (K) in ``"jacket"`` mode: a constant,
                ``f(t)``, ``f(t, args)``, or an inlet port name whose stream
                temperature is the coolant temperature (a coolant stream, or
                a controller's output signal).

        Raises:
            ValueError: unknown ``mode``; ``"specified_duty"`` without
                ``duty``; ``"jacket"`` without ``UA`` or ``Tc``; or ``duty``,
                ``UA``, ``Tc`` given for a mode that does not use them.

        Example:
            >>> import jax.numpy as jnp
            >>> cstr = DynamicCSTR(volume=1.0, rate_fn=lambda C, T, p: jnp.array([0.0]),
            ...                    stoich=jnp.array([[-1.0], [1.0]]), species_order=["A", "B"],
            ...                    mode="jacket", UA=500.0, Tc=lambda t: 300.0 + 0.01 * t)
        """
        modes = ("isothermal", "adiabatic", "specified_duty", "jacket")
        if mode not in modes:
            raise ValueError(f"mode must be one of {modes}, got {mode!r}")
        if mode == "specified_duty" and duty is None:
            raise ValueError("mode='specified_duty' needs duty= (W; constant, f(t) or f(t, args))")
        if mode == "jacket" and (UA is None or Tc is None):
            missing = [k for k, v in (("UA", UA), ("Tc", Tc)) if v is None]
            raise ValueError(f"mode='jacket' needs {' and '.join(missing)}= (Q = UA*(Tc - T))")
        if duty is not None and mode != "specified_duty":
            raise ValueError(f"duty= is used only by mode='specified_duty', not {mode!r}")
        if (UA is not None or Tc is not None) and mode != "jacket":
            raise ValueError(f"UA= and Tc= are used only by mode='jacket', not {mode!r}")

        def as_input(v):
            return v if v is None or callable(v) or isinstance(v, str) else jnp.asarray(v)

        params = {
            "V": jnp.asarray(volume),
            "rate_fn": rate_fn,
            "stoich": jnp.asarray(stoich),
            "species_order": species_order,
            "rate_params": rate_params or {},
            "dH_rxn": jnp.asarray(dH_rxn) if dH_rxn is not None else None,
            "mode": mode,
            "variable_volume": variable_volume,
            "outlet_flow_fn": outlet_flow_fn,
            "duty": as_input(duty),
            "UA": as_input(UA),
            "Tc": as_input(Tc),
        }
        super().__init__(params, name)

    def heat_input(self, t: Array, T: Array, inputs: dict[str, Stream]) -> Array:
        """Heat added to the reactor at time ``t`` and temperature ``T`` (W).

        0 when isothermal or adiabatic, ``duty`` in ``"specified_duty"`` mode
        and ``UA * (Tc - T)`` in ``"jacket"`` mode. Reads the per-call
        ``params``/``args`` while called from :meth:`derivatives`.
        """
        p, args = self.params, self._args
        if p["mode"] == "specified_duty":
            # "Q_spec" is the key older code set through simulate(params=...).
            duty = p["Q_spec"] if p.get("Q_spec") is not None else p["duty"]
            return _value_at(duty, t, args, inputs)
        if p["mode"] == "jacket":
            UA = _value_at(p["UA"], t, args, inputs)
            Tc = _value_at(p["Tc"], t, args, inputs)
            return UA * (Tc - T)
        return jnp.asarray(0.0)

    #: Output port names.
    output_ports = ("outlet",)

    def _signal_ports(self) -> set[str]:
        """Inlet ports named by ``duty``, ``UA`` or ``Tc``: signals, not feed."""
        return {v for v in (self.params.get(k) for k in ("duty", "UA", "Tc"))
                if isinstance(v, str)}

    def output_dependencies(self, inlet_ports) -> dict[str, set[str]]:
        """The outlet reads the state and the process feed, not the ports
        that carry ``duty``, ``UA`` or ``Tc``. That is what lets a controller
        that measures the outlet drive the jacket in one
        ``DynamicFlowsheet`` without an algebraic loop."""
        return {"outlet": set(inlet_ports) - self._signal_ports()}

    def partial_outputs(self, t, state, inputs, params=None) -> dict[str, Stream]:
        """The outlet from the process feed alone (see ``output_dependencies``)."""
        return self.outputs(t, state, inputs, params)

    def _process_inlet(self, inputs: dict[str, Stream]) -> Stream:
        """The process feed: port ``"inlet"``, else the first stream that is
        not a port named by ``duty``, ``UA`` or ``Tc``."""
        if "inlet" in inputs:
            return inputs["inlet"]
        signals = self._signal_ports()
        return next(v for k, v in inputs.items() if k not in signals)

    def _build_state_spec(self) -> StateSpec:
        """Build state spec for moles, optionally volume and temperature."""
        from difflow.dynamic.state import molar_states, thermal_state, volume_state

        spec_parts = []

        if self.params.get("variable_volume", False):
            spec_parts.append(volume_state())

        species = self.params["species_order"]
        spec_parts.append(molar_states(species))

        if self.params["mode"] != "isothermal":
            spec_parts.append(thermal_state())

        spec = spec_parts[0]
        for part in spec_parts[1:]:
            spec = spec + part

        return spec

    def _derivatives(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """Compute dn/dt and optionally dV/dt and dT/dt."""
        p = self.params
        species = p["species_order"]
        variable_volume = p.get("variable_volume", False)

        # Get volume — from state if variable, from params if fixed
        if variable_volume:
            V = state["V"]
        else:
            V = p["V"]

        # Get moles from state
        n = jnp.array([state[f"n_{s}"] for s in species])
        n_total = jnp.sum(n) + 1e-10

        # Compute concentrations
        C = {s: n[i] / V for i, s in enumerate(species)}

        # Get inlet stream (assume single inlet named "inlet")
        inlet = self._process_inlet(inputs)
        inlet_flows = get_flows(inlet)
        F_in = jnp.array([inlet_flows.get(s, 0.0) for s in species])
        F_in_total = jnp.sum(F_in)

        # Outlet flow. By default constant density gives F_out = F_in
        # (constant volume). With an outlet_flow_fn the draw is set
        # independently of the inlet so the level can change under imbalance.
        rho_mol = 55500.0  # mol/m³ (water-like liquid molar density)
        outlet_flow_fn = p.get("outlet_flow_fn")
        if outlet_flow_fn is not None:
            Q_out = outlet_flow_fn(V)               # m³/s
            F_out_total = Q_out * rho_mol           # mol/s
        else:
            F_out_total = F_in_total
        x_out = n / n_total
        F_out = F_out_total * x_out

        # Temperature
        if p["mode"] == "isothermal":
            T = inlet["T"]  # Use inlet T or could use T_spec
        else:
            T = state["T"]

        # Reaction rates
        r = p["rate_fn"](C, T, p["rate_params"])

        # Material balance: dn/dt = F_in - F_out + V * stoich @ r
        dn_dt = F_in - F_out + V * (p["stoich"] @ r)

        # Build derivative vector
        derivs_parts = []

        if variable_volume:
            # Volume balance: dV/dt = Q_in - Q_out. With an independent
            # outlet_flow_fn, Q_out != Q_in, so the level rises/falls.
            Q_in = F_in_total / rho_mol
            Q_out = F_out_total / rho_mol
            dV_dt = Q_in - Q_out
            derivs_parts.append(jnp.array([dV_dt]))

        derivs_parts.append(dn_dt)

        if p["mode"] == "isothermal":
            return jnp.concatenate(derivs_parts)

        # Energy balance for non-isothermal
        Cp = 75.0  # J/mol/K (typical liquid)

        # Heat of reaction
        if p["dH_rxn"] is not None:
            Q_rxn = -V * jnp.sum(r * p["dH_rxn"])  # Positive for exothermic
        else:
            Q_rxn = 0.0

        # Heat from flow
        T_in = inlet["T"]
        Q_flow = F_in_total * Cp * (T_in - T)

        # External heat: 0 (adiabatic), duty, or jacket UA*(Tc - T)
        Q_ext = self.heat_input(t, T, inputs)

        dn_total_dt = jnp.sum(dn_dt)
        dT_dt = safe_divide(Q_flow + Q_rxn + Q_ext, n_total * Cp) - T * safe_divide(dn_total_dt, n_total)

        derivs_parts.append(jnp.array([dT_dt]))
        return jnp.concatenate(derivs_parts)

    def _outputs(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> dict[str, Stream]:
        """Compute outlet stream from reactor state."""
        p = self.params
        species = p["species_order"]

        # Get moles and compute outlet composition
        n = jnp.array([state[f"n_{s}"] for s in species])
        n_total = jnp.sum(n) + 1e-10
        x_out = n / n_total

        # Get inlet for flow rate
        inlet = self._process_inlet(inputs)
        inlet_flows = get_flows(inlet)
        F_in_total = sum(inlet_flows.values())

        # Outlet total flow: independent draw law when given, else = inlet.
        outlet_flow_fn = p.get("outlet_flow_fn")
        if outlet_flow_fn is not None:
            V = state["V"] if p.get("variable_volume", False) else p["V"]
            F_out_total = outlet_flow_fn(V) * 55500.0  # Q_out * rho_mol
        else:
            F_out_total = F_in_total
        outlet_flows = {s: F_out_total * x_out[i] for i, s in enumerate(species)}

        # Temperature
        if p["mode"] == "isothermal":
            T_out = inlet["T"]
        else:
            T_out = state["T"]

        # Pressure (assume constant)
        P = inlet["P"]

        return {"outlet": make_stream(outlet_flows, T_out, P)}

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Initialize from inlet stream composition."""
        p = self.params
        species = p["species_order"]
        V = p["V"]
        variable_volume = p.get("variable_volume", False)

        # Get inlet
        inlet = self._process_inlet(inputs)
        inlet_flows = get_flows(inlet)

        # Initial moles = inlet composition * volume * concentration factor
        F_total = sum(inlet_flows.values())
        x_in = {s: inlet_flows.get(s, 0.0) / F_total for s in species}

        # Assume residence time of 1 minute for initial holdup
        tau = 60.0  # seconds
        n0 = jnp.array([F_total * x_in[s] * tau for s in species])

        parts = []
        if variable_volume:
            parts.append(jnp.array([V]))
        parts.append(n0)

        if p["mode"] != "isothermal":
            T0 = inlet["T"]
            parts.append(jnp.array([T0]))

        return jnp.concatenate(parts)


class DynamicTank(DynamicUnitBase):
    """Dynamic storage tank with level/volume tracking.

    State variables:
    - V: Liquid volume (m³)
    - n_i: Moles of each species (mol)
    - T: Temperature (K) [if non-isothermal]

    Material balance:
        dV/dt = F_in - F_out  (volumetric)
        dn_i/dt = F_in * C_in_i - F_out * C_out_i

    Assumes perfectly mixed (C_out = n / V).
    """

    symbol = "Dynamic Tank"
    equations = [
        r"\frac{dV}{dt} = \dot{V}_\mathrm{in} - \dot{V}_\mathrm{out}",
        r"\frac{dn_i}{dt} = \dot{V}_\mathrm{in}\,C_{i,\mathrm{in}} - \dot{V}_\mathrm{out}\,C_i",
        r"C_i = n_i / V",
    ]
    assumptions = [
        "Perfectly mixed liquid; outlet composition = bulk composition.",
        "Outlet volumetric flow from user-supplied ``outlet_flow_fn(V, t)`` (default constant).",
    ]
    references = ["Luyben, W.L. Process Modeling, Simulation, and Control for Chemical Engineers, 2e."]
    parameter_symbols = {}
    parameter_units = {}
    numerical_method = "ODE integration via diffrax with (V, n_i, T) state."

    def __init__(
        self,
        max_volume: float,
        species_order: list[str],
        outlet_flow_fn: Callable[[Array, Array], Array] | None = None,
        isothermal: bool = True,
        name: str | None = None,
    ):
        """Initialize dynamic tank.

        Args:
            max_volume: Maximum tank volume (m³)
            species_order: List of species names
            outlet_flow_fn: Function(V, t) -> F_out, defaults to constant
            isothermal: Whether to track temperature
            name: Unit name
        """
        params = {
            "V_max": jnp.asarray(max_volume),
            "species_order": species_order,
            "outlet_flow_fn": outlet_flow_fn,
            "isothermal": isothermal,
        }
        super().__init__(params, name)

    def _build_state_spec(self) -> StateSpec:
        """Build state spec for volume, moles, and optionally temperature."""
        from difflow.dynamic.state import (
            volume_state, molar_states, thermal_state
        )

        spec = volume_state()
        spec = spec + molar_states(self.params["species_order"])

        if not self.params["isothermal"]:
            spec = spec + thermal_state()

        return spec

    def _derivatives(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """Compute dV/dt and dn/dt."""
        p = self.params
        species = p["species_order"]

        V = state["V"]
        n = jnp.array([state[f"n_{s}"] for s in species])

        # Concentrations
        C_out = n / jnp.maximum(V, 1e-10)

        # Inlet
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_in_mol = jnp.array([inlet_flows.get(s, 0.0) for s in species])

        # Assume molar density for volumetric conversion
        rho_mol = 55500.0  # mol/m³ (water at ~25°C)
        Q_in = jnp.sum(F_in_mol) / rho_mol

        # Outlet flow
        if p["outlet_flow_fn"] is not None:
            Q_out = p["outlet_flow_fn"](V, t)
        else:
            Q_out = Q_in  # Default: maintain level

        F_out_mol = Q_out * C_out

        # Derivatives
        dV_dt = Q_in - Q_out
        dn_dt = F_in_mol - F_out_mol

        derivs = jnp.concatenate([jnp.array([dV_dt]), dn_dt])

        if not p["isothermal"]:
            # Simplified energy balance
            T = state["T"]
            T_in = inlet["T"]
            Cp = 75.0
            n_total = jnp.sum(n) + 1e-10
            dT_dt = safe_divide(jnp.sum(F_in_mol) * Cp * (T_in - T), n_total * Cp)
            derivs = jnp.concatenate([derivs, jnp.array([dT_dt])])

        return derivs

    def _outputs(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> dict[str, Stream]:
        """Compute outlet stream."""
        p = self.params
        species = p["species_order"]

        V = state["V"]
        n = jnp.array([state[f"n_{s}"] for s in species])
        C_out = n / jnp.maximum(V, 1e-10)

        # Get outlet flow rate
        if p["outlet_flow_fn"] is not None:
            Q_out = p["outlet_flow_fn"](V, t)
        else:
            inlet = inputs.get("inlet") or list(inputs.values())[0]
            inlet_flows = get_flows(inlet)
            Q_out = sum(inlet_flows.values()) / 55500.0  # Assume liquid molar density

        F_out = {s: Q_out * C_out[i] for i, s in enumerate(species)}

        # Temperature
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        if p["isothermal"]:
            T = inlet["T"]
        else:
            T = state["T"]

        return {"outlet": make_stream(F_out, T, inlet["P"])}

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Initialize at half capacity with inlet composition."""
        p = self.params
        species = p["species_order"]

        V0 = p["V_max"] / 2.0

        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_total = sum(inlet_flows.values())
        x_in = {s: inlet_flows.get(s, 0.0) / F_total for s in species}

        # Initial moles based on inlet composition
        rho_mol = 55500.0  # mol/m³ (water at ~25°C)
        n0 = jnp.array([V0 * rho_mol * x_in[s] for s in species])

        state0 = jnp.concatenate([jnp.array([V0]), n0])

        if not p["isothermal"]:
            T0 = inlet["T"]
            state0 = jnp.concatenate([state0, jnp.array([T0])])

        return state0
