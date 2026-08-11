"""Dynamic counter-current heat exchanger (wall-thermal-lag duty model).

difflow's steady-state :class:`~difflow.units.heat_exchanger.EnthalpyCounterCurrentHX`
solves ``Q = UA * LMTD`` with rigorous two-phase enthalpy balances on each side.
This module provides its *dynamic* companion for transient flowsheets.

A real feed-effluent exchanger's dominant time constant is the thermal mass of
the metal wall, not the (short) fluid residence time. :class:`DynamicCounterCurrentHX`
captures that as a reduced-order, first-order lag of the exchanger *duty* toward
its steady-state target::

    dQ/dt = (UA * LMTD(Q) - Q) / tau,     tau = C_wall / UA

At every instant the outlet temperatures are recovered from the per-side
enthalpy balances at the current duty (``H_hot_out = H_hot_in - Q``,
``H_cold_out = H_cold_in + Q``) by inverting the *same* two-phase-aware
``stream_enthalpy_flash`` the steady-state unit uses. At steady state
``dQ/dt = 0`` gives ``Q = UA * LMTD`` with those enthalpy balances -- exactly the
:class:`EnthalpyCounterCurrentHX` fixed point -- so the transient model relaxes
to the validated steady state to solver tolerance. ``tau`` sets only the
transient speed; crucially, giving the exchanger this state also turns a
flowsheet's feed-effluent energy-integration loop from a steady-state tear into
an ODE state, so the dynamic flowsheet needs no algebraic tear solver.

This is a lumped (single-duty) reduced-order model. A spatially distributed
N-node wall+fluid discretization would additionally resolve the internal
temperature profile transient and transport delay; it converges to the same
LMTD steady state as N grows and is the natural refinement if that detail
matters.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
import optimistix as optx

from difflow.streams import Stream, get_flows
from difflow.dynamic.state import StateSpec, StateVar
from difflow.units.heat_exchanger import log_mean_temperature_difference


# ---------------------------------------------------------------------------
# JIT-compiled numeric cores.
#
# Un-jitted, the per-side Newton inversion over ``stream_enthalpy_flash`` is
# re-traced, re-lowered and re-compiled on *every* call -- and an ODE solve
# calls ``derivatives`` repeatedly, so the cost multiplies. Wrapping the cores
# here compiles once per (thermo, shapes) and reuses the executable, matching
# what the steady-state units do (see units/eos_units.py, units/heat_exchanger.py).
# ``thermo`` is static (identity-hashed), so a shared thermo object -- as a
# flowsheet or a session-scoped test fixture provides -- hits the cache.
# ---------------------------------------------------------------------------
def _invert_T(thermo, flows, H_target, P, T_guess):
    """Find T with stream_enthalpy_flash(flows, T, P) = H_target (monotone)."""

    def resid(T, _):
        return thermo.stream_enthalpy_flash(flows, T, P) - H_target

    solver = optx.Newton(rtol=1e-9, atol=1e-4)
    sol = optx.root_find(resid, solver, T_guess, args=None, max_steps=50, throw=False)
    return sol.value


@partial(jax.jit, static_argnames=("thermo",))
def _outlet_temps_core(thermo, hot_flows, cold_flows, T_hot_in, T_cold_in,
                       P_hot, P_cold, Q):
    """Both outlet temperatures at duty ``Q`` from the per-side enthalpy balances."""
    H_hot_in = thermo.stream_enthalpy_flash(hot_flows, T_hot_in, P_hot)
    H_cold_in = thermo.stream_enthalpy_flash(cold_flows, T_cold_in, P_cold)
    T_hot_out = _invert_T(thermo, hot_flows, H_hot_in - Q, P_hot, T_hot_in)
    T_cold_out = _invert_T(thermo, cold_flows, H_cold_in + Q, P_cold, T_cold_in)
    return T_hot_out, T_cold_out


@partial(jax.jit, static_argnames=("thermo",))
def _side_outlet_T_core(thermo, flows, T_in, P, dH):
    """One side's outlet temperature from ``H_out = H_in + dH``.

    ``dH`` is ``+Q`` for the cold side and ``-Q`` for the hot side.
    """
    H_in = thermo.stream_enthalpy_flash(flows, T_in, P)
    return _invert_T(thermo, flows, H_in + dH, P, T_in)


class DynamicCounterCurrentHX:
    """Counter-current HX with a first-order wall-thermal lag on the duty.

    State:
        Q -- exchanger duty (W), the single dynamic state.

    Inputs (``inputs`` dict): ``"hot"`` and ``"cold"`` inlet streams.
    Outputs: ``{"hot_out": Stream, "cold_out": Stream}``.

    The thermo object must provide ``stream_enthalpy_flash(flows, T, P)`` (e.g.
    :class:`difflow.thermo.CubicThermo`), so the energy balance sees the real,
    temperature-dependent heat capacity including any latent heat as a side
    partially vaporizes or condenses -- matching the steady-state unit.
    """

    symbol = "Dynamic counter-current HX"
    equations = [
        r"\frac{dQ}{dt} = \frac{UA\,\mathrm{LMTD}(Q) - Q}{\tau}",
        r"H_{h,\mathrm{out}} = H_{h,\mathrm{in}} - Q,\qquad H_{c,\mathrm{out}} = H_{c,\mathrm{in}} + Q",
    ]
    assumptions = [
        "Lumped wall thermal lag: duty relaxes first-order toward UA*LMTD.",
        "Per-side two-phase enthalpy balances from the supplied thermo/EOS.",
        "Constant UA; tau = C_wall/UA sets the transient speed only.",
    ]
    references = [
        "Incropera, DeWitt, Bergman. Fundamentals of Heat and Mass Transfer, 7e, Ch. 11.",
        "Luyben, W.L. Process Modeling, Simulation, and Control for Chemical Engineers, 2e.",
    ]
    parameter_symbols = {"UA": "UA", "tau": r"\tau"}
    parameter_units = {"UA": "W/K", "tau": "s"}
    numerical_method = "First-order ODE in duty Q; per-side 1-D enthalpy inversion (optimistix Newton) each RHS eval."

    def __init__(self, UA, thermo, tau: float | Array = 30.0, name: str = "hx"):
        """Initialize the dynamic exchanger.

        Args:
            UA: Overall heat-transfer coefficient x area (W/K).
            thermo: Thermo providing ``stream_enthalpy_flash(flows, T, P)``.
            tau: Lumped wall thermal time constant (s), ``C_wall / UA``. Sets the
                transient speed; does not affect the steady state.
            name: Unit name.
        """
        self.UA = jnp.asarray(UA)
        self.thermo = thermo
        self.tau = jnp.asarray(tau)
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([
            StateVar("Q", "generic", "W", "Exchanger duty", bounds=(0.0, None), scale=1e5),
        ])

    @staticmethod
    def _inlets(inputs: dict[str, Stream]) -> tuple[Stream, Stream]:
        return inputs["hot"], inputs["cold"]

    def _outlet_temps(self, Q, hot: Stream, cold: Stream):
        return _outlet_temps_core(
            self.thermo, get_flows(hot), get_flows(cold),
            jnp.asarray(hot["T"]), jnp.asarray(cold["T"]),
            hot["P"], cold["P"], Q,
        )

    def cold_outlet(self, state: Array, cold: Stream) -> Stream:
        """Cold-side outlet from the current duty and the cold inlet alone.

        The cold outlet enthalpy is ``H_cold_in + Q`` -- it does not depend on the
        hot inlet -- so this can be evaluated before the hot stream is known. That
        is what lets a flowsheet feed the preheated cold stream forward (e.g. into
        a downstream heater/reactor) without a within-step algebraic loop.
        """
        Q = state[0]
        T_cold_out = _side_outlet_T_core(
            self.thermo, get_flows(cold), jnp.asarray(cold["T"]), cold["P"], Q
        )
        out = dict(cold)
        out["T"] = T_cold_out
        return out

    def hot_outlet(self, state: Array, hot: Stream) -> Stream:
        """Hot-side outlet from the current duty and the hot inlet alone
        (``H_hot_out = H_hot_in - Q``)."""
        Q = state[0]
        T_hot_out = _side_outlet_T_core(
            self.thermo, get_flows(hot), jnp.asarray(hot["T"]), hot["P"], -Q
        )
        out = dict(hot)
        out["T"] = T_hot_out
        return out

    def derivatives(self, t: Array, state: Array, inputs: dict[str, Stream], params=None) -> Array:
        Q = state[0]
        hot, cold = self._inlets(inputs)
        T_hot_in = jnp.asarray(hot["T"])
        T_cold_in = jnp.asarray(cold["T"])
        T_hot_out, T_cold_out = self._outlet_temps(Q, hot, cold)
        dT1 = T_hot_in - T_cold_out   # hot inlet vs cold outlet
        dT2 = T_hot_out - T_cold_in   # hot outlet vs cold inlet
        LMTD = log_mean_temperature_difference(dT1, dT2)
        dQ_dt = (self.UA * LMTD - Q) / self.tau
        return jnp.reshape(dQ_dt, (1,))

    def outputs(self, t: Array, state: Array, inputs: dict[str, Stream], params=None) -> dict[str, Stream]:
        Q = state[0]
        hot, cold = self._inlets(inputs)
        T_hot_out, T_cold_out = self._outlet_temps(Q, hot, cold)
        hot_out = dict(hot)
        hot_out["T"] = T_hot_out
        cold_out = dict(cold)
        cold_out["T"] = T_cold_out
        return {"hot_out": hot_out, "cold_out": cold_out}

    def initial_state(self, inputs: dict[str, Stream], params=None) -> Array:
        """Cold start: zero duty (no heat transferred yet)."""
        return jnp.zeros(1)

    def __repr__(self) -> str:
        return f"DynamicCounterCurrentHX(name='{self.name}', UA={float(self.UA):.1f} W/K, tau={float(self.tau):.1f} s)"
