"""Equilibrium-limited adiabatic catalyst beds with catalyst heat capacity: shift, Claus, hydrogenation.

A water-gas-shift bed, a Claus converter and a tail-gas hydrogenation
reactor are fixed beds where the gas reaches (or approaches) its
equilibrium within the first part of the bed, and the steady units model
them as adiabatic equilibrium reactors (:class:`difflow.GibbsReactor`). What
they do in time is set by the catalyst's heat capacity: a change in the
inlet temperature or composition moves through the bed as a thermal wave,
taking ``W c_p / (F c_p,gas)`` -- many minutes -- to reach the outlet.

:class:`DynamicEquilibriumBed` keeps that and nothing else. The bed is
``N`` cells; a cell has one state, its temperature ``T_k`` (catalyst and the
gas leaving it). The gas is quasi-steady (it passes the bed in seconds):
the gas entering cell ``k`` -- the previous cell's outflow -- leaves it at
the cell's equilibrium, ``n_k = eq(n_{k-1}, T_k, P)``, the steady unit's
own restricted equilibrium taken ISOTHERMALLY at ``T_k``. The cell's energy
balance::

    C_k dT_k/dt = H(n_{k-1}, T_{k-1}) - H(n_k, T_k)

``H`` is the steady unit's enthalpy (formation basis, so the heat of
reaction is in it). At rest the first cell is at the gas's ADIABATIC
equilibrium -- the steady reactor's answer, by the same equations -- and the
others hold it: the rest state is the steady unit's to the equilibrium
solver's tolerance. The equilibrium reactions conserve elements, and the
cells hold no gas, so element balances close exactly at every instant.

What this leaves out: the approach to equilibrium within a cell is the
steady unit's ``approach_dT`` (a temperature offset), not a rate; gas-solid
heat-transfer resistance (the gas leaves at the catalyst temperature); and
sulfur condensing in a Claus bed (the dew-point margin is reported, as the
steady converter does, not acted on).
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.dynamic.state import StateSpec, StateVar


def catalyst_heat_capacity(volumetric_flow_Nm3_h: float, ghsv_per_h: float, bulk_density: float,
                           cp: float) -> float:
    """``W c_p`` (J/K) of a bed sized by its gas hourly space velocity.

    ``volumetric_flow_Nm3_h / ghsv_per_h`` is the bed volume (m3); times the
    bulk density (kg/m3) the catalyst mass; times ``cp`` (J/(kg K)) its heat
    capacity. Every argument is a design value the caller states.
    """
    return float(volumetric_flow_Nm3_h / ghsv_per_h * bulk_density * cp)


#: Normal molar volume (m3/mol at 273.15 K, 101.325 kPa), ideal gas.
NM3_PER_MOL = 0.0224140


class DynamicEquilibriumBed:
    """An adiabatic equilibrium bed as ``n_cells`` thermal cells (see the module docstring).

    Port ``"inlet"`` (a stream in the steady unit's format), output
    ``"outlet"``. State: the cells' temperatures (scaled by 100 K).

    Args:
        equilibrate: ``eq(n, T, P) -> n_out``, the steady unit's restricted
            equilibrium taken isothermally at ``T`` (flows in the unit's
            species order).
        enthalpy: ``H(n, T) -> W``, the steady unit's enthalpy flow.
        to_vec: Stream -> flow vector.
        to_stream: ``(n, T, P) -> stream``.
        C_bed: Heat capacity of the whole bed's catalyst (J/K), split
            evenly over the cells.
        n_cells: Cells.
        P: Bed inlet pressure (Pa). The equilibrium is taken at the outlet
            pressure ``P - dP``, as :class:`difflow.GibbsReactor` takes it.
        T0: Initial cell temperatures (K), ``(n_cells,)`` or a scalar.
        pre: Optional ``pre(n, T_in) -> (n_reacted, H_in)``: what happens to
            the gas as it enters the bed, before the first cell's equilibrium
            (a Claus bed's COS/CS2 hydrolysis at fixed conversions). ``H_in``
            is the enthalpy entering cell 1; default ``H(n, T_in)``.
        dP: Bed pressure drop (Pa).
        name: Unit name.
    """

    output_ports = ("outlet",)
    output_dependencies = {"outlet": ("inlet",)}

    def __init__(self, equilibrate: Callable, enthalpy: Callable, to_vec: Callable, to_stream: Callable,
                 C_bed: float, n_cells: int = 5, P: float = 1e5, T0: float | Sequence[float] = 600.0,
                 pre: Callable | None = None, dP: float = 0.0, name: str = "bed"):
        self.equilibrate = equilibrate
        self.enthalpy = enthalpy
        self.to_vec = to_vec
        self.to_stream = to_stream
        self.n_cells = int(n_cells)
        self.C_cell = jnp.asarray(C_bed / self.n_cells, dtype=float)
        self.P = jnp.asarray(P, dtype=float)
        self.dP = jnp.asarray(dP, dtype=float)
        self.T0 = np.broadcast_to(np.asarray(T0, dtype=float), (self.n_cells,)).copy()
        self.pre = pre
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar(f"cell{k + 1}.T", "temperature", "K", "", scale=100.0)
                          for k in range(self.n_cells)])

    def initial_state(self, inputs=None, params=None) -> Array:
        return jnp.asarray(self.T0) / 100.0

    def _run(self, inlet, T):
        """``(n_k, H_in_k)`` of every cell: its equilibrium outflow and the enthalpy entering it."""
        n = self.to_vec(inlet)
        T_in = jnp.asarray(inlet["T"], dtype=float)
        if self.pre is None:
            H_in = self.enthalpy(n, T_in)
        else:
            n, H_in = self.pre(n, T_in)

        def body(carry, T_k):
            n_prev, H_prev = carry
            n_k = self.equilibrate(n_prev, T_k, self.P - self.dP)
            return (n_k, self.enthalpy(n_k, T_k)), (n_k, H_prev)

        _, (ns, Hs) = jax.lax.scan(body, (n, H_in), T)
        return ns, Hs

    def derivatives(self, t, state, inputs, params=None) -> Array:
        T = 100.0 * state
        ns, H_in = self._run(inputs["inlet"], T)
        H_out = jax.vmap(self.enthalpy)(ns, T)
        return (H_in - H_out) / self.C_cell / 100.0

    def outputs(self, t, state, inputs, params=None) -> dict:
        T = 100.0 * state
        ns, _ = self._run(inputs["inlet"], T)
        return {"outlet": self.to_stream(ns[-1], T[-1], self.P - self.dP)}

    def thermal_time(self, n, T, equilibrium: bool = True) -> Array:
        """``sum_k C_k / C_flow``: the mean time a thermal step takes to cross the bed (s), at flows ``n``.

        With ``equilibrium`` the gas's heat flow is taken along its
        equilibrium, ``C_flow = d/dT H(eq(n, T), T)``: a gas held at
        equilibrium shifts its composition as it heats, and the heat of
        reaction of that shift is part of the heat the gas carries per
        kelvin (for an exothermic equilibrium such as the water-gas shift
        it adds to it, so the wave moves faster than at frozen
        composition). Without it, at frozen composition.
        """
        T = jnp.asarray(T, dtype=float)
        P = self.P - self.dP
        if equilibrium:
            f = lambda t: self.enthalpy(self.equilibrate(n, t, P), t)  # noqa: E731
        else:
            f = lambda t: self.enthalpy(n, t)  # noqa: E731
        C_flow = jax.jvp(f, (T,), (jnp.asarray(1.0),))[1]
        return self.n_cells * self.C_cell / C_flow


__all__ = ["DynamicEquilibriumBed", "catalyst_heat_capacity", "NM3_PER_MOL"]
