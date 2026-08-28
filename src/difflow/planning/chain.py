"""A two-plant reference chain: NGL recovery feeding a gas-turbine plant.

This is the model behind the scaling measurement in the module documentation
and the worked example, kept in one place so the benchmark, the tests and the
notebook all describe the same problem.

The chain is:

* an **NGL recovery** block — a real cryogenic flash from
  :class:`difflow.units.flash.Flash`, with a Rachford-Rice root find inside it,
  plus recovery and reflux levers.  Because the flash solve is implicit,
  ``jax.jacobian`` of this block returns the *reduced* sensitivity through the
  converged root: exactly the delta vector a planner needs, and exactly what
  one-at-a-time perturbation of a rigorous simulator is used to approximate.
* a **gas-turbine power** block, fed by the residue gas, with a part-load
  efficiency curve and an allocation lever choosing between burning the
  residue for power and selling it as gas.  That lever is bang-bang: it
  switches corner at a finite power price rather than sliding.

The two are coupled through the residue stream, so the downstream block's
inlet is a decision the upstream block makes.

Note:
    The thermodynamics are ideal (Raoult's law with Antoine correlations from
    :mod:`difflow.database`) and the flowsheet is schematic.  It is a
    faithful *shape* of the planning problem — nonlinear, implicit, coupled,
    with a bang-bang lever and a phase boundary in range — not a validated
    plant design.  Do not read design numbers off it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.database import get_species_data
from difflow.planning.block import Block
from difflow.planning.lp import Spec
from difflow.planning.network import Network
from difflow.streams import make_stream
from difflow.thermo import IdealThermo
from difflow.units.flash import Flash, FlashParams

#: Species in the inlet gas, in array order.
SPECIES = ("methane", "ethane", "propane", "n_butane")

#: Nominal inlet gas, mol/s.
FEED = {"methane": 75.0, "ethane": 12.0, "propane": 8.0, "n_butane": 5.0}

#: Reference temperature for the refrigeration duty (K).
T_REFRIG_REF = 300.0

_thermo = IdealThermo({s: get_species_data(s) for s in SPECIES})
_flash = Flash(FlashParams(species_order=list(SPECIES)), _thermo)


def _cold_box(u: Array, feed: Mapping[str, float]):
    """Flash the inlet gas at the cold-box conditions.

    Returns ``(liquid, vapour, info)`` from :class:`difflow.units.flash.Flash`.
    """
    T, P = u[1], u[3]
    inlet = make_stream({s: jnp.asarray(feed[s], dtype=float) for s in SPECIES},
                        T, P)
    return _flash(inlet, T=T, P=P)


def ngl_outputs(u: Array, theta: Mapping[str, Any]) -> Array:
    """NGL recovery block: ``u -> y``.

    Args:
        u: ``[ethane_recovery, T_coldbox, split, P_expander]``.
        theta: ``feed_scale``, ``reflux_gain``, ``refrig_cop``,
            ``recompress_kJ``, ``colfeed_rise``.

    Returns:
        ``[NGL_C2, NGL_C3plus, residue_F, E_refrig, T_colfeed]`` with flows in
        mol/s, duty in MW and temperature in K.
    """
    scale = theta["feed_scale"]
    feed = {s: scale * FEED[s] for s in SPECIES}
    liquid, vapour, info = _cold_box(u, feed)

    recovery, T_cold, P_exp, split = u[0], u[1], u[3], u[2]

    # A cold-reflux lever: `split` sends part of the flash vapour back over
    # the liquid, so heavy components migrate out of the residue.
    gain = theta["reflux_gain"]
    refluxed = {s: split * gain * vapour[f"F_{s}"] for s in SPECIES}
    # Methane is not condensed by the reflux; the C3+ recovery is.
    refluxed["methane"] = 0.0 * refluxed["methane"]
    refluxed["ethane"] = 0.35 * refluxed["ethane"]

    liq = {s: liquid[f"F_{s}"] + refluxed[s] for s in SPECIES}
    vap = {s: vapour[f"F_{s}"] - refluxed[s] for s in SPECIES}

    # Ethane recovery versus rejection: the rejected ethane leaves with the
    # residue gas, where it is sold as fuel instead of as product.
    ngl_c2 = recovery * liq["ethane"]
    rejected_c2 = liq["ethane"] - ngl_c2
    ngl_c3plus = liq["propane"] + liq["n_butane"]
    residue_F = (vap["methane"] + vap["ethane"] + rejected_c2
                 + vap["propane"] + vap["n_butane"] + liq["methane"])

    feed_total = sum(feed.values())
    duty = (theta["refrig_cop"] * feed_total * (T_REFRIG_REF - T_cold)
            + theta["recompress_kJ"] * feed_total
            * jnp.log(4.5e6 / jnp.clip(P_exp, 1.0e5, None)))
    E_refrig = duty / 1.0e6

    T_colfeed = (T_cold + theta["colfeed_rise"] * (1.0 - split)
                 + 6.0 * (recovery - 0.5))
    return jnp.stack([ngl_c2, ngl_c3plus, residue_F, E_refrig, T_colfeed])


def ngl_phases(u: Array, theta: Mapping[str, Any]) -> Array:
    """Vapour fraction of the cold box: the block's phase indicator.

    A delta vector computed across the bubble point describes a different
    function on either side, so the planner needs to know when a proposal
    crosses it.
    """
    scale = theta["feed_scale"]
    feed = {s: scale * FEED[s] for s in SPECIES}
    _, _, info = _cold_box(u, feed)
    return jnp.atleast_1d(info["V_frac"])


def power_outputs(u: Array, theta: Mapping[str, Any]) -> Array:
    """Gas-turbine block: ``u -> y``.

    Args:
        u: ``[fuel_F, alloc]`` — inlet residue (mol/s, linked from the NGL
            block) and the fraction of it burned for power.
        theta: ``w_net`` (J/mol fuel, electrical), ``F_design`` (mol/s),
            ``part_load_k``, ``co2_per_mol`` (kg/mol).

    Returns:
        ``[Power, CO2, gas_sold]`` in MW, kg/s and mol/s.
    """
    fuel_F, alloc = u[0], u[1]
    burned = alloc * fuel_F
    load = burned / theta["F_design"]
    # Part-load penalty, smooth and capped so the curve stays physical when
    # the LP proposes a load outside the design envelope.
    shortfall = jnp.clip(1.0 - load, 0.0, 1.0)
    eta = 1.0 - theta["part_load_k"] * shortfall ** 2
    power = burned * theta["w_net"] * eta / 1.0e6
    co2 = burned * theta["co2_per_mol"]
    return jnp.stack([power, co2, fuel_F - burned])


#: Default NGL block parameters.
NGL_THETA: dict[str, float] = {
    "feed_scale": 1.0,
    "reflux_gain": 0.45,
    "refrig_cop": 260.0,
    "recompress_kJ": 900.0,
    "colfeed_rise": 14.0,
}

#: Default gas-turbine parameters (an F-class combined cycle on residue gas).
POWER_THETA: dict[str, float] = {
    "w_net": 4.6e5,
    "F_design": 60.0,
    "part_load_k": 0.22,
    "co2_per_mol": 0.048,
}

#: Default prices, per period.
PRICES: dict[str, float] = {
    "NGL_C2": 9.0,        # $/kmol-ish, per mol/s of product
    "NGL_C3plus": 26.0,
    "Power": 55.0,        # $/MW
    "gas_sold": 5.5,
    "E_refrig": -18.0,    # refrigeration is a cost
    "CO2": -1.2,
}


def ngl_block(name: str = "ngl", theta: Mapping[str, float] | None = None
              ) -> Block:
    """Build the NGL recovery block."""
    return Block(
        name=name, fn=ngl_outputs,
        u_names=["ethane_recovery", "T_coldbox", "split", "P_expander"],
        y_names=["NGL_C2", "NGL_C3plus", "residue_F", "E_refrig", "T_colfeed"],
        lb=[0.30, 218.0, 0.0, 2.5e6],
        ub=[0.98, 244.0, 1.0, 4.0e6],
        u0=[0.70, 232.0, 0.4, 3.2e6],
        theta=dict(NGL_THETA if theta is None else theta),
        phase_fn=ngl_phases, phase_names=("V_frac",), phase_bounds=(0.0, 1.0),
        jit=True)


def power_block(name: str = "power", theta: Mapping[str, float] | None = None
                ) -> Block:
    """Build the gas-turbine power block."""
    return Block(
        name=name, fn=power_outputs,
        u_names=["fuel_F", "alloc"],
        y_names=["Power", "CO2", "gas_sold"],
        lb=[0.0, 0.0], ub=[200.0, 1.0], u0=[70.0, 0.5],
        theta=dict(POWER_THETA if theta is None else theta), jit=True)


@dataclass
class ChainProblem:
    """A ready-to-plan two-plant chain.

    Attributes:
        network: The block network.
        prices: Prices by qualified variable name.
        specs: Constraints, as :class:`~difflow.planning.lp.Spec` objects.
        horizon: Number of periods.
        n_decisions: ``5 * horizon``.
    """

    network: Network
    prices: dict[str, float]
    specs: list[Spec]
    horizon: int

    @property
    def n_decisions(self) -> int:
        return self.network.n_decisions

    def planner(self, **kwargs):
        """Build a :class:`~difflow.planning.planner.DeltaBasePlanner`."""
        from difflow.planning.planner import DeltaBasePlanner
        opts = dict(prices=self.prices, specs=self.specs)
        opts.update(kwargs)
        return DeltaBasePlanner(self.network, **opts)

    def __repr__(self) -> str:
        return (f"ChainProblem(horizon={self.horizon}, "
                f"n_decisions={self.n_decisions})")


def two_plant_chain(horizon: int = 1,
                    prices: Mapping[str, float] | None = None,
                    power_price: float | Sequence[float] | None = None,
                    colfeed_max: float | None = 236.0,
                    co2_cap: float | None = None) -> ChainProblem:
    """Build the reference chain, optionally replicated over a horizon.

    Each period contributes five decisions — four on the NGL block and the
    power block's allocation lever — so ``n = 5 * horizon``.  That is how a
    planning problem gets large, and it is the sweep used in the scaling
    study.

    Args:
        horizon: Number of periods.  Blocks are named ``ngl@t0``, ``power@t0``,
            ``ngl@t1`` and so on when ``horizon > 1``.
        prices: Override the per-period price map, keyed by bare output name.
        power_price: Shorthand for overriding just the power price; a scalar
            applies to every period, a sequence gives one per period.
        colfeed_max: Upper spec on the deethanizer feed temperature (K), or
            ``None`` for no spec.
        co2_cap: Optional cap on total CO2 across the horizon (kg/s).

    Returns:
        A :class:`ChainProblem`.

    Example:
        >>> problem = two_plant_chain(horizon=4)
        >>> problem.n_decisions
        20
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    base = dict(PRICES if prices is None else prices)

    if power_price is None:
        power_prices = [base["Power"]] * horizon
    elif np.isscalar(power_price):
        power_prices = [float(power_price)] * horizon
    else:
        power_prices = [float(p) for p in power_price]
        if len(power_prices) != horizon:
            raise ValueError(
                f"power_price has {len(power_prices)} entries, expected "
                f"{horizon} to match the horizon")

    blocks, links = [], []
    price_map: dict[str, float] = {}
    specs: list[Spec] = []

    for t in range(horizon):
        suffix = "" if horizon == 1 else f"@t{t}"
        ngl_name, pwr_name = f"ngl{suffix}", f"power{suffix}"
        blocks.append(ngl_block(ngl_name))
        blocks.append(power_block(pwr_name))
        links.append((f"{ngl_name}.residue_F", f"{pwr_name}.fuel_F"))

        for key in ("NGL_C2", "NGL_C3plus", "E_refrig"):
            price_map[f"{ngl_name}.{key}"] = base[key]
        price_map[f"{pwr_name}.Power"] = power_prices[t]
        price_map[f"{pwr_name}.gas_sold"] = base["gas_sold"]
        price_map[f"{pwr_name}.CO2"] = base["CO2"]

        if colfeed_max is not None:
            specs.append(Spec(f"{ngl_name}.T_colfeed", "<=", colfeed_max,
                              name=f"T_colfeed{suffix}"))

    if co2_cap is not None:
        coeffs = {f"power{'' if horizon == 1 else f'@t{t}'}.CO2": 1.0
                  for t in range(horizon)}
        specs.append(Spec(coeffs, "<=", co2_cap, name="CO2_cap"))

    return ChainProblem(network=Network(blocks, links), prices=price_map,
                        specs=specs, horizon=horizon)
