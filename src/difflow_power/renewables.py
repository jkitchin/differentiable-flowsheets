"""Renewable generators: resource models, inverter capability, curtailment.

A wind farm, a solar plant or a run-of-river hydro station enters an
OPF as an ordinary :class:`~difflow_power.network.Generator` with two
things set differently:

- its ceiling ``p_max_mw`` is the power the RESOURCE makes available
  this hour (wind speed, irradiance, river flow), not a nameplate, and
  its floor is zero, so the OPF is free to curtail it;
- its reactive output is bounded jointly with its real output by an
  apparent-power rating, ``P^2 + Q^2 <= S_max^2``
  (:attr:`~difflow_power.network.Generator.s_max_mva`). An inverter is
  rated in current, so at full real output it has little var capability
  left, and a curtailed one has its whole rating available as vars.

So this module adds no physics to the network equations. It provides
the resource models that turn weather into an available MW, and
factories that build the generator from them.

Resource models
---------------

Each is a plain JAX function of the weather, so ``jax.grad`` gives the
sensitivity of available power to wind speed, irradiance or flow.

=========================  =============================================
:func:`wind_turbine_power`  idealised power curve: zero below cut-in,
                            cubic to rated speed, flat to cut-out, zero
                            beyond (the turbine feathers in a storm)
:func:`pv_power`            irradiance and cell temperature to AC MW,
                            PVWatts-style derates and inverter clipping
:func:`hydro_power`         ``rho g Q H eta``
=========================  =============================================

The network data model holds floats, so a factory evaluates the
resource model to a number. To differentiate the optimal cost with
respect to the weather, chain the two halves by hand:
:func:`curtailment` reports what one more MW of available resource is
worth at the optimum (the multiplier on the unit's ceiling, by the
envelope theorem), and ``jax.grad`` of the resource model gives the MW
per unit of weather.

Example:
    >>> import difflow_power as dp
    >>> net = dp.cases.case9()
    >>> net.generators["wind"] = dp.wind_generator(
    ...     "5", wind_speed=10.0, n_turbines=20, turbine_mw=3.0)
    >>> res = dp.solve_acopf(net)
    >>> dp.curtailment(res, ["wind"])["wind"].curtailed_mw  # doctest: +SKIP
    0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow_power.network import Generator

#: air density at which turbine power curves are quoted (kg/m^3; sea
#: level, 15 C, the IEC 61400-12-1 reference)
REFERENCE_AIR_DENSITY = 1.225

#: irradiance at standard test conditions (W/m^2)
STC_IRRADIANCE = 1000.0

#: cell temperature at standard test conditions (C)
STC_TEMPERATURE = 25.0

#: gravitational acceleration (m/s^2)
GRAVITY = 9.80665


# =============================================================================
# Resource models
# =============================================================================


def wind_turbine_power(
    wind_speed: Array | float,
    rated_mw: float,
    *,
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0,
    air_density: Array | float = REFERENCE_AIR_DENSITY,
) -> Array:
    """Electrical output of one turbine (MW) at a hub-height wind speed.

    The idealised power curve: nothing below ``cut_in``; between cut-in
    and ``rated_speed`` the output follows the cubic of the wind speed
    (the kinetic energy flux through the rotor goes as ``rho v^3``);
    held at ``rated_mw`` by pitch control up to ``cut_out``; zero beyond
    it, where the turbine shuts down to protect itself.

    Air density scales the cubic region linearly, so a turbine on a hot
    day or at altitude reaches rated power at a higher wind speed.

    Args:
        wind_speed: hub-height wind speed (m/s). Any shape.
        rated_mw: rated output of the turbine (MW).
        cut_in: speed at which it starts producing (m/s).
        rated_speed: speed at which it reaches rated output (m/s).
        cut_out: speed at which it shuts down (m/s).
        air_density: air density (kg/m^3).

    Returns:
        Output in MW, the shape of ``wind_speed``. Differentiable in the
        wind speed and the density away from the three corners of the
        curve; the derivative is zero in the flat and the off regions.

    Example:
        >>> float(wind_turbine_power(12.0, 3.0))
        3.0
        >>> float(wind_turbine_power(2.0, 3.0))
        0.0
    """
    if not 0.0 <= cut_in < rated_speed < cut_out:
        raise ValueError(
            "need 0 <= cut_in < rated_speed < cut_out, got "
            f"{cut_in}, {rated_speed}, {cut_out}"
        )
    v = jnp.asarray(wind_speed, dtype=jnp.float64)
    fraction = (
        (air_density / REFERENCE_AIR_DENSITY)
        * (v**3 - cut_in**3)
        / (rated_speed**3 - cut_in**3)
    )
    power = rated_mw * jnp.clip(fraction, 0.0, 1.0)
    return jnp.where(v >= cut_out, 0.0, power)


def pv_cell_temperature(
    irradiance: Array | float,
    t_ambient: Array | float,
    *,
    noct: float = 45.0,
) -> Array:
    """Cell temperature (C) from the NOCT model.

    ``T_cell = T_ambient + (NOCT - 20) G / 800``: the nominal operating
    cell temperature is quoted at 800 W/m^2 and 20 C ambient, and the
    rise above ambient is taken as proportional to irradiance.

    Args:
        irradiance: plane-of-array irradiance (W/m^2).
        t_ambient: ambient air temperature (C).
        noct: the module's nominal operating cell temperature (C).

    Returns:
        Cell temperature in C.
    """
    g = jnp.asarray(irradiance, dtype=jnp.float64)
    return t_ambient + (noct - 20.0) * g / 800.0


def pv_power(
    irradiance: Array | float,
    t_ambient: Array | float = 25.0,
    *,
    capacity_mw_dc: float,
    inverter_mw: float | None = None,
    temp_coefficient: float = -0.0047,
    noct: float = 45.0,
    losses: float = 0.14,
    inverter_efficiency: float = 0.96,
) -> Array:
    """AC output of a solar plant (MW) at a given irradiance and temperature.

    The DC output scales with irradiance relative to standard test
    conditions and falls linearly with cell temperature above 25 C; a
    lumped loss fraction covers soiling, wiring, mismatch and
    availability; the inverter converts at a fixed efficiency and CLIPS
    at its rating. The defaults are PVWatts v5's (a standard module at
    -0.47 %/C, 14 % system losses, 96 % inverter efficiency).

    Clipping is real and common: plants are routinely built with more
    DC capacity than inverter (a DC/AC ratio of 1.1 to 1.3), trading
    lost output at noon for more output the rest of the day.

    Args:
        irradiance: plane-of-array irradiance (W/m^2). Any shape.
        t_ambient: ambient temperature (C).
        capacity_mw_dc: module nameplate at standard test conditions
            (MW DC).
        inverter_mw: inverter AC rating (MW), or ``None`` for no
            clipping.
        temp_coefficient: fractional power change per C of cell
            temperature (negative).
        noct: nominal operating cell temperature (C).
        losses: lumped DC-side loss fraction.
        inverter_efficiency: DC to AC conversion efficiency.

    Returns:
        AC output in MW, non-negative.

    Example:
        >>> round(float(pv_power(1000.0, 25.0, capacity_mw_dc=100.0,
        ...                      temp_coefficient=0.0)), 2)
        82.56
    """
    g = jnp.asarray(irradiance, dtype=jnp.float64)
    t_cell = pv_cell_temperature(g, t_ambient, noct=noct)
    p_dc = (
        capacity_mw_dc
        * (g / STC_IRRADIANCE)
        * (1.0 + temp_coefficient * (t_cell - STC_TEMPERATURE))
        * (1.0 - losses)
    )
    p_ac = jnp.maximum(inverter_efficiency * p_dc, 0.0)
    if inverter_mw is not None:
        p_ac = jnp.minimum(p_ac, inverter_mw)
    return p_ac


def hydro_power(
    flow: Array | float,
    head: Array | float,
    *,
    efficiency: float = 0.90,
    water_density: float = 1000.0,
) -> Array:
    """Hydraulic turbine output (MW): ``rho g Q H eta``.

    Args:
        flow: turbined flow (m^3/s).
        head: net head across the turbine (m), i.e. after penstock
            losses.
        efficiency: combined turbine and generator efficiency.
        water_density: kg/m^3.

    Returns:
        Output in MW.

    Example:
        >>> round(float(hydro_power(100.0, 50.0)), 2)
        44.13
    """
    q = jnp.asarray(flow, dtype=jnp.float64)
    return water_density * GRAVITY * q * head * efficiency / 1e6


# =============================================================================
# Generator factories
# =============================================================================


def renewable_generator(
    bus: str,
    available_mw: float,
    s_max_mva: float,
    *,
    cost: Sequence[float] = (0.0,),
    vm_setpoint: float | None = None,
    name: str | None = None,
) -> Generator:
    """A curtailable generator behind a converter rated in MVA.

    Real output may go anywhere from zero to ``available_mw``; reactive
    output anywhere inside the capability circle of radius
    ``s_max_mva``. The power-flow schedule ``p_mw`` is the available
    power, i.e. no curtailment, with zero vars.

    Args:
        bus: bus id it injects at. A PQ bus fixes both its outputs in a
            power flow; a PV bus has it hold voltage.
        available_mw: what the resource can deliver this hour (MW).
        s_max_mva: converter (or machine) apparent-power rating.
        cost: polynomial cost in $/h of MW, highest order first. The
            default zero is a unit with no fuel; a negative linear term
            models a production credit, which the OPF will prefer to
            curtail last.
        vm_setpoint: voltage it regulates to on a PV bus.
        name: human label.

    Returns:
        A :class:`~difflow_power.network.Generator`.
    """
    available = float(available_mw)
    if available < 0.0:
        raise ValueError(f"available_mw must be non-negative, got {available}")
    s_max = float(s_max_mva)
    return Generator(
        bus=bus,
        p_min_mw=0.0,
        p_max_mw=available,
        q_min_mvar=-s_max,
        q_max_mvar=s_max,
        cost=tuple(cost),
        p_mw=min(available, s_max),
        q_mvar=0.0,
        vm_setpoint=vm_setpoint,
        name=name,
        s_max_mva=s_max,
    )


def wind_generator(
    bus: str,
    wind_speed: float,
    *,
    n_turbines: int,
    turbine_mw: float,
    s_max_mva: float | None = None,
    losses: float = 0.0,
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0,
    air_density: float = REFERENCE_AIR_DENSITY,
    cost: Sequence[float] = (0.0,),
    vm_setpoint: float | None = None,
    name: str | None = None,
) -> Generator:
    """A wind farm at one wind speed.

    Args:
        bus: bus id of its point of interconnection.
        wind_speed: hub-height wind speed (m/s).
        n_turbines: number of turbines.
        turbine_mw: rating of each (MW).
        s_max_mva: the farm's converter rating; the nameplate
            ``n_turbines * turbine_mw`` by default (unity power factor
            at full output).
        losses: fraction lost to wakes, collection and availability.
        cut_in, rated_speed, cut_out, air_density: the power curve; see
            :func:`wind_turbine_power`.
        cost, vm_setpoint, name: as :func:`renewable_generator`.
    """
    nameplate = n_turbines * turbine_mw
    available = (
        n_turbines
        * (1.0 - losses)
        * wind_turbine_power(
            wind_speed,
            turbine_mw,
            cut_in=cut_in,
            rated_speed=rated_speed,
            cut_out=cut_out,
            air_density=air_density,
        )
    )
    return renewable_generator(
        bus,
        float(available),
        nameplate if s_max_mva is None else s_max_mva,
        cost=cost,
        vm_setpoint=vm_setpoint,
        name=name,
    )


def solar_generator(
    bus: str,
    irradiance: float,
    t_ambient: float = 25.0,
    *,
    capacity_mw_dc: float,
    inverter_mva: float | None = None,
    temp_coefficient: float = -0.0047,
    noct: float = 45.0,
    losses: float = 0.14,
    inverter_efficiency: float = 0.96,
    cost: Sequence[float] = (0.0,),
    vm_setpoint: float | None = None,
    name: str | None = None,
) -> Generator:
    """A solar plant at one irradiance and ambient temperature.

    The inverter rating does double duty: it clips the real output (see
    :func:`pv_power`) and is the radius of the capability circle.

    Args:
        bus: bus id of its point of interconnection.
        irradiance: plane-of-array irradiance (W/m^2).
        t_ambient: ambient temperature (C).
        capacity_mw_dc: module nameplate (MW DC).
        inverter_mva: inverter rating; ``capacity_mw_dc / 1.2`` by
            default, a typical DC/AC ratio and PVWatts' default.
        temp_coefficient, noct, losses, inverter_efficiency: see
            :func:`pv_power`.
        cost, vm_setpoint, name: as :func:`renewable_generator`.
    """
    inverter = capacity_mw_dc / 1.2 if inverter_mva is None else inverter_mva
    available = pv_power(
        irradiance,
        t_ambient,
        capacity_mw_dc=capacity_mw_dc,
        inverter_mw=inverter,
        temp_coefficient=temp_coefficient,
        noct=noct,
        losses=losses,
        inverter_efficiency=inverter_efficiency,
    )
    return renewable_generator(
        bus,
        float(available),
        inverter,
        cost=cost,
        vm_setpoint=vm_setpoint,
        name=name,
    )


def hydro_generator(
    bus: str,
    flow: float,
    head: float,
    *,
    machine_mva: float,
    efficiency: float = 0.90,
    cost: Sequence[float] = (0.0,),
    vm_setpoint: float | None = None,
    name: str | None = None,
) -> Generator:
    """A hydro unit at a given available flow and net head.

    The available flow is what the river (run-of-river) or the release
    schedule (a reservoir) allows this hour; the OPF may spill some of
    it. ``cost`` is where a reservoir's water value goes. The capability
    is the machine's armature circle; a synchronous machine's field
    heating and stability limits, which cut into the circle at leading
    and lagging power factor, are not modelled.

    Args:
        bus: bus id.
        flow: available turbine flow (m^3/s).
        head: net head (m).
        machine_mva: generator rating.
        efficiency: turbine and generator efficiency.
        cost, vm_setpoint, name: as :func:`renewable_generator`.
    """
    available = hydro_power(flow, head, efficiency=efficiency)
    return renewable_generator(
        bus,
        float(available),
        machine_mva,
        cost=cost,
        vm_setpoint=vm_setpoint,
        name=name,
    )


# =============================================================================
# Curtailment
# =============================================================================


@dataclass
class Curtailment(ParamsMixin):
    """How much of a unit's available output an OPF left unused.

    Attributes:
        available_mw: the unit's ceiling ``p_max_mw``.
        dispatched_mw: its optimal real output.
        curtailed_mw: the difference, never negative.
        value_per_mw: reduction in optimal cost per MW of extra
            available output ($/MWh). Zero for a unit that is curtailed
            (more resource would also go unused); positive for one the
            OPF runs flat out, where it is what that resource is worth
            at this operating point.
    """

    available_mw: float
    dispatched_mw: float
    curtailed_mw: float
    value_per_mw: float


def curtailment(result, generators: Sequence[str] | None = None) -> dict:
    """Curtailment and marginal resource value of units in a solved OPF.

    The value is read from the multiplier on the unit's ceiling, by the
    envelope theorem: ``d(cost*)/d(p_max) = -z``. Multiply it by
    ``jax.grad`` of a resource model to get the value of a unit of
    weather, e.g. $/h per m/s of wind speed.

    A unit whose floor and ceiling coincide (an available output of
    zero) is pinned by an equality instead, and its value is
    ``max(0, -d(cost*)/d(pin))``: more resource is worth something only
    if the OPF would use it.

    Args:
        result: an :class:`~difflow_power.opf.ACOPFResult`.
        generators: unit ids; every unit by default.

    Returns:
        ``{gid: Curtailment}``.
    """
    network, layout, structure = result.network, result.layout, result.structure
    base = network.base_mva
    gids = list(layout.generators) if generators is None else list(generators)
    pg = result.pg_mw
    ineq = structure.inequality_names
    n_eq_physics = 2 * layout.n_bus + 1
    out = {}
    for gid in gids:
        gen = network.generators[gid]
        idx = layout.index(f"pg_{gid}")
        name = f"pg_{gid}_max"
        if name in ineq:
            value = float(result.ipm.z[ineq.index(name)]) / base
        elif idx in structure.fixed_indices:
            row = n_eq_physics + structure.fixed_indices.index(idx)
            value = max(0.0, float(result.ipm.lam[row]) / base)
        else:
            value = 0.0
        out[gid] = Curtailment(
            available_mw=gen.p_max_mw,
            dispatched_mw=pg[gid],
            curtailed_mw=max(0.0, gen.p_max_mw - pg[gid]),
            value_per_mw=value,
        )
    return out
