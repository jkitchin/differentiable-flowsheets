"""Pipe insulation: installed cost and the economic insulation thickness.

The economic thickness balances the annualised cost of the insulation
against the cost of the heat it saves (McCabe, Smith & Harriott and Peters,
Timmerhaus & West both pose the problem this way): the total annual cost
per metre of pipe,

    TAC(t) = CRF(i, n) * C_inst(t) + q'(t) * hours * 3600 * c_heat,

is minimised over the thickness ``t``, with ``q'(t)`` the heat loss of
:func:`difflow.heat_transfer.insulated_pipe_heat_loss`.

Cost model
----------
The installed cost per metre of pipe is linear in the insulated volume,
with optional terms for the cladding (jacketing) area and a fixed cost per
metre:

    C_inst(t) = c_V * pi t (D_o + t) + c_A * pi (D_o + 2 t) + c_L.

Material and most of the application labour scale with the insulated volume,
the weather jacket with its outer area, and access/scaffolding with length.
That is the structure of the textbook treatments; it is **not** a fitted
vendor correlation. **No default unit cost is provided**: published
installed-cost data for pipe insulation (e.g. Turner & Malloy, *Thermal
Insulation Handbook*, 1981; the insulation-cost figure of Peters, Timmerhaus
& West, 5th ed.) could not be checked here, and a number recalled without a
check would be invented. ``cost_per_m3`` (and ``cost_per_m2``,
``cost_per_m``) are required inputs from your own quote or estimating data;
give their ``base_year`` and the cost is escalated by CEPCI
(:func:`difflow.economics.indices.escalate_cost`).

Everything is JAX and differentiable; the optimum thickness is found by a
bracketed root find of ``dTAC/dt`` (``optimistix.Bisection``) and is
differentiable in every numeric input through implicit differentiation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.economics.indices import get_cepci
from difflow.economics.profitability import capital_recovery_factor
from difflow.heat_transfer import insulated_pipe_heat_loss

#: Seconds per hour.
_S_PER_H = 3600.0


def _f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _escalation(base_year: int | None, target_year: int | None) -> float:
    if base_year is None and target_year is None:
        return 1.0
    if base_year is None or target_year is None:
        raise ValueError("give both base_year and target_year to escalate, or neither")
    return get_cepci(target_year) / get_cepci(base_year)


def insulation_volume_per_length(D_o: Array | float, thickness: Array | float) -> Array:
    """Insulation volume per metre of pipe, ``pi t (D_o + t)`` (m^3/m).

    The annulus between ``D_o`` and ``D_o + 2t``:
    ``pi/4 [(D_o + 2t)^2 - D_o^2] = pi t (D_o + t)``.

    Args:
        D_o: Pipe outside diameter (m).
        thickness: Insulation thickness (m).

    Returns:
        Volume per length (m^3/m).

    Example:
        >>> from difflow.economics.insulation import insulation_volume_per_length
        >>> round(float(insulation_volume_per_length(0.1, 0.05)), 6)
        0.023562
    """
    t = _f64(thickness)
    return jnp.pi * t * (_f64(D_o) + t)


def insulation_installed_cost(
    D_o: Array | float,
    thickness: Array | float,
    cost_per_m3: Array | float,
    cost_per_m2: Array | float = 0.0,
    cost_per_m: Array | float = 0.0,
    base_year: int | None = None,
    target_year: int | None = None,
) -> Array:
    """Installed insulation cost per metre of pipe ($/m), linear in the insulated volume.

    ``C = [c_V pi t (D_o + t) + c_A pi (D_o + 2t) + c_L] * CEPCI(target)/CEPCI(base)``
    (see the module docstring for the model and why there is no default unit
    cost). The jacket and fixed terms are only charged where there is
    insulation if you want them to be: they are added for any ``t``, so set
    them to zero for a bare-pipe comparison.

    Args:
        D_o: Pipe outside diameter (m).
        thickness: Insulation thickness (m).
        cost_per_m3: Installed cost per m^3 of insulation ($/m^3), required.
        cost_per_m2: Cladding/jacket cost per m^2 of outer surface ($/m^2).
        cost_per_m: Fixed cost per metre of pipe ($/m).
        base_year: Year of the unit costs (CEPCI); with ``target_year``.
        target_year: Year to escalate to; None for no escalation.

    Returns:
        Installed cost ($ per metre of pipe).

    Raises:
        ValueError: Only one of ``base_year`` / ``target_year`` given, or a
            year without CEPCI data.

    Example:
        >>> from difflow.economics.insulation import insulation_installed_cost
        >>> # illustrative unit cost, not a quoted price
        >>> round(float(insulation_installed_cost(0.1, 0.05, cost_per_m3=2000.0)), 2)
        47.12
    """
    D_o, t = _f64(D_o), _f64(thickness)
    cost = (_f64(cost_per_m3) * insulation_volume_per_length(D_o, t)
            + _f64(cost_per_m2) * jnp.pi * (D_o + 2.0 * t)
            + _f64(cost_per_m))
    return cost * _escalation(base_year, target_year)


def insulation_annual_cost(
    thickness: Array | float,
    D_o: Array | float,
    k_ins: Array | float,
    h_out: Array | float,
    T_pipe: Array | float,
    T_amb: Array | float,
    cost_per_m3: Array | float,
    heat_cost: Array | float,
    hours_per_year: Array | float = 8000.0,
    rate: Array | float = 0.10,
    years: Array | float = 10.0,
    cost_per_m2: Array | float = 0.0,
    cost_per_m: Array | float = 0.0,
    base_year: int | None = None,
    target_year: int | None = None,
) -> dict[str, Array]:
    """Total annual cost per metre of insulated pipe ($/m/yr) and its parts.

    ``TAC = CRF(rate, years) * C_inst + q' * hours * 3600 * heat_cost * 1e-9``.

    Args:
        thickness: Insulation thickness (m).
        D_o: Pipe outside diameter (m).
        k_ins: Insulation conductivity (W/m/K).
        h_out: Outer surface coefficient, convection plus radiation (W/m^2/K).
        T_pipe: Pipe surface temperature (K).
        T_amb: Ambient temperature (K).
        cost_per_m3: Installed insulation cost ($/m^3), required.
        heat_cost: Cost of the lost heat ($/GJ), e.g. the steam price
            divided by the boiler efficiency.
        hours_per_year: Operating hours per year.
        rate: Interest rate for the capital recovery factor (-).
        years: Insulation life (years).
        cost_per_m2, cost_per_m, base_year, target_year: See
            :func:`insulation_installed_cost`.

    Returns:
        Dict with ``total`` ($/m/yr), ``capital`` (annualised insulation,
        $/m/yr), ``heat`` (lost heat, $/m/yr), ``installed_cost`` ($/m) and
        ``heat_loss`` (W/m).
    """
    C = insulation_installed_cost(D_o, thickness, cost_per_m3, cost_per_m2, cost_per_m,
                                  base_year, target_year)
    q = insulated_pipe_heat_loss(D_o, thickness, k_ins, h_out, T_pipe, T_amb)
    capital = capital_recovery_factor(_f64(rate), _f64(years)) * C
    heat = q * _f64(hours_per_year) * _S_PER_H * _f64(heat_cost) * 1e-9
    return {"total": capital + heat, "capital": capital, "heat": heat,
            "installed_cost": C, "heat_loss": q}


def economic_insulation_thickness(
    D_o: Array | float,
    k_ins: Array | float,
    h_out: Array | float,
    T_pipe: Array | float,
    T_amb: Array | float,
    cost_per_m3: Array | float,
    heat_cost: Array | float,
    hours_per_year: Array | float = 8000.0,
    rate: Array | float = 0.10,
    years: Array | float = 10.0,
    cost_per_m2: Array | float = 0.0,
    cost_per_m: Array | float = 0.0,
    base_year: int | None = None,
    target_year: int | None = None,
    t_max: float = 0.5,
    t_min: float = 1.0e-4,
) -> dict[str, Array]:
    """Insulation thickness that minimises the total annual cost (:func:`insulation_annual_cost`).

    Solves ``dTAC/dt = 0`` for ``t`` in ``[t_min, t_max]`` by bisection
    (``optimistix.Bisection``); the returned thickness is differentiable in
    every numeric argument by implicit differentiation (``jax.grad`` of
    ``t_opt`` with respect to the heat price, the conductivity, the unit
    cost, ...). ``dTAC/dt`` is increasing in ``t`` beyond the critical
    radius, so the root is unique there. If it has no sign change in the
    bracket (insulation never pays, or the optimum is thicker than
    ``t_max``) the result sits at a bracket end and ``interior`` is False.

    Args:
        D_o, k_ins, h_out, T_pipe, T_amb, cost_per_m3, heat_cost,
            hours_per_year, rate, years, cost_per_m2, cost_per_m, base_year,
            target_year: As :func:`insulation_annual_cost`.
        t_max: Upper end of the thickness bracket (m), a Python float.
        t_min: Lower end (m), a Python float (above zero).

    Returns:
        Dict with ``thickness`` (m), the :func:`insulation_annual_cost`
        entries at that thickness, ``bare_heat_loss`` (W/m, no insulation)
        and ``interior`` (bool array: the optimum is inside the bracket).

    Example:
        >>> from difflow.economics.insulation import economic_insulation_thickness
        >>> # 4-in steam line at 450 K; ILLUSTRATIVE unit cost and heat price
        >>> out = economic_insulation_thickness(0.1143, 0.05, 10.0, 450.0, 293.15,
        ...                                     cost_per_m3=3000.0, heat_cost=10.0)
        >>> 0.02 < float(out["thickness"]) < 0.2 and bool(out["interior"])
        True
    """
    args = (D_o, k_ins, h_out, T_pipe, T_amb, cost_per_m3, heat_cost, hours_per_year,
            rate, years, cost_per_m2, cost_per_m)

    def tac(t, a):
        return insulation_annual_cost(t, *a, base_year=base_year, target_year=target_year)["total"]

    dtac = jax.grad(tac, argnums=0)

    def residual(t, a):
        return dtac(t, a)

    lo, hi = _f64(t_min), _f64(t_max)
    sol = optx.root_find(
        residual,
        optx.Bisection(rtol=1e-12, atol=1e-12, flip=False),   # dTAC/dt < 0 at t_min, > 0 at t_max
        0.5 * (lo + hi),
        args=args,
        options={"lower": lo, "upper": hi},
        max_steps=200,
        throw=False,
    )
    t_opt = sol.value
    interior = (residual(lo, args) < 0.0) & (residual(hi, args) > 0.0)
    out = insulation_annual_cost(t_opt, *args, base_year=base_year, target_year=target_year)
    out["thickness"] = t_opt
    out["interior"] = interior
    out["bare_heat_loss"] = insulated_pipe_heat_loss(D_o, 0.0, k_ins, h_out, T_pipe, T_amb)
    return out


__all__ = [
    "insulation_volume_per_length",
    "insulation_installed_cost",
    "insulation_annual_cost",
    "economic_insulation_thickness",
]
