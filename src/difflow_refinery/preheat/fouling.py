"""Crude-side fouling: the Ebert-Panchal threshold model.

The fouling resistance of a preheat exchanger grows by deposition, which
is chemical and so Arrhenius in the film temperature, and shrinks by
removal, which goes with the wall shear stress (Ebert and Panchal, 1995)::

    dR_f/dt = alpha Re^beta Pr^(-0.33) exp(-E / (R T_film)) - gamma tau_w

Below the *threshold* -- the film temperature and velocity at which the two
terms cancel -- an exchanger does not foul. The form is the one the
literature uses; **the default constants are illustrative**. They are not
fitted to any crude: they are chosen only so that a hot-end exchanger fouls
at about 1e-11 m^2 K/J (a few 1e-4 m^2 K/W a year) and the cold end not at
all, which is the right order for a crude preheat train. Fit ``alpha``,
``E`` and ``gamma`` to your own monitoring data before reading a cleaning
date off them.

What the model is for here is the *pattern*: which exchangers foul fastest
(the hot end, with low velocity), so that :meth:`PreheatedCrudeUnit.cleaning_ranking
<difflow_refinery.preheat.unit.PreheatedCrudeUnit.cleaning_ranking>` has
fouling resistances to rank that differ for a reason.

Reference:
    Ebert, W. and Panchal, C. B. (1995). Analysis of Exxon crude-oil slip
    stream coking data. In *Fouling Mitigation of Industrial Heat-Exchange
    Equipment*, Begell House, pp. 451-460 (verify the page range).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

R_GAS = 8.314462618


@dataclass
class EbertPanchal(ParamsMixin):
    """Constants of the Ebert-Panchal fouling model. The defaults are illustrative.

    Attributes:
        alpha: Deposition pre-factor (m^2 K/J). Illustrative.
        beta: Reynolds-number exponent of deposition.
        E: Activation energy of deposition (J/mol). Illustrative.
        gamma: Removal constant (m^2 K/J/Pa). Illustrative.
    """

    alpha: float | Array = 2.0e-3
    beta: float | Array = -0.66
    E: float | Array = 48.0e3
    gamma: float | Array = 4.0e-13


def film_temperature(T_bulk, T_wall):
    """The film temperature the deposition term sees: ``T_b + 0.55 (T_w - T_b)``."""
    return T_bulk + 0.55 * (T_wall - T_bulk)


def fouling_rate(model: EbertPanchal, Re, Pr, T_film, tau_w) -> Array:
    """``dR_f/dt`` (m^2 K/J, that is m^2 K/W per second); negative below the threshold."""
    m = model
    dep = m.alpha * jnp.asarray(Re) ** m.beta * jnp.asarray(Pr) ** (-0.33) * jnp.exp(
        -m.E / (R_GAS * jnp.asarray(T_film)))
    return dep - m.gamma * jnp.asarray(tau_w)


def exchanger_film_temperature(exchanger_result: dict, area, h_crude) -> Array:
    """Crude-side film temperature of a solved exchanger.

    The bulk crude temperature is the mean of its inlet and outlet; the
    wall is above it by the mean heat flux over the crude-side film
    coefficient, ``q / h = Q / (A h)``.

    Args:
        exchanger_result: One entry of :attr:`PreheatTrainResult.exchangers
            <difflow_refinery.preheat.train.PreheatTrainResult.exchangers>`.
        area: Its area (m^2).
        h_crude: Crude-side film coefficient (W/m^2/K).
    """
    e = exchanger_result
    T_b = 0.5 * (e["T_cold_in"] + e["T_cold_out"])
    return film_temperature(T_b, T_b + e["Q"] / (area * h_crude))


def fouling_rates(train_result, train_params, model: EbertPanchal | None = None,
                  Re=3.0e4, Pr=10.0, tau_w=2.0) -> dict:
    """Ebert-Panchal ``dR_f/dt`` (m^2 K/J) of every exchanger in a solved train, by name.

    ``Re``, ``Pr`` and ``tau_w`` (Pa) are the crude side's; one value for
    all exchangers or a ``{name: value}`` dict. They are inputs, not
    computed: the train carries no geometry beyond the area. Rates are
    floored at zero -- the model's removal term stops a clean wall from
    fouling, it does not clean a fouled one.
    """
    model = EbertPanchal() if model is None else model

    def pick(v, name):
        return v[name] if isinstance(v, dict) else v

    out = {}
    for ex in train_params.exchangers:
        T_f = exchanger_film_temperature(train_result.exchangers[ex.name], ex.area, ex.h_crude)
        r = fouling_rate(model, pick(Re, ex.name), pick(Pr, ex.name), T_f, pick(tau_w, ex.name))
        out[ex.name] = jnp.maximum(r, 0.0)
    return out


__all__ = ["EbertPanchal", "exchanger_film_temperature", "film_temperature", "fouling_rate", "fouling_rates"]
