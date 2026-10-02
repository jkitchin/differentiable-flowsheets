"""Light-ends fractionation for the alkylation unit: a key-recovery shortcut column.

The depropanizer, deisobutanizer and debutanizer are each a
Fenske-Underwood-Gilliland shortcut column specified by the recoveries of
its two keys and its reflux ratio. The equations are the textbook ones:

* **Relative volatility** ``alpha_i = Psat_i(T) / Psat_HK(T)`` (Raoult),
  Lee-Kesler vapour pressures from the database critical constants
  (Lee & Kesler, AIChE J. 21(3), 510 (1975)), at the column's key
  temperature ``T_col``: the temperature at which ``Psat_LK Psat_HK =
  P^2``, i.e. where the geometric-mean key volatility is one.
* **Non-key split** (Hengstebeck-Geddes): ``log(d_i/b_i) = A + C log
  alpha_i`` with ``A = log(d_HK/b_HK)`` and ``C = [log(d_LK/b_LK) -
  log(d_HK/b_HK)] / log alpha_LK``, so the keys' own splits are reproduced
  (Geddes, AIChE J. 4, 389 (1958); Hengstebeck, *Distillation*, Reinhold,
  1961). The fraction to the distillate is ``sigmoid(A + C log alpha_i)``,
  so every species is conserved exactly.
* **Minimum stages** (Fenske, Ind. Eng. Chem. 24, 482 (1932)):
  ``N_min = log[(d/b)_LK (b/d)_HK] / log alpha_LK``.
* **Minimum reflux** (Underwood, Chem. Eng. Prog. 44, 603 (1948)), feed at
  its bubble point (q = 1): ``sum alpha_i z_i / (alpha_i - theta) = 0`` for
  the root just above the heavy key (``1 < theta < alpha_LK`` when no
  non-key boils between the keys), then ``R_min + 1 = sum alpha_i x_D,i / (alpha_i -
  theta)``.
* **Stages** (Gilliland, Ind. Eng. Chem. 32, 1220 (1940), in Molokanov's
  form, Int. Chem. Eng. 12, 209 (1972)): ``Y = 1 - exp[(1 + 54.4 X) /
  (11 + 117.2 X) (X - 1) / sqrt(X)]``.
* **Duties**, constant molar overflow with saturated-liquid feed and
  products: the vapour rate is ``V = (R + 1) D`` in both sections; the
  condenser duty is ``V`` times the distillate's latent heat at the top
  temperature and the reboiler duty ``V`` times the bottoms' latent heat at
  the bottom temperature (sensible-heat differences between feed and
  products neglected; with unequal molar latent heats at the two ends the
  two duties differ, which is the CMO assumption's own inconsistency).
  Latent heats by Watson's relation from the CRC ``dHvap`` at 298.15 K
  (Watson, Ind. Eng. Chem. 35, 398 (1943)); the end temperatures are the
  Raoult bubble points of the distillate (total condenser) and of the
  bottoms (reboiler).

Bibliographic details of the four classical papers are as commonly cited
(the copies were not opened); the equations themselves are checked in the
tests against their defining properties (the keys' splits reproduced, the
Underwood root, Fenske's limit).

Why not :class:`difflow.units.distillation.ShortcutColumn`. It solves the
same equations with Peng-Robinson bubble points at both column ends, and it
is available here (``AlkylationUnitParams(fractionation="pr_shortcut")``,
forward solves). Differentiating the isobutane recycle through three of
them -- nested Newton bubble-point solves inside the tear's implicit
fixed point -- did not fit in this machine's memory. (Its non-key split
also had a sign error in the Geddes constants when this module was written;
that is fixed in ``ShortcutColumn`` itself.) This column has no inner solve at all apart from two fixed-length bisections,
and its gradient is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream

from difflow_refinery.alkylation.reactor import flows_array, stream_from_array
from difflow_refinery.alkylation.species import ALKYLATION_SPECIES, property_array
from difflow_refinery.correlations import vapor_pressure

_IDX = {s: i for i, s in enumerate(ALKYLATION_SPECIES)}
_TC = jnp.asarray(property_array(ALKYLATION_SPECIES, "Tc"))
_PC = jnp.asarray(property_array(ALKYLATION_SPECIES, "Pc"))
_OMEGA = jnp.asarray(property_array(ALKYLATION_SPECIES, "omega"))
_HV298 = jnp.asarray(property_array(ALKYLATION_SPECIES, "Hvap298"))

#: Bisection passes of the scalar solves (temperatures, Underwood root).
_BISECTIONS = 80


def psat(T: Array) -> Array:
    """Lee-Kesler vapour pressures (Pa) of every alkylation species at ``T``."""
    return vapor_pressure(T, _TC, _PC, _OMEGA)


def latent_heat(T: Array) -> Array:
    """Watson latent heats (J/mol) of every species at ``T``, from CRC dHvap(298.15 K).

    ``dHvap(T) = dHvap(298.15) ((1 - T/Tc) / (1 - 298.15/Tc))^0.38``,
    floored at a reduced temperature of 0.999.
    """
    tr = jnp.minimum(T / _TC, 0.999)
    return _HV298 * ((1.0 - tr) / (1.0 - 298.15 / _TC)) ** 0.38


def _bisect(f, lo, hi, n=_BISECTIONS):
    """Root of the increasing scalar ``f`` on ``[lo, hi]``, with an implicit gradient.

    Fixed-length bisection on stop-gradient copies, then one Newton step on
    the live function -- which attaches ``-f_theta / f_x``, the implicit
    function gradient, exactly as :func:`difflow_refinery.blending.tbp_temperature`
    does.
    """
    fs = lambda x: jax.lax.stop_gradient(f(x))  # noqa: E731

    def body(_, br):
        a, b = br
        m = 0.5 * (a + b)
        neg = fs(m) < 0.0
        return jnp.where(neg, m, a), jnp.where(neg, b, m)

    a, b = jax.lax.fori_loop(0, n, body, (jax.lax.stop_gradient(lo), jax.lax.stop_gradient(hi)))
    x0 = 0.5 * (a + b)
    dfdx = jax.lax.stop_gradient(jax.grad(f)(x0))
    return x0 - f(x0) / dfdx


def boiling_temperature(index: int, P: Array) -> Array:
    """Temperature (K) at which species ``index`` has vapour pressure ``P``."""
    f = lambda T: jnp.log(vapor_pressure(T, _TC[index], _PC[index], _OMEGA[index]) / P)  # noqa: E731
    return _bisect(f, jnp.asarray(150.0), 0.999 * _TC[index])


def bubble_temperature(x: Array, P: Array) -> Array:
    """Raoult bubble point (K) of liquid mole fractions ``x`` at ``P``: ``sum x Psat(T) = P``."""
    # The bracket is not capped at the lightest species' Tc: a trace of
    # propane in a bottoms at 390 K is supercritical, and capping there would
    # stop the bisection short of the root. The Lee-Kesler expression is
    # smooth and increasing past Tr = 1, which is all a bubble point of a
    # mixture with supercritical traces needs.
    f = lambda T: jnp.log(jnp.sum(x * psat(T))) - jnp.log(P)  # noqa: E731
    return _bisect(f, jnp.asarray(150.0), jnp.asarray(700.0))


@dataclass(repr=False)
class KeySplitColumnParams(ParamsMixin):
    """Specs of a key-recovery shortcut column.

    Attributes:
        light_key: Light key species.
        heavy_key: Heavy key species.
        lk_recovery: Fraction of the light key's feed to the distillate.
        hk_recovery: Fraction of the heavy key's feed to the bottoms.
        reflux_ratio: Reflux ratio ``L/D``; should exceed the Underwood
            minimum (``info["R_min"]``, ``info["feasible"]``).
        P: Column pressure (Pa).
    """

    light_key: str
    heavy_key: str
    lk_recovery: float
    hk_recovery: float
    reflux_ratio: float
    P: float


class KeySplitColumn:
    """Fenske-Underwood-Gilliland shortcut column on Lee-Kesler volatilities.

    ``column(feed) -> (distillate, bottoms, info)``; ``info`` holds
    ``alpha`` (array), ``T_col``, ``T_top``, ``T_bot`` (K), ``N_min``,
    ``R_min``, ``N``, ``theta``, ``D``, ``B`` (mol/s), ``Q_condenser`` (W,
    negative: removed) and ``Q_reboiler`` (W), and ``feasible`` (reflux above
    minimum and the light key more volatile than the heavy key).
    """

    def __init__(self, params: KeySplitColumnParams):
        self.params = params
        p = params
        for k in (p.light_key, p.heavy_key):
            if k not in _IDX:
                raise KeyError(f"key {k!r} is not an alkylation species")
        self._lk, self._hk = _IDX[p.light_key], _IDX[p.heavy_key]

    def key_temperature(self, P: Array) -> Array:
        """``T`` with ``Psat_LK(T) Psat_HK(T) = P^2`` (K)."""
        lk, hk = self._lk, self._hk

        def f(T):
            return 0.5 * (jnp.log(vapor_pressure(T, _TC[lk], _PC[lk], _OMEGA[lk]))
                          + jnp.log(vapor_pressure(T, _TC[hk], _PC[hk], _OMEGA[hk]))) - jnp.log(P)
        return _bisect(f, jnp.asarray(150.0), 0.999 * jnp.minimum(_TC[lk], _TC[hk]))

    def __call__(self, feed: Stream):
        p = self.params
        P = jnp.asarray(p.P, dtype=jnp.float64)
        F = flows_array(feed)
        lk, hk = self._lk, self._hk
        T_col = self.key_temperature(P)
        ps = psat(T_col)
        alpha = ps / ps[hk]
        r_lk, r_hk = p.lk_recovery, p.hk_recovery
        log_hk = jnp.log((1.0 - r_hk) / r_hk)          # log(d/b) of the heavy key
        log_lk = jnp.log(r_lk / (1.0 - r_lk))          # log(d/b) of the light key
        A = log_hk
        C = (log_lk - log_hk) / jnp.log(alpha[lk])
        frac = jax.nn.sigmoid(A + C * jnp.log(alpha))
        frac = frac.at[lk].set(r_lk).at[hk].set(1.0 - r_hk)
        d = F * frac
        b = F - d
        D, B, Ft = jnp.sum(d), jnp.sum(b), jnp.sum(F)
        xD, z = d / D, F / Ft

        n_min = (log_lk - log_hk) / jnp.log(alpha[lk])
        # Underwood, q = 1: sum alpha z / (alpha - theta) = 0. The function
        # rises from -inf to +inf between each pair of adjacent volatilities
        # of the species present; the root wanted is the one just above the
        # heavy key -- between the heaviest present species at or below
        # alpha = 1 (the heavy key itself when the feed has it) and the next
        # lighter present species (the light key when no non-key boils
        # between them). Species with z = 0 drop out.
        def und(theta):
            return jnp.sum(alpha * z / (alpha - theta))
        present = z > 1e-12
        lo = jnp.max(jnp.where(present & (alpha <= 1.0 + 1e-12), alpha, 0.0))
        hi = jnp.min(jnp.where(present & (alpha > lo * (1.0 + 1e-9)), alpha, alpha[lk]))
        lo, hi = jax.lax.stop_gradient(lo), jax.lax.stop_gradient(hi)
        theta = _bisect(und, lo + 1e-9 * hi, hi * (1.0 - 1e-9))
        r_min = jnp.sum(alpha * xD / (alpha - theta)) - 1.0
        R = jnp.asarray(p.reflux_ratio, dtype=jnp.float64)
        X = jnp.clip((R - r_min) / (R + 1.0), 1e-6, 1.0)
        Y = 1.0 - jnp.exp((1.0 + 54.4 * X) / (11.0 + 117.2 * X) * (X - 1.0) / jnp.sqrt(X))
        n_stages = (n_min + Y) / (1.0 - Y)

        xB = b / B
        T_top = bubble_temperature(xD, P)
        T_bot = bubble_temperature(xB, P)
        V = (R + 1.0) * D
        q_cond = -V * jnp.sum(xD * latent_heat(T_top))
        q_reb = V * jnp.sum(xB * latent_heat(T_bot))
        info = {
            "alpha": alpha, "T_col": T_col, "T_top": T_top, "T_bot": T_bot,
            "N_min": n_min, "R_min": r_min, "N": n_stages, "theta": theta,
            "D": D, "B": B, "Q_condenser": q_cond, "Q_reboiler": q_reb,
            "feasible": (R > r_min) & (alpha[lk] > 1.0),
        }
        return stream_from_array(d, T_top, P), stream_from_array(b, T_bot, P), info


__all__ = ["KeySplitColumn", "KeySplitColumnParams", "boiling_temperature", "bubble_temperature", "latent_heat", "psat"]
