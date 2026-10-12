"""Crystallization: solubility, equilibrium yield with hydrates, the MSMPR population balance and the ``Crystallizer`` unit.

Library (pure JAX, SI, differentiable)
--------------------------------------
:class:`Solubility` is a small pytree holding a solubility curve ``c_sat(T)`` in
**kg of anhydrous solute per kg of solvent** (the ``per_100g=True`` switch takes
data in g per 100 g water and rescales), as a polynomial in the temperature in
degrees C, a van 't Hoff form ``ln c = A + B/T`` or a PCHIP table, together with
the **hydrate form** of the crystal: ``hydrate_waters`` (moles of water of
crystallization per mole of solute, nu) and the molar masses. An optional
``upper`` curve with ``T_transition`` is a second solid form (for example the
anhydrous salt above a decahydrate transition); the switch between the two is a
logistic blend of width ``width`` K in ``T`` (``c_sat`` and ``nu`` are blended),
never a Python ``if`` on a traced ``T``. The alternative is an explicit user
choice: pass the curve of the form you mean.

:func:`crystal_yield` and :func:`crystal_balance` are the textbook yield
balance. With ``S0`` the solute and ``W0`` the solvent (kg/s) of the feed, ``E``
the solvent evaporated, ``W = W0 - E``, and a crystal (hydrate) of solute mass
fraction ``f_s = M_s / (M_s + nu M_w)`` and water fraction ``f_w = 1 - f_s``::

    solute:  S0 = c W_L + f_s C          water:  W = W_L + f_w C
    C = (S0 - c_sat W) / (f_s - c_sat f_w)

``C`` is the hydrate mass flow. It is clipped at zero with a *soft* clip,
``C = s softplus(C_raw / s)`` with ``s = soft * S0`` (``soft`` defaults to 1e-6),
so an undersaturated feed gives (numerically exactly) zero yield, a finite
gradient and no Python ``if``; away from the corner the soft clip changes
nothing. The mother liquor flows are always computed from the balances with the
clipped ``C``, so both balances close to rounding. A feed richer in solute than
the hydrate stoichiometry (``S0/W > f_s/f_w``) would need an anhydrous solid; the
balance then gives a negative liquor water, which ``crystal_balance`` reports
rather than hides. The clip acts on the numerator ``S0 - c_sat W`` (the solute
above saturation), and the denominator ``f_s - c_sat f_w`` is floored at 1e-9 (a
saturation concentration above the hydrate stoichiometry has no equilibrium
hydrate yield).

MSMPR (mixed-suspension, mixed-product-removal; Randolph & Larson) with
residence time ``tau``, growth rate ``G`` and nuclei density ``n0``::

    n(L) = n0 exp(-L / (G tau))        m_k = int L^k n dL = k! n0 (G tau)^(k+1)
    M_T = 6 rho_c k_v n0 (G tau)^4     L_D = 3 G tau  (mode of L^3 n)
    L_50 = x_50 G tau,  x_50 = 3.672...  (median of the gamma(4) mass law)

:func:`msmpr_distribution`, :func:`msmpr_moments`, :func:`msmpr_dominant_size`,
:func:`msmpr_mass_median_size` (the root of the gamma(4) cumulative fraction at
one half, computed, not typed), :func:`msmpr_mass_cv` (1/sqrt(4) = 50 %),
:func:`msmpr_from_magma`, :func:`fit_msmpr` (the "ln n vs L" line) and
:func:`sieve_population_density` (a screen analysis converted to ``n(L)``,
``n_i = M_T w_i / (rho_c k_v Lbar_i^3 dL_i)``, the data that ``fit_msmpr`` fits).

Kinetics are power laws in the absolute supersaturation ``dC = c - c_sat``
(kg solute per kg solvent): :func:`growth_rate` ``G = k_g dC^g`` (m/s) and
:func:`nucleation_rate` ``B0 = k_b M_T^j dC^b`` (1/m^3/s), with ``M_T`` the
crystal (hydrate) mass per m^3 of liquor. :func:`msmpr_steady_state` closes them:
the solute balance gives ``M_T(dC)``, the third moment gives
``M_T = 6 rho_c k_v B0 G^3 tau^4``, and the scalar root in ``dC`` is found by a
bisection in ``ln(dC / dC_max)`` with an implicit-gradient Newton step
(``difflow.staged._solve_monotone``, so ``jax.grad`` of ``L_D`` with respect to
``tau``, ``k_g``, ``k_b`` is the implicit-function derivative). In the log
form the residual is ``(1 - j) ln M_T - (b + 3 g) ln dC - const``, strictly
decreasing for ``j <= 1``, which is what the bisection needs (``j > 1`` is
rejected). Because ``G`` depends on ``dC`` and ``dC`` on ``tau``, ``L_D(tau)`` is
*not* ``3 G tau`` at fixed ``G``.

Streams (the ``Crystallizer`` convention)
-----------------------------------------
All streams are in **mol/s**, K, Pa. The crystal stream carries the hydrate as
**solute plus bound water**: ``F_<solute>`` is the moles of (anhydrous) solute in
the crystals and ``F_<solvent>`` is ``nu`` times that, the water of
crystallization. Every species of the feed appears in every outlet stream (zero
where absent), so the mole balance closes species by species and a recycle keeps
one species list. The hydrate mass flow, the vapor of an evaporative duty and
the size distribution are in ``info`` (the water evaporated leaves the
flowsheet; it is not a stream). Feeding the crystal stream to a ``Dryer`` is a
dehydration, which the dryer's solid/moisture split does not model.

Water-removal duty: ``Q = m_F cp (T - T_F) + E h_fg(T) - dH_c C`` (W, positive
heat added; negative is cooling). In ``mode="adiabatic"`` (a vacuum
crystallizer) the same balance is solved for ``E`` with ``Q`` set to
``duty`` (0 by default): the feed flashes to ``T`` and its sensible heat and the heat
of crystallization evaporate the solvent. ``cp`` is a constant solution heat capacity,
``h_fg`` comes from :func:`difflow.water.water_latent_heat` and ``dH_c`` is the heat of
crystallization released (J per kg of crystal, user data, default 0).

Not modeled: size-dependent growth, fines destruction or classified product
removal, agglomeration and breakage, polymorph kinetics, batch cooling profiles.
The reactive version is the REE precipitators (``difflow_ree``).

Data caveat: **no solubility data, kinetic constant or hydrate transition
temperature is built in**; everything a test or an example uses is either
synthetic (labelled illustrative) or an explicitly flagged unverified recollection.

References
----------
- Randolph & Larson, Theory of Particulate Processes, 2nd ed. (1988) (MSMPR).
- Mullin, Crystallization, 4th ed. (2001).
- Geankoplis, Transport Processes and Separation Process Principles, Sec. 12.11.
- McCabe, Smith & Harriott, Unit Operations of Chemical Engineering, Ch. 27.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from difflow.params_mixin import ParamsMixin
from difflow.staged import _pchip_eval, _pchip_slopes, _solve_monotone
from difflow.streams import Stream
from difflow.water import water_latent_heat

_TINY = 1e-300
_T0C = 273.15
_C_FLOOR = 1e-9


def _f(x) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


# =============================================================================
# Solubility
# =============================================================================


@jax.tree_util.register_pytree_node_class
class Solubility:
    """Solubility curve ``c_sat(T)`` of a solute with its hydrate form, as a pytree.

    ``c_sat`` is in kg of anhydrous solute per kg of solvent. Build one with
    :meth:`polynomial`, :meth:`vant_hoff` or :meth:`table`.

    Args:
        kind: ``"polynomial"``, ``"vant_hoff"`` or ``"table"``.
        coeffs: Polynomial coefficients (ascending, in ``T - 273.15`` degrees C)
            or ``[A, B]`` of ``ln c = A + B / T`` (T in K).
        T: Table temperatures (K, increasing).
        c: Table solubilities.
        hydrate_waters: Moles of water of crystallization per mole of solute (0
            for an anhydrous crystal).
        solute_MW: Molar mass of the anhydrous solute (g/mol).
        solvent_MW: Molar mass of the solvent (g/mol).
        per_100g: The data are g solute per 100 g solvent (rescaled by 1/100).
        upper: Another :class:`Solubility` for the solid form stable above
            ``T_transition`` (it supplies its own ``hydrate_waters``).
        T_transition: Transition temperature (K) between the two forms.
        width: Width (K) of the logistic blend between the forms.

    Example:
        >>> sol = Solubility.polynomial([0.1, 0.004], hydrate_waters=10, solute_MW=106.0)
        >>> float(sol.c_sat(300.0)) > 0.1
        True
    """

    def __init__(self, kind="polynomial", coeffs=None, T=None, c=None,
                 hydrate_waters=0.0, solute_MW=100.0, solvent_MW=18.015,
                 per_100g=False, upper=None, T_transition=None, width=0.5):
        if kind not in ("polynomial", "vant_hoff", "table"):
            raise ValueError("kind must be 'polynomial', 'vant_hoff' or 'table'")
        scale = 0.01 if per_100g else 1.0
        self.kind = kind
        self.coeffs = None if coeffs is None else _f(coeffs) * (scale if kind == "polynomial" else 1.0)
        if kind == "vant_hoff" and coeffs is not None and per_100g:
            self.coeffs = _f(coeffs).at[0].add(math.log(0.01))
        self.T = None if T is None else _f(T)
        self.c = None if c is None else _f(c) * scale
        self.hydrate_waters = _f(hydrate_waters)
        self.solute_MW = _f(solute_MW)
        self.solvent_MW = _f(solvent_MW)
        self.upper = upper
        self.T_transition = None if T_transition is None else _f(T_transition)
        self.width = _f(width)
        if upper is not None and T_transition is None:
            raise ValueError("an upper form needs T_transition")
        if kind == "table" and (T is None or c is None):
            raise ValueError("a table solubility needs T and c")
        if kind != "table" and coeffs is None:
            raise ValueError(f"a {kind} solubility needs coeffs")
        if kind == "vant_hoff" and len(coeffs) != 2:
            raise ValueError("vant_hoff coeffs are [A, B]")

    # ------------------------------------------------------------ constructors

    @classmethod
    def polynomial(cls, coeffs, **kw) -> "Solubility":
        """``c_sat = sum_i a_i (T - 273.15)^i`` (T in K, ascending ``coeffs``)."""
        return cls("polynomial", coeffs=coeffs, **kw)

    @classmethod
    def vant_hoff(cls, A, B, **kw) -> "Solubility":
        """``ln c_sat = A + B / T`` (T in K)."""
        return cls("vant_hoff", coeffs=[A, B], **kw)

    @classmethod
    def table(cls, T, c, **kw) -> "Solubility":
        """Monotone PCHIP interpolant through ``(T, c)`` (T in K, increasing)."""
        return cls("table", T=T, c=c, **kw)

    # ----------------------------------------------------------------- pytree

    def tree_flatten(self):
        children = (self.coeffs, self.T, self.c, self.hydrate_waters, self.solute_MW,
                    self.solvent_MW, self.upper, self.T_transition, self.width)
        return children, self.kind

    @classmethod
    def tree_unflatten(cls, kind, children):
        obj = object.__new__(cls)
        (obj.coeffs, obj.T, obj.c, obj.hydrate_waters, obj.solute_MW, obj.solvent_MW,
         obj.upper, obj.T_transition, obj.width) = children
        obj.kind = kind
        return obj

    # ------------------------------------------------------------- evaluation

    def _own_c_sat(self, T) -> Array:
        T = _f(T)
        if self.kind == "polynomial":
            t = T - _T0C
            out = jnp.zeros_like(t)
            for a in self.coeffs[::-1]:
                out = out * t + a
        elif self.kind == "vant_hoff":
            out = jnp.exp(self.coeffs[0] + self.coeffs[1] / T)
        else:
            out = _pchip_eval(self.T, self.c, _pchip_slopes(self.T, self.c), T)
        return jnp.maximum(out, _C_FLOOR)

    def _weight(self, T) -> Array:
        """Weight of the ``upper`` form (0 without one)."""
        if self.upper is None:
            return jnp.zeros_like(_f(T))
        return jax.nn.sigmoid((_f(T) - self.T_transition) / self.width)

    def c_sat(self, T: Array | float) -> Array:
        """Saturation concentration (kg anhydrous solute / kg solvent) at ``T`` (K)."""
        lo = self._own_c_sat(T)
        if self.upper is None:
            return lo
        w = self._weight(T)
        return (1.0 - w) * lo + w * self.upper.c_sat(T)

    def hydration(self, T: Array | float) -> Array:
        """Water of crystallization per mole of solute in the stable solid at ``T`` (blended)."""
        if self.upper is None:
            return self.hydrate_waters + jnp.zeros_like(_f(T))
        w = self._weight(T)
        return (1.0 - w) * self.hydrate_waters + w * self.upper.hydration(T)

    def crystal_fractions(self, T: Array | float) -> tuple[Array, Array]:
        """Mass fractions ``(f_solute, f_water)`` of the crystal at ``T``."""
        nu = self.hydration(T)
        mw_w = nu * self.solvent_MW
        tot = self.solute_MW + mw_w
        return self.solute_MW / tot, mw_w / tot

    def per_100g(self, T: Array | float) -> Array:
        """``c_sat`` in g solute per 100 g solvent."""
        return 100.0 * self.c_sat(T)


def supersaturation(c, T, sol: Solubility, kind: Literal["ratio", "absolute", "relative"] = "ratio") -> Array:
    """Supersaturation of a solution of concentration ``c`` (kg solute / kg solvent) at ``T`` (K).

    Args:
        c: Solution concentration, kg anhydrous solute per kg solvent.
        T: Temperature (K).
        sol: The solubility curve.
        kind: ``"ratio"`` ``S = c/c_sat``, ``"absolute"`` ``dC = c - c_sat`` or
            ``"relative"`` ``sigma = (c - c_sat)/c_sat``.

    Returns:
        The supersaturation (negative ``dC`` / ``sigma`` below 1 in ratio form
        mean undersaturated).
    """
    cs = sol.c_sat(T)
    c = _f(c)
    if kind == "ratio":
        return c / cs
    if kind == "absolute":
        return c - cs
    if kind == "relative":
        return (c - cs) / cs
    raise ValueError("kind must be 'ratio', 'absolute' or 'relative'")


# =============================================================================
# Yield balance
# =============================================================================


def _soft_pos(x, s):
    return s * jax.nn.softplus(x / s)


def crystal_balance(feed_solute, feed_water, T_out, sol: Solubility,
                    water_evaporated=0.0, soft: float = 1e-6, c_out=None) -> dict:
    """Solute and water balances of an equilibrium crystallizer.

    Args:
        feed_solute: Anhydrous solute in the feed (kg/s).
        feed_water: Solvent in the feed (kg/s).
        T_out: Outlet temperature (K), where the liquor is saturated.
        sol: Solubility curve with the hydrate form.
        water_evaporated: Solvent removed as vapor (kg/s).
        soft: Soft-clip width as a fraction of ``feed_solute``.
        c_out: Liquor concentration (kg/kg) to use instead of ``c_sat(T_out)``
            (kinetic mode passes ``c_sat + dC``).

    Returns:
        dict of kg/s: ``crystals`` (hydrate), ``crystal_solute``,
        ``crystal_water``, ``liquor_solute``, ``liquor_water``, ``c_sat``,
        ``c_liquor``, ``water`` (feed water after evaporation),
        ``yield_fraction`` (crystal solute / feed solute).
    """
    S0, W0, E = _f(feed_solute), _f(feed_water), _f(water_evaporated)
    W = W0 - E
    c = sol.c_sat(T_out) if c_out is None else _f(c_out)
    f_s, f_w = sol.crystal_fractions(T_out)
    # soft-clip the NUMERATOR (the solute above saturation); dividing by a negative
    # denominator (c_sat above the hydrate stoichiometry) would turn an undersaturated
    # feed into spurious crystals
    excess = _soft_pos(S0 - c * W, soft * S0 + _TINY)
    C = excess / jnp.maximum(f_s - c * f_w, _C_FLOOR)
    crystal_solute = f_s * C
    crystal_water = f_w * C
    liquor_water = W - crystal_water
    liquor_solute = S0 - crystal_solute
    return {
        "crystals": C, "crystal_solute": crystal_solute, "crystal_water": crystal_water,
        "liquor_solute": liquor_solute, "liquor_water": liquor_water,
        "c_sat": sol.c_sat(T_out), "c_liquor": c, "water": W,
        "yield_fraction": crystal_solute / S0,
    }


def crystal_yield(feed_solute, feed_water, T_out, sol: Solubility, water_evaporated=0.0,
                  soft: float = 1e-6) -> Array:
    """Crystal (hydrate) mass flow (kg/s) from the solubility balance.

    ``C = (S0 - c_sat W) / (f_s - c_sat f_w)`` with ``W = W0 - E``, soft-clipped
    at zero (see the module docstring); see :func:`crystal_balance` for the
    full set of flows.

    Example:
        >>> sol = Solubility.polynomial([0.20, 0.0], hydrate_waters=0, solute_MW=100.0)
        >>> round(float(crystal_yield(0.30, 1.0, 300.0, sol)), 6)
        0.1
    """
    return crystal_balance(feed_solute, feed_water, T_out, sol, water_evaporated, soft)["crystals"]


# =============================================================================
# MSMPR
# =============================================================================


def msmpr_distribution(L, n0, G, tau) -> Array:
    """Steady MSMPR population density ``n(L) = n0 exp(-L / (G tau))`` (1/m/m^3)."""
    return _f(n0) * jnp.exp(-_f(L) / (_f(G) * _f(tau)))


def msmpr_moments(n0, G, tau, k=(0, 1, 2, 3)) -> Array:
    """Moments ``m_k = k! n0 (G tau)^(k+1)`` of the MSMPR distribution, one per entry of ``k``."""
    k = _f(k)
    Gt = _f(G) * _f(tau)
    return jnp.exp(gammaln(k + 1.0)) * _f(n0) * Gt ** (k + 1.0)


def msmpr_dominant_size(G, tau) -> Array:
    """Dominant (mass-mode) size ``L_D = 3 G tau``, the maximum of ``L^3 n(L)``."""
    return 3.0 * _f(G) * _f(tau)


@functools.lru_cache(maxsize=1)
def _gamma4_median() -> float:
    """Root of ``1 - exp(-x)(1 + x + x^2/2 + x^3/6) = 1/2`` (Newton, plain floats)."""
    x = 3.5
    for _ in range(100):
        F = 1.0 - math.exp(-x) * (1.0 + x + x * x / 2.0 + x ** 3 / 6.0) - 0.5
        dF = x ** 3 * math.exp(-x) / 6.0
        step = F / dF
        x -= step
        if abs(step) < 1e-15:
            break
    return x


def msmpr_mass_median_size(G, tau) -> Array:
    """Mass median size ``L_50 = x_50 G tau`` with ``x_50 = 3.672...`` computed from the gamma(4) law."""
    return _gamma4_median() * _f(G) * _f(tau)


def msmpr_mass_cv() -> float:
    """Coefficient of variation of the MSMPR mass distribution, ``1/sqrt(4) = 0.5``."""
    return 1.0 / math.sqrt(4.0)


def msmpr_from_magma(M_T, rho_c, k_v, G, tau) -> Array:
    """Nuclei density ``n0 = M_T / (6 rho_c k_v (G tau)^4)`` from the suspension density ``M_T`` (kg/m^3)."""
    return _f(M_T) / (6.0 * _f(rho_c) * _f(k_v) * (_f(G) * _f(tau)) ** 4)


def fit_msmpr(L, n, weights=None) -> tuple[Array, Array]:
    """Fit ``ln n = ln n0 - L/(G tau)`` to a measured distribution.

    Args:
        L: Sizes (m).
        n: Population densities (positive).
        weights: Optional least-squares weights.

    Returns:
        ``(G_tau, n0)``: the characteristic length ``G tau`` (m) and ``n0``.
    """
    L, y = _f(L), jnp.log(_f(n))
    w = jnp.ones_like(L) if weights is None else _f(weights)
    sw = jnp.sum(w)
    Lm, ym = jnp.sum(w * L) / sw, jnp.sum(w * y) / sw
    slope = jnp.sum(w * (L - Lm) * (y - ym)) / jnp.sum(w * (L - Lm) ** 2)
    intercept = ym - slope * Lm
    return -1.0 / slope, jnp.exp(intercept)


def sieve_population_density(mass_fraction, L_upper, L_lower, M_T, crystal_density,
                             shape_factor=math.pi / 6.0) -> tuple[Array, Array]:
    """Population density ``n(L)`` from a screen (sieve) analysis of the product crystals.

    The crystals retained between two screens, of mass fraction ``w_i`` and mean
    size ``Lbar_i = (L_upper + L_lower)/2``, number ``M_T w_i / (rho_c k_v Lbar_i^3)``
    per m^3; dividing by the size interval ``dL_i = L_upper - L_lower`` gives the
    population density (McCabe, Smith & Harriott, Ch. 27; Randolph & Larson)::

        n_i = M_T w_i / (rho_c k_v Lbar_i^3 dL_i)

    Args:
        mass_fraction: Mass fraction of the crystals on each screen (need not sum to 1
            if the pan or the oversize is left out).
        L_upper: Aperture of the screen above each cut, i.e. the upper size (m).
        L_lower: Aperture of the screen that retains the cut, the lower size (m).
        M_T: Magma (suspension) density, kg crystal per m^3 (the basis of ``n``).
        crystal_density: Crystal density (kg/m^3).
        shape_factor: Volume shape factor ``k_v`` (pi/6 for spheres, 1 for cubes).

    Returns:
        ``(L_mean, n)``: the mean size of each cut (m) and its population
        density (number per m of size per m^3).

    Example:
        >>> L, n = sieve_population_density([0.5, 0.5], [2e-3, 1e-3], [1e-3, 0.5e-3], 100.0, 1600.0)
        >>> bool(n[1] > n[0])
        True
    """
    w, Lu, Ll = _f(mass_fraction), _f(L_upper), _f(L_lower)
    Lbar = 0.5 * (Lu + Ll)
    n = _f(M_T) * w / (_f(crystal_density) * _f(shape_factor) * Lbar ** 3 * (Lu - Ll))
    return Lbar, n


# =============================================================================
# Kinetics and the steady MSMPR
# =============================================================================


def _pos_pow(x, p):
    pos = x > 0.0
    safe = jnp.where(pos, x, 1.0)
    return jnp.where(pos, safe ** p, 0.0)


def growth_rate(delta_C, k_g, g) -> Array:
    """Growth rate ``G = k_g dC^g`` (m/s); zero for ``dC <= 0``."""
    return _f(k_g) * _pos_pow(_f(delta_C), _f(g))


def nucleation_rate(delta_C, M_T, k_b, b, j) -> Array:
    """Secondary nucleation ``B0 = k_b M_T^j dC^b`` (1/m^3/s); zero for ``dC <= 0``."""
    return _f(k_b) * _pos_pow(_f(M_T), _f(j)) * _pos_pow(_f(delta_C), _f(b))


def msmpr_steady_state(feed_solute, feed_water, T_out, sol: Solubility, kinetics, tau, *,
                       crystal_density, shape_factor=math.pi / 6.0, liquor_density=1000.0,
                       water_evaporated=0.0, soft: float = 1e-6) -> dict:
    """Steady MSMPR closed by power-law kinetics: solve the supersaturation ``dC``.

    Unknown ``dC = c - c_sat`` (kg/kg) between 0 and the feed's ``dC_max``. The
    solute balance gives the hydrate flow ``C(dC)`` and ``M_T = C / Q_L`` (``Q_L``
    the liquor volume flow, from ``liquor_density``); the third moment requires
    ``M_T = 6 rho_c k_v k_b M_T^j dC^b (k_g dC^g)^3 tau^4``.

    Args:
        feed_solute: Anhydrous solute (kg/s).
        feed_water: Solvent (kg/s).
        T_out: Operating temperature (K).
        sol: Solubility curve with hydrate form.
        kinetics: ``(k_g, g, k_b, b, j)`` with ``j <= 1``.
        tau: Residence time (s).
        crystal_density: Crystal (hydrate) density (kg/m^3).
        shape_factor: Volume shape factor ``k_v`` (default pi/6, spheres).
        liquor_density: Mother-liquor density (kg/m^3).
        water_evaporated: Solvent evaporated (kg/s).
        soft: Soft-clip width of the yield balance.

    Returns:
        dict with ``delta_C``, ``G``, ``B0``, ``n0``, ``M_T`` (kg/m^3), the
        balance (``crystals`` etc., see :func:`crystal_balance`), ``L_D``,
        ``L_50``, ``cv``, ``moments`` (m0..m3), ``active`` (feed is
        supersaturated) and ``closure`` (relative mismatch of the third-moment
        magma density and the balance's, a convergence check).
    """
    try:  # check the raw value: jnp.asarray of a constant is a tracer under jit
        too_big = float(kinetics[4]) > 1.0
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        too_big = False  # traced: cannot check
    k_g, g, k_b, b, j = (_f(v) for v in kinetics)
    if too_big:
        raise ValueError("msmpr_steady_state needs j <= 1 (a single steady state)")
    S0, W0, E = _f(feed_solute), _f(feed_water), _f(water_evaporated)
    tau, rho_c, k_v, rho_L = _f(tau), _f(crystal_density), _f(shape_factor), _f(liquor_density)
    W = W0 - E
    cs = sol.c_sat(T_out)
    f_s, f_w = sol.crystal_fractions(T_out)
    dmax_raw = S0 / W - cs
    active = dmax_raw > 0.0
    dmax = jnp.where(active, dmax_raw, 1.0)
    const = jnp.log(6.0 * rho_c * k_v * k_b) + 4.0 * jnp.log(tau) + 3.0 * jnp.log(k_g)

    def parts(x):
        dC = dmax * jnp.exp(x)
        c = cs + dC
        C = (S0 - c * W) / (f_s - c * f_w)
        Wl = W - f_w * C
        QL = (Wl + c * Wl) / rho_L
        M_T = jnp.maximum(C / QL, _TINY)
        return dC, C, M_T

    def resid(x):
        dC, _, M_T = parts(x)
        return (1.0 - j) * jnp.log(M_T) - (b + 3.0 * g) * jnp.log(dC) - const

    x = _solve_monotone(resid, -60.0, -1e-9)
    dC, C, M_T = parts(x)
    dC = jnp.where(active, dC, 0.0)
    G = growth_rate(dC, k_g, g)
    B0 = nucleation_rate(dC, M_T, k_b, b, j)
    Gs = jnp.where(active, G, 1.0)
    n0 = jnp.where(active, B0 / Gs, 0.0)
    bal = crystal_balance(S0, W0, T_out, sol, E, soft, c_out=cs + dC)
    bal["crystals"] = jnp.where(active, bal["crystals"], 0.0 * bal["crystals"])
    mom = msmpr_moments(n0, Gs, tau)
    M_T_kin = rho_c * k_v * mom[3]
    closure = jnp.where(active, (M_T_kin - M_T) / jnp.maximum(M_T, _TINY), 0.0)
    out = dict(bal)
    out.update({
        "delta_C": dC, "G": G, "B0": B0, "n0": n0, "M_T": jnp.where(active, M_T, 0.0),
        "L_D": msmpr_dominant_size(G, tau), "L_50": msmpr_mass_median_size(G, tau),
        "cv": msmpr_mass_cv(), "moments": mom, "active": active, "closure": closure,
        "tau": tau,
    })
    return out


# =============================================================================
# The unit
# =============================================================================


@dataclass
class CrystallizerParams(ParamsMixin):
    """Parameters of a crystallizer.

    Attributes:
        solute: Solute species name (stream key ``F_<solute>``), the anhydrous formula unit.
        solute_MW: Molar mass of the anhydrous solute (g/mol).
        mode: ``"cooling"`` (cool to ``T``, no evaporation), ``"evaporative"``
            (evaporate at constant temperature: ``T`` defaults to the feed's)
            or ``"cooling_evaporative"`` (both), or ``"adiabatic"`` (a vacuum
            crystallizer: the feed flashes to ``T`` and the solvent evaporated is
            solved from the heat balance with ``Q = duty``, 0 by default).
        T: Outlet temperature (K).
        water_evaporated: Solvent evaporated (kg/s).
        evaporation_fraction: Fraction of the feed solvent evaporated, instead of ``water_evaporated``.
        solvent: Solvent species name.
        solvent_MW: Molar mass of the solvent (g/mol).
        solubility_model: ``"polynomial"`` (in degrees C), ``"vant_hoff"`` (``ln c = A + B/T``) or ``"table"``.
        solubility_coeffs: Polynomial coefficients (ascending) or ``[A, B]``.
        solubility_T: Table temperatures (K).
        solubility_c: Table solubilities.
        solubility_basis: ``"kg_per_kg"`` (kg solute per kg solvent) or ``"g_per_100g"``.
        hydrate_waters: Moles of water of crystallization per mole of solute (0 = anhydrous).
        upper_solubility_model: Model of the second solid form stable above ``transition_T``.
        upper_solubility_coeffs: Coefficients of the second form.
        upper_solubility_T: Table temperatures of the second form (K).
        upper_solubility_c: Table solubilities of the second form.
        upper_hydrate_waters: Hydrate waters of the second form.
        transition_T: Transition temperature (K) between the forms.
        transition_width: Width (K) of the smooth blend between the forms.
        kinetics: ``None`` for equilibrium yield, or ``[k_g, g, k_b, b, j]`` for the MSMPR.
        residence_time: Residence time ``tau`` (s), kinetic mode.
        volume: Crystallizer volume (m^3); if no ``residence_time``, ``tau = volume / Q_P`` with
            ``Q_P = (m_F - E) / liquor_density`` the volume flow of the product slurry (the feed
            minus the solvent evaporated), as in the MSMPR definition of ``tau``.
        crystal_density: Crystal (hydrate) density (kg/m^3), kinetic mode.
        shape_factor: Volume shape factor ``k_v``.
        liquor_density: Mother-liquor density (kg/m^3).
        cp_solution: Solution heat capacity (J/kg/K) for the sensible duty.
        heat_of_crystallization: Heat released per kg of crystal formed (J/kg).
        soft: Soft-clip width of the yield, as a fraction of the feed solute.
        duty: Heat added (W) in ``"adiabatic"`` mode (0 = a truly adiabatic flash;
            negative removes heat through a jacket or cooler as well).
    """

    solute: str
    solute_MW: float
    mode: Literal["cooling", "evaporative", "cooling_evaporative", "adiabatic"] = "cooling"
    T: float | None = None
    water_evaporated: float | None = None
    evaporation_fraction: float | None = None
    solvent: str = "H2O"
    solvent_MW: float = 18.015
    solubility_model: Literal["polynomial", "vant_hoff", "table"] = "polynomial"
    solubility_coeffs: list | None = None
    solubility_T: list | None = None
    solubility_c: list | None = None
    solubility_basis: Literal["kg_per_kg", "g_per_100g"] = "kg_per_kg"
    hydrate_waters: float = 0.0
    upper_solubility_model: Literal["polynomial", "vant_hoff", "table"] = "polynomial"
    upper_solubility_coeffs: list | None = None
    upper_solubility_T: list | None = None
    upper_solubility_c: list | None = None
    upper_hydrate_waters: float = 0.0
    transition_T: float | None = None
    transition_width: float = 0.5
    kinetics: list | None = None
    residence_time: float | None = None
    volume: float | None = None
    crystal_density: float = 2000.0
    shape_factor: float = math.pi / 6.0
    liquor_density: float = 1000.0
    cp_solution: float = 4186.0
    heat_of_crystallization: float = 0.0
    soft: float = 1e-6
    duty: float = 0.0

    def __post_init__(self):
        if self.mode not in ("cooling", "evaporative", "cooling_evaporative", "adiabatic"):
            raise ValueError("mode must be 'cooling', 'evaporative', 'cooling_evaporative' or 'adiabatic'")
        evap = self.water_evaporated is not None or self.evaporation_fraction is not None
        if self.water_evaporated is not None and self.evaporation_fraction is not None:
            raise ValueError("give water_evaporated or evaporation_fraction, not both")
        if self.mode == "cooling" and evap:
            raise ValueError("mode 'cooling' has no evaporation; use 'cooling_evaporative'")
        if self.mode == "adiabatic" and evap:
            raise ValueError("mode 'adiabatic' solves the evaporation from the heat balance; "
                             "do not give water_evaporated or evaporation_fraction")
        if self.mode not in ("cooling", "adiabatic") and not evap:
            raise ValueError(f"mode {self.mode!r} needs water_evaporated or evaporation_fraction")
        if self.mode != "evaporative" and self.T is None:
            raise ValueError(f"mode {self.mode!r} needs T")
        if self.kinetics is not None:
            if len(self.kinetics) != 5:
                raise ValueError("kinetics is [k_g, g, k_b, b, j]")
            try:  # concrete only: under jit/grad even indexing a constant array is traced
                too_big = float(self.kinetics[4]) > 1.0
            except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
                too_big = False
            if too_big:
                raise ValueError("kinetics j must be <= 1")
            if self.residence_time is None and self.volume is None:
                raise ValueError("kinetic mode needs residence_time or volume")
        if (self.upper_solubility_coeffs is not None or self.upper_solubility_T is not None) \
                and self.transition_T is None:
            raise ValueError("an upper solubility form needs transition_T")


def _adiabatic_evaporation(S0, W0, T_in, T, sol: Solubility, p: CrystallizerParams) -> Array:
    """Solvent flashed off (kg/s) so that the duty equals ``p.duty``.

    ``duty = m_F cp (T - T_F) + E h_fg(T) - dH_c C(E)`` with the equilibrium hydrate
    flow ``C(E) = (S0 - c (W0 - E)) / (f_s - c f_w)``, linear in ``E``; solved in
    closed form. The crystal term uses the unclipped ``C``: it is exact whenever
    crystals form, and with ``dH_c = 0`` (the default) exact always. ``E`` is floored
    at zero (a feed too cold to flash at ``T`` evaporates nothing).
    """
    c = sol.c_sat(T)
    f_s, f_w = sol.crystal_fractions(T)
    den = jnp.maximum(f_s - c * f_w, _C_FLOOR)
    dH = _f(p.heat_of_crystallization)
    m_F = S0 + W0
    num = _f(p.duty) - m_F * p.cp_solution * (T - T_in) + dH * (S0 - c * W0) / den
    E = num / (water_latent_heat(T) - dH * c / den)
    return _soft_pos(E, 1e-9 * m_F + _TINY)


def _make_form(model, coeffs, T, c, **kw):
    if model == "table":
        return Solubility.table(T, c, **kw)
    if model == "vant_hoff":
        return Solubility.vant_hoff(coeffs[0], coeffs[1], **kw)
    return Solubility.polynomial(coeffs, **kw)


class Crystallizer:
    """Solubility-driven crystallizer: cooling, evaporative or both; equilibrium or MSMPR kinetics.

    ``Crystallizer(params)(feed) -> (crystals, mother_liquor, info)``. The feed
    is a stream with ``F_<solute>`` and ``F_<solvent>`` (mol/s); an ``Evaporator``
    concentrate goes straight in (set ``solvent`` to the evaporator's solvent
    name). The crystal stream carries the hydrate as solute plus bound water
    (see the module docstring); the mother liquor is saturated at ``T``
    (equilibrium) or at ``c_sat + dC`` (kinetic mode).

    ``info``: ``yield_kg_s`` (solute in the crystals), ``yield_fraction``,
    ``crystal_mass`` (hydrate, kg/s), ``c_sat``, ``c_liquor``,
    ``supersaturation`` (ratio), ``delta_C``, ``water_evaporated`` (kg/s), the
    duty ``Q`` (W; also ``Q_sensible``, ``Q_evaporation``,
    ``Q_crystallization``) and, in kinetic mode, ``G``, ``B0``, ``n0``,
    ``moments``, ``L_D``, ``L_50``, ``cv``, ``M_T`` and ``tau``.
    """

    symbol = "Crystallizer"
    equations = [
        r"S_0 = c\,W_L + f_s C,\qquad W_0 - E = W_L + f_w C",
        r"C = \frac{S_0 - c_{sat}(T)\,(W_0 - E)}{f_s - c_{sat}(T)\,f_w}",
        r"n(L) = n_0 e^{-L/(G\tau)},\qquad m_k = k!\,n_0\,(G\tau)^{k+1}",
        r"L_D = 3 G \tau,\qquad M_T = 6\rho_c k_v n_0 (G\tau)^4",
        r"G = k_g\,\Delta C^{g},\qquad B_0 = k_b\,M_T^{\,j}\,\Delta C^{b}",
    ]
    assumptions = [
        "Equilibrium mode: the mother liquor leaves saturated at the outlet temperature; the crystal is a stoichiometric hydrate (or anhydrous).",
        "The crystal stream carries the solute plus its bound water as F_<solute> and F_<solvent>; evaporated solvent leaves as vapor (reported in info, not a stream).",
        "Undersaturated feeds give zero yield through a soft clip; the switch between two solid forms is a logistic blend in T.",
        "Kinetic mode: steady MSMPR, size-independent growth, power-law growth and secondary nucleation, no fines destruction or classification, no agglomeration or breakage.",
        "Liquor volume flow from a constant liquor density; solution heat capacity constant; heat of crystallization is user data.",
        "No solubility, kinetic or heat data is built in; the user supplies the curves.",
    ]
    references = [
        "Randolph, A.D. and Larson, M.A., Theory of Particulate Processes, 2nd ed. (1988).",
        "Mullin, J.W., Crystallization, 4th ed. (2001).",
        "Geankoplis, C.J., Transport Processes and Separation Process Principles, Sec. 12.11.",
        "McCabe, Smith and Harriott, Unit Operations of Chemical Engineering, Ch. 27.",
    ]
    parameter_symbols = {
        "T": "T", "water_evaporated": "E", "hydrate_waters": r"\nu", "residence_time": r"\tau",
        "crystal_density": r"\rho_c", "shape_factor": "k_v", "cp_solution": "c_p",
        "heat_of_crystallization": r"\Delta H_c", "transition_T": r"T_{tr}",
    }
    parameter_units = {
        "solute_MW": "g/mol", "solvent_MW": "g/mol", "T": "K", "water_evaporated": "kg/s",
        "evaporation_fraction": "-", "solubility_T": "K", "solubility_c": "kg/kg or g/100g",
        "hydrate_waters": "mol/mol", "upper_solubility_T": "K", "upper_solubility_c": "kg/kg or g/100g",
        "upper_hydrate_waters": "mol/mol", "transition_T": "K", "transition_width": "K",
        "residence_time": "s", "volume": "m^3", "crystal_density": "kg/m^3", "shape_factor": "-",
        "liquor_density": "kg/m^3", "cp_solution": "J/(kg K)",
        "heat_of_crystallization": "J/kg", "soft": "-", "duty": "W",
    }
    numerical_method = (
        "Closed-form two-balance yield with a softplus clip; adiabatic (vacuum) evaporation in closed "
        "form from the linear heat balance; kinetic steady state by bisection in "
        "ln(dC/dC_max) with an implicit-gradient Newton step."
    )
    outlet_roles = ["crystals", "mother_liquor"]

    def __init__(self, params: CrystallizerParams):
        self.params = params

    # ------------------------------------------------------------------ helpers

    @property
    def solubility(self) -> Solubility:
        """The :class:`Solubility` described by the parameters."""
        p = self.params
        kw = dict(solute_MW=p.solute_MW, solvent_MW=p.solvent_MW,
                  per_100g=p.solubility_basis == "g_per_100g")
        upper = None
        if p.transition_T is not None:
            upper = _make_form(p.upper_solubility_model, p.upper_solubility_coeffs,
                               p.upper_solubility_T, p.upper_solubility_c,
                               hydrate_waters=p.upper_hydrate_waters, **kw)
        return _make_form(p.solubility_model, p.solubility_coeffs, p.solubility_T, p.solubility_c,
                          hydrate_waters=p.hydrate_waters, upper=upper,
                          T_transition=p.transition_T, width=p.transition_width, **kw)

    # --------------------------------------------------------------------- call

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Crystallize the feed.

        Args:
            feed: Stream with ``F_<solute>``, ``F_<solvent>`` (mol/s), ``T``, ``P``
                (other species stay in the mother liquor).

        Returns:
            crystals: hydrate as solute plus bound water, at the outlet ``T``.
            mother_liquor: the saturated (or kinetically supersaturated) liquor.
            info: see the class docstring.
        """
        p = self.params
        ks, kw_ = f"F_{p.solute}", f"F_{p.solvent}"
        sol = self.solubility
        S0 = _f(feed[ks]) * p.solute_MW / 1000.0
        W0 = _f(feed[kw_]) * p.solvent_MW / 1000.0
        T_in = _f(feed["T"])
        T = T_in if p.T is None else _f(p.T)
        if p.mode == "adiabatic":
            E = _adiabatic_evaporation(S0, W0, T_in, T, sol, p)
        elif p.water_evaporated is not None:
            E = _f(p.water_evaporated)
        elif p.evaporation_fraction is not None:
            E = _f(p.evaporation_fraction) * W0
        else:
            E = _f(0.0)

        if p.kinetics is None:
            bal = crystal_balance(S0, W0, T, sol, E, p.soft)
            extra = {}
        else:
            if p.residence_time is not None:
                tau = _f(p.residence_time)
            else:
                # MSMPR tau is the volume over the PRODUCT (slurry) flow; the evaporated
                # solvent leaves as vapor, so an evaporative crystallizer has a longer tau
                # than the feed flow suggests
                Q_product = (S0 + W0 - E) / _f(p.liquor_density)
                tau = _f(p.volume) / Q_product
            ss = msmpr_steady_state(S0, W0, T, sol, p.kinetics, tau, crystal_density=p.crystal_density,
                                    shape_factor=p.shape_factor, liquor_density=p.liquor_density,
                                    water_evaporated=E, soft=p.soft)
            bal = ss
            extra = {k: ss[k] for k in ("G", "B0", "n0", "moments", "L_D", "L_50", "cv", "M_T",
                                         "tau", "closure")}
            extra["delta_C"] = ss["delta_C"]

        nu = sol.hydration(T)
        mol_solute = bal["crystal_solute"] * 1000.0 / p.solute_MW
        crystals = {k: 0.0 * _f(v) for k, v in feed.items() if k.startswith("F_")}
        liquor = {k: _f(v) for k, v in feed.items() if k.startswith("F_")}
        crystals[ks] = mol_solute
        crystals[kw_] = nu * mol_solute
        liquor[ks] = _f(feed[ks]) - mol_solute
        liquor[kw_] = _f(feed[kw_]) - E * 1000.0 / p.solvent_MW - nu * mol_solute
        for out in (crystals, liquor):
            out["T"], out["P"] = T, _f(feed["P"])

        m_F = S0 + W0
        C = bal["crystals"]
        Q_sens = m_F * p.cp_solution * (T - T_in)
        Q_evap = E * water_latent_heat(T)
        Q_cryst = -p.heat_of_crystallization * C
        info = {
            "yield_kg_s": bal["crystal_solute"], "yield_fraction": bal["yield_fraction"],
            "crystal_mass": C, "crystal_solute": bal["crystal_solute"],
            "crystal_water": bal["crystal_water"], "hydration": nu,
            "c_sat": bal["c_sat"], "c_liquor": bal["c_liquor"],
            "supersaturation": bal["c_liquor"] / bal["c_sat"],
            "delta_C": bal["c_liquor"] - bal["c_sat"], "water_evaporated": E,
            "liquor_solute": bal["liquor_solute"], "liquor_water": bal["liquor_water"],
            "T": T, "Q": Q_sens + Q_evap + Q_cryst, "Q_sensible": Q_sens,
            "Q_evaporation": Q_evap, "Q_crystallization": Q_cryst,
            "solute_balance_error": S0 - bal["crystal_solute"] - bal["liquor_solute"],
            "water_balance_error": W0 - E - bal["crystal_water"] - bal["liquor_water"],
        }
        info.update(extra)
        info["delta_C"] = bal["c_liquor"] - bal["c_sat"]
        return crystals, liquor, info
