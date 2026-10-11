"""Batch distillation: Rayleigh, constant-reflux and constant-composition operation.

A batch of ``F`` mol with light-key fraction ``x_F`` is charged to a still. Vapor
is boiled off at a constant rate ``V`` (mol/s), passes through ``N`` equilibrium
stages above the still, and a total condenser returns reflux and draws
distillate ``D = V/(R+1)``. The model is quasi-steady with no stage or condenser
holdup, constant molar overflow (straight operating lines) and a binary
equilibrium curve ``y*(x)``.

Equations
---------
With ``W`` the still holdup and ``x_W`` its light-key fraction::

    dW/dt = -D                      d(W x_W)/dt = -D x_D
    =>  dln W / d x_W = 1 / (x_D - x_W)

``x_D`` is whatever the column delivers for the still composition ``x_W``:
the stage relation (``N`` steps of ``y_{n+1} = (R x_n + x_D)/(R+1)``,
``x_n = x*(y_n)`` from ``y_1 = x_D``, then the still ``x_W = x*(y_{N+1})``)
inverted for ``x_D``. With ``N = 0`` it is ``x_D = y*(x_W)`` and the integral is
the Rayleigh equation ``ln(F/W) = int_{x_W}^{x_F} dx/(y* - x)``.

* Constant reflux: ``D`` is constant, so ``t = (F - W)(R+1)/V`` and the only
  integral is ``ln(F/W)``, evaluated by Gauss-Legendre quadrature in
  ``s = ln(x/x_F)/ln(x_W/x_F)`` (the integrand ``x/(x_D-x)`` is smooth there
  even as ``x_W`` goes to zero).
* Constant composition: ``x_D`` is fixed, so ``W/F = (x_D - x_F)/(x_D - x_W)``
  exactly, ``R(x_W)`` is the root of the stage relation, and
  ``t = (1/V) int_W^F (R+1) dW'``.

Every stopping criterion is a monotone function of the final still composition
``x_W``; the unit finds it by a bisection on stop-gradient values followed by a
Newton step, so derivatives follow the implicit function theorem
(``dx_W*/d theta = -(dg/d theta)/(dg/d x_W)``, the same statement as
``dt*/d theta = -(dg/d theta)/(dg/dt)``). No Python branch looks at a traced
value; infeasible operation (a purity above the azeotrope, a purity the stages
cannot reach, an unreachable stop) is reported in ``info`` flags and the
numbers stay finite.

Stream convention
-----------------
``charge`` is a stream read as **moles** (``F_i`` in mol, not mol/s), as the
initial contents of a batch vessel, exactly as ``FedBatchReactor`` reads its
initial state. The returned ``still`` and ``distillate`` streams are the final
holdup and the accumulated product, again in moles.

Limits
------
Binary mixtures (one equilibrium curve); no stage or condenser holdup
(``holdup`` must be 0); constant molar overflow; total condenser; the key
component must be the more volatile one (otherwise ``info["feasible"]`` is
False). Multicomponent simple (Rayleigh) distillation is provided separately by
:func:`rayleigh_multicomponent`; a multicomponent tray-to-tray column march is
not implemented.

References
----------
- Rayleigh, Phil. Mag. 42 (1896) 493 (differential distillation).
- Wankat, Separation Process Engineering, Ch. 9; McCabe, Smith & Harriott,
  Ch. 21; Geankoplis, Sec. 11.3 (equations only; no textbook numerical
  example is reproduced -- see ``tests/test_batch_distillation.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from difflow.cache_key import ValueKeyed
from difflow.params_mixin import ParamsMixin
from difflow.phase_diagrams import _gamma, _psat, bubble_T, xy_curve
from difflow.staged import EquilibriumCurve, _solve_monotone, tabulated
from difflow.streams import Stream, get_flows

__all__ = [
    "BatchDistillation",
    "BatchDistillationParams",
    "batch_cuts",
    "equilibrium_curve",
    "rayleigh",
    "rayleigh_multicomponent",
]

_EPS = 1e-12
#: a distillate within this of the azeotrope composition is reported as limited by it
AZEOTROPE_TOL = 1e-2

_gl_x, _gl_w = np.polynomial.legendre.leggauss(64)
_GL_S = (_gl_x + 1.0) / 2.0   # nodes on (0, 1)
_GL_W = _gl_w / 2.0           # weights summing to 1


# ---------------------------------------------------------------------------
# Rayleigh (simple differential distillation)
# ---------------------------------------------------------------------------


def equilibrium_curve(thermo, species, P: float, activity_model=None,
                      n: int = 101) -> EquilibriumCurve:
    """Tabulated ``y*(x)`` of the first species of a binary at constant ``P``.

    A thin wrapper on :func:`difflow.phase_diagrams.xy_curve` and
    :func:`difflow.staged.tabulated` (monotone PCHIP, so the curve and its
    inverse are C1). ``x`` and ``y`` are mole fractions of ``species[0]``.
    """
    d = xy_curve(thermo, list(species), P, n, activity_model)
    return tabulated(d["x"], d["y"])


def _alpha_ln_ratio(alpha, x_F, x_W):
    """Closed form ``ln(F/W)`` for constant relative volatility."""
    return (jnp.log(x_F / x_W) + alpha * jnp.log((1.0 - x_W) / (1.0 - x_F))) / (alpha - 1.0)


def _geometric_quadrature(f, x_F, x_W):
    """``int_{x_W}^{x_F} f(x) dx`` by 64-point Gauss-Legendre in ``ln x``.

    Returns the integral and the node values of ``f`` (for validity checks).
    """
    r = x_W / x_F
    x = x_F * r ** _GL_S
    fx = jax.vmap(f)(x)
    return jnp.log(x_F / x_W) * jnp.sum(_GL_W * x * fx), fx


def _rayleigh_ln_ratio(curve, x_F, x_W):
    """``(ln(F/W), ok)`` for a still going from ``x_F`` to ``x_W``.

    ``ok`` is False when ``y* - x`` changes sign or vanishes between the two
    compositions (an azeotrope or a pinch): the integral diverges there and
    ``ln(F/W)`` is returned as ``inf``.
    """
    r = x_W / x_F
    xs = x_F * r ** _GL_S
    d = curve(xs) - xs
    sgn = jnp.sign(d[0])
    ok = jnp.all(d * sgn > _EPS) & (sgn != 0.0)
    if curve.kind == "alpha":
        val = _alpha_ln_ratio(curve.data["alpha"], x_F, x_W)
    else:
        dd = jnp.where(ok, d, 1.0)
        val = jnp.log(x_F / x_W) * jnp.sum(_GL_W * xs / dd)
    return jnp.where(ok, val, jnp.inf), ok


def rayleigh(curve: EquilibriumCurve, x_F: Array | float, W_over_F: Array | float | None = None,
             x_W: Array | float | None = None) -> dict:
    """Rayleigh equation ``ln(F/W) = int_{x_W}^{x_F} dx/(y* - x)``.

    Give exactly one of ``W_over_F`` (fraction of the charge left in the still)
    or ``x_W`` (final still composition); the other is solved for. Constant
    relative volatility (``constant_alpha``) uses the closed form
    ``ln(F/W) = [ln(x_F/x_W) + alpha ln((1-x_W)/(1-x_F))]/(alpha - 1)``;
    any other :class:`~difflow.staged.EquilibriumCurve` (e.g. from
    :func:`equilibrium_curve`) uses 64-point Gauss-Legendre quadrature in
    ``ln x``. Differentiable in every input, ``x_W`` given ``W_over_F`` by the
    implicit function theorem.

    Args:
        curve: Equilibrium curve ``y*(x)`` of the light key.
        x_F: Charge composition (mole fraction of the key).
        W_over_F: ``W/F`` in ``(0, 1)``, or ``None``.
        x_W: Final still composition, or ``None``.

    Returns:
        dict with ``x_W``, ``W_over_F``, ``ln_F_over_W``, ``x_D_avg`` (the
        accumulated distillate composition ``(F x_F - W x_W)/(F - W)``) and
        ``ok``. ``ok`` is False when ``y* - x`` changes sign between ``x_W``
        and ``x_F`` (an azeotrope): ``ln(F/W)`` is then ``inf`` and
        ``W_over_F`` is 0, not NaN.

    Example:
        >>> from difflow.staged import constant_alpha
        >>> r = rayleigh(constant_alpha(2.5), 0.5, W_over_F=0.4)
        >>> bool(r["ok"]), round(float(r["x_W"]), 3)
        (True, 0.3)
    """
    if (W_over_F is None) == (x_W is None):
        raise ValueError("give exactly one of W_over_F and x_W")
    x_F = jnp.asarray(x_F, float)
    if x_W is None:
        target = -jnp.log(jnp.asarray(W_over_F, float))
        neg = (curve(x_F) - x_F) < 0.0        # beyond an azeotrope the still enriches
        lo = jnp.where(neg, x_F, 1e-9)
        hi = jnp.where(neg, 1.0 - 1e-9, x_F)
        f = lambda xs: jnp.where(neg, -1.0, 1.0) * (_rayleigh_ln_ratio(curve, x_F, xs)[0] - target)
        # the integral is +inf past an azeotrope; clamp so the bisection stays finite
        fs = lambda xs: jnp.minimum(f(xs), 1e6)
        xw = jnp.clip(_solve_monotone(fs, lo, hi, n_iter=60), lo, hi)
    else:
        xw = jnp.asarray(x_W, float)
    lnr, ok = _rayleigh_ln_ratio(curve, x_F, xw)
    w = jnp.exp(-lnr)
    d = 1.0 - w
    xd = (x_F - w * xw) / jnp.where(d > _EPS, d, 1.0)
    return {"x_W": xw, "W_over_F": w, "ln_F_over_W": lnr,
            "x_D_avg": jnp.where(d > _EPS, xd, curve(x_F)), "ok": ok}


def rayleigh_multicomponent(thermo, species, x_F: Array, W_over_F: Array | float, P: Array | float,
                            activity_model=None, n_steps: int = 100) -> dict:
    """Multicomponent simple distillation, ``dx_i/d ln W = y_i - x_i``.

    ``y_i = K_i(T_b, x) x_i`` with ``K_i = gamma_i Psat_i/P`` at the bubble
    temperature of the still liquid; integrated from ``ln W = ln F`` down to
    ``ln(W/F)`` with ``n_steps`` fixed RK4 steps (domain fixed, so no events
    and no Python control flow on traced values).

    Args:
        thermo: ``IdealThermo``-like object with ``Psat``.
        species: Species names in the order of ``x_F``.
        x_F: Charge mole fractions.
        W_over_F: Fraction of the charge left in the still.
        P: Pressure (Pa).
        activity_model: Optional activity model.
        n_steps: RK4 steps.

    Returns:
        dict with ``x_W`` (still composition), ``x_D_avg`` (accumulated
        distillate composition from the component balance), ``T_W`` (final
        still temperature, K) and ``W_over_F``.
    """
    species = tuple(species)
    x0 = jnp.asarray(x_F, float)
    P = jnp.asarray(P, float)
    u_end = jnp.log(jnp.asarray(W_over_F, float))

    def rhs(x):
        T = bubble_T(thermo, species, x, P, activity_model)
        y = x * _gamma(activity_model, x, T) * _psat(thermo, species, T) / P
        return y - x

    h = u_end / n_steps       # negative: ln W decreases

    def step(x, _):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * h * k1)
        k3 = rhs(x + 0.5 * h * k2)
        k4 = rhs(x + h * k3)
        return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, None

    xw, _ = lax.scan(step, x0, None, length=n_steps)
    w = jnp.exp(u_end)
    d = 1.0 - w
    xd = (x0 - w * xw) / jnp.where(d > _EPS, d, 1.0)
    return {"x_W": xw, "x_D_avg": xd, "W_over_F": w,
            "T_W": bubble_T(thermo, species, xw, P, activity_model)}


# ---------------------------------------------------------------------------
# The stage relation
# ---------------------------------------------------------------------------


def _azeotrope(curve: EquilibriumCurve) -> tuple[Array, Array]:
    """``(x_az, found)``: first interior crossing of ``y*(x) = x`` (1.0 if none)."""
    x = jnp.linspace(1e-3, 1.0 - 1e-3, 400)
    d = curve(x) - x
    cross = (d[:-1] > 0.0) & (d[1:] <= 0.0)
    found = jnp.any(cross)
    i = jnp.argmax(cross)
    d0, d1 = d[i], d[i + 1]
    xa = x[i] + (x[i + 1] - x[i]) * d0 / jnp.where(d0 - d1 != 0.0, d0 - d1, 1.0)
    return jnp.where(found, xa, 1.0), found


def _march(curve: EquilibriumCurve, x_D, L, n_stages: int, E):
    """Still composition for distillate ``x_D`` and operating-line slope ``L = R/(R+1)``.

    ``N`` trays from the top (total condenser, ``y_1 = x_D``), each with
    Murphree vapor efficiency ``E``, then the still as an equilibrium stage.
    """
    simple = isinstance(E, (int, float)) and float(E) == 1.0

    def tray(y, _):
        if simple:
            x = curve.inverse(y)
        else:
            f = lambda x_: (1.0 - E) * (L * x_ + (1.0 - L) * x_D) + E * curve(x_) - y
            x = _solve_monotone(f, jnp.asarray(0.0), jnp.asarray(1.0), n_iter=45)
        return L * x + (1.0 - L) * x_D, None

    y, _ = lax.scan(tray, jnp.asarray(x_D, float), None, length=n_stages)
    return curve.inverse(y)


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class BatchDistillationParams(ParamsMixin):
    """Parameters of a batch rectifier (binary, total condenser, no holdup).

    Attributes:
        species_order: The two species names; the charge stream is read in
            this order.
        n_stages: Equilibrium stages above the still. 0 is simple (Rayleigh)
            distillation; the still itself counts as one more stage.
        P: Operating pressure (Pa); sets the equilibrium curve built from the
            thermo package.
        mode: ``"constant_reflux"`` (distillate composition falls with time)
            or ``"constant_composition"`` (reflux rises to hold ``x_D_spec``).
        R: Reflux ratio ``L/D`` for constant reflux.
        x_D_spec: Distillate light-key mole fraction for constant composition.
        light_key: The species whose composition is tracked; it must be the
            more volatile one. Defaults to the first of ``species_order``.
        boilup: Constant boil-up rate ``V`` (mol/s), which converts the
            ln W integration into time.
        stage_efficiency: Murphree vapor efficiency of the trays above the
            still (the still is an equilibrium stage); 1 is ideal.
        holdup: Stage and condenser holdup (mol). Not modelled: must be 0.
    """

    species_order: list[str]
    n_stages: int = 0
    P: float = 101325.0
    mode: Literal["constant_reflux", "constant_composition"] = "constant_reflux"
    R: float = 1.0
    x_D_spec: float = 0.9
    light_key: str | None = None
    boilup: float = 1.0
    stage_efficiency: float = 1.0
    holdup: float = 0.0

    def __post_init__(self):
        if len(self.species_order) != 2:
            raise ValueError("BatchDistillation is binary: species_order needs two species")
        if self.mode not in ("constant_reflux", "constant_composition"):
            raise ValueError("mode must be 'constant_reflux' or 'constant_composition'")
        if self.light_key is not None and self.light_key not in self.species_order:
            raise ValueError(f"light_key {self.light_key!r} not in species_order")
        if int(self.n_stages) != self.n_stages or self.n_stages < 0:
            raise ValueError("n_stages must be a non-negative integer")
        if np.any(np.asarray(self.holdup) != 0.0):
            raise NotImplementedError(
                "stage/condenser holdup is out of scope: the model is quasi-steady, holdup must be 0")


_STOP_KEYS = ("x_W", "x_D_avg", "distillate", "time")


class BatchDistillation(ValueKeyed):
    """Batch rectifier: constant reflux or constant distillate composition.

    A palette unit fed a **charge in moles** (see the module docstring). Binary
    only; the equilibrium curve comes from ``thermo`` (plus an optional
    ``activity_model``, e.g. NRTL for ethanol-water) at ``params.P``, or is
    given directly as ``curve``. With ``n_stages = 0`` constant reflux is simple
    Rayleigh distillation.

    Call ``unit(charge, t_final=None, stop=None)`` -> ``(still, distillate,
    info)``. ``stop`` is data, not control flow: a dict with any of
    ``"x_W"`` (final still composition), ``"x_D_avg"`` (accumulated
    distillate purity), ``"distillate"`` (moles), ``"time"`` (s); the batch
    stops at the first one reached. ``t_final`` is ``stop={"time": t_final}``.
    """

    symbol = "BatchDistillation"
    equations = [
        r"\frac{dW}{dt} = -D = -\frac{V}{R+1},\quad \frac{d(W x_W)}{dt} = -D\,x_D",
        r"\ln\frac{F}{W} = \int_{x_W}^{x_F} \frac{dx}{x_D(x) - x}\quad(x_D = y^* \text{ for } N=0)",
        r"x_W = x^*\big(y_{N+1}\big),\ y_{n+1} = \tfrac{R}{R+1}x_n + \tfrac{x_D}{R+1},\ x_n = x^*(y_n),\ y_1 = x_D",
    ]
    assumptions = [
        "Binary mixture, total condenser, constant molar overflow.",
        "Quasi-steady column: no stage or condenser holdup (holdup must be 0).",
        "The charge stream is read as moles (the batch amount), not a rate.",
        "The tracked key must be the more volatile component; a distillate beyond an azeotrope is infeasible and flagged.",
        "Constant boil-up V converts the balance to time; reboiler heat is V times the mixture latent heat.",
    ]
    references = [
        "Rayleigh, Phil. Mag. 42 (1896) 493.",
        "Wankat, Separation Process Engineering, Ch. 9.",
        "McCabe, Smith & Harriott, Unit Operations of Chemical Engineering, Ch. 21.",
    ]
    parameter_symbols = {"n_stages": "N", "P": "P", "R": "R", "x_D_spec": "x_D", "boilup": "V",
                         "stage_efficiency": "E_{MV}"}
    parameter_units = {"n_stages": "-", "P": "Pa", "R": "-", "x_D_spec": "-", "boilup": "mol/s",
                       "stage_efficiency": "-", "holdup": "mol"}
    numerical_method = ("Gauss-Legendre quadrature of the Rayleigh integral in ln x, bisection + implicit-function "
                        "Newton step for x_D(x_W), R(x_W) and the stopping composition.")
    outlet_roles = ["still", "distillate"]
    #: the charge is in moles; palette editors should show it as a batch amount
    charge_basis = "mol"

    def __init__(self, params: BatchDistillationParams, thermo=None, activity_model=None,
                 curve: EquilibriumCurve | None = None):
        """Create the unit.

        Args:
            params: :class:`BatchDistillationParams`.
            thermo: Thermo package with ``Psat`` for both species (used to build
                the curve, bubble temperatures and latent heats). May be ``None``
                if ``curve`` is given; ``T`` and energy are then NaN.
            activity_model: Optional liquid activity model.
            curve: Equilibrium curve of the key, overriding ``thermo``.
        """
        if thermo is None and curve is None:
            raise ValueError("give thermo (and optionally activity_model) or an equilibrium curve")
        self.params = params
        self.thermo = thermo
        self.activity_model = activity_model
        self.curve = curve
        self._set_value_key(params, thermo, activity_model)

    # -- helpers -----------------------------------------------------------
    def _curve(self) -> EquilibriumCurve:
        if self.curve is not None:
            return self.curve
        p = self.params
        key = p.light_key or p.species_order[0]
        order = [key] + [s for s in p.species_order if s != key]
        return equilibrium_curve(self.thermo, order, p.P, self.activity_model)

    def _temperature(self, x_key: Array, P) -> Array:
        if self.thermo is None:
            return jnp.full_like(x_key, jnp.nan)
        p = self.params
        key = p.light_key or p.species_order[0]
        order = tuple([key] + [s for s in p.species_order if s != key])
        return jax.vmap(lambda xk: bubble_T(self.thermo, order, jnp.stack([xk, 1.0 - xk]), P,
                                            self.activity_model))(x_key)

    def __call__(self, charge: Stream, t_final: Array | float | None = None,
                 stop: dict | None = None, n_traj: int = 41) -> tuple[Stream, Stream, dict]:
        """Run the batch.

        Args:
            charge: Initial still contents, ``F_i`` in **mol** (not mol/s),
                plus ``T`` and ``P``.
            t_final: Stop at this time (s); shorthand for ``stop={"time": ...}``.
            stop: Stopping criteria, a dict with any of ``"x_W"``,
                ``"x_D_avg"``, ``"distillate"`` (mol), ``"time"`` (s); the
                first one reached ends the batch.
            n_traj: Points of the returned trajectory.

        Returns:
            still: Final still holdup (mol), at its bubble temperature.
            distillate: Accumulated distillate (mol), at its bubble temperature.
            info: ``t``, ``W``, ``x_W``, ``x_D``, ``R``, ``T_still`` (trajectory
                arrays of length ``n_traj``); ``distillate_amount``,
                ``x_D_avg``, ``t_batch``, ``boilup_total`` (mol of vapor,
                ``int V dt``), ``Q_reboiler`` (J, NaN without thermo),
                ``recovery`` (key recovered in the distillate),
                ``x_azeotrope`` (NaN if none), ``feasible`` and
                ``azeotrope_limited``, ``stop_reached`` flags.
        """
        p = self.params
        key = p.light_key or p.species_order[0]
        other = [s for s in p.species_order if s != key][0]
        flows = get_flows(charge)
        n_key = jnp.asarray(flows[key], float)
        n_oth = jnp.asarray(flows[other], float)
        F = n_key + n_oth
        x_F = n_key / F
        P = jnp.asarray(charge.get("P", p.P), float)
        stop = dict(stop or {})
        if t_final is not None:
            stop["time"] = t_final
        if not stop:
            raise ValueError("give t_final or a stop criterion; one of " + ", ".join(_STOP_KEYS))
        for k in stop:
            if k not in _STOP_KEYS:
                raise ValueError(f"unknown stop criterion {k!r}; use {_STOP_KEYS}")

        curve = self._curve()
        N = int(p.n_stages)
        E = p.stage_efficiency
        V = jnp.asarray(p.boilup, float)
        x_az, az_found = _azeotrope(curve)
        x_hi = jnp.where(az_found, x_az - 1e-9, 1.0 - 1e-9)
        const_R = p.mode == "constant_reflux"
        spec = jnp.asarray(p.x_D_spec, float)
        S_HI = 1.0 - 1e-9

        def xD_of(xW, L):
            f = lambda xd: _march(curve, xd, L, N, E) - xW
            return jnp.clip(_solve_monotone(f, xW, x_hi, n_iter=45), xW, x_hi)

        def L_of(xW):
            f = lambda s_: _march(curve, spec, s_, N, E) - xW
            return jnp.clip(_solve_monotone(f, jnp.asarray(0.0), jnp.asarray(S_HI), n_iter=45), 0.0, S_HI)

        L_const = jnp.asarray(p.R, float) / (1.0 + jnp.asarray(p.R, float))

        # lower bound of the still composition we may integrate to
        x_tot = _march(curve, spec, jnp.asarray(S_HI), N, E)     # total-reflux still for x_D = spec
        if const_R:
            x_lo = x_F * 1e-4
        else:
            x_lo = jnp.maximum(x_F * 1e-4, x_tot + 1e-3 * jnp.maximum(x_F - x_tot, 0.0))
        x_lo = jnp.minimum(x_lo, x_F * (1.0 - 1e-6))

        def state(xs):
            """W, distillate moles, time, accumulated x_D and validity at still composition ``xs``."""
            if const_R:
                nodes = x_F * (xs / x_F) ** _GL_S
                d = jax.vmap(lambda x: xD_of(x, L_const))(nodes) - nodes
                ok = jnp.all(d > _EPS)
                I = jnp.log(x_F / xs) * jnp.sum(_GL_W * nodes / jnp.where(ok, d, 1.0))
                W = F * jnp.exp(-I)
                Dm = F - W
                t = Dm / ((1.0 - L_const) * V)      # D = V/(R+1) = V (1 - L)
            else:
                W = F * (spec - x_F) / (spec - xs)
                Dm = F - W
                Wk = W + Dm * _GL_S
                xk = spec - (spec - x_F) * F / Wk
                Rk = jax.vmap(lambda xw: (lambda s_: s_ / (1.0 - s_))(L_of(xw)))(xk)
                t = Dm * jnp.sum(_GL_W * (Rk + 1.0)) / V
                ok = jnp.asarray(True)
            # accumulated composition from the component balance; as Dm -> 0 (cancellation) use its
            # limit, the instantaneous distillate at the charge composition
            x_D0 = xD_of(x_F, L_const) if const_R else spec
            thin = Dm < 1e-7 * F
            xda = jnp.where(thin, x_D0, (F * x_F - W * xs) / jnp.where(thin, 1.0, Dm))
            return W, Dm, t, xda, ok

        def progress(xs, name):
            W, Dm, t, xda, _ = state(xs)
            return {"x_W": x_F - xs, "time": t, "distillate": Dm, "x_D_avg": -xda}[name]

        targets = {"x_W": lambda s_: x_F - s_, "time": lambda s_: s_, "distillate": lambda s_: s_,
                   "x_D_avg": lambda s_: -s_}
        xs_list, reached = [], []
        for name, val in stop.items():
            tgt = targets[name](jnp.asarray(val, float))
            if name == "x_W":
                xs_c = jnp.clip(jnp.asarray(val, float), x_lo, x_F)
            else:
                g = lambda xs, _n=name, _t=tgt: progress(xs, _n) - _t
                xs_c = jnp.clip(_solve_monotone(g, x_lo, x_F, n_iter=45), x_lo, x_F)
            xs_list.append(xs_c)
            reached.append(jnp.abs(progress(xs_c, name) - tgt) <= 1e-6 * (1.0 + jnp.abs(tgt)))
        x_stop = xs_list[0]
        for c in xs_list[1:]:
            x_stop = jnp.maximum(x_stop, c)
        stop_reached = jnp.any(jnp.stack(reached))

        # trajectory in x_W, geometric from x_F to x_stop
        s = jnp.linspace(0.0, 1.0, n_traj)
        xs_traj = x_F * (x_stop / x_F) ** s
        W_t, D_t, t_t, xda_t, ok_t = jax.vmap(state)(xs_traj)
        if const_R:
            L_t = jnp.full_like(xs_traj, L_const)
            xD_t = jax.vmap(lambda x: xD_of(x, L_const))(xs_traj)
        else:
            L_t = jax.vmap(L_of)(xs_traj)
            xD_t = jnp.full_like(xs_traj, spec)
        R_t = L_t / (1.0 - L_t)
        W, Dm, t_b, xda, ok_end = state(x_stop)
        T_t = self._temperature(xs_traj, P)

        # reboiler energy: int V * Hvap_mix(x_W, T) dt (trapezoid on the trajectory)
        if self.thermo is None:
            Q = jnp.asarray(jnp.nan)
        else:
            def hmix(xk, T):
                return xk * self.thermo.Hvap(key, T) + (1.0 - xk) * self.thermo.Hvap(other, T)
            h = jax.vmap(hmix)(xs_traj, T_t)
            Q = V * jnp.sum(0.5 * (h[1:] + h[:-1]) * jnp.diff(t_t))

        feasible = ok_end & jnp.all(xD_t > xs_traj) & (x_F < x_hi)
        if not const_R:
            feasible = feasible & (spec < x_hi) & (spec > x_F) & (x_tot < x_F) & (
                jnp.all(L_t < S_HI * (1 - 1e-6))) & (_march(curve, spec, jnp.asarray(0.0), N, E) >= x_F)
        limited = jnp.where(az_found, jnp.max(xD_t) > x_az - AZEOTROPE_TOL, False)
        if not const_R:
            limited = limited | (az_found & (spec >= x_az - AZEOTROPE_TOL))

        x_key_still = x_stop
        still = {f"F_{key}": W * x_key_still, f"F_{other}": W * (1.0 - x_key_still),
                 "T": self._temperature(jnp.atleast_1d(x_key_still), P)[0], "P": P}
        dist = {f"F_{key}": Dm * xda, f"F_{other}": Dm * (1.0 - xda),
                "T": self._temperature(jnp.atleast_1d(xda), P)[0], "P": P}
        info = {
            "t": t_t, "W": W_t, "x_W": xs_traj, "x_D": xD_t, "R": R_t, "T_still": T_t,
            "distillate_amount": Dm, "x_D_avg": xda, "t_batch": t_b, "W_final": W,
            "x_W_final": x_stop, "boilup_total": V * t_b, "Q_reboiler": Q,
            "recovery": Dm * xda / (F * x_F), "charge_moles": F, "x_F": x_F,
            "x_azeotrope": jnp.where(az_found, x_az, jnp.nan),
            "feasible": feasible, "azeotrope_limited": limited, "stop_reached": stop_reached,
        }
        return still, dist, info


def batch_cuts(unit: BatchDistillation, charge: Stream, cut_specs: list) -> dict:
    """Run successive cuts, each charged with the previous still (heads / hearts / tails).

    Args:
        unit: The batch rectifier.
        charge: Initial charge (mol).
        cut_specs: One entry per cut. Each is a stop dict
            (``{"distillate": 20.0}``) or a dict ``{"stop": {...}, "R": ...,
            "x_D_spec": ..., "boilup": ..., "mode": ...}`` whose parameter keys
            override the unit's for that cut (the reflux policy per cut).

    Returns:
        dict with ``cuts`` (list of dicts: ``moles``, ``x_key`` (accumulated
        composition of the key), ``stream``, ``time``, ``info``), ``residue``
        (final still stream), ``total_time``, and ``balance`` (charge minus all
        cuts and residue, per species; zero to round-off).
    """
    key = unit.params.light_key or unit.params.species_order[0]
    cuts, current, total_t = [], charge, 0.0
    cumulative = {s: jnp.zeros(()) for s in unit.params.species_order}
    for spec in cut_specs:
        if "stop" in spec:
            over = {k: v for k, v in spec.items() if k != "stop"}
            stop = spec["stop"]
        else:
            over, stop = {}, spec
        u = unit
        if over:
            u = BatchDistillation(unit.params.update(**over), unit.thermo, unit.activity_model, unit.curve)
        still, dist, info = u(current, stop=stop)
        moles = sum(dist[f"F_{s}"] for s in unit.params.species_order)
        for s in unit.params.species_order:
            cumulative[s] = cumulative[s] + dist[f"F_{s}"]
        cuts.append({"moles": moles, "x_key": dist[f"F_{key}"] / moles, "stream": dist,
                     "time": info["t_batch"], "info": info})
        total_t = total_t + info["t_batch"]
        current = still
    flows0 = get_flows(charge)
    balance = {s: flows0[s] - cumulative[s] - current[f"F_{s}"] for s in unit.params.species_order}
    return {"cuts": cuts, "residue": current, "total_time": total_t, "balance": balance}
