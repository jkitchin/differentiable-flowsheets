"""Countercurrent leaching and washing: constant and variable underflow.

``Leacher`` models a cascade of ideal solid-liquid stages. Solids (the inert
carrier plus the solution it holds) enter stage 1 and leave stage ``N`` as the
*underflow*; fresh solvent enters stage ``N`` and leaves stage 1 as the
*overflow*. In every stage the solution held by the solids has the composition
of the overflow leaving that stage (equilibrium, no solute adsorbed), and the
underflow carries a stated amount of solution per kg of inert.

Flows are **mass flows (kg/s)**: an inert solid has no molar mass, so the
``F_<species>`` entries of the streams of this unit are kg/s, not mol/s.

Notation (Geankoplis / McCabe-Smith-Harriott)::

    B            inert solids (kg/s), the same in every stage
    L_0, m_0     solution and solute entering with the solids (the "stage 0" underflow)
    L_n, y_n     solution in the underflow leaving stage n, its solute mass fraction
    V_n          overflow leaving stage n (V_1 leaves the cascade, V_{N+1} is fresh solvent)

Because every stage is an ideal mixer-settler the overflow and underflow leaving
stage ``n`` have the same composition ``y_n``.

Models
------
``"constant_solution"``
    ``L_n = B u`` kg of *solution* per kg inert for every stage ``n >= 1``
    (``u = params.underflow``). Overall balances across any cut give
    ``V_{n+1} - L_n = V_1 - L_0`` and ``V_{n+1} y_{n+1} - L_n y_n = D`` with a
    constant solute flux ``D = V_1 y_1 - m_0``. So ``V_{n+1} = V`` is the same for
    ``n = 1..N`` and only the first overflow differs, ``V_1 = V + L_0 - L``:
    **the first stage is special**, as in the textbooks, whenever the feed solids
    do not carry the same solution as the underflow. With ``r = L/V``::

        y_{n+1} = r y_n + D/V,        y_{N+1} = r^N y_1 + (D/V) S_N,
        S_N = (r^N - 1)/(r - 1) = sum_{k<N} r^k,    y_1 = (D + m_0)/V_1

    which is solved for ``D`` in closed form. ``N`` enters only through ``r^N`` and
    ``S_N``, so the stage count is a real number and the result is differentiable
    in it; ``r = 1`` is a smooth limit. For pure washing (``L_0 = L``, fresh
    solvent free of solute) the fraction of solute left in the underflow is
    ``(1-w)/(1-w^{N+1})`` with the wash ratio ``w = V/L``
    (:func:`washing_recovery`).
``"constant_solvent"``
    ``L_n = B u`` kg of *solvent* per kg inert. On a solute-per-solvent ratio basis
    the equations are the same, with solvent flows in place of solution flows and
    ``Y = solute/solvent`` in place of ``y``; this is exactly the textbook
    constant-solvent-underflow case.
``"variable"``
    ``L_n = B s(y_n)`` with ``s`` a tabulated underflow curve (kg solution per kg
    inert against the solute mass fraction, ``underflow_y`` and ``underflow_s``,
    interpolated by PCHIP; oilseeds are the classic case). The ``N`` stage
    balances ``V_{n+1} y_{n+1} - L_n y_n = D`` are nonlinear in ``y``; they are
    solved by Newton's method with a fixed iteration count (a fixed-size linear
    solve per step), seeded by the constant-underflow solution, so the result is
    jit-able and differentiable. ``n_stages`` must be a concrete integer.

Design
------
``recovery`` (fraction of the feed solute that ends in the overflow, net of the
solute that comes in with the solvent) with ``design="stages"`` returns the
(continuous) stage count in closed form, and with ``design="solvent"`` scales the
solvent stream (its composition fixed) to meet the recovery at ``n_stages``
(bisection with implicit gradients). ``stage_efficiency`` is an *overall*
efficiency: the number of ideal stages is ``E * n_stages``.

Limits
------
Ideal stages, the solute fully dissolved and in equilibrium with the solution
(no rate limit and no adsorption on the solids); one solute, one solvent, one
inert (other species in the streams are ignored); the inert flow is the same in
every stage; isothermal. Rate-limited (diffusion-controlled) leaching kinetics
is out of scope. The variable-underflow path reproduces McCabe-Smith-Harriott
7e Example 23.1 (oil from meal with benzene: overflow 0.60 oil, four ideal
stages) with the problem data as recalled, not yet checked against the printed
book; see ``tests/test_leaching.py`` for what is checked and against what.

References
----------
- McCabe, Smith & Harriott, Unit Operations of Chemical Engineering, Ch. 23
  (equations; Example 23.1 reproduced from recalled data, see above).
- Geankoplis, Transport Processes and Separation Process Principles, Ch. 12
  (leaching; equations only).
- Treybal, Mass-Transfer Operations, Ch. 13 (leaching).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.staged import (
    _pchip_eval,
    _pchip_slopes,
    _solve_monotone,
    kremser_fraction,
    kremser_stages,
)
from difflow.streams import Stream, get_flows, make_stream

__all__ = [
    "Leacher",
    "LeacherParams",
    "tabulate_underflow",
    "washing_recovery",
    "washing_stages",
]

_N_NEWTON = 16
_TINY = 1e-300


# ---------------------------------------------------------------------------
# Washing helpers (library)
# ---------------------------------------------------------------------------


def washing_recovery(wash_ratio: Array | float, n_stages: Array | float) -> Array:
    """Fraction of the solute washed out of the underflow by ``N`` countercurrent stages.

    Pure washing at constant underflow (the solids carry the same solution in and
    out) with solute-free wash liquid: the fraction left in the underflow is
    ``(1 - w)/(1 - w^(N+1))`` with the wash ratio ``w = V/L`` (wash liquid per
    unit of solution in the underflow), so the fraction removed is the Kremser
    form ``(w^(N+1) - w)/(w^(N+1) - 1)``. ``w = 1`` is the smooth limit
    ``N/(N+1)``. For ``w < 1`` the removal tends to ``w``, not to 1.

    Args:
        wash_ratio: ``w = V/L``, wash solution per solution in the underflow.
        n_stages: Number of ideal stages (continuous).

    Returns:
        Fraction of the solute removed from the underflow, in [0, 1].

    Example:
        >>> from difflow.units.leaching import washing_recovery
        >>> round(float(washing_recovery(1.0, 3.0)), 6)   # w = 1: N/(N+1)
        0.75
    """
    return kremser_fraction(wash_ratio, n_stages)


def washing_stages(wash_ratio: Array | float, recovery: Array | float) -> Array:
    """Stages needed to wash out a ``recovery`` fraction (inverse of :func:`washing_recovery`).

    ``N = ln((w - f)/(1 - f))/ln w - 1``; ``inf`` when ``f >= w`` (``w < 1``
    cannot reach it); not floored at one stage.

    Args:
        wash_ratio: ``w = V/L``.
        recovery: Fraction of the solute to remove, in [0, 1).

    Returns:
        Number of ideal stages (continuous).
    """
    return kremser_stages(wash_ratio, recovery)


def tabulate_underflow(fn: Callable[[Array], Array], y_max: float = 1.0, n: int = 21) -> tuple[list[float], list[float]]:
    """Sample a callable ``s(y)`` into the ``(underflow_y, underflow_s)`` tables.

    ``LeacherParams`` takes the variable underflow as data (so the unit stays
    buildable in the editor); use this to turn a correlation into the tables.

    Args:
        fn: ``s(y)``, kg solution per kg inert at solute mass fraction ``y``.
        y_max: Largest solute mass fraction tabulated.
        n: Number of points (at least 2).

    Returns:
        ``(underflow_y, underflow_s)`` as lists of floats.
    """
    ys = [y_max * i / (n - 1) for i in range(n)]
    return ys, [float(fn(jnp.asarray(y))) for y in ys]


# ---------------------------------------------------------------------------
# Closed-form constant-underflow cascade
# ---------------------------------------------------------------------------


def _geom_sum(ell: Array, N: Array) -> Array:
    """``sum_{k=0}^{N-1} r^k = (r^N - 1)/(r - 1)`` with ``r = e^ell``, smooth at ``ell = 0``."""
    near = jnp.abs(ell) < 1e-6
    e = jnp.where(near, 1.0, ell)
    far = jnp.expm1(N * e) / jnp.expm1(e)
    series = N * (1.0 + 0.5 * (N - 1.0) * ell + (N * N / 6.0 - 1.0 / 6.0 - (N - 1.0) / 4.0) * ell * ell)
    return jnp.where(near, series, far)


def _const_cascade(V, L, V1, m0, y_s, N):
    """Closed form of the constant-underflow cascade (see the module docstring).

    ``V`` carrier (solution or solvent) in the overflows ``V_2..V_{N+1}``, ``L`` carrier in
    the underflows, ``V1`` the first overflow, ``m0`` solute entering with the solids,
    ``y_s`` composition of the fresh solvent. Returns ``(D, y_1, y_N, r)``.
    """
    ell = jnp.log(L / V)
    rN = jnp.exp(N * ell)
    D = (y_s - rN * m0 / V1) / (rN / V1 + _geom_sum(ell, N) / V)
    y1 = (D + m0) / V1
    yN = jnp.exp((N - 1.0) * ell) * y1 + D / V * _geom_sum(ell, N - 1.0)
    return D, y1, yN


def _const_profile(V, L, V1, m0, y_s, N_int: int):
    """``y_n`` for ``n = 1..N_int`` (integer ``N_int``) from the closed form."""
    N = jnp.asarray(float(N_int))
    D, y1, _ = _const_cascade(V, L, V1, m0, y_s, N)
    ell = jnp.log(L / V)
    n = jnp.arange(1, N_int + 1, dtype=float)
    return jnp.exp((n - 1.0) * ell) * y1 + D / V * _geom_sum(ell, n - 1.0)


def _const_stages_for_recovery(V, L, V1, m0, y_s, M1):
    """Closed-form ``N`` for a first-overflow solute flow ``M1`` (``V1 y_1``).

    Returns ``(N, feasible)``. From ``y_{N+1} = r^N y_1 + (D/V)(r^N - 1)/(r - 1)``:
    ``N = ln(1 + (y_s - y_1)/(y_1 + a))/ln r``, ``a = D/(V (r - 1))``; at ``r = 1``,
    ``N = (y_s - y_1) V/D``.
    """
    ell = jnp.log(L / V)
    D = M1 - m0
    y1 = M1 / V1
    near = jnp.abs(ell) < 1e-10
    e = jnp.where(near, 1.0, ell)
    a = D / (V * jnp.expm1(e))
    arg = (y_s - y1) / (y1 + a)
    ok_far = arg > -1.0
    far = jnp.log1p(jnp.where(ok_far, arg, 0.0)) / e
    Dn = jnp.where(jnp.abs(D) > _TINY, D, 1.0)
    nr = (y_s - y1) * V / Dn
    N = jnp.where(near, nr, far)
    feasible = jnp.where(near, jnp.abs(D) > _TINY, ok_far) & (N > 0.0) & jnp.isfinite(N)
    return jnp.where(feasible, N, jnp.inf), feasible


# ---------------------------------------------------------------------------
# Variable underflow: fixed-iteration Newton on the stage balances
# ---------------------------------------------------------------------------


def _variable_cascade(sfun, B, L0, m0, V, m_S, N: int, y_seed):
    """Solve the ``N`` stage balances for ``y_1..y_N`` with ``L_n = B s(y_n)``."""

    def resid(y):
        L = B * sfun(y)
        V1 = V + L0 - L[-1]
        Vn1 = V1 - L0 + L  # overflow entering stage n, n = 1..N  (V_2..V_{N+1})
        ynext = jnp.concatenate([y[1:], jnp.reshape(m_S / V, (1,))])
        D = V1 * y[0] - m0
        return Vn1 * ynext - L * y - D

    def newton(y, _):
        r = resid(y)
        J = jax.jacfwd(resid)(y)
        return y - jnp.linalg.solve(J, r), None

    y, _ = lax.scan(newton, y_seed, None, length=_N_NEWTON)
    return y, resid(y)


# ---------------------------------------------------------------------------
# Params and unit
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class LeacherParams(ParamsMixin):
    """Parameters of a countercurrent leaching or washing cascade.

    Attributes:
        inert_species: Name of the inert solid carrier (``pulp``, ``meal``); the
            stream key is ``F_<name>``. All flows of this unit are kg/s.
        solute: Name of the solute (sugar, oil, a salt).
        solvent: Name of the solvent (water, hexane).
        n_stages: Number of ideal stages (continuous for the constant-underflow
            models, an integer for ``"variable"``); an output when ``design="stages"``.
        underflow_model: ``"constant_solution"`` (``underflow`` is kg solution per kg
            inert), ``"constant_solvent"`` (kg solvent per kg inert) or ``"variable"``
            (the ``underflow_y`` / ``underflow_s`` tables).
        underflow: Underflow ratio ``u`` (kg per kg inert) for the constant models.
        underflow_y: Solute mass fractions of the variable-underflow table
            (increasing).
        underflow_s: Kg of solution per kg of inert at those fractions.
        stage_efficiency: Overall stage efficiency ``E`` in (0, 1]; the ideal stage
            count is ``E * n_stages``.
        recovery: Design specification: fraction of the feed solute that ends in
            the overflow (net of solute in the fresh solvent). ``None`` = rating.
        design: ``"rating"`` (n_stages and the solvent stream are given),
            ``"stages"`` (the stage count is solved for ``recovery``) or
            ``"solvent"`` (the solvent stream is scaled to meet ``recovery``).
    """

    inert_species: str
    solute: str
    solvent: str
    n_stages: float = 3.0
    underflow_model: Literal["constant_solution", "constant_solvent", "variable"] = "constant_solution"
    underflow: float = 1.0
    underflow_y: list[float] | None = None
    underflow_s: list[float] | None = None
    stage_efficiency: float = 1.0
    recovery: float | None = None
    design: Literal["rating", "stages", "solvent"] = "rating"

    def __post_init__(self):
        if self.underflow_model not in ("constant_solution", "constant_solvent", "variable"):
            raise ValueError("underflow_model must be 'constant_solution', 'constant_solvent' or 'variable'")
        if self.design not in ("rating", "stages", "solvent"):
            raise ValueError("design must be 'rating', 'stages' or 'solvent'")
        if self.design != "rating" and self.recovery is None:
            raise ValueError(f"design={self.design!r} needs a recovery")
        if self.underflow_model == "variable":
            if self.underflow_y is None or self.underflow_s is None:
                raise ValueError("underflow_model='variable' needs underflow_y and underflow_s")
            if len(self.underflow_y) != len(self.underflow_s) or len(self.underflow_y) < 2:
                raise ValueError("underflow_y and underflow_s need the same length (at least 2)")
            if self.design == "stages":
                raise ValueError("design='stages' is not available for the variable underflow "
                                 "(n_stages is an integer there); use rating or design='solvent'")
        if len({self.inert_species, self.solute, self.solvent}) != 3:
            raise ValueError("inert_species, solute and solvent must be three different species")


def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer)


_ASSUMPTIONS = [
    "Ideal stages: the solution held by the solids has the composition of the overflow leaving the stage "
    "(no rate limit, no adsorption of solute on the solids).",
    "Inert solids flow is the same in every stage; the underflow carries a stated amount of solution (or "
    "solvent) per unit inert, constant or tabulated against the solute fraction.",
    "One solute, one solvent, one inert; other species in the streams are ignored. Flows are mass flows (kg/s).",
    "Isothermal; the outlet streams take the temperature and pressure of the solids feed.",
    "stage_efficiency is an overall efficiency: ideal stages = E * n_stages.",
]
_EQUATIONS = [
    r"V_{n+1}-L_n = V_1-L_0,\qquad V_{n+1}y_{n+1}-L_n y_n = V_1 y_1-m_0 = D\qquad\text{(balances across any cut)}",
    r"y_{N+1} = r^{N} y_1+\frac{D}{V}\,\frac{r^{N}-1}{r-1},\quad r=\frac{L}{V},\quad V_1=V+L_0-L\qquad\text{(constant underflow, first stage separate)}",
    r"\frac{m_N}{m_0}=\frac{1-w}{1-w^{N+1}},\quad w=\frac{V}{L}\qquad\text{(pure washing, solute-free wash liquid)}",
    r"L_n = B\,s(y_n),\quad \mathbf{r}(\mathbf{y})=V_{n+1}y_{n+1}-L_ny_n-D=0\qquad\text{(variable underflow, Newton)}",
]
_REFERENCES = [
    "McCabe, W.L., Smith, J.C., Harriott, P. Unit Operations of Chemical Engineering, Ch. 23 (leaching and washing).",
    "Geankoplis, C.J. Transport Processes and Separation Process Principles, Ch. 12 (leaching).",
    "Treybal, R.E. Mass-Transfer Operations, Ch. 13.",
]


class Leacher:
    """Countercurrent solid-liquid extraction (leaching) and washing cascade.

    Two inlets, ``solids_feed`` (inert plus the solution it holds) and
    ``solvent`` (fresh solvent, possibly with some solute), and two outlets:
    the ``underflow`` (spent solids with the solution they carry) and the
    ``overflow`` (the leach liquor). See the module docstring for the models.
    The pure-washing case has the closed forms :func:`washing_recovery` and
    :func:`washing_stages`.
    """

    symbol = "Leacher"
    equations = _EQUATIONS
    assumptions = _ASSUMPTIONS
    references = _REFERENCES
    parameter_units = {"n_stages": "-", "underflow": "kg/kg inert", "underflow_s": "kg solution/kg inert",
                       "underflow_y": "kg/kg", "stage_efficiency": "-", "recovery": "-"}
    parameter_symbols = {"n_stages": "N", "underflow": "u", "stage_efficiency": "E", "recovery": r"\phi"}
    numerical_method = ("Closed-form constant-underflow cascade (geometric sums, continuous N); "
                        "fixed-iteration Newton on the stage balances for variable underflow; "
                        "implicit-gradient bisection for the solvent design.")
    outlet_roles = ["underflow", "overflow"]

    def __init__(self, params: LeacherParams):
        self.params = params

    # -- helpers -------------------------------------------------------------
    def _split_inputs(self, solids_feed: Stream, solvent: Stream):
        p = self.params
        sf, vf = get_flows(solids_feed), get_flows(solvent)
        g = lambda d, k: jnp.asarray(d.get(k, 0.0), float)  # noqa: E731
        B = g(sf, p.inert_species)
        feed_solute, feed_solvent = g(sf, p.solute), g(sf, p.solvent)
        fresh_solute, fresh_solvent = g(vf, p.solute), g(vf, p.solvent)
        return B, feed_solute, feed_solvent, fresh_solute, fresh_solvent

    def _table(self):
        p = self.params
        ys = jnp.asarray(p.underflow_y, float)
        ss = jnp.asarray(p.underflow_s, float)
        return ys, ss, _pchip_slopes(ys, ss)

    def __call__(self, solids_feed: Stream, solvent: Stream) -> tuple[Stream, Stream, dict]:
        """Contact the solids (stage 1 to N) with the solvent (stage N to 1).

        Args:
            solids_feed: Solids entering stage 1: the inert (``inert_species``) and
                the solute and solvent in the solution they hold, kg/s.
            solvent: Fresh solvent entering stage N (solvent, and any solute in it), kg/s.

        Returns:
            underflow_out: Spent solids leaving stage N with the solution they carry.
            overflow_out: Leach liquor leaving stage 1.
            info: ``recovery`` (fraction of the feed solute in the overflow, net of the
                solute in the fresh solvent), ``washing_efficiency`` (fractional approach of the
                final underflow solution to the fresh-solvent composition,
                ``1 - (y_N - y_{N+1})/(y_0 - y_{N+1})``), ``solute_to_overflow`` and
                ``solute_in_underflow`` (kg/s), ``solution_to_inert_feed`` and
                ``solution_to_inert_out`` (kg solution per kg inert), ``y_overflow`` /
                ``y_underflow`` (solute mass fraction of the overflow leaving the cascade and of
                the solution in the final underflow), ``n_stages`` (given, or solved for
                ``design="stages"``) and ``N_ideal``, ``solvent_rate`` (fresh solvent, kg/s,
                solved for ``design="solvent"``), ``stage_y`` (mass fraction of the solution
                leaving each stage; concrete integer stage counts only, else ``None``),
                ``stage_L`` and ``stage_V`` (variable underflow), ``converged``, ``feasible``.
        """
        p = self.params
        B, feed_solute, feed_solvent, fresh_solute, fresh_solvent = self._split_inputs(solids_feed, solvent)
        T, P = solids_feed["T"], solids_feed["P"]
        eta = jnp.asarray(p.stage_efficiency, float)
        n_given = jnp.asarray(p.n_stages, float)
        feasible = jnp.asarray(True)
        converged = jnp.asarray(True)
        stage_y = stage_L = stage_V = None
        solvent_scale = jnp.asarray(1.0)
        design = p.design

        L0_solution = feed_solute + feed_solvent
        y0_feed = safe_divide(feed_solute, L0_solution)

        if p.underflow_model == "variable":
            if _is_tracer(p.n_stages):
                raise ValueError("underflow_model='variable' needs a concrete integer n_stages (not a traced value)")
            n_ideal_f = float(p.n_stages) * float(p.stage_efficiency)
            N = int(round(n_ideal_f))
            if abs(n_ideal_f - N) > 1e-9 or N < 1:
                raise ValueError("underflow_model='variable' needs an integer number of ideal stages "
                                 f"E * n_stages >= 1, got {n_ideal_f}")
            ys_t, ss_t, d_t = self._table()

            def sfun(y):
                return _pchip_eval(ys_t, ss_t, d_t, jnp.clip(y, ys_t[0], ys_t[-1]))

            def run(V_fresh):
                m_S = fresh_solute * V_fresh / (fresh_solute + fresh_solvent)
                u0 = sfun(0.5 * (y0_feed + m_S / V_fresh))
                y_s = m_S / V_fresh
                V1s = V_fresh + L0_solution - B * u0
                seed = _const_profile(V_fresh, B * u0, jnp.maximum(V1s, 1e-12), feed_solute, y_s, N)
                y, res = _variable_cascade(sfun, B, L0_solution, feed_solute, V_fresh, m_S, N, seed)
                return y, res, m_S

            V_fresh = fresh_solute + fresh_solvent
            if design == "solvent":
                L_hi = B * jnp.max(ss_t)

                def f(t):
                    Vv = jnp.maximum(L_hi - L0_solution, 0.0) + 1e-9 * (1.0 + L_hi) + jnp.exp(t)
                    y, _, m_S = run(Vv)
                    L = B * sfun(y)
                    V1 = Vv + L0_solution - L[-1]
                    return safe_divide(V1 * y[0] - m_S, feed_solute) - p.recovery

                t = _solve_monotone(f, -25.0, 25.0, 70)
                V_fresh = jnp.maximum(L_hi - L0_solution, 0.0) + 1e-9 * (1.0 + L_hi) + jnp.exp(t)
                solvent_scale = V_fresh / (fresh_solute + fresh_solvent)
            y, res, m_S = run(V_fresh)
            L = B * sfun(y)
            LN = L[-1]
            V1 = V_fresh + L0_solution - LN
            y1, yN = y[0], y[-1]
            M1 = V1 * y1
            under_solute = LN * yN
            over_solute, over_carrier = M1, V1
            under_solvent = LN - under_solute
            over_solvent = V1 - M1
            converged = jnp.max(jnp.abs(res)) < 1e-9 * (1.0 + V_fresh + L0_solution)
            feasible = (V1 > 0.0) & jnp.all(y >= -1e-12)
            N_ideal = jnp.asarray(float(N))
            n_out = jnp.asarray(p.n_stages, float)
            stage_y, stage_L = y, L
            stage_V = jnp.concatenate([jnp.reshape(V1, (1,)), V1 - L0_solution + L])
        else:
            basis_solution = p.underflow_model == "constant_solution"
            u = jnp.asarray(p.underflow, float)
            L = B * u
            if basis_solution:
                carrier_feed = feed_solute + feed_solvent
                carrier_fresh0 = fresh_solute + fresh_solvent
            else:
                carrier_feed, carrier_fresh0 = feed_solvent, fresh_solvent
            m0 = feed_solute
            fresh_solute_flow = fresh_solute
            if design == "solvent":
                N_ideal = eta * n_given

                def f(t):
                    Vv = jnp.maximum(L - carrier_feed, 0.0) + 1e-9 * (1.0 + L) + jnp.exp(t)
                    s = Vv / carrier_fresh0
                    ys = fresh_solute_flow * s / Vv
                    V1 = Vv + carrier_feed - L
                    D, y1, _ = _const_cascade(Vv, L, V1, m0, ys, N_ideal)
                    return safe_divide(V1 * y1 - Vv * ys, m0) - p.recovery

                t = _solve_monotone(f, -25.0, 25.0, 70)
                V = jnp.maximum(L - carrier_feed, 0.0) + 1e-9 * (1.0 + L) + jnp.exp(t)
                solvent_scale = V / carrier_fresh0
                feasible = (f(25.0) >= 0.0) & (f(-25.0) <= 0.0)
            else:
                V = carrier_fresh0
            y_s = safe_divide(fresh_solute_flow * (V / carrier_fresh0), V)
            V1 = V + carrier_feed - L
            if design == "stages":
                M1 = p.recovery * m0 + V * y_s
                N_ideal, feas_n = _const_stages_for_recovery(V, L, V1, m0, y_s, M1)
                feasible = feas_n & (V1 > 0.0)
                N_ideal = jnp.where(feas_n, N_ideal, jnp.nan)  # unreachable: NaN, feasible=False
                n_out = N_ideal / eta
            else:
                N_ideal = eta * n_given
                n_out = n_given
            D, y1, yN = _const_cascade(V, L, V1, m0, y_s, N_ideal)
            feasible = feasible & (V1 > 0.0) & (V > 0.0)
            M1 = V1 * y1
            under_solute = L * yN
            if basis_solution:
                over_solute, over_carrier = M1, V1
                under_solvent = L - under_solute
                over_solvent = V1 - M1
            else:
                over_solute = M1
                over_solvent = V1
                under_solvent = L
                over_carrier = V1
            if not _is_tracer(N_ideal) and bool(jnp.isfinite(N_ideal)):
                n_int = int(jnp.floor(N_ideal + 1e-9))
                if n_int >= 1:
                    stage_y = _const_profile(V, L, V1, m0, y_s, n_int)
            if not basis_solution and stage_y is not None:
                stage_y = stage_y / (1.0 + stage_y)  # ratio -> mass fraction

        # solute mass fraction of the overflow and of the final underflow solution
        y_over = safe_divide(over_solute, over_solute + over_solvent)
        y_under = safe_divide(under_solute, under_solute + under_solvent)
        y_fresh = safe_divide(fresh_solute * solvent_scale, (fresh_solute + fresh_solvent) * solvent_scale)
        recovery = safe_divide(over_solute - fresh_solute * solvent_scale, feed_solute)
        wash_eff = 1.0 - safe_divide(y_under - y_fresh, y0_feed - y_fresh)

        underflow_out = make_stream({p.inert_species: B, p.solute: under_solute, p.solvent: under_solvent}, T, P)
        overflow_out = make_stream({p.solute: over_solute, p.solvent: over_solvent}, T, P)
        info = {
            "recovery": recovery,
            "washing_efficiency": wash_eff,
            "solute_to_overflow": over_solute,
            "solute_in_underflow": under_solute,
            "y_overflow": y_over,
            "y_underflow": y_under,
            "solution_to_inert_feed": L0_solution / B,
            "solution_to_inert_out": (under_solute + under_solvent) / B,
            "n_stages": n_out if p.underflow_model != "variable" else jnp.asarray(p.n_stages, float),
            "N_ideal": N_ideal,
            "solvent_rate": (fresh_solute + fresh_solvent) * solvent_scale,
            "stage_y": stage_y,
            "stage_L": stage_L,
            "stage_V": stage_V,
            "converged": converged,
            "feasible": feasible,
        }
        return underflow_out, overflow_out, info
