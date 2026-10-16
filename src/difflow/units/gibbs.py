r"""Gibbs reactor: chemical equilibrium by Gibbs-energy minimisation (#468).

The outlet of an ideal-gas reactor is the composition that minimises the
total Gibbs energy subject to the element balances,

.. math::

    \min_n \; \sum_i n_i\left[\frac{g_i^\circ(T)}{RT} + \ln\frac{n_i}{N}
        + \ln\frac{P}{P^\circ}\right]
    \quad\text{s.t.}\quad A n = A n^{\mathrm{in}}, \; n \ge 0,

with :math:`P^\circ = 1` bar. It is solved by the **element-potential method**
(Reynolds' STANJAN; Smith & Missen 1982, ch. 6). At the minimum each chemical
potential is a combination of element potentials :math:`\lambda`, so

.. math::

    n_i = N \exp\Big(\sum_k A_{ki}\lambda_k - g_i^\circ/RT - \ln P/P^\circ\Big),

and the unknowns are :math:`\lambda` and :math:`\ln N`, one per element plus
one. Newton's method with a backtracking line search solves the element
balances and :math:`\sum_i n_i = N`. Every :math:`n_i` is an exponential, so
trace species are never negative or NaN.

**Absent and trace elements.** An element that is not fed has no potential to
solve for: its row is dropped and every species containing it is exactly 0.
An element fed at a trace -- below :data:`TRACE_ELEMENT` of the fed atoms,
such as the 1e-27 mol/s of CO2 an amine regenerator passes to a Claus furnace
-- is treated the same way. Its balance row, scaled by its own abundance,
otherwise dominates the Newton step: a furnace fed 1e-9 mol/s of CO2 in 175
mol/s of acid gas and air diverged, with the temperature and every other
species thrown off. The trace's atoms are dropped from the outlet, an
element-balance error at most :data:`TRACE_ELEMENT` of the feed (reported in
``info["element_balance"]``).

**Modes.**

- **Isothermal**: the outlet is at ``T_out``, or at the inlet T if
  ``T_out`` is None.
- **Adiabatic**: ``mode="adiabatic"``. T is a further unknown, and
  :math:`\sum n_i h_i(T) = H_{in} + Q`, where ``Q`` is ``heat_duty``.

**Approach to equilibrium.**

- **Overall**: ``approach_dT`` evaluates :math:`g^\circ` at
  :math:`T + \Delta T`.
- **Per reaction**: ``reactions`` restricts the outlet to
  :math:`n^{in} + \nu\xi`. The element rows are replaced by a basis of the
  reaction invariants (the left null space of :math:`\nu`), and
  :math:`g^\circ/RT` gets the least-norm shift that puts each reaction
  :math:`r` at :math:`K_r(T + \Delta T_r)`. If the reactions span every
  element-conserving change and every :math:`\Delta T_r = 0`, the answer is
  the unrestricted one (tested).

**Gradients.** These come from implicit differentiation at the converged
point, not from unrolling Newton. The solve runs on stop-gradient inputs.
One Newton correction :math:`z^* - J^{-1} r(z^*, \theta)` carries the
derivatives :math:`dz/d\theta = -J^{-1}\partial r/\partial\theta` in both
forward and reverse mode, and changes the value by only round-off.

**Thermodynamics** are pluggable. Any object with ``names``, ``gibbs(T)``
and ``enthalpy(T)`` (J/mol, the 1 bar standard state, shape ``(n,)``) and
``ELEMENTS`` (``{element: (n,) counts}``) works. Two such objects are
:class:`GibbsThermo` here, built from Hf, S0 and a Cp cubic, and the
refinery's ``difflow_refinery.thermochemistry.IdealGasSet``. This module
imports no plugin.

References:

- Smith, W.R. & Missen, R.W., *Chemical Reaction Equilibrium Analysis:
  Theory and Algorithms*, Wiley (1982).
- Reynolds, W.C., "The element potential method for chemical equilibrium
  analysis: implementation in the interactive program STANJAN", Stanford
  University, Dept. of Mechanical Engineering (1986).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, make_stream

R_GAS = 8.314462618        # J/mol/K, CODATA 2018 (exact)
P_STANDARD = 1.0e5         # Pa: the 1 bar standard state
#: An element fed below this fraction of the fed atoms counts as absent (see
#: the module docstring).
TRACE_ELEMENT = 1e-10
T_REF = 298.15


@dataclass
class GibbsThermo:
    """Ideal-gas standard-state data over a species list, for :class:`GibbsReactor`.

    ``Cp = a + bT + cT^2 + dT^3`` (J/mol/K). ``Hf`` (J/mol) and ``S0``
    (J/mol/K, 1 bar) are at 298.15 K.

    Attributes:
        names: Species names, in array order.
        Hf: Enthalpies of formation (J/mol).
        S0: Absolute entropies at 1 bar (J/mol/K).
        cp: Cp coefficients ``(n, 4)``.
        elements: ``{species: {element: count}}``.
    """

    names: tuple
    Hf: Any
    S0: Any
    cp: Any
    elements: dict

    def __post_init__(self):
        self.names = tuple(self.names)
        self.Hf = np.asarray(self.Hf, dtype=float)
        self.S0 = np.asarray(self.S0, dtype=float)
        self.cp = np.asarray(self.cp, dtype=float).reshape(len(self.names), 4)
        els = sorted({e for s in self.names for e in self.elements[s]})
        self.ELEMENTS = {e: np.array([self.elements[s].get(e, 0) for s in self.names],
                                     dtype=float) for e in els}

    def _int_cp(self, T):
        a, b, c, d = (jnp.asarray(self.cp[:, j]) for j in range(4))

        def F(t):
            return a * t + b * t ** 2 / 2 + c * t ** 3 / 3 + d * t ** 4 / 4

        return F(T) - F(T_REF)

    def _int_cp_over_T(self, T):
        a, b, c, d = (jnp.asarray(self.cp[:, j]) for j in range(4))

        def F(t):
            return a * jnp.log(t) + b * t + c * t ** 2 / 2 + d * t ** 3 / 3

        return F(T) - F(T_REF)

    def enthalpy(self, T):
        """``Hf + int Cp dT`` (J/mol), ``(n,)``."""
        return jnp.asarray(self.Hf) + self._int_cp(jnp.asarray(T, dtype=float))

    def gibbs(self, T):
        """Standard (1 bar) ``H - TS`` (J/mol), ``(n,)``."""
        T = jnp.asarray(T, dtype=float)
        return self.enthalpy(T) - T * (jnp.asarray(self.S0) + self._int_cp_over_T(T))


@dataclass(repr=False)
class GibbsReactorParams(ParamsMixin):
    """Parameters of a :class:`GibbsReactor`.

    Attributes:
        thermo: Standard-state thermodynamics over the reactor's species (see
            the module docstring): a :class:`GibbsThermo` or any object with
            ``names``, ``gibbs(T)``, ``enthalpy(T)`` and ``ELEMENTS``.
        mode: ``"isothermal"`` (outlet at ``T_out``, or the inlet T) or
            ``"adiabatic"`` (outlet T from the energy balance).
        T_out: Outlet temperature (K) in isothermal mode; None keeps the
            inlet temperature.
        heat_duty: Heat added (W) in adiabatic mode.
        pressure_drop: Pressure drop (Pa); equilibrium is at the outlet
            pressure.
        approach_dT: Overall approach to equilibrium (K): the standard Gibbs
            energies are evaluated at ``T + approach_dT``. Negative for an
            exothermic reaction that falls short, positive for an
            endothermic one.
        reactions: Optional list of ``{species: nu}``. The outlet is then
            restricted to these reactions (see the module docstring).
        reaction_approach_dT: Per-reaction approach (K), one per entry of
            ``reactions``.
        max_iter: Newton iterations.
        tol: Convergence tolerance on the scaled residual.
    """

    thermo: Any
    mode: str = "isothermal"
    T_out: float | None = None
    heat_duty: float = 0.0
    pressure_drop: float = 0.0
    approach_dT: float = 0.0
    reactions: list | None = None
    reaction_approach_dT: list | None = None
    max_iter: int = 200
    tol: float = 1e-11

    def __post_init__(self):
        if self.mode not in ("isothermal", "adiabatic"):
            raise ValueError(f"mode must be 'isothermal' or 'adiabatic', not {self.mode!r}")
        if self.reaction_approach_dT is not None and self.reactions is None:
            raise ValueError("reaction_approach_dT needs reactions")


def _independent_rows(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """The rows of ``A`` that are linearly independent, kept in order (concrete).

    A dropped (dependent) element is balanced whenever the kept ones are.
    """
    kept = []
    for row in A:
        if not np.any(row):
            continue
        trial = np.array(kept + [row])
        if np.linalg.matrix_rank(trial, tol=tol) == len(trial):
            kept.append(row)
    return np.array(kept).reshape(-1, A.shape[1])


def _null_basis_left(nu: np.ndarray) -> np.ndarray:
    """Rows spanning ``{b : b @ nu = 0}`` for ``nu`` of shape ``(n_species, n_rxn)``."""
    u, s, _ = np.linalg.svd(nu, full_matrices=True)
    rank = int(np.sum(s > 1e-10 * max(s.max(), 1.0)))
    return u[:, rank:].T


class GibbsReactor:
    """Equilibrium reactor by Gibbs-energy minimisation (element-potential method).

    ``GibbsReactor(params)(inlet) -> (outlet, info)``. The outlet carries every
    species of ``params.thermo``. The inlet must not carry a species the
    thermo lacks.

    ``info`` contains:

    - ``converged``: whether the scaled residual met ``tol``.
    - ``residual``: the residual's infinity norm.
    - ``iterations``.
    - ``element_potentials``: ``lambda``, one per conserved row.
    - ``T``: the outlet temperature.
    - ``heat_duty``: W, ``H_out - H_in``.
    - ``element_balance``: per element, outlet minus inlet (mol/s).

    Example:
        >>> from difflow import make_stream, GibbsReactor, GibbsReactorParams, GibbsThermo
        >>> th = GibbsThermo(["A", "B"], Hf=[0.0, 0.0], S0=[0.0, 8.314462618 * 2.0],
        ...                  cp=[[0, 0, 0, 0]] * 2, elements={"A": {"X": 1}, "B": {"X": 1}})
        >>> out, info = GibbsReactor(GibbsReactorParams(thermo=th))(
        ...     make_stream({"A": 1.0}, T=500.0, P=1e5))
        >>> round(float(out["F_B"] / out["F_A"]), 6)       # K = exp(2) for A <-> B
        7.389056
    """

    symbol = "Gibbs Reactor"
    equations = [
        r"\min_n \sum_i n_i\left[g_i^\circ/RT + \ln(n_i/N) + \ln(P/P^\circ)\right]",
        r"A\,n = A\,n^{\mathrm{in}}",
        r"n_i = N\exp\left(\sum_k A_{ki}\lambda_k - g_i^\circ/RT - \ln P/P^\circ\right)",
    ]
    assumptions = [
        "Ideal-gas mixture, 1 bar standard state; one phase.",
        "Equilibrium at the outlet T and P (approach temperatures optional).",
        "Adiabatic mode: no heat loss other than the given heat_duty.",
    ]
    references = [
        "Smith, Missen. Chemical Reaction Equilibrium Analysis: Theory and Algorithms, Wiley 1982.",
        "Reynolds. The element potential method for chemical equilibrium analysis (STANJAN), "
        "Stanford 1986.",
    ]
    parameter_symbols = {"approach_dT": r"\Delta T_{app}", "pressure_drop": r"\Delta P",
                         "heat_duty": "Q"}
    parameter_units = {"T_out": "K", "heat_duty": "W", "pressure_drop": "Pa",
                       "approach_dT": "K", "reaction_approach_dT": "K",
                       "max_iter": "-", "tol": "-"}
    numerical_method = ("Element-potential Newton with backtracking; implicit-function "
                        "gradients at the converged point.")

    def __init__(self, params: GibbsReactorParams):
        """Build the reactor.

        Args:
            params: Thermodynamics, mode and specifications.
        """
        self.params = params
        th = params.thermo
        self.names = tuple(th.names)
        n = len(self.names)
        self.A = np.array([np.asarray(v, dtype=float) for v in th.ELEMENTS.values()]).reshape(-1, n)
        self.element_names = tuple(th.ELEMENTS)
        if params.reactions is not None:
            idx = {s: i for i, s in enumerate(self.names)}
            nu = np.zeros((n, len(params.reactions)))
            for r, rx in enumerate(params.reactions):
                for s, v in rx.items():
                    if s not in idx:
                        raise ValueError(f"reaction {r} names {s!r}, which the thermo lacks")
                    nu[idx[s], r] = float(v)
            if nu.size and np.abs(self.A @ nu).max() > 1e-9:
                raise ValueError("a reaction does not balance the elements")
            if params.reaction_approach_dT is not None and \
                    len(np.atleast_1d(params.reaction_approach_dT)) != len(params.reactions):
                raise ValueError("reaction_approach_dT needs one value per reaction")
            self.nu = nu
            # species in no reaction are inerts: they pass through unsolved
            self.active = np.flatnonzero(np.any(nu != 0, axis=1))
            self.B = _null_basis_left(nu[self.active])
            self.masking = False
        else:
            self.nu = None
            self.active = np.arange(n)
            self.B = _independent_rows(self.A)
            # with element rows a species whose element is not fed is exactly 0
            self.masking = True
        self.inert = np.setdiff1d(np.arange(n), self.active)

    # -- the residual -----------------------------------------------------
    def _g_over_RT(self, T: Array) -> Array:
        """``g/RT`` of every species, approach temperatures applied, ``(n,)``."""
        p = self.params
        th = p.thermo
        Ta = T + p.approach_dT
        g = th.gibbs(Ta) / (R_GAS * Ta)
        if self.nu is not None and p.reaction_approach_dT is not None:
            dT = jnp.atleast_1d(jnp.asarray(p.reaction_approach_dT, dtype=float))
            nu = jnp.asarray(self.nu)
            g_r = jax.vmap(lambda t: th.gibbs(t) / (R_GAS * t))(Ta + dT)  # (m, n)
            c = jnp.sum(nu.T * (g_r - g[None, :]), axis=1)               # nu_r . (g(T_r) - g(T))
            g = g + nu @ jnp.linalg.solve(nu.T @ nu, c)
        return g

    def _absent(self, theta: dict) -> tuple[Array, Array]:
        """Rows with nothing (or a trace) fed, and the species they rule out."""
        m = self.B.shape[0]
        if not self.masking:
            return jnp.zeros(m, bool), jnp.zeros(len(self.active), bool)
        b = lax.stop_gradient(jnp.asarray(self.B) @ theta["n_in"][self.active])
        absent = b <= TRACE_ELEMENT * jnp.sum(b)
        excluded = jnp.any(absent[:, None] & (jnp.asarray(self.B) > 0), axis=0)
        return absent, excluded

    def _moles(self, z: Array, T: Array, theta: dict) -> Array:
        """Moles of every species (inerts as fed) at the potentials ``z``."""
        m = self.B.shape[0]
        lam, lnN = z[:m], z[m]
        g = self._g_over_RT(T)[self.active]
        arg = lnN + jnp.asarray(self.B).T @ lam - g - jnp.log(theta["P"] / P_STANDARD)
        _, excluded = self._absent(theta)
        n_act = jnp.where(excluded, 0.0, jnp.exp(jnp.clip(arg, -700.0, 300.0)))
        return theta["n_in"].at[self.active].set(n_act)

    def _T(self, z: Array, theta: dict) -> Array:
        m = self.B.shape[0]
        return theta["T"] if self.params.mode == "isothermal" else z[m + 1] * 1000.0

    def _residual(self, z: Array, theta: dict) -> Array:
        p = self.params
        m = self.B.shape[0]
        B = jnp.asarray(self.B)
        T = self._T(z, theta)
        n = self._moles(z, T, theta)
        n_in_act = theta["n_in"][self.active]
        absent, _ = self._absent(theta)
        scale = jnp.where(absent, 1.0, jnp.abs(B) @ n_in_act + 1e-300)
        rows = jnp.where(absent, z[:m], (B @ n[self.active] - B @ n_in_act) / scale)
        r = [rows, jnp.atleast_1d(jnp.sum(n) / jnp.exp(z[m]) - 1.0)]
        if p.mode == "adiabatic":
            H_out = jnp.sum(n * p.thermo.enthalpy(T))
            r.append(jnp.atleast_1d((H_out - theta["H_in"] - theta["Q"])
                                    / (jnp.sum(theta["n_in"]) * R_GAS * 1000.0)))
        return jnp.concatenate(r)

    # -- Newton ----------------------------------------------------------
    def _initial(self, theta: dict) -> Array:
        T0 = theta["T"]
        n_in = theta["n_in"]
        N0 = jnp.sum(n_in)
        n_act = n_in[self.active]
        y = (n_act + 1e-3 * N0 / n_act.shape[0]) / (N0 * (1.0 + 1e-3))
        rhs = (self._g_over_RT(T0)[self.active] + jnp.log(theta["P"] / P_STANDARD)
               + jnp.log(y))
        lam = jnp.linalg.lstsq(jnp.asarray(self.B).T, rhs)[0]
        absent, _ = self._absent(theta)
        z = jnp.concatenate([jnp.where(absent, 0.0, lam), jnp.atleast_1d(jnp.log(N0))])
        if self.params.mode == "adiabatic":
            z = jnp.concatenate([z, jnp.atleast_1d(T0 / 1000.0)])
        return z

    def _newton(self, theta: dict):
        p = self.params
        res = lambda z: self._residual(z, theta)            # noqa: E731
        alphas = 0.5 ** jnp.arange(12.0)
        m = self.B.shape[0]
        cap = jnp.concatenate([jnp.full(m + 1, 2.0),
                               jnp.full(1 if p.mode == "adiabatic" else 0, 0.1)])

        def norm(z):
            r = res(z)
            return jnp.where(jnp.all(jnp.isfinite(r)), jnp.max(jnp.abs(r)), jnp.inf)

        def body(state):
            z, k, _ = state
            r = res(z)
            J = jax.jacfwd(res)(z)
            dz = -jnp.linalg.lstsq(J, r)[0]
            dz = dz / jnp.maximum(1.0, jnp.max(jnp.abs(dz) / cap))
            trials = jax.vmap(lambda a: norm(z + a * dz))(alphas)
            ok = trials < norm(z)
            a = jnp.where(jnp.any(ok), alphas[jnp.argmax(ok)], alphas[-1])
            z_new = z + a * dz
            return z_new, k + 1, norm(z_new)

        def cond(state):
            _, k, err = state
            return (k < p.max_iter) & (err > p.tol)

        z0 = self._initial(theta)
        z, k, err = lax.while_loop(cond, body, (z0, jnp.asarray(0), norm(z0)))
        return z, k, err

    # -- the unit --------------------------------------------------------
    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        """Equilibrate the inlet.

        Args:
            inlet: Feed stream; every species must be in ``params.thermo``.

        Returns:
            outlet: Equilibrium effluent (every species of the thermo).
            info: Convergence, element potentials, T, duty, element balance.
        """
        p = self.params
        flows = dict(get_flows(inlet))
        extra = [s for s in flows if s not in self.names]
        if extra:
            raise ValueError(f"inlet species {extra} are not in the reactor's thermo "
                             f"({', '.join(self.names)})")
        n_in = jnp.stack([jnp.asarray(flows.get(s, 0.0), dtype=float) for s in self.names])
        T_in = jnp.asarray(inlet["T"], dtype=float)
        P = jnp.asarray(inlet["P"], dtype=float) - p.pressure_drop
        H_in = jnp.sum(n_in * p.thermo.enthalpy(T_in))
        if p.mode == "isothermal" and p.T_out is not None:
            T = jnp.asarray(p.T_out, dtype=float)
        else:
            T = T_in
        theta = {"n_in": n_in, "T": T, "P": P, "H_in": H_in,
                 "Q": jnp.asarray(p.heat_duty, dtype=float)}

        theta_sg = jax.tree_util.tree_map(lax.stop_gradient, theta)
        z_star, iters, err = self._newton(theta_sg)
        z_star = lax.stop_gradient(z_star)
        J = lax.stop_gradient(jax.jacfwd(lambda z: self._residual(z, theta_sg))(z_star))
        # one Newton correction with J held fixed: exact value, implicit derivatives
        z = z_star - jnp.linalg.lstsq(J, self._residual(z_star, theta))[0]

        m = self.B.shape[0]
        T_out = self._T(z, theta)
        n = self._moles(z, T_out, theta)
        A = jnp.asarray(self.A)
        info = {
            "converged": err <= p.tol,
            "residual": err,
            "iterations": iters,
            "element_potentials": z[:m],
            "T": T_out,
            "heat_duty": jnp.sum(n * p.thermo.enthalpy(T_out)) - H_in,
            "element_balance": dict(zip(self.element_names, A @ n - A @ n_in)),
        }
        return make_stream(dict(zip(self.names, n)), T_out, P), info


__all__ = ["GibbsThermo", "GibbsReactorParams", "GibbsReactor", "R_GAS", "P_STANDARD", "TRACE_ELEMENT"]
