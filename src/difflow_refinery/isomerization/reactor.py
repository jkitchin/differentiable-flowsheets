"""Adiabatic C5/C6 isomerization reactor: approach-to-equilibrium kinetics.

A pseudo-homogeneous, adiabatic plug-flow bed. Every reversible reaction
``A (+ n H2) <=> B`` runs at a first-order *approach-to-equilibrium* rate::

    r_j = theta k_j(T) (F_A - F_B / (K_j(T) p_H2^n))          (mol/s)
    k_j(T) = k_j,ref exp(-Ea_j / R (1/T - 1/T_ref))
    dF_i / dz = sum_j nu_ij r_j,     z in [0, 1] along the bed
    sum_i F_i(z) H_i(T(z)) = sum_i F_i(0) H_i(T_in)            (adiabatic)

with ``theta = 1 / LHSV`` (h) and ``k`` in 1/h, ``K_j`` from
:mod:`.thermochem` and ``p_H2`` in bar. The rate vanishes exactly at
equilibrium, whatever ``k``, so the equilibrium is set by the
thermochemistry alone and the rate constants only set how close to it the
bed gets. The temperature is not integrated: at every point it is the root
of the energy balance, so the bed's enthalpy is conserved exactly however
coarse the steps. Cracking is irreversible and first order in the paraffin.

Reactions (:data:`REACTIONS`):

* C5: ``nC5 <=> iC5`` (neopentane is not formed over these catalysts and
  is not carried);
* C6 paraffins: ``nC6 <=> 2MP``, ``2MP <=> 3MP``, ``2MP <=> 23DMB``,
  ``23DMB <=> 22DMB`` -- the 2,2-DMB step is the slow one;
* naphthenes: ``MCP <=> CH``;
* benzene saturation ``Bz + 3 H2 <=> CH``;
* ring opening ``MCP + H2 <=> 2MP``;
* hydrocracking, a small selectivity loss: ``C6H14 + H2 -> 2 C3H8`` and
  ``C5H12 + H2 -> C2H6 + C3H8``, one rate constant for every paraffin of
  a carbon number.

The C7+ lump is inert.

**The rate constants are illustrative.** The catalyst presets
(:data:`CATALYSTS`) give orders of magnitude and the temperature windows
in which chlorided-alumina, sulfated-zirconia and zeolitic catalysts are
run; they are not fitted to any catalyst's data. Calibrate ``k_scale``
(one factor on every isomerization rate) against a plant's measured
approach to equilibrium before trusting an absolute conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from difflow.params_mixin import ParamsMixin
from difflow_refinery.isomerization import thermochem as tc

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Reaction:
    """One reaction of the network.

    Attributes:
        name: Identifier (``"nC6-2MP"``).
        nu: ``{species label: coefficient}`` (negative for reactants).
        reactant: Label of the species the forward rate is first order in.
        product: Label of the species in the reverse term (None if
            irreversible).
        h2_order: Hydrogen's coefficient on the reactant side (the ``n`` of
            ``p_H2^n`` in the reverse term).
        kind: ``"isom"``, ``"hydrogenation"``, ``"ring_opening"`` or
            ``"cracking"``.
    """

    name: str
    nu: tuple
    reactant: str
    product: Optional[str]
    h2_order: int
    kind: str


def _crack(name, fam_c):
    if fam_c == 6:
        return Reaction(f"crack-{name}", ((name, -1), ("H2", -1), ("C3", 2)), name, None, 1,
                        "cracking")
    return Reaction(f"crack-{name}", ((name, -1), ("H2", -1), ("C2", 1), ("C3", 1)), name, None,
                    1, "cracking")


#: The reaction network, in rate-vector order.
REACTIONS: tuple[Reaction, ...] = (
    Reaction("nC5-iC5", (("nC5", -1), ("iC5", 1)), "nC5", "iC5", 0, "isom"),
    Reaction("nC6-2MP", (("nC6", -1), ("2MP", 1)), "nC6", "2MP", 0, "isom"),
    Reaction("2MP-3MP", (("2MP", -1), ("3MP", 1)), "2MP", "3MP", 0, "isom"),
    Reaction("2MP-23DMB", (("2MP", -1), ("23DMB", 1)), "2MP", "23DMB", 0, "isom"),
    Reaction("23DMB-22DMB", (("23DMB", -1), ("22DMB", 1)), "23DMB", "22DMB", 0, "isom"),
    Reaction("MCP-CH", (("MCP", -1), ("CH", 1)), "MCP", "CH", 0, "isom"),
    Reaction("Bz-CH", (("Bz", -1), ("H2", -3), ("CH", 1)), "Bz", "CH", 3, "hydrogenation"),
    Reaction("MCP-2MP", (("MCP", -1), ("H2", -1), ("2MP", 1)), "MCP", "2MP", 1, "ring_opening"),
    _crack("nC5", 5), _crack("iC5", 5),
    _crack("nC6", 6), _crack("2MP", 6), _crack("3MP", 6), _crack("23DMB", 6), _crack("22DMB", 6),
)
RXN_NAMES = tuple(r.name for r in REACTIONS)
N_RXN = len(REACTIONS)
N_SP = len(tc.SPECIES)


def _stoich():
    nu = np.zeros((N_RXN, N_SP))
    for j, r in enumerate(REACTIONS):
        for lab, c in r.nu:
            nu[j, tc.idx(lab)] = c
    return nu


#: Stoichiometric matrix ``(n_rxn, n_species)``.
NU = jnp.asarray(_stoich())
_REACT = jnp.asarray([tc.idx(r.reactant) for r in REACTIONS])
_PROD = jnp.asarray([tc.idx(r.product) if r.product else 0 for r in REACTIONS])
_REV = jnp.asarray([0.0 if r.product is None else 1.0 for r in REACTIONS])
_H2N = jnp.asarray([float(r.h2_order) for r in REACTIONS])
_KINDS = tuple(r.kind for r in REACTIONS)


def _kind_mask(kind):
    return jnp.asarray([1.0 if k == kind else 0.0 for k in _KINDS])


def assert_balanced() -> None:
    """Raise if any reaction fails to balance C or H."""
    for j, r in enumerate(REACTIONS):
        for atoms, el in ((tc.C_ATOMS, "C"), (tc.H_ATOMS, "H")):
            if abs(float(NU[j] @ atoms)) > 1e-12:
                raise ValueError(f"reaction {r.name} does not balance {el}")


#: Illustrative kinetic presets: ``T_ref`` (K), ``k_ref`` (1/h at T_ref) and
#: ``Ea`` (J/mol) by reaction kind or name, and a typical operating window.
CATALYSTS: dict[str, dict] = {
    "chlorided_alumina": dict(
        T_ref=413.15, window=(393.15, 473.15), H2_HC=0.1,
        k_ref={"nC5-iC5": 3.0, "nC6-2MP": 6.0, "2MP-3MP": 12.0, "2MP-23DMB": 3.0,
               "23DMB-22DMB": 0.8, "MCP-CH": 4.0, "Bz-CH": 40.0, "MCP-2MP": 0.01,
               "cracking": 5e-5},
        Ea={"isom": 100e3, "hydrogenation": 50e3, "ring_opening": 100e3, "cracking": 150e3},
        description="chlorided Pt/alumina (Penex-type): most active, 120-200 C; "
                    "illustrative constants"),
    "sulfated_zirconia": dict(
        T_ref=463.15, window=(433.15, 513.15), H2_HC=1.0,
        k_ref={"nC5-iC5": 2.5, "nC6-2MP": 5.0, "2MP-3MP": 10.0, "2MP-23DMB": 2.5,
               "23DMB-22DMB": 0.6, "MCP-CH": 4.0, "Bz-CH": 40.0, "MCP-2MP": 0.01,
               "cracking": 5e-5},
        Ea={"isom": 100e3, "hydrogenation": 50e3, "ring_opening": 100e3, "cracking": 150e3},
        description="sulfated zirconia (Par-Isom-type): 160-240 C; illustrative constants"),
    "zeolite": dict(
        T_ref=533.15, window=(503.15, 563.15), H2_HC=2.0,
        k_ref={"nC5-iC5": 2.5, "nC6-2MP": 5.0, "2MP-3MP": 10.0, "2MP-23DMB": 2.0,
               "23DMB-22DMB": 0.5, "MCP-CH": 4.0, "Bz-CH": 40.0, "MCP-2MP": 0.01,
               "cracking": 5e-5},
        Ea={"isom": 100e3, "hydrogenation": 50e3, "ring_opening": 100e3, "cracking": 150e3},
        description="Pt/mordenite zeolite (Hysomer-type): 230-290 C, sulfur tolerant; "
                    "illustrative constants"),
}


def _kinetic_arrays(catalyst: str, k_scale, crack_scale):
    cat = CATALYSTS[catalyst]
    k = np.array([cat["k_ref"].get(r.name, cat["k_ref"].get(r.kind)) for r in REACTIONS],
                 dtype=float)
    Ea = np.array([cat["Ea"][r.kind] for r in REACTIONS], dtype=float)
    scale = (jnp.asarray(k_scale, dtype=float) * (1.0 - _kind_mask("cracking"))
             + jnp.asarray(crack_scale, dtype=float) * _kind_mask("cracking"))
    return jnp.asarray(k) * scale, jnp.asarray(Ea), cat["T_ref"]


@dataclass
class IsomerizationReactorParams(ParamsMixin):
    """Parameters for :class:`IsomerizationReactor`.

    Attributes:
        LHSV: Liquid hourly space velocity of the hydrocarbon entering the
            bed (1/h; liquid volume at 60 F per catalyst volume per hour).
        catalyst: A key of :data:`CATALYSTS` (illustrative presets).
        k_scale: Factor on every isomerization, saturation and ring-opening
            rate constant -- the activity knob to fit to a measured approach
            to equilibrium.
        crack_scale: Factor on the hydrocracking rate constant.
        P: Reactor pressure (Pa); sets the hydrogen partial pressure.
        n_steps: Fixed steps of the implicit (L-stable SDIRK) integrator
            along the bed.
        adiabatic: Integrate the energy balance; False holds the inlet T.
    """

    LHSV: float = 2.0
    catalyst: str = "chlorided_alumina"
    k_scale: float = 1.0
    crack_scale: float = 1.0
    P: float = 30e5
    n_steps: int = 40
    adiabatic: bool = True

    def __post_init__(self):
        if self.catalyst not in CATALYSTS:
            raise ValueError(f"catalyst must be one of {sorted(CATALYSTS)}, "
                             f"not {self.catalyst!r}")


def flows_of(stream: Mapping) -> jax.Array:
    """Molar flows of a stream in reactor order (missing species are 0)."""
    unknown = [k[2:] for k in stream if k.startswith("F_") and k[2:] not in tc.NAMES]
    if unknown:
        raise ValueError(f"species {unknown} are not isomerization species {tc.NAMES}")
    return jnp.stack([jnp.asarray(stream.get(f"F_{n}", 0.0), dtype=float) for n in tc.NAMES])


def stream_of(F, T, P, phase=None) -> dict:
    """A difflow stream from reactor-order flows."""
    s = {f"F_{n}": F[i] for i, n in enumerate(tc.NAMES)}
    s.update(T=jnp.asarray(T, dtype=float), P=jnp.asarray(P, dtype=float))
    if phase is not None:
        s["phase"] = phase
    return s


def hydrocarbon_volume(F) -> jax.Array:
    """Liquid volume flow (m^3/h at 60 F) of the hydrocarbons in ``F``
    (hydrogen, ethane and propane excluded)."""
    w = jnp.ones(N_SP).at[jnp.asarray([tc.idx("H2"), tc.idx("C2"), tc.idx("C3")])].set(0.0)
    return 3600.0 * jnp.sum(w * tc.liquid_volume(F))


def _rates(F, T, P, theta, k_ref, Ea, T_ref):
    lnK = -(NU @ tc.gibbs(T)) / (tc.R * T)
    k = theta * k_ref * jnp.exp(-Ea / tc.R * (1.0 / T - 1.0 / T_ref))
    Ftot = jnp.sum(F)
    pH2 = jnp.maximum(F[tc.idx("H2")] / Ftot * P / 1e5, 1e-2)       # bar
    FA = F[_REACT]
    FB = F[_PROD]
    reverse = _REV * FB * jnp.exp(-lnK - _H2N * jnp.log(pH2))
    # A reaction that consumes hydrogen stops as the hydrogen runs out:
    # without this a hydrogen-starved bed drives F_H2 negative.
    FH2 = jnp.maximum(F[tc.idx("H2")], 0.0)
    avail = jnp.where(_H2N > 0, FH2 / (FH2 + 1e-3 * Ftot), 1.0)
    return k * avail * (FA - reverse), k


#: Width of the band over which a step blends from SDIRK to backward Euler
#: (the smallest ratio of a second-stage explicit flow to the step's
#: starting flow; see :func:`integrate`).
BLEND = 0.05


class IsomerizationConvergenceWarning(RuntimeWarning):
    """An implicit integration stage of the reactor did not converge; the
    outlet is not a solution of the bed equations. More ``n_steps`` fixes it."""


def integrate(F0, T0, P, theta, k_ref, Ea, T_ref, n_steps: int, adiabatic: bool = True,
              max_newton: int = 100, newton_tol: float = 1e-12, n_newton_T: int = 8):
    """Integrate along the bed in ``n_steps`` fixed implicit steps.

    Returns ``(F, T, extents, residual)`` at the outlet, the last the
    largest scaled residual of any stage's implicit equation (``inf`` if a
    stage failed): a stage that has not converged shows here.

    The scheme is Alexander's (1977) two-stage, second-order, L-stable
    SDIRK; the fast reactions (benzene saturation, the naphthene and C5
    isomerizations) are stiff against the slow 2,2-DMB step. Each stage is
    a Newton iteration run to ``newton_tol`` on its residual (a
    ``lax.while_loop``, so forward-mode AD only). A Newton step that would
    take a flow through zero is shortened for that species alone, and
    flows are kept non-negative.

    Where SDIRK's second stage starts from an extrapolation that is
    already negative in some flow (a species consumed to near zero within
    the step), its root is too, and the stage cannot converge on the
    non-negative side. The step is then blended continuously into
    backward Euler, which is first order but positivity-preserving:
    ``w = clip(min_i c2_i / F_i / BLEND, 0, 1)`` weights SDIRK. Both
    schemes are Runge-Kutta methods, so any blend conserves every linear
    invariant (mass, atoms, the carbon-number counts) exactly, and the
    weight is continuous, so the outlet is differentiable through the
    switch -- a hard switch makes it jump."""
    T0 = jnp.asarray(T0, dtype=float)
    H0 = jnp.dot(F0, tc.enthalpy(T0))

    def T_of(F, guess):
        # The adiabatic bed's temperature is not integrated: it is the root
        # of sum_i F_i H_i(T) = H0, so the energy balance holds exactly at
        # every stage. Integrating dT/dz instead leaves an error where a
        # fast reaction runs to equilibrium inside one step (4.5 kW, 0.17 K,
        # on the paraffinic charge at equilibrium).
        def newton(_, T):
            return T - (jnp.dot(F, tc.enthalpy(T)) - H0) / jnp.dot(F, tc.cp(T))

        return jax.lax.fori_loop(0, n_newton_T, newton, guess)

    def temperature(y):
        return T_of(y[:N_SP], y[N_SP]) if adiabatic else y[N_SP]

    def rhs(y):
        F, T = y[:N_SP], temperature(y)
        r, _ = _rates(F, T, P, theta, k_ref, Ea, T_ref)
        return jnp.concatenate([NU.T @ r, jnp.zeros(1), r])

    h = 1.0 / n_steps
    g = 1.0 - 1.0 / jnp.sqrt(2.0)
    n_y = N_SP + 1 + N_RXN
    eye = jnp.eye(n_y)
    jac = jax.jacfwd(rhs)
    scale = jnp.concatenate([jnp.full(N_SP + 1, 1.0), jnp.full(N_RXN, 1.0)])
    scale = scale.at[:N_SP].set(jnp.sum(F0)).at[N_SP + 1:].set(jnp.sum(F0)).at[N_SP].set(T0)

    def stage(c, guess, a):
        # Solve Y = c + a f(Y) by Newton, iterated to a tolerance (a
        # while_loop: forward-mode AD runs straight through it; reverse
        # mode does not, so differentiate this reactor with jacfwd/jvp).
        def residual(Y):
            return Y - c - a * rhs(Y)

        def size(Y):
            # Scaled residual; the T slot is only the next guess for
            # temperature() (its rhs is zero), so it is left out.
            G = jnp.abs(residual(Y)) / scale
            return jnp.where(jnp.arange(n_y) == N_SP, 0.0, G).max()

        def newton(state):
            k, Y, _ = state
            d = -jnp.linalg.solve(eye - a * jac(Y), residual(Y))
            # Fraction to the boundary: a full step on a stiff bed (high
            # activity, low LHSV, a benzene-rich charge) can overshoot a
            # flow far below zero, after which the rate law has no
            # equilibrium to return to. The cost is that a species heading
            # to zero shrinks by 95 % per iteration while it throttles the
            # rest, which is why the loop runs to a tolerance rather than a
            # fixed count. At convergence alpha = 1: the converged answer
            # and its derivatives are Newton's.
            # Only a flow whose full step would cross zero limits alpha, and
            # one already at round-off is clipped rather than allowed to
            # throttle the rest (it would hold alpha near zero for good).
            F, dF = Y[:N_SP], d[:N_SP]
            floor = 1e-14 * jnp.sum(jnp.abs(F))
            limits = (F + dF < 0.0) & (F > floor)
            alpha = jnp.minimum(1.0, 0.95 * jnp.min(jnp.where(limits, F / jnp.maximum(-dF, 1e-300),
                                                                jnp.inf)))
            Yn = Y + alpha * d
            Yn = Yn.at[:N_SP].set(jnp.maximum(Yn[:N_SP], 0.0))
            # Converged is judged on the RESIDUAL, not the step: a step cut
            # short by the limiter is small without being near the root.
            return k + 1, Yn, size(Yn)

        def going(state):
            k, _, err = state
            return (k < max_newton) & (err > newton_tol)

        _, Y, _ = jax.lax.while_loop(going, newton, (0, guess, jnp.inf))
        # Two exact Newton steps at the end: the while_loop's exit test is
        # on a primal value, so these make the tangent the converged one.
        # A step that would take a flow negative is not taken (the stage
        # then reports its residual, below, rather than a negative flow).
        for _ in range(2):
            Yp = Y - jnp.linalg.solve(eye - a * jac(Y), residual(Y))
            ok = jnp.all(Yp[:N_SP] >= -1e-9 * jnp.sum(jnp.abs(Y[:N_SP])))
            Y = jnp.where(ok, Yp, Y)
        return Y, jax.lax.stop_gradient(size(Y))

    def step(_, carry):
        y, worst = carry
        # Two-stage, stiffly accurate, L-stable SDIRK (order 2; Alexander
        # 1977), blended into backward Euler on a step it cannot take. L-stability is what lets a stiff pair -- a fast step near
        # equilibrium -- relax onto its equilibrium instead of oscillating.
        Y1, e1 = stage(y, y, h * g)
        c2 = y + h * (1.0 - g) * rhs(Y1)
        # The second stage's explicit part is an extrapolation,
        # c2 = y + ((1 - g)/g)(Y1 - y) with (1 - g)/g = 2.4, so where a
        # species is consumed fast within the step (benzene saturating) c2
        # goes negative and so does the stage's root: the SDIRK step does
        # not preserve positivity. Such a step is taken by backward Euler
        # instead (first order, L-stable, positivity preserving), and to
        # keep the outlet a CONTINUOUS function of the inputs -- a hard
        # switch puts a jump in it wherever a perturbation moves one step
        # across, which finite differences see as a 3 % gradient error --
        # the two are blended over a band: weight w on SDIRK, from 1 where
        # every c2 flow is at least BLEND of its value at the step's start,
        # to 0 where one reaches zero. Both are Runge-Kutta steps, so both
        # conserve every linear invariant (atoms, carbon number), and so
        # does the blend.
        F, c2F = y[:N_SP], c2[:N_SP]
        ratio = jnp.where(F > 1e-12 * jnp.sum(F), c2F / jnp.maximum(F, 1e-300), jnp.inf)
        w = jnp.clip(jnp.min(ratio) / BLEND, 0.0, 1.0)

        def sdirk():
            return stage(c2, Y1, h * g)

        def euler():
            return stage(y, Y1, h)

        def both():
            (Ys, es), (Ye, ee) = sdirk(), euler()
            return w * Ys + (1.0 - w) * Ye, jnp.maximum(es, ee)

        Y2, e2 = jax.lax.cond(w >= 1.0, sdirk,
                              lambda: jax.lax.cond(w <= 0.0, euler, both))
        # keep the T slot as the next guess; a NaN step error counts as failed
        worst = jnp.maximum(worst, jnp.where(jnp.isfinite(e1) & jnp.isfinite(e2),
                                             jnp.maximum(e1, e2), jnp.inf))
        return Y2.at[N_SP].set(temperature(Y2)), worst

    y0 = jnp.concatenate([F0, T0[None], jnp.zeros(N_RXN)])
    y, worst = jax.lax.fori_loop(0, n_steps, step, (y0, jnp.asarray(0.0)))
    return y[:N_SP], temperature(y), y[N_SP + 1:], worst


class IsomerizationReactor:
    """Adiabatic C5/C6 paraffin isomerization reactor (see module docstring).

    Differentiable (forward mode: jacfwd, jvp) in the inlet stream (flows,
    T, P) and every parameter;
    ``__call__`` returns ``(outlet, info)``.

    Example:
        >>> from difflow_refinery.isomerization import (IsomerizationReactor,
        ...     IsomerizationReactorParams)
        >>> rx = IsomerizationReactor(IsomerizationReactorParams(LHSV=2.0))
        >>> feed = {"F_hydrogen": 5.0, "F_n_pentane": 30.0, "F_n_hexane": 20.0,
        ...         "T": 413.15, "P": 30e5}
        >>> out, info = rx(feed)
        >>> float(info["approach"]["nC5-iC5"]) < 1.0
        True
    """

    symbol = "ISOM"
    equations = [
        r"r_j = \theta\, k_j(T)\,\left(F_A - \frac{F_B}{K_j(T)\, p_{H_2}^{n_j}}\right)",
        r"\frac{dF_i}{dz} = \sum_j \nu_{ij} r_j",
        r"\sum_i F_i(z) H_i(T(z)) = \sum_i F_i(0) H_i(T_{in})",
        r"\ln K_j = -\sum_i \nu_{ij} G_i(T) / RT",
    ]
    assumptions = [
        "Pseudo-homogeneous plug flow, adiabatic, no pressure drop",
        "Ideal-gas reaction thermochemistry (dHf, S, Cp of every species)",
        "First-order approach-to-equilibrium rates; ILLUSTRATIVE rate constants",
        "Neopentane not formed; C7+ inert; one cracking constant per carbon number",
    ]
    references = [
        "Prosen, E. J.; Rossini, F. D. (1945) J. Res. NBS 34, 263-269",
        "Scott, D. W. (1974) Chemical thermodynamic properties of hydrocarbons and "
        "related substances, US Bureau of Mines Bulletin 666",
        "Meyers, R. A. (ed.) (2004) Handbook of Petroleum Refining Processes, 3rd ed., "
        "McGraw-Hill, Part 9 (isomerization)",
    ]
    parameter_units = {"LHSV": "1/h", "k_scale": "-", "crack_scale": "-", "P": "Pa",
                       "n_steps": "-"}

    def __init__(self, params: IsomerizationReactorParams):
        self.params = params

    def run(self, F0, T0, LHSV=None, k_scale=None, crack_scale=None, P=None):
        """Integrate the bed from reactor-order flows; returns a result dict."""
        p = self.params
        LHSV = p.LHSV if LHSV is None else LHSV
        P = jnp.asarray(p.P if P is None else P, dtype=float)
        k_ref, Ea, T_ref = _kinetic_arrays(
            p.catalyst, p.k_scale if k_scale is None else k_scale,
            p.crack_scale if crack_scale is None else crack_scale)
        theta = 1.0 / jnp.asarray(LHSV, dtype=float)
        F0 = jnp.asarray(F0, dtype=float)
        T0 = jnp.asarray(T0, dtype=float)
        F, T, xi, worst = integrate(F0, T0, P, theta, k_ref, Ea, T_ref, int(p.n_steps),
                                    p.adiabatic)
        _, kth = _rates(F0, T0, P, theta, k_ref, Ea, T_ref)
        info = reactor_info(F0, F, T0, T, P, xi)
        info["stage_residual"] = worst
        try:
            bad = float(worst) > 1e-6
        except jax.errors.ConcretizationTypeError:   # traced: the caller reads info
            bad = False
        if bad:
            warnings.warn(f"isomerization reactor: an implicit stage did not converge "
                          f"(scaled residual {float(worst):.2e}); raise n_steps",
                          IsomerizationConvergenceWarning, stacklevel=2)
        return dict(F=F, T=T, P=P, extents=xi, F_in=F0, T_in=T0, k_theta_max=jnp.max(kth),
                    info=info)

    def __call__(self, stream: Mapping):
        res = self.run(flows_of(stream), stream["T"], P=stream.get("P", self.params.P))
        return stream_of(res["F"], res["T"], res["P"], phase="vapor"), res["info"]


def reactor_info(F0, F, T0, T, P, xi) -> dict:
    """Diagnostics of a reactor pass: approach to equilibrium per reaction,
    isomer ratios, hydrogen consumption, gas make, balances."""
    lnK = -(NU @ tc.gibbs(T)) / (tc.R * T)
    pH2 = jnp.maximum(F[tc.idx("H2")] / jnp.sum(F) * P / 1e5, 1e-12)
    approach = {}
    for j, r in enumerate(REACTIONS):
        if r.product is None:
            continue
        Q = F[tc.idx(r.product)] / (F[tc.idx(r.reactant)] * pH2 ** r.h2_order)
        approach[r.name] = Q / jnp.exp(lnK[j])
    c5 = F[tc.idx("iC5")] + F[tc.idx("nC5")]
    c6p = sum(F[tc.idx(n)] for n in tc.FAMILIES["C6P"])
    crack = _kind_mask("cracking")
    mass_in, mass_out = jnp.dot(F0, tc.MW), jnp.dot(F, tc.MW)
    return {
        "approach": approach,
        "iC5/C5": F[tc.idx("iC5")] / c5,
        "22DMB/C6P": F[tc.idx("22DMB")] / c6p,
        "DMB/C6P": (F[tc.idx("22DMB")] + F[tc.idx("23DMB")]) / c6p,
        "H2_consumption": F0[tc.idx("H2")] - F[tc.idx("H2")],
        "H2_HC_out": F[tc.idx("H2")] / (jnp.sum(F) - F[tc.idx("H2")]),
        "gas_make": (F[tc.idx("C2")] - F0[tc.idx("C2")]) * tc.MW[tc.idx("C2")] / 1e3
        + (F[tc.idx("C3")] - F0[tc.idx("C3")]) * tc.MW[tc.idx("C3")] / 1e3,
        "cracking_extent": jnp.dot(crack, xi),
        "benzene_conversion": 1.0 - F[tc.idx("Bz")] / jnp.maximum(F0[tc.idx("Bz")], 1e-30),
        "delta_T": T - T0,
        "T_out": T,
        "extents": dict(zip(RXN_NAMES, xi)),
        "balance": balances(F0, F, xi),
        "mass_in": mass_in / 1e3,
        "mass_out": mass_out / 1e3,
    }


def carbon_number_flows(F) -> dict:
    """Moles of each carbon number (2-7), hydrogen apart."""
    out = {}
    for n in range(2, 8):
        m = jnp.asarray([1.0 if (s.C == n) else 0.0 for s in tc.SPECIES])
        out[n] = jnp.dot(m, F)
    return out


def balances(F0, F, xi) -> dict:
    """Relative closure of the total mass, C and H atoms, and every carbon
    number against the reaction extents (the extents say how many C5 and
    C6 molecules were cracked to C2 and C3)."""
    m_in = jnp.dot(F0, tc.MW)
    out = {
        "mass": (jnp.dot(F, tc.MW) - m_in) / m_in,
        "C": (jnp.dot(F, tc.C_ATOMS) - jnp.dot(F0, tc.C_ATOMS)) / jnp.dot(F0, tc.C_ATOMS),
        "H": (jnp.dot(F, tc.H_ATOMS) - jnp.dot(F0, tc.H_ATOMS)) / jnp.dot(F0, tc.H_ATOMS),
    }
    # Predicted change of each carbon number from the extents, then the
    # residual of in + change - out, relative to that carbon number's feed
    # (or to the total, if it had none).
    n_in, n_out = carbon_number_flows(F0), carbon_number_flows(F)
    change = (NU.T @ xi)
    n_change = carbon_number_flows(change)
    tot = jnp.sum(F0)
    for n in n_in:
        out[f"C{n}"] = (n_in[n] + n_change[n] - n_out[n]) / jnp.maximum(n_in[n], 1e-3 * tot)
    return out


__all__ = [
    "Reaction", "REACTIONS", "RXN_NAMES", "NU", "CATALYSTS", "assert_balanced",
    "IsomerizationReactorParams", "IsomerizationReactor", "IsomerizationConvergenceWarning", "flows_of", "stream_of", "hydrocarbon_volume", "integrate", "reactor_info",
    "carbon_number_flows", "balances",
]
