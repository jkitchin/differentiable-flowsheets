"""Adiabatic trickle-bed reactor: beds in series with gas quench, any kinetic model.

The reactor knows nothing about hydrotreating. It integrates the steady
plug-flow balances of a :class:`~.layout.Flows` stream down a catalyst bed,
asks a **kinetic model** for the rates at every point, and keeps the energy
balance; the hydrotreater (:mod:`difflow_refinery.hydrotreating`) and a
hydrocracker supply different kinetic models to the same reactor.

Balances (independent variable ``w``, catalyst mass from the bed inlet, kg)::

    dF/dw = r(F, T, P)                  every gas, cut and attribute flow
    C_eff(F, T) dT/dw = q(F, T, P)      adiabatic

``r`` (mol/s per kg catalyst) and the heat release ``q`` (W per kg catalyst)
are the kinetic model's. ``C_eff = dH/dT`` at fixed flows is the stream's
effective heat capacity *including* the heat of the vaporisation that a
temperature rise causes (the phase split moves with ``T``); it is computed by
forward-mode AD of the stream enthalpy, not approximated.

Phase model (pseudo-homogeneous, the default of issue #306). Gas and liquid
are in equilibrium at every point. The K-values come from a Peng-Robinson
flash (:func:`.separator.pr_flash`) at the **bed inlet**, and are carried down
the bed linearised in temperature, ``ln K(T) = ln K_in + (d ln K/dT)_in (T -
T_in)`` (the slope by forward-mode AD of the flash). The composition the K
-values were computed at is the inlet's; along the bed only the phase split
moves (Rachford-Rice, negative flash), not the K-values' composition
dependence -- an approximation of order the bed's change in H2/H2S partial
pressures, which is a few per cent. Mass-transfer resistances, wetting and
the catalyst effectiveness factor enter only as multipliers the kinetic
model applies (``wetting``, ``effectiveness`` and ``activity`` in the
hydrotreater's parameters); the Korsten-Hoffmann gas-liquid and liquid-solid
film model is NOT implemented (it is listed as an option in the issue).

What a kinetic model sees (:class:`ReactionContext`): the flows, ``T``,
``P``, the equilibrium liquid and vapour compositions ``x`` and ``y`` of the
flashing components, their **fugacity-equivalent liquid concentrations**
``c = x / v_L`` (mol/m^3; ``v_L`` the liquid's molar volume from the cuts'
Rackett volumes, dissolved gases taking none), the attribute concentrations
in that liquid (``c_attr = c_cut * attribute-per-molecule``), and the partial
pressures ``p = y P``. Concentrations are defined through ``x`` even where
the stream is all vapour: there ``x`` is the incipient (dew-point) liquid in
equilibrium with the vapour, ``z / K``, and ``y`` the vapour itself
(:func:`phase_state`), so a rate law written on them is continuous across a
dry-out -- what a naphtha hydrotreater, whose bed is vapour, needs.

Plugging in a kinetic model: anything with

* ``attributes`` / ``attribute_elements`` -- the per-cut attributes it needs
  (see :class:`~.layout.Layout`), and
* ``rates(ctx: ReactionContext, params) -> Rates``

works. ``Rates.gas``, ``.cut``, ``.attr`` are ``d/dw`` of the flows (mol/s per
kg catalyst) and ``.heat`` the heat released (W per kg). Element
conservation is the kinetic model's responsibility;
:func:`check_element_conservation` tests it at a point.

Integration is ``diffrax`` (Tsit5, PID step control on a state scaled by its
inlet values). Gradients are reverse-mode through the solver with
``RecursiveCheckpointAdjoint`` (diffrax's default: exact derivatives of the
discrete solution, checkpointed rather than stored); ``adjoint="backsolve"``
selects the continuous adjoint (``BacksolveAdjoint``) instead.

Beds and quench. Either the quench into each later bed is given (a fraction
of the treat gas; the first bed gets the rest) and each later bed's inlet
temperature follows from the adiabatic mix, a scalar enthalpy balance; or
every bed's inlet temperature is given and the quench each needs is solved
-- a scalar equation in the total quench, since the first bed's gas is what
the quench leaves. The quench-mixing enthalpy uses the upstream bed's
linearised K-values (the extra vaporisation the quench gas itself causes is
neglected; the next bed's inlet flash resets the phase split). Beds run
under ``lax.scan``: one compiled copy of a bed whatever the bed count.

References:
    Kidger, P., "On Neural Differential Equations", PhD thesis, University of
        Oxford (2021) -- diffrax (Tsit5 after Tsitouras 2011).
    Tsitouras, Ch., "Runge-Kutta pairs of order 5(4) satisfying only the first
        column simplifying assumption", Comput. Math. Appl. 62(2), 770-775
        (2011), doi:10.1016/j.camwa.2011.06.002 (unverified).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow_refinery.hydroprocessing.layout import Flows, Layout
from difflow_refinery.hydroprocessing.separator import (
    flash_components, phase_denominator, pr_flash, rachford_rice)
from difflow_refinery.hydroprocessing.solve import newton_scalar
from difflow_refinery.hydroprocessing.thermo import Components

jax.config.update("jax_enable_x64", True)


# =============================================================================
# The kinetic-model interface
# =============================================================================


@dataclass(frozen=True)
class ReactionContext:
    """Everything a kinetic model may read at one point of a bed (a pytree).

    Attributes:
        layout: The stream layout (static).
        comps: The flashing components' constants (gases except water, then cuts).
        flows: Total flows at this point.
        T, P: K, Pa.
        x, y: ``(n_flash,)`` equilibrium liquid and vapour mole fractions.
        beta: Vapour fraction (negative-flash value).
        c: ``(n_flash,)`` fugacity-equivalent liquid concentrations, mol/m^3.
        c_attr: ``(n_cut, n_attr)`` attribute concentrations in that liquid, mol/m^3.
        p: ``(n_flash,)`` partial pressures ``y P``, Pa.
        v_L: Liquid molar volume, m^3/mol.
        per_molecule: ``(n_cut, n_attr)`` attribute per molecule of each cut.
    """

    layout: Layout
    comps: Components
    flows: Flows
    T: Array
    P: Array
    x: Array
    y: Array
    beta: Array
    c: Array
    c_attr: Array
    p: Array
    v_L: Array
    per_molecule: Array

    def flash_index(self, gas: str) -> int:
        """Index of a gas in the flashing-component arrays (``x``, ``c``, ``p``)."""
        return self.comps.names.index(gas)

    def c_gas(self, gas: str) -> Array:
        """Liquid concentration of a gas (mol/m^3), 0 if the layout lacks it."""
        return self.c[self.flash_index(gas)] if gas in self.comps.names else jnp.asarray(0.0)

    def p_gas(self, gas: str) -> Array:
        """Partial pressure of a gas (Pa), 0 if the layout lacks it."""
        return self.p[self.flash_index(gas)] if gas in self.comps.names else jnp.asarray(0.0)

    @property
    def c_cut(self) -> Array:
        """``(n_cut,)`` liquid concentration of each cut's molecules, mol/m^3."""
        return self.c[self.comps.n_gas:]


jax.tree_util.register_dataclass(
    ReactionContext,
    data_fields=["comps", "flows", "T", "P", "x", "y", "beta", "c", "c_attr", "p", "v_L", "per_molecule"],
    meta_fields=["layout"],
)


@dataclass(frozen=True)
class Rates:
    """Rates returned by a kinetic model, per kg of catalyst.

    Attributes:
        gas: ``(n_gas,)`` mol/s/kg.
        cut: ``(n_cut,)`` mol/s/kg.
        attr: ``(n_cut, n_attr)`` mol/s/kg.
        heat: Heat released, W/kg (positive exothermic).
    """

    gas: Array
    cut: Array
    attr: Array
    heat: Array

    def flows(self) -> Flows:
        return Flows(self.gas, self.cut, self.attr)


jax.tree_util.register_dataclass(Rates, data_fields=["gas", "cut", "attr", "heat"], meta_fields=[])


class KineticModel(Protocol):
    """What :class:`TrickleBedReactor` needs from a kinetic model."""

    attributes: tuple[str, ...]
    attribute_elements: tuple[str | None, ...]

    def rates(self, ctx: ReactionContext, params: Any) -> Rates:
        ...


# =============================================================================
# Phase state and enthalpy along a bed
# =============================================================================


@dataclass(frozen=True)
class KModel:
    """Linearised K-values of one bed: ``ln K(T) = lnK0 + dlnK_dT (T - T0)``.

    Attributes:
        lnK0: ``(n_flash,)`` at the bed inlet.
        dlnK_dT: ``(n_flash,)`` 1/K.
        T0: Inlet temperature.
    """

    lnK0: Array
    dlnK_dT: Array
    T0: Array

    def lnK(self, T):
        return self.lnK0 + self.dlnK_dT * (T - self.T0)


jax.tree_util.register_dataclass(KModel, data_fields=["lnK0", "dlnK_dT", "T0"], meta_fields=[])


def k_model_at(flows: Flows, layout: Layout, comps: Components, T, P) -> tuple[KModel, Any]:
    """PR flash at ``(T, P)`` and its K-values' temperature slope (forward-mode AD).

    Returns ``(KModel, flash residual)``; a residual far from zero (or not
    finite) is a flash that failed (see :func:`.separator.pr_flash`).
    """
    z = flash_components(flows, layout, comps)

    def lnK_of(T):
        fr = pr_flash(T, P, z, comps)
        return fr.lnK, fr.residual

    (lnK0, res), (slope, _) = jax.jvp(lnK_of, (jnp.asarray(T, dtype=float),), (jnp.asarray(1.0),))
    return KModel(lnK0, slope, jnp.asarray(T, dtype=float)), res


def phase_state(flows: Flows, layout: Layout, comps: Components, km: KModel, T):
    """``(V, x, y)`` of ``flows`` at ``T`` with the bed's linearised K-values.

    ``V`` is the negative-flash vapour fraction. Inside the two-phase region
    (``0 < V < 1``) ``x`` and ``y`` are the equilibrium liquid and vapour
    (normalised). Outside it they are the stream itself and the incipient
    phase in equilibrium with it: an all-vapour stream (``V >= 1``) has
    ``y = z`` and ``x = z / K`` (the dew-point liquid, unnormalised, so its
    fugacities are the vapour's), an all-liquid one (``V <= 0``) ``x = z``
    and ``y = K z`` (likewise). Both are what the negative flash gives *at*
    the phase boundary, so the two definitions meet continuously there.

    Using the negative flash's own fictitious split beyond the boundary
    instead (the code before #332) put the vapour's hydrogen partial
    pressure six times too low in a vapour-phase naphtha bed (``V = 9``:
    ``y_H2 = 0.06`` against ``z_H2 = 0.37``), and the aromatics equilibrium
    ran backwards.
    """
    z = flash_components(flows, layout, comps)
    zt = jnp.maximum(z, 1e-300)
    zn = zt / jnp.sum(zt)
    K = jnp.exp(km.lnK(T))
    V = rachford_rice(zn, K)
    two = (V > 0.0) & (V < 1.0)
    beta = jnp.clip(V, 0.0, 1.0)
    x = zn / phase_denominator(zn, K, beta)
    y = K * x
    x = jnp.where(two, x / jnp.sum(x), x)
    y = jnp.where(two, y / jnp.sum(y), y)
    return V, x, y


def stream_enthalpy(flows: Flows, layout: Layout, comps: Components, km: KModel, T) -> Array:
    """Enthalpy flow (W) of ``flows`` at ``T`` with the phase split of ``km``.

    Each flashing component's vapour share (``beta K/(1 + beta(K-1))``,
    ``beta`` clipped to [0, 1]) takes the ideal-gas enthalpy, the rest the
    liquid one; water, if present, is vapour.
    """
    z = flash_components(flows, layout, comps)
    V, _, _ = phase_state(flows, layout, comps, km, T)
    beta = jnp.clip(V, 0.0, 1.0)
    K = jnp.exp(km.lnK(T))
    fv = beta * K / (1.0 + beta * (K - 1.0))
    h = fv * comps.h_vapor(T) + (1.0 - fv) * comps.h_liquid(T)
    H = jnp.sum(z * h)
    if "water" in layout.gases:
        from difflow_refinery.thermo import ColumnThermo
        H = H + flows.gas[layout.gas_index("water")] * ColumnThermo.water_h_vapor(T)
    return H


def vapor_enthalpy(flows: Flows, layout: Layout, comps: Components, T) -> Array:
    """Enthalpy flow (W) of ``flows`` as ideal gas at ``T`` (a treat or quench gas)."""
    z = flash_components(flows, layout, comps)
    H = jnp.sum(z * comps.h_vapor(T))
    if "water" in layout.gases:
        from difflow_refinery.thermo import ColumnThermo
        H = H + flows.gas[layout.gas_index("water")] * ColumnThermo.water_h_vapor(T)
    return H


def reaction_context(flows: Flows, layout: Layout, comps: Components, km: KModel, T, P) -> ReactionContext:
    """The :class:`ReactionContext` at one point of a bed."""
    V, x, y = phase_state(flows, layout, comps, km, T)
    vcut = comps.cut_liquid_volume(T)
    k = comps.n_gas
    v_L = jnp.sum(x[k:] * vcut) / jnp.sum(x)          # molar volume of the (incipient) liquid
    c = x / v_L
    pm = flows.per_molecule()
    return ReactionContext(layout=layout, comps=comps, flows=flows, T=jnp.asarray(T), P=jnp.asarray(P),
                           x=x, y=y, beta=V, c=c, c_attr=c[k:, None] * pm, p=y * P, v_L=v_L,
                           per_molecule=pm)


def check_element_conservation(kinetics: KineticModel, ctx: ReactionContext, params) -> Array:
    """``(4,)`` net production of C, H, S, N atoms by ``kinetics`` at ``ctx`` (should be ~0)."""
    r = kinetics.rates(ctx, params)
    return r.flows().elements(ctx.layout)


# =============================================================================
# One bed
# =============================================================================


@dataclass(frozen=True)
class BedResult:
    """One integrated bed.

    Attributes:
        inlet, outlet: Flows.
        T_in, T_out: K.
        xi: ``(n_save,)`` normalised catalyst position (0 inlet, 1 outlet).
        T: ``(n_save,)`` temperature profile.
        profile: :class:`Flows` with a leading ``(n_save,)`` axis.
        k_model: The bed's linearised K-values.
        steps: Accepted solver steps.
        flash_residual: Residual of the bed-inlet PR flash (see :func:`k_model_at`).
    """

    inlet: Flows
    outlet: Flows
    T_in: Array
    T_out: Array
    xi: Array
    T: Array
    profile: Flows
    k_model: KModel
    steps: Array
    flash_residual: Array = 0.0


jax.tree_util.register_dataclass(
    BedResult, data_fields=["inlet", "outlet", "T_in", "T_out", "xi", "T", "profile", "k_model", "steps",
                 "flash_residual"],
    meta_fields=[])


@dataclass(frozen=True)
class ReactorOptions:
    """Integration controls (static).

    Attributes:
        rtol, atol: PID controller tolerances on the scaled state.
        n_save: Profile points saved per bed.
        max_steps: Solver step limit per bed.
        adjoint: ``"checkpoint"`` (RecursiveCheckpointAdjoint, the default:
            reverse mode), ``"backsolve"`` (BacksolveAdjoint, the continuous
            adjoint) or ``"forward"`` (ForwardMode: forward mode only).
        fixed_steps: If set, a fixed number of Tsit5 steps per bed instead of
            adaptive control.
    """

    rtol: float = 1e-9
    atol: float = 1e-11
    n_save: int = 21
    max_steps: int = 4096
    adjoint: str = "checkpoint"
    fixed_steps: int | None = None


def integrate_bed(kinetics: KineticModel, params, layout: Layout, comps: Components,
                  inlet: Flows, T_in, P, W, options: ReactorOptions = ReactorOptions()) -> BedResult:
    """Integrate one adiabatic bed of ``W`` kg of catalyst from ``(inlet, T_in)``."""
    T_in = jnp.asarray(T_in, dtype=float)
    P = jnp.asarray(P, dtype=float)
    W = jnp.asarray(W, dtype=float)
    km, flash_res = k_model_at(inlet, layout, comps, T_in, P)

    y0 = jnp.concatenate([inlet.ravel(), T_in[None]])
    # scale: each block by its own inlet values, floored at 1e-10 of the block's largest
    g, c, a = inlet.gas, inlet.cut, inlet.attr

    def floor(v):
        m = jnp.max(jnp.abs(v)) if v.size else jnp.asarray(1.0)
        return jnp.maximum(jnp.abs(v), 1e-10 * m + 1e-300)

    a_floor = jnp.maximum(jnp.abs(a), 1e-10 * jnp.max(jnp.abs(a), axis=0, keepdims=True) + 1e-300)
    scale = jax.lax.stop_gradient(jnp.concatenate([floor(g), floor(c), a_floor.reshape(-1),
                                                   jnp.asarray([100.0])]))
    n = inlet.ravel().size

    def rhs(xi, u, args):
        W, P = args
        yv = u * scale
        f = Flows.unravel(yv[:n], layout)
        T = yv[n]
        ctx = reaction_context(f, layout, comps, km, T, P)
        r = kinetics.rates(ctx, params)
        _, C_eff = jax.jvp(lambda t: stream_enthalpy(f, layout, comps, km, t), (T,), (jnp.ones_like(T),))
        dT = r.heat / C_eff
        dy = jnp.concatenate([r.flows().ravel(), dT[None]]) * W
        return dy / scale

    term = diffrax.ODETerm(rhs)
    solver = diffrax.Tsit5()
    ts = jnp.linspace(0.0, 1.0, options.n_save)
    if options.fixed_steps:
        controller = diffrax.ConstantStepSize()
        dt0 = 1.0 / options.fixed_steps
    else:
        controller = diffrax.PIDController(rtol=options.rtol, atol=options.atol)
        dt0 = 0.02
    adjoint = {"backsolve": diffrax.BacksolveAdjoint, "forward": diffrax.ForwardMode,
               "checkpoint": diffrax.RecursiveCheckpointAdjoint}[options.adjoint]()
    sol = diffrax.diffeqsolve(term, solver, t0=0.0, t1=1.0, dt0=dt0, y0=y0 / scale, args=(W, P),
                              saveat=diffrax.SaveAt(ts=ts), stepsize_controller=controller,
                              max_steps=options.max_steps, adjoint=adjoint, throw=False)
    ys = sol.ys * scale
    prof = jax.vmap(lambda v: Flows.unravel(v[:n], layout))(ys)
    out = Flows.unravel(ys[-1, :n], layout)
    return BedResult(inlet=inlet, outlet=out, T_in=T_in, T_out=ys[-1, n], xi=ts, T=ys[:, n],
                     profile=prof, k_model=km, steps=sol.stats["num_accepted_steps"],
                     flash_residual=flash_res)


# =============================================================================
# Beds in series with quench
# =============================================================================


@dataclass(frozen=True)
class ReactorResult:
    """Beds in series.

    Attributes:
        beds: One :class:`BedResult` per bed.
        quench: ``(n_beds,)`` quench gas into each bed, as a fraction of the
            treat gas (0 for the first).
        outlet: Last bed's outlet flows.
        T_out: Last bed's outlet temperature.
        wabt: Weight-average bed temperature, ``sum_k W_k (T_in,k + 2 T_out,k)/3 / sum W``
            (the usual inlet/outlet-weighted definition; see docs).
        delta_T: ``(n_beds,)`` temperature rise of each bed.
    """

    beds: tuple
    quench: Array
    outlet: Flows
    T_out: Array
    wabt: Array
    delta_T: Array


jax.tree_util.register_dataclass(ReactorResult, data_fields=["beds", "quench", "outlet", "T_out", "wabt",
                                                             "delta_T"], meta_fields=[])


@dataclass(frozen=True)
class TrickleBedReactor:
    """Adiabatic beds in series with treat-gas quench, around any kinetic model.

    Attributes:
        layout: The stream layout (its attributes must be the kinetic model's).
        kinetics: The kinetic model (see module docstring).
        options: :class:`ReactorOptions`.
    """

    layout: Layout
    kinetics: Any
    options: ReactorOptions = field(default_factory=ReactorOptions)

    def __call__(self, oil: Flows, gas: Flows, T_in: Sequence, T_gas, P, W: Sequence,
                 comps: Components, params, quench: Sequence | None = None,
                 adjoint: str | None = None) -> ReactorResult:
        """Run the beds.

        Args:
            oil: The liquid feed (at the first bed's inlet temperature).
            gas: The total treat gas (first bed plus every quench).
            T_in: Bed inlet temperatures, K: ``n_beds`` of them when the
                quench is solved for (``quench=None``), or only the first
                bed's when quench rates are given.
            T_gas: Quench-gas temperature, K.
            P: Reactor pressure, Pa (no bed pressure drop).
            W: ``(n_beds,)`` catalyst mass in each bed, kg.
            comps: Component constants.
            params: The kinetic model's parameters.
            quench: ``None`` -- solve each later bed's quench (a fraction of
                the treat gas) for its inlet temperature; or ``n_beds - 1``
                quench fractions, the inlet temperatures then following from
                the adiabatic mix.

        With inlet temperatures specified, the first bed's gas is the treat
        gas less every quench, and each quench depends on the bed above it:
        a scalar equation in the total quench fraction, solved by Newton
        (:func:`.solve.newton_scalar`, implicit-function gradient). With quench rates specified each
        mixing temperature is a scalar enthalpy balance, solved the same way.
        """
        lay = self.layout
        n_beds = len(W)
        P = jnp.asarray(P, dtype=float)
        T_gas = jnp.asarray(T_gas, dtype=float)

        kin = self.kinetics
        opts = self.options if adjoint is None else dataclasses.replace(self.options, adjoint=adjoint)
        jac = "fwd" if opts.adjoint == "forward" else "rev"
        T_in = [jnp.asarray(t, dtype=float) for t in T_in]
        Ws = jnp.stack([jnp.asarray(w, dtype=float) for w in W])
        last = jnp.arange(n_beds) == n_beds - 1

        def mix_res(T, args):
            out, km, T_out, qflows, T_gas, comps = args
            target = stream_enthalpy(out, lay, comps, km, T_out) + vapor_enthalpy(qflows, lay, comps, T_gas)
            return (stream_enthalpy(out, lay, comps, km, T) + vapor_enthalpy(qflows, lay, comps, T)
                    - target) / 1e6

        # Beds run under lax.scan: one traced copy of the bed whatever the
        # bed count (compile time is the cost that matters here).
        def run_rates(qs, A):
            """Quench rates given: ``qs`` (n_beds,), qs[k] the quench AFTER bed k (0 for the last)."""
            oil, gas, T0, T_gas, P, Ws, comps, params = A
            q_first = 1.0 - jnp.sum(qs)

            def body(carry, xs):
                stream, T = carry
                Wk, qk = xs
                bed = integrate_bed(kin, params, lay, comps, stream, T, P, Wk, opts)
                qf = gas.scale(qk)
                T_next = newton_scalar(mix_res, jax.lax.stop_gradient(bed.T_out) - 10.0 * (qk > 0),
                                       (bed.outlet, bed.k_model, bed.T_out, qf, T_gas, comps),
                                       tol=1e-12, max_step=50.0, jac=jac).value
                return (bed.outlet + qf, T_next), bed

            _, beds = jax.lax.scan(body, (oil + gas.scale(q_first), T0), (Ws, qs))
            return beds

        def run_temps(s, A):
            """Inlet temperatures given; total quench fraction ``s``; returns (beds, quench after each bed)."""
            oil, gas, Tin, T_gas, P, Ws, comps, params = A
            T_next_all = jnp.concatenate([Tin[1:], Tin[-1:]])

            def body(stream, xs):
                Wk, Tk, Tn, is_last = xs
                bed = integrate_bed(kin, params, lay, comps, stream, Tk, P, Wk, opts)
                km, out = bed.k_model, bed.outlet
                dH = (stream_enthalpy(out, lay, comps, km, bed.T_out) - stream_enthalpy(out, lay, comps, km, Tn))
                per = vapor_enthalpy(gas, lay, comps, Tn) - vapor_enthalpy(gas, lay, comps, T_gas)
                qk = jnp.where(is_last, 0.0, dH / per)
                return out + gas.scale(qk), (bed, qk)

            _, (beds, qs) = jax.lax.scan(body, oil + gas.scale(1.0 - s), (Ws, Tin, T_next_all, last))
            return beds, qs

        if n_beds == 1 or quench is not None:
            qs = jnp.zeros(n_beds)
            if quench is not None and n_beds > 1:
                qv = jnp.stack([jnp.asarray(v, dtype=float) for v in quench])
                if qv.shape != (n_beds - 1,):
                    raise ValueError(f"quench needs {n_beds - 1} fractions")
                qs = qs.at[:-1].set(qv)
            beds = run_rates(qs, (oil, gas, T_in[0], T_gas, P, Ws, comps, params))
        else:
            if len(T_in) != n_beds:
                raise ValueError(f"T_in needs {n_beds} bed inlet temperatures when the quench is solved for")
            A = (oil, gas, jnp.stack(T_in), T_gas, P, Ws, comps, params)
            sol = newton_scalar(lambda s, A: s - jnp.sum(run_temps(s, A)[1]), jnp.asarray(0.1), A,
                                tol=1e-12, max_step=0.5, jac=jac)
            beds, qs = run_temps(sol.value, A)
        q = jnp.concatenate([jnp.zeros(1), qs[:-1]])
        Tin, Tout = beds.T_in, beds.T_out
        wabt = jnp.sum(Ws * (Tin + 2.0 * Tout) / 3.0) / jnp.sum(Ws)
        bed_list = tuple(jax.tree_util.tree_map(lambda a, k=k: a[k], beds) for k in range(n_beds))
        return ReactorResult(beds=bed_list, quench=q, outlet=bed_list[-1].outlet, T_out=Tout[-1],
                             wabt=wabt, delta_T=Tout - Tin)


__all__ = ["ReactionContext", "Rates", "KineticModel", "KModel", "k_model_at", "phase_state",
           "stream_enthalpy", "vapor_enthalpy", "reaction_context", "check_element_conservation",
           "BedResult", "ReactorOptions", "integrate_bed", "ReactorResult", "TrickleBedReactor"]
