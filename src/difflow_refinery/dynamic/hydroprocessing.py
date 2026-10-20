"""Dynamic hydroprocessing units: trickle beds of cells, a charge heater, an HP separator, a recycle loop.

Every class here implements difflow's :class:`~difflow.dynamic.DynamicUnit`
protocol (``state_spec``, ``initial_state``, ``derivatives``, ``outputs``)
on the hydroprocessing stream layout (:mod:`difflow_refinery.hydroprocessing.layout`:
``F_<gas>``, ``F_<cut>``, ``F_<cut>@<attribute>``), so they run in a
:class:`~difflow.dynamic.DynamicFlowsheet`. They reuse the steady units'
rate laws, phase model and enthalpies -- nothing is re-derived.

The bed: N cells per bed
------------------------

Each catalyst bed is ``N`` cells in series. A cell holds a material
**holdup** ``n`` (every gas, cut and attribute, mol) and has one
**temperature** ``T`` shared by its catalyst and fluid (pseudo-homogeneous,
as the steady bed). Two things happen in a cell:

1. *Reaction.* The cell's inflow ``(F_in, T_in)`` -- the previous cell's
   outflow at its temperature -- is carried across the cell's catalyst
   ``W_c`` by ``n_sub`` classical Runge-Kutta steps of the **steady bed
   equations** (:func:`~difflow_refinery.hydroprocessing.reactor.bed_derivatives`:
   ``dF/dW = r``, ``C_eff dT/dW = q``), giving ``(F*, T*)``.
2. *Holdup.* The holdup is fed by ``F*`` and drained by its own outflow::

       dn/dt = F* - F_out(n, T)
       (W_c c_p,cat + C_hold) dT/dt = H(F*, T*) - H(F*, T)

   ``F_out`` splits the holdup into its equilibrium vapour and liquid (the
   bed's K-values, a Rachford-Rice split at ``T``); the vapour leaves with
   residence time ``tau_G`` and the liquid with ``tau_L`` (both per cell,
   fixed at the design point from the bed's gas and liquid holdup volumes).
   The energy balance gives the heat the reacted fluid carries in at
   ``T*`` to the cell's catalyst and holdup at ``T``.

At steady state ``F_out = F*`` and ``T = T*``, so the bed's steady state is
the steady plug-flow bed integrated by ``N x n_sub`` RK4 steps: the
dynamic model's ``t -> infinity`` limit converges to the steady unit at
fourth order in the cell size, whatever the holdups (on the test diesel,
ten RK4 steps per bed already put the bed-outlet sulfur within 5e-5 of
the adaptive steady integration). The transient is first order in the cell
size (the cell series' numerical dispersion is that of a dispersed plug flow
with Peclet number about ``2N``); the reaction in a cell is evaluated on its
inflow, half a cell upstream of the holdup, which is the price of the exact
steady limit. Element balances close exactly along a trajectory: the RK4
step is a linear combination of element-conserving rates, so
``d(elements in holdup)/dt = elements(F_in) - elements(F_out)``.

What is NOT modelled: pressure dynamics (every pressure is fixed; flows are
not pressure-driven), gas-liquid and liquid-solid transfer resistances and
heat transfer between catalyst and fluid (instantaneous, as in the steady
bed), wall heat capacity, catalyst deactivation, liquid maldistribution.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.dynamic.state import StateSpec, StateVar
from difflow_refinery.hydroprocessing.layout import Flows, Layout
from difflow_refinery.hydroprocessing.reactor import (
    KModel, bed_derivatives, k_model_at, stream_enthalpy, vapor_enthalpy)
from difflow_refinery.hydroprocessing.recycle import (
    amine_scrub, compress, knockout, makeup_for_ratio, purge_split)
from difflow_refinery.hydroprocessing.separator import (
    HPSeparator, flash_components, phase_denominator, rachford_rice, split_by_vapor_fraction)
from difflow_refinery.hydroprocessing.solve import newton_scalar
from difflow_refinery.hydroprocessing.thermo import Components

jax.config.update("jax_enable_x64", True)

R_GAS = 8.314462618

#: Catalyst heat capacity, J/(kg K). ILLUSTRATIVE: a gamma-alumina-supported
#: CoMo/NiMo catalyst at 300-400 C. Alumina's own heat capacity rises from
#: about 0.78 kJ/(kg K) at 25 C to about 1.1 at 400 C (NIST-JANAF,
#: Al2O3 alpha; the support is not pure alpha-alumina and the metals and
#: pores change it), so 1000 is the right size, not a measured value.
CATALYST_CP = 1000.0


def _floor_scale(v: Array, rel: float = 1e-10) -> Array:
    """``|v|`` floored at ``rel`` of its largest entry (a state scale)."""
    v = jnp.abs(jnp.asarray(v, dtype=float))
    return jnp.maximum(v, rel * (jnp.max(v) if v.size else 1.0) + 1e-300)


def flows_scale(f: Flows) -> Flows:
    """A scale for a Flows-valued state: each block by its own entries, attributes on their cut too.

    The same floor the steady bed uses (:func:`~difflow_refinery.hydroprocessing.reactor.integrate_bed`),
    so an attribute that is zero in a feed but made in the bed is still resolved.
    """
    g, c, a = f.gas, f.cut, f.attr
    a_floor = jnp.maximum(jnp.abs(a), 1e-10 * jnp.maximum(jnp.max(jnp.abs(a), axis=0, keepdims=True),
                                                         jnp.max(jnp.abs(c)) if c.size else 0.0) + 1e-300)
    return Flows(_floor_scale(g), _floor_scale(c), a_floor)


# =============================================================================
# The holdup split
# =============================================================================


def holdup_outflow(n: Flows, layout: Layout, comps: Components, km: KModel, T, inv_tau_G, inv_tau_L) -> Flows:
    """Outflow (mol/s) of a cell holding ``n`` (mol) at ``T``.

    The holdup is split into equilibrium vapour and liquid with the bed's
    K-values (Rachford-Rice on the holdup's own composition, vapour fraction
    clipped to [0, 1]: a holdup outside the two-phase region is all vapour
    or all liquid). The vapour drains at ``1/tau_G``, the liquid at
    ``1/tau_L``; water (never flashed in this layout) leaves with the vapour.
    Attributes follow their cut. The outflow's phases are the holdup's (same
    ``x`` and ``y``), so a rate law evaluated on the outflow sees the
    holdup's compositions.
    """
    n = Flows(jnp.maximum(n.gas, 0.0), jnp.maximum(n.cut, 0.0), jnp.maximum(n.attr, 0.0))
    z = flash_components(n, layout, comps)
    zt = jnp.maximum(z, 1e-300)
    zn = zt / jnp.sum(zt)
    K = jnp.exp(km.lnK(T))
    beta = jnp.clip(rachford_rice(zn, K), 0.0, 1.0)
    frac = beta * K / phase_denominator(zn, K, beta)            # vapour share of each component
    vap, liq, water = split_by_vapor_fraction(n, layout, comps, frac)
    return vap.scale(inv_tau_G) + liq.scale(inv_tau_L) + water.scale(inv_tau_G)


def holdup_from_outflow(F: Flows, layout: Layout, comps: Components, km: KModel, T, tau_G, tau_L) -> Flows:
    """The holdup whose :func:`holdup_outflow` is ``F`` (the inverse, for initial states).

    The flowing mixture's equilibrium phases are the holdup's, in different
    amounts: the vapour held is ``tau_G`` times the vapour flow and the liquid
    ``tau_L`` times the liquid flow.
    """
    z = flash_components(F, layout, comps)
    zt = jnp.maximum(z, 1e-300)
    zn = zt / jnp.sum(zt)
    K = jnp.exp(km.lnK(T))
    beta = jnp.clip(rachford_rice(zn, K), 0.0, 1.0)
    frac = beta * K / phase_denominator(zn, K, beta)
    vap, liq, water = split_by_vapor_fraction(F, layout, comps, frac)
    return vap.scale(tau_G) + liq.scale(tau_L) + water.scale(tau_G)


def phase_flows(F: Flows, layout: Layout, comps: Components, km: KModel, T) -> tuple[Array, Array, Array]:
    """``(vapour mol/s, liquid mol/s, liquid m^3/mol)`` of a flowing mixture with the bed's K-values."""
    z = flash_components(F, layout, comps)
    zt = jnp.maximum(z, 1e-300)
    zn = zt / jnp.sum(zt)
    K = jnp.exp(km.lnK(T))
    beta = jnp.clip(rachford_rice(zn, K), 0.0, 1.0)
    d = phase_denominator(zn, K, beta)
    x = zn / d
    vcut = comps.cut_liquid_volume(T)
    k = comps.n_gas
    v_L = jnp.sum(x[k:] * vcut) / jnp.maximum(jnp.sum(x), 1e-300)
    Ftot = jnp.sum(z)
    water = F.gas_flow(layout, "water")
    return beta * Ftot + water, (1.0 - beta) * Ftot, v_L


# =============================================================================
# Trickle-bed reactor of cells
# =============================================================================


@dataclass(frozen=True)
class BedGeometry:
    """Holdup of one bed (design values; see :class:`DynamicTrickleBedReactor`).

    Attributes:
        W: Catalyst mass, kg.
        tau_G: Gas residence time of one cell, s.
        tau_L: Liquid residence time of one cell, s.
    """

    W: float
    tau_G: float
    tau_L: float


class DynamicTrickleBedReactor:
    """Adiabatic trickle beds in series with treat-gas quench, each bed ``n_cells`` cells.

    Ports: ``"feed"`` (oil plus the first bed's treat gas, at the bed-1 inlet
    temperature) and ``"quench"`` (the quench gas for every later bed, at its
    own temperature; split between the later beds by ``quench_shares``).
    Output: ``"outlet"`` (the last cell's outflow at its temperature).

    State, per bed and cell: the scaled holdup (every gas, cut and
    attribute) and the temperature. The module docstring gives the model.

    Args:
        layout, kinetics, kin_params, comps: As the steady
            :class:`~difflow_refinery.hydroprocessing.reactor.TrickleBedReactor`
            uses them (the kinetic model and its parameters).
        beds: One :class:`BedGeometry` per bed.
        P: Reactor pressure, Pa.
        quench_shares: Share of the quench stream each later bed gets
            (``n_beds - 1`` entries summing to 1).
        n_cells: Cells per bed.
        n_sub: RK4 steps of the steady bed equations per cell.
        cp_cat: Catalyst heat capacity, J/(kg K).
        scale: Optional state scale (``(n_states,)``); default from
            :meth:`initial_state`'s inputs is not available, so pass the
            design holdups' scale from the builder.
        name: Unit name.
    """

    output_ports = ("outlet",)

    def __init__(self, layout: Layout, kinetics, kin_params, comps: Components, beds: Sequence[BedGeometry],
                 P: float, quench_shares: Sequence[float] = (), n_cells: int = 10, n_sub: int = 1,
                 cp_cat: float = CATALYST_CP, scale: Array | None = None, y0: Array | None = None,
                 name: str = "reactor"):
        self.layout = layout
        self.kinetics = kinetics
        self.kin_params = kin_params
        self.comps = comps
        self.beds = tuple(beds)
        self.P = jnp.asarray(P, dtype=float)
        self.n_beds = len(self.beds)
        if self.n_beds > 1 and len(quench_shares) != self.n_beds - 1:
            raise ValueError(f"quench_shares needs {self.n_beds - 1} entries")
        self.quench_shares = jnp.asarray(tuple(quench_shares) or (0.0,), dtype=float)
        self.n_cells = int(n_cells)
        self.n_sub = int(n_sub)
        self.cp_cat = jnp.asarray(cp_cat, dtype=float)
        self.n_flow = layout.n_gas + layout.n_cut + layout.n_cut * layout.n_attr
        self._scale = None if scale is None else jnp.asarray(scale, dtype=float)
        self._y0 = y0
        self.name = name

    # ----- state layout ---------------------------------------------------------

    @property
    def n_per_cell(self) -> int:
        return self.n_flow + 1

    def state_spec(self) -> StateSpec:
        """One scaled state per (bed, cell, flow entry) and per (bed, cell) temperature."""
        lay = self.layout
        names = [f"F_{g}" for g in lay.gases] + [f"F_{c}" for c in lay.cuts] + \
            [lay.attr_key(c, a) for c in lay.cuts for a in lay.attributes] + ["T"]
        return StateSpec([StateVar(f"bed{b + 1}.cell{c + 1}.{nm}", "generic", "-", "")
                          for b in range(self.n_beds) for c in range(self.n_cells) for nm in names])

    @property
    def n_states(self) -> int:
        return self.n_beds * self.n_cells * self.n_per_cell

    @property
    def scale(self) -> Array:
        if self._scale is None:
            return jnp.ones(self.n_states)
        return self._scale

    def unpack(self, state: Array) -> tuple[Flows, Array]:
        """``(holdups, T)``: Flows with leading ``(n_beds, n_cells)`` axes (mol), and ``(n_beds, n_cells)`` K."""
        x = (state * self.scale).reshape(self.n_beds, self.n_cells, self.n_per_cell)
        lay = self.layout
        f = jax.vmap(jax.vmap(lambda v: Flows.unravel(v[:-1], lay)))(x)
        return f, x[..., -1]

    def pack(self, holdups: Flows, T: Array) -> Array:
        v = jax.vmap(jax.vmap(lambda f: f.ravel()))(holdups)
        x = jnp.concatenate([v, T[..., None]], axis=-1).reshape(-1)
        return x / self.scale

    def initial_state(self, inputs=None, params=None) -> Array:
        if self._y0 is None:
            raise ValueError("build the reactor with y0 (DynamicHydrotreater.from_steady does) "
                             "or pass y0 to simulate")
        return self._y0

    # ----- one bed ----------------------------------------------------------------

    def _segment(self, km: KModel, F_in: Flows, T_in, W, n_sub: int | None = None, unroll: bool = True):
        """``(F*, T*)``: RK4 steps of the steady bed equations across ``W`` kg of catalyst.

        ``n_sub`` overrides :attr:`n_sub` (the rest-state check uses it);
        ``unroll=False`` loops the steps (``lax.fori_loop``: one step compiled,
        for the rest-state check) instead of unrolling them (the dynamics).
        """
        lay, comps, kin, prm, P = self.layout, self.comps, self.kinetics, self.kin_params, self.P
        n_sub = self.n_sub if n_sub is None else int(n_sub)
        h = W / n_sub

        def g(f, T):
            dF, dT = bed_derivatives(kin, prm, lay, comps, km, f, T, P)
            return dF.scale(h), dT * h

        def rk4(f, T):
            k1 = g(f, T)
            k2 = g(f + k1[0].scale(0.5), T + 0.5 * k1[1])
            k3 = g(f + k2[0].scale(0.5), T + 0.5 * k2[1])
            k4 = g(f + k3[0], T + k3[1])
            return (f + (k1[0] + k2[0].scale(2.0) + k3[0].scale(2.0) + k4[0]).scale(1.0 / 6.0),
                    T + (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0)

        f, T = F_in, T_in
        if not unroll:
            return jax.lax.fori_loop(0, n_sub, lambda _, c: rk4(*c), (f, jnp.asarray(T, dtype=float)))
        for _ in range(n_sub):
            f, T = rk4(f, T)
        return f, T

    def rest_profile(self, b: int, F_in: Flows, T_in, n_sub: int | None = None):
        """The rest state of bed ``b`` fed ``(F_in, T_in)``: each cell's outflow and temperature.

        At rest every cell's outflow is its reacted inflow (``F_out = F*``,
        ``T = T*``), so the cell series' steady state is the cells' RK4 steps
        marched down the bed -- whatever the holdups. This is the steady state
        the dynamic model settles to; :func:`check_rest_state` compares it
        with the steady unit. Returns ``(Flows (n_cells, ...), T (n_cells,))``.
        """
        geo = self.beds[b]
        Wc = geo.W / self.n_cells
        km, _ = k_model_at(F_in, self.layout, self.comps, T_in, self.P)

        def step(carry, _):
            f, t = carry
            f2, t2 = self._segment(km, f, t, Wc, n_sub, unroll=False)
            return (f2, t2), (f2, t2)

        _, (F, T) = jax.lax.scan(step, (F_in, jnp.asarray(T_in, dtype=float)), None, length=self.n_cells)
        return F, T

    def _bed(self, b: int, F_in: Flows, T_in, n: Flows, T: Array):
        """Derivatives and outflows of bed ``b`` with inlet ``(F_in, T_in)``.

        Returns ``(dn/dt Flows (n_cells,...), dT/dt (n_cells,), outflows, km)``.
        """
        lay, comps = self.layout, self.comps
        geo = self.beds[b]
        Wc = geo.W / self.n_cells
        km, _ = k_model_at(F_in, lay, comps, T_in, self.P)
        inv_G, inv_L = 1.0 / geo.tau_G, 1.0 / geo.tau_L
        out = jax.vmap(lambda nk, Tk: holdup_outflow(nk, lay, comps, km, Tk, inv_G, inv_L))(n, T)
        # inflow of each cell: the bed inlet, then the previous cell's outflow
        Fi = jax.tree_util.tree_map(lambda a, o: jnp.concatenate([a[None], o[:-1]], axis=0), F_in, out)
        Ti = jnp.concatenate([jnp.atleast_1d(T_in), T[:-1]])
        Fs, Ts = jax.vmap(lambda f, t: self._segment(km, f, t, Wc))(Fi, Ti)
        dn = Fs - out

        def heat(fs, ts, tk, nk):
            drive = stream_enthalpy(fs, lay, comps, km, ts) - stream_enthalpy(fs, lay, comps, km, tk)
            _, C_hold = jax.jvp(lambda t: stream_enthalpy(nk, lay, comps, km, t), (tk,), (jnp.ones_like(tk),))
            return drive / (Wc * self.cp_cat + C_hold)

        dT = jax.vmap(heat)(Fs, Ts, T, n)
        return dn, dT, out, km

    def _run(self, inputs, state, shares: Array | None = None):
        """Every bed's derivatives, outflows and inlet; ``shares`` overrides :attr:`quench_shares`.

        ``shares[b]`` is the fraction of the ``"quench"`` stream mixed in
        after bed ``b`` (a quench controller's valve positions, say).
        """
        lay, comps = self.layout, self.comps
        shares = self.quench_shares if shares is None else shares
        feed = Flows.from_stream(inputs["feed"], lay)
        T_feed = jnp.asarray(inputs["feed"]["T"], dtype=float)
        quench = Flows.from_stream(inputs["quench"], lay) if self.n_beds > 1 else None
        T_q = jnp.asarray(inputs["quench"]["T"], dtype=float) if self.n_beds > 1 else None
        n, T = self.unpack(state)
        dns, dTs, outs, bed_in = [], [], [], []
        F_in, T_in = feed, T_feed
        for b in range(self.n_beds):
            nb = jax.tree_util.tree_map(lambda a: a[b], n)
            dn, dT, out, km = self._bed(b, F_in, T_in, nb, T[b])
            dns.append(dn)
            dTs.append(dT)
            outs.append(out)
            bed_in.append((F_in, T_in))
            if b + 1 < self.n_beds:
                last = jax.tree_util.tree_map(lambda a: a[-1], out)
                qf = quench.scale(shares[b])
                T_last = T[b, -1]

                def mix_res(Tm, args):
                    out_, km_, T_out, qf_, T_gas = args
                    target = stream_enthalpy(out_, lay, comps, km_, T_out) + vapor_enthalpy(qf_, lay, comps, T_gas)
                    return (stream_enthalpy(out_, lay, comps, km_, Tm) + vapor_enthalpy(qf_, lay, comps, Tm)
                            - target) / 1e6

                T_mix = newton_scalar(mix_res, jax.lax.stop_gradient(T_last) - 5.0,
                                      (last, km, T_last, qf, T_q), tol=1e-12, max_step=50.0, jac="fwd").value
                F_in, T_in = last + qf, T_mix
        return dns, dTs, outs, bed_in

    # ----- DynamicUnit ----------------------------------------------------------------

    def derivatives(self, t, state, inputs, params=None):
        dns, dTs, _, _ = self._run(inputs, state)
        dn = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *dns)
        dT = jnp.stack(dTs)
        v = jax.vmap(jax.vmap(lambda f: f.ravel()))(dn)
        x = jnp.concatenate([v, dT[..., None]], axis=-1).reshape(-1)
        return x / self.scale

    def outputs(self, t, state, inputs, params=None):
        _, _, outs, _ = self._run(inputs, state)
        _, T = self.unpack(state)
        last = jax.tree_util.tree_map(lambda a: a[-1], outs[-1])
        return {"outlet": last.to_stream(self.layout, T[-1, -1], self.P)}

    # ----- reporting ------------------------------------------------------------------

    def profiles(self, state, inputs) -> dict:
        """``{"T": (n_beds, n_cells) K, "T_in": (n_beds,) K, "outflow": Flows (n_beds, n_cells, ...)}``."""
        _, _, outs, bed_in = self._run(inputs, state)
        _, T = self.unpack(state)
        out = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *outs)
        return {"T": T, "T_in": jnp.stack([b[1] for b in bed_in]), "outflow": out}

    def wabt(self, state, inputs) -> Array:
        """Weight-average bed temperature on the steady unit's definition, ``sum W (T_in + 2 T_out)/3 / sum W``."""
        p = self.profiles(state, inputs)
        W = jnp.asarray([g.W for g in self.beds])
        return jnp.sum(W * (p["T_in"] + 2.0 * p["T"][:, -1]) / 3.0) / jnp.sum(W)


class DynamicRestStateWarning(UserWarning):
    """A dynamic bed's own rest state is far from the steady unit it was built from.

    The cells' fixed RK4 steps across the catalyst are too coarse for the
    bed's fastest reactions (unstable, or simply inaccurate), so the model
    would settle somewhere else than the steady answer. Use more steps per
    cell (``n_sub``) or more cells.
    """


def _sulfur_wppm(f: Flows, layout: Layout) -> Array:
    """Sulfur on the cuts of ``f``, wppm of the cuts' mass (as ``DynamicHydrotreater.cut_sulfur_wppm``)."""
    from difflow_refinery.composition import SULFUR_CLASSES
    ai = {a: i for i, a in enumerate(layout.attributes)}
    cols = [ai[f"S_{k}"] for k in SULFUR_CLASSES if f"S_{k}" in ai]
    if not cols:
        return jnp.asarray(0.0)
    S = jnp.sum(f.attr[:, cols]) * 32.065e-3
    return 1e6 * S / jnp.sum(f.cut_mass(layout))


def rest_mismatch(reactor: "DynamicTrickleBedReactor", b: int, F_in: Flows, T_in, F_ref: Flows, T_ref,
                  n_sub: int | None = None) -> dict:
    """How far bed ``b``'s rest outlet is from a reference outlet ``(F_ref, T_ref)``.

    Returns ``{"S_wppm", "S_wppm_ref", "rel_S", "dT", "rel_flow"}``: the outlet
    sulfur (wppm of the cuts), its relative error (floored at 0.01 wppm), the
    outlet temperature error (K) and the largest relative error over the outlet
    flows that carry at least 1e-6 of the total.
    """
    F, T = reactor.rest_profile(b, F_in, T_in, n_sub)
    last = jax.tree_util.tree_map(lambda a: a[-1], F)
    lay = reactor.layout
    S, S_ref = float(_sulfur_wppm(last, lay)), float(_sulfur_wppm(F_ref, lay))
    v, v_ref = np.asarray(last.ravel()), np.asarray(F_ref.ravel())
    big = np.abs(v_ref) > 1e-6 * np.sum(np.abs(v_ref))
    rel_flow = float(np.max(np.abs(v - v_ref)[big] / np.abs(v_ref)[big])) if big.any() else 0.0
    return {"S_wppm": S, "S_wppm_ref": S_ref, "rel_S": abs(S - S_ref) / max(abs(S_ref), 0.01),
            "dT": float(T[-1]) - float(T_ref), "rel_flow": rel_flow}


#: Tolerances of the rest-state check: outlet sulfur (relative, floored at
#: 0.01 wppm), outlet temperature (K) and the major outlet flows (relative).
REST_TOL = {"rel_S": 0.02, "dT": 0.5, "rel_flow": 0.01}


def _rest_ok(m: dict, tol: dict = REST_TOL) -> bool:
    return (np.isfinite([m["S_wppm"], m["dT"], m["rel_flow"]]).all() and m["rel_S"] <= tol["rel_S"]
            and abs(m["dT"]) <= tol["dT"] and m["rel_flow"] <= tol["rel_flow"])


def warn_rest_state(rest: Sequence[dict], what: str, n_cells: int, n_sub: int,
                    tol: dict = REST_TOL) -> bool:
    """Warn (:class:`DynamicRestStateWarning`) for the beds whose rest state is off; True if any is.

    ``rest`` is one :func:`rest_mismatch` dict per bed; ``what`` names the unit.
    """
    import warnings
    bad = [(b, m) for b, m in enumerate(rest) if not _rest_ok(m, tol)]
    if bad:
        msg = "; ".join(f"bed {b + 1}: outlet S {m['S_wppm']:.4g} wppm against {m['S_wppm_ref']:.4g}, "
                        f"dT {m['dT']:+.3g} K, flows {m['rel_flow']:.2g}" for b, m in bad)
        warnings.warn(f"the {what}'s rest state (n_cells={n_cells}, n_sub={n_sub}) differs from the steady "
                      f"unit: {msg}. Use a larger n_sub (or n_sub=None) or more cells.",
                      DynamicRestStateWarning, stacklevel=3)
    return bool(bad)


def choose_n_sub(reactor: "DynamicTrickleBedReactor", bed_refs, n_max: int = 64,
                 tol: dict = REST_TOL) -> tuple[int, list]:
    """The fewest RK4 steps per cell whose rest state reproduces the steady beds.

    ``bed_refs`` is one ``(F_in, T_in, F_out_ref, T_out_ref)`` per bed (the
    steady unit's bed inlet and outlet). Tries ``n_sub = 1, 2, 4, ...`` up to
    ``n_max`` and returns the first that meets ``tol`` on every bed, with the
    per-bed mismatches; if none does, returns ``n_max`` (the caller warns).

    Why it is needed: a fixed RK4 step across a cell of a high-conversion bed
    can exceed the step's stability limit for the fastest sulfur class, and the
    marched rest state is then wrong by orders of magnitude, silently.
    """
    n = 1
    while True:
        ms = [rest_mismatch(reactor, b, *ref, n_sub=n) for b, ref in enumerate(bed_refs)]
        if all(_rest_ok(m, tol) for m in ms) or n >= n_max:
            return n, ms
        n = min(2 * n, n_max)


# =============================================================================
# Charge heater, HP separator, recycle loop
# =============================================================================


class FurnaceLag:
    """A fired heater whose outlet temperature lags its setpoint (tube metal and refractory).

    Ports: any streams named in ``inlets`` (summed: every ``F_`` key) and
    ``"signals"``. Output ``"outlet"``: the sum at the heater temperature
    ``T_h``, the one state::

        dT_h/dt = (T_set(t) - T_h) / tau

    with ``T_set`` the signal named ``signal``. Works on any stream whose
    flows are ``F_`` keys (the hydroprocessing layout, the reformer's
    species). The duty a fuel controller would have to fire is not
    modelled; the lag stands for the whole loop (fuel valve, firebox, tube
    metal).
    """

    output_ports = ("outlet",)

    def __init__(self, inlets: Sequence[str], signal: str, tau: float, P: float, T0: float,
                 name: str = "heater"):
        self.inlets = tuple(inlets)
        self.signal = signal
        self.tau = jnp.asarray(tau, dtype=float)
        self.P = jnp.asarray(P, dtype=float)
        self.T0 = float(T0)
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar("T", "temperature", "K", "heater outlet temperature", scale=100.0)])

    def initial_state(self, inputs=None, params=None):
        return jnp.asarray([self.T0 / 100.0])

    def derivatives(self, t, state, inputs, params=None):
        T = 100.0 * state[0]
        return jnp.reshape((inputs["signals"][self.signal] - T) / self.tau / 100.0, (1,))

    def outputs(self, t, state, inputs, params=None):
        out = {}
        for s in self.inlets:
            for k, v in inputs[s].items():
                if k.startswith("F_"):
                    out[k] = out.get(k, 0.0) + v
        out["T"] = 100.0 * state[0]
        out["P"] = self.P
        return {"outlet": out}


class DynamicHPSeparator:
    """High-pressure separator with a liquid holdup.

    Port ``"inlet"`` (the reactor effluent; the effluent cooler is
    instantaneous to ``T``). The effluent is flashed at ``(T, P)`` (PR,
    :class:`~difflow_refinery.hydroprocessing.separator.HPSeparator`) and the
    cuts in the vapour are knocked out into the liquid, as in the steady
    unit. Outputs: ``"vapor"`` and ``"water"`` follow the inlet instantly;
    ``"liquid"`` drains the liquid holdup ``n`` (the one state block)::

        dn/dt = L_in - n / tau

    ``tau`` is the liquid residence time (a level held by a level
    controller acting on the outlet valve, idealised as proportional
    drainage). The vapour space is part of the recycle loop's gas holdup
    (:class:`RecycleGasLoop`).
    """

    output_ports = ("vapor", "liquid", "water")
    output_dependencies = {"vapor": ("inlet",), "water": ("inlet",), "liquid": ()}

    def __init__(self, layout: Layout, comps: Components, T: float, P: float, tau: float,
                 scale: Flows, y0: Array, name: str = "hps"):
        self.layout = layout
        self.comps = comps
        self.T = jnp.asarray(T, dtype=float)
        self.P = jnp.asarray(P, dtype=float)
        self.tau = jnp.asarray(tau, dtype=float)
        self._scale = scale.ravel()
        self._y0 = jnp.asarray(y0)
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar(f"x{i}", "generic", "-", "") for i in range(self._scale.size)])

    def initial_state(self, inputs=None, params=None):
        return self._y0

    def holdup(self, state) -> Flows:
        return Flows.unravel(state * self._scale, self.layout)

    def _flash(self, inlet):
        f = Flows.from_stream(inlet, self.layout)
        vap, liq, water, _ = HPSeparator(self.layout)(f, self.T, self.P, self.comps)
        return knockout(vap, liq) + (water,)

    def derivatives(self, t, state, inputs, params=None):
        _, liq, _ = self._flash(inputs["inlet"])
        n = self.holdup(state)
        return (liq - n.scale(1.0 / self.tau)).ravel() / self._scale

    def _liquid(self, state):
        return self.holdup(state).scale(1.0 / self.tau).to_stream(self.layout, self.T, self.P)

    def outputs(self, t, state, inputs, params=None):
        vap, _, water = self._flash(inputs["inlet"])
        return {"vapor": vap.to_stream(self.layout, self.T, self.P), "liquid": self._liquid(state),
                "water": water.to_stream(self.layout, self.T, self.P)}

    def partial_outputs(self, t, state, inputs, params=None):
        out = {"liquid": self._liquid(state)}
        if "inlet" in inputs:
            vap, _, water = self._flash(inputs["inlet"])
            out["vapor"] = vap.to_stream(self.layout, self.T, self.P)
            out["water"] = water.to_stream(self.layout, self.T, self.P)
        return out


class RecycleGasLoop:
    """The recycle-gas loop: amine scrub, purge, gas inventory, compressor and makeup.

    Ports ``"vapor"`` (HP separator gas) and ``"signals"``. The scrubbed gas
    less the purge enters the loop's gas inventory ``n`` (separator vapour
    space, piping, knock-out drum and scrubber; the one state block). The
    recycle compressor draws ``R = s(t) n / tau`` (``s`` the signal
    ``"compressor"``: 1 running, 0 tripped; ``tau`` the loop's design
    residence time) and the makeup tops the treat gas up to its design H2
    rate (the steady unit's ratio rule, :func:`~difflow_refinery.hydroprocessing.recycle.makeup_for_ratio`)
    up to ``makeup_max`` mol/s (the makeup compressor's capacity). An
    inventory above ``relief_inventory`` is vented (the loop's pressure
    controller to flare) with time constant ``tau_relief``::

        dn/dt = (1 - purge) sweet(vapor) - R - vent

    Outputs: ``"treat_gas"`` (the heater's share, ``1 - q``),
    ``"quench_gas"`` (``q``, the signal ``"quench"``), ``"makeup"``,
    ``"vent"`` (all from the inventory and the signals), ``"purge"`` and
    ``"acid_gas"`` (from the separator gas). The treat gas is at the
    compressor's discharge temperature (ideal-gas compressor,
    :func:`~difflow_refinery.hydroprocessing.recycle.compress`), the makeup
    entering at the same temperature, as in the steady unit.
    """

    output_ports = ("treat_gas", "quench_gas", "makeup", "vent", "purge", "acid_gas")
    output_dependencies = {"treat_gas": ("signals",), "quench_gas": ("signals",), "makeup": ("signals",),
                           "vent": (), "purge": ("vapor",), "acid_gas": ("vapor",)}

    def __init__(self, layout: Layout, comps: Components, *, h2_target, makeup_y, purge, h2s_removal,
                 nh3_removal, tau, P_suction, P_discharge, T_suction, eta, makeup_max, relief_inventory,
                 tau_relief, scale: Array, y0: Array, name: str = "loop"):
        self.layout = layout
        self.comps = comps
        f = lambda v: jnp.asarray(v, dtype=float)
        self.h2_target, self.makeup_y, self.purge = f(h2_target), f(makeup_y), f(purge)
        self.h2s_removal, self.nh3_removal = f(h2s_removal), f(nh3_removal)
        self.tau, self.P_suction, self.P_discharge = f(tau), f(P_suction), f(P_discharge)
        self.T_suction, self.eta = f(T_suction), f(eta)
        self.makeup_max, self.relief_inventory, self.tau_relief = f(makeup_max), f(relief_inventory), f(tau_relief)
        self._scale = jnp.asarray(scale, dtype=float)
        self._y0 = jnp.asarray(y0)
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar(f"n_{g}", "moles", "mol", f"{g} in the loop inventory")
                          for g in self.layout.gases])

    def initial_state(self, inputs=None, params=None):
        return self._y0

    def _gas(self, v) -> Flows:
        lay = self.layout
        return Flows(v, jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))

    def inventory(self, state) -> Array:
        return state * self._scale

    def _supply(self, state, signals):
        n = self.inventory(state)
        R = self._gas(signals["compressor"] * n / self.tau)
        makeup, M = makeup_for_ratio(R, self.layout, self.makeup_y, self.h2_target)
        M_c = jnp.clip(M, 0.0, self.makeup_max)
        makeup = makeup.scale(M_c / jnp.where(M == 0.0, 1.0, M))
        T2, W = compress(R, self.layout, self.comps, self.T_suction, self.P_suction, self.P_discharge, self.eta)
        return R, makeup, T2, W

    def _vent(self, state):
        n = self.inventory(state)
        N = jnp.sum(n)
        rate = jnp.maximum(N - self.relief_inventory, 0.0) / self.tau_relief / jnp.maximum(N, 1e-300)
        return self._gas(n * rate)

    def _sweet(self, vapor_stream):
        lay = self.layout
        vap = Flows.from_stream(vapor_stream, lay)
        sweet, absorbed = amine_scrub(vap, lay, self.h2s_removal, self.nh3_removal)
        rec_in, purge = purge_split(sweet, self.purge)
        return rec_in, purge, absorbed

    def derivatives(self, t, state, inputs, params=None):
        rec_in, _, _ = self._sweet(inputs["vapor"])
        R, _, _, _ = self._supply(state, inputs["signals"])
        dn = rec_in.gas - R.gas - self._vent(state).gas
        return dn / self._scale

    def _state_outputs(self, state, signals):
        lay = self.layout
        R, makeup, T2, _ = self._supply(state, signals)
        G = R + makeup
        q = signals["quench"]
        return {"treat_gas": G.scale(1.0 - q).to_stream(lay, T2, self.P_discharge),
                "quench_gas": G.scale(q).to_stream(lay, T2, self.P_discharge),
                "makeup": makeup.to_stream(lay, T2, self.P_discharge),
                "vent": self._vent(state).to_stream(lay, self.T_suction, self.P_suction)}

    def _vapor_outputs(self, vapor):
        lay = self.layout
        _, purge, absorbed = self._sweet(vapor)
        return {"purge": purge.to_stream(lay, self.T_suction, self.P_suction),
                "acid_gas": absorbed.to_stream(lay, self.T_suction, self.P_suction)}

    def outputs(self, t, state, inputs, params=None):
        out = self._state_outputs(state, inputs["signals"])
        out.update(self._vapor_outputs(inputs["vapor"]))
        return out

    def partial_outputs(self, t, state, inputs, params=None):
        out = {}
        if "signals" in inputs:
            out.update(self._state_outputs(state, inputs["signals"]))
        else:
            out["vent"] = self._vent(state).to_stream(self.layout, self.T_suction, self.P_suction)
        if "vapor" in inputs:
            out.update(self._vapor_outputs(inputs["vapor"]))
        return out

    def compressor(self, state, signals) -> dict:
        """``{"recycle": mol/s, "T_out": K, "power": W, "makeup": mol/s H2}`` at a state."""
        R, makeup, T2, W = self._supply(state, signals)
        return {"recycle": jnp.sum(R.gas), "T_out": T2, "power": W,
                "makeup": makeup.gas[self.layout.gas_index("hydrogen")]}


__all__ = ["CATALYST_CP", "REST_TOL", "DynamicRestStateWarning", "choose_n_sub", "rest_mismatch", "warn_rest_state",
           "flows_scale", "holdup_outflow", "holdup_from_outflow", "phase_flows",
           "BedGeometry", "DynamicTrickleBedReactor", "FurnaceLag", "DynamicHPSeparator", "RecycleGasLoop"]
