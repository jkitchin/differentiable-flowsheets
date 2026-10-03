"""Gas-plant columns, a compressor train and an amine treater as units.

:class:`GasPlantColumn` puts a :class:`~difflow_refinery.gasplant.column.GasColumn`
behind the usual difflow interface -- a ``Params`` dataclass in, streams in,
streams and an ``info`` dict out -- with the column described by trays
(numbered from 1 at the top, as a drawing numbers them) rather than by
stages. The factories :func:`absorber_deethanizer`, :func:`debutanizer`,
:func:`splitter` and :func:`deisobutanizer` return its parameters with the
spec set each column is usually run on; every spec can be swapped (see
:class:`~difflow_refinery.vacuum.column.StageSpec`) and side draws added.

Tray efficiency. ``tray_efficiency=None`` takes O'Connell's (1946)
correlation for the overall efficiency, as fitted by
``E_o = 0.492 (alpha mu_L)^-0.245`` (mu_L in cP; the power-law fit is the
one Seader and Henley quote -- verify), and applies it as the Murphree
vapor efficiency of every tray. ``E_MV = E_o`` holds exactly only when the
stripping factor is one, so the column is a rating model of the same
accuracy as the correlation (about +-25 % on the efficiency) and no better.
Condenser and reboiler are equilibrium stages.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_refinery.gasplant.column import Feed, GasColumn, GasColumnLayout
from difflow_refinery.gasplant.components import GasComponents
from difflow_refinery.gasplant.thermo import PR, SRK, CubicThermo, feed_state
from difflow_refinery.vacuum.column import Route, StageSpec

#: Pressure drop per tray used by the factories (Pa): 0.1 psi.
TRAY_DP = 689.476

EOS = {"PR": PR, "SRK": SRK}


class GasColumnConvergenceWarning(UserWarning):
    """A gas-plant column did not converge."""


def oconnell_efficiency(alpha, mu_cP):
    """O'Connell's overall tray efficiency (fraction).

    ``E_o = 0.492 (alpha mu_L)^-0.245``, with ``alpha`` the key relative
    volatility and ``mu_L`` the liquid viscosity (cP), both at the average
    column conditions. Clipped to ``[0.1, 1]``.

    References:
        O'Connell, H. E. (1946) Trans. AIChE 42, 741-755 (the data and the
        curve). The power-law fit is the one quoted by Seader, Henley and
        Roper, Separation Process Principles (verify the attribution).
    """
    E = 0.492 * (jnp.asarray(alpha) * jnp.asarray(mu_cP)) ** -0.245
    return jnp.clip(E, 0.1, 1.0)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def column_layout(n_trays: int, feeds: Mapping[str, int], condenser: Optional[str] = "total",
                  reboiler: bool = True, side_draws: Mapping[str, int] = (),
                  groups=(), overhead_keys=(), reflux_guess: float = 2.0) -> GasColumnLayout:
    """A :class:`GasColumnLayout` from a tray description.

    Args:
        n_trays: Trays (equilibrium stages with a Murphree efficiency),
            numbered 1 at the top.
        feeds: ``{feed name: tray}``.
        condenser: ``"total"`` (liquid ``distillate``), ``"partial"``
            (vapor ``overhead``) or ``None`` (tray 1's vapor is the
            ``overhead``: an absorber).
        reboiler: A kettle reboiler below the last tray; without one the
            last tray's liquid is the ``bottoms``.
        side_draws: ``{draw name: tray}``: liquid side products.
        groups: Lumps reported as ``<product>.x.<group>``.
        overhead_keys: Components the initial guess sends overhead.
        reflux_guess: Reflux ratio of the initial guess.

    Returns:
        The layout. Stage numbers: the condenser, if any, is stage 0, tray
        ``t`` is stage ``t - 1 + (condenser is not None)``, the reboiler is
        the last stage.
    """
    if n_trays < 1:
        raise ValueError("n_trays must be >= 1")
    off = 0 if condenser is None else 1
    N = off + n_trays + (1 if reboiler else 0)
    draws = dict(side_draws)
    for name, t in list(dict(feeds).items()) + list(draws.items()):
        if not 1 <= int(t) <= n_trays:
            raise ValueError(f"{name}: tray {t} is not in 1..{n_trays}")
    routes = []
    if condenser == "total":
        routes += [Route("reflux", 0, 1, "reference"), Route("distillate", 0, None, "share")]
    elif condenser == "partial":
        routes += [Route("reflux", 0, 1, "reference")]
    elif condenser is not None:
        raise ValueError(f"condenser must be 'total', 'partial' or None, not {condenser!r}")
    by_stage = {}
    for name, t in draws.items():
        by_stage.setdefault(int(t) - 1 + off, []).append(name)
    for j in range(off, N):
        for name in by_stage.get(j, []):
            routes.append(Route(name, j, None, "share"))
        if j < N - 1:
            routes.append(Route(f"stage{j}.down", j, j + 1, "reference"))
        else:
            routes.append(Route("bottoms", j, None, "reference"))
    fds = tuple(Feed(n, int(t) - 1 + off) for n, t in dict(feeds).items())
    return GasColumnLayout(N, fds, tuple(routes), condenser, reboiler,
                           groups=tuple((g, tuple(m)) for g, m in groups),
                           overhead_keys=tuple(overhead_keys), reflux_guess=reflux_guess)


# ---------------------------------------------------------------------------
# Params and the unit
# ---------------------------------------------------------------------------


@dataclass
class GasPlantColumnParams(ParamsMixin):
    """Parameters for :class:`GasPlantColumn`.

    Attributes:
        components: The component set (:func:`~difflow_refinery.gasplant.gas_components`);
            feed streams' species are its names.
        n_trays: Trays, numbered from 1 at the top.
        feed_trays: ``{feed name: tray}``; the unit takes its feed streams in
            this order.
        condenser: ``"total"``, ``"partial"`` or ``None`` (absorber top).
        reboiler: Kettle reboiler below the last tray.
        side_draws: ``{draw name: tray}`` liquid side products.
        top_P: Top (condenser, or tray 1) pressure (Pa).
        tray_dP: Pressure rise per stage going down (Pa).
        specs: Specs, each trading a knob or a draw rate for a target.
        reboiler_duty: Reboiler duty (W), used only if no spec frees it.
        condenser_duty: Partial-condenser duty (W), used only if no spec
            frees it.
        draw_rates: ``{route: kg/s}`` for share routes (``distillate``, side
            draws) that no spec replaces.
        tray_efficiency: Murphree vapor efficiency of every tray; None takes
            O'Connell's correlation (needs ``keys``).
        keys: ``(light key, heavy key)`` for O'Connell's relative volatility.
        liquid_viscosity: Liquid viscosity for O'Connell (cP). The default
            is a typical value for C3-C6 liquids at column conditions, not a
            prediction.
        groups: Lumps reported as ``<product>.x.<group>`` (spec them too).
        overhead_keys: Components the initial guess sends overhead.
        reflux_guess: Reflux ratio of the initial guess.
        eos: ``"PR"`` or ``"SRK"``.
        max_iter: Newton iteration limit per pass.
        tol: Convergence tolerance on the largest scaled residual.
        continuation_steps: Steps from the easy problem to the user's.
    """

    components: GasComponents
    n_trays: int = 20
    feed_trays: Mapping = field(default_factory=lambda: {"feed": 10})
    condenser: Optional[str] = "total"
    reboiler: bool = True
    side_draws: Mapping = field(default_factory=dict)
    top_P: float = 10e5
    tray_dP: float = TRAY_DP
    specs: tuple = ()
    reboiler_duty: Optional[float] = None
    condenser_duty: Optional[float] = None
    draw_rates: Mapping = field(default_factory=dict)
    tray_efficiency: Optional[float] = 1.0
    keys: tuple = ()
    liquid_viscosity: float = 0.1
    groups: tuple = ()
    overhead_keys: tuple = ()
    reflux_guess: float = 2.0
    eos: str = "PR"
    max_iter: int = 60
    tol: float = 1e-9
    continuation_steps: int = 4

    def __post_init__(self):
        self.specs = tuple(self.specs)
        self.groups = tuple((g, tuple(m)) for g, m in self.groups)
        self.keys = tuple(self.keys)
        self.overhead_keys = tuple(self.overhead_keys)


def _freeze(m):
    return tuple(sorted((k, int(v)) for k, v in dict(m).items())) if not isinstance(m, tuple) else m


@lru_cache(maxsize=32)
def _compiled(layout: GasColumnLayout, names: tuple, spec_key: tuple, eos: str,
              max_iter: int, tol: float, continuation_steps: int):
    specs = tuple(StageSpec(o, 0.0, r, s) for (o, r, s) in spec_key)
    cubic = EOS[eos]
    col = GasColumn(layout, names, specs, thermo=lambda c: CubicThermo(c, cubic),
                    max_iter=max_iter, tol=tol, continuation_steps=continuation_steps)

    def run(th):
        x, it, rn = col.solve(th)
        thp = col.prepare(th)
        ctx = col.context(x, thp)
        outs = col.outputs_from(ctx)
        return dict(x=x, iterations=it, residual=rn, outputs=outs,
                    products=col.product_flows(ctx),
                    profiles={"T": ctx["T"], "P": ctx["P"], "L": ctx["L"], "V": ctx["V"],
                              "x": ctx["x"], "y": ctx["y"]},
                    max_residual=jnp.max(jnp.abs(col.residual(x, thp))))

    return col, jax.jit(run)


class GasPlantColumn:
    """A light-ends column on a cubic EOS: deethanizer, debutanizer,
    splitter, absorber.

    Rigorous MESH (Naphtali-Sandholm, log flows) with Peng-Robinson (or
    SRK) K-values and enthalpies for real components and petroleum
    pseudocomponents alike; total or partial condenser or none (absorber),
    reboiler or none, any number of feeds and liquid side draws. Any
    output can be specified in place of a knob (see
    :class:`~difflow_refinery.vacuum.column.StageSpec`). Differentiable
    with respect to the feeds, every knob and target, the component
    constants and the ``kij`` -- by implicit-function gradients of the
    converged equations.

    Example:
        >>> from difflow_refinery.gasplant import gas_components, debutanizer, GasPlantColumn
        >>> comps = gas_components(["propane", "n_butane", "n_pentane", "n_hexane"])
        >>> col = GasPlantColumn(debutanizer(comps, n_trays=12, feed_tray=6))
        >>> feed = {"F_propane": 10., "F_n_butane": 30., "F_n_pentane": 30.,
        ...         "F_n_hexane": 30., "T": 370., "P": 12e5}
        >>> lpg, naphtha, info = col(feed)
        >>> bool(info["converged"])
        True
    """

    symbol = "GPC"
    equations = [
        r"\ln(l_{ij} + v_{ij}) = \ln \sum (\text{liquid routed in} + v_{i,j+1} + f_{ij})",
        r"y_{ij} = \eta_j K_{ij}(T_j, P_j, x_j, y_j)\, x_{ij} + (1 - \eta_j)\, y_{i,j+1}",
        r"\ln K_{ij} = \ln\hat\phi^L_i - \ln\hat\phi^V_i \quad \text{(PR or SRK)}",
        r"\sum_i l_{ij} h^L_j + v_{ij} h^V_j = H_{\mathrm{in},j} + Q_j",
    ]
    assumptions = [
        "Equilibrium condenser and reboiler; trays with a Murphree vapor efficiency",
        "K evaluated at the actual (not equilibrium) vapor for eta < 1",
        "One cubic EOS for every phase and component; kij zero unless tabulated",
        "No hydraulics: flooding, weeping and entrainment are not checked",
    ]
    references = [
        "Peng, D.-Y.; Robinson, D. B. (1976) Ind. Eng. Chem. Fundam. 15, 59-64",
        "Soave, G. (1972) Chem. Eng. Sci. 27, 1197-1203",
        "Naphtali, L. M.; Sandholm, D. P. (1971) AIChE J. 17, 148-153",
        "O'Connell, H. E. (1946) Trans. AIChE 42, 741-755",
        "Poling, B. E.; Prausnitz, J. M.; O'Connell, J. P. (2001) The Properties "
        "of Gases and Liquids, 5th ed., McGraw-Hill",
        "Watkins, R. N. (1979) Petroleum Refinery Distillation, 2nd ed., Gulf",
    ]
    parameter_units = {
        "n_trays": "-", "top_P": "Pa", "tray_dP": "Pa", "reboiler_duty": "W",
        "condenser_duty": "W", "tray_efficiency": "-", "liquid_viscosity": "cP",
        "reflux_guess": "-", "max_iter": "-", "tol": "-", "continuation_steps": "-",
    }

    def __init__(self, params: GasPlantColumnParams):
        self.params = p = params
        self.layout = column_layout(
            p.n_trays, dict(p.feed_trays), p.condenser, p.reboiler, dict(p.side_draws),
            p.groups, p.overhead_keys, p.reflux_guess)
        self.names = tuple(p.components.names)
        self.specs = tuple(p.specs)
        if p.eos not in EOS:
            raise ValueError(f"eos must be one of {sorted(EOS)}, not {p.eos!r}")
        if p.tray_efficiency is None and len(p.keys) != 2:
            raise ValueError("tray_efficiency=None (O'Connell) needs keys=(light, heavy)")
        for k in p.keys:
            p.components.index(k)
        spec_key = tuple((s.output, s.replaces, s.scale) for s in self.specs)
        self.column, self._run = _compiled(self.layout, self.names, spec_key, p.eos,
                                           int(p.max_iter), float(p.tol),
                                           int(p.continuation_steps))
        replaced = {s.replaces for s in self.specs}
        need = []
        if p.reboiler:
            need.append(("reboiler.duty", p.reboiler_duty))
        if p.condenser == "partial":
            need.append(("condenser.duty", p.condenser_duty))
        for r in self.layout.routes:
            if r.kind == "share":
                need.append((f"{r.name}.rate", dict(p.draw_rates).get(r.name)))
        for name, val in need:
            if val is None and name not in replaced:
                raise ValueError(f"{name} has no value and no spec replaces it")
        self.feed_names = tuple(dict(p.feed_trays))
        self.products = self.layout.products

    @property
    def thermo(self):
        return CubicThermo(self.params.components, EOS[self.params.eos])

    def _flows(self, stream):
        unknown = [k for k in stream if k.startswith("F_") and k[2:] not in self.names]
        if unknown:
            raise ValueError(f"feed species {unknown} are not components of this column")
        return jnp.stack([jnp.asarray(stream.get(f"F_{n}", 0.0), dtype=float)
                          for n in self.names])

    def efficiency(self, feeds) -> jax.Array:
        """Tray (Murphree) efficiency: the parameter, or O'Connell's."""
        p = self.params
        if p.tray_efficiency is not None:
            return jnp.asarray(p.tray_efficiency, dtype=float)
        thermo = self.thermo
        F = sum(jnp.sum(f["flows"]) for f in feeds.values())
        T = sum(jnp.sum(f["flows"]) * f["T"] for f in feeds.values()) / F
        lnK = thermo.log_K_wilson(T, jnp.asarray(p.top_P, dtype=float))
        i, j = p.components.index(p.keys[0]), p.components.index(p.keys[1])
        return oconnell_efficiency(jnp.exp(lnK[i] - lnK[j]), p.liquid_viscosity)

    def theta(self, *feeds) -> dict:
        """The differentiable inputs of a solve, as a pytree."""
        p = self.params
        if len(feeds) != len(self.feed_names):
            raise ValueError(f"expected {len(self.feed_names)} feed streams "
                             f"({', '.join(self.feed_names)}), got {len(feeds)}")
        fd = {n: dict(flows=self._flows(s), T=jnp.asarray(s["T"], dtype=float),
                      P=jnp.asarray(s["P"], dtype=float))
              for n, s in zip(self.feed_names, feeds)}

        def val(v):
            return jnp.asarray(0.0 if v is None else v, dtype=float)

        knobs = {"top.P": val(p.top_P), "dP": val(p.tray_dP)}
        if p.condenser == "partial":
            knobs["condenser.duty"] = val(p.condenser_duty)
        if p.reboiler:
            knobs["reboiler.duty"] = val(p.reboiler_duty)
        for r in self.layout.routes:
            if r.kind == "share":
                knobs[f"{r.name}.rate"] = val(dict(p.draw_rates).get(r.name))
        E = self.efficiency(fd)
        lay = self.layout
        off = 0 if lay.condenser is None else 1
        eta = jnp.stack([jnp.asarray(1.0) if (j < off or (lay.reboiler and j == lay.n_stages - 1))
                         else E for j in range(lay.n_stages)]).astype(float)
        return {
            "components": p.components,
            "feeds": fd,
            "knobs": knobs,
            "targets": {s.output: jnp.asarray(s.target, dtype=float) for s in self.specs},
            "eta": eta,
        }

    def solve_theta(self, th) -> dict:
        res = dict(self._run(th))
        res["converged"] = res["residual"] < self.params.tol
        return res

    def solve(self, *feeds) -> dict:
        """Solve and return the raw result (outputs, products, profiles)."""
        res = self.solve_theta(self.theta(*feeds))
        self._warn(res)
        return res

    def _warn(self, res):
        conv = _concrete(res["converged"])
        if conv is not None and not conv:
            warnings.warn(
                f"GasPlantColumn did not converge: largest scaled residual "
                f"{float(res['residual']):.2e} after {int(res['iterations'])} iterations "
                f"(an infeasible spec -- a purity the trays cannot make -- looks like this)",
                GasColumnConvergenceWarning, stacklevel=3)

    def __call__(self, *feeds: Stream) -> tuple:
        """Run the column.

        Returns:
            One stream per product (``layout.products`` order: ``overhead``
            or ``distillate``, side draws, ``bottoms``), then ``info`` with
            ``converged``, ``iterations``, ``residual``, ``outputs`` (every
            column output, SI) and stage ``profiles``.
        """
        res = self.solve(*feeds)
        T, P = res["profiles"]["T"], res["profiles"]["P"]
        streams = []
        for name in self.products:
            fl = res["products"][name]
            j = self.column.product_stage(name)
            st = {f"F_{n}": fl[i] for i, n in enumerate(self.names)}
            st.update(T=T[j], P=P[j], phase="vapor" if name == "overhead" else "liquid")
            streams.append(st)
        info = {k: res[k] for k in ("converged", "iterations", "residual", "outputs",
                                    "profiles", "max_residual")}
        return (*streams, info)


def _concrete(v):
    """A Python value, or None under tracing."""
    try:
        return v.item() if hasattr(v, "item") else v
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _present(comps, names):
    return tuple(n for n in names if n in comps.names)


#: Lumps the factories report and spec. H2S is in none of them: it boils
#: between ethane and propane and splits between fuel gas and LPG, so it is
#: reported on its own (``<product>.x.hydrogen_sulfide``).
C2_MINUS = ("hydrogen", "nitrogen", "carbon_dioxide", "methane", "ethane", "ethylene")
C3 = ("propane", "propylene")
C4 = ("isobutane", "n_butane", "1_butene", "isobutylene", "cis_2_butene", "trans_2_butene")


def _groups(comps):
    heavy = tuple(n for n in comps.names
                  if n not in C2_MINUS + C3 + C4 + ("hydrogen_sulfide", "water"))
    return (("C2-", _present(comps, C2_MINUS)), ("C3", _present(comps, C3)),
            ("C4", _present(comps, C4)), ("C3-", _present(comps, C2_MINUS + C3)),
            ("C4+", _present(comps, C4) + heavy), ("C5+", heavy))


def absorber_deethanizer(components: GasComponents, n_trays: int = 30, lean_oil_tray: int = 1,
                         feed_tray: int = 12, top_P: float = 14e5,
                         c2_in_bottoms: Optional[float] = 0.005,
                         bottoms_T: Optional[float] = None, **kw) -> GasPlantColumnParams:
    """An absorber-deethanizer: no condenser, lean oil on top, reboiler.

    Feeds are ``"lean_oil"`` (tray ``lean_oil_tray``) and ``"feed"`` (the
    compressed gas and unstabilised naphtha, tray ``feed_tray``). The
    lean-oil rate is the absorber's lever: it is the lean-oil stream's own
    flow. The overhead is fuel gas, the bottoms go to the debutanizer.

    Spec: ``c2_in_bottoms`` (mole fraction of C2- in the bottoms) or
    ``bottoms_T`` (K), replacing the reboiler duty.
    """
    if (c2_in_bottoms is None) == (bottoms_T is None):
        raise ValueError("give exactly one of c2_in_bottoms and bottoms_T")
    spec = (StageSpec("bottoms.x.C2-", c2_in_bottoms, replaces="reboiler.duty")
            if bottoms_T is None else StageSpec("bottom.T", bottoms_T, replaces="reboiler.duty"))
    kw.setdefault("keys", ("ethane", "propane") if "propane" in components.names else ())
    kw.setdefault("tray_efficiency", None if kw["keys"] else 1.0)
    return GasPlantColumnParams(
        components=components, n_trays=n_trays,
        feed_trays={"lean_oil": lean_oil_tray, "feed": feed_tray},
        condenser=None, reboiler=True, top_P=top_P, specs=(spec,),
        groups=_groups(components), overhead_keys=_present(components, C2_MINUS), **kw)


def debutanizer(components: GasComponents, n_trays: int = 30, feed_tray: int = 15,
                top_P: float = 10e5, c5_in_lpg: float = 0.01,
                naphtha_rvp: Optional[float] = None, c4_in_naphtha: Optional[float] = 0.01,
                **kw) -> GasPlantColumnParams:
    """A debutanizer (stabiliser): total condenser, reboiler.

    LPG overhead (``distillate``), stabilised naphtha (``bottoms``).
    Specs: ``c5_in_lpg`` (C5+ mole fraction in the LPG), replacing the
    distillate rate, and either ``naphtha_rvp`` (Pa, the D323-analogue
    vapor pressure at 100 F) or ``c4_in_naphtha`` (mole fraction),
    replacing the reboiler duty. ``naphtha_rvp`` wins if both are given.
    """
    if naphtha_rvp is not None:
        s2 = StageSpec("bottoms.rvp", naphtha_rvp, replaces="reboiler.duty")
    elif c4_in_naphtha is not None:
        s2 = StageSpec("bottoms.x.C4", c4_in_naphtha, replaces="reboiler.duty")
    else:
        raise ValueError("give naphtha_rvp or c4_in_naphtha")
    kw.setdefault("keys", ("n_butane", "isopentane") if {"n_butane", "isopentane"}
                  <= set(components.names) else ())
    kw.setdefault("tray_efficiency", None if kw["keys"] else 1.0)
    return GasPlantColumnParams(
        components=components, n_trays=n_trays, feed_trays={"feed": feed_tray},
        condenser="total", reboiler=True, top_P=top_P,
        specs=(StageSpec("distillate.x.C5+", c5_in_lpg, replaces="distillate.rate"), s2),
        groups=_groups(components),
        overhead_keys=_present(components, C2_MINUS + C3 + C4), **kw)


def splitter(components: GasComponents, light: Sequence[str], heavy_spec: tuple,
             light_spec: tuple, n_trays: int = 40, feed_tray: int = 20,
             top_P: float = 17e5, **kw) -> GasPlantColumnParams:
    """A two-product splitter: total condenser, reboiler.

    Tray efficiency is O'Connell's when ``keys=(light_key, heavy_key)`` is
    given, otherwise 1 (``n_trays`` are then theoretical stages) unless
    ``tray_efficiency=`` says otherwise.

    ``light`` names the components the overhead is meant to carry (what the
    initial guess sends up); the group ``light`` and its complement
    ``heavy`` are reported and can be spec'd. ``light_spec`` and
    ``heavy_spec`` are ``(output, target)`` pairs, the first replacing the
    distillate rate, the second the reboiler duty: for a C3/C4 splitter,
    ``light_spec=("distillate.x.heavy", 0.025)`` (C4+ in propane) and
    ``heavy_spec=("bottoms.x.light", 0.02)`` (C3 in butane); for a naphtha
    splitter, the same with the cut placed by ``light``. Side draws go in
    ``side_draws=`` with their rates in ``draw_rates=``.
    """
    light = _present(components, light)
    heavy = tuple(n for n in components.names if n not in light)
    kw.setdefault("keys", ())
    kw.setdefault("tray_efficiency", None if kw["keys"] else 1.0)
    groups = kw.pop("groups", ()) + (("light", light), ("heavy", heavy))
    return GasPlantColumnParams(
        components=components, n_trays=n_trays, feed_trays={"feed": feed_tray},
        condenser="total", reboiler=True, top_P=top_P,
        specs=(StageSpec(light_spec[0], light_spec[1], replaces="distillate.rate"),
               StageSpec(heavy_spec[0], heavy_spec[1], replaces="reboiler.duty")),
        groups=groups, overhead_keys=light, **kw)


def c3c4_splitter(components: GasComponents, propane_purity: float = 0.95,
                  c3_in_butane: float = 0.02, **kw) -> GasPlantColumnParams:
    """A C3/C4 (depropaniser) :func:`splitter`: propane purity
    (``distillate.x.C3``, propane plus propylene) and C3 in the butane."""
    light = _present(components, C2_MINUS + C3)
    if {"propane", "isobutane"} <= set(components.names):
        kw.setdefault("keys", ("propane", "isobutane"))
    p = splitter(components, light, ("bottoms.x.light", c3_in_butane),
                 ("distillate.x.C3", propane_purity), groups=_groups(components), **kw)
    return p


def deisobutanizer(components: GasComponents, n_trays: int = 60, feed_tray: int = 30,
                   top_P: float = 7e5, ic4_purity: float = 0.95,
                   ic4_in_nc4: float = 0.05, **kw) -> GasPlantColumnParams:
    """A deisobutanizer: isobutane overhead to ``ic4_purity`` (mole
    fraction), ``ic4_in_nc4`` isobutane left in the normal butane."""
    light = _present(components, C2_MINUS + C3 + ("isobutane", "isobutylene"))
    kw.setdefault("keys", ("isobutane", "n_butane") if {"isobutane", "n_butane"}
                  <= set(components.names) else ())
    kw.setdefault("tray_efficiency", None if kw["keys"] else 1.0)
    return GasPlantColumnParams(
        components=components, n_trays=n_trays, feed_trays={"feed": feed_tray},
        condenser="total", reboiler=True, top_P=top_P,
        specs=(StageSpec("distillate.x.isobutane", ic4_purity, replaces="distillate.rate"),
               StageSpec("bottoms.x.isobutane", ic4_in_nc4, replaces="reboiler.duty")),
        groups=_groups(components), overhead_keys=light, **kw)


# ---------------------------------------------------------------------------
# Compressor train and amine treating
# ---------------------------------------------------------------------------


def _solve_T(fn, target, T0, n_iter=12):
    """``T`` with ``fn(T) = target``: Newton without gradients, then one
    frozen-derivative step carrying the implicit derivative.

    The Newton slope is ``grad(fn)`` with its RESULT stopped. Taking the
    gradient of the stopped function instead (``grad(stop_gradient(fn))``,
    as this did until the DWSIM comparison) is identically zero: every
    Newton step divided by zero, the clip bounced ``T`` between half and
    twice its value, and the answer was the single final step -- 1.3 K off
    the isentropic temperature and 3.6 % off the stage power on the DWSIM
    case (``tests/refinery/test_dwsim_gasplant.py``)."""
    Ts = jax.lax.stop_gradient(T0)
    tg = jax.lax.stop_gradient(target)
    fs = lambda T: jax.lax.stop_gradient(fn(T))  # noqa: E731

    def body(_, T):
        d = jax.lax.stop_gradient(jax.grad(fn)(T))
        return jnp.clip(T - (fs(T) - tg) / d, 0.5 * T, 2.0 * T)

    Ts = jax.lax.stop_gradient(jax.lax.fori_loop(0, n_iter, body, Ts))
    d = jax.lax.stop_gradient(jax.grad(fn)(Ts))
    return Ts - (fn(Ts) - target) / d


@dataclass
class GasCompressorParams(ParamsMixin):
    """Parameters for :class:`GasCompressor`.

    Attributes:
        components: The component set; stream species are its names.
        outlet_P: Discharge pressure of the last stage (Pa).
        n_stages: Compression stages, at equal pressure ratios.
        efficiency: Isentropic efficiency of each stage.
        cooler_T: Temperature each stage's aftercooler cools to (K); the
            knockout drum after it flashes there.
        cooler_dP: Pressure drop through each cooler and drum (Pa).
        eos: ``"PR"`` or ``"SRK"``.
    """

    components: GasComponents
    outlet_P: float = 14e5
    n_stages: int = 2
    efficiency: float = 0.75
    cooler_T: float = 313.15
    cooler_dP: float = 0.0
    eos: str = "PR"


class GasCompressor:
    """Wet-gas compressor: isentropic stages, aftercoolers, knockout drums.

    Each stage compresses the drum vapor at constant entropy to its
    discharge pressure, takes the actual work as the isentropic work over
    the efficiency, cools to ``cooler_T`` and flashes; the liquid is
    knocked out, the vapor goes on. The condensate of every drum is
    combined into one stream (in a gas plant it joins the absorber feed).

    Real or pseudo components alike, on the cubic EOS. Surge, choke and
    the compressor map are out of scope: the efficiency is a number.
    """

    symbol = "WGC"
    equations = [
        r"s^V(T_s, P_{k+1}, y) = s^V(T_k, P_k, y)",
        r"h_{k+1} = h_k + (h^V(T_s, P_{k+1}, y) - h_k) / \eta_s",
        r"W = \sum_k F_k (h_{k+1} - h_k)",
    ]
    assumptions = [
        "Equal pressure ratio per stage",
        "Isentropic efficiency constant per stage",
        "Compressor inlet is drum vapor (no liquid in the machine)",
    ]
    references = [
        "GPSA Engineering Data Book, 13th ed. (2012), Section 13 (Compressors and Expanders)",
        "Peng, D.-Y.; Robinson, D. B. (1976) Ind. Eng. Chem. Fundam. 15, 59-64",
    ]
    parameter_units = {"outlet_P": "Pa", "n_stages": "-", "efficiency": "-",
                       "cooler_T": "K", "cooler_dP": "Pa"}

    def __init__(self, params: GasCompressorParams):
        self.params = params
        if params.eos not in EOS:
            raise ValueError(f"eos must be one of {sorted(EOS)}, not {params.eos!r}")
        self.names = tuple(params.components.names)

    def run(self, flows, T, P):
        """Array form: ``(gas flows, condensate flows, T, P, info)``."""
        p = self.params
        th = CubicThermo(p.components, EOS[p.eos])
        flows = jnp.asarray(flows, dtype=float)
        n = int(p.n_stages)
        ratio = (jnp.asarray(p.outlet_P) / P) ** (1.0 / n)
        # the inlet may itself be two-phase: knock it out first
        st = feed_state(th, flows, T, P)
        gas = st["beta"] * jnp.sum(flows) * st["y"]
        cond = flows - gas
        Pk, Tk = P, T
        powers, T_dis = [], []
        for _ in range(n):
            y = gas / jnp.sum(gas)
            P2 = Pk * ratio
            h1 = th.h_vapor(Tk, Pk, y)
            s1 = th.s(Tk, Pk, y, "vapor")
            Ts = _solve_T(lambda t: th.s(t, P2, y, "vapor"), s1, Tk * ratio ** 0.25)
            h2 = h1 + (th.h_vapor(Ts, P2, y) - h1) / p.efficiency
            T2 = _solve_T(lambda t: th.h_vapor(t, P2, y), h2, Ts)
            powers.append(jnp.sum(gas) * (h2 - h1))
            T_dis.append(T2)
            Pk = P2 - p.cooler_dP
            Tk = jnp.asarray(p.cooler_T, dtype=float)
            st = feed_state(th, gas, Tk, Pk)
            v = st["beta"] * jnp.sum(gas) * st["y"]
            cond = cond + gas - v
            gas = v
        info = {"power": sum(powers), "stage_power": jnp.stack(powers),
                "discharge_T": jnp.stack(T_dis), "ratio": ratio}
        return gas, cond, Tk, Pk, info

    def __call__(self, stream: Stream) -> tuple[Stream, Stream, dict]:
        """Compress a stream. Returns ``(gas, condensate, info)``; ``info``
        has ``power`` (W), ``stage_power``, ``discharge_T`` (K) and the
        per-stage pressure ``ratio``."""
        unknown = [k for k in stream if k.startswith("F_") and k[2:] not in self.names]
        if unknown:
            raise ValueError(f"species {unknown} are not components")
        f = jnp.stack([jnp.asarray(stream.get(f"F_{n}", 0.0), dtype=float) for n in self.names])
        gas, cond, T, P, info = self.run(f, jnp.asarray(stream["T"], dtype=float),
                                         jnp.asarray(stream["P"], dtype=float))
        mk = lambda fl, ph: {**{f"F_{n}": fl[i] for i, n in enumerate(self.names)},  # noqa: E731
                             "T": T, "P": P, "phase": ph}
        return mk(gas, "vapor"), mk(cond, "liquid"), info


@dataclass
class AmineTreaterParams(ParamsMixin):
    """Parameters for :class:`AmineTreater`.

    Attributes:
        removal: ``{component: fraction removed}``; default 99 % of the
            H2S and none of the CO2.
    """

    removal: Mapping = field(default_factory=lambda: {"hydrogen_sulfide": 0.99})


class AmineTreater:
    """Amine contactor as a fixed removal fraction per component.

    The sweet gas keeps ``1 - removal`` of each listed component and all
    of the rest; the removed part leaves as acid gas. Absorption chemistry,
    solvent circulation and regeneration are out of scope (see
    ``difflow_cc.AmineAbsorber`` for a rate-based contactor).
    """

    symbol = "AMN"
    equations = [r"F^{\mathrm{sweet}}_i = (1 - r_i) F_i, \quad F^{\mathrm{acid}}_i = r_i F_i"]
    assumptions = ["Removal fractions are fixed numbers, not predictions"]
    references = ["Kohl, A. L.; Nielsen, R. B. (1997) Gas Purification, 5th ed., Gulf"]
    parameter_units = {"removal": "-"}

    def __init__(self, params: AmineTreaterParams):
        self.params = params

    def __call__(self, stream: Stream) -> tuple[Stream, Stream, dict]:
        """Returns ``(sweet gas, acid gas, info)``; ``info`` has the
        ``removed`` flows (mol/s)."""
        r = dict(self.params.removal)
        sweet, acid = {}, {}
        for k, v in stream.items():
            if k.startswith("F_"):
                f = jnp.asarray(r.get(k[2:], 0.0), dtype=float)
                sweet[k] = (1.0 - f) * v
                acid[k] = f * v
            else:
                sweet[k] = acid[k] = v
        return sweet, acid, {"removed": {k[2:]: v for k, v in acid.items() if k.startswith("F_")}}
