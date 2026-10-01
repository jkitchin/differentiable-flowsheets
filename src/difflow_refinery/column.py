"""An equation-oriented crude distillation column.

An atmospheric crude column is a main column with no reboiler -- the crude
arrives partly vaporised from the furnace and is stripped with steam -- plus
the things that make it a *crude* column rather than a textbook one:

* **side products** drawn as liquid from intermediate stages, each usually
  finished in a small steam-stripped **side stripper** whose vapour returns
  to the main column one stage above the draw;
* **pumparounds**: liquid drawn from a stage, cooled outside the column and
  returned a stage or two higher, which removes heat part-way up and so sets
  the internal reflux in each section;
* a **condenser** that is total or partial, with the condensed stripping
  steam decanted as free water.

The model is MESH on every equilibrium stage, solved simultaneously (the
Naphtali-Sandholm approach, not the bubble-point method, which fails on a
feed boiling across 600 K). Stages of the main column and of every stripper
are nodes of one network; draws, returns and stripper vapour are edges of it.

Unknowns per stage are the log component liquid flows, the temperature and
the log *total* vapour flow, steam included. The hydrocarbon vapour follows
from Raoult, ``v_i = K_i x_i V``, and the water vapour from what is left,
``w = V (1 - sum K_i x_i)`` -- so the water balance is the stage's summation
equation, and steam lowering the hydrocarbon partial pressure needs no
special case. Water is never in the hydrocarbon liquid: it condenses only in
the overhead drum, as free water at its own vapour pressure.

Degrees of freedom follow a simulator's: a total condenser has one (the
distillate split), a partial condenser two, each side product one (its draw
rate), each pumparound two (rate and return temperature) and a
:class:`Furnace`, when there is one, its coil outlet temperature. They are closed
by :class:`Spec` s -- product rates, reflux ratio, temperatures, pumparound
duties, overflash, furnace outlet temperature or duty -- which the caller
lists, as many as there are freedoms.
Duties are not unknowns: the condenser and pumparound duties are evaluated
from the converged state.

The solve is a damped Newton from a bubble-point initialisation, then one
Newton step from the converged point with the Jacobian held constant,
which is what makes the result differentiable: the gradient of a
product rate or a duty with respect to a spec value, the feed, or the assay
behind the thermo is the implicit-function gradient, not a derivative taped
through the iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow_refinery.thermo import RHO_WATER_60F, WATER_MW, ColumnThermo

Basis = Literal["mole", "mass", "volume"]

#: One barrel, m^3. A rate of ``n`` barrels per day is ``n * BARREL / 86400`` m^3/s.
BARREL = 0.158987294928

# Newton: the largest change taken in one step in a log-flow and in a
# temperature, and the residual (infinity norm, scaled) called converged.
_MAX_LOG_STEP = 1.5
_MAX_T_STEP = 25.0
_LOG_SIGNIFICANT = float(np.log(1e-9))  # flows below 1e-9 of the feed
_BACKTRACKS = 12
# Bubble-point passes in the initialisation.
_INIT_PASSES = 25
# Volatility continuation (#301). A component whose vapour pressure at
# _PSAT_REF_T is below _PSAT_FLOOR -- a vacuum-residue lump boiling near
# 950 C has ~3e-5 Pa there -- stalls the damped Newton from the bubble-point
# start: the merit stops falling within three iterations at a residual of
# order one, although the same column converges in five iterations when
# started from the solution of a column whose lump is 1000x more volatile.
# So such a column is first solved with those vapour pressures raised to
# the floor, and the real one is solved from there. Measured on the test
# crude with a 950 C lump: 3e-3 Pa still stalls, 3e-2 Pa converges; the
# heaviest cut of the default (no heavy end) characterization has 7.6e-2 Pa,
# so a column without such a component never takes this path and its
# numbers are untouched.
_PSAT_REF_T = 600.0
_PSAT_FLOOR = 0.05
_T_SCALE = 100.0  # K, for temperature spec residuals


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SideProduct:
    """A liquid side product, optionally finished in a steam stripper.

    Attributes:
        name: Product name; the key of its stream in the column's result.
        draw_stage: Main-column stage the liquid is drawn from (1 = top).
        stripper_stages: Equilibrium stages in the side stripper. 0 draws the
            product straight from the column.
        steam: Stripping steam to the bottom of the stripper (mol/s). Required
            when there is a stripper: with neither steam nor a reboiler the
            stripper has nothing to strip with.
        return_stage: Main-column stage the stripper vapour returns to.
            Default: the stage above the draw.
    """

    name: str
    draw_stage: int
    stripper_stages: int = 3
    steam: float | Array = 0.0
    return_stage: int | None = None


@dataclass(frozen=True)
class Pumparound:
    """A pumparound: liquid drawn, cooled and returned higher up.

    Attributes:
        name: Pumparound name, used by its specs and in the result.
        draw_stage: Main-column stage the liquid is drawn from.
        return_stage: Main-column stage it is returned to; above the draw.
    """

    name: str
    draw_stage: int
    return_stage: int


@dataclass(frozen=True)
class Spec:
    """One column specification. Build these with the functions below.

    Attributes:
        kind: What is specified.
        target: The product, pumparound or stage it applies to.
        value: The specified value. Traceable: a gradient with respect to it
            is a sensitivity of the column to the spec.
        basis: ``"mole"`` (mol/s), ``"mass"`` (kg/s) or ``"volume"``
            (standard liquid m^3/s at 60 F) for rate specs.
    """

    kind: str
    target: str | int | None
    value: float | Array
    basis: Basis = "mole"


def product_rate(product: str, value, basis: Basis = "volume") -> Spec:
    """Rate of a product: the distillate, a side product, ``"residue"`` or ``"offgas"``."""
    return Spec("product_rate", product, value, basis)


def reflux_ratio(value) -> Spec:
    """Reflux over overhead product (liquid distillate plus offgas), molar."""
    return Spec("reflux_ratio", None, value)


def stage_temperature(stage: int, value) -> Spec:
    """Temperature (K) of a main-column stage; stage 0 is the condenser."""
    return Spec("stage_T", stage, value)


def pumparound_rate(pumparound: str, value, basis: Basis = "volume") -> Spec:
    return Spec("pa_rate", pumparound, value, basis)


def pumparound_duty(pumparound: str, value) -> Spec:
    """Heat removed by a pumparound (W, positive)."""
    return Spec("pa_duty", pumparound, value)


def pumparound_return_temperature(pumparound: str, value) -> Spec:
    return Spec("pa_return_T", pumparound, value)


def pumparound_delta_t(pumparound: str, value) -> Spec:
    """Draw temperature minus return temperature (K)."""
    return Spec("pa_delta_T", pumparound, value)


def overflash(value, basis: Basis = "volume") -> Spec:
    """Liquid falling from the stage above the feed, as a fraction of the feed.

    The overflash is the vapour that rises from the flash zone beyond what
    the side products take, condensed and returned to wash the wash zone. A
    few percent of the feed keeps the trays above the flash zone wet.
    """
    return Spec("overflash", None, value, basis)


def coil_outlet_temperature(value) -> Spec:
    """Furnace coil outlet temperature (K). Needs a :class:`Furnace`."""
    return Spec("furnace_T", None, value)


def furnace_duty(value) -> Spec:
    """Heat absorbed by the crude in the furnace (W). Needs a :class:`Furnace`."""
    return Spec("furnace_duty", None, value)


_SPEC_KINDS = {"product_rate", "reflux_ratio", "stage_T", "pa_rate", "pa_duty",
               "pa_return_T", "pa_delta_T", "overflash", "furnace_T", "furnace_duty"}


@dataclass(frozen=True)
class Furnace:
    """The fired heater in front of the column.

    With a furnace the column's feed is the furnace *inlet* -- the crude as
    the preheat train delivers it -- and the coil outlet temperature becomes
    an unknown of the column, closed by one more spec: the outlet temperature
    itself (:func:`coil_outlet_temperature`), the duty (:func:`furnace_duty`),
    or, as an operator runs it, the :func:`overflash` the outlet temperature
    has to deliver. Solving it with the column rather than in front of it is
    what lets an overflash set the furnace.

    Attributes:
        outlet_P: Coil outlet pressure (Pa), where the crude flashes. Default:
            the feed stage's pressure.
        efficiency: Absorbed over fired duty, for the fuel the furnace burns.
    """

    outlet_P: float | Array | None = None
    efficiency: float | Array = 0.85


@dataclass
class CrudeColumnParams(ParamsMixin):
    """Parameters for :class:`CrudeColumn`.

    Attributes:
        n_stages: Equilibrium stages in the main column, numbered 1 (top)
            to ``n_stages`` (bottom). The condenser is stage 0 and is extra.
        feed_stage: The flash-zone stage the furnace outlet enters.
        specs: The specifications, as many as the column has degrees of
            freedom (see :meth:`CrudeColumn.degrees_of_freedom`).
        side_products: Side products and their strippers.
        pumparounds: Pumparounds.
        P_top: Pressure on stage 1 (Pa).
        P_bottom: Pressure on the bottom stage (Pa); linear in between.
        P_condenser: Pressure in the overhead drum (Pa).
        condenser: ``"total"`` (no vapour product; the drum sits at its bubble
            point) or ``"partial"`` (an offgas, whose rate is then a freedom).
        bottom_steam: Stripping steam to the bottom stage (mol/s).
        steam_T: Temperature of all stripping steam (K).
        distillate_name: Name of the liquid overhead product.
        furnace: A :class:`Furnace` in front of the column, or ``None`` for a
            column fed at a known temperature.
        max_iter: Damped-Newton iterations before giving up (softly).
        tol: Converged when the scaled residual's infinity norm is below this.
    """

    n_stages: int
    feed_stage: int
    specs: tuple[Spec, ...]
    side_products: tuple[SideProduct, ...] = ()
    pumparounds: tuple[Pumparound, ...] = ()
    P_top: float | Array = 1.5e5
    P_bottom: float | Array = 2.0e5
    P_condenser: float | Array = 1.3e5
    condenser: Literal["total", "partial"] = "total"
    bottom_steam: float | Array = 0.0
    steam_T: float | Array = 600.0
    distillate_name: str = "naphtha"
    furnace: Furnace | None = None
    max_iter: int = 80
    tol: float = 1e-9

    def __post_init__(self):
        self.specs = tuple(self.specs)
        self.side_products = tuple(self.side_products)
        self.pumparounds = tuple(self.pumparounds)


# =============================================================================
# Static layout: the stage network as integer arrays
# =============================================================================


@dataclass(frozen=True)
class _Layout:
    n_main: int
    n_total: int  # equilibrium stages, main + strippers
    feed: int  # global index of the feed stage
    vap_dst: np.ndarray  # (M,) node each stage's vapour goes to; M = condenser
    liq_dst: np.ndarray  # (M,) node the net liquid goes to; -1 = a product
    draw_src: np.ndarray  # (n_draw,) side products first, then pumparounds
    draw_dst: np.ndarray  # (n_draw,) -1 = straight to product
    steam_node: np.ndarray  # (n_steam,) nodes receiving steam
    stage_of_main: np.ndarray  # (M,) main-column stage number, or the draw stage for a stripper
    product_nodes: dict  # side-product name -> ("liquid", node) or ("draw", k)
    n_side: int
    n_pa: int
    partial: bool
    furnace: bool

    @property
    def n_draw(self) -> int:
        return self.n_side + self.n_pa


def _layout(p: CrudeColumnParams) -> _Layout:
    N = int(p.n_stages)
    if N < 2:
        raise ValueError("a crude column needs at least two stages")
    if not 1 <= p.feed_stage <= N:
        raise ValueError(f"feed_stage {p.feed_stage} is not a stage of a {N}-stage column")
    if p.condenser not in ("total", "partial"):
        raise ValueError(f"condenser must be 'total' or 'partial', not {p.condenser!r}")

    # node j (0-based) sends vapour to j - 1; stage 1's goes to the condenser,
    # which is numbered after every stage and so is patched in at the end
    vap = list(range(-1, N - 1))
    liq = list(range(1, N)) + [-1]
    stage_of = list(range(1, N + 1))
    draw_src, draw_dst, steam_node, products = [], [], [], {}
    names = set()
    M = N
    for sp in p.side_products:
        if sp.name in names:
            raise ValueError(f"duplicate product name {sp.name!r}")
        names.add(sp.name)
        if not 1 <= sp.draw_stage <= N:
            raise ValueError(f"{sp.name}: draw stage {sp.draw_stage} is not in the column")
        ret = sp.draw_stage - 1 if sp.return_stage is None else sp.return_stage
        n = int(sp.stripper_stages)
        if n > 0:
            if not 1 <= ret <= N:
                raise ValueError(f"{sp.name}: return stage {ret} is not in the column")
            top = M
            for i in range(n):
                vap.append(ret - 1 if i == 0 else M - 1)
                liq.append(M + 1 if i < n - 1 else -1)
                stage_of.append(sp.draw_stage)
                M += 1
            steam_node.append(M - 1)
            draw_src.append(sp.draw_stage - 1)
            draw_dst.append(top)
            products[sp.name] = ("liquid", M - 1)
        else:
            draw_src.append(sp.draw_stage - 1)
            draw_dst.append(-1)
            products[sp.name] = ("draw", len(draw_src) - 1)
    for pa in p.pumparounds:
        if not 1 <= pa.return_stage < pa.draw_stage <= N:
            raise ValueError(f"pumparound {pa.name}: return stage must be above the draw, both in the column")
        draw_src.append(pa.draw_stage - 1)
        draw_dst.append(pa.return_stage - 1)
    for reserved in (p.distillate_name, "residue", "offgas", "water"):
        if reserved in names:
            raise ValueError(f"side product name {reserved!r} is taken")
    vap[0] = M  # the condenser is node M, after every stage
    steam_node = [N - 1] + steam_node
    return _Layout(
        n_main=N, n_total=M, feed=p.feed_stage - 1,
        vap_dst=np.asarray(vap, dtype=int), liq_dst=np.asarray(liq, dtype=int),
        draw_src=np.asarray(draw_src, dtype=int).reshape(-1),
        draw_dst=np.asarray(draw_dst, dtype=int).reshape(-1),
        steam_node=np.asarray(steam_node, dtype=int),
        stage_of_main=np.asarray(stage_of, dtype=int), product_nodes=products,
        n_side=len(p.side_products), n_pa=len(p.pumparounds),
        partial=p.condenser == "partial", furnace=p.furnace is not None,
    )


# =============================================================================
# Result
# =============================================================================


@dataclass(frozen=True)
class CrudeColumnResult:
    """A solved crude column.

    Attributes:
        products: Product streams (difflow stream dicts, ``F_<species>`` in
            mol/s plus ``T`` and ``P``), keyed by product name: the
            distillate, each side product, ``"residue"``, ``"water"`` (the
            decanted free water) and, with a partial condenser, ``"offgas"``.
        T: Main-column stage temperatures (K), stage 1 first.
        T_condenser: Overhead drum temperature (K).
        L, V: Liquid leaving each main-column stage (before draws) and vapour
            leaving it (steam included), mol/s.
        x: ``(n_stages, nc)`` liquid mole fractions on the main column.
        stripper_T: Temperatures of the stripper stages, in layout order.
        condenser_duty: Heat removed in the condenser (W, positive).
        pumparound_duty: Heat removed by each pumparound (W), in order.
        pumparound_rate: Each pumparound's molar rate (mol/s).
        pumparound_return_T: Each pumparound's return temperature (K).
        reflux: Molar reflux to stage 1 (mol/s).
        coil_outlet_T: Temperature the feed enters the flash zone at (K): the
            furnace outlet, solved for when there is a :class:`Furnace`, the
            feed's own ``T`` when there is not.
        feed_vaporized: Molar fraction of the hydrocarbon feed vapour at the
            coil outlet.
        furnace_duty: Heat absorbed in the furnace (W); 0 without one.
        furnace_fired_duty: ``furnace_duty / efficiency`` (W); 0 without one.
        water_saturation: ``y_w P / Psat_w(T)`` on each main stage. Above 1,
            free water would condense on that tray -- a column that has to be
            run hotter or with less steam.
        residual_norm: Scaled residual infinity norm at the returned state.
        converged: Whether that is below the tolerance.
        iterations: Damped-Newton iterations used.
    """

    products: dict
    T: Array
    T_condenser: Array
    L: Array
    V: Array
    x: Array
    stripper_T: Array
    condenser_duty: Array
    pumparound_duty: Array
    pumparound_rate: Array
    pumparound_return_T: Array
    reflux: Array
    coil_outlet_T: Array
    feed_vaporized: Array
    furnace_duty: Array
    furnace_fired_duty: Array
    water_saturation: Array
    residual_norm: Array
    converged: Array
    iterations: Array


jax.tree_util.register_dataclass(
    CrudeColumnResult,
    data_fields=["products", "T", "T_condenser", "L", "V", "x", "stripper_T", "condenser_duty",
                 "pumparound_duty", "pumparound_rate", "pumparound_return_T", "reflux",
                 "coil_outlet_T", "feed_vaporized", "furnace_duty", "furnace_fired_duty",
                 "water_saturation", "residual_norm", "converged", "iterations"],
    meta_fields=[],
)


# =============================================================================
# The column
# =============================================================================


class CrudeColumn:
    """Atmospheric crude column: side strippers, pumparounds, steam, free water.

    Args:
        params: :class:`CrudeColumnParams`.
        thermo: A :class:`~difflow_refinery.thermo.ColumnThermo`, usually
            ``ColumnThermo.from_characterization(crude)``.

    Calling the column on a feed stream returns the product streams;
    :meth:`solve` returns the whole :class:`CrudeColumnResult`. The feed is
    the furnace outlet: its ``T`` and ``P`` fix its enthalpy, its
    ``F_<component>`` must name the thermo's components, and an optional
    ``F_water`` enters as vapour.

    Example:
        >>> from difflow_refinery import Assay, characterize, ColumnThermo
        >>> from difflow_refinery import column as cc
        >>> crude = characterize(Assay([0, 30, 70, 100], [320., 480., 640., 900.], sg=0.85))
        >>> col = cc.CrudeColumn(cc.CrudeColumnParams(
        ...     n_stages=8, feed_stage=7, bottom_steam=0.2,
        ...     specs=(cc.product_rate("naphtha", 0.25, basis="mole"),)),
        ...     ColumnThermo.from_characterization(crude))
        >>> result = col.solve(crude.stream(1.0, T=620.0, P=2.0e5, basis="mole"))
        >>> bool(result.converged)
        True
    """

    def __init__(self, params: CrudeColumnParams, thermo: ColumnThermo):
        self.params = params
        self.thermo = thermo
        self.layout = _layout(params)
        for s in params.specs:
            if s.kind not in _SPEC_KINDS:
                raise ValueError(f"unknown spec kind {s.kind!r}")
        n_specs = len(params.specs)
        if n_specs != self.degrees_of_freedom():
            raise ValueError(
                f"the column has {self.degrees_of_freedom()} degrees of freedom "
                f"({'2' if self.layout.partial else '1'} at the condenser, "
                f"{self.layout.n_side} side-product draw(s), 2 x {self.layout.n_pa} "
                f"pumparound(s){', 1 furnace' if self.layout.furnace else ''}) "
                f"but {n_specs} spec(s) were given"
            )
        self._check_targets()

    def degrees_of_freedom(self) -> int:
        """How many specs the column needs."""
        lay = self.layout
        return (2 if lay.partial else 1) + lay.n_side + 2 * lay.n_pa + int(lay.furnace)

    def _check_targets(self):
        p = self.params
        products = {p.distillate_name, "residue", *(s.name for s in p.side_products)}
        if self.layout.partial:
            products.add("offgas")
        pas = {pa.name for pa in p.pumparounds}
        for s in p.specs:
            if s.kind == "product_rate" and s.target not in products:
                raise ValueError(f"no product {s.target!r}; products are {sorted(products)}")
            if s.kind.startswith("pa_") and s.target not in pas:
                raise ValueError(f"no pumparound {s.target!r}")
            if s.kind == "stage_T" and not 0 <= int(s.target) <= p.n_stages:
                raise ValueError(f"no stage {s.target}")
            if s.kind.startswith("furnace") and not self.layout.furnace:
                raise ValueError(f"a {s.kind} spec needs a Furnace in the column's params")
            if s.kind == "overflash" and p.feed_stage < 2:
                raise ValueError("overflash needs a stage above the feed")
            if s.basis not in ("mole", "mass", "volume"):
                raise ValueError(f"basis must be 'mole', 'mass' or 'volume', not {s.basis!r}")

    # ------------------------------------------------------------------

    def __call__(self, feed: dict) -> dict:
        return self.solve(feed).products

    def solve(self, feed: dict) -> CrudeColumnResult:
        """Solve the column for a feed stream."""
        args = self._args(feed)
        frozen = jax.lax.stop_gradient(args)
        z0 = self._initial_guess(frozen)
        if self._needs_continuation(frozen["thermo"]):
            easy = {**frozen, "thermo": self._raised_volatility(frozen["thermo"])}
            z0, _ = self._damped_newton(self._initial_guess(easy), easy)
        z1, iters = self._damped_newton(z0, frozen)
        # The implicit function theorem written as code: the value is the
        # converged z1, untouched, and the derivative with respect to
        # anything in ``args`` is -J^-1 dr/dargs, from one linear solve with
        # the Jacobian held constant. The iterations are never taped.
        # (optimistix's Newton would give the same derivative but polishes
        # with undamped steps first, and in the logs of components at 1e-30
        # of the feed an undamped step can diverge.)
        z1 = jax.lax.stop_gradient(z1)
        rel = lambda z, a: self._residual(z, a, relative=True)  # noqa: E731
        J = jax.lax.stop_gradient(jax.jacfwd(rel)(z1, frozen))
        dz = jnp.linalg.solve(J, rel(z1, args))
        z = z1 - (dz - jax.lax.stop_gradient(dz))
        norm = jnp.max(jnp.abs(self._residual(z1, frozen)))
        return self._result(z, args, norm, iters)

    @staticmethod
    def _needs_continuation(thermo) -> bool:
        """Whether a component is too involatile to start from the bubble point.

        Decided on concrete numbers. When they cannot be read (under ``jit``
        or a transformation that traces the thermo) the continuation is
        taken: an unneeded one costs an extra Newton run, a missing one
        fails the solve.
        """
        try:
            return bool(np.min(np.asarray(thermo.psat(_PSAT_REF_T))) < _PSAT_FLOOR)
        except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
            return True

    @staticmethod
    def _raised_volatility(thermo):
        """``thermo`` with every vapour pressure below the floor raised to it.

        Lee-Kesler's ``psat`` is proportional to ``Pc`` at fixed ``Tc`` and
        ``omega``, so scaling ``Pc`` lifts the whole curve by the same
        factor and keeps its shape (and the component's enthalpies).
        """
        ratio = _PSAT_FLOOR / thermo.psat(_PSAT_REF_T)
        return replace(thermo, Pc=thermo.Pc * jnp.maximum(ratio, 1.0))

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _args(self, feed: dict) -> dict:
        p, th = self.params, self.thermo
        missing = [n for n in th.names if f"F_{n}" not in feed]
        if missing:
            raise ValueError(f"feed has no flow for {missing[:5]}{'...' if len(missing) > 5 else ''}")
        extra = [k for k in feed if k.startswith("F_") and k[2:] not in th.names and k != "F_water"]
        if extra:
            raise ValueError(f"feed species {extra} are not components of the thermo")
        f = jnp.stack([jnp.asarray(feed[f"F_{n}"], dtype=float) for n in th.names])
        T_in = jnp.asarray(feed["T"], dtype=float)
        P_in = jnp.asarray(feed["P"], dtype=float)
        if p.furnace is not None:
            if p.furnace.outlet_P is None:
                frac = (p.feed_stage - 1) / max(p.n_stages - 1, 1)
                P_F = jnp.asarray(p.P_top, dtype=float) + (
                    jnp.asarray(p.P_bottom, dtype=float) - jnp.asarray(p.P_top, dtype=float)) * frac
            else:
                P_F = jnp.asarray(p.furnace.outlet_P, dtype=float)
            efficiency = jnp.asarray(p.furnace.efficiency, dtype=float)
        else:
            P_F, efficiency = P_in, jnp.asarray(1.0)
        steam = jnp.stack([jnp.asarray(p.bottom_steam, dtype=float)]
                          + [jnp.asarray(sp.steam, dtype=float)
                             for sp in p.side_products if sp.stripper_stages > 0])
        return {
            "thermo": th,
            "f": f,
            "f_water": jnp.asarray(feed.get("F_water", 0.0), dtype=float),
            # T_F/P_F: where the feed flashes; T_in/P_in: what arrives. The
            # same point without a furnace. With one, T_F is a placeholder
            # the solve replaces with its unknown.
            "T_F": T_in,
            "P_F": P_F,
            "T_in": T_in,
            "P_in": P_in,
            "efficiency": efficiency,
            "steam": steam,
            "T_steam": jnp.asarray(p.steam_T, dtype=float),
            "P_top": jnp.asarray(p.P_top, dtype=float),
            "P_bottom": jnp.asarray(p.P_bottom, dtype=float),
            "P_cond": jnp.asarray(p.P_condenser, dtype=float),
            "specs": jnp.stack([jnp.asarray(s.value, dtype=float) for s in p.specs]),
        }

    @staticmethod
    def _has_water(args) -> Array:
        return (jnp.sum(args["steam"]) + args["f_water"]) > 0

    def _pressures(self, args) -> Array:
        lay = self.layout
        frac = (lay.stage_of_main - 1) / max(lay.n_main - 1, 1)
        return args["P_top"] + (args["P_bottom"] - args["P_top"]) * jnp.asarray(frac)

    # ------------------------------------------------------------------
    # Unknowns
    # ------------------------------------------------------------------

    def _sizes(self):
        lay, nc = self.layout, self.thermo.n_components
        return lay.n_total * (nc + 2), nc + 3, lay.n_draw, lay.n_pa + int(lay.furnace)

    def _unpack(self, z, F):
        lay, nc = self.layout, self.thermo.n_components
        n_st, n_c, n_d, n_pa = self._sizes()
        st = z[:n_st].reshape(lay.n_total, nc + 2)
        c = z[n_st:n_st + n_c]
        d = z[n_st + n_c:n_st + n_c + n_d]
        tail = z[n_st + n_c + n_d:]
        tret = tail[:lay.n_pa]
        T_F = tail[lay.n_pa] if lay.furnace else None
        return dict(T_F=T_F,
            l=F * jnp.exp(st[:, :nc]), T=st[:, nc], V=F * jnp.exp(st[:, nc + 1]),
            l0=F * jnp.exp(c[:nc]), T0=c[nc], G=F * c[nc + 1], D=F * c[nc + 2],
            d=F * d, T_ret=tret,
        )

    def _pack(self, liq, T, V, l0, T0, G, D, d, T_ret, F, T_F=None):
        st = jnp.concatenate([jnp.log(liq / F), T[:, None], jnp.log(V / F)[:, None]], axis=1)
        tail = [T_ret] + ([jnp.reshape(T_F, (1,))] if self.layout.furnace else [])
        return jnp.concatenate([st.reshape(-1), jnp.log(l0 / F), jnp.stack([T0, G / F, D / F]),
                                d / F, *tail])

    # ------------------------------------------------------------------
    # The model
    # ------------------------------------------------------------------

    def _feed_split(self, args, T=None, P=None):
        """Raoult flash of the feed: (liquid, vapour) component flows.

        At the flash zone (``T_F``, ``P_F``) unless ``T`` and ``P`` are given.
        """
        th = args["thermo"]
        T = args["T_F"] if T is None else T
        P = args["P_F"] if P is None else P
        f, fw = args["f"], args["f_water"]
        total = jnp.sum(f) + fw
        z, zw = f / total, fw / total
        K = th.K(T, P)

        def rr(psi):
            return jnp.sum(z * (K - 1.0) / (1.0 + psi * (K - 1.0))) + zw / psi

        Ks, zs, zws = (jax.lax.stop_gradient(a) for a in (K, z, zw))

        def rr_s(psi):
            return jnp.sum(zs * (Ks - 1.0) / (1.0 + psi * (Ks - 1.0))) + zws / psi

        def bisect(_, lohi):
            lo, hi = lohi
            mid = 0.5 * (lo + hi)
            pos = rr_s(mid) > 0
            return jnp.where(pos, mid, lo), jnp.where(pos, hi, mid)

        lo, hi = jax.lax.fori_loop(0, 80, bisect, (jnp.asarray(1e-12), jnp.asarray(1.0)))
        psi0 = 0.5 * (lo + hi)
        # one Newton step from the root carries the implicit derivative
        psi = psi0 - rr(psi0) / jax.grad(rr)(psi0)
        all_vap = rr_s(jnp.asarray(1.0)) >= 0
        all_liq = (zws == 0) & (rr_s(jnp.asarray(1e-12)) <= 0)
        psi = jnp.where(all_vap, 1.0, jnp.where(all_liq, 0.0, psi))
        frac_v = jnp.where(all_vap, 1.0, jnp.where(all_liq, 0.0, psi * K / (1.0 + psi * (K - 1.0))))
        return f * (1.0 - frac_v), f * frac_v

    def _state(self, z, args):
        lay, th = self.layout, args["thermo"]
        M = lay.n_total
        F = jnp.sum(args["f"])
        u = self._unpack(z, F)
        liq, T, V = u["l"], u["T"], u["V"]
        P = self._pressures(args)
        L = jnp.sum(liq, axis=1)
        x = liq / L[:, None]
        K = th.K(T[:, None], P[:, None])
        sumKx = jnp.sum(K * x, axis=1)
        v = K * x * V[:, None]
        w = V * (1.0 - sumKx)

        l0, T0, G, D = u["l0"], u["T0"], u["G"], u["D"]
        L0 = jnp.sum(l0)
        x0 = l0 / L0
        K0 = th.K(T0, args["P_cond"])
        # free water in the drum only when there is water to condense; without
        # steam the drum is a plain hydrocarbon bubble point
        y_w0 = jnp.where(self._has_water(args), th.water_psat(T0) / args["P_cond"], 0.0)
        g = K0 * x0 * G

        d, T_ret = u["d"], u["T_ret"]
        src, dst = lay.draw_src, lay.draw_dst
        draw_total = jax.ops.segment_sum(d, src, num_segments=M) if lay.n_draw else jnp.zeros(M)
        phi = 1.0 - draw_total / L
        lnet = liq * phi[:, None]

        hL = th.h_liquid(T[:, None])
        hV = th.h_vapor(T[:, None])
        hwV = th.water_h_vapor(T)

        nodes = M + 2  # stages, condenser (M), and a sink (M + 1) for products
        sink = M + 1
        liq_dst = np.where(lay.liq_dst < 0, sink, lay.liq_dst)
        comp_in = jax.ops.segment_sum(lnet, liq_dst, num_segments=nodes)
        comp_in = comp_in + jax.ops.segment_sum(v, lay.vap_dst, num_segments=nodes)
        water_in = jax.ops.segment_sum(w, lay.vap_dst, num_segments=nodes)
        E_in = jax.ops.segment_sum(jnp.sum(lnet * hL, axis=1), liq_dst, num_segments=nodes)
        E_in = E_in + jax.ops.segment_sum(jnp.sum(v * hV, axis=1) + w * hwV, lay.vap_dst,
                                          num_segments=nodes)
        pa_duty = jnp.zeros(lay.n_pa)
        if lay.n_draw:
            xs = x[src]
            ddst = np.where(dst < 0, sink, dst)
            h_src = hL[src]
            # pumparounds return at their own temperature; side draws arrive as drawn
            h_draw = h_src
            if lay.n_pa:
                h_ret = th.h_liquid(T_ret[:, None])
                h_draw = jnp.concatenate([h_src[:lay.n_side], h_ret], axis=0)
                d_pa = d[lay.n_side:]
                pa_duty = d_pa * jnp.sum(xs[lay.n_side:] * (h_src[lay.n_side:] - h_ret), axis=1)
            comp_in = comp_in + jax.ops.segment_sum(d[:, None] * xs, ddst, num_segments=nodes)
            E_in = E_in + jax.ops.segment_sum(d * jnp.sum(xs * h_draw, axis=1), ddst,
                                              num_segments=nodes)
        # reflux
        reflux = l0 - D * x0
        hL0 = th.h_liquid(T0)
        comp_in = comp_in.at[0].add(reflux)
        E_in = E_in.at[0].add(jnp.sum(reflux * hL0))
        # feed, at the coil outlet
        if lay.furnace:
            args = {**args, "T_F": u["T_F"]}
        E_feed, f_vap = self._feed_enthalpy(args, args["T_F"], args["P_F"])
        Q_f = (E_feed - self._feed_enthalpy(args, args["T_in"], args["P_in"])[0]
               if lay.furnace else jnp.asarray(0.0))
        comp_in = comp_in.at[lay.feed].add(args["f"])
        water_in = water_in.at[lay.feed].add(args["f_water"])
        E_in = E_in.at[lay.feed].add(E_feed)
        # steam
        water_in = water_in.at[lay.steam_node].add(args["steam"])
        E_in = E_in.at[lay.steam_node].add(args["steam"] * th.water_h_vapor(args["T_steam"]))

        E_out = jnp.sum(liq * hL, axis=1) + jnp.sum(v * hV, axis=1) + w * hwV

        # condenser
        W = water_in[M] - y_w0 * G
        hV0 = th.h_vapor(T0)
        Q_c = E_in[M] - (jnp.sum(l0 * hL0) + jnp.sum(g * hV0)
                         + y_w0 * G * th.water_h_vapor(T0) + W * th.water_h_liquid(T0))

        return dict(
            F=F, l=liq, T=T, V=V, L=L, x=x, K=K, v=v, w=w, sumKx=sumKx, P=P, lnet=lnet,
            l0=l0, T0=T0, G=G, D=D, L0=L0, x0=x0, K0=K0, y_w0=y_w0, g=g, W=W,
            d=d, T_ret=T_ret, phi=phi, comp_in=comp_in, water_in=water_in, E_in=E_in,
            E_out=E_out, Q_c=Q_c, pa_duty=pa_duty, reflux=reflux,
            T_F=args["T_F"], Q_f=Q_f, feed_vaporized=jnp.sum(f_vap) / F,
        )

    def _feed_enthalpy(self, args, T, P):
        """Enthalpy flow (W) of the feed at ``T`` and ``P``, and its vapour flows."""
        th = args["thermo"]
        f_liq, f_vap = self._feed_split(args, T, P)
        E = (jnp.sum(f_liq * th.h_liquid(T)) + jnp.sum(f_vap * th.h_vapor(T))
             + args["f_water"] * th.water_h_vapor(T))
        return E, f_vap

    # rates of products and draws, on a basis -----------------------------

    def _rate(self, flows, water, basis, th):
        if basis == "mole":
            return jnp.sum(flows) + water
        if basis == "mass":
            return th.mass(flows) + water * WATER_MW / 1000.0
        return th.std_volume(flows) + water * WATER_MW / 1000.0 / RHO_WATER_60F

    def _product_flows(self, s):
        """Component flows (nc,) and water flow of every product."""
        lay, p = self.layout, self.params
        out = {
            p.distillate_name: (s["D"] * s["x0"], jnp.asarray(0.0)),
            "residue": (s["lnet"][lay.n_main - 1], jnp.asarray(0.0)),
            "water": (jnp.zeros_like(s["x0"]), s["W"]),
        }
        if lay.partial:
            out["offgas"] = (s["g"], s["y_w0"] * s["G"])
        for sp in p.side_products:
            kind, idx = lay.product_nodes[sp.name]
            if kind == "liquid":
                out[sp.name] = (s["lnet"][idx], jnp.asarray(0.0))
            else:
                out[sp.name] = (s["d"][idx] * s["x"][lay.draw_src[idx]], jnp.asarray(0.0))
        return out

    def _residual(self, z, args, relative=False):
        lay = self.layout
        M = lay.n_total
        s = self._state(z, args)
        F = s["F"]
        steam_scale = jnp.sum(args["steam"]) + args["f_water"] + 1e-3 * F
        E_scale = 3e4 * F
        # Two scalings of the component balances, with the same roots.
        #
        # ABSOLUTE (by the component's feed) for the Newton iteration: far
        # from the solution a trace component can have in/out of 1e10, and a
        # relative residual hands that to the line search's merit function.
        #
        # RELATIVE (by what leaves the stage, held constant) for the
        # implicit-gradient solve: a residue pseudo-component reaches the top
        # tray at 1e-30 of its feed, so scaled by its feed both its row and
        # (through l = e^z) its column of the Jacobian are 1e-30 -- a block no
        # pivoting recovers, a condition number of 1e17 and a NaN gradient.
        # A constant row scaling changes neither the Newton direction nor the
        # implicit derivative, and makes every one of those rows order one.
        if relative:
            tiny = 1e-280 * F
            c = jax.lax.stop_gradient(s["l"] + s["v"] + tiny)
            c0 = jax.lax.stop_gradient(s["l0"] + s["g"] + tiny)
        else:
            c = c0 = args["f"] + 1e-9 * F
        r_comp = (s["comp_in"][:M] - s["l"] - s["v"]) / c
        r_water = (s["water_in"][:M] - s["w"]) / steam_scale
        r_E = (s["E_in"][:M] - s["E_out"]) / E_scale
        r_stage = jnp.concatenate([r_comp, r_water[:, None], r_E[:, None]], axis=1).reshape(-1)

        r_cc = (s["comp_in"][M] - s["l0"] - s["g"]) / c0
        r_sum = jnp.sum(s["K0"] * s["x0"]) + s["y_w0"] - 1.0
        r_cond = jnp.concatenate([r_cc, r_sum[None]])

        r_spec = self._spec_residuals(s, args)
        return jnp.concatenate([r_stage, r_cond, r_spec])

    def _spec_residuals(self, s, args):
        lay, p, th = self.layout, self.params, args["thermo"]
        F = s["F"]
        feed_rate = {b: self._rate(args["f"], args["f_water"], b, th) for b in ("mole", "mass", "volume")}
        products = None
        pa_index = {pa.name: i for i, pa in enumerate(p.pumparounds)}
        out = []
        if not lay.partial:
            out.append(s["G"] / F)
        for i, spec in enumerate(p.specs):
            val = args["specs"][i]
            if spec.kind == "product_rate":
                products = products if products is not None else self._product_flows(s)
                flows, water = products[spec.target]
                r = (self._rate(flows, water, spec.basis, th) - val) / feed_rate[spec.basis]
            elif spec.kind == "reflux_ratio":
                R = jnp.sum(s["reflux"]) / (s["D"] + s["G"])
                r = (R - val) / (1.0 + jnp.abs(val))
            elif spec.kind == "stage_T":
                T = s["T0"] if int(spec.target) == 0 else s["T"][int(spec.target) - 1]
                r = (T - val) / _T_SCALE
            elif spec.kind == "furnace_T":
                r = (s["T_F"] - val) / _T_SCALE
            elif spec.kind == "furnace_duty":
                r = (s["Q_f"] - val) / (3e4 * F)
            elif spec.kind == "overflash":
                above = lay.feed - 1
                r = (self._rate(s["lnet"][above], 0.0, spec.basis, th) / feed_rate[spec.basis]) - val
            else:
                k = pa_index[spec.target]
                src = lay.draw_src[lay.n_side + k]
                d = s["d"][lay.n_side + k]
                if spec.kind == "pa_rate":
                    r = (self._rate(d * s["x"][src], 0.0, spec.basis, th) - val) / feed_rate[spec.basis]
                elif spec.kind == "pa_duty":
                    r = (s["pa_duty"][k] - val) / (3e4 * F)
                elif spec.kind == "pa_return_T":
                    r = (s["T_ret"][k] - val) / _T_SCALE
                else:  # pa_delta_T
                    r = (s["T"][src] - s["T_ret"][k] - val) / _T_SCALE
            out.append(r)
        return jnp.stack(out)

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _damped_newton(self, z0, args):
        n_st, n_c, n_d, n_pa = self._sizes()
        lay, nc = self.layout, self.thermo.n_components
        # which unknowns are log flows, and which temperatures, for step caps
        kind = np.zeros(z0.shape[0], dtype=int)  # 0 other, 1 log, 2 T
        st = np.zeros((lay.n_total, nc + 2), dtype=int)
        st[:, :nc] = 1
        st[:, nc] = 2
        st[:, nc + 1] = 1
        kind[:n_st] = st.reshape(-1)
        kind[n_st:n_st + nc] = 1
        kind[n_st + nc] = 2
        kind[n_st + n_c + n_d:] = 2
        is_log, is_T = jnp.asarray(kind == 1), jnp.asarray(kind == 2)

        res = lambda z: self._residual(z, args)  # noqa: E731
        tol = self.params.tol

        def merit(r):
            return 0.5 * jnp.sum(r * r)

        def body(carry):
            z, r, it, _ = carry
            J = jax.jacfwd(res)(z)
            dz = jnp.linalg.solve(J, -r)
            dz = jnp.where(jnp.isfinite(dz), dz, 0.0)
            # The step is shortened to keep the largest change in a log flow
            # and in a temperature within bounds -- judged on the flows that
            # matter. A component at 1e-30 of the feed on a stage it cannot
            # reach has a huge, meaningless Newton step in its log, and
            # letting it set the step length freezes every other unknown.
            # Those are clipped one by one instead.
            significant = is_log & (z > _LOG_SIGNIFICANT)
            big_log = jnp.max(jnp.where(significant, jnp.abs(dz), 0.0))
            big_T = jnp.max(jnp.where(is_T, jnp.abs(dz), 0.0))
            cap = jnp.minimum(1.0, jnp.minimum(_MAX_LOG_STEP / (big_log + 1e-30),
                                               _MAX_T_STEP / (big_T + 1e-30)))
            dz = dz * cap
            dz = jnp.where(is_log, jnp.clip(dz, -_MAX_LOG_STEP, _MAX_LOG_STEP), dz)
            m0 = merit(r)

            def bt_cond(c):
                a, k, m = c
                return (k < _BACKTRACKS) & ~(m < (1.0 - 1e-4 * a) * m0)

            def bt_body(c):
                a, k, _ = c
                a = 0.5 * a
                m = merit(res(z + a * dz))
                return a, k + 1, jnp.where(jnp.isfinite(m), m, jnp.inf)

            m1 = merit(res(z + dz))
            m1 = jnp.where(jnp.isfinite(m1), m1, jnp.inf)
            a, _, _ = jax.lax.while_loop(bt_cond, bt_body, (jnp.asarray(1.0), 0, m1))
            z_new = z + a * dz
            r_new = res(z_new)
            return z_new, r_new, it + 1, jnp.max(jnp.abs(r_new))

        def cond(carry):
            _, _, it, norm = carry
            return (it < self.params.max_iter) & (norm > tol)

        r0 = res(z0)
        z, r, it, _ = jax.lax.while_loop(cond, body, (z0, r0, 0, jnp.max(jnp.abs(r0))))
        return z, it

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initial_guess(self, args):
        """A bubble-point start: product splits, temperatures, then compositions.

        1. The products are sliced off the feed in boiling order at their
           specified (or estimated) rates -- the split a perfect column would
           make.
        2. Each product sets the temperature of the stage it leaves from (dew
           point of the overhead, bubble points of the side draws), at the
           hydrocarbon partial pressure the steam leaves; stages in between
           are interpolated.
        3. Total flows by constant molar overflow from the product rates.
        4. Component flows by solving the (linear, at fixed T and total flows)
           component balances on the whole network, then a few bubble-point
           passes updating T.
        """
        lay, p, th = self.layout, self.params, args["thermo"]
        nc, M, N = th.n_components, lay.n_total, lay.n_main
        f, F = args["f"], jnp.sum(args["f"])
        if lay.furnace:
            args = {**args, "T_F": self._guess_coil_outlet(args)}
        f_liq, f_vap = self._feed_split(args)
        V_feed = jnp.sum(f_vap)
        P = self._pressures(args)
        steam = args["steam"]
        steam_total = jnp.sum(steam) + args["f_water"]
        specs = {(s.kind, s.target): (i, s) for i, s in enumerate(p.specs)}

        # --- 1. product rates and compositions -----------------------------
        order = []  # light to heavy
        if lay.partial:
            order.append("offgas")
        order.append(p.distillate_name)
        for sp in sorted(p.side_products, key=lambda s: s.draw_stage):
            order.append(sp.name)

        factor = {"mole": jnp.ones(nc), "mass": th.MW / 1000.0,
                  "volume": th.MW / (1000.0 * th.SG * RHO_WATER_60F)}
        mean = {b: jnp.sum(f * factor[b]) / F for b in factor}
        amount, basis_of = {}, {}
        for name in order:
            key = ("product_rate", name)
            if key in specs:
                i, s = specs[key]
                amount[name] = args["specs"][i]
                basis_of[name] = s.basis
        light = th.Tb < 250.0
        if lay.partial and "offgas" not in amount:
            amount["offgas"] = jnp.maximum(jnp.sum(jnp.where(light, f, 0.0)), 1e-3 * F)
            basis_of["offgas"] = "mole"
        spec_moles = sum((amount[n] / mean[basis_of[n]] for n in amount), jnp.asarray(0.0))
        free = [n for n in order if n not in amount]
        if free:
            share = jnp.maximum((V_feed - spec_moles) / len(free), 0.02 * F)
            for n in free:
                amount[n], basis_of[n] = share, "mole"

        tb_order = jnp.argsort(th.Tb)
        remaining = f
        comp = {}
        for name in order:
            fac = factor[basis_of[name]]
            r_sorted = remaining[tb_order]
            u_sorted = r_sorted * fac[tb_order]
            before = jnp.cumsum(u_sorted) - u_sorted
            take = jnp.clip((amount[name] - before) / jnp.maximum(u_sorted, 1e-300), 0.0, 1.0)
            taken = jnp.zeros(nc).at[tb_order].set(take * r_sorted)
            taken = jnp.maximum(taken, 1e-12 * F)
            comp[name] = taken
            remaining = jnp.maximum(remaining - taken, 1e-12 * F)
        comp["residue"] = remaining
        rate = {n: jnp.sum(c) for n, c in comp.items()}

        # --- 2. temperatures --------------------------------------------------
        top_rate = rate[p.distillate_name] + (rate["offgas"] if lay.partial else 0.0)
        top_comp = comp[p.distillate_name] + (comp["offgas"] if lay.partial else 0.0)
        if ("reflux_ratio", None) in specs:
            R = jnp.maximum(args["specs"][specs[("reflux_ratio", None)][0]], 0.2)
        else:
            # Without a reflux spec the reflux is whatever the feed's vapour
            # leaves over: what rises from the flash zone and is not drawn
            # off as product comes back down. Half of it, as the pumparounds
            # take some of that heat out lower down.
            above = sum((rate[n] for n in order), jnp.asarray(0.0))
            R = jnp.maximum(0.5 * (V_feed - above), top_rate) / top_rate
        V_top = (1.0 + R) * top_rate

        def y_water(V_hc):
            return steam_total / (steam_total + V_hc + 1e-30)

        def bubble(xn, Peff, lo=150.0, hi=1100.0):
            xn = xn / jnp.sum(xn)

            def body(_, c):
                a, b = c
                m = 0.5 * (a + b)
                hot = jnp.sum(th.K(m, Peff) * xn) > 1.0
                return jnp.where(hot, a, m), jnp.where(hot, m, b)

            a, b = jax.lax.fori_loop(0, 60, body, (jnp.asarray(lo), jnp.asarray(hi)))
            return 0.5 * (a + b)

        def dew(yn, Peff):
            yn = yn / jnp.sum(yn)

            def body(_, c):
                a, b = c
                m = 0.5 * (a + b)
                hot = jnp.sum(yn / th.K(m, Peff)) < 1.0
                return jnp.where(hot, a, m), jnp.where(hot, m, b)

            a, b = jax.lax.fori_loop(0, 60, body, (jnp.asarray(150.0), jnp.asarray(1100.0)))
            return 0.5 * (a + b)

        anchors = {1: dew(top_comp, P[0] * (1.0 - y_water(V_top)))}
        for sp in p.side_products:
            j = sp.draw_stage
            if j not in anchors:
                anchors[j] = bubble(comp[sp.name], P[j - 1] * (1.0 - y_water(2.0 * V_top)))
        T_flash = args["T_F"] - 5.0
        anchors.setdefault(lay.feed + 1, T_flash)
        anchors.setdefault(N, T_flash - jnp.where(args["steam"][0] > 0, 15.0, 0.0))
        stages = sorted(anchors)
        T_anchor = jnp.stack([anchors[j] for j in stages])
        # anchors must run hot to cold going up; force monotone
        T_anchor = jax.lax.cummax(T_anchor)
        T_main = jnp.interp(jnp.arange(1, N + 1, dtype=float), jnp.asarray(stages, dtype=float), T_anchor)
        T = jnp.concatenate([T_main, T_main[lay.stage_of_main[N:] - 1]]) if M > N else T_main

        # --- 3. total flows by constant molar overflow ---------------------
        D = rate[p.distillate_name]
        G = rate["offgas"] if lay.partial else jnp.asarray(0.0)
        reflux_total = R * (D + G)
        # side draws: product plus what the stripper sends back up
        d_side, strip = [], []
        for sp in p.side_products:
            s_k = 0.3 * rate[sp.name] if sp.stripper_stages > 0 else 0.0 * rate[sp.name]
            d_side.append(rate[sp.name] + s_k)
            strip.append(s_k)
        d_pa, T_ret = [], []
        for pa in p.pumparounds:
            key = ("pa_rate", pa.name)
            if key in specs:
                i, s = specs[key]
                d_pa.append(args["specs"][i] / mean[s.basis])
            else:
                d_pa.append(0.5 * top_rate)
            T_draw = T_main[pa.draw_stage - 1]
            if ("pa_return_T", pa.name) in specs:
                T_ret.append(args["specs"][specs[("pa_return_T", pa.name)][0]])
            elif ("pa_delta_T", pa.name) in specs:
                T_ret.append(T_draw - args["specs"][specs[("pa_delta_T", pa.name)][0]])
            else:
                T_ret.append(T_draw - 60.0)
        d = jnp.stack(d_side + d_pa) if lay.n_draw else jnp.zeros(0)
        T_ret = jnp.stack(T_ret) if lay.n_pa else jnp.zeros(0)

        Lnet = []
        V_hc = []  # hydrocarbon vapour leaving each main stage
        residue = rate["residue"]
        v_strip = 0.05 * residue
        for j in range(1, N + 1):
            if j < lay.feed + 1:
                Ln = reflux_total + sum((d_pa[k] for k, pa in enumerate(p.pumparounds)
                                         if pa.return_stage <= j < pa.draw_stage), 0.0)
            elif j < N:
                Ln = residue + v_strip
            else:
                Ln = residue
            Lnet.append(Ln)
        for j in range(1, N + 1):
            if j == 1:
                Vj = reflux_total + D + G
            elif j <= lay.feed + 1:
                k = j - 1  # envelope over stages 1..j-1
                Vj = Lnet[k - 1] + D + G
                for i, sp in enumerate(p.side_products):
                    if sp.draw_stage <= k:
                        Vj = Vj + d_side[i]
                    ret = sp.draw_stage - 1 if sp.return_stage is None else sp.return_stage
                    if sp.stripper_stages > 0 and ret <= k:
                        Vj = Vj - strip[i]
                for i, pa in enumerate(p.pumparounds):
                    if pa.draw_stage <= k:
                        Vj = Vj + d_pa[i]
                    if pa.return_stage <= k:
                        Vj = Vj - d_pa[i]
            else:
                Vj = v_strip
            V_hc.append(Vj)
        Lnet = jnp.stack([jnp.asarray(a, dtype=float) for a in Lnet])
        V_hc = jnp.maximum(jnp.stack([jnp.asarray(a, dtype=float) for a in V_hc]), 1e-4 * F)
        # water vapour leaving each main stage
        w_main = []
        for j in range(1, N + 1):
            wj = steam[0]
            si = 1
            for sp in p.side_products:
                if sp.stripper_stages > 0:
                    ret = sp.draw_stage - 1 if sp.return_stage is None else sp.return_stage
                    if ret >= j:
                        wj = wj + steam[si]
                    si += 1
            if j <= lay.feed + 1:
                wj = wj + args["f_water"]
            w_main.append(wj)
        w_all = list(w_main)
        Lnet_all, V_hc_all = [Lnet], [V_hc]
        si = 1
        for i, sp in enumerate(p.side_products):
            n = int(sp.stripper_stages)
            if n:
                Lnet_all.append(jnp.full(n, rate[sp.name]) + jnp.concatenate([jnp.full(n - 1, strip[i]), jnp.zeros(1)]))
                V_hc_all.append(jnp.full(n, jnp.maximum(strip[i], 1e-4 * F)))
                w_all += [steam[si]] * n
                si += 1
        Lnet = jnp.concatenate(Lnet_all)
        V_hc = jnp.concatenate(V_hc_all)
        w_st = jnp.stack([jnp.asarray(a, dtype=float) for a in w_all])
        draw_total = jax.ops.segment_sum(d, lay.draw_src, num_segments=M) if lay.n_draw else jnp.zeros(M)
        L = jnp.maximum(Lnet + draw_total, 1e-4 * F)
        V = V_hc + w_st
        phi = 1.0 - draw_total / L
        yw = w_st / V

        # condenser
        L0 = reflux_total + D
        rho = reflux_total / L0
        x_top = top_comp / jnp.sum(top_comp)
        has_water = self._has_water(args)

        def cond_T(x0):
            def body(_, c):
                a, b = c
                m = 0.5 * (a + b)
                s = jnp.sum(th.K(m, args["P_cond"]) * x0)
                s = s + jnp.where(has_water, th.water_psat(m) / args["P_cond"], 0.0)
                hot = s > 1.0
                return jnp.where(hot, a, m), jnp.where(hot, m, b)

            a, b = jax.lax.fori_loop(0, 60, body, (jnp.asarray(150.0), jnp.asarray(1100.0)))
            return 0.5 * (a + b)

        T0 = cond_T(x_top)

        # --- 4. component balances at fixed T and flows ------------------
        nodes = M + 1
        C = M
        liq_dst = lay.liq_dst
        mask_l = liq_dst >= 0
        src_idx = np.arange(M)

        def comp_solve(T, T0):
            K = th.K(T[:, None], P[:, None])  # (M, nc)
            S = K * (V / L)[:, None]
            K0 = th.K(T0, args["P_cond"])
            S0 = K0 * G / L0
            diag = jnp.concatenate([1.0 + S, (1.0 + S0)[None]], axis=0)  # (nodes, nc)
            A = jnp.zeros((nc, nodes, nodes))
            A = A.at[:, np.arange(nodes), np.arange(nodes)].set(diag.T)
            A = A.at[:, lay.vap_dst, src_idx].add(-S.T)
            A = A.at[:, liq_dst[mask_l], src_idx[mask_l]].add(-phi[mask_l][None, :])
            if lay.n_draw:
                into = lay.draw_dst >= 0
                if into.any():
                    coef = (d / L[lay.draw_src])[into]
                    A = A.at[:, lay.draw_dst[into], lay.draw_src[into]].add(-coef[None, :])
            A = A.at[:, 0, C].add(-rho)
            b = jnp.zeros((nc, nodes)).at[:, lay.feed].set(f)
            sol = jnp.linalg.solve(A, b[..., None])[..., 0]  # (nc, nodes)
            sol = jnp.maximum(sol, 1e-30 * F)
            return sol[:, :M].T, sol[:, C]

        def pass_(_, TT):
            T, T0 = TT
            liq, l0 = comp_solve(T, T0)
            x = liq / jnp.sum(liq, axis=1, keepdims=True)
            T_new = jax.vmap(lambda xi, Pi: bubble(xi, Pi, 150.0, 1100.0))(x, P * (1.0 - yw))
            T0_new = cond_T(l0 / jnp.sum(l0))
            return 0.5 * (T + T_new), 0.5 * (T0 + T0_new)

        T, T0 = jax.lax.fori_loop(0, _INIT_PASSES, pass_, (T, T0))
        liq, l0 = comp_solve(T, T0)
        return self._pack(liq, T, V, l0, T0, G, D, d, T_ret, F, args["T_F"])

    def _guess_coil_outlet(self, args):
        """A starting furnace outlet temperature, from the spec that sets it.

        The outlet temperature itself if it is specified; the temperature
        that absorbs a specified duty; otherwise the temperature that
        vaporises, at the flash-zone pressure, what the specs take overhead
        and out the side, less the share the stripping steam lifts out of
        the residue, plus any overflash. Both are monotone in T: bisection.
        """
        p, th = self.params, args["thermo"]
        f, F = args["f"], jnp.sum(args["f"])
        specs = {s.kind: (i, s) for i, s in enumerate(p.specs)}
        if "furnace_T" in specs:
            return args["specs"][specs["furnace_T"][0]]
        factor = {"mole": jnp.ones(th.n_components), "mass": th.MW / 1000.0,
                  "volume": th.MW / (1000.0 * th.SG * RHO_WATER_60F)}
        mean = {b: jnp.sum(f * factor[b]) / F for b in factor}
        if "furnace_duty" in specs:
            target = args["specs"][specs["furnace_duty"][0]]
            E_in = self._feed_enthalpy(args, args["T_in"], args["P_in"])[0]

            def excess(T):
                return self._feed_enthalpy(args, T, args["P_F"])[0] - E_in - target
        else:
            want = jnp.asarray(0.0)
            for i, s in enumerate(p.specs):
                if s.kind == "product_rate" and s.target not in ("residue", "water"):
                    want = want + 0.85 * args["specs"][i] / mean[s.basis]
                elif s.kind == "overflash":
                    want = want + args["specs"][i] * F * mean["mole"] / mean[s.basis]
            want = jnp.clip(want, 0.05 * F, 0.9 * F)

            def excess(T):
                return jnp.sum(self._feed_split(args, T, args["P_F"])[1]) - want

        def body(_, c):
            a, b = c
            m = 0.5 * (a + b)
            hot = excess(m) > 0
            return jnp.where(hot, a, m), jnp.where(hot, m, b)

        a, b = jax.lax.fori_loop(0, 50, body, (jnp.asarray(300.0), jnp.asarray(800.0)))
        return 0.5 * (a + b)

    # ------------------------------------------------------------------

    def _result(self, z, args, norm, iters) -> CrudeColumnResult:
        lay, p, th = self.layout, self.params, args["thermo"]
        s = self._state(z, args)
        N = lay.n_main
        names = th.names

        def stream(flows, water, T, P):
            out = {f"F_{n}": flows[i] for i, n in enumerate(names)}
            out["F_water"] = water
            out["T"], out["P"] = T, P
            return out

        products = {}
        for name, (flows, water) in self._product_flows(s).items():
            if name in (p.distillate_name, "water", "offgas"):
                T, P = s["T0"], args["P_cond"]
            elif name == "residue":
                T, P = s["T"][N - 1], s["P"][N - 1]
            else:
                kind, idx = lay.product_nodes[name]
                node = idx if kind == "liquid" else lay.draw_src[idx]
                T, P = s["T"][node], s["P"][node]
            products[name] = stream(flows, water, T, P)

        y_w = s["w"] / s["V"]
        sat = y_w * s["P"] / th.water_psat(s["T"])
        Q_f = s["Q_f"]
        return CrudeColumnResult(
            products=products,
            T=s["T"][:N], T_condenser=s["T0"], L=s["L"][:N], V=s["V"][:N], x=s["x"][:N],
            stripper_T=s["T"][N:],
            condenser_duty=s["Q_c"], pumparound_duty=s["pa_duty"],
            pumparound_rate=s["d"][lay.n_side:], pumparound_return_T=s["T_ret"],
            reflux=jnp.sum(s["reflux"]), coil_outlet_T=s["T_F"], feed_vaporized=s["feed_vaporized"],
            furnace_duty=Q_f, furnace_fired_duty=Q_f / args["efficiency"],
            water_saturation=sat[:N],
            residual_norm=norm, converged=jnp.isfinite(norm) & (norm <= 100.0 * p.tol),
            iterations=iters,
        )


__all__ = [
    "BARREL", "CrudeColumn", "CrudeColumnParams", "CrudeColumnResult", "Furnace", "Pumparound",
    "SideProduct", "Spec", "coil_outlet_temperature", "furnace_duty", "overflash", "product_rate", "pumparound_delta_t",
    "pumparound_duty", "pumparound_rate", "pumparound_return_temperature",
    "reflux_ratio", "stage_temperature",
]
