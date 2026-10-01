"""Crude assays and their characterisation into pseudo-components.

A crude is too many molecules to list, so a simulator describes it by its true
boiling point (TBP) curve -- the cumulative fraction distilled against
temperature in a column of many stages at high reflux -- plus a gravity, and
cuts that curve into narrow boiling ranges. Each range becomes one
*pseudo-component*, a fictitious species with the cut's average boiling point
and specific gravity, whose remaining properties come from
:mod:`difflow_refinery.correlations`.

The two steps here:

* :class:`Assay` holds the data as measured: TBP points, a bulk gravity or a
  gravity curve, and optionally a light-ends analysis (C1-C6 as real species).
* :func:`characterize` cuts it and returns a :class:`Characterization`, whose
  arrays -- boiling points, gravities, critical constants, fractions -- are
  ``jax`` arrays and differentiable with respect to the assay data. A gradient
  of a product yield with respect to a TBP point is therefore one ``jax.grad``,
  and an assay's measurement uncertainty can be pushed through a flowsheet
  with :mod:`difflow.uncertainty`.

:meth:`Characterization.species_data` and :meth:`Characterization.thermo`
turn the result into the ``SpeciesData`` / ``CriticalProperties`` that
difflow's :class:`~difflow.thermo.IdealThermo` and
:class:`~difflow.thermo.CubicThermo` take. Those classes hold Python floats,
so that step needs concrete values and is not traceable; it is the boundary
between the assay and a column built from it.

What this is *not* is an assay library. Curated assays are proprietary data
(see ``docs/pims-integration.md``); this module characterises the curve the
caller brings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.database import get_critical_props, get_species_data
from difflow.eos import CriticalProperties
from difflow.thermo import SpeciesData

from difflow_refinery import correlations as corr

Basis = Literal["volume", "mass"]

#: Standard-condition (60 F) liquid specific gravities of the light ends,
#: GPA 2145. Methane's 0.3 is the conventional value -- it has no liquid at
#: 60 F -- and only matters for a volume-basis light-ends analysis.
LIGHT_END_SG: dict[str, float] = {
    "methane": 0.3,
    "ethane": 0.35584,
    "propane": 0.50736,
    "isobutane": 0.56293,
    "n_butane": 0.58407,
    "neopentane": 0.59670,
    "isopentane": 0.62470,
    "n_pentane": 0.63086,
    "n_hexane": 0.66404,
    "water": 1.0,
}

#: Default cut widths (K) and the temperature (K) up to which each applies:
#: 20 C cuts to 400 C, 40 C cuts to 600 C, 100 C cuts beyond. Narrow where
#: the products are cut from each other, wide in the residue, where no cut
#: point falls.
DEFAULT_CUT_WIDTHS: tuple[tuple[float, float], ...] = (
    (673.15, 20.0),
    (873.15, 40.0),
    (np.inf, 100.0),
)

# The cut-boundary inversion: bisection to round-off, then one Newton step
# for the derivative (see _invert).
_BISECTION_STEPS = 60

# Vapour-pressure (Antoine) fit: window in K, intersected with
# 0.35 < T/Tc < 0.95, and the number of points it is fitted on.
DEFAULT_ANTOINE_WINDOW = (290.0, 750.0)
_ANTOINE_TR = (0.35, 0.95)
_ANTOINE_POINTS = 40
_ANTOINE_P_MIN = 10.0  # Pa


# =============================================================================
# The assay
# =============================================================================


@dataclass(frozen=True)
class Assay:
    """A crude assay: TBP curve, gravity, and optionally light ends.

    Attributes:
        tbp_percent: Cumulative percent distilled (0-100) at each TBP point,
            strictly increasing, starting at 0 (the initial boiling point)
            and ending at 100 (the final boiling point). A curve that stops
            short of 100 % has to be extrapolated before it can be cut, and
            that is a modelling decision left to the caller rather than made
            here silently.
        tbp_T: TBP temperature (K) at each point, strictly increasing.
        basis: Whether the percentages are by ``"volume"`` (liquid volume at
            60 F, the usual assay basis) or by ``"mass"``. Applies to the
            gravity curve and the light ends too.
        sg: Bulk specific gravity (60 F / 60 F) of the whole crude. With no
            ``sg_curve``, each cut's gravity follows from a Watson K constant
            across the crude, chosen so the cuts recombine to this gravity.
        api: Bulk API gravity -- the alternative to ``sg``. Give one.
        sg_curve: ``(mid_percent, SG)`` pairs: the gravity of the material
            at each cumulative percent. Each cut takes the average over its
            span (linear between points, flat beyond the ends). Overrides
            ``sg``/``api`` for the cut gravities.
        light_ends: ``{species: percent of the whole crude}`` for light ends
            measured as real species (difflow database names: ``methane``,
            ``ethane``, ``propane``, ``isobutane``, ``n_butane``,
            ``isopentane``, ``n_pentane``, ...). They replace the bottom of
            the TBP curve: pseudo-components start where the curve reaches
            their total.
        name: A label for reports.
    """

    tbp_percent: Sequence[float] | Array
    tbp_T: Sequence[float] | Array
    basis: Basis = "volume"
    sg: float | Array | None = None
    api: float | Array | None = None
    sg_curve: tuple[Sequence[float], Sequence[float]] | None = None
    light_ends: Mapping[str, float | Array] = field(default_factory=dict)
    name: str = "crude"

    def __post_init__(self):
        if self.basis not in ("volume", "mass"):
            raise ValueError(f"basis must be 'volume' or 'mass', not {self.basis!r}")
        pct = np.asarray(self.tbp_percent, dtype=float) if not _traced(self.tbp_percent) else None
        T = np.asarray(self.tbp_T, dtype=float) if not _traced(self.tbp_T) else None
        if pct is not None:
            if pct.ndim != 1 or pct.size < 2:
                raise ValueError("tbp_percent needs at least two points")
            if abs(pct[0]) > 1e-9 or abs(pct[-1] - 100.0) > 1e-9:
                raise ValueError(
                    f"tbp_percent must run from 0 to 100 (got {pct[0]} to {pct[-1]}); "
                    "extrapolate the curve to its initial and final boiling points first"
                )
            if np.any(np.diff(pct) <= 0):
                raise ValueError("tbp_percent must be strictly increasing")
        if T is not None:
            if pct is not None and T.shape != pct.shape:
                raise ValueError("tbp_T and tbp_percent must have the same length")
            if np.any(np.diff(T) <= 0):
                raise ValueError("tbp_T must be strictly increasing")
        if self.sg_curve is None and (self.sg is None) == (self.api is None):
            raise ValueError("give exactly one of sg or api (or an sg_curve)")
        for species in self.light_ends:
            if species not in LIGHT_END_SG:
                raise ValueError(
                    f"no standard liquid gravity for light end {species!r}; "
                    f"known: {', '.join(LIGHT_END_SG)}"
                )

    @property
    def bulk_sg(self) -> Array | None:
        """The bulk specific gravity given, from ``sg`` or ``api``."""
        if self.sg is not None:
            return jnp.asarray(self.sg, dtype=float)
        if self.api is not None:
            return corr.sg_from_api(jnp.asarray(self.api, dtype=float))
        return None


def _traced(x) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(x))


# =============================================================================
# Monotone interpolation of the TBP curve
# =============================================================================


def _pchip_slopes(x: Array, y: Array) -> Array:
    """Fritsch-Carlson slopes (the scheme scipy's ``PchipInterpolator`` uses).

    For a strictly increasing curve every secant is positive and the interior
    slope is a weighted harmonic mean of its neighbours -- smooth in the data,
    which is what keeps a cut fraction differentiable in the TBP points.
    """
    h = jnp.diff(x)
    delta = jnp.diff(y) / h
    if x.shape[0] == 2:
        return jnp.stack([delta[0], delta[0]])
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    same = delta[:-1] * delta[1:] > 0
    safe_prev = jnp.where(same, delta[:-1], 1.0)
    safe_next = jnp.where(same, delta[1:], 1.0)
    interior = jnp.where(same, (w1 + w2) / (w1 / safe_prev + w2 / safe_next), 0.0)

    def _end(h0, h1, d0, d1):
        d = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        d = jnp.where(jnp.sign(d) != jnp.sign(d0), 0.0, d)
        return jnp.where((jnp.sign(d0) != jnp.sign(d1)) & (jnp.abs(d) > 3.0 * jnp.abs(d0)),
                         3.0 * d0, d)

    first = _end(h[0], h[1], delta[0], delta[1])
    last = _end(h[-1], h[-2], delta[-1], delta[-2])
    return jnp.concatenate([first[None], interior, last[None]])


@dataclass(frozen=True)
class _Pchip:
    """Monotone cubic through ``(x, y)``, with its exact antiderivative."""

    x: Array
    y: Array
    d: Array

    @classmethod
    def fit(cls, x, y):
        x = jnp.asarray(x, dtype=float)
        y = jnp.asarray(y, dtype=float)
        return cls(x, y, _pchip_slopes(x, y))

    def _segment(self, t):
        k = jnp.clip(jnp.searchsorted(self.x, t, side="right") - 1, 0, self.x.shape[0] - 2)
        h = self.x[k + 1] - self.x[k]
        return k, h, (t - self.x[k]) / h

    def __call__(self, t: Array) -> Array:
        k, h, s = self._segment(t)
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return (h00 * self.y[k] + h10 * h * self.d[k]
                + h01 * self.y[k + 1] + h11 * h * self.d[k + 1])

    def derivative(self, t: Array) -> Array:
        k, h, s = self._segment(t)
        return ((6 * s**2 - 6 * s) * self.y[k] / h + (3 * s**2 - 4 * s + 1) * self.d[k]
                + (-6 * s**2 + 6 * s) * self.y[k + 1] / h + (3 * s**2 - 2 * s) * self.d[k + 1])

    def integral(self, t: Array) -> Array:
        """``integral(x[0] -> t)`` of the interpolant, exactly."""
        h = jnp.diff(self.x)
        full = h * ((self.y[:-1] + self.y[1:]) / 2.0 + h * (self.d[:-1] - self.d[1:]) / 12.0)
        cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(full)])
        k, hk, s = self._segment(t)
        H00 = s - s**3 + s**4 / 2
        H10 = s**2 / 2 - 2 * s**3 / 3 + s**4 / 4
        H01 = s**3 - s**4 / 2
        H11 = -(s**3) / 3 + s**4 / 4
        part = hk * (H00 * self.y[k] + H10 * hk * self.d[k]
                     + H01 * self.y[k + 1] + H11 * hk * self.d[k + 1])
        return cumulative[k] + part

    def invert(self, target: Array) -> Array:
        """The ``t`` with ``self(t) = target``, for an increasing interpolant.

        Bisection finds it to round-off with the gradient stopped; one Newton
        step from there then carries the implicit-function derivative
        ``dt = -(d self)/self'`` -- exact at the root, so the result is
        differentiable in both the data and the target.
        """
        lo0 = jax.lax.stop_gradient(self.x[0])
        hi0 = jax.lax.stop_gradient(self.x[-1])
        frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, self)
        tgt = jax.lax.stop_gradient(target)

        def body(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            below = frozen(mid) < tgt
            return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

        lo, hi = jax.lax.fori_loop(0, _BISECTION_STEPS, body,
                                   (jnp.full_like(tgt, lo0), jnp.full_like(tgt, hi0)))
        t0 = 0.5 * (lo + hi)
        return t0 - (self(t0) - target) / self.derivative(t0)


jax.tree_util.register_dataclass(_Pchip, data_fields=["x", "y", "d"], meta_fields=[])


def _linear_integral(xp: Array, fp: Array, t: Array) -> Array:
    """``integral(xp[0] -> t)`` of the linear interpolant, flat beyond the ends."""
    seg = jnp.diff(xp) * (fp[:-1] + fp[1:]) / 2.0
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    tc = jnp.clip(t, xp[0], xp[-1])
    k = jnp.clip(jnp.searchsorted(xp, tc, side="right") - 1, 0, xp.shape[0] - 2)
    s = tc - xp[k]
    slope = (fp[k + 1] - fp[k]) / (xp[k + 1] - xp[k])
    inside = cumulative[k] + fp[k] * s + 0.5 * slope * s**2
    return inside + fp[0] * jnp.minimum(t - xp[0], 0.0) + fp[-1] * jnp.maximum(t - xp[-1], 0.0)


# =============================================================================
# The characterisation
# =============================================================================


@dataclass(frozen=True)
class Characterization:
    """A crude as light ends plus pseudo-components.

    Pseudo-component arrays (``Tb`` ... ``cp_ig_coeffs``) have one entry per
    cut. Whole-crude arrays (``*_fraction``, ``component_MW``,
    ``component_SG``) follow :attr:`names`: the light ends first, in the
    order the assay listed them, then the cuts from lightest to heaviest.

    Every array is a ``jax`` array and differentiable with respect to the
    assay data. Registered as a pytree, so it can be returned from a
    ``jax.jit``-ed or differentiated function.

    Attributes:
        pseudo_names: Species names of the cuts (``pc01``, ``pc02``, ...).
        light_names: Species names of the light ends.
        method: The critical-property correlation used.
        cut_edges: Boundaries of the cuts (K); ``n_cuts + 1`` entries.
        Tb: Normal boiling point of each cut (K): the average of the TBP curve
            over the cut, on the assay's basis.
        SG: Specific gravity of each cut.
        MW, Tc, Pc, omega: Molecular weight (g/mol), critical temperature (K)
            and pressure (Pa), and the acentric factor (for an EOS).
        omega_vp: The acentric factor that puts the Lee-Kesler vapour
            pressure at exactly one atmosphere at ``Tb``. Equal to ``omega``
            below ``Tb/Tc = 0.8``; above it ``omega`` is Kesler-Lee's, which
            does not, and a pseudo-component whose vapour pressure missed its
            own boiling point would be a contradiction.
        Kw: Watson characterisation factor of each cut.
        hvap_nb: Heat of vaporisation at ``Tb`` (J/mol), Riedel.
        cp_liquid_coeffs, cp_ig_coeffs: ``(n_cuts, 4)`` cubic Cp coefficients
            (J/mol/K, T in K), liquid (Kesler-Lee) and ideal-gas
            (Watson-Nelson).
        component_MW, component_SG: MW and liquid SG of every component.
        volume_fraction, mass_fraction, mole_fraction: Composition of the
            whole crude.
    """

    pseudo_names: tuple[str, ...]
    light_names: tuple[str, ...]
    method: str
    cut_edges: Array
    Tb: Array
    SG: Array
    MW: Array
    Tc: Array
    Pc: Array
    omega: Array
    omega_vp: Array
    Kw: Array
    hvap_nb: Array
    cp_liquid_coeffs: Array
    cp_ig_coeffs: Array
    component_MW: Array
    component_SG: Array
    volume_fraction: Array
    mass_fraction: Array
    mole_fraction: Array

    @property
    def names(self) -> tuple[str, ...]:
        """Every component: light ends, then cuts."""
        return self.light_names + self.pseudo_names

    @property
    def n_cuts(self) -> int:
        return len(self.pseudo_names)

    @property
    def bulk_sg(self) -> Array:
        """Specific gravity of the whole crude (ideal mixing by volume)."""
        return jnp.sum(self.volume_fraction * self.component_SG)

    @property
    def bulk_api(self) -> Array:
        return corr.api_from_sg(self.bulk_sg)

    @property
    def bulk_MW(self) -> Array:
        """Number-average molecular weight of the whole crude (g/mol)."""
        return 1.0 / jnp.sum(self.mass_fraction / self.component_MW)

    def vapor_pressure(self, T: Array | float) -> Array:
        """Lee-Kesler vapour pressure of every cut at ``T`` (Pa)."""
        return corr.vapor_pressure(jnp.asarray(T, dtype=float), self.Tc, self.Pc, self.omega_vp)

    def flows(self, total: Array | float, basis: Literal["mass", "mole"] = "mass") -> dict[str, Array]:
        """Molar flows (mol/s) of every component for a crude feed.

        Args:
            total: The feed rate: kg/s for ``basis="mass"``, mol/s for
                ``basis="mole"``.
            basis: What ``total`` measures.
        """
        if basis == "mass":
            moles = jnp.asarray(total, dtype=float) * 1000.0 / self.bulk_MW
        elif basis == "mole":
            moles = jnp.asarray(total, dtype=float)
        else:
            raise ValueError(f"basis must be 'mass' or 'mole', not {basis!r}")
        return {n: moles * self.mole_fraction[i] for i, n in enumerate(self.names)}

    def stream(self, total: Array | float, T: Array | float, P: Array | float,
               basis: Literal["mass", "mole"] = "mass"):
        """A difflow feed stream of this crude (see :meth:`flows`)."""
        from difflow.streams import make_stream

        return make_stream(self.flows(total, basis), T, P)

    # ------------------------------------------------------------------
    # Concrete export to difflow's thermo classes
    # ------------------------------------------------------------------

    def species_data(
        self,
        cp: Literal["liquid", "ideal_gas"] = "liquid",
        antoine_window: tuple[float, float] = DEFAULT_ANTOINE_WINDOW,
    ) -> dict[str, SpeciesData]:
        """``SpeciesData`` for every component, for :class:`~difflow.thermo.IdealThermo`.

        Needs concrete values: call it outside ``jit``/``grad``.

        Args:
            cp: Which heat capacity goes in ``Cp_coeffs``. difflow's two
                thermo packages read that field differently --
                ``IdealThermo`` takes it as the *liquid* Cp and adds Watson's
                heat of vaporisation for a vapour, ``CubicThermo`` takes it as
                the *ideal-gas* Cp and adds the EOS departure for either phase.
                :meth:`thermo` picks the right one. Light ends always come from
                difflow's database as they are.
            antoine_window: Temperature range (K) the Antoine equation is
                fitted to the Lee-Kesler vapour pressure over (see
                :func:`fit_antoine` for how each cut narrows or extends it).
                Recorded as each cut's ``T_antoine_min``/``T_antoine_max``.
        """
        if cp not in ("liquid", "ideal_gas"):
            raise ValueError(f"cp must be 'liquid' or 'ideal_gas', not {cp!r}")
        out = {name: get_species_data(name) for name in self.light_names}
        arr = {k: np.asarray(getattr(self, k)) for k in (
            "Tb", "MW", "Tc", "Pc", "omega_vp", "hvap_nb", "cp_liquid_coeffs", "cp_ig_coeffs")}
        cp_key = "cp_liquid_coeffs" if cp == "liquid" else "cp_ig_coeffs"
        for i, name in enumerate(self.pseudo_names):
            Tb, Tc, Pc, w = arr["Tb"][i], arr["Tc"][i], arr["Pc"][i], arr["omega_vp"][i]
            (A, B, C), (T_lo, T_hi) = fit_antoine(Tc, Pc, w, antoine_window, Tb=Tb)
            out[name] = SpeciesData(
                name=name,
                MW=float(arr["MW"][i]),
                Cp_coeffs=tuple(float(c) for c in arr[cp_key][i]),
                Hvap_coeffs=_watson(float(arr["hvap_nb"][i]), float(Tc), float(Tb)),
                antoine_coeffs=(A, B, C),
                Hf=0.0,
                Cp_vapor_coeffs=tuple(float(c) for c in arr["cp_ig_coeffs"][i]),
                T_antoine_min=T_lo,
                T_antoine_max=T_hi,
            )
        return out

    def critical_properties(self) -> dict[str, CriticalProperties]:
        """``CriticalProperties`` of every component, for a cubic EOS.

        Needs concrete values: call it outside ``jit``/``grad``.
        """
        out = {name: get_critical_props(name) for name in self.light_names}
        for i, name in enumerate(self.pseudo_names):
            out[name] = CriticalProperties(
                name=name, Tc=float(self.Tc[i]), Pc=float(self.Pc[i]),
                omega=float(self.omega[i]), MW=float(self.MW[i]),
            )
        return out

    def thermo(self, eos: Literal["ideal", "pr", "srk"] = "ideal",
               antoine_window: tuple[float, float] = DEFAULT_ANTOINE_WINDOW):
        """A difflow thermo package for this crude.

        Args:
            eos: ``"ideal"`` for :class:`~difflow.thermo.IdealThermo`
                (Raoult K-values from the fitted vapour pressures), ``"pr"``
                or ``"srk"`` for :class:`~difflow.thermo.CubicThermo` over
                Peng-Robinson or SRK.
            antoine_window: See :meth:`species_data`.
        """
        from difflow.eos import PengRobinson, SRK
        from difflow.thermo import CubicThermo, IdealThermo

        if eos == "ideal":
            return IdealThermo(self.species_data("liquid", antoine_window))
        if eos not in ("pr", "srk"):
            raise ValueError(f"eos must be 'ideal', 'pr' or 'srk', not {eos!r}")
        cls = PengRobinson if eos == "pr" else SRK
        return CubicThermo(IdealThermo(self.species_data("ideal_gas", antoine_window)),
                           cls(self.critical_properties()))

    def table(self) -> str:
        """The cuts as a fixed-width text table."""
        rows = [f"{'name':<10} {'NBP C':>7} {'SG':>6} {'MW':>7} {'Tc K':>7} "
                f"{'Pc bar':>7} {'omega':>6} {'Kw':>6} {'vol %':>6} {'wt %':>6}"]
        k = len(self.light_names)
        for i, name in enumerate(self.light_names):
            rows.append(f"{name:<10} {'':>7} {float(self.component_SG[i]):6.4f} "
                        f"{float(self.component_MW[i]):7.2f} {'':>7} {'':>7} {'':>6} {'':>6} "
                        f"{100 * float(self.volume_fraction[i]):6.2f} "
                        f"{100 * float(self.mass_fraction[i]):6.2f}")
        for i, name in enumerate(self.pseudo_names):
            rows.append(
                f"{name:<10} {float(self.Tb[i]) - 273.15:7.1f} {float(self.SG[i]):6.4f} "
                f"{float(self.MW[i]):7.1f} {float(self.Tc[i]):7.1f} {float(self.Pc[i]) / 1e5:7.2f} "
                f"{float(self.omega[i]):6.3f} {float(self.Kw[i]):6.2f} "
                f"{100 * float(self.volume_fraction[k + i]):6.2f} "
                f"{100 * float(self.mass_fraction[k + i]):6.2f}")
        return "\n".join(rows)


jax.tree_util.register_dataclass(
    Characterization,
    data_fields=["cut_edges", "Tb", "SG", "MW", "Tc", "Pc", "omega", "omega_vp", "Kw",
                 "hvap_nb", "cp_liquid_coeffs", "cp_ig_coeffs", "component_MW",
                 "component_SG", "volume_fraction", "mass_fraction", "mole_fraction"],
    meta_fields=["pseudo_names", "light_names", "method"],
)


def _watson(hvap_nb: float, Tc: float, Tb: float, n: float = 0.38) -> tuple[float, float, float]:
    """``SpeciesData.Hvap_coeffs`` ``(A, n, Tc)`` through ``hvap_nb`` at ``Tb``.

    ``Hvap = A (1 - T/Tc)^n``, so ``A`` is the value at 0 K, not at ``Tb``.
    """
    return (hvap_nb / (1.0 - Tb / Tc) ** n, n, Tc)


def default_cut_points(assay: Assay, widths=DEFAULT_CUT_WIDTHS) -> tuple[float, ...]:
    """Interior cut boundaries (K) at round Celsius temperatures.

    Boundaries fall on multiples of the local width (20 C to 400 C, 40 C to
    600 C, 100 C beyond), between the start of the pseudo-component range
    and the final boiling point. A boundary closer than half a width to
    either end is dropped rather than leave a sliver of a cut.

    Needs a concrete assay: the number of cuts is the shape of every array
    that follows, so it cannot depend on a traced value. Under ``jax.grad``
    pass ``cut_points`` to :func:`characterize` explicitly.
    """
    if _traced((assay.tbp_percent, assay.tbp_T, dict(assay.light_ends))):
        raise ValueError(
            "the default cut points need concrete assay data: the number of "
            "cuts fixes the shape of everything downstream. Pass cut_points= "
            "explicitly when differentiating with respect to the assay "
            "(default_cut_points(concrete_assay) gives the usual ones)."
        )
    curve = _Pchip.fit(assay.tbp_T, np.asarray(assay.tbp_percent) / 100.0)
    x_le = sum(float(v) for v in assay.light_ends.values()) / 100.0
    T_start = float(curve.invert(jnp.asarray(x_le))) if x_le > 0 else float(np.asarray(assay.tbp_T)[0])
    T_end = float(np.asarray(assay.tbp_T)[-1])

    points = []
    T_C = 0.0
    while True:
        limit, width = next((lim, w) for lim, w in widths if T_C + 273.15 < lim)
        T_C = (np.floor(T_C / width) + 1) * width
        T = T_C + 273.15
        if T >= T_end - width / 2:
            break
        if T > T_start + width / 2:
            points.append(T)
    return tuple(points)


def characterize(
    assay: Assay,
    cut_points: Sequence[float] | None = None,
    method: str = "twu",
    prefix: str = "pc",
) -> Characterization:
    """Cut an assay into pseudo-components and estimate their properties.

    Each cut takes the average of the TBP curve over its span as its boiling
    point (on the assay's basis) and the average of the gravity curve -- or,
    without one, the gravity that a crude-wide Watson K gives at that boiling
    point, with K chosen so the whole crude recombines to its bulk gravity.

    Differentiable with respect to the assay's numbers (TBP temperatures and
    percentages, gravity, light-end fractions) when ``cut_points`` is given.

    Args:
        assay: The data.
        cut_points: Interior cut boundaries (K). The outer boundaries are
            where the curve reaches the light-ends total, and the final
            boiling point. Default: :func:`default_cut_points`.
        method: Critical-property correlation, one of
            :data:`~difflow_refinery.correlations.CRITICAL_METHODS`.
        prefix: Pseudo-component names are ``f"{prefix}{i:02d}"``, from 1.

    Returns:
        A :class:`Characterization`.
    """
    if cut_points is None:
        cut_points = default_cut_points(assay)
    points = jnp.asarray(cut_points, dtype=float).reshape(-1)

    curve = _Pchip.fit(assay.tbp_T, jnp.asarray(assay.tbp_percent, dtype=float) / 100.0)
    light_names = tuple(assay.light_ends)
    le_frac = jnp.asarray([assay.light_ends[n] for n in light_names], dtype=float).reshape(-1) / 100.0
    x_le = jnp.sum(le_frac)
    T_start = jnp.where(x_le > 0, curve.invert(x_le), curve.x[0])

    if not _traced((points, T_start, curve.x)):
        pts = np.asarray(points)
        if np.any(np.diff(pts) <= 0):
            raise ValueError("cut_points must be strictly increasing")
        if pts.size and (pts[0] <= float(T_start) or pts[-1] >= float(curve.x[-1])):
            raise ValueError(
                f"cut_points must lie strictly between {float(T_start):.1f} K (where the "
                f"pseudo-components start) and the final boiling point {float(curve.x[-1]):.1f} K"
            )

    edges = jnp.concatenate([T_start[None], points, curve.x[-1:]])
    x_edges = curve(edges).at[0].set(x_le).at[-1].set(1.0)
    frac = jnp.diff(x_edges)
    integral = curve.integral(edges)
    # Mean of T over the cut, by parts: int T dx = [T x] - int x dT.
    Tb = (edges[1:] * x_edges[1:] - edges[:-1] * x_edges[:-1] - jnp.diff(integral)) / frac

    le_sg = jnp.asarray([LIGHT_END_SG[n] for n in light_names], dtype=float).reshape(-1)
    le_mw = jnp.asarray([get_species_data(n).MW for n in light_names], dtype=float).reshape(-1)

    if assay.sg_curve is not None:
        mid, sgs = (jnp.asarray(a, dtype=float) / s for a, s in zip(assay.sg_curve, (100.0, 1.0)))
        SG = jnp.diff(_linear_integral(mid, sgs, x_edges)) / frac
    else:
        bulk = assay.bulk_sg
        cube = (1.8 * Tb) ** (1.0 / 3.0)
        if assay.basis == "volume":
            # sum(v SG) = SG_bulk, SG_i = cube_i / Kw
            Kw = jnp.sum(frac * cube) / (bulk - jnp.sum(le_frac * le_sg))
        else:
            # sum(w / SG) = 1 / SG_bulk (volumes add)
            Kw = (1.0 / bulk - jnp.sum(le_frac / le_sg)) / jnp.sum(frac / cube)
        SG = cube / Kw

    MW, Tc, Pc = corr.critical_properties(Tb, SG, method)
    omega = corr.acentric_factor(Tb, Tc, Pc, SG)
    Tbr = Tb / Tc
    omega_vp = (-jnp.log(Pc / corr.P_ATM) - corr._lk_f0(Tbr)) / corr._lk_f1(Tbr)

    comp_sg = jnp.concatenate([le_sg, SG])
    comp_mw = jnp.concatenate([le_mw, MW])
    basis_frac = jnp.concatenate([le_frac, frac])
    if assay.basis == "volume":
        vol = basis_frac
        mass = vol * comp_sg / jnp.sum(vol * comp_sg)
    else:
        mass = basis_frac
        vol = (mass / comp_sg) / jnp.sum(mass / comp_sg)
    mole = (mass / comp_mw) / jnp.sum(mass / comp_mw)

    n_cuts = int(edges.shape[0]) - 1
    return Characterization(
        pseudo_names=tuple(f"{prefix}{i + 1:02d}" for i in range(n_cuts)),
        light_names=light_names,
        method=method,
        cut_edges=edges,
        Tb=Tb,
        SG=SG,
        MW=MW,
        Tc=Tc,
        Pc=Pc,
        omega=omega,
        omega_vp=omega_vp,
        Kw=corr.watson_k(Tb, SG),
        hvap_nb=corr.hvap_at_tb(Tb, Tc, Pc),
        cp_liquid_coeffs=corr.cp_liquid_coeffs(Tb, SG, MW),
        cp_ig_coeffs=corr.cp_ideal_gas_coeffs(Tb, SG, MW),
        component_MW=comp_mw,
        component_SG=comp_sg,
        volume_fraction=vol,
        mass_fraction=mass,
        mole_fraction=mole,
    )


def fit_antoine(
    Tc: float, Pc: float, omega: float,
    window: tuple[float, float] = DEFAULT_ANTOINE_WINDOW,
    Tb: float | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Antoine coefficients fitted to the Lee-Kesler vapour pressure.

    difflow's ``SpeciesData`` carries vapour pressure as Antoine
    ``log10(P/Pa) = A - B/(T + C)``, so a pseudo-component's Lee-Kesler curve
    is fitted to that form by least squares in ``log10 P`` over ``window``
    -- extended up to ``Tb`` if given, and starting no lower than where the
    vapour pressure reaches 10 Pa -- intersected with ``0.35 < T/Tc < 0.95``.
    Concrete (numpy/scipy), not traceable.

    Returns:
        ``((A, B, C), (T_min, T_max))``: the coefficients and the range they
        were fitted over.
    """
    from scipy.optimize import least_squares

    def log10_psat(T):
        Tr = np.asarray(T) / Tc
        f0 = 5.92714 - 6.09648 / Tr - 1.28862 * np.log(Tr) + 0.169347 * Tr**6
        f1 = 15.2518 - 15.6875 / Tr - 13.4721 * np.log(Tr) + 0.43577 * Tr**6
        return np.log10(Pc) + (f0 + omega * f1) / np.log(10.0)

    # Start where the vapour pressure becomes worth having: below
    # _ANTOINE_P_MIN a component's K-value is negligible, and fitting the
    # three-parameter form across many decades of it costs accuracy where the
    # component does distil. End at the window or at Tb, whichever is higher,
    # so a heavy cut's fit always reaches its own boiling point.
    T_lo = max(window[0], _ANTOINE_TR[0] * Tc)
    T_hi = min(max(window[1], Tb if Tb is not None else window[1]), _ANTOINE_TR[1] * Tc)
    if log10_psat(T_lo) < np.log10(_ANTOINE_P_MIN) < log10_psat(T_hi):
        from scipy.optimize import brentq

        T_lo = brentq(lambda t: log10_psat(t) - np.log10(_ANTOINE_P_MIN), T_lo, T_hi)
    if T_hi - T_lo < 20.0:
        # A cut whose critical point sits at the edge of the window: fit over
        # the reduced-temperature range alone rather than over a sliver.
        T_lo, T_hi = _ANTOINE_TR[0] * Tc, _ANTOINE_TR[1] * Tc
    T = np.linspace(T_lo, T_hi, _ANTOINE_POINTS)
    target = log10_psat(T)

    # Clausius-Clapeyron (C = 0) through the end points as the start.
    B0 = (target[0] - target[-1]) / (1.0 / T[-1] - 1.0 / T[0])
    A0 = target[0] + B0 / T[0]

    def resid(p):
        A, B, C = p
        return A - B / (T + C) - target

    sol = least_squares(resid, [A0, B0, 0.0], bounds=([-np.inf, 0.0, -0.9 * T_lo], np.inf),
                        x_scale=[1.0, B0, 10.0])
    return tuple(float(v) for v in sol.x), (float(T_lo), float(T_hi))
