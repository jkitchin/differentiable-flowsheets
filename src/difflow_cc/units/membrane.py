"""Membrane separation units for CO2 capture.

This module provides gas separation membrane models based on the
solution-diffusion mechanism. Both single-stage and multi-stage
configurations are supported.

The models are suitable for:
- Post-combustion CO2 capture (flue gas)
- Natural gas sweetening (CO2/CH4)
- Biogas upgrading
- Pre-combustion hydrogen purification

References:
    Baker RW (2012). Membrane Technology and Applications, 3rd ed.
        Wiley. Chapters 8-9.
    Robeson LM (2008). The upper bound revisited.
        J Membr Sci 320:390-400.
    Merkel TC et al. (2010). Power plant post-combustion carbon
        dioxide capture: An opportunity for membranes.
        J Membr Sci 359:126-139.
"""

__all__ = [
    "MembraneParams",
    "MembraneSeparator",
    "MultistageMembrane",
]

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow.params_mixin import ParamsMixin
from difflow.numerics import safe_divide
from difflow.flowsheet import _concrete
from difflow_cc.database import get_membrane, Membrane


# Unit conversions
# 1 Barrer = 10^-10 cm³(STP)·cm / (cm²·s·cmHg)
# 1 GPU = 10^-6 cm³(STP) / (cm²·s·cmHg) = Barrer / thickness(cm)
# Converting to SI: 1 GPU = 3.35e-10 mol/(m²·s·Pa)

GPU_TO_SI = 3.35e-10  # mol/(m²·s·Pa)
BARRER_TO_SI = 3.35e-16  # mol·m/(m²·s·Pa)

# Gas constant
R = 8.314  # J/(mol*K)
#: Units for every :class:`MembraneParams` field, shared by the single-stage
#: separator and the multistage cascade.
_MEMBRANE_UNITS = {
    "area": "m^2",
    "thickness": "um",
    "pressure_ratio": "-",
    "T_operation": "K",
    "feed_pressure": "Pa",
    "permeate_pressure": "Pa",
    "stage_cut_target": "-",
}


# =============================================================================
# Membrane Parameters
# =============================================================================

@dataclass(repr=False)
class MembraneParams(ParamsMixin):
    """Parameters for membrane separator.

    Attributes:
        membrane_type: Membrane material name from database
        area: Membrane area (m²)
        thickness: Membrane thickness (μm), None uses database default
        pressure_ratio: Feed/permeate pressure ratio
        T_operation: Operating temperature (K)
        feed_pressure: Feed-side pressure (Pa); None (default) uses the
            feed stream pressure
        permeate_pressure: Permeate pressure (Pa), calculated if None

    Notes:
        For polymeric membranes, typical thicknesses are 0.1-1 μm
        for thin-film composites on supports.

        Pressure ratio typically 5-20 for gas separation.
        Higher ratios improve recovery but increase compression cost.

        Temperature affects permeability through Arrhenius relation:
            P = P0 * exp(-Ep/(R*T))
    """
    membrane_type: str
    area: float | Array = 1000.0  # m²
    thickness: float | Array | None = None  # μm (uses default if None)
    pressure_ratio: float | Array = 10.0
    T_operation: float | Array = 298.15  # K
    feed_pressure: float | Array | None = None  # Pa (None: feed stream P)
    permeate_pressure: float | Array | None = None  # Pa

    # Stage cut control
    stage_cut_target: float | Array | None = None  # If set, adjusts area

    def __post_init__(self):
        """Reject pressure ratios and stage cuts that cannot operate.

        Only concrete values are checked (traced values pass through).
        """
        try:
            ratio = float(self.pressure_ratio)
        except Exception:
            ratio = None
        if ratio is not None and ratio <= 1.0:
            # Audit C5: ratios <= 1 were accepted and still "separated".
            raise ValueError(
                f"pressure_ratio must exceed 1 (feed/permeate), got {ratio}"
            )
        if self.stage_cut_target is not None:
            try:
                cut = float(self.stage_cut_target)
            except Exception:
                cut = None
            if cut is not None and not 0.0 < cut < 1.0:
                raise ValueError(
                    f"stage_cut_target must be in (0, 1), got {cut}"
                )


# =============================================================================
# Complete-mixing solver
# =============================================================================

def _permeate_composition(t, F_i, Q, area, P_h, P_l):
    """Permeate mole fractions of a complete-mixing stage.

    With retentate fraction ``t = D/F`` (so stage cut ``1 - t``), the
    species balance ``(1-t) F y_i = A Q_i (x_i P_h - y_i P_l)`` with
    ``x_i = (F_i - (1-t) F y_i) / (t F)`` solves to

        y_i = A Q_i z_i P_h / (F t (1-t) + A Q_i ((1-t) P_h + t P_l)),

    where ``z_i`` is the feed mole fraction. The physical ``t`` is the one
    where the ``y_i`` sum to one.

    Args:
        t: Retentate fraction in [0, 1].
        F_i: Feed flows (n,).
        Q: Permeances (n,), mol/(m2 s Pa).
        area: Membrane area (m2).
        P_h: Feed-side pressure (Pa).
        P_l: Permeate-side pressure (Pa).

    Returns:
        Permeate mole fractions (n,).
    """
    F = jnp.sum(F_i)
    z = F_i / F
    aq = area * Q
    denom = F * t * (1.0 - t) + aq * ((1.0 - t) * P_h + t * P_l)
    return aq * z * P_h / jnp.maximum(denom, 1e-300)


def _bisect_increasing(fn, lo, hi, n_iter=100):
    """Root of a function negative left / positive right of it, by bisection.

    Args:
        fn: Scalar function.
        lo, hi: Bracket.
        n_iter: Number of halvings.

    Returns:
        Converged abscissa (no gradient; see ``_implicit_step``).
    """
    def body(_, b):
        a, c = b
        m = 0.5 * (a + c)
        neg = fn(m) < 0.0
        return jnp.where(neg, m, a), jnp.where(neg, c, m)

    a, c = jax.lax.fori_loop(0, n_iter, body,
                             (jnp.asarray(lo, float), jnp.asarray(hi, float)))
    return jax.lax.stop_gradient(0.5 * (a + c))


def _implicit_step(fn, u0):
    """Value-neutral Newton step from a stop-gradient root ``u0``.

    Gives the exact implicit-function first derivative of the root with
    respect to whatever ``fn`` closes over.
    """
    d = jax.lax.stop_gradient(jax.grad(fn)(u0))
    d = jnp.where(jnp.abs(d) > 1e-300, d, 1.0)
    r = fn(u0)
    return u0 - (r - jax.lax.stop_gradient(r)) / d


def _solve_retentate_fraction(F_i, Q, area, P_h, P_l):
    """Retentate fraction ``t`` of a complete-mixing stage of given area.

    ``g(t) = sum_i y_i(t) - 1`` is convex with ``g(0) = 0`` and
    ``g(1) = (P_h/P_l) * (permeable feed fraction) - 1``, so it has at most
    one root in (0, 1): ``g < 0`` left of it, ``> 0`` right. If ``g >= 0``
    throughout, the area is large enough to permeate everything (t -> 0);
    if ``g < 0`` throughout, nothing can permeate (t = 1).
    """
    def g(t):
        return jnp.sum(_permeate_composition(t, F_i, Q, area, P_h, P_l)) - 1.0

    t0 = _bisect_increasing(g, 0.0, 1.0)
    return jnp.clip(_implicit_step(g, t0), 0.0, 1.0)


def _solve_area_for_cut(F_i, Q, P_h, P_l, t):
    """Area at which a complete-mixing stage has retentate fraction ``t``.

    At fixed ``t`` in (0, 1), ``sum_i y_i`` rises monotonically with area
    from 0 to ``1 / ((1-t) + t P_l/P_h) * (permeable fraction)``, so the
    root in ``s = ln(A / A_ref)`` is bracketed whenever the cut is reachable.
    """
    F = jnp.sum(F_i)
    A_ref = F / (jnp.max(Q) * P_h)  # area of ~one transfer unit

    def h(s):
        A = A_ref * jnp.exp(s)
        return jnp.sum(_permeate_composition(t, F_i, Q, A, P_h, P_l)) - 1.0

    s0 = _bisect_increasing(h, -60.0, 60.0)
    return A_ref * jnp.exp(_implicit_step(h, s0))
# =============================================================================
# Membrane Separator
# =============================================================================

class MembraneSeparator:
    """Single-stage membrane gas separator.

    Uses solution-diffusion model for gas transport:
        J_i = (P_i / δ) * (p_i,feed - p_i,permeate)

    where J_i is molar flux, P_i is permeability, δ is thickness,
    and p_i are partial pressures.

    Example:
        >>> params = MembraneParams(
        ...     membrane_type='Matrimid',
        ...     area=1000,  # m²
        ...     pressure_ratio=10,
        ... )
        >>> membrane = MembraneSeparator(params)
        >>> retentate, permeate, info = membrane(feed)

    The model outputs:
    - Retentate: Depleted stream (remaining feed side)
    - Permeate: Enriched stream (passed through membrane)
    - Stage cut: Fraction of feed that permeates

    For CO2 capture:
    - Permeate is CO2-enriched
    - Retentate is treated gas

    This model assumes perfect mixing on both sides (simplification).
    For counter-current or cross-flow, see extensibility hooks.

    References:
        Baker RW (2012). Membrane Technology and Applications.
        Wijmans JG, Baker RW (1995). The solution-diffusion model:
            a review. J Membr Sci 107:1-21.
    """

    symbol = "Membrane"
    equations = [
        r"J_i = \frac{P_i}{\delta}\,(p_{i,\mathrm{feed}} - p_{i,\mathrm{permeate}})\qquad \text{(solution-diffusion)}",
        r"\alpha_{i/j} = \frac{P_i}{P_j}\qquad \text{(selectivity)}",
        r"\theta = \frac{\dot{n}_\mathrm{permeate}}{\dot{n}_\mathrm{feed}}\qquad \text{(stage cut)}",
    ]
    assumptions = [
        "Solution-diffusion transport with constant permeabilities.",
        "Perfect-mixing (CSTR) on each side unless otherwise specified.",
        "Isothermal, isobaric operation (feed-side pressure drop neglected).",
    ]
    references = [
        "Baker, R.W. Membrane Technology and Applications, 3e, Wiley, 2012.",
        "Wijmans, J.G., Baker, R.W. J. Membr. Sci., 107, 1 (1995).",
    ]
    parameter_symbols = {
        "area": "A",
        "thickness": r"\delta",
        "pressure_ratio": r"P_F/P_P",
    }
    parameter_units = _MEMBRANE_UNITS
    numerical_method = "Per-component flux integration with closed-form perfect-mixing solution."

    def __init__(self, params: MembraneParams):
        """Initialize membrane separator.

        Args:
            params: MembraneParams dataclass
        """
        self.params = params
        self._membrane_data = get_membrane(params.membrane_type)

        # Get thickness (use default if not specified)
        if params.thickness is not None:
            self.thickness = params.thickness
        else:
            self.thickness = self._membrane_data.typical_thickness

    def _permeance(self, species: str, T: Array) -> Array:
        """Calculate permeance at temperature T.

        Permeance = Permeability / thickness

        Args:
            species: Gas species name
            T: Temperature (K)

        Returns:
            Permeance in mol/(m²·s·Pa)
        """
        mem = self._membrane_data
        T = jnp.asarray(T)
        T_ref = 298.15

        # Get base permeability
        P_base = mem.permeability.get(species, 0.0)  # Barrer

        # Temperature correction
        if species in mem.activation_energy:
            Ep = mem.activation_energy[species]  # J/mol
            T_factor = jnp.exp(-Ep / R * (1 / T - 1 / T_ref))
        else:
            T_factor = 1.0

        P_T = P_base * T_factor  # Barrer

        # Convert to permeance: mol/(m²·s·Pa)
        # Barrer → SI permeability, then divide by thickness
        thickness_m = jnp.asarray(self.thickness) * 1e-6  # μm to m
        permeance = P_T * BARRER_TO_SI / thickness_m

        return permeance

    def __call__(
        self,
        feed: Stream,
        P_feed: Array | float | None = None,
        P_permeate: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform membrane separation.

        Args:
            feed: Feed gas stream
            P_feed: Feed pressure (Pa), overrides params if provided
            P_permeate: Permeate pressure (Pa), overrides params if provided

        Returns:
            retentate: Retentate stream (feed side, depleted in CO2)
            permeate: Permeate stream (CO2 enriched)
            info: Dict with operation details:
                - stage_cut: Fraction permeated
                - CO2_recovery: CO2 recovery in permeate
                - CO2_purity: CO2 purity in permeate
                - area_used: Membrane area
        """
        p = self.params
        T = jnp.asarray(p.T_operation)

        # Pressures. Audit C5: the feed stream's pressure used to be ignored
        # (a 1 bar feed separated like a 10 bar one); it is now the default.
        if P_feed is not None:
            P_feed = jnp.asarray(P_feed)
        elif p.feed_pressure is not None:
            P_feed = jnp.asarray(p.feed_pressure)
        else:
            P_feed = jnp.asarray(feed["P"])

        if P_permeate is not None:
            P_permeate = jnp.asarray(P_permeate)
        elif p.permeate_pressure is not None:
            P_permeate = jnp.asarray(p.permeate_pressure)
        else:
            P_permeate = P_feed / jnp.asarray(p.pressure_ratio)

        P_h_c, P_l_c = _concrete(P_feed), _concrete(P_permeate)
        if P_h_c is not None and P_l_c is not None and P_l_c >= P_h_c:
            raise ValueError(
                f"Permeate pressure ({P_l_c} Pa) must be below the feed "
                f"pressure ({P_h_c} Pa): with no transmembrane pressure "
                "ratio above 1 nothing permeates."
            )

        feed_flows = get_flows(feed)
        species = list(feed_flows)
        F_i = jnp.stack([jnp.asarray(feed_flows[s], dtype=float) for s in species])
        F_total = jnp.sum(F_i)
        permeances = {s: self._permeance(s, T) for s in species}
        Q = jnp.stack([jnp.asarray(permeances[s], dtype=float) for s in species])

        # Audit C5: the old model fixed the permeate CO2 fraction from the
        # feed-composition selectivity formula, split the rest over the other
        # species by feed fraction, and capped each species at 99 % of its
        # feed. That broke the solution-diffusion model it states (Matrimid:
        # 99 % CO2 recovery at 0.72 purity with a retentate CO2 partial
        # pressure of 1.16 kPa against 72 kPa in the permeate). It is
        # replaced by the complete-mixing (both sides well mixed) model
        # solved exactly: for every species
        #     F_perm,i = A Q_i (x_i P_feed - y_i P_permeate),
        # with x the retentate and y the permeate composition, which keeps
        # every driving force non-negative and every retentate flow >= 0.
        if p.stage_cut_target is not None:
            # Documented behaviour: the area is adjusted to hit the cut.
            t = 1.0 - jnp.asarray(p.stage_cut_target)
            area = _solve_area_for_cut(F_i, Q, P_feed, P_permeate, t)
        else:
            area = jnp.asarray(p.area)
            t = _solve_retentate_fraction(F_i, Q, area, P_feed, P_permeate)

        y = _permeate_composition(t, F_i, Q, area, P_feed, P_permeate)
        F_perm_i = (1.0 - t) * F_total * y
        F_perm_i = jnp.minimum(F_perm_i, F_i)  # round-off only
        F_ret_i = F_i - F_perm_i

        permeate_flows = {s: F_perm_i[k] for k, s in enumerate(species)}
        retentate_flows = {s: F_ret_i[k] for k, s in enumerate(species)}
        F_perm_total = jnp.sum(F_perm_i)
        stage_cut = safe_divide(F_perm_total, F_total)

        F_CO2_feed = feed_flows.get("CO2", jnp.array(0.0))
        F_CO2_perm = permeate_flows.get("CO2", jnp.array(0.0))
        CO2_recovery = safe_divide(F_CO2_perm, F_CO2_feed)
        CO2_purity = safe_divide(F_CO2_perm, F_perm_total)

        retentate = make_stream(retentate_flows, T, P_feed)
        permeate = make_stream(permeate_flows, T, P_permeate)

        info = {
            "stage_cut": stage_cut,
            "CO2_recovery": CO2_recovery,
            "CO2_purity": CO2_purity,
            "permeate_flow": F_perm_total,
            "retentate_flow": F_total - F_perm_total,
            "area_used": area,
            "pressure_ratio": P_feed / P_permeate,
            "P_feed": P_feed,
            "P_permeate": P_permeate,
            "permeances": permeances,
        }

        return retentate, permeate, info

    def required_area(
        self,
        feed: Stream,
        CO2_recovery_target: Array | float,
    ) -> Array:
        """Calculate membrane area for target CO2 recovery.

        Args:
            feed: Feed gas stream
            CO2_recovery_target: Target CO2 recovery (0-1)

        Returns:
            Required membrane area (m²)
        """
        p = self.params
        T = jnp.asarray(p.T_operation)
        if p.feed_pressure is not None:
            P_feed = jnp.asarray(p.feed_pressure)
        else:
            P_feed = jnp.asarray(feed["P"])
        if p.permeate_pressure is not None:
            P_permeate = jnp.asarray(p.permeate_pressure)
        else:
            P_permeate = P_feed / jnp.asarray(p.pressure_ratio)

        recovery = jnp.asarray(CO2_recovery_target)

        feed_flows = get_flows(feed)
        species = list(feed_flows)
        F_i = jnp.stack([jnp.asarray(feed_flows[s], dtype=float) for s in species])
        Q = jnp.stack([jnp.asarray(self._permeance(s, T), dtype=float)
                       for s in species])
        k = species.index("CO2")
        F = jnp.sum(F_i)
        A_ref = F / (jnp.max(Q) * P_feed)

        # Same complete-mixing model as __call__ (audit C5); CO2 recovery
        # rises monotonically with area from 0 to 1, so the root in
        # s = ln(A / A_ref) is bracketed.
        def h(s):
            A = A_ref * jnp.exp(s)
            t = _solve_retentate_fraction(F_i, Q, A, P_feed, P_permeate)
            y = _permeate_composition(t, F_i, Q, A, P_feed, P_permeate)
            return (1.0 - t) * F * y[k] / F_i[k] - recovery

        s0 = _bisect_increasing(h, -60.0, 60.0, n_iter=80)
        return A_ref * jnp.exp(_implicit_step(h, s0))


# =============================================================================
# Multi-stage Membrane
# =============================================================================

class MultistageMembrane:
    """Multi-stage membrane cascade.

    Cascades multiple membrane stages for higher purity or recovery
    than achievable with a single stage.

    Common configurations:
    - Two-stage with permeate recycle (for high purity)
    - Two-stage with retentate recycle (for high recovery)
    - Three-stage for both high purity and recovery

    Example:
        >>> params = MembraneParams(membrane_type='Matrimid', area=500)
        >>> cascade = MultistageMembrane(params, n_stages=2)
        >>> retentate, permeate, info = cascade(feed)

    References:
        Merkel TC et al. (2010). Power plant post-combustion CO2
            capture: An opportunity for membranes.
            J Membr Sci 359:126-139.
    """

    symbol = "Membrane Cascade"
    equations = [
        r"\theta_\mathrm{overall} = 1 - \prod_k (1 - \theta_k)\qquad \text{(series stage cuts)}",
        r"\mathrm{purity} = \frac{\sum_k \dot{n}_{\mathrm{CO}_2,k}^\mathrm{permeate}}{\sum_k \dot{n}_k^\mathrm{permeate}}",
    ]
    assumptions = [
        "Identical membrane properties across stages unless parameters are overridden per stage.",
        "Series or permeate-recycle configuration selected via constructor argument.",
    ]
    references = ["Merkel, T.C. et al. J. Membr. Sci., 359, 126 (2010)."]
    parameter_symbols = {"area": "A", "pressure_ratio": r"P_F/P_P"}
    parameter_units = _MEMBRANE_UNITS
    numerical_method = "Sequential stage evaluation; optional fixed-point recycle convergence."

    def __init__(
        self,
        params: MembraneParams,
        n_stages: int = 2,
        configuration: Literal["series", "permeate_recycle"] = "series",
        recycle_iterations: int = 100,
        stage_params: list[MembraneParams] | None = None,
    ):
        """Initialize multi-stage membrane.

        Args:
            params: Base membrane parameters (area is per stage)
            n_stages: Number of stages. 'permeate_recycle' is a two-stage
                layout and accepts only ``n_stages=2``.
            configuration: 'series' (each stage treats the previous
                retentate) or 'permeate_recycle' (stage 2 enriches the
                stage-1 permeate; its retentate is recycled to stage 1).
            recycle_iterations: Successive-substitution passes used to
                converge the 'permeate_recycle' loop.
            stage_params: Optional per-stage parameters (one per stage),
                e.g. a smaller second stage in 'permeate_recycle': with one
                shared ``params`` that stage sees only the stage-1 permeate
                on the same area and tends to permeate all of it.

        Raises:
            ValueError: unknown configuration, or 'permeate_recycle' with
                ``n_stages != 2`` (audit (e): it silently ran 2 stages).
        """
        if configuration not in ("series", "permeate_recycle"):
            raise ValueError(
                f"configuration must be 'series' or 'permeate_recycle', "
                f"got {configuration!r}"
            )
        if configuration == "permeate_recycle" and n_stages != 2:
            raise ValueError(
                "configuration='permeate_recycle' is a two-stage cascade; "
                f"got n_stages={n_stages}. Use n_stages=2 or 'series'."
            )
        if n_stages < 1:
            raise ValueError(f"n_stages must be >= 1, got {n_stages}")
        self.recycle_iterations = int(recycle_iterations)
        self.params = params
        self.n_stages = n_stages
        self.configuration = configuration
        if stage_params is None:
            stage_params = [params] * n_stages
        elif len(stage_params) != n_stages:
            raise ValueError(
                f"stage_params has {len(stage_params)} entries for "
                f"n_stages={n_stages}"
            )
        self._stages = [MembraneSeparator(sp) for sp in stage_params]

    def __call__(
        self,
        feed: Stream,
    ) -> tuple[Stream, Stream, dict]:
        """Perform multi-stage separation.

        Args:
            feed: Feed gas stream

        Returns:
            retentate: Final retentate (treated gas)
            permeate: Final permeate (CO2 product)
            info: Dict with stage-by-stage results
        """
        if self.configuration == "series":
            return self._series_operation(feed)
        else:
            return self._permeate_recycle_operation(feed)

    def _series_operation(
        self,
        feed: Stream
    ) -> tuple[Stream, Stream, dict]:
        """Series configuration: each stage operates on previous retentate."""
        current_feed = feed
        stage_infos = []
        total_permeate_flows = {}

        for i, stage in enumerate(self._stages):
            retentate, permeate, info = stage(current_feed)
            stage_infos.append(info)

            # Accumulate permeate
            perm_flows = get_flows(permeate)
            for species, flow in perm_flows.items():
                if species in total_permeate_flows:
                    total_permeate_flows[species] = total_permeate_flows[species] + flow
                else:
                    total_permeate_flows[species] = flow

            current_feed = retentate

        # Final streams
        final_retentate = current_feed
        final_permeate = make_stream(
            total_permeate_flows,
            permeate["T"],
            permeate["P"]
        )

        # Overall metrics
        feed_flows = get_flows(feed)
        F_CO2_feed = feed_flows.get("CO2", jnp.array(0.0))
        F_CO2_perm = total_permeate_flows.get("CO2", jnp.array(0.0))
        F_perm_total = sum(total_permeate_flows.values())

        overall_info = {
            "n_stages": self.n_stages,
            "configuration": self.configuration,
            "overall_CO2_recovery": safe_divide(F_CO2_perm, F_CO2_feed),
            "overall_CO2_purity": safe_divide(F_CO2_perm, F_perm_total),
            "stage_info": stage_infos,
        }

        return final_retentate, final_permeate, overall_info

    def _permeate_recycle_operation(
        self,
        feed: Stream,
    ) -> tuple[Stream, Stream, dict]:
        """Two-stage enriching cascade with stage-2 retentate recycle.

        Stage 1 treats feed + recycle; its permeate is recompressed to the
        stage-1 feed pressure and fed to stage 2, whose permeate is the
        product and whose retentate is recycled to the stage-1 inlet (Merkel
        et al. 2010). Audit (e): this used to skip the recycle (it summed
        the two retentates) and always ran two stages while reporting the
        requested count; the recycle is now converged by successive
        substitution over a fixed iteration count (differentiable by
        unrolling) and the residual is reported.
        """
        stage1, stage2 = self._stages
        feed_flows = get_flows(feed)
        species = list(feed_flows)
        F_feed = jnp.stack([jnp.asarray(feed_flows[s], dtype=float) for s in species])
        T_feed = feed["T"]
        P1 = self.params.feed_pressure
        P1 = jnp.asarray(feed["P"] if P1 is None else P1)

        def run(rec):
            mixed = make_stream({s: F_feed[k] + rec[k] for k, s in enumerate(species)},
                                T_feed, P1)
            ret_1, perm_1, info_1 = stage1(mixed, P_feed=P1)
            ret_2, perm_2, info_2 = stage2(perm_1, P_feed=P1)
            f2 = get_flows(ret_2)
            new_rec = jnp.stack([jnp.asarray(f2.get(s, 0.0)) for s in species])
            return new_rec, (ret_1, perm_1, perm_2, info_1, info_2)

        rec = jax.lax.fori_loop(
            0, self.recycle_iterations, lambda _, r: run(r)[0], jnp.zeros_like(F_feed))
        new_rec, (ret_1, perm_1, perm_2, info_1, info_2) = run(rec)
        residual = jnp.max(jnp.abs(new_rec - rec)) / jnp.maximum(jnp.sum(F_feed), 1e-300)

        final_retentate = ret_1
        final_permeate = perm_2

        perm_flows = get_flows(final_permeate)
        F_CO2_feed = feed_flows.get("CO2", jnp.array(0.0))
        F_CO2_perm = perm_flows.get("CO2", jnp.array(0.0))
        F_perm_total = total_flow(final_permeate)

        overall_info = {
            "n_stages": 2,
            "configuration": self.configuration,
            "overall_CO2_recovery": safe_divide(F_CO2_perm, F_CO2_feed),
            "overall_CO2_purity": safe_divide(F_CO2_perm, F_perm_total),
            "recycle_flow": jnp.sum(new_rec),
            "recycle_residual": residual,
            "interstage_compression": {"from_Pa": perm_1["P"], "to_Pa": P1,
                                       "flow": total_flow(perm_1)},
            "stage_info": [info_1, info_2],
        }

        return final_retentate, final_permeate, overall_info
