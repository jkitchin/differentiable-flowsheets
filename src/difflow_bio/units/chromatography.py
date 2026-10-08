"""Chromatography unit operations for protein purification.

This module provides chromatography models for biopharmaceutical purification:
- ProteinAChromatography: Affinity capture for monoclonal antibodies
- IonExchangeChromatography: Charge-based separation (CEX/AEX)
- SizeExclusionChromatography: Size-based polishing

Key equations:
    Langmuir isotherm: q = q_max * C / (K_d + C)
    Linear isotherm: q = K * C
    Yield = (mass_eluted / mass_loaded) * purity_factor
    DBC = dynamic binding capacity (g/L resin)

where:
    q = bound concentration (g/L resin)
    C = mobile phase concentration (g/L)
    K_d = dissociation constant
"""

from typing import Literal
from dataclasses import dataclass, field

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows
from difflow.numerics import safe_divide


# =============================================================================
# Isotherm Models
# =============================================================================

def langmuir_isotherm(
    C: Array,
    q_max: Array,
    K_d: Array,
) -> Array:
    """Langmuir binding isotherm.

    q = q_max * C / (K_d + C)

    Args:
        C: Mobile phase concentration (g/L)
        q_max: Maximum binding capacity (g/L resin)
        K_d: Dissociation constant (g/L)

    Returns:
        Bound concentration (g/L resin)
    """
    C = jnp.asarray(C)
    return q_max * C / (K_d + C)


def linear_isotherm(
    C: Array,
    K: Array,
) -> Array:
    """Linear binding isotherm.

    q = K * C

    Args:
        C: Mobile phase concentration (g/L)
        K: Partition coefficient (L solution / L resin)

    Returns:
        Bound concentration (g/L resin)
    """
    return K * jnp.asarray(C)


def langmuir_freundlich_isotherm(
    C: Array,
    q_max: Array,
    K_d: Array,
    n: Array,
) -> Array:
    """Langmuir-Freundlich (Sips) isotherm.

    q = q_max * (C/K_d)^n / (1 + (C/K_d)^n)

    Args:
        C: Mobile phase concentration (g/L)
        q_max: Maximum binding capacity (g/L resin)
        K_d: Characteristic constant (g/L)
        n: Heterogeneity parameter

    Returns:
        Bound concentration (g/L resin)
    """
    C = jnp.asarray(C)
    ratio = (C / K_d) ** n
    return q_max * ratio / (1.0 + ratio)


# =============================================================================
# Chromatography Parameters
# =============================================================================

@dataclass(repr=False)
class ProteinAParams(ParamsMixin):
    """Parameters for Protein A affinity chromatography.

    Attributes:
        column_volume: Column volume (L)
        q_max: Maximum binding capacity (g mAb / L resin)
        K_d: Dissociation constant (g/L)
        target_species: Name of target species (mAb/product)
        yield_factor: Elution yield (0-1), accounts for incomplete elution
        impurity_clearance: Dict of impurity -> log reduction value (LRV)
        species_order: List of all species
    """
    column_volume: float | Array
    q_max: float | Array = 35.0  # g/L, typical for Protein A
    K_d: float | Array = 0.1  # g/L, high affinity
    target_species: str = "mAb"
    yield_factor: float | Array = 0.95
    impurity_clearance: dict = field(default_factory=lambda: {
        "HCP": 2.0,  # 2 log reduction for host cell proteins
        "DNA": 3.0,  # 3 log reduction for DNA
        "cells": 4.0,  # 4 log reduction for cells
    })
    species_order: list[str] = None
    # Kinetic dynamic binding capacity (#100). When k_ads is set and a load
    # flow rate is supplied to __call__, DBC is computed from residence time
    # via dynamic_binding_capacity() instead of the static q_max fraction, so
    # DBC falls at high flow rate (short residence time).
    k_ads: float | Array | None = None  # Adsorption rate constant (1/min)
    # Elution pool sizing (#102). When n_plates is set, the elution pool
    # volume is estimated from the peak base width w = 4 V_R / sqrt(N)
    # (N = 16 (t_R/w)^2; Fritz & Gjerde, Ion Chromatography, 4e, Wiley, 2009),
    # and product concentration = mass_eluted / pool_volume is reported.
    n_plates: float | Array | None = None  # Theoretical plate count (-)
    elution_cv: float | Array = 2.0  # Elution retention volume (column volumes)


@dataclass(repr=False)
class IEXParams(ParamsMixin):
    """Parameters for ion exchange chromatography.

    Attributes:
        column_volume: Column volume (L); with the binding capacity it
            limits what the column holds (q x column_volume).
        mode: "bind_elute" or "flow_through"
        q_max: Maximum binding capacity (g/L resin). In bind-elute mode
            target beyond the capacity breaks through to waste; in
            flow-through mode impurity binding is capped by it.
        K_d: Dissociation constant (g/L) of the Langmuir isotherm,
            q = q_max C / (K_d + C); used when the call is given the feed
            concentration or volume, otherwise the feed is taken as
            saturating (q = q_max).
        target_species: Target protein species
        selectivity: Dict of species -> binding selectivity (0=no binding, 1=strong)
        yield_factor: Recovery yield
        species_order: List of species
        impurity_clearance: Dict of impurity -> log reduction value (LRV)
            across the step. An impurity listed here reaches the product
            as 10**(-LRV) of what was loaded, in either mode, and its
            selectivity is ignored. Impurities not listed fall back to the
            selectivity rule. Default: none listed.
    """
    column_volume: float | Array
    mode: Literal["bind_elute", "flow_through"] = "bind_elute"
    q_max: float | Array = 50.0
    K_d: float | Array = 0.5
    target_species: str = "mAb"
    selectivity: dict = field(default_factory=dict)
    yield_factor: float | Array = 0.90
    species_order: list[str] = None
    impurity_clearance: dict = field(default_factory=dict)


#: Representative impurity clearances (LRV) for the polishing steps of a
#: platform mAb process, used by the packaged trains. They are not measured
#: values: they are chosen so that a harvest at ~5% HCP and ~3% aggregate
#: by mass, after a Protein A step at its defaults, ends near the reported
#: end-of-process levels: ~10 ppm HCP against a <100 ppm target (Lenhoff &
#: Herman, BioProcess International, 2023) and aggregate below the usual
#: <1% target (Evans, PharmTech, 2015). The AEX HCP value matches one
#: reported flow-through step, 530 to 15 ppm (Liu et al., mAbs 2:480, 2010,
#: doi:10.4161/mabs.2.5.12645). Replace them with your process's data.
TYPICAL_CEX_CLEARANCE = {"HCP": 0.5, "DNA": 1.0, "aggregates": 0.7}
TYPICAL_AEX_CLEARANCE = {"HCP": 1.5, "DNA": 3.0, "aggregates": 0.1}


@dataclass(repr=False)
class SECParams(ParamsMixin):
    """Parameters for size exclusion chromatography.

    Attributes:
        column_volume: Column volume (L)
        void_fraction: Void volume fraction (0.3-0.4 typical)
        target_species: Target species (excluded from pores)
        aggregate_species: Aggregates (also excluded, elute early)
        fragment_species: Fragments (included in pores, elute late)
        yield_factor: Recovery yield
        species_order: List of species
    """
    column_volume: float | Array
    void_fraction: float | Array = 0.35
    target_species: str = "mAb"
    aggregate_species: str = "aggregates"
    fragment_species: str = "fragments"
    yield_factor: float | Array = 0.95
    resolution: float | Array = 1.5  # Baseline resolution
    species_order: list[str] = None
    # Axial-dispersion / peak-broadening coupling (#156). When True, the
    # cross-fraction carryover between neighbouring size peaks is derived from
    # the resolution using the equal-area Gaussian overlap relation
    # f_overlap = 0.5 * erfc(R_s * sqrt(2)) (Snyder, Kirkland & Dolan,
    # Introduction to Modern Liquid Chromatography, 3e, Wiley, 2010, Ch. 2),
    # so lower resolution (more band broadening) gives poorer separation.
    # When False, the legacy fixed 5% overlap is used (backward compatible).
    use_resolution_overlap: bool = False


def _load_fraction(load_volume, feed_volume, unit: str) -> Array:
    """Fraction of the inlet loaded onto a column, in [0, 1].

    ``load_volume`` is a volume (L), and the stream carries amounts, not a
    volume, so the fraction needs the feed volume too. Bio audit C11: without
    ``feed_volume`` the columns divided the load volume by the stream's total
    amount (L / g), so 10 L "loaded" 49.75% of a 20.1 g feed.

    Args:
        load_volume: Volume loaded (L), or None to load the whole inlet.
        feed_volume: Volume of the inlet (L).
        unit: Unit name for the error message.

    Returns:
        Loaded fraction of every species in the inlet.

    Raises:
        ValueError: When ``load_volume`` is given without ``feed_volume``.
    """
    if load_volume is None:
        return jnp.asarray(1.0)
    if feed_volume is None:
        raise ValueError(
            f"{unit}: load_volume is a volume (L) and the inlet stream carries "
            "amounts, not a volume, so the loaded fraction needs the feed "
            "volume too: pass feed_volume (L), or omit load_volume to load "
            "the whole inlet."
        )
    return jnp.clip(jnp.asarray(load_volume) / jnp.asarray(feed_volume), 0.0, 1.0)


def _feed_concentration(feed_concentration, amount, feed_volume):
    """Feed concentration (g/L) if known, else None (saturating feed)."""
    if feed_concentration is not None:
        return jnp.asarray(feed_concentration)
    if feed_volume is not None:
        return safe_divide(amount, jnp.asarray(feed_volume))
    return None


# =============================================================================
# Protein A Chromatography
# =============================================================================

class ProteinAChromatography:
    """Protein A affinity chromatography for mAb capture.

    Models the capture step of mAb purification using Protein A ligand.
    Includes:
    - Loading with breakthrough curve
    - Wash (impurity removal)
    - Elution (product recovery)

    Simplified model assuming:
    - Instantaneous binding equilibrium
    - Complete wash of unbound material
    - Specified elution yield
    """

    symbol = "Protein A"
    equations = [
        r"q = \frac{q_\mathrm{max}\,K_d\,C}{1 + K_d\,C}\qquad \text{(Langmuir binding)}",
        r"\mathrm{DBC}_{10\%} = \frac{q_\mathrm{max}\,\rho_\mathrm{bed}\,t_r}{1 + k'}\qquad \text{(dynamic binding capacity)}",
        r"Y_\mathrm{load} = 1 - C_\mathrm{breakthrough}/C_0",
    ]
    assumptions = [
        "Instantaneous binding equilibrium within the column.",
        "Complete wash of unbound species.",
        "User-specified elution yield; residual on column neglected.",
    ]
    references = [
        "Hahn, R., Jungbauer, A. J. Chromatogr. A, 1039, 189 (2004).",
        "Harrison, R.G., et al. Bioseparations Science and Engineering, 2e, Oxford, 2015.",
    ]
    parameter_symbols = {
        "column_volume": r"V_\mathrm{bed}",
        "q_max": r"q_\mathrm{max}",
        "K_d": "K_d",
    }
    parameter_units = {
        "column_volume": "L",
        "q_max": "g/L",
        "K_d": "g/L",
        "yield_factor": "-",
        "k_ads": "1/min",
        "n_plates": "-",
        "elution_cv": "CV",
    }
    numerical_method = "Closed-form Langmuir loading + mass-balance accounting across load/wash/elute."

    def __init__(self, params: ProteinAParams):
        """Initialize Protein A column.

        Args:
            params: Column and binding parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        load_volume: float | Array | None = None,
        breakthrough_limit: float | Array = 0.01,
        feed_volume: float | Array = None,
        load_flow_rate: float | Array = None,
        feed_concentration: float | Array = None,
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Run Protein A chromatography cycle.

        Args:
            inlet: Feed stream (concentrated harvest)
            load_volume: Volume of feed to load (L). None (the default)
                loads the entire inlet, which is what a train processing a
                whole batch wants; the column capacity then limits how much
                binds, and any excess breaks through to waste.
            breakthrough_limit: Acceptable breakthrough fraction (0-1)
            feed_volume: Total volume of the feed stream (L). Required with
                ``load_volume``: the loaded fraction is
                ``load_volume / feed_volume``.
            load_flow_rate: Volumetric load flow rate (L/min). When provided
                together with ``params.k_ads``, the dynamic binding capacity is
                computed from the column residence time (t_r = CV / Q) so DBC
                decreases at high flow rate (#100). If None, the static
                q_max-based DBC is used (backward compatible).
            feed_concentration: Target concentration in the feed (g/L), used
                in the Langmuir isotherm (with ``K_d``) for the binding
                capacity. If None it is the target amount over
                ``feed_volume`` when that is given; with neither, a
                saturating feed is assumed (q_eq -> q_max).

        Returns:
            (product, waste): Product (elution) and waste (FT + wash) streams
            info: Dictionary with:
                - 'yield': Target in the product / target in the inlet
                  (unloaded feed counts as lost)
                - 'purity': Product purity
                - 'DBC': Dynamic binding capacity used
                - 'impurity_clearance': LRV for each impurity
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        target_flow = inlet_flows.get(p.target_species, jnp.array(0.0))

        # Loaded fraction, clipped to [0, 1]: a load volume larger than the
        # feed volume otherwise loads more of the target than the feed
        # contains (loading 20 L of an 11.1 L feed closed 8 mol/s out). It
        # needs the feed volume (bio audit C11; see _load_fraction).
        _load_frac = _load_fraction(load_volume, feed_volume, "ProteinAChromatography")
        target_mass_loaded = target_flow * _load_frac

        # Dynamic binding capacity.
        # Kinetic model (#100): when an adsorption rate constant and a load
        # flow rate are given, DBC follows from the residence time so that it
        # drops at high flow rate. Otherwise fall back to the static estimate.
        if p.k_ads is not None and load_flow_rate is not None:
            residence_time = jnp.asarray(p.column_volume) / jnp.asarray(load_flow_rate)
            C_feed = (
                jnp.asarray(feed_concentration)
                if feed_concentration is not None
                else jnp.asarray(p.q_max)  # saturating proxy: q_eq -> q_max
            )
            DBC = dynamic_binding_capacity(
                p.q_max, C_feed, p.K_d, residence_time, p.k_ads
            )
        else:
            residence_time = None
            # Static capacity at low breakthrough: the Langmuir loading at
            # the feed concentration when that is known (from
            # feed_concentration, or the target amount over feed_volume),
            # else a saturating feed (q -> q_max). K_d did nothing on this
            # path before (bio audit C10).
            C_feed = _feed_concentration(feed_concentration, target_flow, feed_volume)
            q_eq = p.q_max if C_feed is None else langmuir_isotherm(C_feed, p.q_max, p.K_d)
            DBC = q_eq * (1.0 - breakthrough_limit)
        max_binding = DBC * p.column_volume

        # Actual bound mass (limited by capacity)
        target_bound = jnp.minimum(target_mass_loaded, max_binding)
        breakthrough_mass = target_mass_loaded - target_bound

        # Elution recovery
        target_eluted = target_bound * p.yield_factor

        # Calculate product stream (elution pool)
        product_flows = {p.target_species: target_eluted}

        # Add impurities (with clearance factors)
        waste_flows = {}
        for species, flow in inlet_flows.items():
            if species == p.target_species:
                # Target: unloaded feed + breakthrough + column losses to waste
                unloaded_target = flow * (1.0 - _load_frac)
                waste_flows[species] = unloaded_target + breakthrough_mass + target_bound * (1.0 - p.yield_factor)
            else:
                # Impurity: apply clearance factor to loaded portion
                mass_loaded = flow * _load_frac
                mass_unloaded = flow - mass_loaded
                clearance = p.impurity_clearance.get(species, 0.0)
                reduction_factor = 10.0 ** (-clearance)

                # Impurity in product
                impurity_in_product = mass_loaded * reduction_factor
                product_flows[species] = impurity_in_product

                # Rest goes to waste (unloaded + loaded-but-cleared)
                waste_flows[species] = mass_unloaded + mass_loaded * (1.0 - reduction_factor)

        product = make_stream(product_flows, inlet["T"], inlet["P"])
        waste = make_stream(waste_flows, inlet["T"], inlet["P"])

        # Calculate purity
        product_total = sum(product_flows.values())
        purity = jnp.where(product_total > 0, target_eluted / product_total, jnp.array(1.0))

        # Yield is product out over product IN, not over what was loaded:
        # the unloaded feed goes to waste, and a step yield that ignores it
        # reported 0.95 while 4.73 of 10 g reached the product (bio audit C11).
        yield_val = safe_divide(target_eluted, target_flow)

        info = {
            "yield": yield_val,
            "load_fraction": _load_frac,
            "purity": purity,
            "DBC": DBC,
            "mass_loaded": target_mass_loaded,
            "mass_bound": target_bound,
            "mass_eluted": target_eluted,
            "impurity_clearance": p.impurity_clearance,
            "capacity_utilization": target_bound / max_binding,
        }
        if residence_time is not None:
            info["residence_time"] = residence_time

        # Elution pool volume from column dimensions (#102). The peak base
        # width in volume units follows from the plate count,
        # N = 16 (V_R / w)^2  =>  w = 4 V_R / sqrt(N), with the retention
        # volume V_R = elution_cv * CV. The pool volume is taken as this base
        # width and sets the eluted product concentration.
        if p.n_plates is not None:
            V_R = jnp.asarray(p.elution_cv) * jnp.asarray(p.column_volume)
            pool_volume = 4.0 * V_R / jnp.sqrt(jnp.asarray(p.n_plates))
            info["elution_pool_volume"] = pool_volume
            info["product_concentration"] = safe_divide(target_eluted, pool_volume)

        return (product, waste), info

    def calculate_load_volume(
        self,
        feed_concentration: float | Array,
        target_utilization: float | Array = 0.8,
    ) -> Array:
        """Calculate recommended load volume.

        Args:
            feed_concentration: mAb concentration in feed (g/L)
            target_utilization: Target fraction of DBC to use (0-1)

        Returns:
            Recommended load volume (L)
        """
        p = self.params
        DBC = p.q_max * 0.9  # 10% breakthrough capacity
        max_mass = DBC * p.column_volume * target_utilization
        return max_mass / jnp.asarray(feed_concentration)


class IonExchangeChromatography:
    """Ion exchange chromatography for intermediate purification.

    Can operate in bind-elute or flow-through mode:
    - Bind-elute: Product binds, impurities flow through
    - Flow-through: Impurities bind, product flows through
    """

    symbol = "IEX"
    equations = [
        r"q = \frac{q_\mathrm{max}\,(K_b\,C)^n}{1 + (K_b\,C)^n}\qquad \text{(Langmuir-Freundlich)}",
        r"\log K = \log K_0 - Z\,\log [\mathrm{salt}]\qquad \text{(steric mass action)}",
    ]
    assumptions = [
        "Local equilibrium within the column.",
        "Gradient elution modelled as a linear salt ramp.",
        "Bind-elute or flow-through operating mode is user-selected.",
    ]
    references = [
        "Brooks, C.A., Cramer, S.M. AIChE J., 38, 1969 (1992).",
        "Jungbauer, A. J. Chromatogr. A, 1065, 3 (2005).",
    ]
    parameter_symbols = {"q_max": r"q_\mathrm{max}", "K_d": "K_d"}
    parameter_units = {
        "column_volume": "L",
        "q_max": "g/L",
        "K_d": "g/L",
        "yield_factor": "-",
    }
    numerical_method = "Isotherm-based mass balance; bind/flow-through selection at __call__."

    def __init__(self, params: IEXParams):
        """Initialize IEX column.

        Args:
            params: Column and binding parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        load_volume: float | Array | None = None,
        feed_volume: float | Array = None,
        feed_concentration: float | Array = None,
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Run ion exchange chromatography.

        The column has a finite capacity, q x ``column_volume``, with q from
        the Langmuir isotherm ``q_max C / (K_d + C)`` at the feed
        concentration C when that is known (``feed_concentration``, or the
        amount over ``feed_volume``), else ``q_max`` (a saturating feed).
        In bind-elute mode the target binds up to that capacity and the
        excess breaks through to waste; in flow-through mode the impurities
        the column would hold are scaled down together when their total
        exceeds it, and the excess stays in the product.

        Args:
            inlet: Feed stream
            load_volume: Volume loaded (L). None (the default) loads the
                entire inlet. Needs ``feed_volume``.
            feed_volume: Volume of the inlet (L); the loaded fraction is
                ``load_volume / feed_volume``.
            feed_concentration: Concentration (g/L) of what binds (the
                target in bind-elute mode, the bound impurities in total in
                flow-through mode), for the isotherm.

        Returns:
            (product, waste): Product and waste streams
            info: Operation details; 'yield' is the target in the product
            over the target in the inlet.

        Raises:
            ValueError: ``load_volume`` without ``feed_volume``.
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        # Loaded fraction, a volume ratio (bio audit C11: this divided the
        # load volume in L by the stream's total amount).
        load_frac = _load_fraction(load_volume, feed_volume, "IonExchangeChromatography")

        product_flows = {}
        waste_flows = {}

        # Bio audit C10: column_volume, q_max and K_d were documented and
        # ignored, so a 1 mL column processed any load. They now set a
        # capacity (see the docstring).
        def capacity(bound_amount):
            C = _feed_concentration(feed_concentration, bound_amount, feed_volume)
            q = p.q_max if C is None else langmuir_isotherm(C, p.q_max, p.K_d)
            return q * p.column_volume

        target = p.target_species
        target_in = inlet_flows.get(target, jnp.array(0.0))
        target_loaded = target_in * load_frac

        if p.mode == "bind_elute":
            cap = capacity(target_in)
            target_bound = jnp.minimum(target_loaded, cap)
            bound_demand = target_loaded
            scale = jnp.asarray(1.0)
        else:  # flow_through: the impurities bind
            cap = None
            demand = {}
            for species, flow in inlet_flows.items():
                if species == target:
                    continue
                loaded = flow * load_frac
                if species in p.impurity_clearance:
                    lrv = jnp.asarray(p.impurity_clearance[species])
                    demand[species] = loaded * (1.0 - 10.0 ** (-lrv))
                else:
                    demand[species] = loaded * jnp.asarray(p.selectivity.get(species, 0.5))
            bound_demand = sum(demand.values()) if demand else jnp.asarray(0.0)
            cap = capacity(sum(inlet_flows[s] for s in demand) if demand else jnp.asarray(0.0))
            # Every bound impurity shares the capacity in proportion to what
            # it would bind; at or below capacity this is exactly 1.
            scale = jnp.minimum(1.0, cap / jnp.maximum(bound_demand, 1e-300))
            target_bound = jnp.asarray(0.0)

        for species, flow in inlet_flows.items():
            mass_loaded = flow * load_frac
            mass_unloaded = flow - mass_loaded

            if species == target:
                if p.mode == "bind_elute":
                    # Bound target elutes at yield_factor; the breakthrough
                    # beyond capacity and the unloaded feed go to waste.
                    product_flows[species] = target_bound * p.yield_factor
                else:
                    product_flows[species] = mass_loaded * p.yield_factor
                waste_flows[species] = flow - product_flows[species]
            elif p.mode == "flow_through":
                to_waste = demand[species] * scale
                product_flows[species] = mass_loaded - to_waste
                waste_flows[species] = mass_unloaded + to_waste
            elif species in p.impurity_clearance:
                # Explicit clearance: 10**-LRV of the loaded impurity
                # reaches the product.
                to_product = mass_loaded * 10.0 ** (-jnp.asarray(p.impurity_clearance[species]))
                product_flows[species] = to_product
                waste_flows[species] = mass_unloaded + mass_loaded - to_product
            else:
                # Bind-elute impurity leakage: a strongly binding impurity
                # co-elutes with the target, leakage ~ selectivity^2.
                selectivity = jnp.asarray(p.selectivity.get(species, 0.5))
                to_product = mass_loaded * jnp.power(selectivity, 2) * (1.0 - p.yield_factor)
                product_flows[species] = to_product
                waste_flows[species] = mass_unloaded + mass_loaded - to_product

        product = make_stream(product_flows, inlet["T"], inlet["P"])
        waste = make_stream(waste_flows, inlet["T"], inlet["P"])

        target_out = product_flows.get(target, jnp.array(0.0))
        product_total = sum(product_flows.values())

        info = {
            # Product out over product IN (bio audit C11), not over the load.
            "yield": safe_divide(target_out, target_in),
            "purity": jnp.where(product_total > 0, target_out / product_total, jnp.array(1.0)),
            "mode": p.mode,
            "load_fraction": load_frac,
            "capacity": cap,
            "capacity_utilization": jnp.minimum(bound_demand, cap) / cap,
        }

        return (product, waste), info


class SizeExclusionChromatography:
    """Size exclusion chromatography for polishing.

    Separates based on molecular size:
    - Large molecules (aggregates) elute first
    - Target protein elutes at intermediate time
    - Small molecules (fragments) elute last

    Only the loaded fraction is separated. Whatever was never loaded onto
    the column (``load_volume`` below ``feed_volume``) bypasses it and stays
    in the product pool, whichever species it is: the target, the
    aggregates and the fragments are treated alike.
    """

    symbol = "SEC"
    equations = [
        r"V_e = V_0 + K_\mathrm{av}\,V_p,\qquad K_\mathrm{av} = \frac{V_e - V_0}{V_t - V_0}",
        r"R_s = \frac{1}{4}\,\frac{V_{e,1}-V_{e,2}}{\sigma_{1,2}}\qquad \text{(resolution)}",
    ]
    assumptions = [
        "No adsorption to the stationary phase.",
        "Gaussian peak approximation for resolution.",
        "Isocratic elution.",
    ]
    references = ["Hagel, L. Size Exclusion Chromatography, in Protein Purification, 2e, Wiley, 2011."]
    parameter_symbols = {"column_volume": "V_t", "void_fraction": r"\varepsilon_0"}
    parameter_units = {
        "column_volume": "L",
        "void_fraction": "-",
        "yield_factor": "-",
        "resolution": "-",
    }
    numerical_method = "Kav-based elution ordering + Gaussian peak overlap."

    def __init__(self, params: SECParams):
        """Initialize SEC column.

        Args:
            params: Column parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        load_volume: float | Array | None = None,
        feed_volume: float | Array = None,
    ) -> tuple[tuple[Stream, Stream, Stream], dict[str, Array]]:
        """Run size exclusion chromatography.

        Args:
            inlet: Feed stream
            load_volume: Volume loaded (L). None (the default) loads the
                entire inlet. Needs ``feed_volume``.
            feed_volume: Volume of the inlet (L); the loaded fraction is
                ``load_volume / feed_volume``.

        Returns:
            (product, aggregates, fragments): Three fractions
            info: Operation details; 'yield' is the target in the product
            over the target in the inlet.

        Raises:
            ValueError: ``load_volume`` without ``feed_volume``.
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        # Load fraction, a volume ratio (bio audit C11: this divided the
        # load volume in L by the stream's total amount).
        load_frac = _load_fraction(load_volume, feed_volume, "SizeExclusionChromatography")

        product_flows = {}
        aggregate_flows = {}
        fragment_flows = {}

        # Overlap (carryover into the product pool) between adjacent size
        # peaks. With the dispersion coupling on (#156), it is set by the
        # resolution; otherwise it is the legacy fixed 5%.
        if p.use_resolution_overlap:
            from jax.scipy.special import erfc
            overlap = 0.5 * erfc(jnp.asarray(p.resolution) * jnp.sqrt(2.0))
        else:
            overlap = jnp.asarray(0.05)

        for species, flow in inlet_flows.items():
            mass_loaded = flow * load_frac
            mass_unloaded = flow - mass_loaded

            if species == p.target_species:
                # Main product - some loss to side fractions; unloaded goes to product
                product_flows[species] = mass_unloaded + mass_loaded * p.yield_factor
                aggregate_flows[species] = mass_loaded * (1.0 - p.yield_factor) * 0.3
                fragment_flows[species] = mass_loaded * (1.0 - p.yield_factor) * 0.7

            elif species == p.aggregate_species:
                # Aggregates - the loaded part mostly to the aggregate
                # fraction, with overlap into product. What was never
                # loaded was not separated and stays in the product pool.
                aggregate_flows[species] = mass_loaded * (1.0 - overlap)
                product_flows[species] = mass_unloaded + mass_loaded * overlap

            elif species == p.fragment_species:
                # Fragments - as for aggregates: unloaded stays in the product
                fragment_flows[species] = mass_loaded * (1.0 - overlap)
                product_flows[species] = mass_unloaded + mass_loaded * overlap

            else:
                # Other species - distribute based on assumed size; unloaded to product
                product_flows[species] = mass_unloaded + mass_loaded * 0.1
                aggregate_flows[species] = mass_loaded * 0.1
                fragment_flows[species] = mass_loaded * 0.8  # Small molecules

        product = make_stream(product_flows, inlet["T"], inlet["P"])
        aggregates = make_stream(aggregate_flows, inlet["T"], inlet["P"])
        fragments = make_stream(fragment_flows, inlet["T"], inlet["P"])

        # Metrics
        # Yield and removal against what came IN (bio audit C11): measured
        # against the loaded amount, the unloaded target that is routed to
        # the product pool pushed the yield above 1 on a partial load.
        target_in = inlet_flows.get(p.target_species, jnp.array(0.0))
        target_out = product_flows.get(p.target_species, jnp.array(0.0))
        product_total = sum(product_flows.values())

        aggregate_in = inlet_flows.get(p.aggregate_species, jnp.array(0.0))
        aggregate_removed = 1.0 - safe_divide(product_flows.get(p.aggregate_species, jnp.array(0.0)), aggregate_in)

        info = {
            "yield": safe_divide(target_out, target_in),
            "load_fraction": load_frac,
            "purity": jnp.where(product_total > 0, target_out / product_total, jnp.array(1.0)),
            "aggregate_removal": aggregate_removed,
            "resolution": jnp.asarray(p.resolution),
            "peak_overlap": overlap,
        }

        return (product, aggregates, fragments), info


# =============================================================================
# Utility Functions
# =============================================================================

def dynamic_binding_capacity(
    q_max: Array,
    C_feed: Array,
    K_d: Array,
    residence_time: Array,
    k_ads: Array,
) -> Array:
    """Estimate dynamic binding capacity (DBC) at 10% breakthrough.

    Simplified model based on mass transfer kinetics.

    Args:
        q_max: Static binding capacity (g/L)
        C_feed: Feed concentration (g/L)
        K_d: Dissociation constant (g/L)
        residence_time: Column residence time (min)
        k_ads: Adsorption rate constant (1/min)

    Returns:
        DBC at 10% breakthrough (g/L)
    """
    # Equilibrium capacity at feed concentration
    q_eq = langmuir_isotherm(C_feed, q_max, K_d)

    # Mass transfer correction (simplified)
    efficiency = 1.0 - jnp.exp(-k_ads * residence_time)

    return q_eq * efficiency * 0.9  # 10% breakthrough


def column_productivity(
    DBC: Array,
    column_volume: Array,
    cycle_time: Array,
) -> Array:
    """Calculate column productivity.

    Productivity = DBC * CV / cycle_time

    Args:
        DBC: Dynamic binding capacity (g/L)
        column_volume: Column volume (L)
        cycle_time: Total cycle time (h)

    Returns:
        Productivity (g/h)
    """
    return DBC * column_volume / cycle_time


def resolution(
    t_R1: Array,
    t_R2: Array,
    w1: Array,
    w2: Array,
) -> Array:
    """Calculate chromatographic resolution between two peaks.

    R_s = 2 * (t_R2 - t_R1) / (w1 + w2)

    Args:
        t_R1: Retention time of peak 1
        t_R2: Retention time of peak 2
        w1: Width of peak 1 at base
        w2: Width of peak 2 at base

    Returns:
        Resolution (R_s > 1.5 for baseline separation)
    """
    return 2.0 * (t_R2 - t_R1) / (w1 + w2)


def plate_count(
    t_R: Array,
    w: Array,
) -> Array:
    """Calculate theoretical plate count from peak.

    N = 16 * (t_R / w)²

    Args:
        t_R: Retention time
        w: Peak width at base

    Returns:
        Number of theoretical plates
    """
    return 16.0 * (t_R / w) ** 2


def hetp(
    L: Array,
    N: Array,
) -> Array:
    """Calculate height equivalent to theoretical plate.

    HETP = L / N

    Args:
        L: Column length (cm)
        N: Number of theoretical plates

    Returns:
        HETP (cm)
    """
    return L / N
