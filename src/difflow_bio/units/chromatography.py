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


@dataclass(repr=False)
class IEXParams(ParamsMixin):
    """Parameters for ion exchange chromatography.

    Attributes:
        column_volume: Column volume (L)
        mode: "bind_elute" or "flow_through"
        q_max: Maximum binding capacity (g/L)
        K_d: Dissociation constant (g/L)
        target_species: Target protein species
        selectivity: Dict of species -> binding selectivity (0=no binding, 1=strong)
        yield_factor: Recovery yield
        species_order: List of species
    """
    column_volume: float | Array
    mode: Literal["bind_elute", "flow_through"] = "bind_elute"
    q_max: float | Array = 50.0
    K_d: float | Array = 0.5
    target_species: str = "mAb"
    selectivity: dict = field(default_factory=dict)
    yield_factor: float | Array = 0.90
    species_order: list[str] = None


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

    def __init__(self, params: ProteinAParams):
        """Initialize Protein A column.

        Args:
            params: Column and binding parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        load_volume: float | Array,
        breakthrough_limit: float | Array = 0.01,
        feed_volume: float | Array = None,
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Run Protein A chromatography cycle.

        Args:
            inlet: Feed stream (concentrated harvest)
            load_volume: Volume of feed to load (L)
            breakthrough_limit: Acceptable breakthrough fraction (0-1)
            feed_volume: Total volume of feed stream (L). If provided, used to
                calculate concentration. If None, assumes load_volume/total_flow
                gives the mass fraction loaded.

        Returns:
            (product, waste): Product (elution) and waste (FT + wash) streams
            info: Dictionary with:
                - 'yield': Product recovery
                - 'purity': Product purity
                - 'DBC': Dynamic binding capacity used
                - 'impurity_clearance': LRV for each impurity
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        total_flow = sum(inlet_flows.values())
        target_flow = inlet_flows.get(p.target_species, jnp.array(0.0))

        # Mass loaded calculation
        if feed_volume is not None:
            # Proper calculation: concentration = mass/volume, then mass = conc * load_vol
            # load_fraction = load_volume / feed_volume
            load_fraction = jnp.asarray(load_volume) / jnp.asarray(feed_volume)
            target_mass_loaded = target_flow * load_fraction
        else:
            # Legacy: assume flows represent concentrations (g/L) and total_flow is volume
            target_mass_loaded = target_flow * load_volume / total_flow

        # Dynamic binding capacity (simplified)
        # DBC = q_max at low breakthrough
        DBC = p.q_max * (1.0 - breakthrough_limit)
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
                # Target: breakthrough + column losses to waste
                waste_flows[species] = breakthrough_mass + target_bound * (1.0 - p.yield_factor)
            else:
                # Impurity: apply clearance factor
                if feed_volume is not None:
                    mass_loaded = flow * load_fraction
                else:
                    mass_loaded = flow * load_volume / total_flow
                clearance = p.impurity_clearance.get(species, 0.0)
                reduction_factor = 10.0 ** (-clearance)

                # Impurity in product
                impurity_in_product = mass_loaded * reduction_factor
                product_flows[species] = impurity_in_product

                # Rest goes to waste
                waste_flows[species] = mass_loaded * (1.0 - reduction_factor)

        product = make_stream(product_flows, inlet["T"], inlet["P"])
        waste = make_stream(waste_flows, inlet["T"], inlet["P"])

        # Calculate purity
        product_total = sum(product_flows.values())
        purity = jnp.where(product_total > 0, target_eluted / product_total, jnp.array(1.0))

        # Calculate yield
        yield_val = jnp.where(target_mass_loaded > 0, target_eluted / target_mass_loaded, jnp.array(0.0))

        info = {
            "yield": yield_val,
            "purity": purity,
            "DBC": DBC,
            "mass_loaded": target_mass_loaded,
            "mass_bound": target_bound,
            "mass_eluted": target_eluted,
            "impurity_clearance": p.impurity_clearance,
            "capacity_utilization": target_bound / max_binding,
        }

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

    def __init__(self, params: IEXParams):
        """Initialize IEX column.

        Args:
            params: Column and binding parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        load_volume: float | Array,
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Run ion exchange chromatography.

        Args:
            inlet: Feed stream
            load_volume: Volume loaded (L)

        Returns:
            (product, waste): Product and waste streams
            info: Operation details
        """
        p = self.params
        inlet_flows = get_flows(inlet)
        total_flow = sum(inlet_flows.values())

        product_flows = {}
        waste_flows = {}

        for species, flow in inlet_flows.items():
            mass = flow * load_volume / total_flow
            selectivity = p.selectivity.get(species, 0.5)  # Default moderate binding
            selectivity = jnp.asarray(selectivity)

            if p.mode == "bind_elute":
                # High selectivity = binds = goes to product
                if species == p.target_species:
                    product_flows[species] = mass * p.yield_factor
                    waste_flows[species] = mass * (1.0 - p.yield_factor)
                else:
                    # Impurities: high selectivity = retained = waste
                    # low selectivity = flow through = waste
                    # Intermediate = some in product
                    to_product = mass * (1.0 - selectivity) * 0.1  # Leakage
                    product_flows[species] = to_product
                    waste_flows[species] = mass - to_product

            else:  # flow_through
                # Target flows through (product), impurities bind (waste)
                if species == p.target_species:
                    product_flows[species] = mass * p.yield_factor
                    waste_flows[species] = mass * (1.0 - p.yield_factor)
                else:
                    # High selectivity = binds = waste
                    to_waste = mass * selectivity
                    product_flows[species] = mass - to_waste
                    waste_flows[species] = to_waste

        product = make_stream(product_flows, inlet["T"], inlet["P"])
        waste = make_stream(waste_flows, inlet["T"], inlet["P"])

        # Calculate metrics
        target_in = inlet_flows.get(p.target_species, jnp.array(1.0)) * load_volume / total_flow
        target_out = product_flows.get(p.target_species, jnp.array(0.0))
        product_total = sum(product_flows.values())

        info = {
            "yield": jnp.where(target_in > 0, target_out / target_in, jnp.array(0.0)),
            "purity": jnp.where(product_total > 0, target_out / product_total, jnp.array(1.0)),
            "mode": p.mode,
        }

        return (product, waste), info


class SizeExclusionChromatography:
    """Size exclusion chromatography for polishing.

    Separates based on molecular size:
    - Large molecules (aggregates) elute first
    - Target protein elutes at intermediate time
    - Small molecules (fragments) elute last
    """

    def __init__(self, params: SECParams):
        """Initialize SEC column.

        Args:
            params: Column parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        load_volume: float | Array,
    ) -> tuple[tuple[Stream, Stream, Stream], dict[str, Array]]:
        """Run size exclusion chromatography.

        Args:
            inlet: Feed stream
            load_volume: Volume loaded (L)

        Returns:
            (product, aggregates, fragments): Three fractions
            info: Operation details
        """
        p = self.params
        inlet_flows = get_flows(inlet)
        total_flow = sum(inlet_flows.values())

        product_flows = {}
        aggregate_flows = {}
        fragment_flows = {}

        for species, flow in inlet_flows.items():
            mass = flow * load_volume / total_flow

            if species == p.target_species:
                # Main product - some loss to side fractions
                product_flows[species] = mass * p.yield_factor
                aggregate_flows[species] = mass * (1.0 - p.yield_factor) * 0.3
                fragment_flows[species] = mass * (1.0 - p.yield_factor) * 0.7

            elif species == p.aggregate_species:
                # Aggregates - mostly to aggregate fraction
                aggregate_flows[species] = mass * 0.95
                product_flows[species] = mass * 0.05  # Some overlap

            elif species == p.fragment_species:
                # Fragments - mostly to fragment fraction
                fragment_flows[species] = mass * 0.95
                product_flows[species] = mass * 0.05

            else:
                # Other species - distribute based on assumed size
                product_flows[species] = mass * 0.1
                aggregate_flows[species] = mass * 0.1
                fragment_flows[species] = mass * 0.8  # Small molecules

        product = make_stream(product_flows, inlet["T"], inlet["P"])
        aggregates = make_stream(aggregate_flows, inlet["T"], inlet["P"])
        fragments = make_stream(fragment_flows, inlet["T"], inlet["P"])

        # Metrics
        target_in = inlet_flows.get(p.target_species, jnp.array(1.0)) * load_volume / total_flow
        target_out = product_flows.get(p.target_species, jnp.array(0.0))
        product_total = sum(product_flows.values())

        aggregate_in = inlet_flows.get(p.aggregate_species, jnp.array(0.0)) * load_volume / total_flow
        aggregate_removed = 1.0 - product_flows.get(p.aggregate_species, jnp.array(0.0)) / (aggregate_in + 1e-10)

        info = {
            "yield": jnp.where(target_in > 0, target_out / target_in, jnp.array(0.0)),
            "purity": jnp.where(product_total > 0, target_out / product_total, jnp.array(1.0)),
            "aggregate_removal": aggregate_removed,
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
