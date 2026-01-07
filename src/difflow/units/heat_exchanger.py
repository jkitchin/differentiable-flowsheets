"""Heat exchanger unit operations.

This module provides differentiable heat exchanger models:
- Heater: Single stream heating with utility
- Cooler: Single stream cooling with utility
- CounterCurrentHX: Two-stream counter-current heat exchanger
- CoCurrentHX: Two-stream co-current (parallel flow) heat exchanger

All models use either LMTD or effectiveness-NTU methods and are
fully differentiable for gradient-based optimization.

Numerical Considerations:
- LMTD singularity: When ΔT₁ ≈ ΔT₂, LMTD → arithmetic mean via smooth blending
- Temperature crossing: Soft enforcement with warnings in info dict
- Cr = 1 edge case: Smooth blending between balanced and general formulas
"""

from dataclasses import dataclass, replace
from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, get_species


# =============================================================================
# Numerical Constants
# =============================================================================

# LMTD blending width: controls smooth transition when dT1 ≈ dT2
# When |ln(dT1/dT2)| < this value, blend toward arithmetic mean
LMTD_BLEND_WIDTH = 0.1

# Cr blending width: controls smooth transition when Cr → 1
# Larger values = smoother gradients, smaller = sharper transition
CR_BLEND_WIDTH = 0.05

# Minimum temperature difference to avoid numerical issues
MIN_DELTA_T = 1e-6


# =============================================================================
# Utility Functions
# =============================================================================


def log_mean_temperature_difference(
    dT1: Array,
    dT2: Array,
) -> Array:
    """Compute log mean temperature difference (LMTD).

    LMTD = (dT1 - dT2) / ln(dT1/dT2)

    Uses a numerically stable formulation:
    - When dT1 ≈ dT2: Uses Taylor expansion to avoid 0/0
    - When dT → 0: Returns small positive value (avoids log(0))
    - Smooth polynomial blending ensures continuous gradients

    The key insight is that LMTD = dT2 * (r-1) / ln(r) where r = dT1/dT2.
    Near r = 1, we use the Taylor expansion:
        (r-1)/ln(r) ≈ 1 + (r-1)/2 + (r-1)²/12 - ...
    This gives LMTD ≈ (dT1 + dT2)/2 + correction terms.

    Args:
        dT1: Temperature difference at one end (K)
        dT2: Temperature difference at other end (K)

    Returns:
        Log mean temperature difference (K), always positive
    """
    # Ensure positive temperature differences (physically meaningful)
    dT1_safe = jnp.maximum(jnp.abs(dT1), MIN_DELTA_T)
    dT2_safe = jnp.maximum(jnp.abs(dT2), MIN_DELTA_T)

    # Compute ratio r = dT1/dT2
    ratio = dT1_safe / dT2_safe

    # For numerical stability, compute (r-1)/ln(r) using different formulas
    # depending on how close r is to 1.
    #
    # When r ≈ 1: Use Taylor expansion (r-1)/ln(r) ≈ 1 + (r-1)/2 + (r-1)²/12
    # When r far from 1: Use direct formula (r-1)/ln(r)
    #
    # We use a smooth blend based on |r - 1|.

    r_minus_1 = ratio - 1.0

    # Taylor expansion for (r-1)/ln(r) near r=1 (up to 4th order for accuracy)
    # (r-1)/ln(r) = 1 + (r-1)/2 + (r-1)²/12 - (r-1)⁴/720 + ...
    taylor_factor = (
        1.0
        + r_minus_1 / 2.0
        + r_minus_1**2 / 12.0
        - r_minus_1**4 / 720.0
    )

    # Direct formula: (r-1)/ln(r)
    # Need to handle r = 1 case where ln(r) = 0
    log_ratio = jnp.log(ratio)
    # Add tiny value to prevent 0/0, but only affects very small |log_ratio|
    direct_factor = r_minus_1 / (log_ratio + 1e-30)

    # Smooth blending using a polynomial weight
    # When |r-1| < threshold, use Taylor; otherwise use direct
    # Using |r-1|² gives C¹ continuity at the blend point
    threshold = 0.1  # Blend when ratio is within 10% of 1
    t = jnp.abs(r_minus_1) / threshold
    # Smooth step: 0 when t < 1, 1 when t > 2, smooth transition between
    blend = jnp.clip(3 * t**2 - 2 * t**3, 0.0, 1.0)
    blend = jnp.where(t > 1, 1.0, blend)

    # Compute factor: blend → 0 uses Taylor, blend → 1 uses direct
    factor = blend * direct_factor + (1.0 - blend) * taylor_factor

    # LMTD = dT2 * factor
    lmtd = dT2_safe * factor

    return lmtd


def effectiveness_counter_current(NTU: Array, Cr: Array) -> Array:
    """Effectiveness for counter-current heat exchanger.

    Uses Taylor expansion near Cr=1 to ensure continuous derivatives
    for gradient-based optimization.

    General formula (Cr < 1):
        ε = (1 - exp(-NTU(1-Cr))) / (1 - Cr·exp(-NTU(1-Cr)))

    Balanced case (Cr = 1):
        ε = NTU / (1 + NTU)

    Near Cr = 1, we use Taylor expansion of the general formula.
    Let x = 1 - Cr (small), then:
        exp(-NTU·x) ≈ 1 - NTU·x + (NTU·x)²/2 - ...
        Numerator: 1 - exp(-NTU·x) ≈ NTU·x - (NTU·x)²/2 + ...
        Denominator: 1 - Cr·exp(-NTU·x) = 1 - (1-x)(1 - NTU·x + ...) ≈ x + NTU·x - NTU·x² + ...

    After careful expansion, ε ≈ NTU/(1+NTU) + corrections in (1-Cr).

    Args:
        NTU: Number of transfer units (UA/Cmin)
        Cr: Heat capacity ratio (Cmin/Cmax), in range [0, 1]

    Returns:
        Effectiveness in range (0, 1)
    """
    # Balanced case: ε = NTU / (1 + NTU)
    eps_balanced = NTU / (1.0 + NTU)

    # For numerical stability near Cr = 1, use Taylor expansion
    # Let x = 1 - Cr, then for small x:
    # ε ≈ NTU/(1+NTU) * (1 + x*NTU/(1+NTU)²/2 + ...)
    #
    # First-order correction term:
    x = 1.0 - Cr
    NTU_safe = jnp.maximum(NTU, 1e-10)
    base = NTU_safe / (1.0 + NTU_safe)

    # Taylor expansion: ε ≈ base * (1 + x * NTU * (1 - base) / (1 + NTU))
    # Simplified: ε ≈ base + base * x * NTU * (1 - base) / (1 + NTU)
    correction = base * x * NTU_safe * (1.0 - base) / (1.0 + NTU_safe)
    eps_taylor = base + correction

    # General case (direct formula)
    # Need to regularize for x → 0
    exp_arg = -NTU * jnp.maximum(x, 1e-10)
    exp_term = jnp.exp(exp_arg)
    # ε = (1 - exp(-NTU*x)) / (1 - (1-x)*exp(-NTU*x))
    numerator = 1.0 - exp_term
    denominator = 1.0 - (1.0 - x) * exp_term
    eps_general = numerator / (denominator + 1e-10)

    # Smooth polynomial blending
    # When |1-Cr| < threshold, use Taylor; otherwise use direct
    threshold = CR_BLEND_WIDTH
    t = jnp.abs(x) / threshold
    # Smooth step function
    blend = jnp.clip(3 * t**2 - 2 * t**3, 0.0, 1.0)
    blend = jnp.where(t > 1, 1.0, blend)

    # blend → 0 uses Taylor, blend → 1 uses general
    eps = blend * eps_general + (1.0 - blend) * eps_taylor

    return jnp.clip(eps, 0.0, 1.0)


def effectiveness_co_current(NTU: Array, Cr: Array) -> Array:
    """Effectiveness for co-current (parallel flow) heat exchanger.

    Formula: ε = (1 - exp(-NTU(1+Cr))) / (1 + Cr)

    This formula is well-behaved for all Cr ∈ [0, 1] since (1 + Cr) ≥ 1.

    Args:
        NTU: Number of transfer units (UA/Cmin)
        Cr: Heat capacity ratio (Cmin/Cmax)

    Returns:
        Effectiveness in range (0, 1)
    """
    eps = (1.0 - jnp.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)
    return jnp.clip(eps, 0.0, 1.0)


def heat_capacity_rate(
    stream: Stream,
    Cp_fn: Callable[[Stream], Array] | None = None,
    Cp_const: float | None = None,
) -> Array:
    """Compute heat capacity rate (m_dot * Cp) for a stream.

    Args:
        stream: Process stream
        Cp_fn: Function to compute Cp from stream (optional)
        Cp_const: Constant Cp value in J/(mol·K) (optional)

    Returns:
        Heat capacity rate in W/K
    """
    flows = get_flows(stream)
    F_total = sum(flows.values())

    if Cp_fn is not None:
        Cp = Cp_fn(stream)
    elif Cp_const is not None:
        Cp = Cp_const
    else:
        # Default: assume ideal gas Cp ≈ 30 J/(mol·K)
        Cp = 30.0

    return F_total * Cp


# =============================================================================
# Single-Stream Heat Exchangers (with utility)
# =============================================================================


@dataclass
class HeaterParams:
    """Parameters for a heater.

    Attributes:
        duty: Heat duty (W). Positive = heating.
        T_out: Outlet temperature (K). Alternative to duty.
        UA: Overall heat transfer coefficient × area (W/K). For rating.
        T_utility: Utility temperature (K). For LMTD calculation.
        Cp: Heat capacity (J/mol·K). If None, uses thermo.
    """
    duty: Array | float | None = None
    T_out: Array | float | None = None
    UA: Array | float | None = None
    T_utility: Array | float | None = None
    Cp: float | None = None

    def update(self, **kwargs) -> "HeaterParams":
        """Return a new HeaterParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., duty=1000.0, T_out=350.0)

        Returns:
            New HeaterParams with updated fields
        """
        return replace(self, **kwargs)

    def __getitem__(self, key: str):
        """Get parameter value by name for dict-like access."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


class Heater:
    """Single-stream heater with utility (steam, hot oil, etc).

    Can operate in three modes:
    1. Specified duty: Q given, calculate T_out
    2. Specified T_out: Calculate required Q
    3. Rating mode: Given UA and T_utility, calculate Q and T_out

    All modes are fully differentiable.
    """

    def __init__(self, params: HeaterParams):
        """Initialize heater.

        Args:
            params: Heater parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        duty: Array | float | None = None,
        T_out: Array | float | None = None,
    ) -> tuple[Stream, dict[str, Array]]:
        """Perform heater calculation.

        Args:
            inlet: Inlet stream
            duty: Heat duty override (W)
            T_out: Outlet temperature override (K)

        Returns:
            outlet: Outlet stream
            info: Dictionary with:
                - 'Q': Heat duty (W)
                - 'T_in': Inlet temperature (K)
                - 'T_out': Outlet temperature (K)
                - 'LMTD': Log mean temperature difference (K), if applicable
        """
        p = self.params

        # Get inlet properties
        T_in = inlet["T"]
        flows = get_flows(inlet)
        F_total = sum(flows.values())

        # Get Cp
        Cp = p.Cp if p.Cp is not None else 75.0  # Default liquid Cp

        # Heat capacity rate
        C = F_total * Cp

        # Determine operating mode
        Q = duty if duty is not None else p.duty
        T_out_spec = T_out if T_out is not None else p.T_out

        if Q is not None:
            # Mode 1: Duty specified
            Q = jnp.asarray(Q)
            T_out_calc = T_in + Q / C

        elif T_out_spec is not None:
            # Mode 2: Outlet temperature specified
            T_out_calc = jnp.asarray(T_out_spec)
            Q = C * (T_out_calc - T_in)

        elif p.UA is not None and p.T_utility is not None:
            # Mode 3: Rating with UA and utility temperature
            UA = jnp.asarray(p.UA)
            T_util = jnp.asarray(p.T_utility)

            # For heater: utility is hotter than process
            # Counter-current approximation:
            # Q = UA * LMTD where LMTD = (T_util - T_in) - (T_util - T_out) / ln(...)
            # This requires iteration, use effectiveness-NTU instead

            # NTU = UA / C (utility has infinite capacity)
            NTU = UA / C
            effectiveness = 1.0 - jnp.exp(-NTU)

            # Q_max = C * (T_utility - T_in)
            Q_max = C * (T_util - T_in)
            Q = effectiveness * Q_max
            T_out_calc = T_in + Q / C

        else:
            raise ValueError(
                "Must specify duty, T_out, or (UA and T_utility)"
            )

        # Build outlet stream
        outlet = dict(inlet)
        outlet["T"] = T_out_calc

        # Compute LMTD if utility temperature is known
        info = {
            "Q": Q,
            "T_in": T_in,
            "T_out": T_out_calc,
        }

        if p.T_utility is not None:
            T_util = jnp.asarray(p.T_utility)
            dT1 = T_util - T_in
            dT2 = T_util - T_out_calc
            info["LMTD"] = log_mean_temperature_difference(dT1, dT2)
            info["UA_required"] = Q / info["LMTD"]

        return outlet, info


@dataclass
class CoolerParams:
    """Parameters for a cooler.

    Attributes:
        duty: Heat duty (W). Positive = cooling (heat removed).
        T_out: Outlet temperature (K). Alternative to duty.
        UA: Overall heat transfer coefficient × area (W/K). For rating.
        T_utility: Utility temperature (K). For LMTD calculation.
        Cp: Heat capacity (J/mol·K). If None, uses default.
    """
    duty: Array | float | None = None
    T_out: Array | float | None = None
    UA: Array | float | None = None
    T_utility: Array | float | None = None
    Cp: float | None = None

    def update(self, **kwargs) -> "CoolerParams":
        """Return a new CoolerParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., duty=1000.0, T_out=300.0)

        Returns:
            New CoolerParams with updated fields
        """
        return replace(self, **kwargs)

    def __getitem__(self, key: str):
        """Get parameter value by name for dict-like access."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


class Cooler:
    """Single-stream cooler with utility (cooling water, refrigerant, etc).

    Same operating modes as Heater but for cooling.
    """

    def __init__(self, params: CoolerParams):
        """Initialize cooler.

        Args:
            params: Cooler parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        duty: Array | float | None = None,
        T_out: Array | float | None = None,
    ) -> tuple[Stream, dict[str, Array]]:
        """Perform cooler calculation.

        Args:
            inlet: Inlet stream
            duty: Heat duty override (W), positive = heat removed
            T_out: Outlet temperature override (K)

        Returns:
            outlet: Outlet stream
            info: Dictionary with Q, T_in, T_out, LMTD (if applicable)
        """
        p = self.params

        T_in = inlet["T"]
        flows = get_flows(inlet)
        F_total = sum(flows.values())

        Cp = p.Cp if p.Cp is not None else 75.0
        C = F_total * Cp

        Q = duty if duty is not None else p.duty
        T_out_spec = T_out if T_out is not None else p.T_out

        if Q is not None:
            # Duty specified (positive = heat removed)
            Q = jnp.asarray(Q)
            T_out_calc = T_in - Q / C

        elif T_out_spec is not None:
            T_out_calc = jnp.asarray(T_out_spec)
            Q = C * (T_in - T_out_calc)

        elif p.UA is not None and p.T_utility is not None:
            UA = jnp.asarray(p.UA)
            T_util = jnp.asarray(p.T_utility)

            NTU = UA / C
            effectiveness = 1.0 - jnp.exp(-NTU)

            Q_max = C * (T_in - T_util)
            Q = effectiveness * Q_max
            T_out_calc = T_in - Q / C

        else:
            raise ValueError(
                "Must specify duty, T_out, or (UA and T_utility)"
            )

        outlet = dict(inlet)
        outlet["T"] = T_out_calc

        info = {
            "Q": Q,
            "T_in": T_in,
            "T_out": T_out_calc,
        }

        if p.T_utility is not None:
            T_util = jnp.asarray(p.T_utility)
            dT1 = T_in - T_util
            dT2 = T_out_calc - T_util
            info["LMTD"] = log_mean_temperature_difference(dT1, dT2)
            info["UA_required"] = Q / info["LMTD"]

        return outlet, info


# =============================================================================
# Two-Stream Heat Exchangers
# =============================================================================


@dataclass
class HeatExchangerParams:
    """Parameters for two-stream heat exchangers.

    Attributes:
        UA: Overall heat transfer coefficient × area (W/K)
        Cp_hot: Hot side heat capacity (J/mol·K). If None, uses default.
        Cp_cold: Cold side heat capacity (J/mol·K). If None, uses default.
        min_approach: Minimum temperature approach (K). For design mode.
    """
    UA: Array | float | None = None
    Cp_hot: float | None = None
    Cp_cold: float | None = None
    min_approach: float = 10.0

    def update(self, **kwargs) -> "HeatExchangerParams":
        """Return a new HeatExchangerParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., UA=500.0)

        Returns:
            New HeatExchangerParams with updated fields
        """
        return replace(self, **kwargs)

    def __getitem__(self, key: str):
        """Get parameter value by name for dict-like access."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


class CounterCurrentHX:
    """Counter-current (shell-and-tube style) heat exchanger.

    Hot fluid flows opposite to cold fluid direction, achieving
    higher effectiveness than co-current flow.

    Uses effectiveness-NTU method for fully differentiable calculations.

    Temperature Crossing:
        If T_hot_in < T_cold_in (hot stream is actually colder), the calculation
        still proceeds but returns Q < 0 (heat flows from cold to hot) and
        sets a 'temperature_crossing' flag in the info dict. This "soft"
        handling allows gradient-based optimizers to recover from infeasible
        intermediate states without causing NaN values.
    """

    def __init__(self, params: HeatExchangerParams):
        """Initialize counter-current heat exchanger.

        Args:
            params: Heat exchanger parameters
        """
        self.params = params

    def __call__(
        self,
        hot_inlet: Stream,
        cold_inlet: Stream,
        UA: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform heat exchanger calculation.

        Args:
            hot_inlet: Hot stream inlet
            cold_inlet: Cold stream inlet
            UA: Override UA value (W/K)

        Returns:
            hot_outlet: Hot stream outlet
            cold_outlet: Cold stream outlet
            info: Dictionary with:
                - 'Q': Heat transferred (W), positive = hot to cold
                - 'effectiveness': Heat exchanger effectiveness
                - 'NTU': Number of transfer units
                - 'LMTD': Log mean temperature difference (K)
                - 'T_hot_out': Hot outlet temperature (K)
                - 'T_cold_out': Cold outlet temperature (K)
                - 'temperature_crossing': True if T_hot_in < T_cold_in (unphysical)
                - 'driving_force': T_hot_in - T_cold_in (should be positive)
        """
        p = self.params

        # Get inlet temperatures
        T_hot_in = hot_inlet["T"]
        T_cold_in = cold_inlet["T"]

        # Temperature crossing check (soft - doesn't prevent calculation)
        # When T_hot_in < T_cold_in, the "hot" stream is actually colder
        # Q_max becomes negative, resulting in heat flow from cold to hot
        driving_force = T_hot_in - T_cold_in
        temperature_crossing = driving_force < 0

        # Get flow rates
        hot_flows = get_flows(hot_inlet)
        cold_flows = get_flows(cold_inlet)
        F_hot = sum(hot_flows.values())
        F_cold = sum(cold_flows.values())

        # Heat capacities
        Cp_hot = p.Cp_hot if p.Cp_hot is not None else 75.0
        Cp_cold = p.Cp_cold if p.Cp_cold is not None else 75.0

        # Heat capacity rates (ensure positive with small regularization)
        C_hot = jnp.maximum(F_hot * Cp_hot, 1e-10)
        C_cold = jnp.maximum(F_cold * Cp_cold, 1e-10)

        C_min = jnp.minimum(C_hot, C_cold)
        C_max = jnp.maximum(C_hot, C_cold)
        Cr = C_min / C_max  # Already safe since C_max >= C_min >= 1e-10

        # Get UA
        UA_val = UA if UA is not None else p.UA
        if UA_val is None:
            raise ValueError("UA must be specified")
        UA_val = jnp.asarray(UA_val)

        # NTU and effectiveness
        NTU = UA_val / C_min
        eps = effectiveness_counter_current(NTU, Cr)

        # Maximum possible heat transfer
        # Note: Q_max can be negative if temperature crossing occurs
        Q_max = C_min * driving_force

        # Actual heat transfer (preserves sign of Q_max)
        Q = eps * Q_max

        # Outlet temperatures
        T_hot_out = T_hot_in - Q / C_hot
        T_cold_out = T_cold_in + Q / C_cold

        # Build outlet streams
        hot_outlet = dict(hot_inlet)
        hot_outlet["T"] = T_hot_out

        cold_outlet = dict(cold_inlet)
        cold_outlet["T"] = T_cold_out

        # LMTD (counter-current)
        # Use absolute values to get meaningful LMTD even with crossing
        dT1 = T_hot_in - T_cold_out  # Hot inlet vs cold outlet
        dT2 = T_hot_out - T_cold_in  # Hot outlet vs cold inlet
        LMTD = log_mean_temperature_difference(dT1, dT2)

        info = {
            "Q": Q,
            "effectiveness": eps,
            "NTU": NTU,
            "Cr": Cr,
            "LMTD": LMTD,
            "T_hot_in": T_hot_in,
            "T_hot_out": T_hot_out,
            "T_cold_in": T_cold_in,
            "T_cold_out": T_cold_out,
            "approach": jnp.minimum(jnp.abs(dT1), jnp.abs(dT2)),
            "driving_force": driving_force,
            "temperature_crossing": temperature_crossing,
        }

        return hot_outlet, cold_outlet, info


class CoCurrentHX:
    """Co-current (parallel flow) heat exchanger.

    Both fluids flow in the same direction. Lower effectiveness
    than counter-current, but useful for some applications.

    Temperature Crossing:
        Same soft handling as CounterCurrentHX - calculation proceeds
        with a warning flag in the info dict.
    """

    def __init__(self, params: HeatExchangerParams):
        """Initialize co-current heat exchanger.

        Args:
            params: Heat exchanger parameters
        """
        self.params = params

    def __call__(
        self,
        hot_inlet: Stream,
        cold_inlet: Stream,
        UA: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform heat exchanger calculation.

        Args:
            hot_inlet: Hot stream inlet
            cold_inlet: Cold stream inlet
            UA: Override UA value (W/K)

        Returns:
            hot_outlet: Hot stream outlet
            cold_outlet: Cold stream outlet
            info: Dictionary with Q, effectiveness, NTU, LMTD, etc.
                  Includes 'temperature_crossing' flag if T_hot_in < T_cold_in.
        """
        p = self.params

        T_hot_in = hot_inlet["T"]
        T_cold_in = cold_inlet["T"]

        # Temperature crossing check
        driving_force = T_hot_in - T_cold_in
        temperature_crossing = driving_force < 0

        hot_flows = get_flows(hot_inlet)
        cold_flows = get_flows(cold_inlet)
        F_hot = sum(hot_flows.values())
        F_cold = sum(cold_flows.values())

        Cp_hot = p.Cp_hot if p.Cp_hot is not None else 75.0
        Cp_cold = p.Cp_cold if p.Cp_cold is not None else 75.0

        # Heat capacity rates (ensure positive)
        C_hot = jnp.maximum(F_hot * Cp_hot, 1e-10)
        C_cold = jnp.maximum(F_cold * Cp_cold, 1e-10)

        C_min = jnp.minimum(C_hot, C_cold)
        C_max = jnp.maximum(C_hot, C_cold)
        Cr = C_min / C_max

        UA_val = UA if UA is not None else p.UA
        if UA_val is None:
            raise ValueError("UA must be specified")
        UA_val = jnp.asarray(UA_val)

        NTU = UA_val / C_min
        eps = effectiveness_co_current(NTU, Cr)

        Q_max = C_min * driving_force
        Q = eps * Q_max

        T_hot_out = T_hot_in - Q / C_hot
        T_cold_out = T_cold_in + Q / C_cold

        hot_outlet = dict(hot_inlet)
        hot_outlet["T"] = T_hot_out

        cold_outlet = dict(cold_inlet)
        cold_outlet["T"] = T_cold_out

        # LMTD (co-current)
        dT1 = T_hot_in - T_cold_in  # Both inlets
        dT2 = T_hot_out - T_cold_out  # Both outlets
        LMTD = log_mean_temperature_difference(dT1, dT2)

        info = {
            "Q": Q,
            "effectiveness": eps,
            "NTU": NTU,
            "Cr": Cr,
            "LMTD": LMTD,
            "T_hot_in": T_hot_in,
            "T_hot_out": T_hot_out,
            "T_cold_in": T_cold_in,
            "T_cold_out": T_cold_out,
            "approach": jnp.abs(dT2),  # Co-current approach is at outlets
            "driving_force": driving_force,
            "temperature_crossing": temperature_crossing,
        }

        return hot_outlet, cold_outlet, info


# =============================================================================
# Design Functions
# =============================================================================


def design_heat_exchanger(
    Q: Array,
    T_hot_in: Array,
    T_hot_out: Array,
    T_cold_in: Array,
    T_cold_out: Array,
    U: Array,
    flow_config: str = "counter_current",
) -> dict[str, Array]:
    """Design a heat exchanger given temperatures and duty.

    Calculates required area from LMTD method.

    Args:
        Q: Heat duty (W)
        T_hot_in: Hot inlet temperature (K)
        T_hot_out: Hot outlet temperature (K)
        T_cold_in: Cold inlet temperature (K)
        T_cold_out: Cold outlet temperature (K)
        U: Overall heat transfer coefficient (W/m²·K)
        flow_config: "counter_current" or "co_current"

    Returns:
        Dictionary with:
        - 'A': Required area (m²)
        - 'LMTD': Log mean temperature difference (K)
        - 'UA': UA value (W/K)
    """
    if flow_config == "counter_current":
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in
    else:  # co_current
        dT1 = T_hot_in - T_cold_in
        dT2 = T_hot_out - T_cold_out

    LMTD = log_mean_temperature_difference(dT1, dT2)
    UA = Q / LMTD
    A = UA / U

    return {
        "A": A,
        "LMTD": LMTD,
        "UA": UA,
        "dT1": dT1,
        "dT2": dT2,
    }


def size_heat_exchanger(
    Q: Array,
    T_hot_in: Array,
    T_cold_in: Array,
    C_hot: Array,
    C_cold: Array,
    U: Array,
    flow_config: str = "counter_current",
) -> dict[str, Array]:
    """Size a heat exchanger given heat capacity rates.

    Uses effectiveness-NTU method in reverse to find required UA/area.
    Smooth blending is used when Cr → 1 to ensure differentiability.

    Args:
        Q: Heat duty (W)
        T_hot_in: Hot inlet temperature (K)
        T_cold_in: Cold inlet temperature (K)
        C_hot: Hot side heat capacity rate (W/K)
        C_cold: Cold side heat capacity rate (W/K)
        U: Overall heat transfer coefficient (W/m²·K)
        flow_config: "counter_current" or "co_current"

    Returns:
        Dictionary with A, UA, NTU, effectiveness, and temperature crossing flag
    """
    # Ensure positive heat capacity rates
    C_hot = jnp.maximum(C_hot, 1e-10)
    C_cold = jnp.maximum(C_cold, 1e-10)

    C_min = jnp.minimum(C_hot, C_cold)
    C_max = jnp.maximum(C_hot, C_cold)
    Cr = C_min / C_max

    # Check for temperature crossing
    driving_force = T_hot_in - T_cold_in
    temperature_crossing = driving_force < 0

    Q_max = C_min * driving_force
    # Clip effectiveness to valid range [0, 1) to avoid log of negative numbers
    eps = jnp.clip(Q / (Q_max + 1e-10 * jnp.sign(Q_max + 1e-20)), 0.0, 0.9999)

    # Invert effectiveness-NTU relationship with smooth blending for Cr → 1
    if flow_config == "counter_current":
        # Balanced case (Cr = 1): NTU = eps / (1 - eps)
        NTU_balanced = eps / (1.0 - eps + 1e-10)

        # General case: NTU = ln((1 - eps*Cr) / (1 - eps)) / (1 - Cr)
        one_minus_Cr = jnp.maximum(1.0 - Cr, 1e-10)
        # Ensure arguments to log are positive
        log_arg = jnp.maximum((1.0 - eps * Cr) / (1.0 - eps + 1e-10), 1e-10)
        NTU_general = jnp.log(log_arg) / one_minus_Cr

        # Smooth blending
        blend_weight = jax.nn.sigmoid((1.0 - Cr - CR_BLEND_WIDTH) / (CR_BLEND_WIDTH / 3))
        NTU = blend_weight * NTU_general + (1.0 - blend_weight) * NTU_balanced
    else:
        # For co-current: NTU = -ln(1 - eps(1+Cr)) / (1 + Cr)
        # This formula is well-behaved since (1 + Cr) >= 1
        one_plus_Cr = 1.0 + Cr
        log_arg = jnp.maximum(1.0 - eps * one_plus_Cr, 1e-10)
        NTU = -jnp.log(log_arg) / one_plus_Cr

    # Ensure NTU is positive
    NTU = jnp.maximum(NTU, 0.0)

    UA = NTU * C_min
    A = UA / U

    # Calculate outlet temperatures
    T_hot_out = T_hot_in - Q / C_hot
    T_cold_out = T_cold_in + Q / C_cold

    return {
        "A": A,
        "UA": UA,
        "NTU": NTU,
        "effectiveness": eps,
        "Cr": Cr,
        "T_hot_out": T_hot_out,
        "T_cold_out": T_cold_out,
        "driving_force": driving_force,
        "temperature_crossing": temperature_crossing,
    }
