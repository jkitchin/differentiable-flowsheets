"""Heat exchanger unit operations.

This module provides differentiable heat exchanger models:
- Heater: Single stream heating with utility
- Cooler: Single stream cooling with utility
- CounterCurrentHX: Two-stream counter-current heat exchanger
- CoCurrentHX: Two-stream co-current (parallel flow) heat exchanger
- CrossFlowHX: Two-stream cross-flow heat exchanger with configurable mixing
  - both_unmixed: Both fluids unmixed (most common, e.g., car radiators)
  - cmax_mixed: Cmax mixed, Cmin unmixed
  - cmin_mixed: Cmin mixed, Cmax unmixed
  - both_mixed: Both fluids mixed

All models use either LMTD or effectiveness-NTU methods and are
fully differentiable for gradient-based optimization.

Numerical Considerations:
- LMTD singularity: When ΔT₁ ≈ ΔT₂, LMTD → arithmetic mean via smooth blending
- Temperature crossing: Soft enforcement with warnings in info dict
- Cr = 1 edge case: Smooth blending between balanced and general formulas
- Cross-flow with Cr → 0: Smooth blending to limiting effectiveness
"""

from dataclasses import dataclass
from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, get_species
from difflow.params_mixin import ParamsMixin
from difflow.constants import LMTD_BLEND_WIDTH, CR_BLEND_WIDTH, MIN_DELTA_T, EPS_DIVISION
from difflow.numerics import safe_divide, safe_log


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
    # Use safe_divide to prevent 0/0
    direct_factor = safe_divide(r_minus_1, log_ratio)

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
    eps_general = safe_divide(numerator, denominator)

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


def effectiveness_crossflow_both_unmixed(NTU: Array, Cr: Array) -> Array:
    """Effectiveness for cross-flow heat exchanger with both fluids unmixed.

    This is the most common cross-flow configuration (e.g., finned-tube HX,
    car radiators). Both fluids flow through separate channels perpendicular
    to each other with no mixing in the flow direction.

    Formula: ε = 1 - exp[(NTU^0.22/Cr) * (exp(-Cr*NTU^0.78) - 1)]

    This is an approximation that can sometimes overpredict at high NTU and Cr.
    We cap the effectiveness at the counter-current value to maintain physical
    consistency (cross-flow should never exceed counter-current effectiveness).

    Numerical handling:
    - For Cr → 0: Uses Taylor expansion to avoid division by zero
    - Smooth blending ensures continuous gradients
    - Capped at counter-current effectiveness

    Args:
        NTU: Number of transfer units (UA/Cmin)
        Cr: Heat capacity ratio (Cmin/Cmax), in range [0, 1]

    Returns:
        Effectiveness in range (0, 1)
    """
    NTU_safe = jnp.maximum(NTU, 1e-10)
    Cr_safe = jnp.maximum(Cr, 1e-10)

    # Standard formula for Cr not too small
    # ε = 1 - exp[(NTU^0.22/Cr) * (exp(-Cr*NTU^0.78) - 1)]
    NTU_022 = NTU_safe**0.22
    NTU_078 = NTU_safe**0.78
    exp_term = jnp.exp(-Cr_safe * NTU_078)
    exponent = (NTU_022 / Cr_safe) * (exp_term - 1.0)
    eps_general = 1.0 - jnp.exp(exponent)

    # For very small Cr (Cmin << Cmax), use limiting case
    # As Cr → 0: ε → 1 - exp(-NTU)
    eps_limit = 1.0 - jnp.exp(-NTU_safe)

    # Smooth blend when Cr < 0.01
    blend_threshold = 0.01
    t = Cr / blend_threshold
    blend = jnp.clip(3 * t**2 - 2 * t**3, 0.0, 1.0)

    eps = blend * eps_general + (1.0 - blend) * eps_limit

    # Physical constraint: cross-flow effectiveness cannot exceed counter-current
    # The approximation formula can sometimes overshoot, so we cap it
    eps_counter = effectiveness_counter_current(NTU, Cr)
    eps = jnp.minimum(eps, eps_counter)

    return jnp.clip(eps, 0.0, 1.0)


def effectiveness_crossflow_cmax_mixed(NTU: Array, Cr: Array) -> Array:
    """Effectiveness for cross-flow with Cmax mixed, Cmin unmixed.

    One fluid (with larger heat capacity rate) flows through a single
    channel and can mix. The other fluid (smaller heat capacity rate)
    flows through separate tubes/channels without mixing.

    Formula: ε = (1/Cr) * [1 - exp(-Cr*(1 - exp(-NTU)))]

    When Cr = 1 (balanced flow), this reduces to ε = NTU/(1+NTU).

    Args:
        NTU: Number of transfer units (UA/Cmin)
        Cr: Heat capacity ratio (Cmin/Cmax), in range [0, 1]

    Returns:
        Effectiveness in range (0, 1)
    """
    NTU_safe = jnp.maximum(NTU, 1e-10)
    Cr_safe = jnp.maximum(Cr, 1e-10)

    # General formula: ε = (1/Cr) * [1 - exp(-Cr*(1 - exp(-NTU)))]
    inner_exp = jnp.exp(-NTU_safe)
    outer_exp = jnp.exp(-Cr_safe * (1.0 - inner_exp))
    eps_general = (1.0 / Cr_safe) * (1.0 - outer_exp)

    # Balanced case (Cr = 1): ε = NTU / (1 + NTU)
    eps_balanced = NTU_safe / (1.0 + NTU_safe)

    # Smooth blending near Cr = 1
    threshold = CR_BLEND_WIDTH
    x = jnp.abs(1.0 - Cr)
    t = x / threshold
    blend = jnp.clip(3 * t**2 - 2 * t**3, 0.0, 1.0)
    blend = jnp.where(t > 1, 1.0, blend)

    eps = blend * eps_general + (1.0 - blend) * eps_balanced

    return jnp.clip(eps, 0.0, 1.0)


def effectiveness_crossflow_cmin_mixed(NTU: Array, Cr: Array) -> Array:
    """Effectiveness for cross-flow with Cmin mixed, Cmax unmixed.

    One fluid (with smaller heat capacity rate) flows through a single
    channel and can mix. The other fluid (larger heat capacity rate)
    flows through separate tubes/channels without mixing.

    Formula: ε = 1 - exp[-(1/Cr)*(1 - exp(-Cr*NTU))]

    When Cr = 1, this reduces to ε = NTU/(1+NTU).
    When Cr → 0, this approaches ε → 1 - exp(-NTU).

    Args:
        NTU: Number of transfer units (UA/Cmin)
        Cr: Heat capacity ratio (Cmin/Cmax), in range [0, 1]

    Returns:
        Effectiveness in range (0, 1)
    """
    NTU_safe = jnp.maximum(NTU, 1e-10)
    Cr_safe = jnp.maximum(Cr, 1e-10)

    # General formula: ε = 1 - exp[-(1/Cr)*(1 - exp(-Cr*NTU))]
    inner_exp = jnp.exp(-Cr_safe * NTU_safe)
    exponent = -(1.0 / Cr_safe) * (1.0 - inner_exp)
    eps_general = 1.0 - jnp.exp(exponent)

    # Balanced case (Cr = 1): ε = NTU / (1 + NTU)
    eps_balanced = NTU_safe / (1.0 + NTU_safe)

    # Smooth blending near Cr = 1
    threshold = CR_BLEND_WIDTH
    x = jnp.abs(1.0 - Cr)
    t = x / threshold
    blend = jnp.clip(3 * t**2 - 2 * t**3, 0.0, 1.0)
    blend = jnp.where(t > 1, 1.0, blend)

    eps = blend * eps_general + (1.0 - blend) * eps_balanced

    return jnp.clip(eps, 0.0, 1.0)


def effectiveness_crossflow_both_mixed(NTU: Array, Cr: Array) -> Array:
    """Effectiveness for cross-flow heat exchanger with both fluids mixed.

    Both fluids can mix freely in their flow direction while flowing
    perpendicular to each other. This configuration is less common in
    practice but can occur in some compact heat exchangers.

    Formula (implicit): 1/ε = 1/(1-exp(-NTU)) + Cr/(1-exp(-Cr*NTU)) - 1/NTU

    For numerical stability, we rearrange and use safe exponentials.

    Args:
        NTU: Number of transfer units (UA/Cmin)
        Cr: Heat capacity ratio (Cmin/Cmax), in range [0, 1]

    Returns:
        Effectiveness in range (0, 1)
    """
    NTU_safe = jnp.maximum(NTU, 1e-10)
    Cr_safe = jnp.maximum(Cr, 1e-10)

    # Formula: 1/ε = 1/(1-exp(-NTU)) + Cr/(1-exp(-Cr*NTU)) - 1/NTU
    # Rearrange for numerical stability
    exp_NTU = jnp.exp(-NTU_safe)
    exp_CrNTU = jnp.exp(-Cr_safe * NTU_safe)

    # Add small epsilon to denominators to avoid division by zero
    term1 = 1.0 / jnp.maximum(1.0 - exp_NTU, 1e-10)
    term2 = Cr_safe / jnp.maximum(1.0 - exp_CrNTU, 1e-10)
    term3 = 1.0 / NTU_safe

    inv_eps = term1 + term2 - term3

    # Ensure inv_eps is positive before taking reciprocal
    inv_eps_safe = jnp.maximum(inv_eps, 1.001)  # ε < 0.999

    eps = 1.0 / inv_eps_safe

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


@dataclass(repr=False)
class HeaterParams(ParamsMixin):
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

    def eo_residuals(
        self,
        inlets: list[Stream],
        outlets: list[Stream],
        **kwargs,
    ) -> Array:
        """Compute residuals for the EO solver.

        Residuals:
            F_out_i - F_in_i = 0        (n_species)
            T_out - T_expected = 0       (1)
            P_out - P_in = 0             (1)

        Args:
            inlets: [inlet_stream]
            outlets: [outlet_stream]
            **kwargs: Optional duty, T_out overrides

        Returns:
            Flat residual array, length n_species + 2
        """
        p = self.params
        inlet = inlets[0]
        outlet = outlets[0]

        inlet_flows = get_flows(inlet)
        outlet_flows = get_flows(outlet)
        species = get_species(inlet)

        # Material balance: flows pass through unchanged
        mat_resid = []
        for s in species:
            mat_resid.append(jnp.atleast_1d(outlet_flows[s] - inlet_flows[s]))

        # Temperature: compute expected outlet T
        duty = kwargs.get('duty', p.duty)
        T_out_spec = kwargs.get('T_out', p.T_out)

        Cp = p.Cp if p.Cp is not None else 75.0
        F_total = sum(inlet_flows.values())
        C = F_total * Cp

        if T_out_spec is not None:
            T_expected = jnp.asarray(T_out_spec)
        elif duty is not None:
            T_expected = inlet["T"] + jnp.asarray(duty) / C
        elif p.UA is not None and p.T_utility is not None:
            UA = jnp.asarray(p.UA)
            T_util = jnp.asarray(p.T_utility)
            NTU = UA / C
            effectiveness = 1.0 - jnp.exp(-NTU)
            Q = effectiveness * C * (T_util - inlet["T"])
            T_expected = inlet["T"] + Q / C
        else:
            T_expected = inlet["T"]

        T_resid = jnp.atleast_1d(outlet["T"] - T_expected)
        P_resid = jnp.atleast_1d(outlet["P"] - inlet["P"])

        return jnp.concatenate(mat_resid + [T_resid, P_resid])


@dataclass(repr=False)
class CoolerParams(ParamsMixin):
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

    def eo_residuals(
        self,
        inlets: list[Stream],
        outlets: list[Stream],
        **kwargs,
    ) -> Array:
        """Compute residuals for the EO solver.

        Residuals:
            F_out_i - F_in_i = 0        (n_species)
            T_out - T_expected = 0       (1)
            P_out - P_in = 0             (1)

        Args:
            inlets: [inlet_stream]
            outlets: [outlet_stream]
            **kwargs: Optional duty, T_out overrides

        Returns:
            Flat residual array, length n_species + 2
        """
        p = self.params
        inlet = inlets[0]
        outlet = outlets[0]

        inlet_flows = get_flows(inlet)
        outlet_flows = get_flows(outlet)
        species = get_species(inlet)

        # Material balance: flows pass through unchanged
        mat_resid = []
        for s in species:
            mat_resid.append(jnp.atleast_1d(outlet_flows[s] - inlet_flows[s]))

        # Temperature: compute expected outlet T
        duty = kwargs.get('duty', p.duty)
        T_out_spec = kwargs.get('T_out', p.T_out)

        Cp = p.Cp if p.Cp is not None else 75.0
        F_total = sum(inlet_flows.values())
        C = F_total * Cp

        if T_out_spec is not None:
            T_expected = jnp.asarray(T_out_spec)
        elif duty is not None:
            # Cooler: duty is positive = heat removed
            T_expected = inlet["T"] - jnp.asarray(duty) / C
        elif p.UA is not None and p.T_utility is not None:
            UA = jnp.asarray(p.UA)
            T_util = jnp.asarray(p.T_utility)
            NTU = UA / C
            effectiveness = 1.0 - jnp.exp(-NTU)
            Q = effectiveness * C * (inlet["T"] - T_util)
            T_expected = inlet["T"] - Q / C
        else:
            T_expected = inlet["T"]

        T_resid = jnp.atleast_1d(outlet["T"] - T_expected)
        P_resid = jnp.atleast_1d(outlet["P"] - inlet["P"])

        return jnp.concatenate(mat_resid + [T_resid, P_resid])


# =============================================================================
# Two-Stream Heat Exchangers
# =============================================================================


@dataclass(repr=False)
class HeatExchangerParams(ParamsMixin):
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
            "flow_arrangement": "counter_current",
            "NTU_very_high": NTU > 10.0,
            "Cr_near_one": Cr > 0.95,
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
            "flow_arrangement": "co_current",
            "NTU_very_high": NTU > 10.0,
            "Cr_near_one": Cr > 0.95,
        }

        return hot_outlet, cold_outlet, info


class CrossFlowHX:
    """Cross-flow heat exchanger with configurable mixing.

    In cross-flow heat exchangers, the two fluids flow perpendicular to
    each other. The effectiveness depends on whether each fluid can mix
    in its flow direction:

    - both_unmixed: Both fluids flow through separate channels (most common)
                    Examples: finned-tube HX, car radiators
    - cmax_mixed: Larger heat capacity stream is mixed, smaller is unmixed
                  Examples: one fluid in shell, other in tubes
    - cmin_mixed: Smaller heat capacity stream is mixed, larger is unmixed
    - both_mixed: Both fluids can mix (less common)
                  Examples: some compact heat exchangers

    Temperature Crossing:
        Same soft handling as CounterCurrentHX - calculation proceeds
        with a warning flag in the info dict.
    """

    def __init__(self, params: HeatExchangerParams, mixing: str = "both_unmixed"):
        """Initialize cross-flow heat exchanger.

        Args:
            params: Heat exchanger parameters
            mixing: Mixing configuration, one of:
                    - "both_unmixed" (default, most common)
                    - "cmax_mixed" (Cmax mixed, Cmin unmixed)
                    - "cmin_mixed" (Cmin mixed, Cmax unmixed)
                    - "both_mixed" (both streams mixed)
        """
        self.params = params
        if mixing not in ["both_unmixed", "cmax_mixed", "cmin_mixed", "both_mixed"]:
            raise ValueError(
                f"Invalid mixing configuration: {mixing}. "
                f"Must be one of: both_unmixed, cmax_mixed, cmin_mixed, both_mixed"
            )
        self.mixing = mixing

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
                - 'mixing': Mixing configuration used
                - 'T_hot_out': Hot outlet temperature (K)
                - 'T_cold_out': Cold outlet temperature (K)
                - 'temperature_crossing': True if T_hot_in < T_cold_in
                - 'driving_force': T_hot_in - T_cold_in (should be positive)
        """
        p = self.params

        # Get inlet temperatures
        T_hot_in = hot_inlet["T"]
        T_cold_in = cold_inlet["T"]

        # Temperature crossing check
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

        # Heat capacity rates (ensure positive)
        C_hot = jnp.maximum(F_hot * Cp_hot, 1e-10)
        C_cold = jnp.maximum(F_cold * Cp_cold, 1e-10)

        C_min = jnp.minimum(C_hot, C_cold)
        C_max = jnp.maximum(C_hot, C_cold)
        Cr = C_min / C_max

        # Get UA
        UA_val = UA if UA is not None else p.UA
        if UA_val is None:
            raise ValueError("UA must be specified")
        UA_val = jnp.asarray(UA_val)

        # NTU and effectiveness based on mixing configuration
        NTU = UA_val / C_min

        if self.mixing == "both_unmixed":
            eps = effectiveness_crossflow_both_unmixed(NTU, Cr)
        elif self.mixing == "cmax_mixed":
            eps = effectiveness_crossflow_cmax_mixed(NTU, Cr)
        elif self.mixing == "cmin_mixed":
            eps = effectiveness_crossflow_cmin_mixed(NTU, Cr)
        else:  # both_mixed
            eps = effectiveness_crossflow_both_mixed(NTU, Cr)

        # Maximum possible heat transfer
        Q_max = C_min * driving_force

        # Actual heat transfer
        Q = eps * Q_max

        # Outlet temperatures
        T_hot_out = T_hot_in - Q / C_hot
        T_cold_out = T_cold_in + Q / C_cold

        # Build outlet streams
        hot_outlet = dict(hot_inlet)
        hot_outlet["T"] = T_hot_out

        cold_outlet = dict(cold_inlet)
        cold_outlet["T"] = T_cold_out

        # LMTD (use counter-current approximation for cross-flow)
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in
        LMTD = log_mean_temperature_difference(dT1, dT2)

        info = {
            "Q": Q,
            "effectiveness": eps,
            "NTU": NTU,
            "Cr": Cr,
            "LMTD": LMTD,
            "mixing": self.mixing,
            "T_hot_in": T_hot_in,
            "T_hot_out": T_hot_out,
            "T_cold_in": T_cold_in,
            "T_cold_out": T_cold_out,
            "approach": jnp.minimum(jnp.abs(dT1), jnp.abs(dT2)),
            "driving_force": driving_force,
            "temperature_crossing": temperature_crossing,
            "flow_arrangement": "crossflow",
            "NTU_very_high": NTU > 10.0,
            "Cr_near_one": Cr > 0.95,
        }

        return hot_outlet, cold_outlet, info


# =============================================================================
# Shell-and-Tube Heat Exchanger with LMTD F-correction
# =============================================================================


def lmtd_correction_factor(
    R: Array,
    P: Array,
    n_shell_passes: int = 1,
) -> Array:
    """LMTD correction factor for multi-pass shell-and-tube heat exchangers.

    Computes the F-factor for 1-2N TEMA shell-and-tube configurations.

    F = sqrt(R^2 + 1) * ln((1-P)/(1-R*P)) / ((R-1) * ln((2-P*(R+1-sqrt(R^2+1))) / (2-P*(R+1+sqrt(R^2+1)))))

    For R = 1 (balanced flow), uses L'Hôpital's limit:
    F = (P * sqrt(2)) / ((1-P) * ln((2-P*(2-sqrt(2))) / (2-P*(2+sqrt(2)))))

    Args:
        R: Heat capacity ratio (Th_in - Th_out) / (Tc_out - Tc_in)
        P: Temperature effectiveness (Tc_out - Tc_in) / (Th_in - Tc_in)
        n_shell_passes: Number of shell passes (default 1)

    Returns:
        F-correction factor (0 < F <= 1)
    """
    R = jnp.asarray(R, dtype=jnp.float64)
    P = jnp.asarray(P, dtype=jnp.float64)

    # For multi-shell-pass, adjust P
    # P_eff for n shell passes: P_1 = ((1 - (R*P - 1)/(P - 1))^(1/n) - 1) / ...
    # Simplified: use single-shell formula with adjusted P for n > 1
    if n_shell_passes > 1:
        # Adjust P for multiple shell passes
        # P_n to P_1 conversion
        ratio = safe_divide(1.0 - R * P, 1.0 - P)
        ratio_n = ratio ** (1.0 / n_shell_passes)
        P = safe_divide(1.0 - ratio_n, R - ratio_n)

    # Clamp P to avoid singularities at boundaries
    P = jnp.clip(P, 1e-6, 1.0 - 1e-6)

    sqrt_R2_1 = jnp.sqrt(R**2 + 1.0)

    # R = 1 singularity: use smooth blending
    # General formula numerator: sqrt(R^2+1) * ln((1-P)/(1-R*P))
    # General formula denominator: (R-1) * ln(A_minus / A_plus)
    A_minus = 2.0 - P * (R + 1.0 - sqrt_R2_1)
    A_plus = 2.0 - P * (R + 1.0 + sqrt_R2_1)

    # Ensure positive log arguments
    A_minus_safe = jnp.maximum(A_minus, 1e-10)
    A_plus_safe = jnp.maximum(A_plus, 1e-10)

    numer = sqrt_R2_1 * jnp.log(jnp.maximum(safe_divide(1.0 - P, 1.0 - R * P), 1e-10))
    denom = (R - 1.0) * jnp.log(safe_divide(A_minus_safe, A_plus_safe))

    # General case
    F_general = safe_divide(numer, denom)

    # R = 1 case: use limit formula
    # F = P*sqrt(2) / ((1-P) * ln((2-P*(2-sqrt(2)))/(2-P*(2+sqrt(2)))))
    sqrt2 = jnp.sqrt(2.0)
    A1 = 2.0 - P * (2.0 - sqrt2)
    A2 = 2.0 - P * (2.0 + sqrt2)
    A1_safe = jnp.maximum(A1, 1e-10)
    A2_safe = jnp.maximum(A2, 1e-10)
    F_balanced = safe_divide(
        P * sqrt2,
        (1.0 - P) * jnp.log(safe_divide(A1_safe, A2_safe))
    )

    # Smooth blend near R = 1
    blend_width = 0.05
    t = jnp.abs(R - 1.0) / blend_width
    blend = jnp.clip(3 * t**2 - 2 * t**3, 0.0, 1.0)

    F = blend * F_general + (1.0 - blend) * F_balanced

    return jnp.clip(F, 0.0, 1.0)


@dataclass(repr=False)
class ShellAndTubeHXParams(ParamsMixin):
    """Parameters for shell-and-tube heat exchanger with F-correction.

    Attributes:
        UA: Overall heat transfer coefficient × area (W/K)
        Cp_hot: Hot side heat capacity (J/mol·K). If None, uses default.
        Cp_cold: Cold side heat capacity (J/mol·K). If None, uses default.
        min_approach: Minimum temperature approach (K).
        n_shell_passes: Number of shell passes (default 1)
    """
    UA: Array | float
    Cp_hot: float | None = None
    Cp_cold: float | None = None
    min_approach: float = 10.0
    n_shell_passes: int = 1


class ShellAndTubeHX:
    """Shell-and-tube heat exchanger with LMTD F-correction.

    Models a 1-2N TEMA shell-and-tube heat exchanger where the
    effective LMTD is reduced by the F-correction factor.

    Q = UA * F * LMTD_counter_current

    The F-factor accounts for the reduced effectiveness of multi-pass
    arrangements compared to pure counter-current flow.

    All calculations are fully differentiable.
    """

    def __init__(self, params: ShellAndTubeHXParams):
        self.params = params

    def __call__(
        self,
        hot_inlet: Stream,
        cold_inlet: Stream,
        UA: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform shell-and-tube heat exchanger calculation.

        Args:
            hot_inlet: Hot stream inlet
            cold_inlet: Cold stream inlet
            UA: Override UA value (W/K)

        Returns:
            hot_outlet: Hot stream outlet
            cold_outlet: Cold stream outlet
            info: Dictionary with Q, F_correction, R, P_param, etc.
        """
        p = self.params

        T_hot_in = hot_inlet["T"]
        T_cold_in = cold_inlet["T"]

        hot_flows = get_flows(hot_inlet)
        cold_flows = get_flows(cold_inlet)
        F_hot = sum(hot_flows.values())
        F_cold = sum(cold_flows.values())

        Cp_hot = p.Cp_hot if p.Cp_hot is not None else 75.0
        Cp_cold = p.Cp_cold if p.Cp_cold is not None else 75.0

        C_hot = jnp.maximum(F_hot * Cp_hot, 1e-10)
        C_cold = jnp.maximum(F_cold * Cp_cold, 1e-10)

        C_min = jnp.minimum(C_hot, C_cold)
        C_max = jnp.maximum(C_hot, C_cold)
        Cr = C_min / C_max

        UA_val = UA if UA is not None else p.UA
        UA_val = jnp.asarray(UA_val)

        # Use effectiveness-NTU for counter-current to get Q
        NTU = UA_val / C_min
        eps_cc = effectiveness_counter_current(NTU, Cr)

        driving_force = T_hot_in - T_cold_in
        Q_max = C_min * driving_force

        # Compute outlet temperatures from counter-current effectiveness
        T_hot_out_cc = T_hot_in - eps_cc * Q_max / C_hot
        T_cold_out_cc = T_cold_in + eps_cc * Q_max / C_cold

        # Compute R and P for F-correction
        dT_hot = T_hot_in - T_hot_out_cc
        dT_cold = T_cold_out_cc - T_cold_in
        R_param = safe_divide(dT_hot, jnp.maximum(dT_cold, 1e-10))
        P_param = safe_divide(dT_cold, jnp.maximum(T_hot_in - T_cold_in, 1e-10))

        # Compute F-correction factor
        F_corr = lmtd_correction_factor(R_param, P_param, p.n_shell_passes)

        # Apply F-correction: effective Q = F * Q_counter_current
        Q = F_corr * eps_cc * Q_max

        # Outlet temperatures
        T_hot_out = T_hot_in - Q / C_hot
        T_cold_out = T_cold_in + Q / C_cold

        hot_outlet = dict(hot_inlet)
        hot_outlet["T"] = T_hot_out

        cold_outlet = dict(cold_inlet)
        cold_outlet["T"] = T_cold_out

        # LMTD (counter-current basis)
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in
        LMTD = log_mean_temperature_difference(dT1, dT2)

        info = {
            "Q": Q,
            "F_correction": F_corr,
            "R": R_param,
            "P_param": P_param,
            "LMTD": LMTD,
            "effectiveness": eps_cc * F_corr,
            "NTU": NTU,
            "Cr": Cr,
            "T_hot_in": T_hot_in,
            "T_hot_out": T_hot_out,
            "T_cold_in": T_cold_in,
            "T_cold_out": T_cold_out,
            "flow_arrangement": "shell_and_tube",
            "F_too_low": F_corr < 0.75,
            "n_shell_passes": p.n_shell_passes,
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
    eps = jnp.clip(safe_divide(Q, Q_max), 0.0, 0.9999)

    # Invert effectiveness-NTU relationship with smooth blending for Cr → 1
    if flow_config == "counter_current":
        # Balanced case (Cr = 1): NTU = eps / (1 - eps)
        NTU_balanced = safe_divide(eps, 1.0 - eps)

        # General case: NTU = ln((1 - eps*Cr) / (1 - eps)) / (1 - Cr)
        one_minus_Cr = jnp.maximum(1.0 - Cr, EPS_DIVISION)
        # Ensure arguments to log are positive
        log_arg = jnp.maximum(safe_divide(1.0 - eps * Cr, 1.0 - eps), EPS_DIVISION)
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
