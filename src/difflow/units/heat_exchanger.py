"""Heat exchanger unit operations.

This module provides differentiable heat exchanger models:
- Heater: Single stream heating with utility
- Cooler: Single stream cooling with utility
- CounterCurrentHX: Two-stream counter-current heat exchanger
- CoCurrentHX: Two-stream co-current (parallel flow) heat exchanger

All models use either LMTD or effectiveness-NTU methods and are
fully differentiable for gradient-based optimization.
"""

from dataclasses import dataclass
from typing import Callable
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, get_species


# =============================================================================
# Utility Functions
# =============================================================================


def log_mean_temperature_difference(
    dT1: Array,
    dT2: Array,
    eps: float = 1e-8,
) -> Array:
    """Compute log mean temperature difference (LMTD).

    LMTD = (dT1 - dT2) / ln(dT1/dT2)

    Uses a numerically stable formulation that handles dT1 ≈ dT2.

    Args:
        dT1: Temperature difference at one end (K)
        dT2: Temperature difference at other end (K)
        eps: Small value for numerical stability

    Returns:
        Log mean temperature difference (K)
    """
    # Avoid division by zero and log(0)
    dT1 = jnp.maximum(dT1, eps)
    dT2 = jnp.maximum(dT2, eps)

    # When dT1 ≈ dT2, LMTD ≈ dT1 (use arithmetic mean as fallback)
    ratio = dT1 / dT2
    lmtd = jnp.where(
        jnp.abs(ratio - 1.0) < 0.01,
        (dT1 + dT2) / 2.0,  # Arithmetic mean for nearly equal
        (dT1 - dT2) / jnp.log(ratio),
    )
    return lmtd


def effectiveness_counter_current(NTU: Array, Cr: Array) -> Array:
    """Effectiveness for counter-current heat exchanger.

    Args:
        NTU: Number of transfer units (UA/Cmin)
        Cr: Heat capacity ratio (Cmin/Cmax)

    Returns:
        Effectiveness (0-1)
    """
    # Special case: Cr = 1 (balanced heat exchanger)
    # ε = NTU / (1 + NTU)
    eps_balanced = NTU / (1.0 + NTU)

    # General case: Cr < 1
    # ε = (1 - exp(-NTU(1-Cr))) / (1 - Cr*exp(-NTU(1-Cr)))
    exp_term = jnp.exp(-NTU * (1.0 - Cr))
    eps_general = (1.0 - exp_term) / (1.0 - Cr * exp_term + 1e-10)

    return jnp.where(Cr > 0.99, eps_balanced, eps_general)


def effectiveness_co_current(NTU: Array, Cr: Array) -> Array:
    """Effectiveness for co-current (parallel flow) heat exchanger.

    Args:
        NTU: Number of transfer units (UA/Cmin)
        Cr: Heat capacity ratio (Cmin/Cmax)

    Returns:
        Effectiveness (0-1)
    """
    # ε = (1 - exp(-NTU(1+Cr))) / (1 + Cr)
    return (1.0 - jnp.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)


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


class CounterCurrentHX:
    """Counter-current (shell-and-tube style) heat exchanger.

    Hot fluid flows opposite to cold fluid direction, achieving
    higher effectiveness than co-current flow.

    Uses effectiveness-NTU method for fully differentiable calculations.
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
                - 'Q': Heat transferred (W)
                - 'effectiveness': Heat exchanger effectiveness
                - 'NTU': Number of transfer units
                - 'LMTD': Log mean temperature difference (K)
                - 'T_hot_out': Hot outlet temperature (K)
                - 'T_cold_out': Cold outlet temperature (K)
        """
        p = self.params

        # Get inlet temperatures
        T_hot_in = hot_inlet["T"]
        T_cold_in = cold_inlet["T"]

        # Get flow rates
        hot_flows = get_flows(hot_inlet)
        cold_flows = get_flows(cold_inlet)
        F_hot = sum(hot_flows.values())
        F_cold = sum(cold_flows.values())

        # Heat capacities
        Cp_hot = p.Cp_hot if p.Cp_hot is not None else 75.0
        Cp_cold = p.Cp_cold if p.Cp_cold is not None else 75.0

        # Heat capacity rates
        C_hot = F_hot * Cp_hot
        C_cold = F_cold * Cp_cold

        C_min = jnp.minimum(C_hot, C_cold)
        C_max = jnp.maximum(C_hot, C_cold)
        Cr = C_min / (C_max + 1e-10)

        # Get UA
        UA_val = UA if UA is not None else p.UA
        if UA_val is None:
            raise ValueError("UA must be specified")
        UA_val = jnp.asarray(UA_val)

        # NTU and effectiveness
        NTU = UA_val / C_min
        eps = effectiveness_counter_current(NTU, Cr)

        # Maximum possible heat transfer
        Q_max = C_min * (T_hot_in - T_cold_in)

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

        # LMTD (counter-current)
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
            "approach": jnp.minimum(dT1, dT2),
        }

        return hot_outlet, cold_outlet, info


class CoCurrentHX:
    """Co-current (parallel flow) heat exchanger.

    Both fluids flow in the same direction. Lower effectiveness
    than counter-current, but useful for some applications.
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

        C_hot = F_hot * Cp_hot
        C_cold = F_cold * Cp_cold

        C_min = jnp.minimum(C_hot, C_cold)
        C_max = jnp.maximum(C_hot, C_cold)
        Cr = C_min / (C_max + 1e-10)

        UA_val = UA if UA is not None else p.UA
        if UA_val is None:
            raise ValueError("UA must be specified")
        UA_val = jnp.asarray(UA_val)

        NTU = UA_val / C_min
        eps = effectiveness_co_current(NTU, Cr)

        Q_max = C_min * (T_hot_in - T_cold_in)
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
            "approach": dT2,  # Co-current approach is at outlets
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

    Args:
        Q: Heat duty (W)
        T_hot_in: Hot inlet temperature (K)
        T_cold_in: Cold inlet temperature (K)
        C_hot: Hot side heat capacity rate (W/K)
        C_cold: Cold side heat capacity rate (W/K)
        U: Overall heat transfer coefficient (W/m²·K)
        flow_config: "counter_current" or "co_current"

    Returns:
        Dictionary with A, UA, NTU, effectiveness
    """
    C_min = jnp.minimum(C_hot, C_cold)
    C_max = jnp.maximum(C_hot, C_cold)
    Cr = C_min / (C_max + 1e-10)

    Q_max = C_min * (T_hot_in - T_cold_in)
    eps = Q / Q_max

    # Invert effectiveness-NTU relationship
    if flow_config == "counter_current":
        # For counter-current:
        # eps = (1 - exp(-NTU(1-Cr))) / (1 - Cr*exp(-NTU(1-Cr)))
        # Solving for NTU:
        NTU = jnp.where(
            Cr > 0.99,
            eps / (1.0 - eps),  # Cr = 1 case
            jnp.log((1.0 - eps * Cr) / (1.0 - eps)) / (1.0 - Cr),
        )
    else:
        # For co-current:
        # eps = (1 - exp(-NTU(1+Cr))) / (1 + Cr)
        NTU = -jnp.log(1.0 - eps * (1.0 + Cr)) / (1.0 + Cr)

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
    }
