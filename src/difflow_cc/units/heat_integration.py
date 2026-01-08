"""Heat integration for carbon capture processes.

This module provides heat exchanger models for energy recovery
in amine-based carbon capture systems:
- Lean/rich heat exchanger (cross-exchanger)
- Intercoolers for absorber columns
- Trim coolers and heaters

All models are JAX-compatible for automatic differentiation.

References:
    Freguia S, Rochelle GT (2003). Modeling of CO2 capture by
        aqueous monoethanolamine. AIChE J 49:1676-1686.
    Abu-Zahra MRM et al. (2007). CO2 capture from power plants:
        Part I. Parametric study of the technical performance.
        Int J Greenh Gas Control 1:37-46.
"""

from dataclasses import dataclass, replace, fields, asdict
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, total_flow


# =============================================================================
# Heat Exchanger Parameters
# =============================================================================

@dataclass
class HeatExchangerParams:
    """Parameters for shell-and-tube heat exchanger.

    Attributes:
        U: Overall heat transfer coefficient (W/m²/K)
        A: Heat transfer area (m²)
        arrangement: 'counter' or 'parallel' flow
        min_approach: Minimum temperature approach (K)
        pressure_drop_hot: Pressure drop on hot side (Pa)
        pressure_drop_cold: Pressure drop on cold side (Pa)
    """
    U: float | Array = 500.0  # W/m²/K
    A: float | Array = 100.0  # m²
    arrangement: Literal["counter", "parallel"] = "counter"
    min_approach: float = 10.0  # K
    pressure_drop_hot: float = 5000.0  # Pa
    pressure_drop_cold: float = 5000.0  # Pa

    def update(self, **kwargs) -> "HeatExchangerParams":
        return replace(self, **kwargs)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def keys(self):
        return (f.name for f in fields(self))

    def items(self):
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def asdict(self) -> dict:
        return asdict(self)


@dataclass
class LeanRichExchangerParams:
    """Parameters for lean/rich solvent heat exchanger.

    The lean/rich exchanger recovers heat from hot lean solvent
    (from stripper) to preheat cold rich solvent (to stripper).

    Attributes:
        U: Overall heat transfer coefficient (W/m²/K)
        A: Heat transfer area (m²)
        effectiveness: Heat exchanger effectiveness (0-1)
        min_approach: Minimum temperature approach (K)
        Cp_solvent: Solvent heat capacity (J/mol/K)
    """
    U: float | Array = 800.0  # W/m²/K (liquid-liquid, high)
    A: float | Array = 200.0  # m²
    effectiveness: float | Array | None = None  # If set, overrides U*A
    min_approach: float = 10.0  # K
    Cp_solvent: float = 75.0  # J/mol/K (approximate for aqueous amine)

    def update(self, **kwargs) -> "LeanRichExchangerParams":
        return replace(self, **kwargs)


@dataclass
class IntercoolerParams:
    """Parameters for absorber intercooler.

    Intercooling removes heat of absorption to maintain
    favorable equilibrium in the absorber column.

    Attributes:
        T_coolant: Coolant temperature (K)
        duty: Fixed cooling duty (W), if None calculated from approach
        approach: Temperature approach to coolant (K)
        location: Fraction of column height from bottom (0-1)
    """
    T_coolant: float | Array = 298.15  # K (25°C cooling water)
    duty: float | Array | None = None
    approach: float = 5.0  # K
    location: float = 0.5  # Middle of column

    def update(self, **kwargs) -> "IntercoolerParams":
        return replace(self, **kwargs)


# =============================================================================
# Heat Exchanger Models
# =============================================================================

class HeatExchanger:
    """General shell-and-tube heat exchanger.

    Uses effectiveness-NTU method for counter-current flow.

    Example:
        >>> params = HeatExchangerParams(U=500, A=50)
        >>> hx = HeatExchanger(params)
        >>> hot_out, cold_out, info = hx(hot_in, cold_in)
    """

    def __init__(self, params: HeatExchangerParams):
        self.params = params

    def __call__(
        self,
        hot_in: Stream,
        cold_in: Stream,
        Cp_hot: float | Array = 75.0,
        Cp_cold: float | Array = 75.0,
    ) -> tuple[Stream, Stream, dict]:
        """Perform heat exchange.

        Args:
            hot_in: Hot stream inlet
            cold_in: Cold stream inlet
            Cp_hot: Heat capacity of hot stream (J/mol/K)
            Cp_cold: Heat capacity of cold stream (J/mol/K)

        Returns:
            hot_out: Hot stream outlet
            cold_out: Cold stream outlet
            info: Dict with Q, effectiveness, LMTD, etc.
        """
        p = self.params

        # Get temperatures and flows
        T_hot_in = jnp.asarray(hot_in.T)
        T_cold_in = jnp.asarray(cold_in.T)
        F_hot = total_flow(hot_in)
        F_cold = total_flow(cold_in)

        # Heat capacity rates (W/K)
        C_hot = F_hot * Cp_hot
        C_cold = F_cold * Cp_cold
        C_min = jnp.minimum(C_hot, C_cold)
        C_max = jnp.maximum(C_hot, C_cold)
        C_r = C_min / (C_max + 1e-10)

        # NTU
        U = jnp.asarray(p.U)
        A = jnp.asarray(p.A)
        NTU = U * A / (C_min + 1e-10)

        # Effectiveness (counter-current)
        if p.arrangement == "counter":
            # For C_r < 1
            effectiveness = jnp.where(
                jnp.abs(C_r - 1.0) < 1e-6,
                NTU / (1 + NTU),  # C_r = 1 case
                (1 - jnp.exp(-NTU * (1 - C_r))) /
                (1 - C_r * jnp.exp(-NTU * (1 - C_r)) + 1e-10)
            )
        else:
            # Parallel flow
            effectiveness = (1 - jnp.exp(-NTU * (1 + C_r))) / (1 + C_r + 1e-10)

        effectiveness = jnp.clip(effectiveness, 0.0, 0.99)

        # Maximum possible heat transfer
        Q_max = C_min * (T_hot_in - T_cold_in)

        # Actual heat transfer
        Q = effectiveness * Q_max

        # Outlet temperatures
        T_hot_out = T_hot_in - Q / (C_hot + 1e-10)
        T_cold_out = T_cold_in + Q / (C_cold + 1e-10)

        # Enforce minimum approach
        min_approach = jnp.asarray(p.min_approach)
        T_cold_out = jnp.minimum(T_cold_out, T_hot_in - min_approach)
        T_hot_out = jnp.maximum(T_hot_out, T_cold_in + min_approach)

        # Recalculate Q with constrained temperatures
        Q = C_hot * (T_hot_in - T_hot_out)

        # LMTD
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in
        LMTD = jnp.where(
            jnp.abs(dT1 - dT2) < 0.1,
            (dT1 + dT2) / 2,
            (dT1 - dT2) / (jnp.log(dT1 / (dT2 + 1e-10)) + 1e-10)
        )

        # Create outlet streams
        hot_flows = get_flows(hot_in)
        cold_flows = get_flows(cold_in)

        P_hot_out = jnp.asarray(hot_in.P) - p.pressure_drop_hot
        P_cold_out = jnp.asarray(cold_in.P) - p.pressure_drop_cold

        hot_out = make_stream(hot_flows, T_hot_out, P_hot_out)
        cold_out = make_stream(cold_flows, T_cold_out, P_cold_out)

        info = {
            "Q": Q,
            "effectiveness": effectiveness,
            "NTU": NTU,
            "LMTD": LMTD,
            "C_min": C_min,
            "C_max": C_max,
            "T_hot_in": T_hot_in,
            "T_hot_out": T_hot_out,
            "T_cold_in": T_cold_in,
            "T_cold_out": T_cold_out,
            "U": U,
            "A": A,
        }

        return hot_out, cold_out, info


class LeanRichExchanger:
    """Lean/rich solvent heat exchanger for amine systems.

    Recovers heat from hot lean solvent leaving the stripper
    to preheat rich solvent entering the stripper.

    This is the most important heat integration in amine systems,
    typically recovering 60-80% of sensible heat.

    Example:
        >>> params = LeanRichExchangerParams(effectiveness=0.85)
        >>> lrhx = LeanRichExchanger(params)
        >>> lean_out, rich_out, info = lrhx(lean_hot, rich_cold)
    """

    def __init__(self, params: LeanRichExchangerParams):
        self.params = params

    def __call__(
        self,
        lean_hot: Stream,
        rich_cold: Stream,
    ) -> tuple[Stream, Stream, dict]:
        """Exchange heat between lean and rich streams.

        Args:
            lean_hot: Hot lean solvent from stripper bottom
            rich_cold: Cold rich solvent from absorber bottom

        Returns:
            lean_cold: Cooled lean solvent (to absorber)
            rich_hot: Heated rich solvent (to stripper)
            info: Heat exchange details
        """
        p = self.params

        T_lean_in = jnp.asarray(lean_hot.T)
        T_rich_in = jnp.asarray(rich_cold.T)

        F_lean = total_flow(lean_hot)
        F_rich = total_flow(rich_cold)

        Cp = jnp.asarray(p.Cp_solvent)

        # Heat capacity rates
        C_lean = F_lean * Cp
        C_rich = F_rich * Cp
        C_min = jnp.minimum(C_lean, C_rich)
        C_max = jnp.maximum(C_lean, C_rich)

        # Determine effectiveness
        if p.effectiveness is not None:
            effectiveness = jnp.asarray(p.effectiveness)
        else:
            # Calculate from U*A
            U = jnp.asarray(p.U)
            A = jnp.asarray(p.A)
            NTU = U * A / (C_min + 1e-10)
            C_r = C_min / (C_max + 1e-10)
            effectiveness = (1 - jnp.exp(-NTU * (1 - C_r))) / \
                           (1 - C_r * jnp.exp(-NTU * (1 - C_r)) + 1e-10)
            effectiveness = jnp.clip(effectiveness, 0.0, 0.95)

        # Maximum heat transfer
        Q_max = C_min * (T_lean_in - T_rich_in)
        Q = effectiveness * Q_max

        # Outlet temperatures
        T_lean_out = T_lean_in - Q / (C_lean + 1e-10)
        T_rich_out = T_rich_in + Q / (C_rich + 1e-10)

        # Enforce minimum approach
        min_approach = jnp.asarray(p.min_approach)
        T_rich_out = jnp.minimum(T_rich_out, T_lean_in - min_approach)
        T_lean_out = jnp.maximum(T_lean_out, T_rich_in + min_approach)

        # Recalculate Q
        Q = C_lean * (T_lean_in - T_lean_out)

        # Heat recovery fraction (of stripper reboiler duty estimate)
        # Reboiler heats rich from T_rich_out to T_reboiler (~120°C)
        T_reboiler_est = 393.15  # K
        Q_reboiler_no_hx = C_rich * (T_reboiler_est - T_rich_in)
        Q_reboiler_with_hx = C_rich * (T_reboiler_est - T_rich_out)
        heat_recovery_fraction = (Q_reboiler_no_hx - Q_reboiler_with_hx) / \
                                  (Q_reboiler_no_hx + 1e-10)

        # Create output streams
        lean_flows = get_flows(lean_hot)
        rich_flows = get_flows(rich_cold)

        lean_cold = make_stream(lean_flows, T_lean_out, lean_hot.P)
        rich_hot = make_stream(rich_flows, T_rich_out, rich_cold.P)

        info = {
            "Q": Q,
            "effectiveness": effectiveness,
            "T_lean_in": T_lean_in,
            "T_lean_out": T_lean_out,
            "T_rich_in": T_rich_in,
            "T_rich_out": T_rich_out,
            "heat_recovery_fraction": heat_recovery_fraction,
            "C_lean": C_lean,
            "C_rich": C_rich,
        }

        return lean_cold, rich_hot, info


class Intercooler:
    """Absorber intercooler for temperature control.

    Removes heat of absorption to maintain favorable VLE
    and increase capture efficiency.

    The absorber temperature bulge occurs due to exothermic
    CO2-amine reaction. Intercooling can increase capacity
    by 5-15% in some cases.

    Example:
        >>> params = IntercoolerParams(T_coolant=298.15, approach=5.0)
        >>> cooler = Intercooler(params)
        >>> stream_out, info = cooler(stream_in)
    """

    def __init__(self, params: IntercoolerParams):
        self.params = params

    def __call__(
        self,
        stream_in: Stream,
        Cp: float | Array = 75.0,
    ) -> tuple[Stream, dict]:
        """Cool the process stream.

        Args:
            stream_in: Hot process stream
            Cp: Heat capacity (J/mol/K)

        Returns:
            stream_out: Cooled stream
            info: Cooling duty and temperatures
        """
        p = self.params

        T_in = jnp.asarray(stream_in.T)
        T_coolant = jnp.asarray(p.T_coolant)
        F = total_flow(stream_in)

        # Target outlet temperature
        approach = jnp.asarray(p.approach)
        T_target = T_coolant + approach

        # Only cool if stream is hotter than target
        T_out = jnp.minimum(T_in, T_target)

        # Cooling duty
        Cp = jnp.asarray(Cp)
        Q = F * Cp * (T_in - T_out)

        # Override with fixed duty if specified
        if p.duty is not None:
            Q = jnp.asarray(p.duty)
            T_out = T_in - Q / (F * Cp + 1e-10)
            T_out = jnp.maximum(T_out, T_coolant + approach)

        # Create output stream
        stream_out = make_stream(get_flows(stream_in), T_out, stream_in.P)

        info = {
            "Q": Q,
            "T_in": T_in,
            "T_out": T_out,
            "T_coolant": T_coolant,
            "cooling_water_flow": Q / (4186 * 10),  # Approximate, 10K rise
        }

        return stream_out, info


class TrimCooler:
    """Trim cooler for final temperature adjustment.

    Cools lean solvent to absorber inlet temperature
    after the lean/rich exchanger.
    """

    def __init__(self, T_target: float | Array, T_coolant: float | Array = 298.15):
        self.T_target = T_target
        self.T_coolant = T_coolant

    def __call__(
        self,
        stream_in: Stream,
        Cp: float | Array = 75.0,
    ) -> tuple[Stream, dict]:
        """Cool stream to target temperature."""
        T_in = jnp.asarray(stream_in.T)
        T_target = jnp.asarray(self.T_target)
        F = total_flow(stream_in)
        Cp = jnp.asarray(Cp)

        T_out = jnp.minimum(T_in, T_target)
        Q = F * Cp * (T_in - T_out)

        stream_out = make_stream(get_flows(stream_in), T_out, stream_in.P)

        info = {
            "Q": Q,
            "T_in": T_in,
            "T_out": T_out,
        }

        return stream_out, info


class TrimHeater:
    """Trim heater for temperature adjustment.

    Heats rich solvent to stripper inlet temperature
    if lean/rich exchanger doesn't achieve target.
    """

    def __init__(self, T_target: float | Array):
        self.T_target = T_target

    def __call__(
        self,
        stream_in: Stream,
        Cp: float | Array = 75.0,
    ) -> tuple[Stream, dict]:
        """Heat stream to target temperature."""
        T_in = jnp.asarray(stream_in.T)
        T_target = jnp.asarray(self.T_target)
        F = total_flow(stream_in)
        Cp = jnp.asarray(Cp)

        T_out = jnp.maximum(T_in, T_target)
        Q = F * Cp * (T_out - T_in)

        stream_out = make_stream(get_flows(stream_in), T_out, stream_in.P)

        info = {
            "Q": Q,
            "T_in": T_in,
            "T_out": T_out,
        }

        return stream_out, info


# =============================================================================
# Integrated Heat Recovery System
# =============================================================================

@dataclass
class HeatRecoverySystemParams:
    """Parameters for complete heat recovery system.

    Includes lean/rich exchanger, trim cooler, and optional intercooler.
    """
    # Lean/rich exchanger
    lrhx_effectiveness: float | Array = 0.85
    lrhx_min_approach: float = 10.0

    # Trim cooler
    T_lean_target: float | Array = 313.15  # K (40°C to absorber)
    T_coolant: float | Array = 298.15  # K (25°C cooling water)

    # Optional intercooler
    use_intercooler: bool = False
    intercooler_location: float = 0.5

    # Heat capacities
    Cp_solvent: float = 75.0  # J/mol/K


class HeatRecoverySystem:
    """Complete heat recovery system for amine capture.

    Combines lean/rich exchanger and trim cooler to minimize
    energy consumption.

    Example:
        >>> params = HeatRecoverySystemParams(lrhx_effectiveness=0.85)
        >>> hrs = HeatRecoverySystem(params)
        >>> lean_to_abs, rich_to_strip, info = hrs(lean_from_strip, rich_from_abs)
    """

    def __init__(self, params: HeatRecoverySystemParams):
        self.params = params

        # Create sub-units
        self.lrhx = LeanRichExchanger(LeanRichExchangerParams(
            effectiveness=params.lrhx_effectiveness,
            min_approach=params.lrhx_min_approach,
            Cp_solvent=params.Cp_solvent,
        ))

        self.trim_cooler = TrimCooler(
            T_target=params.T_lean_target,
            T_coolant=params.T_coolant,
        )

    def __call__(
        self,
        lean_from_stripper: Stream,
        rich_from_absorber: Stream,
    ) -> tuple[Stream, Stream, dict]:
        """Process streams through heat recovery system.

        Args:
            lean_from_stripper: Hot lean solvent (~120°C)
            rich_from_absorber: Cold rich solvent (~40-50°C)

        Returns:
            lean_to_absorber: Cooled lean solvent (~40°C)
            rich_to_stripper: Heated rich solvent (~100-110°C)
            info: Energy flows and temperatures
        """
        p = self.params

        # Lean/rich exchange
        lean_after_lrhx, rich_hot, lrhx_info = self.lrhx(
            lean_from_stripper,
            rich_from_absorber,
        )

        # Trim cooling for lean
        lean_to_absorber, cooler_info = self.trim_cooler(
            lean_after_lrhx,
            Cp=p.Cp_solvent,
        )

        # Compile info
        info = {
            # Lean/rich exchanger
            "Q_lrhx": lrhx_info["Q"],
            "lrhx_effectiveness": lrhx_info["effectiveness"],
            "heat_recovery_fraction": lrhx_info["heat_recovery_fraction"],

            # Trim cooler
            "Q_trim_cooler": cooler_info["Q"],

            # Temperatures
            "T_lean_from_stripper": lrhx_info["T_lean_in"],
            "T_lean_after_lrhx": lrhx_info["T_lean_out"],
            "T_lean_to_absorber": cooler_info["T_out"],
            "T_rich_from_absorber": lrhx_info["T_rich_in"],
            "T_rich_to_stripper": lrhx_info["T_rich_out"],

            # Total cooling duty
            "Q_total_cooling": cooler_info["Q"],
        }

        return lean_to_absorber, rich_hot, info
