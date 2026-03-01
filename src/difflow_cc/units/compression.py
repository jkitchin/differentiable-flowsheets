"""CO2 compression for transport and storage.

This module provides multi-stage compression models for
bringing captured CO2 to pipeline or storage conditions.

Typical targets:
- Pipeline transport: 100-150 bar
- Geological storage: 100-200 bar
- Ship transport: 15 bar (liquefied at -30°C)

All models are JAX-compatible for automatic differentiation.

References:
    IEAGHG (2011). Rotating Equipment for Carbon Dioxide Capture
        and Storage. Report 2011/07.
    McCollum DL, Ogden JM (2006). Techno-Economic Models for
        Carbon Dioxide Compression, Transport, and Storage.
        UCD-ITS-RR-06-14.
"""

__all__ = [
    "CompressorParams",
    "Compressor",
    "CompressionTrainParams",
    "CompressionTrain",
    "Pump",
    "compression_power_estimate",
    "co2_density",
    "is_supercritical",
]

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow.params_mixin import ParamsMixin
from difflow.numerics import safe_divide
from difflow.eos import PengRobinson, CriticalProperties


# Gas constant
R = 8.314  # J/(mol*K)

# CO2 properties
MW_CO2 = 44.01  # g/mol
T_CRIT_CO2 = 304.13  # K
P_CRIT_CO2 = 7.377e6  # Pa
OMEGA_CO2 = 0.225  # Acentric factor for CO2

# Peng-Robinson EOS instance for pure CO2
_CO2_EOS = PengRobinson({
    "CO2": CriticalProperties(
        name="CO2",
        Tc=T_CRIT_CO2,
        Pc=P_CRIT_CO2,
        omega=OMEGA_CO2,
        MW=MW_CO2,
    )
})
# Mole fraction array for pure CO2
_Y_CO2 = jnp.array([1.0])


# =============================================================================
# Compression Parameters
# =============================================================================

@dataclass(repr=False)
class CompressorParams(ParamsMixin):
    """Parameters for a single compressor stage.

    Attributes:
        pressure_ratio: Compression ratio per stage
        eta_isentropic: Isentropic efficiency (0-1)
        eta_mechanical: Mechanical efficiency (0-1)
        eta_motor: Motor/driver efficiency (0-1)
    """
    pressure_ratio: float | Array = 3.0
    eta_isentropic: float | Array = 0.80
    eta_mechanical: float | Array = 0.98
    eta_motor: float | Array = 0.95


@dataclass(repr=False)
class CompressionIntercoolerParams(ParamsMixin):
    """Parameters for interstage cooler in compression train.

    Attributes:
        T_outlet: Target outlet temperature (K)
        T_coolant: Cooling water temperature (K)
        approach: Minimum approach temperature (K)
        pressure_drop: Pressure drop (Pa)
    """
    T_outlet: float | Array = 313.15  # K (40°C)
    T_coolant: float | Array = 298.15  # K (25°C)
    approach: float | Array = 5.0  # K
    pressure_drop: float = 5000.0  # Pa


@dataclass(repr=False)
class CompressionTrainParams(ParamsMixin):
    """Parameters for multi-stage compression train.

    Attributes:
        P_inlet: Inlet pressure (Pa)
        P_outlet: Target outlet pressure (Pa)
        n_stages: Number of compression stages (None = auto-calculate)
        max_pressure_ratio: Maximum pressure ratio per stage
        eta_isentropic: Isentropic efficiency per stage
        eta_mechanical: Mechanical efficiency
        eta_motor: Motor efficiency
        T_intercool: Intercooler outlet temperature (K)
        use_pump: Use pump for supercritical stage
        pump_efficiency: Pump efficiency
    """
    P_inlet: float | Array = 200000.0  # Pa (2 bar)
    P_outlet: float | Array = 15000000.0  # Pa (150 bar)
    n_stages: int | None = None
    max_pressure_ratio: float = 3.5
    eta_isentropic: float | Array = 0.80
    eta_mechanical: float | Array = 0.98
    eta_motor: float | Array = 0.95
    T_intercool: float | Array = 313.15  # K (40°C)
    use_pump: bool = True
    pump_efficiency: float | Array = 0.75


# =============================================================================
# Thermodynamic Functions
# =============================================================================

def co2_gamma(T: Array | float, P: Array | float) -> Array:
    """Heat capacity ratio (Cp/Cv) for CO2.

    Uses correlation valid for gaseous CO2.
    For more accuracy near critical point, use EOS.

    Args:
        T: Temperature (K)
        P: Pressure (Pa)

    Returns:
        Heat capacity ratio gamma
    """
    T = jnp.asarray(T)
    P = jnp.asarray(P)

    # Ideal gas gamma for CO2 is ~1.3
    # Decreases approaching critical point
    T_r = T / T_CRIT_CO2
    P_r = P / P_CRIT_CO2

    # Simplified correlation
    gamma_ideal = 1.289

    # Correction for real gas effects (simplified)
    correction = 1.0 - 0.1 * P_r / (T_r + 0.5)
    gamma = gamma_ideal * jnp.clip(correction, 0.8, 1.0)

    return gamma


def co2_compressibility(T: Array | float, P: Array | float) -> Array:
    """Compressibility factor Z for CO2.

    Uses the Peng-Robinson equation of state via difflow.eos.PengRobinson.

    The PR cubic equation solved is:
        Z^3 - (1-B)*Z^2 + (A - 3B^2 - 2B)*Z - (AB - B^2 - B^3) = 0
    where:
        A = a(T)*P / (R*T)^2
        B = b*P / (R*T)
        a(T) = a_c * [1 + kappa*(1 - sqrt(T/Tc))]^2
        kappa = 0.37464 + 1.54226*omega - 0.26992*omega^2
        a_c = 0.45724 * R^2 * Tc^2 / Pc
        b   = 0.07780 * R * Tc / Pc

    CO2 critical properties: Tc=304.13 K, Pc=7.377e6 Pa, omega=0.225.

    Args:
        T: Temperature (K)
        P: Pressure (Pa)

    Returns:
        Compressibility factor Z
    """
    T = jnp.asarray(T, dtype=float)
    P = jnp.asarray(P, dtype=float)

    # Determine phase: use vapor root above critical temperature or at low
    # reduced pressure; use liquid root otherwise.
    T_r = T / T_CRIT_CO2
    P_r = P / P_CRIT_CO2
    is_vapor = (T_r >= 1.0) | (P_r < 1.0)

    # solve_Z requires string literals for phase, so compute both roots and
    # select the appropriate one with jnp.where (JAX-compatible).
    Z_vap = _CO2_EOS.solve_Z(T, P, _Y_CO2, phase="vapor")
    Z_liq = _CO2_EOS.solve_Z(T, P, _Y_CO2, phase="liquid")

    Z = jnp.where(is_vapor, Z_vap, Z_liq)

    return Z


def co2_density(T: Array | float, P: Array | float) -> Array:
    """CO2 density using compressibility factor.

    Args:
        T: Temperature (K)
        P: Pressure (Pa)

    Returns:
        Density (kg/m³)
    """
    T = jnp.asarray(T)
    P = jnp.asarray(P)

    Z = co2_compressibility(T, P)
    rho = P * MW_CO2 / (Z * R * T * 1000)  # kg/m³

    return rho


def is_supercritical(T: Array | float, P: Array | float) -> Array:
    """Check if CO2 is in supercritical state.

    Args:
        T: Temperature (K)
        P: Pressure (Pa)

    Returns:
        Boolean array (True if supercritical)
    """
    T = jnp.asarray(T)
    P = jnp.asarray(P)

    return (T > T_CRIT_CO2) & (P > P_CRIT_CO2)


# =============================================================================
# Compressor Models
# =============================================================================

class Compressor:
    """Single-stage compressor.

    Models isentropic compression with efficiency correction.

    Example:
        >>> params = CompressorParams(pressure_ratio=3.0, eta_isentropic=0.80)
        >>> comp = Compressor(params)
        >>> outlet, info = comp(inlet)
    """

    def __init__(self, params: CompressorParams):
        self.params = params

    def __call__(
        self,
        inlet: Stream,
    ) -> tuple[Stream, dict]:
        """Compress the gas stream.

        Args:
            inlet: Inlet gas stream

        Returns:
            outlet: Compressed gas stream
            info: Work, temperatures, efficiency
        """
        p = self.params

        T_in = jnp.asarray(inlet["T"])
        P_in = jnp.asarray(inlet["P"])
        F = total_flow(inlet)  # mol/s

        pr = jnp.asarray(p.pressure_ratio)
        eta_is = jnp.asarray(p.eta_isentropic)
        eta_mech = jnp.asarray(p.eta_mechanical)
        eta_motor = jnp.asarray(p.eta_motor)

        P_out = P_in * pr

        # Get gamma at inlet conditions
        gamma = co2_gamma(T_in, P_in)

        # Isentropic outlet temperature
        T_out_is = T_in * jnp.power(pr, (gamma - 1) / gamma)

        # Actual outlet temperature (with efficiency)
        T_out = T_in + (T_out_is - T_in) / eta_is

        # Compressibility factor
        Z_in = co2_compressibility(T_in, P_in)
        Z_out = co2_compressibility(T_out, P_out)
        Z_avg = (Z_in + Z_out) / 2

        # Isentropic work (J/mol) - use inlet Z per standard formula
        W_is = Z_in * R * T_in * gamma / (gamma - 1) * \
               (jnp.power(pr, (gamma - 1) / gamma) - 1)

        # Actual work per mol (J/mol)
        W_actual = W_is / eta_is

        # Shaft power (W)
        P_shaft = F * W_actual / eta_mech

        # Electrical power (W)
        P_elec = P_shaft / eta_motor

        # Create outlet stream
        outlet = make_stream(get_flows(inlet), T_out, P_out)

        info = {
            "T_in": T_in,
            "T_out": T_out,
            "T_out_isentropic": T_out_is,
            "P_in": P_in,
            "P_out": P_out,
            "pressure_ratio": pr,
            "W_isentropic": W_is * F,  # W
            "W_actual": W_actual * F,  # W
            "P_shaft": P_shaft,  # W
            "P_electrical": P_elec,  # W
            "eta_isentropic": eta_is,
            "gamma": gamma,
            "Z_avg": Z_avg,
        }

        return outlet, info


class Intercooler:
    """Interstage cooler for compression train.

    Cools compressed gas to reduce work in subsequent stages.
    """

    def __init__(self, params: CompressionIntercoolerParams):
        self.params = params

    def __call__(
        self,
        inlet: Stream,
    ) -> tuple[Stream, dict]:
        """Cool the gas stream.

        Args:
            inlet: Hot gas from compressor

        Returns:
            outlet: Cooled gas
            info: Heat duty, temperatures
        """
        p = self.params

        T_in = jnp.asarray(inlet["T"])
        P_in = jnp.asarray(inlet["P"])
        F = total_flow(inlet)

        # Target temperature
        T_target = jnp.asarray(p.T_outlet)
        T_coolant = jnp.asarray(p.T_coolant)
        approach = jnp.asarray(p.approach)

        # Limit by approach to coolant
        T_out = jnp.maximum(T_target, T_coolant + approach)
        T_out = jnp.minimum(T_out, T_in)  # Can't heat

        # Pressure drop
        P_out = P_in - p.pressure_drop

        # Cooling duty (approximate Cp for CO2)
        # Cp varies with T, P; use average value
        Cp_avg = 40.0  # J/(mol·K), approximate for CO2
        Q = F * Cp_avg * (T_in - T_out)

        outlet = make_stream(get_flows(inlet), T_out, P_out)

        info = {
            "T_in": T_in,
            "T_out": T_out,
            "Q": Q,
            "cooling_water_flow": Q / (4186 * 10),  # kg/s, 10K rise
        }

        return outlet, info


class Pump:
    """Pump for supercritical/liquid CO2.

    More efficient than compression for dense-phase CO2.
    """

    def __init__(self, P_outlet: float | Array, eta: float | Array = 0.75):
        self.P_outlet = P_outlet
        self.eta = eta

    def __call__(
        self,
        inlet: Stream,
    ) -> tuple[Stream, dict]:
        """Pump dense-phase CO2.

        Args:
            inlet: Dense-phase CO2 stream

        Returns:
            outlet: Pressurized stream
            info: Work and conditions
        """
        T_in = jnp.asarray(inlet["T"])
        P_in = jnp.asarray(inlet["P"])
        P_out = jnp.asarray(self.P_outlet)
        eta = jnp.asarray(self.eta)
        F = total_flow(inlet)

        # Density at inlet
        rho = co2_density(T_in, P_in)  # kg/m³

        # Volumetric flow
        m_dot = F * MW_CO2 / 1000  # kg/s
        V_dot = m_dot / rho  # m³/s

        # Hydraulic power (incompressible approximation)
        dP = P_out - P_in
        W_hydraulic = V_dot * dP  # W

        # Actual power
        W_actual = W_hydraulic / eta

        # Small temperature rise from inefficiency
        Cp = 40.0  # J/(mol·K)
        T_out = T_in + (W_actual - W_hydraulic) / (F * Cp)

        outlet = make_stream(get_flows(inlet), T_out, P_out)

        info = {
            "P_in": P_in,
            "P_out": P_out,
            "T_in": T_in,
            "T_out": T_out,
            "W_hydraulic": W_hydraulic,
            "W_actual": W_actual,
            "rho": rho,
            "eta": eta,
        }

        return outlet, info


class CompressionTrain:
    """Multi-stage CO2 compression train with intercooling.

    Automatically sizes number of stages based on pressure ratio.
    Uses pump for final stage if CO2 becomes supercritical.

    Example:
        >>> params = CompressionTrainParams(
        ...     P_inlet=200000,    # 2 bar
        ...     P_outlet=15000000,  # 150 bar
        ... )
        >>> train = CompressionTrain(params)
        >>> outlet, info = train(co2_stream)
    """

    def __init__(self, params: CompressionTrainParams):
        self.params = params

    def __call__(
        self,
        inlet: Stream,
    ) -> tuple[Stream, dict]:
        """Compress CO2 to target pressure.

        Args:
            inlet: CO2 stream from capture unit

        Returns:
            outlet: Compressed CO2 at target pressure
            info: Stage-by-stage details, total power
        """
        p = self.params

        P_in = jnp.asarray(p.P_inlet)
        P_out_target = jnp.asarray(p.P_outlet)

        # Calculate number of stages
        total_ratio = P_out_target / P_in
        max_pr = p.max_pressure_ratio

        if p.n_stages is not None:
            n_stages = p.n_stages
        else:
            # Calculate minimum stages needed
            n_stages = max(1, int(float(jnp.ceil(jnp.log(total_ratio) / jnp.log(max_pr)))))

        # Equal pressure ratio per stage
        pr_per_stage = jnp.power(total_ratio, 1.0 / n_stages)

        # Initialize
        stream = inlet
        total_power = 0.0
        total_cooling = 0.0
        stage_info = []

        comp_params = CompressorParams(
            pressure_ratio=pr_per_stage,
            eta_isentropic=p.eta_isentropic,
            eta_mechanical=p.eta_mechanical,
            eta_motor=p.eta_motor,
        )
        compressor = Compressor(comp_params)

        cool_params = CompressionIntercoolerParams(
            T_outlet=p.T_intercool,
            T_coolant=p.T_intercool - 15,  # Assume 15K approach
        )
        cooler = Intercooler(cool_params)

        # Compression stages
        for i in range(n_stages):
            # Check if we've reached supercritical and should switch to pump
            T_current = jnp.asarray(stream["T"])
            P_current = jnp.asarray(stream["P"])

            if p.use_pump and float(P_current) > float(P_CRIT_CO2) and \
               float(T_current) < float(T_CRIT_CO2 + 20):
                # Use pump for remaining pressure increase
                pump = Pump(P_out_target, eta=p.pump_efficiency)
                stream, pump_info = pump(stream)
                total_power += pump_info["W_actual"]
                stage_info.append({
                    "stage": i + 1,
                    "type": "pump",
                    **pump_info,
                })
                break

            # Compress
            stream, comp_info = compressor(stream)
            total_power += comp_info["P_electrical"]

            # Intercool (except last stage)
            if i < n_stages - 1:
                stream, cool_info = cooler(stream)
                total_cooling += cool_info["Q"]
                stage_info.append({
                    "stage": i + 1,
                    "type": "compressor",
                    "P_electrical": comp_info["P_electrical"],
                    "T_out_comp": comp_info["T_out"],
                    "T_out_cool": cool_info["T_out"],
                    "P_out": comp_info["P_out"],
                })
            else:
                stage_info.append({
                    "stage": i + 1,
                    "type": "compressor",
                    "P_electrical": comp_info["P_electrical"],
                    "T_out": comp_info["T_out"],
                    "P_out": comp_info["P_out"],
                })

        # Final outlet conditions
        F = total_flow(inlet)
        m_dot_co2 = F * MW_CO2 / 1000  # kg/s

        # Specific power (kWh/tonne CO2)
        # W / (kg/s) = J/kg; J/kg * (1 kWh / 3.6e6 J) * (1000 kg/tonne) = J/kg / 3600
        specific_power = safe_divide(total_power, m_dot_co2) / 3600  # J/kg → kWh/tonne

        info = {
            "n_stages": n_stages,
            "pressure_ratio_per_stage": pr_per_stage,
            "total_power": total_power,  # W
            "total_cooling": total_cooling,  # W
            "specific_power": safe_divide(total_power, m_dot_co2 * 1000),  # kJ/kg
            "specific_power_kWh_tonne": specific_power,
            "P_inlet": P_in,
            "P_outlet": stream["P"],
            "T_inlet": inlet["T"],
            "T_outlet": stream["T"],
            "stages": stage_info,
        }

        return stream, info


# =============================================================================
# Utility Functions
# =============================================================================

def compression_power_estimate(
    F_CO2: Array | float,
    P_in: Array | float,
    P_out: Array | float,
    T_in: Array | float = 313.15,
    eta: float = 0.80,
) -> Array:
    """Quick estimate of compression power.

    Args:
        F_CO2: CO2 molar flow (mol/s)
        P_in: Inlet pressure (Pa)
        P_out: Outlet pressure (Pa)
        T_in: Inlet temperature (K)
        eta: Overall efficiency

    Returns:
        Electrical power (W)
    """
    F_CO2 = jnp.asarray(F_CO2)
    P_in = jnp.asarray(P_in)
    P_out = jnp.asarray(P_out)
    T_in = jnp.asarray(T_in)

    # Number of stages (estimate)
    ratio = P_out / P_in
    n_stages = jnp.ceil(jnp.log(ratio) / jnp.log(3.0))

    # Multi-stage compression with intercooling back to T_in
    gamma = 1.3
    k = (gamma - 1) / gamma

    # Per-stage work with intercooling (equal pressure ratios)
    pr_per_stage = jnp.power(ratio, 1.0 / n_stages)
    W_per_stage = R * T_in / k * (jnp.power(pr_per_stage, k) - 1)
    W_actual = n_stages * W_per_stage

    # Power with efficiency
    P_elec = F_CO2 * W_actual / eta

    return P_elec
