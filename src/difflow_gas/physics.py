"""Gas physics: unit conversions, pipe/resistor coefficients, compressor power.

Steady-state isothermal approximations in the standard benchmark form
(the same physics as the GasLib literature and the ``gaslib_bench``
equation-oriented models, so sequential and simultaneous solutions of
the same network are directly comparable):

* squared-pressure Weymouth pipe law
  ``p_from^2 - p_to^2 = beta q |q|`` with q in kg/s and p in Pa,
* resistors in the same squared-pressure form with coefficient xi,
* generic adiabatic compressor shaft power with an optional smoothed
  |q| (matching the smoothing used in NLP objectives, so reduced-space
  and full-space optimizations minimize the same function).

For operational work one would replace the Nikuradse friction factor
and the constant compressibility with something richer; every
coefficient here is a plain float, so doing that does not disturb the
rest of the plugin.

Conventions: pressure in Pa, mass flow in kg/s, temperature in K.
GasLib XML files use bar, 1000 Nm^3/h and Celsius; the conversion
helpers cover the flow conversion (pressure and temperature are
trivial).
"""

from __future__ import annotations

import math

import jax.numpy as jnp

R_UNIVERSAL = 8314.462618  # J/(kmol K)
PI = math.pi

# Defaults for natural gas around 50 bar / 283 K.
DEFAULT_Z = 0.90          # compressibility factor
DEFAULT_KAPPA = 1.3       # isentropic exponent
DEFAULT_ETA_AD = 0.80     # adiabatic efficiency
DEFAULT_CP = 2200.0       # J/(kg K)
DEFAULT_TEMP_K = 283.15   # 10 C

BAR_TO_PA = 1.0e5
KNM3H_TO_M3S = 1000.0 / 3600.0   # 1000 Nm^3/h -> Nm^3/s

#: smoothing (kg/s) for |q| in objectives; matches the equation-oriented
#: NLP convention sqrt(q^2 + eps^2)
EPS_FLOW = 1.0e-4


def knm3h_to_kg_s(flow_knm3h: float, norm_density_kg_m3: float) -> float:
    """Convert 1000 m^3/h (normal conditions) to kg/s."""
    return flow_knm3h * KNM3H_TO_M3S * norm_density_kg_m3


def kg_s_to_knm3h(flow_kg_s: float, norm_density_kg_m3: float) -> float:
    """Convert kg/s to 1000 m^3/h (normal conditions)."""
    return flow_kg_s / (KNM3H_TO_M3S * norm_density_kg_m3)


def specific_gas_constant(molar_mass_kg_kmol: float) -> float:
    """R_s = R / M in J/(kg K)."""
    return R_UNIVERSAL / molar_mass_kg_kmol


def nikuradse_friction(diameter_m: float, roughness_m: float) -> float:
    """Fully-rough-turbulent Darcy friction factor (Nikuradse).

    lambda = (2 log10(D/k) + 1.138)^-2
    """
    return (2.0 * math.log10(diameter_m / roughness_m) + 1.138) ** -2


def weymouth_beta(
    length_m: float,
    diameter_m: float,
    roughness_m: float,
    gas_temp_k: float = DEFAULT_TEMP_K,
    molar_mass_kg_kmol: float = 18.0,
    z_factor: float = DEFAULT_Z,
) -> float:
    """Coefficient beta for the squared-pressure Weymouth equation.

        p_from^2 - p_to^2 = beta * q * |q|      (Pa^2, q in kg/s)
        beta = 16 lambda L R_s T z / (pi^2 D^5)

    Args:
        length_m: pipe length (m)
        diameter_m: inner diameter (m)
        roughness_m: absolute roughness (m)
        gas_temp_k: gas temperature (K)
        molar_mass_kg_kmol: molar mass (kg/kmol)
        z_factor: compressibility factor

    Returns:
        beta in Pa^2 / (kg/s)^2.
    """
    lam = nikuradse_friction(diameter_m, roughness_m)
    r_s = specific_gas_constant(molar_mass_kg_kmol)
    return (
        16.0 * lam * length_m * r_s * gas_temp_k * z_factor
        / (PI ** 2 * diameter_m ** 5)
    )


def resistor_xi(
    drag_factor: float,
    diameter_m: float,
    gas_temp_k: float = DEFAULT_TEMP_K,
    molar_mass_kg_kmol: float = 18.0,
    z_factor: float = DEFAULT_Z,
) -> float:
    """Coefficient xi for a resistor's squared-pressure drop.

    Standard form ``p_from^2 - p_to^2 = xi * q * |q|`` with the drag
    factor zeta folded in as

        xi = 8 zeta R_s T z / (pi^2 D^4)

    which mirrors the pipe formula with (16 lambda L) replaced by
    (8 zeta). Resistors therefore use the same unit operations as pipes
    with beta replaced by xi.
    """
    r_s = specific_gas_constant(molar_mass_kg_kmol)
    return (
        8.0 * drag_factor * r_s * gas_temp_k * z_factor
        / (PI ** 2 * diameter_m ** 4)
    )


def compressor_power(
    q_kg_s,
    ratio,
    t_in_k: float = DEFAULT_TEMP_K,
    cp: float = DEFAULT_CP,
    kappa: float = DEFAULT_KAPPA,
    eta_ad: float = DEFAULT_ETA_AD,
):
    """Adiabatic compressor shaft power (W), plain |q|.

        P = q c_p T_in (ratio^((k-1)/k) - 1) / eta_ad

    Works with floats, numpy or jax arrays, and symbolic (e.g. Pyomo)
    expressions when ``q_kg_s`` is nonnegative.
    """
    exponent = (kappa - 1.0) / kappa
    return q_kg_s * cp * t_in_k * (ratio ** exponent - 1.0) / eta_ad


def smoothed_power_w(
    q_kg_s,
    ratio,
    t_in_k: float = DEFAULT_TEMP_K,
    cp: float = DEFAULT_CP,
    kappa: float = DEFAULT_KAPPA,
    eta_ad: float = DEFAULT_ETA_AD,
    eps_flow: float = EPS_FLOW,
):
    """Adiabatic shaft power (W) with the smoothed |q| of NLP objectives.

        P = sqrt(q^2 + eps^2) c_p T_in (ratio^((k-1)/k) - 1) / eta_ad

    This is C^1 in q (finite-difference and interior-point solvers need
    that) and equals :func:`compressor_power` to O(eps^2/|q|) away from
    q = 0. Use this form when comparing objectives across sequential
    and equation-oriented solves of the same network.
    """
    exponent = (kappa - 1.0) / kappa
    qsm = jnp.sqrt(q_kg_s ** 2 + eps_flow ** 2)
    return qsm * cp * t_in_k * (ratio ** exponent - 1.0) / eta_ad
