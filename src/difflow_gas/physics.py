"""Gas physics: unit conversions, pipe/resistor coefficients, compressor power.

Steady-state isothermal approximations in the standard benchmark form
(the same physics as the GasLib literature and the ``gaslib_bench``
equation-oriented models, so sequential and simultaneous solutions of
the same network are directly comparable):

* the squared-pressure isothermal pipe law
  ``p_from^2 - p_to^2 = beta q |q|`` with q in kg/s and p in Pa (the
  GasLib literature calls this the "Weymouth" form; see
  :func:`isothermal_pipe_beta` for what that does and does not mean),
* resistors in the same squared-pressure form with coefficient xi,
* generic adiabatic compressor shaft power with an optional smoothed
  |q| (matching the smoothing used in NLP objectives, so reduced-space
  and full-space optimizations minimize the same function).

The pipe law drops the kinetic-energy (acceleration) term of the full
isothermal line equation,
``p1^2 - p2^2 = (q/A)^2 z R_s T [lambda L / D + 2 ln(p1/p2)]``; the
``2 ln(p1/p2)`` term is negligible for transmission lines (lambda L / D
of order 10^3 against 2 ln(p1/p2) of order 1) but not for short, fast
lines. :func:`difflow.compressible.isothermal_outlet_pressure` solves the
full equation. All coefficients are computed with ``jax.numpy`` and are
differentiable in every float argument.

Conventions: pressure in Pa, mass flow in kg/s, temperature in K.
GasLib XML files use bar, 1000 Nm^3/h and Celsius; the conversion
helpers cover the flow conversion (pressure and temperature are
trivial).
"""

from __future__ import annotations

import math

import jax
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

#: Friction laws accepted by :func:`isothermal_pipe_beta`.
PIPE_FRICTION_LAWS = ("nikuradse", "colebrook", "weymouth", "panhandle_a",
                      "panhandle_b", "aga")

#: Base (standard) conditions of the US pipeline correlations: 60 F and
#: 14.73 psia (Menon, *Gas Pipeline Hydraulics*, CRC 2005, Ch. 2).
PIPELINE_BASE_T_K = 288.706
PIPELINE_BASE_P_PA = 101_560.0
_M3_S_TO_SCFD = 35.314667 * 86400.0
_AIR_MOLAR_MASS = 28.9625


def knm3h_to_kg_s(flow_knm3h: float, norm_density_kg_m3: float) -> float:
    """Convert 1000 m^3/h (normal conditions) to kg/s."""
    return flow_knm3h * KNM3H_TO_M3S * norm_density_kg_m3


def kg_s_to_knm3h(flow_kg_s: float, norm_density_kg_m3: float) -> float:
    """Convert kg/s to 1000 m^3/h (normal conditions)."""
    return flow_kg_s / (KNM3H_TO_M3S * norm_density_kg_m3)


def specific_gas_constant(molar_mass_kg_kmol: float) -> float:
    """R_s = R / M in J/(kg K)."""
    return R_UNIVERSAL / molar_mass_kg_kmol


def nikuradse_friction(diameter_m, roughness_m):
    """Fully-rough-turbulent Darcy friction factor (Nikuradse).

    ``lambda = (2 log10(D/k) + 1.138)^-2``, the Re -> infinity limit of
    Colebrook-White (``2 log10(3.7) = 1.136``, so this equals
    ``(2 log10(3.7 D/k))^-2`` to 0.05%). Independent of the Reynolds
    number, so it overestimates nothing in fully rough flow but
    underestimates lambda for smooth or moderately rough lines (by about
    1% on a 0.6 m, 45 um line at Re = 2e7). Differentiable (``jnp``).
    """
    return (2.0 * jnp.log10(jnp.asarray(diameter_m) / roughness_m) + 1.138) ** -2


def weymouth_friction(diameter_m):
    """Darcy friction factor of the Weymouth (1912) gas-pipeline equation.

    ``lambda = 0.032 / d^(1/3)`` with d the inner diameter in inches, i.e.
    ``lambda = 0.009406 / D^(1/3)`` with D in m. Equivalent to the Weymouth
    transmission factor ``F = 2/sqrt(lambda) = 11.18 d^(1/6)`` (Menon, *Gas
    Pipeline Hydraulics*, CRC 2005, Ch. 2). An empirical fit to old
    small-bore lines: it ignores roughness and Reynolds number and is
    conservative (high) for large modern lines.
    """
    d_in = jnp.asarray(diameter_m) / 0.0254
    return 0.032 * d_in ** (-1.0 / 3.0)


def _transmission_to_darcy(F):
    """Darcy friction factor from the transmission factor ``F = 2/sqrt(lambda)``."""
    return 4.0 / F**2


def standard_flow_scfd(mass_flow_kg_s, molar_mass_kg_kmol):
    """Mass flow (kg/s) as standard ft^3/day at 60 F, 14.73 psia (ideal gas).

    The flow unit of the Panhandle correlations; the base density is
    ``P_b M / (R T_b)``.
    """
    rho_b = PIPELINE_BASE_P_PA * molar_mass_kg_kmol / (R_UNIVERSAL * PIPELINE_BASE_T_K)
    return jnp.abs(mass_flow_kg_s) / rho_b * _M3_S_TO_SCFD


def panhandle_a_friction(diameter_m, mass_flow_kg_s, molar_mass_kg_kmol,
                         efficiency=1.0):
    """Darcy friction factor of the Panhandle A gas-pipeline equation.

    Transmission factor ``F = 7.2111 E (Q G / d)^0.07305`` with ``Q`` in
    standard ft^3/day (60 F, 14.73 psia), ``G`` the gas specific gravity
    (``M / 28.9625``), ``d`` the inner diameter in inches and ``E`` the
    pipeline efficiency (Menon, *Gas Pipeline Hydraulics*, CRC 2005,
    Eq. 2.62); ``lambda = 4 / F^2``. A partially turbulent (smooth-pipe)
    fit for large lines, Re of 5e6 to 1.1e7; it ignores roughness and
    predicts a low friction factor (a high capacity) outside that range.
    ``E`` (typically 0.88-0.94) folds the line's age and fittings into the
    correlation; ``E = 1`` is the bare fit.
    """
    d_in = jnp.asarray(diameter_m) / 0.0254
    QG = standard_flow_scfd(mass_flow_kg_s, molar_mass_kg_kmol) * (
        molar_mass_kg_kmol / _AIR_MOLAR_MASS)
    F = 7.2111 * efficiency * (QG / d_in) ** 0.07305
    return _transmission_to_darcy(F)


def panhandle_b_friction(diameter_m, mass_flow_kg_s, molar_mass_kg_kmol,
                         efficiency=1.0):
    """Darcy friction factor of the Panhandle B (revised Panhandle) equation.

    ``F = 16.70 E (Q G / d)^0.01961`` in the units of
    :func:`panhandle_a_friction` (Menon, *Gas Pipeline Hydraulics*, CRC 2005,
    Eq. 2.67); ``lambda = 4 / F^2``. Fitted to large, high-pressure lines
    at Re of 4e6 to 4e7.
    """
    d_in = jnp.asarray(diameter_m) / 0.0254
    QG = standard_flow_scfd(mass_flow_kg_s, molar_mass_kg_kmol) * (
        molar_mass_kg_kmol / _AIR_MOLAR_MASS)
    F = 16.70 * efficiency * (QG / d_in) ** 0.01961
    return _transmission_to_darcy(F)


def aga_friction(diameter_m, roughness_m, mass_flow_kg_s, viscosity_pa_s,
                 drag_factor=0.96, n_iter: int = 30):
    """Darcy friction factor of the AGA (1964) transmission equation.

    The AGA (American Gas Association NB-13) method takes the smaller of two
    transmission factors (Menon, *Gas Pipeline Hydraulics*, CRC 2005,
    Eqs. 2.71-2.73):

    - fully turbulent: ``F = 4 log10(3.7 D / e)`` (rough-pipe law),
    - partially turbulent: ``F = 4 D_f log10(Re / F_t)`` with the smooth-pipe
      factor ``F_t = 4 log10(Re / F_t) - 0.6`` (von Karman),

    and ``lambda = 4 / F^2``. ``D_f`` is the bend-index drag factor, 0.90 to
    0.99 (0.96 is a typical value for a line with moderate bends). ``Re`` is
    ``4 q / (pi D mu)``. ``F_t`` is a contraction solved by ``n_iter``
    substitutions, differentiable throughout.
    """
    D = jnp.asarray(diameter_m)
    Re = 4.0 * jnp.abs(mass_flow_kg_s) / (PI * D * viscosity_pa_s)
    Re = jnp.maximum(Re, 1.0e3)

    def step(_, Ft):
        return 4.0 * jnp.log10(Re / Ft) - 0.6

    Ft = jax.lax.fori_loop(0, n_iter, step, jnp.full_like(Re, 20.0))
    F_partial = 4.0 * drag_factor * jnp.log10(Re / Ft)
    F_full = 4.0 * jnp.log10(3.7 * D / roughness_m)
    return _transmission_to_darcy(jnp.minimum(F_partial, F_full))


def isothermal_pipe_beta(
    length_m,
    diameter_m,
    roughness_m,
    gas_temp_k=DEFAULT_TEMP_K,
    molar_mass_kg_kmol=18.0,
    z_factor=DEFAULT_Z,
    friction: str = "nikuradse",
    mass_flow_kg_s=None,
    viscosity_pa_s=None,
    efficiency=1.0,
    drag_factor=0.96,
):
    """Coefficient beta of the squared-pressure isothermal pipe law.

        p_from^2 - p_to^2 = beta * q * |q|      (Pa^2, q in kg/s)
        beta = 16 lambda L R_s T z / (pi^2 D^5)

    This is the isothermal line equation without its kinetic-energy term
    ``2 ln(p_from/p_to)`` (see the module docstring), the form the GasLib
    literature calls "Weymouth". Strictly, the Weymouth *equation* is this
    law with Weymouth's diameter-only friction correlation
    (``friction="weymouth"``); the default here is the fully rough
    Nikuradse factor of the GasLib benchmarks.

    Args:
        length_m: pipe length (m)
        diameter_m: inner diameter (m)
        roughness_m: absolute roughness (m); unused by ``"weymouth"``
        gas_temp_k: gas temperature (K)
        molar_mass_kg_kmol: molar mass (kg/kmol)
        z_factor: compressibility factor
        friction: ``"nikuradse"`` (fully rough, Re-independent; default and
            the historical behaviour), ``"colebrook"`` (Colebrook-White at
            the Reynolds number of ``mass_flow_kg_s``, via
            :func:`difflow.fluids.friction_factor`), or ``"weymouth"``
            (:func:`weymouth_friction`), ``"panhandle_a"`` /
            ``"panhandle_b"`` (:func:`panhandle_a_friction`,
            :func:`panhandle_b_friction`, flow-dependent), or ``"aga"``
            (:func:`aga_friction`, AGA fully/partially turbulent).
        mass_flow_kg_s: mass flow (kg/s); required for ``"colebrook"``,
            ``"panhandle_a"``, ``"panhandle_b"`` and ``"aga"``. beta then
            depends on the flow, so a network solve that holds beta fixed
            must update it.
        viscosity_pa_s: gas dynamic viscosity (Pa s); required for
            ``"colebrook"`` and ``"aga"``.
        efficiency: Panhandle pipeline efficiency ``E`` (default 1).
        drag_factor: AGA drag factor ``D_f`` (default 0.96).

    Returns:
        beta in Pa^2 / (kg/s)^2 (a JAX scalar, differentiable in every
        float argument).

    Raises:
        ValueError: unknown ``friction`` or missing Colebrook inputs.

    Example:
        >>> beta = isothermal_pipe_beta(50e3, 0.6, 45e-6, 288.15, 16.04, 0.851,
        ...                             friction="colebrook",
        ...                             mass_flow_kg_s=121.5, viscosity_pa_s=1.2e-5)
    """
    if friction == "nikuradse":
        lam = nikuradse_friction(diameter_m, roughness_m)
    elif friction == "weymouth":
        lam = weymouth_friction(diameter_m)
    elif friction == "colebrook":
        if mass_flow_kg_s is None or viscosity_pa_s is None:
            raise ValueError(
                "friction='colebrook' needs mass_flow_kg_s and viscosity_pa_s "
                "(the Reynolds number)"
            )
        from difflow.fluids import friction_factor

        D = jnp.asarray(diameter_m)
        Re = 4.0 * jnp.abs(mass_flow_kg_s) / (PI * D * viscosity_pa_s)
        lam = friction_factor(Re, roughness_m / D, method="colebrook")
    elif friction in ("panhandle_a", "panhandle_b"):
        if mass_flow_kg_s is None:
            raise ValueError(f"friction={friction!r} needs mass_flow_kg_s")
        fn = panhandle_a_friction if friction == "panhandle_a" else panhandle_b_friction
        lam = fn(diameter_m, mass_flow_kg_s, molar_mass_kg_kmol, efficiency)
    elif friction == "aga":
        if mass_flow_kg_s is None or viscosity_pa_s is None:
            raise ValueError(
                "friction='aga' needs mass_flow_kg_s and viscosity_pa_s "
                "(the Reynolds number)"
            )
        lam = aga_friction(diameter_m, roughness_m, mass_flow_kg_s,
                           viscosity_pa_s, drag_factor)
    else:
        raise ValueError(
            f"Unknown friction law {friction!r}; use one of {PIPE_FRICTION_LAWS}"
        )
    r_s = specific_gas_constant(molar_mass_kg_kmol)
    return (
        16.0 * lam * length_m * r_s * gas_temp_k * z_factor
        / (PI ** 2 * jnp.asarray(diameter_m) ** 5)
    )


#: Historical name of :func:`isothermal_pipe_beta` (same signature and
#: defaults, Nikuradse friction). Kept for existing callers; "Weymouth"
#: refers to the squared-pressure form, not Weymouth's friction factor.
weymouth_beta = isothermal_pipe_beta


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
