"""Objective functions for Nd/Dy separation optimization.

All functions are JAX-differentiable for gradient-based optimization.
"""

import jax.numpy as jnp
from jax import Array

from .single_stage import SingleStageLLE, D2EHPADistribution


# =============================================================================
# Model Instance (shared)
# =============================================================================

_model = SingleStageLLE(
    distribution=D2EHPADistribution(),
    efficiency=0.95,
)


# =============================================================================
# Core Objective Functions
# =============================================================================

def dy_purity(
    pH: Array,
    OA_ratio: Array,
    T: Array = jnp.array(298.15),
    conc: Array = jnp.array(0.5),
    F_Nd_feed: Array = jnp.array(0.01),
    F_Dy_feed: Array = jnp.array(0.01),
    F_aq: Array = jnp.array(1.0),
) -> Array:
    """Calculate Dy purity in extract phase.

    Args:
        pH: Operating pH
        OA_ratio: Organic-to-aqueous flow ratio
        T: Temperature (K)
        conc: D2EHPA concentration (M)
        F_Nd_feed: Nd feed flow (mol/s)
        F_Dy_feed: Dy feed flow (mol/s)
        F_aq: Aqueous carrier flow

    Returns:
        Dy purity (mole fraction) in extract
    """
    F_org = F_aq * OA_ratio

    result = _model(
        F_Nd_feed=F_Nd_feed,
        F_Dy_feed=F_Dy_feed,
        F_aq=F_aq,
        F_org=F_org,
        pH=pH,
        T=T,
        conc=conc,
    )

    return result.purity_Dy_org


def dy_recovery(
    pH: Array,
    OA_ratio: Array,
    T: Array = jnp.array(298.15),
    conc: Array = jnp.array(0.5),
    F_Nd_feed: Array = jnp.array(0.01),
    F_Dy_feed: Array = jnp.array(0.01),
    F_aq: Array = jnp.array(1.0),
) -> Array:
    """Calculate Dy recovery to extract phase.

    Returns:
        Dy recovery fraction (0-1)
    """
    F_org = F_aq * OA_ratio

    result = _model(
        F_Nd_feed=F_Nd_feed,
        F_Dy_feed=F_Dy_feed,
        F_aq=F_aq,
        F_org=F_org,
        pH=pH,
        T=T,
        conc=conc,
    )

    return result.recovery_Dy


def nd_purity(
    pH: Array,
    OA_ratio: Array,
    T: Array = jnp.array(298.15),
    conc: Array = jnp.array(0.5),
    F_Nd_feed: Array = jnp.array(0.01),
    F_Dy_feed: Array = jnp.array(0.01),
    F_aq: Array = jnp.array(1.0),
) -> Array:
    """Calculate Nd purity in raffinate phase.

    Returns:
        Nd purity (mole fraction) in raffinate
    """
    F_org = F_aq * OA_ratio

    result = _model(
        F_Nd_feed=F_Nd_feed,
        F_Dy_feed=F_Dy_feed,
        F_aq=F_aq,
        F_org=F_org,
        pH=pH,
        T=T,
        conc=conc,
    )

    return result.purity_Nd_aq


def nd_recovery(
    pH: Array,
    OA_ratio: Array,
    T: Array = jnp.array(298.15),
    conc: Array = jnp.array(0.5),
    F_Nd_feed: Array = jnp.array(0.01),
    F_Dy_feed: Array = jnp.array(0.01),
    F_aq: Array = jnp.array(1.0),
) -> Array:
    """Calculate Nd recovery to raffinate phase.

    Returns:
        Nd recovery fraction (0-1)
    """
    F_org = F_aq * OA_ratio

    result = _model(
        F_Nd_feed=F_Nd_feed,
        F_Dy_feed=F_Dy_feed,
        F_aq=F_aq,
        F_org=F_org,
        pH=pH,
        T=T,
        conc=conc,
    )

    return result.recovery_Nd


# =============================================================================
# Cost Functions
# =============================================================================

def separation_cost(
    pH: Array,
    OA_ratio: Array,
    T: Array = jnp.array(298.15),
    conc: Array = jnp.array(0.5),
    F_aq: Array = jnp.array(1.0),
    hours_per_year: float = 8000.0,
    extractant_price: float = 8.0,  # $/kg D2EHPA
    acid_price: float = 0.15,  # $/kg HCl
    base_price: float = 0.50,  # $/kg NaOH
    electricity_price: float = 0.07,  # $/kWh
) -> Array:
    """Calculate annual operating cost for separation.

    Simplified cost model including:
    - Extractant makeup (0.1% loss per contact)
    - Acid/base for pH control
    - Mixing power

    Args:
        pH: Operating pH
        OA_ratio: Organic-to-aqueous ratio
        T: Temperature (K)
        conc: D2EHPA concentration (M)
        F_aq: Aqueous flow rate (kg/s)
        hours_per_year: Operating hours
        extractant_price: D2EHPA cost ($/kg)
        acid_price: HCl cost ($/kg)
        base_price: NaOH cost ($/kg)
        electricity_price: Electricity cost ($/kWh)

    Returns:
        Annual operating cost ($/year)
    """
    F_org = F_aq * OA_ratio
    seconds_per_year = hours_per_year * 3600.0

    # Extractant makeup (0.1% entrainment loss)
    # D2EHPA MW = 322.43 g/mol, density ~0.975 g/mL
    extractant_loss_rate = 0.001  # fraction per contact
    org_volume_rate = F_org / 0.8  # L/s (assuming kerosene density ~0.8 kg/L)
    extractant_mass_rate = org_volume_rate * conc * 0.322  # kg/s D2EHPA
    extractant_makeup = extractant_mass_rate * extractant_loss_rate * extractant_price
    extractant_annual = extractant_makeup * seconds_per_year

    # Acid/base for pH control
    # Higher pH requires more base; lower pH may need acid addition
    # Simplified: cost proportional to deviation from natural pH (~2)
    pH_deviation = jnp.abs(pH - 2.0)
    reagent_rate = F_aq * 0.001 * pH_deviation  # kg/s reagent
    reagent_price = jnp.where(pH > 2.0, base_price, acid_price)
    reagent_annual = reagent_rate * reagent_price * seconds_per_year

    # Mixing power
    # Power ~ 0.5 kW per L/s of total flow
    total_flow = F_aq / 1.0 + F_org / 0.8  # L/s
    mixing_power = 0.5 * total_flow  # kW
    electricity_annual = mixing_power * electricity_price * hours_per_year

    # Temperature control (heating/cooling)
    # Cost increases with deviation from ambient (298 K)
    T_deviation = jnp.abs(T - 298.15)
    thermal_cost_rate = T_deviation * 0.001 * F_aq  # $/s, rough estimate
    thermal_annual = thermal_cost_rate * seconds_per_year

    total_annual = (
        extractant_annual +
        reagent_annual +
        electricity_annual +
        thermal_annual
    )

    return total_annual


def capital_cost(
    F_aq: Array = jnp.array(1.0),
    OA_ratio: Array = jnp.array(1.0),
    mixer_residence_time: float = 120.0,  # seconds
    settler_residence_time: float = 300.0,  # seconds
) -> Array:
    """Estimate capital cost for single mixer-settler.

    Uses factored estimation based on vessel volume.

    Args:
        F_aq: Aqueous flow rate (kg/s)
        OA_ratio: Organic-to-aqueous ratio
        mixer_residence_time: Mixer residence time (s)
        settler_residence_time: Settler residence time (s)

    Returns:
        Installed capital cost ($)
    """
    F_org = F_aq * OA_ratio

    # Volume flows (L/s)
    V_aq = F_aq / 1.0  # Assuming density 1 kg/L
    V_org = F_org / 0.8  # Assuming density 0.8 kg/L
    V_total = V_aq + V_org

    # Vessel volumes (m³)
    V_mixer = V_total * mixer_residence_time / 1000.0
    V_settler = V_total * settler_residence_time / 1000.0

    # Cost correlation (simplified Guthrie-type)
    # C = a + b * V^n
    # Mixer: agitated vessel
    mixer_cost = 15000 + 8000 * jnp.power(V_mixer, 0.6)

    # Settler: horizontal vessel
    settler_cost = 10000 + 5000 * jnp.power(V_settler, 0.6)

    # Auxiliaries (pumps, piping): 30%
    auxiliary = (mixer_cost + settler_cost) * 0.30

    # Installation factor (Lang): 3.5
    installed_cost = (mixer_cost + settler_cost + auxiliary) * 3.5

    return installed_cost


def annualized_cost(
    pH: Array,
    OA_ratio: Array,
    T: Array = jnp.array(298.15),
    conc: Array = jnp.array(0.5),
    F_aq: Array = jnp.array(1.0),
    discount_rate: float = 0.10,
    plant_life: float = 20.0,
) -> Array:
    """Calculate total annualized cost (CAPEX + OPEX).

    Args:
        pH: Operating pH
        OA_ratio: Organic-to-aqueous ratio
        T: Temperature (K)
        conc: D2EHPA concentration (M)
        F_aq: Aqueous flow rate (kg/s)
        discount_rate: Discount rate for annualization
        plant_life: Plant lifetime (years)

    Returns:
        Total annualized cost ($/year)
    """
    # Operating cost
    opex = separation_cost(pH, OA_ratio, T, conc, F_aq)

    # Capital cost (annualized)
    capex = capital_cost(F_aq, OA_ratio)
    crf = (discount_rate * jnp.power(1 + discount_rate, plant_life)) / \
          (jnp.power(1 + discount_rate, plant_life) - 1)
    annualized_capex = capex * crf

    return opex + annualized_capex


def cost_per_kg_dy(
    pH: Array,
    OA_ratio: Array,
    T: Array = jnp.array(298.15),
    conc: Array = jnp.array(0.5),
    F_Nd_feed: Array = jnp.array(0.01),
    F_Dy_feed: Array = jnp.array(0.01),
    F_aq: Array = jnp.array(1.0),
    hours_per_year: float = 8000.0,
) -> Array:
    """Calculate separation cost per kg of Dy recovered.

    Args:
        pH: Operating pH
        OA_ratio: Organic-to-aqueous ratio
        T: Temperature (K)
        conc: D2EHPA concentration (M)
        F_Nd_feed: Nd feed flow (mol/s)
        F_Dy_feed: Dy feed flow (mol/s)
        F_aq: Aqueous flow rate (kg/s)
        hours_per_year: Operating hours

    Returns:
        Cost per kg Dy recovered ($/kg)
    """
    # Annual cost
    annual_cost = annualized_cost(pH, OA_ratio, T, conc, F_aq)

    # Dy recovery
    recovery = dy_recovery(pH, OA_ratio, T, conc, F_Nd_feed, F_Dy_feed, F_aq)

    # Annual Dy production (kg/year)
    Dy_MW = 162.5  # g/mol
    seconds_per_year = hours_per_year * 3600.0
    annual_Dy_kg = F_Dy_feed * recovery * Dy_MW / 1000.0 * seconds_per_year

    return annual_cost / (annual_Dy_kg + 1e-10)


# =============================================================================
# Combined Objective for Optimization
# =============================================================================

def weighted_objective(
    params: Array,
    w_purity: float = 1.0,
    w_recovery: float = 0.5,
    w_cost: float = 0.0001,
    minimize: bool = True,
) -> Array:
    """Weighted multi-objective function.

    params = [pH, OA_ratio, T, conc]

    Objective = w_purity * purity + w_recovery * recovery - w_cost * cost

    Args:
        params: Array of [pH, OA_ratio, T, conc]
        w_purity: Weight on Dy purity (maximize)
        w_recovery: Weight on Dy recovery (maximize)
        w_cost: Weight on cost (minimize)
        minimize: If True, return negative (for minimization)

    Returns:
        Weighted objective value
    """
    pH, OA_ratio, T, conc = params[0], params[1], params[2], params[3]

    purity = dy_purity(pH, OA_ratio, T, conc)
    recovery = dy_recovery(pH, OA_ratio, T, conc)
    cost = annualized_cost(pH, OA_ratio, T, conc)

    # Normalize cost to similar scale as purity/recovery
    cost_normalized = cost / 1e6  # Cost in millions

    obj = w_purity * purity + w_recovery * recovery - w_cost * cost_normalized

    if minimize:
        return -obj
    return obj


def constrained_purity_objective(
    params: Array,
    min_recovery: float = 0.80,
    penalty_weight: float = 100.0,
) -> Array:
    """Maximize purity subject to minimum recovery constraint.

    Uses penalty method for constraint handling.

    Args:
        params: Array of [pH, OA_ratio, T, conc]
        min_recovery: Minimum required Dy recovery
        penalty_weight: Penalty coefficient for constraint violation

    Returns:
        Negative purity plus penalty (for minimization)
    """
    pH, OA_ratio, T, conc = params[0], params[1], params[2], params[3]

    purity = dy_purity(pH, OA_ratio, T, conc)
    recovery = dy_recovery(pH, OA_ratio, T, conc)

    # Constraint violation penalty
    violation = jnp.maximum(min_recovery - recovery, 0.0)
    penalty = penalty_weight * violation**2

    # Minimize negative purity (maximize purity)
    return -purity + penalty
