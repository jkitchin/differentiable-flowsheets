"""Protein stability and degradation models.

Provides kinetic models for common biopharmaceutical degradation
pathways during manufacturing and storage.

All functions use JAX numpy for automatic differentiation compatibility.

Models implemented:
- Aggregation: First-order and Lumry-Eyring models
- Deamidation: pH/temperature dependent Asn deamidation
- Oxidation: Met oxidation kinetics
- Fragmentation: Peptide bond cleavage

References:
    Manning MC et al. (2010). Pharm Res 27:544.
    Roberts CJ (2007). J Phys Chem B 111:13447.
    Wang W (1999). Int J Pharm 185:129.
"""

__all__ = [
    "aggregation_rate",
    "aggregation_arrhenius",
    "aggregate_fraction",
    "stretched_exponential_fraction",
    "lumry_eyring_fraction",
    "deamidation_rate",
    "deamidation_ph_dependent",
    "deamidation_fraction",
    "oxidation_rate",
    "oxidation_peroxide",
    "oxidation_fraction",
    "fragmentation_rate",
    "fragmentation_fraction",
    "total_degradation",
    "shelf_life",
    "DegradationModel",
    "DegradationParams",
    "get_degradation_model",
]

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Physical Constants
# =============================================================================

R = 8.314  # J/mol/K - Gas constant


# =============================================================================
# Aggregation Models
# =============================================================================

def aggregation_rate(
    C: Array | float,
    k_agg: Array | float,
    n: Array | float = 1.0,
) -> Array:
    """Calculate aggregation rate.

    dA/dt = k_agg * C^n

    First-order (n=1) or higher-order aggregation kinetics.

    Args:
        C: Protein concentration (g/L)
        k_agg: Aggregation rate constant (1/h for n=1)
        n: Reaction order (default 1)

    Returns:
        Aggregation rate (g/L/h)

    Notes:
        n=1: Conformational change limited (unfolding)
        n=2: Collision limited (association)
    """
    C = jnp.asarray(C)
    k_agg = jnp.asarray(k_agg)
    n = jnp.asarray(n)

    return k_agg * jnp.power(C, n)


def aggregation_arrhenius(
    T: Array | float,
    k_ref: Array | float,
    E_a: Array | float,
    T_ref: Array | float = 298.15,
) -> Array:
    """Temperature-dependent aggregation rate via Arrhenius.

    k(T) = k_ref * exp(-E_a/R * (1/T - 1/T_ref))

    Args:
        T: Temperature (K)
        k_ref: Rate constant at reference temperature (1/h)
        E_a: Activation energy (J/mol)
        T_ref: Reference temperature (K), default 25°C

    Returns:
        Rate constant at temperature T

    Notes:
        Typical E_a for protein aggregation: 80-150 kJ/mol
    """
    T = jnp.asarray(T)
    k_ref = jnp.asarray(k_ref)
    E_a = jnp.asarray(E_a)
    T_ref = jnp.asarray(T_ref)

    return k_ref * jnp.exp(-E_a / R * (1.0 / T - 1.0 / T_ref))


def aggregate_fraction(
    t: Array | float,
    k_agg: Array | float,
) -> Array:
    """Calculate fraction aggregated over time.

    f_agg = 1 - exp(-k_agg * t)

    First-order kinetics integrated form.

    Args:
        t: Time (h)
        k_agg: Aggregation rate constant (1/h)

    Returns:
        Fraction aggregated (0-1)
    """
    t = jnp.asarray(t)
    k_agg = jnp.asarray(k_agg)

    return 1.0 - jnp.exp(-k_agg * t)


def stretched_exponential_fraction(
    t: Array | float,
    k: Array | float,
    beta: Array | float = 1.0,
) -> Array:
    """Degraded fraction with stretched-exponential (KWW/Weibull) kinetics.

    f = 1 - exp(-(k*t)^beta)

    Generalizes the first-order integrated form (beta = 1) to the
    Kohlrausch-Williams-Watts / Weibull kinetics often observed for protein
    degradation in heterogeneous environments, where a distribution of rate
    constants gives non-first-order time dependence:
      - beta < 1: dispersive / broad rate distribution (fast then tailing)
      - beta > 1: sigmoidal with an induction lag

    Args:
        t: Time (h)
        k: Characteristic rate constant (1/h)
        beta: Stretch exponent (dimensionless, > 0). beta = 1 -> first order.

    Returns:
        Degraded fraction (0-1)

    References:
        Manning MC et al. (2010). Pharm Res 27:544 (non-first-order protein
        degradation kinetics).
    """
    t = jnp.asarray(t)
    k = jnp.asarray(k)
    beta = jnp.asarray(beta)
    # Guard the power for t=0 (0^beta is 0 for beta>0; keep it well-defined).
    return 1.0 - jnp.exp(-jnp.power(jnp.maximum(k * t, 0.0), beta))


def lumry_eyring_fraction(
    t: Array | float,
    k_unfold: Array | float,
    k_agg: Array | float,
) -> Array:
    """Aggregated fraction from the Lumry-Eyring two-step model.

    Native -> Unfolded (k_unfold) -> Aggregate (k_agg), treated as consecutive
    irreversible first-order steps. The integrated aggregate fraction is the
    classic consecutive-reaction product:

        f_A = 1 - (k_agg e^{-k_unfold t} - k_unfold e^{-k_agg t})/(k_agg - k_unfold)

    with the degenerate limit f_A = 1 - (1 + k t) e^{-k t} when the two rate
    constants coincide. Unlike a single first-order decay, this reproduces the
    lag phase set by the unfolding step before aggregation accelerates.

    Args:
        t: Time (h)
        k_unfold: Unfolding rate constant N -> U (1/h)
        k_agg: Aggregation rate constant U -> A (1/h)

    Returns:
        Aggregated fraction (0-1)

    References:
        Roberts CJ (2007). J Phys Chem B 111:13447 (Lumry-Eyring aggregation).
    """
    t = jnp.asarray(t)
    k1 = jnp.asarray(k_unfold)
    k2 = jnp.asarray(k_agg)
    denom = k2 - k1
    safe_denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    f_general = 1.0 - (k2 * jnp.exp(-k1 * t) - k1 * jnp.exp(-k2 * t)) / safe_denom
    f_equal = 1.0 - (1.0 + k1 * t) * jnp.exp(-k1 * t)
    return jnp.where(jnp.abs(denom) < 1e-12, f_equal, f_general)


# =============================================================================
# Deamidation Models
# =============================================================================

def deamidation_rate(
    k_deam: Array | float,
    C: Array | float = 1.0,
) -> Array:
    """Calculate deamidation rate.

    dD/dt = k_deam * C

    First-order Asn/Gln deamidation kinetics.

    Args:
        k_deam: Deamidation rate constant (1/h)
        C: Protein concentration (g/L)

    Returns:
        Deamidation rate (g/L/h)

    Notes:
        Asn-Gly sequences are most susceptible.
        Half-lives range from days (Asn-Gly) to years (Asn-Pro).
    """
    k_deam = jnp.asarray(k_deam)
    C = jnp.asarray(C)

    return k_deam * C


def deamidation_ph_dependent(
    pH: Array | float,
    T: Array | float,
    k_ref: Array | float,
    pH_ref: Array | float = 7.0,
    T_ref: Array | float = 298.15,
    E_a: Array | float = 85000.0,
    pH_slope: Array | float = 0.5,
) -> Array:
    """pH and temperature dependent deamidation rate.

    k = k_ref * 10^(pH_slope*(pH-pH_ref)) * exp(-E_a/R*(1/T-1/T_ref))

    Args:
        pH: Solution pH
        T: Temperature (K)
        k_ref: Rate at reference conditions (1/h)
        pH_ref: Reference pH (default 7.0)
        T_ref: Reference temperature (K)
        E_a: Activation energy (J/mol)
        pH_slope: pH sensitivity factor

    Returns:
        Deamidation rate constant (1/h)

    Notes:
        Deamidation accelerates at higher pH (base-catalyzed).
        Typical E_a: 80-90 kJ/mol
    """
    pH = jnp.asarray(pH)
    T = jnp.asarray(T)
    k_ref = jnp.asarray(k_ref)
    pH_ref = jnp.asarray(pH_ref)
    T_ref = jnp.asarray(T_ref)
    E_a = jnp.asarray(E_a)
    pH_slope = jnp.asarray(pH_slope)

    # pH factor (increases with pH)
    ph_factor = jnp.power(10.0, pH_slope * (pH - pH_ref))

    # Temperature factor
    t_factor = jnp.exp(-E_a / R * (1.0 / T - 1.0 / T_ref))

    return k_ref * ph_factor * t_factor


def deamidation_fraction(
    t: Array | float,
    k_deam: Array | float,
) -> Array:
    """Calculate fraction deamidated over time.

    f_deam = 1 - exp(-k_deam * t)

    Args:
        t: Time (h)
        k_deam: Deamidation rate constant (1/h)

    Returns:
        Fraction deamidated (0-1)
    """
    t = jnp.asarray(t)
    k_deam = jnp.asarray(k_deam)

    return 1.0 - jnp.exp(-k_deam * t)


# =============================================================================
# Oxidation Models
# =============================================================================

def oxidation_rate(
    k_ox: Array | float,
    C: Array | float = 1.0,
) -> Array:
    """Calculate oxidation rate.

    dO/dt = k_ox * C

    First-order Met/Trp oxidation kinetics.

    Args:
        k_ox: Oxidation rate constant (1/h)
        C: Protein concentration (g/L)

    Returns:
        Oxidation rate (g/L/h)

    Notes:
        Met oxidation to Met sulfoxide is most common.
        Surface-exposed Met residues are most susceptible.
    """
    k_ox = jnp.asarray(k_ox)
    C = jnp.asarray(C)

    return k_ox * C


def oxidation_peroxide(
    C_H2O2: Array | float,
    C: Array | float,
    k_ox: Array | float,
) -> Array:
    """Peroxide-mediated oxidation rate.

    dO/dt = k_ox * C * C_H2O2

    Second-order oxidation with peroxide concentration.

    Args:
        C_H2O2: Hydrogen peroxide concentration (mM)
        C: Protein concentration (g/L)
        k_ox: Second-order rate constant (L/mmol/h)

    Returns:
        Oxidation rate (g/L/h)

    Notes:
        Polysorbate degradation can generate peroxides.
        Typical H2O2 levels: 0.01-1 mM
    """
    C_H2O2 = jnp.asarray(C_H2O2)
    C = jnp.asarray(C)
    k_ox = jnp.asarray(k_ox)

    return k_ox * C * C_H2O2


def oxidation_fraction(
    t: Array | float,
    k_ox: Array | float,
) -> Array:
    """Calculate fraction oxidized over time.

    f_ox = 1 - exp(-k_ox * t)

    Args:
        t: Time (h)
        k_ox: Oxidation rate constant (1/h)

    Returns:
        Fraction oxidized (0-1)
    """
    t = jnp.asarray(t)
    k_ox = jnp.asarray(k_ox)

    return 1.0 - jnp.exp(-k_ox * t)


# =============================================================================
# Fragmentation Models
# =============================================================================

def fragmentation_rate(
    k_frag: Array | float,
    C: Array | float = 1.0,
) -> Array:
    """Calculate fragmentation rate.

    dF/dt = k_frag * C

    First-order peptide bond cleavage kinetics.

    Args:
        k_frag: Fragmentation rate constant (1/h)
        C: Protein concentration (g/L)

    Returns:
        Fragmentation rate (g/L/h)

    Notes:
        Asp-Pro bonds are particularly labile.
        Hinge region cleavage common in IgG.
    """
    k_frag = jnp.asarray(k_frag)
    C = jnp.asarray(C)

    return k_frag * C


def fragmentation_fraction(
    t: Array | float,
    k_frag: Array | float,
) -> Array:
    """Calculate fraction fragmented over time.

    f_frag = 1 - exp(-k_frag * t)

    Args:
        t: Time (h)
        k_frag: Fragmentation rate constant (1/h)

    Returns:
        Fraction fragmented (0-1)
    """
    t = jnp.asarray(t)
    k_frag = jnp.asarray(k_frag)

    return 1.0 - jnp.exp(-k_frag * t)


# =============================================================================
# Combined Degradation Models
# =============================================================================

def total_degradation(
    t: Array | float,
    k_agg: Array | float = 0.0,
    k_deam: Array | float = 0.0,
    k_ox: Array | float = 0.0,
    k_frag: Array | float = 0.0,
) -> dict:
    """Calculate total degradation from all pathways.

    Assumes independent first-order kinetics for each pathway.

    Args:
        t: Time (h)
        k_agg: Aggregation rate constant (1/h)
        k_deam: Deamidation rate constant (1/h)
        k_ox: Oxidation rate constant (1/h)
        k_frag: Fragmentation rate constant (1/h)

    Returns:
        Dict with fractions and total purity
    """
    f_agg = aggregate_fraction(t, k_agg)
    f_deam = deamidation_fraction(t, k_deam)
    f_ox = oxidation_fraction(t, k_ox)
    f_frag = fragmentation_fraction(t, k_frag)

    # Remaining native protein (assuming independent pathways)
    # Product of remaining fractions from each pathway
    f_native = (1.0 - f_agg) * (1.0 - f_deam) * (1.0 - f_ox) * (1.0 - f_frag)

    return {
        "aggregation": float(f_agg),
        "deamidation": float(f_deam),
        "oxidation": float(f_ox),
        "fragmentation": float(f_frag),
        "native_fraction": float(f_native),
        "total_degraded": float(1.0 - f_native),
    }


def shelf_life(
    target_purity: Array | float,
    k_total: Array | float,
) -> Array:
    """Calculate shelf life to reach target purity.

    t = -ln(target_purity) / k_total

    First-order decay model.

    Args:
        target_purity: Target remaining fraction (e.g., 0.95 for 95%)
        k_total: Total first-order degradation rate (1/h)

    Returns:
        Time to reach target purity (h)

    Example:
        >>> # Time to 95% purity with k=0.001/h
        >>> t = shelf_life(0.95, 0.001)  # ~51 hours
    """
    target_purity = jnp.asarray(target_purity)
    k_total = jnp.asarray(k_total)

    return -jnp.log(target_purity) / k_total


# =============================================================================
# Degradation Model Class
# =============================================================================

@dataclass(repr=False)
class DegradationParams(ParamsMixin):
    """Parameters for degradation model.

    Attributes:
        k_agg: Aggregation rate constant (1/h)
        k_deam: Deamidation rate constant (1/h)
        k_ox: Oxidation rate constant (1/h)
        k_frag: Fragmentation rate constant (1/h)
        E_a_agg: Aggregation activation energy (J/mol)
        E_a_deam: Deamidation activation energy (J/mol)
        T_ref: Reference temperature (K)
        pH_ref: Reference pH
    """
    k_agg: float | Array = 1e-4  # 1/h at reference
    k_deam: float | Array = 1e-5  # 1/h at reference
    k_ox: float | Array = 1e-5  # 1/h at reference
    k_frag: float | Array = 1e-6  # 1/h at reference
    E_a_agg: float | Array = 100000.0  # J/mol
    E_a_deam: float | Array = 85000.0  # J/mol
    T_ref: float | Array = 278.15  # 5°C storage
    pH_ref: float | Array = 6.0  # Typical formulation pH


class DegradationModel:
    """Unified interface for protein degradation modeling.

    Calculates degradation rates accounting for temperature,
    pH, and time effects on multiple pathways.

    Example:
        >>> model = DegradationModel(DegradationParams(k_agg=1e-4))
        >>> purity = model.predict_purity(t=720, T=278.15)  # 30 days at 5°C
    """

    def __init__(self, params: DegradationParams):
        """Initialize degradation model.

        Args:
            params: Degradation model parameters
        """
        self.params = params

    def aggregation_at_T(
        self,
        T: Array | float,
    ) -> Array:
        """Get aggregation rate at temperature T.

        Args:
            T: Temperature (K)

        Returns:
            Aggregation rate constant (1/h)
        """
        return aggregation_arrhenius(
            T,
            self.params.k_agg,
            self.params.E_a_agg,
            self.params.T_ref,
        )

    def deamidation_at_conditions(
        self,
        T: Array | float,
        pH: Array | float,
    ) -> Array:
        """Get deamidation rate at temperature and pH.

        Args:
            T: Temperature (K)
            pH: Solution pH

        Returns:
            Deamidation rate constant (1/h)
        """
        return deamidation_ph_dependent(
            pH,
            T,
            self.params.k_deam,
            self.params.pH_ref,
            self.params.T_ref,
            self.params.E_a_deam,
        )

    def predict_degradation(
        self,
        t: Array | float,
        T: Array | float = None,
        pH: Array | float = None,
    ) -> dict:
        """Predict degradation at given conditions.

        Args:
            t: Time (h)
            T: Temperature (K), uses T_ref if None
            pH: Solution pH, uses pH_ref if None

        Returns:
            Dict with degradation fractions
        """
        p = self.params

        if T is None:
            T = p.T_ref
        if pH is None:
            pH = p.pH_ref

        # Temperature-adjusted rates
        k_agg_T = self.aggregation_at_T(T)
        k_deam_T = self.deamidation_at_conditions(T, pH)

        # Simple Arrhenius for others (assume same E_a as aggregation)
        T_factor = jnp.exp(-p.E_a_agg / R * (1.0 / T - 1.0 / p.T_ref))
        k_ox_T = p.k_ox * T_factor
        k_frag_T = p.k_frag * T_factor

        return total_degradation(t, k_agg_T, k_deam_T, k_ox_T, k_frag_T)

    def predict_purity(
        self,
        t: Array | float,
        T: Array | float = None,
        pH: Array | float = None,
    ) -> Array:
        """Predict product purity at given time and conditions.

        Args:
            t: Time (h)
            T: Temperature (K)
            pH: Solution pH

        Returns:
            Native fraction (0-1)
        """
        result = self.predict_degradation(t, T, pH)
        return jnp.array(result["native_fraction"])

    def estimate_shelf_life(
        self,
        target_purity: Array | float = 0.95,
        T: Array | float = None,
    ) -> Array:
        """Estimate shelf life at storage temperature.

        Args:
            target_purity: Target remaining fraction
            T: Storage temperature (K)

        Returns:
            Estimated shelf life (h)
        """
        if T is None:
            T = self.params.T_ref

        # Get total rate at storage temperature
        k_total = (
            self.aggregation_at_T(T)
            + self.params.k_deam  # Simplified
            + self.params.k_ox
            + self.params.k_frag
        )

        return shelf_life(target_purity, k_total)


def get_degradation_model(
    **kwargs,
) -> DegradationModel:
    """Create degradation model.

    Args:
        **kwargs: Model parameters

    Returns:
        DegradationModel instance
    """
    params = DegradationParams(**kwargs)
    return DegradationModel(params)
