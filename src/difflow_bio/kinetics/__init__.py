"""Kinetics module for biopharmaceutical processes.

Provides modular, reusable kinetic models for:
- Cell growth kinetics (Monod, Contois, logistic, Tessier, Moser, Andrews)
- Product formation kinetics (Luedeking-Piret, growth/non-growth associated)
- Substrate uptake and metabolism
- Oxygen consumption and CO2 evolution

All functions are JAX-compatible for automatic differentiation.

References:
    Monod J (1949). Annu Rev Microbiol 3:371.
    Luedeking R, Piret EL (1959). J Biochem Microbiol Technol Eng 1:393.
    Contois DE (1959). J Gen Microbiol 21:40.
    Pirt SJ (1965). Proc R Soc Lond B Biol Sci 163:224.
"""

from difflow_bio.kinetics.growth import (
    monod,
    monod_inhibition,
    contois,
    logistic,
    tessier,
    moser,
    andrews,
    death_rate,
    net_growth_rate,
    GrowthModel,
    GrowthModelParams,
    get_growth_model,
)

from difflow_bio.kinetics.production import (
    luedeking_piret,
    growth_associated,
    non_growth_associated,
    overflow_production,
    product_inhibited_production,
    substrate_limited_production,
    ProductionModel,
    ProductionModelParams,
    get_production_model,
)

from difflow_bio.kinetics.metabolism import (
    substrate_uptake_rate,
    oxygen_uptake_rate,
    co2_evolution_rate,
    maintenance_energy,
    specific_substrate_uptake,
    yield_coefficient,
    metabolic_quotient,
    MetabolismModel,
    MetabolismParams,
    get_metabolism_model,
)

__all__ = [
    # Growth
    "monod",
    "monod_inhibition",
    "contois",
    "logistic",
    "tessier",
    "moser",
    "andrews",
    "death_rate",
    "net_growth_rate",
    "GrowthModel",
    "GrowthModelParams",
    "get_growth_model",
    # Production
    "luedeking_piret",
    "growth_associated",
    "non_growth_associated",
    "overflow_production",
    "product_inhibited_production",
    "substrate_limited_production",
    "ProductionModel",
    "ProductionModelParams",
    "get_production_model",
    # Metabolism
    "substrate_uptake_rate",
    "oxygen_uptake_rate",
    "co2_evolution_rate",
    "maintenance_energy",
    "specific_substrate_uptake",
    "yield_coefficient",
    "metabolic_quotient",
    "MetabolismModel",
    "MetabolismParams",
    "get_metabolism_model",
]
