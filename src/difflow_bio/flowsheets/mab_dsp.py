"""Standard monoclonal antibody downstream processing train.

Industry-standard 3-column process:
1. Protein A capture (affinity)
2. Cation exchange polish (CEX)
3. Anion exchange flow-through (AEX)

With TFF for concentration/buffer exchange between steps.

    Harvest
       ↓
  ┌─────────┐
  │ Protein │
  │    A    │ ← Capture
  └────┬────┘
       ↓
  ┌─────────┐
  │  TFF    │ ← Concentrate/Buffer Exchange
  └────┬────┘
       ↓
  ┌─────────┐
  │  CEX    │ ← Intermediate Polish
  └────┬────┘
       ↓
  ┌─────────┐
  │  AEX    │ ← Final Polish (flow-through)
  └────┬────┘
       ↓
  ┌─────────┐
  │  TFF    │ ← Final Formulation
  └────┬────┘
       ↓
    Product

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows
from difflow_bio.units.chromatography import (
    ProteinAChromatography, ProteinAParams,
    IonExchangeChromatography, IEXParams,
)
from difflow_bio.units.filtration import TFF
from difflow.numerics import safe_divide


@dataclass(repr=False)
class mAbDSPParams(ParamsMixin):
    """Parameters for mAb DSP train.

    Attributes:
        species_order: List of species names
        target_species: Name of mAb species
        proa_column_volume: Protein A column volume (L)
        proa_q_max: Protein A binding capacity (g/L)
        cex_column_volume: CEX column volume (L)
        cex_q_max: CEX binding capacity (g/L)
        aex_column_volume: AEX column volume (L)
        tff_area: TFF membrane area (m²)
        concentration_factor: Target concentration factor
    """
    species_order: list[str] = None
    target_species: str = "mAb"
    proa_column_volume: float | Array = 10.0
    proa_q_max: float | Array = 35.0
    proa_K_d: float | Array = 0.1
    proa_yield: float | Array = 0.95
    cex_column_volume: float | Array = 20.0
    cex_q_max: float | Array = 50.0
    cex_K_d: float | Array = 0.5
    cex_yield: float | Array = 0.90
    aex_column_volume: float | Array = 15.0
    aex_yield: float | Array = 0.95
    tff_area: float | Array = 5.0
    concentration_factor: float | Array = 10.0
    final_concentration_g_L: float | Array = 100.0


class mAbDSPTrain:
    """Standard monoclonal antibody downstream processing train.

    Industry-standard platform process with:
    - Protein A capture
    - CEX intermediate polish
    - AEX final polish (flow-through mode)
    - TFF for concentration/formulation

    Example:
        >>> params = mAbDSPParams(
        ...     species_order=["mAb", "HCP", "DNA", "aggregates"],
        ...     proa_column_volume=10.0,
        ... )
        >>> train = mAbDSPTrain(params)
        >>> results = train(harvest)
        >>> print(f"Overall yield: {results['overall_yield']:.1%}")
    """

    def __init__(self, params: mAbDSPParams):
        """Initialize DSP train.

        Args:
            params: DSP train parameters
        """
        self.params = params

        # Protein A capture
        self._proa = ProteinAChromatography(ProteinAParams(
            column_volume=params.proa_column_volume,
            q_max=params.proa_q_max,
            K_d=params.proa_K_d,
            target_species=params.target_species,
            yield_factor=params.proa_yield,
            species_order=params.species_order,
        ))

        # CEX intermediate polish (bind-elute)
        self._cex = IonExchangeChromatography(IEXParams(
            column_volume=params.cex_column_volume,
            mode="bind_elute",
            q_max=params.cex_q_max,
            K_d=params.cex_K_d,
            target_species=params.target_species,
            yield_factor=params.cex_yield,
            selectivity={params.target_species: 1.0, "aggregates": 0.3},
            species_order=params.species_order,
        ))

        # AEX final polish (flow-through)
        self._aex = IonExchangeChromatography(IEXParams(
            column_volume=params.aex_column_volume,
            mode="flow_through",
            target_species=params.target_species,
            yield_factor=params.aex_yield,
            selectivity={params.target_species: 0.0, "HCP": 0.9, "DNA": 1.0},
            species_order=params.species_order,
        ))

        # TFF for concentration
        self._tff = TFF(
            membrane_area=params.tff_area,
            MWCO=30.0,
            rejection={params.target_species: 0.995},
        )

    def __call__(
        self,
        harvest: Stream,
        return_intermediates: bool = False,
    ) -> dict:
        """Run complete DSP train.

        Args:
            harvest: Clarified harvest stream
            return_intermediates: Return streams from each step

        Returns:
            Dictionary with:
            - product: Final product stream
            - overall_yield: Overall mAb yield
            - step_yields: Yield from each step
            - purity: Final product purity (optional)
            - intermediates: Intermediate streams (if requested)
        """
        p = self.params
        target = p.target_species

        harvest_flows = get_flows(harvest)
        mab_in = harvest_flows.get(target, 0.0)

        intermediates = {"harvest": harvest}

        # Step 1: Protein A capture
        (proa_eluate, proa_waste), proa_info = self._proa(harvest, load_volume=p.proa_column_volume)
        proa_flows = get_flows(proa_eluate)
        proa_yield = safe_divide(proa_flows.get(target, 0.0), mab_in)
        intermediates["proa_eluate"] = proa_eluate

        # Step 2: TFF concentration (post-ProA)
        (tff1_out, _tff1_perm), tff1_info = self._tff.concentrate(
            proa_eluate,
            concentration_factor=p.concentration_factor,
        )
        intermediates["tff1_concentrate"] = tff1_out

        # Step 3: CEX polish
        (cex_eluate, cex_waste), cex_info = self._cex(tff1_out, load_volume=p.cex_column_volume)
        cex_flows = get_flows(cex_eluate)
        cex_yield = safe_divide(cex_flows.get(target, 0.0), proa_flows.get(target, 0.0))
        intermediates["cex_eluate"] = cex_eluate

        # Step 4: AEX flow-through polish
        (aex_product, aex_bound), aex_info = self._aex(cex_eluate, load_volume=p.aex_column_volume)
        aex_flows = get_flows(aex_product)
        aex_yield = safe_divide(aex_flows.get(target, 0.0), cex_flows.get(target, 0.0))
        intermediates["aex_product"] = aex_product

        # Step 5: Final TFF formulation
        (final_product, _tff2_perm), tff2_info = self._tff.concentrate(
            aex_product,
            concentration_factor=safe_divide(p.final_concentration_g_L, aex_flows.get(target, 1.0)),
        )
        intermediates["final_product"] = final_product

        # Calculate overall metrics
        final_flows = get_flows(final_product)
        mab_out = final_flows.get(target, 0.0)
        overall_yield = safe_divide(mab_out, mab_in)

        # Purity (mAb as fraction of total protein)
        total_protein = sum(
            float(final_flows.get(s, 0.0))
            for s in [target, "HCP", "aggregates"]
            if s in final_flows or s == target
        )
        purity = safe_divide(float(mab_out), total_protein)

        result = {
            "product": final_product,
            "overall_yield": float(overall_yield),
            "step_yields": {
                "proa": float(proa_yield),
                "cex": float(cex_yield),
                "aex": float(aex_yield),
            },
            "purity": purity,
        }

        if return_intermediates:
            result["intermediates"] = intermediates

        return result

    def calculate_resin_usage(
        self,
        harvest: Stream,
        batches_per_year: int = 50,
    ) -> dict:
        """Calculate annual resin usage.

        Args:
            harvest: Harvest stream
            batches_per_year: Annual batch count

        Returns:
            Resin usage and cost estimates
        """
        p = self.params
        harvest_flows = get_flows(harvest)
        mab_per_batch = float(harvest_flows.get(p.target_species, 0.0))

        # Load calculations
        proa_load = mab_per_batch / (p.proa_q_max * 0.8)  # 80% of max
        cex_load = mab_per_batch * float(p.proa_yield) / (p.cex_q_max * 0.8)

        # Cycles per batch (if column is smaller than needed)
        proa_cycles = max(1, int(jnp.ceil(proa_load / p.proa_column_volume)))
        cex_cycles = max(1, int(jnp.ceil(cex_load / p.cex_column_volume)))

        annual_proa_cycles = proa_cycles * batches_per_year
        annual_cex_cycles = cex_cycles * batches_per_year

        return {
            "proa_cycles_per_batch": proa_cycles,
            "cex_cycles_per_batch": cex_cycles,
            "annual_proa_cycles": annual_proa_cycles,
            "annual_cex_cycles": annual_cex_cycles,
            "proa_column_volume_L": float(p.proa_column_volume),
            "cex_column_volume_L": float(p.cex_column_volume),
        }


def design_mab_dsp(
    harvest_volume_L: float,
    harvest_titer_g_L: float,
    annual_batches: int = 50,
    target_yield: float = 0.70,
) -> mAbDSPParams:
    """Design mAb DSP train for given harvest.

    Args:
        harvest_volume_L: Harvest volume per batch (L)
        harvest_titer_g_L: mAb titer in harvest (g/L)
        annual_batches: Batches per year
        target_yield: Target overall yield

    Returns:
        Recommended mAbDSPParams
    """
    mab_per_batch = harvest_volume_L * harvest_titer_g_L

    # Size Protein A for ~20 g/L load (10 cycles max per batch)
    proa_load_target = 20.0  # g/L
    proa_cv = mab_per_batch / proa_load_target

    # CEX sized for concentrated eluate (~5x concentration)
    cex_cv = proa_cv * 0.8  # Slightly smaller

    # AEX flow-through, size for throughput
    aex_cv = proa_cv * 0.6

    # TFF area: ~50 L/m²/h flux, 4h processing
    tff_area = harvest_volume_L / (50 * 4)

    return mAbDSPParams(
        species_order=["mAb", "HCP", "DNA", "aggregates", "fragments"],
        target_species="mAb",
        proa_column_volume=proa_cv,
        cex_column_volume=cex_cv,
        aex_column_volume=aex_cv,
        tff_area=max(1.0, tff_area),
    )
