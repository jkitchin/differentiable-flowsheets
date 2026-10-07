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

from dataclasses import dataclass, field

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows
from difflow_bio.units.chromatography import (
    ProteinAChromatography, ProteinAParams,
    IonExchangeChromatography, IEXParams,
    TYPICAL_CEX_CLEARANCE, TYPICAL_AEX_CLEARANCE,
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
        cex_column_volume: CEX column volume (L); with cex_q_max it caps
            the mAb the column binds, and it sets the CEX pool volume.
        cex_q_max: CEX binding capacity (g/L)
        aex_column_volume: AEX column volume (L); with aex_q_max it caps
            the impurity the flow-through column can hold.
        aex_q_max: AEX binding capacity (g/L)
        tff_area: TFF membrane area (m²). Sets the TFF processing times
            reported in ``tff_process_time_h``; the split does not depend
            on it.
        concentration_factor: Concentration factor of the post-Protein A
            TFF step
        final_concentration_g_L: mAb concentration the final TFF step
            concentrates the AEX pool to (g/L)
        proa_elution_cv: Protein A elution pool, in column volumes
            (5 CV, Petrides 2015 sec. 11.6.3, p. 65); the first TFF's
            feed volume.
        cex_elution_cv: CEX gradient elution pool, in column volumes
            (5 CV, same source); the AEX load and pool volume, so the
            final TFF's feed volume.
        cex_clearance: CEX impurity -> LRV. Defaults to
            TYPICAL_CEX_CLEARANCE, representative values, not measured ones.
        aex_clearance: AEX impurity -> LRV. Defaults to
            TYPICAL_AEX_CLEARANCE.
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
    aex_q_max: float | Array = 50.0
    aex_yield: float | Array = 0.95
    tff_area: float | Array = 5.0
    concentration_factor: float | Array = 10.0
    final_concentration_g_L: float | Array = 100.0
    proa_elution_cv: float | Array = 5.0
    cex_elution_cv: float | Array = 5.0
    cex_clearance: dict = field(default_factory=lambda: dict(TYPICAL_CEX_CLEARANCE))
    aex_clearance: dict = field(default_factory=lambda: dict(TYPICAL_AEX_CLEARANCE))


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
            impurity_clearance=params.cex_clearance,
            species_order=params.species_order,
        ))

        # AEX final polish (flow-through)
        self._aex = IonExchangeChromatography(IEXParams(
            column_volume=params.aex_column_volume,
            q_max=params.aex_q_max,
            mode="flow_through",
            target_species=params.target_species,
            yield_factor=params.aex_yield,
            selectivity={params.target_species: 0.0, "HCP": 0.9, "DNA": 1.0},
            impurity_clearance=params.aex_clearance,
            species_order=params.species_order,
        ))

        # TFF for concentration
        self._tff = TFF(
            membrane_area=params.tff_area,
            MWCO=30.0,
            # UF/DF is not credited with impurity clearance: HCP and DNA are retained
            # like the product, and aggregates, larger than the monomer, slightly
            # better. Unlisted species default to zero rejection, which washed
            # 90% of the HCP and aggregates out in the 10x concentration.
            rejection={params.target_species: 0.995, "HCP": 0.995, "DNA": 0.995,
                       "aggregates": 0.999},
        )

    def __call__(
        self,
        harvest: Stream,
        return_intermediates: bool = False,
    ) -> dict:
        """Run complete DSP train.

        Stream amounts are taken as grams per batch. Volumes, which the
        streams do not carry, come from the columns: the Protein A and CEX
        elution pools are ``proa_elution_cv`` and ``cex_elution_cv`` column
        volumes, and the AEX flow-through pool is taken equal to its load
        (the CEX pool). The final TFF concentrates the AEX pool to
        ``final_concentration_g_L``.

        Args:
            harvest: Clarified harvest stream
            return_intermediates: Return streams from each step

        Returns:
            Dictionary with:
            - product: Final product stream
            - side_streams: Every other outlet (``proa_waste``,
              ``tff1_permeate``, ``cex_waste``, ``aex_bound``,
              ``tff2_permeate``); product + side streams = harvest for
              every species.
            - overall_yield: Overall mAb yield
            - step_yields: Each step's mAb out over its own mAb in
              (``proa``, ``tff1``, ``cex``, ``aex``, ``tff2``)
            - purity, hcp_ppm, aggregate_fraction
            - final_concentration_g_L, final_tff_concentration_factor
            - tff_process_time_h: hours per TFF step at the computed flux
              over ``tff_area``
            - intermediates: Intermediate streams (if requested)

        Example:
            >>> res = mAbDSPTrain(mAbDSPParams(species_order=SP))(harvest)
            >>> res["product"], res["side_streams"]["proa_waste"]
        """
        p = self.params
        target = p.target_species

        def amount(stream):
            return get_flows(stream).get(target, jnp.asarray(0.0))

        mab_in = amount(harvest)
        intermediates = {"harvest": harvest}

        # Step 1: Protein A capture. Each column loads the whole batch; the
        # column volume sets its capacity, not how much of the feed it sees.
        (proa_eluate, proa_waste), proa_info = self._proa(harvest)
        intermediates["proa_eluate"] = proa_eluate
        proa_pool_L = p.proa_elution_cv * p.proa_column_volume

        # Step 2: TFF concentration (post-ProA)
        (tff1_out, tff1_perm), tff1_info = self._tff.concentrate(
            proa_eluate,
            concentration_factor=p.concentration_factor,
            feed_volume=proa_pool_L,
        )
        intermediates["tff1_concentrate"] = tff1_out

        # Step 3: CEX polish
        (cex_eluate, cex_waste), cex_info = self._cex(tff1_out)
        intermediates["cex_eluate"] = cex_eluate
        cex_pool_L = p.cex_elution_cv * p.cex_column_volume

        # Step 4: AEX flow-through polish
        (aex_product, aex_bound), aex_info = self._aex(cex_eluate)
        intermediates["aex_product"] = aex_product

        # Step 5: Final TFF formulation. Bio audit (a): the concentration
        # factor was final_concentration_g_L / mAb amount, g/L over g, so
        # a 1000 g harvest got CF = 0.12. CF is a volume ratio: the AEX
        # pool volume over the volume that holds its mAb at the target
        # concentration; a pool already above the target is not diluted.
        target_volume_L = safe_divide(amount(aex_product), p.final_concentration_g_L)
        cf_final = jnp.maximum(safe_divide(cex_pool_L, target_volume_L), 1.0)
        (final_product, tff2_perm), tff2_info = self._tff.concentrate(
            aex_product,
            concentration_factor=cf_final,
            feed_volume=cex_pool_L,
        )
        intermediates["final_product"] = final_product

        # Calculate overall metrics
        final_flows = get_flows(final_product)
        mab_out = amount(final_product)
        overall_yield = safe_divide(mab_out, mab_in)

        # Purity (mAb as fraction of total protein). Values stay JAX arrays
        # so the train can be differentiated, jitted and vmapped (#360).
        total_protein = sum(
            final_flows.get(s, 0.0) for s in (target, "HCP", "aggregates")
        )
        purity = safe_divide(mab_out, total_protein)
        aggregates_out = final_flows.get("aggregates", 0.0)

        result = {
            "product": final_product,
            # Bio audit (a): these five outlets were computed and dropped,
            # with 19.8% of a 100 g harvest and all the cleared impurity.
            "side_streams": {
                "proa_waste": proa_waste,
                "tff1_permeate": tff1_perm,
                "cex_waste": cex_waste,
                "aex_bound": aex_bound,
                "tff2_permeate": tff2_perm,
            },
            "overall_yield": overall_yield,
            # Each step against its own inlet (bio audit (a): cex was taken
            # against the ProA eluate, absorbing the TFF1 loss, and the
            # final TFF loss appeared in no step).
            "step_yields": {
                "proa": safe_divide(amount(proa_eluate), mab_in),
                "tff1": safe_divide(amount(tff1_out), amount(proa_eluate)),
                "cex": safe_divide(amount(cex_eluate), amount(tff1_out)),
                "aex": safe_divide(amount(aex_product), amount(cex_eluate)),
                "tff2": safe_divide(mab_out, amount(aex_product)),
            },
            "purity": purity,
            # The two numbers a drug substance specification is written in
            "hcp_ppm": 1e6 * safe_divide(final_flows.get("HCP", 0.0), mab_out),
            "aggregate_fraction": safe_divide(aggregates_out, mab_out + aggregates_out),
            "final_tff_concentration_factor": cf_final,
            "final_concentration_g_L": safe_divide(mab_out, cex_pool_L / cf_final),
            "tff_process_time_h": {
                "tff1": tff1_info["process_time_h"],
                "tff2": tff2_info["process_time_h"],
            },
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

        A reporting helper: cycle counts are integers (ceil), so this is
        not differentiable and is not meant to be traced.

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
