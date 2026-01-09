"""REE extraction unit operations.

Multi-stage counter-current extraction cascades for REE separation.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, lax

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.loading import LoadingIsotherm, get_loading_isotherm
from difflow_ree.equilibrium.speciation import REESpeciation


# =============================================================================
# REE Extractor Parameters
# =============================================================================

@dataclass(repr=False)
class REEExtractorParams(ParamsMixin):
    """Parameters for REE extraction cascade.

    Attributes:
        n_stages: Number of extraction stages
        extractant: Extractant name (D2EHPA, PC88A, etc.)
        elements: REE elements to track
        pH: Operating pH (typically 1-5 for REE extraction)
        extractant_conc: Extractant concentration (M)
        include_loading: Whether to account for extractant loading
        include_speciation: Whether to account for aqueous speciation
        speciation_medium: Aqueous medium type for speciation
        ligand_conc: Ligand concentration for speciation (M)
    """
    n_stages: int | float | Array
    extractant: str
    elements: tuple[str, ...]
    pH: float | Array = 3.0
    extractant_conc: float = 0.5
    include_loading: bool = True
    include_speciation: bool = False
    speciation_medium: str = "sulfate"
    ligand_conc: float = 0.5

    def __post_init__(self):
        """Validate extractor parameters."""
        from difflow_ree.database import get_extractant_database, get_ree_database

        # Validate extractant exists
        extractant_db = get_extractant_database()
        valid_extractants = extractant_db.list_extractants()
        if self.extractant not in valid_extractants:
            raise ValueError(
                f"Unknown extractant: '{self.extractant}'. "
                f"Available: {valid_extractants}"
            )

        # Validate elements are valid REE
        ree_db = get_ree_database()
        valid_elements = ree_db.list_elements()
        for elem in self.elements:
            if elem not in valid_elements:
                raise ValueError(
                    f"Unknown REE element: '{elem}'. "
                    f"Valid elements: {valid_elements}"
                )

        # Validate bounds
        if hasattr(self.n_stages, '__float__'):
            if float(self.n_stages) < 1:
                raise ValueError(f"n_stages must be >= 1, got {self.n_stages}")
        if self.extractant_conc <= 0:
            raise ValueError(
                f"extractant_conc must be > 0, got {self.extractant_conc}"
            )


class REEExtractor:
    """Multi-stage REE extraction cascade.

    Counter-current extraction using the Kremser equation
    with optional loading and speciation corrections.

    Example:
        >>> params = REEExtractorParams(
        ...     n_stages=10,
        ...     extractant="D2EHPA",
        ...     elements=("La", "Ce", "Nd", "Dy"),
        ...     pH=3.0,
        ... )
        >>> extractor = REEExtractor(params)
        >>> raffinate, extract, info = extractor(feed, solvent)
    """

    def __init__(self, params: REEExtractorParams):
        """Initialize extractor.

        Args:
            params: Extractor parameters
        """
        self.params = params
        self._distribution = REEDistribution(
            extractant=params.extractant,
            elements=params.elements,
            concentration=params.extractant_conc,
        )
        if params.include_loading:
            self._isotherm = get_loading_isotherm(
                params.extractant,
                params.extractant_conc,
            )
        else:
            self._isotherm = None

        if params.include_speciation:
            self._speciation = REESpeciation(
                elements=params.elements,
                medium=params.speciation_medium,
            )
        else:
            self._speciation = None

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float = 298.15,
        pH: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform multi-stage extraction.

        Args:
            feed: Aqueous feed stream (REE solution)
            solvent: Organic solvent stream
            T: Temperature (K)
            pH: Operating pH (overrides params if provided)

        Returns:
            raffinate: Aqueous outlet (depleted)
            extract: Organic outlet (loaded)
            info: Stage profiles and diagnostics
        """
        p = self.params
        pH = pH if pH is not None else p.pH
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)

        # Get aqueous and organic carrier flows
        # Assume "H2O" is aqueous carrier, "Organic" is organic carrier
        F_aq = feed_flows.get("H2O", 1.0)
        F_org = solvent_flows.get("Organic", 1.0)

        # Get distribution coefficients
        D_values = self._distribution.get_D_all(pH, T)

        # Apply speciation correction if enabled
        if self._speciation is not None:
            for elem in p.elements:
                alpha = self._speciation.free_fraction(elem, p.ligand_conc, pH)
                D_values[elem] = D_values[elem] * alpha

        # Solve extraction using Kremser equation
        n_stages = jnp.asarray(p.n_stages, dtype=jnp.float64)

        raffinate_flows = {"H2O": F_aq}
        extract_flows = {"Organic": F_org}
        stage_profiles = {}

        for elem in p.elements:
            D = D_values[elem]
            F_in = jnp.asarray(feed_flows.get(elem, 0.0))
            F_solvent = jnp.asarray(solvent_flows.get(elem, 0.0))

            # Apply loading correction if enabled
            if self._isotherm is not None:
                # Estimate average loading (iterative would be more accurate)
                avg_loading = F_in * 0.5 / F_org  # Rough estimate
                D = self._isotherm.apparent_D(D, avg_loading)

            # Extraction factor E = D * (F_org / F_aq)
            E = D * F_org / F_aq

            # Kremser equation for counter-current extraction
            E_Np1 = jnp.power(E, n_stages + 1)

            frac_remaining = jnp.where(
                jnp.abs(E - 1.0) < 1e-6,
                1.0 / (n_stages + 1),
                (E - 1.0) / (E_Np1 - 1.0 + 1e-10)
            )
            frac_remaining = jnp.clip(frac_remaining, 0.0, 1.0)

            F_raffinate = F_in * frac_remaining + F_solvent * frac_remaining
            F_extract = F_in + F_solvent - F_raffinate

            raffinate_flows[elem] = jnp.maximum(F_raffinate, 0.0)
            extract_flows[elem] = jnp.maximum(F_extract, 0.0)

            stage_profiles[elem] = {
                "D": D,
                "E": E,
                "recovery": 1.0 - frac_remaining,
            }

        # Create output streams
        P = feed["P"]
        raffinate = make_stream(raffinate_flows, T, P)
        extract = make_stream(extract_flows, T, P)

        info = {
            "n_stages": n_stages,
            "pH": pH,
            "T": T,
            "profiles": stage_profiles,
            "D_values": D_values,
        }

        return raffinate, extract, info


# =============================================================================
# Mixer-Settler Unit
# =============================================================================

@dataclass(repr=False)
class MixerSettlerParams(ParamsMixin):
    """Parameters for single mixer-settler stage.

    Attributes:
        extractant: Extractant name
        elements: REE elements to track
        pH: Operating pH
        extractant_conc: Extractant concentration (M)
        mixer_residence_time: Mixer residence time (s)
        settler_residence_time: Settler residence time (s)
        stage_efficiency: Murphree stage efficiency (0-1)
    """
    extractant: str
    elements: tuple[str, ...]
    pH: float = 3.0
    extractant_conc: float = 0.5
    mixer_residence_time: float = 120.0  # 2 minutes typical
    settler_residence_time: float = 300.0  # 5 minutes typical
    stage_efficiency: float = 0.95


class REEMixerSettler:
    """Single mixer-settler stage for REE extraction.

    Models one equilibrium stage with efficiency factor.

    Example:
        >>> params = MixerSettlerParams(
        ...     extractant="D2EHPA",
        ...     elements=("Nd", "Dy"),
        ...     pH=3.0,
        ... )
        >>> stage = REEMixerSettler(params)
        >>> aq_out, org_out, info = stage(aq_in, org_in)
    """

    def __init__(self, params: MixerSettlerParams):
        """Initialize mixer-settler.

        Args:
            params: Stage parameters
        """
        self.params = params
        self._distribution = REEDistribution(
            extractant=params.extractant,
            elements=params.elements,
            concentration=params.extractant_conc,
        )

    def __call__(
        self,
        aqueous_in: Stream,
        organic_in: Stream,
        T: Array | float = 298.15,
        pH: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform single-stage extraction.

        Args:
            aqueous_in: Inlet aqueous stream
            organic_in: Inlet organic stream
            T: Temperature (K)
            pH: Operating pH

        Returns:
            aqueous_out: Outlet aqueous stream
            organic_out: Outlet organic stream
            info: Stage information
        """
        p = self.params
        pH = pH if pH is not None else p.pH
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)

        aq_flows = get_flows(aqueous_in)
        org_flows = get_flows(organic_in)

        F_aq = aq_flows.get("H2O", 1.0)
        F_org = org_flows.get("Organic", 1.0)

        D_values = self._distribution.get_D_all(pH, T)

        aq_out_flows = {"H2O": F_aq}
        org_out_flows = {"Organic": F_org}

        for elem in p.elements:
            D = D_values[elem]
            F_aq_in = jnp.asarray(aq_flows.get(elem, 0.0))
            F_org_in = jnp.asarray(org_flows.get(elem, 0.0))

            # Total solute
            F_total = F_aq_in + F_org_in

            # Equilibrium distribution
            # At equilibrium: c_org = D * c_aq
            # Mass balance: F_aq * c_aq + F_org * c_org = F_total
            # Solving: c_aq = F_total / (F_aq + D * F_org)
            c_aq_eq = F_total / (F_aq + D * F_org)
            c_org_eq = D * c_aq_eq

            F_aq_eq = c_aq_eq * F_aq
            F_org_eq = c_org_eq * F_org

            # Apply stage efficiency
            eta = p.stage_efficiency
            F_aq_out = F_aq_in + eta * (F_aq_eq - F_aq_in)
            F_org_out = F_org_in + eta * (F_org_eq - F_org_in)

            aq_out_flows[elem] = jnp.maximum(F_aq_out, 0.0)
            org_out_flows[elem] = jnp.maximum(F_org_out, 0.0)

        P = aqueous_in["P"]
        aqueous_out = make_stream(aq_out_flows, T, P)
        organic_out = make_stream(org_out_flows, T, P)

        info = {
            "D_values": D_values,
            "efficiency": p.stage_efficiency,
        }

        return aqueous_out, organic_out, info


# =============================================================================
# Multi-Stage Cascade Builder
# =============================================================================

def build_extraction_cascade(
    n_stages: int,
    params: MixerSettlerParams,
) -> list[REEMixerSettler]:
    """Build a list of mixer-settler stages.

    Args:
        n_stages: Number of stages
        params: Parameters for each stage

    Returns:
        List of mixer-settler units
    """
    return [REEMixerSettler(params) for _ in range(n_stages)]


def solve_cascade_sequential(
    stages: list[REEMixerSettler],
    feed: Stream,
    solvent: Stream,
    T: float = 298.15,
    pH: float | None = None,
) -> tuple[Stream, Stream, list[dict]]:
    """Solve co-current cascade sequentially.

    Args:
        stages: List of mixer-settler stages
        feed: Aqueous feed
        solvent: Organic solvent
        T: Temperature (K)
        pH: Operating pH

    Returns:
        raffinate: Final aqueous outlet
        extract: Final organic outlet
        stage_info: List of stage information dicts
    """
    aq = feed
    org = solvent
    stage_info = []

    for stage in stages:
        aq, org, info = stage(aq, org, T, pH)
        stage_info.append(info)

    return aq, org, stage_info
