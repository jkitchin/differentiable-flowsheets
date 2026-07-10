"""REE extraction unit operations.

Multi-stage counter-current extraction cascades for REE separation.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, lax

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.loading import LoadingIsotherm, get_loading_isotherm
from difflow_ree.equilibrium.speciation import REESpeciation
from difflow_ree.kinetics.extraction_kinetics import approach_to_equilibrium


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
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
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
    diluent: str = "kerosene"
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

    symbol = "REE Extraction"
    equations = [
        r"\mathrm{RE}^{3+}_{(aq)} + 3\,\overline{\mathrm{HA}} \rightleftharpoons \overline{\mathrm{REA}_3} + 3\,\mathrm{H}^+\qquad \text{(cation-exchange extractant)}",
        r"K_{\mathrm{ex},i} = \frac{[\overline{\mathrm{REA}_3}]\,[\mathrm{H}^+]^3}{[\mathrm{RE}^{3+}]\,[\overline{\mathrm{HA}}]^3}",
        r"D_i = K_{\mathrm{ex},i}\,\frac{[\overline{\mathrm{HA}}]^3}{[\mathrm{H}^+]^3}",
        r"\frac{x_{N+1}}{x_1} = \frac{E-1}{E^{N+1}-1}\qquad \text{(Kremser, counter-current)}",
    ]
    assumptions = [
        "Counter-current equilibrium stages with constant phase flows.",
        "Distribution coefficients depend on pH and optional extractant loading (Langmuir).",
        "Aqueous and organic phases immiscible; no third phase.",
    ]
    references = [
        "Rydberg, J., Musikas, C., Choppin, G.R. Principles and Practices of Solvent Extraction, Marcel Dekker, 1992.",
        "Xie, F., Zhang, T.A., Dreisinger, D., Doyle, F. Miner. Eng., 56, 10 (2014).",
    ]
    parameter_symbols = {
        "n_stages": "N",
        "extractant_conc": "[HA]",
        "pH": r"\mathrm{pH}",
        "O_A_ratio": "O/A",
    }
    parameter_units = {
        "extractant_conc": "mol/L",
        "pH": "-",
        "O_A_ratio": "-",
    }
    numerical_method = "Kremser closed form with pH-dependent distribution ratios; optional loading via Langmuir isotherm."

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
        # Use total aqueous flow (all species not in organic phase)
        # For concentrated solutions, using only H2O underestimates F_aq
        organic_species = {p.extractant, p.diluent}
        F_aq = sum(
            jnp.asarray(v) for k, v in feed_flows.items()
            if k not in organic_species
        )
        F_aq = jnp.maximum(F_aq, 1e-10)  # Prevent division by zero
        F_extractant = solvent_flows.get(p.extractant, 0.0)
        F_diluent = solvent_flows.get(p.diluent, 1.0)
        F_org = F_extractant + F_diluent

        # Get distribution coefficients
        D_values = self._distribution.get_D_all(pH, T)

        # Apply speciation correction if enabled
        if self._speciation is not None:
            for elem in p.elements:
                alpha = self._speciation.free_fraction(elem, p.ligand_conc, pH)
                D_values[elem] = D_values[elem] * alpha

        # Solve extraction using Kremser equation
        n_stages = jnp.asarray(p.n_stages, dtype=jnp.float64)

        raffinate_flows = dict(feed_flows)
        extract_flows = dict(solvent_flows)
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

            # Adjust for initial organic loading (loaded solvent reduces capacity)
            if self._isotherm is not None:
                initial_loading = F_solvent / jnp.maximum(F_org, 1e-10)
                capacity = self._isotherm.max_ree_conc
                E = E * jnp.maximum(1.0 - initial_loading / capacity, 0.0)
            else:
                # Simple loading correction without isotherm:
                # Reduce E based on ratio of existing loading to feed
                loading_ratio = F_solvent / jnp.maximum(F_in + F_solvent, 1e-10)
                E = E * (1.0 - loading_ratio)

            # Kremser equation for counter-current extraction
            E_Np1 = jnp.power(E, n_stages + 1)

            frac_remaining = jnp.where(
                jnp.abs(E - 1.0) < 1e-6,
                1.0 / (n_stages + 1),
                safe_divide(E - 1.0, E_Np1 - 1.0)
            )
            frac_remaining = jnp.clip(frac_remaining, 0.0, 1.0)

            F_raffinate = F_in * frac_remaining
            F_extract = F_solvent + F_in * (1.0 - frac_remaining)

            raffinate_flows[elem] = jnp.maximum(F_raffinate, 0.0)
            extract_flows[elem] = jnp.maximum(F_extract, 0.0)

            stage_profiles[elem] = {
                "D": D,
                "E": E,
                "recovery": 1.0 - frac_remaining,
            }

        # Enforce total extractant loading capacity (Bug #112)
        # The Kremser equation can predict extraction beyond physical capacity
        # when multiple REE are extracted simultaneously
        if self._isotherm is not None:
            total_newly_extracted = sum(
                extract_flows[elem] - jnp.asarray(solvent_flows.get(elem, 0.0))
                for elem in p.elements
            )
            max_capacity = self._isotherm.max_ree_conc * F_org
            scale = jnp.where(
                total_newly_extracted > max_capacity,
                max_capacity / jnp.maximum(total_newly_extracted, 1e-10),
                1.0,
            )
            for elem in p.elements:
                F_solvent_elem = jnp.asarray(solvent_flows.get(elem, 0.0))
                newly_extracted = extract_flows[elem] - F_solvent_elem
                extract_flows[elem] = F_solvent_elem + newly_extracted * scale
                raffinate_flows[elem] = (
                    jnp.asarray(feed_flows.get(elem, 0.0))
                    + F_solvent_elem
                    - extract_flows[elem]
                )

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
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        pH: Operating pH
        extractant_conc: Extractant concentration (M)
        mixer_residence_time: Mixer residence time (s)
        settler_residence_time: Settler residence time (s)
        stage_efficiency: Murphree stage efficiency (0-1)
    """
    extractant: str
    elements: tuple[str, ...]
    diluent: str = "kerosene"
    pH: float = 3.0
    extractant_conc: float = 0.5
    mixer_residence_time: float = 120.0  # 2 minutes typical
    settler_residence_time: float = 300.0  # 5 minutes typical
    stage_efficiency: float = 0.95
    # Extraction kinetics (#118). When k_extraction (overall rate constant,
    # 1/s) is set, the effective stage efficiency is the kinetic approach to
    # equilibrium 1 - exp(-k * mixer_residence_time) rather than the fixed
    # Murphree stage_efficiency, so slow kinetics / short mixing under-shoot
    # equilibrium. None keeps the constant efficiency (backward compatible).
    k_extraction: float | None = None
    # Phase entrainment (#110). Fractions of one phase physically carried into
    # the opposite outlet (organic droplets in aqueous, aqueous droplets in
    # organic), carrying their dissolved REE across and reducing separation.
    # Typical values 0.001-0.01; 0 disables (backward compatible).
    entrainment_org_in_aq: float = 0.0
    entrainment_aq_in_org: float = 0.0
    # Third-phase formation (#117). If set, an organic loading (mol REE per mol
    # extractant) above this limit flags third-phase onset in info. None
    # disables the check (backward compatible).
    third_phase_loading_limit: float | None = None


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

    symbol = "Mixer-Settler"
    equations = [
        r"E_M = \frac{y_\mathrm{out} - y_\mathrm{in}}{y^\ast - y_\mathrm{in}}\qquad \text{(Murphree stage efficiency)}",
        r"y^\ast_i = D_i\,x_i\qquad (D_i = D_i(\mathrm{pH}, [\mathrm{HA}]))",
    ]
    assumptions = [
        "Single stage with configurable Murphree efficiency.",
        "Mixer reaches near-equilibrium; settler provides phase separation only.",
    ]
    references = [
        "Perry's Chemical Engineers' Handbook, 9e, Sec. 15.",
    ]
    parameter_symbols = {
        "stage_efficiency": r"E_M",
        "mixer_residence_time": r"\tau_\mathrm{mix}",
        "settler_residence_time": r"\tau_\mathrm{set}",
    }
    parameter_units = {
        "mixer_residence_time": "s",
        "settler_residence_time": "s",
    }

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
        F_extractant = org_flows.get(p.extractant, 0.0)
        F_diluent = org_flows.get(p.diluent, 1.0)
        F_org = F_extractant + F_diluent

        D_values = self._distribution.get_D_all(pH, T)

        # Effective stage efficiency: kinetic approach to equilibrium when a
        # rate constant is supplied (#118), else the fixed Murphree value.
        if p.k_extraction is not None:
            eta = approach_to_equilibrium(p.mixer_residence_time, p.k_extraction)
        else:
            eta = jnp.asarray(p.stage_efficiency)

        # Start with full copies to preserve non-extracted species (Bug #53)
        aq_out_flows = dict(aq_flows)
        org_out_flows = dict(org_flows)

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
            F_aq_out = F_aq_in + eta * (F_aq_eq - F_aq_in)
            F_org_out = F_org_in + eta * (F_org_eq - F_org_in)

            # Clip aqueous and derive organic to preserve mass balance
            F_aq_out = jnp.maximum(F_aq_out, 0.0)
            F_org_out = F_total - F_aq_out
            aq_out_flows[elem] = F_aq_out
            org_out_flows[elem] = jnp.maximum(F_org_out, 0.0)

        # Phase entrainment (#110): a fraction of each phase (and every species
        # it carries) is entrained into the opposite outlet, so dissolved REE
        # crosses back and separation degrades. Mass is conserved species-wise.
        f_oa = jnp.asarray(p.entrainment_org_in_aq)
        f_ao = jnp.asarray(p.entrainment_aq_in_org)
        if p.entrainment_org_in_aq or p.entrainment_aq_in_org:
            all_species = set(aq_out_flows) | set(org_out_flows)
            entrained_aq = {}
            entrained_org = {}
            for s in all_species:
                a = jnp.asarray(aq_out_flows.get(s, 0.0))
                o = jnp.asarray(org_out_flows.get(s, 0.0))
                entrained_aq[s] = a * (1.0 - f_ao) + o * f_oa
                entrained_org[s] = o * (1.0 - f_oa) + a * f_ao
            aq_out_flows = entrained_aq
            org_out_flows = entrained_org

        P = aqueous_in["P"]
        aqueous_out = make_stream(aq_out_flows, T, P)
        organic_out = make_stream(org_out_flows, T, P)

        info = {
            "D_values": D_values,
            "efficiency": eta,
        }
        if p.k_extraction is not None:
            info["kinetic_efficiency"] = eta
            info["mixer_residence_time"] = jnp.asarray(p.mixer_residence_time)

        # Third-phase formation check (#117): organic loading vs the limit.
        if p.third_phase_loading_limit is not None:
            total_org_ree = sum(jnp.asarray(org_out_flows.get(e, 0.0)) for e in p.elements)
            F_extractant = org_flows.get(p.extractant, 1.0)
            loading = total_org_ree / jnp.maximum(jnp.asarray(F_extractant), 1e-30)
            info["organic_loading"] = loading
            info["third_phase_formed"] = loading > p.third_phase_loading_limit

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
