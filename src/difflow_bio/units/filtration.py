"""Membrane filtration unit operations for protein processing.

This module provides ultrafiltration and diafiltration models:
- Ultrafiltration: Concentration of proteins by membrane separation
- Diafiltration: Buffer exchange with UF membrane

Key equations:
    Flux (with concentration polarization): J = Lp * (TMP - delta_pi)
    Film model: C_wall = C_bulk * exp(J / k_mass)
    Osmotic pressure difference: delta_pi = sigma * (C_wall - C_permeate)
    Linearized: J = Lp * TMP / (1 + Lp * sigma * C_bulk / k_mass)
    Rejection: R = 1 - C_permeate / C_retentate
    Concentration factor: CF = V_initial / V_final
    Diafiltration: C/C_0 = exp(-N_dv * (1-R)) for permeable solutes

where:
    J = permeate flux (L/m²/h or LMH)
    TMP = transmembrane pressure (bar)
    R = rejection coefficient
    N_dv = number of diavolumes
    k_mass = mass transfer coefficient (m/s)
    sigma = osmotic pressure coefficient (Pa·m³/kg)
"""

from dataclasses import dataclass, field

from difflow.params_mixin import ParamsMixin
import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, make_stream, get_flows
from difflow.numerics import safe_divide


# =============================================================================
# Filtration Parameters
# =============================================================================

@dataclass(repr=False)
class UltrafiltrationParams(ParamsMixin):
    """Parameters for ultrafiltration.

    Attributes:
        membrane_area: Membrane area (m²). Sets the processing time reported
            when the call is given ``feed_volume``
            (``info['process_time_h']`` = permeate volume / (flux x area)).
            The species split does not depend on it: sieving coefficients
            are per unit area, so a smaller membrane reaches the same
            concentration factor, only more slowly.
        MWCO: Molecular weight cutoff (kDa), the molecular weight rejected
            at 90% (the manufacturers' nominal rating). Sets the rejection
            of every species not listed in ``rejection`` from its molecular
            weight (:func:`rejection_from_mw`).
        rejection: Dict of species name -> rejection coefficient (0-1).
            Overrides the MWCO-derived value for the species listed.
        molecular_weights: Dict of species name -> molecular weight (kDa),
            consulted before :data:`difflow_bio.database.BIO_SPECIES_MW_KDA`
            and the core species database. A species found in none of
            them, and not in ``rejection``, is treated as freely permeable
            (R = 0), the right default for buffer components.
        Lp: Membrane permeability (L/m²/h/bar), optional
        k_mass: Mass transfer coefficient for concentration polarization (m/s).
                Controls the build-up of solute at the membrane wall.
                Typical value for protein UF: 5e-6 m/s.
        sigma: Osmotic pressure coefficient (Pa·m³/kg).
               Relates wall concentration to osmotic back-pressure.
               Typical value for protein UF: 1000 Pa·m³/kg.
        species_order: List of species names
    """
    membrane_area: float | Array
    MWCO: float | Array = 30.0  # kDa, typical for mAb
    rejection: dict = field(default_factory=dict)
    Lp: float | Array = 50.0  # L/m²/h/bar, typical for UF membrane
    k_mass: float | Array = 5e-6  # m/s, mass transfer coefficient
    sigma: float | Array = 1000.0  # Pa·m³/kg, osmotic pressure coefficient
    # Fouling coupling (#155). Resistance-in-series flux decline with
    # cumulative permeate throughput: Lp_eff = Lp / (1 + fouling_coefficient *
    # V_permeate) (Cheryan, Ultrafiltration and Microfiltration Handbook, 2e,
    # CRC Press, 1998, Ch. 4). Default 0 disables it (backward compatible).
    fouling_coefficient: float | Array = 0.0  # per unit permeate volume (1/L)
    species_order: list[str] = None
    molecular_weights: dict = field(default_factory=dict)


@dataclass(repr=False)
class DiafiltrationParams(ParamsMixin):
    """Parameters for diafiltration (buffer exchange).

    Attributes:
        membrane_area: Membrane area (m²). Sets the processing time reported
            when the call is given ``feed_volume``; the exchange itself
            depends on the diavolumes, not the area.
        MWCO: Molecular weight cutoff (kDa, rejected at 90%). Sets the
            rejection of species not listed in ``rejection``.
        rejection: Dict of species -> rejection coefficient, overriding
            the MWCO-derived value.
        molecular_weights: Dict of species -> molecular weight (kDa); see
            :class:`UltrafiltrationParams`.
        Lp: Membrane permeability (L/m²/h/bar)
        k_mass: Mass transfer coefficient for concentration polarization (m/s).
                Controls the build-up of solute at the membrane wall.
                Typical value for protein UF: 5e-6 m/s.
        sigma: Osmotic pressure coefficient (Pa·m³/kg).
               Relates wall concentration to osmotic back-pressure.
               Typical value for protein UF: 1000 Pa·m³/kg.
        species_order: List of species names
    """
    membrane_area: float | Array
    MWCO: float | Array = 30.0
    rejection: dict = field(default_factory=dict)
    Lp: float | Array = 50.0
    k_mass: float | Array = 5e-6  # m/s, mass transfer coefficient
    sigma: float | Array = 1000.0  # Pa·m³/kg, osmotic pressure coefficient
    fouling_coefficient: float | Array = 0.0  # per unit permeate volume (1/L)
    species_order: list[str] = None
    molecular_weights: dict = field(default_factory=dict)


def species_rejection(params, species: str) -> Array:
    """Rejection coefficient a membrane applies to one species.

    An explicit ``params.rejection`` entry wins; otherwise the rejection
    follows the membrane MWCO against the species' molecular weight
    (``params.molecular_weights``, then the bio and core species tables).
    A species with no molecular weight anywhere is freely permeable.

    Bio audit C4: the units used ``rejection.get(species, 0.0)``, so with the
    default empty ``rejection`` every species, the 150 kDa mAb included,
    passed a 30 kDa membrane: UF at CF 5 put 8 of 10 g of mAb in the permeate
    and 5 diavolumes of DF washed out 9.93 of 10 g.

    Args:
        params: ``UltrafiltrationParams`` or ``DiafiltrationParams``.
        species: Species name.

    Returns:
        Rejection coefficient in [0, 1].
    """
    if species in params.rejection:
        return jnp.asarray(params.rejection[species])
    mws = params.molecular_weights or {}
    if species in mws:
        mw = mws[species]
    else:
        from difflow_bio.database import get_species_mw_kda
        mw = get_species_mw_kda(species)
    if mw is None:
        return jnp.asarray(0.0)
    return rejection_from_mw(jnp.asarray(mw), jnp.asarray(params.MWCO))


def _process_time_h(permeate_volume_L, flux_LMH, area_m2):
    """Hours to pass ``permeate_volume_L`` at ``flux_LMH`` through ``area_m2``."""
    return safe_divide(permeate_volume_L, flux_LMH * area_m2)


# =============================================================================
# Ultrafiltration
# =============================================================================

class Ultrafiltration:
    """Ultrafiltration for protein concentration.

    Uses a semi-permeable membrane to concentrate proteins (retained)
    while allowing water and small molecules to pass through (permeate).

    Operates in batch concentration mode or continuous mode.
    """

    symbol = "UF"
    equations = [
        r"J = L_p\,(\Delta P - \sigma\,\Delta\Pi)\qquad \text{(Kedem-Katchalsky)}",
        r"R = 1 - \frac{C_p}{C_r}\qquad \text{(rejection coefficient)}",
        r"\mathrm{CF} = \frac{V_\mathrm{in}}{V_\mathrm{retentate}}\qquad \text{(concentration factor)}",
    ]
    assumptions = [
        "Steady-state local permeation; uniform TMP across the module.",
        "Constant sieving (rejection) coefficients per species.",
        "Negligible fouling within the run (resistance-in-series optional).",
    ]
    references = [
        "Kedem, O., Katchalsky, A. Biochim. Biophys. Acta, 27, 229 (1958).",
        "Cheryan, M. Ultrafiltration and Microfiltration Handbook, CRC Press, 1998.",
    ]
    parameter_symbols = {"membrane_area": "A", "MWCO": r"\mathrm{MWCO}", "Lp": "L_p"}
    parameter_units = {
        "membrane_area": "m^2",
        "MWCO": "kDa",
        "Lp": "L/m^2/h/bar",
        "k_mass": "m/s",
        "sigma": "Pa*m^3/kg",
        "fouling_coefficient": "1/L",
    }
    numerical_method = "Closed-form sieving + resistance-in-series flux model."

    def __init__(self, params: UltrafiltrationParams):
        """Initialize ultrafiltration unit.

        Args:
            params: UF parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        concentration_factor: float | Array,
        TMP: float | Array = 1.0,
        mode: str = "batch",
        feed_concentration: float | Array = None,
        feed_volume: float | Array = None,
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Perform ultrafiltration.

        Args:
            inlet: Feed stream
            concentration_factor: Target CF = V_in / V_retentate
            TMP: Transmembrane pressure (bar)
            mode: "batch" for batch concentration, "continuous" for steady-state
            feed_concentration: Retained-solute concentration in the feed
                (kg/m³). When provided, the flux uses the Kedem-Katchalsky
                osmotic form J = Lp_eff·(TMP − σ·C_ret/1e5) with the retentate
                concentration C_ret = feed_concentration·CF, so flux falls as
                the solution concentrates (concentration-polarization feedback,
                #99). When None, the linearized polarization factor is used
                (backward compatible).
            feed_volume: Feed volume (L). When given, the permeate volume
                and the processing time at the computed flux over
                ``membrane_area`` are reported. The stream carries amounts,
                not a volume, so the time cannot be computed without it.

        Returns:
            (retentate, permeate): Tuple of outlet streams
            info: Dictionary with:
                - 'flux': Permeate flux (L/m²/h)
                - 'recovery': Per-species fraction kept in the retentate
                - 'rejection': Per-species rejection coefficient used
                - 'volume_reduction': V_permeate / V_feed
                - 'process_time_h', 'permeate_volume_L': only with
                  ``feed_volume``
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        CF = jnp.asarray(concentration_factor)

        # Calculate volumes (assuming unit density for simplicity)
        # Total inlet flow represents volume per unit time
        total_flow_in = sum(inlet_flows.values())

        # Volume fractions
        volume_reduction = 1.0 - 1.0 / CF
        retentate_volume_frac = 1.0 / CF
        permeate_volume_frac = volume_reduction

        # Permeate flux with concentration polarization correction (film model).
        #
        # The film model relates the wall concentration to the bulk:
        #   C_wall = C_bulk * exp(J / k_mass)
        # The osmotic back-pressure from the polarized layer is:
        #   delta_pi = sigma * C_bulk * (exp(J / k_mass) - 1)
        # Corrected flux:
        #   J = Lp * (TMP - delta_pi)
        #
        # Linearised (first-order in J) for JAX-compatible closed-form solution:
        #   J_eff = Lp * TMP / (1 + Lp * sigma * C_bulk / k_mass)
        #
        # Convert Lp from L/(m²·h·bar) to SI (m/(Pa·s)) for the polarization
        # factor, which uses sigma [Pa·m³/kg] and k_mass [m/s]:
        #   1 L/(m²·h·bar) = 1e-3 m³ / (m² · 3600 s · 1e5 Pa) = 2.778e-12 m/(Pa·s)
        # With Lp_SI, sigma, and k_mass all in SI the factor is dimensionless:
        #   polarization_factor = 1 + Lp_SI * sigma / k_mass
        # Note: sigma (Pa·m³/kg) already incorporates the bulk concentration
        # effect (sigma = osmotic_coeff * C_bulk effectively), so C_bulk does
        # not appear separately in the polarization factor expression.
        # Fouling (#155): effective permeability declines with cumulative
        # permeate throughput via a resistance-in-series factor.
        V_permeate = total_flow_in * permeate_volume_frac
        fouling_factor = 1.0 + p.fouling_coefficient * V_permeate
        Lp_eff = p.Lp / fouling_factor

        if feed_concentration is not None:
            # Concentration-polarization feedback (#99): the osmotic back-
            # pressure grows with the retentate concentration, which rises
            # with CF, so flux declines as the batch concentrates.
            C_ret = jnp.asarray(feed_concentration) * CF  # kg/m³
            delta_pi_bar = p.sigma * C_ret / 1e5          # Pa -> bar
            J = Lp_eff * jnp.maximum(TMP - delta_pi_bar, 0.0)  # L/m²/h
        else:
            Lp_SI = Lp_eff * 2.778e-12  # m/(Pa·s)
            polarization_factor = 1.0 + Lp_SI * p.sigma / p.k_mass
            delta_pi_bar = None
            J = Lp_eff * TMP / polarization_factor  # L/m²/h (effective flux)

        # Split species based on rejection: explicit, else from the MWCO
        # against the species' molecular weight (bio audit C4).
        retentate_flows = {}
        permeate_flows = {}
        rejection = {}

        for species, flow in inlet_flows.items():
            R = species_rejection(p, species)
            rejection[species] = R

            if mode == "batch":
                # Batch concentration with constant rejection R:
                # C_ret = C_0 * CF^R  (concentration increases for retained species)
                # M_ret = C_ret * V_ret = C_0 * CF^R * V_0/CF = M_0 * CF^(R-1)
                # retained_frac = CF^(R-1)
                retained_frac = CF ** (R - 1.0)
                retained_frac = jnp.clip(retained_frac, 0.0, 1.0)

                retentate_flows[species] = flow * retained_frac
                permeate_flows[species] = flow * (1.0 - retained_frac)

            else:  # continuous mode
                # Steady state: simple rejection split
                retentate_flows[species] = flow * retentate_volume_frac * (1.0 + R * (CF - 1.0))
                permeate_flows[species] = flow - retentate_flows[species]
                permeate_flows[species] = jnp.maximum(permeate_flows[species], 0.0)

        retentate = make_stream(retentate_flows, inlet["T"], inlet["P"])
        permeate = make_stream(permeate_flows, inlet["T"], inlet["P"])

        # Retentate recovery of every species (each now has a rejection).
        recovery = {
            species: safe_divide(retentate_flows[species], flow)
            for species, flow in inlet_flows.items()
        }

        info = {
            "flux": J,
            "concentration_factor": CF,
            "volume_reduction": volume_reduction,
            "recovery": recovery,
            "rejection": rejection,
            "retentate_volume_fraction": retentate_volume_frac,
            "fouling_factor": fouling_factor,
            "membrane_area": jnp.asarray(p.membrane_area),
        }
        if delta_pi_bar is not None:
            info["osmotic_pressure_bar"] = delta_pi_bar
        if feed_volume is not None:
            # The area sets how long the batch takes, not the split.
            permeate_volume_L = jnp.asarray(feed_volume) * permeate_volume_frac
            info["permeate_volume_L"] = permeate_volume_L
            info["process_time_h"] = _process_time_h(
                permeate_volume_L, J, jnp.asarray(p.membrane_area))

        return (retentate, permeate), info


class Diafiltration:
    """Diafiltration for buffer exchange.

    Adds buffer while removing permeate to exchange the buffer
    composition while maintaining constant volume.

    Two modes:
    - Constant volume diafiltration (CVD): Add buffer = Remove permeate
    - Discontinuous diafiltration: Batch dilution then concentration
    """

    symbol = "Diafiltration"
    equations = [
        r"\frac{C_\mathrm{final}}{C_0} = \exp\!\bigl(-(1-R)\,N\bigr)\qquad \text{(CVD, }N\text{ diavolumes)}",
        r"C_\mathrm{buffer,final} = C_\mathrm{buffer,in}\,(1 - e^{-N})",
    ]
    assumptions = [
        "Constant-volume diafiltration (buffer addition rate equals permeate removal).",
        "Constant sieving coefficients and TMP.",
    ]
    references = [
        "Cheryan, M. Ultrafiltration and Microfiltration Handbook, CRC Press, 1998.",
    ]
    parameter_symbols = {"membrane_area": "A", "MWCO": r"\mathrm{MWCO}"}
    parameter_units = {
        "membrane_area": "m^2",
        "MWCO": "kDa",
        "Lp": "L/m^2/h/bar",
        "k_mass": "m/s",
        "sigma": "Pa*m^3/kg",
        "fouling_coefficient": "1/L",
    }
    numerical_method = "Analytical buffer-exchange kinetics (CVD)."

    def __init__(self, params: DiafiltrationParams):
        """Initialize diafiltration unit.

        Args:
            params: DF parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        buffer: Stream,
        n_diavolumes: float | Array | None = None,
        TMP: float | Array = 1.0,
        feed_volume: float | Array = None,
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Perform constant-volume diafiltration.

        Args:
            inlet: Feed stream (retentate side).
            buffer: Diafiltration buffer. With ``n_diavolumes=None`` (the
                default) this is the buffer actually fed: its flows enter
                the balance and set the diavolumes, N = total buffer /
                total inlet (the same amount-as-volume proxy the unit uses
                for the retentate). With ``n_diavolumes`` given, only its
                composition is used and it is scaled to N x the inlet; the
                buffer that entered is then reported as
                ``info['buffer_consumed']`` so the balance can be closed.
            n_diavolumes: Number of diavolumes (buffer volume / retentate
                volume), or None to take it from the buffer stream.
            TMP: Transmembrane pressure (bar).
            feed_volume: Retentate volume (L). When given, the permeate
                volume and the processing time over ``membrane_area`` are
                reported.

        Returns:
            (retentate, permeate): Tuple of outlet streams. Retentate +
            permeate = inlet + ``info['buffer_consumed']`` for every
            species.
            info: Dictionary with the flux, diavolumes, per-species
            exchange efficiency and rejection, and the buffer consumed.

        Example:
            >>> df = Diafiltration(DiafiltrationParams(membrane_area=1.0))
            >>> (ret, perm), info = df(feed, buffer)  # N from the buffer
        """
        p = self.params
        inlet_flows = get_flows(inlet)
        buffer_flows = get_flows(buffer)

        # Total volume (sum of flows as proxy)
        V_initial = sum(inlet_flows.values())
        total_buffer_flow = sum(buffer_flows.values())

        # Bio audit C4: the buffer stream entered only as a composition, so
        # 100 units of buffer fed became 5 x 110 = 550 units in the balance
        # and the flowsheet's buffer stream never closed. Now the buffer fed
        # is the buffer consumed unless the caller fixes N explicitly.
        if n_diavolumes is None:
            n_dv = safe_divide(total_buffer_flow, V_initial)
            scale = jnp.asarray(1.0)
        else:
            n_dv = jnp.asarray(n_diavolumes)
            scale = n_dv * V_initial / jnp.maximum(total_buffer_flow, 1e-30)
        buffer_added = {s: f * scale for s, f in buffer_flows.items()}

        # Permeate flux with concentration polarization correction (film model).
        #
        # Linearised film-model approximation (same as Ultrafiltration):
        #   J = Lp * TMP / (1 + Lp_SI * sigma / k_mass)
        # Lp converted to SI (m/(Pa·s)) so the factor is dimensionless.
        # Fouling (#155): permeability declines with cumulative permeate
        # (= n_dv diavolumes of the initial volume).
        fouling_factor = 1.0 + p.fouling_coefficient * (n_dv * V_initial)
        Lp_eff = p.Lp / fouling_factor
        Lp_SI = Lp_eff * 2.778e-12  # L/(m²·h·bar) → m/(Pa·s)
        polarization_factor = 1.0 + Lp_SI * p.sigma / p.k_mass
        J = Lp_eff * TMP / polarization_factor  # L/m²/h (effective flux)

        # Constant-volume diafiltration, per diavolume N and sieving
        # coefficient s = 1 - R:  dC/dN = C_buffer - s C.  Hence
        #   initial solute kept:  M_0 exp(-s N)
        #   buffer solute kept:   C_buffer V (1 - exp(-s N)) / s
        #                       = buffer_added (1 - exp(-x)) / x,  x = s N.
        # The second form is continuous through R -> 1 (all added solute
        # stays); the old code used (1 - exp(-x)) without the 1/s and
        # switched to "all retained" at R = 0.99, a jump in R.
        species_all = list(inlet_flows) + [s for s in buffer_flows if s not in inlet_flows]
        retentate_flows = {}
        permeate_flows = {}
        rejection = {}
        exchange_efficiency = {}
        for species in species_all:
            flow = inlet_flows.get(species, jnp.asarray(0.0))
            added = buffer_added.get(species, jnp.asarray(0.0))
            R = species_rejection(p, species)
            rejection[species] = R
            x = n_dv * (1.0 - R)
            remaining_frac = jnp.exp(-x)
            big = x > 1e-8
            x_safe = jnp.where(big, x, 1.0)
            wash_in = jnp.where(big, -jnp.expm1(-x_safe) / x_safe, 1.0 - 0.5 * x)
            retentate_flows[species] = flow * remaining_frac + added * wash_in
            # Exact complement, so retentate + permeate = inlet + buffer.
            permeate_flows[species] = flow * (1.0 - remaining_frac) + added * (1.0 - wash_in)
            if species in inlet_flows:
                exchange_efficiency[species] = 1.0 - remaining_frac

        retentate = make_stream(retentate_flows, inlet["T"], inlet["P"])
        permeate = make_stream(permeate_flows, inlet["T"], inlet["P"])

        info = {
            "flux": J,
            "n_diavolumes": n_dv,
            "exchange_efficiency": exchange_efficiency,
            "rejection": rejection,
            "buffer_volume_added": n_dv * V_initial,
            "buffer_consumed": make_stream(
                {s: buffer_added[s] for s in buffer_flows}, buffer["T"], buffer["P"]
            ),
            "fouling_factor": fouling_factor,
            "membrane_area": jnp.asarray(p.membrane_area),
        }
        if feed_volume is not None:
            permeate_volume_L = n_dv * jnp.asarray(feed_volume)
            info["permeate_volume_L"] = permeate_volume_L
            info["process_time_h"] = _process_time_h(
                permeate_volume_L, J, jnp.asarray(p.membrane_area))

        return (retentate, permeate), info


# =============================================================================
# TFF (Tangential Flow Filtration) - Combined UF/DF
# =============================================================================

class TFF:
    """Tangential Flow Filtration system for UF and DF operations.

    Combines ultrafiltration and diafiltration in a single unit,
    as commonly used in bioprocessing for:
    1. Initial concentration
    2. Diafiltration (buffer exchange)
    3. Final concentration
    """

    symbol = "TFF"
    equations = [
        r"J = L_p\,(\Delta P - \sigma\,\Delta\Pi)\qquad \text{(UF step)}",
        r"\frac{C_\mathrm{final}}{C_0} = \mathrm{CF}^{R} \exp\!\bigl(-(1-R)\,N\bigr)\qquad \text{(combined concentration + DF)}",
    ]
    assumptions = [
        "Tangential (cross-flow) operation; cake layer suppressed by shear.",
        "Constant membrane properties across the three stages.",
    ]
    references = ["Zeman, L.J., Zydney, A.L. Microfiltration and Ultrafiltration, Marcel Dekker, 1996."]
    parameter_symbols = {"MWCO": r"\mathrm{MWCO}", "Lp": "L_p"}
    parameter_units = {"MWCO": "kDa", "Lp": "L/m^2/h/bar"}
    numerical_method = "Sequential composition of UF + CVD + UF models."

    def __init__(
        self,
        membrane_area: float | Array,
        MWCO: float | Array = 30.0,
        rejection: dict = None,
        Lp: float | Array = 50.0,
        k_mass: float | Array = 5e-6,
        sigma: float | Array = 1000.0,
        fouling_coefficient: float | Array = 0.0,
        molecular_weights: dict = None,
    ):
        """Initialize TFF system.

        Args:
            membrane_area: Membrane area (m²); sets the processing times
                the stages report when given a feed volume.
            MWCO: Molecular weight cutoff (kDa, rejected at 90%); sets the
                rejection of species not listed in ``rejection``.
            rejection: Dict of species -> rejection coefficient
            molecular_weights: Dict of species -> molecular weight (kDa),
                forwarded to both stages.
            Lp: Membrane permeability (L/m²/h/bar)
            k_mass: Mass transfer coefficient (m/s) for concentration
                polarization. Forwarded to the UF and DF stages (#99).
            sigma: Osmotic pressure coefficient (Pa·m³/kg), forwarded to the
                UF and DF stages (#99).
            fouling_coefficient: Fouling coefficient (1/L permeate) for the
                resistance-in-series flux decline (#155), forwarded to both
                stages.
        """
        rejection = rejection or {}
        molecular_weights = molecular_weights or {}

        # Keep the arguments, not just what was built from them. TFF is a
        # composition of two stages and holds no Params of its own, so
        # `difflow.serialize` -- which reads a constructor argument off the
        # instance by attribute of the same name -- had nothing to read,
        # and a flowsheet containing a TFF could not be saved and loaded
        # back at all. The other six are recorded for the same reason,
        # though only the required one is carried by the file today.
        self.membrane_area = membrane_area
        self.MWCO = MWCO
        self.rejection = rejection
        self.Lp = Lp
        self.k_mass = k_mass
        self.sigma = sigma
        self.fouling_coefficient = fouling_coefficient
        self.molecular_weights = molecular_weights

        self.uf = Ultrafiltration(UltrafiltrationParams(
            membrane_area=membrane_area,
            MWCO=MWCO,
            rejection=rejection,
            Lp=Lp,
            k_mass=k_mass,
            sigma=sigma,
            fouling_coefficient=fouling_coefficient,
            molecular_weights=molecular_weights,
        ))

        self.df = Diafiltration(DiafiltrationParams(
            membrane_area=membrane_area,
            MWCO=MWCO,
            rejection=rejection,
            Lp=Lp,
            k_mass=k_mass,
            sigma=sigma,
            fouling_coefficient=fouling_coefficient,
            molecular_weights=molecular_weights,
        ))

    def concentrate(
        self,
        inlet: Stream,
        concentration_factor: float | Array,
        TMP: float | Array = 1.0,
        feed_volume: float | Array = None,
    ) -> tuple[tuple[Stream, Stream], dict]:
        """Concentrate feed by ultrafiltration.

        Args:
            inlet: Feed stream
            concentration_factor: Target CF
            TMP: Transmembrane pressure (bar)
            feed_volume: Feed volume (L); when given, the processing time
                over the membrane area is reported.

        Returns:
            (retentate, permeate): Outlet streams
            info: Operation details
        """
        return self.uf(inlet, concentration_factor, TMP, feed_volume=feed_volume)

    def diafilter(
        self,
        inlet: Stream,
        buffer: Stream,
        n_diavolumes: float | Array | None = None,
        TMP: float | Array = 1.0,
    ) -> tuple[tuple[Stream, Stream], dict]:
        """Exchange buffer by diafiltration.

        Args:
            inlet: Feed stream
            buffer: Buffer fed (or its composition when ``n_diavolumes``
                is given); see :meth:`Diafiltration.__call__`
            n_diavolumes: Number of diavolumes, or None to take it from
                the buffer stream
            TMP: Transmembrane pressure (bar)

        Returns:
            (retentate, permeate): Outlet streams
            info: Operation details
        """
        return self.df(inlet, buffer, n_diavolumes, TMP)

    def uf_df_uf(
        self,
        inlet: Stream,
        buffer: Stream,
        CF_initial: float | Array,
        n_diavolumes: float | Array,
        CF_final: float | Array,
        TMP: float | Array = 1.0,
    ) -> tuple[tuple[Stream, Stream, Stream, Stream], dict]:
        """Complete UF/DF/UF process.

        Standard process:
        1. Concentrate to CF_initial
        2. Diafilter with n_diavolumes
        3. Concentrate to CF_final

        Args:
            inlet: Feed stream
            buffer: Buffer composition for diafiltration (scaled to
                ``n_diavolumes``; what entered is ``info['buffer_consumed']``)
            CF_initial: Initial concentration factor
            n_diavolumes: Diavolumes for buffer exchange
            CF_final: Final concentration factor
            TMP: Transmembrane pressure

        Returns:
            (final_retentate, uf1_permeate, df_permeate, uf2_permeate):
                the product and the three permeates, so that their sum
                equals ``inlet`` + ``info['buffer_consumed']`` species by
                species. Only the retentate used to be returned and the
                permeates, with the product they carry, were dropped
                (bio audit (d): 3.59 of 100 g mAb).
            info: Aggregate information from all steps

        Example:
            >>> (product, p1, p2, p3), info = tff.uf_df_uf(feed, buffer, 2.0, 5.0, 5.0)
        """
        # Step 1: Initial concentration
        (ret1, perm1), info1 = self.uf(inlet, CF_initial, TMP)

        # Step 2: Diafiltration
        (ret2, perm2), info2 = self.df(ret1, buffer, n_diavolumes, TMP)

        # Step 3: Final concentration
        (ret3, perm3), info3 = self.uf(ret2, CF_final, TMP)

        info = {
            "step1_uf": info1,
            "step2_df": info2,
            "step3_uf": info3,
            "total_CF": CF_initial * CF_final,
            "n_diavolumes": n_diavolumes,
            "buffer_consumed": info2["buffer_consumed"],
        }

        return (ret3, perm1, perm2, perm3), info


# =============================================================================
# Utility Functions
# =============================================================================

def concentration_polarization(
    C_bulk: Array,
    J: Array,
    k_m: Array,
    R: Array,
) -> Array:
    """Calculate wall concentration with concentration polarization.

    C_wall = C_bulk * exp(J / k_m) / (1 - R + R * exp(J / k_m))

    Args:
        C_bulk: Bulk concentration
        J: Permeate flux (m/s or consistent units with k_m)
        k_m: Mass transfer coefficient (m/s)
        R: Rejection coefficient

    Returns:
        Wall (membrane surface) concentration
    """
    exp_term = jnp.exp(J / k_m)
    return C_bulk * exp_term / (1.0 - R + R * exp_term)


def gel_layer_flux(
    C_bulk: Array,
    C_gel: Array,
    k_m: Array,
) -> Array:
    """Calculate flux limited by gel layer formation.

    J = k_m * ln(C_gel / C_bulk)

    Args:
        C_bulk: Bulk concentration
        C_gel: Gel concentration (limiting)
        k_m: Mass transfer coefficient

    Returns:
        Limiting permeate flux
    """
    return k_m * jnp.log(C_gel / C_bulk)


def diavolumes_required(
    initial_conc: Array,
    target_conc: Array,
    rejection: Array,
) -> Array:
    """Calculate diavolumes needed for target concentration.

    n_dv = -ln(C_target / C_initial) / (1 - R)

    Args:
        initial_conc: Initial concentration of species to remove
        target_conc: Target concentration
        rejection: Rejection coefficient of species

    Returns:
        Required number of diavolumes
    """
    ratio = target_conc / initial_conc
    return safe_divide(-jnp.log(ratio), 1.0 - rejection)


def rejection_from_mw(
    MW: Array,
    MWCO: Array,
) -> Array:
    """Estimate rejection coefficient from molecular weight.

    Logistic sieving curve in log molecular weight, anchored so that a
    solute AT the cutoff is rejected at 90%, which is how membrane makers
    rate a nominal MWCO (Cheryan, Ultrafiltration and Microfiltration
    Handbook, 1998, Ch. 3):

    R = 1 / (1 + exp(-k*(log(MW) - log(MWCO))) / 9)

    With k = 3 a solute five times the cutoff is rejected at >99.9%, the
    basis of the usual rule of choosing an MWCO 3 to 5 times below the
    product's molecular weight, and a small solute (< 1 kDa on a 30 kDa
    membrane) passes freely. The curve used to put R = 0.5 at the MWCO,
    which left a 150 kDa mAb at R = 0.992 on a 30 kDa membrane and lost
    ~4% of it over five diavolumes (bio audit C4).

    Args:
        MW: Molecular weight (Da or kDa, consistent with MWCO)
        MWCO: Molecular weight cutoff

    Returns:
        Estimated rejection coefficient (0-1)

    Example:
        >>> round(float(rejection_from_mw(30.0, 30.0)), 3)
        0.9
    """
    k = 3.0  # Steepness factor
    log_ratio = jnp.log(MW) - jnp.log(MWCO)
    return jax.nn.sigmoid(k * log_ratio + jnp.log(9.0))
