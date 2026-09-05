"""Saponified extractants and the counter-ion balance (#197).

WHY THIS EXISTS
---------------
Before this module there was no sodium, ammonium or magnesium anywhere in the
extraction path and no representation of a pre-neutralized organic. That is not
a missing advanced option; it is how industrial rare-earth circuits actually
run. Rather than dose base into every mixer -- which causes local pH excursions
that precipitate hydroxides and stabilize emulsions -- a plant neutralizes 30
to 50% of the acidic extractant *before* it enters the cascade,

.. math::

    \\mathrm{HA}_{org} + \\mathrm{NaOH}
        \\rightarrow \\mathrm{NaA}_{org} + \\mathrm{H_2O}

so extraction is a counter-ion exchange rather than a proton exchange:

.. math::

    \\mathrm{RE}^{3+} + 3\\,\\mathrm{NaA}_{org}
        \\rightleftharpoons \\mathrm{RE(A)_3}_{org} + 3\\,\\mathrm{Na}^+

WHAT GOES WRONG WITHOUT IT
--------------------------
A model without saponification predicts a **pH collapse down the extraction
section that a real plant does not have**: every trivalent ion extracted
releases three protons into an aqueous phase with nothing to absorb them. Such
a model under-predicts loading, over-predicts the stage count required, and
mis-ranks extractants -- while looking entirely plausible, because each stage
is internally consistent. :func:`ph_profile_flatness` measures the collapse,
and ``tests/ree/test_saponification.py`` asserts the difference rather than
describing it.

THE ORGANIC IS THE BUFFER
-------------------------
The mechanism that holds a long cascade at a nearly flat pH is that the
organic phase is a buffer. ``(HA)2`` and its counter-ion salt ``M(HA2)`` are an
acid / conjugate-base pair whose proton lives in the *aqueous* phase:

.. math::

    [\\mathrm{M(HA_2)}] = K\\,\\frac{[\\mathrm{M}^+][\\mathrm{(HA)_2}]}
                                    {[\\mathrm{H}^+]}
    \\quad\\Longrightarrow\\quad
    \\mathrm{pH} = \\mathrm{p}K + \\log_{10}\\frac{S}{1-S} - \\log_{10}[\\mathrm{M}^+]

with ``S`` the saponification degree -- Henderson-Hasselbalch for the organic
phase (:func:`organic_buffer_pH`), whose capacity
``beta = ln(10) E_T S (1 - S)`` (:func:`organic_buffer_capacity`) is what the
released protons are spent against. A free organic ``A-`` is deliberately
**not** a species: a bare anion is not stable in a low-dielectric diluent, it
is always paired with its counter-ion, and pairing it is exactly what
``M(HA2)`` is. So the ``HA``/``A-`` acid-base equilibrium the issue asks for is
present, in the only form in which it is physical, as one row of the tableau.

ONE ADDED ROW, NO NEW SOLVER
----------------------------
#196 made the counter-ion a conserved component of every shipped network
precisely so that this issue would be one species row in
``data/reaction_networks.yaml`` and no change to
:mod:`difflow_ree.equilibrium.mass_action`. That held:
``cation_exchange_dimer_saponified`` is ``cation_exchange_dimer`` plus

.. code-block:: yaml

    - name: "M(HA2)"
      phase: organic
      charge: 0
      stoichiometry: {"M+": 1, "(HA)2": 1, "H+": -1}

and everything below is bookkeeping around the unchanged
:func:`~difflow_ree.equilibrium.mass_action.solve_section`.
:class:`SaponifiedSection` is a subclass of
:class:`~difflow_ree.equilibrium.mass_action.MassActionSection` that overrides
exactly two things: which species flows the *solvent* stream contributes (a
saponified solvent brings counter-ion in, and with it a negative proton
component), and writing the outlet organic's counter-ion into the extract
stream so the counter-ion is conserved across the unit's own interface.

SAPONIFICATION DEGREE IS THE MANIPULATED VARIABLE
-------------------------------------------------
Along with phase ratio per section and scrub/strip acid strength, the degree is
what an operator actually adjusts, so it has to be a handle a control or RTO
layer can reach. It is carried on the **stream** -- ``schema.saponified_organic``
writes it as a counter-ion flow -- so it can be a tracer and can be
differentiated through, and :func:`saponification_degree_for_pH` inverts the
section for the degree that hits a pH target, posed as one more row of the same
root find so the derivative comes out of one implicit differentiation.

THE SAME BALANCE PREDICTS THE EFFLUENT
--------------------------------------
The counter-ion balance that makes the cascade correct is the equation that
predicts the raffinate load, so kilograms of base per kilogram of rare-earth
oxide costs no extra machinery:
:attr:`~difflow_ree.equilibrium.network.ReactionNetwork.base_equivalents_per_mole_ree`
reads three equivalents per mole of rare earth off the tableau, and
:mod:`difflow_ree.economics.saponification` turns that into kg of base, kg of
ammonium nitrogen, kg of dissolved salt and dollars.

WHERE THE CONSTANT COMES FROM
-----------------------------
Nowhere measurable in this repository. :func:`saponification_log_K` inverts

.. math::

    K = \\frac{S}{1-S}\\,\\frac{[\\mathrm{H}^+]}{[\\mathrm{M}^+]}

at a *stated* reference degree and condition, so the constant is a restatement
of a declared operating point rather than a number taken from a paper -- the
same discipline the correlation calibration of #196 follows. Supply a measured
constant for design numbers.

References:
    Banda, R., Jeon, H., Lee, M. Separation of Nd from mixed chloride solutions
    with Pr by extraction with saponified PC 88A and scrubbing.
    *J. Ind. Eng. Chem.* 21 (2015) 436-442. doi:10.1016/j.jiec.2014.03.002

    Liao, C. et al. Clean separation technologies of rare earth resources in
    China. *J. Rare Earths* 31 (2013) 331-336.
    doi:10.1016/S1002-0721(12)60281-6

    Both were consulted qualitatively, for the 30-50% saponification range and
    for the effluent consequences of the ammonium route. Neither is the source
    of any number here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from difflow.eo_solver import solve_residual_system
from difflow.streams import Stream, get_flows, make_stream
from difflow_ree.database import get_extractant
from difflow_ree.equilibrium.mass_action import (
    MassActionParams,
    MassActionSection,
    MassActionSolution,
    aqueous_component_totals,
    floor_totals,
    make_section_residual,
    organic_component_totals,
    section_scales,
)
from difflow_ree.equilibrium.network import (
    NetworkTemplate,
    Reaction,
    ReactionNetwork,
    get_network_template,
    network_for_extractant,
)
from difflow_ree.equilibrium.schema import (
    ALL_COUNTER_ION_CHARGES,
    counter_ion_charge_of,
)


_LN10 = float(np.log(10.0))

#: Saponification degrees industrial circuits run at. Quoted as a *range* on
#: purpose: it is an operating decision, not a property, and both endpoints are
#: routinely used. Below it the organic cannot buffer the section; above it the
#: organic entrains water and third phases become a risk.
INDUSTRIAL_DEGREE_RANGE = (0.30, 0.50)


# =============================================================================
# Calibrating the exchange constant
# =============================================================================

def saponification_log_K(
    degree: float = 0.35,
    pH: float = 3.0,
    counter_ion_conc: float = 0.1,
    counter_ion_charge: int = 1,
    extractant_conc: float = 1.0,
) -> float:
    """``log10 K`` that puts the organic at a stated degree and condition.

    The saponification equilibrium is

    .. math::

        \\mathrm{M}^{z+} + z\\,\\overline{\\mathrm{(HA)_2}}
          \\rightleftharpoons \\overline{\\mathrm{M(HA_2)}_z} + z\\,\\mathrm{H}^+

    and with ``S`` the fraction of extractant equivalents neutralized, the
    organic holds ``[salt] = S E_T / z`` and ``[(HA)2] = (1 - S) E_T``, so

    .. math::

        K = \\frac{S E_T / z}{[\\mathrm{M}^{z+}]\\,\\bigl((1-S)E_T\\bigr)^{z}}
            \\,[\\mathrm{H}^+]^{z}

    which for a monovalent counter-ion collapses to the extractant-independent

    .. math::

        K = \\frac{S}{1-S}\\,\\frac{[\\mathrm{H}^+]}{[\\mathrm{M}^+]}

    **This is a calibration, not a measurement.** It restates a declared
    operating point as a constant, exactly as
    :func:`~difflow_ree.equilibrium.network.log_K_from_correlation` restates
    the correlation. Nothing in this repository measures a saponification
    constant. Supply your own for design numbers.

    Args:
        degree: Saponification degree ``S``, strictly between 0 and 1.
        pH: Aqueous pH (concentration scale) of the reference contact.
        counter_ion_conc: Free aqueous counter-ion concentration (M) at the
            reference contact.
        counter_ion_charge: Counter-ion charge ``z``; 1 for Na/NH4/K, 2 for
            Mg/Ca.
        extractant_conc: Total extractant concentration on the **component**
            basis (M, i.e. dimers for a dimeric extractant). It cancels
            exactly for ``z = 1`` and only matters for a divalent counter-ion.

    Returns:
        ``log10 K``.

    Raises:
        ValueError: If ``degree`` is not strictly inside (0, 1), or a
            concentration is non-positive.

    Example:
        >>> round(saponification_log_K(0.35, pH=3.0, counter_ion_conc=0.1), 4)
        -2.2688
    """
    if not 0.0 < float(degree) < 1.0:
        raise ValueError(
            f"saponification degree must be strictly between 0 and 1 to define "
            f"an exchange constant (S = 0 means no salt at all, S = 1 means no "
            f"free acid at all), got {degree} (#197)."
        )
    if counter_ion_conc <= 0 or extractant_conc <= 0:
        raise ValueError(
            f"saponification_log_K needs positive reference concentrations, "
            f"got counter_ion_conc={counter_ion_conc}, "
            f"extractant_conc={extractant_conc} (#197)."
        )
    z = int(counter_ion_charge)
    S = float(degree)
    h = 10.0 ** (-float(pH))
    salt = S * float(extractant_conc) / z
    free = (1.0 - S) * float(extractant_conc)
    K = salt * h ** z / (float(counter_ion_conc) * free ** z)
    return float(np.log10(K))


def organic_buffer_pH(
    log10_K: float | Array,
    degree: float | Array,
    counter_ion_conc: float | Array,
    counter_ion_charge: int = 1,
    extractant_conc: float | Array = 1.0,
) -> Array:
    """Henderson-Hasselbalch for the organic phase (#197).

    Inverse of :func:`saponification_log_K`: the aqueous pH an organic at a
    given saponification degree holds the aqueous phase at. This is *why* a
    saponified cascade has a flat pH profile -- the section sits on the flat
    part of a titration curve whose total is the whole extractant inventory.

    Args:
        log10_K: Saponification constant.
        degree: Saponification degree, strictly inside (0, 1).
        counter_ion_conc: Free aqueous counter-ion concentration (M).
        counter_ion_charge: Counter-ion charge.
        extractant_conc: Extractant concentration on the component basis (M);
            only enters for a divalent counter-ion.

    Returns:
        pH on the concentration scale.

    Example:
        >>> float(round(organic_buffer_pH(-2.2688, 0.35, 0.1), 4))
        3.0
    """
    z = int(counter_ion_charge)
    S = jnp.asarray(degree, dtype=jnp.float64)
    E = jnp.asarray(extractant_conc, dtype=jnp.float64)
    m = jnp.asarray(counter_ion_conc, dtype=jnp.float64)
    salt = S * E / z
    free = (1.0 - S) * E
    # K = salt * h**z / (m * free**z)  ->  z*log10 h = log10 K + log10 m
    #                                                 + z log10 free - log10 salt
    log_h = (
        jnp.asarray(log10_K, dtype=jnp.float64)
        + jnp.log10(m)
        + z * jnp.log10(free)
        - jnp.log10(salt)
    ) / z
    return -log_h


def organic_buffer_capacity(
    extractant_conc: float | Array,
    degree: float | Array,
) -> Array:
    """Buffer capacity of the saponified organic (mol/L per pH unit).

    The classical Van Slyke capacity of a conjugate pair whose total is the
    extractant inventory,

    .. math::

        \\beta = \\ln(10)\\,E_T\\,S\\,(1 - S)

    maximal at ``S = 0.5`` and vanishing at either end. This is the quantity
    that decides how many stages of proton release the organic can absorb
    before the pH moves, so it is the honest measure of "the organic is a
    buffer".

    Args:
        extractant_conc: Total extractant concentration on the component basis
            (M).
        degree: Saponification degree.

    Returns:
        Buffer capacity in mol of protons per litre of organic per pH unit.

    Example:
        >>> f"{float(organic_buffer_capacity(0.25, 0.35)):.5f}"
        '0.13096'
    """
    E = jnp.asarray(extractant_conc, dtype=jnp.float64)
    S = jnp.asarray(degree, dtype=jnp.float64)
    return _LN10 * E * S * (1.0 - S)


# =============================================================================
# A divalent counter-ion is a different tableau
# =============================================================================

def divalent_counter_ion_template(
    template: str | NetworkTemplate = "cation_exchange_dimer_saponified",
    counter_ion_charge: int = 2,
    name: str | None = None,
) -> NetworkTemplate:
    """Derive the magnesia-saponification tableau from the monovalent one.

    A divalent counter-ion is a different network for exactly the reason a
    divalent anion is: the counter-ion component carries charge ``+z``, the
    salt species neutralizes ``z`` extractant equivalents and releases ``z``
    protons, so the row is ``M(HA2)2``, not ``M(HA2)``. Deriving it from the
    shipped monovalent template rather than shipping a fifth near-copy in the
    YAML keeps one source for the stoichiometry (#197).

    Args:
        template: Monovalent saponified template, by name or object.
        counter_ion_charge: Charge ``z`` of the counter-ion (2 for Mg, Ca).
        name: Name for the derived template; None appends ``"_z<z>"``.

    Returns:
        A :class:`~difflow_ree.equilibrium.network.NetworkTemplate` that can
        be passed anywhere a template name can.

    Raises:
        ValueError: If the source template is not saponified, or the charge is
            not at least 1.

    Example:
        >>> t = divalent_counter_ion_template()
        >>> [s.name for s in t.species]
        ['RE(HA2)3', 'M(HA2)2']
        >>> [c.charge for c in t.components if c.role == "counter_ion"]
        [2]
    """
    template = get_network_template(template)
    if not template.is_saponified:
        raise ValueError(
            f"Network {template.name!r} has no counter-ion species, so there "
            f"is nothing to give a different charge. Start from a saponified "
            f"template such as 'cation_exchange_dimer_saponified' (#197)."
        )
    z = int(counter_ion_charge)
    if z < 1:
        raise ValueError(f"counter_ion_charge must be >= 1, got {z} (#197).")

    counter = next(c for c in template.components if c.role == "counter_ion")
    extractant = next(c for c in template.components if c.role == "extractant")
    proton = next(c for c in template.components if c.role == "proton")
    new_counter_name = f"M{z}+" if z > 1 else counter.name

    components = tuple(
        dataclasses.replace(c, name=new_counter_name, charge=z)
        if c.role == "counter_ion" else c
        for c in template.components
    )

    species = []
    for spec in template.species:
        if spec.stoichiometry.get(counter.name, 0) <= 0:
            species.append(spec)
            continue
        species.append(
            Reaction(
                # "M(HA2)" -> "M(HA2)2": z formula units of the monovalent
                # salt, held together by one divalent cation.
                name=spec.name if z == 1 else f"{spec.name}{z}",
                phase=spec.phase,
                charge=0,
                stoichiometry={
                    new_counter_name: 1,
                    extractant.name: z,
                    proton.name: -z,
                },
                log10_K=spec.log10_K,
            )
        )
    return dataclasses.replace(
        template,
        name=name or f"{template.name}_z{z}",
        components=components,
        species=tuple(species),
    )


# =============================================================================
# Reading the degree back out of a solved section
# =============================================================================

def saponification_degree_profile(
    network: ReactionNetwork,
    solution: MassActionSolution,
) -> Array:
    """Per-stage saponification degree of the organic phase (#197).

    The degree is an **output** of the same component balances that give the
    pH profile: the counter-ion total partitions between free ``M+`` in the
    aqueous phase and the salt in the organic, and the section decides the
    split. A stage-0 degree well below the inlet degree is the section telling
    you it has spent its buffer.

    Args:
        network: The saponified reaction network.
        solution: A solved section.

    Returns:
        ``(n_stages,)`` array of the fraction of extractant equivalents held
        as the counter-ion salt. All zeros for an unsaponified network.

    Example:
        >>> S = saponification_degree_profile(net, sol)   # doctest: +SKIP
    """
    if not network.counter_ion_species_index:
        return jnp.zeros(solution.u.shape[0], dtype=jnp.float64)
    c = solution.concentrations(network)
    names = network.species_names
    # Extractant component concentration in the organic: free plus everything
    # bound to it, read through the tableau rather than assumed constant.
    total = jnp.exp(solution.u[:, network.extractant_index])
    for j, spec_name in enumerate(names):
        nu_e = float(network.nu[j, network.extractant_index])
        if nu_e:
            total = total + nu_e * c[spec_name]
    neutralized = jnp.zeros_like(total)
    for j in network.counter_ion_species_index:
        neutralized = neutralized + abs(
            float(network.nu[j, network.proton_index])
        ) * c[names[j]]
    return neutralized / total


def ph_profile_flatness(pH_profile: Array) -> Array:
    """Peak-to-peak spread of a pH profile (pH units).

    The single number in which #197's central claim is falsifiable: an
    unsaponified section's profile collapses by the protons it releases, a
    saponified one's does not. Reported as peak-to-peak rather than as a
    standard deviation because it is the *excursion* that decides whether a
    stage precipitates hydroxides or drops below the extractant's working
    range.

    Args:
        pH_profile: ``(n_stages,)`` pH values, e.g. ``info["pH_profile"]``.

    Returns:
        ``max(pH) - min(pH)``. Smaller is flatter.

    Example:
        >>> f"{float(ph_profile_flatness(jnp.array([3.0, 2.9, 2.85]))):.4f}"
        '0.1500'
    """
    p = jnp.asarray(pH_profile, dtype=jnp.float64)
    return jnp.max(p) - jnp.min(p)


# =============================================================================
# Parameters and the section
# =============================================================================

@dataclass(repr=False)
class SaponifiedParams(MassActionParams):
    """Parameters for a saponified extraction section (#197).

    Everything :class:`~difflow_ree.equilibrium.mass_action.MassActionParams`
    carries, plus the saponification handle.

    Attributes:
        saponification_degree: Fraction of the extractant's exchangeable
            protons neutralized in the solvent **entering** the section. This
            is the default used when the solvent stream does not declare its
            own counter-ion loading; a stream built by
            :meth:`~difflow_ree.equilibrium.schema.REEStreamSchema.saponified_organic`
            or produced by a :class:`~difflow_ree.units.saponification.Saponifier`
            overrides it, and that is the differentiable path -- the degree
            travels in the stream, so it can be a tracer.
        saponification_log10_K: The counter-ion exchange constant. None
            calibrates it with :func:`saponification_log_K` from the reference
            degree and condition below, defaulting to the extractant record's
            ``saponification`` block.
        saponification_reference_degree: Degree the constant is calibrated at.
            None uses the extractant record's declared degree.
        saponification_reference_pH: Reference contact pH for the calibration.
            None uses the record's.
        saponification_reference_counter_ion: Reference free counter-ion
            concentration (M) for the calibration. None uses the record's.
        base: Reagent the counter-ion is dosed as (``"NaOH"``, ``"NH3"``,
            ``"Mg(OH)2"``, ...). None picks the default base for the
            counter-ion; it is used only by the economics layer.

    Example:
        >>> p = SaponifiedParams(
        ...     n_stages=4, extractant="D2EHPA", elements=("Nd", "Dy"),
        ...     aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
        ...     saponification_degree=0.35,
        ... )
        >>> p["saponification_degree"]
        0.35
    """

    saponification_degree: float | Array = 0.35
    saponification_log10_K: float | None = None
    saponification_reference_degree: float | None = None
    saponification_reference_pH: float | None = None
    saponification_reference_counter_ion: float | None = None
    base: str | None = None

    def __post_init__(self) -> None:
        """Validate the section and the saponification handle.

        Raises:
            ValueError: On anything
                :class:`~difflow_ree.equilibrium.mass_action.MassActionParams`
                rejects, a concrete degree outside [0, 1], a counter-ion of
                ``None``, or a solvating extractant, which has no acidic
                proton to neutralize.
        """
        super().__post_init__()
        if self.counter_ion is None:
            raise ValueError(
                "A saponified section needs a counter_ion: the base that "
                "neutralizes the extractant leaves its cation behind, and "
                "without a counter-ion component there is nothing to conserve "
                "it (#197)."
            )
        if self.counter_ion not in ALL_COUNTER_ION_CHARGES:
            raise ValueError(
                f"Unknown counter_ion {self.counter_ion!r}. Recognised: "
                f"{sorted(ALL_COUNTER_ION_CHARGES)} (#197)."
            )
        ext = get_extractant(self.extractant)
        if ext.mechanism != "cation_exchange":
            raise ValueError(
                f"Extractant {self.extractant!r} extracts by "
                f"{ext.mechanism!r}, so it has no acidic proton for a base to "
                f"neutralize and cannot be saponified. Use "
                f"MassActionSection for it (#197)."
            )
        # Only a concrete degree can be range-checked; the whole point of
        # carrying it here is that it may also arrive as a tracer.
        try:
            value = float(self.saponification_degree)
        except Exception:  # pragma: no cover - tracer
            return
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"saponification_degree is a fraction of the extractant's "
                f"exchangeable protons and must lie in [0, 1], got {value}. "
                f"Industrial circuits run at "
                f"{INDUSTRIAL_DEGREE_RANGE[0]}-{INDUSTRIAL_DEGREE_RANGE[1]} "
                f"(#197)."
            )


class SaponifiedSection(MassActionSection):
    """Counter-current section run on pre-neutralized organic (#197).

    A :class:`~difflow_ree.equilibrium.mass_action.MassActionSection` on the
    saponified reaction network, which differs from its parent in exactly two
    places:

    1. the **solvent** contributes a counter-ion salt species, so a saponified
       solvent brings counter-ion into the section and, with it, a *negative*
       proton component -- the protons the base removed. That is the whole
       mechanism by which the pH profile stops collapsing;
    2. the **extract** carries the counter-ion still bound to the organic when
       it leaves, so the counter-ion is conserved across the unit's own
       interface and not merely inside the solver.

    Everything else -- the residual, the globalization, the implicit
    differentiation, the structural conservation -- is the parent's, unchanged.

    Note:
        :func:`~difflow_ree.equilibrium.mass_action.base_addition_for_pH` and
        :func:`~difflow_ree.equilibrium.mass_action.base_addition_bounds` call
        ``organic_component_totals`` directly rather than through this class,
        so they do **not** see a saponified solvent's counter-ion loading and
        will answer as if the organic were bare. Use
        :func:`saponification_degree_for_pH` instead: on a saponified circuit
        the degree, not aqueous dosing, is the handle anyway.

    Attributes:
        params: The :class:`SaponifiedParams`.
        schema: The stream schema, which now has an ``Na_org``-style organic
            counter-ion key.
        network: The saponified
            :class:`~difflow_ree.equilibrium.network.ReactionNetwork`.
        log10_K_saponification: The constant actually used, calibrated or
            supplied.

    Example:
        >>> params = SaponifiedParams(
        ...     n_stages=3, extractant="D2EHPA", elements=("Nd", "Dy"),
        ...     aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
        ...     saponification_degree=0.35,
        ... )
        >>> section = SaponifiedSection(params)
        >>> section.network.is_saponified
        True
    """

    symbol = "REE Saponified Extraction Section"
    equations = [
        r"\overline{\mathrm{HA}} + \mathrm{MOH} \rightarrow \overline{\mathrm{MA}} + \mathrm{H_2O}",
        r"\mathrm{RE}^{3+} + 3\,\overline{\mathrm{M(HA_2)}} \rightleftharpoons \overline{\mathrm{RE}(\mathrm{HA}_2)_3} + 3\,\mathrm{M}^+",
        r"[\overline{\mathrm{M(HA_2)}}] = K\,[\mathrm{M}^+][\overline{\mathrm{(HA)_2}}]/[\mathrm{H}^+]",
        r"\mathrm{pH} = \mathrm{p}K + \log_{10}\frac{S}{1-S} - \log_{10}[\mathrm{M}^+]",
    ]
    assumptions = [
        "Equilibrium stages; constant volumetric phase flows across the section.",
        "The organic conjugate base exists only as its counter-ion salt; no free organic A-.",
        "Conditional equilibrium constants at the medium's ionic strength; no activity model.",
        "Phases immiscible; no third phase, which a high saponification degree can cause.",
    ]
    references = [
        "Banda, R., Jeon, H., Lee, M. J. Ind. Eng. Chem. 21, 436 (2015). doi:10.1016/j.jiec.2014.03.002",
        "Liao, C. et al. J. Rare Earths 31, 331 (2013). doi:10.1016/S1002-0721(12)60281-6",
    ]
    numerical_method = (
        "Unchanged from MassActionSection: section-scope Newton on the "
        "component balances in log concentration, via optimistix."
    )

    def __init__(self, params: SaponifiedParams):
        """Select the saponified network and calibrate its constant.

        Args:
            params: Section parameters.

        Raises:
            ValueError: If the schema's counter-ion charge disagrees with the
                network's counter-ion component charge.
        """
        ext = get_extractant(params.extractant)
        z = counter_ion_charge_of(params.counter_ion)

        template = get_network_template(
            params.network
            or network_for_extractant(params.extractant, saponified=True)
        )
        if z != 1:
            template = divalent_counter_ion_template(
                template, counter_ion_charge=z
            )
        salt = next(
            s for s in template.species
            if any(
                s.stoichiometry.get(c.name, 0) > 0
                for c in template.components if c.role == "counter_ion"
            )
        )

        log_K = params.saponification_log10_K
        if log_K is None:
            log_K = ext.saponification_log10_K
        if log_K is None:
            degree = params.saponification_reference_degree
            if degree is None:
                degree = ext.saponification_degree or 0.35
            ref_pH = params.saponification_reference_pH
            if ref_pH is None:
                ref_pH = ext.saponification_reference_pH
            ref_M = params.saponification_reference_counter_ion
            if ref_M is None:
                ref_M = ext.saponification_reference_counter_ion
            log_K = saponification_log_K(
                degree=degree,
                pH=ref_pH,
                counter_ion_conc=ref_M,
                counter_ion_charge=z,
                extractant_conc=(
                    params.extractant_conc
                    / (2.0 if template.extractant_basis == "dimer" else 1.0)
                ),
            )
        self.log10_K_saponification = float(log_K)

        # The parent calibrates the per-element constants only if `log10_K` is
        # empty, so the saponification constant has to travel with them.
        element_K = dict(params.log10_K) if params.log10_K else None
        if element_K is None:
            from difflow_ree.equilibrium.network import log_K_from_correlation

            dist_kwargs = (
                {"nitrate_conc": params.anion_conc or ext.reference_nitrate}
                if ext.requires_nitrate else {}
            )
            element_K = log_K_from_correlation(
                template,
                params.elements,
                params.extractant,
                calibration_pH=params.calibration_pH,
                extractant_conc=params.extractant_conc,
                anion_conc=(
                    params.anion_conc
                    if params.anion_conc is not None
                    else (ext.reference_nitrate or 1.0)
                ),
                **dist_kwargs,
            )
        element_K = dict(element_K)
        element_K.setdefault(salt.name, self.log10_K_saponification)

        super().__init__(
            dataclasses.replace(params, network=template, log10_K=element_K)
        )
        # Keep the caller's own params object visible: the parent stored the
        # rewritten copy, and the network/log10_K rewrite is an implementation
        # detail nobody should read back as a specification.
        self.params = params
        self._salt_index = self.network.species_names.index(salt.name)
        self._salt_name = salt.name

        network_charge = self.network.components[
            self.network.counter_ion_index
        ].charge
        if network_charge != self.schema.counter_ion_charge:
            raise ValueError(
                f"Counter-ion {params.counter_ion!r} has charge "
                f"{self.schema.counter_ion_charge:+d} but the reaction network "
                f"declares its counter-ion component with charge "
                f"{network_charge:+d}. A divalent counter-ion needs the "
                f"tableau from divalent_counter_ion_template (#197)."
            )

    # -- the two things that differ from the parent ----------------------

    @property
    def salt_species(self) -> str:
        """Name of the saponified-extractant species, e.g. ``"M(HA2)"``."""
        return self._salt_name

    @property
    def counter_ion_per_salt(self) -> float:
        """Moles of counter-ion in one mole of the salt species."""
        return float(
            self.network.nu[self._salt_index, self.network.counter_ion_index]
        )

    @property
    def equivalents_per_salt(self) -> float:
        """Base equivalents held by one mole of the salt species."""
        return abs(
            float(self.network.nu[self._salt_index, self.network.proton_index])
        )

    def solvent_counter_ion(self, solvent: Stream) -> Array:
        """Counter-ion (mol/s) the solvent stream brings into the section.

        Read from the stream's organic counter-ion key when it declares one --
        that is the differentiable path, and it is what a
        :class:`~difflow_ree.units.saponification.Saponifier` writes -- and
        otherwise computed from ``params.saponification_degree``, so a section
        can be run standalone with the degree as a plain parameter.

        Args:
            solvent: Organic solvent stream.

        Returns:
            Scalar molar flow of counter-ion bound to the organic phase.
        """
        flows = get_flows(solvent)
        key = self.schema.organic_counter_ion
        if key is not None and key in flows:
            return jnp.asarray(flows[key], dtype=jnp.float64)
        f_ext = jnp.asarray(
            flows.get(self.schema.extractant, 0.0), dtype=jnp.float64
        )
        equivalents = jnp.asarray(
            self.params.saponification_degree, dtype=jnp.float64
        ) * f_ext / self.network.monomers_per_component
        return equivalents / self.schema.counter_ion_charge

    def _salt_flows(self, solvent: Stream) -> dict[str, Array]:
        """Species flows the solvent contributes beyond the loaded elements."""
        return {
            self._salt_name:
                self.solvent_counter_ion(solvent) / self.counter_ion_per_salt
        }

    def component_totals(
        self,
        feed: Stream,
        solvent: Stream,
        base_addition: Array | float | None = None,
    ) -> tuple[Array, Array]:
        """Component totals entering the section from each phase.

        The override is the whole of #197's equilibrium model: the solvent's
        saponified extractant is read through the tableau, so it contributes
        ``+1`` counter-ion and ``-1`` proton per equivalent. A saponified
        solvent therefore enters carrying a *negative* proton component, at
        the solvent end of the cascade, which is exactly where a real circuit
        puts it -- and is not the same as dosing the equivalent base into the
        aqueous feed at the other end.

        Args:
            feed: Aqueous feed stream.
            solvent: Organic solvent stream.
            base_addition: Overrides ``params.base_addition``; this is dosing
                into the *aqueous* feed, and is independent of saponification.

        Returns:
            ``(feed_totals, solvent_totals)``, each ``(n_components,)`` mol/s.
        """
        if base_addition is None:
            base_addition = self.params.base_addition
        return (
            aqueous_component_totals(
                self.network, self.schema, feed, base_addition
            ),
            organic_component_totals(
                self.network, self.schema, solvent,
                extra_species_flows=self._salt_flows(solvent),
            ),
        )

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float = 298.15,
        base_addition: Array | float | None = None,
        u0: Array | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Run the section.

        Args:
            feed: Aqueous feed stream.
            solvent: Organic solvent stream, carrying its counter-ion loading
                in ``schema.organic_counter_ion`` if it has one.
            T: Temperature (K).
            base_addition: Strong-base dosing into the aqueous feed (mol/s),
                separate from saponification.
            u0: Explicit starting point for the solve.

        Returns:
            ``(raffinate, extract, info)``. The extract now carries the
            counter-ion still bound to the organic, so the counter-ion is
            conserved across the unit interface. ``info`` adds
            ``saponification_degree_profile`` and ``saponification_degree``
            (both **outputs**), ``saponification_degree_in``,
            ``counter_ion_in``, ``counter_ion_released`` (into the raffinate,
            which is the reagent duty and the effluent load), ``pH_flatness``
            and ``base_equivalents_per_mole_ree``.
        """
        raffinate, extract, info = super().__call__(
            feed, solvent, T, base_addition, u0
        )
        net = self.network
        solution = info["solution"]
        Q_org = jnp.asarray(
            self.params.organic_volumetric_flow, dtype=jnp.float64
        )
        c = solution.concentrations(net)

        m_in = self.solvent_counter_ion(solvent)
        m_out = Q_org * c[self._salt_name][0] * self.counter_ion_per_salt

        key = self.schema.organic_counter_ion
        extract_flows = dict(get_flows(extract))
        extract_flows[key] = m_out
        extract = make_stream(extract_flows, extract["T"], extract["P"])

        degrees = saponification_degree_profile(net, solution)
        aqueous_in = jnp.asarray(
            get_flows(feed).get(self.schema.counter_ion, 0.0),
            dtype=jnp.float64,
        )
        info = dict(info)
        info.update({
            # The degree is an OUTPUT here, exactly as the pH is (#197).
            "saponification_degree_profile": degrees,
            "saponification_degree": degrees[0],
            "saponification_degree_in": (
                m_in * self.schema.counter_ion_charge
                * net.monomers_per_component
                / jnp.maximum(
                    jnp.asarray(
                        get_flows(solvent).get(self.schema.extractant, 0.0),
                        dtype=jnp.float64,
                    ),
                    1e-300,
                )
            ),
            "counter_ion_in": m_in + aqueous_in,
            "counter_ion_out_organic": m_out,
            "counter_ion_released": m_in - m_out,
            "pH_flatness": ph_profile_flatness(info["pH_profile"]),
            "base_equivalents_per_mole_ree": net.base_equivalents_per_mole_ree,
            "log10_K_saponification": self.log10_K_saponification,
            "counter_ion": self.params.counter_ion,
        })
        return raffinate, extract, info


# =============================================================================
# The manipulated variable, inverted
# =============================================================================

def saponification_degree_for_pH(
    section: SaponifiedSection,
    feed: Stream,
    solvent: Stream,
    target_pH: Array | float,
    stage: int = -1,
    T: Array | float = 298.15,
    bracket: tuple[float, float] = (1e-6, 0.999),
    n_bisection_steps: int = 40,
    tol: float = 1e-10,
    max_steps: int = 20,
) -> tuple[Array, Array]:
    """Invert the section for the saponification degree that hits a pH (#197).

    Saponification degree is the primary manipulated variable of a rare-earth
    circuit, so a control or RTO layer needs the map from a pH specification
    onto it -- not onto "stage pH setpoints", which is a plant that does not
    exist. This is the organic-side twin of
    :func:`~difflow_ree.equilibrium.mass_action.base_addition_for_pH` and is
    posed the same way, and for the same reasons: bisection under
    ``stop_gradient`` locates the root, then a single *augmented* root find --
    the section's own component balances with the degree as one more unknown
    and "the pH at this stage equals the target" as one more row -- produces
    both the answer and its derivative from one implicit differentiation.

    Cost is roughly ``n_bisection_steps`` section solves. It is a design-time
    utility; differentiate through it once and use the linearization if you
    need it in a loop.

    Args:
        section: A configured :class:`SaponifiedSection`.
        feed: Aqueous feed stream.
        solvent: Organic solvent stream; its own counter-ion loading is
            *replaced* by the degree being solved for.
        target_pH: Desired pH (concentration scale).
        stage: Stage index whose pH is specified; -1 is the raffinate end.
        T: Temperature (K).
        bracket: ``(S_lo, S_hi)`` degrees to search between. The endpoints are
            kept strictly inside (0, 1): at either end the buffer pair is
            incomplete and the constant is undefined.
        n_bisection_steps: Bisection iterations.
        tol: Tolerance for the augmented Newton polish.
        max_steps: Maximum Newton steps in the polish.

    Returns:
        ``(degree, feasible)``. ``feasible`` is False -- with the degree
        clipped to the nearest bracket end -- when the target needs more or
        less saponification than the bracket allows, which is a statement
        about the specification, not a solver failure.

    Example:
        >>> S, ok = saponification_degree_for_pH(sec, feed, solv, 2.6)  # doctest: +SKIP
    """
    net = section.network
    schema = section.schema
    params = section.params
    n_stages = int(params.n_stages)
    n_comp = net.n_components
    target = jnp.asarray(target_pH, dtype=jnp.float64)
    S_lo = jnp.asarray(bracket[0], dtype=jnp.float64)
    S_hi = jnp.asarray(bracket[1], dtype=jnp.float64)

    f_ext = jnp.asarray(
        get_flows(solvent).get(schema.extractant, 0.0), dtype=jnp.float64
    )
    per_salt_M = section.counter_ion_per_salt
    z = schema.counter_ion_charge

    def salt_flow(S):
        """Moles of the salt species carried by the solvent at degree S."""
        return (
            S * f_ext / net.monomers_per_component / z / per_salt_M
        )

    def solvent_totals_at(S):
        return organic_component_totals(
            net, schema, solvent,
            extra_species_flows={section.salt_species: salt_flow(S)},
        )

    def pH_at(S):
        sol = section.solve(
            feed,
            schema.saponified_organic(
                f_ext,
                S,
                monomers_per_component=net.monomers_per_component,
                counter_ion_charge=z,
                diluent_flow=get_flows(solvent).get(schema.diluent, 0.0),
                element_flows={
                    el: get_flows(solvent).get(el, 0.0) for el in net.elements
                },
                T=solvent["T"],
                P=solvent["P"],
            ),
            T=T,
        )
        return jnp.where(sol.feasible, sol.pH(net)[stage], -jnp.inf)

    def bisect(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        # pH rises monotonically with the degree: more base, less free acid.
        too_low = pH_at(mid) < target
        return jnp.where(too_low, mid, lo), jnp.where(too_low, hi, mid)

    lo, hi = lax.fori_loop(0, n_bisection_steps, bisect, (S_lo, S_hi))
    S0 = lax.stop_gradient(0.5 * (lo + hi))

    feed_totals_0 = aqueous_component_totals(
        net, schema, feed, params.base_addition
    )
    scale = lax.stop_gradient(
        section_scales(net, feed_totals_0, solvent_totals_at(S0))
    )
    feed_totals = floor_totals(net, feed_totals_0, scale)
    residual_fn, _ = make_section_residual(net, n_stages, params.anion_closure)
    start = lax.stop_gradient(
        section.solve(
            feed,
            schema.saponified_organic(
                f_ext, S0,
                monomers_per_component=net.monomers_per_component,
                counter_ion_charge=z,
                diluent_flow=get_flows(solvent).get(schema.diluent, 0.0),
                element_flows={
                    el: get_flows(solvent).get(el, 0.0) for el in net.elements
                },
                T=solvent["T"], P=solvent["P"],
            ),
            T=T,
        ).u.reshape(-1)
    )

    def augmented(zz, args):
        u = zz[:-1]
        S = zz[-1]
        inner = residual_fn(
            u,
            {
                "ln_K": section.network.ln_K(),
                "Q_aq": jnp.asarray(
                    params.aqueous_volumetric_flow, dtype=jnp.float64
                ),
                "Q_org": jnp.asarray(
                    params.organic_volumetric_flow, dtype=jnp.float64
                ),
                "feed_totals": feed_totals,
                "solvent_totals": floor_totals(
                    net, solvent_totals_at(S), scale
                ),
                "scale": scale,
            },
        )
        pH = -u.reshape(n_stages, n_comp)[stage, net.proton_index] / _LN10
        return jnp.concatenate([inner, jnp.reshape(pH - target, (1,))])

    z0 = jnp.concatenate([start, jnp.reshape(S0, (1,))])
    z_sol, _, feasible = solve_residual_system(
        augmented, z0, None, rtol=tol, atol=tol, max_steps=max_steps,
        feasible_tol=1e-8,
    )
    S_raw = z_sol[-1]
    S = jnp.clip(jnp.where(jnp.isfinite(S_raw), S_raw, S_hi), S_lo, S_hi)
    slack = 1e-9 * jnp.abs(S_hi - S_lo)
    feasible = jnp.logical_and(
        feasible,
        jnp.logical_and(S_raw >= S_lo - slack, S_raw <= S_hi + slack),
    )
    return S, feasible
