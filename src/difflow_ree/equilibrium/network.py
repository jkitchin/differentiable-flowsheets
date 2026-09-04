"""Reaction networks for REE solvent extraction, carried as data (#196).

`difflow_ree` had no mass-action equilibrium model: the reaction
``RE3+ + 3 HA <-> REA3 + 3 H+`` and its constant ``K_ex`` appeared only in a
LaTeX docstring, while what ran was an empirical ``log10(D)`` correlation with
multiplicative corrections (#189-#195). This module is the data model for the
closed alternative, and the reason it is a *data model* rather than four
branches is the point of issue #196: cation exchange, saponified cation
exchange (#197), solvating extraction and anion exchange are rows in
``data/reaction_networks.yaml``, not four code paths.

THE TABLEAU
-----------
A network declares

- **components**: a chemically independent basis whose totals are conserved,
  each with a phase, an integer charge and a role
  (``rare_earth`` / ``proton`` / ``counter_ion`` / ``anion`` / ``extractant``);
- **species**: everything formed from that basis, each with integer
  stoichiometric coefficients against the components, a phase, a charge and
  one ``log10 K``.

Mass action is then

.. math::

    \\log_{10}[S_j] = \\log_{10} K_j + \\sum_c \\nu_{jc}\\,\\log_{10}[C_c]

and the conserved total of component ``c`` over a stage is

.. math::

    T_c = \\sum_j \\nu_{jc}\\,[S_j]\\,Q_{\\mathrm{phase}(j)}

summed over every species *including the free components themselves*, whose
stoichiometry is the identity row and whose ``log10 K`` is zero. This is the
standard equilibrium tableau. Writing it this way means the mass-action
expressions hold identically by construction, so the residual set handed to
the solver is the component balances alone and the system is square: one
unknown (a log free-component concentration) and one balance per component.

NEGATIVE COEFFICIENTS
---------------------
``H+`` enters the cation-exchange complex with coefficient ``-3`` because the
complex releases three protons per trivalent ion. The ``H+`` *component*
therefore means "proton in excess of the reference state in which the
extractant is fully protonated", and a loaded organic phase carries a negative
H component. That bookkeeping is exact, and it is precisely what makes the pH
profile an output of the model rather than a parameter of it.

WHERE THE CONSTANTS COME FROM
-----------------------------
:func:`log_K_from_correlation` inverts the existing L1 correlation
(:class:`~difflow_ree.equilibrium.distribution.REEDistribution`) at a stated
reference condition. That is the only source available in this repository, and
it inherits that source's provenance: the ``ph_coefficients`` of D2EHPA, PC88A
and Cyanex272 are hand-tuned with no literature source (see the header of
``data/extractants.yaml``), so a constant derived from them is illustrative.
Supply measured constants for design numbers.

The calibration is exact only at the reference condition, and the residual
disagreement is a real and reportable statement about the correlation rather
than a defect of the closure: mass action forces
``d log10 D / d pH = protons_released`` (3.0), while the tabulated ``b``
coefficients are 2.20-2.90. :func:`correlation_ph_slope_defect` returns that
gap so it can be tested and quoted instead of discovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
import yaml
from jax import Array

from difflow_ree.database import get_extractant


DATA_DIR = Path(__file__).parent.parent / "data"

#: Mechanisms a reaction network may declare.  This is deliberately a superset
#: of :data:`difflow_ree.database.EXTRACTION_MECHANISMS`: the correlation layer
#: only knows the two mechanisms it has coefficient blocks for, whereas the
#: network layer can express any mechanism whose stoichiometry can be written
#: down, which is the whole point of carrying it as data (#196).
NETWORK_MECHANISMS = (
    "cation_exchange",
    "solvating",
    "anion_exchange",
)

#: Component roles.  ``rare_earth`` and ``extractant`` are required; the other
#: three are what let :mod:`difflow_ree.equilibrium.mass_action` find the
#: proton, counter-ion and anion balances without hard-coding species names.
COMPONENT_ROLES = (
    "rare_earth",
    "proton",
    "counter_ion",
    "anion",
    "extractant",
    "other",
)

PHASES = ("aqueous", "organic")

#: Template name used for the per-element expansion of the rare-earth
#: component.  ``per_element`` rows reference it by this name.
_RE_TEMPLATE = "RE3+"

_LN10 = float(np.log(10.0))


# =============================================================================
# Rows of the table
# =============================================================================

@dataclass(frozen=True)
class Component:
    """One conserved component: a column of the tableau.

    Attributes:
        name: Component name after per-element expansion (e.g. ``"Nd3+"``).
        phase: ``"aqueous"`` or ``"organic"`` -- the phase the *free*
            component species lives in, which decides which volumetric flow
            its concentration is multiplied by.
        charge: Integer formal charge. Used by the aqueous charge balance and
            by the build-time charge-consistency check.
        role: One of :data:`COMPONENT_ROLES`.
        element: REE symbol when this component came from a ``per_element``
            row, else None.
    """
    name: str
    phase: str
    charge: int
    role: str = "other"
    element: str | None = None


@dataclass(frozen=True)
class Reaction:
    """A formation reaction: one row of the tableau.

    A species is *defined* by how it is formed from the components, so the
    species and its formation reaction are the same row.

    Attributes:
        name: Species name after per-element expansion (e.g.
            ``"Nd(HA2)3"``).
        phase: ``"aqueous"`` or ``"organic"``.
        charge: Integer formal charge; must equal ``sum_c nu_c * charge_c``.
        stoichiometry: Component name -> integer coefficient. Negative
            coefficients are normal (three protons released per RE(III)).
        log10_K: Base-10 formation constant, or None when it must be supplied
            at build time (the YAML rows carry None: see the module
            docstring).
        element: REE symbol when this species came from a ``per_element`` row,
            else None.
    """
    name: str
    phase: str
    charge: int
    stoichiometry: Mapping[str, int]
    log10_K: float | None = None
    element: str | None = None


@dataclass(frozen=True)
class NetworkTemplate:
    """An unexpanded network as it is written in the YAML file.

    Attributes:
        name: Key in ``data/reaction_networks.yaml``.
        description: Free text.
        mechanism: One of :data:`NETWORK_MECHANISMS`.
        extractant_basis: ``"dimer"`` or ``"monomer"``; what one mole of the
            extractant component is. Checked against the extractant record's
            own ``stoichiometry.basis`` when a network is built for a named
            extractant (#191, #196).
        components: Component rows, possibly carrying ``per_element``.
        species: Species rows, possibly carrying ``per_element``.
        per_element_components: Names of the component rows that expand.
        per_element_species: Names of the species rows that expand.
    """
    name: str
    description: str
    mechanism: str
    extractant_basis: str
    components: tuple[Component, ...]
    species: tuple[Reaction, ...]
    per_element_components: frozenset[str]
    per_element_species: frozenset[str]


# =============================================================================
# The expanded, numeric network
# =============================================================================

@dataclass(frozen=True, eq=False)
class ReactionNetwork:
    """An expanded reaction network with its numeric tableau.

    Everything on this object except :attr:`log10_K` is *static*: it is
    closed over by the residual builder and never traced. ``log10_K`` is
    carried separately through the solver's ``args`` so a cascade can be
    differentiated with respect to the equilibrium constants.

    Attributes:
        name: Template name.
        mechanism: One of :data:`NETWORK_MECHANISMS`.
        elements: REE symbols in tableau order.
        extractant_basis: ``"dimer"`` or ``"monomer"``.
        monomers_per_component: Extractant monomers in one mole of the
            extractant component (2 for a dimer basis, 1 for a monomer
            basis). A solvent stream declares its extractant as a *monomer*
            molar flow, so this is the divisor that converts it to the
            component basis.
        components: Expanded component rows, in tableau column order.
        species: Expanded species rows, in tableau row order.
        nu: ``(n_species, n_components)`` float64 stoichiometric matrix.
        log10_K: ``(n_species,)`` float64 formation constants.
        species_is_aqueous / component_is_aqueous: boolean masks.
        component_charges / species_charges: float64 charge vectors.
        element_component_index: Column index of each element's free ion.
        element_species_index: Row index of each element's extracted complex.

    Example:
        >>> net = cation_exchange_network("D2EHPA", ("Nd",), calibration_pH=3.0)
        >>> net.n_components, net.n_species
        (5, 1)
        >>> net.component_names[net.proton_index]
        'H+'
    """
    name: str
    mechanism: str
    elements: tuple[str, ...]
    extractant_basis: str
    monomers_per_component: float
    components: tuple[Component, ...]
    species: tuple[Reaction, ...]
    nu: np.ndarray
    log10_K: Array
    species_is_aqueous: np.ndarray
    component_is_aqueous: np.ndarray
    component_charges: np.ndarray
    species_charges: np.ndarray
    element_component_index: tuple[int, ...]
    element_species_index: tuple[int, ...]
    proton_index: int
    counter_ion_index: int | None
    anion_index: int
    extractant_index: int

    # -- convenience -----------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of conserved components (= number of unknowns per stage)."""
        return len(self.components)

    @property
    def n_species(self) -> int:
        """Number of formed species (free components excluded)."""
        return len(self.species)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in tableau column order."""
        return tuple(c.name for c in self.components)

    @property
    def species_names(self) -> tuple[str, ...]:
        """Species names in tableau row order."""
        return tuple(s.name for s in self.species)

    def component_index(self, name: str) -> int:
        """Column index of a component by name.

        Args:
            name: Component name, e.g. ``"H+"`` or ``"Nd3+"``.

        Returns:
            Index into the component axis.

        Raises:
            KeyError: If no component has that name.
        """
        try:
            return self.component_names.index(name)
        except ValueError:
            raise KeyError(
                f"Network {self.name!r} has no component {name!r}. "
                f"Components: {list(self.component_names)}."
            ) from None

    def with_log_K(self, log10_K: Array | Sequence[float]) -> "ReactionNetwork":
        """Return a copy carrying different equilibrium constants.

        Args:
            log10_K: New ``(n_species,)`` constants.

        Returns:
            A new :class:`ReactionNetwork`; the tableau is shared.

        Raises:
            ValueError: If the length does not match :attr:`n_species`.
        """
        arr = jnp.asarray(log10_K, dtype=jnp.float64)
        if arr.shape != (self.n_species,):
            raise ValueError(
                f"log10_K must have shape ({self.n_species},) for network "
                f"{self.name!r}, got {arr.shape}."
            )
        return ReactionNetwork(**{**self.__dict__, "log10_K": arr})

    def ln_K(self) -> Array:
        """Natural-log formation constants.

        The solver works in natural log concentration, so the base-10 data is
        converted once here rather than at every residual evaluation.

        Returns:
            ``(n_species,)`` array of ``ln K``.
        """
        return self.log10_K * _LN10

    def describe(self) -> str:
        """Human-readable statement of the tableau.

        Returns:
            Multi-line string listing components and formation reactions.
            Useful in a notebook to check what was actually built.
        """
        lines = [
            f"ReactionNetwork {self.name!r} (mechanism={self.mechanism}, "
            f"basis={self.extractant_basis})",
            f"  elements: {list(self.elements)}",
            "  components (unknown per stage = ln of each free concentration):",
        ]
        for c in self.components:
            lines.append(
                f"    {c.name:<12s} phase={c.phase:<8s} z={c.charge:+d} "
                f"role={c.role}"
            )
        lines.append("  species (mass action, log10 K):")
        for j, s in enumerate(self.species):
            terms = " + ".join(
                f"{v:+d} {k}" for k, v in s.stoichiometry.items()
            )
            lines.append(
                f"    {s.name:<12s} phase={s.phase:<8s} = {terms}   "
                f"log10 K = {float(self.log10_K[j]):.4f}"
            )
        return "\n".join(lines)


# =============================================================================
# YAML loading
# =============================================================================

_TEMPLATE_CACHE: dict[str, NetworkTemplate] = {}


def _load_templates(yaml_path: Path | None = None) -> dict[str, NetworkTemplate]:
    """Load and validate every network template from YAML.

    Args:
        yaml_path: Path to ``reaction_networks.yaml``; None uses the packaged
            file.

    Returns:
        Mapping of template name to :class:`NetworkTemplate`.

    Raises:
        ValueError: If a row declares an unknown phase, role or mechanism, or
            references a component that is not declared.
    """
    global _TEMPLATE_CACHE
    if yaml_path is None and _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE

    path = yaml_path if yaml_path is not None else DATA_DIR / "reaction_networks.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    templates: dict[str, NetworkTemplate] = {}
    for name, body in raw["networks"].items():
        mechanism = body["mechanism"]
        if mechanism not in NETWORK_MECHANISMS:
            raise ValueError(
                f"Network {name!r} declares mechanism {mechanism!r}; "
                f"supported: {list(NETWORK_MECHANISMS)} (#196)."
            )
        basis = body["extractant_basis"]
        if basis not in ("monomer", "dimer"):
            raise ValueError(
                f"Network {name!r}: extractant_basis must be 'monomer' or "
                f"'dimer', got {basis!r}."
            )

        comps: list[Component] = []
        per_elem_comps: set[str] = set()
        for row in body["components"]:
            _check_enum(name, row["name"], "phase", row["phase"], PHASES)
            role = row.get("role", "other")
            _check_enum(name, row["name"], "role", role, COMPONENT_ROLES)
            comps.append(
                Component(
                    name=row["name"],
                    phase=row["phase"],
                    charge=int(row["charge"]),
                    role=role,
                )
            )
            if row.get("per_element", False):
                per_elem_comps.add(row["name"])

        declared = {c.name for c in comps}
        specs: list[Reaction] = []
        per_elem_specs: set[str] = set()
        for row in body.get("species", []):
            _check_enum(name, row["name"], "phase", row["phase"], PHASES)
            stoich = {k: int(v) for k, v in row["stoichiometry"].items()}
            missing = sorted(set(stoich) - declared)
            if missing:
                raise ValueError(
                    f"Network {name!r}, species {row['name']!r}: "
                    f"stoichiometry references undeclared components "
                    f"{missing}. Declared: {sorted(declared)} (#196)."
                )
            specs.append(
                Reaction(
                    name=row["name"],
                    phase=row["phase"],
                    charge=int(row["charge"]),
                    stoichiometry=stoich,
                    log10_K=row.get("log10_K"),
                )
            )
            if row.get("per_element", False):
                per_elem_specs.add(row["name"])

        templates[name] = NetworkTemplate(
            name=name,
            description=body.get("description", ""),
            mechanism=mechanism,
            extractant_basis=basis,
            components=tuple(comps),
            species=tuple(specs),
            per_element_components=frozenset(per_elem_comps),
            per_element_species=frozenset(per_elem_specs),
        )

    if yaml_path is None:
        _TEMPLATE_CACHE = templates
    return templates


def _check_enum(net: str, row: str, field_name: str, value: str, allowed) -> None:
    """Raise a located error for an out-of-vocabulary YAML field.

    Args:
        net: Network name, for the message.
        row: Row name, for the message.
        field_name: Field being checked.
        value: Offending value.
        allowed: Permitted values.

    Raises:
        ValueError: Always, when ``value not in allowed``.
    """
    if value not in allowed:
        raise ValueError(
            f"Network {net!r}, row {row!r}: {field_name}={value!r} is not one "
            f"of {list(allowed)} (#196)."
        )


def list_networks() -> list[str]:
    """Names of the reaction networks shipped in ``reaction_networks.yaml``.

    Returns:
        Sorted list of template names.

    Example:
        >>> "cation_exchange_dimer" in list_networks()
        True
    """
    return sorted(_load_templates())


def get_network_template(name: str) -> NetworkTemplate:
    """Fetch one unexpanded network template.

    Args:
        name: Template name, see :func:`list_networks`.

    Returns:
        The :class:`NetworkTemplate`.

    Raises:
        KeyError: If the name is unknown.
    """
    templates = _load_templates()
    if name not in templates:
        raise KeyError(
            f"Unknown reaction network {name!r}. Available: "
            f"{sorted(templates)} (#196)."
        )
    return templates[name]


# =============================================================================
# Expansion
# =============================================================================

def build_network(
    template: str | NetworkTemplate,
    elements: Sequence[str],
    log10_K: Mapping[str, float] | Sequence[float] | Array,
    monomers_per_component: float | None = None,
) -> ReactionNetwork:
    """Expand a template over the tracked elements and attach constants.

    Args:
        template: Template name or a :class:`NetworkTemplate`.
        elements: REE symbols to track, in the order they should appear.
        log10_K: Formation constants. Either a mapping keyed by element
            symbol (for a network whose only species row is per-element) or
            keyed by expanded species name, or a flat sequence in species
            order.
        monomers_per_component: Extractant monomers per mole of the extractant
            component. None derives it from the template basis (2 for
            ``dimer``, 1 for ``monomer``).

    Returns:
        The expanded :class:`ReactionNetwork`.

    Raises:
        ValueError: If ``elements`` is empty, a species' declared charge does
            not match its stoichiometry, or a constant is missing.

    Example:
        >>> net = build_network(
        ...     "cation_exchange_dimer", ("Nd", "Dy"),
        ...     log10_K={"Nd": -6.0, "Dy": -4.0},
        ... )
        >>> net.component_names
        ('Nd3+', 'Dy3+', 'H+', 'M+', 'X-', '(HA)2')
    """
    if isinstance(template, str):
        template = get_network_template(template)
    elements = tuple(elements)
    if not elements:
        raise ValueError("build_network requires at least one element (#196).")

    if monomers_per_component is None:
        monomers_per_component = 2.0 if template.extractant_basis == "dimer" else 1.0

    # --- components ---------------------------------------------------
    components: list[Component] = []
    # Map the *template* component name seen in a per-element species row to
    # the expanded column, for the element currently being expanded.
    element_component_index: list[int] = []
    template_to_index: dict[str, int] = {}
    for comp in template.components:
        if comp.name in template.per_element_components:
            for el in elements:
                element_component_index.append(len(components))
                components.append(
                    Component(
                        name=f"{el}{_charge_suffix(comp)}",
                        phase=comp.phase,
                        charge=comp.charge,
                        role=comp.role,
                        element=el,
                    )
                )
        else:
            template_to_index[comp.name] = len(components)
            components.append(comp)

    per_element_column = {
        el: element_component_index[i] for i, el in enumerate(elements)
    }

    # --- species ------------------------------------------------------
    species: list[Reaction] = []
    element_species_index: list[int] = []
    for spec in template.species:
        if spec.name in template.per_element_species:
            for el in elements:
                stoich = {}
                for key, coeff in spec.stoichiometry.items():
                    if key in template.per_element_components:
                        stoich[components[per_element_column[el]].name] = coeff
                    else:
                        stoich[key] = coeff
                element_species_index.append(len(species))
                species.append(
                    Reaction(
                        name=spec.name.replace("RE", el, 1),
                        phase=spec.phase,
                        charge=spec.charge,
                        stoichiometry=stoich,
                        element=el,
                    )
                )
        else:
            species.append(spec)

    # --- numeric tableau ---------------------------------------------
    names = [c.name for c in components]
    index = {n: i for i, n in enumerate(names)}
    nu = np.zeros((len(species), len(components)), dtype=np.float64)
    for j, spec in enumerate(species):
        for key, coeff in spec.stoichiometry.items():
            nu[j, index[key]] = float(coeff)

    comp_charge = np.array([c.charge for c in components], dtype=np.float64)
    spec_charge = np.array([s.charge for s in species], dtype=np.float64)

    # Real check on the data file: a species' declared charge must equal the
    # charge its stoichiometry implies.  This catches a mistyped coefficient,
    # which is otherwise invisible until the charge balance drifts.
    implied = nu @ comp_charge
    bad = np.nonzero(np.abs(implied - spec_charge) > 1e-9)[0]
    if bad.size:
        detail = ", ".join(
            f"{species[j].name}: declared {spec_charge[j]:+.0f}, "
            f"stoichiometry implies {implied[j]:+.0f}"
            for j in bad
        )
        raise ValueError(
            f"Network {template.name!r} is not charge consistent: {detail} "
            "(#196)."
        )

    log_K = _resolve_log_K(template.name, species, elements, log10_K)

    roles = {c.role: i for i, c in enumerate(components) if c.role != "rare_earth"}
    for required in ("proton", "anion", "extractant"):
        if required not in roles:
            raise ValueError(
                f"Network {template.name!r} declares no component with role "
                f"{required!r}; the mass-action closure needs it to find the "
                f"{required} balance (#196)."
            )

    return ReactionNetwork(
        name=template.name,
        mechanism=template.mechanism,
        elements=elements,
        extractant_basis=template.extractant_basis,
        monomers_per_component=float(monomers_per_component),
        components=tuple(components),
        species=tuple(species),
        nu=nu,
        log10_K=log_K,
        species_is_aqueous=np.array(
            [s.phase == "aqueous" for s in species], dtype=bool
        ),
        component_is_aqueous=np.array(
            [c.phase == "aqueous" for c in components], dtype=bool
        ),
        component_charges=comp_charge,
        species_charges=spec_charge,
        element_component_index=tuple(element_component_index),
        element_species_index=tuple(element_species_index),
        proton_index=roles["proton"],
        counter_ion_index=roles.get("counter_ion"),
        anion_index=roles["anion"],
        extractant_index=roles["extractant"],
    )


def _charge_suffix(comp: Component) -> str:
    """Suffix used when expanding a per-element component name.

    ``"RE3+"`` becomes ``"Nd3+"``: the element symbol replaces the ``RE``
    placeholder and the charge decoration is kept.

    Args:
        comp: Template component row.

    Returns:
        The part of the template name that follows the ``RE`` placeholder.
    """
    if comp.name.startswith(_RE_TEMPLATE[:2]):
        return comp.name[2:]
    return comp.name


def _resolve_log_K(
    net_name: str,
    species: Sequence[Reaction],
    elements: Sequence[str],
    log10_K,
) -> Array:
    """Turn a user-supplied constant spec into a dense ``(n_species,)`` array.

    Args:
        net_name: Network name, for error messages.
        species: Expanded species rows.
        elements: Tracked elements.
        log10_K: Mapping by element symbol or by species name, or a flat
            sequence / array in species order.

    Returns:
        ``(n_species,)`` float64 JAX array.

    Raises:
        ValueError: If a species has no constant, or a flat sequence has the
            wrong length.
    """
    n = len(species)
    if isinstance(log10_K, Mapping):
        values = []
        for spec in species:
            if spec.name in log10_K:
                values.append(log10_K[spec.name])
            elif spec.element is not None and spec.element in log10_K:
                values.append(log10_K[spec.element])
            elif spec.log10_K is not None:
                values.append(spec.log10_K)
            else:
                raise ValueError(
                    f"Network {net_name!r}: no log10_K supplied for species "
                    f"{spec.name!r}. Key it by species name or by element "
                    f"symbol; tracked elements are {list(elements)} (#196)."
                )
        return jnp.asarray(values, dtype=jnp.float64)

    arr = jnp.asarray(log10_K, dtype=jnp.float64)
    if arr.shape != (n,):
        raise ValueError(
            f"Network {net_name!r}: log10_K given as a sequence must have "
            f"length {n} (one per species, order "
            f"{[s.name for s in species]}), got shape {arr.shape} (#196)."
        )
    return arr


# =============================================================================
# Calibration against the L1 correlation
# =============================================================================

def log_K_from_correlation(
    template: str | NetworkTemplate,
    elements: Sequence[str],
    extractant: str,
    calibration_pH: float = 3.0,
    T: float = 298.15,
    extractant_conc: float | None = None,
    anion_conc: float = 1.0,
    monomers_per_component: float | None = None,
    **distribution_kwargs: Any,
) -> dict[str, float]:
    """Calibrate one formation constant per element from the L1 correlation.

    This is the "continuation from the correlation" of #196 applied to the
    *parameters* rather than to the state: the closed model starts life
    reproducing the correlation exactly at one stated reference condition,
    and departs from it only where mass action says it must.

    In the dilute limit, where the free extractant is the total extractant and
    the released protons do not move the pH, the tableau gives

    .. math::

        D_i = \\frac{[\\mathrm{complex}_i]}{[\\mathrm{RE}_i^{3+}]}
            = K_i \\prod_{c \\neq \\mathrm{RE}_i} [C_c]^{\\nu_{ic}}

    so inverting at the reference condition is a closed-form algebraic step:

    .. math::

        \\log_{10} K_i = \\log_{10} D_i^{\\mathrm{corr}}
            - \\sum_{c \\neq \\mathrm{RE}_i} \\nu_{ic}\\,\\log_{10}[C_c]^{\\mathrm{ref}}

    The reference free concentrations are the proton at ``calibration_pH``,
    the extractant at ``extractant_conc`` converted to the component basis,
    and the anion at ``anion_conc``.

    Args:
        template: Network template name or object.
        elements: REE symbols.
        extractant: Extractant name, passed to
            :class:`~difflow_ree.equilibrium.distribution.REEDistribution`.
        calibration_pH: Reference pH (concentration scale) at which the
            correlation and the closure are made to agree. Choose the pH the
            cascade will actually run at: the correlation's pH slope is 2.2-2.9
            while mass action forces exactly ``protons_released``, so the two
            models separate away from this point (see
            :func:`correlation_ph_slope_defect`).
        T: Reference temperature (K).
        extractant_conc: Total extractant concentration (M, monomer basis).
            None uses the record's ``typical_concentration``.
        anion_conc: Reference free anion concentration (M). Only enters for
            networks whose complex contains the anion (solvating, anion
            exchange); it is exactly cancelled for cation exchange, where the
            anion coefficient is zero.
        monomers_per_component: Override for the monomer-to-component
            conversion; None derives it from the template basis.
        **distribution_kwargs: Forwarded to
            :class:`~difflow_ree.equilibrium.distribution.REEDistribution`
            (``medium``, ``mechanism``, ``nitrate_conc``, ...).

    Returns:
        Mapping of element symbol to ``log10 K``.

    Raises:
        ValueError: If the template's extractant basis disagrees with the
            extractant record's declared basis (#191).

    Example:
        >>> K = log_K_from_correlation(
        ...     "cation_exchange_dimer", ("Nd",), "D2EHPA", calibration_pH=3.0
        ... )
        >>> round(K["Nd"], 3)
        -7.454
    """
    from difflow_ree.equilibrium.distribution import REEDistribution

    if isinstance(template, str):
        template = get_network_template(template)
    ext = get_extractant(extractant)

    # The basis is stated in two places -- on the extractant record (#191) and
    # on the network row -- and a disagreement silently doubles or halves the
    # capacity.  Check rather than trust either one.
    if template.extractant_basis != ext.stoichiometry_basis:
        raise ValueError(
            f"Network {template.name!r} declares extractant_basis="
            f"{template.extractant_basis!r} but extractant {extractant!r} "
            f"declares stoichiometry.basis={ext.stoichiometry_basis!r}. Pick "
            f"the network whose basis matches the record, or fix the record "
            f"(#191, #196)."
        )

    if monomers_per_component is None:
        monomers_per_component = 2.0 if template.extractant_basis == "dimer" else 1.0
    if extractant_conc is None:
        extractant_conc = ext.typical_concentration

    net = build_network(
        template,
        elements,
        log10_K={el: 0.0 for el in elements},
        monomers_per_component=monomers_per_component,
    )

    dist = REEDistribution(
        extractant=extractant,
        elements=tuple(elements),
        concentration=extractant_conc,
        **distribution_kwargs,
    )

    # Reference free concentrations, one per component column.
    ref = np.ones(net.n_components, dtype=np.float64)
    ref[net.proton_index] = 10.0 ** (-float(calibration_pH))
    ref[net.anion_index] = float(anion_conc)
    ref[net.extractant_index] = float(extractant_conc) / monomers_per_component
    if net.counter_ion_index is not None:
        ref[net.counter_ion_index] = 1.0
    log_ref = np.log10(ref)

    out: dict[str, float] = {}
    for i, el in enumerate(elements):
        j = net.element_species_index[i]
        col = net.element_component_index[i]
        nu_row = net.nu[j].copy()
        nu_row[col] = 0.0  # the RE column is D's own denominator
        D = dist.get_D(
            el,
            pH=None if net.mechanism == "solvating" else calibration_pH,
            T=T,
        )
        out[el] = float(np.log10(float(D)) - float(nu_row @ log_ref))
    return out


def correlation_ph_slope_defect(extractant: str, element: str) -> float:
    """Gap between the correlation's pH slope and the mass-action slope.

    Mass action forces ``d log10 D / d pH = protons_released`` exactly: three
    for a trivalent ion on an acidic extractant. The tabulated correlations
    use element-specific slopes ``b`` between 2.20 and 2.90, so calibrating
    ``K`` at one pH and evaluating at another leaves a *predictable* gap,

    .. math::

        \\log_{10} D^{\\mathrm{closed}} - \\log_{10} D^{\\mathrm{corr}}
          = (p - b)\\,(\\mathrm{pH} - \\mathrm{pH}_{\\mathrm{ref}})
            - c\\,(\\mathrm{pH}^2 - \\mathrm{pH}_{\\mathrm{ref}}^2)

    with ``c`` the correlation's quadratic term. This function returns
    ``p - b``. It exists so that departure from the correlation can be
    asserted quantitatively in a test rather than absorbed into a loose
    tolerance, and so a user can see whether the correlation they are
    calibrating from is mass-action consistent at all.

    Args:
        extractant: Extractant name.
        element: REE symbol.

    Returns:
        ``protons_released - b`` in log10 units per pH unit. Zero would mean
        the correlation is exactly mass-action consistent in pH.

    Raises:
        ValueError: If the extractant carries no ``ph_coefficients`` block.

    Example:
        >>> round(correlation_ph_slope_defect("D2EHPA", "Nd"), 2)
        0.55
    """
    ext = get_extractant(extractant)
    if not ext.ph_coefficients:
        raise ValueError(
            f"Extractant {extractant!r} carries no 'ph_coefficients' block, "
            f"so it has no pH slope to compare against (#195, #196)."
        )
    return float(ext.stoichiometry_protons - ext.ph_coefficients[element].b)


def network_for_extractant(extractant: str) -> str:
    """Template name matching an extractant record's mechanism and basis.

    Args:
        extractant: Extractant name.

    Returns:
        A key of ``reaction_networks.yaml``.

    Raises:
        ValueError: If no shipped template matches the record.

    Example:
        >>> network_for_extractant("D2EHPA")
        'cation_exchange_dimer'
        >>> network_for_extractant("TBP")
        'solvating_nitrate'
    """
    ext = get_extractant(extractant)
    for name, tmpl in _load_templates().items():
        if tmpl.mechanism != ext.mechanism:
            continue
        if tmpl.extractant_basis == ext.stoichiometry_basis:
            return name
    raise ValueError(
        f"No shipped reaction network matches extractant {extractant!r} "
        f"(mechanism={ext.mechanism!r}, basis={ext.stoichiometry_basis!r}). "
        f"Available: {list_networks()} (#196)."
    )


def cation_exchange_network(
    extractant: str,
    elements: Sequence[str],
    calibration_pH: float = 3.0,
    T: float = 298.15,
    extractant_conc: float | None = None,
    log10_K: Mapping[str, float] | None = None,
    **distribution_kwargs: Any,
) -> ReactionNetwork:
    """Build the network for an extractant, calibrated from the correlation.

    This is the one-line entry point: it picks the template matching the
    extractant record's mechanism and basis, calibrates one constant per
    element from the L1 correlation unless constants are supplied, and
    returns the expanded network.

    Despite the name it is not restricted to cation exchange -- it dispatches
    on the record's mechanism, so a solvating extractant gets the solvating
    template (#195, #196).

    Args:
        extractant: Extractant name.
        elements: REE symbols to track.
        calibration_pH: Reference pH for the calibration; see
            :func:`log_K_from_correlation`.
        T: Reference temperature (K).
        extractant_conc: Total extractant concentration (M, monomer basis).
        log10_K: Measured constants, keyed by element symbol. Supplying these
            bypasses the calibration entirely and is what a user with real
            data should do.
        **distribution_kwargs: Forwarded to the correlation.

    Returns:
        The expanded :class:`ReactionNetwork`.

    Example:
        >>> net = cation_exchange_network("D2EHPA", ("Nd", "Dy"))
        >>> net.n_components
        6
    """
    template = get_network_template(network_for_extractant(extractant))
    ext = get_extractant(extractant)
    if extractant_conc is None:
        extractant_conc = ext.typical_concentration
    # A nitrate-requiring extractant's correlation is driven by the salting
    # anion, so it cannot be evaluated without one (#195).
    if ext.requires_nitrate and "nitrate_conc" not in distribution_kwargs:
        distribution_kwargs = dict(
            distribution_kwargs, nitrate_conc=ext.reference_nitrate
        )
    if log10_K is None:
        log10_K = log_K_from_correlation(
            template,
            elements,
            extractant,
            calibration_pH=calibration_pH,
            T=T,
            extractant_conc=extractant_conc,
            anion_conc=(
                ext.reference_nitrate if ext.reference_nitrate else 1.0
            ),
            **distribution_kwargs,
        )
    return build_network(template, elements, log10_K)
