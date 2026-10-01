"""Lumped cracking kinetics for the FCC riser.

A lumped scheme describes catalytic cracking as a handful of pseudo-species
("lumps") defined by boiling range, linked by irreversible reactions. Three
schemes are provided, all on one machinery (:class:`LumpScheme`):

======================  =====================================================
``weekman_nace_3``      gas oil -> gasoline -> (gas + coke); Weekman & Nace
                        (1970).
``lee_4``               gas oil -> gasoline, gas, coke; gasoline -> gas,
                        coke; Lee, Chen, Huang & Pan (1989).
``ancheyta_5``          gas oil -> gasoline, LPG, dry gas, coke; gasoline ->
                        LPG, dry gas, coke; Ancheyta-Juarez, Lopez-Isunza &
                        Aguilar-Rodriguez (1999).
======================  =====================================================

Rate laws (the form all three papers use): gas-oil cracking is SECOND order
in the gas-oil mass fraction, every other reaction FIRST order in its
reactant's mass fraction (Weekman & Nace 1970)::

    r_ij = k_ij(T) * phi * y_i^n_i ,   n_gas_oil = 2,  n_other = 1

``phi`` is the catalyst activity (deactivation, :func:`deactivation`).

Basis of the rate constants HERE. ``k`` is in ``1/s`` per unit
catalyst-to-oil ratio: the riser integrates ``dy/dt_c = (C/O) * r`` along
the catalyst residence time ``t_c`` (see :mod:`difflow_refinery.fcc.riser`),
which is the catalyst-holdup-per-oil formulation of Weekman (1968). Papers
report constants on several different bases (per weight hourly space
velocity, per gas concentration, per catalyst mass); none of them is the
one used here without conversion.

THE DEFAULT RATE CONSTANTS ARE ILLUSTRATIVE. They are NOT taken from any
table in the papers above -- those could not be checked against the papers
for this implementation, and the papers' constants are in their own bases
and fitted to one feed and one catalyst each. They were chosen so that a
typical vacuum gas oil at a typical riser outlet temperature gives yields in
the ranges commonly quoted for VGO FCC units (conversion ~70 wt%, gasoline
~45-50 wt%, coke ~4-6 wt%). The 3- and 4-lump defaults are aggregations of
the 5-lump set (same total rates), so the three schemes agree on gas-oil
conversion at the same conditions. Fit them to unit or MAT data before
reading any yield as a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

R_GAS = 8.314462618  # J/mol/K (CODATA 2018 exact value)

#: Reference temperature (K) at which the rate constants ``k_ref`` are
#: given: 500 C, the middle of the 480-520 C MAT range Ancheyta et al.
#: (1999) worked at (abstract).
T_REF_KINETICS = 773.15

#: Product groups every scheme's lumps are mapped onto, in this order.
GROUPS = ("gas_oil", "gasoline", "lpg", "dry_gas", "coke")


@dataclass(frozen=True)
class Reaction:
    """One lumped reaction ``reactant -> product``.

    Attributes:
        reactant: Lump consumed.
        product: Lump formed.
        k_ref: Rate constant at :data:`T_REF_KINETICS` (1/s per unit C/O).
        Ea: Apparent activation energy (J/mol).
    """

    reactant: str
    product: str
    k_ref: float
    Ea: float


@dataclass(frozen=True, eq=False)
class LumpScheme:
    """A lumped kinetic scheme.

    Attributes:
        name: Identifier.
        lumps: Lump names; the first is the feed lump (gas oil).
        reactions: The reaction network.
        order: Reaction order in the reactant per lump (2 for gas oil, 1
            otherwise in all three schemes).
        groups: ``{lump: {group: share}}`` -- how each lump's mass maps onto
            the product groups :data:`GROUPS`. A lump that is itself a group
            maps 1:1. A share given as a callable is evaluated on the unit's
            split parameters (``dry_gas_share``, ``coke_share``) at run
            time, so it stays traceable.
        source: Citation for the scheme's network.
    """

    name: str
    lumps: tuple[str, ...]
    reactions: tuple[Reaction, ...]
    order: tuple[int, ...]
    groups: Mapping[str, Mapping[str, object]]
    source: str

    @property
    def n(self) -> int:
        return len(self.lumps)

    def index(self, lump: str) -> int:
        return self.lumps.index(lump)

    def theta(self) -> dict[str, Array]:
        """The default kinetic parameters as arrays (``k_ref``, ``Ea``)."""
        return {
            "k_ref": jnp.asarray([r.k_ref for r in self.reactions], dtype=float),
            "Ea": jnp.asarray([r.Ea for r in self.reactions], dtype=float),
        }

    def stoichiometry(self) -> tuple[np.ndarray, np.ndarray]:
        """``(src, dst)`` index arrays of the reactions."""
        src = np.asarray([self.index(r.reactant) for r in self.reactions])
        dst = np.asarray([self.index(r.product) for r in self.reactions])
        return src, dst

    def rates(self, y: Array, T: Array, phi: Array, k_ref: Array, Ea: Array,
              multiplier: Array = 1.0) -> Array:
        """``dy/dt_c`` per unit C/O (mass fraction per s), every lump.

        Mass is conserved exactly: each reaction moves mass from one lump to
        another (lumps are on a mass basis, so a stoichiometric coefficient
        of one in mass is the whole stoichiometry).
        """
        src, dst = self.stoichiometry()
        k = k_ref * jnp.exp(-Ea / R_GAS * (1.0 / T - 1.0 / T_REF_KINETICS))
        order = jnp.asarray(self.order, dtype=float)
        yr = jnp.maximum(y[src], 0.0)
        r = multiplier * phi * k * yr ** order[src]
        dy = jnp.zeros_like(y)
        dy = dy.at[src].add(-r)
        dy = dy.at[dst].add(r)
        return dy

    def group_matrix(self, shares: Mapping[str, Array]) -> Array:
        """``(n_lumps, 5)`` matrix: lump mass -> product-group mass."""
        rows = []
        for lump in self.lumps:
            spec = self.groups[lump]
            row = []
            for g in GROUPS:
                v = spec.get(g, 0.0)
                if callable(v):
                    v = v(shares)
                row.append(jnp.asarray(v, dtype=float))
            rows.append(jnp.stack(row))
        return jnp.stack(rows)


# -----------------------------------------------------------------------------
# Default (illustrative) constants
# -----------------------------------------------------------------------------

#: Illustrative 5-lump constants: ``(reactant, product, k_ref [1/s per C/O]
#: at 500 C, Ea [J/mol])``. NOT from Ancheyta et al. (1999) Table(s): see the
#: module docstring. Activation energies are of the magnitude reported for
#: lumped FCC schemes (40-90 kJ/mol), ordered so that dry gas has the
#: highest and coke the lowest, the ordering the lumped-kinetics literature
#: generally reports (unverified for any one paper).
ILLUSTRATIVE_5LUMP: tuple[tuple[str, str, float, float], ...] = (
    ("gas_oil", "gasoline", 0.0900, 60.0e3),
    ("gas_oil", "lpg", 0.0150, 75.0e3),
    ("gas_oil", "dry_gas", 0.0035, 90.0e3),
    ("gas_oil", "coke", 0.0070, 40.0e3),
    ("gasoline", "lpg", 0.0070, 70.0e3),
    ("gasoline", "dry_gas", 0.0008, 90.0e3),
    ("gasoline", "coke", 0.0004, 50.0e3),
)


def _k(table, reactant, products):
    return sum(k for a, b, k, _ in table if a == reactant and b in products)


def _ea(table, reactant, products):
    """Rate-weighted activation energy of an aggregated reaction."""
    rows = [(k, e) for a, b, k, e in table if a == reactant and b in products]
    return sum(k * e for k, e in rows) / sum(k for k, _ in rows)


def ancheyta_5(table=ILLUSTRATIVE_5LUMP) -> LumpScheme:
    """The 5-lump scheme of Ancheyta-Juarez, Lopez-Isunza & Aguilar-Rodriguez (1999).

    Network: gas oil -> gasoline, LPG (C3-C4), dry gas (C2 and lighter,
    with H2), coke; gasoline -> LPG, dry gas, coke. Seven reaction
    constants; with the deactivation constant that makes the "eight kinetic
    constants" of the paper's abstract. The network as coded follows the
    abstract's description; the exact reaction set was not checked against
    the paper's scheme figure (unverified). Default constants are
    :data:`ILLUSTRATIVE_5LUMP`, not the paper's.
    """
    lumps = ("gas_oil", "gasoline", "lpg", "dry_gas", "coke")
    return LumpScheme(
        name="ancheyta_5",
        lumps=lumps,
        reactions=tuple(Reaction(a, b, k, e) for a, b, k, e in table),
        order=(2, 1, 1, 1, 1),
        groups={l: {l: 1.0} for l in lumps},
        source="Ancheyta-Juarez et al., Appl. Catal. A 177 (1999) 227-235",
    )


def lee_4(table=ILLUSTRATIVE_5LUMP) -> LumpScheme:
    """The 4-lump scheme of Lee, Chen, Huang & Pan (1989).

    Weekman & Nace's gas+coke lump split in two: gas oil -> gasoline, gas
    (C1-C4), coke; gasoline -> gas, coke. Default constants aggregate
    :data:`ILLUSTRATIVE_5LUMP` (gas = LPG + dry gas); the gas lump is split
    into LPG and dry gas by the parameter ``dry_gas_share``.
    """
    gas = ("lpg", "dry_gas")
    rx = (
        Reaction("gas_oil", "gasoline", _k(table, "gas_oil", ("gasoline",)), _ea(table, "gas_oil", ("gasoline",))),
        Reaction("gas_oil", "gas", _k(table, "gas_oil", gas), _ea(table, "gas_oil", gas)),
        Reaction("gas_oil", "coke", _k(table, "gas_oil", ("coke",)), _ea(table, "gas_oil", ("coke",))),
        Reaction("gasoline", "gas", _k(table, "gasoline", gas), _ea(table, "gasoline", gas)),
        Reaction("gasoline", "coke", _k(table, "gasoline", ("coke",)), _ea(table, "gasoline", ("coke",))),
    )
    return LumpScheme(
        name="lee_4",
        lumps=("gas_oil", "gasoline", "gas", "coke"),
        reactions=rx,
        order=(2, 1, 1, 1),
        groups={"gas_oil": {"gas_oil": 1.0}, "gasoline": {"gasoline": 1.0},
                "gas": {"lpg": lambda s: 1.0 - s["dry_gas_share"],
                        "dry_gas": lambda s: s["dry_gas_share"]},
                "coke": {"coke": 1.0}},
        source="Lee, Chen, Huang & Pan, Can. J. Chem. Eng. 67 (1989) 615 (unverified)",
    )


def weekman_nace_3(table=ILLUSTRATIVE_5LUMP) -> LumpScheme:
    """The 3-lump scheme of Weekman & Nace (1970).

    Gas oil -> gasoline (K1), gas oil -> C (K3), gasoline -> C (K2), C being
    light gas plus coke; gas-oil cracking second order, gasoline cracking
    first order (Weekman & Nace 1970). Default constants aggregate
    :data:`ILLUSTRATIVE_5LUMP`; the C lump is split into coke (share
    ``coke_share``) and gas, the gas into LPG and dry gas by
    ``dry_gas_share``.
    """
    c = ("lpg", "dry_gas", "coke")
    rx = (
        Reaction("gas_oil", "gasoline", _k(table, "gas_oil", ("gasoline",)), _ea(table, "gas_oil", ("gasoline",))),
        Reaction("gas_oil", "gas_coke", _k(table, "gas_oil", c), _ea(table, "gas_oil", c)),
        Reaction("gasoline", "gas_coke", _k(table, "gasoline", c), _ea(table, "gasoline", c)),
    )
    return LumpScheme(
        name="weekman_nace_3",
        lumps=("gas_oil", "gasoline", "gas_coke"),
        reactions=rx,
        order=(2, 1, 1),
        groups={"gas_oil": {"gas_oil": 1.0}, "gasoline": {"gasoline": 1.0},
                "gas_coke": {
                    "coke": lambda s: s["coke_share"],
                    "lpg": lambda s: (1.0 - s["coke_share"]) * (1.0 - s["dry_gas_share"]),
                    "dry_gas": lambda s: (1.0 - s["coke_share"]) * s["dry_gas_share"]}},
        source="Weekman & Nace, AIChE J. 16 (1970) 397-404",
    )


SCHEMES = {"weekman_nace_3": weekman_nace_3, "lee_4": lee_4, "ancheyta_5": ancheyta_5}
_DEFAULT_SCHEMES: dict[str, LumpScheme] = {}


def get_scheme(scheme: "str | LumpScheme") -> LumpScheme:
    """A :class:`LumpScheme` by name (or the scheme itself)."""
    if isinstance(scheme, LumpScheme):
        return scheme
    if scheme == "jacob_10":
        raise NotImplementedError(
            "the Jacob et al. (1976) 10-lump scheme is not implemented; "
            "see docs/unit-operations-refinery.md, 'FCC: not done'")
    if scheme not in SCHEMES:
        raise ValueError(f"unknown kinetic scheme {scheme!r}; known: {', '.join(SCHEMES)}")
    # One instance per name, so compiled solves are shared between units.
    if scheme not in _DEFAULT_SCHEMES:
        _DEFAULT_SCHEMES[scheme] = SCHEMES[scheme]()
    return _DEFAULT_SCHEMES[scheme]


# -----------------------------------------------------------------------------
# Deactivation and coke
# -----------------------------------------------------------------------------

def deactivation(kind: str, t_c: Array, coke_on_cat: Array, alpha: Array) -> Array:
    """Catalyst activity ``phi`` (0..1).

    * ``"time"``: ``phi = exp(-alpha t_c)``, the exponential time-on-stream
      decay of Weekman (1968), ``alpha`` in 1/s.
    * ``"coke"``: ``phi = exp(-alpha C_c)``, an exponential in coke on
      catalyst ``C_c`` (mass fraction), ``alpha`` dimensionless. This form
      is common in the riser-modelling literature; no single source is
      claimed for it here.
    """
    if kind == "time":
        return jnp.exp(-alpha * t_c)
    if kind == "coke":
        return jnp.exp(-alpha * coke_on_cat)
    raise ValueError(f"deactivation must be 'time' or 'coke', not {kind!r}")


def voorhies_coke(t_c: Array, A: Array, n: Array) -> Array:
    """Coke on catalyst ``C_c = A t_c^n`` (Voorhies 1945).

    Voorhies' empirical coke-time law (mass fraction coke on catalyst
    against catalyst residence time; Voorhies reported ``n`` near 0.5).
    Provided as a diagnostic -- e.g. to compare the kinetic coke lump with
    a Voorhies fit of the same data -- and not used by the riser, whose
    coke is a kinetic lump. ``A`` and ``n`` have no defaults: they are fitted
    per catalyst and feed.
    """
    return A * jnp.asarray(t_c, dtype=float) ** n
