"""The reforming reaction network and its rate laws.

The network is Smith's (1959) four reactions -- naphthene dehydrogenation to
aromatics, naphthene ring opening to paraffins (the reverse of
dehydrocyclization), and hydrocracking of paraffins and of naphthenes --
written per carbon number, C6 to C10, in the manner of Krane et al. (1959),
with three further steps the carbon-number view needs:

* ``isomerization`` of each n-paraffin to its iso-paraffin, because the
  octane of a paraffin lump depends on its branching;
* ``ring_expansion`` of methylcyclopentane to cyclohexane, because the C6
  naphthene in a naphtha is mostly the five-ring isomer, which must expand
  before it can dehydrogenate -- the reason benzene forms slowly;
* ``dealkylation`` of an aromatic to the next lighter one and methane.

With ``n`` the carbon number and ``P`` the total pressure (bar), the
reactions and their rates (mol/s per kg catalyst; partial pressures ``p`` in
bar) are:

====================  ===========================================  ==========================================
family                reaction                                     rate
====================  ===========================================  ==========================================
dehydrogenation       N_n = A_n + 3 H2                             k (p_N - p_A p_H2^3 / K)
ring_expansion        MCP = CH (C6 only)                           k (p_MCP - p_CH / K)
ring_opening          N_n + H2 = nP_n,  N_n + H2 = iP_n            k (p_N p_H2 - p_P / K)
isomerization         nP_n = iP_n                                  k (p_nP - p_iP / K)
hydrocracking_P       P_n + H2 -> lighter paraffins                k p_P / P
hydrocracking_N       N_n + 2 H2 -> lighter paraffins              k p_N / P
dealkylation          A_n + H2 -> A_(n-1) + CH4  (n >= 7)          k p_A p_H2 / P
====================  ===========================================  ==========================================

The forms of the first, third and the two hydrocracking rates are Smith's
(1959): first order in the reactant with the equilibrium approach for the
reversible steps, and hydrocracking first order in the reactant's *mole
fraction* (``p / P``), which is how Smith made it independent of pressure.
For the C6 ring opening the naphthene is methylcyclopentane (the ring a C6
paraffin closes to). The isomerization, ring-expansion and dealkylation
forms are this project's, written the same way (ILLUSTRATIVE).

Every equilibrium constant ``K`` is computed from the species' standard
Gibbs energies (:func:`difflow_refinery.reforming.thermo.ln_K`), not taken
from a kinetic paper, so the dehydrogenation equilibrium limits correctly at
low pressure and high temperature. Smith's own fitted ``Kp`` expressions are
not used.

Hydrocracking products: a paraffin ``C_n`` (or a naphthene after ring
opening) is split at one C-C bond chosen uniformly, so it yields
``2/(n-1)`` mol of each lighter paraffin ``C_k``, ``k = 1 .. n-1``; for
``k >= 4`` a fraction ``iso_fraction`` of it is the branched isomer. That
distribution conserves carbon and hydrogen exactly and is ILLUSTRATIVE --
acid-site cracking makes less methane and ethane than uniform scission, and
metal-site hydrogenolysis more.

Rate constants are Arrhenius in the reference-temperature form::

    k = activity * A * f(n) * exp(-E/R (1/T - 1/T_ref)),   T_ref = 773.15 K

Activation energies of the four Smith reactions are Smith's (1959): the
temperature coefficients of his expressions, 34750, 59600 and 62300 (in
degrees Rankine) for dehydrogenation, ring opening and both hydrocrackings,
converted to kelvin by dividing by 1.8. Those numbers are as Smith's model is
commonly reproduced in the reforming literature; the original paper was not
available to check them (unverified). The pre-exponentials ``A`` and the
carbon-number factors ``f(n)`` are NOT Smith's or Krane's: Smith's were for a
1950s monometallic catalyst, and Krane et al.'s table of rate constants
could not be checked. They were chosen by this project so that the model
behaves like a modern semi-regenerative unit (first-reactor temperature
drop, reformate octane and yields in the ranges commonly quoted for one) and
are ILLUSTRATIVE until fitted to a plant. The ``f(n)`` trends -- heavier
paraffins cyclize and crack faster, the C6 paraffins barely cyclize -- are
the trends Krane et al. reported, not their values.

KNOWN DEFECT of these illustrative constants: the rich (high-naphthene)
feed makes LESS net hydrogen than the lean (high-paraffin) one -- 2.4
against 2.8 wt% of feed at the default design -- the opposite of commercial
experience. The ring-opening pre-exponential and its carbon-number factors
make C7+ paraffin dehydrocyclization (4 H2 per aromatic formed, against 3
from a naphthene) fast enough that the lean feed converts most of its
paraffins, while the rich feed has few to convert and also loses hydrogen to
naphthene hydrocracking (2 H2 each). The constants were tuned for octane and
yield, not hydrogen. This is a defect of the parameter set, not of the
model, and fitting the kinetics is what removes it.

Coke make (kg coke per kg catalyst per second) is an illustrative
precursor-over-hydrogen form, ``k_c (p_N + p_A) / p_H2`` with an Arrhenius
``k_c``: coke rises with severity and falls with the hydrogen partial
pressure, which is the dependence the deactivation literature (e.g. Taskar &
Riggs 1997) describes; the form and constants are this project's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming import thermo as th

#: Reference temperature of the Arrhenius form (K).
T_REF_KINETICS = 773.15

#: Reaction families.
FAMILIES: tuple[str, ...] = ("dehydrogenation", "ring_expansion", "ring_opening", "isomerization",
                             "hydrocracking_P", "hydrocracking_N", "dealkylation")

#: Families with an equilibrium approach term.
REVERSIBLE: tuple[str, ...] = ("dehydrogenation", "ring_expansion", "ring_opening", "isomerization")

# Smith (1959) temperature coefficients, degrees Rankine (unverified, see module docstring).
SMITH_E_RANKINE: dict[str, float] = {
    "dehydrogenation": 34750.0, "ring_opening": 59600.0, "hydrocracking": 62300.0}


def _smith_E(key: str) -> float:
    return SMITH_E_RANKINE[key] / 1.8 * th.R


@dataclass
class ReformingKinetics(ParamsMixin):
    """Rate parameters of the reforming network.

    Attributes:
        A: Pre-exponential factor at ``T_ref`` per family (mol/s/kg cat per
            unit of the rate law's driving force, bar-based). ILLUSTRATIVE.
        E: Activation energy per family (J/mol). Smith (1959) for
            dehydrogenation, ring opening and the hydrocrackings
            (unverified transcription); this project's for the others.
        carbon_factor: Multiplier ``f(n)`` per family and carbon number
            (C6..C10). ILLUSTRATIVE; the trends are Krane et al.'s.
        iso_fraction: Branched share of the C4+ paraffins hydrocracking makes.
        activity: Catalyst activity, multiplying every rate constant (and the
            coke rate). 1 for fresh catalyst; the quantity a tracker would
            estimate from plant data as the catalyst deactivates.
        coke_A: Coke-make pre-exponential at ``T_ref`` (kg coke/kg cat/s).
        coke_E: Coke-make activation energy (J/mol).
    """

    A: dict = field(default_factory=lambda: {
        "dehydrogenation": 0.5,
        "ring_expansion": 0.02,
        "ring_opening": 0.5,
        "isomerization": 0.1,
        "hydrocracking_P": 0.1,
        "hydrocracking_N": 5.0e-3,
        "dealkylation": 5.0e-6,
    })
    E: dict = field(default_factory=lambda: {
        "dehydrogenation": _smith_E("dehydrogenation"),
        "ring_expansion": 150.0e3,
        "ring_opening": _smith_E("ring_opening"),
        "isomerization": 120.0e3,
        "hydrocracking_P": _smith_E("hydrocracking"),
        "hydrocracking_N": _smith_E("hydrocracking"),
        "dealkylation": 200.0e3,
    })
    carbon_factor: dict = field(default_factory=lambda: {
        "dehydrogenation": (0.3, 1.0, 1.2, 1.4, 1.6),
        "ring_opening": (0.2, 1.0, 1.5, 2.0, 2.5),
        "isomerization": (1.0, 1.0, 1.0, 1.0, 1.0),
        "hydrocracking_P": (0.3, 1.0, 2.5, 5.0, 8.0),
        "hydrocracking_N": (0.5, 1.0, 1.3, 1.6, 1.9),
        "dealkylation": (0.0, 1.0, 1.0, 1.0, 1.0),
        "ring_expansion": (1.0, 0.0, 0.0, 0.0, 0.0),
    })
    iso_fraction: float = 0.5
    activity: float = 1.0
    coke_A: float = 1.7e-7
    coke_E: float = 120.0e3


jax.tree_util.register_dataclass(
    ReformingKinetics,
    data_fields=["A", "E", "carbon_factor", "activity", "coke_A", "coke_E"],
    meta_fields=["iso_fraction"],
)


@dataclass(frozen=True)
class Reaction:
    """One reaction of the network (static structure)."""

    name: str
    family: str
    carbon: int
    reactant: str           # the species the rate is first order in
    product: str | None     # the equilibrium partner (reversible families)


def _cracked_products(n: int, iso: float, nu: np.ndarray):
    """Add the uniform-scission products of one C_n paraffin to ``nu`` (in place)."""
    w = 2.0 / (n - 1)
    for k in range(1, n):
        if k <= 3:
            nu[sp.INDEX[sp.paraffin(k, False)]] += w
        else:
            nu[sp.INDEX[sp.paraffin(k, True)]] += w * iso
            nu[sp.INDEX[sp.paraffin(k, False)]] += w * (1.0 - iso)


def build_network(iso_fraction: float = 0.5) -> tuple[tuple[Reaction, ...], np.ndarray]:
    """The reactions and their stoichiometry matrix ``(n_reactions, N_SPECIES)``.

    ``iso_fraction`` enters the cracking product rows, so it is a static
    (Python float) choice, not a traced parameter.
    """
    I = sp.INDEX
    H = I["H2"]
    rxns: list[Reaction] = []
    rows: list[np.ndarray] = []

    def add(rxn, coeffs):
        nu = np.zeros(sp.N_SPECIES)
        for k, v in coeffs.items():
            nu[I[k]] += v
        rxns.append(rxn)
        rows.append(nu)
        return nu

    for n in sp.CARBON_NUMBERS:
        N, A, nP, iP = f"N{n}", f"A{n}", f"nP{n}", f"iP{n}"
        add(Reaction(f"dehydrogenation_{n}", "dehydrogenation", n, N, A), {N: -1, A: 1, "H2": 3})
        ring = "N5_6" if n == 6 else N
        if n == 6:
            add(Reaction("ring_expansion_6", "ring_expansion", 6, "N5_6", "N6"), {"N5_6": -1, "N6": 1})
        for P in (nP, iP):
            add(Reaction(f"ring_opening_{P}", "ring_opening", n, ring, P), {ring: -1, "H2": -1, P: 1})
        add(Reaction(f"isomerization_{n}", "isomerization", n, nP, iP), {nP: -1, iP: 1})
        for P in (nP, iP):
            nu = add(Reaction(f"hydrocracking_{P}", "hydrocracking_P", n, P, None), {P: -1, "H2": -1})
            _cracked_products(n, iso_fraction, nu)
        for Nk in ((["N5_6", "N6"]) if n == 6 else [N]):
            nu = add(Reaction(f"hydrocracking_{Nk}", "hydrocracking_N", n, Nk, None), {Nk: -1, "H2": -2})
            _cracked_products(n, iso_fraction, nu)
        if n >= 7:
            add(Reaction(f"dealkylation_{n}", "dealkylation", n, A, None),
                {A: -1, "H2": -1, f"A{n - 1}": 1, "C1": 1})
    nu = np.stack(rows)
    # Every reaction conserves carbon and hydrogen exactly.
    assert np.allclose(nu @ sp.ELEMENTS.T, 0.0, atol=1e-12)
    del H
    return tuple(rxns), nu


class RateModel:
    """Vectorised rates of a network for given kinetic parameters.

    Built once per network (static structure: index arrays and the
    stoichiometry); the parameters are passed at call time, so they can be
    traced.
    """

    def __init__(self, iso_fraction: float = 0.5):
        self.reactions, self.nu = build_network(iso_fraction)
        self.iso_fraction = iso_fraction
        R = self.reactions
        I = sp.INDEX
        self.nu_j = jnp.asarray(self.nu)
        self.reactant = np.array([I[r.reactant] for r in R])
        self.product = np.array([I[r.product] if r.product else 0 for r in R])
        self.family = [r.family for r in R]
        self.carbon_slot = np.array([sp.CARBON_NUMBERS.index(r.carbon) for r in R])
        self.reversible = np.array([r.family in REVERSIBLE for r in R])
        # Which reversible reactions have hydrogen as a product (dehydrogenation,
        # H2 cubed) or as a reactant (ring opening, H2 to the first power).
        self.is_dehydro = np.array([r.family == "dehydrogenation" for r in R])
        self.is_opening = np.array([r.family == "ring_opening" for r in R])
        self.is_cracking = np.array([r.family in ("hydrocracking_P", "hydrocracking_N") for r in R])
        self.is_dealk = np.array([r.family == "dealkylation" for r in R])
        # Equilibrium rows: the reversible reactions' own stoichiometry.
        self.nu_eq = jnp.asarray(self.nu)
        self.coke_precursors = np.array([sp.SPECIES[k].kind in ("N", "A") for k in sp.NAMES])

    def rate_constants(self, kin: ReformingKinetics, T: Array) -> Array:
        """``(n_reactions,)`` rate constants at ``T``."""
        A = jnp.stack([jnp.asarray(kin.A[f], dtype=float) for f in self.family])
        E = jnp.stack([jnp.asarray(kin.E[f], dtype=float) for f in self.family])
        fac = jnp.stack([jnp.asarray(kin.carbon_factor[f][s], dtype=float)
                         for f, s in zip(self.family, self.carbon_slot)])
        return (jnp.asarray(kin.activity, dtype=float) * A * fac
                * jnp.exp(-E / th.R * (1.0 / T - 1.0 / T_REF_KINETICS)))

    def equilibrium_constants(self, T: Array) -> Array:
        """``(n_reactions,)`` equilibrium constants (1-bar basis); 1 for irreversible ones."""
        lnK = th.ln_K(self.nu_eq, T)
        return jnp.where(self.reversible, jnp.exp(jnp.where(self.reversible, lnK, 0.0)), 1.0)

    def rates(self, kin: ReformingKinetics, p: Array, T: Array, P: Array) -> Array:
        """``(n_reactions,)`` rates (mol/s/kg cat) at partial pressures ``p`` (bar)."""
        k = self.rate_constants(kin, T)
        K = self.equilibrium_constants(T)
        pH = p[sp.INDEX["H2"]]
        pr = p[self.reactant]
        pp = p[self.product]
        dehydro = pr - pp * pH**3 / K
        opening = pr * pH - pp / K
        simple = pr - pp / K
        reversible = jnp.where(self.is_dehydro, dehydro, jnp.where(self.is_opening, opening, simple))
        irreversible = jnp.where(self.is_dealk, pr * pH / P, pr / P)
        drive = jnp.where(self.reversible, reversible, irreversible)
        return k * drive

    def coke_rate(self, kin: ReformingKinetics, p: Array, T: Array) -> Array:
        """Coke make (kg coke/kg cat/s)."""
        kc = (jnp.asarray(kin.activity) * jnp.asarray(kin.coke_A)
              * jnp.exp(-jnp.asarray(kin.coke_E) / th.R * (1.0 / T - 1.0 / T_REF_KINETICS)))
        return kc * jnp.sum(jnp.where(self.coke_precursors, p, 0.0)) / p[sp.INDEX["H2"]]

    def heats_of_reaction(self, T: Array) -> Array:
        """``(n_reactions,)`` standard enthalpies of reaction at ``T`` (J/mol)."""
        return th.reaction_enthalpy(self.nu_j, T)
