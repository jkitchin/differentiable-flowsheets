"""Lumped hydrotreating kinetics: HDS by sulfur class, HDN, aromatics and olefin saturation, a cracking leak.

A :class:`HDTKinetics` is a kinetic model for
:class:`~difflow_refinery.hydroprocessing.reactor.TrickleBedReactor`. Its
per-cut attributes (:data:`HDT_ATTRIBUTES`) are carbon and hydrogen atoms,
sulfur atoms in each of the composition module's five
:data:`~difflow_refinery.composition.SULFUR_CLASSES`, nitrogen atoms in its two
:data:`~difflow_refinery.composition.NITROGEN_CLASSES`, and counts of
molecules that are mono-, di- and poly-aromatic, olefinic and naphthenic (the
paraffins are the rest).

Reactions (each per cut, on the cut's own attribute concentrations; ``c`` are
the fugacity-equivalent liquid concentrations of
:class:`~difflow_refinery.hydroprocessing.reactor.ReactionContext`, mol/m^3)::

    HDS, class j:   S_j + nu_j H2 -> H2S       r = f k_j c_Sj h(c_H2) / D^2
                                               D = 1 + K_H2S c_H2S + K_N c_Nbasic
    HDN, class j:   N_j + nu_j H2 -> NH3       r = f k_j c_Nj h(c_H2) / (1 + K_H2S c_H2S)
    poly-aromatic + 2 H2 <-> di-aromatic       r = f k (c_A3 - c_A2 / (K3 (pH2/p0)^2))
    di-aromatic   + 2 H2 <-> mono-aromatic     r = f k (c_A2 - c_A1 / (K2 (pH2/p0)^2))
    mono-aromatic + 3 H2 <-> naphthene         r = f k (c_A1 - c_Nn / (K1 (pH2/p0)^3))
    olefin        +   H2 -> paraffin           r = f k c_O h(c_H2)
    cut molecule  +   H2 -> lighter molecule + C1-C4 gas   (the cracking leak)

with ``f = activity * effectiveness * wetting``, Arrhenius ``k(T) = k_ref
exp(-E/R (1/T - 1/T_ref))``, ``h(c_H2) = (c_H2/c_ref)^m``, the aromatics
rates also multiplied by ``h``, and adsorption constants ``K(T) = K_ref
exp(-dH_ads/R (1/T - 1/T_ref))`` (``dH_ads < 0``: adsorption weakens as T
rises). Every rate is per kg of catalyst.

* **The HDS form** is the Langmuir-Hinshelwood-Hougen-Watson rate with a
  squared H2S-inhibition denominator, the form of Korsten & Hoffmann (1996)
  and, with a richer denominator, of Froment, Depauw & Vanrysselberghe (1994)
  and Vanrysselberghe & Froment (1996). It is first order in each class: a
  sum of first-order classes with different constants is what gives a lumped
  total-sulfur rate its apparent order above one (the reason Korsten &
  Hoffmann fitted an order of about 1.6-1.8 to total sulfur). Basic nitrogen
  is in the denominator because it adsorbs on the same sites. ``hds_form=
  "power"`` gives the nth-order power-law fallback
  ``r = f k c_ref,S (c_Sj/c_ref,S)^n h(c_H2)`` with no inhibition.
* **Aromatics saturation** is first order and reversible; the equilibrium
  constants come from the ideal-gas thermochemistry of model compounds
  (:data:`AROMATIC_THERMO`), ``K = exp(-(dH - T dS)/(R T))`` with pressures in
  bar. Saturation is exothermic and loses three or two moles of gas, so the
  equilibrium recedes with temperature and aromatics pass through a minimum.
* **H2 stoichiometry** per class (:data:`HDS_H2`, :data:`HDN_H2`) is that of a
  model compound of the class, and the hydrogen not leaving as H2S or NH3
  goes onto the cut (``H += 2 nu - 2`` per sulfur, ``2 nu - 3`` per
  nitrogen). The molecule count of a cut does not change (a desulfurized
  molecule keeps its carbon skeleton). Element balances are exact.
* **Heats of reaction** per event (:data:`HDS_HEAT`, ...) are ideal-gas
  reaction enthalpies at 298 K of the same model compounds, so stoichiometry
  and heat are consistent. They neglect the heats of vaporisation of the
  reacting species (the bed is mostly liquid) and the temperature dependence.
* **The cracking leak** moves a molecule of cut ``i`` to the cut whose carbon
  number per molecule is nearest to ``i``'s less the gas fragment's
  (:attr:`HDTKinetics.crack_targets`, fixed at construction), splitting off
  one C1-C4 molecule in the proportions :data:`CRACK_GAS_SPLIT`, with one H2.
  The fragment takes the parent's sulfur, nitrogen and type attributes.

Where the numbers come from -- read this before trusting a ppm:

* The RATE FORMS are the literature's (above). The RATE CONSTANTS, activation
  energies and adsorption constants in :class:`HDTKineticParams` are
  ILLUSTRATIVE: chosen here so that a straight-run diesel at 350 C, LHSV 1/h,
  50 bar and 300 Nm3/m3 desulfurizes to a few hundred ppm and to ULSD at
  370-380 C, the right order of magnitude for a CoMo catalyst. They are not
  Korsten & Hoffmann's, not Froment's, and not any commercial catalyst's.
  Product sulfur to 10 ppm is predictive only after the activity and the
  refractory-class constants are fitted to the unit's own data
  (``difflow.estimation``, or ``difflow.reconciliation.tracking`` for the
  activity as it drifts).
* The THERMOCHEMISTRY (heats of reaction, equilibrium) is computed from
  ideal-gas formation enthalpies and entropies of model compounds as
  tabulated in the ``chemicals`` package (C. Bell et al., which transcribes
  TRC/ATcT/CRC sources; the per-compound primary source is unverified). The
  derivations are in the docs and pinned in the tests. Lumping a class of
  real molecules to one model compound is the approximation (class (b) of
  the issue's sources table).

References:
    Korsten, H. and Hoffmann, U., "Three-phase reactor model for
        hydrotreating in pilot trickle-bed reactors", AIChE J. 42(5),
        1350-1360 (1996), doi:10.1002/aic.690420515 (pages and DOI as
        recalled; not checked against the paper -- unverified).
    Froment, G.F., Depauw, G.A. and Vanrysselberghe, V., "Kinetic modeling
        and reactor simulation in hydrodesulfurization of oil fractions",
        Ind. Eng. Chem. Res. 33(12), 2975-2988 (1994), doi:10.1021/ie00036a011
        (title, pages and DOI unverified).
    Vanrysselberghe, V. and Froment, G.F., "Hydrodesulfurization of
        dibenzothiophene on a CoMo/Al2O3 catalyst: reaction network and
        kinetics", Ind. Eng. Chem. Res. 35(10), 3311-3318 (1996),
        doi:10.1021/ie960099t (pages and DOI unverified).
    Girgis, M.J. and Gates, B.C., "Reactivities, reaction networks, and
        kinetics in high-pressure catalytic hydroprocessing", Ind. Eng. Chem.
        Res. 30(9), 2021-2058 (1991), doi:10.1021/ie00057a001 (unverified) --
        the class reactivity order and the DDS/HYD routes.
    Mederos, F.S., Elizalde, I. and Ancheyta, J., "Steady-state and dynamic
        reactor models for hydrotreatment of oil fractions: a review", Catal.
        Rev. Sci. Eng. 51(4), 485-607 (2009), doi:10.1080/01614940903048661
        (DOI unverified).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow_refinery.composition import NITROGEN_CLASSES, SULFUR_CLASSES
from difflow_refinery.hydroprocessing.layout import GAS_ELEMENTS, Layout
from difflow_refinery.hydroprocessing.reactor import Rates, ReactionContext

jax.config.update("jax_enable_x64", True)

R_GAS = 8.314462618
P0_BAR = 1e5

#: Aromatic classes tracked per cut (molecules with 1, 2, 3+ aromatic rings).
AROMATIC_CLASSES: tuple[str, ...] = ("mono", "di", "poly")

#: The hydrotreater's per-cut attributes, in array order.
HDT_ATTRIBUTES: tuple[str, ...] = (
    ("C", "H") + tuple(f"S_{c}" for c in SULFUR_CLASSES) + tuple(f"N_{c}" for c in NITROGEN_CLASSES)
    + tuple(f"A_{c}" for c in AROMATIC_CLASSES) + ("olefins", "naphthenes"))
#: Element counted by each attribute (``None``: a molecule count).
HDT_ATTRIBUTE_ELEMENTS: tuple[str | None, ...] = (
    ("C", "H") + ("S",) * len(SULFUR_CLASSES) + ("N",) * len(NITROGEN_CLASSES)
    + (None,) * (len(AROMATIC_CLASSES) + 2))

# -----------------------------------------------------------------------------
# Model-compound thermochemistry (ideal gas, 298.15 K): formation enthalpy
# (J/mol) and absolute entropy (J/mol/K), as tabulated in the `chemicals`
# package (Bell et al. 2016-, v1.5.2): Hfg and S0g. Primary sources per
# compound unverified (see the module docstring).
# -----------------------------------------------------------------------------
MODEL_COMPOUNDS: dict[str, tuple[float, float | None]] = {
    "hydrogen": (0.0, 130.7),
    "hydrogen_sulfide": (-20600.0, 205.8),
    "ammonia": (-45558.0, 192.8),
    "benzene": (83180.0, 269.2),
    "cyclohexane": (-122080.0, 298.19),
    "naphthalene": (150600.0, 333.1),
    "tetralin": (26000.0, 366.22),
    "phenanthrene": (207500.0, 396.01),
    "tetrahydrophenanthrene": (92300.0, None),
    "diethyl_sulfide": (-83500.0, 368.1),
    "ethane": (-83780.0, 229.2),
    "thiophene": (114900.0, 278.8),
    "n_butane": (-125850.0, 304.4),
    "benzothiophene": (166300.0, 212.76),
    "ethylbenzene": (29900.0, 360.6),
    "dibenzothiophene": (205100.0, None),
    "biphenyl": (181400.0, 391.24),
    "cyclohexylbenzene": (-16700.0, 429.36),
    "quinoline": (200500.0, 366.04),
    "propylbenzene": (7900.0, 397.86),
    "carbazole": (200700.0, 244.95),
    "1_hexene": (-43500.0, 383.84),
    "n_hexane": (-166940.0, 388.82),
}


def _dH(products: dict, reactants: dict) -> float:
    H = lambda side: sum(n * MODEL_COMPOUNDS[s][0] for s, n in side.items())
    return H(products) - H(reactants)


def _dS(products: dict, reactants: dict) -> float:
    S = lambda side: sum(n * MODEL_COMPOUNDS[s][1] for s, n in side.items())
    return S(products) - S(reactants)


#: H2 consumed per sulfur atom removed, by class: diethyl sulfide + 2 H2 ->
#: 2 ethane + H2S; thiophene + 4 H2 -> n-butane + H2S; benzothiophene + 3 H2
#: -> ethylbenzene + H2S; dibenzothiophene by 80 % direct desulfurization
#: (DBT + 2 H2 -> biphenyl + H2S) and 20 % hydrogenation (DBT + 5 H2 ->
#: cyclohexylbenzene + H2S); the hindered 4,6-alkyl DBTs 35 % / 65 % (they
#: react mainly through the hydrogenation route). The route shares are
#: ILLUSTRATIVE, set by the qualitative finding (Girgis & Gates 1991,
#: Vanrysselberghe & Froment 1996) that CoMo removes DBT mainly by direct
#: desulfurization and 4,6-DMDBT mainly after ring hydrogenation.
HDS_ROUTE_DDS: tuple[float, ...] = (1.0, 1.0, 1.0, 0.80, 0.35)
_DH_DBT_DDS = _dH({"biphenyl": 1, "hydrogen_sulfide": 1}, {"dibenzothiophene": 1})
_DH_DBT_HYD = _dH({"cyclohexylbenzene": 1, "hydrogen_sulfide": 1}, {"dibenzothiophene": 1})
HDS_H2: tuple[float, ...] = (2.0, 4.0, 3.0, 0.80 * 2 + 0.20 * 5, 0.35 * 2 + 0.65 * 5)
#: Heat of reaction per sulfur atom removed (J/mol), same model compounds.
HDS_HEAT: tuple[float, ...] = (
    _dH({"ethane": 2, "hydrogen_sulfide": 1}, {"diethyl_sulfide": 1}),
    _dH({"n_butane": 1, "hydrogen_sulfide": 1}, {"thiophene": 1}),
    _dH({"ethylbenzene": 1, "hydrogen_sulfide": 1}, {"benzothiophene": 1}),
    0.80 * _DH_DBT_DDS + 0.20 * _DH_DBT_HYD,
    0.35 * _DH_DBT_DDS + 0.65 * _DH_DBT_HYD,
)
#: H2 per nitrogen atom: basic as quinoline + 4 H2 -> propylbenzene + NH3;
#: non-basic as carbazole + 5 H2 -> cyclohexylbenzene + NH3.
HDN_H2: tuple[float, ...] = (4.0, 5.0)
HDN_HEAT: tuple[float, ...] = (
    _dH({"propylbenzene": 1, "ammonia": 1}, {"quinoline": 1}),
    _dH({"cyclohexylbenzene": 1, "ammonia": 1}, {"carbazole": 1}),
)
#: Aromatic saturation steps (poly -> di, di -> mono, mono -> naphthene):
#: H2 per step and the model-compound thermochemistry ``(dH J/mol, dS J/mol/K)``:
#: phenanthrene + 2 H2 -> 1,2,3,4-tetrahydrophenanthrene (dH; its entropy is
#: not tabulated, so dS is taken from the di step), naphthalene + 2 H2 ->
#: tetralin, benzene + 3 H2 -> cyclohexane.
HDA_H2: tuple[float, ...] = (2.0, 2.0, 3.0)
_DS_DI = _dS({"tetralin": 1}, {"naphthalene": 1, "hydrogen": 2})
AROMATIC_THERMO: tuple[tuple[float, float], ...] = (
    (_dH({"tetrahydrophenanthrene": 1}, {"phenanthrene": 1}), _DS_DI),
    (_dH({"tetralin": 1}, {"naphthalene": 1}), _DS_DI),
    (_dH({"cyclohexane": 1}, {"benzene": 1}), _dS({"cyclohexane": 1}, {"benzene": 1, "hydrogen": 3})),
)
#: Olefin saturation, 1-hexene + H2 -> n-hexane.
OLEFIN_HEAT: float = _dH({"n_hexane": 1}, {"1_hexene": 1})
#: Cracking leak per event, n-hexane + H2 -> n-butane + ethane.
CRACK_HEAT: float = _dH({"n_butane": 1, "ethane": 1}, {"n_hexane": 1})
#: Gas molecule split off per cracking event, by species (mole shares;
#: ILLUSTRATIVE: a hydrotreater's off-gas is mostly C3-C4).
CRACK_GAS_SPLIT: dict[str, float] = {
    "methane": 0.10, "ethane": 0.15, "propane": 0.35, "isobutane": 0.15, "n_butane": 0.25}


@dataclass
class HDTKineticParams(ParamsMixin):
    """Rate constants and adsorption constants of :class:`HDTKinetics`.

    Rate constants are first-order, m^3 of liquid per kg catalyst per s, at
    ``T_ref``. All ILLUSTRATIVE (see the module docstring): the forms are the
    literature's, the numbers are chosen for CoMo-like trends.

    Attributes:
        activity: Catalyst activity multiplier a(t) on every rate (1 fresh).
        effectiveness: Catalyst effectiveness factor (multiplier).
        wetting: Catalyst wetting efficiency (multiplier).
        T_ref: Reference temperature of every Arrhenius and van 't Hoff term (K).
        c_ref: Reference H2 concentration of ``h(c_H2) = (c_H2/c_ref)^m`` (mol/m^3).
        hds_form: ``"lhhw"`` (default) or ``"power"`` (the nth-order fallback).
        hds_k: Rate constants per sulfur class (sulfides, thiophenes,
            benzothiophenes, dibenzothiophenes, hindered DBTs), m^3/kg/s.
        hds_E: Activation energies per sulfur class, J/mol.
        hds_h2_order: Order ``m`` in H2 of the HDS rates.
        hds_n: Order in the class concentration of the ``"power"`` form.
        c_ref_s: Reference sulfur concentration of the ``"power"`` form (mol/m^3).
        K_h2s: H2S adsorption constant at ``T_ref``, m^3/mol.
        dH_h2s: H2S adsorption enthalpy, J/mol (negative).
        K_nbasic: Basic-nitrogen adsorption constant at ``T_ref``, m^3/mol.
        dH_nbasic: Basic-nitrogen adsorption enthalpy, J/mol (negative).
        hdn_k: Rate constants (basic, non-basic), m^3/kg/s.
        hdn_E: Activation energies, J/mol.
        hdn_h2_order: Order in H2 of HDN.
        hda_k: Forward rate constants (poly->di, di->mono, mono->naphthene), m^3/kg/s.
        hda_E: Their activation energies, J/mol.
        hda_h2_order: Order in H2 of the forward aromatics rates.
        olefin_k: Olefin saturation rate constant, m^3/kg/s.
        olefin_E: Its activation energy, J/mol.
        crack_k: Cracking-leak rate constant (per cut molecule), m^3/kg/s.
        crack_E: Its activation energy, J/mol.
    """

    activity: float = 1.0
    effectiveness: float = 1.0
    wetting: float = 1.0
    T_ref: float = 623.15
    c_ref: float = 1000.0
    hds_form: str = "lhhw"
    hds_k: tuple = (4.0e-5, 2.0e-5, 1.2e-5, 1.0e-5, 5.0e-6)
    hds_E: tuple = (90e3, 100e3, 110e3, 125e3, 140e3)
    hds_h2_order: float = 0.5
    hds_n: float = 1.0
    c_ref_s: float = 10.0
    K_h2s: float = 0.02
    dH_h2s: float = -40e3
    K_nbasic: float = 0.10
    dH_nbasic: float = -40e3
    hdn_k: tuple = (4.0e-7, 7.0e-7)
    hdn_E: tuple = (130e3, 120e3)
    hdn_h2_order: float = 0.5
    hda_k: tuple = (4.0e-6, 1.5e-6, 3.0e-7)
    hda_E: tuple = (80e3, 90e3, 100e3)
    hda_h2_order: float = 1.0
    olefin_k: float = 5.0e-5
    olefin_E: float = 60e3
    crack_k: float = 3.0e-10
    crack_E: float = 200e3


jax.tree_util.register_dataclass(
    HDTKineticParams,
    data_fields=[f for f in HDTKineticParams.__dataclass_fields__ if f != "hds_form"],
    meta_fields=["hds_form"],
)


#: An ILLUSTRATIVE parameter set for a NAPHTHA hydrotreater (#332): the
#: diesel constants of :class:`HDTKineticParams` with
#:
#: * HDS of sulfides, thiophenes and benzothiophenes ten times faster -- in
#:   a naphtha those classes are mercaptans, light sulfides and
#:   (alkyl)thiophenes, which react far faster than the diesel-range
#:   molecules the default constants stand for (the class reactivity order
#:   of Girgis & Gates 1991, unverified; the factor is chosen here);
#: * HDN a hundred times faster (light pyrroles and pyridines; a reformer
#:   feed needs N as well as S below about 0.5 wppm);
#: * aromatics saturation ten times slower on every step, as on a CoMo
#:   naphtha catalyst at 20-35 bar, where benzene and the alkylbenzenes pass
#:   through largely unsaturated. Slowing the di and poly steps also tames
#:   an artefact of the lumping: the di <-> mono equilibrium is that of
#:   naphthalene/tetralin, so in a cut with no di-aromatics its reverse rate
#:   turns an alkylbenzene into "di-aromatic" at low H2 pressure, which no
#:   real naphtha does.
#:
#: Chosen so that a straight-run naphtha at 30 bar, 320 C bed inlet, LHSV
#: 4 1/h and 100 Nm3/m3 reaches below 0.5 wppm S (a reformer feed): see the
#: docs ("A naphtha hydrotreater"). Not any catalyst's constants.
NAPHTHA_HDT_PARAMS = HDTKineticParams(
    hds_k=(4.0e-4, 2.0e-4, 1.2e-4, 1.0e-5, 5.0e-6),
    hdn_k=(4.0e-5, 7.0e-5),
    hda_k=(4.0e-7, 1.5e-7, 3.0e-8),
)


def _arr(k, E, T, T_ref):
    return jnp.asarray(k, dtype=float) * jnp.exp(-jnp.asarray(E, dtype=float) / R_GAS * (1.0 / T - 1.0 / T_ref))


@dataclass(frozen=True)
class HDTKinetics:
    """The hydrotreating kinetic model (see the module docstring).

    Attributes:
        layout: The stream layout; must carry :data:`HDT_ATTRIBUTES` and the
            gases hydrogen, hydrogen_sulfide, ammonia and those of
            :data:`CRACK_GAS_SPLIT`.
        crack_targets: For each cut, the cut a cracked molecule lands in
            (static; from :func:`crack_targets`).
    """

    layout: Layout
    crack_targets: tuple[int, ...]
    attributes: tuple[str, ...] = HDT_ATTRIBUTES
    attribute_elements: tuple = HDT_ATTRIBUTE_ELEMENTS

    def __post_init__(self):
        # The layout's attributes must START with HDT_ATTRIBUTES; a unit that
        # tracks more per-cut attributes (the hydrocracker's "cracked" count)
        # appends them, and the cracking leak carries them with the molecule.
        if tuple(self.layout.attributes[:len(HDT_ATTRIBUTES)]) != HDT_ATTRIBUTES:
            raise ValueError("the layout's attributes must be (or begin with) HDT_ATTRIBUTES")
        need = ("hydrogen", "hydrogen_sulfide", "ammonia") + tuple(CRACK_GAS_SPLIT)
        missing = [g for g in need if g not in self.layout.gases]
        if missing:
            raise ValueError(f"the layout needs gases {missing}")
        if len(self.crack_targets) != self.layout.n_cut:
            raise ValueError("crack_targets needs one entry per cut")

    def rates(self, ctx: ReactionContext, p: HDTKineticParams) -> Rates:
        lay = self.layout
        T = ctx.T
        Tr = p.T_ref
        ai = {a: i for i, a in enumerate(HDT_ATTRIBUTES)}
        gi = {g: lay.gas_index(g) for g in lay.gases}
        f = p.activity * p.effectiveness * p.wetting
        cA = ctx.c_attr
        c_h2 = ctx.c_gas("hydrogen")
        c_h2s = ctx.c_gas("hydrogen_sulfide")
        pH2 = ctx.p_gas("hydrogen") / P0_BAR
        h2r = jnp.maximum(c_h2, 0.0) / p.c_ref

        d_attr = jnp.zeros((lay.n_cut, lay.n_attr))
        d_gas = jnp.zeros(lay.n_gas)
        d_cut = jnp.zeros(lay.n_cut)
        heat = jnp.asarray(0.0)

        # --- HDS -----------------------------------------------------------
        s_idx = jnp.asarray([ai[f"S_{c}"] for c in SULFUR_CLASSES])
        cS = cA[:, s_idx]                                              # (nc, 5)
        kS = _arr(p.hds_k, p.hds_E, T, Tr)                             # (5,)
        K_h2s = p.K_h2s * jnp.exp(-p.dH_h2s / R_GAS * (1.0 / T - 1.0 / Tr))
        K_nb = p.K_nbasic * jnp.exp(-p.dH_nbasic / R_GAS * (1.0 / T - 1.0 / Tr))
        c_nb = jnp.sum(cA[:, ai["N_basic"]])
        if p.hds_form == "power":
            rS = f * kS * p.c_ref_s * (jnp.maximum(cS, 0.0) / p.c_ref_s) ** p.hds_n * h2r ** p.hds_h2_order
        elif p.hds_form == "lhhw":
            D = 1.0 + K_h2s * c_h2s + K_nb * c_nb
            rS = f * kS * cS * h2r ** p.hds_h2_order / D**2
        else:
            raise ValueError(f"hds_form must be 'lhhw' or 'power', not {p.hds_form!r}")
        nuS = jnp.asarray(HDS_H2)
        d_attr = d_attr.at[:, s_idx].add(-rS)
        d_attr = d_attr.at[:, ai["H"]].add(rS @ (2.0 * nuS - 2.0))
        d_gas = d_gas.at[gi["hydrogen_sulfide"]].add(jnp.sum(rS))
        d_gas = d_gas.at[gi["hydrogen"]].add(-jnp.sum(rS @ nuS))
        heat = heat - jnp.sum(rS @ jnp.asarray(HDS_HEAT))

        # --- HDN -----------------------------------------------------------
        n_idx = jnp.asarray([ai[f"N_{c}"] for c in NITROGEN_CLASSES])
        cN = cA[:, n_idx]
        kN = _arr(p.hdn_k, p.hdn_E, T, Tr)
        rN = f * kN * cN * h2r ** p.hdn_h2_order / (1.0 + K_h2s * c_h2s)
        nuN = jnp.asarray(HDN_H2)
        d_attr = d_attr.at[:, n_idx].add(-rN)
        d_attr = d_attr.at[:, ai["H"]].add(rN @ (2.0 * nuN - 3.0))
        d_gas = d_gas.at[gi["ammonia"]].add(jnp.sum(rN))
        d_gas = d_gas.at[gi["hydrogen"]].add(-jnp.sum(rN @ nuN))
        heat = heat - jnp.sum(rN @ jnp.asarray(HDN_HEAT))

        # --- aromatics (reversible) -----------------------------------------
        kA = _arr(p.hda_k, p.hda_E, T, Tr) * h2r ** p.hda_h2_order     # (3,)
        dHs = jnp.asarray([t[0] for t in AROMATIC_THERMO])
        dSs = jnp.asarray([t[1] for t in AROMATIC_THERMO])
        nuA = jnp.asarray(HDA_H2)
        lnK = -(dHs - T * dSs) / (R_GAS * T)
        lnK_p = lnK + nuA * jnp.log(jnp.maximum(pH2, 1e-30))        # products/reactant at equilibrium
        inv = jnp.exp(-lnK_p)
        a3, a2, a1, nn = (cA[:, ai["A_poly"]], cA[:, ai["A_di"]], cA[:, ai["A_mono"]], cA[:, ai["naphthenes"]])
        r3 = f * kA[0] * (a3 - a2 * inv[0])
        r2 = f * kA[1] * (a2 - a1 * inv[1])
        r1 = f * kA[2] * (a1 - nn * inv[2])
        d_attr = d_attr.at[:, ai["A_poly"]].add(-r3)
        d_attr = d_attr.at[:, ai["A_di"]].add(r3 - r2)
        d_attr = d_attr.at[:, ai["A_mono"]].add(r2 - r1)
        d_attr = d_attr.at[:, ai["naphthenes"]].add(r1)
        d_attr = d_attr.at[:, ai["H"]].add(2.0 * nuA[0] * r3 + 2.0 * nuA[1] * r2 + 2.0 * nuA[2] * r1)
        d_gas = d_gas.at[gi["hydrogen"]].add(-jnp.sum(nuA[0] * r3 + nuA[1] * r2 + nuA[2] * r1))
        heat = heat - jnp.sum(dHs[0] * r3 + dHs[1] * r2 + dHs[2] * r1)

        # --- olefins -------------------------------------------------------
        rO = f * _arr(p.olefin_k, p.olefin_E, T, Tr) * cA[:, ai["olefins"]] * h2r ** p.hds_h2_order
        d_attr = d_attr.at[:, ai["olefins"]].add(-rO)
        d_attr = d_attr.at[:, ai["H"]].add(2.0 * rO)
        d_gas = d_gas.at[gi["hydrogen"]].add(-jnp.sum(rO))
        heat = heat - OLEFIN_HEAT * jnp.sum(rO)

        # --- cracking leak ---------------------------------------------------
        rC = f * _arr(p.crack_k, p.crack_E, T, Tr) * ctx.c_cut                 # (nc,) molecules
        pm = ctx.per_molecule
        names = tuple(CRACK_GAS_SPLIT)
        shares = jnp.asarray([CRACK_GAS_SPLIT[g] for g in names])
        nC = sum(CRACK_GAS_SPLIT[g] * GAS_ELEMENTS[g]["C"] for g in names)
        nH = sum(CRACK_GAS_SPLIT[g] * GAS_ELEMENTS[g]["H"] for g in names)
        tgt = jnp.asarray(self.crack_targets)
        moved = rC[:, None] * pm                                              # (nc, na)
        d_attr = d_attr - moved
        d_attr = d_attr.at[tgt].add(moved)
        d_cut = d_cut - rC
        d_cut = d_cut.at[tgt].add(rC)
        d_attr = d_attr.at[tgt, ai["C"]].add(-nC * rC)
        d_attr = d_attr.at[tgt, ai["H"]].add((2.0 - nH) * rC)
        for g, s in zip(names, shares):
            d_gas = d_gas.at[gi[g]].add(s * jnp.sum(rC))
        d_gas = d_gas.at[gi["hydrogen"]].add(-jnp.sum(rC))
        heat = heat - CRACK_HEAT * jnp.sum(rC)

        return Rates(gas=d_gas, cut=d_cut, attr=d_attr, heat=heat)


def crack_targets(carbon_per_molecule) -> tuple[int, ...]:
    """For each cut, the lighter cut nearest in carbon number to its cracked fragment.

    ``carbon_per_molecule`` is concrete (``(n_cut,)``, lightest first). A
    fragment of cut ``i`` has ``c_i - n_C(gas)`` carbons; the target is the
    cut ``j <= i`` whose carbon number is nearest to that.
    """
    c = np.asarray(carbon_per_molecule, dtype=float)
    nC = sum(CRACK_GAS_SPLIT[g] * GAS_ELEMENTS[g]["C"] for g in CRACK_GAS_SPLIT)
    out = []
    for i in range(c.size):
        frag = c[i] - nC
        j = int(np.argmin(np.abs(c[: i + 1] - frag)))
        out.append(j)
    return tuple(out)


__all__ = ["AROMATIC_CLASSES", "HDT_ATTRIBUTES", "HDT_ATTRIBUTE_ELEMENTS", "MODEL_COMPOUNDS",
           "HDS_ROUTE_DDS", "HDS_H2", "HDS_HEAT", "HDN_H2", "HDN_HEAT", "HDA_H2", "AROMATIC_THERMO",
           "OLEFIN_HEAT", "CRACK_HEAT", "CRACK_GAS_SPLIT", "HDTKineticParams", "NAPHTHA_HDT_PARAMS", "HDTKinetics",
           "crack_targets"]
