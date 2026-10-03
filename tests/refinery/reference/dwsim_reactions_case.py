"""Inputs shared by ``dwsim_reactions_generate.py`` and
``tests/refinery/test_dwsim_reactions.py``: the reaction-thermochemistry
cases run on DWSIM 9.0.5, and difflow's constants for them.

DWSIM has no hydrotreater, FCC, reformer or alkylation kinetic model; what
is compared is the thermochemistry those units rest on -- formation
enthalpies and Gibbs energies, ideal-gas heat capacities, the equilibria they
imply, and energy balances -- with DWSIM's Gibbs, equilibrium and conversion
reactors. Each case is run twice in DWSIM where it can be:

* ``"dwsim"`` -- DWSIM's own database compounds (ChemSep, ChEDL Thermo, its
  "User" table): its formation data, its Cp. That compares the *data* (and
  DWSIM's model, (b) of the brief).
* ``"hypo"`` -- every species a hypothetical compound carrying difflow's
  own ``H_f``, ``S`` (as ``G_f``), Cp and critical constants
  (:class:`.dwsim_reactors.ThermoHypo`). That compares the *implementation*
  ((a)): same numbers, DWSIM's reactor.

Every constant here is read from difflow at call time; the generator freezes
it in the JSON and the tests check it has not moved (staleness), so a change
to difflow's thermochemistry is a regeneration, not a disagreement.
"""

from __future__ import annotations

#: Psat multiplier exponent for the "ideal-gas" runs: DWSIM's Raoult's-law
#: package with every vapour pressure multiplied by ``exp(PSAT_SHIFT)`` never
#: forms a liquid, and its vapour is an ideal gas -- the model of difflow's
#: isomerization and reformer thermochemistry. (Without it Raoult condenses
#: the C5/C6 charge at 30 bar and the adiabatic temperature is a different
#: problem.)
PSAT_SHIFT = 40.0

# ---------------------------------------------------------------------------
# DWSIM names
# ---------------------------------------------------------------------------

#: Isomerization species (difflow.database names) -> DWSIM.
ISOM_DW = {
    "hydrogen": "Hydrogen", "ethane": "Ethane", "propane": "Propane", "isobutane": "Isobutane",
    "n_butane": "N-butane", "isopentane": "Isopentane", "n_pentane": "N-pentane",
    "2_2_dimethylbutane": "2,2-dimethylbutane", "2_3_dimethylbutane": "2,3-dimethylbutane",
    "2_methylpentane": "2-methylpentane", "3_methylpentane": "3-methylpentane",
    "n_hexane": "N-hexane", "methylcyclopentane": "Methylcyclopentane",
    "cyclohexane": "Cyclohexane", "benzene": "Benzene", "n_heptane": "N-heptane",
}

#: Reformer species keys -> DWSIM.
REFORMER_DW = {
    "H2": "Hydrogen", "C1": "Methane", "C2": "Ethane", "C3": "Propane", "iC4": "Isobutane",
    "nC4": "N-butane", "iC5": "Isopentane", "nC5": "N-pentane",
    "nP6": "N-hexane", "iP6": "2-methylpentane", "N5_6": "Methylcyclopentane",
    "N6": "Cyclohexane", "A6": "Benzene",
    "nP7": "N-heptane", "iP7": "2-methylhexane", "N7": "Methylcyclohexane", "A7": "Toluene",
    "nP8": "N-octane", "iP8": "2-methylheptane", "N8": "Ethylcyclohexane", "A8": "Ethylbenzene",
    "nP9": "N-nonane", "iP9": "2-methyloctane", "N9": "N-propylcyclohexane", "A9": "N-propylbenzene",
    "nP10": "N-decane", "iP10": "2-methylnonane", "N10": "N-butylcyclohexane",
    "A10": "N-butylbenzene",
}

#: Hydroprocessing model compounds (``hydrotreating.kinetics.MODEL_COMPOUNDS``
#: keys) -> DWSIM, ``None`` where DWSIM 9.0.5 has no such compound
#: (searched by CAS and by name).
HDT_DW = {
    "hydrogen": "Hydrogen", "hydrogen_sulfide": "Hydrogen sulfide", "ammonia": "Ammonia",
    "benzene": "Benzene", "cyclohexane": "Cyclohexane", "naphthalene": "Naphthalene",
    "tetralin": "1,2,3,4-Tetrahydronaphthalene", "phenanthrene": "Phenanthrene",
    "tetrahydrophenanthrene": None, "diethyl_sulfide": "Diethyl sulfide", "ethane": "Ethane",
    "thiophene": "Thiophene", "n_butane": "N-butane", "benzothiophene": "Benzothiophene",
    "ethylbenzene": "Ethylbenzene", "dibenzothiophene": None, "biphenyl": "Biphenyl",
    "cyclohexylbenzene": None, "quinoline": None, "propylbenzene": "N-propylbenzene",
    "carbazole": None, "1_hexene": "1-hexene", "n_hexane": "N-hexane",
}

#: Element formulas of the hydroprocessing model compounds.
HDT_FORMULA = {
    "hydrogen": {"H": 2}, "hydrogen_sulfide": {"H": 2, "S": 1}, "ammonia": {"N": 1, "H": 3},
    "benzene": {"C": 6, "H": 6}, "cyclohexane": {"C": 6, "H": 12},
    "naphthalene": {"C": 10, "H": 8}, "tetralin": {"C": 10, "H": 12},
    "phenanthrene": {"C": 14, "H": 10}, "tetrahydrophenanthrene": {"C": 14, "H": 14},
    "diethyl_sulfide": {"C": 4, "H": 10, "S": 1}, "ethane": {"C": 2, "H": 6},
    "thiophene": {"C": 4, "H": 4, "S": 1}, "n_butane": {"C": 4, "H": 10},
    "benzothiophene": {"C": 8, "H": 6, "S": 1}, "ethylbenzene": {"C": 8, "H": 10},
    "dibenzothiophene": {"C": 12, "H": 8, "S": 1}, "biphenyl": {"C": 12, "H": 10},
    "cyclohexylbenzene": {"C": 12, "H": 16}, "quinoline": {"C": 9, "H": 7, "N": 1},
    "propylbenzene": {"C": 9, "H": 12}, "carbazole": {"C": 12, "H": 9, "N": 1},
    "1_hexene": {"C": 6, "H": 12}, "n_hexane": {"C": 6, "H": 14},
}

#: Alkylation species -> DWSIM (difflow.dwsim_import.DWSIM_NAMES).
ALKY_DW = {
    "propane": "Propane", "isobutane": "Isobutane", "n_butane": "N-butane",
    "isopentane": "Isopentane", "n_pentane": "N-pentane", "propylene": "Propylene",
    "1_butene": "1-butene", "cis_2_butene": "Cis-2-butene", "trans_2_butene": "Trans-2-butene",
    "isobutylene": "Isobutene", "1_pentene": "1-pentene", "2_methyl_2_butene": "2-methyl-2-butene",
    "2_3_dimethylpentane": "2,3-dimethylpentane", "2_4_dimethylpentane": "2,4-dimethylpentane",
    "2_2_4_trimethylpentane": "2,2,4-trimethylpentane",
    "2_3_4_trimethylpentane": "2,3,4-trimethylpentane",
    "2_5_dimethylhexane": "2,5-dimethylhexane", "2_2_5_trimethylhexane": "2,2,5-trimethylhexane",
    "n_dodecane": "N-dodecane",
}

#: FCC regenerator gases -> DWSIM. Coke carbon is a hypothetical compound
#: ``COKE_C`` with ``H_f = 0`` (graphite, difflow's coke convention):
#: DWSIM's own "Graphite" (its "User" table) has a constant-vapour-pressure
#: equation (number 1) that the ideal-gas setting cannot lift, so it enters
#: as a liquid and its enthalpy carries a heat of vaporisation. Entering at
#: 298.15 K and burnt completely, the carbon's other properties play no part.
#: Coke hydrogen is hydrogen.
FCC_DW = {"carbon": "COKE_C", "hydrogen": "Hydrogen", "oxygen": "Oxygen", "nitrogen": "Nitrogen",
          "carbon_dioxide": "Carbon dioxide", "carbon_monoxide": "Carbon monoxide", "water": "Water"}

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

#: Temperatures at which the formation functions are tabulated (K).
T_TABLE = (298.15, 400.0, 413.15, 420.0, 450.0, 480.0, 500.0, 550.0, 573.15, 623.15, 693.15,
           700.0, 773.15, 973.15, 1003.15)

#: Reformer reactions (reformer keys), compared per reaction.
REFORMER_REACTIONS = {
    "N6 = A6 + 3 H2": {"N6": -1, "A6": 1, "H2": 3},
    "N7 = A7 + 3 H2": {"N7": -1, "A7": 1, "H2": 3},
    "N8 = A8 + 3 H2": {"N8": -1, "A8": 1, "H2": 3},
    "N9 = A9 + 3 H2": {"N9": -1, "A9": 1, "H2": 3},
    "N10 = A10 + 3 H2": {"N10": -1, "A10": 1, "H2": 3},
    "N5_6 = N6": {"N5_6": -1, "N6": 1},
    "nP7 = N7 + H2": {"nP7": -1, "N7": 1, "H2": 1},
    "nP7 = A7 + 4 H2": {"nP7": -1, "A7": 1, "H2": 4},
    "nP8 = A8 + 4 H2": {"nP8": -1, "A8": 1, "H2": 4},
    "nP6 = iP6": {"nP6": -1, "iP6": 1},
    "nP7 = iP7": {"nP7": -1, "iP7": 1},
    "nP8 = iP8": {"nP8": -1, "iP8": 1},
    "nP7 + H2 = C3 + iC4": {"nP7": -1, "H2": -1, "C3": 1, "iC4": 1},
    "A7 + H2 = A6 + C1": {"A7": -1, "H2": -1, "A6": 1, "C1": 1},
}

#: Trace amount (mol/s) given to every product species in an equilibrium
#: feed: DWSIM's equilibrium reactor fails on an absent species ("negative
#: mole fractions"), and its Gibbs reactor starts from the feed.
TRACE = 1.0e-4

#: The isomerization reactor's reversible network (``isomerization.REACTIONS``
#: without the irreversible cracking): ``{name: nu}``.
ISOM_REACTIONS = {
    "nC5 = iC5": {"n_pentane": -1, "isopentane": 1},
    "nC6 = 2MP": {"n_hexane": -1, "2_methylpentane": 1},
    "2MP = 3MP": {"2_methylpentane": -1, "3_methylpentane": 1},
    "2MP = 23DMB": {"2_methylpentane": -1, "2_3_dimethylbutane": 1},
    "23DMB = 22DMB": {"2_3_dimethylbutane": -1, "2_2_dimethylbutane": 1},
    "MCP = CH": {"methylcyclopentane": -1, "cyclohexane": 1},
    "Bz + 3 H2 = CH": {"benzene": -1, "hydrogen": -3, "cyclohexane": 1},
    "MCP + H2 = 2MP": {"methylcyclopentane": -1, "hydrogen": -1, "2_methylpentane": 1},
}

#: Reformer equilibria (ideal gas, isothermal): feed (mol/s, reformer keys)
#: at each (T, P), and the independent reactions DWSIM's equilibrium
#: reactor is given.
REFORMER_EQUILIBRIA = {
    "mch_toluene": {"feed": {"H2": 5.0, "N7": 1.0, "A7": 0.0}, "T": (700.0, 773.15),
                    "P": (10.0e5, 25.0e5),
                    "reactions": {"N7 = A7 + 3 H2": {"N7": -1, "A7": 1, "H2": 3}}},
    "c6_rings": {"feed": {"H2": 5.0, "N5_6": 0.5, "N6": 0.5, "A6": 0.0}, "T": (700.0, 773.15),
                 "P": (10.0e5, 25.0e5),
                 "reactions": {"N5_6 = N6": {"N5_6": -1, "N6": 1},
                               "N6 = A6 + 3 H2": {"N6": -1, "A6": 1, "H2": 3}}},
    "c7_dehydrocyclization": {"feed": {"H2": 5.0, "nP7": 1.0, "iP7": 0.0, "N7": 0.0, "A7": 0.0},
                              "T": (773.15,), "P": (10.0e5, 25.0e5),
                              "reactions": {"nP7 = iP7": {"nP7": -1, "iP7": 1},
                                            "nP7 = N7 + H2": {"nP7": -1, "N7": 1, "H2": 1},
                                            "N7 = A7 + 3 H2": {"N7": -1, "A7": 1, "H2": 3}}},
}

#: One adiabatic reformer bed for the energy-balance check.
REFORMER_BED = {"feed": "rich_naphtha", "kg_s": 10.0, "H2_HC": 5.0, "T_in": 773.15, "P": 15.0e5,
                "LHSV": 1.5, "catalyst_density": 700.0, "catalyst_split": 0.15}

#: Hydroprocessing model reactions (MODEL_COMPOUNDS keys): stoichiometry and
#: which difflow constant they stand behind.
HDT_REACTIONS = {
    "HDS sulfide: Et2S + 2 H2 = 2 C2H6 + H2S": (
        {"diethyl_sulfide": -1, "hydrogen": -2, "ethane": 2, "hydrogen_sulfide": 1}, "HDS_HEAT[0]"),
    "HDS thiophene: C4H4S + 4 H2 = nC4 + H2S": (
        {"thiophene": -1, "hydrogen": -4, "n_butane": 1, "hydrogen_sulfide": 1}, "HDS_HEAT[1]"),
    "HDS benzothiophene: C8H6S + 3 H2 = EB + H2S": (
        {"benzothiophene": -1, "hydrogen": -3, "ethylbenzene": 1, "hydrogen_sulfide": 1},
        "HDS_HEAT[2], RESIDUE_S_HEAT"),
    "HDS DBT (DDS): DBT + 2 H2 = biphenyl + H2S": (
        {"dibenzothiophene": -1, "hydrogen": -2, "biphenyl": 1, "hydrogen_sulfide": 1},
        "HDS_HEAT[3], HDS_HEAT[4]"),
    "HDS DBT (HYD): DBT + 5 H2 = CHB + H2S": (
        {"dibenzothiophene": -1, "hydrogen": -5, "cyclohexylbenzene": 1, "hydrogen_sulfide": 1},
        "HDS_HEAT[3], HDS_HEAT[4]"),
    "HDN basic: quinoline + 4 H2 = PB + NH3": (
        {"quinoline": -1, "hydrogen": -4, "propylbenzene": 1, "ammonia": 1}, "HDN_HEAT[0]"),
    "HDN non-basic: carbazole + 5 H2 = CHB + NH3": (
        {"carbazole": -1, "hydrogen": -5, "cyclohexylbenzene": 1, "ammonia": 1}, "HDN_HEAT[1]"),
    "HDA mono: Bz + 3 H2 = CH": (
        {"benzene": -1, "hydrogen": -3, "cyclohexane": 1}, "AROMATIC_THERMO[2], CCR_HEAT_PER_H2"),
    "HDA di: naphthalene + 2 H2 = tetralin": (
        {"naphthalene": -1, "hydrogen": -2, "tetralin": 1}, "AROMATIC_THERMO[1]"),
    "HDA poly: phenanthrene + 2 H2 = THP": (
        {"phenanthrene": -1, "hydrogen": -2, "tetrahydrophenanthrene": 1}, "AROMATIC_THERMO[0]"),
    "olefin: 1-hexene + H2 = nC6": (
        {"1_hexene": -1, "hydrogen": -1, "n_hexane": 1}, "OLEFIN_HEAT"),
    "cracking: nC6 + H2 = nC4 + C2": (
        {"n_hexane": -1, "hydrogen": -1, "n_butane": 1, "ethane": 1}, "CRACK_HEAT, CONVERSION_HEAT"),
}

#: Temperature at which DWSIM's conversion reactor measures each
#: hydroprocessing heat (isothermal, ideal gas) -- a hydrotreater's.
HDT_T = 623.15

#: Aromatics saturation equilibria (Gibbs reactor, ideal gas, isothermal).
AROMATIC_EQUILIBRIA = {
    "benzene": {"feed": {"hydrogen": 20.0, "benzene": 1.0, "cyclohexane": 0.0},
                "T": (573.15, 623.15, 693.15), "P": (30.0e5, 100.0e5),
                "reactions": {"Bz + 3 H2 = CH": {"benzene": -1, "hydrogen": -3, "cyclohexane": 1}},
                "step": 2},
    "naphthalene": {"feed": {"hydrogen": 20.0, "naphthalene": 1.0, "tetralin": 0.0},
                    "T": (573.15, 623.15, 693.15), "P": (30.0e5, 100.0e5),
                    "reactions": {"Np + 2 H2 = tetralin": {"naphthalene": -1, "hydrogen": -2,
                                                           "tetralin": 1}},
                    "step": 1},
}

#: Alkylation: liquid heats of reaction by DWSIM's conversion reactor
#: (isothermal, Peng-Robinson, liquid) at these conditions.
ALKY_T = (283.15, 298.15)
ALKY_P = 10.0e5
#: Single-product alkylation reactions (alkylation species).
ALKY_REACTIONS = {
    "iC4 + 1-butene = 2,2,4-TMP": {"isobutane": -1, "1_butene": -1, "2_2_4_trimethylpentane": 1},
    "iC4 + cis-2-butene = 2,2,4-TMP": {"isobutane": -1, "cis_2_butene": -1,
                                       "2_2_4_trimethylpentane": 1},
    "iC4 + trans-2-butene = 2,2,4-TMP": {"isobutane": -1, "trans_2_butene": -1,
                                         "2_2_4_trimethylpentane": 1},
    "iC4 + isobutylene = 2,2,4-TMP": {"isobutane": -1, "isobutylene": -1,
                                      "2_2_4_trimethylpentane": 1},
    "iC4 + 2-butenes = 2,3,4-TMP": {"isobutane": -1, "trans_2_butene": -1,
                                    "2_3_4_trimethylpentane": 1},
    "iC4 + propylene = 2,4-DMP": {"isobutane": -1, "propylene": -1, "2_4_dimethylpentane": 1},
    "iC4 + 2 butene = C12 (heavy end)": {"isobutane": -1, "1_butene": -2, "n_dodecane": 1},
}

#: FCC regenerator: coke burned in air.
FCC = {"coke_kg_s": 1.0, "coke_hydrogen": 0.07, "flue_o2": 0.02, "co_co2": (0.0, 0.5),
       "T_rg": (973.15, 1003.15), "P": 2.5e5}


# ---------------------------------------------------------------------------
# difflow's constants (inputs, frozen in the JSON)
# ---------------------------------------------------------------------------

def isom_constants() -> dict:
    """The isomerization thermochemistry: ``Hf``, ``S``, ``cp`` (cubic),
    ``C``, ``H``, ``MW`` and the database's ``Tc``, ``Pc``, ``omega``."""
    from difflow.database import get_critical_props
    from difflow_refinery.isomerization import thermochem as tc

    out = {}
    for s in tc.SPECIES:
        cr = get_critical_props(s.name)
        out[s.name] = {"Hf": s.Hf, "S": s.S, "cp": list(s.cp), "C": s.C, "H": s.H, "MW": s.MW,
                       "Tc": float(cr.Tc), "Pc": float(cr.Pc), "omega": float(cr.omega)}
    return out


def reformer_constants() -> dict:
    from difflow_refinery.reforming import species as sp

    return {k: {"Hf": s.Hf, "S": s.S0, "cp": list(s.cp), "C": s.carbon, "H": s.hydrogen,
                "MW": s.MW, "Tc": s.Tc, "Pc": s.Pc, "omega": s.omega}
            for k, s in sp.SPECIES.items()}


def hdt_constants() -> dict:
    """``MODEL_COMPOUNDS`` and the per-class constants built from them."""
    from difflow_refinery.hydrotreating import kinetics as k
    from difflow_refinery.residue import kinetics as rk

    return {"model_compounds": {n: {"Hf": v[0], "S": v[1]} for n, v in k.MODEL_COMPOUNDS.items()},
            "HDS_HEAT": list(k.HDS_HEAT), "HDS_H2": list(k.HDS_H2),
            "HDN_HEAT": list(k.HDN_HEAT), "HDN_H2": list(k.HDN_H2),
            "AROMATIC_THERMO": [list(t) for t in k.AROMATIC_THERMO], "HDA_H2": list(k.HDA_H2),
            "OLEFIN_HEAT": k.OLEFIN_HEAT, "CRACK_HEAT": k.CRACK_HEAT,
            "RESIDUE_S_HEAT": rk.RESIDUE_S_HEAT, "CCR_HEAT_PER_H2": rk.CCR_HEAT_PER_H2,
            "CONVERSION_HEAT": rk.CONVERSION_HEAT}


def alky_constants() -> dict:
    """Per species ``Hf_gas``, ``Hvap298``, ``Hf_liquid`` and formula; the
    route-A stoichiometry (default selectivity) and liquid heats per olefin."""
    from difflow_refinery.alkylation import reactor as ar
    from difflow_refinery.alkylation.species import ALKYLATION_SPECIES, OLEFINS, species

    sp = {n: {"Hf_gas": species(n).Hf_gas, "Hvap298": species(n).Hvap298,
              "Hf_liquid": species(n).Hf_liquid, "C": species(n).n_C, "H": species(n).n_H,
              "MW": species(n).MW, "Tc": species(n).Tc, "Pc": species(n).Pc,
              "omega": species(n).omega}
          for n in ALKYLATION_SPECIES}
    nu_A, nu_H, _, _ = ar._route_tables(ar.DEFAULT_SELECTIVITY)
    routes = {o: {"A": {n: float(v) for n, v in zip(ALKYLATION_SPECIES, nu_A[j]) if v != 0.0},
                  "H": {n: float(v) for n, v in zip(ALKYLATION_SPECIES, nu_H[j]) if v != 0.0}}
              for j, o in enumerate(OLEFINS)}
    rxr = ar.AlkylationReactor(ar.AlkylationReactorParams())
    heats = {o: list(v) for o, v in rxr.heats_of_reaction().items()}
    return {"species": sp, "routes": routes, "route_heats": heats}


def fcc_constants() -> dict:
    """The regenerator's gas data and the flue gas of :data:`FCC`'s coke, by
    difflow's own combustion arithmetic, per CO/CO2 ratio."""
    from difflow_refinery.fcc import regenerator as rg
    from difflow_refinery.fcc import species as sp

    m = FCC["coke_kg_s"]
    el = {"C": m * (1 - FCC["coke_hydrogen"]), "H": m * FCC["coke_hydrogen"], "S": 0.0, "N": 0.0}
    out = {"HF_298": dict(sp.HF_298), "CP_IG": {k: list(v) for k, v in sp.CP_IG.items()},
           "ATOMIC_WEIGHT": dict(sp.ATOMIC_WEIGHT), "AIR_O2": sp.AIR_O2, "elements_kg_s": el,
           "burn": {}}
    for r in FCC["co_co2"]:
        p = {"co_co2": r, "flue_o2": FCC["flue_o2"]}
        c = rg.combustion(p, el, 973.15, "spec", "flue_o2")
        out["burn"][repr(r)] = {"air": {k: float(v) for k, v in c["air"].items()},
                                "flue": {k: float(v) for k, v in c["flue"].items()}}
    return out


def reformer_bed() -> dict:
    """difflow's first reformer bed on :data:`REFORMER_BED`: inlet and outlet
    flows (mol/s, reformer order), temperatures, and the catalyst mass. This
    is an INPUT to DWSIM's conversion reactor (its outlet composition)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    from difflow_refinery.reforming import feed as fd
    from difflow_refinery.reforming import species as sp
    from difflow_refinery.reforming.kinetics import ReformingKinetics
    from difflow_refinery.reforming.reactor import integrate_bed

    c = REFORMER_BED
    feed = getattr(fd, c["feed"])(c["kg_s"])
    F = np.asarray(feed.flows, dtype=float).copy()
    hc = F.sum() - F[sp.INDEX["H2"]]
    F[sp.INDEX["H2"]] += c["H2_HC"] * hc
    V_cat = float(feed.volume_flow) * 3600.0 / c["LHSV"]
    W = c["catalyst_density"] * V_cat * c["catalyst_split"]
    F_out, T_out, coke, _ = integrate_bed(jnp.asarray(F), jnp.asarray(c["T_in"]), jnp.asarray(c["P"]),
                                          jnp.asarray(W), ReformingKinetics())
    return {"names": list(sp.NAMES), "F_in": [float(v) for v in F],
            "F_out": [float(v) for v in F_out], "T_in": c["T_in"], "T_out": float(T_out),
            "P": c["P"], "W": W}
