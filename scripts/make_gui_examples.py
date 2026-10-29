"""Write the plugin examples (src/difflow/workspace/examples/*.json).

Each example is built here with the plugin's own unit classes and saved
through :func:`difflow.serialize.to_dict`, so the file is exactly what
**File > Flowsheet JSON** would write. Re-run after a change to a unit's
parameters or to the file format:

    python scripts/make_gui_examples.py

``tests/test_gui_examples.py`` opens every example on each commit; it does
not solve them. Check a regenerated example's numbers by hand against the
reference its docstring names.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from difflow import Flowsheet  # noqa: E402
from difflow.flowsheet import Unit  # noqa: E402
from difflow.serialize import to_dict  # noqa: E402
from difflow.units import Mixer  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "difflow" / "workspace" / "examples"


def write(key: str, fs: Flowsheet, title: str, description: str,
          nodes: dict, solver: dict | None = None,
          code_context: str | None = None) -> Path:
    fs.view = {"title": title, "description": description, "nodes": nodes}
    if solver:
        fs.view["solver"] = solver
    if code_context:
        fs.view["code_context"] = code_context
    path = OUT / f"{key}.json"
    path.write_text(json.dumps(to_dict(fs), indent=1) + "\n")
    print(f"wrote {path.relative_to(OUT.parents[3])}")
    return path


# ---------------------------------------------------------------------
# Gas: GasLib-11
# ---------------------------------------------------------------------

def gaslib11() -> Path:
    """GasLib-11 with its standard nomination (Schmidt et al. 2017).

    3 entries, 3 exits, 5 inner nodes, 8 pipes, 1 valve, 2 compressor
    stations. Every pipe is 55 km of 500 mm line at 0.1 mm roughness;
    the gas is 18.5674 kg/kmol at 10 C with a norm density of 0.785
    kg/m^3. The nomination fixes every boundary flow, so the only
    implicit quantity is how flow from N01 divides between pipe02 and
    the valve V01 around the loop N01-N02-N04-N03; the valve flow is the
    one tear. entry01 is the slack node at 50 bar, and the compressor
    ratios (1.10, 1.05) are unit parameters.
    """
    from difflow_gas import (BackPipe, Compressor, FlowMinus, FlowSplit,
                             GasPipe, Junction, OpenValve,
                             PressureDrivenPipe, SourceHead, TearSplit,
                             gas_stream)
    from difflow_gas.physics import knm3h_to_kg_s, weymouth_beta

    T = 283.15                  # 10 C
    rho0 = 0.785                # norm density, kg/m^3
    P_slack = 50.0e5
    beta = weymouth_beta(length_m=55e3, diameter_m=0.5, roughness_m=1e-4,
                         gas_temp_k=T, molar_mass_kg_kmol=18.5674)
    kg_s = {n: knm3h_to_kg_s(q, rho0) for n, q in {
        "entry01": 160.0, "entry02": 140.0, "entry03": 0.0,
        "exit01": 100.0, "exit02": 120.0, "exit03": 80.0}.items()}

    fs = Flowsheet(species_order=["gas"], default_flow=5.0, default_T=T,
                   default_P=P_slack)
    for entry in ("entry01", "entry02", "entry03"):
        fs.add_feed(f"{entry}_feed", gas_stream(kg_s[entry], T, P_slack))

    def U(name, op, inlets, outlets):
        fs.add_unit(Unit(name, op, inlets, outlets))

    U("src_entry01", SourceHead(P_slack), ["entry01_feed"], ["entry01"])
    U("pipe01", GasPipe(beta), ["entry01"], ["entry03_arrival"])
    U("node_entry03", Junction(), ["entry03_arrival", "entry03_feed"],
      ["entry03"])
    U("cs01", Compressor(1.10), ["entry03"], ["N01"])
    U("split_N01", TearSplit(), ["N01", "v01_tear"], ["v01_in", "pipe02_in"])
    U("v01", OpenValve(), ["v01_in"], ["v01_out"])
    U("pipe02", GasPipe(beta), ["pipe02_in"], ["N02"])
    U("split_N02", FlowSplit(kg_s["exit01"]), ["N02"],
      ["pipe04_in", "pipe05_in"])
    U("pipe04", GasPipe(beta), ["pipe04_in"], ["exit01"])
    U("pipe05", GasPipe(beta), ["pipe05_in"], ["N04_a"])
    U("node_N03", Junction(), ["v01_out", "entry02_feed"], ["N03"])
    U("pipe03_back", BackPipe(beta), ["N03", "entry02_feed"], ["entry02"])
    U("pipe06", PressureDrivenPipe(beta), ["N03", "N04_a"], ["pipe06_out"])
    U("v01_update", FlowMinus(), ["pipe06_out", "entry02_feed"],
      ["v01_calc"])
    U("node_N04", Junction(), ["N04_a", "pipe06_out"], ["N04"])
    U("cs02", Compressor(1.05), ["N04"], ["N05"])
    U("split_N05", FlowSplit(kg_s["exit02"]), ["N05"],
      ["pipe07_in", "pipe08_in"])
    U("pipe07", GasPipe(beta), ["pipe07_in"], ["exit02"])
    U("pipe08", GasPipe(beta), ["pipe08_in"], ["exit03"])
    fs.add_recycle("v01_calc", "v01_tear")

    # GasLib node coordinates (x right, y up), and each unit at the
    # midpoint of the arc it models, so the canvas reads like the map.
    xy = {"entry01": (0, 0), "entry03": (200, 0), "N01": (400, 0),
          "N02": (600, 100), "N03": (600, -100), "N04": (800, 0),
          "N05": (1000, 0), "exit01": (600, 300), "exit02": (1141, 141),
          "exit03": (1141, -141), "entry02": (600, -300)}

    def mid(a, b, dx=0, dy=0):
        return ((xy[a][0] + xy[b][0]) / 2 + dx, (xy[a][1] + xy[b][1]) / 2 + dy)

    at = {
        "feed:entry01_feed": (xy["entry01"][0] - 120, 0),
        "feed:entry03_feed": (200, 110),
        "feed:entry02_feed": (xy["entry02"][0] - 160, -300),
        "src_entry01": xy["entry01"],
        "pipe01": mid("entry01", "entry03"),
        "node_entry03": xy["entry03"],
        "cs01": mid("entry03", "N01"),
        "split_N01": xy["N01"],
        "v01": mid("N01", "N03"),
        "pipe02": mid("N01", "N02"),
        "split_N02": xy["N02"],
        "pipe04": mid("N02", "exit01"),
        "pipe05": mid("N02", "N04"),
        "node_N03": xy["N03"],
        "pipe03_back": mid("entry02", "N03"),
        "pipe06": mid("N03", "N04"),
        "v01_update": mid("N03", "N04", dy=-110),
        "node_N04": xy["N04"],
        "cs02": mid("N04", "N05"),
        "split_N05": xy["N05"],
        "pipe07": mid("N05", "exit02"),
        "pipe08": mid("N05", "exit03"),
    }
    scale = 1.25
    nodes = {k: {"x": 60 + scale * (x + 120), "y": 60 + scale * (320 - y)}
             for k, (x, y) in at.items()}

    return write(
        "04_gaslib11",
        fs,
        title="GasLib-11 gas network",
        description=(
            "The GasLib-11 benchmark with its standard nomination: 8 pipes, "
            "a valve and 2 compressor stations around one loop, entry01 "
            "held at 50 bar."
        ),
        nodes=nodes,
        # Signed flows: the valve flow is a tear and may reverse while
        # iterating, so it must not be clipped at zero.
        solver={"clip_negative_flows": False, "tol": 1e-10, "max_iter": 200},
    )


# ---------------------------------------------------------------------
# Bio: mAb downstream processing
# ---------------------------------------------------------------------

def bio() -> Path:
    """Platform mAb downstream train for one 2000 L fed-batch harvest.

    Disc-stack centrifuge -> Protein A capture -> CEX bind-elute polish
    -> AEX flow-through polish -> UF concentration. Stream entries are
    grams per batch (the plugin's convention, see
    ``difflow_bio.flowsheets.mab_dsp.mAbDSPTrain``); there is no solvent
    species, because the columns would carry it into the eluate.

    Sources: harvest, centrifuge, Protein A and CEX sizing follow
    ``examples/bio/04_mab_downstream.ipynb`` (2000 L at 5 g/L, 20 g/L wet
    cells, 0.1 g/L DNA; 100-disc stack at 6000 rpm, 1000 L/h, 15 um
    cells; 300 L Protein A at 40 g/L; 200 L CEX at 60 g/L). HCP (5 %)
    and aggregate (3 %) of the mAb are the basis of
    ``TYPICAL_CEX_CLEARANCE``/``TYPICAL_AEX_CLEARANCE``, used here. The
    AEX and UF settings follow ``mAbDSPTrain`` (q_max 50 g/L, yield
    0.95; UF/DF rejections, 10x concentration).
    """
    from difflow import Flowsheet, make_stream
    from difflow.flowsheet import Unit
    from difflow_bio import (DiscStackCentrifuge, DiscStackParams,
                             IEXParams, IonExchangeChromatography,
                             ProteinAChromatography, ProteinAParams,
                             Ultrafiltration, UltrafiltrationParams)
    from difflow_bio.units.chromatography import (TYPICAL_AEX_CLEARANCE,
                                                  TYPICAL_CEX_CLEARANCE)

    species = ["cells", "mAb", "HCP", "DNA", "aggregates"]
    V_harvest = 2000.0               # L per batch
    titer = 5.0                      # g/L mAb
    harvest = make_stream({
        "cells": 20.0 * V_harvest,   # wet cell weight, g
        "mAb": titer * V_harvest,
        "HCP": 0.05 * titer * V_harvest,
        "DNA": 0.1 * V_harvest,
        "aggregates": 0.03 * titer * V_harvest,
    }, T=298.15, P=101325.0)

    concentrate_fraction = 0.05      # liquid to the solids discharge
    V_clarified = V_harvest * (1.0 - concentrate_fraction)

    fs = Flowsheet(species_order=species, default_flow=1.0,
                   default_T=298.15, default_P=101325.0)
    fs.add_feed("harvest", harvest)

    fs.add_unit(Unit(
        "centrifuge",
        DiscStackCentrifuge(DiscStackParams(
            n_discs=100, r_outer=0.15, r_inner=0.05, half_angle=0.698,
            rpm=6000.0, efficiency=0.8, species_order=species,
            cell_species="cells")),
        ["harvest"], ["cell_sludge", "clarified"],
        {"Q": 2.78e-4,               # m^3/s: 2000 L in about 2 h
         "d_particle": 15e-6,        # m, CHO cell
         "rho_particle": 1050.0, "rho_fluid": 1000.0, "viscosity": 1e-3,
         "concentrate_fraction": concentrate_fraction},
    ))
    fs.add_unit(Unit(
        "protein_a",
        ProteinAChromatography(ProteinAParams(
            column_volume=300.0, q_max=40.0, K_d=0.05, target_species="mAb",
            yield_factor=0.95,
            impurity_clearance={"HCP": 2.0, "DNA": 3.0, "cells": 4.0},
            species_order=species)),
        ["clarified"], ["proa_eluate", "proa_flowthrough"],
        {"feed_volume": V_clarified, "breakthrough_limit": 0.01},
    ))
    fs.add_unit(Unit(
        "cex",
        IonExchangeChromatography(IEXParams(
            column_volume=200.0, mode="bind_elute", q_max=60.0, K_d=0.3,
            target_species="mAb", yield_factor=0.92,
            selectivity={"mAb": 1.0, "cells": 0.0},
            impurity_clearance=dict(TYPICAL_CEX_CLEARANCE),
            species_order=species)),
        ["proa_eluate"], ["cex_eluate", "cex_waste"],
    ))
    fs.add_unit(Unit(
        "aex",
        IonExchangeChromatography(IEXParams(
            column_volume=50.0, mode="flow_through", q_max=50.0,
            target_species="mAb", yield_factor=0.95,
            selectivity={"mAb": 0.0, "cells": 1.0},
            impurity_clearance=dict(TYPICAL_AEX_CLEARANCE),
            species_order=species)),
        ["cex_eluate"], ["aex_flowthrough", "aex_bound"],
    ))
    fs.add_unit(Unit(
        "uf",
        Ultrafiltration(UltrafiltrationParams(
            membrane_area=10.0, MWCO=30.0,
            rejection={"mAb": 0.995, "HCP": 0.995, "DNA": 0.995,
                       "aggregates": 0.999, "cells": 1.0},
            species_order=species)),
        ["aex_flowthrough"], ["drug_substance", "uf_permeate"],
        # The CEX pool is 5 CV of the 200 L column (Petrides 2015).
        {"concentration_factor": 10.0, "TMP": 1.0, "feed_volume": 1000.0},
    ))

    order = ["feed:harvest", "centrifuge", "protein_a", "cex", "aex", "uf"]
    nodes = {k: {"x": 80 + 220 * i, "y": 220} for i, k in enumerate(order)}

    return write(
        "05_bio_downstream",
        fs,
        title="mAb downstream purification",
        description=(
            "A 2000 L monoclonal antibody harvest (grams per batch) "
            "clarified in a disc-stack centrifuge, captured on Protein A, "
            "polished by cation- and anion-exchange chromatography and "
            "concentrated 10x by ultrafiltration."
        ),
        nodes=nodes,
    )


# ---------------------------------------------------------------------
# Carbon capture: 30 wt% MEA absorber and stripper on coal flue gas
# ---------------------------------------------------------------------

def carbon_capture() -> Path:
    """Post-combustion capture with 30 wt% MEA on a coal flue gas, closed loop.

    The flue gas is a pulverised-coal boiler's after FGD and the direct
    contact cooler: 13.5 % CO2, 4.5 % O2, 7.3 % H2O (saturated at 40 C)
    and N2, at 40 C and 1 atm; 1000 mol/s is about a 25 MWe slip stream
    (5.9 kg/s CO2). The absorber is the plugin's own base design
    (``src/difflow_cc/examples/05_integrated_capture_plant.ipynb``: MEA
    30 wt%, 15 stages, 40 C), with the circulation raised to L/G = 4.2
    so the rich-end pinch (rich loading ~0.48 from 13.5 % CO2) still
    allows ~92 % capture. The stripper runs its defaults (120 C reboiler,
    2 bar, 40 C condenser, 2 mol steam per mol CO2).

    The loop is closed: the stripper's lean solvent goes back to the
    absorber through ``SolventMakeup``, which restores the design amine
    and water circulation (the water the treated gas and the CO2 product
    carry away is made up), so the lean loading is solved for rather than
    given. It converges to the same answer as ``solve_amine_loop``.
    """
    from difflow_cc import (AbsorberParams, AmineAbsorber, AmineStripper,
                            SolventMakeup, SolventMakeupParams,
                            StripperParams)
    from difflow_cc.database import get_solvent

    F = 1000.0                                      # mol/s flue gas
    y = {"CO2": 0.135, "O2": 0.045, "H2O": 0.073}
    y["N2"] = 1.0 - sum(y.values())
    species = ["CO2", "N2", "O2", "H2O", "Amine", "CO2_absorbed"]

    absorber = AbsorberParams(
        solvent="MEA", n_stages=15, solvent_conc=30.0, L_G_ratio=4.2,
        T_gas_in=313.15, T_liquid_in=313.15, P_absorber=101325.0)
    # The design circulation, as AmineAbsorber builds it from L/G and wt%.
    w, MW_amine, MW_water = 0.30, get_solvent("MEA").MW, 18.0
    x_amine = (w / MW_amine) / (w / MW_amine + (1 - w) / MW_water)
    F_liquid = absorber.L_G_ratio * F

    fs = Flowsheet(species_order=species, default_flow=0.0,
                   default_T=313.15, default_P=101325.0)
    fs.add_feed("flue_gas", {**{f"F_{k}": F * v for k, v in y.items()},
                             "T": 313.15, "P": 101325.0})
    fs.add_unit(Unit("absorber", AmineAbsorber(absorber),
                     ["flue_gas", "lean_return"],
                     ["treated_gas", "rich_solvent"]))
    fs.add_unit(Unit(
        "stripper",
        AmineStripper(StripperParams(
            solvent="MEA", n_stages=10, T_reboiler=393.15,
            P_stripper=200000.0, steam_ratio=2.0, T_condenser=313.15,
            target_lean_loading=0.20)),
        ["rich_solvent"], ["lean_solvent", "co2_product"]))
    fs.add_unit(Unit(
        "makeup",
        SolventMakeup(SolventMakeupParams(
            amine_flow=F_liquid * x_amine,
            water_flow=F_liquid * (1 - x_amine),
            T=absorber.T_liquid_in, P=absorber.P_absorber)),
        ["lean_solvent"], ["lean_makeup"]))
    fs.add_recycle("lean_makeup", "lean_return")

    nodes = {
        "feed:flue_gas": {"x": 80, "y": 200},
        "absorber": {"x": 320, "y": 180},
        "stripper": {"x": 580, "y": 220},
        "makeup": {"x": 450, "y": 400},
    }

    return write(
        "06_carbon_capture",
        fs,
        title="Amine CO2 capture (MEA), closed loop",
        description=(
            "A 30 wt% MEA absorber and stripper with the lean solvent "
            "recycled through a makeup block, capturing 92 % of the CO2 "
            "in a 13.5 % CO2 coal flue gas."
        ),
        nodes=nodes,
        solver={"tol": 1e-10, "max_iter": 200},
    )


# ---------------------------------------------------------------------
# Power: a radial distribution feeder
# ---------------------------------------------------------------------

def power() -> Path:
    """The trunk of ``difflow_power.cases.radial_feeder`` as a ladder.

    ``radial_feeder()`` is a 7-bus, 12.47 kV feeder (10 MVA base) whose
    trunk s-n1-n2-n3-n4 carries two laterals. A ladder flowsheet cannot
    branch, so the laterals (buses l1, l2, branches b1, b2, loads dl1,
    dl2) are dropped, exactly as ``tests/power/test_flowsheet.py``'s
    ``ladder_network`` does, and the flowsheet is the plugin's own
    :func:`difflow_power.build_ladder_flowsheet`: SlackSource at the
    substation (1.02 pu), a SeriesBranch per trunk section, a
    constant-power LoadDraw at each bus, and LadderClose correcting the
    substation infeed (the one tear) by the leftover power at the open
    end. Power is signed, so the tear must not be clipped at zero.
    """
    import difflow_power as dp
    from difflow_power.network import PowerNetwork

    net = dp.cases.radial_feeder()
    lateral = {"buses": ("l1", "l2"), "branches": ("b1", "b2"),
               "loads": ("dl1", "dl2")}
    trunk = PowerNetwork(
        name="radial feeder trunk (12.47 kV)",
        base_mva=net.base_mva,
        buses={k: v for k, v in net.buses.items()
               if k not in lateral["buses"]},
        branches={k: v for k, v in net.branches.items()
                  if k not in lateral["branches"]},
        generators=net.generators,
        loads={k: v for k, v in net.loads.items()
               if k not in lateral["loads"]},
    )
    fs, order = dp.build_ladder_flowsheet(trunk)

    # Left to right along the feeder: each bus's load below the line,
    # the series branch between buses, the closure back under the source.
    dx = 190
    at = {"feed:infeed": (40, 140), "src_s": (190, 140), "load_s": (190, 300)}
    x = 190
    for bus, branch in zip(order[1:], ("t0", "t1", "t2", "t3")):
        x += dx
        at[f"line_{branch}"] = (x, 140)
        at[f"load_{bus}"] = (x + dx // 2, 300)
        x += dx // 2
    at["close"] = (x + dx, 300)
    missing = {u.name for u in fs.units} - set(at)
    assert not missing, missing
    nodes = {k: {"x": float(px), "y": float(py)} for k, (px, py) in at.items()}

    return write(
        "07_power_feeder",
        fs,
        title="Radial distribution feeder (ladder)",
        description=(
            "A four-section 12.47 kV distribution feeder solved "
            "sequentially, with the substation infeed as the one tear "
            "and LadderClose driving the open-end leftover power to zero."
        ),
        nodes=nodes,
        # Signed power flows: the infeed tear must not be clipped.
        solver={"clip_negative_flows": False, "tol": 1e-12, "max_iter": 100},
    )


# ---------------------------------------------------------------------
# REE: didymium (Pr/Nd) from a bastnasite liquor on PC88A
# ---------------------------------------------------------------------

def ree() -> Path:
    """Pr/Nd (didymium) solvent extraction and oxalate precipitation.

    The operating point is the plugin's own worked design in
    examples/09_ree_ndfeb_magnet.ipynb (section 4): a Mountain Pass
    bastnasite chloride liquor with the Ce already removed, 0.30 M in
    total rare earth at 5000 t REO/yr (8000 h/yr), on 0.5 M PC88A
    (record basis) in kerosene. Extraction 10 stages at pH 1.15 and
    O/A 1.2 (the La/Pr cut, PC88A window [0.1, 2.5]); scrub 6 stages at
    pH 1.05 with scrub/organic 0.3; strip 5 stages at pH 0.2 with
    strip/organic 0.4. The scrub liquor is refluxed to the extraction
    feed through a mixer (the circuit's recycle_scrub_liquor=True,
    #377), which is what lifts Pr recovery from 45 % to 79 %. The strip liquor is precipitated with 0.5 M
    oxalic acid at 1.5x stoichiometry (PrecipitatorParams' own "50 %
    excess typical"; 1.5 C2O4 per RE).
    """
    from difflow.streams import make_stream
    from difflow_ree import (OxalatePrecipitator, PrecipitatorParams,
                             REEExtractor, REEExtractorParams, REEScrubber,
                             REEStripper, ScrubberParams, StripperParams,
                             ree_oxide_mass_flow)

    # Notebook 09: Mountain Pass REO with Ce removed, mol % of contained RE.
    mol_pct = {"La": 65.44, "Pr": 8.62, "Nd": 23.84, "Sm": 1.62,
               "Gd": 0.41, "Dy": 0.061}
    elements = tuple(mol_pct)
    extractant, diluent, conc = "PC88A", "kerosene", 0.5
    T, P = 298.15, 101325.0

    total = sum(mol_pct.values())
    x = {e: v / total for e, v in mol_pct.items()}
    kg_per_mol = sum(x[e] * float(ree_oxide_mass_flow({e: 1.0}))
                     for e in elements)
    ree_mol_s = 5000.0 * 1000.0 / kg_per_mol / (8000.0 * 3600.0)
    F_aq = ree_mol_s / 0.30 * 55.5          # water, 55.5 mol per litre
    F_org = 1.2 * F_aq                      # O/A 1.2, on the diluent (#373)

    species = ["H2O", *elements, extractant, diluent, "C2O4"]
    fs = Flowsheet(species_order=species, default_T=T, default_P=P)
    zero = {e: 0.0 for e in elements}
    fs.add_feed("leach_liquor", make_stream(
        {"H2O": F_aq, **{e: x[e] * ree_mol_s for e in elements}}, T, P))
    fs.add_feed("solvent", make_stream(
        {diluent: F_org, extractant: conc * F_org, **zero}, T, P))
    fs.add_feed("scrub_acid", make_stream({"H2O": 0.3 * F_org, **zero}, T, P))
    fs.add_feed("strip_acid", make_stream({"H2O": 0.4 * F_org, **zero}, T, P))

    common = dict(extractant=extractant, elements=elements, diluent=diluent,
                  extractant_conc=conc)
    # The scrub liquor (the Pr/Nd washed off with the co-extracted La) is
    # returned to the extraction feed, as in a plant and as the circuit's
    # recycle_scrub_liquor=True does (#377). The tear is the mixed
    # extraction feed, which carries every species of the flowsheet.
    fs.add_unit(Unit("extraction", REEExtractor(REEExtractorParams(
        n_stages=10, pH=1.15, **common)),
        ["extraction_feed_guess", "solvent"], ["raffinate", "loaded_organic"]))
    fs.add_unit(Unit("scrubbing", REEScrubber(ScrubberParams(
        n_stages=6, pH=1.05, target_elements=("Pr", "Nd"), **common)),
        ["loaded_organic", "scrub_acid"], ["scrub_liquor", "scrubbed_organic"]))
    fs.add_unit(Unit("stripping", REEStripper(StripperParams(
        n_stages=5, pH=0.2, **common)),
        ["scrubbed_organic", "strip_acid"], ["strip_liquor", "barren_organic"]))
    fs.add_unit(Unit("feed_mixer", Mixer(species), ["leach_liquor", "scrub_liquor"],
                     ["extraction_feed"]))
    fs.add_recycle("extraction_feed", "extraction_feed_guess")

    # Oxalic acid sized on the strip liquor the three sections deliver.
    strip_ree = sum(float(v) for e, v in
                    fs.solve(tol=1e-11, max_iter=200)["strip_liquor"].items()
                    if e[2:] in elements)
    F_ox = 1.5 * 1.5 * strip_ree
    fs.add_feed("oxalic_acid", make_stream(
        {"H2O": F_ox / 0.5 * 55.5, "C2O4": F_ox}, T, P))
    fs.add_unit(Unit("precipitation", OxalatePrecipitator(PrecipitatorParams(
        elements=elements, precipitant_excess=1.5)),
        ["strip_liquor", "oxalic_acid"], ["filtrate", "oxalate_cake"]))

    nodes = {
        "feed:leach_liquor": {"x": 60, "y": 100},
        "feed_mixer": {"x": 220, "y": 160},
        "feed:solvent": {"x": 220, "y": 340},
        "extraction": {"x": 400, "y": 240},
        "feed:scrub_acid": {"x": 580, "y": 60},
        "scrubbing": {"x": 600, "y": 240},
        "feed:strip_acid": {"x": 800, "y": 60},
        "stripping": {"x": 820, "y": 240},
        "feed:oxalic_acid": {"x": 1020, "y": 60},
        "precipitation": {"x": 1040, "y": 240},
    }

    code_context = (
        "# Didymium (Pr/Nd) on PC88A: the worked design of\n"
        "# examples/09_ree_ndfeb_magnet.ipynb. These two names are what the\n"
        "# REE units need and are not numbers, so the palette can drop more.\n"
        f"extractant = {extractant!r}\n"
        f"elements = {elements!r}\n"
    )

    return write(
        "08_ree_separation",
        fs,
        title="Didymium (Pr/Nd) solvent extraction",
        description=(
            "A Ce-free bastnasite chloride liquor split on PC88A: Pr and Nd "
            "extracted from La, scrubbed with the scrub liquor refluxed, "
            "stripped and precipitated as oxalate."
        ),
        nodes=nodes,
        code_context=code_context,
        # The loop gain is about 0.24; a tight step tolerance keeps the
        # element balance closed to ~1e-11 rather than ~1e-9.
        solver={"tol": 1e-11, "max_iter": 200},
    )


# ---------------------------------------------------------------------
# Refinery: crude front end
# ---------------------------------------------------------------------

def refinery() -> Path:
    """Crude front end: desalter, preflash drum, CDU and VDU.

    95,000 BPD of the crude of examples/36_crude_to_vacuum (SG 0.86,
    1.8 wt% S, with a heavy end, so the CDU and the VDU share one
    characterization and the CDU residue feeds the VDU with no re-cut;
    here with a TBP point added at 3 vol% so the curve meets the light
    ends) arrives at 143.5 C and 15 bar with 0.2 vol% BS&W: the crude after
    exchanger E3, the train pressure and the tank BS&W of
    examples/37_crude_preheat_train. The desalter (5 vol% wash water,
    95 % salt removal) and the adiabatic 3 bar preflash drum are that
    notebook's; its exchangers E4 and E5 between them (3.24 + 2.19 MW)
    are the drum's duty here. The atmospheric column is example 36's (30
    stages, three side strippers, two pumparounds, 5 % overflash, cuts as
    fractions of the crude's standard volume). The preflash vapour (about
    4 vol% of the crude in example 37, mostly C3-C5) leaves before the column instead of
    entering it, so the naphtha draw is example 36's 20 % less that. The
    VDU runs at the plugin's defaults, as in example 36. Every parameter
    is a plain dataclass, so the file needs no code context.
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp

    import difflow_refinery as dr
    from difflow import Flowsheet
    from difflow.flowsheet import Unit
    from difflow_refinery import column as cc
    from difflow_refinery.vacuum import VacuumColumn, VacuumColumnParams

    C = 273.15
    BPD = 95_000.0
    # The 3 vol% point (the n-pentane boiling point) closes the TBP curve onto
    # the light ends; without it the 3-5 vol% between them joins the first cut,
    # and characterize warns.
    assay = dr.Assay(
        [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
        [t + C for t in (36, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680)],
        sg=0.86, light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
        heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0,
        ccr_wt=6.0, nickel_vanadium_wppm=60.0, asphaltenes_wt=3.0)
    char = dr.characterize(assay)
    th = dr.ColumnThermo.from_characterization(char)
    kg_s = BPD * cc.BARREL / 86400.0 * float(char.bulk_sg) * 999.016
    sized = char.stream(kg_s, T=300.0, P=1e5, basis="mass")
    flows = jnp.stack([jnp.asarray(sized[f"F_{n}"]) for n in th.names])
    Vf = float(th.std_volume(flows))              # m^3/s at 15 C

    column = cc.CrudeColumnParams(
        n_stages=30, feed_stage=27, P_top=1.5e5, P_bottom=1.9e5,
        P_condenser=1.3e5, bottom_steam=150.0, steam_T=C + 260.0,
        side_products=(cc.SideProduct("kero", 9, 4, steam=40.0),
                       cc.SideProduct("diesel", 16, 4, steam=40.0),
                       cc.SideProduct("ago", 22, 3, steam=20.0)),
        pumparounds=(cc.Pumparound("pa1", 12, 10), cc.Pumparound("pa2", 19, 17)),
        specs=(cc.product_rate("naphtha", (0.20 - 0.041) * Vf),
               cc.product_rate("kero", 0.11 * Vf),
               cc.product_rate("diesel", 0.17 * Vf),
               cc.product_rate("ago", 0.05 * Vf),
               cc.pumparound_duty("pa1", 15e6), cc.pumparound_delta_t("pa1", 60.0),
               cc.pumparound_duty("pa2", 20e6), cc.pumparound_delta_t("pa2", 60.0),
               cc.overflash(0.05)))

    desalter = dr.Desalter(dr.DesalterUnitParams(
        assay=assay, desalter=dr.DesalterParams()))
    preflash = dr.PreflashDrum(dr.PreflashDrumUnitParams(
        assay=assay, drum=dr.PreflashDrumParams(P=3.0e5, duty=5.43e6)))
    cdu = dr.CrudeDistillationUnit(dr.CrudeDistillationUnitParams(
        assay=assay, column=column))
    vdu = VacuumColumn(VacuumColumnParams(components=char.pseudo_components()))

    feed = desalter.feed(float(jnp.sum(flows)), T=C + 143.5, P=15.0e5,
                         water=0.002)

    # CDU water is F_water, VDU steam is F_H2O (difflow_refinery/CLAUDE.md)
    fs = Flowsheet(species_order=list(char.names) + ["water", "H2O"])
    fs.add_feed("crude", {k: float(v) for k, v in feed.items()})
    fs.add_unit(Unit("desalter", desalter, ["crude"],
                     ["desalted_crude", "brine"]))
    fs.add_unit(Unit("preflash", preflash, ["desalted_crude"],
                     ["preflash_vapor", "topped_crude", "preflash_water"]))
    fs.add_unit(Unit("cdu", cdu, ["topped_crude"],
                     ["naphtha", "kero", "diesel", "ago", "atm_residue",
                      "sour_water"]))
    fs.add_unit(Unit("vdu", vdu, ["atm_residue"],
                     ["vac_overhead", "lvgo", "hvgo", "slop", "vac_residue"]))

    y = 200
    nodes = {
        "feed:crude": {"x": 60, "y": y},
        "desalter": {"x": 260, "y": y},
        "preflash": {"x": 460, "y": y},
        "cdu": {"x": 680, "y": y},
        "vdu": {"x": 920, "y": y + 120},
    }
    return write(
        "09_refinery_crude_unit",
        fs,
        title="Crude unit: desalter to vacuum column",
        description=(
            "95,000 BPD of a sour 0.86 SG crude through a desalter, a "
            "preflash drum, the atmospheric column and the vacuum column "
            "on its residue."
        ),
        nodes=nodes,
    )


if __name__ == "__main__":
    gaslib11()
    bio()
    carbon_capture()
    power()
    ree()
    refinery()
