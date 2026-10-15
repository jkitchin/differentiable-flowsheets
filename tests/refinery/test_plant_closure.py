"""Plant-level closure of the sulfur and hydrogen systems (#475).

One crude (the hydrotreating/hydrocracking test assay, 1.8 wt% S) feeds a
kerosene hydrotreater, a hydrocracker on its vacuum gas oil and an FCC on the
same VGO cut; a CCR reformer runs the lean naphtha with 0.5 wppm S and an
SMR at the IEAGHG 2017/02 base case is the swing hydrogen plant.

Hydrogen: CCR net gas + SMR (swing) -> header -> hydrotreater and
hydrocracker, closed on the header's makeup purity
(:func:`~difflow_refinery.hydrogen.close_hydrotreater_loop`).

Sulfur: the H2S of the SAME converged units -> the refinery's amine system ->
Claus + SCOT. Routing (stated, not modelled):

- the recycle-gas scrubbers' acid gas (``acid_gas``: H2S and NH3) shares
  the amine regenerator, so it goes straight to the SRU feed;
- off gas and purge of the hydroprocessors, the FCC dry gas and the
  reformer's net/fuel-gas H2S (a trace) are compressed to one fuel-gas
  contactor (313 K, 7 bar);
- the off gas and purge also carry vaporised oil lumps with ORGANIC sulfur
  (~1e-5 mol/s here); an amine contactor does not take it, so it is booked
  as burned with the fuel gas, not sent to treating;
- sour water would go to a sour-water stripper; here it carries no sulfur
  and is a separate sink.

The closure counts sulfur atoms (mol/s) from the four units' feeds to
liquid products, FCC flue SO2, sweet fuel gas, amine flash gas, liquid
sulfur and the SO2 the SRU emits. Each unit uses its own S molar mass (FCC
32.06, reformer 32.065); the closure is in moles.

The fixtures are module-scoped full solves; each test only reads them.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

import difflow_refinery as dr
import difflow_refinery.hydrogen as h2
from difflow import make_stream
from difflow_refinery.fcc import FCCFeed, FCCParams, FCCUnit
from difflow_refinery.fcc import species as fsp
from difflow_refinery.hydrocracking.unit import Hydrocracker, HydrocrackerParams
from difflow_refinery.hydrogen import smr
from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams, straight_run_cut
from difflow_refinery.hydrotreating.feed import select_cuts
from difflow_refinery.reforming import lean_naphtha
from difflow_refinery.reforming.ccr import CCRReformer, ccr_reformer_params
from difflow_refinery.reforming.sulfur import S_MW as REFORMER_S_MW
from difflow_refinery.sulfur import claus as cl
from difflow_refinery.sulfur import tgtu as tg
from difflow_refinery.treating import AmineUnitParams, Contactor, TrayParams, solve_amine_unit

jax.config.update("jax_enable_x64", True)

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_A = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]
VGO = (343.0 + 273.15, 550.0 + 273.15)

#: IEAGHG 2017/02 base case natural gas (stream 2: 1455.8 kmol/h, 128 C, 3.71 MPa).
IEAGHG_NG = {"methane": 0.89, "ethane": 0.07, "propane": 0.01, "n_butane": 0.0011,
             "carbon_dioxide": 0.02, "nitrogen": 0.0089, "hydrogen_sulfide": 5e-6}

TRAY = TrayParams(area=200.0, k_G=1e-5, k_L=1e-4)     # as tests/refinery/test_treating.py
FUEL_GAS_T, FUEL_GAS_P = 313.15, 7e5

# Hydroprocessor outlets (`Flows`); the out-lists of their balances.
HDT_OUT = ("purge", "acid_gas", "hps_water", "off_gas", "wild_naphtha", "sour_water", "product")
HCU_OUT = ("purge", "acid_gas", "hps_water", "off_gas", "lpg", "light_naphtha", "heavy_naphtha",
           "kerosene", "diesel", "uco_bleed", "sour_water")
TO_FUEL_GAS = ("off_gas", "purge")
WATER = ("hps_water", "sour_water")


def _char():
    a = dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_A]), sg=0.86,
                 light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dr.characterize(a, cut_points=dr.default_cut_points(a), composition=True)


def _vgo(char, crude_kg_s=150.0):
    """LVGO + HVGO of a VacuumColumn on an idealized atmospheric residue."""
    from difflow_refinery.vacuum import VacuumColumn, VacuumColumnParams, atmospheric_residue
    out = dict(zip(["vac_overhead", "lvgo", "hvgo", "slop", "vac_residue", "info"],
                   VacuumColumn(VacuumColumnParams(components=char.pseudo_components()))(
                       atmospheric_residue(char, crude_kg_s))))
    vgo = {}
    for cut in ("lvgo", "hvgo"):
        for k, v in out[cut].items():
            if k.startswith("F_"):
                vgo[k] = vgo.get(k, 0.0) + v
    vgo["T"], vgo["P"] = jnp.asarray(298.15), jnp.asarray(101325.0)
    return vgo


def _S(flows, layout):
    return float(flows.elements(layout)[2])


def _gas(flows, layout):
    return {g: flows.gas[i] for i, g in enumerate(layout.gases)}


class _Rebuilt:
    """A hydrocracker for :func:`close_hydrotreater_loop`: its ``makeup`` shapes
    the compiled solve (``Hydrocracker.solve`` refuses a new one), so each new
    makeup builds a new unit."""

    def __init__(self, char, feed, params):
        self.char, self.unit = char, Hydrocracker(char, feed, params)
        self.layout = self.unit.layout

    def solve(self, feed, params=None, warn=True):
        if params is not None and params.makeup != self.unit.params.makeup:
            self.unit = Hydrocracker(self.char, feed, params)
        return self.unit.solve(feed, params=params, warn=warn)


# ----- the units ---------------------------------------------------------------


@pytest.fixture(scope="module")
def char():
    return _char()


@pytest.fixture(scope="module")
def ccr():
    return CCRReformer(ccr_reformer_params(feed_sulfur_wppm=0.5)).solve(lean_naphtha())


@pytest.fixture(scope="module")
def smr_plant():
    p = smr.SMRParams(n_tubes=240, psa=h2.PSA(recovery=0.90, purity=0.9999))
    ng = make_stream({k: v * 1455.8 / 3.6 for k, v in IEAGHG_NG.items()}, 401.15, 37.1e5)
    return smr.smr_plant(ng, p)


@pytest.fixture(scope="module")
def consumers(char):
    kc = select_cuts(char, 150 + 273.15, 250 + 273.15)
    kfeed = straight_run_cut(char, rate=20.0, cuts=kc)
    kp = HydrotreaterParams(T_in=(593.15,), quench=None, bed_fractions=(1.0,), P=35e5, h2_oil=150.0,
                            lhsv=2.5, stripper_feed_T=473.15)
    vgo = _vgo(char)
    hp = HydrocrackerParams()
    return {"kht": (Hydrotreater(char, kfeed, kp), kfeed, kp),
            "hcu": (_Rebuilt(char, vgo, hp), vgo, hp)}


@pytest.fixture(scope="module")
def loop(ccr, smr_plant, consumers):
    plant = h2.H2Plant.from_smr(smr_plant, P=30e5)
    cons = [h2.Consumer(n, 50.0, P=p.P, min_purity=0.85) for n, (_, _, p) in consumers.items()]
    net = h2.HydrogenNetwork(producers=[h2.Producer.from_reformer(ccr.reformer, name="ccr")],
                             consumers=cons,
                             headers=[h2.Header("main", P=20e5, min_purge=1.0, swing=[plant])])
    return h2.close_hydrotreater_loop(net, consumers, tol=1e-7)


@pytest.fixture(scope="module")
def fcc(char):
    feed = FCCFeed.from_characterization(char, 50.0, *VGO)
    return feed, FCCUnit(FCCParams()).solve(feed)


# ----- hydrogen -----------------------------------------------------------------


class TestHydrogenClosure:
    def test_units_converged(self, ccr, smr_plant, loop):
        assert ccr.converged and bool(smr_plant["converged"]) and loop.converged
        for r in loop.units.values():
            assert bool(r.converged)

    def test_header_and_units_balance(self, loop):
        for k, v in loop.header.balances.items():
            assert abs(float(v)) < 1e-12, k
        for name, r in loop.units.items():
            for k, v in r.balances.items():
                assert abs(float(v)) < 1e-8, (name, k)

    def test_h2_closes(self, ccr, loop):
        """CCR net H2 + SMR H2 - makeups = surplus (purged), on the solved units' own numbers
        (to the loop's tolerance: the header carries each unit's linear purity response)."""
        o = loop.header.outputs
        made = float(ccr.reformer.outputs()["h2.net_mol_s"]) + float(o["h2_plant.h2"])
        used = sum(float(r.outputs["h2.makeup"]) for r in loop.units.values())   # an H2 flow
        # the surplus is a small difference of ~800 mol/s flows the loop closes to
        # its 1e-7 tolerance: judge it on the throughput, not on itself
        assert float(o["h2.surplus"]) == pytest.approx(made - used, abs=1e-6 * made)
        for n, r in loop.units.items():   # the header balances what each unit reports
            assert float(o[f"{n}.makeup_h2"]) == pytest.approx(float(r.outputs["h2.makeup"]), rel=1e-6)
        assert float(o["h2.surplus"]) >= -1e-9
        for n in loop.units:  # each unit was solved at the header's purity
            y = loop.params[n].makeup["hydrogen"] / sum(loop.params[n].makeup.values())
            assert y == pytest.approx(float(o[f"{n}.purity"]), abs=1e-6)

    def test_smr_swings_inside_its_capacity(self, smr_plant, loop):
        """The CCR alone is short; the SMR (IEAGHG capacity, 4461.5 kmol/h
        product) is part-loaded, never past its capacity."""
        used = float(loop.header.outputs["h2_plant.h2"])
        assert 0.0 < used < float(smr_plant["H2"])

    def test_makeup_is_the_header_gas(self, loop):
        """What each consumer takes is the header's mixture of CCR net gas
        (~85 mol% H2) and SMR product (99.99%): purity strictly between."""
        y = float(loop.header.outputs["kht.purity"])
        assert 0.85 < y < 0.9999


# ----- sulfur -------------------------------------------------------------------


@pytest.fixture(scope="module")
def sulfur(loop, consumers, fcc, ccr):
    """Every unit's sulfur, routed through the amine unit and the SRU."""
    rows, fuel, scrub = {}, {}, {"F_hydrogen_sulfide": 0.0, "F_ammonia": 0.0}

    def add_gas(d):
        for g, v in d.items():
            fuel[f"F_{g}"] = fuel.get(f"F_{g}", 0.0) + v

    for name, outs in (("kht", HDT_OUT), ("hcu", HCU_OUT)):
        unit, r = consumers[name][0], loop.units[name]
        lay, st = unit.layout, r.streams
        rows[name] = {"feed": _S(st["feed"], lay), "out": sum(_S(st[k], lay) for k in outs),
                      "scrubber": _S(st["acid_gas"], lay),
                      "fuel_gas": sum(_S(st[k], lay) for k in TO_FUEL_GAS),
                      "sour_water": sum(_S(st[k], lay) for k in WATER)}
        rows[name]["liquid"] = rows[name]["out"] - rows[name]["scrubber"] - rows[name]["fuel_gas"] \
            - rows[name]["sour_water"]
        # the off-gas/purge also carry vaporised oil lumps with ORGANIC sulfur:
        # an amine contactor does not take it; it burns with the fuel gas
        rows[name]["fuel_organic"] = rows[name]["fuel_gas"] - sum(
            float(st[k].gas_flow(lay, "hydrogen_sulfide")) for k in TO_FUEL_GAS)
        rows[name]["fuel_gas"] -= rows[name]["fuel_organic"]
        scrub["F_hydrogen_sulfide"] += st["acid_gas"].gas_flow(lay, "hydrogen_sulfide")
        scrub["F_ammonia"] += st["acid_gas"].gas_flow(lay, "ammonia")
        for k in TO_FUEL_GAS:
            add_gas(_gas(st[k], lay))

    feed, fr = fcc
    s_mol = 1000.0 / fsp.ATOMIC_WEIGHT["S"]
    dry = fr["streams"]["dry_gas"]
    rows["fcc"] = {"feed": float(feed.sulfur * feed.mass) * s_mol,
                   "fuel_gas": float(dry["hydrogen_sulfide"]),
                   "flue_SO2": float(fr["streams"]["flue_gas"]["sulfur_dioxide"]),
                   "liquid": sum(float(fr["outputs"][f"{p}.sulfur"] * fr["outputs"][f"{p}.mass"])
                                 for p in ("gasoline", "lco", "slurry")) * s_mol}
    add_gas(dry)

    s = ccr.reformer.sulfur()
    rows["ccr"] = {"feed": float(s["S.feed"]) * 1000.0 / REFORMER_S_MW,
                   "liquid": float(s["S.reformate"]) * 1000.0 / REFORMER_S_MW,
                   "fuel_gas": float(s["H2S.net_gas"] + s["H2S.fuel_gas"])}
    add_gas({"hydrogen_sulfide": s["H2S.net_gas"] + s["H2S.fuel_gas"]})

    fuel["T"], fuel["P"] = jnp.asarray(FUEL_GAS_T), jnp.asarray(FUEL_GAS_P)
    amine = solve_amine_unit(AmineUnitParams("MDEA", 0.45, (Contactor("fuel", 250.0, 10, TRAY),),
                                             Q_reboiler=1.2e7, n_regenerator=8), {"fuel": fuel})
    ag = dict(amine.acid_gas)
    ag["F_hydrogen_sulfide"] = ag["F_hydrogen_sulfide"] + scrub["F_hydrogen_sulfide"]
    ag["F_ammonia"] = ag.get("F_ammonia", 0.0) + scrub["F_ammonia"]

    beds = (cl.ConverterParams(523.15, hydrolysis=(0.9, 0.7)), cl.ConverterParams(483.15))
    claus = cl.ClausParams(converters=beds, T_condensers=(443.15, 438.15, 413.15), T_air=473.15)
    tgtu = tg.TGTUParams(tray=TRAY, L=30.0, n_stages=12, P_absorber=1.2e5)
    sru = tg.solve_sru(ag, claus, tgtu)
    return {"rows": rows, "fuel": fuel, "amine": amine, "acid_gas": ag, "sru": sru}


class TestSulfurClosure:
    def test_units_balance_sulfur(self, sulfur, fcc, ccr):
        rows = sulfur["rows"]
        for n in ("kht", "hcu"):
            assert rows[n]["out"] == pytest.approx(rows[n]["feed"], rel=1e-8)
        assert abs(float(fcc[1]["balances"]["S"])) < 1e-8
        assert abs(float(ccr.reformer.balances()["sulfur"])) < 1e-8

    def test_amine_conserves_h2s(self, sulfur):
        a, H = sulfur["amine"], "F_hydrogen_sulfide"
        assert bool(a.converged)
        out = a.sweet["fuel"][H] + a.flash_gas[H] + a.acid_gas[H]
        assert float(out) == pytest.approx(float(sulfur["fuel"][H]), rel=1e-9)

    def test_sweet_fuel_gas(self, sulfur):
        """The fuel-gas contactor takes the H2S to the ppm level (Kohl &
        Nielsen 1997 ch. 2: MDEA fuel-gas treating to < 4 ppmv-class specs is
        routine; 100 ppmv is the loose bound asserted here)."""
        sw = sulfur["amine"].sweet["fuel"]
        tot = sum(float(v) for k, v in sw.items() if k.startswith("F_"))
        assert float(sw["F_hydrogen_sulfide"]) / tot < 100e-6

    def test_sru_closes(self, sulfur):
        sru = sulfur["sru"]
        assert bool(sru["converged"])
        S_in = float(cl.sulfur_in(cl.to_vec(sulfur["acid_gas"])))
        assert float(sru["claus"]["sulfur"] + sru["SO2_emitted"]) == pytest.approx(S_in, rel=1e-9)

    def test_sru_recovery(self, sulfur):
        """Claus (2 beds) + SCOT with the acid gas recycled: AP-42 8.13 gives
        about 99.9 %; the #472 benchmark on a clean acid gas is 99.987 %."""
        assert float(sulfur["sru"]["recovery"]) > 0.999

    def test_plant_sulfur_closes(self, sulfur):
        """Feed S of the four units = liquid products + FCC flue SO2 + sour
        water + organic S burned in fuel gas + sweet fuel gas + amine flash gas
        + liquid sulfur + SRU SO2."""
        rows, a, sru, H = sulfur["rows"], sulfur["amine"], sulfur["sru"], "F_hydrogen_sulfide"
        s_in = sum(r["feed"] for r in rows.values())
        fcc = rows["fcc"]
        retained = (rows["kht"]["liquid"] + rows["hcu"]["liquid"] + fcc["liquid"] + rows["ccr"]["liquid"]
                    + rows["kht"]["sour_water"] + rows["hcu"]["sour_water"] + fcc["flue_SO2"]
                    + rows["kht"]["fuel_organic"] + rows["hcu"]["fuel_organic"])
        gas_side = float(a.sweet["fuel"][H] + a.flash_gas[H] + sru["claus"]["sulfur"] + sru["SO2_emitted"])
        assert retained + gas_side == pytest.approx(s_in, rel=1e-8)
        # and the gas side alone: what the units sent to treating
        sent = sum(r.get("fuel_gas", 0.0) + r.get("scrubber", 0.0) for r in rows.values())
        assert gas_side == pytest.approx(sent, rel=1e-8)

    def test_where_the_sulfur_goes(self, sulfur):
        """Most of the crude's sulfur ends as liquid sulfur; what leaves the
        refinery as SO2 (FCC flue + SRU stack) is a few percent at most."""
        rows, sru = sulfur["rows"], sulfur["sru"]
        s_in = sum(r["feed"] for r in rows.values())
        assert float(sru["claus"]["sulfur"]) / s_in > 0.5
        assert (rows["fcc"]["flue_SO2"] + float(sru["SO2_emitted"])) / s_in < 0.1
