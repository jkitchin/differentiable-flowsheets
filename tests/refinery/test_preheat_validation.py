"""The preheat train's drum and exchangers against IDAES unit models (#313).

``reference/preheat_generate.py`` writes ``reference/preheat_reference.json``
from an IDAES ``Flash`` (the adiabatic preflash drum), IDAES state blocks
(the wet crude at drum conditions) and an IDAES ``HeatExchanger``
(counter-current, exact LMTD) at the inlets of the base case's two hottest
exchangers. These tests read it and need neither IDAES nor IPOPT.

**This is an independent implementation, not an independent model.** IDAES
is given the same property model -- Raoult over Lee-Kesler vapour pressures,
the cubic ideal-gas Cp, a Watson liquid enthalpy, water immiscible and
vapour-only -- on the same pseudo-component constants. What it checks is
everything difflow assembles on top of that: the three-phase split's
Rachford-Rice with water's infinite K, the adiabatic drum's enthalpy
balance, and the exchanger's duty, LMTD and two-phase cold side. It says
nothing about whether Raoult is right for a crude at 12 bar -- the column's
Peng-Robinson comparison (``test_validation.py``) is where that is measured.
The free-water branch of the split is not here: IDAES's package carries
water as vapour-only, so the per-commit tests check that branch by hand
(``test_preheat.py::TestThermo``).

No published preheat-train case study is reproduced. Polley, Wilson,
Yeap and Pugh (2002) was the one the issue named; none was found that
gives a train's full data (assay, exchanger areas and U, hot-stream rates)
in a form that could be set up here, so there is no comparison with it.

Tolerances, with the measured agreement in brackets:

* drum temperature, 1e-5 K (2e-6 K) and vapour fraction 1e-6 relative (3e-8);
* drum vapour composition, 1e-8 absolute (2e-9);
* wet vapour fraction and vapour water fraction, 1e-9 relative (2e-12);
* exchanger duty, 1e-8 relative (3.5e-10); outlet temperatures 1e-6 K (8e-8 K).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.preheat import crude_split

from .reference import preheat_case as pc

pytestmark = pytest.mark.release

jax.config.update("jax_enable_x64", True)

REF = json.loads((Path(__file__).parent / "reference" / "preheat_reference.json").read_text())


@pytest.fixture(scope="module")
def thermo():
    crude = dr.characterize(pc.assay("base"))
    return dr.ColumnThermo.from_characterization(crude)


class TestDrum:
    def test_adiabatic_flash(self, thermo):
        d, inp = REF["drum"], REF["drum"]["inputs"]
        op = dr.PreflashDrum(dr.PreflashDrumUnitParams(pc.assay("base"), dr.PreflashDrumParams(P=inp["P"])))
        r = op.solve(op.feed(inp["rate"], T=inp["T_in"], P=inp["P_in"], water=0.0))
        assert float(r["T"]) == pytest.approx(d["T"], abs=1e-5)
        assert float(r["vapor_fraction"]) == pytest.approx(d["vapor_fraction"], rel=1e-6)
        y = np.array([float(r["vapor"][f"F_{n}"]) for n in thermo.names])
        np.testing.assert_allclose(y / y.sum(), d["y"], atol=1e-8)

    @pytest.mark.parametrize("k", range(2))
    def test_wet_split_with_all_water_vapour(self, thermo, k):
        w = REF["wet"][k]
        zw = w["z_water"]
        f = jnp.asarray(REF["components"]["mole_fraction"]) * (1 - zw)
        sp = crude_split(thermo, f, jnp.asarray(zw), w["T"], w["P"])
        assert float(sp["water_liquid"]) == 0.0
        V = float(jnp.sum(sp["vapor"]) + sp["water_vapor"])
        assert V == pytest.approx(w["vapor_fraction"], rel=1e-9)
        assert float(sp["water_vapor"]) / V == pytest.approx(w["y_water"], rel=1e-9)


class TestExchangers:
    @pytest.mark.parametrize("name", ["E7", "E8"])
    def test_duty_and_outlets(self, thermo, name):
        e = REF["exchangers"][name]
        inp = e["inputs"]
        src = inp["hot_source"]
        tp = dr.PreheatTrainParams((dr.PreheatExchanger(name, inp["UA"], 1.0),), (dr.HotStream(src, (name,)),),
                                   (name,), tank_water=0.0, P_crude=inp["cold_P"])
        r = dr.PreheatTrain(tp, thermo).solve(jnp.asarray(inp["cold_flows"]), inp["cold_T"],
                                              {src: (jnp.asarray(inp["hot_flows"]), inp["hot_T"])})
        assert bool(r.converged)
        x = r.exchangers[name]
        assert float(x["Q"]) == pytest.approx(e["Q"], rel=1e-8)
        assert float(x["T_cold_out"]) == pytest.approx(e["T_cold_out"], abs=1e-6)
        assert float(x["T_hot_out"]) == pytest.approx(e["T_hot_out"], abs=1e-6)
