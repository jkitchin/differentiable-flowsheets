"""The gas plant against DWSIM 9.0.5: rigorous columns, the compressor
train and the vapour pressures.

DWSIM ran in the generator (``reference/dwsim_gasplant_generate.py``) and
wrote ``reference/dwsim_gasplant_reference.json``; this file reads it and
needs neither DWSIM nor .NET. The cases are in
``reference/dwsim_gasplant_case.py``.

Comparison (a), same model and constants -- every DWSIM component a
hypothetical compound on difflow's constants, difflow's kij (zero for the
columns), the liquid density from the EOS:

* **Columns.** DWSIM's ``DistillationColumn`` with the reflux and boilup
  ratios fixed, like the IDAES reference -- and DWSIM converges the C3/C4
  splitter, which IDAES's ``TrayColumn`` could not, so the splitter gets
  its column-level check here. Agreement is bounded by DWSIM's column
  tolerance (loop tolerance 1e-9; its Wang-Henke closes component balances
  only to about 1e-8), not by difflow. The duties differ by 5e-5 relative,
  all of it DWSIM's R = 8.314 and its midpoint-rule ideal-gas enthalpy
  (both found for the smoke test): recomputed from DWSIM's own stage states
  with those two reproduced, the duties agree to 1e-13.
* **Compressor.** Stage power, discharge temperatures and the knock-out
  split. This comparison found a bug in
  :class:`~difflow_refinery.gasplant.GasCompressor`: its temperature solve
  took Newton steps on ``grad(stop_gradient(fn))``, which is zero, so the
  answer was one final Newton step from a clipped bounce -- 1.3 K low on
  the isentropic temperature and 3.6 % low on the first stage's power. Fixed
  in ``gasplant/units.py`` (``_solve_T``); the regression is
  ``test_gasplant.py::test_solve_T_converges``.
* **Vapour pressure.** The bubble pressure at 100 F (TVP) and difflow's
  ASTM D323 construction rebuilt on DWSIM's flashes.

Comparison (b), DWSIM's own data -- database compounds and DWSIM's kij; and
DWSIM's own RVP, a correlation on the TVP (the classic UI's cold-flow
utility), against difflow's D323 construction. These measure the model,
not the code, and are reported rather than tuned away.

``release`` throughout but for the per-commit checks on the file
(:class:`TestReferenceFile`, :class:`TestReferenceIsCurrent`).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .reference import dwsim_gasplant_case as dc

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent / "reference"
REF = json.loads((HERE / "dwsim_gasplant_reference.json").read_text())
COLUMNS = sorted(dc.COLUMNS)
DB_COLUMNS = sorted(n for n in COLUMNS if "dwsim_data" in REF["columns"][n])
RVPS = sorted(dc.RVP)
R_DWSIM = 8.314
R = 8.314462618

REGENERATE = ("difflow's constants (or the cases) no longer match the ones the DWSIM gas-plant "
              "reference was built on; regenerate it: PYTHONPATH=src:tests python -m "
              "refinery.reference.dwsim_gasplant_generate (needs DWSIM 9.0.5; see "
              "scripts/install_dwsim.sh)")


def _jsonable(x):
    return json.loads(json.dumps(x))


def _thermo(comp):
    from difflow_refinery.gasplant import CubicThermo
    from difflow_refinery.gasplant.thermo import PR

    return CubicThermo(dc.gas_components_of(comp), PR)


def _h_dwsim(th, cp, T, P, x, phase):
    """DWSIM's molar enthalpy on difflow's EOS: midpoint-rule ideal gas plus
    the departure at R = 8.314 (both from ``test_dwsim_smoke.py``)."""
    from .test_dwsim_smoke import _dwsim_h_ig

    x = np.asarray(x, float)
    return (float(np.dot(x, _dwsim_h_ig(cp, T)))
            + R_DWSIM / R * float(th.h_departure(T, P, jnp.asarray(x), phase)))


# ---------------------------------------------------------------------------
# Per commit: the file, and whether it is still about difflow's constants
# ---------------------------------------------------------------------------


class TestReferenceFile:
    def test_it_records_where_it_came_from(self):
        p = REF["provenance"]
        for key in ("generated", "script", "command", "dwsim", "dotnet", "pythonnet",
                    "reference_simulator", "difflow_commit"):
            assert p[key], key
        assert "9.0.5" in p["dwsim"]
        assert (HERE / "dwsim_gasplant_generate.py").exists()

    @pytest.mark.parametrize("name", COLUMNS)
    def test_columns_ran_the_like_for_like_model(self, name):
        c = REF["columns"][name]
        d = c["same"]["dwsim"]
        assert d["property_package"] == "Peng-Robinson (PR)"
        assert not np.any(d["kij_matrix"])
        assert d["options"]["LiquidDensityCalculationMode_Subcritical"] == "EOS"
        for i, k in enumerate(d["constants"]):
            assert k["is_hypo"]
            for key in ("MW", "Tc", "Pc", "omega"):
                assert k[key] == pytest.approx(c["components"][key][i], rel=1e-14)
        s = c["same"]["settings"]
        assert s["stages"] == c["case"]["n_trays"] + 2
        assert s["condenser"] == "Total_Condenser" and s["column_pressure_drop"] == 0.0

    @pytest.mark.parametrize("name", COLUMNS)
    def test_dwsim_met_the_specs_and_closed_its_balances(self, name):
        """DWSIM's column hit both ratios and closed each component to its
        tolerance: 5e-16 (Naphtali-Sandholm), 7.8e-9 (Wang-Henke)."""
        c = REF["columns"][name]
        for key in ("same",) + (("dwsim_data",) if "dwsim_data" in c else ()):
            r = c[key]
            assert r["reflux_ratio"] == pytest.approx(c["case"]["reflux_ratio"], rel=1e-8)
            assert r["boilup_ratio"] == pytest.approx(c["case"]["boilup_ratio"], rel=1e-8)
            assert r["component_balance"] < 1e-7, (key, r["component_balance"])

    def test_the_database_runs_use_dwsims_kij(self):
        """(b) is DWSIM's model: the C3/C4 splitter's database run carries
        DWSIM's nonzero kij (ethane/propylene, propane/propylene, ...)."""
        d = REF["columns"]["c3c4_splitter"]["dwsim_data"]["dwsim"]
        assert d["kij"] == "dwsim"
        assert np.any(d["kij_matrix"])
        assert not any(k["is_hypo"] for k in d["constants"])

    def test_the_compressor_runs_on_difflows_kij(self):
        c = REF["compressor"]
        assert c["same"]["dwsim"]["kij"] == "given"
        np.testing.assert_allclose(c["same"]["dwsim"]["kij_matrix"], c["components"]["kij"],
                                   atol=1e-15)
        assert np.any(c["components"]["kij"])
        for st in c["same"]["stages"]:
            assert st["efficiency_percent"] == pytest.approx(100 * c["case"]["efficiency"])


class TestReferenceIsCurrent:
    """The constants are an input to both sides; if difflow's have moved, the
    comparison is against a different mixture -- regenerate, never loosen."""

    @pytest.mark.parametrize("name", COLUMNS)
    def test_column_constants(self, name):
        case = dc.COLUMNS[name]
        assert REF["columns"][name]["case"] == _jsonable(case), REGENERATE
        self._same(dc.component_data(case["names"], case["pseudo"]),
                   REF["columns"][name]["components"])

    def test_compressor_constants(self):
        assert REF["compressor"]["case"] == _jsonable(dc.COMPRESSOR), REGENERATE
        self._same(dc.component_data(dc.COMPRESSOR["names"]), REF["compressor"]["components"])

    @pytest.mark.parametrize("name", RVPS)
    def test_rvp_constants(self, name):
        case = dc.RVP[name]
        assert REF["rvp"][name]["case"] == _jsonable(case), REGENERATE
        self._same(dc.component_data(case["names"], case["pseudo"]), REF["rvp"][name]["components"])

    @staticmethod
    def _same(now, ref):
        assert now["names"] == ref["names"], REGENERATE
        for key in ("MW", "Tc", "Pc", "omega", "cp_ig", "kij"):
            np.testing.assert_allclose(now[key], ref[key], rtol=1e-12, atol=1e-15,
                                       err_msg=f"{key}: {REGENERATE}")

    def test_the_idaes_cases_are_the_ones_reused(self):
        """The debutanizer and C3/C4 splitter are the IDAES reference's cases."""
        from .reference import gasplant_case as gc

        for name in ("debutanizer", "c3c4_splitter"):
            for k, v in gc.CASES[name].items():
                if k != "reference":
                    assert dc.COLUMNS[name][k] == v, (name, k)


# ---------------------------------------------------------------------------
# Release: columns
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=COLUMNS)
def column(request):
    name = request.param
    c = REF["columns"][name]
    col, feed = dc.difflow_column(c["case"], c["components"])
    D, B, info = col(feed)
    assert bool(info["converged"]), name
    return name, c, D, B, info


def _x(stream, names):
    f = np.array([float(stream[f"F_{n}"]) for n in names])
    return f / f.sum()


@pytest.mark.release
@pytest.mark.slow
class TestColumns:

    def test_k_values_at_dwsims_stage_states(self, column):
        """difflow's PR ``ln K`` at DWSIM's own (T, P, x, y) on every
        equilibrium stage, no column in the way. Measured 5.1e-7
        (debutanizer), 3.5e-7 (C3/C4 splitter), 1.2e-6 (naphtha splitter) --
        DWSIM's truncated sqrt(2) in the fugacity's log term (the smoke
        test reproduces it to 1e-12)."""
        name, c, *_ = column
        th = _thermo(c["components"])
        worst = 0.0
        for s in c["same"]["stages"][1:]:
            lnK = th.log_K(s["T"], s["P"], jnp.asarray(s["x"]), jnp.asarray(s["y"]))
            worst = max(worst, float(np.max(np.abs(np.exp(np.asarray(lnK)) / np.asarray(s["K"]) - 1))))
        assert worst < 5e-6, (name, worst)

    def test_temperature_profile(self, column):
        """Measured worst stage: 5.8e-5 K (debutanizer, Naphtali-Sandholm),
        3.5e-5 K (C3/C4 splitter), 2.0e-4 K (naphtha splitter) -- DWSIM's
        loop tolerance of 1e-9; IDAES and DWSIM are as far apart on the
        debutanizer."""
        name, c, D, B, info = column
        T = np.asarray(info["profiles"]["T"])
        T_ref = np.array([s["T"] for s in c["same"]["stages"]])
        assert np.max(np.abs(T - T_ref)) < 1e-3, (name, np.max(np.abs(T - T_ref)))

    def test_flow_and_composition_profiles(self, column):
        """Liquid and vapour leaving each tray and the reboiler, and their
        compositions (the condenser's are bookkept differently: DWSIM's L
        there is the reflux, difflow's the whole condensate). Measured
        flows 3.7e-6, 2.1e-6 and 1.3e-5 relative, mole fractions 4.2e-7,
        2.1e-7 and 6.2e-6 (debutanizer, C3/C4, naphtha splitter)."""
        name, c, D, B, info = column
        pr = info["profiles"]
        st = c["same"]["stages"]
        for j in range(1, len(st)):
            assert float(pr["L"][j]) == pytest.approx(st[j]["L"], rel=5e-5), (name, j)
            assert float(pr["V"][j]) == pytest.approx(st[j]["V"], rel=5e-5), (name, j)
            np.testing.assert_allclose(np.asarray(pr["x"][j]), st[j]["x"], atol=2e-5)
            np.testing.assert_allclose(np.asarray(pr["y"][j]), st[j]["y"], atol=2e-5)

    def test_product_compositions(self, column):
        """Measured, on components above 1e-4 of their product: 5.9e-6
        (debutanizer), 2.1e-6 (C3/C4 splitter) and 4.2e-5 relative (naphtha
        splitter: its 375 K cut at 8.5e-4 of the distillate, Wang-Henke's
        balance closure); 3e-7 absolute worst."""
        name, c, D, B, info = column
        names = c["components"]["names"]
        for prod, stream in (("distillate", c["same"]["distillate"]), ("bottoms", c["same"]["bottoms"])):
            ref = np.asarray(stream["z"])
            got = _x(D if prod == "distillate" else B, names)
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6, err_msg=f"{name} {prod}")
        assert float(sum(D[f"F_{n}"] for n in names)) == pytest.approx(
            c["same"]["distillate"]["F"], rel=2e-5)

    def test_duties(self, column):
        """Plain, difflow is 5e-5 above DWSIM on both duties: R and the
        midpoint rule. Measured (condenser, reboiler): 5.2e-5 and 5.1e-5
        (debutanizer), 5.5e-5 and 5.1e-5 (C3/C4), 5.1e-5 and 4.7e-5 (naphtha
        splitter); held to 1e-4."""
        name, c, D, B, info = column
        o = info["outputs"]
        assert float(o["condenser.duty"]) == pytest.approx(c["same"]["condenser_duty"], rel=1e-4)
        assert float(o["reboiler.duty"]) == pytest.approx(c["same"]["reboiler_duty"], rel=1e-4)


@pytest.mark.release
@pytest.mark.parametrize("name", COLUMNS)
def test_dwsims_duties_are_its_own_enthalpies(name):
    """The 5e-5 explained: DWSIM's condenser and reboiler duties recomputed
    from DWSIM's own stage states (flows, T, compositions) with difflow's PR
    but DWSIM's R and midpoint-rule ideal-gas enthalpy. Measured 9e-14
    relative worst; with difflow's exact enthalpy instead, the 5e-5 comes back."""
    c = REF["columns"][name]
    th = _thermo(c["components"])
    cp = np.asarray(c["components"]["cp_ig"])
    st, P = c["same"]["stages"], c["case"]["P"]
    Dn, Bn = c["same"]["distillate"]["F"], c["same"]["bottoms"]["F"]

    def duties(h):
        qc = st[1]["V"] * h(st[1]["T"], st[1]["y"], "vapor") - (st[0]["L"] + Dn) * h(
            st[0]["T"], st[0]["x"], "liquid")
        qr = (st[-1]["V"] * h(st[-1]["T"], st[-1]["y"], "vapor") + Bn * h(st[-1]["T"], st[-1]["x"], "liquid")
              - st[-2]["L"] * h(st[-2]["T"], st[-2]["x"], "liquid"))
        return qc, qr

    qc, qr = duties(lambda T, x, ph: _h_dwsim(th, cp, T, P, x, ph))
    assert qc == pytest.approx(c["same"]["condenser_duty"], rel=1e-10)
    assert qr == pytest.approx(c["same"]["reboiler_duty"], rel=1e-10)
    qc2, _ = duties(lambda T, x, ph: float(th.h(T, P, jnp.asarray(x), ph)))
    assert 2e-5 < qc2 / c["same"]["condenser_duty"] - 1 < 1e-4


@pytest.mark.release
@pytest.mark.parametrize("name", DB_COLUMNS)
def test_dwsims_own_data_moves_the_split(name):
    """Comparison (b): the same column on DWSIM's database constants and
    kij. The model's sensitivity, reported in the docs; what is pinned here
    is its size, so a regeneration that changes it is noticed. Measured:
    debutanizer, isopentane in the distillate +2.1 points (3.2 % to 5.3 %)
    and the condenser duty +6.1 %; C3/C4 splitter, propylene in the
    distillate +1.3 points and the condenser duty -0.17 %."""
    c = REF["columns"][name]
    a, b = c["same"], c["dwsim_data"]
    names = c["components"]["names"]
    dz = np.asarray(b["distillate"]["z"]) - np.asarray(a["distillate"]["z"])
    key, shift, duty = {"debutanizer": ("isopentane", 0.0211, 0.0613),
                        "c3c4_splitter": ("propylene", 0.0134, -0.0017)}[name]
    assert dz[names.index(key)] == pytest.approx(shift, abs=1e-3), dict(zip(names, dz.round(5)))
    assert float(np.max(np.abs(dz))) == pytest.approx(shift, abs=1e-3)
    assert b["condenser_duty"] / a["condenser_duty"] - 1 == pytest.approx(duty, abs=2e-3)


# ---------------------------------------------------------------------------
# Release: compressor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compressed():
    c = REF["compressor"]
    unit, feed = dc.difflow_compressor(c["components"])
    return c, unit(feed)


@pytest.mark.release
class TestCompressor:

    def test_stage_power_and_discharge_temperature(self, compressed):
        """Measured: power 5.1e-5 (stage 1) and 4.9e-5 (stage 2) relative,
        discharge temperatures 2.7 and 2.8 mK -- R and the midpoint rule in
        DWSIM's enthalpy and entropy. Before the ``_solve_T`` fix difflow's
        first stage was 3.6 % low and 1.5 K cold."""
        c, (gas, cond, info) = compressed
        ref = c["same"]["stages"]
        for k, st in enumerate(ref):
            assert float(info["stage_power"][k]) == pytest.approx(st["power"], rel=2e-4), k
            assert float(info["discharge_T"][k]) == pytest.approx(st["discharge_T"], abs=0.02), k

    def test_knockout_split(self, compressed):
        """Gas and condensate leaving the train; measured 1.9e-6 relative
        worst (the n-hexane left in the gas) and 3.9e-7 in the condensate."""
        c, (gas, cond, info) = compressed
        names = c["components"]["names"]
        last = c["same"]["stages"][-1]
        g = np.array([float(gas[f"F_{n}"]) for n in names])
        np.testing.assert_allclose(g, [last["vapor"]["flows"]["DF_" + n] for n in names], rtol=2e-5)
        liq = sum(np.array([st["liquid"]["flows"]["DF_" + n] for n in names]) for st in c["same"]["stages"])
        liq = liq + np.array([c["same"]["inlet_drum"]["liquid"]["flows"]["DF_" + n] for n in names])
        np.testing.assert_allclose([float(cond[f"F_{n}"]) for n in names], liq, rtol=2e-5)
        assert float(gas["T"]) == pytest.approx(last["vapor"]["T"])
        assert float(gas["P"]) == pytest.approx(last["vapor"]["P"], rel=1e-12)

    def test_dwsims_own_data(self, compressed):
        """Comparison (b): DWSIM's compounds and kij. Measured: difflow's
        total power 0.40 % below DWSIM's, its condensate 0.86 % above -- the
        constants and kij (DWSIM's H2S/C1-C3 and C1/C4-C6 pairs), not the
        implementation."""
        c, (gas, cond, info) = compressed
        b = c["dwsim_data"]["stages"]
        assert float(info["power"]) / sum(s["power"] for s in b) - 1 == pytest.approx(-0.0040, abs=1e-3)
        n = sum(float(cond[f"F_{k}"]) for k in c["components"]["names"])
        assert n / sum(s["liquid"]["F"] for s in b) - 1 == pytest.approx(0.0086, abs=2e-3)


# ---------------------------------------------------------------------------
# Release: vapour pressure
# ---------------------------------------------------------------------------


@pytest.mark.release
@pytest.mark.parametrize("name", RVPS)
class TestVaporPressure:

    def _flows(self, name):
        c = REF["rvp"][name]
        return c, dc.gas_components_of(c["components"]), jnp.asarray(c["case"]["z"])

    def test_true_vapor_pressure(self, name):
        """Bubble pressure at 100 F: measured 8.6e-7 and 1.1e-6 relative
        (DWSIM's flash residual and its truncated sqrt(2))."""
        from difflow_refinery.gasplant.products import T_100F, true_vapor_pressure

        c, comps, z = self._flows(name)
        assert T_100F == pytest.approx(310.927, abs=1e-3)
        # DWSIM's 100 F is 310.95 K; the reference is at DWSIM's
        tvp = float(true_vapor_pressure(z, comps, T=310.95))
        assert tvp == pytest.approx(c["same"]["tvp"], rel=1e-5)

    def test_d323_construction(self, name):
        """difflow's D323 construction against the same construction on
        DWSIM's flashes and DWSIM's PR liquid root: measured 8.6e-7 and
        1.1e-6 relative."""
        from difflow_refinery.gasplant import CubicThermo
        from difflow_refinery.gasplant.thermo import PR, vapor_pressure_vl

        c, comps, z = self._flows(name)
        rvp = float(vapor_pressure_vl(CubicThermo(comps, PR), z, 310.95, 4.0))
        assert rvp == pytest.approx(c["same"]["d323"]["rvp"], rel=1e-5)

    def test_dwsims_rvp_is_a_correlation(self, name):
        """DWSIM's own RVP (its cold-flow utility) is a correlation on the
        TVP, not the D323 construction. Recomputed here from its TVP. It is
        33 % (debutanizer bottoms) and 18 % (stabilized naphtha) below
        difflow's construction, and it is not a vapour-pressure model: it
        returns the TVP itself only at 2.1 psi
        (:func:`test_the_correlation_crosses_the_tvp_once`), so it puts the
        RVP of a pure component, which must equal its vapour pressure, below
        it."""
        from .reference.dwsim_lightends import dwsim_rvp_from_tvp

        c, *_ = self._flows(name)
        r = c["same"]
        assert dwsim_rvp_from_tvp(r["tvp"]) == pytest.approx(r["rvp_dwsim_correlation"], rel=1e-14)
        gap = {"debutanizer_bottoms": -0.333, "stabilized_naphtha": -0.177}[name]
        assert r["rvp_dwsim_correlation"] / r["d323"]["rvp"] - 1 == pytest.approx(gap, abs=0.005)


@pytest.mark.release
def test_the_correlation_crosses_the_tvp_once():
    """``RVP(TVP) = TVP`` where ``ln t + 0.15279 = 2.7738 log10 t`` (t in
    psi): t = 2.11 psi. Above it the correlation returns less than the TVP."""
    from .reference.dwsim_lightends import dwsim_rvp_from_tvp

    psi = 6894.76
    t = np.exp((12.972789480266567 - 12.82) / (2.773803987779386 / np.log(10) - 1))
    assert t == pytest.approx(2.11, abs=0.01)
    assert dwsim_rvp_from_tvp(t * psi) == pytest.approx(t * psi, rel=1e-12)
    assert dwsim_rvp_from_tvp(10 * psi) < 10 * psi
