"""difflow_refinery's crude side against DWSIM 9.0.5.

DWSIM ran in the generator (``reference/dwsim_cdu_generate.py``) and wrote
``reference/dwsim_cdu_reference.json``; this file reads it and needs neither
DWSIM nor .NET. The cases are ``reference/dwsim_cdu_case.py``; the DWSIM
unit-operation helpers ``reference/dwsim_columns.py``.

Each comparison says what it tests:

* **Same model, same constants** (an implementation check): DWSIM's Raoult's
  Law package with the ideal options on difflow's constants, every component
  a ``FlatCompound`` (so DWSIM's latent heat is Watson's with difflow's
  exponent). Differences are reproduced, not tolerated: DWSIM's ideal-gas
  enthalpy is the midpoint-rule integral of Cp; its liquid enthalpy carries a
  ``P v_L`` term (``P/rho_L`` on its Rackett density); its Watson latent heat
  is not smoothed near ``Tc`` (difflow's is, so that the energy balance has a
  derivative; ``thermo._watson_base``).
* **DWSIM's own data and correlations** (a model check): DWSIM's
  distillation-curve characterization, its correlations, its PR /
  Grayson-Streed / Lee-Kesler-Plocker on its own petroleum fractions. Those
  differences are documented findings, pinned at their measured size so a
  change on either side shows, never tuned away.

``release``: a model change moves these. The per-commit checks (the file is
intact and still about difflow's constants and characterizations) are in
:class:`TestReferenceFile` and :class:`TestReferenceIsCurrent`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .reference import dwsim_cdu_case as dc

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent / "reference"
REF = json.loads((HERE / "dwsim_cdu_reference.json").read_text())
INP = REF["inputs"]
COMP = INP["components"]
NLE = COMP["n_light_ends"]

REGENERATE = ("difflow's constants, characterizations or the DWSIM cases no longer match the "
              "ones the DWSIM reference was built on; regenerate it: PYTHONPATH=src:tests python "
              "-m refinery.reference.dwsim_cdu_generate (needs DWSIM 9.0.5; see "
              "scripts/install_dwsim.sh), and re-read 'Validation against DWSIM: "
              "characterization and crude unit' in docs/unit-operations-refinery.md")

ASSAYS = ("test_crude", "heavy_crude")


# ---------------------------------------------------------------------------
# helpers: difflow's model, and DWSIM's three reproducible departures from it
# ---------------------------------------------------------------------------

def _thermo():
    from difflow_refinery.thermo import ColumnThermo

    a = {k: jnp.asarray(COMP[k], dtype=float) for k in ("MW", "SG", "Tb", "Tc", "Pc",
                                                        "omega_vp", "hvap_A", "cp_ig")}
    return ColumnThermo(names=tuple(COMP["names"]), MW=a["MW"], SG=a["SG"], Tb=a["Tb"],
                        Tc=a["Tc"], Pc=a["Pc"], omega_vp=a["omega_vp"], hvap_A=a["hvap_A"],
                        cp_ig=a["cp_ig"])


def _dwsim_h_ig(cp, T, T0=298.15):
    """DWSIM's ideal-gas enthalpy (J/mol): midpoint rule on Cp,
    ``round(|T-T0|/10)`` intervals clipped to [10, 100] (as in
    ``test_dwsim_smoke``)."""
    cp = np.asarray(cp)
    span = abs(T - T0)
    n = 2 if span < 1 else 4 if span < 3 else 6 if span < 5 else min(max(round(span / 10), 10), 100)
    d = (T - T0) / n
    t = T0 + d / 2 + d * np.arange(n)
    cpv = cp[:, 0:1] + cp[:, 1:2] * t + cp[:, 2:3] * t ** 2 + cp[:, 3:4] * t ** 3
    return cpv.sum(axis=1) * d


def _dwsim_hvap(T):
    """DWSIM's latent heat of a FlatCompound: ``A (1 - Tr)^0.38``, zero at
    and above Tc -- difflow's Watson form without its smoothing."""
    A, Tc = np.asarray(COMP["hvap_A"]), np.asarray(COMP["Tc"])
    x = 1.0 - T / Tc
    return np.where(x > 0, A * np.abs(x) ** 0.38, 0.0)


def _dwsim_h(point):
    """DWSIM's mixture enthalpy of a flash (J/mol), rebuilt from difflow's
    constants and DWSIM's three conventions."""
    T, P, vf = point["T"], point["P"], point["vapor_fraction"]
    hig = _dwsim_h_ig(COMP["cp_ig"], T)
    hv = float(np.dot(point["y"], hig)) if point["y"] is not None else 0.0
    hl = 0.0
    if point["x"] is not None:
        pv = P / 1000.0 / point["rho_liq"] * point["MW_liq"]          # kJ/kg * g/mol = J/mol
        hl = float(np.dot(point["x"], hig - _dwsim_hvap(T))) + pv
    return vf * hv + (1.0 - vf) * hl


def _rr(th, z, T, P):
    """difflow's Raoult flash: (vapour fraction, x, y)."""
    K = np.asarray(th.K(T, P))
    z = np.asarray(z) / np.sum(z)

    def g(b):
        return np.sum(z * (K - 1) / (1 + b * (K - 1)))

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
    b = 0.5 * (lo + hi)
    x = z / (1 + b * (K - 1))
    return b, x, K * x


def _saturation_T(th, z, P, kind):
    """difflow's bubble (``sum z K = 1``) or dew (``sum z/K = 1``) temperature."""
    z = np.asarray(z) / np.sum(z)

    def f(T):
        K = np.asarray(th.K(T, P))
        return math.log(np.sum(z * K)) if kind == "bubble" else -math.log(np.sum(z / K))

    lo, hi = 200.0, 1200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
    return 0.5 * (lo + hi)


def _difflow_h(th, z, T, P):
    """difflow's ColumnThermo enthalpy of the flashed mixture (J/mol)."""
    b, x, y = _rr(th, z, T, P)
    return b * float(np.dot(y, th.h_vapor(T))) + (1 - b) * float(np.dot(x, th.h_liquid(T))), b


# ---------------------------------------------------------------------------
# Per commit: the file, and whether it is still about difflow's numbers
# ---------------------------------------------------------------------------


class TestReferenceFile:
    def test_it_records_where_it_came_from(self):
        p = REF["provenance"]
        for key in ("generated", "script", "command", "dwsim", "dotnet", "pythonnet",
                    "reference_simulator", "difflow_commit", "sections_generated"):
            assert p[key], key
        assert "9.0.5" in p["dwsim"]
        for f in ("dwsim_cdu_generate.py", "dwsim_cdu_case.py", "dwsim_columns.py"):
            assert (HERE / f).exists(), f

    def test_every_section_is_there(self):
        for s in ("characterization", "thermo", "vacuum", "column"):
            assert s in REF, s

    def test_the_same_model_ran_on_difflows_constants(self):
        d = REF["thermo"]["same_model"]["dwsim"]
        assert d["property_package"] == "Raoult's Law"
        assert d["options"]["LiquidFugacity_UsePoyntingCorrectionFactor"] == "False"
        assert d["options"]["UseHenryConstants"] == "False"
        assert d["flash_settings"]["PVFlash_TryIdealCalcOnFailure"] == "False"
        for i, c in enumerate(d["constants"]):
            assert not c["is_hypo"] and not c["is_pf"], c["name"]
            for key in ("MW", "Tc", "Pc"):
                assert c[key] == pytest.approx(COMP[key][i], rel=1e-14), (c["name"], key)
            assert c["omega"] == pytest.approx(COMP["omega_vp"][i], rel=1e-14)


class TestReferenceIsCurrent:
    """Every difflow number the comparison is about is an input frozen into
    the file; if difflow's have moved, regenerate -- never loosen."""

    def test_the_cases_are_the_ones_dwsim_ran(self):
        assert INP["test_crude"] == json.loads(json.dumps(dc.TEST_CRUDE)), REGENERATE
        assert INP["heavy_crude"] == json.loads(json.dumps(dc.HEAVY_CRUDE)), REGENERATE
        assert INP["column_A"] == json.loads(json.dumps(dc.COLUMN_A)), REGENERATE
        assert INP["char_runs"] == json.loads(json.dumps(dc.DWSIM_CHAR_RUNS)), REGENERATE
        assert INP["heat_path"] == json.loads(json.dumps(dc.HEAT_PATH)), REGENERATE
        assert INP["vacuum_flash"] == json.loads(json.dumps(dc.VACUUM_FLASH)), REGENERATE

    def test_the_crude_constants_are_difflows(self):
        from .reference import case

        unit, crude, thermo, Vf, flows = dc.difflow_crude()
        now = case.component_data(crude, thermo)
        assert now["names"] == COMP["names"], REGENERATE
        for key in ("MW", "SG", "Tb", "Tc", "Pc", "omega_vp", "omega_eos", "cp_ig", "hvap_nb"):
            np.testing.assert_allclose(now[key], COMP[key], rtol=1e-12, err_msg=f"{key}: {REGENERATE}")
        np.testing.assert_allclose(np.asarray(thermo.hvap_A), COMP["hvap_A"], rtol=1e-12,
                                   err_msg=REGENERATE)
        np.testing.assert_allclose(flows, INP["feed_flows"], rtol=1e-12, err_msg=REGENERATE)

    def test_the_characterizations_are_difflows(self):
        now = dc.difflow_characterizations()
        for key in ASSAYS:
            for m, tab in now[key].items():
                ref = INP["difflow_characterization"][key][m]
                for k in ("cut_edges", "Tb", "SG", "MW", "Tc", "Pc", "omega", "cp_ig",
                          "volume_fraction"):
                    np.testing.assert_allclose(tab[k], ref[k], rtol=1e-12,
                                               err_msg=f"{key} {m} {k}: {REGENERATE}")

    def test_the_vacuum_feed_is_the_cdu_references_residue(self):
        cdu = json.loads((HERE / "cdu_reference.json").read_text())
        np.testing.assert_allclose(REF["vacuum"]["feed_flows"],
                                   cdu["layer3"]["column"]["product_flows"]["residue"], rtol=0)


# ---------------------------------------------------------------------------
# Release: 1. characterization
# ---------------------------------------------------------------------------

def _runs(key, run):
    """DWSIM's rows and cut bookkeeping for difflow's cuts (the test crude's
    first DWSIM cut is the light-ends region below difflow's first edge), and
    difflow's table; ``n`` cuts are common to both (the heavy crude's last
    DWSIM cut stops at the last data point, difflow's does not)."""
    C = REF["characterization"][key]
    off = 1 if key == "test_crude" else 0
    rows, cuts = C["runs"][run]["rows"][off:], C["runs"][run]["cuts"][off:]
    tab = INP["difflow_characterization"][key]["twu" if run == "default" else "riazi_daubert_1987"]
    n = len(rows) - (1 if key == "heavy_crude" else 0)
    return rows, cuts, tab, n


def _col(rows, k, n=None):
    return np.array([r[k] for r in rows[:n]])


@pytest.mark.release
@pytest.mark.parametrize("key", ASSAYS)
class TestCorrelationsAtTheSameCut:
    """DWSIM's correlations evaluated at difflow's own (Tb, SG): the
    correlations alone, the pipeline out of the way."""

    def _at(self, key):
        from difflow_refinery import correlations as corr

        pm = REF["characterization"][key]["property_methods_at_difflow_cuts"]
        tab = INP["difflow_characterization"][key]["twu"]
        m = len(pm["MW_Winn"])
        Tb, SG = np.array(tab["Tb"][:m]), np.array(tab["SG"][:m])
        return pm, tab, Tb, SG, corr

    def test_riazi_daubert_1985_is_difflows_riazi_daubert_1987(self, key):
        """The same equations (API TDB extended form); measured 3e-16."""
        pm, _, Tb, SG, corr = self._at(key)
        _, Tc, Pc = (np.asarray(v) for v in corr.critical_properties(Tb, SG, "riazi_daubert_1987"))
        np.testing.assert_allclose(pm["Tc_RiaziDaubert"], Tc, rtol=1e-13)
        np.testing.assert_allclose(pm["Pc_RiaziDaubert"], Pc, rtol=1e-13)

    def test_riazi_1986_mw_truncates_one_coefficient(self, key):
        """DWSIM's "Riazi (1986)" MW is the extended Riazi-Daubert MW with
        ``-7.78 SG`` for ``-7.78712 SG`` in the exponent: ``exp(0.00712 SG)``
        high, 0.50-0.76 % here. Reproduced to 1e-15."""
        pm, _, Tb, SG, corr = self._at(key)
        MW, _, _ = (np.asarray(v) for v in corr.critical_properties(Tb, SG, "riazi_daubert_1987"))
        ratio = np.asarray(pm["MW_Riazi"]) / MW
        np.testing.assert_allclose(ratio, np.exp(0.00712 * SG), rtol=1e-13)
        assert 0.0049 < ratio.min() - 1 and ratio.max() - 1 < 0.0077

    def test_lee_kesler_tc_is_the_same_correlation(self, key):
        """DWSIM writes Kesler-Lee's Tc in kelvin with rounded constants:
        within 0.012 K of difflow's Rankine form (measured 0.0108 / 0.0116 K)."""
        pm, _, Tb, SG, corr = self._at(key)
        _, Tc, _ = (np.asarray(v) for v in corr.critical_properties(Tb, SG, "lee_kesler"))
        assert np.max(np.abs(np.asarray(pm["Tc_LeeKesler"]) - Tc)) < 0.015

    def test_lee_kesler_pc_is_ten_times_too_high(self, key):
        """A DWSIM bug: its "Lee-Kesler (1976)" Pc multiplies a pressure in
        bar by ``1e6 * 0.986923`` where ``1e5`` converts it to Pa -- 9.869x
        Kesler-Lee's Pc (the remaining 1e-3 is its rounded kelvin constants)."""
        pm, _, Tb, SG, corr = self._at(key)
        _, _, Pc = (np.asarray(v) for v in corr.critical_properties(Tb, SG, "lee_kesler"))
        ratio = np.asarray(pm["Pc_LeeKesler"]) / Pc
        np.testing.assert_allclose(ratio, 9.86923, rtol=1.5e-3)

    def test_lee_kesler_mw_has_a_sign_error(self, key):
        """A DWSIM bug: in "Lee-Kesler (1974)" MW the last term's
        ``(1 - 0.80882 SG + 0.02226 SG^2)`` is written with ``- 0.02226``.
        With that sign flipped difflow's Kesler-Lee MW reproduces DWSIM's
        within 0.6 g/mol (its rounded kelvin constants); as published it is
        13-180 g/mol away -- a negative MW for the lightest cuts."""
        pm, _, Tb, SG, corr = self._at(key)
        R = 1.8 * Tb

        def kl(sign):
            return (-12272.6 + 9486.4 * SG + (4.6523 - 3.3287 * SG) * R
                    + (1.0 - 0.77084 * SG - 0.02058 * SG ** 2) * (1.3437 - 720.79 / R) * 1e7 / R
                    + (1.0 - 0.80882 * SG + sign * 0.02226 * SG ** 2) * (1.8828 - 181.98 / R)
                    * 1e12 / R ** 3)

        MW, _, _ = (np.asarray(v) for v in corr.critical_properties(Tb, SG, "lee_kesler"))
        np.testing.assert_allclose(kl(1.0), MW, rtol=1e-12)
        dw = np.asarray(pm["MW_LeeKesler"])
        assert np.max(np.abs(dw - kl(-1.0))) < 0.6
        assert np.min(np.abs(dw - MW)) > 10.0

    def test_lee_kesler_omega_is_difflows_below_tbr_0_78(self, key):
        """Both use Lee-Kesler's vapour-pressure inversion; difflow switches to
        Kesler-Lee's (Kw, Tbr) form above Tbr = 0.8 (blended), DWSIM never
        does. Measured 2e-5 below Tbr 0.78 (the blend's tail); up to 0.22
        above."""
        pm, _, Tb, SG, corr = self._at(key)
        _, Tc, Pc = (np.asarray(v) for v in corr.critical_properties(Tb, SG, "riazi_daubert_1987"))
        w = np.asarray(corr.acentric_factor(Tb, Tc, Pc, SG))
        lo = Tb / Tc < 0.78
        assert lo.sum() >= 10
        assert np.max(np.abs(np.asarray(pm["omega_LeeKesler_RD"])[lo] - w[lo])) < 5e-5
        assert np.max(np.abs(np.asarray(pm["omega_LeeKesler_RD"]) - w)) < 0.25

    def test_winn_mw_against_twu(self, key):
        """DWSIM's default MW (Winn 1956) against difflow's default (Twu
        1984), same (Tb, SG): a model difference, -21 % to +12 % across the
        cuts (Winn high in the middle distillates, low for the heavy end)."""
        pm, tab, *_ = self._at(key)
        r = np.asarray(pm["MW_Winn"]) / np.asarray(tab["MW"][:len(pm["MW_Winn"])]) - 1
        assert -0.21 < r.min() < -0.18 and 0.11 < r.max() < 0.12


@pytest.mark.release
@pytest.mark.parametrize("key", ASSAYS)
@pytest.mark.parametrize("run", ["default", "rd"])
class TestDWSIMsCharacterizationPipeline:
    """DWSIM's characterization run on difflow's cut temperatures,
    reproduced step by step: each difference from difflow's is traced to one
    step, and each step is checked to be what it is."""

    def test_cut_boiling_points_are_its_polynomial_at_the_midpoint(self, key, run):
        """DWSIM fits a 6th-order polynomial T(x) to the TBP curve and gives a
        cut the temperature at its MIDPOINT fraction; difflow interpolates
        monotonically and gives the MEAN temperature over the cut. The
        polynomial is up to 17 K (test crude) / 23 K (heavy crude) off the
        curve at a cut's midpoint; mean against midpoint is up to 10 / 17 K
        on difflow's own curve; net, cut Tb differ by up to 7.2 / 6.5 K."""
        from difflow_refinery.assay import tbp_curve

        rows, cuts, tab, n = _runs(key, run)
        tbpm = np.array([c["tbpm"] for c in cuts])
        np.testing.assert_allclose(_col(rows, "Tb"), tbpm, rtol=1e-14)
        spec = dc.TEST_CRUDE if key == "test_crude" else dc.HEAVY_CRUDE
        curve = tbp_curve(dc.assay_of(spec))
        mid = np.asarray(curve.invert(jnp.asarray([c["fvm"] for c in cuts[:n]])))
        fit_err = np.max(np.abs(tbpm[:n] - mid))
        mean_mid = np.max(np.abs(np.asarray(tab["Tb"][:n]) - mid))
        net = np.max(np.abs(tbpm[:n] - np.asarray(tab["Tb"][:n])))
        limits = {"test_crude": (18.0, 11.0, 7.5), "heavy_crude": (24.0, 17.0, 7.0)}[key]
        assert fit_err < limits[0] and mean_mid < limits[1] and net < limits[2]

    def test_mw_is_computed_before_the_gravities_are_rescaled(self, key, run):
        """DWSIM estimates each cut's SG from its Tb (Riazi-Al Sahhaf's
        ``d15(MW)`` on a Tb-only MW guess), computes the MW from THAT SG, and
        only then rescales every SG by one factor to the bulk gravity
        (mass-weighted). Its MW is reproduced to round-off from the
        unscaled SG, which is 0.3 % (test crude) / 9 % (heavy crude) below
        the SG the cut ends up with."""
        rows, cuts, _, _ = _runs(key, run)
        tbpm = np.array([c["tbpm"] for c in cuts])
        sgu = np.array([c["sg_unscaled"] for c in cuts])
        if run == "default":       # Winn (1956)
            mw = 0.00005805 * tbpm ** 2.3776 / sgu ** 0.9371
        else:                      # Riazi (1986), DWSIM's coefficients
            mw = 42.965 * np.exp(0.0002097 * tbpm - 7.78 * sgu + 0.00208476 * tbpm * sgu) \
                * tbpm ** 1.26007 * sgu ** 4.98308
        np.testing.assert_allclose(_col(rows, "MW"), mw, rtol=1e-13)
        scale = _col(rows, "SG") / sgu
        assert np.ptp(scale) < 1e-12
        expect = {"test_crude": 1.00323, "heavy_crude": 1.09412}[key]
        assert scale[0] == pytest.approx(expect, abs=1e-5)

    def test_critical_properties_are_riazi_daubert_at_its_tb_and_sg(self, key, run):
        """Tc and Pc are the Riazi-Daubert 1985 = difflow 1987 equations at
        DWSIM's (Tb, rescaled SG), to round-off."""
        from difflow_refinery import correlations as corr

        rows, _, _, _ = _runs(key, run)
        _, Tc, Pc = (np.asarray(v) for v in
                     corr.critical_properties(_col(rows, "Tb"), _col(rows, "SG"), "riazi_daubert_1987"))
        np.testing.assert_allclose(_col(rows, "Tc"), Tc, rtol=1e-13)
        np.testing.assert_allclose(_col(rows, "Pc"), Pc, rtol=1e-13)

    def test_the_net_differences_from_difflow(self, key, run):
        """What a user sees, cut by cut, DWSIM against difflow (``default``:
        DWSIM's defaults against difflow's Twu; ``rd``: DWSIM's Riazi-Daubert
        options against difflow's ``riazi_daubert_1987``). Pinned at the
        measured relative ranges; the docs table has them per cut. The SG
        shapes differ (DWSIM's d15(MW) curve scaled by mass, difflow's
        constant Watson K by volume); most of the heavy crude's SG difference
        is DWSIM normalising its 1-60 wt % cuts to the WHOLE crude's gravity
        (the 40 wt % residue that would carry the rest is not in them)."""
        rows, _, tab, n = _runs(key, run)
        ranges = {   # measured (min, max) of DWSIM / difflow - 1 over the common cuts
            ("test_crude", "default"): {"SG": (-0.0323, -0.0005), "MW": (-0.1456, 0.1280),
                                        "Tc": (-0.0123, 0.0019), "Pc": (-0.1694, -0.0015),
                                        "omega": (-0.1271, 0.3172)},
            ("test_crude", "rd"): {"SG": (-0.0323, -0.0005), "MW": (0.0045, 0.0612),
                                   "Tc": (-0.0189, 0.0), "Pc": (-0.1682, -0.0005),
                                   "omega": (-0.0393, 0.3669)},
            ("heavy_crude", "default"): {"SG": (0.0549, 0.0812), "MW": (0.0155, 0.1419),
                                         "Tc": (0.0153, 0.0361), "Pc": (0.1045, 0.1815),
                                         "omega": (-0.2677, 0.0851)},
            ("heavy_crude", "rd"): {"SG": (0.0549, 0.0812), "MW": (0.0051, 0.0421),
                                    "Tc": (0.0176, 0.0322), "Pc": (0.0850, 0.2381),
                                    "omega": (-0.1944, -0.0559)},
        }[(key, run)]
        for k, (lo, hi) in ranges.items():
            r = _col(rows, k, n) / np.asarray(tab[k][:n]) - 1
            assert lo - 2e-3 < r.min() and r.max() < hi + 2e-3, (k, r.min(), r.max())

    def test_dwsims_fractions_cover_only_the_curve(self, key, run):
        """The cut fractions are DWSIM's polynomial differences: within 0.011
        (volume fraction, test crude) / 0.004 (mass, heavy crude) of
        difflow's. DWSIM's cuts span the curve's first to last point only:
        the heavy crude's 40 wt % above 565 C is in no pseudo-component (it
        would have to be added by hand); difflow's HeavyEnd extends the curve
        to 800 C and lumps the rest."""
        rows, cuts, tab, n = _runs(key, run)
        frac = np.array([c["fvf"] - c["fv0"] for c in cuts])
        basis = "volume_fraction" if key == "test_crude" else "mass_fraction"
        df = np.asarray(tab[basis])[len(tab["light_names"]):]
        assert np.max(np.abs(frac[:n] - df[:n])) < (0.011 if key == "test_crude" else 0.004)
        if key == "heavy_crude":
            assert cuts[-1]["fvf"] == pytest.approx(0.60, abs=2e-3)   # measured 0.5990
            assert tab["residue_lump"] and tab["mass_fraction"][-1] > 0.15

    def test_ideal_gas_cp(self, key, run):
        """DWSIM's petroleum-fraction Cp is Lee-Kesler's (Watson K, omega);
        difflow's is Watson-Nelson's. On each side's own cut: -34 % to +20 % at 300 K, -17 % to +12 % at 600 K (pinned); the large ends are
        where the two characterizations' MW and omega differ most."""
        rows, _, tab, n = _runs(key, run)
        r3 = _col(rows, "cp_ig_300", n) / np.array([c[0] for c in tab["cp_ig"][:n]]) - 1
        r6 = _col(rows, "cp_ig_600", n) / np.array([c[1] for c in tab["cp_ig"][:n]]) - 1
        assert -0.35 < r3.min() and r3.max() < 0.20
        assert -0.17 < r6.min() and r6.max() < 0.12


# ---------------------------------------------------------------------------
# Release: 2. thermodynamics on the crude
# ---------------------------------------------------------------------------

SAME = REF["thermo"]["same_model"]
OWN = REF["thermo"]["dwsim_own"]


@pytest.mark.release
class TestSameModelOnTheCrude:
    """difflow's ColumnThermo against DWSIM's Raoult's Law on difflow's
    constants (an implementation check), on the CDU case's whole crude."""

    def test_pure_component_functions(self):
        """Psat (Lee-Kesler in DIPPR-101 form) to 4e-14, ideal-gas Cp exact;
        latent heat exactly ``A (1-Tr)^0.38`` (zero at and above Tc). Against
        difflow's smoothed Watson it differs by up to 0.25 % more than 30 K
        below Tc (``eps^2/4x`` in ``1 - Tr``)."""
        th = _thermo()
        worst = 0.0
        for i, n in enumerate(COMP["names"]):
            for T, v in SAME["pure"][n].items():
                T = float(T)
                assert v["psat"] == pytest.approx(float(th.psat(T)[i]), rel=1e-12), (n, T)
                a, b, c, d = COMP["cp_ig"][i]
                assert v["cp_ig"] == pytest.approx(a + b * T + c * T * T + d * T ** 3, rel=1e-13)
                assert v["hvap"] == pytest.approx(_dwsim_hvap(T)[i], rel=1e-12, abs=1e-9), (n, T)
                if T < COMP["Tc"][i] - 30.0:
                    worst = max(worst, abs(v["hvap"] / float(th.dhvap(T)[i]) - 1))
        assert 1e-3 < worst < 3e-3

    @pytest.mark.parametrize("i", [0, 1])
    def test_bubble_points(self, i):
        """351.06 K at 1 bar, 383.03 K at 2 bar; agree to 3e-13 K."""
        p = SAME["bubble"][i]
        assert _saturation_T(_thermo(), INP["feed_flows"], p["P"], "bubble") == pytest.approx(
            p["T"], abs=1e-9)

    @pytest.mark.parametrize("i", [0, 1])
    def test_dew_points_dwsim_stops_short(self, i):
        """820.75 K at 1 bar, 848.17 K at 2 bar from DWSIM: 0.126 K above and
        0.055 K below difflow's root of ``sum z/K = 1`` (820.63 / 848.22 K). The difference is
        DWSIM's: at its temperature ``sum z/K - 1`` is -3.3e-3 / +1.3e-3 (its
        PV flash stops at the loop tolerance in T, and the heaviest cuts'
        K of 1e-3 make the dew equation stiff); difflow's root is exact to
        bisection."""
        th = _thermo()
        p = SAME["dew"][i]
        T = _saturation_T(th, INP["feed_flows"], p["P"], "dew")
        assert abs(T - p["T"]) < 0.15
        z = np.asarray(INP["feed_flows"]) / np.sum(INP["feed_flows"])
        assert abs(np.sum(z / np.asarray(th.K(T, p["P"]))) - 1) < 1e-12
        resid = np.sum(z / np.asarray(th.K(p["T"], p["P"]))) - 1
        assert 1e-3 < abs(resid) < 4e-3

    def test_flash_zone(self):
        """The crude at the CDU reference's coil outlet (586.30 K) and flash
        zone pressure: 69.62 % vaporized, to 6e-13; phase compositions 7e-14."""
        fz = SAME["flash_zone"]
        b, x, y = _rr(_thermo(), INP["feed_flows"], fz["T"], fz["P"])
        assert b == pytest.approx(fz["vapor_fraction"], abs=1e-11)
        np.testing.assert_allclose(x, fz["x"], atol=1e-12)
        np.testing.assert_allclose(y, fz["y"], atol=1e-12)
        assert fz["vapor_fraction"] == pytest.approx(0.69623, abs=1e-5)

    def test_enthalpy_along_the_heating_path(self):
        """DWSIM's enthalpy at five points from the furnace inlet (240 C,
        6 bar) to the coil outlet is reproduced to 1e-4 J/mol from difflow's
        constants and DWSIM's three conventions (midpoint-rule ideal-gas
        enthalpy, unsmoothed Watson, ``P v_L`` in the liquid); the stream's
        number and ``DW_CalcEnthalpy`` agree. The vapour fractions agree to
        2e-14."""
        th = _thermo()
        for p in SAME["path"]:
            assert _dwsim_h(p) == pytest.approx(p["h"], abs=1e-4), p["T"]
            assert p["h_direct"] == pytest.approx(p["h"], abs=1e-4)
            b, _, _ = _rr(th, INP["feed_flows"], p["T"], p["P"])
            assert b == pytest.approx(p["vapor_fraction"], abs=1e-11)

    def test_furnace_duty_and_why_it_differs(self):
        """Furnace inlet to coil outlet for the 742 mol/s crude: difflow 42.405
        MW, DWSIM 42.268 MW (0.32 % lower). All of the 136.48 kW is the three
        conventions, computed term by term: DWSIM's ``P v_L`` (97.37 kW --
        156 J/mol of liquid at the 6 bar inlet, 24 J/mol at 1.86 bar), the
        Watson smoothing difflow keeps for a derivative (39.31 kW) and DWSIM's
        midpoint-rule quadrature (-0.20 kW). Nothing unexplained (residual
        < 1 W)."""
        th = _thermo()
        z = np.asarray(INP["feed_flows"])
        F = z.sum()
        p0, p1 = SAME["path"][0], SAME["path"][-1]
        h0, _ = _difflow_h(th, z, p0["T"], p0["P"])
        h1, _ = _difflow_h(th, z, p1["T"], p1["P"])
        duty_df = F * (h1 - h0)
        assert duty_df == pytest.approx(42.4048e6, rel=1e-5)
        assert SAME["furnace_duty_W"] == pytest.approx(42.2683e6, rel=1e-5)
        parts = {}
        for key in ("quad", "watson", "pv"):
            v = []
            for p in (p0, p1):
                T, P, vf = p["T"], p["P"], p["vapor_fraction"]
                x, y = np.asarray(p["x"]), np.asarray(p["y"])
                d_ig = np.asarray(th.h_vapor(T)) - _dwsim_h_ig(COMP["cp_ig"], T)
                v.append({"quad": vf * y @ d_ig + (1 - vf) * x @ d_ig,
                          "watson": (1 - vf) * x @ (_dwsim_hvap(T) - np.asarray(th.dhvap(T))),
                          "pv": -(1 - vf) * P / 1000.0 / p["rho_liq"] * p["MW_liq"]}[key])
            parts[key] = F * (v[1] - v[0])
        assert parts["pv"] == pytest.approx(97.37e3, rel=1e-3)
        assert parts["watson"] == pytest.approx(39.31e3, rel=1e-3)
        assert parts["quad"] == pytest.approx(-0.205e3, rel=1e-2)
        assert duty_df - SAME["furnace_duty_W"] == pytest.approx(sum(parts.values()), abs=1.0)


@pytest.mark.release
class TestDWSIMsOwnModelsOnTheCrude:
    """DWSIM's own characterization of the same crude (default options, its
    cuts plus database propane/n-butane/n-pentane, the same standard volume
    split and mass flow) on DWSIM's PR, Grayson-Streed, Lee-Kesler-Plocker
    and Raoult's Law: a model check. Pinned findings, not agreement."""

    PKG = OWN["packages"]

    def test_the_feed_is_the_same_crude(self):
        f = OWN["feed"]
        mass = np.dot(INP["feed_flows"], COMP["MW"])
        assert f["mass_flow_g_s"] == pytest.approx(mass, rel=1e-12)
        assert f["names"][:3] == ["Propane", "N-butane", "N-pentane"]
        assert sum(f["volume_fraction"]) == pytest.approx(1.0, abs=1e-12)

    def test_bubble_points(self):
        """PR on DWSIM's fractions: 356.03 / 393.25 K at 1 / 2 bar, 5.0 / 10.2 K
        above difflow's Raoult (351.06 / 383.03 K); Grayson-Streed 359.32 /
        397.92 K. DWSIM's Lee-Kesler-Plocker returns DWSIM's Raoult bubble
        point (348.60 / 371.68 K) to every digit -- with
        ``PVFlash_TryIdealCalcOnFailure`` off -- unexplained: its TP flashes
        differ from Raoult's (below)."""
        exp = {"PR": (356.030, 393.246), "GS": (359.318, 397.924),
               "LKP": (348.600, 371.685), "RAOULT": (348.600, 371.685)}
        for pkg, (b1, b2) in exp.items():
            got = [q["T"] for q in self.PKG[pkg]["bubble"]]
            assert got == pytest.approx([b1, b2], abs=2e-3), pkg
        assert [q["T"] for q in self.PKG["LKP"]["bubble"]] == [q["T"] for q in self.PKG["RAOULT"]["bubble"]]

    def test_no_dwsim_package_finds_the_crudes_dew_point(self):
        """Every DWSIM package's PV flash at vapour fraction one fails on its
        own characterization of the crude ("Unable to calculate PV Flash")
        at both pressures; difflow's (and DWSIM's own Raoult on difflow's
        constants, above) is 820.75 / 848.17 K."""
        for pkg, v in self.PKG.items():
            for q in v["dew"]:
                assert q["T"] is None and "Unable to calculate PV Flash" in q["error"], pkg

    def test_flash_zone_vaporization(self):
        """At the coil outlet and flash-zone pressure, against difflow's
        0.6962: PR 0.6960, Grayson-Streed 0.6853, Raoult (DWSIM's fractions)
        0.7017, Lee-Kesler-Plocker 0.7786."""
        exp = {"PR": 0.69596, "GS": 0.68528, "LKP": 0.77864, "RAOULT": 0.70170}
        for pkg, vf in exp.items():
            assert self.PKG[pkg]["flash_zone"]["vapor_fraction"] == pytest.approx(vf, abs=1e-4), pkg

    def test_furnace_duty(self):
        """``DW_CalcEnthalpy`` on each phase (what DWSIM's column uses),
        against difflow's 42.40 MW: PR 42.50 (+0.23 %), Grayson-Streed 42.30
        (-0.24 %), Lee-Kesler-Plocker 38.34 (-9.6 %), Raoult 27.14 (-36 %):
        DWSIM gives its petroleum fractions NO latent heat under Raoult's Law
        (``AUX_HVAPi`` has no branch for an "Petroleum Assay" compound and
        returns zero), so its Raoult liquid is ideal-gas enthalpy."""
        exp = {"PR": 42.5025e6, "GS": 42.3011e6, "LKP": 38.3360e6, "RAOULT": 27.1438e6}
        for pkg, q in exp.items():
            assert self.PKG[pkg]["furnace_duty_direct_W"] == pytest.approx(q, rel=1e-4), pkg

    def test_gs_and_lkp_stream_enthalpies_are_not_their_own_function(self):
        """A DWSIM inconsistency: under Grayson-Streed and Lee-Kesler-Plocker
        the enthalpy a material stream reports for these fractions is not
        what the package's ``DW_CalcEnthalpy`` returns for the same phases
        (vapour near -1 kJ/mol at 513 K): the stream-based duty is 15.06 /
        11.09 MW against 42.30 / 38.34 MW. PR's and Raoult's agree to 6 W."""
        for pkg in ("GS", "LKP"):
            v = self.PKG[pkg]
            assert v["furnace_duty_W"] < 0.4 * v["furnace_duty_direct_W"], pkg
        for pkg in ("PR", "RAOULT"):
            v = self.PKG[pkg]
            assert v["furnace_duty_W"] == pytest.approx(v["furnace_duty_direct_W"], abs=10.0), pkg


# ---------------------------------------------------------------------------
# Release: 4. the vacuum feed
# ---------------------------------------------------------------------------


@pytest.mark.release
class TestVacuumFeedFlash:
    """The CDU reference's atmospheric residue at vacuum flash-zone
    conditions (673 K / 50 and 100 mmHg, 693 K / 75 mmHg)."""

    def test_same_model(self):
        """Raoult on difflow's constants: vapour fractions 0.8349, 0.7526,
        0.8501 to 4e-12; liquid compositions 2e-12."""
        th = _thermo()
        z = REF["vacuum"]["feed_flows"]
        for p in REF["vacuum"]["same_model"]["points"]:
            b, x, _ = _rr(th, z, p["T"], p["P"])
            assert b == pytest.approx(p["vapor_fraction"], abs=1e-10)
            np.testing.assert_allclose(x, p["x"], atol=1e-10)

    def test_peng_robinson_on_the_same_constants(self):
        """PR (kij 0, EOS liquid) on the same Tc, Pc and EOS omega vaporizes
        4.6-4.7 points more of the residue than Raoult/Lee-Kesler at every
        point: a model difference at the conditions where the heaviest cuts'
        vapour pressures are an extrapolation in both."""
        th = _thermo()
        z = REF["vacuum"]["feed_flows"]
        for p in REF["vacuum"]["pr_same_constants"]["points"]:
            b, _, _ = _rr(th, z, p["T"], p["P"])
            assert 0.044 < p["vapor_fraction"] - b < 0.049, p["T"]
            assert p["equilibrium_residual"] < 1e-7
