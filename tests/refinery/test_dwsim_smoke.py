"""difflow's Peng-Robinson against DWSIM 9.0.5's, on the same constants.

The end-to-end check of the DWSIM harness (``reference/dwsim_session.py``):
DWSIM ran in the generator (``reference/dwsim_smoke_generate.py``) and wrote
``reference/dwsim_smoke_reference.json``; this file reads it. DWSIM, .NET and
pythonnet are not needed here.

Same model, same constants, both sides: PR with the 1976 kappa, ``kij = 0``,
every DWSIM component a hypothetical compound carrying difflow's MW, Tc, Pc,
omega and ideal-gas Cp cubic, the liquid density from the EOS without
Peneloux translation. It checks the implementation, not PR.

Four differences between the two implementations were found (from DWSIM's
IL), and each is reproduced and tested for what it is rather than absorbed
into a tolerance; with them reproduced the two agree to round-off:

* **R.** DWSIM's PR uses ``R = 8.314``, difflow ``8.314462618``. ``A`` and
  ``B`` (so Z, phi and K) do not depend on R; the departure enthalpy and the
  density are proportional to it: 5.6e-5 relative, up to 2 J/mol here.
* **sqrt(2).** DWSIM's fugacity routine (``ThermoPlugs.PR.CalcLnFugCPU``)
  writes ``1 + sqrt 2``, ``1 - sqrt 2`` and ``2 sqrt 2`` in the log term as
  2.414213, -0.414213 and 2.828426 (its cubic for Z, and its enthalpy, use
  the full values): 1.4e-6 in a liquid phi. :func:`_ln_phi_dwsim` is
  difflow's expression with those constants and reproduces DWSIM's phi to
  2e-12.
* **Ideal-gas enthalpy by quadrature.** DWSIM integrates Cp by the midpoint
  rule (:func:`_dwsim_h_ig`), not analytically: up to 0.23 J/mol (1e-5 of
  the ideal-gas enthalpy) on the naphtha at 410 K.
* **DWSIM's own convergence.** At loop tolerances of 1e-10 DWSIM's Nested
  Loops flash still ends up to 1.6e-5 from its own equilibrium condition
  (``equilibrium_residual`` in the file: ``max |ln(y/x) - ln(phi_L/phi_V)|``
  over DWSIM's own numbers, light ends at 280 K); its tolerance does not
  bound that. The flash comparison is held to ten times that residual.

``release``: a model change, not a typo, would move these. The per-commit
checks (the file is intact; the constants are still difflow's) are in
:class:`TestReferenceFile` and :class:`TestReferenceIsCurrent` and are not
release-marked.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .reference import dwsim_smoke_case as sc

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent / "reference"
REF = json.loads((HERE / "dwsim_smoke_reference.json").read_text())
CASES = sorted(sc.CASES)

#: DWSIM's gas constant in its PR package (J/mol/K).
R_DWSIM = 8.314

REGENERATE = ("difflow's constants (or the smoke cases) no longer match the ones the DWSIM "
              "reference was built on; regenerate it: PYTHONPATH=src:tests python -m "
              "refinery.reference.dwsim_smoke_generate (needs DWSIM 9.0.5; see "
              "scripts/install_dwsim.sh)")


def _thermo(case):
    return sc.difflow_thermo(REF["cases"][case]["components"])


def _ln_phi_dwsim(th, T, P, x, phase):
    """difflow's PR ``ln phi`` with DWSIM's log term: Z from the exact cubic,
    ``ln((Z + 2.414213 B)/(Z - 0.414213 B))/2.828426``."""
    m = th._mix(T, P, jnp.asarray(x))
    Z = th.Z(m, phase)
    A, B, am, bm = m["A"], m["B"], m["am"], m["bm"]
    bb = th.b_i / bm
    L = jnp.log((Z + 2.414213 * B) / (Z - 0.414213 * B)) / 2.828426
    return bb * (Z - 1.0) - jnp.log(Z - B) - (A / B * L) * (2.0 * m["sum_xa"] / am - bb)


def _dwsim_h_ig(cp, T, T0=298.15):
    """DWSIM's ideal-gas enthalpy of each component (J/mol): the midpoint
    rule on Cp, as ``PropertyPackage.AUX_INT_CPDTi`` does it -- ``n =
    round(|T - T0|/10)`` intervals clipped to [10, 100] (2, 4, 6 below 1, 3,
    5 K). Its error is ``-(T - T0) h^2 Cp''/24``."""
    span = abs(T - T0)
    n = 2 if span < 1 else 4 if span < 3 else 6 if span < 5 else min(max(round(span / 10), 10), 100)
    d = (T - T0) / n
    t = T0 + d / 2 + d * np.arange(n)
    cpv = cp[:, 0:1] + cp[:, 1:2] * t + cp[:, 2:3] * t ** 2 + cp[:, 3:4] * t ** 3
    return cpv.sum(axis=1) * d


def _points(case):
    return [r for r in REF["cases"][case]["dwsim"]["tp"] if r["K"] is not None]


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
        assert (HERE / "dwsim_smoke_generate.py").exists()

    @pytest.mark.parametrize("case", CASES)
    def test_dwsim_ran_the_like_for_like_model(self, case):
        d = REF["cases"][case]["dwsim"]["dwsim"]
        assert d["property_package"] == "Peng-Robinson (PR)"
        assert d["kij"] == "zero"
        assert not np.any(d["kij_matrix"])
        assert d["options"]["LiquidDensityCalculationMode_Subcritical"] == "EOS"
        assert d["options"]["LiquidDensity_UsePenelouxVolumeTranslation"] == "False"
        assert float(d["flash_settings"]["PTFlash_External_Loop_Tolerance"]) <= 1e-10
        # every component a hypo on difflow's numbers
        comp = REF["cases"][case]["components"]
        for i, c in enumerate(d["constants"]):
            assert c["is_hypo"]
            for key in ("MW", "Tc", "Pc", "omega"):
                assert c[key] == pytest.approx(comp[key][i], rel=1e-14), (c["name"], key)

    @pytest.mark.parametrize("case", CASES)
    def test_every_point_is_two_phase_and_flashed(self, case):
        pts = REF["cases"][case]["dwsim"]["tp"]
        assert len(pts) == len(sc.CASES[case]["tp"])
        for r in pts:
            assert r["phases"] == ["vapor", "liquid"], (r["T"], r["phases"])
            assert 0.05 < r["vapor_fraction"] < 0.95
            assert sum(r["x"]) == pytest.approx(1.0, abs=1e-12)
            assert sum(r["y"]) == pytest.approx(1.0, abs=1e-12)


class TestReferenceIsCurrent:
    """The constants are an input to both sides; if difflow's have moved, the
    comparison is against a different mixture -- regenerate, never loosen."""

    @pytest.mark.parametrize("case", CASES)
    def test_the_constants_are_the_ones_the_reference_used(self, case):
        ref = REF["cases"][case]["components"]
        now = sc.component_data(sc.CASES[case])
        assert now["names"] == ref["names"], REGENERATE
        for key in ("MW", "Tc", "Pc", "omega", "cp_ig"):
            np.testing.assert_allclose(now[key], ref[key], rtol=1e-12, err_msg=f"{key}: {REGENERATE}")

    @pytest.mark.parametrize("case", CASES)
    def test_the_case_is_the_one_the_reference_flashed(self, case):
        assert REF["cases"][case]["case"] == json.loads(json.dumps(sc.CASES[case])), REGENERATE


# ---------------------------------------------------------------------------
# Release: the comparison
# ---------------------------------------------------------------------------


@pytest.mark.release
@pytest.mark.parametrize("case", CASES)
class TestAgainstDWSIM:

    def test_ideal_gas_cp_is_difflows_cubic(self, case):
        """The hypo plumbing: DWSIM evaluates difflow's Cp cubic exactly."""
        c = REF["cases"][case]
        for i, n in enumerate(c["components"]["names"]):
            a, b, cc, d = c["components"]["cp_ig"][i]
            for T, v in c["dwsim"]["pure"][n].items():
                T = float(T)
                assert v["cp_ig"] == pytest.approx(a + b * T + cc * T * T + d * T ** 3,
                                                   rel=1e-12), (n, T)

    @pytest.mark.parametrize("phase", ["vapor", "liquid"])
    def test_fugacity_coefficients(self, case, phase):
        """At DWSIM's own phase compositions, no flash in the way. With
        DWSIM's log-term constants reproduced: 2.2e-12 worst (measured). With
        difflow's exact PR: 1.4e-6 (liquid), 7e-10 (vapour) -- all of it the
        truncated sqrt(2)."""
        th = _thermo(case)
        key, ckey = ("phi_vap", "y") if phase == "vapor" else ("phi_liq", "x")
        for r in _points(case):
            c, phi = r[ckey], np.asarray(r[key])
            emu = np.exp(np.asarray(_ln_phi_dwsim(th, r["T"], r["P"], c, phase)))
            exact = np.exp(np.asarray(th.ln_phi(r["T"], r["P"], jnp.asarray(c), phase)))
            np.testing.assert_allclose(emu, phi, rtol=1e-10, err_msg=f"T={r['T']}")
            np.testing.assert_allclose(exact, phi, rtol=3e-6, err_msg=f"T={r['T']}")

    def test_compressibility_factors(self, case):
        """Z of each phase at DWSIM's compositions; measured 1e-12 worst."""
        th = _thermo(case)
        for r in _points(case):
            for ph, comp, Z in (("vapor", r["y"], r["Z_vap"]), ("liquid", r["x"], r["Z_liq"])):
                m = th._mix(r["T"], r["P"], jnp.asarray(comp))
                assert float(th.Z(m, ph)) == pytest.approx(Z, rel=1e-10), (r["T"], ph)

    def test_dwsims_own_flash_residual(self, case):
        """How converged DWSIM's answer is, from DWSIM's numbers alone; this
        bounds every flash comparison below. Measured 1.6e-5 worst
        (light_ends at 280 K, n-pentane)."""
        for r in _points(case):
            assert r["equilibrium_residual"] < 5e-5, r["T"]

    def test_the_flash(self, case):
        """difflow's own TP flash of the same feed against DWSIM's.
        Measured: vapour fraction 6.5e-6, phase compositions 1.9e-6 (both
        at light_ends 280 K, where DWSIM's own residual is 1.6e-5); 3.4e-7
        and 1.9e-7 on the naphtha."""
        from difflow_refinery.gasplant.thermo import feed_state

        th = _thermo(case)
        z = jnp.asarray(sc.CASES[case]["z"])
        for r in _points(case):
            fs = feed_state(th, z, r["T"], r["P"])
            tol = 10 * max(r["equilibrium_residual"], 1e-7)
            assert float(fs["beta"]) == pytest.approx(r["vapor_fraction"], abs=tol), r["T"]
            np.testing.assert_allclose(np.asarray(fs["x"]), r["x"], atol=tol, err_msg=f"T={r['T']}")
            np.testing.assert_allclose(np.asarray(fs["y"]), r["y"], atol=tol, err_msg=f"T={r['T']}")

    def test_phase_enthalpies(self, case):
        """Same reference on both sides (ideal gas at 298.15 K, no heat of
        formation). Two known differences, each reproduced rather than
        tolerated: DWSIM's R in the departure, and DWSIM's ideal-gas
        enthalpy, which is the midpoint-rule integral of Cp
        (:func:`_dwsim_h_ig`), not the analytic one. With both, the phases
        agree to 2.3e-8 J/mol (measured). Plain, difflow is up to 2.1 J/mol
        off DWSIM: the R difference up to 2.0 J/mol, the quadrature up to
        0.23 J/mol (1e-5 of the ideal-gas enthalpy, naphtha at 410 K)."""
        th = _thermo(case)
        cp = np.asarray(REF["cases"][case]["components"]["cp_ig"])
        scale = R_DWSIM / 8.314462618
        for r in _points(case):
            T, P = r["T"], r["P"]
            for ph, comp, h in (("vapor", r["y"], r["h_vap"]), ("liquid", r["x"], r["h_liq"])):
                c = jnp.asarray(comp)
                h_ig = float(jnp.dot(c, th.h_ig(T)))
                h_ig_dw = float(np.dot(comp, _dwsim_h_ig(cp, T)))
                h_dep = float(th.h_departure(T, P, c, ph))
                assert h_ig_dw + scale * h_dep == pytest.approx(h, abs=1e-5), (T, ph)
                # and the unexplained part of the plain difference is that small too
                plain = float(th.h(T, P, c, ph)) - h
                assert plain - (h_ig - h_ig_dw) - (1 - scale) * h_dep == pytest.approx(0, abs=1e-5)
                assert abs(h_ig - h_ig_dw) < 3e-5 * max(abs(h_ig), 1.0) + 1e-3

    def test_phase_densities(self, case):
        """EOS densities (no volume translation): ``rho = P MW / (Z R T)``,
        so DWSIM's is difflow's times ``R / 8.314`` (5.6e-5). Rescaled,
        measured 1e-12 relative."""
        th = _thermo(case)
        mw = np.asarray(REF["cases"][case]["components"]["MW"])
        for r in _points(case):
            for ph, comp, rho in (("vapor", r["y"], r["rho_vap"]), ("liquid", r["x"], r["rho_liq"])):
                v = float(th.molar_volume(r["T"], r["P"], jnp.asarray(comp), ph))
                rho_df = float(np.dot(comp, mw)) / v / 1000.0
                assert rho_df * 8.314462618 / R_DWSIM == pytest.approx(rho, rel=1e-10), (r["T"], ph)
                assert rho_df / rho - 1 == pytest.approx(R_DWSIM / 8.314462618 - 1, abs=1e-10)

    def test_ph_flash_returns_to_its_temperature(self, case):
        """DWSIM's PH flash from a PT point's enthalpy lands back on its
        temperature: the harness's PH path works."""
        for p in REF["cases"][case]["dwsim"]["ph"]:
            assert p["result"]["T"] == pytest.approx(p["T_from"], abs=1e-6)
