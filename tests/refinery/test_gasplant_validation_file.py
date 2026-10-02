"""The gas plant's committed IDAES reference: cheap checks run on every commit (#312).

Split from ``test_gasplant_validation.py``, which is ``release`` throughout,
so that a deleted or truncated ``gasplant_reference.json``, or component
constants that have drifted from the ones the reference was built on, are
caught per commit and not first at release time.

Needs neither IDAES nor IPOPT: it reads the JSON and difflow's component
table.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

import jax.numpy as jnp

from difflow_refinery.gasplant import CubicThermo, gas_components
from difflow_refinery.gasplant.thermo import PR, feed_state

from .reference import gasplant_case as gc

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent / "reference"
REF = json.loads((HERE / "gasplant_reference.json").read_text())

COLUMN_CASES = sorted(n for n, c in gc.CASES.items() if c.get("reference", "column") == "column")
POINT_CASES = sorted(n for n, c in gc.CASES.items() if c.get("reference") == "state_points")

REGENERATE = ("difflow's gas plant component constants (or the cases) no longer match the ones "
              "the IDAES reference was built on; regenerate it: PYTHONPATH=src:tests python -m "
              "refinery.reference.gasplant_generate (needs idaes and ipopt), and re-read the gas "
              "plant Validation section of docs/unit-operations-refinery.md against the new numbers")


class TestReferenceFile:
    def test_it_records_where_it_came_from(self):
        p = REF["provenance"]
        for key in ("generated", "script", "command", "idaes", "pyomo", "ipopt",
                    "reference_simulator", "difflow_commit"):
            assert p[key], key
        assert "independently" in p["reference_simulator"]
        assert (HERE / "gasplant_generate.py").exists()

    @pytest.mark.parametrize("case", COLUMN_CASES)
    def test_every_case_is_there_and_solved(self, case):
        r = REF["cases"][case]["idaes"]
        assert r["solve"]["status"] == "optimal"
        n = gc.CASES[case]["n_trays"]
        assert len(r["stages"]) == n + 2
        assert all(s["K"] is not None for s in r["stages"][1:])

    @pytest.mark.parametrize("case", COLUMN_CASES)
    def test_the_reference_balances_its_own_mass(self, case):
        c = gc.CASES[case]
        r = REF["cases"][case]["idaes"]
        for i, n in enumerate(c["names"]):
            out = r["distillate"][n] + r["bottoms"][n]
            assert out == pytest.approx(c["F"] * c["z"][i], rel=1e-6, abs=1e-9), n

    @pytest.mark.parametrize("case", COLUMN_CASES)
    def test_the_reference_meets_its_specs(self, case):
        """Reflux and boilup ratios are what IDAES held fixed: its stage
        flows must reproduce them."""
        c = gc.CASES[case]
        r = REF["cases"][case]["idaes"]
        assert r["reflux_ratio"] == pytest.approx(c["reflux_ratio"], rel=1e-8)
        assert r["boilup_ratio"] == pytest.approx(c["boilup_ratio"], rel=1e-8)


    @pytest.mark.parametrize("case", POINT_CASES)
    def test_every_state_point_is_solved_and_two_phase(self, case):
        pts = REF["cases"][case]["idaes_points"]
        assert len(pts) == gc.CASES[case]["n_trays"] + 1          # trays and reboiler
        for p in pts:
            assert p["status"] == "optimal", p["stage"]
            assert 0.05 < p["vapor_fraction"] < 0.95, p["stage"]
            assert max(abs(k - 1) for k in p["K"]) > 0.05, p["stage"]   # not a trivial x == y


@pytest.mark.parametrize("case", POINT_CASES)
class TestStatePoints:
    """difflow's PR against IDAES's at IDAES's own flash solutions, on the
    stage states of a column IDAES cannot converge. No column on either
    side, so this is cheap enough for every commit. Measured on
    ``c3c4_splitter`` (21 points, 317-352 K, 17 bar, vapor fraction
    0.41-0.75): K 7.1e-7 relative, vapor fraction 3.3e-5, phase compositions
    1.1e-6, molar enthalpies 2e-9 J/mol."""

    def _thermo(self, case):
        return CubicThermo(gas_components(gc.CASES[case]["names"]), PR)

    def test_k_values(self, case):
        th = self._thermo(case)
        for p in REF["cases"][case]["idaes_points"]:
            lnK = th.log_K(p["T"], p["P"], jnp.asarray(p["x"]), jnp.asarray(p["y"]))
            np.testing.assert_allclose(np.exp(np.asarray(lnK)), p["K"], rtol=1e-5,
                                       err_msg=p["stage"])

    def test_the_flash(self, case):
        """difflow's own TP flash of the same z lands on IDAES's split."""
        th = self._thermo(case)
        for p in REF["cases"][case]["idaes_points"]:
            fs = feed_state(th, jnp.asarray(p["z"]), p["T"], p["P"])
            assert float(fs["beta"]) == pytest.approx(p["vapor_fraction"], abs=1e-4), p["stage"]
            np.testing.assert_allclose(np.asarray(fs["x"]), p["x"], atol=1e-5, err_msg=p["stage"])
            np.testing.assert_allclose(np.asarray(fs["y"]), p["y"], atol=1e-5, err_msg=p["stage"])

    def test_enthalpies(self, case):
        """Same reference state on both sides (ideal gas, 298.15 K, zero), so
        the absolute values compare; 1e-3 J/mol against values of order
        1e4."""
        th = self._thermo(case)
        for p in REF["cases"][case]["idaes_points"]:
            T, P = p["T"], p["P"]
            assert float(th.h_liquid(T, P, jnp.asarray(p["x"]))) == pytest.approx(p["h_liq"], abs=1e-3)
            assert float(th.h_vapor(T, P, jnp.asarray(p["y"]))) == pytest.approx(p["h_vap"], abs=1e-3)


class TestReferenceIsCurrent:
    """The reference was built on difflow's constants (an input to both
    sides); if they have moved, the comparison is against a different
    mixture, and the right response is to regenerate, not loosen."""

    @pytest.mark.parametrize("case", sorted(gc.CASES))
    def test_the_constants_are_the_ones_the_reference_used(self, case):
        ref = REF["cases"][case]["components"]
        now = gc.component_data(gc.CASES[case]["names"])
        assert now["names"] == ref["names"], REGENERATE
        for key in ("MW", "Tc", "Pc", "omega", "cp_ig"):
            np.testing.assert_allclose(now[key], ref[key], rtol=1e-12, err_msg=f"{key}: {REGENERATE}")

    @pytest.mark.parametrize("case", sorted(gc.CASES))
    def test_the_case_is_the_one_the_reference_solved(self, case):
        assert REF["cases"][case]["case"] == json.loads(json.dumps(gc.CASES[case])), REGENERATE
