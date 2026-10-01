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

from .reference import gasplant_case as gc

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent / "reference"
REF = json.loads((HERE / "gasplant_reference.json").read_text())

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

    @pytest.mark.parametrize("case", sorted(gc.CASES))
    def test_every_case_is_there_and_solved(self, case):
        r = REF["cases"][case]["idaes"]
        assert r["solve"]["status"] == "optimal"
        n = gc.CASES[case]["n_trays"]
        assert len(r["stages"]) == n + 2
        assert all(s["K"] is not None for s in r["stages"][1:])

    @pytest.mark.parametrize("case", sorted(gc.CASES))
    def test_the_reference_balances_its_own_mass(self, case):
        c = gc.CASES[case]
        r = REF["cases"][case]["idaes"]
        for i, n in enumerate(c["names"]):
            out = r["distillate"][n] + r["bottoms"][n]
            assert out == pytest.approx(c["F"] * c["z"][i], rel=1e-6, abs=1e-9), n

    @pytest.mark.parametrize("case", sorted(gc.CASES))
    def test_the_reference_meets_its_specs(self, case):
        """Reflux and boilup ratios are what IDAES held fixed: its stage
        flows must reproduce them."""
        c = gc.CASES[case]
        r = REF["cases"][case]["idaes"]
        assert r["reflux_ratio"] == pytest.approx(c["reflux_ratio"], rel=1e-8)
        assert r["boilup_ratio"] == pytest.approx(c["boilup_ratio"], rel=1e-8)


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
