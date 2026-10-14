"""The vacuum column's committed reference: cheap checks run on every commit (#294).

Split from ``test_vdu_validation.py`` because that module is ``release``
throughout, and a class-level ``pytestmark = []`` does not take a
module-level mark off (pytest merges marks; it does not override them). So
a deleted or truncated ``vdu_reference.json``, or a characterisation that has
drifted away from the one the reference was built on, is caught here per
commit, not first at release time.

Needs neither Pyomo nor IPOPT: it reads the JSON and, for the staleness
check, runs difflow's characterisation (about two seconds).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from .reference import vdu_case as vc

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent / "reference"
REF = json.loads((HERE / "vdu_reference.json").read_text())

REGENERATE = ("difflow's characterisation (or the vacuum case) no longer matches the one the "
              "vacuum reference was built on; regenerate it: PYTHONPATH=src:tests python -m "
              "refinery.reference.vdu_generate (needs pyomo and ipopt), and re-read the vacuum "
              "Validation section of docs/unit-operations-refinery.md against the new numbers")


def _plain(x):
    return json.loads(json.dumps(x))


class TestReferenceFile:
    def test_it_records_where_it_came_from(self):
        p = REF["provenance"]
        for key in ("generated", "script", "command", "pyomo", "ipopt", "property_methods",
                    "reference_simulator", "difflow_commit"):
            assert p[key], key
        assert "DWSIM" in p["reference_simulator"]
        assert (HERE / "vdu_generate.py").exists()
        assert (HERE / "vdu_mesh.py").exists()

    def test_every_part_is_there_and_solved(self):
        assert REF["column"]["solve"]["status"] == "optimal"
        assert REF["column"]["solve"]["start"] == "engineering_guess"
        assert REF["murphree"]["solve"]["status"] == "optimal"
        for name in ("exact_fixed_point", "published_branches"):
            assert REF["variants"][name]["solve"]["status"] == "optimal", name
        for knob in ("furnace_T", "flash_zone_P", "steam_per_feed"):
            assert REF["sensitivities"][knob]["d"], knob

    @pytest.mark.parametrize("part", ["column", "murphree"])
    def test_the_reference_balances_its_own_mass(self, part):
        """Every kilogram of residue feed leaves in a product: a reference
        that loses mass would make every rate comparison meaningless."""
        o = REF[part]["outputs"]
        out = sum(o[f"{p}.rate"] for p in ("overhead", "lvgo", "hvgo", "slop", "residue"))
        assert out == pytest.approx(o["feed.rate"], rel=1e-9)

    @pytest.mark.parametrize("part", ["column", "murphree"])
    def test_the_reference_meets_its_specs(self, part):
        o = REF[part]["outputs"]
        specs = vc.SPECS if part == "column" else vc.EFFICIENCY_SPECS
        assert o["top.T"] == pytest.approx(specs["top.T"], abs=1e-8)
        assert o["overflash"] == pytest.approx(specs["overflash"], rel=1e-8)
        assert o["lvgo.T95"] == pytest.approx(specs["lvgo.T95"], abs=1e-6)


class TestReferenceIsCurrent:
    """The reference was built on difflow's pseudo-components (an input to
    both models); if they have moved, every comparison is against a
    different residue, and the right response is to regenerate, not loosen."""

    def test_the_component_table_is_the_one_the_reference_used(self):
        comp, feed = vc.characterize()
        assert comp["names"] == REF["components"]["names"], REGENERATE
        for key in vc.COMPONENT_FIELDS:
            np.testing.assert_allclose(comp[key], REF["components"][key], rtol=1e-9,
                                       err_msg=f"{key}: {REGENERATE}")
        np.testing.assert_allclose(feed, REF["case"]["feed_mol_s"], rtol=1e-9, err_msg=REGENERATE)

    def test_the_case_is_the_one_the_reference_solved(self):
        c = REF["case"]
        assert c["crude"] == _plain(vc.CRUDE), REGENERATE
        assert c["layout"] == _plain(vc.LAYOUT), REGENERATE
        assert c["specs"] == _plain(vc.SPECS), REGENERATE
        assert c["efficiency_case"] == _plain(vc.EFFICIENCY_CASE), REGENERATE
        assert c["efficiency_specs"] == _plain(vc.EFFICIENCY_SPECS), REGENERATE
