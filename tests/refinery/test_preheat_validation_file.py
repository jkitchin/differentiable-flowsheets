"""The preheat train's committed IDAES reference: cheap checks run on every commit (#313).

Split from ``test_preheat_validation.py`` because that module is
``release`` throughout, and a class-level ``pytestmark = []`` does not take a
module-level mark off. So a deleted or truncated ``preheat_reference.json``,
or a characterisation that has drifted from the one the reference was built
on, is caught here per commit.

Needs neither IDAES nor IPOPT: it reads the JSON and, for the staleness
check, runs difflow's characterisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from .reference import preheat_case as pc
from .reference import preheat_generate as pg

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent / "reference"
REF = json.loads((HERE / "preheat_reference.json").read_text())

REGENERATE = ("difflow's characterisation (or the preheat case) no longer matches the one the "
              "preheat reference was built on; regenerate it: PYTHONPATH=src:tests python -m "
              "refinery.reference.preheat_generate (needs idaes and ipopt)")


class TestReferenceFile:
    def test_it_records_where_it_came_from(self):
        p = REF["provenance"]
        for key in ("generated", "script", "command", "idaes", "pyomo", "ipopt",
                    "property_methods", "unit_models", "difflow_commit"):
            assert p[key], key
        # what it is: the same model, implemented by IDAES
        assert "not an independent one" in p["property_methods"]
        assert (HERE / "preheat_generate.py").exists()

    def test_every_part_solved(self):
        assert REF["drum"]["solve"]["status"] == "optimal"
        assert all(w["ok"] for w in REF["wet"])
        for name in pg.EXCHANGERS:
            assert REF["exchangers"][name]["solve"]["status"] == "optimal", name

    def test_the_cases_are_the_ones_in_the_generator(self):
        assert REF["drum"]["inputs"] == json.loads(json.dumps(pg.DRUM)), REGENERATE
        assert [(w["T"], w["P"]) for w in REF["wet"]] == [tuple(p) for p in pg.WET["points"]]

    def test_the_drum_reference_flashes(self):
        """The drum check is only a check of the flash if the pressure drop
        does something: the inlet at 15 bar is barely vapour (about 1 %, the
        light ends) and the outlet a third."""
        d = REF["drum"]
        assert d["inlet_vapor_fraction"] < 0.02
        assert 0.2 < d["vapor_fraction"] < 0.5
        assert d["vapor"] + d["liquid"] == pytest.approx(pg.DRUM["rate"], rel=1e-9)
        assert d["T"] < pg.DRUM["T_in"]  # the flash cools it

    def test_the_exchangers_are_the_hot_end_and_e8_starts_to_boil(self):
        e7, e8 = REF["exchangers"]["E7"], REF["exchangers"]["E8"]
        for e in (e7, e8):
            assert e["inputs"]["bypass"] == 0.0  # IDAES's exchanger has none
            assert e["T_cold_out"] > e["inputs"]["cold_T"] and e["T_hot_out"] < e["inputs"]["hot_T"]
        assert e7["cold_out_vapor_fraction"] < 1e-6 < e8["cold_out_vapor_fraction"]


class TestReferenceIsCurrent:
    def test_the_component_table_is_the_one_the_reference_used(self):
        import difflow_refinery as dr

        from .reference import case

        a = pc.assay("base")
        crude = dr.characterize(a)
        cd = case.component_data(crude, dr.ColumnThermo.from_characterization(crude))
        assert cd["names"] == REF["components"]["names"], REGENERATE
        for key in ("MW", "SG", "Tb", "Tc", "Pc", "omega_vp", "hvap_nb", "mole_fraction"):
            np.testing.assert_allclose(cd[key], REF["components"][key], rtol=1e-9,
                                       err_msg=f"{key}: {REGENERATE}")
