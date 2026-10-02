"""The isomerization unit's committed IDAES reference: cheap checks on every commit (#311).

Split from ``test_isomerization_validation.py``, which is ``release``
throughout, so that a deleted or truncated ``isom_reference.json``, or
thermochemistry that has drifted from the constants the reference was built
on, is caught per commit and not first at release time.

Needs neither IDAES nor IPOPT: it reads the JSON and difflow's species
table. The isomer-family check is here rather than in the release file
because difflow's side of it is a closed form, as cheap as reading the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from difflow_refinery.isomerization import family_equilibrium
from difflow_refinery.isomerization import thermochem as tc

from .reference import isom_generate as gen

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent / "reference"
REF = json.loads((HERE / "isom_reference.json").read_text())

REGENERATE = ("difflow's isomerization thermochemistry (or the constructed feeds, or the cases) "
              "no longer match what the IDAES reference was built on; regenerate it: "
              "PYTHONPATH=src:tests python -m refinery.reference.isom_generate (needs idaes and "
              "ipopt), and re-read the isomerization Validation section of "
              "docs/unit-operations-refinery.md against the new numbers")


def all_solves():
    for fam, rows in REF["families"].items():
        for r in rows:
            yield f"families/{fam}/{r['T']}", r["solve"]
    for r in REF["c6_ring"]:
        yield f"c6_ring/{r['T']}", r["solve"]
    for k, r in REF["adiabatic"].items():
        yield f"adiabatic/{k}", r["solve"]


class TestReferenceFile:
    def test_it_records_where_it_came_from(self):
        p = REF["provenance"]
        for key in ("generated", "script", "command", "idaes", "pyomo", "ipopt",
                    "difflow_commit", "model", "elements"):
            assert p[key], key
        assert "not an independent one" in p["model"]
        assert (HERE / "isom_generate.py").exists()

    def test_every_case_is_there(self):
        assert set(REF["families"]) == set(tc.FAMILIES)
        for rows in REF["families"].values():
            assert [r["T"] for r in rows] == list(gen.FAMILY_T)
        assert [r["T"] for r in REF["c6_ring"]] == list(gen.C6_RING["T"])
        assert set(REF["adiabatic"]) == set(gen.ADIABATIC["feeds"])

    @pytest.mark.parametrize("case,solve", list(all_solves()))
    def test_every_solve_is_optimal(self, case, solve):
        assert solve["status"] == "optimal", case


class TestNoDrift:
    """The reference holds for the constants it was given. If they move,
    the answer checks would fail for a reason unrelated to the model."""

    @pytest.mark.parametrize("name", tc.NAMES)
    def test_component_constants_are_the_ones_the_reference_used(self, name):
        s = tc.SPECIES[tc.idx(name)]
        r = REF["components"][name]
        assert (s.Hf, s.S, s.C, s.H) == (r["Hf"], r["S"], r["C"], r["H"]), REGENERATE
        assert list(s.cp) == r["cp"], REGENERATE

    @pytest.mark.parametrize("kind", gen.ADIABATIC["feeds"])
    def test_the_adiabatic_charge_is_the_one_the_reference_used(self, kind):
        now = gen.adiabatic_charge(kind)
        then = REF["adiabatic"][kind]["feed"]
        assert set(now) == set(then), REGENERATE
        for n, v in then.items():
            assert now[n] == pytest.approx(v, rel=1e-12), REGENERATE


class TestTheReferenceIsSelfConsistent:
    """Properties of the IDAES answers that hold whatever difflow computes."""

    @staticmethod
    def _atoms(flows):
        C = sum(v * tc.C_ATOMS[tc.idx(n)] for n, v in flows.items())
        H = sum(v * tc.H_ATOMS[tc.idx(n)] for n, v in flows.items())
        return float(C), float(H)

    @pytest.mark.parametrize("case", ["c6_ring/0", "c6_ring/1", "adiabatic/paraffinic",
                                      "adiabatic/benzene_rich"])
    def test_atoms_are_conserved(self, case):
        kind, key = case.split("/")
        r = REF[kind][int(key)] if kind == "c6_ring" else REF[kind][key]
        cin, hin = self._atoms(r["feed"])
        cout, hout = self._atoms(r["flows"])
        assert cout == pytest.approx(cin, rel=1e-8)
        assert hout == pytest.approx(hin, rel=1e-8)

    @pytest.mark.parametrize("kind", gen.ADIABATIC["feeds"])
    def test_the_adiabatic_answer_closes_difflows_energy_balance(self, kind):
        """IDAES's adiabatic outlet, priced with difflow's enthalpies: the
        same constants, so the inlet and outlet enthalpy must agree."""
        r = REF["adiabatic"][kind]

        def H(flows, T):
            F = jnp.asarray([flows.get(n, 0.0) for n in tc.NAMES])
            return float(jnp.dot(F, tc.enthalpy(T)))

        h_in = H(r["feed"], gen.ADIABATIC["T_in"])
        h_out = H(r["flows"], r["T"])
        assert abs(h_out - h_in) < 1e-6 * sum(r["feed"].values()) * 1e3

    @pytest.mark.parametrize("fam", sorted(tc.FAMILIES))
    def test_the_families_match_the_closed_form(self, fam):
        for r in REF["families"][fam]:
            x = family_equilibrium(fam, r["T"])
            for n, v in r["x"].items():
                assert float(x[n]) == pytest.approx(v, abs=1e-10), (fam, r["T"], n)

    def test_the_isothermal_ring_case_releases_heat(self):
        """Benzene saturation is exothermic: holding the bed isothermal
        takes heat out."""
        for r in REF["c6_ring"]:
            assert r["heat_duty"] < 0
