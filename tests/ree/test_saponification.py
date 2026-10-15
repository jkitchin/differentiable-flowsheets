"""Saponifier unit, saponification duty and their validation (#197)."""

import jax

jax.config.update("jax_enable_x64", True)

import pytest

from difflow_ree.economics.saponification import compare_counter_ions
from difflow_ree.equilibrium.saponification import SaponifiedParams
from difflow_ree.units.saponification import Saponifier, SaponifierParams


def test_saponifier_smoke():
    unit = Saponifier(SaponifierParams(extractant="D2EHPA",
                                       saponification_degree=0.35))
    organic = unit.schema.make_organic(0.5 / unit.monomers_per_basis_unit,
                                       diluent_flow=4.0)
    out, spent, info = unit(organic)
    assert float(info["saponification_degree"]) == pytest.approx(0.35)
    assert float(info["counter_ion_imbalance"]) == pytest.approx(0.0, abs=1e-12)
    assert float(out["F_Na_org"]) > 0.0


def test_compare_counter_ions_smoke():
    duties = compare_counter_ions({"Nd": 1.0})
    assert set(duties) == {"Na", "NH4", "Mg"}
    assert all(float(d.kg_base_per_kg_reo) > 0.0 for d in duties.values())


def test_saponifier_rejects_impossible_configurations():
    with pytest.raises(ValueError, match="no acidic proton"):
        SaponifierParams(extractant="TBP")
    with pytest.raises(ValueError, match="cannot disagree"):
        SaponifierParams(extractant="D2EHPA", counter_ion="Na", base="NH3")
    with pytest.raises(ValueError, match="base_utilization"):
        SaponifierParams(extractant="D2EHPA", base_utilization=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SaponifierParams(extractant="D2EHPA", saponification_degree=1.5)


def test_saponified_section_rejects_a_solvating_extractant():
    with pytest.raises(ValueError, match="cannot be saponified"):
        SaponifiedParams(
            n_stages=2, extractant="TBP", elements=("Nd",),
            aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
            anion="NO3",
        )
