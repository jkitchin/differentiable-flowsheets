"""REE mass-action equilibrium closure (#196): smoke and validation."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from difflow.eo_solver import solve_residual_system
from difflow_ree.equilibrium.mass_action import MassActionParams, MassActionSection
from difflow_ree.equilibrium.network import (
    Reaction,
    build_network,
    get_network_template,
    list_networks,
    log_K_from_correlation,
)
from difflow_ree.equilibrium.schema import REEStreamSchema
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams

CAL_PH = 0.5


def _section(**kw):
    return MassActionSection(MassActionParams(
        n_stages=kw.pop("n_stages", 2), extractant="D2EHPA",
        elements=kw.pop("elements", ("Nd",)),
        aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0, **kw))


def _streams(elements=("Nd",)):
    schema = REEStreamSchema(elements=elements, extractant="D2EHPA")
    feed = schema.make_aqueous({e: 0.02 for e in elements}, acid=0.01, water=55.0)
    return feed, schema.make_organic(0.5, diluent_flow=4.0)


def test_every_shipped_network_builds():
    for name in list_networks():
        net = build_network(name, ("Nd",), log10_K={"Nd": -6.0})
        assert net.n_components > 0


def test_section_solve_smoke():
    section = _section(calibration_pH=CAL_PH)
    feed = section.schema.make_aqueous({"Nd": 0.02}, acid=0.01, water=55.0)
    solvent = section.schema.make_organic(0.5, diluent_flow=4.0)
    sol = section.solve(feed, solvent)
    assert bool(sol.feasible)
    assert np.all(np.isfinite(np.asarray(sol.u)))


def test_charge_inconsistent_network_is_rejected():
    template = get_network_template("cation_exchange_dimer")
    broken = dataclasses.replace(
        template,
        species=(Reaction(name="RE(HA2)3", phase="organic", charge=0,
                          stoichiometry={"RE3+": 1, "(HA)2": 3, "H+": -2}),),
        per_element_species=frozenset({"RE(HA2)3"}),
    )
    with pytest.raises(ValueError, match="not charge consistent"):
        build_network(broken, ("Nd",), log10_K={"Nd": -6.0})


def test_basis_mismatch_between_record_and_network_is_rejected():
    with pytest.raises(ValueError, match="extractant_basis"):
        log_K_from_correlation("cation_exchange_monomer", ("Nd",), "D2EHPA",
                               calibration_pH=CAL_PH)


def test_divalent_anion_is_refused():
    with pytest.raises(ValueError, match="declares its anion component"):
        _section(anion="SO4")


def test_closed_model_refuses_a_specified_ph():
    feed, solvent = _streams()
    extractor = REEExtractor(REEExtractorParams(
        n_stages=2, extractant="D2EHPA", elements=("Nd",), pH=CAL_PH,
        model="mass_action", aqueous_volumetric_flow=1.0,
        organic_volumetric_flow=1.0))
    with pytest.raises(ValueError, match="pH is an OUTPUT"):
        extractor(feed, solvent, pH=CAL_PH)


def test_correlation_level_refuses_base_addition():
    feed, solvent = _streams()
    extractor = REEExtractor(REEExtractorParams(
        n_stages=2, extractant="D2EHPA", elements=("Nd",), pH=CAL_PH))
    with pytest.raises(ValueError, match="base_addition"):
        extractor(feed, solvent, base_addition=0.01)


def test_extractor_params_validation():
    with pytest.raises(ValueError, match="volumetric_flow"):
        REEExtractorParams(n_stages=2, extractant="D2EHPA", elements=("Nd",),
                           model="mass_action")
    with pytest.raises(ValueError, match="model must be one of"):
        REEExtractorParams(n_stages=2, extractant="D2EHPA", elements=("Nd",),
                           model="magic")


def test_schema_rejects_unknown_ions():
    with pytest.raises(ValueError, match="Unknown anion"):
        REEStreamSchema(elements=("Nd",), extractant="D2EHPA", anion="ClO4")
    with pytest.raises(ValueError, match="Unknown counter_ion"):
        REEStreamSchema(elements=("Nd",), extractant="D2EHPA", counter_ion="Cs")


def test_solve_residual_system_smoke():
    z, norm, feasible = solve_residual_system(
        lambda z, a: z ** 2 - a, jnp.array([1.0]), 2.0)
    assert bool(feasible) and z.shape == (1,)
