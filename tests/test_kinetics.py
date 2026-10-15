"""Smoke and validation tests for difflow.kinetics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import CSTR, CSTRParams, get_flows, import_reactions, make_stream
from difflow.kinetics import KineticsSpecError, ReactionSet, mass_action_kinetics

SPECIES = ["A", "B"]


def first_order():
    """A -> B."""
    return [{
        "equation": "A -> B",
        "reactants": {"A": 1.0}, "products": {"B": 1.0},
        "rate_params": {"A": 1.0e6, "Ea": 50_000.0, "n": 0.0},
    }]


def reversible():
    """A <=> B."""
    return [{
        "equation": "A <=> B",
        "reactants": {"A": 1.0}, "products": {"B": 1.0},
        "rate_params": {"A": 2.0, "Ea": 0.0, "n": 0.0},
        "reversible": True, "K_eq": 4.0,
    }]


def test_mass_action_smoke():
    kin = mass_action_kinetics(first_order(), SPECIES)
    assert isinstance(kin, ReactionSet) and "stoich" in kin
    np.testing.assert_allclose(np.asarray(kin.stoich), [[-1.0], [1.0]])
    assert bool(jnp.all(jnp.isfinite(kin.rates({"A": 1.0, "B": 0.0}, 350.0))))
    assert kin.equations and kin.summary()
    assert set(kin.params_kwargs()) == {"rate_fn", "stoich", "rate_params", "species_order"}


@pytest.mark.parametrize("mode", ["forward_only", "equilibrium"])
def test_reverse_modes_run(mode):
    kin = mass_action_kinetics(reversible(), SPECIES, reverse=mode)
    assert np.isfinite(float(kin.rates({"A": 3.0, "B": 8.0}, 300.0)[0]))


def test_cstr_with_mass_action_and_grad():
    kin = mass_action_kinetics(first_order(), SPECIES)
    feed = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)

    def product_flow(pre_exponential):
        rp = {**kin.rate_params, "A": jnp.array([pre_exponential])}
        params = CSTRParams(V=1.5, molar_density=1000.0, rate_fn=kin.rate_fn,
                            stoich=kin.stoich, rate_params=rp, species_order=SPECIES)
        out = CSTR(params)(feed)
        return get_flows(out[0] if isinstance(out, tuple) else out)["B"]

    assert np.isfinite(float(product_flow(1.0e6)))
    assert np.isfinite(float(jax.grad(product_flow)(1.0e6)))


def test_cantera_import_is_accepted():
    reactions = import_reactions("tests/data/test_mechanism.yaml")
    kin = mass_action_kinetics(reactions, reverse="forward_only")
    assert kin.stoich.shape == (len(kin.species_order), len(reactions))


def _no_k_eq():
    rxn = reversible()
    del rxn[0]["K_eq"]
    return rxn


@pytest.mark.parametrize("kwargs,match", [
    (dict(reactions=reversible(), species_order=SPECIES), "reversible"),
    (dict(reactions=_no_k_eq(), species_order=SPECIES, reverse="equilibrium"), "K_eq"),
    (dict(reactions=[{**first_order()[0], "type": "falloff"}], species_order=SPECIES),
     "unsupported reaction type"),
    (dict(reactions=first_order(), species_order=["A"]), "not in species_order"),
    (dict(reactions=[], species_order=SPECIES), "no reactions"),
    (dict(reactions=[{"equation": "A -> B", "rate_params": {"A": 1.0}}], species_order=SPECIES),
     "no reactants and no products"),
    (dict(reactions=first_order(), species_order=SPECIES, orders=[{"Z": 1.0}]),
     "not in species_order"),
], ids=["reversible", "k_eq", "type", "species", "empty", "no_species", "orders"])
def test_bad_specification_is_refused(kwargs, match):
    with pytest.raises(KineticsSpecError, match=match):
        kw = dict(kwargs)
        mass_action_kinetics(kw.pop("reactions"), **kw)


def test_bad_reverse_mode_is_refused():
    with pytest.raises(ValueError, match="reverse="):
        mass_action_kinetics(first_order(), SPECIES, reverse="maybe")
