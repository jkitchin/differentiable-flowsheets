"""Ideal-thermo records for the species that had critical constants only (issue #442)."""

import warnings

import jax
import jax.numpy as jnp

from difflow import IdealThermo
from difflow.database import (
    SOURCE_CITATIONS,
    _CH2_IDEAL,
    get_species_data,
    list_species,
)

jax.config.update("jax_enable_x64", True)

NEW = sorted(_CH2_IDEAL)


def test_the_gap_is_closed_except_helium_and_acetylene():
    missing = set(list_species("critical")) - set(list_species("ideal"))
    assert missing == {"helium", "acetylene"}


def test_every_new_record_builds_and_is_finite():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in NEW:
            assert name in SOURCE_CITATIONS
            th = IdealThermo({name: get_species_data(name)})
            assert jnp.isfinite(th.H_pure(name, 400.0, "vapor")), name


def test_vapor_enthalpy_grad_is_finite():
    name = NEW[0]
    th = IdealThermo({name: get_species_data(name)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = jax.grad(lambda T: th.H_pure(name, T, "vapor"))(400.0)
    assert jnp.isfinite(g)
