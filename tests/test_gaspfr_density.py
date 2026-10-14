"""GasPFR physical-bed path: rho_gas derived from, or checked against, the inlet stream."""

import warnings

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import IdealThermo, make_stream
from difflow.database import get_species_data
from difflow.units.pfr import GasPFR, GasPFRDensityWarning, GasPFRParams

P0, T0 = 2.0e5, 400.0
BED = dict(d_p=3e-3, voidage=0.4, mu=1.8e-5, bed_area=0.05)
MW = {"A": 28.0, "B": 44.0}


def _params(**kw):
    return GasPFRParams(
        V=0.05, rate_fn=lambda C, T, p: jnp.zeros(1),
        stoich=jnp.array([[-1.0], [1.0]]), rate_params={},
        species_order=["A", "B"], **kw,
    )


def _inlet():
    return make_stream({"A": 1.0, "B": 0.5}, T0, P0)


def test_rho_derived_from_MW_or_thermo():
    thermo = IdealThermo({"A": get_species_data("nitrogen"),
                          "B": get_species_data("carbon_dioxide")})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, a = GasPFR(_params(**BED, MW=MW))(_inlet())
        _, b = GasPFR(_params(**BED), thermo=thermo)(_inlet())
    assert jnp.isfinite(a["pressure_drop"]) and jnp.isfinite(b["pressure_drop"])


def test_mismatched_rho_warns():
    with pytest.warns(GasPFRDensityWarning, match="rho_gas"):
        GasPFR(_params(**BED, rho_gas=3.0, MW=MW))(_inlet())


def test_no_density_anywhere_raises():
    with pytest.raises(ValueError, match="rho_gas"):
        GasPFR(_params(**BED))(_inlet())
    with pytest.raises(ValueError, match="MW"):
        GasPFR(_params(**BED, MW={"A": 28.0}))(_inlet())
