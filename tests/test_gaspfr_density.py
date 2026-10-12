"""GasPFR physical-bed path: rho_gas derived from, or checked against, the inlet stream.

Mirrors the ``u_s0`` handling: ``rho_gas=None`` takes the inlet density
``mass flow / Q0`` (``P0 MW / (R T0)`` with the default ideal-gas ``Q0``),
from ``GasPFRParams.MW`` or the unit's thermo; a given ``rho_gas`` that differs
from the stream's by more than 2 % raises ``GasPFRDensityWarning``.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import IdealThermo, make_stream
from difflow.database import get_species_data
from difflow.units.pfr import (
    RHO_GAS_RTOL,
    GasPFR,
    GasPFRBedWarning,
    GasPFRDensityWarning,
    GasPFRParams,
)

P0, T0 = 2.0e5, 400.0
BED = dict(d_p=3e-3, voidage=0.4, mu=1.8e-5, bed_area=0.05)


def _params(**kw):
    return GasPFRParams(
        V=0.05,
        rate_fn=lambda C, T, p: jnp.zeros(1),
        stoich=jnp.array([[-1.0], [1.0]]),
        rate_params={},
        species_order=["A", "B"],
        **kw,
    )


def _inlet():
    return make_stream({"A": 1.0, "B": 0.5}, T0, P0)


def _rho_ideal(MW_mix):
    return P0 * MW_mix * 1e-3 / (8.314 * T0)          # GasPFR's R = 8.314


MW = {"A": 28.0, "B": 44.0}
MW_MIX = (1.0 * 28.0 + 0.5 * 44.0) / 1.5


def test_warning_is_bed_warning_subclass():
    assert issubclass(GasPFRDensityWarning, GasPFRBedWarning)


@pytest.mark.slow
def test_rho_from_MW_matches_explicit_value():
    rho = _rho_ideal(MW_MIX)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, derived = GasPFR(_params(**BED, MW=MW))(_inlet())
        _, explicit = GasPFR(_params(**BED, rho_gas=rho, MW=MW))(_inlet())
    assert float(derived["pressure_drop"]) == pytest.approx(float(explicit["pressure_drop"]), rel=1e-10)
    assert float(derived["pressure_drop"]) > 0


def test_scalar_MW():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, a = GasPFR(_params(**BED, MW=30.0))(_inlet())
        _, b = GasPFR(_params(**BED, rho_gas=_rho_ideal(30.0)))(_inlet())
    assert float(a["pressure_drop"]) == pytest.approx(float(b["pressure_drop"]), rel=1e-10)


def test_rho_from_thermo():
    thermo = IdealThermo({"A": get_species_data("nitrogen"), "B": get_species_data("carbon_dioxide")})
    mw = (1.0 * thermo.species["A"].MW + 0.5 * thermo.species["B"].MW) / 1.5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, a = GasPFR(_params(**BED), thermo=thermo)(_inlet())
        _, b = GasPFR(_params(**BED, rho_gas=_rho_ideal(mw)))(_inlet())
    assert float(a["pressure_drop"]) == pytest.approx(float(b["pressure_drop"]), rel=1e-10)


def test_rho_from_given_volumetric_flow():
    # A non-ideal Q0 changes the stream density to mass / Q0.
    Q0 = 0.9 * 1.5 * 8.314 * T0 / P0
    rho = 1.5 * MW_MIX * 1e-3 / Q0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, a = GasPFR(_params(**BED, MW=MW))(_inlet(), volumetric_flow=Q0)
        _, b = GasPFR(_params(**BED, rho_gas=rho, MW=MW))(_inlet(), volumetric_flow=Q0)
    assert float(a["pressure_drop"]) == pytest.approx(float(b["pressure_drop"]), rel=1e-10)


def test_mismatched_rho_warns_and_is_used():
    rho = _rho_ideal(MW_MIX)
    with pytest.warns(GasPFRDensityWarning, match="rho_gas"):
        _, bad = GasPFR(_params(**BED, rho_gas=1.10 * rho, MW=MW))(_inlet())
    _, good = GasPFR(_params(**BED, rho_gas=1.10 * rho))(_inlet())   # no MW: nothing to check
    assert float(bad["pressure_drop"]) == pytest.approx(float(good["pressure_drop"]), rel=1e-10)


def test_within_tolerance_is_silent():
    rho = _rho_ideal(MW_MIX) * (1.0 + 0.5 * RHO_GAS_RTOL)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GasPFR(_params(**BED, rho_gas=rho, MW=MW))(_inlet())


def test_backward_compatible_without_molar_mass():
    # rho_gas given, no MW and no thermo: no check, no warning (old behaviour).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GasPFR(_params(**BED, rho_gas=2.0))(_inlet())


def test_no_density_anywhere_raises():
    with pytest.raises(ValueError, match="rho_gas"):
        GasPFR(_params(**BED))(_inlet())
    with pytest.raises(ValueError, match="alpha_for"):
        _params(**BED, u_s0=0.5).effective_alpha
    with pytest.raises(ValueError, match="MW"):
        GasPFR(_params(**BED, MW={"A": 28.0}))(_inlet())


@pytest.mark.slow
def test_gradient_through_derived_density():
    def dp(P):
        inlet = make_stream({"A": 1.0, "B": 0.5}, T0, P)
        _, info = GasPFR(_params(**BED, MW=MW))(inlet)
        return info["pressure_drop"]

    g = jax.grad(dp)(P0)
    h = 1.0
    fd = (dp(P0 + h) - dp(P0 - h)) / (2 * h)
    assert jnp.isfinite(g)
    assert float(g) == pytest.approx(float(fd), rel=1e-4)
