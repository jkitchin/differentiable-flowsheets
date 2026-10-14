"""NaN, not a number nobody measured, for elements the record has no data for (#384).

D2EHPA has no coefficients for Ho to Lu (#269) and nothing below pH 0 for the
heavies Gd, Tb, Dy and Y, where heavy-REE stripping has to run. With
``no_data="nan"`` those answer NaN; every element with data is untouched.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from difflow import get_flows, make_stream
from difflow_ree.database import get_extractant
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)
from difflow_ree.flowsheets.extract_strip import ExtractStripCircuit, ExtractStripParams

ELS = ("La", "Nd", "Gd", "Dy", "Y", "Ho", "Lu")
HEAVY_WITH_COEFFS = ("Gd", "Dy", "Y")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _dist(**kw):
    return REEDistribution("D2EHPA", ELS, no_data="nan", **kw)


def test_the_record_names_the_heavies_that_are_only_extrapolated():
    assert set(get_extractant("D2EHPA").unmeasured_outside_window) == {"Gd", "Tb", "Dy", "Y"}
    assert get_extractant("PC88A").unmeasured_outside_window == ()


def test_default_behaviour_is_unchanged():
    with pytest.raises(ValueError, match="no coefficients for Ho, Lu"):
        REEDistribution("D2EHPA", ELS)
    d = REEDistribution("D2EHPA", ("Nd", "Dy"), on_out_of_range="ignore")
    assert d.no_data == "warn"
    assert jnp.isfinite(d.get_D("Dy", -0.78))          # extrapolated, as before


@pytest.mark.parametrize("el", ["Ho", "Lu"])
def test_an_element_with_no_coefficients_is_nan_everywhere(el):
    d = _dist()
    assert not d.has_data(el)
    for pH in (-1.0, 0.5, 1.5):
        assert jnp.isnan(d.get_D(el, pH))
    assert jnp.all(jnp.isnan(d.get_D(el, jnp.array([0.2, 1.0]))))


@pytest.mark.parametrize("el", HEAVY_WITH_COEFFS)
def test_a_heavy_is_data_inside_the_window_and_nan_below_it(el):
    d = _dist()
    assert jnp.isfinite(d.get_D(el, 1.0))
    assert jnp.isnan(d.get_D(el, -0.78))
    assert jnp.isnan(jax.jit(lambda p: d.get_D(el, p))(-0.5))
    assert jnp.isfinite(jax.grad(lambda p: d.get_D(el, p))(1.0))


@pytest.mark.parametrize("el", ["La", "Nd"])
def test_an_element_with_data_is_never_nan(el):
    plain = REEDistribution("D2EHPA", ("La", "Nd"), on_out_of_range="ignore")
    for mode in ("nan", "raise"):
        d = REEDistribution("D2EHPA", ELS if mode == "nan" else ("La", "Nd"),
                            no_data=mode, on_out_of_range="ignore")
        for pH in (-1.17, -0.5, 1.0, 2.5):
            assert float(d.get_D(el, pH)) == float(plain.get_D(el, pH))


@pytest.mark.parametrize("el", HEAVY_WITH_COEFFS)
def test_raise_mode_refuses_a_heavy_outside_the_window(el):
    d = REEDistribution("D2EHPA", ("Nd", el), no_data="raise")
    assert jnp.isfinite(d.get_D(el, 1.0))
    with pytest.raises(ValueError, match="no data for"):
        d.get_D(el, -0.78)
    with pytest.raises(ValueError, match="no data for"):
        d.get_D(el, jnp.array([0.5, -0.1]))
    assert jnp.isnan(jax.jit(lambda p: d.get_D(el, p))(-0.5))   # nothing to inspect
    # an element with data is not affected
    assert jnp.isfinite(d.get_D("Nd", -0.78))


def test_raise_mode_still_refuses_an_element_with_no_coefficients():
    with pytest.raises(ValueError, match="no coefficients for Ho"):
        REEDistribution("D2EHPA", ("Nd", "Ho"), no_data="raise")


def _feed(els):
    return make_stream({"H2O": 100.0, **{e: 1.0 for e in els}}, 298.15, 101325.0)


def test_nan_is_not_ignored_by_a_calculation_that_uses_it():
    """If a NaN reaches the calculation, the result says so: for every element
    that shares a limiter with it, not only the one with no data."""
    circuit = ExtractScrubStripCircuit(ExtractScrubStripParams(
        extractant="D2EHPA", elements=ELS, target_elements=("Gd", "Dy"), no_data="nan"))
    r = circuit(_feed(ELS))
    for stream in ("raffinate", "product", "barren_organic"):
        flows = get_flows(r[stream])
        assert jnp.isnan(flows["Ho"]), stream
        assert jnp.isnan(flows["Nd"]), stream     # shares the loading limiter


def test_a_circuit_in_raise_mode_errors_instead_of_computing_without_data():
    els = ("La", "Nd", "Gd", "Dy")
    circuit = ExtractScrubStripCircuit(ExtractScrubStripParams(
        extractant="D2EHPA", elements=els, target_elements=("Gd", "Dy"), no_data="raise"))
    with pytest.raises(ValueError, match="no data for"):
        circuit(_feed(els))
    # an element with no coefficients is refused when the circuit is built
    with pytest.raises(ValueError, match="no coefficients for Ho"):
        ExtractScrubStripCircuit(ExtractScrubStripParams(
            extractant="D2EHPA", elements=("La", "Ho"), target_elements=("La",),
            no_data="raise"))


def test_the_extract_strip_circuit_takes_it_too():
    r = ExtractStripCircuit(ExtractStripParams(
        extractant="D2EHPA", elements=("La", "Ho"), no_data="nan"))(_feed(("La", "Ho")))
    assert jnp.isnan(get_flows(r["product"])["Ho"])


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="no_data"):
        REEDistribution("D2EHPA", ("Nd",), no_data="zero")
