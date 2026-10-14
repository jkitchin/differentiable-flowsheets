"""Tests for difflow.publish."""

import json

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (
    CSTR,
    CSTRParams,
    Flowsheet,
    SweepAxis,
    Unit,
    get_flows,
    make_stream,
    mass_action_kinetics,
)
from difflow.publish import publish, sweep, to_html

SPECIES = ["A", "B"]


def payload_of(page: str) -> dict:
    """What the page hands its front end, read back out of the file."""
    body = page.split("window.DIFFLOW = ", 1)[1].split(";</script>", 1)[0]
    return json.loads(body)


@pytest.fixture(scope="module")
def flowsheet():
    kin = mass_action_kinetics([{
        "equation": "A -> B",
        "reactants": {"A": 1.0}, "products": {"B": 1.0},
        "rate_params": {"A": 1.0e6, "Ea": 50_000.0, "n": 0.0},
    }], SPECIES)
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0))
    fs.add_unit(Unit("reactor", CSTR(CSTRParams(
        V=1.0, molar_density=1000.0, **kin.params_kwargs()
    )), ["feed"], ["out"]))
    return fs


def product(streams):
    return get_flows(streams["out"])["B"]


def test_sweep_and_publish(flowsheet, tmp_path):
    result = sweep(flowsheet, [SweepAxis("reactor.V", 0.5, 3.0, n=3)],
                   {"</script><b>x": product})
    assert result.shape == (3,)
    assert np.all(np.isfinite(np.asarray(result.values["</script><b>x"])))
    assert np.all(np.isfinite(np.asarray(result.gradients["</script><b>x"]["reactor.V"])))
    page = to_html(result, title="t", flowsheet=flowsheet)
    assert "</script><b>x" not in page
    data = payload_of(page)
    assert data["sweep"]["outputs"][0]["name"] == "</script><b>x"
    assert [u["name"] for u in data["topology"]["units"]] == ["reactor"]
    for marker in ('id="sliders"', 'id="chart"'):
        assert marker in page
    for pattern in ("<script src", "<link ", "@import url(", ' src="http', ' href="http'):
        assert pattern not in page, f"page fetches via {pattern!r}"
    path = publish(flowsheet, [SweepAxis("reactor.V", 0.5, 3.0, n=2)],
                   {"Product": product}, tmp_path / "model.html", title="Reactor",
                   gradients=False)
    assert path.read_text().startswith("<!doctype html>")


def test_sweep_validation(flowsheet):
    with pytest.raises(ValueError, match="at least one axis"):
        sweep(flowsheet, [], {"product": product})
    with pytest.raises(ValueError, match="at least one output"):
        sweep(flowsheet, [SweepAxis("reactor.V", 1.0, 2.0)], {})
    with pytest.raises(ValueError, match="at least 2 points"):
        SweepAxis("reactor.V", 1.0, 2.0, n=1)
    with pytest.raises(ValueError, match="hi must exceed lo"):
        SweepAxis("reactor.V", 2.0, 1.0)
