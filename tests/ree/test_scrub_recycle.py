"""The scrub liquor is recycled to the extraction feed (#377).

Without reflux the heavy circuit's scrub liquor carried co-extracted heavies
on into the light product (63 % of the default train's Gd). Recycling it as a
closed loop, solved with a tear, sends them back through the extraction.
"""

import warnings

import pytest

from difflow import get_flows, make_stream
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)
from difflow_ree.flowsheets.full_train import (
    FullSeparationTrain,
    GroupSeparator,
    SeparationTrainParams,
)

ELEMENTS = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Y")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _feed(elements=ELEMENTS, water=100.0):
    return make_stream({"H2O": water, **{e: 1.0 for e in elements}}, 298.15, 101325.0)


def _circuit(recycle):
    els = ("La", "Nd", "Gd", "Dy")
    return ExtractScrubStripCircuit(ExtractScrubStripParams(
        extractant="D2EHPA", elements=els, target_elements=("Gd", "Dy"),
        recycle_scrub_liquor=recycle,
    )), _feed(els)


def test_recycle_is_off_for_a_stand_alone_circuit_by_default():
    assert ExtractScrubStripParams(
        extractant="D2EHPA", elements=("La", "Nd")).recycle_scrub_liquor is False
    c, feed = _circuit(False)
    assert c(feed)["recycle"] is None


def test_the_loop_converges_and_closes_the_balance():
    c, feed = _circuit(True)
    r = c(feed)
    assert r["recycle"]["converged"]
    assert 1 < r["recycle"]["iterations"] < 100
    for e, closure in r["mass_balance"]["closure"].items():
        assert float(closure) == pytest.approx(1.0, abs=1e-8), e
    # the converged liquor is internal: the recycled loop's tear stream
    liquor = get_flows(r["scrub_liquor"])
    for e in ("La", "Nd", "Gd", "Dy"):
        assert float(liquor[e]) == pytest.approx(r["recycle"]["scrub_liquor_ree"][e], abs=1e-8)


def test_recycle_raises_the_recovery_of_the_targets():
    open_r = _circuit(False)[0](_circuit(False)[1])
    closed_r = _circuit(True)[0](_circuit(True)[1])
    for e in ("Gd", "Dy"):
        assert float(closed_r["target_recovery"][e]) >= float(open_r["target_recovery"][e])
    assert float(closed_r["target_recovery"]["Gd"]) > float(open_r["target_recovery"]["Gd"]) + 0.05
    assert float(closed_r["target_purity"]) > 0.99


def test_the_default_train_no_longer_leaks_gd_to_the_light_product():
    def gd_split(recycle):
        train = FullSeparationTrain(SeparationTrainParams(
            elements=ELEMENTS, include_ce_removal=False,
            recycle_scrub_liquor=recycle))
        res = train(_feed())
        light = float(get_flows(res["products"]["light_REE"])["Gd"])
        heavy = float(get_flows(res["products"]["heavy_REE"])["Gd"])
        mb = res["mass_balance"]   # REE left on the barren solvent is the holdup
        assert float(mb["total_out"]) + float(mb["holdup"]) == pytest.approx(
            float(mb["total_in"]), rel=1e-8)
        return light, heavy

    light_open, heavy_open = gd_split(False)
    light_closed, heavy_closed = gd_split(True)
    assert light_open > 0.5            # the defect: 63 % of the Gd in the light product
    assert light_closed < 0.15
    assert heavy_closed > heavy_open + 0.4
    # the train recycles by default
    assert SeparationTrainParams().recycle_scrub_liquor is True
    assert GroupSeparator(ELEMENTS).recycle_scrub_liquor is True


def test_a_loop_that_cannot_converge_says_so():
    els = ("La", "Nd", "Gd", "Dy")
    c = ExtractScrubStripCircuit(ExtractScrubStripParams(
        extractant="D2EHPA", elements=els, target_elements=("Gd", "Dy"),
        recycle_scrub_liquor=True, recycle_max_iter=2, recycle_tol=1e-14))
    with pytest.warns(RuntimeWarning, match="did not converge"):
        r = c(_feed(els))
    assert r["recycle"]["converged"] is False
