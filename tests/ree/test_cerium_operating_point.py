"""Cerium removal at the oxidizer's documented operating point (audit R4).

``CeriumOxidizer`` documents pH 8 as favourable and 80 C as typical, but its
pH factor ran to pH 10, so the defaults converted 40 % of the Ce against a
stated 95 %; and ``FullSeparationTrain`` passed its own temperature (298.15 K,
the solvent-extraction temperature) to the oxidizer, which cut a default
train's Ce removal to 8 %.
"""

import warnings

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.flowsheets.full_train import FullSeparationTrain, SeparationTrainParams
from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams

COMP = dict(La=0.025, Ce=0.045, Pr=0.005, Nd=0.015, Sm=0.002, Eu=0.0005,
            Gd=0.001, Tb=0.0002, Dy=0.0005, Y=0.002)
ELS = tuple(COMP)


def _feed():
    return make_stream({"H2O": 55.5, **COMP}, 298.15, 101325.0)


def test_defaults_give_the_documented_conversion():
    _, solid, info = CeriumOxidizer(CeriumOxidizerParams(elements=ELS))(_feed())
    # ce_conversion at pH >= 8 and 353.15 K, scaled by air's efficiency.
    assert info["ce_conversion"] == pytest.approx(0.95 * 0.85)
    assert float(get_flows(solid)["Ce"]) == pytest.approx(0.95 * 0.85 * COMP["Ce"])


def test_pH_factor_saturates_at_the_operating_pH():
    ox = CeriumOxidizer(CeriumOxidizerParams(elements=ELS))
    assert ox(_feed(), pH=8.0)[2]["ce_conversion"] == \
        pytest.approx(ox(_feed(), pH=10.0)[2]["ce_conversion"])
    assert ox(_feed(), pH=7.0)[2]["ce_conversion"] == \
        pytest.approx(0.5 * ox(_feed(), pH=8.0)[2]["ce_conversion"])
    assert ox(_feed(), pH=6.0)[2]["ce_conversion"] == 0.0


def test_train_runs_the_oxidizer_at_its_own_temperature():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = FullSeparationTrain(SeparationTrainParams())(_feed())
    ce = r["info"]["ce_removal"]
    assert ce["T"] == pytest.approx(353.15)
    assert ce["ce_conversion"] == pytest.approx(0.95 * 0.85)
    assert float(get_flows(r["products"]["CeO2"])["Ce"]) / COMP["Ce"] > 0.8
