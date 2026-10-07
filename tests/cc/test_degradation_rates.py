"""Plausible annual loss at the degradation models' defaults (#376).

The defaults lost ~1900x the amine inventory per year (about 571,000 kg/m3/yr
for 30 wt% MEA) and 100 % of a zeolite's capacity in a year. These pin the
annual loss to the ranges reported for amine oxidative/thermal degradation
(Rochelle 2009; Sexton & Rochelle 2011; Davis & Rochelle 2009) and for
adsorbent capacity fade (Choi 2009; Sayari 2011; Hedin 2010).
"""

import pytest

from difflow_cc.degradation.adsorbent_degradation import (
    STABILITY_PARAMS,
    AdsorbentDegradationParams,
    capacity_fade,
)
from difflow_cc.degradation.amine_degradation import (
    AmineDegradationParams,
    total_amine_loss,
)

# fraction of the amine inventory lost per year at the default operating point
AMINE_BAND = {
    "MEA": (0.3, 1.5),
    "DEA": (0.2, 1.0),
    "AMP": (0.1, 0.6),
    "MDEA": (0.03, 0.3),
    "PZ": (0.03, 0.3),
}
# percent of capacity lost in a year (8000 h, 48 cycles/day)
ADSORBENT_BAND = {
    "zeolite": (1.0, 10.0),
    "carbon": (1.0, 10.0),
    "MOF": (8.0, 35.0),
    "amine_silica": (10.0, 45.0),
}


@pytest.mark.parametrize("solvent", AMINE_BAND)
def test_amine_annual_loss_is_a_fraction_of_the_inventory(solvent):
    loss = total_amine_loss(AmineDegradationParams(solvent=solvent))
    lo, hi = AMINE_BAND[solvent]
    assert lo <= float(loss["total_fraction_yr"]) <= hi
    # 30 wt% of ~1000 kg/m3 is 300 kg amine per m3 of solvent
    assert float(loss["total_kg_m3_yr"]) < 1.5 * 300.0


def test_amine_stability_ordering_and_oxidation_dominates():
    frac = {
        s: float(total_amine_loss(AmineDegradationParams(solvent=s))["total_fraction_yr"])
        for s in AMINE_BAND
    }
    assert frac["MEA"] > frac["DEA"] > frac["AMP"] > frac["MDEA"]
    assert frac["MEA"] > frac["PZ"]
    mea = total_amine_loss(AmineDegradationParams(solvent="MEA"))
    assert mea["oxidative_kg_m3_yr"] > mea["thermal_kg_m3_yr"] > mea["CO2_induced_kg_m3_yr"]


def test_hotter_stripper_degrades_more():
    cool = total_amine_loss(AmineDegradationParams(T_stripper=383.15))
    hot = total_amine_loss(AmineDegradationParams(T_stripper=403.15))
    assert float(hot["thermal_kg_m3_yr"]) > float(cool["thermal_kg_m3_yr"])


@pytest.mark.parametrize("material", ADSORBENT_BAND)
def test_adsorbent_annual_capacity_loss_is_plausible(material):
    assert material in STABILITY_PARAMS
    fade = capacity_fade(8000.0, AdsorbentDegradationParams(material_type=material))
    lo, hi = ADSORBENT_BAND[material]
    assert lo <= float(fade["capacity_loss_percent"]) <= hi
    assert float(fade["thermal_contribution"]) > 0.5


def test_zeolite_is_the_most_durable_default():
    loss = {
        m: float(capacity_fade(8000.0, AdsorbentDegradationParams(material_type=m))["capacity_loss_percent"])
        for m in ADSORBENT_BAND
    }
    assert loss["zeolite"] < loss["MOF"] < loss["amine_silica"]
