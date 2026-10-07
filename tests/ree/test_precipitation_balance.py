"""Precipitators respect their reagent and conserve every species (audit R10).

Before: oxalate/carbonate conversion scaled with sqrt(excess) uncapped, so
0.5x oxalate precipitated 70.5 % of the REE (0.127 mol oxalate needed, 0.090
fed); the filtrate was rebuilt as {H2O, elements}, so HCl, HNO3, Fe, the
excess reagent and the precipitant's water left through no outlet; the
hydroxide route never read its precipitant (0 and 0.36 mol NaOH both 99.5 %)
and reported pH_final = -16; Ho/Er/Tm/Yb/Lu raised a bare KeyError.
"""

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.units.precipitation import (
    CarbonatePrecipitator,
    HydroxidePrecipitator,
    OxalatePrecipitator,
    PrecipitatorParams,
)

ELS = ("Nd", "Dy")
REE = 0.12


def _feed(**extra):
    return make_stream({"H2O": 55.5, "Nd": 0.10, "Dy": 0.02, **extra}, 298.15, 101325.0)


def _species_total(*streams):
    out = {}
    for s in streams:
        for k, v in get_flows(s).items():
            out[k] = out.get(k, 0.0) + float(v)
    return out


@pytest.mark.parametrize("cls,reagent", [(OxalatePrecipitator, "C2O4"),
                                         (CarbonatePrecipitator, "CO3")])
@pytest.mark.parametrize("excess", [0.2, 0.5, 1.0, 1.5])
def test_no_more_precipitate_than_the_reagent_binds(cls, reagent, excess):
    supplied = 1.5 * REE * excess
    rg = make_stream({"H2O": 10.0, reagent: supplied}, 298.15, 101325.0)
    _, solid, _ = cls(PrecipitatorParams(elements=ELS))(_feed(), rg)
    s = get_flows(solid)
    precipitated = sum(float(s[e]) for e in ELS)
    assert 1.5 * precipitated <= supplied * (1 + 1e-12)
    if excess < 1:
        assert 1.5 * precipitated == pytest.approx(supplied)  # reagent-limited


@pytest.mark.parametrize("cls,reagent", [(OxalatePrecipitator, "C2O4"),
                                         (CarbonatePrecipitator, "CO3"),
                                         (HydroxidePrecipitator, "OH")])
def test_every_species_leaves_somewhere(cls, reagent):
    feed = make_stream({"H2O": 100.0, "La": 1.0, "Nd": 2.0, "Fe": 0.3,
                        "HNO3": 1.0}, 298.15, 101325.0)
    rg = make_stream({reagent: 6.75, "H2O": 10.0}, 298.15, 101325.0)
    filt, solid, _ = cls(PrecipitatorParams(elements=("La", "Nd")))(feed, rg)
    out = _species_total(filt, solid)
    for k in ("La", "Nd", "Fe"):
        assert out[k] == pytest.approx(float(get_flows(feed)[k]))
    if cls is HydroxidePrecipitator:
        # HNO3 + OH- -> NO3- + H2O: nitrate and the water it makes are kept.
        assert out["HNO3"] + out["NO3"] == pytest.approx(1.0)
        assert out["H2O"] == pytest.approx(110.0 + out["NO3"])
        assert out["OH"] == pytest.approx(6.75 - out["NO3"])
    else:
        assert out["HNO3"] == pytest.approx(1.0)
        assert out["H2O"] == pytest.approx(110.0)
        assert out[reagent] == pytest.approx(6.75)


def test_hydroxide_is_limited_by_the_base_supplied():
    def run(naoh):
        base = make_stream({"H2O": 10.0, "NaOH": naoh}, 298.15, 101325.0)
        _, solid, info = HydroxidePrecipitator(PrecipitatorParams(elements=ELS))(
            _feed(), base)
        return sum(float(get_flows(solid)[e]) for e in ELS), info

    none, _ = run(0.0)
    half, info = run(0.18)
    full, _ = run(0.36 * 1.5)
    assert none == 0.0
    assert 3 * half == pytest.approx(0.18)
    assert float(info["reagent_scale"]) < 1.0
    assert full == pytest.approx(0.995 * REE, rel=1e-6)


def test_hydroxide_neutralises_acid_first():
    base = make_stream({"H2O": 10.0, "NaOH": 0.36}, 298.15, 101325.0)
    filt, solid, info = HydroxidePrecipitator(PrecipitatorParams(elements=ELS))(
        _feed(HCl=0.5), base)
    # 0.36 mol base does not even cover 0.5 mol HCl: nothing precipitates,
    # and the filtrate is acidic, at the pH of the 0.14 mol excess acid.
    assert sum(float(get_flows(solid)[e]) for e in ELS) == 0.0
    f = get_flows(filt)
    assert float(f["Na"]) == pytest.approx(0.36)
    assert float(f["Cl"]) == pytest.approx(0.36)
    V_L = 0.018015 * 65.5
    assert float(info["pH_final"]) == pytest.approx(-__import__("math").log10(0.14 / V_L), abs=1e-6)


def test_hydroxide_pH_final_is_the_setpoint_when_base_reaches_it():
    """It used to report -16 after any real precipitation."""
    base = make_stream({"H2O": 10.0, "NaOH": 1.0}, 298.15, 101325.0)
    _, _, info = HydroxidePrecipitator(PrecipitatorParams(elements=ELS))(
        _feed(), base, pH=9.0)
    assert float(info["pH_final"]) == pytest.approx(9.0)


@pytest.mark.parametrize("cls", [OxalatePrecipitator, CarbonatePrecipitator,
                                 HydroxidePrecipitator])
def test_missing_solubility_data_is_named(cls):
    with pytest.raises(ValueError, match=r"No \w+ solubility product .*'Ho'"):
        cls(PrecipitatorParams(elements=("Nd", "Ho")))
