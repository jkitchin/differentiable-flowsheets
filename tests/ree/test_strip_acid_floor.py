"""The strip acid floor on strip pHs the code chooses.

The cut rule (equilibrium/operating_points.py) put heavy-REE stripping on
D2EHPA at pH -1.17, 14.6 M H+ on the plugin's concentration scale: beyond
any real strip liquor (concentrated HCl is ~12 M, plants strip the heavies
with 4-6 M) and 1.2 pH units below the [0, 2] window the coefficients were
fitted over. ``max_strip_acid`` (default 6 M, pH -0.78) bounds a strip pH
the code chooses; the strip then runs at the floor, the circuits report
what stays on the barren organic, the design helpers add strip stages at
the floor where that meets their target and warn
(:class:`StripAcidLimitWarning`) where it cannot. A pH the caller gives is
never clamped. The floor is a stopgap until D2EHPA is refitted down to
strong acid (#384).
"""

import math
import warnings

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.equilibrium.operating_points import (
    MAX_STRIP_ACID,
    StripAcidLimitWarning,
    circuit_phase_ratios,
    cut_pHs,
    strip_pH_floor,
)
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
    design_extract_scrub_strip,
)
from difflow_ree.flowsheets.extract_strip import (
    ExtractStripCircuit,
    ExtractStripParams,
    design_extract_strip,
)
from difflow_ree.units.stripping import StripperParams

ELS = ("La", "Ce", "Nd", "Sm", "Gd", "Dy", "Y")
HEAVY = ("Gd", "Dy", "Y")
FLOOR = -math.log10(6.0)


def _feed(elements=ELS):
    return make_stream({"H2O": 1.0, **{e: 0.001 for e in elements}},
                       298.15, 101325.0)


def _quiet(fn, *args, **kwargs):
    """Call ``fn`` with the out-of-window extrapolation report silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _floor_warnings(record):
    return [w for w in record if issubclass(w.category, StripAcidLimitWarning)]


def test_floor_is_minus_log10_of_the_acid():
    assert MAX_STRIP_ACID == 6.0
    assert strip_pH_floor() == pytest.approx(FLOOR)
    assert strip_pH_floor(12.0) == pytest.approx(-math.log10(12.0))
    with pytest.raises(ValueError, match="positive"):
        strip_pH_floor(0.0)


def test_cut_reports_the_unfloored_strip_and_the_floor():
    oa = circuit_phase_ratios(1.0, 0.2, 0.5, 0.5)
    kw = dict(extraction_OA=oa["extraction"], scrub_OA=oa["scrubbing"],
              strip_OA=oa["stripping"])
    raw = cut_pHs("D2EHPA", ELS, HEAVY, max_strip_acid=None, **kw)
    floored = cut_pHs("D2EHPA", ELS, HEAVY, **kw)
    # The cut: about pH -1.17, 14.6 M H+.
    assert raw.stripping == pytest.approx(-1.165, abs=5e-3)
    assert floored.strip_acid_limited
    assert floored.strip_cut == raw.stripping
    assert floored.stripping == pytest.approx(FLOOR)
    assert floored.strip_elements == HEAVY
    # The floor moves only the strip.
    assert (floored.extraction, floored.scrubbing) == (raw.extraction, raw.scrubbing)


def test_heavy_d2ehpa_defaults_strip_at_the_floor_and_say_what_stays():
    with pytest.warns(StripAcidLimitWarning, match=r"pH -1\.17.*Y 34\.7 %"):
        p = ExtractScrubStripParams(extractant="D2EHPA", elements=ELS,
                                    target_elements=HEAVY)
    assert p.stripping_pH >= FLOOR - 1e-12
    assert p.stripping_pH == pytest.approx(FLOOR)
    r = _quiet(ExtractScrubStripCircuit(p), _feed())
    retained = {e: float(v) for e, v in r["strip_retained"].items()}
    # What the floored strip leaves on the barren organic, reported: 34.7 %
    # of the Y entering the strip, 0.39 % of the Dy. With the ~15 M strip it
    # was nothing.
    assert retained["Y"] == pytest.approx(0.347, abs=2e-3)
    assert retained["Dy"] == pytest.approx(0.0039, abs=2e-4)
    barren = get_flows(r["barren_organic"])
    product = get_flows(r["product"])
    assert retained["Y"] == pytest.approx(
        float(barren["Y"]) / (float(barren["Y"]) + float(product["Y"])), rel=1e-12)

    with pytest.warns(StripAcidLimitWarning, match="ExtractStripParams"):
        q = ExtractStripParams(extractant="D2EHPA", elements=ELS)
    assert q.stripping_pH == pytest.approx(FLOOR)
    r = _quiet(ExtractStripCircuit(q), _feed())
    assert float(r["strip_retained"]["Y"]) > 0.3


@pytest.mark.parametrize("ext", ["PC88A", "Cyanex272"])
def test_a_cut_within_the_acid_limit_is_unchanged(ext):
    oa = circuit_phase_ratios(1.0, 0.2, 0.5, 0.5)
    raw = cut_pHs(ext, ELS, HEAVY, extraction_OA=oa["extraction"],
                  scrub_OA=oa["scrubbing"], strip_OA=oa["stripping"],
                  max_strip_acid=None)
    assert raw.stripping > FLOOR  # PC88A -0.47 (2.9 M), Cyanex272 0.59
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        p = ExtractScrubStripParams(extractant=ext, elements=ELS,
                                    target_elements=HEAVY)
        q = ExtractStripParams(extractant=ext, elements=ELS)
    assert not _floor_warnings(record)
    assert p.stripping_pH == raw.stripping
    r = _quiet(ExtractScrubStripCircuit(p), _feed())
    assert max(float(v) for v in r["strip_retained"].values()) < 1e-3
    assert q.stripping_pH == cut_pHs(
        ext, ELS, extraction_OA=1.0, strip_OA=2.0, max_strip_acid=None).stripping


def test_an_explicit_strip_pH_below_the_floor_is_used_as_given():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        p = ExtractScrubStripParams(extractant="D2EHPA", elements=ELS,
                                    target_elements=HEAVY, stripping_pH=-1.2)
        q = ExtractStripParams(extractant="D2EHPA", elements=ELS,
                               stripping_pH=-1.2)
        s = StripperParams(n_stages=5, extractant="D2EHPA", elements=ELS,
                           pH=-1.2)
    assert not _floor_warnings(record)
    assert p.stripping_pH == -1.2 and q.stripping_pH == -1.2 and s.pH == -1.2
    # The stripper reports the 15.8 M the pH implies; it does not refuse it.
    assert s.acid_conc == pytest.approx(10 ** 1.2)


def test_stripper_refuses_an_acid_beyond_the_limit_it_would_derive_pH_from():
    with pytest.raises(ValueError, match="max_strip_acid=6 M"):
        StripperParams(n_stages=5, extractant="D2EHPA", elements=ELS,
                       acid_conc=8.0)
    p = StripperParams(n_stages=5, extractant="D2EHPA", elements=ELS,
                       acid_conc=8.0, max_strip_acid=12.0)
    assert p.pH == pytest.approx(-math.log10(8.0))
    p = StripperParams(n_stages=5, extractant="D2EHPA", elements=ELS,
                       acid_conc=6.0)
    assert p.pH == pytest.approx(FLOOR)


def test_design_adds_strip_stages_at_the_floor_when_that_meets_the_target():
    """Nd/Dy on D2EHPA: the cut wants pH -0.96 (9.2 M); at the floor Dy still
    strips (S > 1), so two more strip stages reach 99 % with no warning."""
    feed = {"Nd": 0.01, "Dy": 0.01}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        floored = design_extract_strip(feed, "D2EHPA", target_recovery=0.99)
        raw = design_extract_strip(feed, "D2EHPA", target_recovery=0.99,
                                   max_strip_acid=None)
    assert not _floor_warnings(record)
    assert floored.stripping_pH == pytest.approx(FLOOR)
    assert raw.stripping_pH < FLOOR
    assert floored.n_stripping_stages > raw.n_stripping_stages
    r = _quiet(ExtractStripCircuit(floored), _feed(("Nd", "Dy")))
    assert float(r["recovery"]) >= 0.99


def test_design_warns_when_the_floor_cannot_meet_the_target():
    """With Y the floor leaves S < 1, which no stage count fixes."""
    with pytest.warns(StripAcidLimitWarning, match="barren organic keeps.*Y"):
        p = _quiet_except_floor(design_extract_strip,
                                {"Nd": 0.01, "Dy": 0.01, "Y": 0.01}, "D2EHPA")
    assert p.stripping_pH == pytest.approx(FLOOR)
    assert p.n_stripping_stages == 20
    with pytest.warns(StripAcidLimitWarning, match="design_extract_scrub_strip"):
        q = _quiet_except_floor(design_extract_scrub_strip,
                                {"Nd": 0.01, "Sm": 0.01, "Dy": 0.01, "Y": 0.01},
                                ("Dy", "Y"), "D2EHPA")
    assert q.stripping_pH == pytest.approx(FLOOR)


def _quiet_except_floor(fn, *args, **kwargs):
    """Silence everything but StripAcidLimitWarning (pytest.warns catches it)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("always", StripAcidLimitWarning)
        return fn(*args, **kwargs)
