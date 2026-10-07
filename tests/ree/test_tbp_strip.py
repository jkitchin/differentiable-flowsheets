"""A TBP circuit strips at its own, low nitrate (2026 operating-point audit, R8).

TBP's D is driven by nitrate, not pH. One ``nitrate_conc`` was threaded to
every section and the strip pH does not enter D, so the strip D equalled the
extraction D: whatever loaded, stayed loaded.
"""

import warnings

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)
from difflow_ree.flowsheets.extract_strip import ExtractStripCircuit, ExtractStripParams

COMP = {"La": 0.02, "Nd": 0.02, "Dy": 0.02}
ELS = tuple(COMP)


def _feed():
    return make_stream({"H2O": 55.5, **COMP}, 298.15, 101325.0)


def _run(**kw):
    # 30 % v/v TBP (1.1 M) at 6 M nitrate, the top of the documented window,
    # with a generous solvent ratio: TBP is a weak extractant.
    params = ExtractStripParams(extractant="TBP", elements=ELS, nitrate_conc=6.0,
                                extractant_conc=1.1, solvent_to_feed_ratio=3.0,
                                n_extraction_stages=10, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return params, ExtractStripCircuit(params)(_feed())


def test_strip_runs_at_low_nitrate_by_default():
    params, r = _run()
    assert params.strip_nitrate_conc == ExtractStripParams.DEFAULT_STRIP_NITRATE
    loaded = {e: COMP[e] - float(get_flows(r["raffinate"])[e]) for e in ELS}
    assert loaded["Dy"] > 0.5 * COMP["Dy"]  # it extracts
    product = get_flows(r["product"])
    for e in ELS:
        # and what loaded comes off in the strip
        assert float(product[e]) > 0.99 * loaded[e], e


def test_one_nitrate_everywhere_does_not_strip():
    """The old behaviour, kept reachable: strip D == extraction D."""
    _, r = _run(strip_nitrate_conc=6.0)
    product = get_flows(r["product"])
    barren = get_flows(r["barren_organic"])
    assert float(barren["Dy"]) > float(product["Dy"])


def test_scrub_circuit_and_cation_exchange_defaults():
    p = ExtractScrubStripParams(extractant="TBP", elements=ELS, nitrate_conc=6.0)
    assert p.strip_nitrate_conc == pytest.approx(1.0)
    # Nitrate does not enter a cation-exchange D: nothing is invented.
    p = ExtractScrubStripParams(extractant="D2EHPA", elements=ELS)
    assert p.strip_nitrate_conc is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = ExtractScrubStripCircuit(ExtractScrubStripParams(
            extractant="TBP", elements=ELS, nitrate_conc=6.0, extractant_conc=1.1))
    assert c._stripper.params.nitrate_conc == pytest.approx(1.0)
