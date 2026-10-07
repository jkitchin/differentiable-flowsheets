"""Non-REE species leave through some outlet (2026 conservation audit, b/c).

REEScrubber and REEStripper rebuilt their outlets from the extractant, the
diluent, the aqueous carrier and ``elements``, so a saponification
counter-ion on the organic (``Na_org``) vanished: an ExtractScrubStripModule
fed solvent with Na_org = 3.0 returned no Na anywhere. CeriumOxidizer's
filtrate and SplitShellModule's raffinate were rebuilt as {H2O, elements},
dropping acid and impurity metals in the feed.
"""

import warnings

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.flowsheets.extract_scrub_strip import ExtractScrubStripParams
from difflow_ree.flowsheets.modules import ExtractScrubStripModule, SplitShellModule
from difflow_ree.flowsheets.split_shell import SplitShellParams
from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams
from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams
from difflow_ree.units.stripping import REEStripper, StripperParams


def _total(streams, key):
    return sum(float(get_flows(s).get(key, 0.0)) for s in streams)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def test_scrubber_and_stripper_carry_spectators():
    org = make_stream({"D2EHPA": 5.0, "kerosene": 10.0, "Na_org": 0.7,
                       "Nd": 0.05}, 298.15, 101325.0)
    aq = make_stream({"H2O": 5.0, "HCl": 0.4, "Fe": 0.01}, 298.15, 101325.0)
    for unit in (REEScrubber(ScrubberParams(n_stages=3, extractant="D2EHPA",
                                            elements=("Nd",), pH=0.5)),
                 REEStripper(StripperParams(n_stages=3, extractant="D2EHPA",
                                            elements=("Nd",), pH=0.0))):
        a, o, info = unit(org, aq)
        assert float(get_flows(o)["Na_org"]) == pytest.approx(0.7)
        assert "Na_org" not in get_flows(a)
        assert float(get_flows(a)["HCl"]) == pytest.approx(0.4)
        assert float(get_flows(a)["Fe"]) == pytest.approx(0.01)
        assert info["dropped_species"] == ()


def test_module_returns_the_counter_ion():
    """Before: Na_org = 3.0 in, no Na anywhere out."""
    els = ("La", "Nd")
    mod = ExtractScrubStripModule("sep", ExtractScrubStripParams(
        extractant="Cyanex272", elements=els, target_elements=("Nd",),
        extraction_pH=2.4, scrubbing_pH=2.3, stripping_pH=1.9,
        extractant_conc=0.3))
    feed = make_stream({"H2O": 100.0, "La": 3.0, "Nd": 3.0, "Fe": 0.5},
                       298.15, 101325.0)
    solvent = make_stream({"Cyanex272": 30.0, "kerosene": 100.0, "Na_org": 3.0},
                          298.15, 101325.0)
    out = mod(feed, solvent)[:4]
    assert _total(out, "Na_org") == pytest.approx(3.0)
    assert float(get_flows(out[3])["Na_org"]) == pytest.approx(3.0)
    assert _total(out, "Fe") == pytest.approx(0.5)


def test_cerium_filtrate_keeps_the_feed_species():
    feed = make_stream({"H2O": 100.0, "La": 1.0, "Ce": 2.0, "Fe": 0.3,
                        "HNO3": 1.0}, 298.15, 101325.0)
    filt, _, _ = CeriumOxidizer(CeriumOxidizerParams(elements=("La", "Ce")))(feed)
    f = get_flows(filt)
    assert float(f["Fe"]) == pytest.approx(0.3)
    assert float(f["HNO3"]) == pytest.approx(1.0)
    assert float(f["H2O"]) == pytest.approx(100.0)


def test_split_shell_raffinate_keeps_the_feed_species():
    els = ("La", "Ce", "Pr", "Nd")
    feed = make_stream({"H2O": 100.0, "HNO3": 2.0, "Fe": 0.1,
                        **{e: 1.0 for e in els}}, 298.15, 101325.0)
    solvent = make_stream({"D2EHPA": 50.0, "kerosene": 100.0}, 298.15, 101325.0)
    out = SplitShellModule("ss", SplitShellParams(
        extractant="D2EHPA", elements=els, n_stages=12, split_points=(4, 8)))(
        feed, solvent)
    raff = get_flows(out[-2])
    assert float(raff["HNO3"]) == pytest.approx(2.0)
    assert float(raff["Fe"]) == pytest.approx(0.1)
