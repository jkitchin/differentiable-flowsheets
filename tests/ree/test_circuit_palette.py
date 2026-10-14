"""The REE circuits as palette operations: stream outlets a flowsheet can wire.

``ExtractStripCircuit`` and its siblings return one results dict, which a
flowsheet handed downstream as if it were a stream: wiring the circuit
into a ``Heater`` died on ``KeyError: 'T'`` (audit of the core outlets).
The palette now registers stream-returning entries under the circuits'
names, each running the circuit unchanged.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_ree as R
from difflow.flowsheet import Flowsheet, Unit, _is_stream
from difflow.plugins import OperationRegistry
from difflow.streams import make_stream
from difflow.units.heat_exchanger import Heater, HeaterParams

EL = ("La", "Nd", "Dy")


@pytest.fixture(scope="module")
def feed():
    return make_stream({"H2O": 10.0, "La": 0.01, "Nd": 0.02, "Dy": 0.01},
                       298.15, 101325.0)


def test_the_palette_names_resolve_to_the_stream_entries():
    registry = OperationRegistry()
    R.register(registry)
    assert registry.get("ExtractStripCircuit") is R.ExtractStripUnit
    assert registry.get("ExtractScrubStripCircuit") is R.ExtractScrubStripUnit
    assert registry.get("SplitShellCascade") is R.SplitShellUnit
    assert registry.get("FullSeparationTrain") is R.SeparationTrainUnit


def test_extract_strip_wires_into_a_heater(feed):
    params = R.ExtractStripParams(extractant="D2EHPA", elements=EL,
                                  n_extraction_stages=5, n_stripping_stages=3)
    fs = Flowsheet(species_order=["H2O", *EL])
    fs.add_feed("feed", feed)
    fs.add_unit(Unit("esc", R.ExtractStripUnit(params), ["feed"],
                     ["raffinate", "product", "barren"]))
    fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=320.0, Cp=75.0)),
                     ["product"], ["hot"]))
    streams = fs.solve()
    assert float(streams["hot"]["T"]) == pytest.approx(320.0)

    # the numbers are the circuit's own
    results = R.ExtractStripCircuit(params)(feed)
    for name, key in (("raffinate", "raffinate"), ("product", "product"),
                      ("barren", "barren_organic")):
        for k, v in results[key].items():
            assert float(streams[name][k]) == pytest.approx(float(v)), (name, k)
    assert "recovery" in fs.last_solve_unit_info["esc"]


def test_extract_scrub_strip_returns_four_streams(feed):
    unit = R.ExtractScrubStripUnit(R.ExtractScrubStripParams(
        extractant="D2EHPA", elements=EL, target_elements=("Dy",)))
    *streams, info = unit(feed)
    assert len(streams) == 4 and all(map(_is_stream, streams))
    assert "target_recovery" in info


def test_split_shell_products_are_streams(feed):
    """The cascade's products carry bare element keys; the outlets must not."""
    unit = R.SplitShellUnit(R.SplitShellParams(
        extractant="D2EHPA", elements=EL, n_stages=12, split_points=(4, 8)))
    solvent = make_stream({"D2EHPA": 0.5, "kerosene": 1.0,
                           "La": 0.0, "Nd": 0.0, "Dy": 0.0}, 298.15, 101325.0)
    *streams, info = unit(feed, solvent)
    assert len(streams) == 4    # three side-draws and the raffinate
    assert all(map(_is_stream, streams))
    # every element of the feed leaves through some outlet
    for e in EL:
        out = sum(float(s.get(f"F_{e}", 0.0)) for s in streams)
        assert out == pytest.approx(float(feed[f"F_{e}"]), rel=1e-9), e


@pytest.mark.parametrize("kw,names", [
    ({}, ("light_REE", "middle_REE", "heavy_REE", "CeO2")),
    ({"include_ce_removal": False}, ("light_REE", "middle_REE", "heavy_REE")),
    ({"group_separation": False}, ("ce_depleted", "CeO2")),
])
def test_separation_train_outlets_follow_its_steps(kw, names):
    elements = ("La", "Ce", "Nd", "Sm", "Dy")
    feed = make_stream({"H2O": 10.0, **{e: 0.01 for e in elements}},
                       298.15, 101325.0)
    unit = R.SeparationTrainUnit(R.SeparationTrainParams(elements=elements, **kw))
    assert unit.outlet_names == names
    *streams, info = unit(feed)
    assert len(streams) == len(names) and all(map(_is_stream, streams))
    assert "mass_balance" in info
