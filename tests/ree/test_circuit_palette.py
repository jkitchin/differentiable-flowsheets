"""The REE circuits and their palette (stream-returning) wrappers."""

import warnings

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_ree as R
from difflow.flowsheet import _is_stream
from difflow.plugins import OperationRegistry
from difflow.streams import get_flows, make_stream

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
    assert registry.get("REEExtractor") is R.REEExtractor


def test_extract_strip_circuit_smoke(feed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = R.ExtractStripCircuit(R.ExtractStripParams(
            extractant="D2EHPA", elements=EL, n_extraction_stages=3,
            n_stripping_stages=2))(feed)
    for key in ("raffinate", "product", "barren_organic"):
        assert all(jnp.isfinite(v) for v in get_flows(results[key]).values())


def test_split_shell_unit_returns_streams(feed):
    unit = R.SplitShellUnit(R.SplitShellParams(
        extractant="D2EHPA", elements=EL, n_stages=12, split_points=(4, 8)))
    solvent = make_stream({"D2EHPA": 0.5, "kerosene": 1.0}, 298.15, 101325.0)
    *streams, info = unit(feed, solvent)
    assert len(streams) == 4 and all(map(_is_stream, streams))
    for e in EL:
        out = sum(float(s.get(f"F_{e}", 0.0)) for s in streams)
        assert out == pytest.approx(float(feed[f"F_{e}"]), rel=1e-9), e
