"""REE separation-train superstructure (#202): wiring, validation, smoke."""

import math

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import make_stream
from difflow_ree.equilibrium.distribution import stages_fenske, stages_kremser
from difflow_ree.flowsheets import (
    ExtractScrubStripModule,
    ExtractScrubStripParams,
    OperatingLimits,
    PortMismatchError,
    PrecipitationModule,
    SeparationTrain,
    TopologyError,
    list_modules,
    screen_separation,
)
from difflow_ree.units.precipitation import PrecipitatorParams

ELEMENTS = ("La", "Nd")


def _feed(la=1.5, nd=1.5):
    return make_stream({"H2O": 100.0, "La": la, "Nd": nd}, 298.15, 101325.0)


def _ess_module(name="sep", **overrides):
    params = dict(
        extractant="Cyanex272", elements=ELEMENTS, target_elements=("Nd",),
        n_extraction_stages=3, n_scrubbing_stages=1, n_stripping_stages=1,
        extraction_pH=2.4, scrubbing_pH=2.3, stripping_pH=1.9,
        extractant_conc=0.3, solvent_to_feed_ratio=1.0,
        scrub_to_solvent_ratio=0.1, strip_to_solvent_ratio=0.5,
    )
    params.update(overrides)
    return ExtractScrubStripModule(name, ExtractScrubStripParams(**params),
                                   limits=OperatingLimits(third_phase_loading=0.65))


def _closed_train():
    train = SeparationTrain("closed")
    train.add_module(_ess_module())
    train.add_feed("leach", _feed(), "sep.feed")
    train.connect("sep.barren_organic", "sep.solvent")
    return train


def test_library_lists_the_modules():
    assert set(list_modules()) == {"extract_scrub_strip", "split_shell",
                                   "cerium_oxidation", "precipitation",
                                   "saponification"}


def test_closed_loop_solves_and_reports_constraints():
    train = _closed_train()
    result = train.solve()
    constraints = train.constraints(result)
    assert jnp.all(jnp.isfinite(constraints.vector()))
    assert "sep.third_phase" in constraints.names


def test_phase_mismatch_is_rejected():
    train = SeparationTrain("t")
    train.add_module(_ess_module())
    train.add_module(PrecipitationModule("precip",
                                         PrecipitatorParams(elements=ELEMENTS)))
    with pytest.raises(PortMismatchError, match="phase mismatch"):
        train.connect("sep.barren_organic", "sep.feed")
    with pytest.raises(PortMismatchError, match="phase mismatch"):
        train.connect("precip.solid", "sep.feed")
    assert train.connections == []


def test_species_loss_is_refused_unless_asked_for():
    train = SeparationTrain("t")
    train.add_module(_ess_module("wide"))
    train.add_module(_ess_module("narrow", elements=("La",),
                                 target_elements=("La",)))
    with pytest.raises(PortMismatchError, match="Nd"):
        train.connect("wide.raffinate", "narrow.feed")
    train.connect("wide.raffinate", "narrow.feed", allow_species_loss=True)
    assert len(train.connections) == 1


def test_topology_errors():
    train = SeparationTrain("t")
    train.add_module(_ess_module())
    train.add_feed("leach", _feed(), "sep.feed")
    with pytest.raises(TopologyError, match="already has a source"):
        train.add_feed("second", _feed(), "sep.feed")
    with pytest.raises(TopologyError, match="sep.solvent.*no source"):
        train.validate()


def test_two_tears_sharing_one_source_are_refused():
    train = SeparationTrain("shared_tear")
    train.add_module(_ess_module("b"))
    train.add_module(_ess_module("a"))
    train.add_feed("leach", _feed(), "b.feed")
    train.connect("b.raffinate", "a.feed")
    train.connect("a.barren_organic", "a.solvent")
    train.connect("a.barren_organic", "b.solvent")
    with pytest.raises(TopologyError, match=r"feeds both"):
        train.to_flowsheet()


def test_round_trip_and_unknown_format_version():
    from difflow.serialize import SerializationError

    train = _closed_train()
    clone = SeparationTrain.from_json(train.to_json())
    assert list(clone.modules) == list(train.modules)
    data = train.to_dict()
    data["format_version"] = 999
    with pytest.raises(SerializationError, match="format version"):
        SeparationTrain.from_dict(data)


def test_stage_bounds_and_screen_smoke():
    assert math.isfinite(float(stages_fenske(100.0, 0.99, 0.99)))
    assert math.isfinite(float(stages_kremser(2.0, 1.0, 0.9)))
    verdict = screen_separation("Cyanex272", "Nd", "La", installed_stages=3,
                                purity=0.99, pH=2.3)
    assert isinstance(verdict.admissible, bool)
