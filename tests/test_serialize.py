"""Tests for difflow.serialize.

The load-bearing test is `test_reloaded_flowsheet_solves_identically`: a
file format that reproduces the structure but not the answer is worse
than none, because the difference is invisible until it matters.

The rest check that what cannot be written faithfully is refused with a
message naming the culprit, rather than dropped.
"""

import json

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (
    Flash,
    FlashParams,
    Flowsheet,
    Heater,
    HeaterParams,
    IdealThermo,
    Splitter,
    Unit,
    get_species_data,
    make_stream,
    serialize,
)
from difflow.serialize import FORMAT_VERSION, SerializationError

SPECIES = ["water", "ethanol"]


@pytest.fixture(scope="module")
def thermo():
    return IdealThermo({n: get_species_data(n) for n in SPECIES})


@pytest.fixture
def flowsheet(thermo):
    """Heater into a flash: covers a params-only unit and a thermo one."""
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream(
        {"water": 1.0, "ethanol": 0.5}, T=350.0, P=101325.0
    ))
    fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=360.0)), ["feed"], ["hot"]))
    fs.add_unit(Unit(
        "flash", Flash(FlashParams(species_order=SPECIES), thermo),
        ["hot"], ["liq", "vap"],
    ))
    return fs


# =============================================================================
# Round trip
# =============================================================================


class TestRoundTrip:
    def test_reloaded_flowsheet_solves_identically(self, flowsheet):
        """Structure is not enough; the answer has to survive too."""
        reloaded = serialize.from_json(serialize.to_json(flowsheet))

        original, restored = flowsheet.solve(), reloaded.solve()
        assert set(original) == set(restored)
        for stream in original:
            for key, value in original[stream].items():
                if isinstance(value, str):
                    continue
                assert float(value) == float(restored[stream][key]), (
                    f"{stream}.{key} differs"
                )

    def test_structure_survives(self, flowsheet):
        reloaded = serialize.from_json(serialize.to_json(flowsheet))
        assert reloaded.species_order == flowsheet.species_order
        assert [u.name for u in reloaded.units] == ["heat", "flash"]
        assert [u.inlet_names for u in reloaded.units] == [["feed"], ["hot"]]
        assert [u.outlet_names for u in reloaded.units] == [["hot"], ["liq", "vap"]]

    def test_parameters_survive(self, flowsheet):
        reloaded = serialize.from_json(serialize.to_json(flowsheet))
        assert float(reloaded.units[0].operation.params.T_out) == 360.0
        assert reloaded.units[1].operation.params.species_order == SPECIES

    def test_feeds_survive(self, flowsheet):
        reloaded = serialize.from_json(serialize.to_json(flowsheet))
        feed = reloaded.feeds["feed"]
        assert float(feed["F_water"]) == 1.0
        assert float(feed["T"]) == 350.0

    def test_serialization_is_stable(self, flowsheet):
        """Writing, reading and writing again gives the same bytes."""
        once = serialize.to_json(flowsheet)
        twice = serialize.to_json(serialize.from_json(once))
        assert once == twice

    def test_recycles_survive(self, thermo):
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.5},
                                        T=350.0, P=101325.0))
        fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=360.0)),
                         ["feed", "recycle"], ["hot"]))
        fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                         ["hot"], ["liq", "vap"]))
        fs.add_recycle("liq", "recycle")

        reloaded = serialize.from_json(serialize.to_json(fs))
        assert reloaded.recycles == fs.recycles

    def test_defaults_survive(self):
        fs = Flowsheet(species_order=["A"], default_flow=1e-6,
                       default_T=310.0, default_P=2.0e5)
        reloaded = serialize.from_json(serialize.to_json(fs))
        assert reloaded.default_flow == 1e-6
        assert reloaded.default_T == 310.0
        assert reloaded.default_P == 2.0e5

    def test_output_is_real_json(self, flowsheet):
        data = json.loads(serialize.to_json(flowsheet))
        assert data["format_version"] == FORMAT_VERSION
        assert data["units"][0]["operation"] == "Heater"
        assert "difflow_version" in data


# =============================================================================
# Thermodynamics
# =============================================================================


class TestThermo:
    def test_thermo_is_written_and_rebuilt(self, flowsheet):
        """About half the core units need a thermo object, not just params."""
        reloaded = serialize.from_json(serialize.to_json(flowsheet))
        rebuilt = reloaded.units[1].operation.thermo
        assert isinstance(rebuilt, IdealThermo)
        assert list(rebuilt.species) == SPECIES

    def test_species_data_survives_as_a_namedtuple(self, flowsheet, thermo):
        """SpeciesData is a NamedTuple, which a plain tuple encoding erases."""
        reloaded = serialize.from_json(serialize.to_json(flowsheet))
        rebuilt = reloaded.units[1].operation.thermo.species["water"]
        original = thermo.species["water"]
        assert type(rebuilt) is type(original)
        assert rebuilt._fields == original._fields

    def test_an_unsupported_thermo_is_refused_with_a_way_out(self, flowsheet):
        class ExoticThermo:
            pass

        flowsheet.units[1].operation.thermo = ExoticThermo()
        with pytest.raises(SerializationError, match="extras="):
            serialize.to_json(flowsheet)

    def test_extras_supplies_what_the_file_lacks(self, flowsheet, thermo):
        """A thermo the format cannot write can be passed back on load."""
        data = serialize.to_dict(flowsheet)
        data["units"][1]["constructor"] = {}          # simulate a file without it

        with pytest.raises(SerializationError, match="requires thermo"):
            serialize.from_dict(data)

        reloaded = serialize.from_dict(
            data, extras={"flash": {"thermo": thermo}}
        )
        assert reloaded.units[1].operation.thermo is thermo

    def test_extras_overrides_a_stored_thermo(self, flowsheet, thermo):
        reloaded = serialize.from_json(
            serialize.to_json(flowsheet), extras={"flash": {"thermo": thermo}}
        )
        assert reloaded.units[1].operation.thermo is thermo


# =============================================================================
# What it refuses
# =============================================================================


class TestRefusals:
    def test_a_callable_parameter_is_refused(self, thermo):
        """A file that silently lost a rate law would reload as a different model."""
        from difflow import CSTR, CSTRParams

        def rate_fn(C, T, p):
            return jnp.array([0.0])

        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("rx", CSTR(CSTRParams(
            V=1.0, rate_fn=rate_fn, stoich=jnp.zeros((2, 1)),
            rate_params={}, species_order=SPECIES,
        )), ["feed"], ["out"]))

        with pytest.raises(SerializationError, match="callable"):
            serialize.to_json(fs)

    def test_the_refusal_names_the_field(self, thermo):
        from difflow import CSTR, CSTRParams

        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("rx", CSTR(CSTRParams(
            V=1.0, rate_fn=lambda C, T, p: jnp.array([0.0]),
            stoich=jnp.zeros((2, 1)), rate_params={}, species_order=SPECIES,
        )), ["feed"], ["out"]))
        with pytest.raises(SerializationError) as exc:
            serialize.to_json(fs)
        assert "rate_fn" in str(exc.value)
        assert "rx" in str(exc.value)

    def test_an_unregistered_operation_is_refused(self):
        class HomeMadeUnit:
            def __init__(self):
                self.params = None

        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("mystery", HomeMadeUnit(), ["feed"], ["out"]))
        with pytest.raises(SerializationError, match="not in the operation registry"):
            serialize.to_json(fs)

    def test_a_future_format_version_is_refused(self, flowsheet):
        data = serialize.to_dict(flowsheet)
        data["format_version"] = FORMAT_VERSION + 1
        with pytest.raises(SerializationError, match="not supported"):
            serialize.from_dict(data)

    def test_an_unknown_operation_name_is_refused(self, flowsheet):
        data = serialize.to_dict(flowsheet)
        data["units"][0]["operation"] = "NoSuchUnit"
        with pytest.raises(SerializationError, match="not registered"):
            serialize.from_dict(data)


# =============================================================================
# Values
# =============================================================================


class TestValueEncoding:
    def test_arrays_keep_their_type_and_shape(self):
        from difflow.serialize import _decode_value, _encode_value

        original = jnp.arange(6.0).reshape(2, 3)
        restored = _decode_value(_encode_value(original, "test"))
        assert restored.shape == original.shape
        assert jnp.allclose(restored, original)

    def test_a_plain_list_stays_a_list(self):
        from difflow.serialize import _decode_value, _encode_value

        assert _decode_value(_encode_value(["a", "b"], "test")) == ["a", "b"]

    def test_nested_dicts_and_none(self):
        from difflow.serialize import _decode_value, _encode_value

        value = {"a": 1.0, "b": None, "c": {"d": [1.0, 2.0]}}
        assert _decode_value(_encode_value(value, "test")) == value

    def test_an_unencodable_object_is_refused(self):
        from difflow.serialize import _encode_value

        class Opaque:
            pass

        with pytest.raises(SerializationError, match="no JSON form"):
            _encode_value(Opaque(), "test")


# =============================================================================
# Files
# =============================================================================


class TestFiles:
    def test_save_and_load(self, flowsheet, tmp_path):
        path = serialize.save(flowsheet, tmp_path / "plant.json")
        assert path.exists()
        reloaded = serialize.load(path)
        assert [u.name for u in reloaded.units] == ["heat", "flash"]

    def test_the_file_is_human_readable(self, flowsheet, tmp_path):
        path = serialize.save(flowsheet, tmp_path / "plant.json")
        text = path.read_text()
        assert "\n" in text, "indented, so it can be diffed"
        assert '"operation": "Heater"' in text

    def test_load_accepts_extras(self, flowsheet, thermo, tmp_path):
        path = serialize.save(flowsheet, tmp_path / "plant.json")
        reloaded = serialize.load(path, extras={"flash": {"thermo": thermo}})
        assert reloaded.units[1].operation.thermo is thermo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
