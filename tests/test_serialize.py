"""Tests for difflow.serialize.

The load-bearing test is `test_reloaded_flowsheet_solves_identically`: a
file format that reproduces the structure but not the answer is worse
than none, because the difference is invisible until it matters.

The rest check that what cannot be written faithfully is refused with a
message naming the culprit, rather than dropped.
"""

import dataclasses
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


class TestDataBuiltRateLaw:
    """A rate law built from data survives, unlike a hand-written one.

    mass_action_kinetics records the specification that produced its
    closure, so the file can carry the reactions and rebuild the
    identical function.
    """

    @pytest.fixture
    def reactor_flowsheet(self):
        from difflow import CSTR, CSTRParams, mass_action_kinetics

        kin = mass_action_kinetics([{
            "equation": "water -> ethanol",
            "reactants": {"water": 1.0}, "products": {"ethanol": 1.0},
            "rate_params": {"A": 1.0e3, "Ea": 40_000.0, "n": 0.0},
        }], SPECIES)
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream(
            {"water": 1.0, "ethanol": 0.1}, T=350.0, P=101325.0
        ))
        fs.add_unit(Unit("reactor", CSTR(CSTRParams(
            V=1.0, molar_density=1000.0, **kin.params_kwargs()
        )), ["feed"], ["out"]))
        return fs

    def test_a_reactor_round_trips(self, reactor_flowsheet):
        reloaded = serialize.from_json(serialize.to_json(reactor_flowsheet))
        assert callable(reloaded.units[0].operation.params.rate_fn)

    def test_the_reloaded_reactor_solves_identically(self, reactor_flowsheet):
        reloaded = serialize.from_json(serialize.to_json(reactor_flowsheet))
        original, restored = reactor_flowsheet.solve(), reloaded.solve()
        for stream in original:
            for key, value in original[stream].items():
                if isinstance(value, str):
                    continue
                assert float(value) == float(restored[stream][key])

    def test_the_reactions_are_what_is_written(self, reactor_flowsheet):
        """The file carries the specification, not an opaque blob."""
        data = json.loads(serialize.to_json(reactor_flowsheet))
        spec = data["units"][0]["params"]["rate_fn"]["$callable"]
        assert spec["factory"] == "mass_action_kinetics"
        assert spec["attr"] == "rate_fn"
        assert spec["kwargs"]["reactions"][0]["equation"] == "water -> ethanol"


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
# The view block (format version 2)
# =============================================================================


class TestView:
    """Presentation state rides along, and nothing numeric reads it."""

    def test_the_view_block_round_trips(self, flowsheet):
        flowsheet.view = {
            "nodes": {"reactor": {"x": 120.0, "y": 40.0},
                      "feed:feed": {"x": 20.0, "y": 40.0}},
            "code_context": "thermo = IdealThermo(...)\n",
        }
        back = serialize.from_dict(serialize.to_dict(flowsheet))
        assert back.view == flowsheet.view

    def test_a_flowsheet_with_no_view_writes_no_view_key(self, flowsheet):
        """A file that has never been opened in an editor is unchanged."""
        assert "view" not in serialize.to_dict(flowsheet)
        assert serialize.from_dict(serialize.to_dict(flowsheet)).view == {}

    def test_a_version_1_file_still_loads(self, flowsheet):
        """Files written before the view block existed keep opening."""
        data = serialize.to_dict(flowsheet)
        data["format_version"] = 1
        data.pop("view", None)
        back = serialize.from_dict(data)
        assert [u.name for u in back.units] == [u.name for u in flowsheet.units]
        assert back.view == {}

    def test_the_written_version_is_the_current_one(self, flowsheet):
        assert serialize.to_dict(flowsheet)["format_version"] == 2
        assert FORMAT_VERSION == 2
        assert set(serialize.SUPPORTED_VERSIONS) == {1, 2}

    def test_the_view_survives_an_apply_params(self, flowsheet):
        """_apply_params rebuilds the flowsheet; the canvas must survive it."""
        flowsheet.view = {"nodes": {"reactor": {"x": 1.0, "y": 2.0}}}
        applied = flowsheet._apply_params({})
        assert applied.view == flowsheet.view
        applied.view["nodes"]["reactor"]["x"] = 99.0
        assert flowsheet.view["nodes"]["reactor"]["x"] == 1.0, "copied, not shared"

    def test_the_view_holds_arbitrary_json(self, flowsheet):
        """The editor puts its planning selection here too; nothing validates."""
        flowsheet.view = {"planning": {"u": ["reactor.V"], "y": ["out.total_flow"],
                                       "bounds": {"reactor.V": [0.5, 5.0]},
                                       "radius": 0.3}}
        assert serialize.from_dict(serialize.to_dict(flowsheet)).view == flowsheet.view


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
# References into a namespace
# =============================================================================


class TestRefs:
    """`{"$ref": "thermo"}` --- how a file names an object it cannot hold.

    The editor's code context is one namespace; a script that calls
    `load(..., refs=locals())` is another. Neither is privileged: the
    mechanism lives here so a file carrying references can be opened
    with nothing but `serialize`.
    """

    def test_a_known_object_is_written_as_its_name(self, flowsheet, thermo):
        data = serialize.to_dict(flowsheet, refs={"thermo": thermo})
        assert data["units"][1]["constructor"] == {"thermo": {"$ref": "thermo"}}

    def test_it_comes_back_as_the_same_object(self, flowsheet, thermo):
        data = serialize.to_dict(flowsheet, refs={"thermo": thermo})
        back = serialize.from_dict(data, refs={"thermo": thermo})
        assert back.units[1].operation.thermo is thermo

    def test_a_different_object_of_the_same_type_is_not_a_reference(
        self, flowsheet, thermo
    ):
        """Identity, not equality: two thermos are two objects."""
        other = IdealThermo({n: get_species_data(n) for n in SPECIES})
        data = serialize.to_dict(flowsheet, refs={"thermo": other})
        assert data["units"][1]["constructor"]["thermo"] != {"$ref": "thermo"}

    def test_numbers_and_short_strings_are_never_captured(self, flowsheet):
        """Small ints and identifier-like strings are interned, so an
        identity scan run before the primitive branch would rewrite every
        1.0 in the file into whatever the namespace happened to call it."""
        data = serialize.to_dict(
            flowsheet, refs={"one": 1, "hot": "hot", "name": "Heater"}
        )
        heat = data["units"][0]
        assert heat["operation"] == "Heater"
        assert heat["outlets"] == ["hot"]
        assert json.dumps(data).count('"$ref"') == 0

    def test_a_reference_the_namespace_lacks_is_refused_by_name(self, flowsheet,
                                                                thermo):
        data = serialize.to_dict(flowsheet, refs={"thermo": thermo})
        with pytest.raises(SerializationError) as excinfo:
            serialize.from_dict(data)
        assert "thermo" in str(excinfo.value)
        assert "code context" in str(excinfo.value)

    def test_a_file_with_references_round_trips_on_disk(self, flowsheet, thermo,
                                                        tmp_path):
        path = serialize.save(flowsheet, tmp_path / "plant.json",
                              refs={"thermo": thermo})
        assert '"$ref": "thermo"' in path.read_text()
        reloaded = serialize.load(path, refs={"thermo": thermo})
        assert reloaded.units[1].operation.thermo is thermo

    def test_a_reference_survives_a_solve(self, flowsheet, thermo, tmp_path):
        """The point of the whole mechanism: the file still runs."""
        path = serialize.save(flowsheet, tmp_path / "plant.json",
                              refs={"thermo": thermo})
        before = flowsheet.solve()
        after = serialize.load(path, refs={"thermo": thermo}).solve()
        for key, value in before["vap"].items():
            if not isinstance(value, str):
                assert float(after["vap"][key]) == float(value), f"vap.{key}"

    def test_the_namespace_does_not_leak_between_calls(self, flowsheet, thermo):
        """It is a context, not a global: the next call has none of it."""
        serialize.to_dict(flowsheet, refs={"thermo": thermo})
        plain = serialize.to_dict(flowsheet)
        assert json.dumps(plain).count('"$ref"') == 0

    def test_private_names_are_not_offered(self, flowsheet, thermo):
        data = serialize.to_dict(flowsheet, refs={"_thermo": thermo})
        assert data["units"][1]["constructor"]["thermo"] != {"$ref": "_thermo"}


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


class TestPluginTypes:
    """Every shipped plugin has to be reachable by name.

    A nested ``Params`` is written as ``{"$type": "BranchParams", ...}``
    and rebuilt by searching the packages difflow ships. A plugin missing
    from that list fails only on the way *back* in, so a flowsheet saves
    without complaint and refuses to load.
    """

    @pytest.mark.parametrize("name", [
        "HeaterParams",                     # difflow
        "BioreactorParams",                 # difflow_bio
        "PipeParams",                       # difflow_gas
        "BranchParams",                     # difflow_power
    ])
    def test_a_plugin_params_class_resolves_by_name(self, name):
        found = serialize._lookup_type(name)
        assert dataclasses.is_dataclass(found)

    def test_every_shipped_plugin_is_searched(self):
        from importlib.metadata import entry_points

        plugins = {f"difflow_{e.name}"
                   for e in entry_points(group="difflow.plugins")}
        assert plugins <= set(serialize._PACKAGES), (
            "a plugin outside _PACKAGES saves fine and refuses to load")


# =============================================================================
# Units that build their own Params
# =============================================================================


class TestUnpackedParams:
    """Units that take their numbers as plain constructor arguments.

    ``Compressor(ratio)``, ``FlowSplit(w)``, ``GasPipe(beta)`` and most of
    the gas plugin build their own ``Params`` rather than being handed
    one. ``to_dict`` still writes those numbers under ``"params"``,
    because it writes whatever the instance calls ``.params`` --- so the
    value is in the file, one field away from where the loader used to
    look for it. It demanded them under ``"constructor"`` instead and
    refused a file that plainly carried them.
    """

    CASES = [
        ("Compressor", dict(ratio=1.3), "ratio", 1.3),
        ("FlowSplit", dict(w=2.0), "w", 2.0),
        ("SourceHead", dict(P_set=5.0e6), "P_set", 5.0e6),
        ("GasPipe", dict(beta=7.0), "beta", 7.0),
        ("BackPipe", dict(beta=7.0), "beta", 7.0),
        ("PressureDrivenPipe", dict(beta=7.0), "beta", 7.0),
    ]

    def _round_trip(self, name, kwargs):
        from difflow.catalog import _default_registry

        cls = _default_registry().list_operations()[name].cls
        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit(name.lower(), cls(**kwargs), ["a"], ["b"]))
        return fs, serialize.from_json(serialize.to_json(fs))

    @pytest.mark.parametrize("name,kwargs,field,value", CASES)
    def test_the_value_comes_back(self, name, kwargs, field, value):
        _, back = self._round_trip(name, kwargs)
        assert getattr(back.units[0].operation.params, field) == value

    @pytest.mark.parametrize("name,kwargs,field,value", CASES)
    def test_the_file_carries_it_under_params(self, name, kwargs, field, value):
        """Where the loader now reads it from, stated as its own fact."""
        fs, _ = self._round_trip(name, kwargs)
        written = json.loads(serialize.to_json(fs))["units"][0]
        assert written["params"][field] == value
        assert field not in written["constructor"]

    def test_defaults_the_unit_filled_in_survive_too(self):
        """Not just the required argument: the whole Params it built."""
        _, back = self._round_trip("Compressor", dict(ratio=1.3))
        params = back.units[0].operation.params
        assert (params.eta_ad, params.kappa, params.cp) == (0.8, 1.3, 2200.0)

    def test_a_non_params_argument_still_travels_as_a_constructor_extra(self):
        """The two channels compose: `ratio` by params, `direction` by extras."""
        _, back = self._round_trip(
            "CompressorBoost", dict(ratio=1.2, direction=-1)
        )
        assert back.units[0].operation.params.ratio == 1.2
        assert back.units[0].operation.direction == -1

    def test_extras_on_load_still_outrank_the_file(self):
        from difflow.catalog import _default_registry

        cls = _default_registry().list_operations()["GasPipe"].cls
        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("pipe", cls(beta=7.0), ["a"], ["b"]))
        back = serialize.from_json(
            serialize.to_json(fs), extras={"pipe": {"beta": 9.0}}
        )
        assert back.units[0].operation.params.beta == 9.0

    def test_a_genuinely_absent_argument_is_still_refused(self):
        """The message has to keep meaning what it says.

        ``Mixer(species_order)`` is an object the file does not carry, and
        widening the loader to read "params" must not turn that into a
        silent success or a different error.
        """
        from difflow.catalog import _default_registry

        cls = _default_registry().list_operations()["Mixer"].cls
        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("mix", cls(species_order=SPECIES), ["a"], ["b"]))
        text = serialize.to_json(fs)
        spec = json.loads(text)
        spec["units"][0]["constructor"] = {}
        with pytest.raises(SerializationError, match="requires species_order"):
            serialize.from_dict(spec)

    def test_tff_records_the_arguments_it_was_built_with(self):
        """A composite still has to answer for its own constructor.

        TFF holds two stages and no Params, so the writer --- which reads
        a constructor argument off the instance by attribute --- found
        nothing to write and a TFF flowsheet could not be saved at all.
        """
        from difflow_bio import TFF

        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("tff", TFF(membrane_area=4.0), ["a"], ["b"]))
        back = serialize.from_json(serialize.to_json(fs))
        assert back.units[0].operation.membrane_area == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
