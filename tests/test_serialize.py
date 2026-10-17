"""Tests for difflow.serialize: round trips, and refusals that name the culprit."""

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

# =============================================================================
# Thermodynamics
# =============================================================================


class TestThermo:
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

    def test_the_reloaded_reactor_solves_identically(self, reactor_flowsheet):
        reloaded = serialize.from_json(serialize.to_json(reactor_flowsheet))
        original, restored = reactor_flowsheet.solve(), reloaded.solve()
        for stream in original:
            for key, value in original[stream].items():
                if isinstance(value, str):
                    continue
                assert float(value) == float(restored[stream][key])

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

    def test_a_version_1_file_still_loads(self, flowsheet):
        """Files written before the view block existed keep opening."""
        data = serialize.to_dict(flowsheet)
        data["format_version"] = 1
        data.pop("view", None)
        back = serialize.from_dict(data)
        assert [u.name for u in back.units] == [u.name for u in flowsheet.units]
        assert back.view == {}

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

    def test_an_unencodable_object_is_refused(self):
        from difflow.serialize import _encode_value

        class Opaque:
            pass

        with pytest.raises(SerializationError, match="no JSON form"):
            _encode_value(Opaque(), "test")


class TestNonFiniteFloats:
    """A saved file has to be JSON that something other than Python reads.

    Python's `json` writes the non-finite floats as the bare tokens
    `Infinity` / `NaN`, which are not JSON: `JSON.parse`, `jq` and every
    strict reader reject the whole file. And this is the common case --
    `mass_action_kinetics` puts `inf` in `K_eq` for every irreversible
    reaction.
    """

    @staticmethod
    def _strict(text):
        def reject(token):
            raise AssertionError(f"bare {token} is not valid JSON")

        return json.loads(text, parse_constant=reject)

    def test_a_saved_flowsheet_with_an_irreversible_reaction_is_valid_json(self):
        from difflow.kinetics import mass_action_kinetics
        from difflow import CSTR, CSTRParams

        reactions = mass_action_kinetics(
            [{"equation": "A -> B",
              "reactants": {"A": 1.0}, "products": {"B": 1.0},
              "rate_params": {"A": 0.5, "Ea": 0.0}}],
            species_order=["A", "B"],
        )
        assert bool(jnp.isinf(reactions.rate_params["K_eq"]).all())

        fs = Flowsheet(species_order=["A", "B"])
        fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0))
        fs.add_unit(Unit(
            "r",
            CSTR(CSTRParams(V=1.0, rate_fn=reactions.rate_fn,
                            stoich=reactions.stoich,
                            rate_params=reactions.rate_params,
                            species_order=["A", "B"], molar_density=55500.0)),
            ["feed"], ["out"],
        ))
        text = serialize.to_json(fs)

        self._strict(text)
        back = serialize.from_json(text)
        assert bool(jnp.isinf(
            back.units[0].operation.params.rate_params["K_eq"]).all())

    def test_files_written_with_the_bare_tokens_still_load(self, flowsheet):
        """Python's reader accepts them, so files written before the tag
        -- which hold `Infinity` -- keep loading unchanged."""
        legacy = serialize.to_json(flowsheet).replace(
            '"T_out": 360.0', '"T_out": Infinity')

        back = serialize.from_json(legacy)

        assert back.units[0].operation.params.T_out == float("inf")


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

# =============================================================================
# Files
# =============================================================================


class TestPluginTypes:
    """Every shipped plugin has to be reachable by name.

    A nested ``Params`` is written as ``{"$type": "BranchParams", ...}``
    and rebuilt by searching the packages difflow ships. A plugin missing
    from that list fails only on the way *back* in, so a flowsheet saves
    without complaint and refuses to load.
    """

    def test_a_plugin_params_class_resolves_by_name(self):
        found = serialize._lookup_type("BranchParams")
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

    def _round_trip(self, name, kwargs):
        from difflow.catalog import _default_registry

        cls = _default_registry().list_operations()[name].cls
        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit(name.lower(), cls(**kwargs), ["a"], ["b"]))
        return fs, serialize.from_json(serialize.to_json(fs))

    def test_the_value_comes_back(self):
        _, back = self._round_trip("GasPipe", dict(beta=7.0))
        assert back.units[0].operation.params.beta == 7.0

    def test_a_non_params_argument_still_travels_as_a_constructor_extra(self):
        """The two channels compose: `ratio` by params, `direction` by extras."""
        _, back = self._round_trip(
            "CompressorBoost", dict(ratio=1.2, direction=-1)
        )
        assert back.units[0].operation.params.ratio == 1.2
        assert back.units[0].operation.direction == -1

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



class TestOptionalConstructorArguments:
    """An optional constructor argument set away from its default is written.

    Only required ones used to be, so ``Mixer(species, thermo)`` reloaded
    as ``Mixer(species)``: it mixed temperatures without an enthalpy
    balance, and the documented isomerisation recycle reloaded to a
    reactor-feed temperature 1.8e-3 K off the original.
    """

    def _mixer_flowsheet(self, mixer):
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("a", make_stream({"water": 1.0, "ethanol": 0.0},
                                     T=300.0, P=101325.0))
        fs.add_feed("b", make_stream({"water": 0.0, "ethanol": 1.0},
                                     T=360.0, P=101325.0))
        fs.add_unit(Unit("mix", mixer, ["a", "b"], ["out"]))
        return fs

    def test_mixer_thermo_survives_and_solves_identically(self, thermo):
        from difflow import Mixer

        fs = self._mixer_flowsheet(Mixer(SPECIES, thermo))
        back = serialize.from_json(serialize.to_json(fs))
        assert back.units[0].operation.thermo is not None
        assert float(fs.solve()["out"]["T"]) == float(back.solve()["out"]["T"])

    def test_an_argument_at_its_default_is_not_written(self):
        from difflow import Mixer

        spec = serialize.to_dict(self._mixer_flowsheet(Mixer(SPECIES)))
        assert set(spec["units"][0]["constructor"]) == {"species_order"}

    def test_an_unwritable_one_warns_rather_than_refuses(self):
        from difflow import Mixer

        class ExoticThermo:
            pass

        fs = self._mixer_flowsheet(Mixer(SPECIES, ExoticThermo()))
        with pytest.warns(serialize.SerializationWarning, match="thermo"):
            text = serialize.to_json(fs)
        assert serialize.from_json(text).units[0].operation.thermo is None

    def test_codegen_emits_it_as_a_keyword(self, thermo):
        from difflow import Mixer, codegen

        source = codegen.to_python(self._mixer_flowsheet(Mixer(SPECIES, thermo)))
        assert "thermo=thermo_mix" in source

    def test_a_ref_is_written_for_an_optional_argument(self, thermo):
        from difflow import Mixer

        fs = self._mixer_flowsheet(Mixer(SPECIES, thermo))
        text = serialize.to_json(fs, refs={"thermo": thermo})
        assert '"$ref": "thermo"' in text
        back = serialize.from_json(text, refs={"thermo": thermo})
        assert back.units[0].operation.thermo is thermo

    def test_a_hand_written_bpr_fn_warns_rather_than_vanishing(self):
        """The evaporators kept it only inside a private helper, unseen."""
        from difflow.units.evaporator import Evaporator, EvaporatorParams

        op = Evaporator(EvaporatorParams(solute_species=["ethanol"],
                                         solute_MW=[46.0], A=10.0),
                        bpr_fn=lambda x, Tw, P: 0.0)
        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("evap", op, ["a"], ["b"]))
        with pytest.warns(serialize.SerializationWarning, match="bpr_fn"):
            serialize.to_json(fs)

    def test_membrane_stage_params_survive(self):
        """MultistageMembrane kept per-stage params only as private stages."""
        from difflow_cc import MembraneParams, MultistageMembrane

        stages = [MembraneParams(membrane_type="Matrimid"),
                  MembraneParams(membrane_type="Matrimid", area=2.0)]
        op = MultistageMembrane(stages[0], n_stages=2, stage_params=stages)
        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("mem", op, ["a"], ["b"]))
        back = serialize.from_json(serialize.to_json(fs)).units[0].operation
        assert back.stage_params == stages

    def test_every_optional_argument_is_readable_by_name(self):
        """The convention optional_extras relies on, for every operation.

        An optional constructor argument the instance keeps only under
        another name is invisible to the writer, so it is dropped on save
        without a word: the evaporators' ``bpr_fn`` and the membrane's
        ``stage_params`` were. It must be an attribute of the same name,
        a dataclass field, or a field of the unit's ``Params``.
        """
        import inspect
        import re

        from difflow.catalog import _default_registry, _params_class
        from difflow.plugins import load_plugins
        from difflow.serialize import _takes_params_first

        load_plugins()
        unreadable = {}
        for name, info in _default_registry().list_operations().items():
            cls = info.cls
            try:
                sig = inspect.signature(cls.__init__)
            except (TypeError, ValueError):
                continue
            optional = [
                a for a, p in sig.parameters.items()
                if a != "self" and p.default is not inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            if not optional:
                continue
            source = "".join(
                inspect.getsource(k) for k in cls.__mro__
                if k.__module__.split(".")[0].startswith("difflow")
            )
            known = set(re.findall(r"self\.(\w+)\s*(?::[^=\n]+)?=", source))
            known |= set(re.findall(r"def (\w+)\(self\)", source))
            if not _takes_params_first(sig):
                # builds its own Params, which is written: `X(dp_pa=dp_pa)`
                known |= set(re.findall(r"\b(\w+)=\1\b", source))
            for owner in (cls, _params_class(cls)):
                if owner is not None and dataclasses.is_dataclass(owner):
                    known |= {f.name for f in dataclasses.fields(owner)}
            missing = [a for a in optional if a not in known]
            if missing:
                unreadable[name] = missing
        assert not unreadable, unreadable


class TestEquationOfState:
    """PengRobinson/SRK are written as the arguments that build them.

    Before, ``EOSFlash`` (whose ``eos`` is required) could not be saved at
    all, and ``Flash(eos=...)`` warned and reloaded as an ideal flash.
    """

    GASES = ["methane", "propane"]

    def _eos_flash(self, eos):
        from difflow.units.flash import EOSFlash, EOSFlashParams

        fs = Flowsheet(species_order=self.GASES)
        fs.add_feed("f", make_stream({"methane": 1.0, "propane": 1.0},
                                     T=250.0, P=2.0e6))
        fs.add_unit(Unit("fl", EOSFlash(EOSFlashParams(species_order=self.GASES),
                                        eos), ["f"], ["l", "v"]))
        return fs

    def _props(self):
        from difflow import get_critical_props

        return {s: get_critical_props(s) for s in self.GASES}

    def test_an_eos_flash_reloads_and_solves_identically(self):
        from difflow import PengRobinson

        fs = self._eos_flash(PengRobinson(self._props()))
        back = serialize.from_json(serialize.to_json(fs))
        original, restored = fs.solve(), back.solve()
        for stream in original:
            for key, value in original[stream].items():
                if not isinstance(value, str):
                    assert float(value) == float(restored[stream][key])

    def test_srk_with_a_kij_dict_comes_back_equal(self):
        """Tuple keys have no JSON form; they travel as [i, j, value]."""
        from difflow import SRK

        eos = SRK(self._props(), k_ij={("methane", "propane"): 0.02})
        back = serialize.from_json(serialize.to_json(self._eos_flash(eos)))
        rebuilt = back.units[0].operation.eos
        assert type(rebuilt) is SRK and rebuilt == eos

    def test_an_optional_eos_no_longer_warns(self, thermo):
        import warnings

        from difflow import PengRobinson

        props = self._props()
        fs = Flowsheet(species_order=self.GASES)
        fs.add_unit(Unit("fl", Flash(FlashParams(species_order=self.GASES),
                                     IdealThermo({s: get_species_data(s)
                                                  for s in self.GASES}),
                                     eos=PengRobinson(props)), ["a"], ["l", "v"]))
        with warnings.catch_warnings():
            warnings.simplefilter("error", serialize.SerializationWarning)
            back = serialize.from_json(serialize.to_json(fs))
        assert type(back.units[0].operation.eos) is PengRobinson

    def test_a_subclass_is_not_written_as_its_base(self):
        """Rebuilding it as PengRobinson would quietly drop what it adds."""
        from difflow import PengRobinson

        class Tuned(PengRobinson):
            pass

        with pytest.raises(SerializationError, match="Tuned"):
            serialize.to_json(self._eos_flash(Tuned(self._props())))

    def test_an_unknown_kind_is_refused_by_name(self):
        from difflow import PengRobinson

        spec = serialize.to_dict(self._eos_flash(PengRobinson(self._props())))
        spec["units"][0]["constructor"]["eos"]["$eos"] = "VanDerWaals"
        with pytest.raises(SerializationError, match="VanDerWaals"):
            serialize.from_dict(spec)
