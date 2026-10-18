"""difflow.serialize beyond the steady-state flowsheet basics.

Thermo kinds, functions and classes by import path (and the trust rule
for loading them), standalone objects, saved results, dynamic flowsheets,
format versions, and the constructor-call recording that makes the
dynamic units writable.
"""

import json
import textwrap

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (
    ActivityThermo,
    CubicThermo,
    Flowsheet,
    Heater,
    HeaterParams,
    IdealThermo,
    PengRobinson,
    Unit,
    codegen,
    get_critical_props,
    get_species_data,
    make_stream,
    mass_action_kinetics,
    serialize,
)
from difflow.serialize import SerializationError

GASES = ["methane", "propane"]


@pytest.fixture(scope="module")
def ideal():
    return IdealThermo({s: get_species_data(s) for s in GASES})


@pytest.fixture(scope="module")
def eos():
    return PengRobinson({s: get_critical_props(s) for s in GASES})


def round_trip(value, **load):
    return serialize.from_value(json.loads(json.dumps(serialize.to_value(value))),
                                **load)


@pytest.fixture
def user_module(tmp_path, monkeypatch):
    """A module of the user's own, importable but outside difflow."""
    (tmp_path / "mylab.py").write_text(textwrap.dedent('''
        from dataclasses import dataclass
        import jax.numpy as jnp
        from difflow import Flowsheet
        from difflow.params_mixin import ParamsMixin

        def first_order(c, T, p):
            return jnp.array([p["k"] * c["A"]])

        @dataclass
        class ScaleParams(ParamsMixin):
            factor: float = 1.0

        class Scale:
            def __init__(self, params: ScaleParams):
                self.params = params

            def __call__(self, inlet):
                return {k: v * self.params.factor if k.startswith("F_") else v
                        for k, v in inlet.items()}

        class LabFlowsheet(Flowsheet):
            pass

        class Curve:
            """Keeps its constructor arguments: rebuildable from them."""
            def __init__(self, a, b=1.0):
                self.a, self.b = a, b

        class Shifted:
            """Keeps a value *derived* from its argument under its name."""
            def __init__(self, x):
                self.x = x + 1.0
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    import mylab
    return importlib.reload(mylab)


# =============================================================================
# Thermo kinds
# =============================================================================


class TestThermo:
    def test_an_ideal_thermo_reloads_equal(self, ideal):
        """SpeciesData's tuples came back as lists, so it never compared equal."""
        assert round_trip(ideal) == ideal

    def test_a_cubic_thermo_reloads_equal(self, ideal, eos):
        thermo = CubicThermo(ideal, eos)
        assert round_trip(thermo) == thermo

    def test_an_activity_thermo_reloads_equal(self, ideal):
        from difflow.activity import WilsonParams

        wilson = WilsonParams(species=tuple(GASES), V=jnp.array([1.0, 1.2]),
                              dlam=jnp.array([[0.0, 100.0], [50.0, 0.0]]))
        thermo = ActivityThermo(ideal, wilson)
        assert round_trip(thermo) == thermo

    def test_a_dataclass_thermo_is_written_field_by_field(self):
        from difflow.combustion import IdealGasThermo

        thermo = IdealGasThermo(species=("methane", "oxygen", "carbon_dioxide", "water"))
        assert round_trip(thermo).species == thermo.species

    def test_an_unknown_thermo_is_refused_with_the_way_out(self):
        class ExoticThermo:
            pass

        with pytest.raises(SerializationError, match="extras="):
            serialize.to_value(ExoticThermo())


# =============================================================================
# Functions and classes by import path
# =============================================================================


class TestImportPaths:
    def test_a_difflow_function_is_written_by_path(self):
        from difflow.kinetics import mass_action_kinetics as fn

        data = serialize.to_value(fn)
        assert data == {"$callable": {"function":
                                      "difflow.kinetics:mass_action_kinetics"}}
        assert serialize.from_value(data) is fn

    def test_a_lambda_is_refused_with_the_alternatives(self):
        with pytest.raises(SerializationError, match="refs="):
            serialize.to_value(lambda x: x)

    def test_your_own_function_needs_trust_to_load(self, user_module):
        data = serialize.to_value(user_module.first_order)
        with pytest.raises(SerializationError, match="trust=True"):
            serialize.from_value(data)
        assert serialize.from_value(data, trust=True) is user_module.first_order

    def test_your_own_unit_params_and_flowsheet_class_round_trip(self, user_module):
        fs = user_module.LabFlowsheet(species_order=["A"])
        fs.add_feed("f", make_stream({"A": 1.0}, T=300.0, P=101325.0))
        fs.add_unit(Unit("s", user_module.Scale(user_module.ScaleParams(2.0)),
                         ["f"], ["o"]))
        back = serialize.from_json(serialize.to_json(fs), trust=True)
        assert type(back) is user_module.LabFlowsheet
        assert type(back.units[0].operation) is user_module.Scale
        assert back.units[0].operation.params.factor == 2.0

    def test_an_object_rebuilt_from_its_attributes(self, user_module):
        curve = user_module.Curve(2.0, b=3.0)
        back = round_trip(curve, trust=True)
        assert type(back) is user_module.Curve and (back.a, back.b) == (2.0, 3.0)

    def test_an_attribute_that_is_not_its_argument_is_refused(self, user_module):
        """Rebuilt from x it would come back x + 2: refused, not written wrong."""
        with pytest.raises(SerializationError, match="no JSON form"):
            serialize.to_value(user_module.Shifted(1.0))

    def test_codegen_imports_your_module(self, user_module):
        fs = Flowsheet(species_order=["A"])
        fs.add_unit(Unit("s", user_module.Scale(user_module.ScaleParams(2.0)),
                         ["a"], ["b"]))
        assert "from mylab import Scale, ScaleParams" in codegen.to_python(fs)


# =============================================================================
# Traced values
# =============================================================================


def test_a_traced_value_is_refused_by_name():
    def build(T):
        fs = Flowsheet(species_order=["water"])
        fs.add_unit(Unit("h", Heater(HeaterParams(T_out=T)), ["a"], ["b"]))
        with pytest.raises(SerializationError, match="traced value"):
            serialize.to_json(fs)
        return T

    jax.grad(build)(300.0)


# =============================================================================
# Standalone objects
# =============================================================================


class TestObjects:
    def test_a_reaction_set_round_trips(self):
        kin = mass_action_kinetics([{"reactants": {"A": 1.0}, "products": {"B": 1.0},
                                     "rate_params": {"A": 0.1}}], species_order=["A", "B"])
        back = serialize.loads_object(serialize.dumps_object(kin))
        assert back.equations == kin.equations
        c = {"A": 1.0, "B": 0.0}
        assert float(back.rate_fn(c, 350.0, back.rate_params)[0]) == \
            float(kin.rate_fn(c, 350.0, kin.rate_params)[0])

    def test_save_object_and_load_object(self, tmp_path, ideal):
        path = serialize.save_object(ideal, tmp_path / "thermo.json")
        assert serialize.load_object(path) == ideal

    def test_a_flowsheet_document_is_not_an_object(self):
        text = serialize.to_json(Flowsheet(species_order=["A"]))
        with pytest.raises(SerializationError, match="looks like a flowsheet"):
            serialize.loads_object(text)

    def test_save_refuses_a_plain_value(self):
        with pytest.raises(SerializationError, match="save_object"):
            serialize.to_dict({"not": "a model"})


# =============================================================================
# Saved results
# =============================================================================


def _recycle(ideal_water):
    from difflow import Mixer, Splitter

    species = ["water", "ethanol"]
    fs = Flowsheet(species_order=species)
    fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.5},
                                    T=350.0, P=101325.0))
    fs.add_unit(Unit("mix", Mixer(species), ["feed", "back"], ["mixed"]))
    fs.add_unit(Unit("split", Splitter(species), ["mixed"], ["back", "out"],
                     params={"split_frac": 0.5}))
    fs.add_recycle("back", "back")
    return fs


class TestResults:
    def test_results_reload_and_warm_start(self):
        fs = _recycle(None)
        streams = fs.solve(tol=1e-10, max_iter=200)
        back = serialize.from_json(serialize.to_json(fs, results=streams))
        assert float(back.saved_results["out"]["F_water"]) == \
            float(streams["out"]["F_water"])
        assert back.saved_solve_info["converged"] is True
        assert list(serialize.warm_start(back)) == ["back"]
        back.solve(tol=1e-10, max_iter=200, tear_initial=serialize.warm_start(back))
        assert back.last_solve_iterations <= 1

    def test_no_results_writes_no_block(self):
        assert "results" not in serialize.to_dict(_recycle(None))

    def test_a_flowsheet_without_results_warm_starts_from_nothing(self):
        assert serialize.warm_start(_recycle(None)) == {}


# =============================================================================
# Dynamic flowsheets
# =============================================================================


class TestDynamic:
    def _build(self, rate):
        from difflow.dynamic import DynamicCSTR, DynamicFlowsheet, DynamicTank

        fs = DynamicFlowsheet(species_order=["A", "B"])
        fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0))
        fs.add_unit(DynamicCSTR(volume=1.0, rate_fn=rate,
                                stoich=jnp.array([[-1.0], [1.0]]),
                                species_order=["A", "B"], rate_params={"k": 0.1},
                                name="reactor"),
                    inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(DynamicTank(max_volume=10.0, species_order=["A", "B"],
                                name="storage"),
                    inlet_names=["reactor_out"], outlet_names=["product"])
        return fs

    def test_it_reloads_and_simulates_identically(self, user_module):
        fs = self._build(user_module.first_order)
        back = serialize.from_json(serialize.to_json(fs), trust=True)
        a = fs.simulate(t_span=(0.0, 50.0), method="RK4", n_steps=50)
        b = back.simulate(t_span=(0.0, 50.0), method="RK4", n_steps=50)
        assert bool(jnp.all(a.y_final == b.y_final))

    def test_a_local_rate_law_travels_by_ref(self):
        def rate(c, T, p):
            return jnp.array([p["k"] * c["A"]])

        text = serialize.to_json(self._build(rate), refs={"rate": rate})
        back = serialize.from_json(text, refs={"rate": rate})
        assert back.units[0].unit.params["rate_fn"] is rate

    def test_codegen_says_what_to_use_instead(self):
        with pytest.raises(codegen.CodegenError, match="serialize.save"):
            codegen.to_python(self._build(None))


# =============================================================================
# Format versions
# =============================================================================


class TestVersions:
    def test_a_plain_flowsheet_is_still_version_2(self):
        """So a release that reads only 2 keeps opening it."""
        fs = Flowsheet(species_order=["water"])
        fs.add_unit(Unit("h", Heater(HeaterParams(T_out=360.0)), ["a"], ["b"]))
        assert serialize.to_dict(fs)["format_version"] == 2

    def test_an_eos_makes_it_version_3(self, eos):
        """A version 2 reader would build the unit with a dict for an EOS."""
        assert serialize.version_for({"x": serialize.to_value(eos)}) == 3


# =============================================================================
# Constructor-call recording
# =============================================================================


class TestInitArgs:
    def test_the_outermost_call_is_recorded(self):
        from difflow.init_args import init_args, record_init_args

        @record_init_args
        class Base:
            def __init__(self, params, name=None):
                self.params = params

        class Child(Base):
            def __init__(self, volume, name=None):
                super().__init__({"V": volume}, name)

        record_init_args(Child)
        assert init_args(Child(2.0, name="c")) == {"volume": 2.0, "name": "c"}

    def test_a_class_that_does_not_record_reports_none(self):
        from difflow.init_args import init_args

        assert init_args(object()) is None
