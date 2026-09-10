"""Tests for difflow.catalog.

The catalog is derived from the code by introspection rather than from
a hand-maintained table, so these tests mostly check that the
derivation agrees with what the units actually do --- particularly the
port arity, where the trap is counting the info dict every unit returns
as though it were an outlet stream.
"""

import importlib
import inspect
import json
import re

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import difflow
from difflow.catalog import (
    CORE_NAME_OVERRIDES,
    OperationSchema,
    catalog,
    core_operations,
    describe_class,
    describe_operation,
    register_core_operations,
)
from difflow.plugins import OperationRegistry


@pytest.fixture(scope="module")
def cat():
    return catalog()


# =============================================================================
# Core registration
# =============================================================================


class TestCoreRegistration:
    def test_core_units_are_in_the_catalog(self, cat):
        """The registry previously held only plugin units."""
        for name in ["CSTR", "PFR", "Flash", "Mixer", "Splitter", "Heater",
                     "Cooler", "ShortcutColumn", "CounterCurrentHX"]:
            assert name in cat, f"{name} missing from the catalog"
            assert cat[name].plugin == "core"

    def test_registration_happens_on_import(self):
        """`import difflow` is enough; no explicit call required."""
        assert "CSTR" in catalog()

    def test_plugin_units_are_still_there(self, cat):
        plugins = {s.plugin for s in cat.values()}
        assert {"core", "difflow_gas", "difflow_bio",
                "difflow_cc", "difflow_ree"} <= plugins

    def test_core_operations_are_discovered_not_listed(self):
        """Discovery reads __all__, so a new unit joins automatically."""
        found = core_operations()
        assert len(found) > 25
        for cls in found.values():
            assert cls.__module__.startswith("difflow.units")

    def test_base_classes_are_excluded(self):
        found = core_operations()
        assert "UnitBase" not in found
        assert "ReactorBase" not in found

    def test_the_compressor_name_clash_is_resolved(self, cat):
        """difflow_gas registers its own Compressor; neither may be lost.

        Plugins load after the core, so registering both under the bare
        name would silently overwrite the EOS one.
        """
        assert CORE_NAME_OVERRIDES["Compressor"] == "EOSCompressor"
        assert cat["EOSCompressor"].plugin == "core"
        assert cat["EOSCompressor"].module.endswith("eos_units")
        assert cat["Compressor"].plugin == "difflow_gas"

    def test_registering_into_a_fresh_registry(self):
        registry = OperationRegistry()
        count = register_core_operations(registry)
        assert count == len(registry.list_operations()) == len(core_operations())


# =============================================================================
# Ports
# =============================================================================


class TestPorts:
    @pytest.mark.parametrize("name,inlets,n_outlets", [
        ("CSTR", ["inlet"], 1),
        ("PFR", ["inlet"], 1),
        ("Flash", ["inlet"], 2),
        ("EOSFlash", ["inlet"], 2),
        ("Heater", ["inlet"], 1),
        ("ShortcutColumn", ["feed"], 2),
        ("CounterCurrentHX", ["hot_inlet", "cold_inlet"], 2),
        ("MultistageCascade", ["feed", "solvent"], 2),
    ])
    def test_arity_matches_the_unit(self, cat, name, inlets, n_outlets):
        ports = cat[name].ports
        assert ports.inlets == inlets
        assert ports.n_outlets == n_outlets

    def test_the_info_dict_is_not_counted_as_an_outlet(self, cat):
        """Every unit returns (outlets..., info); info is `dict[str, Array]`.

        Stream is `dict[str, Array | float]`, so a loose match on the
        return annotation counts the info payload as a third stream.
        """
        assert cat["Flash"].ports.n_outlets == 2       # not 3
        assert cat["CSTR"].ports.n_outlets == 1        # not 2

    def test_a_mixer_is_variadic(self, cat):
        ports = cat["Mixer"].ports
        assert ports.variadic
        assert ports.n_inlets is None
        assert ports.n_outlets == 1

    def test_a_fixed_arity_unit_reports_its_inlet_count(self, cat):
        assert cat["CounterCurrentHX"].ports.n_inlets == 2
        assert not cat["CounterCurrentHX"].ports.variadic

    def test_string_annotations_are_understood(self, cat):
        """Modules using `from __future__ import annotations` keep strings."""
        sig = inspect.signature(difflow.Compressor.__call__)
        assert isinstance(sig.return_annotation, str), "premise: a string annotation"
        assert cat["EOSCompressor"].ports.inlets == ["inlet"]
        assert cat["EOSCompressor"].ports.n_outlets == 1

    def test_unknown_arity_is_reported_not_guessed(self, cat):
        """Splitter returns a bare `tuple`, so its outlet count is unknown."""
        assert cat["Splitter"].ports.n_outlets is None

    def test_a_lone_stream_return_is_one_outlet(self, cat):
        """The gas units return a Stream, not a (Stream, info) tuple.

        Reading only tuples left eleven of them with no outlet at all,
        and the canvas drew them from its fallback rather than from the
        signature.
        """
        for name in ("GasPipe", "Compressor", "OpenValve", "SourceHead"):
            assert cat[name].ports.n_outlets == 1, name
            assert cat[name].ports.inlets == ["inlet"], name

    def test_a_class_with_no_call_of_its_own_has_no_ports(self, cat):
        """`cls.__call__` on such a class reaches the metaclass slot.

        Its signature is `(*args, **kwargs)`, which reads back as a unit
        taking any number of inlets -- drawing a mixer where the catalog
        holds a model object with named methods.
        """
        for name in ("LLEEquilibrium", "TFF"):
            spec = cat[name]
            cls = getattr(importlib.import_module(spec.module), spec.class_name)
            assert "__call__" not in cls.__dict__, f"premise: {name}"
            ports = spec.ports
            assert ports.inlets == [], name
            assert not ports.variadic, name

    def test_the_ports_left_unknown_are_the_ones_that_cannot_be_known(self, cat):
        """Guard against the count creeping back up.

        What remains is honest: the REE circuits return a dict of named
        streams, Splitter's width is a call argument, and two entries are
        model objects with no `__call__` at all.
        """
        unknown = {n for n, s in cat.items() if s.ports.n_outlets is None}
        assert unknown == {
            "ExtractStripCircuit", "ExtractScrubStripCircuit",
            "FullSeparationTrain", "SplitShellCascade",
            "Splitter", "LLEEquilibrium", "TFF",
        }


# =============================================================================
# Parameters
# =============================================================================


class TestParameters:
    def test_required_and_optional_are_distinguished(self, cat):
        spec = cat["CSTR"]
        assert "V" in spec.required_parameters()
        assert "T_damping" not in spec.required_parameters()

    def test_defaults_are_captured(self, cat):
        by_name = {p.name: p for p in cat["CSTR"].parameters}
        assert by_name["T_damping"].default == "0.3"
        assert by_name["V"].default is None

    def test_a_params_annotation_in_quotes_still_resolves(self, cat):
        """`params: "SplitParams"` arrives here as a string WITH its quotes.

        Under ``from __future__ import annotations`` an already-quoted
        annotation is stringified a second time, so the lookup name is
        ``'SplitParams'`` and not ``SplitParams``. The failure is silent
        and total: no Params class means no parameters, so the operation
        offers an empty inspector and cannot be dropped at all.
        """
        pytest.importorskip("difflow_power")
        spec = cat.get("PowerSplit")
        if spec is None:
            pytest.skip("difflow_power not registered")
        assert spec.params_class == "SplitParams"
        assert [p.name for p in spec.parameters] == ["fraction"]

    def test_every_operation_that_takes_params_found_the_class(self, cat):
        """An operation whose ``__init__`` wants a Params must resolve one.

        The quoted-annotation case was invisible precisely because a
        missing Params class looks like an operation that takes none.
        """
        import inspect

        from difflow.catalog import _default_registry

        unresolved = []
        for name, info in _default_registry().list_operations().items():
            try:
                sig = inspect.signature(info.cls.__init__)
            except (TypeError, ValueError):
                continue
            args = [p for n, p in sig.parameters.items() if n != "self"]
            if not args:
                continue
            wants = "Params" in str(args[0].annotation)
            if wants and cat[name].params_class is None:
                unresolved.append((name, args[0].annotation))
        assert not unresolved

    def test_callable_fields_are_flagged(self, cat):
        """The fields a form cannot fill in."""
        assert set(cat["CSTR"].callable_parameters) == {
            "rate_fn", "eos", "H_mix_fn", "K_eq_fn"
        }
        assert not cat["CSTR"].is_declarative

    def test_most_operations_are_declarative(self, cat):
        non_declarative = [n for n, s in cat.items() if not s.is_declarative]
        assert len(non_declarative) < 10, non_declarative
        # every one of them is a reactor, and it is always the rate law
        for name in non_declarative:
            assert any(
                "fn" in p for p in cat[name].callable_parameters
            ), f"{name}: {cat[name].callable_parameters}"

    def test_params_class_is_found(self, cat):
        assert cat["CSTR"].params_class == "CSTRParams"
        assert cat["Flash"].params_class == "FlashParams"

    def test_a_unit_without_params_has_none(self, cat):
        assert cat["Mixer"].parameters == []


# =============================================================================
# Buildability
# =============================================================================


class TestBuildability:
    """What a form --- difflow.gui's palette --- can construct unaided."""

    def test_constructor_extras_are_reported(self, cat):
        assert cat["Flash"].constructor_extras == ["thermo"]
        assert cat["Mixer"].constructor_extras == ["species_order"]
        assert cat["Heater"].constructor_extras == []

    def test_an_object_argument_blocks_building(self, cat):
        assert not cat["Flash"].is_buildable, "a thermo is not data"

    def test_a_required_callable_blocks_building(self, cat):
        assert not cat["CSTR"].is_buildable, "a rate law is not data"

    def test_an_optional_callable_does_not_block_building(self, cat):
        """Weaker than is_declarative, deliberately.

        A field like Flash's optional ``eos`` is no obstacle to building
        the unit --- only to filling that one field --- so it must not
        cost the operation its place in the palette.
        """
        optional_only = [
            n for n, s in cat.items()
            if s.callable_parameters and s.is_buildable
        ]
        for name in optional_only:
            spec = cat[name]
            assert not [
                p.name for p in spec.parameters if p.required and p.is_callable
            ], name
            assert not spec.is_declarative, name

    def test_a_useful_number_of_operations_are_buildable(self, cat):
        """A palette that could add almost nothing would not be worth one."""
        buildable = [n for n, s in cat.items() if s.is_buildable]
        assert len(buildable) > len(cat) // 2, f"only {len(buildable)} of {len(cat)}"

    def test_buildability_is_in_the_json_payload(self, cat):
        payload = cat["Flash"].to_dict()
        assert payload["buildable"] is False
        assert payload["constructor_extras"] == ["thermo"]


# =============================================================================
# Schema
# =============================================================================


class TestSchema:
    def test_is_json_serializable(self, cat):
        payload = json.dumps({n: s.to_dict() for n, s in cat.items()})
        assert len(payload) > 1000
        assert json.loads(payload)["Flash"]["ports"]["n_outlets"] == 2

    def test_equations_are_carried_through(self, cat):
        assert cat["CSTR"].equations, "CSTR declares equations"
        assert any("\\" in e for e in cat["CSTR"].equations), "LaTeX expected"

    def test_describe_class_works_on_an_unregistered_class(self):
        """Usable on a plugin's units before they are wired in."""
        spec = describe_class(difflow.Flash, category="separations")
        assert isinstance(spec, OperationSchema)
        assert spec.name == "Flash"
        assert spec.ports.n_outlets == 2

    def test_describe_operation_rejects_an_unknown_name(self):
        with pytest.raises(KeyError, match="no operation registered"):
            describe_operation("NotAUnit")

    def test_catalog_filters_by_category(self, cat):
        reactors = catalog(category="reactors")
        assert "CSTR" in reactors and "PFR" in reactors
        assert "Flash" not in reactors
        assert all(s.category == "reactors" for s in reactors.values())

    def test_categories_are_assigned_to_core_units(self, cat):
        assert cat["CSTR"].category == "reactors"
        assert cat["CounterCurrentHX"].category == "heat_transfer"
        assert cat["ShortcutColumn"].category == "distillation"

    def test_description_is_a_single_line(self, cat):
        for name, spec in cat.items():
            assert "\n" not in spec.description, name


# =============================================================================
# Parameter documentation
# =============================================================================


class TestParameterDocumentation:
    """#213: nothing populates the ``description`` field metadata the
    catalog read, so every parameter of every operation carried
    ``description=None``. The descriptions are written --- in the
    ``Params`` class's own ``Attributes:`` section and the comments
    around its fields --- and :mod:`difflow.docstrings` is what carries
    them across. These tests keep that pathway from silently going quiet
    again. Units are not read from the prose; see :class:`TestMetadata`.
    """

    def test_a_representative_sample_is_described(self, cat):
        sample = {
            ("CSTR", "V"): "Reactor volume",
            ("Flash", "species_order"): "species",
            ("ShortcutColumn", "light_key"): "light key",
            ("DistillationColumn", "n_stages"): "stages",
            ("CounterCurrentHX", "UA"): "heat transfer",
            ("Heater", "duty"): "heat duty",
        }
        for (op, param), expected in sample.items():
            spec = next(p for p in cat[op].parameters if p.name == param)
            assert spec.description, f"{op}.{param} has no description"
            assert expected.lower() in spec.description.lower(), \
                f"{op}.{param}: {spec.description!r}"

    def test_almost_every_parameter_is_described(self, cat):
        """Not all of them: a handful of fields are genuinely undocumented.

        The bar is a fraction rather than a count so that adding a unit
        does not fail the suite, while the pathway breaking does ---
        before #213 this was exactly 0.
        """
        params = [p for spec in cat.values() for p in spec.parameters]
        described = [p for p in params if p.description]
        assert len(described) / len(params) > 0.9, (
            f"only {len(described)}/{len(params)} parameters are described; "
            "the docstring pathway looks broken"
        )

    def test_field_metadata_wins_over_the_docstring(self):
        """Option 2 from #213 stays available per field."""
        from dataclasses import dataclass, field

        from difflow.params_mixin import ParamsMixin

        @dataclass
        class ExplicitParams(ParamsMixin):
            """Summary.

            Attributes:
                P: Column pressure (Pa)
            """

            P: float = field(
                default=101325.0, metadata={"description": "Overridden"},
            )

        class Explicit:
            """A unit."""

            def __init__(self, params: ExplicitParams):
                self.params = params

        assert describe_class(Explicit).parameters[0].description == "Overridden"

    def test_a_comment_documents_a_field_the_docstring_misses(self, cat):
        """35 fields are documented beside themselves, not in the section."""
        by_name = {p.name: p for p in cat["CSTR"].parameters}
        assert by_name["eos"].description.startswith("Optional cubic EOS")
        assert by_name["outlet_volumetric_basis"].description.startswith(
            "Volumetric-flow basis"
        )

    def test_descriptions_survive_the_json_payload(self, cat):
        payload = json.loads(json.dumps(cat["CSTR"].to_dict()))
        volume = next(p for p in payload["parameters"] if p["name"] == "V")
        assert volume["description"] == "Reactor volume (m^3)"


# =============================================================================
# The metadata contract
# =============================================================================


class TestMetadata:
    """The catalog reads the metadata contract the units already carry.

    ``symbol``, ``assumptions``, ``references``, ``numerical_method``
    and the per-parameter units and symbols were declared on the unit
    classes for the report writer and read by nobody else; an inspector
    wants exactly them, and a second hand-maintained table would drift.
    """

    def test_the_whole_docstring_is_carried_not_just_its_first_line(self, cat):
        spec = cat["CSTR"]
        assert spec.description == spec.doc.splitlines()[0]
        assert len(spec.doc.splitlines()) > 1
        assert not spec.doc.startswith(" "), "cleaned of its indentation"

    def test_every_operation_has_a_docstring(self, cat):
        for name, spec in cat.items():
            assert spec.doc.strip(), name

    def test_symbol_falls_back_to_the_class_name(self, cat):
        assert cat["CSTR"].symbol == "CSTR"
        for name, spec in cat.items():
            assert spec.symbol, name

    def test_assumptions_and_references_come_through(self, cat):
        assert len(cat["CSTR"].assumptions) >= 3
        assert cat["CSTR"].references
        assert cat["CSTR"].numerical_method

    def test_parameter_units_come_from_the_class(self, cat):
        """No `Params` field declares `metadata={"units": ...}`, but 54
        unit classes declare `parameter_units`. Reading them is the
        difference between a form with units on it and one without."""
        by_name = {p.name: p for p in cat["CSTR"].parameters}
        assert by_name["V"].units == "m^3"
        assert by_name["dH_rxn"].units == "J/mol"
        assert by_name["rate_fn"].units is None, "a callable has no units"

    def test_parameter_symbols_come_through(self, cat):
        by_name = {p.name: p for p in cat["CSTR"].parameters}
        assert by_name["V"].symbol == "V"
        assert by_name["stoich"].symbol == r"\nu_{ij}"

    def test_a_field_that_declares_its_own_units_wins(self):
        """The field is more specific than the class-level table."""
        import dataclasses

        from difflow.catalog import _parameters
        from difflow.report.metadata import UnitMetadata

        @dataclasses.dataclass
        class P:
            V: float = dataclasses.field(
                default=1.0, metadata={"units": "L", "description": "volume"}
            )
            T: float = 300.0

        meta = UnitMetadata(symbol="X", description="",
                            parameter_units={"V": "m^3", "T": "K"})
        specs = {p.name: p for p in _parameters(P, meta)}
        assert specs["V"].units == "L"
        assert specs["V"].description == "volume"
        assert specs["T"].units == "K"

    def test_a_meaningful_share_of_parameters_now_carry_units(self, cat):
        with_units = sum(1 for s in cat.values()
                         for p in s.parameters if p.units)
        assert with_units > 100, "was zero before the classes were read"

    def test_no_metadata_entry_names_a_field_that_does_not_exist(self):
        # `parameter_units` and `parameter_symbols` are keyed by field name
        # and read with .get(), so a key that no longer matches a field is
        # silently dropped -- the unit simply never appears and nothing
        # says why. Thirty-five such keys had accumulated: `qmax` where the
        # field is `q_max`, `area` where it is `membrane_area`, `T_top` on
        # an absorber whose temperatures are `T_gas_in`/`T_liquid_in`.
        import dataclasses
        from difflow.catalog import _default_registry, _params_class

        dead = []
        for name, info in _default_registry().list_operations().items():
            params_cls = _params_class(info.cls)
            if params_cls is None or not dataclasses.is_dataclass(params_cls):
                continue                # nothing for the keys to disagree with
            fields = {f.name for f in dataclasses.fields(params_cls)}
            for attr in ("parameter_units", "parameter_symbols"):
                for key in getattr(info.cls, attr, None) or {}:
                    if key not in fields:
                        dead.append(f"{name}.{attr}[{key!r}]")
        assert not dead, f"metadata naming fields that do not exist: {dead}"

    def test_every_numeric_parameter_carries_a_unit(self, cat):
        # A number in a form without a unit is a number the reader has to
        # guess at, and the same table now feeds the delta-vector export,
        # where a mislabelled column is worse than an absent one. The
        # exceptions are listed rather than tolerated in bulk, so a new
        # field cannot join them by accident.
        allowed = {
            # a cached arity of the kinetic callable, not a quantity
            ("ContinuousBioreactor", "_kinetic_arity"),
            ("FedBatchBioreactor", "_kinetic_arity"),
        }
        numeric = re.compile(r"\b(float|int|Array|jnp|ndarray|Scalar)\b")
        missing = [
            (name, p.name) for name, spec in cat.items() for p in spec.parameters
            if not p.units and not p.is_callable
            and numeric.search(p.type)
            and "str" not in p.type and "bool" not in p.type
        ]
        assert not set(missing) - allowed, sorted(set(missing) - allowed)
        assert not allowed - set(missing), (
            "these gained a unit; drop them from the allowlist: "
            f"{sorted(allowed - set(missing))}")

    def test_the_new_fields_survive_to_dict(self, cat):
        payload = json.loads(json.dumps(cat["CSTR"].to_dict()))
        for key in ("symbol", "doc", "assumptions", "references",
                    "numerical_method"):
            assert key in payload, key
        assert payload["parameters"][0]["symbol"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
