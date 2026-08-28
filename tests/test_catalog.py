"""Tests for difflow.catalog.

The catalog is derived from the code by introspection rather than from
a hand-maintained table, so these tests mostly check that the
derivation agrees with what the units actually do --- particularly the
port arity, where the trap is counting the info dict every unit returns
as though it were an outlet stream.
"""

import inspect
import json

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

    def test_callable_fields_are_flagged(self, cat):
        """The fields a form cannot fill in."""
        assert set(cat["CSTR"].callable_parameters()) == {
            "rate_fn", "eos", "H_mix_fn", "K_eq_fn"
        }
        assert not cat["CSTR"].is_declarative

    def test_most_operations_are_declarative(self, cat):
        non_declarative = [n for n, s in cat.items() if not s.is_declarative]
        assert len(non_declarative) < 10, non_declarative
        # every one of them is a reactor, and it is always the rate law
        for name in non_declarative:
            assert any(
                "fn" in p for p in cat[name].callable_parameters()
            ), f"{name}: {cat[name].callable_parameters()}"

    def test_params_class_is_found(self, cat):
        assert cat["CSTR"].params_class == "CSTRParams"
        assert cat["Flash"].params_class == "FlashParams"

    def test_a_unit_without_params_has_none(self, cat):
        assert cat["Mixer"].parameters == []


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
