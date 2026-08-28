"""Tests for difflow.codegen.

The load-bearing test is `test_generated_script_reproduces_the_solve`:
generated source that runs but computes something else would be worse
than no generator at all.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (
    CSTR,
    CSTRParams,
    Flash,
    FlashParams,
    Flowsheet,
    Heater,
    HeaterParams,
    IdealThermo,
    Unit,
    codegen,
    get_species_data,
    make_stream,
    mass_action_kinetics,
)
from difflow.codegen import CodegenError

SPECIES = ["water", "ethanol"]


@pytest.fixture(scope="module")
def thermo():
    return IdealThermo({n: get_species_data(n) for n in SPECIES})


@pytest.fixture
def flowsheet(thermo):
    """A reactor with a data-built rate law, then a flash."""
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
    )), ["feed"], ["rx"]))
    fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                     ["rx"], ["liq", "vap"]))
    return fs


def run_generated(source: str):
    """Execute generated source and hand back its flowsheet."""
    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)
    return namespace["fs"]


# =============================================================================
# The generated script
# =============================================================================


class TestGeneratedScript:
    def test_generated_script_reproduces_the_solve(self, flowsheet):
        """Source that runs but computes something else is worse than none."""
        rebuilt = run_generated(codegen.to_python(flowsheet))

        original, generated = flowsheet.solve(), rebuilt.solve()
        assert set(original) == set(generated)
        for stream in original:
            for key, value in original[stream].items():
                if isinstance(value, str):
                    continue
                assert float(value) == float(generated[stream][key]), (
                    f"{stream}.{key} differs"
                )

    def test_it_is_valid_python(self, flowsheet):
        compile(codegen.to_python(flowsheet), "<generated>", "exec")

    def test_structure_survives(self, flowsheet):
        rebuilt = run_generated(codegen.to_python(flowsheet))
        assert rebuilt.species_order == SPECIES
        assert [u.name for u in rebuilt.units] == ["reactor", "flash"]
        assert [u.outlet_names for u in rebuilt.units] == [["rx"], ["liq", "vap"]]

    def test_recycles_survive(self, thermo):
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.1},
                                        T=350.0, P=101325.0))
        fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=360.0)),
                         ["feed", "recycle"], ["hot"]))
        fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                         ["hot"], ["liq", "vap"]))
        fs.add_recycle("liq", "recycle")

        rebuilt = run_generated(codegen.to_python(fs))
        assert rebuilt.recycles == fs.recycles

    def test_the_solve_block_is_optional(self, flowsheet):
        assert "__main__" in codegen.to_python(flowsheet)
        assert "__main__" not in codegen.to_python(flowsheet, include_solve=False)

    def test_imports_only_what_it_uses(self, flowsheet):
        source = codegen.to_python(flowsheet)
        assert "CSTR" in source and "Flash" in source
        assert "Splitter" not in source, "unused names should not be imported"


# =============================================================================
# Readability
# =============================================================================


class TestReadability:
    def test_a_data_built_rate_law_is_hoisted_not_inlined(self, flowsheet):
        """Inlining buries the reactor in one unreadable line."""
        source = codegen.to_python(flowsheet)
        assert "kinetics_reactor = mass_action_kinetics(" in source
        assert "**kinetics_reactor.params_kwargs()" in source

    def test_hoisting_omits_the_arrays_the_factory_derives(self, flowsheet):
        """stoich and the order matrices are rebuilt, not repeated."""
        source = codegen.to_python(flowsheet)
        assert "order_f" not in source
        assert "stoich=" not in source

    def test_thermo_is_hoisted_and_uses_the_database(self, flowsheet):
        source = codegen.to_python(flowsheet)
        assert "thermo_flash = IdealThermo(" in source
        assert "get_species_data" in source

    def test_it_carries_a_docstring(self, flowsheet):
        assert codegen.to_python(flowsheet).startswith('"""Flowsheet generated by difflow')


# =============================================================================
# Value rendering
# =============================================================================


class TestRendering:
    def test_infinity_is_rendered_as_an_expression(self):
        """`repr` writes `inf`, which is not a name in the generated module."""
        from difflow.codegen import _render_nested

        assert _render_nested([float("inf")]) == "[float('inf')]"
        assert _render_nested([-float("inf")]) == "[-float('inf')]"
        assert _render_nested([float("nan")]) == "[float('nan')]"
        # and it has to survive a round trip through compile()
        assert eval(_render_nested([float("inf"), 1.0])) == [float("inf"), 1.0]

    def test_nested_arrays_keep_their_shape(self):
        from difflow.codegen import _render_nested

        assert _render_nested([[1.0, 2.0], [3.0, 4.0]]) == "[[1.0, 2.0], [3.0, 4.0]]"

    def test_an_array_parameter_survives(self, thermo):
        """A unit carrying a raw array must still generate and run."""
        kin = mass_action_kinetics([{
            "equation": "2 water -> ethanol",
            "reactants": {"water": 2.0}, "products": {"ethanol": 1.0},
            "rate_params": {"A": 5.0, "Ea": 0.0, "n": 0.0},
        }], SPECIES)
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.0},
                                        T=350.0, P=101325.0))
        fs.add_unit(Unit("rx", CSTR(CSTRParams(
            V=1.0, molar_density=1000.0,
            dH_rxn=jnp.array([-5.0e4]), **kin.params_kwargs(),
        )), ["feed"], ["out"]))

        rebuilt = run_generated(codegen.to_python(fs))
        assert float(rebuilt.units[0].operation.params.dH_rxn[0]) == -5.0e4


# =============================================================================
# What it refuses
# =============================================================================


class TestRefusals:
    def test_a_hand_written_callable_is_refused(self):
        """It cannot be written as source, so it is not silently dropped."""
        def rate_fn(C, T, p):
            return jnp.array([0.0])

        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("rx", CSTR(CSTRParams(
            V=1.0, rate_fn=rate_fn, stoich=jnp.zeros((2, 1)),
            rate_params={}, species_order=SPECIES,
        )), ["feed"], ["out"]))

        with pytest.raises(CodegenError, match="does not record how it was built"):
            codegen.to_python(fs)

    def test_an_unregistered_operation_is_refused(self):
        class HomeMadeUnit:
            def __init__(self):
                self.params = None

        fs = Flowsheet(species_order=SPECIES)
        fs.add_unit(Unit("mystery", HomeMadeUnit(), ["feed"], ["out"]))
        with pytest.raises(CodegenError, match="not in the operation registry"):
            codegen.to_python(fs)

    def test_an_unsupported_thermo_is_refused(self, flowsheet):
        class ExoticThermo:
            pass

        flowsheet.units[1].operation.thermo = ExoticThermo()
        with pytest.raises(CodegenError, match="cannot be written as source"):
            codegen.to_python(flowsheet)


# =============================================================================
# Files
# =============================================================================


class TestFiles:
    def test_save_script(self, flowsheet, tmp_path):
        path = codegen.save_script(flowsheet, tmp_path / "plant.py")
        assert path.exists()
        rebuilt = run_generated(path.read_text())
        assert [u.name for u in rebuilt.units] == ["reactor", "flash"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
