"""SolventMakeup, and the absorber's optional lean-solvent inlet."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import Flowsheet, make_stream  # noqa: E402
from difflow.catalog import catalog  # noqa: E402
from difflow.flowsheet import Unit  # noqa: E402
from difflow.workspace import edit  # noqa: E402
from difflow_cc import (  # noqa: E402
    AbsorberParams,
    AmineAbsorber,
    AmineStripper,
    SolventMakeup,
    SolventMakeupParams,
    StripperParams,
)

AMINE, WATER = 470.97, 3729.03


def makeup():
    return SolventMakeup(SolventMakeupParams(
        amine_flow=AMINE, water_flow=WATER, T=313.15, P=101325.0))


def _lean(co2=94.0):
    return make_stream({"Amine": 470.0, "H2O": 3720.0, "CO2_absorbed": co2}, 380.0, 2e5)


def test_unit_restores_the_circulation():
    out, info = makeup()(_lean())
    assert float(out["F_Amine"]) == pytest.approx(AMINE)
    assert float(out["F_H2O"]) == pytest.approx(WATER)
    assert bool(jnp.isfinite(info["lean_loading"]))


def test_grad():
    g = jax.grad(lambda c: makeup()(_lean(c))[0]["F_CO2_absorbed"])(90.0)
    assert bool(jnp.isfinite(g))


def test_closed_loop_on_a_flowsheet():
    gas = make_stream({"CO2": 135.0, "O2": 45.0, "H2O": 73.0, "N2": 747.0},
                      313.15, 101325.0)
    fs = Flowsheet(species_order=["CO2", "N2", "O2", "H2O", "Amine", "CO2_absorbed"],
                   default_flow=0.0, default_T=313.15, default_P=101325.0)
    fs.add_feed("flue_gas", gas)
    fs.add_unit(Unit("absorber", AmineAbsorber(AbsorberParams(
        solvent="MEA", n_stages=15, L_G_ratio=4.2)),
        ["flue_gas", "lean_return"], ["treated_gas", "rich"]))
    fs.add_unit(Unit("stripper", AmineStripper(StripperParams(solvent="MEA", n_stages=10)),
                     ["rich"], ["lean", "co2_product"]))
    fs.add_unit(Unit("makeup", makeup(), ["lean"], ["lean_makeup"]))
    fs.add_recycle("lean_makeup", "lean_return")
    s = fs.solve(tol=1e-8, max_iter=200)
    assert float(s["lean_return"]["F_Amine"]) == pytest.approx(AMINE)
    assert bool(jnp.isfinite(s["treated_gas"]["F_CO2"]))


class TestTheOptionalInlet:
    """The absorber's lean solvent is a port the editor can add and drop."""

    @pytest.fixture
    def ports(self):
        return catalog()["AmineAbsorber"].to_dict()["ports"]

    def test_the_catalog_reports_it_as_an_optional_inlet(self, ports):
        assert ports["inlets"] == ["gas_in", "solvent_in"]
        assert ports["optional_inlets"] == ["solvent_in"]
        assert catalog()["AmineAbsorber"].ports.min_inlets == 1

    def test_the_lean_inlet_is_added_and_removed(self, ports):
        fs = Flowsheet(species_order=["CO2", "N2"])
        fs.add_unit(Unit("absorber", AmineAbsorber(AbsorberParams(
            solvent="MEA")), ["absorber_in"], ["gas", "rich"]))
        added = edit.add_inlet(fs, "absorber", ports)["stream"]
        assert edit.unit(fs, "absorber").inlet_names == ["absorber_in", added]
        with pytest.raises(edit.EditError, match="at most 2 inlets"):
            edit.add_inlet(fs, "absorber", ports)
        edit.remove_inlet(fs, "absorber", added, ports)
        with pytest.raises(edit.EditError, match="needs its inlet"):
            edit.remove_inlet(fs, "absorber", "absorber_in", ports)

    def test_a_bad_declaration_is_refused_where_it_is_made(self):
        from difflow.catalog import describe_class

        class Wrong:
            optional_inlets = ("missing",)

            def __call__(self, gas_in: dict, solvent_in=None):
                return gas_in

        with pytest.raises(TypeError, match="does not take"):
            describe_class(Wrong)
