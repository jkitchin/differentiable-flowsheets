"""SolventMakeup, and the absorber's optional lean-solvent inlet.

The loop drawn on a flowsheet --- absorber, stripper, makeup, recycle ---
must reach the answer ``solve_amine_loop`` gives for the same design: the
makeup is that function's closure made into a unit, so any disagreement
is a bug in one or the other.
"""

import jax
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from difflow import Flowsheet, make_stream  # noqa: E402
from difflow.catalog import catalog  # noqa: E402
from difflow.flowsheet import Unit  # noqa: E402
from difflow.gui import edit  # noqa: E402
from difflow_cc import (  # noqa: E402
    AbsorberParams,
    AmineAbsorber,
    AmineStripper,
    SolventMakeup,
    SolventMakeupParams,
    StripperParams,
    solve_amine_loop,
)

AMINE, WATER = 470.97, 3729.03


def makeup():
    return SolventMakeup(SolventMakeupParams(
        amine_flow=AMINE, water_flow=WATER, T=313.15, P=101325.0))


class TestTheUnit:
    def test_it_restores_the_circulation_and_carries_the_loading(self):
        lean = make_stream({"Amine": 470.0, "H2O": 3720.0,
                            "CO2_absorbed": 0.2 * 470.0}, 380.0, 2e5)
        out, info = makeup()(lean)
        assert float(out["F_Amine"]) == pytest.approx(AMINE)
        assert float(out["F_H2O"]) == pytest.approx(WATER)
        assert float(out["F_CO2_absorbed"]) == pytest.approx(0.2 * AMINE)
        assert float(out["T"]) == 313.15 and float(out["P"]) == 101325.0
        assert float(info["lean_loading"]) == pytest.approx(0.2)
        assert float(info["makeup_water"]) == pytest.approx(WATER - 3720.0)
        assert float(info["makeup_amine"]) == pytest.approx(AMINE - 470.0)

    def test_gradients(self):
        def loaded(co2):
            lean = make_stream({"Amine": 470.0, "H2O": 3720.0,
                                "CO2_absorbed": co2}, 380.0, 2e5)
            return makeup()(lean)[0]["F_CO2_absorbed"]
        check_grads(loaded, (90.0,), order=1, modes=["rev"])


class TestTheClosedLoop:
    def test_the_flowsheet_agrees_with_solve_amine_loop(self):
        gas = make_stream({"CO2": 135.0, "O2": 45.0, "H2O": 73.0,
                           "N2": 747.0}, 313.15, 101325.0)
        ap = AbsorberParams(solvent="MEA", n_stages=15, solvent_conc=30.0,
                            L_G_ratio=4.2, T_gas_in=313.15,
                            T_liquid_in=313.15, P_absorber=101325.0)
        sp = StripperParams(solvent="MEA", n_stages=10)
        _, co2_ref, ref = solve_amine_loop(gas, ap, sp)

        fs = Flowsheet(species_order=["CO2", "N2", "O2", "H2O", "Amine",
                                      "CO2_absorbed"],
                       default_flow=0.0, default_T=313.15,
                       default_P=101325.0)
        fs.add_feed("flue_gas", gas)
        fs.add_unit(Unit("absorber", AmineAbsorber(ap),
                         ["flue_gas", "lean_return"],
                         ["treated_gas", "rich"]))
        fs.add_unit(Unit("stripper", AmineStripper(sp), ["rich"],
                         ["lean", "co2_product"]))
        fs.add_unit(Unit("makeup", SolventMakeup(SolventMakeupParams(
            amine_flow=float(ref["amine_flow"]),
            water_flow=float(ref["water_flow"]),
            T=313.15, P=101325.0)), ["lean"], ["lean_makeup"]))
        fs.add_recycle("lean_makeup", "lean_return")
        s = fs.solve(tol=1e-10, max_iter=200)

        lean = s["lean_return"]
        loading = float(lean["F_CO2_absorbed"] / lean["F_Amine"])
        assert loading == pytest.approx(float(ref["lean_loading"]), abs=1e-9)
        capture = 1.0 - float(s["treated_gas"]["F_CO2"]) / 135.0
        assert capture == pytest.approx(float(ref["capture_efficiency"]),
                                        abs=1e-9)
        assert float(s["co2_product"]["F_CO2"]) == pytest.approx(
            float(co2_ref["F_CO2"]), rel=1e-9)


class TestTheOptionalInlet:
    """The absorber's lean solvent is a port the editor can add and drop."""

    @pytest.fixture
    def ports(self):
        return catalog()["AmineAbsorber"].to_dict()["ports"]

    def test_the_catalog_reports_it_as_an_optional_inlet(self, ports):
        assert ports["inlets"] == ["gas_in", "solvent_in"]
        assert ports["optional_inlets"] == ["solvent_in"]
        spec = catalog()["AmineAbsorber"]
        assert "solvent_in" not in [p.name for p in spec.call_parameters]
        assert spec.ports.min_inlets == 1

    def test_a_composition_dict_is_not_mistaken_for_one(self):
        """``Stream | None`` is also how a concentration dict is written."""
        spec = catalog()["FedBatchReactor"]
        assert spec.ports.optional_inlets == []
        assert "feed_composition" in [p.name for p in spec.call_parameters]

    def test_a_dropped_absorber_starts_with_its_gas_inlet_only(self, ports):
        inlets, outlets = edit.default_ports("absorber", ports, set())
        assert len(inlets) == 1 and len(outlets) == 2

    def test_the_lean_inlet_is_added_and_removed(self, ports):
        fs = Flowsheet(species_order=["CO2", "N2"])
        fs.add_unit(Unit("absorber", AmineAbsorber(AbsorberParams(
            solvent="MEA")), ["absorber_in"], ["gas", "rich"]))
        added = edit.add_inlet(fs, "absorber", ports)["stream"]
        assert edit.unit(fs, "absorber").inlet_names == ["absorber_in", added]
        with pytest.raises(edit.EditError, match="at most 2 inlets"):
            edit.add_inlet(fs, "absorber", ports)
        with pytest.raises(edit.EditError, match="needs its inlet"):
            edit.remove_inlet(fs, "absorber", "absorber_in", ports)
        edit.remove_inlet(fs, "absorber", added, ports)
        assert edit.unit(fs, "absorber").inlet_names == ["absorber_in"]
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
