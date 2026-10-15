"""Smoke tests for the carbon capture plugin (difflow_cc): databases,
equilibrium and kinetics functions, the amine, membrane and adsorption
units, gradients, input validation and plugin registration."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import get_flows, make_stream  # noqa: E402
from difflow_cc import (  # noqa: E402
    AbsorberParams,
    AdsorptionParams,
    AmineAbsorber,
    AmineStripper,
    MembraneParams,
    MembraneSeparator,
    MultistageMembrane,
    PSAUnit,
    StripperParams,
    TSAUnit,
    TVSAUnit,
    VSAUnit,
    solve_amine_loop,
)


def _finite(x):
    return bool(jnp.all(jnp.isfinite(jnp.asarray(x))))


def _flue(P=101325.0):
    return make_stream({"CO2": 1.0, "N2": 9.0}, T=313.15, P=P)


def _rich():
    return make_stream({"H2O": 70.0, "Amine": 30.0, "CO2_absorbed": 12.0},
                       T=313.15, P=101325.0)


class TestDatabases:
    def test_lookups(self):
        from difflow_cc import (get_adsorbent, get_membrane, get_solvent,
                                list_adsorbents, list_membranes, list_solvents)
        assert "MEA" in list_solvents()
        assert "Zeolite_13X" in list_adsorbents()
        assert "Matrimid" in list_membranes()
        assert get_solvent("MEA").name == "MEA"
        assert get_adsorbent("Zeolite_13X") is not None
        assert get_membrane("Matrimid") is not None

    def test_unknown_solvent_raises(self):
        from difflow_cc import get_solvent
        with pytest.raises(KeyError):
            get_solvent("UNKNOWN_SOLVENT")


class TestEquilibriumAndKinetics:
    def test_isotherms(self):
        from difflow_cc import (dual_site_langmuir, get_isotherm, langmuir,
                                sips, toth, working_capacity_PSA)
        vals = [langmuir(1e4, 5.0, 1e-5), sips(1e4, 5.0, 1e-5, 0.8),
                toth(1e4, 5.0, 1e-5, 0.5),
                dual_site_langmuir(1e4, 3.0, 1e-4, 2.0, 1e-6)]
        iso = get_isotherm("Zeolite_13X", "CO2")
        vals += [iso(15000.0, 298.15),
                 working_capacity_PSA(iso, P_ads=101325, P_des=10000, T=298.15)]
        assert all(_finite(v) and float(v) > 0 for v in vals)

    def test_vle(self):
        from difflow_cc import AmineVLE, co2_equilibrium_pressure, henry_constant
        assert _finite(henry_constant(T=298.15, solvent="MEA"))
        assert _finite(co2_equilibrium_pressure(loading=0.3, T=313.15, solvent="MEA"))
        assert AmineVLE("MEA", C_amine=5000).loading_capacity > 0

    def test_kinetics(self):
        from difflow_cc.kinetics import enhancement_factor, overall_mass_transfer
        E = enhancement_factor(T=298.15, solvent="MEA", C_amine=5000.0, kL=1e-4)
        K = overall_mass_transfer(k_G=0.01, k_L=1e-4, E=10.0, H=3e6, P=101325.0)
        assert _finite(E) and _finite(K)

    def test_isotherm_grad(self):
        from difflow_cc import langmuir
        assert _finite(jax.grad(lambda P: langmuir(P, 5.0, 1e-5))(1e4))


class TestAmine:
    def test_absorber(self):
        gas_out, solvent_out, info = AmineAbsorber(
            AbsorberParams(solvent="MEA", n_stages=10, L_G_ratio=3.0))(_flue())
        assert 0 < float(info["capture_efficiency"]) < 1
        assert _finite(gas_out["F_CO2"]) and _finite(solvent_out["F_CO2_absorbed"])

    def test_stripper(self):
        lean, co2, info = AmineStripper(StripperParams(
            solvent="MEA", T_reboiler=393.15, target_lean_loading=0.2))(_rich())
        assert float(info["reboiler_duty"]) > 0
        assert _finite(info["specific_energy"]) and _finite(co2["F_CO2"])

    def test_solve_amine_loop(self):
        gas = make_stream({"CO2": 13.0, "N2": 87.0}, T=313.15, P=101325.0)
        _, _, info = solve_amine_loop(
            gas, AbsorberParams(solvent="MEA", n_stages=15, L_G_ratio=3.5),
            StripperParams(solvent="MEA", n_stages=10))
        assert bool(info["converged"])

    def test_absorber_grad(self):
        def captured(L_G):
            ab = AmineAbsorber(AbsorberParams(solvent="MEA", n_stages=10, L_G_ratio=L_G))
            return ab(_flue())[2]["CO2_captured"]
        assert _finite(jax.grad(captured)(1.0))

    def test_solvent_in_without_amine_rejected(self):
        ab = AmineAbsorber(AbsorberParams(solvent="MEA"))
        with pytest.raises(ValueError, match="Amine"):
            ab(_flue(), make_stream({"H2O": 10.0}, T=313.15, P=101325.0))

    def test_cold_reboiler_rejected(self):
        with pytest.raises(ValueError, match="below the rich solvent"):
            AmineStripper(StripperParams(solvent="MEA", T_reboiler=300.0))(_rich())

    def test_loop_solvent_mismatch_rejected(self):
        with pytest.raises(ValueError, match="solvent"):
            solve_amine_loop(_flue(), AbsorberParams(solvent="MEA"),
                             StripperParams(solvent="PZ"))


class TestMembrane:
    @staticmethod
    def _params(area=1000.0):
        return MembraneParams(membrane_type="Matrimid", area=area,
                              pressure_ratio=10.0, feed_pressure=1e6)

    def test_separator_conserves_species(self):
        feed = _flue(P=1e6)
        ret, perm, info = MembraneSeparator(self._params())(feed)
        assert 0 < float(info["stage_cut"]) < 1
        for sp, f in get_flows(feed).items():
            assert float(get_flows(ret)[sp] + get_flows(perm)[sp]) == pytest.approx(
                float(f), rel=1e-8)

    @pytest.mark.parametrize("config", ["series", "permeate_recycle"])
    def test_multistage(self, config):
        _, _, info = MultistageMembrane(self._params(500.0), n_stages=2,
                                        configuration=config)(_flue(P=1e6))
        assert _finite(info["overall_CO2_purity"])

    def test_grad(self):
        def recovery(area):
            return MembraneSeparator(self._params(area))(_flue(P=1e6))[2]["CO2_recovery"]
        assert _finite(jax.grad(recovery)(50.0))

    def test_pressure_ratio_rejected(self):
        with pytest.raises(ValueError, match="pressure_ratio"):
            MembraneParams(membrane_type="Matrimid", pressure_ratio=1.0)

    def test_permeate_pressure_above_feed_rejected(self):
        m = MembraneSeparator(MembraneParams(membrane_type="Matrimid"))
        with pytest.raises(ValueError, match="below the feed"):
            m(_flue(P=1e5), P_permeate=2e5)

    def test_recycle_three_stages_rejected(self):
        with pytest.raises(ValueError, match="two-stage"):
            MultistageMembrane(MembraneParams(membrane_type="Matrimid"),
                               n_stages=3, configuration="permeate_recycle")

    def test_unconverged_recycle_warns(self):
        p = MembraneParams(membrane_type="Matrimid", area=500.0)
        cas = MultistageMembrane(
            p, n_stages=2, configuration="permeate_recycle", recycle_iterations=1,
            stage_params=[p, MembraneParams(membrane_type="Matrimid", area=5.0)])
        with pytest.warns(RuntimeWarning, match="recycle not converged"):
            _, _, info = cas(_flue(P=1e6))
        assert not bool(info["recycle_converged"])


class TestAdsorption:
    @pytest.mark.parametrize("unit, kw, P", [
        (PSAUnit, dict(adsorbent="Zeolite_13X", P_adsorption=5e5, P_desorption=1e5), 5e5),
        (VSAUnit, dict(adsorbent="Zeolite_13X", cycle_type="VSA",
                       P_adsorption=101325.0, P_desorption=1e4), 101325.0),
        (TSAUnit, dict(adsorbent="PEI_Silica", cycle_type="TSA",
                       T_adsorption=298.15, T_desorption=373.15), 101325.0),
        (TVSAUnit, dict(adsorbent="Mg_MOF_74", cycle_type="TVSA", T_adsorption=298.15,
                        T_desorption=353.15, P_desorption=1e4), 101325.0),
    ])
    def test_units_conserve_species(self, unit, kw, P):
        feed = make_stream({"CO2": 1.5, "N2": 8.5}, T=298.15, P=P)
        prod, off, info = unit(AdsorptionParams(bed_mass=100.0, **kw))(feed)
        assert _finite(info["recovery"]) and float(info["working_capacity"]) > 0
        for sp, f in get_flows(feed).items():
            assert float(get_flows(prod)[sp] + get_flows(off)[sp]) == pytest.approx(
                float(f), rel=1e-8)

    def test_psa_grad(self):
        feed = make_stream({"CO2": 1.0, "N2": 4.0}, T=298.15, P=5e5)

        def captured(P_ads):
            psa = PSAUnit(AdsorptionParams(adsorbent="Zeolite_13X", P_adsorption=P_ads,
                                           P_desorption=1e5, bed_mass=100.0))
            return psa(feed)[2]["CO2_captured"]
        assert _finite(jax.grad(captured)(5e5))

    def test_tsa_cold_desorption_flagged_not_nan(self):
        feed = make_stream({"CO2": 0.5, "N2": 9.5}, T=298.15, P=101325.0)
        _, _, info = TSAUnit(AdsorptionParams(adsorbent="PEI_Silica",
                                              T_desorption=280.0))(feed)
        assert not bool(info["feasible"])
        assert _finite(info["purity"]) and _finite(info["heating_power"])


def test_register():
    import difflow_cc

    class Registry:
        def __init__(self):
            self.names = []

        def register(self, name, **kw):
            self.names.append(name)

    reg = Registry()
    info = difflow_cc.register(reg)
    assert reg.names == info["operations"]
    assert "AmineAbsorber" in reg.names
