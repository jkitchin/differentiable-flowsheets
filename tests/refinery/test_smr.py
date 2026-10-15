"""SMR hydrogen plant (#473): Xu-Froment kinetics, reformer tube, shift, furnace, plant."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import GibbsReactor, GibbsReactorParams, make_stream
from difflow_refinery import thermochemistry as tc
from difflow_refinery.hydrogen import PSA, H2Plant
from difflow_refinery.hydrogen import smr

jax.config.update("jax_enable_x64", True)

TUBE_FEED = {"methane": 0.4, "water": 1.2, "hydrogen": 0.04, "carbon_monoxide": 0.0,
             "carbon_dioxide": 0.01, "nitrogen": 0.004, "hydrogen_sulfide": 0.0}


def tube_inlet(**kw):
    f = dict(TUBE_FEED)
    T = kw.pop("T", 893.15)
    P = kw.pop("P", 25e5)
    f.update(kw)
    return make_stream(f, T, P)


#: IEAGHG 2017/02 base case, heat and material balance stream 2 (natural gas
#: feedstock, 128 C, 3.71 MPa, 1455.8 kmol/h); n-pentane (0.0001) lumped into
#: n-butane; H2S 5 ppmv (note 1 of the balance).
IEAGHG_NG = {"methane": 0.89, "ethane": 0.07, "propane": 0.01, "n_butane": 0.0011,
             "carbon_dioxide": 0.02, "nitrogen": 0.0089, "hydrogen_sulfide": 5e-6}


def natural_gas(F=0.45, T=401.15, P=37.1e5):
    """The IEAGHG natural gas, ``F`` mol/s (one tube's worth by default)."""
    return make_stream({k: v * F for k, v in IEAGHG_NG.items()}, T, P)


def elements(n, names):
    tab = tc.IdealGasSet(list(names))
    return {e: float(jnp.sum(jnp.asarray(n) * tab.ELEMENTS[e])) for e in ("C", "H", "O", "N", "S")}


# ---------------------------------------------------------------- kinetics
class TestKinetics:
    def test_constants_as_published(self):
        """Xu & Froment (1989) Tables 3-4, as printed."""
        assert smr.XU_FROMENT == {
            "k1": (4.225e15, 240.1e3), "k2": (1.955e6, 67.13e3), "k3": (1.020e15, 243.9e3),
            "K_CO": (8.23e-5, -70.65e3), "K_H2": (6.12e-9, -82.90e3),
            "K_CH4": (6.65e-4, -38.28e3), "K_H2O": (1.77e5, 88.68e3)}
        assert "10.1002/aic.690350109" in smr.XU_FROMENT_SOURCE

    @pytest.mark.parametrize("T", [700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0])
    def test_table_equilibrium_matches_xu_froment(self, T):
        """The formation table's K1, K2 against the expressions Xu & Froment
        used: two independent thermodynamic bases agree to 11% in K (ln K
        within 0.098 over 700-1200 K, worst near 850 K)."""
        ours = np.log(np.asarray(smr.equilibrium_constants(T)))
        theirs = np.log(np.asarray(smr.xu_froment_equilibrium(T)))
        assert np.max(np.abs(ours - theirs)) < 0.11

    def test_rates_vanish_at_equilibrium(self):
        eq, _ = GibbsReactor(GibbsReactorParams(thermo=smr.GAS, mode="isothermal", T_out=1100.0))(
            tube_inlet(P=20e5))
        n = smr.to_vec(eq)
        p = n / jnp.sum(n) * 20.0
        r = smr.xu_froment_rates(p, 1100.0)
        r_feed = smr.xu_froment_rates(smr.to_vec(tube_inlet()) / 1.654 * 20.0 + 1e-6, 1100.0)
        assert np.all(np.abs(np.asarray(r)) < 1e-6 * np.max(np.abs(np.asarray(r_feed))))

    def test_rates_forward_on_feed(self):
        n = smr.to_vec(tube_inlet())
        r = smr.xu_froment_rates(n / jnp.sum(n) * 25.0, 900.0)
        assert float(r[0]) > 0 and float(r[2]) > 0


# ---------------------------------------------------------------- tube
@pytest.fixture(scope="module")
def tube():
    return smr.reformer_tube(tube_inlet(), smr.ReformerTubeParams(), T_out=1123.15, n_profile=41)


class TestTube:
    def test_outlet_temperature_and_duty(self, tube):
        out, info = tube
        assert bool(info["converged"])
        assert float(out["T"]) == pytest.approx(1123.15, abs=1e-3)
        n0, n1 = smr.to_vec(tube_inlet()), smr.to_vec(out)
        dH = smr.enthalpy(n1, out["T"]) - smr.enthalpy(n0, 893.15)
        assert float(info["Q"]) == pytest.approx(float(dH), rel=1e-6)

    def test_element_balance(self, tube):
        out, _ = tube
        a = elements(smr.to_vec(tube_inlet()), smr.SYNGAS)
        b = elements(smr.to_vec(out), smr.SYNGAS)
        for e in a:
            assert b[e] == pytest.approx(a[e], abs=1e-10)

    def test_short_of_equilibrium_and_slip(self, tube):
        out, info = tube
        assert 0.0 < float(info["approach"]) < 15.0
        n = smr.to_vec(out)
        slip = n[0] / (jnp.sum(n) - n[1])
        assert 0.02 < float(slip) < 0.06

    def test_profile(self, tube):
        """The cold inlet dip: the endotherm outruns the flux at the inlet."""
        _, info = tube
        T = np.asarray(info["profile"]["T"])
        assert T[1] < T[0] and T[-1] == pytest.approx(1123.15, abs=1e-3)
        assert np.all(np.diff(np.asarray(info["profile"]["n"])[:, 0]) <= 1e-12)

    def test_equilibrium_limit(self):
        """With 33x the effectiveness the outlet is the Gibbs equilibrium at
        the outlet T and P (the kinetics' K are the table's), to the 0.1 K
        approach left (dlnK1/dT ~ 0.02/K moves CH4 ~0.1%)."""
        tp = smr.ReformerTubeParams(eta=(1.0, 1.0, 1.0))
        out, info = smr.reformer_tube(tube_inlet(), tp, T_out=1123.15)
        eq, _ = GibbsReactor(GibbsReactorParams(thermo=smr.GAS, mode="isothermal", T_out=1123.15))(
            make_stream(TUBE_FEED, 893.15, out["P"]))
        for k in ("methane", "water", "hydrogen", "carbon_monoxide", "carbon_dioxide"):
            assert float(out[f"F_{k}"]) == pytest.approx(float(eq[f"F_{k}"]), rel=2e-3)
        assert abs(float(info["approach"])) < 0.2

    def test_wall_mode(self):
        tp = smr.ReformerTubeParams(mode="wall", T_wall=(1050.0, 1180.0), U=300.0)
        out, info = smr.reformer_tube(tube_inlet(), tp)
        assert bool(info["converged"])
        assert 900.0 < float(out["T"]) < 1180.0
        n0, n1 = smr.to_vec(tube_inlet()), smr.to_vec(out)
        dH = smr.enthalpy(n1, out["T"]) - smr.enthalpy(n0, 893.15)
        assert float(info["Q"]) == pytest.approx(float(dH), rel=1e-12)

    @pytest.mark.parametrize("change,direction", [
        (dict(water=1.6), -1),     # more steam, less slip
        (dict(P=35e5), +1),        # more pressure, more slip
    ])
    def test_slip_trends(self, tube, change, direction):
        base = float(tube[0]["F_methane"])
        out, _ = smr.reformer_tube(tube_inlet(**change), smr.ReformerTubeParams(), T_out=1123.15)
        assert np.sign(float(out["F_methane"]) - base) == direction

    def test_slip_falls_with_outlet_T(self, tube):
        out, _ = smr.reformer_tube(tube_inlet(), smr.ReformerTubeParams(), T_out=1173.15)
        assert float(out["F_methane"]) < float(tube[0]["F_methane"])

    @pytest.mark.parametrize("adjoint", ["reverse", "forward"])
    def test_gradient_vs_fd(self, adjoint):
        """d(CH4 out, Q)/d(T_out, steam): implicit step through the duty solve."""
        tp = smr.ReformerTubeParams(adjoint=adjoint)

        def f(x):
            out, info = smr.reformer_tube(tube_inlet(water=x[1]), tp, T_out=x[0])
            return jnp.stack([out["F_methane"], info["Q"] / 1e5])

        x = jnp.array([1123.15, 1.2])
        J = (jax.jacfwd if adjoint == "forward" else jax.jacrev)(f)(x)
        h = jnp.array([0.05, 1e-4])
        for j in range(2):
            e = jnp.zeros(2).at[j].set(h[j])
            fd = (f(x + e) - f(x - e)) / (2 * h[j])
            np.testing.assert_allclose(np.asarray(J[:, j]), np.asarray(fd), rtol=2e-4, atol=1e-9)


# ---------------------------------------------------------------- front/back end
class TestUnits:
    def test_desulfurize(self):
        out, removed = smr.desulfurize(natural_gas(), 0.99)
        h = float(natural_gas()["F_hydrogen_sulfide"])
        assert float(removed) == pytest.approx(0.99 * h)
        assert float(out["F_hydrogen_sulfide"]) == pytest.approx(0.01 * h)

    def test_prereformer_converts_c2plus(self):
        ng = natural_gas()
        mixed = dict(ng, F_water=3.0 * smr.carbon_flow(ng))
        out, info = smr.prereform(mixed, 763.15)
        # equilibrium leaves ppm C2+ (2 CH4 = C2H6 + H2: ~1e-5 at 700 K);
        # IEAGHG specify <= 500 ppmv
        assert float(info["C2plus"]) / float(jnp.sum(smr.to_vec(out))) < 50e-6
        a = elements(smr.to_vec(mixed, smr.SPECIES), smr.SPECIES)
        b = elements(smr.to_vec(out), smr.SYNGAS)
        for e in ("C", "H", "O", "N"):
            assert b[e] == pytest.approx(a[e], rel=1e-6)
        assert float(out["T"]) < 763.15        # net endothermic with this gas

    def test_shift_equilibrium(self):
        g = make_stream({"methane": 0.05, "water": 0.8, "hydrogen": 1.3, "carbon_monoxide": 0.25,
                         "carbon_dioxide": 0.1, "nitrogen": 0.0, "hydrogen_sulfide": 0.0}, 613.15, 24e5)
        out, info = smr.shift(g, 613.15)
        n = smr.to_vec(out)
        i = {k: j for j, k in enumerate(smr.SYNGAS)}
        Kp = n[i["carbon_dioxide"]] * n[i["hydrogen"]] / (n[i["carbon_monoxide"]] * n[i["water"]])
        assert float(Kp) == pytest.approx(float(smr.equilibrium_constants(out["T"])[1]), rel=1e-8)
        assert float(out["T"]) > 613.15      # exothermic, adiabatic

    def test_knock_out_saturates(self):
        g = make_stream({"methane": 0.05, "water": 0.8, "hydrogen": 1.3, "carbon_monoxide": 0.02,
                         "carbon_dioxide": 0.3, "nitrogen": 0.0, "hydrogen_sulfide": 0.0}, 450.0, 22e5)
        out, cond, duty = smr.knock_out(g, 313.15)
        from difflow.water import water_saturation_pressure
        n = smr.to_vec(out)
        assert float(n[1] / jnp.sum(n) * 22e5) == pytest.approx(float(water_saturation_pressure(313.15)), rel=1e-10)
        assert float(cond + n[1]) == pytest.approx(0.8)
        assert float(duty) > 0

    def test_lhv_methane(self):
        """LHV of CH4 from the formation table vs 802.3 kJ/mol (the CODATA/
        NIST-JANAF Hf of CH4, CO2, H2O(g): -74.87, -393.51, -241.83)."""
        assert float(smr.lhv(["methane"])[0]) == pytest.approx(802.3e3, rel=2e-3)
        assert float(smr.lhv(["hydrogen"])[0]) == pytest.approx(241.83e3, rel=1e-3)

    def test_furnace_closes_duty(self):
        tail = {"hydrogen": 0.15, "methane": 0.09, "carbon_monoxide": 0.06, "carbon_dioxide": 0.33,
                "water": 0.01}
        fuel = {"methane": 1.0}
        fz = smr.furnace(1.0e5, tail, fuel, y_O2=0.02, T_stack=423.15)
        assert abs(float(fz["closure"])) < 1e-6
        flue = np.asarray(fz["flue"])
        assert flue[3] / flue.sum() == pytest.approx(0.02, rel=1e-12)
        assert float(fz["CO2"]) == pytest.approx(0.09 + 0.06 + 0.33 + float(fz["fuel"]), rel=1e-12)


# ---------------------------------------------------------------- plant
SHORTCUT = PSA(recovery=0.89, purity=0.999)


@pytest.fixture(scope="module")
def plant():
    return smr.smr_plant(natural_gas(), smr.SMRParams(psa=SHORTCUT))


class TestPlant:
    def test_converged(self, plant):
        assert bool(plant["converged"])
        assert float(plant["fuel"]) > 0

    def test_element_balance(self, plant):
        """Feed + steam = product + tail + condensate + ZnO, and furnace C = CO2."""
        ng = natural_gas()
        a = elements(smr.to_vec(ng, smr.SPECIES), smr.SPECIES)
        a["H"] += 2 * float(plant["steam"])
        a["O"] += float(plant["steam"])
        prod = plant["psa"]["product"]
        tail = plant["psa"]["tail_gas"]
        out = elements(jnp.stack([prod[k] + tail[k] for k in smr.SYNGAS]), smr.SYNGAS)
        out["H"] += 2 * float(plant["condensate"]) + 2 * float(plant["H2S_removed"])
        out["O"] += float(plant["condensate"])
        out["S"] += float(plant["H2S_removed"])
        for e in a:
            assert out[e] == pytest.approx(a[e], rel=1e-9, abs=1e-14)
        C_fuel = float(plant["fuel"]) * float(smr.carbon_flow(ng) + ng["F_carbon_dioxide"]) / float(
            sum(ng.get(f"F_{k}", 0.0) for k in smr.SPECIES))
        C_tail = sum(float(tail[k]) for k in ("methane", "carbon_monoxide", "carbon_dioxide"))
        assert float(plant["CO2"]) == pytest.approx(C_fuel + C_tail, rel=1e-10)

    def test_typical_values(self, plant):
        assert 0.03 < float(plant["CH4_slip"]) < 0.06
        assert 0.0 < float(plant["approach"]) < 15.0
        assert float(plant["H2"]) == pytest.approx(0.89 * float(plant["ko"]["F_hydrogen"]), rel=1e-12)

    def test_h2plant_from_smr(self, plant):
        h = H2Plant.from_smr(plant, ng_price=4.0)
        assert float(h.capacity) == pytest.approx(float(plant["H2"]))
        assert float(h.purity) == pytest.approx(0.999)
        assert float(h.marginal_cost) == pytest.approx(float(plant["ng_lhv"] / plant["H2"]) * 4e-6, rel=1e-12)

    def test_cycle_model_psa(self):
        r = smr.smr_plant(natural_gas(), smr.SMRParams())
        for v in r["psa"]["balance"].values():
            assert abs(float(v)) < 1e-12
        assert 0.0 < float(r["recovery"]) < 1.0

    def test_gradients_vs_fd(self):
        """d(H2, CO2, NG)/d(S/C, T_out) through the whole plant, forward mode."""
        def f(x):
            p = smr.SMRParams(S_C=x[0], T_out=x[1], psa=SHORTCUT,
                              tube=smr.ReformerTubeParams(adjoint="forward"))
            r = smr.smr_plant(natural_gas(), p)
            return jnp.stack([r["H2"], r["CO2"], r["ng_lhv"] / 1e5])

        x = jnp.array([3.0, 1123.15])
        J = jax.jacfwd(f)(x)
        h = jnp.array([1e-3, 0.05])
        for j in range(2):
            e = jnp.zeros(2).at[j].set(h[j])
            fd = (f(x + e) - f(x - e)) / (2 * h[j])
            np.testing.assert_allclose(np.asarray(J[:, j]), np.asarray(fd), rtol=5e-4)
        assert float(J[0, 1]) > 0        # hotter outlet, more H2


# ---------------------------------------------------------------- IEAGHG 2017/02 base case
class TestIEAGHG:
    """Against the IEAGHG 2017/02 base case (Amec Foster Wheeler, Feb 2017),
    heat and material balance streams 2-7 and 12-13 and the equipment list.
    Fuel use is NOT compared: the combustion-air preheat temperature is not
    given, and the supplementary fuel moves with it."""

    #: stream 5, HTS inlet: 320 C, 2.80 MPa, 8370.3 kmol/h
    HTS_IN = {"carbon_dioxide": 0.0492, "carbon_monoxide": 0.1156, "hydrogen": 0.5171,
              "nitrogen": 0.0015, "methane": 0.0238, "water": 0.2927, "hydrogen_sulfide": 0.0}

    def hts_inlet(self):
        F = 8370.3 / 3.6
        return make_stream({k: v * F for k, v in self.HTS_IN.items()}, 593.15, 2.80e6)

    def test_hts_outlet(self):
        """Stream 6: 412 C, CO 3.66%, CO2 12.83%, H2 59.61% (wet). The
        equilibrium gives 413.0 C and 3.47% CO: the plant is ~0.2% CO short
        of equilibrium."""
        out, _ = smr.shift(self.hts_inlet(), 593.15, pressure_drop=0.03e6)
        n = smr.to_vec(out)
        y = n / jnp.sum(n)
        i = {k: j for j, k in enumerate(smr.SYNGAS)}
        assert float(out["T"]) - 273.15 == pytest.approx(412.0, abs=2.0)
        assert float(y[i["carbon_monoxide"]]) == pytest.approx(0.0366, abs=0.003)
        assert float(y[i["carbon_dioxide"]]) == pytest.approx(0.1283, abs=0.003)
        assert float(y[i["hydrogen"]]) == pytest.approx(0.5961, abs=0.003)

    def test_reformer_outlet_at_stated_temperature(self):
        """The reformer outlet gas (stream 5, before the shift) is at
        reforming equilibrium between the stated 900 and 950 C."""
        n = smr.to_vec(self.hts_inlet())
        assert float(smr.approach_to_equilibrium(n, 1173.15, 2.9e6)) < 0
        assert float(smr.approach_to_equilibrium(n, 1223.15, 2.9e6)) > 0

    @pytest.fixture(scope="class")
    def base(self):
        ng = natural_gas(F=1455.8 / 3.6)
        p = smr.SMRParams(n_tubes=240, psa=PSA(recovery=0.90, purity=0.9999))
        return smr.smr_plant(ng, p)

    def test_plant(self, base):
        """H2 4490.6 kmol/h (product 4461.5 + recycle 29.1, stream 12-13);
        dry CH4 slip "3.3 to 4%"; radiant duty 82.63 MMkcal/h (96.1 MW);
        PSA inlet H2 75.63% (stream 7); HTS outlet 412 C; prereformer C2+
        <= 500 ppmv. PSA recovery 0.90 = 4490.6 / (0.7563 x 6596.9)."""
        r = base
        assert bool(r["converged"])
        assert float(r["H2"]) * 3.6 == pytest.approx(4490.6, rel=0.02)
        assert 0.033 <= float(r["CH4_slip"]) <= 0.040
        assert float(r["reformer_duty"]) == pytest.approx(82.63e6 * 4184.0 / 3600.0, rel=0.05)
        ko = smr.to_vec(r["ko"])
        assert float(ko[2] / jnp.sum(ko)) == pytest.approx(0.7563, abs=0.01)
        assert float(r["hts"]["T"]) - 273.15 == pytest.approx(412.0, abs=3.0)
        pre = smr.to_vec(r["prereformer"])
        assert float(r["prereformer_info"]["C2plus"] / jnp.sum(pre)) < 500e-6


# ---------------------------------------------------------------- planning
class TestPlanning:
    def test_plant_reverse_mode(self):
        """jax.grad through the whole plant (tube on the checkpointed adjoint)
        agrees with the forward-mode derivative."""
        def f(T, adjoint):
            p = smr.SMRParams(T_out=T, psa=SHORTCUT, tube=smr.ReformerTubeParams(adjoint=adjoint))
            return smr.smr_plant(natural_gas(), p)["H2"]

        g_rev = jax.grad(f)(1193.15, "reverse")
        g_fwd = jax.jacfwd(f)(1193.15, "forward")
        assert float(g_rev) == pytest.approx(float(g_fwd), rel=1e-6)

    def test_smr_block_links_to_network(self):
        from difflow_refinery.hydrogen import link_smr, smr_block
        b = smr_block(natural_gas(), smr.SMRParams(psa=SHORTCUT), jit=False)
        assert b.u_names == ["ng", "S_C", "T_out"]
        assert b.y_names == ["h2.net_mol_s", "h2.purity", "ng.lhv_MW", "co2.mol_s"]
        y = np.asarray(b.fn(jnp.asarray(b.u0)))
        r = smr.smr_plant(natural_gas(), smr.SMRParams(psa=SHORTCUT))
        np.testing.assert_allclose(y, [float(r["H2"]), 99.9, float(r["ng_lhv"]) / 1e6, float(r["CO2"])],
                                   rtol=1e-9)
        assert link_smr() == [("smr.h2.net_mol_s", "h2.smr.h2"), ("smr.h2.purity", "h2.smr.purity")]
        assert b.metadata["u_units"] == ["mol/s", "mol/mol", "K"]
