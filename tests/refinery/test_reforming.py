"""Catalytic reformer (#309).

What is pinned:

* the thermochemistry: coded heats of formation and entropies against the
  values they were transcribed from, and against ``difflow.database`` where
  that has the species; heats of reaction and equilibrium constants follow
  from them (no separate parameters); Smith's (1959) activation energies as
  converted from his Rankine coefficients;
* the network conserves carbon and hydrogen reaction by reaction, and a long
  bed relaxes the dehydrogenation to the equilibrium the Gibbs energies set;
* the feed mapping conserves mass (PIONA and characterization paths), and
  the naphthene lever sets what it says;
* a single bed is adiabatic and element-conserving to round-off, and its
  outlet temperature is differentiable (AD against central differences);
* the flowsheet (slow): converges with H2 recycle on a lean and a rich
  naphtha from the default initialization; overall mass, C, H and energy
  close to 1e-8; the first reactor has the largest temperature drop; RON and
  H2 make rise and C5+ yield falls with WAIT; aromatics rise as pressure
  falls; implicit gradients of yield, RON, net H2 and the first-reactor dT
  in WAIT, separator pressure, H2/HC and naphthene content match central
  differences; the planning block evaluates and differentiates.

NOT reproduced (and not claimed): a published commercial-reformer simulation
(Padmavathi & Chaudhuri 1997, Taskar & Riggs 1997) -- their papers could not
be obtained to check feed, conditions and outlet data; and an IDAES
GibbsReactor comparison -- IDAES is not available in the test environment.
"""

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.reforming import (
    CatalyticReformer, NaphthaFeed, ReformerParams, ReformingKinetics, lean_naphtha, rich_naphtha)
from difflow_refinery.reforming import kinetics as kn
from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming import thermo as th
from difflow_refinery.reforming.products import reformate_properties
from difflow_refinery.reforming.reactor import integrate_bed

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Species and thermochemistry
# =============================================================================


class TestSpecies:
    def test_formulas_and_molar_masses(self):
        for k, s in sp.SPECIES.items():
            mw = 12.0107 * s.carbon + 1.00794 * s.hydrogen
            assert s.MW == pytest.approx(mw, rel=2e-4), k
            if s.kind in ("nP", "iP", "light"):
                assert s.hydrogen == 2 * s.carbon + 2, k
            elif s.kind == "N":
                assert s.hydrogen == 2 * s.carbon, k
            elif s.kind == "A":
                assert s.hydrogen == 2 * s.carbon - 6, k

    @pytest.mark.parametrize("key,Hf,S0", [
        # API TDB ideal-gas Hf, Yaws S0, as read from chemicals 1.5.2 (module docstring).
        ("C1", -74520.0, 186.60), ("nP6", -166950.0, 388.74), ("N5_6", -106690.0, 339.90),
        ("N6", -123130.0, 297.31), ("A6", 82930.0, 269.18), ("N7", -154770.0, 343.50),
        ("A7", 50170.0, 321.08), ("A8", 29790.0, 361.24), ("iP7", -194500.0, 420.52),
        ("H2", 0.0, 130.68),
    ])
    def test_pinned_thermochemistry(self, key, Hf, S0):
        assert sp.SPECIES[key].Hf == Hf
        assert sp.SPECIES[key].S0 == S0

    def test_heats_of_formation_agree_with_difflow_database(self):
        from difflow.database import _IDEAL_THERMO_DATA as db
        pairs = {"C1": "methane", "C2": "ethane", "C3": "propane", "iC4": "isobutane",
                 "nC4": "n_butane", "iC5": "isopentane", "nC5": "n_pentane", "nP6": "n_hexane",
                 "iP6": "2_methylpentane", "N5_6": "methylcyclopentane", "N6": "cyclohexane",
                 "A6": "benzene", "nP7": "n_heptane", "A7": "toluene", "A8": "ethylbenzene",
                 "nP8": "n_octane"}
        for k, name in pairs.items():
            assert abs(sp.SPECIES[k].Hf - db[name]["Hf"]) < 1.0e3, (k, sp.SPECIES[k].Hf, db[name]["Hf"])

    def test_crosscheck_table_is_within_its_stated_spread(self):
        for k, vals in sp.HF_CROSSCHECK.items():
            assert vals["API_TDB"] * 1e3 == sp.SPECIES[k].Hf
            spread = max(vals.values()) - min(vals.values())
            assert spread < 1.5, k

    def test_cp_fits_are_smooth_and_physical(self):
        T = np.linspace(298.15, 1000.0, 50)
        cp = np.array([np.asarray(th.cp(t)) for t in T])
        assert np.all(cp > 20.0)
        assert np.all(np.diff(cp[:, 1:], axis=0) > 0)   # every hydrocarbon's Cp rises with T
        # TRC correlation at 298.15 K as evaluated when the fit was made (module docstring):
        ref = {"A6": 82.54, "N6": 106.33, "nP7": 165.19, "A7": 103.80}
        for k, v in ref.items():
            assert float(th.cp(298.15)[sp.INDEX[k]]) == pytest.approx(v, rel=0.015), k

    def test_primary_reference_fuels(self):
        assert sp.SPECIES["nP7"].RON == 0.0 and sp.SPECIES["nP7"].MON == 0.0
        assert sp.SPECIES["nP7"].octane_source == "reference"
        assert {s.octane_source for s in sp.SPECIES.values()} <= {"reference", "recalled", "estimate"}


class TestThermo:
    def test_heat_of_dehydrogenation_from_heats_of_formation(self):
        rm = kn.RateModel()
        i = [r.name for r in rm.reactions].index("dehydrogenation_6")
        # cyclohexane = benzene + 3 H2 at 298.15 K: 82.93 + 123.13 kJ/mol.
        assert float(rm.heats_of_reaction(298.15)[i]) == pytest.approx(206.06e3, abs=1.0)

    def test_equilibrium_constant_is_the_gibbs_energy(self):
        rm = kn.RateModel()
        nu = np.asarray(rm.nu)
        dH = nu @ sp.HF
        dS = nu @ sp.S0
        lnK = -(dH - 298.15 * dS) / (th.R * 298.15)
        np.testing.assert_allclose(np.asarray(th.ln_K(nu, 298.15)), lnK, rtol=1e-12, atol=1e-9)

    def test_vant_hoff(self):
        # d ln K / dT = dH / (R T^2), with dH the coded heat of reaction.
        rm = kn.RateModel()
        T = 760.0
        dlnK = jax.jacfwd(lambda t: th.ln_K(rm.nu_j, t))(T)
        np.testing.assert_allclose(np.asarray(dlnK), np.asarray(rm.heats_of_reaction(T)) / (th.R * T**2),
                                   rtol=1e-10, atol=1e-14)

    def test_dehydrogenation_is_favoured_and_more_so_at_high_T(self):
        rm = kn.RateModel()
        i = [r.name for r in rm.reactions].index("dehydrogenation_7")
        K700, K800 = (float(rm.equilibrium_constants(T)[i]) for T in (700.0, 800.0))
        assert K700 > 1e4 and K800 > 50 * K700

    def test_smith_activation_energies(self):
        k = ReformingKinetics()
        assert k.E["dehydrogenation"] == pytest.approx(34750.0 / 1.8 * th.R)
        assert k.E["ring_opening"] == pytest.approx(59600.0 / 1.8 * th.R)
        assert k.E["hydrocracking_P"] == pytest.approx(62300.0 / 1.8 * th.R)

    def test_enthalpy_is_cubic_thermo_plus_heats_of_formation(self):
        F = jnp.asarray(lean_naphtha().flows).at[sp.INDEX["H2"]].set(400.0)
        flows = {k: F[i] for i, k in enumerate(sp.NAMES)}
        a = th.cubic_thermo().stream_enthalpy(flows, 760.0, "vapor", 15e5) + F @ jnp.asarray(sp.HF)
        assert float(th.total_enthalpy(F, 760.0, 15e5)) == pytest.approx(float(a), rel=1e-12)


class TestNetwork:
    def test_every_reaction_conserves_carbon_and_hydrogen(self):
        for iso in (0.0, 0.5, 1.0):
            _, nu = kn.build_network(iso)
            np.testing.assert_allclose(nu @ sp.ELEMENTS.T, 0.0, atol=1e-12)

    def test_families_present(self):
        fam = {r.family for r in kn.RateModel().reactions}
        assert fam == set(kn.FAMILIES)


# =============================================================================
# Feed
# =============================================================================


class TestFeed:
    def test_piona_mass_and_fractions(self):
        f = lean_naphtha(12.0)
        assert float(f.mass_flow) == pytest.approx(12.0, rel=1e-12)
        g = f.group_fractions("mass")
        assert sum(float(v) for v in g.values()) == pytest.approx(1.0)
        assert float(rich_naphtha().n_plus_2a()) > float(lean_naphtha().n_plus_2a()) + 25

    def test_group_lever(self):
        f = lean_naphtha()
        g = f.with_group_fraction("naphthenes", 0.35)
        assert float(g.group_fractions("volume")["naphthenes"]) == pytest.approx(0.35, rel=1e-12)
        assert float(g.volume_flow) == pytest.approx(float(f.volume_flow), rel=1e-12)

    def test_from_characterization_conserves_mass_and_differentiates(self):
        import difflow_refinery as dr
        from difflow_refinery.composition import CompositionRangeWarning

        assay = dr.Assay([0, 5, 10, 30, 50, 70, 90, 95, 100],
                         [t + 273.15 for t in [20, 55, 90, 190, 290, 400, 540, 600, 700]], sg=0.83,
                         light_ends={"propane": 0.4, "n_butane": 1.2})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CompositionRangeWarning)
            char = dr.characterize(assay, composition=True)
        Tb = np.concatenate([np.zeros(len(char.light_names)), np.asarray(char.Tb)])
        mask = jnp.asarray((Tb > 345.0) & (Tb < 455.0), dtype=float)   # heavy naphtha cuts
        assert float(jnp.sum(mask)) >= 3
        F = jnp.asarray(char.mole_fraction) * 100.0 * mask

        def build(iso):
            return NaphthaFeed.from_characterization(char, F, iso_fraction=iso)

        feed = build(0.5)
        m_in = float(F @ jnp.asarray(char.component_MW)) / 1000.0
        assert float(feed.mass_flow) == pytest.approx(m_in, rel=1e-12)
        g = feed.group_fractions("volume")
        assert 0.0 < float(g["naphthenes"]) < 1.0 and 0.0 < float(g["aromatics"]) < 1.0
        d = jax.jacfwd(lambda x: build(x).flows[sp.INDEX["iP8"]])(0.5)
        fd = (build(0.5 + 1e-4).flows[sp.INDEX["iP8"]] - build(0.5 - 1e-4).flows[sp.INDEX["iP8"]]) / 2e-4
        assert float(d) == pytest.approx(float(fd), rel=1e-8)


# =============================================================================
# One bed
# =============================================================================


def _bed_feed():
    F = jnp.asarray(lean_naphtha().flows) * 1.0
    return F.at[sp.INDEX["H2"]].set(5.0 * float(jnp.sum(F)))


class TestBed:
    def test_adiabatic_and_element_conserving(self):
        F = _bed_feed()
        kin = ReformingKinetics()
        Fo, To, coke, _ = integrate_bed(F, 773.15, 15e5, 3000.0, kin)
        assert float(To) < 773.15 - 30.0
        E = jnp.asarray(sp.ELEMENTS)
        np.testing.assert_allclose(np.asarray(E @ Fo), np.asarray(E @ F), rtol=1e-12)
        H_in, H_out = th.total_enthalpy(F, 773.15, 15e5), th.total_enthalpy(Fo, To, 15e5)
        assert abs(float(H_out - H_in)) < 1e-9 * abs(float(H_in))
        assert float(coke) > 0.0

    def test_long_bed_reaches_dehydrogenation_equilibrium(self):
        # Only the dehydrogenations, a very long bed: the outlet quotient is K(T_out).
        A = {f: 0.0 for f in kn.FAMILIES}
        A["dehydrogenation"] = 0.5
        kin = dataclasses.replace(ReformingKinetics(), A=A)
        F = _bed_feed()
        Fo, To, _, _ = integrate_bed(F, 773.15, 15e5, 2.0e5, kin, rtol=1e-10)
        rm = kn.RateModel()
        i = [r.name for r in rm.reactions].index("dehydrogenation_7")
        y = Fo / jnp.sum(Fo)
        p = y * 15.0
        Q = p[sp.INDEX["A7"]] * p[sp.INDEX["H2"]] ** 3 / p[sp.INDEX["N7"]]
        assert float(Q) == pytest.approx(float(rm.equilibrium_constants(To)[i]), rel=1e-4)

    def test_outlet_temperature_gradient(self):
        F = _bed_feed()
        kin = ReformingKinetics()

        def To(T):
            return integrate_bed(F, T, 15e5, 3000.0, kin, rtol=1e-10)[1]

        g = jax.jacfwd(To)(773.15)
        fd = (To(773.15 + 1e-2) - To(773.15 - 1e-2)) / 2e-2
        assert float(g) == pytest.approx(float(fd), rel=1e-6)


# =============================================================================
# The flowsheet
# =============================================================================


@pytest.fixture(scope="module")
def lean():
    return CatalyticReformer(ReformerParams()).solve(lean_naphtha())


@pytest.fixture(scope="module")
def rich():
    return CatalyticReformer(ReformerParams()).solve(rich_naphtha())


@pytest.mark.slow
class TestFlowsheet:
    @pytest.mark.parametrize("which", ["lean", "rich"])
    def test_converges_and_balances_close(self, which, request):
        res = request.getfixturevalue(which)
        assert res.converged
        for k, v in res.balances().items():
            assert abs(float(v)) < 1e-8, (k, float(v))

    @pytest.mark.release
    @pytest.mark.parametrize("which", ["lean", "rich"])
    def test_first_reactor_drop_is_largest(self, which, request):
        dT = [float(r["dT"]) for r in request.getfixturevalue(which).reactors]
        assert dT[0] < 0 and dT[0] == min(dT)

    @pytest.mark.release
    def test_products_are_plausible(self, lean, rich):
        for res in (lean, rich):
            o = {k: float(v) for k, v in res.outputs().items()}
            assert 60.0 < o["c5plus.yield_vol"] < 100.0
            assert o["h2.net_mol_s"] > 0 and o["h2.purity"] > 70.0
            assert o["lpg.kg_s"] > 0 and o["fuel_gas.kg_s"] > 0
            assert o["reformate.aromatics_vol"] > 40.0
        assert float(rich.outputs()["reformate.RON"]) > float(lean.outputs()["reformate.RON"])

    @pytest.mark.release
    def test_severity_trends(self, lean):
        hot = CatalyticReformer(ReformerParams().with_wait(783.15)).solve(lean_naphtha(),
                                                                           tear_initial=lean.tear)
        a, b = lean.outputs(), hot.outputs()
        assert float(b["reformate.RON"]) > float(a["reformate.RON"])
        assert float(b["h2.net_mol_s"]) > float(a["h2.net_mol_s"])
        assert float(b["c5plus.yield_vol"]) < float(a["c5plus.yield_vol"])

    @pytest.mark.release
    def test_aromatics_rise_as_pressure_falls(self, lean):
        low = CatalyticReformer(ReformerParams(P_separator=8.0e5)).solve(lean_naphtha(),
                                                                         tear_initial=lean.tear)
        assert (float(low.outputs()["reformate.aromatics_vol"])
                > float(lean.outputs()["reformate.aromatics_vol"]))

    def test_reformate_goes_to_the_blend_pool(self, lean):
        from difflow_refinery import BlendComponent, BlendPool
        from difflow_refinery.reforming.products import blend_component

        ref = blend_component("reformate", lean.flows("reformate"))
        alky = BlendComponent.from_properties("alkylate", SG=0.70, RON=96.0, MON=93.5, RVP_psi=4.5,
                                              olefins_vol=0.5, aromatics_vol=0.5, benzene_vol=0.0)
        pool = BlendPool("gasoline", specs=[("RON", ">=", 91.0), ("aromatics_vol", "<=", 35.0)])
        out = pool([ref, alky], [0.5, 0.5])
        assert np.isfinite(float(out.properties["RON"]))
        assert float(out.properties["aromatics_vol"]) == pytest.approx(
            0.5 * float(ref.properties["aromatics_vol"]) + 0.25, rel=1e-9)


@pytest.mark.slow
def test_ron_target_sets_the_wait(lean):
    reformer = CatalyticReformer(ReformerParams())
    wait, res = reformer.wait_for_ron(lean_naphtha(), 95.0, tol=1e-3)
    assert float(res.outputs()["reformate.RON"]) == pytest.approx(95.0, abs=1e-3)
    assert float(wait) > float(ReformerParams().wait)      # base RON is below 95


@pytest.mark.slow
@pytest.mark.release
def test_implicit_gradients_match_central_differences(lean):
    feed0 = lean_naphtha()
    N0 = float(feed0.group_fractions("volume")["naphthenes"])
    keys = ["reformate.yield_vol", "reformate.RON", "h2.net_mol_s", "rx1.dT"]
    tear = CatalyticReformer(ReformerParams(rtol=1e-10)).solve(feed0, tol=1e-11).tear

    def f(x):
        wait, Psep, h2hc, nfrac = x
        p = ReformerParams(rtol=1e-10, P_separator=Psep * 1e5, H2_HC=h2hc).with_wait(wait + 273.15)
        r = CatalyticReformer(p).solve(feed0.with_group_fraction("naphthenes", nfrac),
                                       tear_initial=tear, tol=1e-11, max_iter=400,
                                       on_nonconvergence="ignore")
        o = r.outputs()
        return jnp.stack([o[k] for k in keys])

    x0 = jnp.array([500.0, 12.0, 5.0, N0])
    J = jax.jacfwd(f)(x0)
    h = [0.02, 0.01, 0.005, 0.001]
    for j in range(4):
        e = jnp.zeros(4).at[j].set(h[j])
        fd = (f(x0 + e) - f(x0 - e)) / (2 * h[j])
        np.testing.assert_allclose(np.asarray(J[:, j]), np.asarray(fd), rtol=1e-5, atol=1e-8)
    # Signs a planner relies on: RON up and yield down with WAIT; RON down with pressure.
    assert J[1, 0] > 0 and J[0, 0] < 0 and J[2, 0] > 0 and J[1, 1] < 0


@pytest.mark.slow
def test_reformer_block(lean):
    from difflow_refinery.reforming.planning import reformer_block

    blk = reformer_block(CatalyticReformer(ReformerParams()), lean_naphtha(), ["wait", "H2_HC"],
                         ["reformate.RON", "h2.net_mol_s", "c5plus.yield_vol"], jit=False)
    y0 = blk.evaluate(blk.u0)
    o = lean.outputs()
    np.testing.assert_allclose(np.asarray(y0), [float(o["reformate.RON"]), float(o["h2.net_mol_s"]),
                                                float(o["c5plus.yield_vol"])], rtol=1e-7)


def test_reformate_properties_of_pure_toluene():
    F = jnp.zeros(sp.N_SPECIES).at[sp.INDEX["A7"]].set(1.0)
    p = reformate_properties(F)
    assert float(p["RON"]) == pytest.approx(sp.SPECIES["A7"].RON)
    assert float(p["aromatics_vol"]) == pytest.approx(100.0)
    assert float(p["SG"]) == pytest.approx(872.5 / 999.016, rel=1e-9)
