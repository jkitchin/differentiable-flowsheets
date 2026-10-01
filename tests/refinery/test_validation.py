"""The crude unit checked against an independent reference (issue #298).

What the reference is, and why it is not DWSIM: the issue asked for a
commercial or open simulator's crude case. DWSIM, HYSYS and PRO/II were not
available; IDAES (the U.S. DOE's equation-oriented process modelling
platform, on Pyomo and IPOPT) was. ``reference/generate.py`` built the
reference with it and wrote ``reference/cdu_reference.json``, with the tool
versions in its ``provenance`` block. These tests read that file and need
neither IDAES nor IPOPT.

Four layers, each with its own tolerance and the reason for it:

1. **Characterisation** against *published* numbers -- worked examples from
   ASTM MNL50 (Riazi 2005) and Ahmed (2016), each cited where it is
   asserted, and the docstring examples of Caleb Bell's ``chemicals``.
2. **Thermodynamics** against IDAES's generic property framework: the ideal
   model difflow states, assembled from IDAES's own state-block equations and
   solved by IPOPT at the reference column's stage states; and Peng-Robinson
   on the same constants, whose *differences* are pinned here as documented
   findings rather than tolerated or tuned away.
3. **The column** against a separately written equation-oriented model
   (``reference/mesh.py``: every stage, stripper, pumparound, condenser and
   the furnace flash as Pyomo constraints) solved by IPOPT from a starting
   point that owes nothing to difflow's solution, with the same specs.
4. **Gradients**: ``jax.grad`` through difflow's solve against central
   finite differences of the reference column.

The whole module is ``release``: its subject is the answer, and a deliberate
model change is what moves it. When one does -- a new characterisation, a
new property method -- :class:`TestReferenceIsCurrent` fails first and says
to regenerate (see ``reference/generate.py``'s docstring for the command).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .reference import case
from .reference import formulas as fm

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.release

REF_PATH = Path(__file__).parent / "reference" / "cdu_reference.json"
REF = json.loads(REF_PATH.read_text())
COMP = REF["components"]
NAMES = COMP["names"]
NLE = COMP["n_light_ends"]
PRODUCTS = ("naphtha", "kero", "diesel", "ago", "residue")

REGENERATE = ("difflow's characterisation no longer matches the reference's component table; "
              "regenerate it: PYTHONPATH=src:tests python -m refinery.reference.generate "
              "(needs idaes-pse and ipopt), and re-read the Validation section of "
              "docs/unit-operations-refinery.md against the new numbers")


@pytest.fixture(scope="module")
def difflow_case():
    return case.difflow_case()


@pytest.fixture(scope="module")
def thermo(difflow_case):
    return difflow_case[2]


@pytest.fixture(scope="module")
def solved(difflow_case):
    unit, _, _, Vf = difflow_case
    return case.solve_difflow(unit, Vf)


class TestReferenceFile:
    """Cheap and not ``release``: a deleted or truncated reference is wiring."""

    pytestmark = []

    def test_it_records_where_it_came_from(self):
        p = REF["provenance"]
        for key in ("generated", "script", "idaes", "pyomo", "ipopt", "property_methods",
                    "reference_simulator", "difflow_commit"):
            assert p[key], key
        assert "DWSIM" in p["reference_simulator"]
        assert Path(__file__).parent.joinpath("reference", "generate.py").exists()

    def test_every_layer_is_there(self):
        for key in ("layer1", "layer2", "layer3", "layer4"):
            assert REF[key], key
        assert REF["layer3"]["solve"]["status"] == "optimal"


class TestReferenceIsCurrent:
    """The reference was built on difflow's pseudo-components (an input to
    both models); if they have moved, every comparison below is against a
    different crude, and the right response is to regenerate, not loosen."""

    def test_the_component_table_is_the_one_the_reference_used(self, difflow_case):
        unit, crude, thermo, _ = difflow_case
        now = case.component_data(crude, thermo)
        assert now["names"] == NAMES, REGENERATE
        for key in ("MW", "SG", "Tb", "Tc", "Pc", "omega_vp", "omega_eos", "hvap_nb",
                    "volume_fraction", "mole_fraction"):
            np.testing.assert_allclose(now[key], COMP[key], rtol=1e-9, err_msg=f"{key}: {REGENERATE}")
        np.testing.assert_allclose(now["cp_ig"], COMP["cp_ig"], rtol=1e-9, err_msg=REGENERATE)
        np.testing.assert_allclose(case.feed_flows(unit, thermo), REF["case"]["feed_mol_s"],
                                   rtol=1e-9, err_msg=REGENERATE)

    def test_the_case_is_the_one_the_reference_solved(self):
        assert REF["case"]["specs"] == json.loads(json.dumps(case.SPECS))
        assert REF["case"]["layout"] == json.loads(json.dumps(case.LAYOUT))


# =============================================================================
# Layer 1: characterisation against published values
# =============================================================================

#: ``(source, method, Tb K, SG, quantity, published value, rel tol, why)``.
#: Pc in bar, Tc in K. Tolerances are the printed precision unless the reason
#: says otherwise.
_R = 5 / 9  # degR -> K
PUBLISHED = [
    # Riazi, ASTM MNL50 (2005), Example 2.7 / Table 2.11: n-hexatriacontane
    ("MNL50 Ex. 2.7", "riazi_daubert_1980", 770.2, 0.8172, "MW", 445.6, 1e-3, "printed to 0.1"),
    ("MNL50 Ex. 2.7", "riazi_daubert_1980", 770.2, 0.8172, "Tc", 885.8, 1e-3, "printed to 0.1 K"),
    ("MNL50 Ex. 2.7", "riazi_daubert_1980", 770.2, 0.8172, "Pc", 7.3, 1e-2, "printed to 0.1 bar"),
    ("MNL50 Ex. 2.7", "riazi_daubert_1987", 770.2, 0.8172, "MW", 512.7, 1e-3, "printed to 0.1"),
    ("MNL50 Ex. 2.7", "riazi_daubert_1987", 770.2, 0.8172, "Tc", 879.3, 1e-3, "printed to 0.1 K"),
    # Twu: our M and Tc reproduce the table; our Vc reproduces the printed
    # 2010 cm3/mol and Pc0 = 6.015 bar, but the printed Pc (6.02 here, 6.03
    # in Table 2.12) is Pc0 without Twu's f_P correction, which takes ours to
    # 5.965. 1.5 % covers that, and is the stated reason for it.
    ("MNL50 Ex. 2.7", "twu", 770.2, 0.8172, "MW", 513.8, 3e-3, "M from Twu's iterative "
     "Tb(M) inversion; the example rounds the intermediate theta"),
    ("MNL50 Ex. 2.7", "twu", 770.2, 0.8172, "Tc", 882.1, 5e-4, "printed to 0.1 K"),
    ("MNL50 Ex. 2.7", "twu", 770.2, 0.8172, "Pc", 6.02, 1.5e-2, "the printed Pc omits f_P "
     "(see comment)"),
    # MNL50 Example 2.5: n-butylbenzene
    ("MNL50 Ex. 2.5", "riazi_daubert_1980", 456.45, 0.866, "MW", 133.2, 1e-3, "printed to 0.1"),
    ("MNL50 Ex. 2.5", "riazi_daubert_1987", 456.45, 0.866, "MW", 139.2, 1e-3, "printed to 0.1"),
    ("MNL50 Ex. 2.5", "lee_kesler", 456.45, 0.866, "MW", 143.4, 1e-3, "printed to 0.1"),
    # Ahmed, Equations of State and PVT Analysis, 2nd ed. (2016), Example 2.2:
    # Tb 198 F, SG 0.7365, answers in degR and psia, printed to 3 figures
    ("Ahmed Ex. 2.2", "riazi_daubert_1980", (198 + 459.67) * _R, 0.7365, "MW", 96, 5e-3, "3 figures"),
    ("Ahmed Ex. 2.2", "riazi_daubert_1980", (198 + 459.67) * _R, 0.7365, "Tc", 990 * _R, 1e-3, "3 figures"),
    ("Ahmed Ex. 2.2", "riazi_daubert_1980", (198 + 459.67) * _R, 0.7365, "Pc", 467 * 0.0689476, 2e-3, "3 figures"),
    ("Ahmed Ex. 2.2", "riazi_daubert_1987", (198 + 459.67) * _R, 0.7365, "MW", 97, 5e-3, "3 figures"),
    ("Ahmed Ex. 2.2", "riazi_daubert_1987", (198 + 459.67) * _R, 0.7365, "Tc", 986 * _R, 1e-3, "3 figures"),
    ("Ahmed Ex. 2.2", "riazi_daubert_1987", (198 + 459.67) * _R, 0.7365, "Pc", 466 * 0.0689476, 2e-3, "3 figures"),
    ("Ahmed Ex. 2.2", "lee_kesler", (198 + 459.67) * _R, 0.7365, "MW", 98.6, 1e-3, "printed to 0.1"),
    ("Ahmed Ex. 2.2", "lee_kesler", (198 + 459.67) * _R, 0.7365, "Tc", 981 * _R, 1e-3, "3 figures"),
    ("Ahmed Ex. 2.2", "lee_kesler", (198 + 459.67) * _R, 0.7365, "Pc", 470 * 0.0689476, 2e-3, "3 figures"),
]


class TestPublishedCharacterisation:
    @pytest.mark.parametrize("src,method,Tb,SG,q,expected,rtol,why", PUBLISHED,
                             ids=[f"{r[0]}-{r[1]}-{r[4]}" for r in PUBLISHED])
    def test_critical_properties(self, src, method, Tb, SG, q, expected, rtol, why):
        from difflow_refinery.correlations import critical_properties

        MW, Tc, Pc = (float(v) for v in critical_properties(Tb, SG, method=method))
        got = {"MW": MW, "Tc": Tc, "Pc": Pc / 1e5}[q]
        assert got == pytest.approx(expected, rel=rtol), f"{src} {method} {q}: {why}"

    def test_tbp_to_d86_reproduces_mnl50_example_3_3(self):
        """MNL50 Example 3.3 / Table 3.8: a kerosene's TBP (Riazi-Daubert
        column) to ASTM D86. Printed to 0.1 C; ours agrees within 0.1 C."""
        from difflow_refinery.blending import tbp_to_d86

        tbp_C = {0: 134.1, 10: 160.6, 30: 188.2, 50: 208.9, 70: 230.2, 90: 254.7}
        d86_C = {0: 165.6, 10: 176.7, 30: 193.3, 50: 206.7, 70: 222.8, 90: 242.8}
        for p, t in tbp_C.items():
            assert float(tbp_to_d86(t + 273.15, p)) - 273.15 == pytest.approx(d86_C[p], abs=0.1), p

    def test_watson_k_reproduces_the_api_databook_example(self):
        """API TDB Procedure 2B4.1 example as quoted in ``chemicals.Watson_K``:
        MeABP 580 F, 34.5 API gives K = 11.8846 (to the printed digits)."""
        from difflow_refinery.correlations import sg_from_api, watson_k

        Tb = (580 + 459.67) * _R
        assert float(watson_k(Tb, sg_from_api(34.5))) == pytest.approx(11.88457, rel=1e-5)

    def test_lee_kesler_and_riedel_reproduce_their_worked_examples(self):
        """Lee-Kesler vapour pressure, ethylbenzene at 347.2 K (Tc 617.1 K,
        Pc 36 bar, omega 0.299): 13078.69 Pa. Riedel's latent heat at Tb,
        pyridine (Tb 388.4 K, Tc 620 K, Pc 56.3 bar): 35089.8 J/mol, against
        35090 measured. Both are the examples ``chemicals`` 1.5 documents."""
        from difflow_refinery.correlations import hvap_at_tb, vapor_pressure

        assert float(vapor_pressure(347.2, 617.1, 36e5, 0.299)) == pytest.approx(13078.69, rel=1e-6)
        assert float(hvap_at_tb(388.4, 620.0, 56.3e5)) == pytest.approx(35089.8, rel=1e-5)

    def test_chemicals_agrees_on_every_pseudo_component(self):
        """The same equations from an independent code base (``chemicals``),
        on this crude's own constants: Lee-Kesler Psat at four temperatures,
        the Lee-Kesler acentric factor that Psat uses, and Riedel's latent
        heat for the cuts."""
        l1 = REF["layer1"]
        if not l1:
            pytest.skip("reference generated without chemicals")
        Tc, Pc, w, Tb = (np.asarray(COMP[k]) for k in ("Tc", "Pc", "omega_vp", "Tb"))
        for T, ref in l1["Lee_Kesler_Psat"].items():
            np.testing.assert_allclose(fm.lee_kesler_psat(fm.NUMPY, float(T), Tc, Pc, w), ref, rtol=1e-10)
        from difflow_refinery.correlations import vapor_pressure

        for T, ref in l1["Lee_Kesler_Psat"].items():
            np.testing.assert_allclose(vapor_pressure(float(T), Tc, Pc, w), ref, rtol=1e-10)
        # omega_vp is the Lee-Kesler inversion for every component (the EOS
        # omega is the one that switches to Kesler-Lee above Tbr = 0.8)
        np.testing.assert_allclose(COMP["omega_vp"][NLE:], l1["LK_omega"][NLE:], rtol=1e-8)
        np.testing.assert_allclose(COMP["hvap_nb"][NLE:], l1["Riedel_hvap"][NLE:], rtol=1e-10)


# =============================================================================
# Layer 2: the property model against IDAES
# =============================================================================


def _stage_state(stage):
    s = REF["layer2"]["ideal"]["smoothed"][stage]
    return s


STAGES = [k for k in REF["layer2"]["ideal"]["smoothed"]]


class TestIdealModelAgainstIDAES:
    """IDAES's generic framework with an ideal EOS, Raoult VLE, the same pure
    component methods (``reference/idaes_thermo.py``), solved by IPOPT at each
    stage's T and P and overall composition. It is the same model, so the
    agreement should be at the solver's precision: what this checks is that
    difflow's arrays implement the equations it states, not the equations."""

    @pytest.mark.parametrize("stage", STAGES)
    def test_k_values(self, thermo, stage):
        s = _stage_state(stage)
        assert s["ok"]
        T, P = _stage_TP(stage)
        K = np.asarray(thermo.psat(jnp.asarray(T))) / P
        # each K is y_i/x_i of IPOPT's solution, whose tolerance is absolute
        # on the mole fractions: a component at 1e-12 (the floor a stripped-out
        # one is held at) carries no K to compare. Above 1e-8, 1e-6 relative.
        live = np.asarray(s["x"]) > 1e-8
        assert live.sum() >= 10
        np.testing.assert_allclose(np.asarray(s["K"])[live], K[live], rtol=1e-6)

    @pytest.mark.parametrize("stage", STAGES)
    def test_phase_enthalpies(self, thermo, stage):
        s = _stage_state(stage)
        T, _ = _stage_TP(stage)
        x, y, yw = np.asarray(s["x"]), np.asarray(s["y"]), s["yw"]
        hL = float(np.sum(x * np.asarray(thermo.h_liquid(jnp.asarray(T)))))
        # IDAES's vapour fractions run over hydrocarbons and steam together
        hV = float(np.sum(y * np.asarray(thermo.h_vapor(jnp.asarray(T))))
                   + yw * float(thermo.water_h_vapor(jnp.asarray(T))))
        # J/mol on ~1e5: 1e-6 relative is well inside the solver tolerance's
        # footprint and well outside a wrong datum or a wrong latent heat
        assert s["h_liq"] == pytest.approx(hL, rel=1e-6)
        assert s["h_vap"] == pytest.approx(hV, rel=1e-6)

    @pytest.mark.parametrize("stage", STAGES)
    def test_the_stage_splits_back_into_its_own_liquid_and_vapour(self, stage):
        """Flashing a stage's total contents at its T and P must return the
        reference column's L and V: equilibrium and summation are the same
        equations in both, solved twice by different routes."""
        s = _stage_state(stage)
        st = REF["layer3"]["column"]
        i = int(stage[1:]) - 1
        assert s["L"] == pytest.approx(st["L"][i], rel=1e-5)
        assert s["V"] == pytest.approx(st["V"][i], rel=1e-5)

    def test_crude_flash_at_the_coil_outlet(self, thermo):
        """The crude alone at the reference column's coil outlet T and the
        feed-stage pressure: IDAES's vapour fraction against Rachford-Rice on
        difflow's K, and its bubble and dew points against difflow's
        psat (sum z K = 1 and sum z / K = 1)."""
        c = REF["layer2"]["crude"]
        assert c["ok"]
        z = np.asarray(COMP["mole_fraction"])
        K = np.asarray(thermo.psat(jnp.asarray(c["T"]))) / c["P"]
        assert c["feed_vaporized"] == pytest.approx(fm.rachford_rice(z, K), abs=1e-7)
        Kb = np.asarray(thermo.psat(jnp.asarray(c["T_bubble"]))) / c["P"]
        Kd = np.asarray(thermo.psat(jnp.asarray(c["T_dew"]))) / c["P"]
        assert float(np.sum(z * Kb)) == pytest.approx(1.0, abs=1e-6)
        assert float(np.sum(z / Kd)) == pytest.approx(1.0, abs=1e-6)

    def test_the_watson_smoothing_moves_only_the_supercritical_light_ends(self, thermo):
        """difflow floors Watson's ``1 - T/Tc`` smoothly (eps = 0.01) so the
        latent heat of a component above its Tc has a derivative. IDAES with
        the exact ``max(1 - T/Tc, 0)`` gives the cost: at every stage the
        liquid enthalpy moves by less than 0.1 %, because only the light ends
        near or above Tc see the floor and they are a trace of the liquid."""
        for stage in STAGES:
            a = REF["layer2"]["ideal"]["smoothed"][stage]
            b = REF["layer2"]["ideal"]["exact"][stage]
            assert abs(a["h_liq"] - b["h_liq"]) < 1e-3 * abs(a["h_liq"]), stage


def _stage_TP(stage):
    i = int(stage[1:]) - 1
    lay = case.LAYOUT
    T = REF["layer3"]["column"]["stage_T"][i]
    P = lay["P_top"] + i * (lay["P_bottom"] - lay["P_top"]) / (lay["n_stages"] - 1)
    return T, P


class TestPengRobinsonDifferences:
    """Peng-Robinson on the same Tc, Pc, omega and ideal-gas Cp, solved by
    IDAES at the reference's stage states (hydrocarbons at their partial
    pressure) and at the furnace inlet and coil outlet. These are the
    *modelling* differences -- what choosing Raoult and Watson over a cubic
    costs on this crude -- and they are pinned, not tolerated: if a change
    moves them out of these bands, the Validation section of
    docs/unit-operations-refinery.md says the wrong thing and needs
    rewriting along with the band."""

    pr = REF["layer2"]["pr"]

    def test_every_state_converged_two_phase(self):
        for k, s in self.pr.items():
            assert s["ok"], (k, s["termination"])
            assert 0.0 < s["beta"] < 1.0, k

    def test_the_vapour_fraction_on_the_stages_agrees_within_two_mole_percent(self):
        """At every stage state and at the coil outlet PR and Raoult vaporise
        the same hydrocarbons to within 0.021 (measured worst: stage 22); at
        the coil outlet, where the furnace closes, 0.0035."""
        for k, s in self.pr.items():
            if k != "furnace_inlet":
                assert abs(s["beta"] - s["beta_ideal"]) < 0.025, k
        s = self.pr["coil_outlet"]
        assert abs(s["beta"] - s["beta_ideal"]) < 0.005

    def test_but_not_at_the_furnace_inlet(self):
        """At 240 C and 6 bar Raoult vaporises 22 % of the crude (molar),
        PR 15 %: the light ends, which carry that vapour, are supercritical
        or nearly so, and that is where the two part (next test)."""
        s = self.pr["furnace_inlet"]
        assert 0.05 < s["beta_ideal"] - s["beta"] < 0.09

    def test_the_furnace_enthalpy_rise_is_four_percent_higher_under_pr(self):
        """The furnace duty at a fixed coil outlet is ``H(COT) - H(inlet)``.
        PR's is 4.0 % above Raoult/Watson's on this crude (59.3 against 57.0
        kJ/mol): the size of the furnace-duty difference a PR crude case
        would show, from the property model alone."""
        pr = self.pr
        d_pr = pr["coil_outlet"]["h"] - pr["furnace_inlet"]["h"]
        d_id = pr["coil_outlet"]["h_ideal"] - pr["furnace_inlet"]["h_ideal"]
        assert 0.02 < d_pr / d_id - 1 < 0.06

    def test_raoult_overpredicts_the_supercritical_light_ends(self):
        """Lee-Kesler Psat extrapolated past Tc keeps rising steeply; a cubic's
        fugacity does not. At the flash zone propane's Raoult K is about 60
        times PR's, and every supercritical component's is larger --
        harmless for the products (the light ends go overhead either way),
        but the reason not to read a light-ends K-value off this model."""
        s = self.pr["coil_outlet"]
        ratio = np.asarray(s["K"]) / fm.Props(COMP).K(s["T"], s["P_hc"])
        sup = np.asarray(COMP["Tc"]) < s["T"]
        assert ratio[0] < 0.05
        assert np.all(ratio[sup] < 1.0)

    def test_middle_cuts_agree_and_the_heaviest_diverge(self):
        """For the cuts boiling 420-640 K, which make the side products, PR
        and Raoult K agree within -30 %/+25 % at the flash zone. For the
        heaviest residue cut PR's K is about 20 times Raoult's: Lee-Kesler's
        Psat at Tr ~ 0.5 with omega above 1 against a cubic with the same
        omega. Residue leaves as liquid under both, so this shows in the
        overflash composition, not the yields."""
        s = self.pr["coil_outlet"]
        ratio = np.asarray(s["K"]) / fm.Props(COMP).K(s["T"], s["P_hc"])
        Tb = np.asarray(COMP["Tb"])
        mid = (Tb > 420) & (Tb < 640)
        assert np.all((ratio[mid] > 0.7) & (ratio[mid] < 1.25)), ratio[mid]
        assert ratio[-1] > 10.0

    def test_liquid_enthalpy_at_the_same_composition(self):
        """PR's departure function against Watson's latent heat, both at PR's
        liquid composition so that composition is taken out. On the stages
        above the flash zone (light and middle cuts) they agree within 1.5
        kJ/mol; where residue is in the liquid (flash zone, stripping
        section) PR's liquid sits 13-18 kJ/mol higher. Watson anchored at a
        Riedel latent heat and PR with an extrapolated omega are both
        extrapolations for a 1000 K boiling cut; nothing here says which is
        right, only how far apart they are."""
        for k, s in self.pr.items():
            d = s["h_liq"] - s["h_liq_ideal_same_x"]
            if k in ("s1", "s9", "s12", "s16", "s19", "s22"):
                assert abs(d) < 1500, (k, d)
            elif k in ("s27", "s30", "coil_outlet"):
                assert 10e3 < d < 20e3, (k, d)


class TestWater:
    def test_wagner_pruss_against_iapws(self, thermo):
        """difflow's water Psat (the Wagner-Pruss auxiliary equation) against
        IAPWS-95 itself (as ``chemicals`` evaluates it), 300-550 K: within
        0.004 %. Against IAPWS-IF97 region 4, the industrial formulation and a
        different equation, within 0.02 % -- the two IAPWS formulations are
        that far apart from each other here."""
        w = REF["layer2"]["water"]
        got = np.asarray([float(thermo.water_psat(jnp.asarray(T))) for T in w["T"]])
        np.testing.assert_allclose(got, w["Psat_IF97"], rtol=2e-4)
        if "Psat_IAPWS95" in w:
            np.testing.assert_allclose(got, w["Psat_IAPWS95"], rtol=5e-5)

    def test_watson_latent_heat_against_iapws95(self, thermo):
        """Water's latent heat from Watson anchored at 40.66 kJ/mol at Tb,
        against IAPWS-95 by Clausius-Clapeyron: exact at Tb, 1.3 % high at
        300 K and 2.6 % low at 550 K (n = 0.38 is fitted to organics).
        Water never condenses in the column as modelled (it is vapour-only
        and the water-saturation result flags where it would), so this
        reaches the answer only through the condenser's decanted water."""
        w = REF["layer2"]["water"]
        if "hvap_IAPWS95" not in w:
            pytest.skip("reference generated without chemicals")
        got = np.asarray([float(thermo.water_h_vapor(jnp.asarray(T)) - thermo.water_h_liquid(jnp.asarray(T)))
                          for T in w["T"]])
        np.testing.assert_allclose(got, w["hvap_IAPWS95"], rtol=3e-2)


# =============================================================================
# Layer 3: the column
# =============================================================================


@pytest.mark.slow
class TestColumnAgainstReference:
    """Same specs, same property model, independently written equations and
    an independent solver started from a crude guess. The model is the same,
    so the tolerances are the two solvers' precision -- loosen one and a
    transcription error in either column is free to hide inside it."""

    ref = REF["layer3"]["column"]

    def test_both_converged(self, solved):
        assert bool(solved.converged)
        assert REF["layer3"]["solve"]["status"] == "optimal"

    def test_temperature_profile(self, solved):
        np.testing.assert_allclose(np.asarray(solved.column.T), self.ref["stage_T"], atol=1e-3)
        assert float(solved.column.T_condenser) == pytest.approx(self.ref["T_condenser"], abs=1e-3)
        assert float(solved.column.coil_outlet_T) == pytest.approx(self.ref["coil_outlet_T"], abs=1e-3)

    def test_stripper_and_draw_temperatures(self, solved):
        flat = [t for d in ("kero", "diesel", "ago") for t in self.ref["stripper_T"][d]]
        np.testing.assert_allclose(np.asarray(solved.column.stripper_T), flat, atol=1e-3)

    def test_duties(self, solved):
        c = solved.column
        assert float(c.condenser_duty) == pytest.approx(self.ref["condenser_duty"], rel=1e-5)
        assert float(c.furnace_duty) == pytest.approx(self.ref["furnace_duty"], rel=1e-5)
        assert float(c.furnace_fired_duty) == pytest.approx(self.ref["fired_duty"], rel=1e-5)
        np.testing.assert_allclose(np.asarray(c.pumparound_duty),
                                   [self.ref["pumparound_duty"][p] for p in ("pa1", "pa2")], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(c.pumparound_return_T),
                                   [self.ref["pumparound_return_T"][p] for p in ("pa1", "pa2")], atol=1e-3)
        assert float(c.feed_vaporized) == pytest.approx(self.ref["feed_vaporized"], abs=1e-6)

    def test_yields_gravity_and_tbp(self, solved):
        for p in PRODUCTS:
            got, ref = solved.properties[p], self.ref["products"][p]
            assert float(got.yield_volume) == pytest.approx(ref["volume_yield"], rel=1e-5), p
            assert float(got.sg) == pytest.approx(ref["SG"], rel=1e-6), p
            assert float(got.api) == pytest.approx(ref["API"], abs=1e-4), p
            for pct, T in ref["TBP"].items():
                assert float(got.tbp_at(int(pct))) == pytest.approx(T, abs=1e-2), (p, pct)

    def test_gaps(self, solved):
        g = solved.gaps(list(PRODUCTS))
        for k, v in self.ref["gaps"].items():
            a, b = k.split("-")
            assert float(g[(a, b)]) == pytest.approx(v, abs=2e-2), k

    def test_steam_and_water(self, solved):
        """Steam's approach to saturation (``y_w P / Psat_w``), stage by stage:
        below 1 everywhere, so the vapour-only water model is self-consistent
        in this case."""
        # the reference divides by IAPWS-IF97's Psat, difflow by Wagner-Pruss;
        # they differ by up to 0.02 % (TestWater), and that is all that differs
        np.testing.assert_allclose(np.asarray(solved.column.water_saturation),
                                   self.ref["water_saturation"], rtol=3e-4)
        assert self.ref["max_water_saturation"] < 1.0


# =============================================================================
# Layer 4: gradients
# =============================================================================


@pytest.mark.slow
class TestGradientsAgainstReference:
    """``jax.grad`` through difflow's implicit-function solve against central
    differences of the reference column (step 2e-3 in each spec fraction,
    re-solved warm by IPOPT). The difference quotient carries an O(h^2)
    truncation and the solve tolerance over 2h; measured agreement is 2e-5
    and 3e-5, so 0.1 % leaves room for neither model to be wrong."""

    @pytest.fixture(scope="class")
    def unit_vf(self, difflow_case):
        return difflow_case[0], difflow_case[3]

    def test_diesel_api_with_diesel_yield(self, unit_vf):
        unit, Vf = unit_vf
        ref = REF["layer4"]["d_diesel_api_d_diesel_yield"]

        def f(y):
            return case.solve_difflow(unit, Vf, volume_yield={"diesel": y}).properties["diesel"].api

        assert float(jax.grad(f)(ref["x0"])) == pytest.approx(ref["value"], rel=1e-3)

    def test_fired_duty_with_overflash(self, unit_vf):
        unit, Vf = unit_vf
        ref = REF["layer4"]["d_fired_duty_d_overflash"]

        def f(of):
            return case.solve_difflow(unit, Vf, overflash=of).column.furnace_fired_duty

        assert float(jax.grad(f)(ref["x0"])) == pytest.approx(ref["value"], rel=1e-3)


# =============================================================================
# Regeneration
# =============================================================================


@pytest.mark.slow
def test_regeneration_reproduces_the_committed_reference(tmp_path):
    """Optional: rebuild the reference (without finite differences) and check
    it lands on the committed numbers. Needs IDAES and its IPOPT."""
    pytest.importorskip("idaes")
    import shutil

    from .reference import generate

    ipopt = shutil.which("ipopt") or generate.DEFAULT_IPOPT
    if not Path(ipopt).exists():
        pytest.skip("no ipopt")
    out = tmp_path / "ref.json"
    generate.main(["--ipopt", ipopt, "--out", str(out), "--skip-fd"])
    new = json.loads(out.read_text())["layer3"]["column"]
    old = REF["layer3"]["column"]
    np.testing.assert_allclose(new["stage_T"], old["stage_T"], atol=1e-6)
    assert new["fired_duty"] == pytest.approx(old["fired_duty"], rel=1e-8)
