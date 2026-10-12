"""Tests for liquid-liquid extraction unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    MultistageCascade,
    CascadeParams,
    DifferentialContactor,
    ContactorParams,
    LLEEquilibrium,
    DistributionCoeffs,
    get_K_values,
    separation_factor,
    make_stream,
    get_flows,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_equilibrium():
    """Simple extraction equilibrium with two solutes."""
    K_coeffs = DistributionCoeffs(
        species=("A", "B"),
        K0=(2.0, 0.5),  # A prefers organic, B prefers aqueous
    )

    return LLEEquilibrium(
        solutes=["A", "B"],
        aqueous_carrier="H2O",
        organic_carrier="Solvent",
        K_coeffs=K_coeffs,
    )


class TestDistributionCoeffs:
    def test_get_K_values_constant(self):
        """Test constant K-values (no temperature dependence)."""
        coeffs = DistributionCoeffs(
            species=("A", "B"),
            K0=(2.0, 0.5),
        )
        K = get_K_values(coeffs, T=jnp.array(298.15))

        assert float(K["A"]) == pytest.approx(2.0)
        assert float(K["B"]) == pytest.approx(0.5)

    def test_get_K_values_temperature_dependent(self):
        """Test temperature-dependent K-values."""
        # K(T) = K0 * exp(-dH/R * (1/T - 1/Tref))
        # With positive dH and T > Tref, (1/T - 1/Tref) < 0
        # So -dH/R * (1/T - 1/Tref) > 0, meaning K increases with T
        coeffs = DistributionCoeffs(
            species=("A",),
            K0=(2.0,),
            dH=(10000.0,),  # Positive dH means K increases with T
            Tref=298.15,
        )

        K_ref = get_K_values(coeffs, T=jnp.array(298.15))
        K_hot = get_K_values(coeffs, T=jnp.array(350.0))

        # At Tref, K should equal K0
        assert float(K_ref["A"]) == pytest.approx(2.0)
        # With positive dH, K should increase at higher T
        assert float(K_hot["A"]) > float(K_ref["A"])


class TestSeparationFactor:
    def test_separation_factor(self):
        """Test separation factor calculation."""
        K1 = jnp.array(4.0)
        K2 = jnp.array(2.0)

        SF = separation_factor(K1, K2)
        assert float(SF) == pytest.approx(2.0)


class TestMultistageCascade:
    def test_cascade_creation(self, simple_equilibrium):
        """Test cascade can be created."""
        params = CascadeParams(
            n_stages=5,
            equilibrium=simple_equilibrium,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(params)
        assert cascade is not None

    def test_cascade_mass_balance(self, simple_equilibrium):
        """Test mass balance in cascade."""
        params = CascadeParams(
            n_stages=5,
            equilibrium=simple_equilibrium,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(params)

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15,
            P=101325.0,
        )

        raffinate, extract, info = cascade(feed, solvent)

        # Check mass balance for each solute
        feed_flows = get_flows(feed)
        raff_flows = get_flows(raffinate)
        ext_flows = get_flows(extract)

        for species in ["A", "B"]:
            total_in = float(feed_flows.get(species, 0.0))
            total_out = float(raff_flows.get(species, 0.0)) + float(ext_flows.get(species, 0.0))
            assert total_out == pytest.approx(total_in, rel=1e-6)

    def test_cascade_extraction_direction(self, simple_equilibrium):
        """Test that species with high K are extracted more."""
        params = CascadeParams(
            n_stages=5,
            equilibrium=simple_equilibrium,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(params)

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15,
            P=101325.0,
        )

        raffinate, extract, info = cascade(feed, solvent)

        ext_flows = get_flows(extract)
        feed_flows = get_flows(feed)

        # A (K=2) should be extracted more than B (K=0.5)
        recovery_A = float(ext_flows["A"]) / float(feed_flows["A"])
        recovery_B = float(ext_flows["B"]) / float(feed_flows["B"])

        assert recovery_A > recovery_B

    @pytest.mark.release
    def test_cascade_differentiability(self, simple_equilibrium):
        """Test that cascade is differentiable w.r.t. n_stages."""

        def recovery_A(n_stages):
            params = CascadeParams(
                n_stages=n_stages,
                equilibrium=simple_equilibrium,
                flow_config="counter_current",
            )
            cascade = MultistageCascade(params)

            feed = make_stream(
                {"H2O": 100.0, "A": 1.0, "B": 0.0, "Solvent": 0.0},
                T=298.15,
                P=101325.0,
            )
            solvent = make_stream(
                {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
                T=298.15,
                P=101325.0,
            )

            _, extract, _ = cascade(feed, solvent)
            return extract["F_A"]

        # Compute gradient
        grad_n = jax.grad(recovery_A)(jnp.array(5.0))

        # Gradient should be positive (more stages = more extraction)
        assert float(grad_n) > 0


class TestDifferentialContactor:
    def test_contactor_creation(self, simple_equilibrium):
        """Test contactor can be created."""
        params = ContactorParams(
            length=2.0,
            area=0.1,
            equilibrium=simple_equilibrium,
            HETP=0.5,
        )
        contactor = DifferentialContactor(params)
        assert contactor is not None

    def test_contactor_mass_balance(self, simple_equilibrium):
        """Test mass balance in differential contactor."""
        params = ContactorParams(
            length=2.0,
            area=0.1,
            equilibrium=simple_equilibrium,
            HETP=0.5,
        )
        contactor = DifferentialContactor(params)

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15,
            P=101325.0,
        )

        raffinate, extract, info = contactor(feed, solvent)

        # Check mass balance
        feed_flows = get_flows(feed)
        raff_flows = get_flows(raffinate)
        ext_flows = get_flows(extract)

        for species in ["A", "B"]:
            total_in = float(feed_flows.get(species, 0.0))
            total_out = float(raff_flows.get(species, 0.0)) + float(ext_flows.get(species, 0.0))
            assert total_out == pytest.approx(total_in, rel=1e-4)


class TestCounterCurrentProfiles:
    """Tests for profile population in _solve_counter_current (GitHub issue #61)."""

    def test_counter_current_profiles_populated(self, simple_equilibrium):
        """Profiles dicts must not be empty after calling counter-current cascade."""
        n_stages = 5
        params = CascadeParams(
            n_stages=n_stages,
            equilibrium=simple_equilibrium,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(params)

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15,
            P=101325.0,
        )

        _, _, info = cascade(feed, solvent)
        profiles = info["profiles"]

        # Profiles must contain "x" and "y" keys
        assert "x" in profiles
        assert "y" in profiles

        # Each solute must appear in the profiles
        for species in ["A", "B"]:
            assert species in profiles["x"]
            assert species in profiles["y"]
            # Stage 0 (feed boundary) + n_stages values = n_stages + 1 entries
            assert len(profiles["x"][species]) == n_stages + 1
            assert len(profiles["y"][species]) == n_stages + 1

    def test_counter_current_profiles_monotone(self, simple_equilibrium):
        """Aqueous concentration (x) should decrease monotonically from stage 0 to N."""
        n_stages = 5
        params = CascadeParams(
            n_stages=n_stages,
            equilibrium=simple_equilibrium,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(params)

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 0.0, "Solvent": 0.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15,
            P=101325.0,
        )

        _, _, info = cascade(feed, solvent)
        x_A = [float(v) for v in info["profiles"]["x"]["A"]]

        # x should decrease: feed-end value > value after N stages
        assert x_A[0] > x_A[-1]

    def test_co_current_profiles_populated(self, simple_equilibrium):
        """Profiles dicts must not be empty after calling co-current cascade."""
        n_stages = 4
        params = CascadeParams(
            n_stages=n_stages,
            equilibrium=simple_equilibrium,
            flow_config="co_current",
        )
        cascade = MultistageCascade(params)

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15,
            P=101325.0,
        )

        _, _, info = cascade(feed, solvent)
        profiles = info["profiles"]

        assert "x" in profiles
        assert "y" in profiles

        for species in ["A", "B"]:
            assert species in profiles["x"]
            assert species in profiles["y"]
            # Stage 0 (feed boundary) + n_stages values = n_stages + 1 entries
            assert len(profiles["x"][species]) == n_stages + 1
            assert len(profiles["y"][species]) == n_stages + 1


# =============================================================================
# Partially miscible LLE: ternary_lle_flash and CascadeParams(model="rigorous") (#430)
#
# What is and is not verified
# ---------------------------
# - No published acetic acid-water-MIBK data or NRTL set is used. The ternary below is a
#   SYNTHETIC type-I system with illustrative NRTL parameters (a and b partially miscible,
#   the solute prefers b); species named water / mibk / acetic_acid in the wex17-2 test only
#   label the roles. The recalled literature parameters could not be verified, so none are
#   used and no agreement with Sherwood-Evans-Longcor or DECHEMA data is claimed.
# - ternary_lle_flash is checked against ternary_lle tie lines (an independent Newton
#   continuation), against the binodal polygon on random feeds, and by check_grads.
# - The rigorous cascade is checked against (i) the immiscible Kremser cascade in the limit
#   of vanishing mutual solubility and dilute solute (NRTL tau_ab -> large), including that
#   the discrepancy falls with dilution, (ii) a single flash for N = 1, (iii) an
#   independent block Gauss-Seidel sweep over stage flashes, (iv) mass balance, and (v) the
#   Hunter-Nash stage count of the same equilibrium.
# =============================================================================

import numpy as np  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from difflow.activity import activity_gamma  # noqa: E402
from difflow.phase_diagrams import hunter_nash, ternary_lle  # noqa: E402
from difflow.units.lle import NRTLParams, ternary_lle_flash  # noqa: E402


def _synthetic_ternary(tau_ab=2.5):
    tau = jnp.array([[0.0, tau_ab, 0.5], [tau_ab, 0.0, 0.3], [0.8, 0.6, 0.0]])
    return NRTLParams(("water", "mibk", "acetic_acid"), tau, jnp.zeros((3, 3)), 0.3 * (1 - jnp.eye(3)))


@pytest.fixture(scope="module")
def tmodel():
    return _synthetic_ternary()


@pytest.fixture(scope="module")
def tdata(tmodel):
    return ternary_lle(tmodel, 300.0, n_tie=30)


def _eq(model):
    return LLEEquilibrium(solutes=["acetic_acid"], aqueous_carrier="water", organic_carrier="mibk",
                          nrtl_params=model, activity_model="NRTL")


def _streams(water=80.0, acid=20.0, mibk=100.0):
    feed = make_stream({"water": water, "mibk": 0.0, "acetic_acid": acid}, 300.0, 1e5)
    solv = make_stream({"water": 0.0, "mibk": mibk, "acetic_acid": 0.0}, 300.0, 1e5)
    return feed, solv


@pytest.mark.slow
class TestTernaryLLEFlash:
    @pytest.mark.slow
    def test_reproduces_ternary_lle_tie_lines(self, tmodel, tdata):
        tl = np.asarray(tdata["tie_lines"])
        for i, f in ((2, 0.3), (9, 0.5), (15, 0.7)):
            z = (1 - f) * tl[i, 0] + f * tl[i, 1]
            r = ternary_lle_flash(tmodel, jnp.asarray(z), 300.0)
            assert bool(r["two_phase"]) and bool(r["converged"])
            np.testing.assert_allclose(r["x_I"], tl[i, 0], atol=1e-8)
            np.testing.assert_allclose(r["x_II"], tl[i, 1], atol=1e-8)
            np.testing.assert_allclose(r["beta"], f, atol=1e-8)       # lever rule

    @pytest.mark.slow
    def test_isoactivity_and_balance(self, tmodel):
        z = jnp.array([0.4, 0.5, 0.1])
        r = ternary_lle_flash(tmodel, z, 300.0)
        aI = r["x_I"] * activity_gamma(tmodel, r["x_I"], 300.0)
        aII = r["x_II"] * activity_gamma(tmodel, r["x_II"], 300.0)
        np.testing.assert_allclose(aI, aII, rtol=1e-10)
        np.testing.assert_allclose((1 - r["beta"]) * r["x_I"] + r["beta"] * r["x_II"], z, atol=1e-13)
        np.testing.assert_allclose([r["x_I"].sum(), r["x_II"].sum()], 1.0, atol=1e-13)
        np.testing.assert_allclose(r["K"], r["x_II"] / r["x_I"], rtol=1e-13)

    def test_flows_in_flows_out(self, tmodel):
        z = jnp.array([40.0, 50.0, 10.0])
        r = ternary_lle_flash(tmodel, z, 300.0)
        np.testing.assert_allclose(r["amount_I"] * r["x_I"] + r["amount_II"] * r["x_II"], z, rtol=1e-12)

    @pytest.mark.slow
    def test_agrees_with_the_binodal_on_random_feeds(self, tmodel):
        mpl_path = pytest.importorskip("matplotlib.path")
        d = ternary_lle(tmodel, 300.0, n_tie=60)
        poly = mpl_path.Path(np.asarray(d["binodal"])[:, 1:])      # (c1, c2): affine image of the triangle
        rng = np.random.default_rng(0)
        n_two = 0
        for z in rng.dirichlet(np.ones(3), size=40):
            r = ternary_lle_flash(tmodel, jnp.asarray(z), 300.0)
            inside = poly.contains_point(z[1:])
            assert bool(r["two_phase"]) == bool(inside), z
            n_two += int(inside)
        assert 5 < n_two < 35                                       # both outcomes exercised

    @pytest.mark.slow
    def test_single_phase_is_masked(self, tmodel):
        z = jnp.array([0.97, 0.02, 0.01])
        r = ternary_lle_flash(tmodel, z, 300.0)
        assert not bool(r["two_phase"])
        np.testing.assert_allclose(r["x_I"], z)
        np.testing.assert_allclose(r["x_II"], z)
        g = jax.grad(lambda zz: ternary_lle_flash(tmodel, zz, 300.0)["x_I"][0])(z)
        assert np.all(np.isfinite(np.asarray(g)))

    @pytest.mark.slow
    def test_gradients_wrt_z(self, tmodel):
        z = jnp.array([0.4, 0.5, 0.1])
        for out in ("x_I", "x_II", "beta"):
            f = jax.jit(lambda zz, out=out: ternary_lle_flash(tmodel, zz, 300.0)[out])
            check_grads(f, (z,), order=1, modes=["rev", "fwd"], rtol=1e-5, atol=1e-7)

    def test_gradients_wrt_nrtl_tau(self, tmodel):
        z = jnp.array([0.4, 0.5, 0.1])

        def f(tau):
            m = NRTLParams(tmodel.species, tau, tmodel.b, tmodel.alpha)
            r = ternary_lle_flash(m, z, 300.0)
            return jnp.concatenate([r["x_I"], r["x_II"], r["beta"][None]])
        check_grads(jax.jit(f), (tmodel.a,), order=1, modes=["rev"], rtol=1e-5, atol=1e-7)

    @pytest.mark.slow
    def test_gradient_wrt_T_and_jit_vmap(self):
        tau_a = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        tau_b = jnp.array([[0.0, 750.0, 150.0], [750.0, 0.0, 90.0], [240.0, 180.0, 0.0]])   # tau = b/T
        m = NRTLParams(("a", "b", "s"), tau_a, tau_b, 0.3 * (1 - jnp.eye(3)))
        z = jnp.array([0.4, 0.5, 0.1])
        f = jax.jit(lambda T: ternary_lle_flash(m, z, T)["beta"])
        check_grads(f, (300.0,), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)
        zs = jnp.array([[0.4, 0.5, 0.1], [0.5, 0.45, 0.05]])
        out = jax.jit(jax.vmap(lambda zz: ternary_lle_flash(m, zz, 300.0)["beta"]))(zs)
        assert out.shape == (2,) and np.all(np.asarray(out) > 0)

    @pytest.mark.slow
    def test_guess_and_pair(self, tmodel):
        z = jnp.array([0.4, 0.5, 0.1])
        r0 = ternary_lle_flash(tmodel, z, 300.0)
        r1 = ternary_lle_flash(tmodel, z, 300.0, guess=(r0["x_I"] * 0.99 + 0.0033, r0["x_II"] * 0.99 + 0.0033))
        np.testing.assert_allclose(r1["x_I"], r0["x_I"], atol=1e-9)
        # swapping the pair labels swaps the phases
        r2 = ternary_lle_flash(tmodel, z, 300.0, pair=(1, 0))
        np.testing.assert_allclose(r2["x_I"], r0["x_II"], atol=1e-8)

    def test_accepts_lle_equilibrium(self, tmodel):
        r = ternary_lle_flash(_eq(tmodel), jnp.array([0.4, 0.5, 0.1]), 300.0)
        assert bool(r["two_phase"])


def _gauss_seidel_cascade(model, F, S, N, sweeps=3000):
    """Independent block solve: forward sweep of single-stage flashes, extract feeds lagged."""
    F, S = np.asarray(F), np.asarray(S)
    E = [np.asarray(S) * 0.5 + 1e-3 for _ in range(N)] + [S]            # E_1..E_N, E_{N+1} = S
    R = [None] * N
    for _ in range(sweeps):
        E_old = [e.copy() for e in E]
        R_prev = F
        for n in range(N):
            tot = R_prev + E[n + 1]
            fl = ternary_lle_flash(model, jnp.asarray(tot), 300.0)
            R[n] = np.asarray(fl["amount_I"] * fl["x_I"])
            E[n] = np.asarray(fl["amount_II"] * fl["x_II"])
            R_prev = R[n]
        if max(np.max(np.abs(E[n] - E_old[n])) for n in range(N)) < 1e-12:
            break
    return np.array(R), np.array(E[:N])


class TestRigorousCascade:
    @pytest.mark.slow
    def test_converges_and_conserves_mass_wex17_2_analogue(self, tmodel):
        # three countercurrent stages: acid-like solute from the aqueous feed into the mibk-like solvent
        feed, solv = _streams()
        casc = MultistageCascade(CascadeParams(3, _eq(tmodel), model="rigorous"))
        raff, ext, info = casc(feed, solv)
        assert bool(info["converged"]) and bool(info["two_phase"]) and float(info["residual"]) < 1e-10
        for sp in ("water", "mibk", "acetic_acid"):
            fin = float(feed[f"F_{sp}"]) + float(solv[f"F_{sp}"])
            fout = float(raff[f"F_{sp}"]) + float(ext[f"F_{sp}"])
            assert abs(fin - fout) < 1e-10, sp
        # mutual solubility is represented: the raffinate holds mibk and the extract water
        assert float(raff["F_mibk"]) > 0.5 and float(ext["F_water"]) > 1.0
        assert float(raff["F_acetic_acid"]) < 20.0 * 0.2

    @pytest.mark.slow
    def test_stage_profile_is_isoactive_and_balanced(self, tmodel):
        feed, solv = _streams()
        _, _, info = MultistageCascade(CascadeParams(3, _eq(tmodel), model="rigorous"))(feed, solv)
        P = info["profiles"]
        R, E = np.asarray(P["R"]), np.asarray(P["E"])
        xR, xE = R / R.sum(1, keepdims=True), E / E.sum(1, keepdims=True)
        for n in range(3):
            aR = xR[n] * np.asarray(activity_gamma(tmodel, jnp.asarray(xR[n]), 300.0))
            aE = xE[n] * np.asarray(activity_gamma(tmodel, jnp.asarray(xE[n]), 300.0))
            np.testing.assert_allclose(aR, aE, rtol=1e-9)
        F = np.array([80.0, 0.0, 20.0]); S = np.array([0.0, 100.0, 0.0])
        Rin = np.vstack([F, R[:-1]]); Ein = np.vstack([E[1:], S])
        np.testing.assert_allclose(Rin + Ein, R + E, atol=1e-9)
        assert set(P["x"]) == {"water", "mibk", "acetic_acid"} and len(P["x"]["water"]) == 4

    @pytest.mark.slow
    def test_one_stage_equals_one_flash(self, tmodel):
        feed, solv = _streams()
        raff, ext, info = MultistageCascade(CascadeParams(1, _eq(tmodel), model="rigorous"))(feed, solv)
        fl = ternary_lle_flash(tmodel, jnp.array([80.0, 100.0, 20.0]), 300.0)
        np.testing.assert_allclose([raff["F_water"], raff["F_mibk"], raff["F_acetic_acid"]],
                                   fl["amount_I"] * fl["x_I"], rtol=1e-10)
        np.testing.assert_allclose([ext["F_water"], ext["F_mibk"], ext["F_acetic_acid"]],
                                   fl["amount_II"] * fl["x_II"], rtol=1e-10)

    @pytest.mark.slow
    def test_matches_an_independent_gauss_seidel_stage_sweep(self, tmodel):
        feed, solv = _streams()
        raff, ext, _ = MultistageCascade(CascadeParams(3, _eq(tmodel), model="rigorous"))(feed, solv)
        R, E = _gauss_seidel_cascade(tmodel, [80.0, 0.0, 20.0], [0.0, 100.0, 0.0], 3)
        np.testing.assert_allclose([raff["F_water"], raff["F_mibk"], raff["F_acetic_acid"]], R[-1], rtol=1e-6)
        np.testing.assert_allclose([ext["F_water"], ext["F_mibk"], ext["F_acetic_acid"]], E[0], rtol=1e-6)

    @pytest.mark.slow
    def test_more_stages_extract_more(self, tmodel):
        feed, solv = _streams()
        left = [float(MultistageCascade(CascadeParams(n, _eq(tmodel), model="rigorous"))(feed, solv)[0]["F_acetic_acid"])
                for n in (1, 2, 4)]
        assert left[0] > left[1] > left[2]

    @pytest.mark.slow
    def test_immiscible_limit_equals_the_kremser_cascade(self):
        """Mutual solubility -> 0 (tau_ab = 16) and a dilute solute: the rigorous solve is Kremser's."""
        m = _synthetic_ternary(16.0)
        ga = activity_gamma(m, jnp.array([1 - 1e-12, 1e-18, 1e-12]), 300.0)[2]
        gb = activity_gamma(m, jnp.array([1e-18, 1 - 1e-12, 1e-12]), 300.0)[2]
        K = float(ga / gb)                                    # infinite-dilution K = x_org/x_aq
        eq_k = LLEEquilibrium(solutes=["acetic_acid"], aqueous_carrier="water", organic_carrier="mibk",
                              K_coeffs=DistributionCoeffs(("acetic_acid",), (K,)))
        errs = []
        for dil in (1e-3, 1e-4):
            feed, solv = _streams(water=100.0, acid=100.0 * dil)
            r, _, info = MultistageCascade(CascadeParams(3, _eq(m), model="rigorous"))(feed, solv)
            r0, _, _ = MultistageCascade(CascadeParams(3, eq_k))(feed, solv)
            assert bool(info["converged"])
            errs.append(abs(float(r["F_acetic_acid"]) / float(r0["F_acetic_acid"]) - 1.0))
            # the carriers stay put to the mutual solubility (1e-7 mole fraction here)
            assert float(r["F_mibk"]) < 1e-4
        # the discrepancy is the finite dilution (solute-dependent K), and it falls with dilution
        assert errs[1] < 5e-5 and errs[1] < errs[0] / 5

    @pytest.mark.slow
    def test_hunter_nash_agrees_with_the_rigorous_stage_count(self, tmodel, tdata):
        feed, solv = _streams()
        raff, _, _ = MultistageCascade(CascadeParams(3, _eq(tmodel), model="rigorous"))(feed, solv)
        R = np.array([float(raff["F_water"]), float(raff["F_mibk"]), float(raff["F_acetic_acid"])])
        x_RN = R / R.sum()
        hn = hunter_nash(tdata, np.array([0.8, 0.0, 0.2]), np.array([0.0, 1.0, 0.0]), x_RN, S_over_F=1.0)
        assert hn["feasible"]
        assert abs(hn["n_stages"] - 3.0) < 0.5            # the issue's graphical tolerance
        assert abs(hn["n_stages"] - 3.0) < 0.01           # observed: ~1e-5 stages (same equilibrium model)

    @pytest.mark.slow
    def test_gradients_wrt_solvent_feed_and_tau(self, tmodel):
        feed, _ = _streams()

        def rec(mibk, acid, tau):
            m = NRTLParams(tmodel.species, tau, tmodel.b, tmodel.alpha)
            fd = make_stream({"water": 80.0, "mibk": 0.0, "acetic_acid": acid}, 300.0, 1e5)
            sv = make_stream({"water": 0.0, "mibk": mibk, "acetic_acid": 0.0}, 300.0, 1e5)
            r, _, _ = MultistageCascade(CascadeParams(3, _eq(m), model="rigorous"))(fd, sv)
            return r["F_acetic_acid"]
        f = jax.jit(rec)
        check_grads(f, (100.0, 20.0, tmodel.a), order=1, modes=["rev"], rtol=1e-4, atol=1e-8)
        assert float(jax.grad(f, 0)(100.0, 20.0, tmodel.a)) < 0.0      # more solvent, less solute left

    def test_default_model_is_immiscible(self):
        eq = LLEEquilibrium(solutes=["A"], aqueous_carrier="H2O", organic_carrier="S",
                            K_coeffs=DistributionCoeffs(("A",), (2.0,)))
        assert CascadeParams(3, eq).model == "immiscible"

    def test_unsupported_configurations_raise(self, tmodel):
        feed, solv = _streams()
        with pytest.raises(ValueError, match="model must be"):
            CascadeParams(3, _eq(tmodel), model="exact")
        with pytest.raises(ValueError, match="counter_current"):
            MultistageCascade(CascadeParams(3, _eq(tmodel), flow_config="co_current", model="rigorous"))(feed, solv)
        with pytest.raises(ValueError, match="integer"):
            MultistageCascade(CascadeParams(2.5, _eq(tmodel), model="rigorous"))(feed, solv)
        eqk = LLEEquilibrium(solutes=["A"], aqueous_carrier="water", organic_carrier="mibk",
                             K_coeffs=DistributionCoeffs(("A",), (2.0,)))
        with pytest.raises(ValueError, match="NRTL"):
            MultistageCascade(CascadeParams(3, eqk, model="rigorous"))(feed, solv)
        ms = _eq(tmodel)
        ms.mutual_solubility = {"aqueous_in_organic": 0.01}
        with pytest.raises(ValueError, match="supersedes"):
            MultistageCascade(CascadeParams(3, ms, model="rigorous"))(feed, solv)
        feed0 = make_stream({"water": 80.0, "mibk": 0.0, "acetic_acid": 0.0}, 300.0, 1e5)
        with pytest.raises(ValueError, match="zero flow"):
            MultistageCascade(CascadeParams(3, _eq(tmodel), model="rigorous"))(feed0, solv)

    @pytest.mark.slow
    def test_single_phase_stages_are_flagged_not_converged(self, tmodel):
        # too little solvent: the stages mix into one liquid, which the two-phase stage model cannot represent
        feed, solv = _streams(water=80.0, acid=20.0, mibk=5.0)
        _, _, info = MultistageCascade(CascadeParams(3, _eq(tmodel), model="rigorous"))(feed, solv)
        assert not bool(info["converged"])
