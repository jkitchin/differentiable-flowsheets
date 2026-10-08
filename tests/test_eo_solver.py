"""Tests for the equation-oriented (EO) solver."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    Flash,
    FlashParams,
    Mixer,
    Splitter,
    Heater,
    HeaterParams,
    Cooler,
    CoolerParams,
    IdealThermo,
    SpeciesData,
    Flowsheet,
    Unit,
    make_stream,
    get_flows,
    EOSolver,
    EOSolveResult,
    EOStateLayout,
)

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_thermo():
    """Simple two-component thermodynamics."""
    species_data = {
        "A": SpeciesData(
            "A",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 500.0),
            antoine_coeffs=(10.0, 3000.0, -50.0),
        ),
        "B": SpeciesData(
            "B",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 450.0),
            antoine_coeffs=(10.0, 2800.0, -40.0),
        ),
    }
    return IdealThermo(species_data)


@pytest.fixture
def simple_rate_fn():
    """First-order reaction rate function: A -> B."""
    def rate_fn(C, T, params):
        k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def cstr_setup(simple_thermo, simple_rate_fn):
    """Set up a CSTR with standard parameters."""
    stoich = jnp.array([[-1.0], [+1.0]])  # A -> B
    params = CSTRParams(
        V=jnp.array(1.0),
        rate_fn=simple_rate_fn,
        stoich=stoich,
        rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
        species_order=["A", "B"],
    )
    cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
    inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
    return cstr, inlet


# =============================================================================
# State Layout Tests
# =============================================================================


class TestEOStateLayout:
    def test_pack_unpack_roundtrip(self):
        """Test that pack then unpack recovers original streams."""
        layout = EOStateLayout(
            species_order=["A", "B"],
            stream_names=["s1", "s2"],
        )
        streams = {
            "s1": make_stream({"A": 5.0, "B": 3.0}, T=350.0, P=101325.0),
            "s2": make_stream({"A": 2.0, "B": 8.0}, T=400.0, P=202650.0),
        }
        x = layout.pack(streams)
        recovered = layout.unpack(x)

        assert len(recovered) == 2
        for name in ["s1", "s2"]:
            for key in streams[name]:
                assert jnp.allclose(
                    recovered[name][key], streams[name][key], atol=1e-10
                ), f"Mismatch in {name}[{key}]"

    def test_total_vars(self):
        """Test total variable count."""
        layout = EOStateLayout(
            species_order=["A", "B", "C"],
            stream_names=["s1", "s2"],
        )
        # 3 species + T + P = 5 per stream, 2 streams = 10
        assert layout.total_vars == 10

    def test_system_is_square(self):
        """Test that n_vars equals expected for simple systems."""
        layout = EOStateLayout(
            species_order=["A", "B"],
            stream_names=["reactor_out"],
        )
        # 1 stream with 2 species + T + P = 4 variables
        assert layout.total_vars == 4

    def test_stream_slice(self):
        """Test stream slice indexing."""
        layout = EOStateLayout(
            species_order=["A", "B"],
            stream_names=["s1", "s2"],
        )
        s = layout.stream_slice("s2")
        assert s == slice(4, 8)


# =============================================================================
# Unit Residual Tests
# =============================================================================


class TestCSTRResiduals:
    def test_cstr_residuals_at_solution(self, cstr_setup):
        """Verify CSTR residuals are zero at known SM solution."""
        cstr, inlet = cstr_setup
        outlet, info = cstr(inlet, T_spec=350.0)

        residuals = cstr.eo_residuals(
            [inlet], [outlet], T_spec=350.0
        )
        assert jnp.max(jnp.abs(residuals)) < 1e-6, (
            f"Residuals not zero at solution: {residuals}"
        )

    def test_cstr_residuals_nonzero_away_from_solution(self, cstr_setup):
        """Verify CSTR residuals are nonzero away from solution."""
        cstr, inlet = cstr_setup
        # Use a wrong outlet
        wrong_outlet = make_stream({"A": 5.0, "B": 5.0}, T=350.0, P=101325.0)
        residuals = cstr.eo_residuals(
            [inlet], [wrong_outlet], T_spec=350.0
        )
        assert jnp.max(jnp.abs(residuals)) > 0.1


class TestFlashResiduals:
    def test_flash_residuals_at_solution(self, simple_thermo):
        """Verify Flash residuals are zero at known SM solution."""
        params = FlashParams(species_order=["A", "B"])
        flash = Flash(params, simple_thermo)

        inlet = make_stream({"A": 5.0, "B": 5.0}, T=350.0, P=101325.0)
        liquid, vapor, info = flash(inlet)

        residuals = flash.eo_residuals([inlet], [liquid, vapor])
        assert jnp.max(jnp.abs(residuals)) < 1e-4, (
            f"Flash residuals not zero at solution: max={jnp.max(jnp.abs(residuals))}"
        )


class TestMixerResiduals:
    def test_mixer_residuals_at_solution(self):
        """Verify Mixer residuals are zero at known solution."""
        mixer = Mixer(species_order=["A", "B"])

        inlet1 = make_stream({"A": 5.0, "B": 3.0}, T=300.0, P=101325.0)
        inlet2 = make_stream({"A": 2.0, "B": 4.0}, T=350.0, P=101325.0)
        outlet, _info = mixer(inlet1, inlet2)

        residuals = mixer.eo_residuals([inlet1, inlet2], [outlet])
        assert jnp.max(jnp.abs(residuals)) < 1e-6


class TestSplitterResiduals:
    def test_splitter_residuals_at_solution(self):
        """Verify Splitter residuals are zero at known solution."""
        splitter = Splitter(species_order=["A", "B"])

        inlet = make_stream({"A": 10.0, "B": 5.0}, T=300.0, P=101325.0)
        out1, out2, _info = splitter(inlet, split_frac=0.6)

        residuals = splitter.eo_residuals(
            [inlet], [out1, out2], split_frac=0.6
        )
        assert jnp.max(jnp.abs(residuals)) < 1e-6


class TestHeaterResiduals:
    def test_heater_residuals_at_solution(self):
        """Verify Heater residuals are zero at known solution."""
        params = HeaterParams(T_out=400.0)
        heater = Heater(params)

        inlet = make_stream({"A": 5.0, "B": 5.0}, T=300.0, P=101325.0)
        outlet, info = heater(inlet)

        residuals = heater.eo_residuals([inlet], [outlet])
        assert jnp.max(jnp.abs(residuals)) < 1e-6


class TestCoolerResiduals:
    def test_cooler_residuals_at_solution(self):
        """Verify Cooler residuals are zero at known solution."""
        params = CoolerParams(T_out=280.0)
        cooler = Cooler(params)

        inlet = make_stream({"A": 5.0, "B": 5.0}, T=350.0, P=101325.0)
        outlet, info = cooler(inlet)

        residuals = cooler.eo_residuals([inlet], [outlet])
        assert jnp.max(jnp.abs(residuals)) < 1e-6


# =============================================================================
# EO Solver Tests
# =============================================================================


class TestEOSolverSimple:
    def test_sequential_no_recycle(self, simple_thermo, simple_rate_fn):
        """EO matches SM for simple chain without recycles."""
        stoich = jnp.array([[-1.0], [+1.0]])
        cstr_params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        cstr = CSTR(cstr_params, thermo=simple_thermo, mode="isothermal")

        # Build flowsheet
        fs = Flowsheet(species_order=["A", "B"])
        feed = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)
        fs.add_unit(Unit("reactor", cstr, ["feed"], ["reactor_out"],
                         params={"T_spec": 350.0}))

        # SM solution
        sm_streams = fs.solve()

        # EO solution
        eo_streams = fs.solve_eo(use_sm_init=False)

        # Compare
        for key in sm_streams["reactor_out"]:
            assert jnp.allclose(
                sm_streams["reactor_out"][key],
                eo_streams["reactor_out"][key],
                atol=1e-4,
            ), f"Mismatch in reactor_out[{key}]: SM={sm_streams['reactor_out'][key]}, EO={eo_streams['reactor_out'][key]}"

    def test_heater_chain(self):
        """EO solves a simple heater chain."""
        fs = Flowsheet(species_order=["A", "B"])
        feed = make_stream({"A": 5.0, "B": 5.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)

        heater = Heater(HeaterParams(T_out=400.0))
        fs.add_unit(Unit("heater", heater, ["feed"], ["hot_out"]))

        sm_streams = fs.solve()
        eo_streams = fs.solve_eo(use_sm_init=False)

        assert jnp.allclose(
            sm_streams["hot_out"]["T"],
            eo_streams["hot_out"]["T"],
            atol=1e-4,
        )


class TestEOFallbackForUnitsWithoutEOResiduals:
    """#206: the residual builder's fallback branch, for any unit operation that
    does not define `eo_residuals`.

    Nothing in the suite exercised this branch before, which is how a plain
    NameError survived in it: the branch read a bare `feed_names`, but that name
    is a local of `EOSolver.__init__`, neither a closure cell nor a module
    global, so it raised as soon as the branch was reached.
    """

    @staticmethod
    def _scaler_flowsheet():
        """A flowsheet whose only unit deliberately has no `eo_residuals`."""

        class Scaler:
            """Multiplies every species flow by `factor`. No eo_residuals."""

            def __call__(self, inlet, factor=1.0):
                flows = get_flows(inlet)
                return make_stream(
                    {k: v * factor for k, v in flows.items()},
                    T=inlet["T"],
                    P=inlet["P"],
                )

        assert not hasattr(Scaler, "eo_residuals"), (
            "this test is only meaningful for an operation without eo_residuals"
        )

        fs = Flowsheet(species_order=["A", "B"])
        fs.add_feed("feed", make_stream({"A": 2.0, "B": 3.0}, T=310.0, P=101325.0))
        fs.add_unit(Unit("scaler", Scaler(), ["feed"], ["scaled"], params={"factor": 2.0}))
        return fs

    def test_fallback_branch_does_not_raise(self):
        """Regression: this raised `NameError: name 'feed_names' is not defined`."""
        fs = self._scaler_flowsheet()
        streams = fs.solve_eo(use_sm_init=False)
        assert "scaled" in streams

    def test_fallback_branch_matches_the_sequential_solve(self):
        """The branch must also be correct, not merely non-raising."""
        fs = self._scaler_flowsheet()
        sm = get_flows(fs.solve()["scaled"])
        eo = get_flows(fs.solve_eo(use_sm_init=False)["scaled"])
        for species in ("A", "B"):
            assert jnp.allclose(eo[species], sm[species], atol=1e-8)
        # And the answer is the one hand-arithmetic gives: 2x the feed.
        assert jnp.allclose(eo["A"], 4.0, atol=1e-8)
        assert jnp.allclose(eo["B"], 6.0, atol=1e-8)


class TestEOSolverRecycle:
    def test_cstr_with_recycle(self, simple_thermo, simple_rate_fn):
        """EO matches SM for CSTR with simple recycle via splitter."""
        stoich = jnp.array([[-1.0], [+1.0]])
        cstr_params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        cstr = CSTR(cstr_params, thermo=simple_thermo, mode="isothermal")
        mixer = Mixer(species_order=["A", "B"])
        splitter = Splitter(species_order=["A", "B"])

        fs = Flowsheet(species_order=["A", "B"])
        feed = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)

        fs.add_unit(Unit("mixer", mixer, ["feed", "recycle"], ["mixed"]))
        fs.add_unit(Unit("reactor", cstr, ["mixed"], ["reactor_out"],
                         params={"T_spec": 350.0}))
        fs.add_unit(Unit("splitter", splitter, ["reactor_out"],
                         ["product", "recycle"],
                         params={"split_frac": 0.7}))
        fs.add_recycle("recycle", "recycle")

        # SM solution
        sm_streams = fs.solve(tol=1e-8, max_iter=200)

        # EO solution (use SM init for reliability)
        eo_streams = fs.solve_eo(use_sm_init=True, tol=1e-8)

        # Compare product stream
        for key in sm_streams["product"]:
            assert jnp.allclose(
                sm_streams["product"][key],
                eo_streams["product"][key],
                atol=1e-3,
            ), f"Mismatch in product[{key}]"


# =============================================================================
# Differentiability Tests
# =============================================================================


class TestEODifferentiability:
    @pytest.mark.release
    def test_eo_grad_wrt_params(self, simple_thermo, simple_rate_fn):
        """Test that grad works through EO solve."""
        stoich = jnp.array([[-1.0], [+1.0]])

        def objective(V):
            cstr_params = CSTRParams(
                V=V,
                rate_fn=simple_rate_fn,
                stoich=stoich,
                rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
                species_order=["A", "B"],
            )
            cstr = CSTR(cstr_params, thermo=simple_thermo, mode="isothermal")

            fs = Flowsheet(species_order=["A", "B"])
            feed = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
            fs.add_feed("feed", feed)
            fs.add_unit(Unit("reactor", cstr, ["feed"], ["reactor_out"],
                             params={"T_spec": 350.0}))

            streams = fs.solve_eo(use_sm_init=False)
            # Return product B flow
            return streams["reactor_out"]["F_B"]

        V = jnp.array(1.0)
        grad_val = jax.grad(objective)(V)

        # Gradient should be positive (more volume -> more conversion -> more B)
        assert jnp.isfinite(grad_val), "Gradient is not finite"
        assert grad_val > 0, f"Expected positive gradient, got {grad_val}"


# =============================================================================
# EOSolveResult Tests
# =============================================================================


class TestEOSolveResult:
    def test_result_attributes(self, simple_thermo, simple_rate_fn):
        """Test that EOSolveResult has expected attributes."""
        stoich = jnp.array([[-1.0], [+1.0]])
        cstr_params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        cstr = CSTR(cstr_params, thermo=simple_thermo, mode="isothermal")

        fs = Flowsheet(species_order=["A", "B"])
        feed = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)
        fs.add_unit(Unit("reactor", cstr, ["feed"], ["reactor_out"],
                         params={"T_spec": 350.0}))

        solver = EOSolver(fs)
        result = solver.solve(use_sm_init=False)

        assert isinstance(result, EOSolveResult)
        assert isinstance(result.streams, dict)
        assert isinstance(result.converged, bool)
        assert isinstance(result.residual_norm, float)
        assert result.wall_time > 0
        assert "feed" in result.streams
        assert "reactor_out" in result.streams


class TestEOFromSMInit:
    def test_eo_from_sm_init_fast(self, simple_thermo, simple_rate_fn):
        """EO converges quickly when initialized from SM solution."""
        stoich = jnp.array([[-1.0], [+1.0]])
        cstr_params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        cstr = CSTR(cstr_params, thermo=simple_thermo, mode="isothermal")

        fs = Flowsheet(species_order=["A", "B"])
        feed = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)
        fs.add_unit(Unit("reactor", cstr, ["feed"], ["reactor_out"],
                         params={"T_spec": 350.0}))

        solver = EOSolver(fs)
        result = solver.solve(use_sm_init=True, tol=1e-8)

        assert result.converged
        assert result.residual_norm < 1e-8


# =============================================================================
# Cubic-EOS units: EOSFlash, EnthalpyCounterCurrentHX, non-isothermal CSTR (#389)
# =============================================================================

from functools import lru_cache

from difflow.eos import CriticalProperties, PengRobinson
from difflow.thermo import CubicThermo
from difflow.units.flash import EOSFlash, EOSFlashParams
from difflow.units.heat_exchanger import EnthalpyCounterCurrentHX, EnthalpyHXParams

C4_SPECIES = ["propane", "butane", "isobutane"]
C4_P = 8e5


@lru_cache(maxsize=None)
def _c4_eos_and_thermo():
    """PR EOS and CubicThermo for propane / n-butane / isobutane.

    Memoized: the units' JIT caches key on the thermo identity, so sharing
    one object compiles each solve once across the tests here.
    """
    species = {
        "propane": SpeciesData(name="propane", MW=44.10, Cp_coeffs=(73.0, 0.0, 0.0, 0.0),
                               Hvap_coeffs=(18000.0, 0.38, 369.8),
                               antoine_coeffs=(13.72, 1872.5, -25.16)),
        "butane": SpeciesData(name="butane", MW=58.12, Cp_coeffs=(98.0, 0.0, 0.0, 0.0),
                              Hvap_coeffs=(22000.0, 0.38, 425.1),
                              antoine_coeffs=(13.98, 2292.4, -27.86)),
        "isobutane": SpeciesData(name="isobutane", MW=58.12, Cp_coeffs=(96.0, 0.0, 0.0, 0.0),
                                 Hvap_coeffs=(21000.0, 0.38, 407.8),
                                 antoine_coeffs=(13.82, 2181.8, -24.28)),
    }
    crit = {
        "propane": CriticalProperties(name="propane", Tc=369.8, Pc=4.25e6, omega=0.152, MW=44.10),
        "butane": CriticalProperties(name="butane", Tc=425.1, Pc=3.80e6, omega=0.200, MW=58.12),
        "isobutane": CriticalProperties(name="isobutane", Tc=407.8, Pc=3.64e6, omega=0.184, MW=58.12),
    }
    eos = PengRobinson(crit)
    return eos, CubicThermo(IdealThermo(species), eos)


def _isomerization_cstr():
    """Adiabatic vapor-phase n-butane -> isobutane CSTR on the PR EOS."""
    eos, thermo = _c4_eos_and_thermo()

    def rate_fn(C, T, p):
        return jnp.array([p["k0"] * jnp.exp(-p["Ea"] / (8.314 * T)) * C["butane"]])

    params = CSTRParams(
        V=0.5, rate_fn=rate_fn, stoich=jnp.array([[0.0], [-1.0], [1.0]]),
        rate_params={"k0": 2e3, "Ea": 40000.0}, species_order=C4_SPECIES,
        dH_rxn=jnp.array([-8000.0]), eos=eos, reaction_phase="vapor",
    )
    return CSTR(params, thermo=thermo, mode="adiabatic")


def _c4_stream(T, flows=(0.5, 1.0, 0.1)):
    return make_stream(dict(zip(C4_SPECIES, flows)), T=T, P=C4_P)


class TestEOSFlashResiduals:
    """EOSFlash.eo_residuals vanishes at the sequential flash, in every regime."""

    @pytest.mark.parametrize("T", [325.0, 300.0, 400.0])  # two-phase, liquid, vapor
    def test_zero_at_sm_solution_with_regular_jacobian(self, T):
        eos, _ = _c4_eos_and_thermo()
        flash = EOSFlash(EOSFlashParams(species_order=C4_SPECIES), eos)
        feed = _c4_stream(T)
        liq, vap, _ = flash(feed)

        layout = EOStateLayout(C4_SPECIES, ["liq", "vap"])

        def r(x):
            s = layout.unpack(x)
            return flash.eo_residuals([feed], [s["liq"], s["vap"]])

        x = layout.pack({"liq": liq, "vap": vap})
        assert r(x).shape == x.shape
        assert float(jnp.max(jnp.abs(r(x)))) < 1e-6
        J = jax.jacfwd(r)(x)
        assert bool(jnp.all(jnp.isfinite(J)))
        # Regular in every regime: the single-phase rows pin the empty
        # phase instead of leaving its composition undetermined.
        assert float(jnp.linalg.cond(J)) < 1e12


class TestCSTRNonIsothermalResiduals:
    """The adiabatic CSTR's EO energy row is its energy balance (#389).

    It used to be ``T_out - T_out``: identically zero, so an adiabatic or
    specified-duty reactor had no energy equation in EO form at all.
    """

    def test_energy_row_vanishes_at_sm_solution_and_depends_on_T(self):
        cstr = _isomerization_cstr()
        feed = _c4_stream(460.0)
        out, _ = cstr(feed)

        layout = EOStateLayout(C4_SPECIES, ["out"])

        def r(x):
            return cstr.eo_residuals([feed], [layout.unpack(x)["out"]])

        x = layout.pack({"out": out})
        res = r(x)
        i_T = len(C4_SPECIES)
        assert float(jnp.max(jnp.abs(res[:i_T]))) < 1e-8
        # The energy row is in K. The sequential solve closes T with a fixed
        # point at rtol=1e-6 (~5e-4 K at 511 K), so that is all its answer
        # can satisfy; the EO solve itself drives this row to ~1e-13.
        assert abs(float(res[i_T])) < 1e-2
        J = jax.jacfwd(r)(x)
        assert float(J[i_T, i_T]) != 0.0
        assert float(jnp.linalg.cond(J)) < 1e12


@pytest.mark.slow
class TestEnthalpyHXResiduals:
    def test_zero_at_sm_solution(self):
        _, thermo = _c4_eos_and_thermo()
        hx = EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=150.0), thermo)
        hot, cold = _c4_stream(510.0), _c4_stream(340.0)
        hot_out, cold_out, _ = hx(hot, cold)

        layout = EOStateLayout(C4_SPECIES, ["h", "c"])

        def r(x):
            s = layout.unpack(x)
            return hx.eo_residuals([hot, cold], [s["h"], s["c"]])

        x = layout.pack({"h": hot_out, "c": cold_out})
        # The sequential unit converges Q to atol=1e-3 W, so its answer
        # satisfies the K-scaled energy rows to about that over C.
        assert float(jnp.max(jnp.abs(r(x)))) < 1e-3
        assert float(jnp.linalg.cond(jax.jacfwd(r)(x))) < 1e12


@pytest.mark.slow
class TestCubicEOSFlowsheetIsOpenEquation:
    """FEHE + heater + adiabatic PR CSTR + cooler + EOSFlash solves EO (#389)."""

    def _flowsheet(self):
        eos, thermo = _c4_eos_and_thermo()
        fs = Flowsheet(species_order=C4_SPECIES)
        fs.add_feed("feed", _c4_stream(340.0))
        fs.add_unit(Unit("fehe", EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=150.0), thermo),
                         ["rx_out", "feed"], ["hot_out", "preheated"]))
        fs.add_unit(Unit("heater", Heater(HeaterParams(T_out=460.0), thermo),
                         ["preheated"], ["rx_in"]))
        fs.add_unit(Unit("reactor", _isomerization_cstr(), ["rx_in"], ["rx_out"]))
        fs.add_unit(Unit("cooler", Cooler(CoolerParams(T_out=320.0), thermo),
                         ["hot_out"], ["cooled"]))
        fs.add_unit(Unit("flash", EOSFlash(EOSFlashParams(species_order=C4_SPECIES), eos),
                         ["cooled"], ["liq", "vap"]))
        fs.add_recycle("rx_out", "rx_out")
        return fs

    def test_as_residual_accepts_and_solve_eo_matches_solve(self, monkeypatch):
        from difflow import eo_solver
        from difflow.solvers import as_residual

        fs = self._flowsheet()
        view = as_residual(fs)  # require_eo_residuals must not refuse it
        assert view.n_unknowns == len(EOSolver(fs).layout.stream_names) * (len(C4_SPECIES) + 2)

        sm = fs.solve(tol=1e-9, max_iter=200)

        def no_fallback(*a, **k):
            raise AssertionError("EO residual took the forward-run fallback")

        monkeypatch.setattr(eo_solver, "_parse_unit_result", no_fallback)
        result = EOSolver(fs).solve(tol=1e-8)
        assert result.converged
        assert result.residual_norm < 1e-8

        # The flash split is two-phase here, so this also checks the
        # fugacity rows, not just the single-phase ones.
        assert float(result.streams["liq"]["F_propane"]) > 0.01
        assert float(result.streams["vap"]["F_propane"]) > 0.01
        for name in ["preheated", "rx_out", "hot_out", "liq", "vap"]:
            for key, val in sm[name].items():
                assert float(result.streams[name][key]) == pytest.approx(
                    float(val), rel=1e-4, abs=1e-4
                ), f"{name}[{key}]"
