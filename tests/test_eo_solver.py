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
