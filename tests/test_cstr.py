"""Tests for CSTR unit operation."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    IdealThermo,
    SpeciesData,
    make_stream,
    get_flows,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


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
    """First-order reaction rate function."""
    def rate_fn(C, T, params):
        k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
        return jnp.array([k * C["A"]])
    return rate_fn


class TestCSTR:
    def test_cstr_creation(self, simple_thermo, simple_rate_fn):
        """Test CSTR can be created."""
        stoich = jnp.array([[-1.0], [+1.0]])  # A → B
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
        assert cstr is not None

    def test_cstr_isothermal(self, simple_thermo, simple_rate_fn):
        """Test isothermal CSTR operation."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=350.0)

        # Check mass balance (A + B should be conserved)
        flows_in = get_flows(inlet)
        flows_out = get_flows(outlet)
        total_in = float(flows_in["A"]) + float(flows_in["B"])
        total_out = float(flows_out["A"]) + float(flows_out["B"])

        assert total_out == pytest.approx(total_in, rel=1e-6)

        # Check some B was produced
        assert float(flows_out["B"]) > 0

        # Check conversion is reported
        assert "conversion" in info
        assert info["conversion"]["A"] > 0

    @pytest.mark.release
    def test_cstr_differentiability(self, simple_thermo, simple_rate_fn):
        """Test that CSTR is differentiable w.r.t. volume."""
        stoich = jnp.array([[-1.0], [+1.0]])

        def outlet_B(V):
            params = CSTRParams(
                V=V,
                rate_fn=simple_rate_fn,
                stoich=stoich,
                rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
                species_order=["A", "B"],
            )
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
            outlet, _ = cstr(inlet, T_spec=350.0)
            return outlet["F_B"]

        # Compute gradient
        grad_V = jax.grad(outlet_B)(jnp.array(1.0))

        # Gradient should be positive (more volume = more conversion = more B)
        assert float(grad_V) > 0


class TestCSTRParamsUpdate:
    """Test CSTRParams update() and __getitem__ methods."""

    def test_update_returns_new_instance(self, simple_rate_fn):
        """Test that update() returns a new instance."""
        stoich = jnp.array([[-1.0], [+1.0]])
        original = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        updated = original.update(V=jnp.array(2.0))

        # Should be different instances
        assert updated is not original
        # Updated value should change
        assert float(updated.V) == 2.0
        # Original should be unchanged
        assert float(original.V) == 1.0

    def test_update_preserves_other_fields(self, simple_rate_fn):
        """Test that update() preserves fields not being updated."""
        stoich = jnp.array([[-1.0], [+1.0]])
        original = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        updated = original.update(V=jnp.array(5.0))

        # Other fields should be preserved
        assert updated.rate_fn is original.rate_fn
        assert jnp.allclose(updated.stoich, original.stoich)
        assert updated.species_order == original.species_order

    def test_update_multiple_fields(self, simple_rate_fn):
        """Test that update() can update multiple fields at once."""
        stoich = jnp.array([[-1.0], [+1.0]])
        original = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        new_stoich = jnp.array([[-2.0], [+2.0]])
        updated = original.update(V=jnp.array(3.0), stoich=new_stoich)

        assert float(updated.V) == 3.0
        assert jnp.allclose(updated.stoich, new_stoich)

    def test_getitem_access(self, simple_rate_fn):
        """Test that __getitem__ provides dict-like read access."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        # Dict-like access should work
        assert float(params["V"]) == 1.0
        assert params["species_order"] == ["A", "B"]
        assert jnp.allclose(params["stoich"], stoich)

    def test_getitem_invalid_key(self, simple_rate_fn):
        """Test that __getitem__ raises KeyError for invalid keys."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        with pytest.raises(KeyError):
            _ = params["nonexistent_field"]

    def test_update_with_jax_grad(self, simple_thermo, simple_rate_fn):
        """Test that update() works with JAX automatic differentiation."""
        stoich = jnp.array([[-1.0], [+1.0]])
        base_params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        def outlet_B_with_update(V):
            # Use update() to create new params - should be JAX compatible
            params = base_params.update(V=V)
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
            outlet, _ = cstr(inlet, T_spec=350.0)
            return outlet["F_B"]

        # Compute gradient - should work without errors
        grad_V = jax.grad(outlet_B_with_update)(jnp.array(1.0))

        # Gradient should be positive and finite
        assert jnp.isfinite(grad_V)
        assert float(grad_V) > 0

    def test_keys_returns_field_names(self, simple_rate_fn):
        """Test that keys() returns all field names."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        keys = list(params.keys())

        # Should contain all dataclass fields
        assert "V" in keys
        assert "rate_fn" in keys
        assert "stoich" in keys
        assert "rate_params" in keys
        assert "species_order" in keys

    def test_contains_existing_field(self, simple_rate_fn):
        """Test that 'field in params' works for existing fields."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        assert "V" in params
        assert "rate_fn" in params
        assert "stoich" in params

    def test_contains_nonexistent_field(self, simple_rate_fn):
        """Test that 'field in params' returns False for nonexistent fields."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        assert "nonexistent" not in params
        assert "volume" not in params  # V, not volume

    def test_asdict_returns_dict(self, simple_rate_fn):
        """Test that asdict() returns a dictionary with all fields."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        d = params.asdict()

        assert isinstance(d, dict)
        assert float(d["V"]) == 1.0
        assert d["species_order"] == ["A", "B"]
        assert jnp.allclose(d["stoich"], stoich)

    def test_values_returns_field_values(self, simple_rate_fn):
        """Test that values() returns all field values."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        values = list(params.values())

        # Should have same number of values as fields
        assert len(values) == len(list(params.keys()))
        # First value should be V
        assert float(values[0]) == 1.0

    def test_items_returns_key_value_pairs(self, simple_rate_fn):
        """Test that items() returns (key, value) pairs."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        items = list(params.items())

        # Should have same count as fields
        assert len(items) == len(list(params.keys()))
        # Each item should be a (key, value) tuple
        for key, value in items:
            assert isinstance(key, str)
        # Check specific values
        items_dict = dict(items)
        assert float(items_dict["V"]) == 1.0
        assert items_dict["species_order"] == ["A", "B"]

    def test_iter_over_keys(self, simple_rate_fn):
        """Test that iterating over params yields keys."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        # list(params) should give field names
        field_names = list(params)
        assert "V" in field_names
        assert "rate_fn" in field_names
        assert "stoich" in field_names

    def test_len_returns_field_count(self, simple_rate_fn):
        """Test that len() returns number of fields."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        # CSTRParams fields: 7 original + molar_density, eos, reaction_phase,
        # H_mix_fn, K_eq_fn, outlet_volumetric_basis
        assert len(params) == 13

    def test_dict_conversion_roundtrip(self, simple_rate_fn):
        """Test that dict(params) works like asdict()."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )

        # dict(params) should work due to __iter__ and __getitem__
        d = dict(params.items())
        assert float(d["V"]) == 1.0
        assert d["species_order"] == ["A", "B"]


class TestConcentrationBasis:
    """#227: the molar density that sets residence time must not be silently water.

    Concentration is ``C_i = F_i / Q_v`` with ``Q_v = F_total / rho``, so the
    density sets ``tau = V * rho / F_total`` and, through it, the conversion.
    These tests pin down where that density comes from and that a guessed one
    is announced.
    """

    species = ["n_butane", "isobutane"]

    @staticmethod
    def _rate_fn(C, T, params):
        return jnp.array([params["k"] * C["n_butane"]])

    @classmethod
    def _params(cls, **kwargs):
        return CSTRParams(
            V=12.0,
            rate_fn=cls._rate_fn,
            stoich=jnp.array([[-1.0], [+1.0]]),
            rate_params={"k": 1e-4},
            species_order=cls.species,
            **kwargs,
        )

    @staticmethod
    def _feed():
        # 350 K / 10 bar: a real C4 liquid (~8800 mol/m^3 by Peng-Robinson),
        # about 6x lighter than the 55500 mol/m^3 water fallback.
        return make_stream({"n_butane": 90.0, "isobutane": 10.0}, 350.0, 10e5)

    @classmethod
    def _eos(cls):
        from difflow.eos import PengRobinson
        from difflow.database import get_critical_props

        return PengRobinson({c: get_critical_props(c) for c in cls.species})

    @classmethod
    def _cubic_thermo(cls, eos):
        from difflow.thermo import CubicThermo
        from difflow.database import get_species_data

        return CubicThermo(IdealThermo({c: get_species_data(c) for c in cls.species}), eos)

    def test_default_density_warns(self):
        """With no eos and no molar_density the water fallback is announced."""
        from difflow.units.cstr import CSTRDensityWarning

        cstr = CSTR(self._params())
        with pytest.warns(CSTRDensityWarning, match="55500"):
            _, info = cstr(self._feed())
        assert float(info["molar_density"]) == pytest.approx(55500.0)

    def test_fallback_warns_once_per_reactor(self):
        """A reactor called in a loop should not warn on every call."""
        import warnings

        from difflow.units.cstr import CSTRDensityWarning

        cstr = CSTR(self._params())
        feed = self._feed()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cstr(feed)
            cstr(feed)
        assert sum(issubclass(w.category, CSTRDensityWarning) for w in caught) == 1

    def test_explicit_molar_density_is_silent(self):
        """An explicitly chosen density is not a guess, so it does not warn."""
        import warnings

        from difflow.units.cstr import CSTRDensityWarning

        cstr = CSTR(self._params(molar_density=7000.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error", CSTRDensityWarning)
            _, info = cstr(self._feed())
        assert float(info["molar_density"]) == pytest.approx(7000.0)

    def test_explicit_volumetric_flow_is_not_a_fallback(self):
        """A caller-supplied Q_v sets the basis; report the density it implies."""
        import warnings

        from difflow.units.cstr import CSTRDensityWarning

        cstr = CSTR(self._params())
        with warnings.catch_warnings():
            warnings.simplefilter("error", CSTRDensityWarning)
            _, info = cstr(self._feed(), volumetric_flow=0.01)
        # 100 mol/s through 0.01 m^3/s is 10000 mol/m^3, not 55500.
        assert float(info["molar_density"]) == pytest.approx(10000.0)

    def test_eos_sets_the_basis_and_is_silent(self):
        """With an EOS the density is the real one at reactor conditions."""
        import warnings

        from difflow.units.cstr import CSTRDensityWarning

        eos = self._eos()
        feed = self._feed()
        cstr = CSTR(self._params(eos=eos, reaction_phase="liquid"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", CSTRDensityWarning)
            _, info = cstr(feed)

        y = jnp.array([0.9, 0.1])
        expected = eos.density(feed["T"], feed["P"], y, phase="liquid")
        assert float(info["molar_density"]) == pytest.approx(float(expected))
        # A C4 liquid is nowhere near water's 55500 mol/m^3.
        assert 5000.0 < float(info["molar_density"]) < 20000.0

    def test_water_fallback_over_converts_relative_to_the_eos(self):
        """The 8x density error of #227 is an 8x residence-time error."""
        feed = self._feed()
        eos = self._eos()

        with pytest.warns(UserWarning):
            fallback_out, fallback_info = CSTR(self._params())(feed)
        eos_out, eos_info = CSTR(self._params(eos=eos, reaction_phase="liquid"))(feed)

        ratio = float(fallback_info["molar_density"]) / float(eos_info["molar_density"])
        assert ratio > 4.0
        # Higher density is a longer tau, so the water basis leaves less n-butane.
        assert float(get_flows(fallback_out)["n_butane"]) < float(
            get_flows(eos_out)["n_butane"]
        )

    def test_eos_without_reaction_phase_is_an_error(self):
        """A phase-dependent density has no defensible default phase."""
        with pytest.raises(ValueError, match="reaction_phase"):
            self._params(eos=self._eos())

    def test_unknown_reaction_phase_is_an_error(self):
        with pytest.raises(ValueError, match="reaction_phase"):
            self._params(eos=self._eos(), reaction_phase="supercritical")

    def test_cubic_thermo_eos_is_used_when_the_phase_is_named(self):
        """A CubicThermo's own EOS sets the basis rather than being ignored."""
        import warnings

        from difflow.units.cstr import CSTRDensityWarning

        eos = self._eos()
        feed = self._feed()
        cstr = CSTR(
            self._params(reaction_phase="liquid"),
            thermo=self._cubic_thermo(eos),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", CSTRDensityWarning)
            _, info = cstr(feed)

        y = jnp.array([0.9, 0.1])
        expected = eos.density(feed["T"], feed["P"], y, phase="liquid")
        assert float(info["molar_density"]) == pytest.approx(float(expected))

    def test_reaction_phase_with_nothing_to_apply_it_to_warns(self):
        """A phase that changes no number should say so."""
        from difflow.units.cstr import CSTRDensityWarning

        with pytest.warns(CSTRDensityWarning, match="reaction_phase"):
            CSTR(self._params(reaction_phase="liquid"))

    def test_eo_residuals_use_the_same_basis_as_the_solve(self):
        """The EO residual must vanish at the sequential solution."""
        eos = self._eos()
        feed = self._feed()
        cstr = CSTR(self._params(eos=eos, reaction_phase="liquid"))
        outlet, _ = cstr(feed)

        resid = cstr.eo_residuals([feed], [outlet])
        # Flows are ~100 mol/s, so 1e-6 is a converged material balance.
        assert float(jnp.max(jnp.abs(resid[:2]))) < 1e-6
