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

        # CSTRParams has 10 fields (7 original + molar_density, H_mix_fn, K_eq_fn)
        assert len(params) == 10

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
