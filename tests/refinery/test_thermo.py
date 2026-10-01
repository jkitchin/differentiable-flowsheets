"""Tests for the column thermodynamics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import Assay, ColumnThermo, characterize, water_vapor_pressure
from difflow_refinery.thermo import LIGHT_ENDS

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def thermo():
    assay = Assay([0, 10, 50, 90, 100], [280.0, 350.0, 550.0, 800.0, 950.0], sg=0.86,
                  light_ends={"propane": 0.5, "n_butane": 1.0})
    return ColumnThermo.from_characterization(characterize(assay))


class TestWater:
    @pytest.mark.parametrize("T, P", [(298.15, 3169.9), (373.124, 101325.0), (473.15, 1.5549e6)])
    def test_steam_tables(self, T, P):
        assert float(water_vapor_pressure(T)) == pytest.approx(P, rel=1e-3)

    def test_latent_heat_at_the_normal_boiling_point(self):
        T = 373.124
        dh = float(ColumnThermo.water_h_vapor(T) - ColumnThermo.water_h_liquid(T))
        assert dh == pytest.approx(40660.0, rel=1e-6)


class TestColumnThermo:
    def test_components_boil_at_their_boiling_points(self, thermo):
        # Lee-Kesler with the acentric factor that boils each one at Tb
        psat = jax.vmap(lambda T, i: thermo.psat(T)[i])(thermo.Tb, jnp.arange(thermo.n_components))
        pseudo = np.asarray(psat)[2:]
        np.testing.assert_allclose(pseudo, 101325.0, rtol=1e-6)
        # the light ends use their tabulated acentric factors: close, not exact
        np.testing.assert_allclose(np.asarray(psat)[:2], 101325.0, rtol=0.05)

    def test_latent_heat_at_tb(self, thermo):
        dh = jax.vmap(lambda T, i: thermo.dhvap(T)[i])(thermo.Tb, jnp.arange(thermo.n_components))
        expected = [LIGHT_ENDS["propane"][5], LIGHT_ENDS["n_butane"][5]]
        np.testing.assert_allclose(np.asarray(dh)[:2], expected, rtol=1e-10)

    def test_supercritical_light_end_is_smooth(self, thermo):
        """Propane goes critical at 97 C, mid-column: no kink, no NaN."""
        Tc = float(thermo.Tc[0])
        T = jnp.linspace(Tc - 20.0, Tc + 20.0, 41)
        dh = jax.vmap(lambda t: thermo.dhvap(t)[0])(T)
        slope = jax.vmap(jax.grad(lambda t: thermo.dhvap(t)[0]))(T)
        assert bool(jnp.all(jnp.isfinite(slope)))
        assert bool(jnp.all(dh > 0))
        assert bool(jnp.all(jnp.diff(dh) < 0))
        # the smooth floor leaves a tail above Tc: a few percent of dHvap(Tb)
        assert float(dh[-1]) < 0.08 * LIGHT_ENDS["propane"][5]
        assert float(jnp.max(jnp.abs(slope))) < 1e4

    def test_k_values_order_with_boiling_point(self, thermo):
        K = np.asarray(thermo.K(500.0, 1.5e5))
        order = np.argsort(np.asarray(thermo.Tb))
        assert np.all(np.diff(K[order]) < 0)

    def test_volume_and_mass(self, thermo):
        flows = jnp.zeros(thermo.n_components).at[0].set(1000.0)  # 1 kmol/s propane
        assert float(thermo.mass(flows)) == pytest.approx(44.097, rel=1e-4)
        # propane SG 0.507: about 87 m^3 per kmol
        assert float(thermo.std_volume(flows)) == pytest.approx(44.097 / (0.50736 * 999.016), rel=1e-3)

    def test_is_a_pytree(self, thermo):
        f = jax.jit(lambda th, T: th.h_liquid(T))
        np.testing.assert_allclose(f(thermo, 400.0), thermo.h_liquid(400.0), rtol=1e-12)

    def test_unknown_light_end(self):
        crude = characterize(Assay([0, 100], [300.0, 900.0], sg=0.85, light_ends={"water": 1.0}))
        with pytest.raises(ValueError, match="light ends"):
            ColumnThermo.from_characterization(crude)
