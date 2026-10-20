"""Sulfur species, the wide-range Cp set and liquid sulfur (#467).

Smoke, API and identity tests only (the test policy in CLAUDE.md). The
values come from the NIST-JANAF Thermochemical Tables (4th ed., 1998); the
comparisons with those tables are not per-commit tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import thermochemistry as tc

jax.config.update("jax_enable_x64", True)

SULFUR = ["S2", "S6", "S8", "H2S", "SO2", "COS", "CS2"]


class TestRows:
    def test_aliases(self):
        for n in range(2, 9):
            assert tc.resolve(f"S{n}") == f"sulfur_s{n}"
        assert tc.resolve("COS") == "carbonyl_sulfide"
        assert tc.resolve("CS2") == "carbon_disulfide"
        assert tc.species("S8").MW == pytest.approx(8 * 32.06)

    @pytest.mark.parametrize("basis", ["table", "wide"])
    def test_smoke(self, basis):
        g = tc.IdealGasSet(SULFUR, cp_basis=basis)
        for T in (600.0, 1500.0):
            assert np.all(np.isfinite(np.asarray(g.cp(T))))
        for k in SULFUR:
            assert np.isfinite(float(tc.gibbs(k, 800.0)))
        p = np.asarray(tc.sulfur_vapor_pressures(700.0, cp_basis=basis))
        assert np.all(np.isfinite(p)) and np.all(p > 0)


class TestWideCp:
    def test_every_entry_is_a_table_species(self):
        for k in tc.WIDE_CP:
            assert k in tc.TABLE

    def test_basis(self):
        g = tc.IdealGasSet(["H2", "CH4"], cp_basis="wide")
        assert np.allclose(g.HF, tc.IdealGasSet(["H2", "CH4"]).HF)
        with pytest.raises(KeyError):
            tc.IdealGasSet(["benzene"], cp_basis="wide")
        with pytest.raises(ValueError):
            tc.IdealGasSet(["H2"], cp_basis="hot")


class TestLiquidSulfur:
    def test_nodes_are_the_table(self):
        H = np.asarray(tc.liquid_sulfur_enthalpy(tc._SL_T)) - tc._SL_HF
        S = np.asarray(tc.liquid_sulfur_entropy(tc._SL_T))
        assert np.allclose(H, tc._SL_H, atol=1e-6)
        assert np.allclose(S, tc._SL_S, atol=1e-9)

    def test_consistency_and_grads(self):
        T = jnp.linspace(300.0, 1490.0, 47)
        dG = jax.vmap(jax.grad(tc.liquid_sulfur_gibbs))(T)
        assert np.allclose(dG, -np.asarray(tc.liquid_sulfur_entropy(T)), atol=1e-8)
        assert np.isfinite(float(jax.grad(jax.grad(tc.liquid_sulfur_gibbs))(615.0)))
