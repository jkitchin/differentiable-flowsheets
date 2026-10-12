"""Tests for :class:`difflow.ActivityThermo` (modified-Raoult K-values).

Oracles: the modified-Raoult K-value written out by hand from the NRTL
equations (no library activity code), the Raoult fallback, and the
ethanol-water column closing its balances in both the CMO and MESH paths.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import (ActivityThermo, IdealThermo, NRTLParams, SpeciesData,
                     make_stream)
from difflow.units.distillation import (DistillationColumn,
                                        DistillationColumnParams)

jax.config.update("jax_enable_x64", True)

A = np.array([[0.0, -0.8009], [3.4578, 0.0]])
B = np.array([[0.0, 246.18], [-586.0809, 0.0]])


def _sd(name, antoine, Tc, MW):
    return SpeciesData(name=name, MW=MW, Cp_coeffs=(80.0, 0, 0, 0),
                       Hvap_coeffs=(4.0e4, 0.38, Tc), antoine_coeffs=antoine)


@pytest.fixture(scope="module")
def ideal():
    return IdealThermo({
        "ethanol": _sd("ethanol", (10.24677, 1598.673, -46.424), 514.0, 46.07),
        "water": _sd("water", (9.6543, 1435.264, -64.848), 647.0, 18.015),
    })


@pytest.fixture(scope="module")
def thermo(ideal):
    nrtl = NRTLParams(("ethanol", "water"), a=jnp.array(A), b=jnp.array(B),
                      alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]))
    return ActivityThermo(ideal, nrtl)


def _nrtl_gamma_by_hand(x1, T):
    tau12 = A[0, 1] + B[0, 1] / T
    tau21 = A[1, 0] + B[1, 0] / T
    G12, G21 = np.exp(-0.3 * tau12), np.exp(-0.3 * tau21)
    x2 = 1.0 - x1
    l1 = x2**2 * (tau21 * (G21 / (x1 + x2 * G21))**2 + tau12 * G12 / (x2 + x1 * G12)**2)
    l2 = x1**2 * (tau12 * (G12 / (x2 + x1 * G12))**2 + tau21 * G21 / (x1 + x2 * G21)**2)
    return np.exp(l1), np.exp(l2)


class TestKValues:
    def test_modified_raoult_by_hand(self, thermo, ideal):
        T, P, x1 = 355.0, 101325.0, 0.2
        g1, g2 = _nrtl_gamma_by_hand(x1, T)
        K0 = np.asarray(ideal.K_values_array(T, P))
        K = np.asarray(thermo.K_values_array(T, P, jnp.array([x1, 1 - x1])))
        assert K == pytest.approx([g1 * K0[0], g2 * K0[1]], rel=1e-10)

    def test_no_composition_is_raoult(self, thermo, ideal):
        K = thermo.K_values_array(360.0, 101325.0)
        assert np.asarray(K) == pytest.approx(np.asarray(ideal.K_values_array(360.0, 101325.0)))

    def test_dict_api_and_delegation(self, thermo, ideal):
        Kd = thermo.K_values(355.0, 101325.0, {"ethanol": 0.3, "water": 0.7})
        Ka = thermo.K_values_array(355.0, 101325.0, jnp.array([0.3, 0.7]))
        assert float(Kd["water"]) == pytest.approx(float(Ka[1]))
        assert thermo.species_order == ideal.species_order
        assert thermo.K_depends_on_composition
        f = {"ethanol": 1.0, "water": 2.0}
        assert float(thermo.stream_enthalpy(f, 350.0, "liquid", 101325.0)) == pytest.approx(
            float(ideal.stream_enthalpy(f, 350.0, "liquid", 101325.0)))

    def test_grads(self, thermo):
        def f(x1, T):
            return jnp.sum(thermo.K_values_array(T, 101325.0, jnp.array([x1, 1 - x1])) ** 2)
        check_grads(f, (0.3, 355.0), order=1, modes=["rev", "fwd"])

    @pytest.mark.slow
    def test_azeotrope_is_where_K_is_one(self, thermo):
        # at the NRTL azeotrope (x ~ 0.89, about 351.3 K) both K-values are ~1
        from difflow.phase_diagrams import find_azeotrope
        r = find_azeotrope(thermo.ideal, ["ethanol", "water"], P=101325.0,
                           activity_model=thermo.activity_model)
        x = float(r["x"])
        K = np.asarray(thermo.K_values_array(r["T"], 101325.0, jnp.array([x, 1 - x])))
        assert K == pytest.approx([1.0, 1.0], abs=2e-3)


class TestColumn:
    @pytest.mark.slow
    @pytest.mark.parametrize("use_mesh", [False, True])
    def test_ethanol_water_column_closes(self, thermo, use_mesh):
        p = DistillationColumnParams(species_order=["ethanol", "water"], n_stages=12,
                                     feed_stage=8, P=101325.0, q=1.0)
        feed = make_stream({"ethanol": 0.1, "water": 0.9}, T=360.0, P=101325.0)
        D, Bt, info = DistillationColumn(p, thermo)(feed, R=2.0, D_spec=0.1 * 0.10 / 0.75,
                                                     use_mesh=use_mesh)
        assert float(info["balance_error_rel"]) < 5e-3
        xD = float(D["F_ethanol"] / (D["F_ethanol"] + D["F_water"]))
        # nonideal: far richer than Raoult would allow at this reflux, and
        # below the 0.89 azeotrope
        assert 0.6 < xD < 0.89
        ideal_col = DistillationColumn(p, thermo.ideal)(feed, R=2.0, D_spec=0.1 * 0.10 / 0.75,
                                                       use_mesh=use_mesh)[0]
        xD_ideal = float(ideal_col["F_ethanol"] / (ideal_col["F_ethanol"] + ideal_col["F_water"]))
        assert xD > xD_ideal
