"""Kent-Eisenberg H2S-CO2-amine equilibrium (``difflow_cc.equilibrium.acid_gas``, #471).

Code checks only (the test policy in CLAUDE.md): the speciation closes its
balances and solves its equations, the inverse inverts it, it converges over
the range of the held-out data, and its derivatives are finite. The
comparisons with literature constants and VLE data (fitted and held out),
and the refit, are in ``tests/reference/amine_ke_fit.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_cc.equilibrium import acid_gas as ag
from tests.cc import acid_gas_data as D

jax.config.update("jax_enable_x64", True)

T25 = 298.15
LOG10 = np.log(10.0)


def pK(c, T=T25):
    return -float(ag._lnk(c, T)) / LOG10


@pytest.mark.parametrize("amine,aC,aS", [
    ("MDEA", 0.1, 0.05), ("MDEA", 0.6, 0.0), ("MDEA", 0.0, 0.4),
    ("DEA", 0.3, 0.1), ("DEA", 0.05, 0.0), ("DEA", 0.0, 0.8),
])
def test_speciation_closes_its_balances(amine, aC, aS):
    m = ag.amine_molality(0.4, amine)
    out = ag.ke_speciate(amine, 330.0, m, aC, aS)
    assert bool(out["converged"])
    sp = out["species"]
    a, c, s, pos, neg = (float(v) for v in ag._sums(sp))
    m = float(m)
    np.testing.assert_allclose(a, m, rtol=1e-10)
    np.testing.assert_allclose(c, max(aC, ag.TRACE) * m, rtol=1e-10)
    np.testing.assert_allclose(s, max(aS, ag.TRACE) * m, rtol=1e-10)
    np.testing.assert_allclose(pos, neg, rtol=1e-10)
    # every reaction is at its equilibrium
    lnK = ag._constants(ag.ke_amine(amine), 330.0, max(aC, ag.TRACE) + max(aS, ag.TRACE),
                        np.log(m))
    np.testing.assert_allclose(np.log(sp["H+"] * sp["OH-"]), lnK["w"], atol=1e-10)
    np.testing.assert_allclose(np.log(sp["H+"] * sp["HS-"] / sp["H2S"]), lnK["d"], atol=1e-10)
    np.testing.assert_allclose(np.log(sp["H+"] * sp["Am"] / sp["AmH+"]), lnK["f"], atol=1e-10)


@pytest.mark.parametrize("amine", ["MDEA", "DEA"])
def test_trace_loading_is_henrys_law(amine):
    """At infinite dilution of one acid gas (the other sets the pH) its
    pressure is linear in its loading, down to loadings a stripped bottoms
    reaches (~1e-17, #475); TRACE only guards the log, far below that."""
    m = ag.amine_molality(0.45, amine)
    k = [float(ag.ke_speciate(amine, 395.0, m, a, 0.01)["p_CO2"]) / a for a in (1e-12, 1e-17, 1e-24)]
    np.testing.assert_allclose(k[1:], k[0], rtol=1e-8)
    assert ag.TRACE < 1e-24


@pytest.mark.parametrize("amine", ["MDEA", "DEA"])
def test_unloaded_amine_is_the_weak_base(amine):
    # closed form: [OH-] = sqrt(Kb m) with Kb = Kw / K_f, the thermodynamic
    # K_f (Posey 1996). The apparent correction at zero loading is
    # d0 + d1 ln(alpha0); it moves pH by half of it over ln 10, so it must
    # stay bounded (a bare ln(alpha) moved it by 3 units)
    am = ag.KE_AMINES[amine]
    m = float(ag.amine_molality(0.4, amine))
    out = ag.ke_speciate(amine, T25, m, 0.0, 0.0)
    assert bool(out["converged"]) and np.isfinite(float(out["pH"]))
    shift = -0.5 * (am.apparent[0] + am.apparent[1] * np.log(ag.ALPHA0)) / LOG10
    assert abs(shift) < 1.0


class TestChemistry:
    @pytest.mark.parametrize("amine", ["MDEA", "DEA"])
    def test_loading_inverts_the_pressures(self, amine):
        pc, ps = ag.ke_partial_pressures(amine, 323.15, 0.4, 0.25, 0.12)
        out = ag.ke_loading(amine, 323.15, 0.4, pc, ps)
        assert bool(out["converged"])
        np.testing.assert_allclose(float(out["alpha_CO2"]), 0.25, rtol=1e-8)
        np.testing.assert_allclose(float(out["alpha_H2S"]), 0.12, rtol=1e-8)

    def test_loading_with_one_gas_absent(self):
        out = ag.ke_loading("MDEA", 313.15, 0.5, 0.0, 1e3)
        assert float(out["alpha_CO2"]) < 1e-10
        assert 0.0 < float(out["alpha_H2S"]) < 1.0

    def test_unknown_amine(self):
        with pytest.raises(ValueError, match="no Kent-Eisenberg"):
            ag.ke_amine("MEA")


def _finite_grads(f, x):
    for J in (jax.jacfwd(f)(x), jax.jacrev(f)(x)):
        assert np.all(np.isfinite(np.asarray(J)))


class TestGradients:
    def test_pressures(self):
        def f(x):
            pc, ps = ag.ke_partial_pressures("DEA", x[0], 0.35, x[1], x[2])
            return jnp.log(pc) + 2.0 * jnp.log(ps)
        _finite_grads(f, jnp.array([340.0, 0.3, 0.1]))

    def test_loading(self):
        def f(x):
            out = ag.ke_loading("MDEA", x[0], 0.45, x[1], x[2])
            return out["alpha_CO2"] + 3.0 * out["alpha_H2S"]
        _finite_grads(f, jnp.array([320.0, 5e3, 2e3]))

    def test_pressures_at_regenerator_conditions(self):
        # 50 wt% MDEA near a stripper bottom: 393 K, lean loadings (in logs,
        # so the finite-difference step is relative to a small loading)
        def f(x):
            pc, ps = ag.ke_partial_pressures("MDEA", x[0], 0.489, jnp.exp(x[1]), jnp.exp(x[2]))
            return jnp.log(pc) + 2.0 * jnp.log(ps)
        _finite_grads(f, jnp.array([393.15, np.log(0.003), np.log(0.002)]))

    def test_jit(self):
        f = jax.jit(lambda T: ag.ke_partial_pressures("MDEA", T, 0.5, 0.2, 0.1)[1])
        np.testing.assert_allclose(
            float(f(330.0)), float(ag.ke_partial_pressures("MDEA", 330.0, 0.5, 0.2, 0.1)[1]),
            rtol=1e-12)


@pytest.mark.parametrize("gas", ["H2S", "CO2"])
def test_converges_over_the_jou_range(gas):
    """Speciation and its inverse converge at every held-out Jou (1982) point:
    25-120 C, 12-49 wt% MDEA, loadings 4e-4 to 3.2, up to 6.6 MPa."""
    rows = np.array(D.JOU_H2S_MDEA if gas == "H2S" else D.JOU_CO2_MDEA)
    T, w, a, p = (jnp.asarray(rows[:, i]) for i in range(4))

    def fwd(T, w, a):
        aC, aS = (a, 0.0) if gas == "CO2" else (0.0, a)
        return ag.ke_speciate("MDEA", T, ag.amine_molality(w, "MDEA"), aC, aS)["converged"]

    def inv(T, w, p):
        pc, ps = (1e3 * p, 0.0) if gas == "CO2" else (0.0, 1e3 * p)
        return ag.ke_loading("MDEA", T, w, pc, ps)["converged"]

    assert bool(jnp.all(jax.vmap(fwd)(T, w, a)))
    assert bool(jnp.all(jax.vmap(inv)(T, w, p)))
