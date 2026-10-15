"""Derivatives of the SMR, SRU and amine loop against central differences (#471-#473).

Run from the repository root, by hand::

    PYTHONPATH=src:. python -m pytest tests/reference/gradient_checks.py -v

Not part of the suite (the file name keeps pytest from collecting it):
each check compiles a derivative through a nested solve, minutes apiece,
past ``DIFFLOW_TEST_BUDGET``; the suite keeps a finite-gradient test per
unit. ``src/difflow_refinery/plant.py`` (``AD_MODES``) cites these.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.hydrogen import smr
from difflow_refinery.sulfur import tgtu as tg
from difflow_refinery.treating import solve_amine_unit
from tests.refinery.test_smr import SHORTCUT, natural_gas, tube_inlet
from tests.refinery.test_sulfur import TRAY, acid_gas, params
from tests.refinery.test_treating import LOOP, loop_gas

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("adjoint", ["reverse", "forward"])
def test_smr_tube(adjoint):
    """d(CH4 out, Q)/d(T_out, steam): implicit step through the duty solve."""
    tp = smr.ReformerTubeParams(adjoint=adjoint)

    def f(x):
        out, info = smr.reformer_tube(tube_inlet(water=x[1]), tp, T_out=x[0])
        return jnp.stack([out["F_methane"], info["Q"] / 1e5])

    x = jnp.array([1123.15, 1.2])
    J = (jax.jacfwd if adjoint == "forward" else jax.jacrev)(f)(x)
    h = jnp.array([0.05, 1e-4])
    for j in range(2):
        e = jnp.zeros(2).at[j].set(h[j])
        fd = (f(x + e) - f(x - e)) / (2 * h[j])
        np.testing.assert_allclose(np.asarray(J[:, j]), np.asarray(fd), rtol=2e-4, atol=1e-9)


def test_smr_plant_forward():
    """d(H2, CO2, NG)/d(S/C, T_out) through the whole plant, forward mode."""
    def f(x):
        p = smr.SMRParams(S_C=x[0], T_out=x[1], psa=SHORTCUT,
                          tube=smr.ReformerTubeParams(adjoint="forward"))
        r = smr.smr_plant(natural_gas(), p)
        return jnp.stack([r["H2"], r["CO2"], r["ng_lhv"] / 1e5])

    x = jnp.array([3.0, 1123.15])
    J = jax.jacfwd(f)(x)
    h = jnp.array([1e-3, 0.05])
    for j in range(2):
        e = jnp.zeros(2).at[j].set(h[j])
        fd = (f(x + e) - f(x - e)) / (2 * h[j])
        np.testing.assert_allclose(np.asarray(J[:, j]), np.asarray(fd), rtol=5e-4)


def test_smr_plant_reverse_matches_forward():
    """jax.grad through the whole plant (tube on the checkpointed adjoint)
    agrees with the forward-mode derivative."""
    def f(T, adjoint):
        p = smr.SMRParams(T_out=T, psa=SHORTCUT, tube=smr.ReformerTubeParams(adjoint=adjoint))
        return smr.smr_plant(natural_gas(), p)["H2"]

    g_rev = jax.grad(f)(1193.15, "reverse")
    g_fwd = jax.jacfwd(f)(1193.15, "forward")
    assert float(g_rev) == pytest.approx(float(g_fwd), rel=1e-6)


def test_sru():
    tp = tg.TGTUParams(tray=TRAY, L=30.0, n_stages=12, P_absorber=1.2e5)

    def emitted(h):
        return tg.solve_sru(dict(acid_gas(), F_hydrogen_sulfide=h), params(2), tp)["SO2_emitted"]

    vg = jax.jit(jax.value_and_grad(emitted))     # one compile serves the FD points too
    g = float(vg(90.0)[1])
    fd = (float(vg(90.001)[0]) - float(vg(89.999)[0])) / 0.002
    assert g == pytest.approx(fd, rel=1e-5)


def test_amine_loop():
    def f(x):
        r = solve_amine_unit(LOOP, loop_gas(), circulation=(x[0],), Q_reboiler=x[1] * 1e6)
        return jnp.log(r.sweet["fuel"]["F_hydrogen_sulfide"])

    x0 = jnp.array([120.0, 5.0])
    J = np.asarray(jax.jacfwd(f)(x0))
    for i, h in enumerate((1.0, 0.05)):
        e = jnp.zeros(2).at[i].set(h)
        fd = (float(f(x0 + e)) - float(f(x0 - e))) / (2 * h)
        assert J[i] == pytest.approx(fd, rel=5e-3, abs=1e-6)
