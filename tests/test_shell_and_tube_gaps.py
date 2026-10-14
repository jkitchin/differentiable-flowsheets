"""Issue #435: wall-state ``shell_h``, optional viscosity correction, N shells in series,
F = 0 warning. Smoke and contract tests only."""

import warnings

import jax
import numpy as np
import pytest

from difflow import heat_transfer as ht
from difflow import shell_and_tube as st
from difflow.units.heat_exchanger import LowCorrectionFactorWarning

COLD = {"T": 303.0, "m_dot": 18.0}
CROSS_HOT = {"T": 340.0, "m_dot": 5.0}
CROSS_COLD = {"T": 300.0, "m_dot": 5.0}
CROSS_Q = 5.0 * 2050.0 * 24.0


def make_params(**kw):
    base = dict(
        tube_OD=0.01905, tube_ID=0.01483,
        rho_hot=780.0, mu_hot=4.0e-4, k_hot=0.13, Cp_hot=2200.0,
        rho_cold=830.0, mu_cold=2.9e-3, k_cold=0.13, Cp_cold=2050.0,
        tube_side="cold", tube_length=4.88, R_fi=1e-4, R_fo=1e-4, k_wall=45.0,
    )
    base.update(kw)
    return st.ShellAndTubeDesignParams(**base)


def cross_params(N):
    return make_params(Cp_hot=2050.0, n_shell_passes=N, tube_velocity=1.2, tube_length=None)


def test_wall_state_shell_h_smoke():
    seen = {}

    def shell_h(g):
        seen.update(g)
        return ht.mostinski(g["q_flux"], 4600.0, 1000.0)

    p = make_params(cold_isothermal=True, tube_side="hot", shell_h=shell_h, tube_length=3.0)
    r = st.ShellAndTubeDesign(p)({"T": 480.0, "m_dot": 4.0}, {"T": 400.0, "m_dot": 1.0},
                                 Q=4.0e5, warn=False)
    assert bool(r["converged"]) and np.isfinite(float(r["h_o"]))
    assert {"T_wall", "dT_film", "q_flux", "T_shell", "T_tube"} <= set(seen)


def test_viscosity_correction_is_opt_in():
    base = ht.internal_nusselt(2.0e4, 6.0)
    assert float(ht.internal_nusselt(2.0e4, 6.0, mu_ratio=0.5)) == float(base)
    corr = ht.internal_nusselt(2.0e4, 6.0, mu_ratio=1.7, viscosity_correction=True)
    assert np.isfinite(float(corr)) and float(corr) != float(base)


def test_two_shells_in_series_smoke():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = st.ShellAndTubeDesign(cross_params(2))(CROSS_HOT, CROSS_COLD, Q=CROSS_Q)
    assert bool(r["converged"]) and int(r["n_shell_passes"]) == 2
    assert float(r["area"]) == pytest.approx(2.0 * float(r["area_per_shell"]), rel=1e-12)


def test_F_zero_warns_and_is_silent_under_jit():
    d = st.ShellAndTubeDesign(cross_params(1))
    with pytest.warns(LowCorrectionFactorWarning, match="F = 0"):
        r = d(CROSS_HOT, CROSS_COLD, Q=CROSS_Q)
    assert bool(r["F_zero"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jax.jit(lambda Q: d(CROSS_HOT, CROSS_COLD, Q=Q)["F"])(CROSS_Q)
