"""Tests for difflow.shell_and_tube: smoke, API contracts and input validation."""

import warnings

import jax
import numpy as np
import pytest

from difflow import shell_and_tube as st
from difflow.units.heat_exchanger import effectiveness_shell_and_tube, lmtd_correction_factor

DO, DI = 0.01905, 0.01483
PT = 1.25 * DO
HOT = {"T": 473.0, "m_dot": 5.0}
COLD = {"T": 303.0, "m_dot": 18.0}


def make_params(**kw):
    base = dict(
        tube_OD=DO, tube_ID=DI,
        rho_hot=780.0, mu_hot=4.0e-4, k_hot=0.13, Cp_hot=2200.0,
        rho_cold=830.0, mu_cold=2.9e-3, k_cold=0.13, Cp_cold=2050.0,
        tube_side="cold", tube_length=4.88, R_fi=1e-4, R_fo=1e-4, k_wall=45.0,
    )
    base.update(kw)
    return st.ShellAndTubeDesignParams(**base)


@pytest.fixture(scope="module")
def design():
    d = st.ShellAndTubeDesign(make_params())
    return d, d(HOT, COLD, T_hot_out=373.0)


def test_geometry_and_correlation_helpers_smoke():
    vals = [
        st.equivalent_diameter(PT, DO, "triangular"),
        st.bundle_diameter(300.0, DO, 2, "triangular"),
        st.tube_count(0.5, DO, PT, "triangular", 2.0),
        st.tube_inside_diameter(0.75 * 0.0254, 16),
        st.kern_shell_side(5.0, 780.0, 4.0e-4, 0.13, 2200.0, shell_ID=0.5, tube_OD=DO,
                           pitch=PT, baffle_spacing=0.2)["Re"],
        st.tube_side_pressure_drop(18.0, 830.0, 2.9e-3, 300.0, 2.0, DI, 4.88)["dP"],
        lmtd_correction_factor(0.5, 0.3, 1),
        effectiveness_shell_and_tube(1.0, 0.5),
    ]
    assert all(np.isfinite(float(v)) for v in vals)


def test_geometry_validation():
    with pytest.raises(ValueError):
        st.equivalent_diameter(PT, DO, "hexagonal")
    with pytest.raises(KeyError, match="BWG"):
        st.tube_inside_diameter(0.0254, 99)


def test_design_smoke(design):
    _, r = design
    assert bool(r["converged"])
    for k in ("U", "area", "F", "LMTD", "Q", "n_tubes", "shell_ID", "dP_tube", "dP_shell"):
        assert np.isfinite(float(r[k])), k
    assert float(r["U"] * r["area"] * r["F"] * r["LMTD"]) == pytest.approx(float(r["Q"]), rel=1e-8)


def test_design_spec_validation(design):
    d, _ = design
    with pytest.raises(ValueError, match="exactly one"):
        d(HOT, COLD)
    with pytest.raises(ValueError, match="unknown override"):
        d(HOT, COLD, T_hot_out=373.0, bogus=1.0)
    with pytest.raises(ValueError, match="tube_side"):
        st.ShellAndTubeDesign(make_params(tube_side="middle"))
    with pytest.raises(ValueError, match="tube_velocity or tube_length"):
        st.ShellAndTubeDesign(make_params(tube_length=None))(HOT, COLD, T_hot_out=373.0)


def test_rate_and_round_design_smoke(design):
    d, r = design
    g = st.round_design(r)
    assert g["n_tube_passes"] in st.STANDARD_TUBE_PASSES
    rt = d.rate(HOT, COLD, g["n_tubes"], g["tube_length"], g["shell_ID"], g["baffle_spacing"],
                g["n_tube_passes"])
    assert np.isfinite(float(rt["Q"])) and np.isfinite(float(rt["T_hot_out"]))


def test_isothermal_shell_smoke():
    r = st.ShellAndTubeDesign(make_params(hot_isothermal=True, shell_h=6000.0, tube_length=3.0))(
        {"T": 400.0, "m_dot": 1.0}, COLD, Q=8.0e5)
    assert float(r["F"]) == 1.0 and np.isfinite(float(r["area"]))


def test_pressure_drop_limits_warn():
    d = st.ShellAndTubeDesign(make_params(dP_tube_max=1e3, dP_shell_max=1e3))
    with pytest.warns(UserWarning, match="pressure drop"):
        r = d(HOT, COLD, T_hot_out=373.0)
    assert bool(r["dP_tube_exceeded"]) and bool(r["dP_shell_exceeded"])


def test_grad_area_finite_and_silent():
    d = st.ShellAndTubeDesign(make_params(dP_tube_max=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = jax.grad(lambda lB: d(HOT, COLD, T_hot_out=373.0, baffle_spacing=lB)["area"])(0.2)
    assert np.isfinite(float(g))


def test_params_dict_like():
    p = make_params()
    assert p["tube_OD"] == DO and "tube_ID" in p
    assert p.update(tube_length=6.0).tube_length == 6.0
