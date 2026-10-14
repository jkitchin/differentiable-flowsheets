"""Smoke tests for agitated-vessel (jacket and coil) heat transfer."""

import jax
import numpy as np
import pytest

from difflow.heat_transfer import (
    agitated_coil_h,
    agitated_vessel_nusselt,
    agitator_reynolds,
    jacketed_vessel_h,
)

RHO, N, DA, DT, MU, CP, K = 1000.0, 2.0, 0.5, 1.5, 1.0e-3, 4180.0, 0.6


def test_agitated_vessel_smoke():
    for v in (
        agitator_reynolds(RHO, N, DA, MU),
        agitated_vessel_nusselt(5e5, 7.0, 1.3),
        jacketed_vessel_h(RHO, N, DA, DT, MU, CP, K, impeller="flat_blade_turbine"),
        agitated_coil_h(1e5, 5.0, K, DT, 1.2),
        agitated_coil_h(1e5, 5.0, K, DT, method="oldshue_gretton", D_a=DA, d_o=0.0334),
    ):
        assert np.isfinite(float(v))


def test_unknown_impeller():
    with pytest.raises(KeyError, match="impeller"):
        agitated_vessel_nusselt(1e4, 5.0, impeller="propeller_x")


def test_coil_errors():
    with pytest.raises(ValueError, match="D_a"):
        agitated_coil_h(1e5, 5.0, K, DT, method="oldshue_gretton")
    with pytest.raises(ValueError, match="method"):
        agitated_coil_h(1e5, 5.0, K, DT, method="dittus")


def test_jacket_grad_is_finite():
    g = jax.grad(lambda n: jacketed_vessel_h(RHO, n, DA, DT, MU, CP, K))(N)
    assert np.isfinite(float(g))
