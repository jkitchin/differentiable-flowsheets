"""Smoke tests for bubbling fluidized-bed correlations and Geldart groups."""

import jax
import numpy as np
import pytest

from difflow.particles import (
    bubble_rise_velocity,
    bubble_velocity,
    bubbling_bed_expansion,
    darton_bubble_diameter,
    geldart_group,
    mori_wen_bubble_diameter,
    mori_wen_initial_bubble_diameter,
    mori_wen_max_bubble_diameter,
)

jax.config.update("jax_enable_x64", True)

U0, UMF, DT = 0.30, 0.05, 0.8          # m/s, m/s, m


def test_correlations_smoke():
    vals = [
        mori_wen_max_bubble_diameter(U0, UMF, DT),
        mori_wen_initial_bubble_diameter(U0, UMF),
        mori_wen_initial_bubble_diameter(U0, UMF, "perforated", 2000.0),
        mori_wen_bubble_diameter(0.4, U0, UMF, DT),
        darton_bubble_diameter(0.5, U0, UMF, 1e-4),
        bubble_rise_velocity(0.12),
        bubble_velocity(U0, UMF, 0.12),
        bubbling_bed_expansion(U0, UMF, 0.12, 1.2)["H"],
    ]
    assert all(np.isfinite(float(v)) and float(v) > 0 for v in vals)


def test_bed_height_grad_finite():
    def height(u0):
        d_b = mori_wen_bubble_diameter(0.5, u0, UMF, DT)
        return bubbling_bed_expansion(u0, UMF, d_b, 1.0)["H"]

    assert np.isfinite(float(jax.grad(height)(U0)))


def test_mori_wen_errors():
    with pytest.raises(ValueError, match="N_or"):
        mori_wen_initial_bubble_diameter(U0, UMF, "perforated")
    with pytest.raises(ValueError, match="distributor"):
        mori_wen_initial_bubble_diameter(U0, UMF, "bubble_cap")


def test_geldart_group_smoke():
    assert geldart_group(60e-6, 1500.0, 1.2) in "ABCD"
    assert geldart_group(60e-6, 1500.0, 1.2, method="geldart1973") in "ABCD"
    with pytest.raises(ValueError, match="method"):
        geldart_group(60e-6, 1500.0, 1.2, method="chart")
