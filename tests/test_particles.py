"""Smoke and validation tests for difflow.particles."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.particles import (
    bed_expansion,
    drag_coefficient,
    ergun_pressure_gradient,
    fluidization_window,
    geldart_group,
    hindered_settling_velocity,
    kozeny_carman,
    minimum_bubbling_velocity,
    minimum_fluidization_velocity,
    richardson_zaki_exponent,
    terminal_velocity,
)

AIR = (300e-6, 2600.0, 1.2, 1.8e-5)


def test_drag_and_terminal_velocity_smoke():
    for m in ("haider_levenspiel", "turton_levenspiel", "schiller_naumann"):
        assert np.isfinite(float(drag_coefficient(100.0, 1.0, m)))
    with pytest.raises(ValueError):
        drag_coefficient(1.0, 1.0, "nope")
    v = jax.jit(jax.vmap(lambda d: terminal_velocity(d, 2650.0, 998.0, 1e-3)))(
        jnp.array([1e-5, 1e-3]))
    assert np.all(np.isfinite(np.asarray(v)))


def test_hindered_settling():
    assert np.isfinite(float(richardson_zaki_exponent(10.0)))
    assert np.isfinite(float(hindered_settling_velocity(1.0, 0.5, n=4.65)))
    with pytest.raises(ValueError):
        hindered_settling_velocity(1.0, 0.5)


def test_ergun_and_kozeny_carman_smoke():
    dp, info = ergun_pressure_gradient(1.0, 3e-3, 0.4, 1.2, 1.8e-5, D_bed=0.03)
    assert float(dp) == pytest.approx(float(info["viscous"] + info["inertial"]))
    assert np.isfinite(float(kozeny_carman(1.0, 1e-3, 0.4, 1e-3)))


def test_fluidization_smoke():
    assert np.isfinite(float(minimum_fluidization_velocity(*AIR, 0.45, 0.8)))
    assert np.isfinite(float(minimum_fluidization_velocity(*AIR, method="wen_yu")))
    with pytest.raises(ValueError):
        minimum_fluidization_velocity(*AIR, method="x")
    u_mf, u_t = fluidization_window(*AIR, 0.45, 0.8)
    eps, H = bed_expansion(0.5 * float(u_t), float(u_t), n=2.39, voidage_mf=0.45, H_mf=1.0)
    assert np.isfinite(float(eps)) and np.isfinite(float(H))
    assert geldart_group(300e-6, 2600.0, 1.2) in "ABCD"
    assert np.isfinite(float(minimum_bubbling_velocity(70e-6, 1.2, 1.8e-5)))


def test_grads_finite():
    fns = [
        lambda d: terminal_velocity(d, 2650.0, 998.0, 1e-3),
        lambda d: ergun_pressure_gradient(1.0, d, 0.4, 1.2, 1.8e-5)[0],
        lambda d: minimum_fluidization_velocity(d, 2600.0, 1.2, 1.8e-5, 0.45, 0.8),
    ]
    for f in fns:
        assert np.isfinite(float(jax.grad(f)(1e-3)))


class TestGasPFRPhysicalBed:
    BED = dict(d_p=3e-3, voidage=0.4, mu=1.8e-5, rho_gas=1.2, u_s0=1.0, bed_area=0.5)

    def _params(self, **kw):
        from difflow.units.pfr import GasPFRParams

        return GasPFRParams(V=1.0, rate_fn=lambda C, T, p: jnp.zeros(1),
                            stoich=jnp.array([[-1.0], [1.0]]), rate_params={},
                            species_order=["A", "B"], **kw)

    def test_effective_alpha(self):
        assert float(self._params(**self.BED).effective_alpha) > 0
        assert self._params(alpha=123.0).effective_alpha == 123.0
        assert self._params().effective_alpha == 0.0

    def test_u_s0_from_stream(self):
        from difflow import make_stream
        from difflow.units.pfr import GasPFR

        inlet = make_stream({"A": 1.0, "B": 0.0}, 400.0, 2.0e5)
        bed = {**self.BED, "u_s0": None}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, info = GasPFR(self._params(**bed))(inlet)
        assert float(info["pressure_drop"]) > 0
        with pytest.raises(ValueError, match="alpha_for"):
            self._params(**bed).effective_alpha

    def test_validation(self):
        with pytest.raises(ValueError):
            self._params(alpha=1.0, **self.BED)
        with pytest.raises(ValueError):
            self._params(d_p=3e-3)
