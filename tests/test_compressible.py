"""Smoke tests for difflow.compressible: nozzles, isothermal and Fanno gas flow,
polytropic efficiency and staging."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.compressible import (
    adiabatic_outlet_mach,
    choked_mass_flux,
    critical_pressure_ratio,
    fanno_parameter,
    fanno_ratios,
    isentropic_efficiency_from_polytropic,
    isothermal_distance,
    isothermal_limit_mach,
    isothermal_outlet_pressure,
    isothermal_pipe_flow,
    isothermal_pipe_mass_flux,
    isothermal_pressure_profile,
    mach_number,
    nozzle_flow,
    nozzle_mass_flow,
    optimal_stage_ratios,
    polytropic_efficiency_from_isentropic,
    polytropic_exponent,
    speed_of_sound,
    stagnation_ratios,
)

jax.config.update("jax_enable_x64", True)

AIR = 0.02897


def _finite(*vals):
    return all(np.isfinite(float(v)) for v in vals)


def test_scalar_relations_smoke():
    a = speed_of_sound(300.0, AIR, 1.4)
    assert _finite(a, mach_number(100.0, 300.0, AIR, 1.4), critical_pressure_ratio(1.4),
                   stagnation_ratios(0.5, 1.4).P0_P, choked_mass_flux(8e5, 320.0, AIR, 1.4),
                   fanno_parameter(0.3, 1.4), fanno_ratios(0.3, 1.4).P_Pstar,
                   isothermal_limit_mach(1.4), polytropic_exponent(0.8, 1.4),
                   isentropic_efficiency_from_polytropic(4.0, 0.8, 1.4),
                   polytropic_efficiency_from_isentropic(4.0, 0.75, 1.4, False))


def test_nozzle_smoke_and_grad():
    r = nozzle_flow(10e5, 293.15, 8e5, 7.85e-5, AIR, 1.4, 0.85)
    assert _finite(r.m_dot, r.P_throat) and not bool(r.choked)
    g = jax.grad(lambda Pb: nozzle_mass_flow(10e5, 293.15, Pb, 7.85e-5, AIR, 1.4, 0.85))(8e5)
    assert np.isfinite(float(g))


def test_isothermal_pipe_smoke_and_grad():
    args = dict(T=288.0, L=5e4, D=0.5, f=0.012, MW=0.0175)
    G = isothermal_pipe_mass_flux(60e5, 40e5, **args)
    out = isothermal_outlet_pressure(float(G), 60e5, 288.0, 5e4, 0.5, 0.012, 0.0175)
    assert _finite(G, out.P2)
    g = jax.grad(lambda P1: isothermal_pipe_mass_flux(P1, 40e5, **args))(60e5)
    assert np.isfinite(float(g))


def test_isothermal_pipe_flow_and_profile_smoke():
    r = isothermal_pipe_flow(60e5, 40e5, 288.15, 50e3, 0.6, 45e-6, 1.2e-5, 0.018, Z=0.851)
    assert _finite(r.G, r.m_dot, r.f)
    prof = isothermal_pressure_profile(jnp.linspace(0.0, 50e3, 3), r.G, 60e5, 288.15, 0.6, r.f,
                                       0.018, 1.3, 0.851)
    assert prof.P2.shape == (3,)
    back = isothermal_distance(prof.P2[1:], r.G, 60e5, 288.15, 0.6, r.f, 0.018, 0.851)
    assert back.shape == (2,)


def test_fanno_smoke_and_grads():
    out = adiabatic_outlet_mach(0.3, 2.0, 1.4)
    assert _finite(out.M2, out.P2_P1, out.T2_T1)
    # choked branch gradient is finite (regression)
    g = jax.grad(lambda fl: adiabatic_outlet_mach(0.5, fl, 1.4).P2_P1)(2.0)
    assert np.isfinite(float(g))


def test_jit_vmap():
    f = jax.jit(jax.vmap(lambda L: isothermal_outlet_pressure(50.0, 60e5, 288.0, L, 0.5, 0.012, 0.0175).P2))
    assert f(jnp.array([1e4, 5e4])).shape == (2,)


def test_staging():
    r = optimal_stage_ratios(1e5, 110e5, 4)
    assert r.shape == (4,)
    with pytest.raises(ValueError):
        optimal_stage_ratios(1e5, 1e6, 0)
