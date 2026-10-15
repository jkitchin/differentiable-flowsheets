"""Smoke tests for Tier 2 core unit-operation enhancements (diagnostics in info)."""

import jax
import jax.numpy as jnp
import pytest

from difflow.streams import get_flows, make_stream
from difflow.thermo import IdealThermo, SpeciesData


def _rate_fn(C, T, params):
    return jnp.array([params['k'] * C['A']])


def _ab(P=101325.0):
    return make_stream({'A': 1.0, 'B': 0.0}, 350.0, P)


def test_pfr_info_and_pressure_drop():
    from difflow.units.pfr import PFR, PFRParams
    pfr = PFR(PFRParams(V=1.0, rate_fn=_rate_fn, stoich=jnp.array([[-1], [1]]),
                        rate_params={'k': 0.5}, species_order=['A', 'B'], dP_dV=1e4))
    outlet, info = pfr(_ab(P=2e5), volumetric_flow=0.01)
    assert info['solution_method'] == 'numerical'
    assert float(outlet['P']) == pytest.approx(2e5 - 1e4)


def test_hx_info_flags():
    from difflow.units.heat_exchanger import CounterCurrentHX, HeatExchangerParams
    hx = CounterCurrentHX(HeatExchangerParams(UA=100.0, Cp_hot=75.0, Cp_cold=75.0))
    _, _, info = hx(make_stream({'water': 2.0}, 400.0, 101325.0),
                    make_stream({'water': 1.0}, 300.0, 101325.0))
    assert info['flow_arrangement'] == 'counter_current'
    assert {'NTU_very_high', 'Cr_near_one'} <= set(info)


def test_shell_and_tube_hx_smoke_and_grad():
    from difflow.units.heat_exchanger import ShellAndTubeHX, ShellAndTubeHXParams

    def q(ua):
        hx = ShellAndTubeHX(ShellAndTubeHXParams(UA=ua, Cp_hot=75.0, Cp_cold=75.0))
        _, _, info = hx(make_stream({'water': 2.0}, 400.0, 101325.0),
                        make_stream({'water': 1.0}, 300.0, 101325.0))
        assert {'F_correction', 'F_too_low'} <= set(info)
        return info['Q']

    assert jnp.isfinite(jax.grad(q)(500.0))


def test_shortcut_column_diagnostics():
    from difflow.units.distillation import ShortcutColumn, ShortcutColumnParams
    data = {s: SpeciesData(s, MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                           Hvap_coeffs=(30000.0, 0.38, 500.0),
                           antoine_coeffs=(10.0, b, -40.0))
            for s, b in (('A', 2800.0), ('B', 3000.0))}
    col = ShortcutColumn(ShortcutColumnParams(species_order=['A', 'B'], light_key='A',
                                              heavy_key='B'), IdealThermo(data))
    _, _, info = col(make_stream({'A': 0.5, 'B': 0.5}, 365.0, 101325.0), R=2.0)
    assert {'alpha_variation', 'feasible', 'alpha_insufficient',
            'negative_flows_detected'} <= set(info)


def test_cstr_density_warning_and_optional_hooks():
    from difflow.units.cstr import CSTR, CSTRDensityWarning, CSTRParams
    common = dict(V=1.0, rate_fn=_rate_fn, stoich=jnp.array([[-1], [1]]),
                  rate_params={'k': 0.1}, species_order=['A', 'B'])
    with pytest.warns(CSTRDensityWarning):
        CSTR(CSTRParams(**common))(_ab())
    _, info = CSTR(CSTRParams(molar_density=1000.0, H_mix_fn=lambda f, T: jnp.asarray(100.0),
                              K_eq_fn=lambda T: jnp.asarray(1e6),
                              outlet_volumetric_basis=True, **common))(_ab())
    assert float(info['H_mix']) == pytest.approx(100.0)
    assert {'K_eq', 'X_eq_est', 'equilibrium_exceeded'} <= set(info)


def test_lle_mutual_solubility_smoke():
    from difflow.units.lle import (CascadeParams, DistributionCoeffs, LLEEquilibrium,
                                   MultistageCascade)
    eq = LLEEquilibrium(solutes=['solute'], aqueous_carrier='water', organic_carrier='hexane',
                        K_coeffs=DistributionCoeffs(species=('solute',), K0=(3.0,)),
                        mutual_solubility={'aqueous_in_organic': 0.01,
                                           'organic_in_aqueous': 0.005})
    raff, ext, info = MultistageCascade(CascadeParams(n_stages=3, equilibrium=eq))(
        make_stream({'water': 10.0, 'solute': 1.0}, 298.15, 101325.0),
        make_stream({'hexane': 5.0}, 298.15, 101325.0))
    assert info['profiles']['mutual_solubility_active']
    assert float(get_flows(ext).get('water', 0.0)) > 0
