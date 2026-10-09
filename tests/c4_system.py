"""Shared cubic-EOS test system: propane / n-butane / isobutane (#389, #390).

A Peng-Robinson EOS and CubicThermo for the C4 mixture, and an adiabatic
vapor-phase n-butane -> isobutane CSTR on them. The steady-state EO tests and
the dynamic flowsheet tests build the same feed-effluent flowsheet from these,
so the dynamic long-time limit can be checked against the steady state.
"""

import jax.numpy as jnp

from difflow import CSTR, CSTRParams, IdealThermo, SpeciesData, make_stream
from functools import lru_cache

from difflow.eos import CriticalProperties, PengRobinson
from difflow.thermo import CubicThermo

C4_SPECIES = ["propane", "butane", "isobutane"]
C4_P = 8e5


@lru_cache(maxsize=None)
def _c4_eos_and_thermo():
    """PR EOS and CubicThermo for propane / n-butane / isobutane.

    Memoized: the units' JIT caches key on the thermo identity, so sharing
    one object compiles each solve once across the tests here.
    """
    species = {
        "propane": SpeciesData(name="propane", MW=44.10, Cp_coeffs=(73.0, 0.0, 0.0, 0.0),
                               Hvap_coeffs=(18000.0, 0.38, 369.8),
                               antoine_coeffs=(13.72, 1872.5, -25.16)),
        "butane": SpeciesData(name="butane", MW=58.12, Cp_coeffs=(98.0, 0.0, 0.0, 0.0),
                              Hvap_coeffs=(22000.0, 0.38, 425.1),
                              antoine_coeffs=(13.98, 2292.4, -27.86)),
        "isobutane": SpeciesData(name="isobutane", MW=58.12, Cp_coeffs=(96.0, 0.0, 0.0, 0.0),
                                 Hvap_coeffs=(21000.0, 0.38, 407.8),
                                 antoine_coeffs=(13.82, 2181.8, -24.28)),
    }
    crit = {
        "propane": CriticalProperties(name="propane", Tc=369.8, Pc=4.25e6, omega=0.152, MW=44.10),
        "butane": CriticalProperties(name="butane", Tc=425.1, Pc=3.80e6, omega=0.200, MW=58.12),
        "isobutane": CriticalProperties(name="isobutane", Tc=407.8, Pc=3.64e6, omega=0.184, MW=58.12),
    }
    eos = PengRobinson(crit)
    return eos, CubicThermo(IdealThermo(species), eos)


def _isomerization_cstr():
    """Adiabatic vapor-phase n-butane -> isobutane CSTR on the PR EOS."""
    eos, thermo = _c4_eos_and_thermo()

    def rate_fn(C, T, p):
        return jnp.array([p["k0"] * jnp.exp(-p["Ea"] / (8.314 * T)) * C["butane"]])

    params = CSTRParams(
        V=0.5, rate_fn=rate_fn, stoich=jnp.array([[0.0], [-1.0], [1.0]]),
        rate_params={"k0": 2e3, "Ea": 40000.0}, species_order=C4_SPECIES,
        dH_rxn=jnp.array([-8000.0]), eos=eos, reaction_phase="vapor",
    )
    return CSTR(params, thermo=thermo, mode="adiabatic")


def _c4_stream(T, flows=(0.5, 1.0, 0.1)):
    return make_stream(dict(zip(C4_SPECIES, flows)), T=T, P=C4_P)
