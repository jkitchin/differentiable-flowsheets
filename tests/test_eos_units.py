"""Smoke test for the EOS-consistent process units (issue #171).

Every unit here pays a large one-off JAX compile (a nested bracketed flash), so
only the JT valve is exercised: forward pass and a finite gradient.
"""

import jax
import jax.numpy as jnp
import pytest

from difflow.database import get_critical_props, get_species_data
from difflow.eos import PengRobinson
from difflow.streams import get_flows, make_stream
from difflow.thermo import CubicThermo, IdealThermo
from difflow.units.eos_units import JTValve, JTValveParams


def test_jt_valve_runs_and_is_differentiable():
    thermo = CubicThermo(IdealThermo({"methane": get_species_data("methane")}),
                         PengRobinson({"methane": get_critical_props("methane")}))
    feed = make_stream({"methane": 1.0}, 305.0, 60e5)

    def T_out(P):
        out, _ = JTValve(JTValveParams(P_out=P), thermo)(feed)
        return out["T"], out

    (T, out), g = jax.jit(jax.value_and_grad(T_out, has_aux=True))(jnp.array(20e5))
    assert bool(jnp.isfinite(T)) and bool(jnp.isfinite(g))
    assert float(get_flows(out)["methane"]) == pytest.approx(1.0)
