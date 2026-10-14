"""CSTR, PFR, GasPFR and FedBatchReactor implement the DynamicUnit protocol."""

import jax
import jax.numpy as jnp
import pytest

from difflow.dynamic import DynamicUnit, StateSpec, integrate, integrate_unit
from difflow.streams import make_stream
from difflow.units.cstr import CSTR, CSTRParams
from difflow.units.fed_batch import FedBatchParams, FedBatchReactor
from difflow.units.pfr import PFR, GasPFR, GasPFRParams, PFRParams

KW = dict(rate_fn=lambda C, T, p: jnp.array([p["k"] * C["A"]]),
          stoich=jnp.array([[-1.0], [1.0]]), rate_params={"k": 0.1},
          species_order=["A", "B"])
INLET = {"inlet": make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)}


@pytest.mark.parametrize("unit", [
    CSTR(CSTRParams(V=1.0, **KW)), PFR(PFRParams(V=1.0, **KW)),
    GasPFR(GasPFRParams(V=1.0, **KW)), FedBatchReactor(FedBatchParams(V0=1.0, **KW)),
], ids=["CSTR", "PFR", "GasPFR", "FedBatch"])
def test_units_are_dynamic_units(unit):
    assert isinstance(unit, DynamicUnit)
    spec = unit.state_spec()
    assert isinstance(spec, StateSpec) and spec.n_states > 0


def test_cstr_dynamic_interface():
    cstr = CSTR(CSTRParams(V=1.0, **KW))
    y0 = cstr.initial_state(INLET)
    assert y0.shape == (2,)
    assert jnp.all(jnp.isfinite(cstr.derivatives(0.0, y0, INLET)))
    assert "outlet" in cstr.outputs(0.0, y0, INLET)
    result = integrate_unit(cstr, inputs=INLET, t_span=(0.0, 60.0), method="RK4", n_steps=10)
    assert result.info.success

    def b_final(y0):
        f = lambda t, y: cstr.derivatives(t, y, INLET)  # noqa: E731
        return integrate(f, y0, (0.0, 10.0), method="RK4", n_steps=10).y_final[1]

    assert jnp.all(jnp.isfinite(jax.grad(b_final)(jnp.array([60.0, 0.0]))))


def test_pfr_dynamic_interface():
    pfr = PFR(PFRParams(V=1.0, **KW))
    y0 = pfr.initial_state(INLET)
    assert jnp.allclose(pfr.derivatives(0.0, y0, INLET), 0.0)  # pseudo-steady-state
    assert "outlet" in pfr.outputs(0.0, y0, INLET)


def test_fed_batch_dynamic_interface():
    reactor = FedBatchReactor(FedBatchParams(V0=1.0, **KW))
    y0 = reactor.initial_state(inputs={}, params={"C0": {"A": 100.0, "B": 0.0}, "T0": 350.0})
    assert y0.shape == (3,)
    dy = reactor.derivatives(0.0, y0, {}, params={"T_spec": 350.0})
    assert jnp.all(jnp.isfinite(dy))
    out = reactor.outputs(0.0, y0, {}, params={"T_spec": 350.0})
    assert {"contents", "V", "n", "C"} <= set(out)
