"""The gas plant columns against IDAES's TrayColumn on Peng-Robinson (#312).

``release`` throughout: each case is a full column solve per side, and the
reference is the committed ``reference/gasplant_reference.json`` (IDAES is
not needed to run this; see ``reference/gasplant_generate.py`` to rebuild
it). The per-commit checks on the file itself are in
``test_gasplant_validation_file.py``.

What this is: an independent *implementation* of the same model -- PR with
zero kij on the same constants, equilibrium trays, a total condenser at the
bubble point, a kettle reboiler, the reflux and boilup ratios fixed. It
checks difflow's EOS, MESH equations and solver against IDAES's. It does not
say how well PR describes these mixtures.

Three layers, from the most local to the least:

* K-values point by point, difflow's ``ln phi_L - ln phi_V`` evaluated at
  IDAES's own (T, P, x, y) on every stage -- no column in the way;
* the stage temperature profile;
* the products' compositions and the duties, the issue's 1 % criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.gasplant import CubicThermo, gas_components
from difflow_refinery.gasplant.thermo import PR

from .reference import gasplant_case as gc

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.release, pytest.mark.slow]

REF = json.loads((Path(__file__).parent / "reference" / "gasplant_reference.json").read_text())


@pytest.fixture(scope="module", params=sorted(gc.CASES))
def solved(request):
    name = request.param
    case = gc.CASES[name]
    col, feed = gc.difflow_column(case)
    D, B, info = col(feed)
    assert bool(info["converged"]), name
    return name, case, REF["cases"][name]["idaes"], D, B, info


def _x(stream, names):
    f = np.array([float(stream[f"F_{n}"]) for n in names])
    return f / f.sum()


def test_k_values_point_by_point(solved):
    """Measured: worst relative difference 1.3e-7 (debutanizer); the two
    implementations agree to IPOPT's tolerance, not just to 1 %."""
    name, case, ref, *_ = solved
    th = CubicThermo(gas_components(case["names"]), PR)
    worst = 0.0
    for s in ref["stages"][1:]:
        lnK = th.log_K(s["T"], s["P"], jnp.asarray(s["x"]), jnp.asarray(s["y"]))
        worst = max(worst, float(np.max(np.abs(np.exp(np.asarray(lnK)) / np.asarray(s["K"]) - 1))))
    assert worst < 1e-5, (name, worst)


def test_temperature_profile(solved):
    name, case, ref, D, B, info = solved
    T = np.asarray(info["profiles"]["T"])
    T_ref = np.array([s["T"] for s in ref["stages"]])
    assert np.max(np.abs(T - T_ref)) < 0.25, name


def test_product_compositions_within_one_percent(solved):
    name, case, ref, D, B, info = solved
    names = case["names"]
    for prod, stream in (("distillate", D), ("bottoms", B)):
        f_ref = np.array([ref[prod][n] for n in names])
        np.testing.assert_allclose(_x(stream, names), f_ref / f_ref.sum(), rtol=0.01, atol=1e-5,
                                   err_msg=f"{name} {prod}")


def test_duties_within_one_percent(solved):
    """IDAES signs a condenser's duty negative (heat in); difflow reports
    the heat removed."""
    name, case, ref, D, B, info = solved
    o = info["outputs"]
    assert float(o["condenser.duty"]) == pytest.approx(-ref["condenser_duty"], rel=0.01), name
    assert float(o["reboiler.duty"]) == pytest.approx(ref["reboiler_duty"], rel=0.01), name


def test_the_specs_are_the_same_specs(solved):
    name, case, ref, D, B, info = solved
    o = info["outputs"]
    assert float(o["reflux_ratio"]) == pytest.approx(ref["reflux_ratio"], rel=1e-6)
    assert float(o["boilup_ratio"]) == pytest.approx(ref["boilup_ratio"], rel=1e-6)
