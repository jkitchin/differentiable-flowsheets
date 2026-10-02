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

For ``debutanizer``, three layers, from the most local to the least:

* K-values point by point, difflow's ``ln phi_L - ln phi_V`` evaluated at
  IDAES's own (T, P, x, y) on every stage -- no column in the way;
* the stage temperature profile;
* the products' compositions and the duties (the issue asks for 1 %; they
  agree to 1e-7).

``c3c4_splitter`` has no IDAES column to compare against (IDAES's
``TrayColumn`` initialization does not converge it; see
``reference/gasplant_generate.py``). Its reference is IDAES's flash at each
of difflow's stage states, which ``test_gasplant_validation_file.py``
checks per commit against difflow's thermodynamics; what is left here is
that difflow's column still puts its stages at those states.
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


COLUMN_CASES = sorted(n for n, c in gc.CASES.items() if c.get("reference", "column") == "column")
POINT_CASES = sorted(n for n, c in gc.CASES.items() if c.get("reference") == "state_points")


@pytest.fixture(scope="module", params=COLUMN_CASES)
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
    """Measured: worst relative difference 1.4e-13 (debutanizer)."""
    name, case, ref, *_ = solved
    th = CubicThermo(gas_components(case["names"]), PR)
    worst = 0.0
    for s in ref["stages"][1:]:
        lnK = th.log_K(s["T"], s["P"], jnp.asarray(s["x"]), jnp.asarray(s["y"]))
        worst = max(worst, float(np.max(np.abs(np.exp(np.asarray(lnK)) / np.asarray(s["K"]) - 1))))
    assert worst < 1e-6, (name, worst)


def test_temperature_profile(solved):
    """Measured: 6.8e-7 K worst stage (debutanizer)."""
    name, case, ref, D, B, info = solved
    T = np.asarray(info["profiles"]["T"])
    T_ref = np.array([s["T"] for s in ref["stages"]])
    assert np.max(np.abs(T - T_ref)) < 1e-4, name


def test_product_compositions(solved):
    """The issue asks for 1 %; measured 7.5e-8 relative worst (debutanizer),
    so the test holds it to 1e-5 -- a 1 % band would let a real regression
    through. Before the generator tightened SmoothVLE's smoothing the gap
    was 0.7 %, all of it IDAES's condenser (see the generator)."""
    name, case, ref, D, B, info = solved
    names = case["names"]
    for prod, stream in (("distillate", D), ("bottoms", B)):
        f_ref = np.array([ref[prod][n] for n in names])
        np.testing.assert_allclose(_x(stream, names), f_ref / f_ref.sum(), rtol=1e-5, atol=1e-9,
                                   err_msg=f"{name} {prod}")


def test_duties(solved):
    """IDAES signs a condenser's duty negative (heat in); difflow reports
    the heat removed. Measured: 1.2e-8 (condenser) and 2.4e-8 (reboiler)
    relative (debutanizer); held to 1e-5."""
    name, case, ref, D, B, info = solved
    o = info["outputs"]
    assert float(o["condenser.duty"]) == pytest.approx(-ref["condenser_duty"], rel=1e-5), name
    assert float(o["reboiler.duty"]) == pytest.approx(ref["reboiler_duty"], rel=1e-5), name


def test_the_specs_are_the_same_specs(solved):
    name, case, ref, D, B, info = solved
    o = info["outputs"]
    assert float(o["reflux_ratio"]) == pytest.approx(ref["reflux_ratio"], rel=1e-6)
    assert float(o["boilup_ratio"]) == pytest.approx(ref["boilup_ratio"], rel=1e-6)


@pytest.mark.parametrize("name", POINT_CASES)
def test_the_state_points_are_still_difflows_stages(name):
    """The state-point reference is only a check on the column if the column
    still sits at those points: same T to 1e-6 K, same overall stage
    composition to 1e-8."""
    case = gc.CASES[name]
    col, feed = gc.difflow_column(case)
    _, _, info = col(feed)
    assert bool(info["converged"])
    pr = info["profiles"]
    for j, p in enumerate(REF["cases"][name]["idaes_points"], start=1):
        L, V = float(pr["L"][j]), float(pr["V"][j])
        z = (L * np.asarray(pr["x"][j]) + V * np.asarray(pr["y"][j])) / (L + V)
        assert float(pr["T"][j]) == pytest.approx(p["T"], abs=1e-6), p["stage"]
        np.testing.assert_allclose(z, p["z"], atol=1e-8, err_msg=p["stage"])
