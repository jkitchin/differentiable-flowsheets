"""GibbsReactor against IDAES's GibbsReactor (#468).

``tests/reference/gibbs_reference.json`` is written by
``tests/reference/gibbs_generate.py`` (IDAES + IPOPT). This test reads it
without IDAES. The reference is REGENERATED when it goes stale, never
loosened.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from difflow import GibbsReactor, GibbsReactorParams, make_stream
from tests.test_gibbs_reactor import janaf_thermo

jax.config.update("jax_enable_x64", True)

REF = json.loads((Path(__file__).parent / "reference" / "gibbs_reference.json").read_text())
NAMES = tuple(REF["components"])
REGENERATE = ("the GibbsReactor test thermo (janaf_thermo) or the cases changed: regenerate "
              "with `PYTHONPATH=src:. python tests/reference/gibbs_generate.py` "
              "(IDAES + IPOPT); never loosen the comparison")
CASES = [(kind, i) for kind, rows in REF["results"].items() for i in range(len(rows))]


@pytest.fixture(scope="module")
def th():
    return janaf_thermo(NAMES)


def test_it_records_where_it_came_from():
    p = REF["provenance"]
    assert p["idaes"] and p["ipopt"] and p["difflow_commit"]
    assert "independent implementation" in p["model"]


def test_constants_are_the_ones_the_reference_used(th):
    for i, n in enumerate(NAMES):
        r = REF["components"][n]
        assert r["Hf"] == float(th.Hf[i]), REGENERATE
        assert r["S"] == float(th.S0[i]), REGENERATE
        # janaf_thermo fits cp with np.polyfit, whose least squares lands in
        # different last bits on different LAPACK builds (1.8e-13 apart on
        # Linux OpenBLAS vs macOS Accelerate); a changed fit moves far more.
        np.testing.assert_allclose(r["cp"], th.cp[i], rtol=1e-10, err_msg=REGENERATE)


@pytest.mark.parametrize("kind,i", CASES)
def test_every_idaes_solve_is_optimal(kind, i):
    assert REF["results"][kind][i]["solve"]["status"] == "optimal"


@pytest.mark.parametrize("kind,i", CASES)
def test_matches_idaes(th, kind, i):
    r = REF["results"][kind][i]
    mode = "adiabatic" if kind == "adiabatic" else "isothermal"
    out, info = GibbsReactor(GibbsReactorParams(thermo=th, mode=mode))(
        make_stream(r["feed"], T=r["T"], P=r["P"]))
    assert bool(info["converged"])
    np.testing.assert_allclose(float(out["T"]), r["T_out"], atol=1e-4)
    for n in NAMES:
        np.testing.assert_allclose(float(out[f"F_{n}"]), r["flows"][n],
                                   rtol=1e-5, atol=1e-9, err_msg=n)
    if kind == "adiabatic":
        assert abs(r["heat_duty"]) < 1e-6
    else:
        np.testing.assert_allclose(float(info["heat_duty"]), r["heat_duty"], rtol=1e-5)
