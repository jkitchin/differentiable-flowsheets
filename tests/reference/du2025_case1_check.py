"""Du et al. (2025) Case 1 as published, checked against its own solution (#530).

Not part of the suite: it needs the benchmark data, which difflow does not
vendor (the paper is CC BY-NC-SA and the data repository has no license
file). Fetch them first, then run by hand from the repository root::

    python -c "from difflow_refinery.benchmarks import du2025; du2025.fetch()"
    PYTHONPATH=src:. python -m pytest tests/reference/du2025_case1_check.py -v

``fetch`` writes to ``$DIFFLOW_DU2025_DATA`` (default
``~/.cache/difflow/du2025``). The published solution is a GAMS ``.gdx``;
reading it needs ``pip install gamspy-base 'gamsapi[transfer]'``, or a JSON
export of it (``du2025.export_solution_json``) next to it as
``solution_baron.json``. The local solve needs ``pounce-solver``.

What it checks, at the published BARON/ANTIGONE point:

1. the loader's counts match the set/parameter files;
2. every equation of the model closes (mass balance on every unit, mixer and
   splitter included), and no inequality or bound is violated, to the
   solver's tolerance;
3. the profit equals the paper's Table 3 (34,167,968);
4. the JAX Jacobian matches central differences on random columns;
5. a local NLP solve started there stays there (same profit).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.benchmarks import du2025

jax.config.update("jax_enable_x64", True)

CASE = du2025.find_case(1)
pytestmark = pytest.mark.skipif(CASE is None, reason="benchmark data not fetched")


def _solution_path() -> Path:
    for name in ("solution_baron.json", "solution_baron.gdx"):
        if (CASE / name).exists():
            return CASE / name
    pytest.skip("no published solution file")


@pytest.fixture(scope="module")
def setup():
    data = du2025.load_case(CASE)
    model = du2025.build_model(data)
    try:
        sol = du2025.load_solution(_solution_path())
    except ImportError as exc:
        pytest.skip(str(exc))
    return data, model, model.x_from_solution(sol), sol


def test_counts(setup):
    data, *_ = setup
    c = data.counts()
    assert (c["S"], c["UPF"], c["UPD"], c["UMIX"], c["USPL"], c["UBLD"]) == (364, 20, 5, 32, 61, 12)
    assert (c["S_M"], c["S_P"], c["C"]) == (31, 37, 27)


def test_published_point_feasible(setup):
    _, model, x, _ = setup
    v = model.violations(x)
    assert v["eq"] < 1e-6, model.worst(x)
    assert v["ineq"] < 1e-6 and v["bounds"] < 1e-9


def test_published_profit(setup):
    _, model, x, sol = setup
    assert float(model.profit(x)) == pytest.approx(sol["Profit"][()], rel=1e-12)
    assert float(model.profit(x)) == pytest.approx(du2025.CASE1_PUBLISHED_PROFIT, abs=1.0)


def test_jacobian_matches_fd(setup):
    _, model, x, _ = setup
    J = np.asarray(jax.jacfwd(model.eq)(jnp.asarray(x)))
    rng = np.random.default_rng(0)
    h = 1e-6
    for j in rng.choice(model.n, 40, replace=False):
        e = np.zeros(model.n)
        e[j] = h * max(1.0, abs(x[j]))
        fd = (np.asarray(model.eq(x + e)) - np.asarray(model.eq(x - e))) / (2 * e[j])
        np.testing.assert_allclose(J[:, j], fd, rtol=1e-6, atol=1e-7)


def test_local_solve_stays(setup):
    pytest.importorskip("pounce")
    _, model, x, _ = setup
    xs, info = du2025.solve_local(model, x, options=du2025.WARM_START_OPTIONS)
    assert model.violations(xs)["eq"] < 1e-6
    assert info["profit"] == pytest.approx(float(model.profit(x)), rel=1e-4)


# --------------------------------------------------------------------------- #
# #532: property map and reconstructed assays                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def tbp(setup):
    from difflow_refinery.benchmarks.du2025.assays import fit_tbp
    return fit_tbp(setup[0])


def test_nickel_plus_vanadium(setup):
    """The relation behind Q22/Q23/Q25 (property map, 'high')."""
    fq = setup[0].param("FQcut")
    pairs = [(v, fq[(m, c, f, "Q22")] + fq[(m, c, f, "Q23")])
             for (m, c, f, q), v in fq.items() if q == "Q25"]
    ok = [abs(a - b) <= 1e-3 * max(1.0, a) for a, b in pairs]
    assert len(pairs) == 325 and sum(ok) >= 324      # one perturbed entry


def test_reference_unit_reproduced(tbp):
    from difflow_refinery.benchmarks.du2025.assays import SG_TOL, YIELD_TOL
    s = tbp.summary()["UCDU0"]
    assert s["max_yield_err"] < YIELD_TOL and s["max_sg_err"] < SG_TOL


def test_characterized_assay_reproduces_cuts(tbp):
    import warnings
    from difflow_refinery.assay import characterize
    from difflow_refinery.benchmarks.du2025.assays import SG_TOL, YIELD_TOL, crude_assay
    t = tbp.tables[0]
    for i in range(len(tbp.crudes)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ch = characterize(crude_assay(tbp, i), cut_points=tbp.cut_points[t.unit])
        mf, sg, k = np.asarray(ch.mass_fraction), np.asarray(ch.SG), len(t.cuts) - 1
        y = np.concatenate([mf[:k], [mf[k:].sum()]])
        sgc = np.concatenate([sg[:k], [mf[k:].sum() / (mf[k:] / sg[k:]).sum()]])
        assert np.max(np.abs(y - t.yields[i])) < YIELD_TOL
        assert np.max(np.abs(sgc - t.props["SPG"][i])) < SG_TOL


def test_cut_trends(tbp):
    """Sulfur, nitrogen and CCR rise with boiling point in the benchmark data;
    gravity does in all but a few (crude, cut) steps, which are recorded."""
    from difflow_refinery.benchmarks.du2025.assays import monotone_violations
    assert monotone_violations(tbp, "Q0") == {"UCDU0": 0, "UCDU1": 0}
    assert monotone_violations(tbp, "Q18") == {"UCDU0": 0, "UCDU1": 0}
    assert sum(monotone_violations(tbp, "Q21").values()) <= 1
    assert sum(monotone_violations(tbp, "SPG").values()) <= 8
