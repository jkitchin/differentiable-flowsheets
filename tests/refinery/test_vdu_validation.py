"""The vacuum column checked against an independent simulator (issue #294).

What the reference is, and why it is not DWSIM: the issue asked for DWSIM,
or IDAES if a suitable column exists. DWSIM was not available, and IDAES has
no vacuum-column model with pumparounds, a wash bed, entrainment routes and
Murphree beds. So, as for the crude unit (#298), the column is written
independently in Pyomo -- IDAES's modelling layer -- as an equation-oriented
MESH model (``reference/vdu_mesh.py``) and solved by IDAES's IPOPT build,
on a property model transcribed from the published correlations
(``reference/vdu_formulas.py``). ``reference/vdu_generate.py`` writes
``reference/vdu_reference.json``; these tests read it and need neither Pyomo
nor IPOPT.

The two models share only the input: difflow's characterised residue (the
pseudo-component table and the feed's molar flows). The formulations differ
in every place they can (the table in ``vdu_mesh``'s docstring): mole
fractions and total flows against log component flows, absolute route flows
against softmax draws, an explicit summation, the pumparound return
temperature as an unknown instead of a duty, the LVGO end point as a smooth
cumulative-mass equation, IPOPT against a damped Newton. And the reference
starts from an engineering guess built from the feed flash and the specs,
not from difflow's answer.

What is compared, and the tolerance each is held to (measured agreement in
brackets, so the margin is visible):

* product rates, 1e-6 relative (<= 3e-7);
* stage and product-TBP temperatures, 1e-4 K (<= 2e-5 K);
* pumparound and furnace duties, 1e-6 relative (<= 2e-7);
* product specific gravities, 1e-8 (<= 3e-9);
* sensitivities to furnace outlet temperature, flash-zone pressure and
  stripping steam -- ``jax.jacfwd`` through difflow's solve against central
  differences of the reference -- 2e-4 relative (<= 4.1e-5, which is the
  differences' own truncation error).

The residual 1e-7-level differences are not noise to be explained away: the
reference converts Maxwell-Bonnell's Watson-K correction in Rankine with the
coefficient 2.5/1.8 written out, difflow in SI with 1.3889 -- a 4e-5
relative difference in one coefficient, which is the size of difference that
lands in the products. Both models carry four fixed-point passes of that
correction; :class:`TestWhatTheModelChoicesCost` reports what solving it
exactly, and using the published piecewise branches instead of the blend,
would change. Those are properties of the stated model, not disagreements
between the simulators.

The whole module is ``release``: its subject is the answer, and a deliberate
model change is what moves it. The cheap checks -- the file is intact, the
characterisation is the one it was built on -- are in
``test_vdu_validation_file.py`` and run on every commit.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .reference import vdu_case as vc

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.release

REF = json.loads((Path(__file__).parent / "reference" / "vdu_reference.json").read_text())
FEED = np.asarray(REF["case"]["feed_mol_s"])
PRODUCTS = ("overhead", "lvgo", "hvgo", "slop", "residue")
PERCENTS = ("T05", "T10", "T50", "T90", "T95")


@pytest.fixture(scope="module")
def comp():
    """difflow's characterisation now (``TestReferenceIsCurrent`` in the
    file module checks it is the reference's)."""
    return vc.characterize()[0]


@pytest.fixture(scope="module")
def equilibrium(comp):
    return vc.solve_difflow(comp, FEED)[-1]


@pytest.fixture(scope="module")
def murphree(comp):
    return vc.solve_difflow(comp, FEED, efficiency=vc.EFFICIENCY_CASE,
                            specs=vc.EFFICIENCY_SPECS)[-1]


def _check_column(info, ref):
    assert bool(info["converged"])
    o, r = info["outputs"], ref["outputs"]
    for p in PRODUCTS:
        # the overhead is 7e-4 kg/s of light vapour; relative to it the
        # reference's 7.5e-10 kg/s difference is 1e-6, so it gets an absolute floor
        assert float(o[f"{p}.rate"]) == pytest.approx(r[f"{p}.rate"], rel=1e-6, abs=1e-8), p
        assert float(info["properties"][p]["sg"]) == pytest.approx(r[f"{p}.sg"], abs=1e-8), p
        for pct in PERCENTS:
            assert float(o[f"{p}.{pct}"]) == pytest.approx(r[f"{p}.{pct}"], abs=1e-4), (p, pct)
    for route in ("lvgo_pa", "hvgo_pa", "wash_oil", "entrained", "entrained_bypass"):
        assert float(o[f"{route}.rate"]) == pytest.approx(r[f"{route}.rate"], rel=1e-6), route
    np.testing.assert_allclose(np.asarray(info["profiles"]["T"]), ref["T"], rtol=0, atol=1e-4)
    for k in ("lvgo_pa.duty", "hvgo_pa.duty", "furnace.duty"):
        assert float(o[k]) == pytest.approx(r[k], rel=1e-6), k
    for k in ("lvgo_pa.return_T", "hvgo_pa.return_T", "flash_zone.T"):
        assert float(o[k]) == pytest.approx(r[k], abs=1e-4), k
    assert float(o["furnace.vapor_fraction"]) == pytest.approx(r["furnace.vapor_fraction"],
                                                               rel=1e-6)


@pytest.mark.slow
class TestColumnAgainstReference:
    def test_equilibrium_beds(self, equilibrium):
        _check_column(equilibrium, REF["column"])

    def test_murphree_beds(self, murphree):
        """LVGO 80 %, HVGO 70 %, wash 50 %, stripping 40 %, on a 520 C LVGO
        end point: the efficiency path, where the beds' below-stage vapour
        enters every equilibrium relation."""
        _check_column(murphree, REF["murphree"])

    def test_the_efficiencies_move_the_answer_far_more_than_the_models_differ(self):
        """So the Murphree comparison is a test of the efficiency path, not of
        a column that barely changed: the LVGO rate more than triples."""
        a, b = REF["column"]["outputs"], REF["murphree"]["outputs"]
        assert b["lvgo.rate"] > 3.0 * a["lvgo.rate"]
        assert b["hvgo.rate"] < 0.5 * a["hvgo.rate"]


@pytest.mark.slow
class TestWhatTheModelChoicesCost:
    """Not a comparison with difflow: the reference re-solved with the two
    numerical choices difflow's model states, undone. Pinned so the docs'
    numbers stay true, and so a change that makes them matter is noticed."""

    def _rel(self, variant, key):
        a = REF["variants"][variant]["column"]["outputs"][key]
        b = REF["column"]["outputs"][key]
        return abs(a - b) / abs(b)

    def test_four_watson_passes_are_as_good_as_the_fixed_point(self):
        """The truncated fixed point moves ln Psat by up to 6e-3 at the top
        stage, but the products by under 1e-7 -- well inside the comparison."""
        for k in ("lvgo.rate", "hvgo.rate", "residue.rate", "lvgo_pa.duty", "hvgo_pa.duty"):
            assert self._rel("exact_fixed_point", k) < 1e-7, k

    def test_the_branch_blend_costs_a_tenth_of_a_percent(self):
        """Blending Maxwell-Bonnell's branches (for differentiability) instead
        of switching at the published joins moves duties and the LVGO rate
        by 8-9e-4: invisible against 'a few percent', and stated."""
        for k in ("lvgo.rate", "hvgo.rate", "lvgo_pa.duty", "hvgo_pa.duty", "furnace.duty"):
            assert 1e-5 < self._rel("published_branches", k) < 2e-3, k


@pytest.mark.slow
class TestGradientsAgainstReference:
    """``jax.jacfwd`` through difflow's solve against central differences of
    the reference column, re-solved by IPOPT at each perturbed point."""

    @pytest.mark.parametrize("knob", ["furnace_T", "flash_zone_P", "steam_per_feed"])
    def test_sensitivities(self, comp, knob):
        fd = REF["sensitivities"][knob]["d"]
        names = list(fd)

        def q(x):
            o = vc.solve_difflow(comp, FEED, **{knob: x})[-1]["outputs"]
            return jnp.stack([o[k] for k in names])

        J = np.asarray(jax.jacfwd(q)(jnp.asarray(vc.LAYOUT[knob])))
        for k, g in zip(names, J):
            # an absolute floor for the furnace duty's steam sensitivity, which is zero
            assert g == pytest.approx(fd[k], rel=2e-4, abs=1e-6), (knob, k)


@pytest.mark.slow
def test_regeneration_reproduces_the_committed_reference(tmp_path):
    """Optional: rebuild the reference (without the variants or finite
    differences) and check it lands on the committed numbers. Needs Pyomo
    and an IPOPT."""
    pytest.importorskip("pyomo")
    import shutil

    from .reference import vdu_generate

    ipopt = shutil.which("ipopt") or vdu_generate.DEFAULT_IPOPT
    if not Path(ipopt).exists():
        pytest.skip("no ipopt")
    out = tmp_path / "ref.json"
    vdu_generate.main(["--ipopt", ipopt, "--out", str(out), "--skip-fd", "--skip-variants"])
    new = json.loads(out.read_text())
    for part in ("column", "murphree"):
        np.testing.assert_allclose(new[part]["T"], REF[part]["T"], atol=1e-6)
        for p in PRODUCTS:
            k = f"{p}.rate"
            assert new[part]["outputs"][k] == pytest.approx(REF[part]["outputs"][k],
                                                            rel=1e-8, abs=1e-12), (part, k)
