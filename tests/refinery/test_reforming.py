"""Catalytic reformer: smoke and gradient tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.reforming import CatalyticReformer, ReformerParams, ReformingKinetics, lean_naphtha
from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming.products import reformate_properties
from difflow_refinery.reforming.reactor import integrate_bed
from difflow_refinery.reforming.sulfur import sulfur_balance

jax.config.update("jax_enable_x64", True)


def test_feed_smoke():
    f = lean_naphtha(12.0)
    assert float(f.mass_flow) == pytest.approx(12.0, rel=1e-12)
    g = f.with_group_fraction("naphthenes", 0.35)
    assert float(g.group_fractions("volume")["naphthenes"]) == pytest.approx(0.35, rel=1e-12)


def test_bed_smoke_and_jacfwd():
    F = jnp.asarray(lean_naphtha().flows)
    F = F.at[sp.INDEX["H2"]].set(5.0 * float(jnp.sum(F)))
    kin = ReformingKinetics()
    Fo, To, coke, _ = integrate_bed(F, 773.15, 15e5, 3000.0, kin)
    assert np.all(np.isfinite(np.asarray(Fo))) and float(To) < 773.15
    g = jax.jacfwd(lambda T: integrate_bed(F, T, 15e5, 3000.0, kin)[1])(773.15)
    assert np.isfinite(float(g))


def test_reformate_properties_smoke():
    p = reformate_properties(jnp.zeros(sp.N_SPECIES).at[sp.INDEX["A7"]].set(1.0))
    assert float(p["aromatics_vol"]) == pytest.approx(100.0)


def test_sulfur_balance_smoke_and_grad():
    def bal(S):
        return sulfur_balance(feed_mass=10.0, feed_S_wppm=S, conversion=0.95, V=900.0, L=110.0,
                              T_sep=311.15, P_sep=12e5, recycle_fraction=0.8, reformate_mass=8.5,
                              net_gas_moles=180.0, recycle_moles=720.0)

    S = bal(0.5)
    assert float(S["S.reformate"] + S["S.net_gas"] + S["S.fuel_gas"]) == pytest.approx(
        float(S["S.feed"]), rel=1e-12)
    assert np.isfinite(float(jax.grad(lambda x: bal(x)["reformate.S_wppm"])(0.5)))


def test_reformer_smoke():
    # three tear iterations: the code path, not the converged answer (compile dominates)
    res = CatalyticReformer(ReformerParams()).solve(lean_naphtha(), max_iter=3, on_nonconvergence="ignore")
    o = res.outputs()
    assert np.isfinite(float(o["reformate.RON"])) and np.isfinite(float(o["h2.net_mol_s"]))
    assert np.isfinite(float(res.blend_component().properties["RON"]))


def test_bed_solver_validation():
    F = jnp.asarray(lean_naphtha().flows)
    with pytest.raises(ValueError, match="solver_name"):
        integrate_bed(F, 773.15, 15e5, 3000.0, ReformingKinetics(), solver_name="euler")
    with pytest.raises(ValueError, match="max_steps"):
        integrate_bed(F, 773.15, 15e5, 3000.0, ReformingKinetics(), max_steps=0)
    with pytest.raises(ValueError, match="solver_name"):
        ReformerParams(bed_solver="euler")
    with pytest.raises(ValueError, match="max_steps"):
        ReformerParams(bed_max_steps=0)


def test_bed_step_cap_passed_through():
    # #479: the step cap and solver reach the beds (a too-small cap raises, it is not ignored)
    p = ReformerParams(bed_max_steps=2)
    fs, ops, _ = CatalyticReformer(p).build(lean_naphtha(), p)
    assert ops["rx1"].params.max_steps == 2 and ops["rx1"].params.solver_name == "tsit5"
    F = jnp.asarray(lean_naphtha().flows)
    F = F.at[sp.INDEX["H2"]].set(5.0 * float(jnp.sum(F)))
    with pytest.raises(Exception, match="max_steps"):
        integrate_bed(F, 773.15, 15e5, 3000.0, ReformingKinetics(), max_steps=2)
