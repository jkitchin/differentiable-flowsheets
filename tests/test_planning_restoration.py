"""Mechanics tests for feasibility restoration (difflow.planning.restoration):
which rows the phase-one model relaxes, and that the row taxonomy is total."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import Block, DeltaBasePlanner, Network
from difflow.planning.lp import LPModel, Spec
from difflow.planning.restoration import (
    RELAXABLE_PREFIXES, STRUCTURAL_PREFIXES,
    restoration_model, restoration_violation,
)


def _planner(net, prices, specs, **kw):
    return DeltaBasePlanner(net, prices=prices, specs=specs, radius=0.4,
                            vertex_seeding=False, **kw)


def infeasible_block():
    """One lever; u0 = 5 and the spec needs u <= 1, outside a 0.1 radius."""
    return Block(name="b", fn=lambda u: jnp.array([u[0]]),
                 u_names=["x"], y_names=["y"], lb=[0.0], ub=[10.0])


# -- 1. what the phase-one model relaxes, and what it does not ------------

def test_restoration_model_relaxes_only_the_spec_rows():
    net = Network([infeasible_block()])
    planner = _planner(net, {"b.y": 1.0},
                       [Spec("b.y", "<=", 1.0, elastic=False)])
    state = net.evaluate(jnp.asarray(net.decision_start()), None)
    lp = planner.build_lp(planner.linearize(state), state, 0.1)
    phase1 = restoration_model(lp)

    artificials = [c for c in phase1.columns if c.startswith("artificial[")]
    # Every row of this model is a spec row, so all of them are relaxed.
    assert len(artificials) == lp.A_ub.shape[0]
    assert all("spec[" in a for a in artificials)
    # Equality rows -- the model row y = y0 + J(u - u0) -- keep their width
    # in the new column space but gain no artificial of their own.
    assert phase1.A_eq.shape == (lp.A_eq.shape[0], phase1.n_cols)
    assert np.all(phase1.A_eq[:, lp.n_cols:] == 0.0)


def test_a_structural_inequality_row_is_not_relaxed():
    lp = LPModel(
        columns=["u", "lam0", "z0"],
        c=np.array([1.0, 0.0, 0.0]),
        A_ub=np.array([[0.0, 1.0, -1.0],     # pw_sos2: structural
                       [1.0, 0.0, 0.0]]),    # spec: the caller's requirement
        b_ub=np.array([0.0, 5.0]),
        A_eq=np.zeros((0, 3)), b_eq=np.zeros(0),
        ub_names=["pw_sos2[blk.lambda[0]]", "spec[purity>=]"],
        lb=np.zeros(3), ub=np.array([10.0, 1.0, 1.0]),
    )
    phase1 = restoration_model(lp)

    artificials = [c for c in phase1.columns if c.startswith("artificial[")]
    assert artificials == ["artificial[spec[purity>=]]"]
    # The structural row is carried through with no way to violate it.
    assert not phase1.A_ub[0, lp.n_cols:].any()
    assert phase1.A_ub[1, lp.n_cols:] == pytest.approx([-1.0])


def test_an_unrecognised_inequality_row_is_refused():
    lp = LPModel(
        columns=["u", "v"],
        c=np.array([1.0, 0.0]),
        A_ub=np.array([[1.0, 0.0],      # spec: recognised
                       [0.0, 1.0]]),    # something new: not
        b_ub=np.array([5.0, 1.0]),
        A_eq=np.zeros((0, 2)), b_eq=np.zeros(0),
        ub_names=["spec[purity>=]", "cut[blk.branch]"],
        lb=np.zeros(2), ub=np.array([10.0, 10.0]),
    )
    with pytest.raises(ValueError, match="cannot classify"):
        restoration_model(lp)



def test_an_unlabelled_inequality_row_is_refused():
    lp = LPModel(
        columns=["u"],
        c=np.array([1.0]),
        A_ub=np.array([[1.0], [1.0]]),
        b_ub=np.array([5.0, 3.0]),
        A_eq=np.zeros((0, 1)), b_eq=np.zeros(0),
        ub_names=["spec[purity>=]"],          # only one of the two rows
        lb=np.zeros(1), ub=np.array([10.0]),
    )
    with pytest.raises(ValueError, match="ub_names labels 1"):
        restoration_model(lp)


def test_every_row_assemble_emits_is_classified():
    net = Network([infeasible_block()])
    planner = _planner(net, {"b.y": 1.0},
                       [Spec("b.y", "<=", 1.0, elastic=False)])
    state = net.evaluate(jnp.asarray(net.decision_start()), None)
    lp = planner.build_lp(planner.linearize(state), state, 0.1)

    assert lp.ub_names, "the test model should emit inequality rows"
    known = RELAXABLE_PREFIXES + STRUCTURAL_PREFIXES
    assert all(n.startswith(known) for n in lp.ub_names), lp.ub_names
    restoration_model(lp)       # does not raise


def test_restoration_model_is_feasible_where_the_lp_is_not():
    net = Network([infeasible_block()])
    planner = _planner(net, {"b.y": 1.0},
                       [Spec("b.y", "<=", 1.0, elastic=False)])
    state = net.evaluate(jnp.asarray(net.decision_start()), None)
    lp = planner.build_lp(planner.linearize(state), state, 0.1)

    assert not lp.solve().success
    sol = restoration_model(lp).solve()
    assert sol.success
    assert np.isfinite(restoration_violation(sol))


def test_restoration_violation_reports_inf_on_failure():
    net = Network([infeasible_block()])
    planner = _planner(net, {"b.y": 1.0},
                       [Spec("b.y", "<=", 1.0, elastic=False)])
    state = net.evaluate(jnp.asarray(net.decision_start()), None)
    lp = planner.build_lp(planner.linearize(state), state, 0.1)
    assert restoration_violation(lp.solve()) == float("inf")


def test_a_model_with_no_spec_rows_is_returned_without_artificials():
    net = Network([infeasible_block()])
    planner = _planner(net, {"b.y": 1.0}, [])
    state = net.evaluate(jnp.asarray(net.decision_start()), None)
    phase1 = restoration_model(planner.build_lp(
        planner.linearize(state), state, 0.1))
    assert not [c for c in phase1.columns if c.startswith("artificial[")]
    assert np.all(phase1.c == 0.0)


# -- 2. two hard specs on one variable (difflow issue 515) ----------------

def test_two_specs_on_one_variable_both_score_and_restore():
    """A >= and a <= on one variable share the default name; keyed by it,
    the violated >= vanished, restoration judged the start feasible and the
    loop ran to max_iter reporting zero violation."""
    blk = Block(name="t", fn=lambda u: jnp.array([u[0] - u[1]]),
                u_names=["made", "sold"], y_names=["level"],
                lb=[0.0, 0.0], ub=[1.0, 2.0], u0=[1.0, 1.9])
    specs = [Spec("t.level", ">=", 0.0, elastic=False),
             Spec("t.level", "<=", 0.5, elastic=False)]
    planner = _planner(Network([blk]), {"t.sold": 1.0}, specs)
    assert planner.violations({"t.level": -0.9}) == pytest.approx(
        {"t.level>=0": 0.9, "t.level<=0.5": 0.0})
    assert [s.name for s in specs] == ["t.level", "t.level"]  # not renamed in place

    res = planner.solve()
    assert res.reason != "max_iter"
    level = float(res.values["t.level"])
    assert res.violations["t.level>=0"] == pytest.approx(max(0.0, -level))
    assert res.converged and level >= -1e-6


def test_duplicate_chosen_spec_names_are_refused():
    net = Network([infeasible_block()])
    with pytest.raises(ValueError, match="two specs are named"):
        _planner(net, {"b.y": 1.0}, [Spec("b.y", ">=", 0.0, name="cap"),
                                     Spec("b.y", "<=", 1.0, name="cap")])
