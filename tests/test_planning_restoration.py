"""Feasibility restoration: the phase-one step for an infeasible subproblem.

An inelastic spec is the right setting for anything physical, and an inelastic
spec can make the trust-region subproblem infeasible. Before this module the
loop's only response was to shrink the radius, which cannot restore
feasibility -- a box that already excludes the feasible set excludes it harder
when it is smaller -- so the run ground down to `radius_min` and reported
`lp_infeasible` with every decision still on its starting value.

What is pinned here:

1. ``test_restoration_model_relaxes_only_the_spec_rows`` -- the design
   decision. Model and link rows are definitional and stay hard, so a
   structurally broken model is reported rather than absorbed.
2. ``test_dead_end_is_recovered`` -- the case that motivated the module.
3. ``test_disabling_restoration_reproduces_the_dead_end`` -- the before
   picture, kept executable.
4. ``test_restoration_is_judged_on_the_nonlinear_model`` -- the bug found
   while building it: a phase-one LP handed a big enough region proposes a
   point it *predicts* feasible and the blocks are not, so restoration needs
   its own trust region and its own acceptance test.
5. ``test_an_unrecognised_inequality_row_is_refused`` -- the row taxonomy is
   total. The relaxable rows are picked by a positive match, so silence is
   the default for anything new; a third row kind added to `assemble` would
   be left hard and phase one would minimise the wrong thing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import Block, DeltaBasePlanner, Network
from difflow.planning.lp import LPModel, Spec
from difflow.planning.planner import TrustRegionOptions
from difflow.planning.restoration import (
    RELAXABLE_PREFIXES, STRUCTURAL_PREFIXES,
    restoration_model, restoration_violation,
)

from tests.test_planning_multiperiod import storage_network, _levels


def _opts(**kw):
    base = dict(radius=0.4, radius_min=1e-4)
    base.update(kw)
    return TrustRegionOptions(**base)


def _planner(net, prices, specs, **kw):
    return DeltaBasePlanner(net, prices=prices, specs=specs, radius=0.4,
                            vertex_seeding=False, **kw)


def infeasible_block():
    """One lever, and a spec its starting trust region cannot reach.

    `u0` defaults to the midpoint, 5.0; the spec demands the output be at
    most 1.0, which needs u <= 1.0; a radius of 0.1 clips the region to
    [4.0, 6.0]. The LP is infeasible from the first cycle.
    """
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
    """The SOS2 adjacency of a piecewise block is not the caller's to break.

    `A_ub` holds two unrelated kinds of row: the caller's specs, and the
    adjacency rows a piecewise block emits to say that lambda_k may only be
    nonzero on the chosen interval. The second kind defines the encoding, so
    it belongs with the model and link equalities on the side phase one does
    not touch. An artificial there would let restoration buy a lower
    *predicted* violation by breaking the piecewise model, and report the
    discount as progress.
    """
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
    """A row that fits neither side of the taxonomy is an error, not a default.

    `restoration_model` picks the relaxable rows by a *positive* match on the
    name, so anything it does not recognise is quietly left hard. If a third
    kind of `A_ub` row is ever added and it is the caller's to relax, phase
    one cannot buy down the violation on it: restoration reports no progress,
    `_restore` exhausts `max_restoration`, and the planner returns
    `restoration_failed` on a problem that was recoverable -- with nothing
    anywhere to say an artificial was never created for that row.

    That is the opposite polarity from the structural-row bug above and it is
    the one the nonlinear acceptance test cannot catch: there is nothing wrong
    with the point restoration returns, there just isn't one. So the taxonomy
    is asserted total here, and adding a row kind to `assemble.py` fails at
    the moment it is added.
    """
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

    # The message names the offending row and where to declare it.
    with pytest.raises(ValueError, match=r"cut\[blk\.branch\]"):
        restoration_model(lp)
    with pytest.raises(ValueError, match="RELAXABLE_PREFIXES"):
        restoration_model(lp)


def test_an_unlabelled_inequality_row_is_refused():
    """An unlabelled row cannot be classified at all, so it is refused too.

    `ub_names` is what the taxonomy reads. A row missing from it is not a
    third kind, it is an unanswerable question -- and it would take the same
    silent path: left hard, never relaxed, never mentioned.
    """
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
    """The prefixes cover what `assemble` actually produces.

    The check above only pays off if the two recognised prefixes are the ones
    in use, so a real assembled LP -- specs and a piecewise block, the two
    `ub_names` producers -- is run through the classification here. A rename
    in `assemble.py` breaks this rather than silently un-relaxing every spec.
    """
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
    # The best it can do inside the clipped region is u = 4, y = 4.
    assert restoration_violation(sol) == pytest.approx(3.0, abs=1e-6)


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


# -- 2 and 3. the dead end, before and after ------------------------------

def test_dead_end_is_recovered():
    """The storage model whose opening level sits outside its trust region."""
    net, prices, specs = storage_network(start_as_decision=True)
    res = _planner(net, prices, specs, options=_opts()).solve()

    assert max(res.violations.values(), default=0.0) < 1e-4
    assert res.reason != "lp_infeasible"
    assert any(h.restoration for h in res.history)
    levels = _levels(net, res.decisions)
    assert np.all(levels >= -1e-4), f"negative inventory: {levels}"


def test_disabling_restoration_reproduces_the_dead_end():
    """The before picture: shrinking the radius cannot restore feasibility."""
    net, prices, specs = storage_network(start_as_decision=True)
    res = _planner(net, prices, specs,
                   options=_opts(max_restoration=0)).solve()

    assert res.reason == "lp_infeasible"
    assert max(res.violations.values(), default=0.0) > 1.0
    assert not any(h.restoration for h in res.history)


def test_restoration_drives_violation_down_monotonically():
    net, prices, specs = storage_network(start_as_decision=True)
    res = _planner(net, prices, specs, options=_opts()).solve()
    accepted = [h.violation for h in res.history
                if h.restoration and h.accepted]
    assert accepted, "no restoration step was accepted"
    assert accepted == sorted(accepted, reverse=True), (
        f"accepted restoration steps must reduce violation: {accepted}")


# -- 4. the bug that made restoration need a trust region -----------------

def test_restoration_is_judged_on_the_nonlinear_model():
    """A step the phase-one LP predicts feasible can be worse in reality.

    Measured while building this: at a doubled radius the phase-one LP took
    *predicted* violation to zero on the storage model while the true
    violation rose, because a delta vector stops describing a block long
    before the bounds do. Restoration must therefore reject such a step --
    the same rule the main acceptance test follows -- and it shows up in the
    history as a rejected restoration cycle.
    """
    net, prices, specs = storage_network(start_as_decision=True)
    res = _planner(net, prices, specs, options=_opts()).solve()
    cycles = [h for h in res.history if h.restoration]

    for h in cycles:
        if h.accepted:
            continue
        # A rejected cycle is one the LP liked and the blocks did not.
        assert h.realised >= min(c.violation for c in cycles
                                 if c.accepted) - 1e-12

    # And the end state is genuinely feasible, not merely predicted so.
    assert max(res.violations.values(), default=0.0) < 1e-4


def test_restoration_does_not_leave_the_optimisation_radius_shrunk():
    """Restoration keeps its own radius; the main loop resumes from its own.

    Searching for a feasible point can drive the working region very small,
    and that smallness is a fact about the search, not about where the
    objective model is trustworthy. Resuming from it makes the planner crawl.
    """
    net, prices, specs = storage_network(start_as_decision=True)
    res = _planner(net, prices, specs, options=_opts()).solve()
    after = [h for h in res.history if not h.restoration
             and h.index > min(c.index for c in res.history if c.restoration)]
    assert after, "the loop never resumed after restoration"
    assert max(h.radius for h in after) >= 0.1
