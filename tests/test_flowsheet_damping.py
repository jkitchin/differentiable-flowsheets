"""Tests for ``damping`` on Flowsheet.solve.

``damping`` was accepted, documented and *silently ignored*: it was passed
down to ``_solve_with_recycle_damped`` and never read, so the
``acceleration="none"`` path -- which is also the path a traced solve falls
back to -- iterated the undamped map whatever the caller asked for. A loop
whose tear map overshoots therefore had no stabilising knob at all on a
plain Flowsheet, which is why ``difflow_gas`` had to write its own damped
fixed point.

These tests pin the fix and the three conventions that make it safe to use:
the answer does not depend on the damping, the *reported* residual is the
undamped one, and ``tol`` still means what it says.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from difflow import ConvergenceWarning
from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream


class Overshoot:
    """``g(x) = 3 F - 2 x``: one tear, eigenvalue -2, fixed point ``x = F``.

    Plain substitution triples the error and flips its sign every step, so
    the undamped iteration diverges from any start but the answer itself.
    Damped by ``alpha``, the iteration is ``x <- (1 - 3 alpha) x + 3 alpha F``,
    which contracts for ``alpha < 2/3`` -- the ``alpha < 2/(1 + m)`` bound
    with ``m = 2``. Linear on purpose: every number below is exact.
    """

    def __call__(self, feed, tear):
        return make_stream(
            {"A": 3.0 * feed["F_A"] - 2.0 * tear["F_A"]},
            feed["T"],
            feed["P"],
        )


class Contraction:
    """``g(x) = F + 0.9 sqrt(x + 0.1)``: monotone and already contractive.

    Damping cannot help this one; it can only slow it down, which is what
    makes it the right map for showing that ``damping`` is read at all.
    """

    def __call__(self, feed, tear):
        return make_stream(
            {"A": feed["F_A"] + 0.9 * jnp.sqrt(tear["F_A"] + 0.1)},
            feed["T"],
            feed["P"],
        )


def _loop(operation) -> Flowsheet:
    fs = Flowsheet(species_order=["A"], default_flow=0.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", operation, ["feed", "tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


#: A start away from the fixed point, so the iteration has work to do.
GUESS = {"tear": make_stream({"A": 0.0}, 300.0, 1e5)}


def _solve(fs, **kwargs):
    """Solve, ignoring the non-convergence warning the caller is testing for."""
    kwargs.setdefault("acceleration", "none")
    kwargs.setdefault("tear_initial", GUESS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return fs.solve(**kwargs)


class TestDampingIsRead:
    """The regression: it used to make no difference whatsoever."""

    def test_it_changes_the_iteration_count(self):
        counts = {}
        for damping in (1.0, 0.5, 0.3):
            fs = _loop(Contraction())
            _solve(fs, damping=damping, tol=1e-10, max_iter=2000)
            assert fs.last_solve_converged is True
            counts[damping] = fs.last_solve_iterations

        # Strictly more iterations as the step shrinks. Before the fix all
        # three were the same number, which is the whole bug.
        assert counts[1.0] < counts[0.5] < counts[0.3]

    def test_an_overshooting_loop_converges_only_when_damped(self):
        """The point of the knob, on a map plain substitution cannot solve."""
        undamped = _loop(Overshoot())
        _solve(undamped, damping=1.0, tol=1e-10, max_iter=200)
        assert undamped.last_solve_converged is False
        assert undamped.last_solve_residual > 1e30   # diverged, not just slow

        damped = _loop(Overshoot())
        streams = _solve(damped, damping=0.3, tol=1e-10, max_iter=200)
        assert damped.last_solve_converged is True
        assert float(streams["loop_out"]["F_A"]) == pytest.approx(1.0, abs=1e-9)

    def test_zero_and_negative_are_refused(self):
        """At alpha = 0 the iteration is the identity and every point is a
        fixed point, so the solve would report instant success anywhere."""
        for damping in (0.0, -0.3):
            with pytest.raises(ValueError, match="damping must be positive"):
                _loop(Contraction()).solve(damping=damping)

    def test_it_is_refused_before_the_no_recycle_shortcut(self):
        """A flowsheet with no recycles returns early; the value is still
        nonsense and saying so at the call is better than a silent no-op."""
        fs = Flowsheet(species_order=["A"], default_flow=0.0)
        fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
        with pytest.raises(ValueError, match="damping must be positive"):
            fs.solve(damping=0.0)


class TestDefaultIsUndamped:
    """``damping=1.0`` is plain substitution, and is what an existing caller
    who never set the argument was already getting."""

    def test_the_default_is_one(self):
        import inspect

        assert inspect.signature(Flowsheet.solve).parameters["damping"].default == 1.0

    def test_the_default_matches_an_explicit_one(self):
        default = _loop(Contraction())
        _solve(default, tol=1e-10, max_iter=2000)

        explicit = _loop(Contraction())
        _solve(explicit, damping=1.0, tol=1e-10, max_iter=2000)

        assert default.last_solve_iterations == explicit.last_solve_iterations
        assert default.last_solve_residual == explicit.last_solve_residual


class TestTheAnswerDoesNotMoveWithIt:
    """Damping changes how the solve gets there, not where it lands."""

    def test_the_fixed_point_is_the_same(self):
        values = []
        for damping in (1.0, 0.5, 0.3, 0.1):
            fs = _loop(Contraction())
            streams = _solve(fs, damping=damping, tol=1e-12, max_iter=20000)
            assert fs.last_solve_converged is True
            values.append(float(streams["loop_out"]["F_A"]))

        for value in values[1:]:
            assert value == pytest.approx(values[0], abs=1e-9)

    def test_the_gradient_is_the_same(self):
        """Optimistix differentiates the converged solution, so the implicit
        gradient is exact whatever the step fraction was."""
        gradients = []
        for damping in (1.0, 0.3):
            fs = _loop(Contraction())

            def objective(feed_F, fs=fs, damping=damping):
                fs.feeds["feed"] = make_stream({"A": feed_F}, 300.0, 1e5)
                return fs.solve(
                    acceleration="none", damping=damping,
                    tol=1e-12, max_iter=3000,
                )["loop_out"]["F_A"]

            gradients.append(float(jax.grad(objective)(1.0)))

        assert gradients[1] == pytest.approx(gradients[0], rel=1e-9)

        h = 1e-6
        fs = _loop(Contraction())

        def eager(feed_F):
            fs.feeds["feed"] = make_stream({"A": feed_F}, 300.0, 1e5)
            return float(_solve(fs, damping=0.3, tol=1e-12, max_iter=3000)
                         ["loop_out"]["F_A"])

        finite_difference = (eager(1.0 + h) - eager(1.0 - h)) / (2 * h)
        assert gradients[0] == pytest.approx(finite_difference, rel=1e-6)


class TestTheReportedResidualIsUndamped:
    """``last_solve_residual`` is ``|g(x) - x|``, not the ``alpha``-times-
    smaller step the solver actually took -- otherwise a heavily damped solve
    would look converged by exactly the factor it was damped by."""

    @pytest.mark.parametrize("max_iter", [2, 3, 4, 5])
    def test_it_is_the_undamped_residual(self, max_iter):
        # x_k = 1 - 0.1^k for alpha = 0.3 from x_0 = 0, so the undamped
        # residual |3 - 3 x_k| is 3 * 0.1^k and the damped step is 0.9 * 0.1^k.
        fs = _loop(Overshoot())
        _solve(fs, damping=0.3, tol=1e-14, max_iter=max_iter)

        k = fs.last_solve_iterations
        assert fs.last_solve_residual == pytest.approx(3.0 * 0.1 ** k, rel=1e-6)

    def test_tol_still_means_tol(self):
        """The tolerance handed to optimistix carries the damping factor, so a
        damped solve is converged to the tolerance asked for and not to
        ``tol / alpha``."""
        tol = 1e-8
        for damping in (1.0, 0.5, 0.3):
            fs = _loop(Contraction())
            streams = _solve(fs, damping=damping, tol=tol, max_iter=20000)
            assert fs.last_solve_converged is True

            # The criterion the verdict uses, stated on the answer itself.
            x = float(streams["loop_out"]["F_A"])
            assert fs.last_solve_residual <= tol * (1.0 + abs(x))


class TestTheTracedPath:
    """A traced solve falls back to ``acceleration="none"``, so it is the one
    path where damping is not optional."""

    def test_damping_reaches_it(self):
        fs = _loop(Overshoot())

        def objective(feed_F, damping):
            fs.feeds["feed"] = make_stream({"A": feed_F}, 300.0, 1e5)
            return fs.solve(
                damping=damping, tol=1e-10, max_iter=200,
            )["loop_out"]["F_A"]

        damped = jax.jit(lambda F: objective(F, 0.3))(1.0)
        assert fs.last_solve_method == "fixed_point (traced)"
        assert float(damped) == pytest.approx(1.0, abs=1e-9)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            undamped = jax.jit(lambda F: objective(F, 1.0))(1.0)
        assert abs(float(undamped)) > 1e30   # the same solve, undamped
