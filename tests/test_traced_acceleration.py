"""Traced Anderson/Wegstein: a requested acceleration survives jit/grad (#440).

Under ``jax.jit`` / ``jax.jacfwd`` ``Flowsheet.solve`` used to replace the
requested acceleration with plain substitution.  It now runs
:class:`difflow.initialization.AcceleratedFixedPoint` inside optimistix's
fixed point: the same Anderson arithmetic in ``lax.while_loop`` and the
same implicit derivative.  The unnamed default keeps plain substitution
under tracing (``test_flowsheet_damping.py`` pins that).
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.flowsheet import Flowsheet, Unit
from difflow.initialization import AcceleratedFixedPoint
from difflow.streams import make_stream

GUESS = {"tear": make_stream({"A": 0.0}, 300.0, 1e5)}


class Slow:
    """``g(x) = F + k x / (1 + k)``: gain ``k/(1+k)``, ``x* = F (1 + k)``."""

    def __init__(self, k, counter=None):
        self.k = k
        self.counter = counter

    def __call__(self, feed, tear):
        if self.counter is not None:
            jax.debug.callback(lambda: self.counter.append(1))
        return make_stream(
            {"A": feed["F_A"] + self.k * tear["F_A"] / (1.0 + self.k)},
            feed["T"], feed["P"],
        )


class Curved:
    """A nonlinear contraction, ``g(x) = F + 0.9 sqrt(x + 0.1)``."""

    def __call__(self, feed, tear):
        return make_stream(
            {"A": feed["F_A"] + 0.9 * jnp.sqrt(tear["F_A"] + 0.1)},
            feed["T"], feed["P"],
        )


def _loop(op, F=1.0):
    fs = Flowsheet(species_order=["A"], default_flow=0.0)
    fs.add_feed("feed", make_stream({"A": F}, 300.0, 1e5))
    fs.add_unit(Unit("loop", op, ["feed", "tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


def _solve_at(op, F, **kw):
    fs = _loop(op, F)
    kw.setdefault("tol", 1e-12)
    kw.setdefault("tear_initial", GUESS)
    kw.setdefault("error_probe", 0)
    return fs, fs.solve(**kw)["loop_out"]["F_A"]


@pytest.mark.parametrize("acceleration", ["anderson", "wegstein"])
@pytest.mark.parametrize("op", [Slow(9.0), Curved()], ids=["linear", "sqrt"])
def test_traced_equals_eager(op, acceleration):
    _, eager = _solve_at(op, 1.0, acceleration=acceleration)
    jitted = jax.jit(lambda F: _solve_at(op, F, acceleration=acceleration)[1])
    plain = _solve_at(op, 1.0, acceleration="none", max_iter=2000)[1]
    assert float(jitted(1.0)) == pytest.approx(float(eager), rel=1e-9)
    assert float(jitted(1.0)) == pytest.approx(float(plain), rel=1e-9)


def test_fewer_passes_than_plain_substitution():
    """Passes counted where they happen: in the unit, once per evaluation."""
    counts = {}
    for accel in ("none", "anderson"):
        counter = []
        op = Slow(19.0, counter)          # gain 0.95

        def run(F, accel=accel, op=op):
            return _solve_at(op, F, acceleration=accel, max_iter=2000,
                             tol=1e-10)[1]

        out = jax.block_until_ready(jax.jit(run)(1.0))
        jax.effects_barrier()
        assert float(out) == pytest.approx(20.0, rel=1e-8)
        counts[accel] = len(counter)
    assert counts["anderson"] * 10 < counts["none"], counts


def test_acceleration_request_is_not_silently_dropped():
    box = {}

    def run(F, **kw):
        fs, out = _solve_at(Slow(9.0), F, **kw)
        box["fs"] = fs
        return out

    jax.jit(lambda F: run(F, acceleration="anderson"))(1.0)
    assert box["fs"].last_solve_method == "fixed_point (traced)"
    assert box["fs"].last_solve_acceleration == "anderson"
    jax.jit(lambda F: run(F, acceleration="wegstein"))(1.0)
    assert box["fs"].last_solve_acceleration == "wegstein"
    # An explicit "none" is plain substitution, and says so; so does the
    # unnamed default, which keeps its historical traced behaviour.
    jax.jit(lambda F: run(F, acceleration="none"))(1.0)
    assert box["fs"].last_solve_acceleration == "none"
    jax.jit(lambda F: run(F))(1.0)
    assert box["fs"].last_solve_acceleration == "none"
    # Eagerly the default is Anderson, as ever.
    run(1.0)
    assert box["fs"].last_solve_acceleration == "anderson"


@pytest.mark.slow
@pytest.mark.parametrize("acceleration", ["anderson", "wegstein", "none"])
def test_jacobian_matches_the_analytic_and_plain_path(acceleration):
    def f(F):
        return _solve_at(Slow(9.0), F, acceleration=acceleration,
                         max_iter=2000)[1]

    # x* = 10 F, so dx*/dF = 10 on every path: the derivative is the
    # implicit one, not that of the iterations.
    assert float(jax.jacfwd(f)(1.0)) == pytest.approx(10.0, rel=1e-8)
    assert float(jax.jacrev(f)(1.0)) == pytest.approx(10.0, rel=1e-8)
    assert float(jax.grad(f)(1.0)) == pytest.approx(10.0, rel=1e-8)
    assert float(jax.jit(jax.jacfwd(f))(1.0)) == pytest.approx(10.0, rel=1e-8)


def test_jacobian_of_a_nonlinear_loop_matches_finite_differences():
    def f(F):
        return _solve_at(Curved(), F, acceleration="anderson")[1]

    h = 1e-6
    fd = (f(1.0 + h) - f(1.0 - h)) / (2 * h)
    assert float(jax.jacfwd(f)(1.0)) == pytest.approx(float(fd), rel=1e-6)
    plain = jax.jacfwd(
        lambda F: _solve_at(Curved(), F, acceleration="none", max_iter=2000)[1]
    )(1.0)
    assert float(jax.jacfwd(f)(1.0)) == pytest.approx(float(plain), rel=1e-8)


def test_gradient_through_a_closed_over_tracer():
    """The #419 case: ``k`` is in the operation's closure, where the
    detector cannot see it, and the named acceleration still gets through."""
    def objective(k):
        fs = _loop(Slow(k))
        return fs.solve(acceleration="anderson", tol=1e-12,
                        tear_initial=GUESS, error_probe=0)["loop_out"]["F_A"]

    g = jax.grad(objective)(9.0)          # x* = 1 + k, so d/dk = 1
    assert float(g) == pytest.approx(1.0, rel=1e-6)


def test_history_is_fixed_size_and_does_not_recompile():
    solver = AcceleratedFixedPoint(atol=1e-10, depth=4)
    state = solver.init(None, jnp.zeros(7), None, {}, None, None, frozenset())
    assert state.dF.shape == (4, 7) and state.dG.shape == (4, 7)

    traces = []

    def run(F):
        traces.append(1)                  # Python: once per trace
        return _solve_at(Slow(9.0), F, acceleration="anderson")[1]

    jr = jax.jit(run)
    for F in (1.0, 2.0, 3.5):
        assert float(jr(F)) == pytest.approx(10.0 * F, rel=1e-9)
    assert len(traces) == 1


def test_clip_negative_flows_binds_on_the_accelerated_guess_only():
    """The clip projects the extrapolated guess; the fixed point it finds is
    the unclipped map's (here 10 F > 0, so a bound that fired would show)."""
    for clip in (True, False):
        out = jax.jit(lambda F, clip=clip: _solve_at(
            Slow(9.0), F, acceleration="anderson",
            clip_negative_flows=clip)[1])(1.0)
        assert float(out) == pytest.approx(10.0, rel=1e-9)
