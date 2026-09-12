"""How often does difflow's recycle solver actually converge?

Nothing in the project answered that.  :mod:`difflow.benchmarks` compared SM
against EO on *one* flowsheet the caller supplied, and the test suite asserts
that the flowsheets it happens to contain converge --- which is a selection of
flowsheets chosen, understandably, because they converge.  There was no corpus
of deliberately hard ones and no pass-rate number, so "initialization strategy
X helped" was not a claim anyone could check.

This module is that corpus and that number.  It runs every case in
:data:`CORPUS` across every acceleration and every initialization strategy,
and reports how many solves *passed* --- where passing means the solver said
it converged **and** the answer closes its material balance.  The two are not
the same thing, and keeping them apart is most of the point: a tear iteration
that stalls inside its tolerance, or one whose negative-flow clipping invents
a fixed point that is not a solution, reports success either way.
:attr:`Outcome.converged` is the solver's verdict and :attr:`Outcome.correct`
is the audit of it.

Why it exists now
-----------------
It is the measurement that #247 (tear initialization never fires) asked for
and did not get: that fix changed every recycle's starting point, and whether
it changed any *outcome* is an empirical question.  Run the benchmark with
``initializations=("default", "unit")`` and the two columns are the before
and after.

It is also the gate on #251 (a piecewise-linear MILP initializer).  A MILP
initial guess is a large piece of machinery to justify, and it can only be
justified against a failure rate someone has measured.  If the cheap fixes
already close the gap, the number says so and the MILP is not needed.

Usage
-----
::

    from difflow.convergence import run_benchmark

    report = run_benchmark()
    print(report.as_text())
    print(report.pass_rate)

Narrow it while iterating --- the full grid is a few hundred solves and each
one pays JAX compilation::

    report = run_benchmark(cases=["high_gain_recycle"],
                           accelerations=("anderson",))

The corpus
----------
Each :class:`Case` says in its own ``difficulty`` field what makes it hard,
because a benchmark whose cases are hard for unstated reasons measures
something nobody can act on.  The eleven cases cover the families #251 names
--- high loop gain, sharp splits, multi-loop flowsheets, phase-regime
switching, near-pinch columns --- plus a trace-magnitude tear, a signed tear,
and two linear maps whose answers are known exactly and so can distinguish
"the solver was wrong" from "the case is hard".

Two of them are controls, meant to pass under every setting: an ordinary
two-phase flash recycle, and a rigorous twelve-stage MESH column with 95% of
its distillate returned.  A corpus where everything fails cannot tell a hard
corpus from a broken solver.

What it measured, and what that is for
--------------------------------------
**Every case is solved by at least one acceleration.**  Across eleven cases
and three methods there is no flowsheet here that all three fail, which is
the benchmark's main finding and a negative one.  It is the fact #251 turns
on: a globally consistent MILP initial guess is aimed at failures that this
corpus does not contain.  Extending the corpus until it does --- or failing
to, deliberately and on the record --- is how that issue gets decided.
"""

from __future__ import annotations

import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

import jax.numpy as jnp

from difflow.flowsheet import (
    ConvergenceWarning,
    Flowsheet,
    TearToleranceWarning,
    Unit,
)
from difflow.initialization import TearInitializationWarning
from difflow.streams import Stream, get_flows, make_stream

__all__ = [
    "Case",
    "Outcome",
    "Report",
    "CORPUS",
    "ACCELERATIONS",
    "INITIALIZATIONS",
    "get_case",
    "cases",
    "run_case",
    "run_benchmark",
    "total_mole_balance",
]


#: The acceleration settings :meth:`Flowsheet.solve` offers.
ACCELERATIONS: tuple[str, ...] = ("none", "wegstein", "anderson")

#: The initialization strategies compared.  ``"default"`` is what every
#: recycle in difflow got before #247 --- ``default_flow`` (0.01 mol/s) in
#: every species.  ``"unit"`` is the propagation pass that #247 wired up.
#: ``"feed"`` is the guess a user writes by hand when the default fails:
#: the combined feed, handed to every tear.
INITIALIZATIONS: tuple[str, ...] = ("default", "unit", "feed")


# =============================================================================
# Balance auditing
# =============================================================================

def _product_streams(fs: Flowsheet) -> list[str]:
    """Streams that leave the flowsheet: produced, consumed by nothing.

    A recycle source is produced and then consumed at the other end of the
    tear, so it is not a product even though no ``inlet_names`` mentions it.
    """
    produced: set[str] = set()
    consumed: set[str] = set()
    for unit in fs.units:
        produced.update(unit.outlet_names)
        consumed.update(unit.inlet_names)
    return sorted(produced - consumed - set(fs.recycles))


def total_mole_balance(fs: Flowsheet, streams: dict[str, Stream]) -> float:
    """Relative error in the overall mole balance, feeds against products.

    The default audit.  It is valid for every case in :data:`CORPUS`: none
    of them change the total mole count, either because nothing reacts or
    because the reaction is ``A -> B``.  A case with a mole-changing
    reaction must supply its own ``check``.

    Args:
        fs: The flowsheet that was solved.
        streams: The streams :meth:`Flowsheet.solve` returned.

    Returns:
        ``|in - out| / max(in, 1)``.  Non-finite values (a diverged solve
        that ran to ``inf``) come back as ``inf`` rather than raising, so a
        failing case still reports a number.
    """
    total_in = sum(float(jnp.sum(jnp.asarray(list(get_flows(s).values()))))
                   for s in fs.feeds.values())
    total_out = 0.0
    for name in _product_streams(fs):
        if name not in streams:
            continue
        total_out += float(
            jnp.sum(jnp.asarray(list(get_flows(streams[name]).values()))))
    err = abs(total_in - total_out) / max(abs(total_in), 1.0)
    return err if jnp.isfinite(jnp.asarray(err)) else float("inf")


# =============================================================================
# Cases
# =============================================================================

@dataclass(frozen=True)
class Case:
    """One deliberately hard flowsheet.

    Attributes:
        name: Identifier, unique within :data:`CORPUS`.
        build: Returns a *fresh* :class:`~difflow.flowsheet.Flowsheet`.  It
            must build a new one on every call: :meth:`Flowsheet.solve`
            records state on the object, and a case reused across settings
            would carry one run's verdict into the next.
        difficulty: What makes this one hard, in a sentence.  Required ---
            an unexplained hard case measures something nobody can act on.
        tags: Free-form labels for filtering (``"recycle"``, ``"flash"``).
        check: Audits a solution, returning a relative error.  Defaults to
            :func:`total_mole_balance`.
        tol: Balance error above which the answer is called wrong, however
            confidently the solver reported convergence.
    """

    name: str
    build: Callable[[], Flowsheet]
    difficulty: str
    tags: tuple[str, ...] = ()
    check: Callable[[Flowsheet, dict[str, Stream]], float] = total_mole_balance
    tol: float = 1e-4


# -- shared pieces ------------------------------------------------------------

def _mixer(s1: Stream, s2: Stream) -> Stream:
    """Adiabatic-ish mixer: flows add, T is flow-averaged."""
    f1, f2 = get_flows(s1), get_flows(s2)
    n1 = sum(f1.values())
    n2 = sum(f2.values())
    total = n1 + n2
    # jnp.where rather than a Python branch: total is traced under grad.
    w = jnp.where(total > 0, n1 / jnp.where(total > 0, total, 1.0), 0.5)
    return make_stream({k: f1[k] + f2[k] for k in f1},
                       w * s1["T"] + (1 - w) * s2["T"], s1["P"])


def _splitter(recycle_frac: float) -> Callable:
    """Split into ``(recycle, product)``."""
    def split(inlet: Stream):
        flows = get_flows(inlet)
        rec = make_stream({k: v * recycle_frac for k, v in flows.items()},
                          inlet["T"], inlet["P"])
        prod = make_stream({k: v * (1.0 - recycle_frac) for k, v in flows.items()},
                           inlet["T"], inlet["P"])
        return rec, prod
    return split


def _first_order_cstr(k: float, species: Sequence[str]):
    """``A -> B``, ``r = k C_A``, isothermal."""
    from difflow.database import get_species_data
    from difflow.thermo import IdealThermo
    from difflow.units.cstr import CSTR, CSTRParams

    thermo = IdealThermo({
        species[0]: get_species_data("n_hexane")._replace(name=species[0]),
        species[1]: get_species_data("n_heptane")._replace(name=species[1]),
    })

    def rate_fn(C, T, params):
        return jnp.array([params["k"] * C[species[0]]])

    params = CSTRParams(
        V=jnp.array(1.0),
        rate_fn=rate_fn,
        stoich=jnp.array([[-1.0], [+1.0]]),
        rate_params={"k": jnp.array(k)},
        species_order=list(species),
        molar_density=55500.0,
    )
    return CSTR(params, thermo=thermo, mode="isothermal")


def _ideal_thermo(species: Sequence[str]):
    """Raoult K-values over ``species``, from the shipped database."""
    from difflow.database import get_species_data
    from difflow.thermo import IdealThermo

    return IdealThermo({s: get_species_data(s) for s in species})


def _flash(species: Sequence[str]):
    """An ideal-thermo flash over ``species``, from the shipped database."""
    from difflow.units.flash import Flash, FlashParams

    return Flash(FlashParams(species_order=list(species)),
                 thermo=_ideal_thermo(species))


def _passthrough(stream: Stream) -> Stream:
    return make_stream(get_flows(stream), stream["T"], stream["P"])


def _cooler(dT: float) -> Callable:
    def cool(stream: Stream) -> Stream:
        return make_stream(get_flows(stream), stream["T"] - dT, stream["P"])
    return cool


# -- the corpus ---------------------------------------------------------------

_PAIR = ("n_hexane", "n_heptane")


def _build_overshoot() -> Flowsheet:
    """``g(x) = 6F - 2x``: one tear, eigenvalue -2, fixed point ``x = 2F``.

    The fixed point is deliberately *not* the feed.  With ``3F - 2x`` it
    would be ``x = F`` exactly, and the ``"feed"`` initialization would
    start on the answer and score a win it had not earned.
    """
    class Overshoot:
        def __call__(self, feed: Stream, tear: Stream) -> Stream:
            return make_stream({"A": 6.0 * feed["F_A"] - 2.0 * tear["F_A"]},
                               feed["T"], feed["P"])

    fs = Flowsheet(["A"], default_flow=0.01)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 101325.0))
    fs.add_unit(Unit("loop", Overshoot(), ["feed", "tear"], ["loop_out"]))
    fs.add_unit(Unit("out", _passthrough, ["loop_out"], ["product"]))
    fs.add_recycle("loop_out", "tear")
    return fs


def _overshoot_check(fs: Flowsheet, streams: dict[str, Stream]) -> float:
    """The exact answer is known: ``x = 2F = 2``.

    The mole balance cannot audit this one --- the map is not a material
    balance --- so the case brings its own check, which is what
    ``Case.check`` is for.
    """
    if "product" not in streams:
        return float("inf")
    got = float(streams["product"]["F_A"])
    return (abs(got - 2.0) / 2.0
            if jnp.isfinite(jnp.asarray(got)) else float("inf"))


def _build_high_gain() -> Flowsheet:
    """Mixer -> CSTR (2% conversion) -> 99% recycle split."""
    species = ["A", "B"]
    fs = Flowsheet(species, default_flow=0.01)
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, 350.0, 101325.0))
    fs.add_unit(Unit("mix", _mixer, ["feed", "rec"], ["mixed"]))
    fs.add_unit(Unit("cstr", _first_order_cstr(0.02, species), ["mixed"], ["rout"]))
    fs.add_unit(Unit("split", _splitter(0.99), ["rout"], ["rec_src", "product"]))
    fs.add_recycle("rec_src", "rec")
    return fs


def _build_flash_recycle() -> Flowsheet:
    """Flash below the bubble point, 99% of the liquid recycled.

    Subcooled, so the flash passes essentially everything to the liquid and
    the loop gain is the split fraction itself.  Deliberately the same gain
    as ``high_gain_recycle``, through a real unit rather than an analytic
    map: the two together separate "the solver cannot do 0.99" from
    "the flash is what broke".
    """
    flash = _flash(_PAIR)
    T, P = 340.0, 101325.0

    fs = Flowsheet(list(_PAIR), default_flow=0.01)
    fs.add_feed("feed", make_stream({_PAIR[0]: 1.0, _PAIR[1]: 1.0}, T, P))
    fs.add_unit(Unit("mix", _mixer, ["feed", "rec"], ["mixed"]))
    fs.add_unit(Unit("flash", lambda s: flash(s, T=T, P=P), ["mixed"],
                     ["liq", "vap"]))
    fs.add_unit(Unit("split", _splitter(0.99), ["liq"], ["rec_src", "bottoms"]))
    fs.add_recycle("rec_src", "rec")
    return fs


def _build_phase_coupled_flash() -> Flowsheet:
    """The flash picks the temperature that decides its own phase split.

    The flash runs at its inlet temperature, and its inlet is the feed mixed
    with cooled recycled liquid.  How much liquid comes back therefore sets
    the mix temperature, which sets the phase split, which sets how much
    liquid comes back.  Tuned to sit just inside the two-phase window, where
    that loop has two regimes to choose between.

    This is the case that catches Anderson.  At these settings Wegstein
    converges in under 30 iterations to a recycle of about 18 mol/s;
    Anderson stops near 0.4 and reports non-convergence, having gone
    somewhere else rather than run out of road.
    """
    flash = _flash(_PAIR)
    P = 101325.0

    fs = Flowsheet(list(_PAIR), default_flow=0.01, default_T=300.0)
    fs.add_feed("feed", make_stream({_PAIR[0]: 1.0, _PAIR[1]: 1.0}, 358.0, P))
    fs.add_unit(Unit("mix", _mixer, ["feed", "rec"], ["mixed"]))
    fs.add_unit(Unit("flash", lambda s: flash(s, T=s["T"], P=P), ["mixed"],
                     ["liq", "vap"]))
    fs.add_unit(Unit("cool", _cooler(5.0), ["liq"], ["cooled"]))
    fs.add_unit(Unit("split", _splitter(0.9), ["cooled"], ["rec_src", "bottoms"]))
    fs.add_recycle("rec_src", "rec")
    return fs


def _build_two_phase_flash() -> Flowsheet:
    """The control: a flash recycle inside the two-phase window.

    A benchmark made only of cases that fail cannot tell a hard corpus from
    a broken solver.  This one is an ordinary flash recycle with a real
    vapour purge, and every setting should pass it.
    """
    flash = _flash(_PAIR)
    T, P = 360.0, 101325.0

    fs = Flowsheet(list(_PAIR), default_flow=0.01)
    fs.add_feed("feed", make_stream({_PAIR[0]: 1.0, _PAIR[1]: 1.0}, T, P))
    fs.add_unit(Unit("mix", _mixer, ["feed", "rec"], ["mixed"]))
    fs.add_unit(Unit("flash", lambda s: flash(s, T=T, P=P), ["mixed"],
                     ["liq", "vap"]))
    fs.add_unit(Unit("split", _splitter(0.9), ["liq"], ["rec_src", "bottoms"]))
    fs.add_recycle("rec_src", "rec")
    return fs


def _build_trace_recycle() -> Flowsheet:
    """Loop gain 0.97 on a tear whose true magnitude is about 1e-6.

    ``default_flow`` is 0.01, so the default guess starts four orders of
    magnitude above the answer.  Here to separate the two things a starting
    point can be wrong about: its *magnitude* and its *direction*.
    """
    def loop(feed: Stream, tear: Stream) -> Stream:
        f, t = get_flows(feed), get_flows(tear)
        return make_stream({"A": f["A"] * 1e-6 + t["A"] * 0.97},
                           feed["T"], feed["P"])

    fs = Flowsheet(["A"], default_flow=0.01)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 101325.0))
    fs.add_unit(Unit("loop", loop, ["feed", "tear"], ["loop_out"]))
    fs.add_unit(Unit("out", _passthrough, ["loop_out"], ["product"]))
    fs.add_recycle("loop_out", "tear")
    return fs


def _trace_check(fs: Flowsheet, streams: dict[str, Stream]) -> float:
    """``x = 1e-6 F / (1 - 0.97)``, again exactly known."""
    if "product" not in streams:
        return float("inf")
    got = float(streams["product"]["F_A"])
    want = 1e-6 / (1.0 - 0.97)
    return (abs(got - want) / want
            if jnp.isfinite(jnp.asarray(got)) else float("inf"))


def _build_two_loop() -> Flowsheet:
    """Two recycles around one mixer, nested: the inner split feeds the outer.

    ``s_in`` returns 42.5% of the mixer outlet directly; ``s_out`` returns
    90% of what is left, another 51.8%.  The combined gain is 0.943 and the
    mixer carries about 17 times the feed.

    The corpus was otherwise all single-tear.  Two tears is the case #251
    calls "globally consistent around the loop" -- substitution has to
    settle both at once, and the accelerated methods see one combined tear
    vector rather than two loops.
    """
    def merge(feed: Stream, r_in: Stream, r_out: Stream) -> Stream:
        f, a, b = get_flows(feed), get_flows(r_in), get_flows(r_out)
        return make_stream({"A": f["A"] + a["A"] + b["A"]},
                           feed["T"], feed["P"])

    fs = Flowsheet(["A"], default_flow=0.01)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 101325.0))
    fs.add_unit(Unit("mix", merge, ["feed", "r_in", "r_out"], ["mixed"]))
    fs.add_unit(Unit("s_in", _splitter(0.425), ["mixed"], ["r_in_src", "mid"]))
    fs.add_unit(Unit("s_out", _splitter(0.9), ["mid"], ["r_out_src", "product"]))
    fs.add_recycle("r_in_src", "r_in")
    fs.add_recycle("r_out_src", "r_out")
    return fs


def _build_sharp_split() -> Flowsheet:
    """A 0.1% purge: 999 parts recycled for every one that leaves."""
    def combine(feed: Stream, tear: Stream) -> Stream:
        f, t = get_flows(feed), get_flows(tear)
        return make_stream({k: f[k] + t[k] for k in f}, feed["T"], feed["P"])

    fs = Flowsheet(["A"], default_flow=0.01)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 101325.0))
    fs.add_unit(Unit("mix", combine, ["feed", "tear"], ["mixed"]))
    fs.add_unit(Unit("split", _splitter(0.999), ["mixed"], ["rec_src", "purge"]))
    fs.add_recycle("rec_src", "tear")
    return fs


#: Coupling matrix for :func:`_build_signed_tear`, spectral radius 0.75.
#: Written out rather than reseeded so the case does not depend on NumPy's
#: random stream staying put across versions.
_SIGNED_M = (
    (+0.060542, -0.063612, +0.308380, +0.050512, -0.257938, +0.174117),
    (+0.627909, +0.456043, -0.338866, -0.609332, -0.300122, +0.019900),
    (-1.119561, -0.105354, -0.599938, -0.352605, -0.262074, -0.152307),
    (+0.198211, +0.501997, -0.061893, +0.657987, -0.320308, +0.169261),
    (+0.435044, +0.045269, -0.358014, -0.443834, -0.220407, +0.106030),
    (-0.486157, -0.100723, -0.076671, +0.260431, +0.103364, +0.171121),
)

#: ``(I - M)^-1 . 1``, the exact answer.  Three components are negative.
#:
#: In full float64, not rounded.  The six-decimal version this started as
#: was itself 4.4e-6 away from the answer, which is a hundred times what a
#: converged solve of this case leaves behind -- so ``_signed_check`` was
#: measuring the constant rather than the solver, and the case's loose
#: ``tol`` was set to accommodate that (#264).
_SIGNED_ANSWER = (
    1.1516238190546417,
    -0.4376153446104903,
    -1.2302575893785253,
    3.785692361299334,
    0.3657687695442713,
    1.9330348572361187,
)

_SIGNED_SPECIES = tuple(f"S{i}" for i in range(6))


def _build_signed_tear() -> Flowsheet:
    """A tear that legitimately carries a signed quantity.

    ``x <- f + M x`` with the spectral radius at 0.75, and three components
    of the answer negative.  Not a contrivance: ``difflow_gas`` works in
    signed flows throughout, because a pipe's direction is an unknown, and
    its docs already say to solve with ``clip_negative_flows=False``.

    The corpus needs it because it is the only case here that inverts the
    usual ranking.  ``clip_negative_flows`` defaults to ``True`` and is
    applied by the Wegstein and Anderson paths but **not** by the
    unaccelerated one, so on this map the two accelerated methods have a
    projection applied to their iterates that the plain one does not --
    and they are the two that fail.
    """
    species = list(_SIGNED_SPECIES)
    M = jnp.asarray(_SIGNED_M)

    def loop(feed: Stream, tear: Stream) -> Stream:
        x = jnp.array([tear[f"F_{s}"] for s in species])
        f = jnp.array([feed[f"F_{s}"] for s in species])
        y = f + M @ x
        return make_stream({s: y[i] for i, s in enumerate(species)},
                           feed["T"], feed["P"])

    fs = Flowsheet(species, default_flow=0.01)
    fs.add_feed("feed", make_stream({s: 1.0 for s in species}, 300.0, 101325.0))
    fs.add_unit(Unit("loop", loop, ["feed", "tear"], ["loop_out"]))
    fs.add_unit(Unit("out", _passthrough, ["loop_out"], ["product"]))
    fs.add_recycle("loop_out", "tear")
    return fs


def _signed_check(fs: Flowsheet, streams: dict[str, Stream]) -> float:
    """Against ``(I - M)^-1 f``, which is known in closed form.

    The mole balance cannot audit a signed tear -- the map is not a
    material balance and the "flows" are not moles.
    """
    if "product" not in streams:
        return float("inf")
    scale = max(abs(v) for v in _SIGNED_ANSWER)
    worst = 0.0
    for species, want in zip(_SIGNED_SPECIES, _SIGNED_ANSWER):
        got = float(streams["product"][f"F_{species}"])
        if not jnp.isfinite(jnp.asarray(got)):
            return float("inf")
        worst = max(worst, abs(got - want))
    return worst / scale


def _build_regime_switch() -> Flowsheet:
    """The solution sits exactly on a switch between two linear regimes.

    Below the threshold the loop is strongly expanding, above it strongly
    contracting, and the two branches meet at the fixed point.  Plain
    substitution can only chatter across the join.

    This is the disjunction #251 proposes to hand to a MILP's binaries --
    a unit picking a branch, with the answer on the boundary.  It is in the
    corpus to find out whether difflow needs binaries to solve one.
    """
    threshold = 2.0

    def loop(feed: Stream, tear: Stream) -> Stream:
        x = tear["F_A"]
        below = 4.0 * x - 6.0     # expanding;  g(2) = 2
        above = 0.2 * x + 1.6     # contracting; g(2) = 2
        return make_stream({"A": jnp.where(x < threshold, below, above)},
                           feed["T"], feed["P"])

    fs = Flowsheet(["A"], default_flow=0.01)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 101325.0))
    fs.add_unit(Unit("loop", loop, ["feed", "tear"], ["loop_out"]))
    fs.add_unit(Unit("out", _passthrough, ["loop_out"], ["product"]))
    fs.add_recycle("loop_out", "tear")
    return fs


def _regime_check(fs: Flowsheet, streams: dict[str, Stream]) -> float:
    """The switch point itself, ``x = 2``."""
    if "product" not in streams:
        return float("inf")
    got = float(streams["product"]["F_A"])
    return (abs(got - 2.0) / 2.0
            if jnp.isfinite(jnp.asarray(got)) else float("inf"))


def _build_column_recycle() -> Flowsheet:
    """A rigorous MESH column with 95% of its distillate recycled.

    #251 names near-pinch columns as a hard family, so the corpus should
    contain one and say what happens.  A real
    :class:`~difflow.units.distillation.DistillationColumn` -- twelve
    stages, MESH after a CMO warm start -- inside a high-recycle loop.

    The heaviest unit here by a wide margin, and it costs a JAX compile.
    """
    from difflow.units.distillation import (
        DistillationColumn, DistillationColumnParams)

    species = list(_PAIR)
    params = DistillationColumnParams(species_order=species, n_stages=12,
                                      feed_stage=6, P=101325.0, q=1.0)
    column = DistillationColumn(params, _ideal_thermo(_PAIR))

    def column_op(feed: Stream):
        total = sum(get_flows(feed).values())
        distillate, bottoms, _ = column(feed, R=1.5, D_spec=0.5 * total)
        return distillate, bottoms

    fs = Flowsheet(species, default_flow=0.01)
    fs.add_feed("feed", make_stream({species[0]: 1.0, species[1]: 1.0},
                                    360.0, 101325.0))
    fs.add_unit(Unit("mix", _mixer, ["feed", "rec"], ["mixed"]))
    fs.add_unit(Unit("col", column_op, ["mixed"], ["dist", "bot"]))
    fs.add_unit(Unit("split", _splitter(0.95), ["dist"], ["rec_src", "product"]))
    fs.add_recycle("rec_src", "rec")
    return fs


#: The corpus.  Ordered easiest-to-read, not easiest-to-solve.
CORPUS: tuple[Case, ...] = (
    Case(
        name="overshoot_loop",
        build=_build_overshoot,
        difficulty=(
            "Linear tear map with eigenvalue -2: plain substitution triples "
            "the error and flips its sign every step, so it diverges from "
            "any start but the answer. Linear on purpose -- the fixed point "
            "is exactly 2.0, so a wrong answer is unambiguous."),
        tags=("recycle", "divergent", "analytic"),
        check=_overshoot_check,
        tol=1e-6,
    ),
    Case(
        name="high_gain_recycle",
        build=_build_high_gain,
        difficulty=(
            "CSTR at about 2% per-pass conversion with 99% of the effluent "
            "recycled: loop gain just under one, so the error decays by "
            "about 1% per substitution."),
        tags=("recycle", "reactor", "high-gain"),
    ),
    Case(
        name="flash_recycle",
        build=_build_flash_recycle,
        difficulty=(
            "Flash held well below its bubble point (sum(z K) = 0.65 at "
            "340 K), so nearly everything leaves as liquid and 99% of that "
            "is recycled. The purge is the only thing bounding the loop."),
        tags=("recycle", "flash", "high-gain"),
    ),
    Case(
        name="phase_coupled_flash",
        build=_build_phase_coupled_flash,
        difficulty=(
            "The flash runs at its own inlet temperature, which the cooled "
            "recycle sets, so the loop chooses its phase regime as it goes. "
            "Only Wegstein solves it: Anderson walks off somewhere else "
            "entirely and plain substitution runs out of iterations. The "
            "case that stops 'anderson' being a free win."),
        tags=("recycle", "flash", "phase-flip"),
    ),
    Case(
        name="two_phase_flash",
        build=_build_two_phase_flash,
        difficulty=(
            "Not hard: an ordinary flash recycle inside the two-phase "
            "window with a real vapour purge. The control -- if this one "
            "fails, the solver is broken, not the case."),
        tags=("recycle", "flash", "control"),
    ),
    Case(
        name="trace_recycle",
        build=_build_trace_recycle,
        difficulty=(
            "Loop gain 0.97 on a tear whose converged value is about 3e-5, "
            "against a default guess of 0.01 -- the starting point is "
            "wrong by three orders of magnitude in scale alone."),
        tags=("recycle", "trace", "high-gain", "analytic"),
        check=_trace_check,
        tol=1e-4,
    ),
    Case(
        name="two_loop_recycle",
        build=_build_two_loop,
        difficulty=(
            "Two recycles around one mixer, returning 0.425 and 0.518 of "
            "its outlet for a combined loop gain of 0.943, so the solver "
            "has to settle two tears at once rather than one. The only "
            "multi-tear case in the corpus."),
        tags=("recycle", "multi-loop", "high-gain"),
    ),
    Case(
        name="sharp_split_purge",
        build=_build_sharp_split,
        difficulty=(
            "A 0.1% purge: 999 parts recycled for every one leaving, so the "
            "loop gain is 0.999 and the converged recycle is a thousand "
            "times the feed. The sharp-split family, pushed a decade past "
            "high_gain_recycle."),
        tags=("recycle", "sharp-split", "high-gain"),
    ),
    Case(
        name="signed_tear",
        build=_build_signed_tear,
        difficulty=(
            "A tear carrying a signed quantity, as difflow_gas does: "
            "x <- f + M x at spectral radius 0.75, with three of the six "
            "components of the answer negative. The one case here that "
            "inverts the ranking -- clip_negative_flows defaults to True "
            "and is applied by Wegstein and Anderson but not by plain "
            "substitution, and it is the two accelerated methods that "
            "fail. Passing clip_negative_flows=False fixes Anderson."),
        tags=("recycle", "signed", "clipping", "analytic"),
        check=_signed_check,
        # 1e-6, the same as the other analytic cases.  This was 1e-4, to
        # accommodate a 1.2e-6 error attributed to M being non-normal --
        # ||(I - M)^-1|| amplifying the tear tolerance rather than
        # 1/(1 - rho).  It was not that: the reference above was rounded to
        # six decimals and the case was measuring the rounding (#264).
        # Plain substitution lands 7.6e-9 from the answer, which is 0.57
        # times its own tear residual, not 450 times it.
        tol=1e-6,
    ),
    Case(
        name="regime_switch",
        build=_build_regime_switch,
        difficulty=(
            "Two linear branches meeting exactly at the fixed point, "
            "expanding below the switch and contracting above it, so "
            "substitution chatters across the join forever. This is the "
            "disjunction #251 proposes handing to a MILP's binaries, and "
            "it is here to find out whether difflow needs them."),
        tags=("recycle", "phase-flip", "nonsmooth", "analytic"),
        check=_regime_check,
        tol=1e-6,
    ),
    Case(
        name="column_recycle",
        build=_build_column_recycle,
        difficulty=(
            "A rigorous twelve-stage MESH column with 95% of its "
            "distillate recycled -- the near-pinch column family #251 "
            "names. Not hard, as it turns out: every setting solves it. "
            "The heaviest unit in the corpus by a wide margin."),
        tags=("recycle", "column", "control"),
    ),
)


def get_case(name: str) -> Case:
    """Look a case up by name."""
    for case in CORPUS:
        if case.name == name:
            return case
    raise KeyError(
        f"unknown case {name!r}; the corpus is "
        f"{', '.join(c.name for c in CORPUS)}")


def cases(names: Iterable[str] | None = None,
          tags: Iterable[str] | None = None) -> list[Case]:
    """Select cases by name or tag.

    Args:
        names: Case names to keep.  ``None`` keeps all.
        tags: Keep only cases carrying at least one of these tags.

    Returns:
        The selected cases, in corpus order.
    """
    selected = list(CORPUS) if names is None else [get_case(n) for n in names]
    if tags is not None:
        wanted = set(tags)
        selected = [c for c in selected if wanted & set(c.tags)]
    return selected


# =============================================================================
# Running
# =============================================================================

def _feed_guess(fs: Flowsheet) -> dict[str, Stream]:
    """The combined feed, offered to every tear.

    The guess a user writes by hand when the default fails.  It is usually
    too large by the recycle ratio, which is the point: it probes whether
    being wrong in *scale* matters as much as being wrong in kind.
    """
    flows: dict[str, float] = {s: 0.0 for s in fs.species_order}
    T = fs.default_T
    P = fs.default_P
    for stream in fs.feeds.values():
        for species, value in get_flows(stream).items():
            if species in flows:
                flows[species] = flows[species] + value
        T, P = stream["T"], stream["P"]
    combined = make_stream(flows, T, P)
    return {dest: combined for dest in fs.recycles.values()}


def _solve_kwargs(fs: Flowsheet, initialization: str) -> dict:
    """Translate an initialization strategy into ``solve`` arguments."""
    if initialization == "default":
        return {"use_initialization": False}
    if initialization == "unit":
        return {"use_initialization": True}
    if initialization == "feed":
        return {"use_initialization": False, "tear_initial": _feed_guess(fs)}
    raise ValueError(
        f"unknown initialization {initialization!r}; expected one of "
        f"{', '.join(INITIALIZATIONS)}")


@dataclass
class Outcome:
    """The result of one solve, at one setting.

    Attributes:
        case: Case name.
        acceleration: The ``acceleration`` passed to :meth:`Flowsheet.solve`.
        initialization: The strategy name from :data:`INITIALIZATIONS`.
        converged: The solver's own verdict
            (``Flowsheet.last_solve_converged``).  ``None`` if it raised.
        correct: Whether the case's ``check`` came back within its ``tol``.
            Kept apart from ``converged`` on purpose: a solve can report
            success and land somewhere that does not close its balance, and
            that is the failure mode worth knowing about.
        balance_error: What the ``check`` returned.
        iterations: Tear iterations used.
        residual: Final tear residual.
        method: What actually ran --- ``last_solve_method``, which is not
            always the requested acceleration (a traced solve falls back).
        gain: Measured signed gain of the tear map at the solution
            (``last_solve_gain``), or ``None`` where there was nothing to
            measure.
        error_estimate: The error the step residual hides
            (``last_solve_error_estimate``).  Recorded because it is the
            benchmark's independent check on the ``WRONG`` verdicts: a
            solve that reports success and does not audit should be one the
            solver could have known about, and on ``trace_recycle`` under
            Wegstein this reads 3.0e-7 against a residual of 9.3e-9 (#264).
        wall_time: Seconds, including JAX compilation on first touch, so
            useful for ranking rather than as an absolute cost.
        error: ``"TypeName: message"`` if the solve raised, else ``None``.
        traceback: Full traceback when it raised, for debugging a case.
    """

    case: str
    acceleration: str
    initialization: str
    converged: bool | None = None
    correct: bool = False
    balance_error: float = float("inf")
    iterations: int | None = None
    residual: float | None = None
    method: str | None = None
    gain: float | None = None
    error_estimate: float | None = None
    wall_time: float = 0.0
    error: str | None = None
    traceback: str | None = None

    @property
    def passed(self) -> bool:
        """Converged *and* landed on an answer that audits."""
        return bool(self.converged) and self.correct

    @property
    def verdict(self) -> str:
        """One word for the report table."""
        if self.error is not None:
            return "raised"
        if not self.converged:
            return "no-conv"
        return "pass" if self.correct else "WRONG"


def run_case(case: Case, acceleration: str = "anderson",
             initialization: str = "unit", tol: float = 1e-8,
             max_iter: int = 100, **solve_kwargs) -> Outcome:
    """Solve one case at one setting and audit the answer.

    Neither a non-convergence warning nor an exception propagates: both are
    outcomes to be counted, not errors to stop on.  A benchmark that dies on
    its hardest case reports the pass rate of the cases before it.

    Args:
        case: The case to run.
        acceleration: Passed to :meth:`Flowsheet.solve`.
        initialization: One of :data:`INITIALIZATIONS`.
        tol: Tear tolerance.
        max_iter: Iteration cap.
        **solve_kwargs: Anything else for :meth:`Flowsheet.solve`.

    Returns:
        An :class:`Outcome`.
    """
    fs = case.build()
    kwargs = dict(_solve_kwargs(fs, initialization))
    kwargs.update(solve_kwargs)
    outcome = Outcome(case=case.name, acceleration=acceleration,
                      initialization=initialization)

    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            # Non-convergence is counted, not announced: this reads
            # last_solve_converged directly, so on_nonconvergence="ignore"
            # is the verdict being taken rather than thrown away.
            # TearInitializationWarning fires on nearly every case here by
            # design -- the propagation pass cannot run a unit whose inlet
            # is still unknown -- and 54 copies of it say nothing.
            #
            # Deliberately narrow: a blanket UserWarning filter would also
            # swallow the modelling warnings (a defaulted CSTR density, a
            # defaulted Cp) that mean a newly added case is wrong rather
            # than hard.
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", TearInitializationWarning)
            # Same reasoning for the tolerance warning: the number behind
            # it is recorded in Outcome.error_estimate, which is the point
            # of running the probe here at all.
            warnings.simplefilter("ignore", TearToleranceWarning)
            streams = fs.solve(acceleration=acceleration, tol=tol,
                               max_iter=max_iter,
                               on_nonconvergence="ignore", **kwargs)
    except Exception as exc:  # noqa: BLE001 - a raise is a result here
        outcome.wall_time = time.perf_counter() - start
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.traceback = traceback.format_exc()
        return outcome

    outcome.wall_time = time.perf_counter() - start
    outcome.converged = fs.last_solve_converged
    outcome.iterations = fs.last_solve_iterations
    outcome.residual = fs.last_solve_residual
    outcome.method = fs.last_solve_method
    outcome.gain = fs.last_solve_gain
    outcome.error_estimate = fs.last_solve_error_estimate

    try:
        outcome.balance_error = float(case.check(fs, streams))
    except Exception as exc:  # noqa: BLE001 - an unauditable answer is wrong
        outcome.balance_error = float("inf")
        outcome.error = f"check raised: {type(exc).__name__}: {exc}"
    outcome.correct = outcome.balance_error <= case.tol
    return outcome


@dataclass
class Report:
    """Every outcome of one benchmark run, and the rates over them.

    Attributes:
        outcomes: One per (case, acceleration, initialization).
        tol: The tear tolerance every solve was asked for.
        max_iter: The iteration cap every solve was given.
    """

    outcomes: list[Outcome] = field(default_factory=list)
    tol: float = 1e-8
    max_iter: int = 100

    @property
    def pass_rate(self) -> float:
        """Fraction of solves that converged and audited."""
        if not self.outcomes:
            return float("nan")
        return sum(o.passed for o in self.outcomes) / len(self.outcomes)

    @property
    def convergence_rate(self) -> float:
        """Fraction the *solver* called converged, audit aside.

        The gap between this and :attr:`pass_rate` is the answer to "how
        often does a solve claim success and mean it?"
        """
        if not self.outcomes:
            return float("nan")
        return sum(bool(o.converged) for o in self.outcomes) / len(self.outcomes)

    def by(self, field_name: Literal["case", "acceleration", "initialization"]
           ) -> dict[str, tuple[int, int]]:
        """Pass counts grouped by one field: ``{value: (passed, total)}``."""
        groups: dict[str, tuple[int, int]] = {}
        for outcome in self.outcomes:
            key = getattr(outcome, field_name)
            passed, total = groups.get(key, (0, 0))
            groups[key] = (passed + outcome.passed, total + 1)
        return groups

    def mean_iterations(self, only_passed: bool = True) -> dict[str, float]:
        """Mean tear iterations per acceleration.

        Averaged over passing solves by default: a failed solve used
        ``max_iter`` by definition, and mixing those in measures the cap
        rather than the method.
        """
        sums: dict[str, list[int]] = {}
        for outcome in self.outcomes:
            if only_passed and not outcome.passed:
                continue
            if outcome.iterations is None:
                continue
            sums.setdefault(outcome.acceleration, []).append(outcome.iterations)
        return {k: sum(v) / len(v) for k, v in sums.items() if v}

    def failures(self) -> list[Outcome]:
        """Every outcome that did not pass."""
        return [o for o in self.outcomes if not o.passed]

    def as_text(self, width: int = 78) -> str:
        """The report as a table, for a terminal or a notebook."""
        lines: list[str] = []
        lines.append("difflow convergence benchmark")
        lines.append("=" * width)
        lines.append(
            f"{len(self.outcomes)} solves  "
            f"(tol={self.tol:.0e}, max_iter={self.max_iter})")
        lines.append(
            f"pass rate        {self.pass_rate:6.1%}   "
            "(converged AND the answer audits)")
        lines.append(
            f"convergence rate {self.convergence_rate:6.1%}   "
            "(the solver's own verdict)")
        lines.append("")

        lines.append("by acceleration")
        lines.append("-" * width)
        for key, (passed, total) in sorted(self.by("acceleration").items()):
            lines.append(f"  {key:<12s} {passed:>3d}/{total:<3d}  {passed / total:6.1%}")
        means = self.mean_iterations()
        if means:
            lines.append("  mean iterations over passing solves: " + ", ".join(
                f"{k}={v:.1f}" for k, v in sorted(means.items())))
        lines.append("")

        lines.append("by initialization")
        lines.append("-" * width)
        for key, (passed, total) in sorted(self.by("initialization").items()):
            lines.append(f"  {key:<12s} {passed:>3d}/{total:<3d}  {passed / total:6.1%}")
        lines.append("")

        lines.append("by case")
        lines.append("-" * width)
        for key, (passed, total) in self.by("case").items():
            lines.append(f"  {key:<22s} {passed:>3d}/{total:<3d}  {passed / total:6.1%}")
        lines.append("")

        lines.append("detail")
        lines.append("-" * width)
        header = (f"  {'case':<22s} {'accel':<9s} {'init':<8s} "
                  f"{'verdict':<8s} {'iters':>5s}  {'balance':>9s}"
                  f"  {'err est':>9s}")
        lines.append(header)
        for outcome in self.outcomes:
            iters = "-" if outcome.iterations is None else str(outcome.iterations)
            err = outcome.balance_error
            bal = "-" if err != err else (  # NaN-safe
                "inf" if err == float("inf") else f"{err:.2e}")
            # The solver's own estimate of how far out it is, beside the
            # audit's verdict on the same solve: on a WRONG row the two
            # should agree, and where they do the solver could have said so.
            est = ("-" if outcome.error_estimate is None
                   else f"{outcome.error_estimate:.2e}")
            lines.append(
                f"  {outcome.case:<22s} {outcome.acceleration:<9s} "
                f"{outcome.initialization:<8s} {outcome.verdict:<8s} "
                f"{iters:>5s}  {bal:>9s}  {est:>9s}")

        raised = [o for o in self.outcomes if o.error]
        if raised:
            lines.append("")
            lines.append("raised")
            lines.append("-" * width)
            for outcome in raised:
                lines.append(
                    f"  {outcome.case} / {outcome.acceleration} / "
                    f"{outcome.initialization}: {outcome.error}")
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """Write every outcome to CSV, one row per solve."""
        import csv

        fields = ["case", "acceleration", "initialization", "converged",
                  "correct", "passed", "balance_error", "iterations",
                  "residual", "method", "wall_time", "error"]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for outcome in self.outcomes:
                row = {f: getattr(outcome, f, None) for f in fields}
                row["passed"] = outcome.passed
                writer.writerow(row)

    def __repr__(self) -> str:
        return (f"Report({len(self.outcomes)} solves, "
                f"pass_rate={self.pass_rate:.1%})")


def run_benchmark(cases: Sequence[str] | Sequence[Case] | None = None,
                  accelerations: Sequence[str] = ACCELERATIONS,
                  initializations: Sequence[str] = INITIALIZATIONS,
                  tol: float = 1e-8, max_iter: int = 100,
                  progress: bool = False, **solve_kwargs) -> Report:
    """Run the corpus across every setting and report the pass rate.

    Args:
        cases: Case names or :class:`Case` objects.  ``None`` runs
            :data:`CORPUS`.
        accelerations: Which ``acceleration`` settings to try.
        initializations: Which strategies from :data:`INITIALIZATIONS`.
        tol: Tear tolerance for every solve.
        max_iter: Iteration cap for every solve.
        progress: Print each outcome as it lands.  The full grid takes
            minutes, most of it JAX compilation, and a silent run is hard
            to tell from a hung one.
        **solve_kwargs: Anything else for :meth:`Flowsheet.solve`.

    Returns:
        A :class:`Report`.

    Example:
        >>> report = run_benchmark(cases=["two_phase_flash"],  # doctest: +SKIP
        ...                        accelerations=("anderson",),
        ...                        initializations=("unit",))
        >>> report.pass_rate                                  # doctest: +SKIP
        1.0
    """
    if cases is None:
        selected = list(CORPUS)
    else:
        selected = [c if isinstance(c, Case) else get_case(c) for c in cases]

    report = Report(tol=tol, max_iter=max_iter)
    for case in selected:
        for acceleration in accelerations:
            for initialization in initializations:
                outcome = run_case(case, acceleration=acceleration,
                                   initialization=initialization, tol=tol,
                                   max_iter=max_iter, **solve_kwargs)
                report.outcomes.append(outcome)
                if progress:
                    print(f"{outcome.case:<22s} {acceleration:<9s} "
                          f"{initialization:<8s} {outcome.verdict}", flush=True)
    return report


def main() -> None:
    """``python -m difflow.convergence``: run the corpus and print the report."""
    report = run_benchmark(progress=True)
    print()
    print(report.as_text())


if __name__ == "__main__":
    main()
