"""Chaining library units into one differentiable function (#334).

The refinery library units -- hydrotreater, reformer, residue desulfurizer,
FCC, hydrocracker and the rest -- are Python objects whose ``solve`` returns
a result pytree, and the adapters between them (``gas_plant_feed``,
``fractionate``, ``NaphthaFeed.from_hydrotreater``, ``fuel_oil_blend``, the
hydrogen network) are pure JAX. So a chain of units is just a Python
function, and the only thing that needs care is the derivative: the units do
not all support the same AD mode (:data:`AD_MODES`).

This module is that care, and deliberately no more than it:

* :class:`Stage` -- one step of a chain: a function from an array pytree to
  an array pytree, and the AD modes it supports;
* :class:`Chain` -- stages composed; :meth:`Chain.jacobian` takes the
  derivative of the whole chain in the cheapest way the stages allow;
* :data:`AD_MODES` / :func:`ad_mode_table` -- every unit's AD mode, in one
  place;
* :func:`central_difference` and :func:`timed` -- the finite-difference check
  and the compile-cost measurement that go with a cross-unit gradient.

How :meth:`Chain.jacobian` differentiates
-----------------------------------------
``method="fwd"`` is ``jax.jacfwd`` of the composition and ``"rev"`` is
``jax.jacrev``: one trace through every unit, possible only when every stage
supports that mode. ``"auto"`` picks one of them when all stages share it (by
shape, :func:`difflow.planning.linearize.choose_ad_mode`), and otherwise
falls back to ``"chain"``.

``"chain"`` is the chain rule by unit Jacobians, for a MIXED chain -- a
forward-only unit (the reformer: ``diffrax.ForwardMode`` beds inside an
``optimistix`` implicit fixed point) next to a reverse-only one (the
hydrotreater's default ``RecursiveCheckpointAdjoint`` beds, a
``custom_vjp``). Neither ``jax.jacfwd`` nor ``jax.jacrev`` can trace such a
chain end to end: the first fails in the reverse-only unit ("can't apply
forward-mode autodiff (jvp) to a custom_vjp function") and the second in the
forward-only one (a ``while_loop`` cannot be transposed). ``"chain"``
evaluates the stages one at a time on concrete values and carries the
Jacobian ``dx_k/dx_0`` along (forward accumulation)::

    J_0 = I
    J_k = jvp(f_k, x_{k-1}) applied to each column of J_{k-1}   (forward stage)
    J_k = jacrev(f_k)(x_{k-1}) @ J_{k-1}                        (reverse-only stage)

A forward stage costs one tangent per chain INPUT; a reverse-only stage one
cotangent per output of that stage, so keep the interfaces after a
reverse-only unit narrow (a ``NaphthaFeed`` is 23 numbers, a whole
hydrotreater result is thousands). Each stage compiles on its own, which
is also what makes the cost of each one visible (:class:`ChainJacobian`
``.timings``).

Two things this does not do, by design: it is not a ``Flowsheet`` (the units
already contain their own recycles; a chain between them has none -- a
recycle between library units would be a tear around two compiled solves,
which the hydrogen loop of :mod:`difflow_refinery.hydrogen.loop` shows is a
concrete Python iteration, not a traced one), and it does not cache or jit
the composition (each unit jit-compiles its own solve, and the stage
Jacobians are jit-compiled per stage).

The hydrotreater and the residue desulfurizer can be made forward-capable:
``HydrotreaterParams(reactor=ReactorOptions(adjoint="forward"))`` (and
``RDSParams(reactor=...)``) integrate the beds with ``diffrax.ForwardMode``;
the Newton loops already use a forward copy, and the final implicit step
``x* - J^-1 f(x*)`` is differentiable in both modes. Then a chain of those
units and the reformer is all forward, and ``method="fwd"`` traces it end to
end. What it costs is reverse mode on that unit (``jax.grad`` then fails).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from difflow.planning.linearize import choose_ad_mode

jax.config.update("jax_enable_x64", True)

MODES = ("fwd", "rev")


# =============================================================================
# The AD-mode table
# =============================================================================


@dataclass(frozen=True)
class ADMode:
    """How one unit (or adapter) of the plugin can be differentiated.

    Attributes:
        name: What it is.
        obj: The class or function (dotted path under ``difflow_refinery``).
        modes: The AD modes it supports as solved by default, a subset of
            ``("fwd", "rev")``.
        mechanism: What sets the mode.
        switch: How to get the other mode, if there is a way.
        evidence: The test that exercises the mode(s).
    """

    name: str
    obj: str
    modes: tuple
    mechanism: str
    switch: str = ""
    evidence: str = ""


#: Every unit's AD mode, in one place (the table of the refinery docs,
#: "Chaining units"). ``modes`` is what works with the unit's DEFAULT
#: options; ``switch`` says what changes it.
AD_MODES: dict[str, ADMode] = {m.name: m for m in (
    ADMode("crude unit", "unit.CrudeUnit / column.CrudeColumn", ("fwd", "rev"),
           "EO MESH Newton in lax.while_loop on stop-gradient inputs, then one Newton step with the "
           "converged Jacobian (implicit-function derivative)",
           evidence="tests/refinery/test_column.py, test_unit.py (jax.grad); test_column.py::TestTransforms (vmap)"),
    ADMode("vacuum unit", "vacuum.VacuumColumn / vacuum.StageColumn", ("fwd", "rev"),
           "stage-network Newton, implicit step reusing the converged Jacobian",
           evidence="tests/refinery/test_vacuum.py (jax.jacfwd and jax.grad)"),
    ADMode("gas plant columns", "gasplant.GasPlantColumn", ("fwd", "rev"),
           "the vacuum StageColumn machinery on CubicThermo: three Newton passes, then the implicit step",
           evidence="tests/refinery/test_gasplant.py (jax.jacfwd of gasplant_block)"),
    ADMode("gas compressor, amine treater", "gasplant.GasCompressor, gasplant.AmineTreater", ("fwd", "rev"),
           "closed-form stages and removal fractions (pure JAX)"),
    ADMode("hydrotreater", "hydrotreating.Hydrotreater", ("rev",),
           "beds integrated by diffrax with RecursiveCheckpointAdjoint (a custom_vjp, reverse only); the "
           "recycle tear's Newton takes its Jacobian from a forward-adjoint copy and ends with one "
           "implicit step on the reverse-mode residual",
           switch="HydrotreaterParams(reactor=ReactorOptions(adjoint='forward')) integrates the beds with "
                  "diffrax.ForwardMode: then forward mode only (jax.jacfwd), same values",
           evidence="tests/refinery/test_hydrotreating.py (jax.jacrev); tests/refinery/test_plant.py (forward)"),
    ADMode("residue desulfurizer", "residue.ResidueDesulfurizer", ("rev",),
           "trickle beds with RecursiveCheckpointAdjoint (reverse only), quench mixing by implicit Newton",
           switch="RDSParams(reactor=ReactorOptions(adjoint='forward')) for forward mode only",
           evidence="tests/refinery/test_residue.py (jax.grad of fuel-oil sulfur)"),
    ADMode("hydrocracker", "hydrocracking.Hydrocracker", ("rev",),
           "checkpointed bed adjoints, and the UCO recycle's Anderson fixed point with a GMRES adjoint "
           "(a custom_vjp); forward mode is not available",
           evidence="tests/refinery/test_hydrocracking.py (jax.jacrev)"),
    ADMode("catalytic reformer", "reforming.CatalyticReformer", ("fwd",),
           "beds integrated with diffrax.ForwardMode inside the Flowsheet's traced recycle path "
           "(optimistix fixed point, implicit differentiation, which needs JVPs of the loop)",
           switch="none for the whole unit: CatalyticReformer(adjoint='reverse') makes a stand-alone "
                  "reactor reverse-capable, but the recycle's implicit derivative still needs JVPs",
           evidence="tests/refinery/test_reforming.py::test_implicit_gradients_match_central_differences "
                    "(jax.jacfwd)"),
    ADMode("FCC", "fcc.FCCUnit", ("fwd", "rev"),
           "riser by constant-step Tsit5 with diffrax.DirectAdjoint (both modes), riser-regenerator "
           "heat balance by optimistix Newton with implicit adjoint",
           evidence="tests/refinery/test_fcc.py::TestGradients::test_reverse_mode_matches_forward"),
    ADMode("isomerization", "isomerization.IsomerizationUnit", ("fwd",),
           "with a DIH the recycle is converged by the Flowsheet's Python Anderson loop and differentiated "
           "at its solution by a jax.custom_jvp",
           evidence="tests/refinery/test_isomerization_dih_gradients.py (jax.jacfwd)"),
    ADMode("alkylation", "alkylation.AlkylationUnit", ("fwd",),
           "DIB-overhead tear: the Flowsheet's traced optimistix fixed point; the planning block "
           "differentiates it with jax.jacfwd",
           evidence="tests/refinery/test_alkylation.py (jax.jvp, jax.jacfwd)"),
    ADMode("preheat train", "preheat.PreheatedCrudeUnit", ("fwd", "rev"),
           "outer Newton in lax.while_loop with an implicit step (preheat._newton.implicit_step)",
           evidence="tests/refinery/test_preheat.py (jax.grad and jax.jacfwd)"),
    ADMode("blend pool", "blending.BlendPool / BlendComponent.from_stream", ("fwd", "rev"),
           "closed-form blending rules and property estimates (pure JAX)",
           evidence="tests/refinery/test_blending.py, test_properties.py"),
    ADMode("adapters", "gasplant.gas_plant_feed, hydrotreating.HydrotreaterResult.fractionate, "
           "reforming.NaphthaFeed.from_hydrotreater, "
           "residue.fuel_oil_blend", ("fwd", "rev"),
           "sums, sigmoid splits and lumping (pure JAX); gas_plant_feed's cut SELECTION is static "
           "(pass cuts= under a transform)",
           evidence="tests/refinery/test_gasplant_feed.py, test_hydrotreated_naphtha_feeds.py"),
    ADMode("hydrogen network", "hydrogen.HydrogenNetwork", ("fwd", "rev"),
           "header balances, swing clips and the PSA purity target in pure JAX (the clips are kinks); "
           "close_hydrotreater_loop is a concrete Python iteration that returns a linear purity response",
           evidence="tests/refinery/test_hydrogen.py, test_hydrogen_loop.py"),
)}


def ad_mode_table(markdown: bool = True) -> str:
    """:data:`AD_MODES` as a table (GitHub markdown, or plain text)."""
    rows = [(m.name, m.obj, " + ".join(m.modes), m.mechanism, m.switch or "--") for m in AD_MODES.values()]
    head = ("unit", "object", "modes", "why", "other mode")
    if markdown:
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    return "\n".join(f"{r[0]}: {r[2]} -- {r[3]}" + (f" (other mode: {r[4]})" if r[4] != "--" else "")
                     for r in rows)


# =============================================================================
# Stages and chains
# =============================================================================


@dataclass(frozen=True)
class Stage:
    """One step of a :class:`Chain`.

    Attributes:
        name: Name (labels timings and errors).
        fn: ``fn(x) -> y``, ``x`` and ``y`` pytrees of float arrays. Must be
            a pure function of ``x``: anything else it reads is a constant of
            the chain.
        modes: The AD modes ``fn`` supports, a non-empty subset of
            ``("fwd", "rev")`` -- for a unit, its :data:`AD_MODES` entry
            (``"rev"`` for a default hydrotreater, ``"fwd"`` for the reformer).
    """

    name: str
    fn: Callable
    modes: tuple = MODES

    def __post_init__(self):
        modes = (self.modes,) if isinstance(self.modes, str) else tuple(self.modes)
        bad = [m for m in modes if m not in MODES]
        if not modes or bad:
            raise ValueError(f"stage {self.name!r}: modes must be a non-empty subset of {MODES}, got {modes}")
        object.__setattr__(self, "modes", tuple(m for m in MODES if m in modes))


def _block(tree):
    return jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a,
                                  tree)


@dataclass(frozen=True)
class ChainJacobian:
    """A chain's value and Jacobian at one point.

    Attributes:
        value: The chain's output (the last stage's pytree).
        jacobian: ``(n_out, n_in)`` array, ``d ravel(value) / d ravel(x)``
            (``jax.flatten_util.ravel_pytree`` order).
        method: ``"fwd"``, ``"rev"`` or ``"chain"`` -- what was used.
        timings: Seconds per step: ``{"total": s}`` for ``fwd``/``rev``,
            ``{stage: s}`` plus ``"total"`` for ``chain``. Each includes the
            stage's compilation the first time a stage is differentiated at
            this shape; :func:`timed` separates the two.
        unravel_in, unravel_out: Map a flat vector back to the input/output
            pytree (``unravel_out(J[:, j])`` is the derivative of every output
            with respect to input ``j``).
    """

    value: Any
    jacobian: jax.Array
    method: str
    timings: dict
    unravel_in: Callable = field(repr=False, default=None)
    unravel_out: Callable = field(repr=False, default=None)

    def column(self, j: int):
        """Derivative of the output pytree with respect to flat input ``j``."""
        return self.unravel_out(self.jacobian[:, j])


class Chain:
    """Stages composed left to right: ``Chain(a, b, c)(x) == c.fn(b.fn(a.fn(x)))``.

    Example::

        chain = Chain(Stage("nht", lambda T: heavy_naphtha_feed(T), modes="rev"),
                      Stage("reformer", lambda feed: reformate_ron(feed), modes="fwd"))
        res = chain.jacobian(jnp.asarray(593.15))       # method "chain": mixed modes
        res.jacobian, res.timings
    """

    def __init__(self, *stages: Stage):
        if len(stages) == 1 and isinstance(stages[0], (list, tuple)):
            stages = tuple(stages[0])
        if not stages:
            raise ValueError("a chain needs at least one stage")
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique: {names}")
        self.stages = tuple(stages)

    def __call__(self, x):
        for s in self.stages:
            x = s.fn(x)
        return x

    @property
    def modes(self) -> tuple:
        """AD modes every stage supports (empty for a mixed chain)."""
        return tuple(m for m in MODES if all(m in s.modes for s in self.stages))

    def choose(self, n_in: int, n_out: int, method: str = "auto") -> str:
        """The method :meth:`jacobian` would use for ``n_in`` inputs and ``n_out`` outputs."""
        if method == "chain":
            return method
        common = self.modes
        if method in MODES:
            if method not in common:
                need = [s.name for s in self.stages if method not in s.modes]
                raise ValueError(f"method={method!r} needs every stage to support it; {need} do not "
                                 f"(use method='chain', the chain rule by unit Jacobians)")
            return method
        if method != "auto":
            raise ValueError(f"unknown method {method!r}; one of 'auto', 'fwd', 'rev', 'chain'")
        if len(common) == 2:
            return choose_ad_mode(n_in, n_out)
        return common[0] if common else "chain"

    def jacobian(self, x, method: str = "auto") -> ChainJacobian:
        """Value and Jacobian of the chain at ``x`` (see the module docstring for the methods).

        ``"auto"`` needs the output size to choose between ``fwd`` and
        ``rev`` and gets it by evaluating the chain once, unless the chain is
        mixed (then ``chain``) or supports one mode only.
        """
        x_flat, unravel_in = ravel_pytree(x)
        if method == "auto" and len(self.modes) == 2:
            y0 = _block(self(x))
            n_out = ravel_pytree(y0)[0].size
            method = self.choose(x_flat.size, n_out, "auto")
        else:
            method = self.choose(x_flat.size, 0, method)
        if method in MODES:
            t0 = time.perf_counter()
            out_struct = {}

            def flat(v):
                y = self(unravel_in(v))
                yf, un = ravel_pytree(y)
                out_struct["unravel"] = un
                return yf, y

            jac = jax.jacfwd if method == "fwd" else jax.jacrev
            J, y = jac(flat, has_aux=True)(x_flat)
            J, y = _block(J), _block(y)
            return ChainJacobian(value=y, jacobian=J, method=method,
                                 timings={"total": time.perf_counter() - t0},
                                 unravel_in=unravel_in, unravel_out=out_struct["unravel"])
        # the chain rule by unit Jacobians, forward accumulation
        timings = {}
        t_all = time.perf_counter()
        xk = x
        J = jnp.eye(x_flat.size)
        for s in self.stages:
            t0 = time.perf_counter()
            xf, unravel = ravel_pytree(xk)
            box = {}

            def f(v, s=s, unravel=unravel, box=box):
                y = s.fn(unravel(v))
                yf, un = ravel_pytree(y)
                box["unravel"] = un
                return yf

            if "fwd" in s.modes:
                def push(t, f=f, xf=xf):
                    return jax.jvp(f, (xf,), (t,))[1]
                J = jax.vmap(push, in_axes=1, out_axes=1)(J)
                xk = _block(s.fn(xk))
            else:
                yf, pull = jax.vjp(f, xf)
                Jk = jax.vmap(lambda c, pull=pull: pull(c)[0])(jnp.eye(yf.size))
                J = Jk @ J
                xk = _block(box["unravel"](yf))
            J = _block(J)
            timings[s.name] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_all
        return ChainJacobian(value=xk, jacobian=J, method="chain", timings=timings,
                             unravel_in=unravel_in, unravel_out=ravel_pytree(xk)[1])


# =============================================================================
# Checking and measuring
# =============================================================================


def central_difference(fn: Callable, x, h) -> np.ndarray:
    """``(n_out, n_in)`` central-difference Jacobian of ``fn`` at ``x`` (pytrees raveled).

    ``h`` is a step per flat input (scalar or array). Costs ``2 n_in``
    evaluations of ``fn`` on concrete values.
    """
    x_flat, unravel = ravel_pytree(x)
    h = np.broadcast_to(np.asarray(h, dtype=float), x_flat.shape)
    cols = []
    for j in range(x_flat.size):
        e = jnp.zeros_like(x_flat).at[j].set(h[j])
        yp = ravel_pytree(fn(unravel(x_flat + e)))[0]
        ym = ravel_pytree(fn(unravel(x_flat - e)))[0]
        cols.append(np.asarray((yp - ym) / (2.0 * h[j])))
    return np.stack(cols, axis=1)


def timed(fn: Callable, *args, repeat: int = 2) -> tuple[Any, list[float]]:
    """``(result, [seconds per call])`` of ``repeat`` calls of ``fn(*args)``.

    The first call includes tracing and XLA compilation; the later ones hit
    the jit caches the units keep, so ``t[0] - t[-1]`` is the compile cost
    (for a function whose units are jit-compiled -- the refinery units'
    solves are; Python-level iteration such as the reformer's concrete
    recycle loop is paid on every call).
    """
    ts, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = _block(fn(*args))
        ts.append(time.perf_counter() - t0)
    return out, ts


__all__ = ["ADMode", "AD_MODES", "ad_mode_table", "Stage", "Chain", "ChainJacobian", "central_difference",
           "timed", "MODES"]
