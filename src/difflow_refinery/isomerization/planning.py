"""The isomerization unit as a delta-base planning block.

:func:`isom_block` does for the C5/C6 unit what
:func:`~difflow_refinery.planning.cdu_block` does for the crude unit: wraps
an :class:`~difflow_refinery.isomerization.plant.IsomerizationUnit` and its
fresh feed as a :class:`~difflow.planning.Block` whose delta vectors are the
AD Jacobian of the converged unit -- through the DIH recycle by the implicit
function theorem (:meth:`IsomerizationUnit.outputs
<difflow_refinery.isomerization.plant.IsomerizationUnit.outputs>`).

Levers
    ``T_in`` (reactor inlet, C), ``LHSV`` (1/h) and ``x_nC6`` (the fresh
    feed's n-hexane mole fraction, :func:`~.feed.with_nc6_fraction`).

Outputs
    Any of :data:`~difflow_refinery.isomerization.plant.OUTPUT_NAMES`, in
    planner units (duties in MW, temperatures differences in K, RVP in
    kPa), plus ``isomerate_V``: the isomerate volume flow (m3/h at 60 F),
    named so that :func:`link_isom` can feed it to a
    :meth:`BlendPool.as_block <difflow_refinery.blending.BlendPool.as_block>`
    lever of the same name.

The block is NOT jit-compiled and its AD mode is forward: a configuration
with a DIH is converged by the flowsheet's Anderson loop, a Python loop
that cannot be traced, and differentiated at its solution by a
``jax.custom_jvp``.

What crosses the link is the isomerate's VOLUME. The pool's components are
:class:`~difflow_refinery.blending.BlendComponent` objects with fixed
properties, so the isomerate's octane in the pool is the one at the
linearisation point; rebuild the component from
:meth:`IsomerizationUnit.blend_component` at each new base point.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from difflow.planning.block import Block

from difflow_refinery.isomerization.feed import nc6_fraction
from difflow_refinery.isomerization.plant import OUTPUT_NAMES, OUTPUT_UNITS, IsomerizationUnit

_C = 273.15

#: Lever names and their planner units.
LEVERS = {"T_in": "C", "LHSV": "1/h", "x_nC6": "-"}

#: Default block outputs.
DEFAULT_OUTPUTS = ("isomerate_V", "RON", "MON", "RVP", "yield_volume", "H2_makeup",
                   "gas_make", "dih_duty", "dip_duty")

_SCALE = {"W": (1e-6, "MW"), "Pa": (1e-3, "kPa"), "m3/h": (1.0, "m3/h")}


def _planner(name):
    if name == "isomerate_V":
        return 1.0, "m3/h"
    si = OUTPUT_UNITS[name]
    return _SCALE.get(si, (1.0, si))


class IsomPlanningModel:
    """The ``u -> y`` map behind :func:`isom_block`."""

    def __init__(self, unit: IsomerizationUnit, feed: Mapping, levers: Sequence[str],
                 outputs: Sequence[str] | None = None, T_in: float | None = None,
                 LHSV: float | None = None):
        bad = sorted(set(levers) - set(LEVERS))
        if bad:
            raise ValueError(f"unknown levers {bad}; the levers are {sorted(LEVERS)}")
        self.unit, self.feed = unit, feed
        self.levers = list(levers)
        self.outputs = list(DEFAULT_OUTPUTS if outputs is None else outputs)
        bad = sorted(set(self.outputs) - set(OUTPUT_NAMES) - {"isomerate_V"})
        if bad:
            raise ValueError(f"unknown outputs {bad}")
        p = unit.params
        self.base = {"T_in": (p.T_in if T_in is None else T_in) - _C,
                     "LHSV": p.reactor.LHSV if LHSV is None else LHSV,
                     "x_nC6": float(nc6_fraction(feed))}
        self.u0 = [float(self.base[n]) for n in self.levers]
        self.output_units = {n: _planner(n)[1] for n in self.outputs}

    def __call__(self, u):
        val = dict(self.base)
        for i, n in enumerate(self.levers):
            val[n] = u[i]
        x6 = val["x_nC6"] if "x_nC6" in self.levers else None
        y = self.unit.outputs(self.feed, val["T_in"] + _C, val["LHSV"], x_nc6=x6)
        out = []
        for n in self.outputs:
            k = "isomerate_volume" if n == "isomerate_V" else n
            out.append(y[OUTPUT_NAMES.index(k)] * _planner(n)[0])
        return jnp.stack(out)


def isom_block(unit: IsomerizationUnit, feed: Mapping, levers: Sequence[str] = ("T_in",),
               outputs: Sequence[str] | None = None, *, name: str = "isom",
               T_in: float | None = None, LHSV: float | None = None,
               bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.1,
               **kwargs: Any) -> Block:
    """A :class:`~difflow.planning.Block` for an isomerization unit.

    Args:
        unit: The :class:`~difflow_refinery.isomerization.plant.IsomerizationUnit`.
        feed: Its fresh feed stream: the linearisation point.
        levers: Any of ``T_in`` (C), ``LHSV`` (1/h), ``x_nC6`` (-).
        outputs: Output names (see the module docstring); default
            :data:`DEFAULT_OUTPUTS`.
        name: Block name.
        T_in, LHSV: Base inlet temperature (K) and LHSV (1/h); default the
            unit's own.
        bounds: ``{lever: (lo, hi)}`` in planner units; otherwise
            ``u0 (1 -/+ span)``.
        span: Default relative half-width of a lever's bounds.
        **kwargs: Passed to :class:`~difflow.planning.Block`. ``jit`` is
            forced off and ``ad_mode`` to ``"fwd"`` (see the module
            docstring).
    """
    model = IsomPlanningModel(unit, feed, levers, outputs, T_in, LHSV)
    bounds = dict(bounds or {})
    unknown = sorted(set(bounds) - set(levers))
    if unknown:
        raise ValueError(f"bounds given for {unknown}, which are not levers")
    lb, ub = [], []
    for lv, x in zip(model.levers, model.u0):
        if lv in bounds:
            lo, hi = bounds[lv]
        else:
            w = abs(x) * span if x != 0 else span
            lo, hi = x - w, x + w
        if not lo <= x <= hi:
            raise ValueError(f"{lv}: base value {x:g} is outside its bounds ({lo:g}, {hi:g})")
        lb.append(lo)
        ub.append(hi)
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata.setdefault("source", "difflow_refinery.isomerization.isom_block")
    metadata.setdefault("u_units", [LEVERS[n] for n in model.levers])
    metadata.setdefault("y_units", [model.output_units[n] for n in model.outputs])
    kwargs.pop("jit", None)
    kwargs["ad_mode"] = "fwd"
    return Block(name=name, fn=model, u_names=list(model.levers), y_names=list(model.outputs),
                 lb=lb, ub=ub, u0=model.u0, jit=False, metadata=metadata, **kwargs)


def link_isom(isom: Block, pool: Block) -> list[tuple[str, str]]:
    """The link from the isomerate volume to a blend pool's ``isomerate_V``
    lever (:meth:`BlendPool.as_block
    <difflow_refinery.blending.BlendPool.as_block>` names its levers
    ``<component>_V``; the component must be called ``"isomerate"``)."""
    if "isomerate_V" not in isom.y_names:
        raise ValueError(f"{isom.name} does not report isomerate_V")
    if "isomerate_V" not in pool.u_names:
        raise ValueError(f"{pool.name} has no isomerate_V lever; name the blend "
                         "component 'isomerate'")
    return [(f"{isom.name}.isomerate_V", f"{pool.name}.isomerate_V")]


__all__ = ["DEFAULT_OUTPUTS", "LEVERS", "IsomPlanningModel", "isom_block", "link_isom"]
