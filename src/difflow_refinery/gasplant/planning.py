"""A gas plant column as a delta-base planning block.

:func:`gasplant_block` does for a light-ends column what
:func:`~difflow_refinery.planning.cdu_block` does for the crude unit: wraps
a :class:`~difflow_refinery.gasplant.units.GasPlantColumn` and its feeds as
a :class:`~difflow.planning.Block` whose delta vectors are the AD Jacobian
of the converged column (implicit-function gradients), refreshed every
trust-region cycle.

Levers
    * every spec *target* by the spec's own name -- ``distillate.x.C5+``,
      ``bottoms.rvp``, ``bottoms.x.C2-``, ``bottom.T`` -- so moving the
      LPG's C5+ limit or the naphtha's RVP is one lever;
    * every column knob a spec has not replaced: ``top.P``,
      ``reboiler.duty``, ``condenser.duty`` (partial condenser),
      ``<draw>.rate`` (side draws);
    * per feed, ``<feed>.mol`` (its total rate, composition held),
      ``<feed>.T``, and ``<feed>.F_<component>`` (one component's rate).

Outputs
    Every column output (:meth:`GasColumn.outputs_from
    <difflow_refinery.gasplant.column.GasColumn.outputs_from>`), in planner
    units.

Units, by the name's last part (also in ``Block.metadata``):

======================================  ======================================
``T``, ``*_T``                          C
``P``, ``rvp``, ``tvp``                 kPa (absolute)
``duty``, ``energy_balance``            MW
``rate``                                kg/h
``mol``, ``F_<component>``              kmol/h
fractions, recoveries, ratios           ``-``
======================================  ======================================

Non-convergence is masked to NaN, which the planner refuses, exactly as in
:func:`~difflow_refinery.planning.cdu_block`.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from difflow.planning.block import Block

from difflow_refinery.gasplant.units import GasPlantColumn

_C = 273.15


def planner_units(name: str) -> str:
    """The planner unit of an output or lever, from its name."""
    tail = name.rsplit(".", 1)[-1]
    parts = name.split(".")
    if len(parts) >= 3 and parts[-2] in ("x", "recovery"):
        return "-"
    if tail == "T" or tail.endswith("_T"):
        return "C"
    if tail in ("P", "rvp", "tvp"):
        return "kPa"
    if tail in ("duty", "energy_balance"):
        return "MW"
    if tail == "rate":
        return "kg/h"
    if tail == "mol" or tail.startswith("F_"):
        return "kmol/h"
    return "-"


_TO = {"C": lambda v: v - _C, "kPa": lambda v: v / 1e3, "MW": lambda v: v / 1e6,
       "kg/h": lambda v: v * 3600.0, "kmol/h": lambda v: v * 3.6, "-": lambda v: v}
_FROM = {"C": lambda v: v + _C, "kPa": lambda v: v * 1e3, "MW": lambda v: v * 1e6,
         "kg/h": lambda v: v / 3600.0, "kmol/h": lambda v: v / 3.6, "-": lambda v: v}


class GasPlantPlanningModel:
    """The pure ``u -> y`` map behind :func:`gasplant_block`.

    Callable on a lever array (planner units), returning the outputs
    (planner units), NaN everywhere when the column does not converge.
    :meth:`solve` gives the column's full result at a lever array.
    """

    def __init__(self, column: GasPlantColumn, feeds: Sequence[Mapping],
                 levers: Sequence[str], outputs: Sequence[str] | None,
                 mask_nonconverged: bool = True):
        self.column = column
        self.mask_nonconverged = bool(mask_nonconverged)
        self.th0 = column.theta(*feeds)
        base = column.solve_theta(self.th0)
        if not bool(base["converged"]):
            raise ValueError("the column does not converge at its base point; "
                             "a planning block has to start from a solution")
        self.base = base
        avail = self.available_levers()
        levers = list(levers)
        unknown = [n for n in levers if n not in avail]
        if unknown:
            raise ValueError(f"no lever(s) {unknown}; available: {sorted(avail)}")
        if len(set(levers)) != len(levers):
            raise ValueError("duplicate levers")
        self.levers = levers
        self.output_units = {n: planner_units(n) for n in base["outputs"]}
        if outputs is None:
            outputs = [n for n in self.default_outputs() if n not in set(levers)]
        outputs = list(outputs)
        unknown = [n for n in outputs if n not in self.output_units]
        if unknown:
            raise ValueError(f"no output(s) {unknown}; available: {sorted(self.output_units)}")
        both = sorted(set(outputs) & set(levers))
        if both:
            raise ValueError(f"{both} are levers: as outputs they would be the lever itself")
        self.outputs = outputs
        self.u0 = np.array([float(_TO[planner_units(n)](avail[n])) for n in levers])

    def available_levers(self) -> dict[str, float]:
        """``{lever: base value in SI}``."""
        th = self.th0
        out = {}
        for name, v in th["targets"].items():
            out[name] = v
        replaced = {s.replaces for s in self.column.specs}
        for name, v in th["knobs"].items():
            if name not in replaced and name != "dP":
                out[name] = v
        for fname, fd in th["feeds"].items():
            out[f"{fname}.mol"] = jnp.sum(fd["flows"])
            out[f"{fname}.T"] = fd["T"]
            for i, c in enumerate(self.column.names):
                out[f"{fname}.F_{c}"] = fd["flows"][i]
        return {k: float(v) for k, v in out.items()}

    def default_outputs(self) -> list[str]:
        """Product rates, duties, end temperatures, and the spec'd qualities."""
        names = list(self.column.products)
        out = []
        for p in names:
            out += [f"{p}.mol", f"{p}.rate"]
        for k in ("reboiler.duty", "condenser.duty", "top.T", "bottom.T", "reflux_ratio"):
            if k in self.output_units:
                out.append(k)
        out += [s.output for s in self.column.specs]
        for p in names:
            if f"{p}.rvp" in self.output_units:
                out.append(f"{p}.rvp")
        seen = set()
        return [n for n in out if not (n in seen or seen.add(n))]

    def theta_of(self, u):
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        th = self.th0
        targets = dict(th["targets"])
        knobs = dict(th["knobs"])
        feeds = {k: dict(v) for k, v in th["feeds"].items()}
        for i, name in enumerate(self.levers):
            v = _FROM[planner_units(name)](u[i])
            if name in targets:
                targets[name] = v
                continue
            if name in knobs:
                knobs[name] = v
                continue
            fname, rest = name.split(".", 1)
            fd = feeds[fname]
            if rest == "mol":
                fl = th["feeds"][fname]["flows"]
                fd["flows"] = fd["flows"] * (v / jnp.sum(fl))
            elif rest == "T":
                fd["T"] = v
            else:
                c = self.column.names.index(rest[2:])
                fd["flows"] = fd["flows"].at[c].set(v)
        return dict(th, targets=targets, knobs=knobs, feeds=feeds)

    def solve(self, u) -> dict:
        """The column's full result at a lever array (planner units)."""
        return self.column.solve_theta(self.theta_of(u))

    def __call__(self, u):
        res = self.solve(u)
        o = res["outputs"]
        y = jnp.stack([jnp.asarray(_TO[self.output_units[n]](o[n]), dtype=float)
                       for n in self.outputs])
        if not self.mask_nonconverged:
            return y
        return jnp.where(res["converged"], y, jnp.nan)


def gasplant_block(column: GasPlantColumn, feeds: Sequence[Mapping], levers: Sequence[str],
                   outputs: Sequence[str] | None = None, *, name: str = "gasplant",
                   bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.1,
                   jit: bool = True, mask_nonconverged: bool = True, **kwargs: Any) -> Block:
    """A :class:`~difflow.planning.Block` for a gas plant column.

    Args:
        column: The :class:`~difflow_refinery.gasplant.units.GasPlantColumn`.
        feeds: Its feed streams (``{"F_<comp>": mol/s, "T": K, "P": Pa}``),
            in the column's ``feed_trays`` order: the linearisation point.
        levers: Lever names (see the module docstring), e.g.
            ``["distillate.x.C5+", "bottoms.rvp", "top.P", "feed.mol"]``.
        outputs: Output names; default
            :meth:`GasPlantPlanningModel.default_outputs`.
        name: Block name.
        bounds: ``{lever: (lo, hi)}`` in planner units; otherwise
            ``u0 (1 -/+ span)``.
        span: Default relative half-width of a lever's bounds.
        jit: Compile the block (the column's solve is compiled anyway).
        mask_nonconverged: NaN outputs when the column does not converge.
            Leave it on.
        **kwargs: Passed to :class:`~difflow.planning.Block`.

    Returns:
        A :class:`~difflow.planning.Block`; ``block.fn`` is the
        :class:`GasPlantPlanningModel`.
    """
    model = GasPlantPlanningModel(column, feeds, levers, outputs, mask_nonconverged)
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
    metadata.setdefault("source", "difflow_refinery.gasplant.gasplant_block")
    metadata.setdefault("u_units", [planner_units(n) for n in model.levers])
    metadata.setdefault("y_units", [model.output_units[n] for n in model.outputs])
    return Block(name=name, fn=model, u_names=list(model.levers), y_names=list(model.outputs),
                 lb=lb, ub=ub, u0=model.u0, jit=jit, metadata=metadata, **kwargs)
