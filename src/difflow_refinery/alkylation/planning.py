"""The alkylation unit as a delta-base planning block.

:func:`alky_block` wraps an :class:`~difflow_refinery.alkylation.unit.AlkylationUnit`
on a given olefin feed as a :class:`difflow.planning.Block`, the way
:func:`difflow_refinery.planning.cdu_block` wraps the crude unit. Its delta
vectors are the unit's own derivatives -- through the reactor correlations,
the shortcut columns and the implicitly differentiated isobutane recycle --
and the trust-region planner refreshes them every cycle.

Levers (planner units):

====================  =========  ==============================================
``io_ratio``          ``-``      external isobutane/olefin ratio, vol/vol
``reactor.T``         ``C``      reactor temperature
``acid_strength``     ``wt%``    spent-acid strength (H2SO4)
``space_velocity``    ``1/h``    olefin space velocity
``olefin.bpd``        ``bbl/d``  fresh olefin feed rate (composition held)
====================  =========  ==============================================

Outputs are any of :data:`~difflow_refinery.alkylation.unit.OUTPUT_UNITS`;
``alkylate.bpd``, ``alkylate.RON``, ``alkylate.MON`` and ``alkylate.RVP_psi``
link to a :class:`~difflow_refinery.blending.BlendPool` component. A planner
that wants an RVP or an end-point target writes a row on the output, as for
``cdu_block``'s cut points.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import jax.numpy as jnp

from difflow.planning.block import Block
from difflow.streams import Stream

from difflow_refinery.alkylation.reactor import flows_array, stream_from_array, to_bpd
from difflow_refinery.alkylation.species import ALKYLATION_SPECIES, OLEFINS, property_array
from difflow_refinery.alkylation.unit import OUTPUT_UNITS, AlkylationUnit

_C0 = 273.15

#: Lever name -> planner unit.
ALKY_LEVERS: dict[str, str] = {
    "io_ratio": "-", "reactor.T": "C", "acid_strength": "wt%",
    "space_velocity": "1/h", "olefin.bpd": "bbl/d",
}

#: Outputs reported when ``outputs`` is not given.
DEFAULT_OUTPUTS = ("alkylate.bpd", "alkylate.MON", "alkylate.RON", "alkylate.RVP_psi",
                   "isobutane.makeup_bpd", "acid.klb_d", "dib.reboiler",
                   "refrigeration.duty")

_V60 = property_array(ALKYLATION_SPECIES, "v60")
_OLEFIN_IDX = jnp.asarray([ALKYLATION_SPECIES.index(o) for o in OLEFINS])


def _base_value(unit: AlkylationUnit, feed: Stream, lever: str) -> float:
    p = unit.params
    if lever == "io_ratio":
        return float(p.makeup.io_ratio)
    if lever == "reactor.T":
        return float(p.reactor.T) - _C0
    if lever == "acid_strength":
        return float(p.reactor.acid_strength)
    if lever == "space_velocity":
        return float(p.reactor.space_velocity)
    if lever == "olefin.bpd":
        F = flows_array(feed)
        return float(to_bpd(jnp.sum(F[_OLEFIN_IDX] * _V60[_OLEFIN_IDX])))
    raise KeyError(f"unknown lever {lever!r}; levers are {sorted(ALKY_LEVERS)}")


def alky_block(unit: AlkylationUnit, feed: Stream, levers: Sequence[str] = ("io_ratio", "reactor.T"),
               outputs: Sequence[str] | None = None, *, name: str = "alky",
               bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.1,
               jit: bool = False) -> Block:
    """A :class:`~difflow.planning.Block` for an alkylation unit on ``feed``.

    Args:
        unit: The unit; its params are the base point.
        feed: The fresh olefin feed (a difflow stream of alkylation species).
        levers: Lever names (:data:`ALKY_LEVERS`).
        outputs: Output names (:data:`~difflow_refinery.alkylation.unit.OUTPUT_UNITS`);
            default :data:`DEFAULT_OUTPUTS`.
        name: Block name.
        bounds: ``{lever: (lo, hi)}`` in planner units; a lever without one
            gets ``u0 (1 -/+ span)``.
        span: Default relative half-width of a lever's bounds.
        jit: Compile the block. Off by default: compiling the traced
            recycle with three Peng-Robinson shortcut columns takes minutes.

    Returns:
        The block; ``metadata["u_units"]``/``["y_units"]`` carry the units.
    """
    levers = list(levers)
    outputs = list(outputs) if outputs is not None else list(DEFAULT_OUTPUTS)
    for lv in levers:
        if lv not in ALKY_LEVERS:
            raise KeyError(f"unknown lever {lv!r}; levers are {sorted(ALKY_LEVERS)}")
    for out in outputs:
        if out not in OUTPUT_UNITS:
            raise KeyError(f"unknown output {out!r}; outputs are {sorted(OUTPUT_UNITS)}")
    u0 = [_base_value(unit, feed, lv) for lv in levers]
    bounds = dict(bounds or {})
    lb = [bounds.get(lv, (v * (1 - span), v * (1 + span)))[0] for lv, v in zip(levers, u0)]
    ub = [bounds.get(lv, (v * (1 - span), v * (1 + span)))[1] for lv, v in zip(levers, u0)]
    F0 = flows_array(feed)
    base_bpd = _base_value(unit, feed, "olefin.bpd")
    p0 = unit.params

    def fn(u):
        reactor, makeup, F = p0.reactor, p0.makeup, F0
        for i, lv in enumerate(levers):
            if lv == "io_ratio":
                makeup = replace(makeup, io_ratio=u[i])
            elif lv == "reactor.T":
                reactor = replace(reactor, T=u[i] + _C0)
            elif lv == "acid_strength":
                reactor = replace(reactor, acid_strength=u[i])
            elif lv == "space_velocity":
                reactor = replace(reactor, space_velocity=u[i])
            elif lv == "olefin.bpd":
                F = F0 * (u[i] / base_bpd)
        res = AlkylationUnit(replace(p0, reactor=reactor, makeup=makeup)).solve(
            stream_from_array(F, feed["T"], feed["P"]), on_nonconvergence="ignore")
        return jnp.stack([jnp.asarray(res.outputs[o], dtype=jnp.float64) for o in outputs])

    return Block(name=name, fn=fn, u_names=levers, y_names=outputs, lb=lb, ub=ub, u0=u0,
                 jit=jit, metadata={"u_units": {lv: ALKY_LEVERS[lv] for lv in levers},
                                    "y_units": {o: OUTPUT_UNITS[o] for o in outputs}})


__all__ = ["ALKY_LEVERS", "DEFAULT_OUTPUTS", "alky_block"]
