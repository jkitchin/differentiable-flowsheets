r"""Closing the header on the hydrotreaters: makeup purity fed back into the units.

The hydrotreaters' makeup demand depends on the purity they receive (a
leaner makeup brings in methane that the purge must carry out, and its H2
with it), and with a swing source on the header the purity depends on the
demands. :func:`close_hydrotreater_loop` solves that loop by substitution
on the purity:

1. balance the network with the consumers' current demands;
2. solve every hydrotreater with ``HydrotreaterParams.makeup`` set to its
   header's composition (folded onto the unit's gases,
   :func:`~difflow_refinery.hydrogen.network.fold_composition`);
3. put each unit's ``h2.makeup`` back on its :class:`Consumer`, with a
   secant ``d_demand_d_purity`` once two passes at different purities exist;
4. re-balance; stop when no header purity moved by more than ``tol``.

With only fixed supplies (purge as the swing) the purity does not depend on
the demands, so pass 1 already closes the loop: one solve per hydrotreater.
With an import or H2 plant as the swing, a pass or two more (each a re-solve
of a compiled unit, about a second).

AD mode
-------
The loop itself is concrete Python (the hydrotreater's ``makeup`` is a
dict of floats, read by ``makeup_vector`` with ``float()``, so it is not a
traced input of :meth:`Hydrotreater.solve`). The returned network carries
each unit's response as a LINEAR one, ``d_j(y) = d_j0 + (dd_j/dy)(y - y_j)``
(a delta-base model, cf. :mod:`difflow.planning`), so
``result.network.solve()`` is differentiable in either mode with the units'
purity response included to first order. A producer's gradient (e.g. the
reformer's, forward mode -- its beds are ``diffrax.ForwardMode``) composes
with it: rebuild the producer under ``jax.jacfwd`` and solve
``result.network.replace(producers=...)``. Pass ``response_step`` to get the
slope from one extra solve per unit when the loop itself does not produce
two purities.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Mapping

from difflow_refinery.hydrogen.network import Consumer, H2NetworkResult, HydrogenNetwork


class HydrogenLoopWarning(UserWarning):
    """The header/hydrotreater loop did not close within ``max_passes``, or a unit did not converge."""


@dataclass
class HydrogenLoopResult:
    """A closed header/hydrotreater loop.

    Attributes:
        network: The :class:`HydrogenNetwork` with each consumer's demand (and
            its linear purity response) from the last unit solves.
        header: Its balanced :class:`H2NetworkResult`.
        units: ``{consumer: HydrotreaterResult}`` at the purity in ``header``.
        params: ``{consumer: HydrotreaterParams}`` they were solved with
            (``makeup`` = the header's composition).
        passes: Hydrotreater solve passes used.
        history: Per pass ``{consumer: (purity used, h2.makeup)}``.
        converged: Whether every purity moved less than ``tol`` on the last pass
            and every unit converged.
    """

    network: HydrogenNetwork
    header: H2NetworkResult
    units: dict
    params: dict
    passes: int
    history: list
    converged: bool

    def table(self) -> str:
        lines = [self.header.table()]
        for c, r in self.units.items():
            o = r.outputs
            lines.append(f"  {c}: makeup purity in {100 * self.params[c].makeup.get('hydrogen', 0.0):.2f} mol%, "
                         f"recycle purity {100 * float(o['recycle.h2_purity']):.1f} %, reactor pH2 "
                         f"{float(o['reactor.pH2_in']) / 1e5:.1f} bar, product S {float(o['product.S_wppm']):.1f} wppm")
        return "\n".join(lines)


def close_hydrotreater_loop(network: HydrogenNetwork, hydrotreaters: Mapping, *, max_passes: int = 6,
                            tol: float = 1e-6, response_step: float | None = None,
                            warn: bool = True) -> HydrogenLoopResult:
    """Feed the header's makeup purity back into the hydrotreaters until it stops moving.

    Args:
        network: The :class:`HydrogenNetwork`; every key of ``hydrotreaters``
            names one of its consumers (their ``h2_demand`` is a starting guess).
        hydrotreaters: ``{consumer: (Hydrotreater, feed, HydrotreaterParams)}``.
        max_passes: Most hydrotreater solve passes.
        tol: Purity tolerance (mole fraction).
        response_step: If set, each unit is solved once more at purity
            ``y - response_step`` after the loop closes, for its
            ``d_demand_d_purity`` (otherwise a secant over the passes, or zero
            when the loop closed in one pass).
        warn: Warn on non-convergence.
    """
    names = [c.name for c in network.consumers]
    missing = [k for k in hydrotreaters if k not in names]
    if missing:
        raise ValueError(f"{missing} are not consumers of the network ({names})")
    net = network
    res = net.solve()
    history, units, params = [], {}, {}
    prev = {}
    converged = False
    passes = 0
    ok_units = True
    for passes in range(1, max_passes + 1):
        used, rec = {}, {}
        for name, (unit, feed, p0) in hydrotreaters.items():
            comp = res.makeup_composition(name, unit.layout.gases)
            p = dataclasses.replace(p0, makeup=comp)
            r = unit.solve(feed, params=p, warn=warn)
            ok_units = ok_units and bool(r.converged)
            y = float(res.outputs[f"{name}.purity"])
            d = float(r.outputs["h2.makeup"])
            units[name], params[name] = r, p
            used[name] = y
            rec[name] = (y, d)
            slope = 0.0
            if name in prev and abs(prev[name][0] - y) > 1e-12:
                slope = (d - prev[name][1]) / (y - prev[name][0])
            net = net.with_consumer(name, h2_demand=d, purity_ref=y, d_demand_d_purity=slope)
            prev[name] = (y, d)
        history.append(rec)
        res = net.solve()
        moved = max(abs(float(res.outputs[f"{n}.purity"]) - used[n]) for n in used)
        if moved < tol:
            converged = True
            break
    if response_step is not None:
        for name, (unit, feed, p0) in hydrotreaters.items():
            _, d0 = prev[name]
            comp = dict(params[name].makeup)
            h = comp.get("hydrogen", 0.0)
            imp = 1.0 - h
            if imp > 0.0:   # leaner makeup, impurities in the same proportions
                comp = {k: (h - response_step if k == "hydrogen" else v * (imp + response_step) / imp)
                        for k, v in comp.items()}
            else:
                comp = {"hydrogen": h - response_step, "methane": response_step}
            r = unit.solve(feed, params=dataclasses.replace(params[name], makeup=comp), warn=warn)
            slope = (d0 - float(r.outputs["h2.makeup"])) / response_step
            net = net.with_consumer(name, d_demand_d_purity=slope)
        res = net.solve()
    converged = converged and ok_units
    if warn and not converged:
        warnings.warn(f"hydrogen loop: {'a unit did not converge' if not ok_units else 'purity still moving'} "
                      f"after {passes} passes", HydrogenLoopWarning, stacklevel=2)
    return HydrogenLoopResult(network=net, header=res, units=units, params=params, passes=passes,
                              history=history, converged=converged)


__all__ = ["close_hydrotreater_loop", "HydrogenLoopResult", "HydrogenLoopWarning", "Consumer"]
