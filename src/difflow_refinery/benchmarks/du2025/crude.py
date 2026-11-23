"""The benchmark's crude units as swing-cut blocks (#531).

:func:`cut_table` builds a :class:`~difflow_refinery.swing_cut.SwingCutTable`
from the published ``y``, ``FQcut``, ``phi`` and ``SC`` of one crude unit.
:func:`~difflow_refinery.swing_cut.swing_cut_block` on that table, with the
published crude rates and ``phi``, reproduces the crude unit's outputs at the
published solution (Eqs. 4-5).
"""

from __future__ import annotations

import numpy as np

from difflow_refinery.benchmarks.du2025.data import Du2025Data
from difflow_refinery.swing_cut import SwingCutTable


def cut_table(data: Du2025Data, unit: str) -> tuple[SwingCutTable, np.ndarray]:
    """The published cut table of crude unit ``unit`` and its ``phi``.

    Args:
        data: The benchmark case.
        unit: A crude unit, e.g. ``"UCDU0"``.

    Returns:
        ``(table, phi)``: crudes are the unit's inlet streams (``FQcut``'s
        index), cuts light to heavy by mean SG, products the unit's outlets,
        properties every ``FQcut`` index the benchmark blends (SPG, ``Qw``,
        ``Qv``; SG under the data's SPG index), and
        ``phi`` in ``table.swings`` order.
    """
    batches = sorted({m for (u, m, s) in data.sets["IM"] if u == unit})
    if len(batches) != 1:
        raise NotImplementedError("one batch per crude unit")
    m = batches[0]
    y, fq, phi = data.param("y"), data.param("FQcut"), data.param("phi")
    spg = data.elements("SPG")[0]
    crudes = tuple(s for (u, mm, s) in data.sets["IM"] if (u, mm) == (unit, m))
    products = tuple(s for (u, mm, s) in data.sets["OM"] if (u, mm) == (unit, m))
    cuts = sorted({k[2] for k in y if k[:2] == (unit, m)})
    cuts = tuple(sorted(cuts, key=lambda c: np.mean([fq[(m, c, f, spg)] for f in crudes])))
    yl = np.array([[y.get((unit, m, c, f), 0.0) for c in cuts] for f in crudes])
    # only properties the benchmark blends (SPG, Qw, Qv); e.g. Q45 is traced
    # but neither, so its published values are free variables
    blended = {spg} | set(data.elements("Qw")) | set(data.elements("Qv"))
    qs = sorted({k[3] for k in fq if k[0] == m and k[3] in blended},
                key=lambda q: int(q[1:]))
    props = {q: np.array([[fq.get((m, c, f, q), np.nan) for c in cuts] for f in crudes])
             for q in qs}
    own = {p: p for p in products if p in cuts}
    swings = tuple((s, s2) for (u, mm, s, s2) in data.sets["SC"] if (u, mm) == (unit, m))
    ph = np.array([phi[(unit, m, s, s2)] for (s, s2) in swings])
    volume = frozenset(q for q in data.elements("Qv"))
    return SwingCutTable(cuts, products, own, swings, crudes, yl, props, spg, volume), ph


def published_rates(table: SwingCutTable, unit: str, sol: dict) -> np.ndarray:
    """The crude rates ``FVM[unit, m, crude]`` of a published solution."""
    fvm = sol["FVM"]
    out = []
    for c in table.crudes:
        out.append(sum(v for k, v in fvm.items() if k[0] == unit and k[2] == c))
    return np.array(out)
