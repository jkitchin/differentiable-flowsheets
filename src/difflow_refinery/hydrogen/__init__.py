"""The refinery hydrogen network (#329): header, producers, consumers, PSA, purge, import/export.

A library, not a palette operation. See :mod:`.network` for the equations,
:mod:`.loop` for closing the header on the hydrotreaters, and
:mod:`.planning` for ``h2_block``.

>>> import difflow_refinery.hydrogen as h2
>>> net = h2.HydrogenNetwork(
...     producers=[h2.Producer.of_purity("reformer", 120.0, 0.88)],
...     consumers=[h2.Consumer("nht", 40.0, min_purity=0.85),
...                h2.Consumer("dht", 70.0, min_purity=0.85)],
...     headers=[h2.Header("main", swing=[h2.Import(purity=0.999)])])
>>> res = net.solve()
>>> round(float(res.outputs["h2.surplus"]), 6)
10.0
"""

from difflow_refinery.hydrogen.network import (
    HEADER_GASES, NM3_H_PER_MOL_S, OUTPUT_UNITS, PSA, PSA_TARGET_PASSES, REFORMER_SPECIES, Consumer,
    H2NetworkResult, H2Plant, Header, HydrogenNetwork, Import, Producer, dropped_flows, fold_composition,
    gas_vector, output_units)
from difflow_refinery.hydrogen.loop import HydrogenLoopResult, HydrogenLoopWarning, close_hydrotreater_loop

__all__ = [
    "HEADER_GASES", "NM3_H_PER_MOL_S", "OUTPUT_UNITS", "PSA", "PSA_TARGET_PASSES", "REFORMER_SPECIES",
    "Consumer", "H2NetworkResult", "H2Plant", "Header", "HydrogenNetwork", "Import", "Producer",
    "dropped_flows", "fold_composition", "gas_vector", "output_units", "HydrogenLoopResult",
    "HydrogenLoopWarning", "close_hydrotreater_loop", "h2_block",
]


def h2_block(*args, **kwargs):
    """:func:`difflow_refinery.hydrogen.planning.h2_block` (imported lazily: it needs ``difflow.planning``)."""
    from difflow_refinery.hydrogen.planning import h2_block as _b
    return _b(*args, **kwargs)
