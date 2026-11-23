"""Delayed coking (#534): correlation yields on pooled residue feeds.

A library, not a palette operation. See :mod:`difflow_refinery.coking.unit`
for the model and which constants are sourced and which are ILLUSTRATIVE.

>>> from difflow_refinery.coking import CokerFeed, DelayedCoker
>>> vr = CokerFeed.from_properties(8.0, sg=1.03, sulfur_wt=3.5, ccr_wt=22.0)
>>> slurry = CokerFeed.from_properties(2.0, sg=1.08, sulfur_wt=1.2, ccr_wt=6.0)
>>> out = DelayedCoker().solve([vr, slurry])
>>> abs(float(out["balance.mass"])) < 1e-12
True
"""

from difflow_refinery.coking.planning import DEFAULT_OUTPUTS, coker_block
from difflow_refinery.coking.unit import (CCR_RANGE, PRODUCTS, CokerFeed, CokerParams,
                                          CokerRangeWarning, DelayedCoker)

__all__ = ["CCR_RANGE", "DEFAULT_OUTPUTS", "PRODUCTS", "CokerFeed", "CokerParams",
           "CokerRangeWarning", "DelayedCoker", "coker_block"]
