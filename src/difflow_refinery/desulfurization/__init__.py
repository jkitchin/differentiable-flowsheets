"""Gasoline desulfurization (#535): S Zorb-type reactive adsorption.

A library, not a palette operation. Every rate and sorbent constant is
ILLUSTRATIVE; see :mod:`difflow_refinery.desulfurization.szorb`.
"""

from difflow_refinery.desulfurization.szorb import (FCC_GASOLINE_SULFUR_SPLIT, SZorbFeed,
                                                    SZorbParams, SZorbRangeWarning, SZorbUnit)

__all__ = ["FCC_GASOLINE_SULFUR_SPLIT", "SZorbFeed", "SZorbParams", "SZorbRangeWarning",
           "SZorbUnit"]
