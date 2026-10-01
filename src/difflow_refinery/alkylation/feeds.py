"""Example olefin feeds for the alkylation unit.

ILLUSTRATIVE compositions, not taken from a source: a mixed C3/C4 olefin
stream of the kind an FCC gas plant sends to alkylation (propylene, the four
butenes, some amylenes, with the isobutane, n-butane and propane that ride
along) and a C4-only cut. Real feeds are unit-specific; the FCC of #308 will
emit its own stream in this same shape -- a plain difflow stream of real
species flows (mol/s) -- and plugs in where these do.
"""

from __future__ import annotations

from difflow.streams import Stream

from difflow_refinery.alkylation.reactor import feed_stream

#: Mixed C3/C4 FCC olefins, mole fractions (illustrative).
C3C4_COMPOSITION: dict[str, float] = {
    "propane": 0.08, "propylene": 0.25, "isobutane": 0.22, "n_butane": 0.07,
    "1_butene": 0.07, "isobutylene": 0.09, "cis_2_butene": 0.07,
    "trans_2_butene": 0.10, "isopentane": 0.02, "1_pentene": 0.01,
    "2_methyl_2_butene": 0.02,
}

#: C4-only olefin cut, mole fractions (illustrative).
C4_COMPOSITION: dict[str, float] = {
    "propane": 0.01, "isobutane": 0.33, "n_butane": 0.10, "1_butene": 0.12,
    "isobutylene": 0.15, "cis_2_butene": 0.11, "trans_2_butene": 0.16,
    "isopentane": 0.02,
}


def _scaled(comp: dict[str, float], total: float, T: float, P: float) -> Stream:
    s = sum(comp.values())
    return feed_stream({k: total * v / s for k, v in comp.items()}, T, P)


def c3c4_olefin_feed(total: float = 100.0, T: float = 311.0, P: float = 6.0e5) -> Stream:
    """Mixed C3/C4 FCC olefin feed, ``total`` mol/s (illustrative composition)."""
    return _scaled(C3C4_COMPOSITION, total, T, P)


def c4_olefin_feed(total: float = 100.0, T: float = 311.0, P: float = 6.0e5) -> Stream:
    """C4-only olefin feed, ``total`` mol/s (illustrative composition)."""
    return _scaled(C4_COMPOSITION, total, T, P)


__all__ = ["C3C4_COMPOSITION", "C4_COMPOSITION", "c3c4_olefin_feed", "c4_olefin_feed"]
