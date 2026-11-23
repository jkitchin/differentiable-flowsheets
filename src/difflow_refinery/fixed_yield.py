"""Property-carrying fixed-yield unit blocks (#537).

Some refinery units do not yet need a mechanistic model in a plant study:
a solvent deasphalter, an aromatics extraction unit, a gas fractionation
unit, an MTBE unit. Each one is a set of base yields plus property
transfers, the fixed-yield form of Du et al. (2025), Eqs. 8 and 11. A
:class:`FixedYieldBlock` evaluates that form on :class:`PropertyStream`
s (a mass rate plus named properties), so the block sits between rigorous
units.

* **Flows** (Eq. 8). Each outlet gets ``gamma_out * sum(inlet mass) /
  sum(gamma_in)``. The inlets are expected in the ``gamma_in`` proportions,
  and :meth:`FixedYieldBlock.feed_ratio_error` reports how far they are
  from them. Where the yields do not sum to the inlet coefficients, as in
  the benchmark (``Upf19``'s sum to 0.99), the difference is an explicit
  ``loss`` outlet, so the mass balance closes.
* **Properties** (Eq. 11). An outlet property is either transferred,
  ``alpha * the inlet property`` from a named inlet, or fixed to a value.
  An outlet property that is neither is not reported: the benchmark leaves
  it free, and the block does not invent one.

:meth:`PropertyStream.to_blend_component` hands a stream to
:class:`~difflow_refinery.blending.BlendPool`, and
:meth:`PropertyStream.to_coker_feed` to the coker. The property names are
the blend pool's (``SG``, ``RON``, ``MON``, ``S_ppm``, ``olefins_vol``,
...) plus ``S_wt``, ``CCR_wt``, ``NiV_wppm`` and ``N_wppm``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PropertyStream:
    """A lumped stream: mass rate (kg/s) and ``{property: value}``."""

    mass: Array
    props: dict = field(default_factory=dict)

    def to_blend_component(self, name: str):
        """A :class:`~difflow_refinery.blending.BlendComponent` (needs ``SG``)."""
        from difflow_refinery.blending import PROPERTY_RULES, BlendComponent
        if "SG" not in self.props:
            raise ValueError("a blend component needs SG")
        p = {k: v for k, v in self.props.items() if k in PROPERTY_RULES}
        if "S_ppm" not in p and "S_wt" in self.props:
            p["S_ppm"] = jnp.asarray(self.props["S_wt"]) * 1e4
        return BlendComponent.from_properties(name, **p)

    def to_coker_feed(self):
        """A :class:`~difflow_refinery.coking.CokerFeed` (needs SG, S_wt, CCR_wt)."""
        from difflow_refinery.coking import CokerFeed
        missing = [k for k in ("SG", "S_wt", "CCR_wt") if k not in self.props]
        if missing:
            raise ValueError(f"a coker feed needs {missing}")
        p = self.props
        return CokerFeed.from_properties(self.mass, sg=p["SG"], sulfur_wt=p["S_wt"],
                                         ccr_wt=p["CCR_wt"], nitrogen_wppm=p.get("N_wppm", 0.0),
                                         nickel_vanadium_wppm=p.get("NiV_wppm", 0.0))


@dataclass(frozen=True)
class FixedYieldBlock:
    """Base yields plus property transfer.

    Attributes:
        name: Label.
        inlets: ``{inlet: gamma_in}``.
        outlets: ``{outlet: gamma_out}``.
        transfer: ``{outlet: {property: (inlet, alpha)}}``.
        fixed: ``{outlet: {property: value}}``.
    """

    name: str
    inlets: Mapping[str, float]
    outlets: Mapping[str, float]
    transfer: Mapping[str, Mapping[str, tuple[str, float]]] = field(default_factory=dict)
    fixed: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.inlets or not self.outlets:
            raise ValueError("a block needs at least one inlet and one outlet")
        if any(g <= 0 for g in self.inlets.values()):
            raise ValueError("inlet coefficients must be positive")
        if any(g < 0 for g in self.outlets.values()):
            raise ValueError("outlet yields must be nonnegative")
        for o, tr in self.transfer.items():
            if o not in self.outlets:
                raise ValueError(f"transfer to unknown outlet {o!r}")
            for q, (i, _) in tr.items():
                if i not in self.inlets:
                    raise ValueError(f"{o}.{q} transfers from unknown inlet {i!r}")

    @property
    def loss(self) -> float:
        """Share of the inlet mass the yields do not account for."""
        return 1.0 - sum(self.outlets.values()) / sum(self.inlets.values())

    def feed_ratio_error(self, feeds: Mapping[str, PropertyStream]) -> Array:
        """Largest relative deviation of the inlets from the ``gamma_in`` ratios."""
        total = sum(jnp.asarray(feeds[i].mass) for i in self.inlets)
        g = sum(self.inlets.values())
        return jnp.max(jnp.stack([jnp.abs(jnp.asarray(feeds[i].mass) / total - gi / g)
                                  for i, gi in self.inlets.items()]))

    def __call__(self, feeds: Mapping[str, PropertyStream]) -> dict[str, PropertyStream]:
        """Outlet streams, plus ``"loss"`` when the yields leave some mass.

        Raises:
            KeyError: If an inlet is missing from ``feeds``.
        """
        missing = [i for i in self.inlets if i not in feeds]
        if missing:
            raise KeyError(f"{self.name}: no feed for inlet(s) {missing}")
        total = sum(jnp.asarray(feeds[i].mass, float) for i in self.inlets)
        scale = total / sum(self.inlets.values())
        out = {}
        for o, g in self.outlets.items():
            props = {q: jnp.asarray(v, float) for q, v in self.fixed.get(o, {}).items()}
            for q, (i, a) in self.transfer.get(o, {}).items():
                if q not in feeds[i].props:
                    raise KeyError(f"{self.name}: {o}.{q} transfers from {i}.{q}, which the feed lacks")
                props[q] = a * jnp.asarray(feeds[i].props[q], float)
            out[o] = PropertyStream(g * scale, props)
        loss = total - sum(s.mass for s in out.values())
        if abs(self.loss) > 1e-12:
            out["loss"] = PropertyStream(loss, {})
        return out


def balance(block: FixedYieldBlock, feeds: Mapping[str, PropertyStream],
            products: Mapping[str, PropertyStream]) -> Array:
    """Relative mass balance error (outlets incl. loss against inlets)."""
    fin = sum(jnp.asarray(feeds[i].mass) for i in block.inlets)
    return (sum(p.mass for p in products.values()) - fin) / fin


__all__: Sequence[str] = ("FixedYieldBlock", "PropertyStream", "balance")
