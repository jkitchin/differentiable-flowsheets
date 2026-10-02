"""Light-naphtha feeds for the isomerization unit.

A crude distillation unit reports its naphtha as light ends (real
species up to n-hexane, when the assay has them) plus pseudo-components
by boiling range. The isomerization reactor needs molecules: which C6 is
n-hexane and which is 2,2-dimethylbutane, how much benzene is in the cut.
A TBP assay does not say, so **the speciation here is constructed, not
measured**: the light ends pass through as themselves, and the
pseudo-components boiling below the light-naphtha cut point are split
into the C6 isomers, the naphthenes, benzene and a C7+ lump by an assumed
composition (:class:`NaphthaSpeciation`). The cut itself is ideal -- the
naphtha splitter between the CDU and the isomerization unit is not
modelled.

Two speciations are provided, both assumptions of this module and both
stated as such wherever a feed is built from them:

* :data:`PARAFFINIC` -- a light naphtha from a paraffinic crude, about
  1.5 wt% benzene in the C6 cut;
* :data:`BENZENE_RICH` -- a naphthenic crude's light naphtha, with
  about 5 wt% benzene and more cyclohexane and methylcyclopentane in the
  C6 cut. This is the feed that loads the benzene-saturation step and
  raises the reactor temperature rise.

Their numbers are in the range of the light straight-run C5/C6 analyses
quoted in refining texts (verify against a real PIONA before relying on
any of them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp

from difflow_refinery.isomerization import thermochem as tc

#: Hydrocarbon species a light-naphtha feed may carry (no hydrogen).
FEED_SPECIES: tuple[str, ...] = tuple(n for n in tc.NAMES if n != "hydrogen")


@dataclass(frozen=True)
class NaphthaSpeciation:
    """An assumed composition of the light-naphtha pseudo-components.

    Attributes:
        name: Identifier.
        description: What the composition stands for and that it is assumed.
        cut_mass: ``{species: mass fraction}`` of the pseudo-component
            material below the cut point; sums to one.
        whole_mass: ``{species: mass fraction}`` of a whole light-naphtha
            feed of this kind (light ends included), for
            :func:`constructed_feed` when no CDU is run; sums to one.
    """

    name: str
    description: str
    cut_mass: tuple
    whole_mass: tuple

    def __post_init__(self):
        for attr in ("cut_mass", "whole_mass"):
            d = dict(getattr(self, attr))
            bad = [k for k in d if k not in FEED_SPECIES]
            if bad:
                raise ValueError(f"{attr}: unknown species {bad}")
            if abs(sum(d.values()) - 1.0) > 1e-9:
                raise ValueError(f"{attr} sums to {sum(d.values())}, not 1")


PARAFFINIC = NaphthaSpeciation(
    name="paraffinic",
    description="ASSUMED light straight-run naphtha of a paraffinic crude; "
                "about 1.5 wt% benzene in the C6 cut",
    cut_mass=(("2_methylpentane", 0.20), ("3_methylpentane", 0.14),
              ("2_3_dimethylbutane", 0.03), ("2_2_dimethylbutane", 0.01),
              ("n_hexane", 0.22), ("methylcyclopentane", 0.12), ("cyclohexane", 0.07),
              ("benzene", 0.015), ("n_heptane", 0.195)),
    whole_mass=(("n_butane", 0.015), ("isobutane", 0.005), ("isopentane", 0.17),
                ("n_pentane", 0.25), ("2_2_dimethylbutane", 0.006),
                ("2_3_dimethylbutane", 0.018), ("2_methylpentane", 0.12),
                ("3_methylpentane", 0.085), ("n_hexane", 0.17),
                ("methylcyclopentane", 0.06), ("cyclohexane", 0.03), ("benzene", 0.011),
                ("n_heptane", 0.06)),
)

BENZENE_RICH = NaphthaSpeciation(
    name="benzene_rich",
    description="ASSUMED light straight-run naphtha of a naphthenic crude; "
                "about 5 wt% benzene in the C6 cut",
    cut_mass=(("2_methylpentane", 0.16), ("3_methylpentane", 0.11),
              ("2_3_dimethylbutane", 0.025), ("2_2_dimethylbutane", 0.008),
              ("n_hexane", 0.17), ("methylcyclopentane", 0.16), ("cyclohexane", 0.11),
              ("benzene", 0.05), ("n_heptane", 0.207)),
    whole_mass=(("n_butane", 0.012), ("isobutane", 0.004), ("isopentane", 0.13),
                ("n_pentane", 0.19), ("2_2_dimethylbutane", 0.005),
                ("2_3_dimethylbutane", 0.015), ("2_methylpentane", 0.10),
                ("3_methylpentane", 0.07), ("n_hexane", 0.14),
                ("methylcyclopentane", 0.11), ("cyclohexane", 0.075), ("benzene", 0.04),
                ("n_heptane", 0.109)),
)

#: The speciations by name.
SPECIATIONS: dict[str, NaphthaSpeciation] = {s.name: s for s in (PARAFFINIC, BENZENE_RICH)}


def _speciation(s) -> NaphthaSpeciation:
    return SPECIATIONS[s] if isinstance(s, str) else s


def _stream(F, T, P) -> dict:
    s = {f"F_{n}": F[i] for i, n in enumerate(tc.NAMES)}
    s.update(T=jnp.asarray(T, dtype=float), P=jnp.asarray(P, dtype=float))
    return s


def constructed_feed(speciation="paraffinic", mass_rate: float = 10.0,
                     T: float = 313.15, P: float = 5e5) -> dict:
    """A light-naphtha feed from a speciation's whole-feed composition.

    The composition is ASSUMED (see :class:`NaphthaSpeciation`), not from
    any assay; use :func:`light_naphtha_from_cdu` to start from a crude.

    Args:
        speciation: ``"paraffinic"``, ``"benzene_rich"`` or a
            :class:`NaphthaSpeciation`.
        mass_rate: Feed rate (kg/s).
        T, P: Stream conditions (K, Pa).
    """
    sp = _speciation(speciation)
    w = dict(sp.whole_mass)
    F = jnp.asarray([mass_rate * w.get(n, 0.0) / (float(tc.MW[i]) / 1e3)
                     for i, n in enumerate(tc.NAMES)])
    return _stream(F, T, P)


def light_naphtha_from_cdu(naphtha: Mapping, characterization, speciation="paraffinic",
                           cut_T: float = 358.15, T: float = 313.15,
                           P: float = 5e5) -> tuple[dict, dict]:
    """The light-naphtha cut of a CDU naphtha product, speciated.

    Light ends that are isomerization species pass through as themselves
    (methane and water are dropped). Pseudo-components with a normal
    boiling point at or below ``cut_T`` are taken into the cut -- an ideal
    naphtha splitter -- and their mass is split by the speciation's
    ``cut_mass``, an ASSUMED composition. Heavier cuts are left to the
    heavy naphtha.

    Args:
        naphtha: The CDU naphtha product stream (``F_<species>`` in mol/s).
        characterization: The :class:`~difflow_refinery.assay.Characterization`
            whose ``pseudo_names``, ``Tb`` and ``MW`` the stream's cuts are.
        speciation: ``"paraffinic"``, ``"benzene_rich"`` or a
            :class:`NaphthaSpeciation`.
        cut_T: Light-naphtha end point (K); 85 C by default.
        T, P: Conditions given to the feed stream.

    Returns:
        ``(feed stream, info)``; ``info`` has the cut's ``mass_rate``
        (kg/s), ``pseudo_mass`` speciated, the ``cuts`` taken and the
        ``speciation`` used.
    """
    sp = _speciation(speciation)
    char = characterization
    F = {n: 0.0 for n in tc.NAMES}
    for k, v in naphtha.items():
        if k.startswith("F_") and k[2:] in F and k[2:] != "hydrogen":
            F[k[2:]] += v
    taken, pseudo_mass = [], 0.0
    for i, name in enumerate(char.pseudo_names):
        if float(char.Tb[i]) <= cut_T and f"F_{name}" in naphtha:
            taken.append(name)
            pseudo_mass = pseudo_mass + naphtha[f"F_{name}"] * char.MW[i] / 1e3
    for n, w in sp.cut_mass:
        F[n] = F[n] + pseudo_mass * w / (float(tc.MW[tc.idx(n)]) / 1e3)
    Fa = jnp.stack([jnp.asarray(F[n], dtype=float) for n in tc.NAMES])
    return _stream(Fa, T, P), {"mass_rate": jnp.dot(Fa, tc.MW) / 1e3,
                               "pseudo_mass": pseudo_mass, "cuts": tuple(taken),
                               "speciation": sp.name}


def hydrocarbon_flows(stream: Mapping) -> jnp.ndarray:
    """Reactor-order flows of a stream (mol/s); missing species are 0."""
    return jnp.stack([jnp.asarray(stream.get(f"F_{n}", 0.0), dtype=float) for n in tc.NAMES])


def with_nc6_fraction(stream: Mapping, x_nc6) -> dict:
    """The stream with its n-hexane mole fraction (hydrogen excluded) set
    to ``x_nc6``, every other hydrocarbon scaled in proportion and the
    total molar flow kept. Differentiable in ``x_nc6``; this is the
    ``feed n-C6 fraction`` lever of :func:`~.planning.isom_block`."""
    F = hydrocarbon_flows(stream)
    i6, ih = tc.idx("n_hexane"), tc.idx("hydrogen")
    hc = jnp.ones_like(F).at[ih].set(0.0)
    tot = jnp.sum(F * hc)
    others = tot - F[i6]
    scale = (1.0 - x_nc6) * tot / others
    Fn = jnp.where(hc > 0, F * scale, F).at[i6].set(x_nc6 * tot)
    return _stream(Fn, stream["T"], stream["P"])


def nc6_fraction(stream: Mapping):
    """n-Hexane mole fraction of the hydrocarbons in ``stream``."""
    F = hydrocarbon_flows(stream)
    return F[tc.idx("n_hexane")] / (jnp.sum(F) - F[tc.idx("hydrogen")])


__all__ = [
    "FEED_SPECIES", "NaphthaSpeciation", "PARAFFINIC", "BENZENE_RICH", "SPECIATIONS",
    "constructed_feed", "light_naphtha_from_cdu", "hydrocarbon_flows",
    "with_nc6_fraction", "nc6_fraction",
]
