"""The benchmark's anonymized property indices, identified (#532).

The Du et al. (2025) data name properties ``Q0``-``Q169`` and never say what
they are. :data:`PROPERTY_MAP` records an identification for every index
the Case 1 data use, the evidence for it, and a confidence:

* ``"high"`` -- a defining set (``SPG``, ``Qp``), an exact arithmetic
  relation between indices, or a product specification only one property
  matches (e.g. gasoline sulfur max 0.001 wt%);
* ``"medium"`` -- value ranges and trends along the boiling range agree with
  one property and the specifications do not contradict it;
* ``"low"`` -- a plausible guess;
* ``"unknown"`` -- recorded so the index is accounted for, not identified.

The evidence was read from the Case 1 files (cut properties ``FQcut`` from
light to heavy cut, fixed values ``FQ0``, specifications ``FQMin``/``FQMax``/
``FQBMin``/``FQBMax``). It has not been confirmed by the benchmark's authors.

``blend_key`` names the matching quality of
:data:`difflow_refinery.blending.PROPERTY_RULES`, with ``to_blend`` the
factor from the benchmark's unit to that one. Note that the benchmark blends
every property except SG by MASS (``Qw``), whereas difflow's blend pool
blends aromatics, olefins and benzene by volume and octane by RT-70: the
same index, a different rule.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropertyInfo:
    """One benchmark property index, identified.

    Attributes:
        index: ``"Q0"`` ... ``"Q169"``.
        name: What the index is taken to be.
        unit: Its unit in the benchmark data.
        confidence: ``"high"``, ``"medium"``, ``"low"`` or ``"unknown"``.
        evidence: Why.
        blend_key: Matching :data:`~difflow_refinery.blending.PROPERTY_RULES`
            key, or ``None``.
        to_blend: Factor from ``unit`` to the ``blend_key`` unit.
        assay_field: Matching :class:`~difflow_refinery.assay.Assay`
            contaminant field (``sulfur_wt``, ``nitrogen_wppm``, ``ccr_wt``,
            ``nickel_vanadium_wppm``), or ``None``.
    """

    index: str
    name: str
    unit: str
    confidence: str
    evidence: str
    blend_key: str | None = None
    to_blend: float = 1.0
    assay_field: str | None = None


_P = PropertyInfo

#: Every index the Case 1 data use, identified. See the module docstring.
PROPERTY_MAP: dict[str, PropertyInfo] = {p.index: p for p in (
    _P("Q168", "specific gravity", "-", "high",
       "the SPG set; 0.54 (light naphtha) to 1.17; jet/diesel max 0.87",
       "SG", 1.0),
    _P("Q169", "component share of a virtual batch", "%", "high",
       "the only Qp index; w = 100 for the component; FCC/hydrotreater feed "
       "residue-ratio limits 5-94 %"),
    _P("Q0", "sulfur", "wt%", "high",
       "gasoline max 0.001 (10 ppm), jet/diesel products max 0.001-0.18; "
       "rises 0.03 -> 1.7 from light naphtha to vacuum residue",
       "S_ppm", 1e4, "sulfur_wt"),
    _P("Q2", "research octane number", "-", "high",
       "gasoline min 92 / 95; reformate fixed 102; on naphtha cuts only",
       "RON", 1.0),
    _P("Q4", "motor octane number", "-", "high",
       "min 87 / 90 alongside Q2; always below Q2 on the same stream",
       "MON", 1.0),
    _P("Q13", "benzene", "vol%", "high",
       "gasoline max 0.7 (China VI: 0.8); fixed 0-5 on gasoline components",
       "benzene_vol", 1.0),
    _P("Q51", "olefins", "vol%", "high",
       "gasoline max 13.5 (China VI-B: 15); FCC gasoline fixed 27.5-38, "
       "reformate/alkylate 0",
       "olefins_vol", 1.0),
    _P("Q11", "aromatics", "vol%", "medium",
       "gasoline max 34 (China VI: 35); naphtha cuts 4-27; diesel products "
       "max 7 and 19, which fit polycyclic and total aromatics -- one index "
       "may carry both", "aromatics_vol", 1.0),
    _P("Q52", "MTBE / oxygenates", "vol%", "medium",
       "gasoline max 14; the MTBE stream fixed 99.6; 0 elsewhere"),
    _P("Q14", "cetane index", "-", "high",
       "diesel/jet products min 47 / 49; rises 24 -> 50 kerosene to gas oil"),
    _P("Q18", "total nitrogen", "wppm", "high",
       "rises 200 -> 8400 from gas oil to vacuum residue; hydrocracker feed "
       "max 3000", "N_ppm", 1.0, "nitrogen_wppm"),
    _P("Q19", "basic nitrogen", "wppm", "medium",
       "one gas-oil swing cut only, about 1/3 of Q18 there"),
    _P("Q21", "Conradson carbon residue", "wt%", "high",
       "0.01-0.02 on distillates, 17-28 on vacuum residue; coker feed max "
       "30, residue-hydrotreater feed max 21; delta-base unit Upd3 moves "
       "its yields with it at a base of 18.7 on a SG 1.01 feed",
       "CCR_wt", 1.0, "ccr_wt"),
    _P("Q22", "nickel", "wppm", "high",
       "Q22 + Q23 = Q25 on every cut; residue 50-115", None, 1.0),
    _P("Q23", "vanadium", "wppm", "high",
       "Q22 + Q23 = Q25 on every cut; residue 4-252", None, 1.0),
    _P("Q25", "nickel + vanadium", "wppm", "high",
       "equals Q22 + Q23 on every cut", None, 1.0, "nickel_vanadium_wppm"),
    _P("Q5", "naphthenes + 2 aromatics (reforming potential)", "vol%", "medium",
       "naphtha cuts only, 25-78, higher for heavy naphtha; reformer feed "
       "min 28"),
    _P("Q10", "paraffins (probable)", "vol%", "low",
       "naphtha cuts only, 30-79; no specification"),
    _P("Q6", "Reid vapour pressure (gasoline components)", "kPa", "low",
       "fixed 60-77 on gasoline components, but ~1.5 on the one naphtha "
       "cut that carries it -- the two may not be the same quantity"),
    _P("Q8", "unidentified", "-", "unknown", "traced on 7 streams, fixed 18.2"),
    _P("Q15", "unidentified (viscosity-like)", "-", "unknown",
       "middle distillates, rises 0.01 -> 5.4 with boiling point"),
    _P("Q16", "unidentified, = 0.85 x Q15", "-", "unknown",
       "Q16 / Q15 = 0.850 on every cut: a second scale of Q15"),
    _P("Q17", "unidentified", "-", "unknown", "naphtha/kerosene cuts, 0.03-0.09"),
    _P("Q36", "unidentified", "-", "unknown", "kerosene/diesel cuts, -0.04-0.99"),
    _P("Q37", "unidentified", "-", "unknown",
       "transferred through hydrotreaters with Q14, Q15, Q17, Q51"),
    _P("Q42", "unidentified", "-", "unknown", "fixed 3.0; 5 streams"),
    _P("Q44", "unidentified", "-", "unknown", "naphtha/kerosene cuts, 0.18-1.24"),
    _P("Q45", "unidentified (not weight-blended)", "-", "unknown",
       "the one traced index outside Qw; values 0.15 to 2.7e12, not monotone"),
    _P("Q47", "viscosity blending index (probable)", "-", "low",
       "gas oils and residue, rises 0.3 -> 2.4 with boiling point; may be "
       "negative, as an index is"),
    _P("Q53", "unidentified", "-", "unknown", "one stream, fixed 1.0"),
)}


def describe(index: str) -> str:
    """``"Q0: sulfur (wt%, high)"``; ``"Q99: not used"`` for an unlisted one."""
    p = PROPERTY_MAP.get(index)
    if p is None:
        return f"{index}: not used in Case 1"
    return f"{index}: {p.name} ({p.unit}, {p.confidence})"


def by_name() -> dict[str, str]:
    """``{blend_key: index}`` for the identified properties with a blend rule."""
    return {p.blend_key: p.index for p in PROPERTY_MAP.values() if p.blend_key}


def named(values: dict[str, float], *, min_confidence: str = "medium") -> dict[str, float]:
    """``{index: value}`` -> ``{name: value}`` for identified indices.

    Args:
        values: A stream's properties keyed by index, e.g. from the
            published solution's ``FQ`` or the data's ``FQ0``.
        min_confidence: Drop indices identified below this level
            (``"high"``, ``"medium"``, ``"low"``, ``"unknown"``).

    Example:
        >>> named({"Q0": 0.5, "Q168": 0.9, "Q8": 3.0})
        {'sulfur': 0.5, 'specific gravity': 0.9}
    """
    order = ("high", "medium", "low", "unknown")
    keep = set(order[:order.index(min_confidence) + 1])
    return {PROPERTY_MAP[q].name: v for q, v in values.items()
            if q in PROPERTY_MAP and PROPERTY_MAP[q].confidence in keep}
