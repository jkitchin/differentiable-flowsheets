"""Section operating pHs read off the distribution curves.

An element leaves for the other phase of a counter-current section when its
factor ``D * (O/A)`` crosses one. A section therefore separates two groups
only at a pH where that crossing falls between them, and it extracts (or
strips) everything only at a pH where every element is well to one side of
it. This module turns those statements into pH values, for any extractant
whose ``D`` moves with pH, so that no circuit default depends on where a
refit happens to put an extractant's fitted window.

Why this exists. Until the 2026 operating-point audit the circuit defaults
were fixed fractions of the extractant's fitted pH window
(:func:`difflow_ree.database.default_pH`: extraction at the top, scrubbing a
quarter up, stripping at the bottom, #270). After the #270 refit those pHs
did not sit between the groups. Stripping at the bottom of D2EHPA's window
left 67 % of the Sm and 99.8 % of the Y on the barren organic of a default
``ExtractStripCircuit``, and a default ``ExtractScrubStripCircuit`` asked for
Gd/Tb/Dy/Y returned a product of 0.09 % target purity. ``GroupSeparator``
had already been moved onto D cuts (#372); this is that logic, shared and
tested, so the circuits, the design functions and the scan helpers use the
same rule.

The rule (factors are ``D * (O/A)`` at the section's own phase ratio):

=================  ==========================================================
section            condition
=================  ==========================================================
extraction (cut)   geometric-mean factor of the boundary pair = 1. The pair
                   is the least extractable target and the most extractable
                   element to reject.
scrubbing (cut)    the same pair's geometric-mean factor = 1 at the scrub
                   phase ratio (``D`` equals the scrub aqueous/organic ratio).
extraction (bulk)  the least extractable element has factor 10: everything
                   loads. Used when there is no boundary (no targets, or
                   every element is a target).
stripping          the most strongly held element to be stripped has factor
                   0.1: everything comes off.
=================  ==========================================================

A ``D`` that does not move with pH (a solvating extractant such as TBP,
driven by nitrate) has no pH cut; :func:`cut_pHs` returns ``None`` and the
caller keeps its own fallback.

The pHs are NOT clamped to the fitted window. Stripping the heavy REE from
D2EHPA needs strong acid, below the window the coefficients were fitted
over; the distribution model reports that extrapolation when the section
runs, which is the same policy :meth:`REEDistribution._check_ph_range`
applies everywhere else (clamping would silently relocate an operating
point).

One bound IS applied, to strip pHs this module chooses and only to those:
the strip liquor cannot be stronger than ``max_strip_acid`` (default
:data:`MAX_STRIP_ACID`, 6 M, pH ``-log10(6) = -0.78`` on the concentration
scale the correlations use). The cut rule put heavy-REE stripping on D2EHPA
at pH -1.17 (14.6 M H+ at strip O/A 2), beyond any real strip liquor and more
than a pH unit outside the [0, 2] window the coefficients were fitted over.
When the cut asks for stronger acid the strip runs at the floor,
:class:`OperatingPHs` says so (``strip_acid_limited``, ``strip_cut``),
:func:`strip_retention` gives what the Kremser solve leaves on the barren
organic there, and the circuits and design helpers warn with
:class:`StripAcidLimitWarning`. This is a stopgap until D2EHPA is refitted
down to strong acid (#384). A pH the caller gives is never floored.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

#: Target ``D * (O/A)`` factor for each kind of section (see module docstring).
CUT_FACTOR = {
    "extraction": 1.0,       # boundary pair, geometric mean
    "scrubbing": 1.0,        # boundary pair, geometric mean, scrub ratio
    "bulk_extraction": 10.0,  # least extractable element
    "stripping": 0.1,        # most strongly held element
}

#: pH bracket for the bisection. Wide on purpose: the answer is reported, not
#: clamped, and a pH outside the fitted window is a finding, not an error.
PH_BRACKET = (-4.0, 10.0)


#: Strongest strip liquor (M H+) a strip pH chosen by the cut rule may call
#: for. 6 M is about the strongest HCl/HNO3 strip used in practice; the
#: plugin's pH is -log10 of the proton concentration, so the floor is
#: pH -0.778.
MAX_STRIP_ACID = 6.0


class OperatingPointWarning(UserWarning):
    """The requested split has no clean pH cut (for example the targets are
    not the more extractable group, so no single pH extracts them and rejects
    the rest)."""


class StripAcidLimitWarning(UserWarning):
    """A strip chosen by the cut rule needs acid stronger than
    ``max_strip_acid``, so it runs at the acid floor and leaves REE on the
    barren organic (or a design cannot reach its target there)."""


def strip_pH_floor(max_strip_acid: float = MAX_STRIP_ACID) -> float:
    """The lowest strip pH the cut rule may choose: ``-log10(max_strip_acid)``.

    Args:
        max_strip_acid: Strongest strip liquor allowed (M H+).

    Returns:
        The pH floor on the concentration scale.

    Raises:
        ValueError: If ``max_strip_acid`` is not positive.

    Example:
        >>> round(strip_pH_floor(6.0), 3)
        -0.778
    """
    if not max_strip_acid > 0:
        raise ValueError(f"max_strip_acid must be positive, got {max_strip_acid}")
    return -math.log10(max_strip_acid)


def circuit_phase_ratios(
    solvent_to_feed_ratio: float,
    scrub_to_solvent_ratio: float | None,
    strip_to_solvent_ratio: float,
    extractant_conc: float,
) -> dict[str, float]:
    """The ``O/A`` each section of a circuit actually runs at.

    The circuits build their solvent as ``{diluent: F_org, extractant:
    extractant_conc * F_org}`` with ``F_org = F_aq * solvent_to_feed_ratio``.
    The organic carrier flow is the diluent's, a volume like the aqueous
    flow; the extractant entry is a charge in moles per unit volume and does
    not count (#373, it used to add ``extractant_conc`` to the organic flow,
    so a stated O/A of 1 ran at 1.5 with 0.5 M extractant). The extraction
    ``O/A`` is ``solvent_to_feed_ratio`` and the scrub and strip ``O/A`` are
    the reciprocals of their A/O ratios, whatever the extractant
    concentration. ``extractant_conc`` is kept in the signature for callers.

    Args:
        solvent_to_feed_ratio: Circuit ``solvent_to_feed_ratio``.
        scrub_to_solvent_ratio: Circuit ``scrub_to_solvent_ratio``, or None
            for a circuit without a scrub.
        strip_to_solvent_ratio: Circuit ``strip_to_solvent_ratio``.
        extractant_conc: Extractant concentration in the organic (M), on the
            extractant record's own basis: DIMER for the dimeric D2EHPA, PC88A
            and Cyanex272 (0.5 M dimer = 1.0 M formal), molecules (monomer) for
            TBP and naphthenic acid. The loading capacity is this divided by
            ``Extractant.basis_units_per_ree`` (3 for every shipped record;
            #374).

    Returns:
        ``{"extraction": O/A, "scrubbing": O/A or None, "stripping": O/A}``.

    Example:
        >>> circuit_phase_ratios(1.0, 0.2, 0.5, 0.5)["stripping"]
        2.0
    """
    return {
        "extraction": solvent_to_feed_ratio,
        "scrubbing": None if not scrub_to_solvent_ratio else 1.0 / scrub_to_solvent_ratio,
        "stripping": 1.0 / strip_to_solvent_ratio,
    }


def pH_where(
    distribution,
    elements: tuple[str, ...],
    factor: float,
    phase_ratio: float = 1.0,
    how: str = "mean",
    bracket: tuple[float, float] = PH_BRACKET,
) -> float:
    """The pH at which ``D * phase_ratio`` of ``elements`` equals ``factor``.

    ``D`` rises with pH for a cation-exchange extractant, so the crossing is
    found by bisection on ``log10 D``.

    Args:
        distribution: A :class:`~difflow_ree.equilibrium.distribution.REEDistribution`
            covering ``elements``.
        elements: The elements whose ``D`` is combined.
        factor: The target ``D * (O/A)``.
        phase_ratio: The section's ``O/A``.
        how: ``"mean"`` (geometric mean of ``D``), ``"max"`` (the largest
            ``D``) or ``"min"`` (the smallest).
        bracket: pH search interval.

    Returns:
        The pH, to about 1e-12. A crossing outside ``bracket`` returns the
        nearer end.

    Example:
        >>> from difflow_ree.equilibrium.distribution import REEDistribution
        >>> d = REEDistribution("D2EHPA", ("Nd",), on_out_of_range="ignore")
        >>> round(float(d.get_D("Nd", pH_where(d, ("Nd",), 1.0))), 6)
        1.0
    """
    import numpy as np

    if how not in ("mean", "max", "min"):
        raise ValueError(f"how must be 'mean', 'max' or 'min', got {how!r}")
    target = math.log10(factor / phase_ratio)
    reduce = {"mean": np.mean, "max": np.max, "min": np.min}[how]

    def log_D(pHs):
        D = distribution.get_D_all(pH=pHs)
        logs = np.stack([np.log10(np.maximum(np.asarray(D[e], dtype=float), 1e-300))
                         * np.ones_like(pHs) for e in elements], axis=-1)
        return reduce(logs, axis=-1)

    # Grid refinement, vectorised: one get_D_all call per pass instead of one
    # per bisection step (a scalar bisection cost ~2 s per cut, and every
    # circuit with an unset pH computes three). Each pass narrows the bracket
    # to one grid interval, 100x; five passes reach ~1e-12 in pH.
    a, b = bracket
    for n in (1401, 101, 101, 101, 101):
        grid = np.linspace(a, b, n)
        below = log_D(grid) < target
        if below.all():
            return float(b)
        if not below.any():
            return float(a)
        # D rises with pH: the last grid point still below the target
        # brackets the crossing with the next.
        k = int(np.nonzero(below)[0].max())
        k = min(k, n - 2)
        a, b = grid[k], grid[k + 1]
    return float(0.5 * (a + b))


def is_pH_driven(distribution, elements: tuple[str, ...]) -> bool:
    """Whether ``D`` of any of ``elements`` moves with pH.

    False for a solvating extractant (TBP), whose ``D`` is a function of
    nitrate, so no pH cut exists for it.
    """
    lo = distribution.get_D_all(pH=0.0)
    hi = distribution.get_D_all(pH=2.0)
    return any(
        abs(float(hi[e]) - float(lo[e])) > 1e-12 * max(abs(float(lo[e])), 1.0)
        for e in elements
    )


@dataclass(frozen=True)
class OperatingPHs:
    """Section pHs and how they were found.

    Attributes:
        extraction: Extraction pH.
        scrubbing: Scrubbing pH (equals ``extraction`` when there is no
            boundary to scrub across).
        stripping: Stripping pH.
        basis: ``"cut"`` (a target/rejected boundary) or ``"bulk"`` (extract
            and strip everything).
        boundary: The boundary pair ``(least extractable target, most
            extractable rejected)``, or None for ``"bulk"``.
        strip_acid_limited: True when the strip cut needed acid stronger
            than ``max_strip_acid`` and ``stripping`` is the acid floor.
        strip_cut: The strip pH the cut rule asked for, before the floor
            (equal to ``stripping`` when the floor did not bind).
        strip_elements: The elements the strip cut was placed on (the
            targets, or every element for ``"bulk"``).
    """

    extraction: float
    scrubbing: float
    stripping: float
    basis: str
    boundary: tuple[str, str] | None = None
    strip_acid_limited: bool = False
    strip_cut: float | None = None
    strip_elements: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, float]:
        """``{"extraction": ..., "scrubbing": ..., "stripping": ...}``."""
        return {"extraction": self.extraction, "scrubbing": self.scrubbing,
                "stripping": self.stripping}


def cut_pHs(
    extractant: str,
    elements: tuple[str, ...],
    targets: tuple[str, ...] = (),
    *,
    extraction_OA: float,
    strip_OA: float,
    scrub_OA: float | None = None,
    extractant_conc: float = 0.5,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
    distribution=None,
    max_strip_acid: float | None = MAX_STRIP_ACID,
) -> OperatingPHs | None:
    """Extraction, scrubbing and stripping pH from the ``D`` curves.

    With a boundary (some but not all of ``elements`` are ``targets``) the
    extraction and scrub sit where the boundary pair's geometric-mean factor
    is one at each section's own ``O/A``, and the strip where the most
    strongly held target has factor 0.1. Without one, the extraction sits
    where the least extractable element has factor 10 and the strip where the
    most strongly held element has factor 0.1; the scrub then runs at the
    extraction pH, where it returns nothing. See the module docstring for why.

    Args:
        extractant: Extractant name.
        elements: Every element the circuit carries.
        targets: The elements meant for the product; ``()`` for none.
        extraction_OA: Extraction ``O/A``, as the unit computes it (see
            :func:`circuit_phase_ratios`).
        strip_OA: Stripping ``O/A``.
        scrub_OA: Scrubbing ``O/A``; needed only for a boundary cut.
        extractant_conc: Extractant concentration in the organic (M), on the
            extractant record's own basis: DIMER for the dimeric D2EHPA, PC88A
            and Cyanex272 (0.5 M dimer = 1.0 M formal), molecules (monomer) for
            TBP and naphthenic acid. The loading capacity is this divided by
            ``Extractant.basis_units_per_ree`` (3 for every shipped record;
            #374). ``D`` depends on it.
        nitrate_conc: Aqueous nitrate (M), for a solvating extractant.
        mechanism: Mechanism override; see ``REEDistribution``.
        distribution: An existing ``REEDistribution`` to reuse (it must cover
            ``elements``); None builds one.
        max_strip_acid: Strongest strip liquor (M H+) the strip may call for;
            a strip cut below ``-log10(max_strip_acid)`` is raised to it and
            flagged (``strip_acid_limited``). None applies no floor.

    Returns:
        :class:`OperatingPHs`, or None when ``D`` does not move with pH (the
        caller keeps its fallback).

    Warns:
        OperatingPointWarning: If the targets are not the more extractable
            group, so no single pH extracts them and rejects the rest.

    Example:
        >>> ph = cut_pHs("D2EHPA", ("Nd", "Dy"), ("Dy",), extraction_OA=1.5,
        ...              scrub_OA=7.5, strip_OA=3.0)
        >>> ph.stripping < ph.scrubbing < ph.extraction
        True
    """
    from difflow_ree.database import get_extractant
    from difflow_ree.equilibrium.distribution import REEDistribution

    elements = tuple(elements)
    targets = tuple(e for e in targets if e in elements)
    # A solvating record has no pH cut, and building its distribution without
    # nitrate raises: decide from the record before building anything.
    if (mechanism or get_extractant(extractant).mechanism) == "solvating":
        return None
    if distribution is None:
        distribution = REEDistribution(
            extractant, elements, concentration=extractant_conc,
            nitrate_conc=nitrate_conc, mechanism=mechanism,
            on_out_of_range="ignore")
    if not is_pH_driven(distribution, elements):
        return None

    floor = None if max_strip_acid is None else strip_pH_floor(max_strip_acid)

    def floored(strip, strip_elements):
        # The acid floor applies to this strip pH because the code chose it;
        # a caller's explicit pH never comes through here.
        limited = floor is not None and strip < floor
        return dict(stripping=floor if limited else strip,
                    strip_acid_limited=bool(limited), strip_cut=strip,
                    strip_elements=tuple(strip_elements))

    rejected = tuple(e for e in elements if e not in targets)
    if not targets or not rejected:
        ext = pH_where(distribution, elements, CUT_FACTOR["bulk_extraction"],
                       extraction_OA, how="min")
        strip = pH_where(distribution, elements, CUT_FACTOR["stripping"],
                         strip_OA, how="max")
        return OperatingPHs(ext, ext, basis="bulk", **floored(strip, elements))

    if scrub_OA is None:
        raise ValueError("a boundary cut needs scrub_OA")
    # Order the elements by D at the middle of the fitted window; for the
    # cation-exchange records every element has the same pH slope, so the
    # order does not depend on the pH chosen.
    lo, hi = distribution._ext_data.valid_ph_range
    D_mid = distribution.get_D_all(pH=0.5 * (lo + hi))
    weakest_target = min(targets, key=lambda e: float(D_mid[e]))
    strongest_rejected = max(rejected, key=lambda e: float(D_mid[e]))
    if float(D_mid[weakest_target]) <= float(D_mid[strongest_rejected]):
        warnings.warn(
            f"On {extractant}, target {weakest_target} is no more extractable "
            f"than rejected {strongest_rejected}, so no pH extracts every "
            f"target and rejects every other element: the product will carry "
            f"rejected elements, or lose targets, whatever the pH. The cut is "
            f"placed on that pair anyway.",
            OperatingPointWarning, stacklevel=2)
    pair = (weakest_target, strongest_rejected)
    strip = pH_where(distribution, targets, CUT_FACTOR["stripping"], strip_OA,
                     how="max")
    return OperatingPHs(
        pH_where(distribution, pair, CUT_FACTOR["extraction"], extraction_OA),
        pH_where(distribution, pair, CUT_FACTOR["scrubbing"], scrub_OA),
        basis="cut",
        boundary=pair,
        **floored(strip, targets),
    )


def strip_retention(distribution, elements, pH: float, strip_OA: float,
                    n_stages) -> dict[str, float]:
    """Fraction of each element a counter-current strip leaves on the organic.

    The Kremser fraction ``(S - 1) / (S**(N+1) - 1)`` with ``S = 1 / (D O/A)``,
    the same one :class:`~difflow_ree.units.stripping.REEStripper` evaluates
    for a strip liquor that enters clean. Below ``S = 1`` no stage count
    strips the element: the fraction tends to ``1 - S``.

    Args:
        distribution: An ``REEDistribution`` covering ``elements``.
        elements: Elements to report.
        pH: Strip pH.
        strip_OA: Strip ``O/A``.
        n_stages: Strip stage count.

    Returns:
        ``{element: fraction left on the barren organic}``.
    """
    factors = section_factors(distribution, tuple(elements), [pH], strip_OA)[0]
    left = kremser_fraction(1.0 / factors, n_stages)
    return {e: float(f) for e, f in zip(elements, left)}


def warn_strip_acid_limited(extractant: str, cuts: OperatingPHs,
                            retention: dict[str, float], n_stages,
                            max_strip_acid: float, where: str,
                            stacklevel: int = 3) -> None:
    """Report a strip held at the acid floor (:class:`StripAcidLimitWarning`).

    Args:
        extractant: Extractant name, for the message.
        cuts: The :class:`OperatingPHs` whose strip was floored.
        retention: :func:`strip_retention` at the floor.
        n_stages: The strip stage count the retention is for.
        max_strip_acid: The acid limit (M).
        where: What is reporting, for the message.
        stacklevel: Passed to :func:`warnings.warn`.
    """
    from difflow_ree.database import get_extractant

    worst = max(retention, key=retention.get)
    held = ", ".join(f"{e} {100 * f:.3g} %" for e, f in retention.items()
                     if f > 1e-3) or "none above 0.1 %"
    # Say so only when it is true: a floor set high enough can sit inside
    # the window even when the cut does not.
    lo = get_extractant(extractant).valid_ph_range[0]
    window = (f" Both pHs are below the fitted window of the {extractant} "
              f"coefficients (pH >= {lo:g}), so these are extrapolations."
              if cuts.stripping < lo else "")
    warnings.warn(
        f"{where}: stripping {', '.join(cuts.strip_elements)} from "
        f"{extractant} needs pH {cuts.strip_cut:.2f} "
        f"({10 ** -cuts.strip_cut:.3g} M H+), stronger than max_strip_acid "
        f"= {max_strip_acid:g} M, so the strip runs at pH "
        f"{cuts.stripping:.2f}. With {n_stages:g} strip stages the barren "
        f"organic keeps {held} (worst: {worst}).{window} Give stripping_pH "
        f"explicitly, or raise max_strip_acid, to override.",
        StripAcidLimitWarning, stacklevel=stacklevel)


def report_strip_floor(extractant: str, cuts: OperatingPHs, strip_OA: float,
                       n_stages, max_strip_acid: float, where: str, *,
                       extractant_conc: float = 1.0,
                       nitrate_conc: float | None = None,
                       mechanism: str | None = None,
                       stacklevel: int = 4) -> dict[str, float]:
    """Retention at a floored strip, warned about; what the circuits call.

    Args:
        extractant: Extractant name.
        cuts: :class:`OperatingPHs` with ``strip_acid_limited`` set.
        strip_OA: Strip ``O/A``.
        n_stages: Strip stage count.
        max_strip_acid: The acid limit (M).
        where: What is reporting, for the message.
        extractant_conc: Extractant concentration on the record basis (M).
        nitrate_conc: Aqueous nitrate (M).
        mechanism: Mechanism override.
        stacklevel: Passed to :func:`warnings.warn`.

    Returns:
        :func:`strip_retention` for ``cuts.strip_elements`` at the floor.
    """
    from difflow_ree.equilibrium.distribution import REEDistribution

    dist = REEDistribution(extractant, cuts.strip_elements,
                           concentration=extractant_conc,
                           nitrate_conc=nitrate_conc, mechanism=mechanism,
                           on_out_of_range="ignore")
    retention = strip_retention(dist, cuts.strip_elements, cuts.stripping,
                                strip_OA, n_stages)
    warn_strip_acid_limited(extractant, cuts, retention, n_stages,
                            max_strip_acid, where, stacklevel=stacklevel)
    return retention


def kremser_fraction(factor, n_stages):
    """Fraction of a solute NOT transferred by a counter-current section.

    ``(f - 1) / (f**(N+1) - 1)`` with ``f`` the transfer factor of the phase
    the solute leaves (``E = D O/A`` for an extraction: the fraction left in
    the raffinate; ``S = 1/(D O/A)`` for a scrub or strip: the fraction left
    on the organic). The same expression the units evaluate through
    :func:`difflow_ree.units.kremser.kremser_two_inlet` with a solute-free
    second inlet, written with numpy so a design search can vectorise it.

    Args:
        factor: Transfer factor(s), array-like.
        n_stages: Stage count(s), array-like, broadcast against ``factor``.

    Returns:
        numpy array of fractions in [0, 1].
    """
    import numpy as np

    f = np.asarray(factor, dtype=float)
    n = np.asarray(n_stages, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fn = np.power(f, n + 1.0)
        out = np.where(np.abs(f - 1.0) < 1e-9, 1.0 / (n + 1.0), (f - 1.0) / (fn - 1.0))
    # f -> inf: fn overflows to inf, (f-1)/inf = 0, which is the limit.
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


def section_factors(distribution, elements: tuple[str, ...], pHs, phase_ratio: float):
    """``D * (O/A)`` for every element at every pH, as a numpy array.

    Args:
        distribution: An ``REEDistribution`` covering ``elements``.
        elements: Element order of the last axis.
        pHs: Iterable of pH values.
        phase_ratio: The section's ``O/A``.

    Returns:
        Array of shape ``(len(pHs), len(elements))``.
    """
    import numpy as np

    pHs = np.atleast_1d(np.asarray(pHs, dtype=float))
    D = distribution.get_D_all(pH=pHs)  # one vectorised call, not one per pH
    cols = [np.broadcast_to(np.asarray(D[e], dtype=float), pHs.shape)
            for e in elements]
    return np.stack(cols, axis=-1) * phase_ratio
