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


class OperatingPointWarning(UserWarning):
    """The requested split has no clean pH cut (for example the targets are
    not the more extractable group, so no single pH extracts them and rejects
    the rest)."""


def circuit_phase_ratios(
    solvent_to_feed_ratio: float,
    scrub_to_solvent_ratio: float | None,
    strip_to_solvent_ratio: float,
    extractant_conc: float,
) -> dict[str, float]:
    """The ``O/A`` each section of a circuit actually runs at.

    The circuits build their solvent as ``{diluent: F_org, extractant:
    extractant_conc * F_org}`` with ``F_org = F_aq * solvent_to_feed_ratio``,
    and the units count the extractant moles as organic flow, so the
    extraction ``O/A`` is ``solvent_to_feed_ratio * (1 + extractant_conc)``.
    The scrub and strip make-up flows are sized from the diluent flow alone,
    so their ``O/A`` is ``(1 + extractant_conc) / ratio``. Whether the
    extractant moles should count as organic flow at all is an open modelling
    question (the audit's R5); this function reports what the units do today
    so that a cut computed from it lands where the section really operates.

    Args:
        solvent_to_feed_ratio: Circuit ``solvent_to_feed_ratio``.
        scrub_to_solvent_ratio: Circuit ``scrub_to_solvent_ratio``, or None
            for a circuit without a scrub.
        strip_to_solvent_ratio: Circuit ``strip_to_solvent_ratio``.
        extractant_conc: Extractant concentration (M).

    Returns:
        ``{"extraction": O/A, "scrubbing": O/A or None, "stripping": O/A}``.

    Example:
        >>> circuit_phase_ratios(1.0, 0.2, 0.5, 0.5)["stripping"]
        3.0
    """
    org = 1.0 + extractant_conc
    return {
        "extraction": solvent_to_feed_ratio * org,
        "scrubbing": None if not scrub_to_solvent_ratio else org / scrub_to_solvent_ratio,
        "stripping": org / strip_to_solvent_ratio,
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
    """

    extraction: float
    scrubbing: float
    stripping: float
    basis: str
    boundary: tuple[str, str] | None = None

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
        extractant_conc: Extractant concentration (M); ``D`` depends on it.
        nitrate_conc: Aqueous nitrate (M), for a solvating extractant.
        mechanism: Mechanism override; see ``REEDistribution``.
        distribution: An existing ``REEDistribution`` to reuse (it must cover
            ``elements``); None builds one.

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

    rejected = tuple(e for e in elements if e not in targets)
    if not targets or not rejected:
        ext = pH_where(distribution, elements, CUT_FACTOR["bulk_extraction"],
                       extraction_OA, how="min")
        strip = pH_where(distribution, elements, CUT_FACTOR["stripping"],
                         strip_OA, how="max")
        return OperatingPHs(ext, ext, strip, "bulk")

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
    return OperatingPHs(
        pH_where(distribution, pair, CUT_FACTOR["extraction"], extraction_OA),
        pH_where(distribution, pair, CUT_FACTOR["scrubbing"], scrub_OA),
        pH_where(distribution, targets, CUT_FACTOR["stripping"], strip_OA,
                 how="max"),
        "cut",
        pair,
    )


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
