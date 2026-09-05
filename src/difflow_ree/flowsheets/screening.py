"""Cheap admissibility screening for candidate topologies (#202).

Rigorous cascade solves and costing are the expensive part of a topology
search, and most candidates do not deserve either. The Fenske minimum
stage count is a *rigorous lower bound* on how many equilibrium stages a
given split needs at a given separation factor --- it is the total-reflux
limit, so no finite solvent-to-feed ratio can beat it --- and it costs two
logarithms. A candidate whose installed stage count is below that bound
cannot meet its purity target no matter how it is operated, so it can be
struck out before anything is solved, let alone costed.

That is the whole of what this module does: it is a *filter*, not a
search. The discrete search itself belongs outside difflow (#202); route
out through :mod:`difflow.solvers` (#203).

Example:
    >>> from difflow_ree.flowsheets.screening import screen_separation
    >>> v = screen_separation("D2EHPA", "Nd", "La", installed_stages=10,
    ...                       purity=0.99)
    >>> v.admissible
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field

from difflow.params_mixin import ParamsMixin
from difflow_ree.equilibrium.distribution import REEDistribution, stages_fenske


@dataclass(repr=False)
class ScreeningVerdict(ParamsMixin):
    """Whether one separation duty is worth solving rigorously.

    Attributes:
        name: What was screened, usually a module name.
        admissible: Whether ``installed_stages >= minimum_stages``.
        minimum_stages: The Fenske lower bound.
        installed_stages: The stage count the candidate declares.
        slack: ``installed_stages - minimum_stages``; negative means the
            duty is unreachable with that many stages, at any flow ratio.
        separation_factor: The ``alpha`` used.
        keys: The ``(extract key, raffinate key)`` pair.
        reason: Human-readable explanation.
    """

    name: str
    admissible: bool
    minimum_stages: float
    installed_stages: float
    slack: float
    separation_factor: float = 0.0
    keys: tuple[str, str] = ("", "")
    reason: str = ""

    def to_dict(self) -> dict:
        """A plain JSON-ready dictionary."""
        return {
            "name": self.name,
            "admissible": self.admissible,
            "minimum_stages": self.minimum_stages,
            "installed_stages": self.installed_stages,
            "slack": self.slack,
            "separation_factor": self.separation_factor,
            "keys": list(self.keys),
            "reason": self.reason,
        }


def minimum_stages(
    alpha: float,
    purity_extract: float,
    purity_raffinate: float | None = None,
) -> float:
    """The Fenske bound, as a Python float.

    A thin concrete wrapper over
    :func:`difflow_ree.equilibrium.distribution.stages_fenske` for the
    screening loop, which is ordinary Python and has no reason to hold a
    JAX scalar.

    Args:
        alpha: Separation factor between the two keys.
        purity_extract: Target purity (or split) of the extract product.
        purity_raffinate: Target purity of the raffinate product;
            defaults to ``purity_extract``.

    Returns:
        Minimum theoretical stages.
    """
    return float(stages_fenske(alpha, purity_extract, purity_raffinate))


def separation_factor(
    extractant: str,
    extract_key: str,
    raffinate_key: str,
    pH: float = 3.0,
    *,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
) -> float:
    """``D_extract_key / D_raffinate_key`` for one extractant.

    Args:
        extractant: Extractant name.
        extract_key: The element the cascade sends to the organic.
        raffinate_key: The element it leaves in the aqueous.
        pH: Operating pH for the correlation.
        nitrate_conc: Nitrate concentration (M) for solvating
            extractants; see #195.
        mechanism: Explicit mechanism override; see #195.

    Returns:
        The separation factor.
    """
    dist = REEDistribution(
        extractant=extractant,
        elements=(extract_key, raffinate_key),
        nitrate_conc=nitrate_conc,
        mechanism=mechanism,
    )
    return float(dist.get_separation_factor(extract_key, raffinate_key, pH))


def screen_separation(
    extractant: str,
    extract_key: str,
    raffinate_key: str,
    installed_stages: float,
    purity: float = 0.99,
    purity_raffinate: float | None = None,
    pH: float = 3.0,
    *,
    name: str = "",
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
) -> ScreeningVerdict:
    """Screen one separation duty against its Fenske minimum.

    Args:
        extractant: Extractant name.
        extract_key: Element reporting to the organic.
        raffinate_key: Element reporting to the aqueous.
        installed_stages: Stages the candidate declares for this duty.
        purity: Target purity (or split) of the extract product.
        purity_raffinate: Target purity of the raffinate; defaults to
            ``purity``.
        pH: Operating pH for the separation factor.
        name: Label for the verdict.
        nitrate_conc: Nitrate concentration (M); see #195.
        mechanism: Mechanism override; see #195.

    Returns:
        The :class:`ScreeningVerdict`.
    """
    alpha = separation_factor(
        extractant, extract_key, raffinate_key, pH,
        nitrate_conc=nitrate_conc, mechanism=mechanism,
    )
    n_min = minimum_stages(alpha, purity, purity_raffinate)
    slack = float(installed_stages) - n_min
    admissible = slack >= 0.0
    reason = (
        f"alpha({extract_key}/{raffinate_key}) = {alpha:.4g} at pH {pH}; "
        f"Fenske minimum {n_min:.3g} stages vs "
        f"{float(installed_stages):.3g} installed"
    )
    if not admissible:
        reason += (
            " -- unreachable at any solvent-to-feed ratio, so the candidate "
            "is rejected before it is solved or costed (#202)"
        )
    return ScreeningVerdict(
        name=name or f"{extract_key}/{raffinate_key}",
        admissible=admissible,
        minimum_stages=n_min,
        installed_stages=float(installed_stages),
        slack=slack,
        separation_factor=alpha,
        keys=(extract_key, raffinate_key),
        reason=reason,
    )


@dataclass(repr=False)
class ScreeningReport(ParamsMixin):
    """The verdicts for a whole candidate train.

    Attributes:
        verdicts: One per screened duty.
    """

    verdicts: tuple[ScreeningVerdict, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.verdicts = tuple(self.verdicts)

    def __len__(self) -> int:
        return len(self.verdicts)

    def __iter__(self):
        return iter(self.verdicts)

    @property
    def admissible(self) -> bool:
        """Whether every duty clears its Fenske bound."""
        return all(v.admissible for v in self.verdicts)

    def rejected(self) -> tuple[ScreeningVerdict, ...]:
        """The duties that failed, worst slack first."""
        bad = [v for v in self.verdicts if not v.admissible]
        return tuple(sorted(bad, key=lambda v: v.slack))

    def to_dict(self) -> dict:
        """A plain JSON-ready dictionary."""
        return {
            "admissible": self.admissible,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }

    def summary(self) -> str:
        """A one-line-per-duty text table."""
        if not self.verdicts:
            return "nothing screened"
        return "\n".join(
            f"{v.name}: {'ADMISSIBLE' if v.admissible else 'REJECTED'} "
            f"(N_min={v.minimum_stages:.3g}, installed="
            f"{v.installed_stages:.3g})"
            for v in self.verdicts
        )


def screen_train(
    train,
    duties: dict[str, tuple[str, str]],
    purity: float = 0.99,
    purity_raffinate: float | None = None,
) -> ScreeningReport:
    """Screen every named separation module of a train.

    Args:
        train: A :class:`~difflow_ree.flowsheets.train.SeparationTrain`.
        duties: ``{module name: (extract key, raffinate key)}``. Only the
            named modules are screened; others (precipitation, cerium
            oxidation) have no key pair to screen.
        purity: Target purity of each extract product.
        purity_raffinate: Target purity of each raffinate; defaults to
            ``purity``.

    Returns:
        The :class:`ScreeningReport`.

    Raises:
        KeyError: If a named module is not in the train.
    """
    verdicts = []
    for name, (extract_key, raffinate_key) in duties.items():
        module = train.modules[name]
        p = module.params
        # The scrubbing section is what actually sets the product purity
        # of an extract-scrub-strip circuit, so it is the stage count the
        # bound is compared against; extraction sets recovery.
        installed = float(getattr(
            p, "n_scrubbing_stages", getattr(p, "n_stages", 0)
        ))
        verdicts.append(screen_separation(
            p.extractant, extract_key, raffinate_key, installed,
            purity=purity, purity_raffinate=purity_raffinate,
            pH=float(getattr(p, "scrubbing_pH", getattr(p, "pH", 3.0))),
            name=name,
            nitrate_conc=getattr(p, "nitrate_conc", None),
            mechanism=getattr(p, "mechanism", None),
        ))
    return ScreeningReport(tuple(verdicts))


def screen_topologies(
    candidates,
    duties_for,
    purity: float = 0.99,
) -> list[tuple[object, ScreeningReport]]:
    """Filter an enumeration of candidate topologies.

    The intended use is the cheap half of an external discrete search:
    enumerate topologies outside difflow, screen them here, and solve
    only the survivors.

    Args:
        candidates: Iterable of
            :class:`~difflow_ree.flowsheets.train.SeparationTrain`.
        duties_for: Callable mapping a candidate to its ``duties`` dict
            for :func:`screen_train`.
        purity: Target purity.

    Returns:
        ``[(candidate, report)]`` for the admissible candidates only, in
        input order.
    """
    survivors = []
    for candidate in candidates:
        report = screen_train(candidate, duties_for(candidate), purity=purity)
        if report.admissible:
            survivors.append((candidate, report))
    return survivors
