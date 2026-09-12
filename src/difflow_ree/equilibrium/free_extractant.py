"""Free extractant: the quantity the correlation's ``[HA]`` actually means (#267).

``Q1`` Eq. 2.88, the working correlation the naphthenic-acid form is built on,
is

    lg D = lg K_ex + 3 lg (HA)o + 3 pH

and Eq. 2.89 defines ``(HA)o`` explicitly as what is **left**::

    (HA)o = [HA]_0 - 3 (REA3)

total charged minus three monomers per extracted RE(III).  :class:`~difflow_ree.
equilibrium.distribution.REEDistribution` applies the same functional form with
``[HA]`` the total charged, which agrees at low loading and diverges where the
cascade is working hardest --- and the error flatters the model, because total
overstates what is available and so overpredicts ``D``.

Two things live here:

* :func:`solve_free_extractant` closes the loop.  ``c_org = D((HA)_free) c_aq``
  and ``(HA)_free = [HA]_0 - m c_org`` is a scalar fixed point, monotone on a
  bracketed interval, so it is solved rather than approximated --- and solved
  through :func:`optimistix.root_find`, so it differentiates by the implicit
  function theorem rather than by unrolling.
* :func:`implied_loading_fraction` and :func:`check_loading_capacity` answer
  the cheaper question the issue asks for as a minimum: is the loading this
  ``D`` implies even possible?  ``Z1`` Table 4.36 system 2 (0.13 mol/L
  naphthenic acid, 0.240 mol/L RE at pH 4.96) implies about 0.066 mol/L RE in
  the organic, which at three monomers per RE(III) needs 0.198 mol/L against
  0.13 available: **152 % of capacity**.  Eq. 2.89 returns a negative free
  extractant there and ``log10`` of it is undefined; difflow_ree never formed
  the quantity, so it returned a finite ``D`` instead.  That rejection used to
  have to be made by hand.

Relationship to :mod:`difflow_ree.equilibrium.loading`: ``LoadingIsotherm.
apparent_D`` caps an answer *after* ``D`` is computed, where this changes the
input that produced it.  They are not the same correction and must not be
composed --- #190 and #204 removed exactly that double count.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow_ree.database import Extractant, get_extractant

__all__ = [
    "ExtractantCapacityWarning",
    "FreeExtractantResult",
    "implied_loading_fraction",
    "check_loading_capacity",
    "solve_free_extractant",
]


class ExtractantCapacityWarning(UserWarning):
    """A loading was requested that the extractant charged cannot hold.

    The extractant balance in monomer equivalents is

        [HA]_total = [HA]_free + m [RE-complex]

    so ``m c_org > [HA]_total`` leaves no free extractant --- a negative one,
    in fact, which is what ``Q1`` Eq. 2.89 returns and what has no logarithm.
    A correlation written against *total* extractant does not notice: it takes
    ``log10([HA]_total / C_ref)`` and hands back a finite ``D`` (#267).
    """


@dataclass(frozen=True)
class FreeExtractantResult:
    """What :func:`solve_free_extractant` found.

    Attributes:
        D: Distribution coefficient at the self-consistent free extractant.
        c_org: Organic-phase REE concentration (M).
        free_extractant: ``[HA]_free`` (M), on the same basis as the
            concentration handed to the correlation.
        loading_fraction: ``m c_org / [HA]_total``, dimensionless; 1.0 is a
            saturated solvent.
        D_total_basis: What the correlation says against *total* extractant
            --- the number difflow_ree returns without this module. Carried so
            the size of the difference is reportable rather than folklore.
    """

    D: Array
    c_org: Array
    free_extractant: Array
    loading_fraction: Array
    D_total_basis: Array

    @property
    def overprediction(self) -> Array:
        """``D_total_basis / D``: how much the total basis flatters the model."""
        return self.D_total_basis / self.D


def _record(extractant: str | Extractant) -> Extractant:
    return (extractant if isinstance(extractant, Extractant)
            else get_extractant(extractant))


def implied_loading_fraction(
    extractant: str | Extractant,
    c_org: Array | float,
    extractant_conc: Array | float,
) -> Array:
    """Fraction of the charged extractant that a given organic loading uses.

    Args:
        extractant: Name or record; only its ``monomers_per_ree`` is read.
        c_org: Organic-phase REE concentration (M).
        extractant_conc: Total extractant charged (M), same basis as the
            record's ``reference_concentration``.

    Returns:
        ``m c_org / [HA]_total``. Above 1.0 the loading is impossible, not
        merely high.
    """
    m = _record(extractant).monomers_per_ree
    return m * jnp.asarray(c_org) / jnp.asarray(extractant_conc)


def check_loading_capacity(
    extractant: str | Extractant,
    c_org: Array | float,
    extractant_conc: Array | float,
    action: str = "warn",
) -> Array:
    """Check an organic loading against the extractant's stoichiometric capacity.

    This is the one-line check the issue asks for as a minimum, and it is what
    would have rejected ``Z1`` Table 4.36 system 2 automatically rather than
    by hand.

    Args:
        extractant: Name or record.
        c_org: Organic-phase REE concentration (M).
        extractant_conc: Total extractant charged (M).
        action: ``"warn"`` (default), ``"raise"`` or ``"ignore"``.

    Returns:
        The loading fraction, so a caller can use it without computing it
        twice.

    Raises:
        ValueError: If the loading exceeds capacity and ``action="raise"``,
            or if ``action`` is not one of the three.
    """
    if action not in ("warn", "raise", "ignore"):
        raise ValueError(
            f"action must be 'warn', 'raise' or 'ignore', got {action!r}."
        )
    theta = implied_loading_fraction(extractant, c_org, extractant_conc)
    if action == "ignore":
        return theta

    # Under tracing there is no value to compare, and a check that cannot be
    # made is not a check that failed.
    try:
        worst = float(jnp.max(theta))
    except Exception:  # noqa: BLE001 - a tracer has no concrete maximum
        return theta
    if worst <= 1.0:
        return theta

    record = _record(extractant)
    message = (
        f"Organic loading is {worst:.1%} of {record.name}'s stoichiometric "
        f"capacity: {record.monomers_per_ree:g} extractant monomers are bound "
        f"per mol REE, so holding this much REE needs more extractant than is "
        f"charged. The free extractant [HA]_0 - m c_org is NEGATIVE, and the "
        "correlation's log10([HA]) is undefined there -- a correlation written "
        "against TOTAL extractant does not notice and returns a finite D "
        "anyway (#267). Either the loading, the extractant concentration or "
        "the D it came from is wrong."
    )
    if action == "raise":
        raise ValueError(message)
    warnings.warn(message, ExtractantCapacityWarning, stacklevel=2)
    return theta


def solve_free_extractant(
    dist,
    element: str,
    c_aq: Array | float,
    pH: Array | float | None = None,
    T: Array | float = 298.15,
    max_steps: int = 64,
    **get_D_kwargs,
) -> FreeExtractantResult:
    """Solve ``D`` and the organic loading against *free* extractant (#267).

    Closes ``Q1`` Eq. 2.88 with Eq. 2.89::

        c_org      = D([HA]_free) * c_aq
        [HA]_free  = [HA]_total - m * c_org

    Substituting gives one scalar equation in ``[HA]_free``::

        F(u) = [HA]_total - m c_aq D_ref (u / [HA]_total)^n - u

    which is strictly decreasing in ``u`` --- both terms fall as ``u`` rises
    --- and changes sign between ``u = 0`` (where ``F = [HA]_total > 0``) and
    ``u = [HA]_total`` (where ``F = -m c_aq D < 0``).  One root, bracketed, on
    a smooth monotone function: :func:`optimistix.root_find` with Newton finds
    it and carries the implicit-differentiation rule, so a gradient through
    this comes from the converged answer rather than from an unrolled
    iteration.

    Args:
        dist: A :class:`~difflow_ree.equilibrium.distribution.REEDistribution`.
            Its ``concentration`` is ``[HA]_total``.
        element: REE symbol.
        c_aq: Aqueous-phase REE concentration (M).
        pH: Passed to ``dist.get_D``; required for cation exchange.
        T: Temperature (K).
        max_steps: Iteration cap for the root find.
        **get_D_kwargs: Anything else ``dist.get_D`` takes
            (``ionic_strength``, ``nitrate_conc``, ``counter_ion_conc``).

    Returns:
        A :class:`FreeExtractantResult`.

    Raises:
        ValueError: If the extractant concentration or ``c_aq`` is not
            positive.
    """
    record = get_extractant(dist.extractant)
    total = float(dist.concentration)
    if total <= 0.0:
        raise ValueError(
            f"Extractant concentration must be positive, got {total}."
        )
    m = record.monomers_per_ree
    n = record.concentration_exponent
    c_aq = jnp.asarray(c_aq)

    # D at the record's own reference concentration: the concentration term is
    # n*log10([HA]/C_ref), so factoring it out leaves a clean power law in the
    # free extractant and keeps every other term (temperature, activity,
    # overrides) exactly as get_D computes it.
    D_ref = dist.get_D(element, pH, T, **get_D_kwargs) * jnp.power(
        record.reference_concentration / total, n
    )
    D_total = dist.get_D(element, pH, T, **get_D_kwargs)

    def residual(u, args):
        """F(u), monotone decreasing, with one root in (0, total]."""
        # The clip keeps a Newton step that overshoots below zero from
        # raising a fractional power of a negative number; the root itself
        # is interior, so the clip is inactive there and the implicit
        # derivative is unaffected.
        u_safe = jnp.clip(u, 1e-12 * total, total)
        D_u = D_ref * jnp.power(u_safe / record.reference_concentration, n)
        return total - m * c_aq * D_u - u

    solution = optx.root_find(
        residual,
        optx.Newton(rtol=1e-10, atol=1e-12),
        # Start from a solvent that is mostly free: at low loading that is
        # nearly the answer, and F is monotone so there is nowhere else to go.
        jnp.asarray(total, dtype=jnp.result_type(float)),
        args=None,
        max_steps=max_steps,
        throw=False,
    )
    free = jnp.clip(solution.value, 0.0, total)
    D = D_ref * jnp.power(free / record.reference_concentration, n)
    c_org = D * c_aq
    return FreeExtractantResult(
        D=D,
        c_org=c_org,
        free_extractant=free,
        loading_fraction=m * c_org / total,
        D_total_basis=D_total,
    )
