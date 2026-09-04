"""REE extraction unit operations.

Multi-stage counter-current extraction cascades for REE separation.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, lax

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.loading import LoadingIsotherm, get_loading_isotherm
from difflow_ree.equilibrium.speciation import REESpeciation
from difflow_ree.kinetics.extraction_kinetics import approach_to_equilibrium


# =============================================================================
# Phase flow bookkeeping
# =============================================================================

#: Relative floor applied to a phase carrier flow before it is used as a
#: denominator.  It is *relative* to the total flow of the streams involved,
#: never an absolute molar flow, so every ratio in this module is invariant to
#: the unit the flows are expressed in (#189).  The absolute ``1e-10`` floor it
#: replaced silently destroyed that invariance below about ``1e-11`` mol/s: a
#: cascade expressed in mol/s and the identical cascade expressed in Tmol/s
#: returned different recoveries (0.3255 vs 0.0330 for the D2EHPA/Nd case in
#: ``tests/ree/test_ree_loading_capacity.py``).  With a relative floor the
#: supported range is the whole float64 range, limited only by underflow of the
#: flows themselves.
_PHASE_FLOW_FLOOR_REL = 1e-12


#: The two levels ``REEExtractor`` can run at (#196), behind one interface.
#:
#: ``"correlation"`` (default, unchanged) evaluates ``log10(D)`` from
#: ``REEDistribution`` at a *specified* pH and solves the cascade with Kremser
#: plus the smooth capacity limiter.
#:
#: ``"mass_action"`` solves the whole section's component balances against a
#: reaction network carried as data, in log concentration, via
#: ``difflow.eo_solver.solve_residual_system``.
#:
#: The interface is shared so cascade code is level-agnostic, but TWO THINGS
#: GENUINELY DIFFER and are not hidden:
#:
#: 1. **State width.** The closed model reads and writes an acid, counter-ion
#:    and anion balance the correlation ignores. The superset is declared once
#:    in ``difflow_ree.equilibrium.schema.REEStreamSchema``; the correlation
#:    path passes through what it does not use, as it always has.
#: 2. **Degrees of freedom.** ``pH`` is an INPUT to the correlation and an
#:    OUTPUT of the closed model, whose corresponding input is base addition
#:    (or, from #197, saponification degree). Under ``model="mass_action"``
#:    the ``pH`` field becomes the *calibration* pH -- where the two levels
#:    are made to agree -- and the operating pH comes back in
#:    ``info["pH_profile"]``. A design specified at one level is therefore not
#:    directly expressible at the other; map between them explicitly with
#:    ``difflow_ree.equilibrium.mass_action.base_addition_for_pH``.
EXTRACTOR_MODELS = ("correlation", "mass_action")


def _concrete(x) -> float | None:
    """Return ``x`` as a Python float, or None if it is a JAX tracer.

    Used for guards that must fire on obviously-broken input but must not
    force a traced value to a concrete one.  Under ``jit``/``grad`` the value
    is a tracer and the guard is skipped, which is the price of keeping the
    unit traceable; the guard still fires on every eager call, which is how
    the mis-specified stream is caught in practice.
    """
    try:
        return float(x)
    except Exception:  # TracerArrayConversionError, ConcretizationTypeError
        return None


def _soft_saturation(
    total: Array | float,
    capacity: Array | float,
    k: float,
) -> Array:
    """Smooth ``min(1, capacity / total)`` with sharpness ``k``.

    Mathematically ``(1 + (total / capacity)**k)**(-1/k)``: 1 well below
    capacity, ``capacity / total`` well above it, and C-infinity everywhere,
    including at ``total == capacity`` where the hard ``jnp.where`` clamp it
    replaced put a kink in the gradient (#193).

    Evaluated in the scaled form ``c / (c**k + t**k)**(1/k)`` with ``c`` and
    ``t`` divided by their own maximum, so neither ``total**k`` nor
    ``capacity**k`` can overflow and ``capacity == 0`` gives a clean 0 rather
    than ``inf``/``nan``.  The degenerate ``total == capacity == 0`` returns 1
    (nothing to limit), guarded with the double-``where`` idiom so reverse-mode
    AD does not see a ``0 ** (1/k)`` derivative.

    Args:
        total: Quantity to be limited (a molar flow, or a loading fraction)
        capacity: The limit, in the same units as ``total``
        k: Sharpness; larger tracks ``min`` more closely, at the cost of
            curvature ``k / 4`` in log-log at the crossing point

    Returns:
        Multiplier in (0, 1]
    """
    total = jnp.asarray(total, dtype=jnp.float64)
    capacity = jnp.asarray(capacity, dtype=jnp.float64)
    # Nothing to limit, and the scaled form would be 0/0.
    degenerate = jnp.logical_and(total <= 0.0, capacity <= 0.0)
    magnitude = jnp.maximum(jnp.maximum(total, capacity), 0.0)
    magnitude = jnp.where(degenerate, 1.0, magnitude)
    c = capacity / magnitude
    t = total / magnitude
    # max(c, t) == 1 whenever not degenerate, so u is in [1, 2]: no overflow
    # from total**k, and no underflow-to-zero of the sum.
    u = jnp.power(c, k) + jnp.power(t, k)
    u = jnp.where(degenerate, 1.0, u)
    return jnp.where(degenerate, 1.0, c / jnp.power(u, 1.0 / k))


def _smooth_free_fraction(theta: Array | float, k: float) -> Array:
    """Smooth ``max(1 - theta, 0)``: the extractant left free at loading theta.

    ``1 - theta * softmin(theta, 1)`` with the same power-mean softmin as
    :func:`_soft_saturation`, so both limiters in this module are driven by the
    single ``capacity_sharpness`` knob.

    Why not ``jnp.maximum(1 - theta, 0)`` (#193): that is a hard kink with a
    *dead lever* beyond it.  Past ``theta = 1`` the clipped free fraction is
    identically zero, so ``E`` is identically zero, so ``d(recovery)/d(anything
    upstream)`` is exactly 0 and a gradient-based optimizer -- or a
    ``difflow.planning`` delta vector -- sees a column of zeros and never moves
    the lever again.  That is precisely the pathology
    ``difflow.planning.health`` flags.  A recycled extract-strip solvent
    travels this path, so it is not a corner case.

    This form is strictly positive for every finite ``theta``: it behaves as
    ``1 - theta`` for ``theta << 1``, equals ``1 - 2**(-1/k)`` at saturation,
    and decays as ``theta**(-k) / k`` beyond it.  Its derivative is
    ``-(1 + theta**k)**(-(k+1)/k)``, which is negative everywhere and never
    identically zero.

    Evaluated in two algebraically identical branches, because the literal
    expression loses the answer to cancellation once it falls below float64
    epsilon: at ``theta = 101`` with ``k = 8`` the true value is 1.2e-17 and
    ``1 - theta * (...)`` rounds to exactly 0.0, resurrecting the dead lever
    this function exists to remove.  Above ``theta = 1`` the equivalent
    ``-expm1(-log1p(theta**-k) / k)`` is used instead, which is accurate down
    to the underflow of ``theta**-k`` (about ``theta = 1e38`` for ``k = 8``).
    The two branches agree to the last bit at ``theta = 1``.

    Args:
        theta: Dimensionless loading fraction (0 = clean, 1 = saturated)
        k: Sharpness, shared with the capacity limiter

    Returns:
        Free-extractant fraction in (0, 1]
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    beyond = theta > 1.0
    # Double-where so neither branch's derivative is evaluated on the input
    # that would make it non-finite (theta**-k overflows for small theta).
    theta_lo = jnp.where(beyond, 1.0, theta)
    theta_hi = jnp.where(beyond, theta, 1.0)
    below = 1.0 - theta_lo * _soft_saturation(theta_lo, 1.0, k)
    above = -jnp.expm1(-jnp.log1p(jnp.power(theta_hi, -k)) / k)
    return jnp.where(beyond, above, below)


def _phase_flows(
    flows: dict,
    extractant: str,
    diluent: str,
    require: tuple[str, ...] = ("aqueous", "organic"),
    stream_name: str = "stream",
) -> tuple[Array, Array]:
    """Split a stream's molar flows into aqueous and organic carrier flows.

    One definition shared by :class:`REEExtractor` and
    :class:`REEMixerSettler` so the two cannot disagree (#192). The organic
    phase is the extractant plus the diluent; everything else (water, acid,
    dissolved REE, spectators) is aqueous. Using only ``H2O`` for the aqueous
    flow underestimates it for the concentrated leach liquors this package
    targets, and a silent ``1.0`` default for a missing phase hides a
    mis-specified stream behind a plausible-looking number, so a phase that is
    required but absent raises instead.

    Two checks, not one.  The key check is on the *Python-level* key set and
    always runs.  Checking keys alone was not enough: ``{"H2O": 0.0, "D2EHPA":
    1.0, "kerosene": 5.0}`` has an aqueous key, so it passed, and then a zero
    ``F_aq`` was clamped to a magic ``1e-10`` and produced an extraction factor
    of order ``1e10``.  So a required phase whose flows are *concretely* zero
    or negative raises as well.  That second check needs the value, so it is
    skipped when the value is a tracer (under ``jit``/``grad``); the relative
    floor in the callers is what keeps the traced path finite.

    Args:
        flows: Species molar flows, as returned by ``get_flows``
        extractant: Extractant species name
        diluent: Organic diluent species name
        require: Phases that must have at least one contributing species;
            a subset of ``("aqueous", "organic")``
        stream_name: Name used in the error message

    Returns:
        (F_aq, F_org): Total aqueous and total organic molar flows. A phase
        with no contributing species (and not in ``require``) returns 0.

    Raises:
        ValueError: If a required phase has no contributing species, or if its
            contributing species carry a concretely zero (or negative) total
            flow.
    """
    organic_species = {extractant, diluent}
    org_keys = [k for k in flows if k in organic_species]
    aq_keys = [k for k in flows if k not in organic_species]

    present = sorted(flows)
    if "aqueous" in require and not aq_keys:
        raise ValueError(
            f"{stream_name} has no aqueous phase: every species present is an "
            f"organic carrier. Species present: {present}; organic carriers "
            f"(extractant, diluent): {sorted(organic_species)}."
        )
    if "organic" in require and not org_keys:
        raise ValueError(
            f"{stream_name} has no organic phase: neither the extractant "
            f"'{extractant}' nor the diluent '{diluent}' is present. "
            f"Species present: {present}."
        )

    zero = jnp.asarray(0.0)
    F_aq = sum((jnp.asarray(flows[k]) for k in aq_keys), zero)
    F_org = sum((jnp.asarray(flows[k]) for k in org_keys), zero)

    for phase, keys, value in (
        ("aqueous", aq_keys, F_aq),
        ("organic", org_keys, F_org),
    ):
        if phase not in require:
            continue
        total = _concrete(value)
        if total is not None and not total > 0.0:
            raise ValueError(
                f"{stream_name} declares an {phase} phase but its total flow "
                f"is {total}. The phase ratio D * F_org / F_aq is undefined "
                f"for a phase with no flow; a previous release silently "
                f"floored it and returned an extraction factor of order 1e10. "
                f"Contributing species: {sorted(keys)}; all species present: "
                f"{present}."
            )

    return F_aq, F_org


# =============================================================================
# REE Extractor Parameters
# =============================================================================

@dataclass(repr=False)
class REEExtractorParams(ParamsMixin):
    """Parameters for REE extraction cascade.

    Attributes:
        n_stages: Number of extraction stages
        extractant: Extractant name (D2EHPA, PC88A, etc.)
        elements: REE elements to track
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        pH: Operating pH (typically 1-5 for REE extraction)
        extractant_conc: Extractant concentration (M)
        include_loading: Whether to account for extractant loading
        include_speciation: Whether to account for aqueous speciation
        speciation_medium: Aqueous medium type for speciation
        ligand_conc: Ligand concentration for speciation (M)
        nitrate_conc: Aqueous nitrate concentration (M), required for
            solvating extractants such as TBP (#195)
        mechanism: Explicit extraction-mechanism override passed to the unit's
            REEDistribution, one of "cation_exchange" or "solvating". None
            (default) takes the mechanism from the extractant record (#195).
        capacity_sharpness: Sharpness k of the two smooth loading limiters
            (#193): the total-capacity limiter
            ``scale = (1 + (total/capacity)**k)**(-1/k)`` applied to the newly
            extracted total, and the entering-solvent free fraction
            ``1 - theta * (1 + theta**k)**(-1/k)``. Both replace hard clamps
            (``jnp.where`` and ``jnp.maximum(1 - theta, 0)``) whose gradients
            kink at, and die beyond, the constraint an economic optimum sits
            on.

            The smoothing is *not* free, and the cost is a small unconditional
            haircut below capacity rather than none at all. Exact values of
            ``scale``:

            ==========  ==========  ==========  ==========
            total/cap   k = 4       k = 8       k = 16
            ==========  ==========  ==========  ==========
            0.50        0.98496     0.99951     0.9999990
            0.75        0.93358     0.98814     0.99938
            1.00        0.84090     0.91700     0.95760
            ==========  ==========  ==========  ==========

            Trade-off: the log-log slope ``d ln scale / d ln ratio`` is bounded
            in [-1, 0] for every k, so k does not affect first-derivative
            magnitude; what grows with k is the *curvature*, whose maximum
            ``|d^2 ln scale / d(ln ratio)^2| = k/4`` sits at the crossing
            point. Small k gives a well-conditioned but visibly wrong model
            below capacity; large k gives a faithful model with a sharper
            (though still C-infinity) turn that a line search must resolve.

            The default was raised from 4 to 8 deliberately. At k = 4 a routine
            run at half capacity lost 1.5% of the extraction unconditionally,
            and ``include_loading`` defaults to True so every default result
            carried it; at k = 8 that becomes 0.05%, below any physical
            uncertainty in the correlations, while the curvature only doubles.
            Raise it further (16, 32) to approach ``min()`` when the solve has
            converged and the extra curvature is affordable; lower it to 2-4
            when an optimizer is far from the solution and needs the gentler
            surface.

            Flowsheet note: the REE flowsheet Params in
            ``difflow_ree.flowsheets`` do not yet expose this knob, so reach it
            through the extractor they build,
            ``circuit._extractor.params = circuit._extractor.params.update(
            capacity_sharpness=16)``.
    """
    n_stages: int | float | Array
    extractant: str
    elements: tuple[str, ...]
    diluent: str = "kerosene"
    pH: float | Array = 3.0
    extractant_conc: float = 0.5
    include_loading: bool = True
    include_speciation: bool = False
    speciation_medium: str = "sulfate"
    ligand_conc: float = 0.5
    nitrate_conc: float | None = None
    mechanism: str | None = None
    capacity_sharpness: int = 8

    # -- Closed mass-action model (#196) ---------------------------------
    # See EXTRACTOR_MODELS and the class docstring of REEExtractor. These are
    # ignored entirely by the default correlation path.
    model: str = "correlation"
    aqueous_volumetric_flow: float | None = None
    organic_volumetric_flow: float | None = None
    counter_ion: str | None = "Na"
    anion: str = "Cl"
    reaction_network: str | None = None
    log10_K: dict | None = None
    base_addition: float | Array = 0.0

    def __post_init__(self):
        """Validate extractor parameters."""
        from difflow_ree.database import get_extractant_database, get_ree_database

        if self.model not in EXTRACTOR_MODELS:
            raise ValueError(
                f"model must be one of {list(EXTRACTOR_MODELS)}, got "
                f"{self.model!r} (#196)."
            )
        if self.model == "mass_action":
            missing = [
                name for name in
                ("aqueous_volumetric_flow", "organic_volumetric_flow")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"model='mass_action' needs {missing} (L/s). The closed "
                    f"model works in concentrations, so it needs the phase "
                    f"volumes that the Kremser correlation could do without: "
                    f"an equilibrium constant is not a function of a flow "
                    f"ratio. There is no defensible way to guess them from "
                    f"molar flows without a density model, so they are "
                    f"required rather than defaulted (#196)."
                )

        # Validate extractant exists
        extractant_db = get_extractant_database()
        valid_extractants = extractant_db.list_extractants()
        if self.extractant not in valid_extractants:
            raise ValueError(
                f"Unknown extractant: '{self.extractant}'. "
                f"Available: {valid_extractants}"
            )

        # Validate elements are valid REE
        ree_db = get_ree_database()
        valid_elements = ree_db.list_elements()
        for elem in self.elements:
            if elem not in valid_elements:
                raise ValueError(
                    f"Unknown REE element: '{elem}'. "
                    f"Valid elements: {valid_elements}"
                )

        # Validate bounds
        if hasattr(self.n_stages, '__float__'):
            if float(self.n_stages) < 1:
                raise ValueError(f"n_stages must be >= 1, got {self.n_stages}")
        if self.extractant_conc <= 0:
            raise ValueError(
                f"extractant_conc must be > 0, got {self.extractant_conc}"
            )
        if self.capacity_sharpness <= 0:
            raise ValueError(
                f"capacity_sharpness must be > 0, got {self.capacity_sharpness}"
            )


class REEExtractor:
    """Multi-stage REE extraction cascade, at either modelling level (#196).

    ``model="correlation"`` (the default, and unchanged) is counter-current
    extraction by the Kremser equation with ``log10(D)`` from
    :class:`~difflow_ree.equilibrium.distribution.REEDistribution`, evaluated
    at a specified pH, plus the smooth capacity limiter.

    ``model="mass_action"`` solves the section's component balances against a
    reaction network carried as data -- see
    :class:`difflow_ree.equilibrium.mass_action.MassActionSection`. Both are
    reached through this one class and return
    ``(raffinate, extract, info)``, so cascade code does not have to know
    which level it is running at.

    What the shared interface does **not** hide, and must not (see
    :data:`EXTRACTOR_MODELS`): the closed model's state is wider (it carries
    an acid, counter-ion and anion balance), and its degrees of freedom are
    different. Under ``model="mass_action"`` the :attr:`REEExtractorParams.pH`
    field is the *calibration* pH, not an operating specification: the
    operating pH is an output, in ``info["pH_profile"]``, and the
    corresponding input is ``base_addition``. Use
    :func:`difflow_ree.equilibrium.mass_action.base_addition_for_pH` to map a
    pH-specified design onto the closed model's inputs.

    Example:
        >>> params = REEExtractorParams(
        ...     n_stages=10,
        ...     extractant="D2EHPA",
        ...     elements=("La", "Ce", "Nd", "Dy"),
        ...     pH=3.0,
        ... )
        >>> extractor = REEExtractor(params)
        >>> raffinate, extract, info = extractor(feed, solvent)

        The same cascade with the closed model, where pH becomes an output:

        >>> closed = REEExtractor(params.update(
        ...     model="mass_action",
        ...     aqueous_volumetric_flow=1.0,
        ...     organic_volumetric_flow=1.0,
        ... ))
        >>> raffinate, extract, info = closed(feed, solvent)
        >>> info["pH_profile"]                       # doctest: +SKIP
    """

    symbol = "REE Extraction"
    equations = [
        r"\mathrm{RE}^{3+}_{(aq)} + 3\,\overline{\mathrm{HA}} \rightleftharpoons \overline{\mathrm{REA}_3} + 3\,\mathrm{H}^+\qquad \text{(cation-exchange extractant)}",
        r"K_{\mathrm{ex},i} = \frac{[\overline{\mathrm{REA}_3}]\,[\mathrm{H}^+]^3}{[\mathrm{RE}^{3+}]\,[\overline{\mathrm{HA}}]^3}",
        r"D_i = K_{\mathrm{ex},i}\,\frac{[\overline{\mathrm{HA}}]^3}{[\mathrm{H}^+]^3}",
        r"\frac{x_{N+1}}{x_1} = \frac{E-1}{E^{N+1}-1}\qquad \text{(Kremser, counter-current)}",
    ]
    assumptions = [
        "Counter-current equilibrium stages with constant phase flows.",
        "Distribution coefficients depend on pH and optional extractant loading (Langmuir).",
        "Aqueous and organic phases immiscible; no third phase.",
    ]
    references = [
        "Rydberg, J., Musikas, C., Choppin, G.R. Principles and Practices of Solvent Extraction, Marcel Dekker, 1992.",
        "Xie, F., Zhang, T.A., Dreisinger, D., Doyle, F. Miner. Eng., 56, 10 (2014).",
    ]
    parameter_symbols = {
        "n_stages": "N",
        "extractant_conc": "[HA]",
        "pH": r"\mathrm{pH}",
        "O_A_ratio": "O/A",
    }
    parameter_units = {
        "extractant_conc": "mol/L",
        "pH": "-",
        "O_A_ratio": "-",
    }
    numerical_method = "Kremser closed form with pH-dependent distribution ratios; optional loading via Langmuir isotherm."

    def __init__(self, params: REEExtractorParams):
        """Initialize extractor.

        Args:
            params: Extractor parameters
        """
        self.params = params
        self._distribution = REEDistribution(
            extractant=params.extractant,
            elements=params.elements,
            concentration=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,
            mechanism=params.mechanism,
        )
        if params.include_loading:
            self._isotherm = get_loading_isotherm(
                params.extractant,
                params.extractant_conc,
            )
        else:
            self._isotherm = None

        if params.include_speciation:
            self._speciation = REESpeciation(
                elements=params.elements,
                medium=params.speciation_medium,
            )
        else:
            self._speciation = None

        # Closed mass-action level (#196). Built here so an unbuildable
        # network (bad element, basis mismatch) raises at construction rather
        # than on the first call.
        self._section = None
        if params.model == "mass_action":
            from difflow_ree.equilibrium.mass_action import (
                MassActionParams,
                MassActionSection,
            )
            self._section = MassActionSection(MassActionParams(
                n_stages=int(params.n_stages),
                extractant=params.extractant,
                elements=tuple(params.elements),
                aqueous_volumetric_flow=params.aqueous_volumetric_flow,
                organic_volumetric_flow=params.organic_volumetric_flow,
                diluent=params.diluent,
                counter_ion=params.counter_ion,
                anion=params.anion,
                extractant_conc=params.extractant_conc,
                # The correlation's operating pH becomes the CALIBRATION pH
                # here; the operating pH is an output (#196).
                calibration_pH=float(params.pH),
                log10_K=params.log10_K,
                network=params.reaction_network,
                base_addition=params.base_addition,
            ))

    @property
    def section(self):
        """The underlying ``MassActionSection``, or None at the L1 level.

        Returns:
            :class:`difflow_ree.equilibrium.mass_action.MassActionSection`
            when ``params.model == "mass_action"``, else None. Reach through
            it for the reaction network, the tableau and the raw log-space
            solution.
        """
        return self._section

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float = 298.15,
        pH: Array | float | None = None,
        base_addition: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform multi-stage extraction, at whichever level was configured.

        Args:
            feed: Aqueous feed stream (REE solution)
            solvent: Organic solvent stream
            T: Temperature (K)
            pH: Operating pH (overrides params if provided). **Correlation
                level only.** Under ``model="mass_action"`` the pH is an
                output, so passing one here is refused rather than silently
                ignored (#196).
            base_addition: Strong monoacidic base dosed into the aqueous feed
                (mol/s). **Closed level only**; this is the input that
                replaces the specified pH.

        Returns:
            raffinate: Aqueous outlet (depleted)
            extract: Organic outlet (loaded)
            info: Stage profiles and diagnostics. With ``include_loading``
                enabled it also reports the capacity condition (#193):
                ``theta_total`` (dimensionless organic loading, 1.0 =
                saturated), ``theta_solvent``, ``free_fraction_in``,
                ``capacity`` (F_extractant / m as a molar flow),
                ``uncapped_extracted``, ``capacity_scale``,
                ``capacity_clamped_fraction`` and ``capacity_sharpness``.

                Under ``model="mass_action"``, ``info`` instead carries
                ``pH_profile`` (an output), ``pH``, ``residual_norm``,
                ``feasible``, ``theta``, ``free_extractant``, ``D``,
                ``charge_imbalance`` and ``solution``.

        Raises:
            ValueError: If the feed has no aqueous species or the solvent has
                no organic carrier (#192), or if a ``pH`` is supplied to the
                closed model, where it is an output (#196).
        """
        p = self.params

        if self._section is not None:
            if pH is not None:
                raise ValueError(
                    "REEExtractor(model='mass_action') was called with an "
                    "explicit pH, but at this level pH is an OUTPUT of the "
                    "proton balance, not an input: every trivalent ion "
                    "extracted releases three protons, and the profile is "
                    "solved for. The input that replaces it is "
                    "base_addition (mol/s of strong base into the aqueous "
                    "feed). To reproduce a design specified at a given pH, "
                    "solve the inverse problem explicitly with "
                    "difflow_ree.equilibrium.mass_action.base_addition_for_pH "
                    "(#196). params.pH is used as the constant-calibration "
                    "pH for the equilibrium constants."
                )
            return self._section(
                feed, solvent, T=T, base_addition=base_addition
            )
        if base_addition is not None:
            raise ValueError(
                "base_addition is only meaningful for "
                "REEExtractor(model='mass_action'): the correlation level has "
                "no acid balance to dose into, and its pH is a parameter "
                "(#196)."
            )

        pH = pH if pH is not None else p.pH
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)

        # Get aqueous and organic carrier flows from the one shared
        # definition (#192): organic = extractant + diluent, aqueous =
        # everything else. Raises rather than defaulting when a phase is
        # missing from the stream it is required in.
        F_aq, _ = _phase_flows(
            feed_flows, p.extractant, p.diluent,
            require=("aqueous",), stream_name="REEExtractor feed",
        )
        _, F_org = _phase_flows(
            solvent_flows, p.extractant, p.diluent,
            require=("organic",), stream_name="REEExtractor solvent",
        )
        # Guard the denominator with a floor *relative* to the streams' own
        # total flow, not the absolute 1e-10 that used to sit here (#189).
        # An absolute floor is a hidden unit: it made recoveries depend on
        # whether the cascade was written in mol/s or Tmol/s. This one is
        # invariant to a uniform rescale by construction, so #189's invariance
        # claim holds over the whole float64 range. ``_phase_flows`` has
        # already raised if F_aq is concretely zero; this only catches the
        # traced path.
        F_aq = jnp.maximum(F_aq, _PHASE_FLOW_FLOOR_REL * (F_aq + F_org))
        # The loading capacity is a molar flow of extractant, F_extractant / m
        # (#191, #193), so with include_loading enabled the solvent has to say
        # how much extractant it carries. Without it the capacity would be
        # identically zero and the unit would silently return zero recovery.
        # This is a Python-level key check, not a check on a traced value.
        if self._isotherm is not None and p.extractant not in solvent_flows:
            raise ValueError(
                f"REEExtractor solvent does not declare a flow of the "
                f"extractant '{p.extractant}', but include_loading=True needs "
                f"it to compute the loading capacity F_extractant / m. "
                f"Species present: {sorted(solvent_flows)}. Add the extractant "
                f"to the solvent stream, or set include_loading=False."
            )
        F_extractant = jnp.asarray(solvent_flows.get(p.extractant, 0.0))

        # Get distribution coefficients
        D_values = self._distribution.get_D_all(pH, T)

        # Apply speciation correction if enabled
        if self._speciation is not None:
            for elem in p.elements:
                alpha = self._speciation.free_fraction(elem, p.ligand_conc, pH)
                D_values[elem] = D_values[elem] * alpha

        # Solve extraction using Kremser equation
        n_stages = jnp.asarray(p.n_stages, dtype=jnp.float64)

        raffinate_flows = dict(feed_flows)
        extract_flows = dict(solvent_flows)
        stage_profiles = {}

        # Where each loading effect lives (#189, #190, #191).
        #
        # Free-extractant depletion, D proportional to [HA]_free^n, is applied
        # in EXACTLY ONE place: the ``n * log10(concentration / C_ref)`` term
        # in ``REEDistribution.get_D`` (difflow_ree/equilibrium/distribution.py),
        # where the correlation carries the extractant-concentration
        # dependence. The per-element ``LoadingIsotherm.apparent_D`` call that
        # used to sit in this loop applied the same (1 - theta)^m physics a
        # second time, so D fell off as the 2m-th power of free extractant
        # instead of the m-th (#190). It has been removed from the stage path;
        # ``apparent_D`` remains supported public API for callers holding a D
        # that does not already carry the concentration term.
        #
        # What remains here is finite capacity, which is a different effect:
        # extractant already carrying REE cannot carry more (theta_solvent
        # below), and the stage total cannot exceed F_extractant / m (the
        # smooth limiter after the loop, #193). Both are expressed as genuine
        # dimensionless loading fractions,
        #
        #     theta = m * (REE molar flow in organic) / (extractant molar flow)
        #
        # which is invariant to the unit the flows are expressed in (#189).
        # Converging a single mass-action extractant balance would subsume all
        # of this and is tracked separately (#196).
        if self._isotherm is not None:
            m = self._isotherm.m
            k_sharp = float(p.capacity_sharpness)
            F_ree_solvent = sum(
                jnp.asarray(solvent_flows.get(e, 0.0)) for e in p.elements
            )
            # Relative floor again (#189): F_org is the same stream's own
            # carrier flow, so this is a unit-invariant guard, not a magic
            # molar flow.
            F_ext_safe = jnp.maximum(
                F_extractant, _PHASE_FLOW_FLOOR_REL * F_org
            )
            theta_solvent = m * F_ree_solvent / F_ext_safe
            # Smooth, never dead (#193). ``jnp.maximum(1 - theta, 0)`` kinked
            # here and was exactly zero beyond theta = 1, which killed the
            # gradient on the recycled-solvent path an extract-strip circuit
            # travels; see _smooth_free_fraction.
            free_fraction_in = _smooth_free_fraction(theta_solvent, k_sharp)

        for elem in p.elements:
            D = D_values[elem]
            F_in = jnp.asarray(feed_flows.get(elem, 0.0))
            F_solvent = jnp.asarray(solvent_flows.get(elem, 0.0))

            # Extraction factor E = D * (F_org / F_aq)
            E = D * F_org / F_aq

            # Adjust for initial organic loading (loaded solvent reduces the
            # extractant left free to bind more REE). theta_solvent is the
            # stage-level dimensionless loading of the entering solvent (#189).
            if self._isotherm is not None:
                E = E * free_fraction_in
            else:
                # Simple loading correction without isotherm:
                # Reduce E based on ratio of existing loading to feed. Guarded
                # with a where rather than an absolute 1e-10 floor, which was
                # a hidden unit here too (#189): with both flows below 1e-10
                # the floored ratio was not the ratio at all.
                denom = F_in + F_solvent
                safe_denom = jnp.where(denom > 0.0, denom, 1.0)
                loading_ratio = jnp.where(
                    denom > 0.0, F_solvent / safe_denom, 0.0
                )
                E = E * (1.0 - loading_ratio)

            # Kremser equation for counter-current extraction
            E_Np1 = jnp.power(E, n_stages + 1)

            frac_remaining = jnp.where(
                jnp.abs(E - 1.0) < 1e-6,
                1.0 / (n_stages + 1),
                safe_divide(E - 1.0, E_Np1 - 1.0)
            )
            frac_remaining = jnp.clip(frac_remaining, 0.0, 1.0)

            F_raffinate = F_in * frac_remaining
            F_extract = F_solvent + F_in * (1.0 - frac_remaining)

            raffinate_flows[elem] = jnp.maximum(F_raffinate, 0.0)
            extract_flows[elem] = jnp.maximum(F_extract, 0.0)

            stage_profiles[elem] = {
                "D": D,
                "E": E,
                "recovery": 1.0 - frac_remaining,
            }

        # Enforce total extractant loading capacity (Bug #112, smoothed #193)
        # The Kremser equation can predict extraction beyond physical capacity
        # when multiple REE are extracted simultaneously.
        #
        # The capacity is a molar flow, F_extractant / m, in the same units as
        # the extracted total: the old ``max_ree_conc * F_org`` multiplied a
        # concentration by a flow and used the whole organic flow, diluent
        # included, as if it were extractant (#191/#193).
        #
        # The limiter is the smooth saturation
        #     scale = (1 + (total/capacity)**k)**(-1/k)
        # which is 1 well below capacity and capacity/total well above it, but
        # unlike the hard jnp.where clamp it replaces is C-infinity, so
        # jax.grad is continuous at the capacity constraint an optimizer sits
        # on. It also guarantees total * scale < capacity strictly, so the
        # extractant balance never binds more extractant than was fed.
        capacity_info = {}
        if self._isotherm is not None:
            total_newly_extracted = sum(
                extract_flows[elem] - jnp.asarray(solvent_flows.get(elem, 0.0))
                for elem in p.elements
            )
            total_newly_extracted = jnp.maximum(total_newly_extracted, 0.0)
            max_capacity = F_extractant / self._isotherm.m
            k = k_sharp
            scale = _soft_saturation(total_newly_extracted, max_capacity, k)
            for elem in p.elements:
                F_solvent_elem = jnp.asarray(solvent_flows.get(elem, 0.0))
                newly_extracted = extract_flows[elem] - F_solvent_elem
                extract_flows[elem] = F_solvent_elem + newly_extracted * scale
                raffinate_flows[elem] = (
                    jnp.asarray(feed_flows.get(elem, 0.0))
                    + F_solvent_elem
                    - extract_flows[elem]
                )

            # Make the capacity condition observable (#193): a design pinned
            # against the capacity wall looks converged otherwise.
            F_ree_org_out = sum(extract_flows[e] for e in p.elements)
            capacity_info = {
                # Dimensionless loading of the organic outlet, 1.0 = saturated
                "theta_total": (
                    self._isotherm.m * F_ree_org_out / F_ext_safe
                ),
                # Loading of the entering solvent, on the same basis
                "theta_solvent": theta_solvent,
                # Extractant capacity as a molar flow, F_extractant / m
                "capacity": max_capacity,
                # Uncapped Kremser prediction, same units as capacity
                "uncapped_extracted": total_newly_extracted,
                # Limiter multiplier applied to the newly extracted total
                "capacity_scale": scale,
                # Fraction of the uncapped prediction the limiter removed
                "capacity_clamped_fraction": 1.0 - scale,
                "capacity_sharpness": jnp.asarray(k),
                # Free-extractant fraction of the *entering* solvent, the
                # smooth replacement for max(1 - theta_solvent, 0) (#193)
                "free_fraction_in": free_fraction_in,
            }

        # Create output streams
        P = feed["P"]
        raffinate = make_stream(raffinate_flows, T, P)
        extract = make_stream(extract_flows, T, P)

        info = {
            "n_stages": n_stages,
            "pH": pH,
            "T": T,
            "profiles": stage_profiles,
            "D_values": D_values,
        }
        info.update(capacity_info)

        return raffinate, extract, info


# =============================================================================
# Mixer-Settler Unit
# =============================================================================

@dataclass(repr=False)
class MixerSettlerParams(ParamsMixin):
    """Parameters for single mixer-settler stage.

    Attributes:
        extractant: Extractant name
        elements: REE elements to track
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        pH: Operating pH
        extractant_conc: Extractant concentration (M)
        mixer_residence_time: Mixer residence time (s)
        settler_residence_time: Settler residence time (s)
        stage_efficiency: Murphree stage efficiency (0-1)
    """
    extractant: str
    elements: tuple[str, ...]
    diluent: str = "kerosene"
    pH: float = 3.0
    extractant_conc: float = 0.5
    mixer_residence_time: float = 120.0  # 2 minutes typical
    settler_residence_time: float = 300.0  # 5 minutes typical
    stage_efficiency: float = 0.95
    # Aqueous nitrate concentration (M), required for solvating extractants
    # such as TBP whose D is nitrate- rather than pH-driven (#195).
    nitrate_conc: float | None = None
    # Explicit extraction-mechanism override ("cation_exchange"/"solvating")
    # for this stage's REEDistribution; None takes it from the record (#195).
    mechanism: str | None = None
    # Extraction kinetics (#118). When k_extraction (overall rate constant,
    # 1/s) is set, the effective stage efficiency is the kinetic approach to
    # equilibrium 1 - exp(-k * mixer_residence_time) rather than the fixed
    # Murphree stage_efficiency, so slow kinetics / short mixing under-shoot
    # equilibrium. None keeps the constant efficiency (backward compatible).
    k_extraction: float | None = None
    # Phase entrainment (#110). Fractions of one phase physically carried into
    # the opposite outlet (organic droplets in aqueous, aqueous droplets in
    # organic), carrying their dissolved REE across and reducing separation.
    # Typical values 0.001-0.01; 0 disables (backward compatible).
    entrainment_org_in_aq: float = 0.0
    entrainment_aq_in_org: float = 0.0
    # Third-phase formation (#117). If set, an organic loading (mol REE per mol
    # extractant) above this limit flags third-phase onset in info, and info
    # also carries the smooth signed margin ``limit - loading`` so the boundary
    # can be posed as an inequality constraint rather than only read as a
    # boolean (#193). None disables the check (backward compatible).
    third_phase_loading_limit: float | None = None


class REEMixerSettler:
    """Single mixer-settler stage for REE extraction.

    Models one equilibrium stage with efficiency factor.

    Example:
        >>> params = MixerSettlerParams(
        ...     extractant="D2EHPA",
        ...     elements=("Nd", "Dy"),
        ...     pH=3.0,
        ... )
        >>> stage = REEMixerSettler(params)
        >>> aq_out, org_out, info = stage(aq_in, org_in)
    """

    symbol = "Mixer-Settler"
    equations = [
        r"E_M = \frac{y_\mathrm{out} - y_\mathrm{in}}{y^\ast - y_\mathrm{in}}\qquad \text{(Murphree stage efficiency)}",
        r"y^\ast_i = D_i\,x_i\qquad (D_i = D_i(\mathrm{pH}, [\mathrm{HA}]))",
    ]
    assumptions = [
        "Single stage with configurable Murphree efficiency.",
        "Mixer reaches near-equilibrium; settler provides phase separation only.",
    ]
    references = [
        "Perry's Chemical Engineers' Handbook, 9e, Sec. 15.",
    ]
    parameter_symbols = {
        "stage_efficiency": r"E_M",
        "mixer_residence_time": r"\tau_\mathrm{mix}",
        "settler_residence_time": r"\tau_\mathrm{set}",
    }
    parameter_units = {
        "mixer_residence_time": "s",
        "settler_residence_time": "s",
    }

    def __init__(self, params: MixerSettlerParams):
        """Initialize mixer-settler.

        Args:
            params: Stage parameters
        """
        self.params = params
        self._distribution = REEDistribution(
            extractant=params.extractant,
            elements=params.elements,
            concentration=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,
            mechanism=params.mechanism,
        )

    def __call__(
        self,
        aqueous_in: Stream,
        organic_in: Stream,
        T: Array | float = 298.15,
        pH: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform single-stage extraction.

        Args:
            aqueous_in: Inlet aqueous stream
            organic_in: Inlet organic stream
            T: Temperature (K)
            pH: Operating pH

        Returns:
            aqueous_out: Outlet aqueous stream
            organic_out: Outlet organic stream
            info: Stage information. With ``third_phase_loading_limit`` set it
                also reports ``organic_loading``, the boolean
                ``third_phase_formed`` and the smooth signed
                ``third_phase_margin`` (#193).

        Raises:
            ValueError: If the aqueous inlet has no aqueous species, if the
                organic inlet has no organic carrier (#192), or if
                ``third_phase_loading_limit`` is set and the organic inlet
                does not declare a flow of the extractant.
        """
        p = self.params
        pH = pH if pH is not None else p.pH
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)

        aq_flows = get_flows(aqueous_in)
        org_flows = get_flows(organic_in)

        # Same phase-flow definition as REEExtractor (#192). The aqueous flow
        # is the whole aqueous phase, not just H2O, which for a concentrated
        # leach liquor is a large difference and used to make the two units
        # disagree on identical inputs.
        F_aq, _ = _phase_flows(
            aq_flows, p.extractant, p.diluent,
            require=("aqueous",), stream_name="REEMixerSettler aqueous inlet",
        )
        _, F_org = _phase_flows(
            org_flows, p.extractant, p.diluent,
            require=("organic",), stream_name="REEMixerSettler organic inlet",
        )
        # Relative, unit-invariant floor rather than an absolute 1e-10 (#189);
        # see _PHASE_FLOW_FLOOR_REL.
        F_aq = jnp.maximum(F_aq, _PHASE_FLOW_FLOOR_REL * (F_aq + F_org))

        # The third-phase check reports an organic loading, mol REE per mol
        # extractant, so it needs the extractant's molar flow. Dividing by a
        # missing flow floored at 1e-30 reported loadings of order 1e29 and
        # called it a converged answer, which is worse than the silent default
        # #192 objected to. REEExtractor already raises for the same condition;
        # this is the same contract. Python-level key check, so it is safe
        # under jit and grad.
        if p.third_phase_loading_limit is not None and p.extractant not in org_flows:
            raise ValueError(
                f"REEMixerSettler organic inlet does not declare a flow of the "
                f"extractant '{p.extractant}', but third_phase_loading_limit "
                f"is set and the loading it is compared against is "
                f"(mol REE in organic) / (mol extractant). Species present: "
                f"{sorted(org_flows)}. Add the extractant to the organic "
                f"stream, or set third_phase_loading_limit=None."
            )
        F_extractant = jnp.asarray(org_flows.get(p.extractant, 0.0))

        D_values = self._distribution.get_D_all(pH, T)

        # Effective stage efficiency: kinetic approach to equilibrium when a
        # rate constant is supplied (#118), else the fixed Murphree value.
        if p.k_extraction is not None:
            eta = approach_to_equilibrium(p.mixer_residence_time, p.k_extraction)
        else:
            eta = jnp.asarray(p.stage_efficiency)

        # Start with full copies to preserve non-extracted species (Bug #53)
        aq_out_flows = dict(aq_flows)
        org_out_flows = dict(org_flows)

        for elem in p.elements:
            D = D_values[elem]
            F_aq_in = jnp.asarray(aq_flows.get(elem, 0.0))
            F_org_in = jnp.asarray(org_flows.get(elem, 0.0))

            # Total solute
            F_total = F_aq_in + F_org_in

            # Equilibrium distribution
            # At equilibrium: c_org = D * c_aq
            # Mass balance: F_aq * c_aq + F_org * c_org = F_total
            # Solving: c_aq = F_total / (F_aq + D * F_org)
            c_aq_eq = F_total / (F_aq + D * F_org)
            c_org_eq = D * c_aq_eq

            F_aq_eq = c_aq_eq * F_aq
            F_org_eq = c_org_eq * F_org

            # Apply stage efficiency
            F_aq_out = F_aq_in + eta * (F_aq_eq - F_aq_in)
            F_org_out = F_org_in + eta * (F_org_eq - F_org_in)

            # Clip aqueous and derive organic to preserve mass balance
            F_aq_out = jnp.maximum(F_aq_out, 0.0)
            F_org_out = F_total - F_aq_out
            aq_out_flows[elem] = F_aq_out
            org_out_flows[elem] = jnp.maximum(F_org_out, 0.0)

        # Phase entrainment (#110): a fraction of each phase (and every species
        # it carries) is entrained into the opposite outlet, so dissolved REE
        # crosses back and separation degrades. Mass is conserved species-wise.
        f_oa = jnp.asarray(p.entrainment_org_in_aq)
        f_ao = jnp.asarray(p.entrainment_aq_in_org)
        if p.entrainment_org_in_aq or p.entrainment_aq_in_org:
            all_species = set(aq_out_flows) | set(org_out_flows)
            entrained_aq = {}
            entrained_org = {}
            for s in all_species:
                a = jnp.asarray(aq_out_flows.get(s, 0.0))
                o = jnp.asarray(org_out_flows.get(s, 0.0))
                entrained_aq[s] = a * (1.0 - f_ao) + o * f_oa
                entrained_org[s] = o * (1.0 - f_oa) + a * f_ao
            aq_out_flows = entrained_aq
            org_out_flows = entrained_org

        P = aqueous_in["P"]
        aqueous_out = make_stream(aq_out_flows, T, P)
        organic_out = make_stream(org_out_flows, T, P)

        info = {
            "D_values": D_values,
            "efficiency": eta,
        }
        if p.k_extraction is not None:
            info["kinetic_efficiency"] = eta
            info["mixer_residence_time"] = jnp.asarray(p.mixer_residence_time)

        # Third-phase formation check (#117): organic loading vs the limit.
        if p.third_phase_loading_limit is not None:
            total_org_ree = sum(jnp.asarray(org_out_flows.get(e, 0.0)) for e in p.elements)
            # The key check above guarantees the extractant is declared; a
            # declared-but-zero flow can still arrive as a traced value, so the
            # denominator keeps a floor -- relative to the organic carrier flow
            # of the same stream, so it carries no hidden unit (#189).
            loading = total_org_ree / jnp.maximum(
                F_extractant, _PHASE_FLOW_FLOOR_REL * F_org
            )
            info["organic_loading"] = loading
            info["third_phase_formed"] = loading > p.third_phase_loading_limit
            # Signed margin so the limit can be used as a smooth inequality
            # constraint in an optimization, not only read as a diagnostic
            # boolean (#193). Positive is feasible; negative means a third
            # phase has formed, and the magnitude says by how much. The
            # boolean has zero gradient, so an optimizer would otherwise walk
            # straight through the boundary because crossing it is profitable
            # in the model.
            info["third_phase_margin"] = (
                jnp.asarray(p.third_phase_loading_limit) - loading
            )

        return aqueous_out, organic_out, info


# =============================================================================
# Multi-Stage Cascade Builder
# =============================================================================

def build_extraction_cascade(
    n_stages: int,
    params: MixerSettlerParams,
) -> list[REEMixerSettler]:
    """Build a list of mixer-settler stages.

    Args:
        n_stages: Number of stages
        params: Parameters for each stage

    Returns:
        List of mixer-settler units
    """
    return [REEMixerSettler(params) for _ in range(n_stages)]


def solve_cascade_sequential(
    stages: list[REEMixerSettler],
    feed: Stream,
    solvent: Stream,
    T: float = 298.15,
    pH: float | None = None,
) -> tuple[Stream, Stream, list[dict]]:
    """Solve co-current cascade sequentially.

    Args:
        stages: List of mixer-settler stages
        feed: Aqueous feed
        solvent: Organic solvent
        T: Temperature (K)
        pH: Operating pH

    Returns:
        raffinate: Final aqueous outlet
        extract: Final organic outlet
        stage_info: List of stage information dicts
    """
    aq = feed
    org = solvent
    stage_info = []

    for stage in stages:
        aq, org, info = stage(aq, org, T, pH)
        stage_info.append(info)

    return aq, org, stage_info
