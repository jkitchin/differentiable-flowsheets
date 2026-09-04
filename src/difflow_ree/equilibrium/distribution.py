"""Distribution coefficient models for REE extraction.

Distribution coefficient D = [REE]_org / [REE]_aq

The driving variable depends on the extraction *mechanism*, which is carried
by the extractant record rather than assumed by this module (#195):

- ``cation_exchange`` (D2EHPA, PC88A, Cyanex272)::

      RE3+ + p (HA)2  <->  RE(HA2)p + p H+
      log10(D) = a + b*pH + c*pH^2 + d*(1/T - 1/Tref)

- ``solvating`` (TBP, and the diglycolamide/malonamide families to come)::

      RE(NO3)3 + m S  <->  RE(NO3)3.mS        (no protons released)
      log10(D) = a + b*s + c*s^2 + d*(1/T - 1/Tref),
      s = log10([NO3-] / [NO3-]_ref)

pH SCALE (#194): ``pH`` is ``-log10([H+])``, the *concentration* scale, matching
the header of ``data/extractants.yaml``. The correlations are conditional
constants fitted at the ionic strength of their source experiments, so the
defensible default is no activity correction at all (``ionic_strength=None``).

ACTIVITY RANGE (#194): when an ionic strength *is* supplied, the value handed to
the activity model is clamped at that model's documented validity limit. Davies'
bracket ``sqrt(I)/(1+sqrt(I)) - 0.3 I`` changes sign at I = 1.940363884733242 M,
above which the correction would MULTIPLY D (6.5x at 3 M, 42.5x at 4 M) instead
of reducing it -- precisely the 2-4 M chloride regime #194 is about. Clamping is
arithmetic, so it holds for tracers too, where no Python check can see the value.
Raw extrapolation is available on request via ``extrapolate_activity_model=True``.

All functions are JAX-compatible for automatic differentiation.
"""

import warnings
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.numerics import safe_divide
from difflow_ree.database import (
    get_extractant_database,
    get_extractant,
    EXTRACTION_MECHANISMS,
    PHCoefficients,
)
from difflow_ree.equilibrium.speciation import (
    ACTIVITY_MODELS,
    AQUEOUS_MEDIA,
    NITRATE_BEARING_MEDIA,
    activity_coefficient,
)


# How an out-of-range / not-applicable activity correction is reported (#194).
OutOfRangeAction = Literal["warn", "raise", "ignore"]


def _concrete_bounds(value) -> tuple[float, float] | None:
    """Return ``(min, max)`` of ``value`` as floats, or None for a JAX tracer.

    Works for Python scalars, numpy scalars, and *concrete* JAX/numpy arrays of
    any shape -- a concrete array is inspectable and must be range-checked, it
    is only ``float()`` that refuses it because it is non-scalar (#194). None
    is returned only when the value genuinely cannot be inspected, i.e. it is
    an abstract tracer under ``jit``/``grad``/``vmap``. Every JAX abstraction
    error raised here (``TracerArrayConversionError``,
    ``ConcretizationTypeError``) is a ``TypeError`` subclass.

    Args:
        value: Scalar, array or tracer.

    Returns:
        ``(minimum, maximum)`` as Python floats, or None if abstract or empty.
    """
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return float(arr.min()), float(arr.max())


# =============================================================================
# Distribution Coefficient Model
# =============================================================================

@dataclass
class REEDistribution:
    """REE distribution coefficient calculator.

    Which correlation drives ``D`` is decided by the *extraction mechanism*
    carried on the extractant record, not assumed by this class (#195):
    ``cation_exchange`` extractants are driven by pH via ``ph_coefficients``,
    ``solvating`` extractants by the nitrate concentration via
    ``nitrate_coefficients``. There is no silent fall-back between the two, and
    a record need not carry both: TBP carries only ``nitrate_coefficients``, so
    asking for it with ``mechanism="cation_exchange"`` raises.
    Two things are actually checked, and they are the only two the data can
    support: the *driving ion* must have a positive concentration (a solvating
    extractant with no nitrate raises), and, when the caller states a
    :attr:`medium`, it must satisfy the record's ``requires_nitrate`` flag.

    ACTIVITY CONVENTION (#194). ``pH`` is on the *concentration* scale,
    ``pH = -log10([H+])``. The tabulated correlations are conditional constants
    fitted at the ionic strength of their source experiments, so
    ``ionic_strength=None`` -- no activity correction at all -- is a first-class
    and usually the *correct* choice: for a 2-4 M chloride leach liquor a
    conditional constant used at the liquor's own ionic strength is the standard
    and defensible treatment, and every implemented activity model is out of its
    validity range there. Supplying ``ionic_strength`` requests a correction to
    a different ionic strength; that correction is
    ``gamma_RE3+ / gamma_H+**p`` with ``p = Extractant.stoichiometry_protons``,
    which is the residual activity factor once the concentration-scale ``[H+]**-p``
    dependence is already carried by the ``b*pH`` term.

    THE DAVIES SIGN INVERSION, AND WHY THE MODEL INPUT IS CLAMPED (#194).
    Davies writes ``log10(gamma) = -A z^2 f(I)`` with
    ``f(I) = sqrt(I)/(1+sqrt(I)) - 0.3 I``. ``f`` is *not* monotone: it peaks
    near I = 0.4 M and changes sign at

        I = 1.940363884733242 M

    (the root of ``0.3 I + 0.3 sqrt(I) - 1 = 0``; see
    :data:`~difflow_ree.equilibrium.speciation.DAVIES_SIGN_CHANGE_IONIC_STRENGTH`).
    Above that ionic strength ``gamma_RE/gamma_H**3 = 10**(-6 A f)`` exceeds 1
    and the "correction" *multiplies* D instead of reducing it: 0.228 at
    I = 0.1 M, 0.245 at 1.0 M, 1.0 at 1.9404 M, 6.49 at 3.0 M, 42.5 at 4.0 M.
    That is precisely the 2-4 M chloride regime issue #194 was filed about, so
    it is not an acceptable silent behaviour and a warning is not a sufficient
    guard (warnings are routinely filtered, and cannot fire at all for a value
    that is an abstract tracer).

    The chosen guard is therefore in the *arithmetic*, not only in the report:
    the ionic strength handed to the activity model is clamped to that model's
    documented ``max_ionic_strength`` (0.5 M for Davies) unless
    :attr:`extrapolate_activity_model` is set. Beyond the documented range the
    correction saturates at its end-of-range value instead of reversing; it is
    never inverted, and ``d D / d I`` is exactly zero there, which is the
    honest statement that the model carries no information about that regime.
    This is the only option of the three considered that also holds under
    ``jit``/``grad``/``vmap``, where the ionic strength is an abstract tracer
    and *no* Python-level check can inspect it: refusing to trace would have
    broken gradient-based design studies outright, and an opt-in flag for
    traced values would have left the inverted branch reachable by anyone who
    set the flag for an in-range sweep. Raw, possibly inverted Davies remains
    available, but only by explicitly asking for it with
    ``extrapolate_activity_model=True``.

    Attributes:
        extractant: Name of extractant (D2EHPA, PC88A, Cyanex272, TBP)
        elements: List of REE symbols to include
        concentration: Extractant concentration (M)
        nitrate_conc: Aqueous nitrate concentration (M). Required for
            solvating extractants such as TBP, whose distribution ratio is
            driven by the salting anion rather than by pH (#195). Ignored by
            cation-exchange extractants. ``None`` (or a non-positive value)
            means no nitrate is available to drive the correlation.
        medium: Aqueous medium, one of
            :data:`~difflow_ree.equilibrium.speciation.AQUEOUS_MEDIA`
            (``"sulfate"``, ``"chloride"``, ``"nitrate"``, ``"mixed"``), or
            ``None`` (default) to leave it unstated. This is the *only* medium
            concept the data supports: the extractant records carry a single
            medium constraint, ``stoichiometry.requires_nitrate``, so declaring
            ``medium="chloride"`` or ``"sulfate"`` for a nitrate-requiring
            extractant is detected and raises. No record declares a chloride or
            sulfate *incompatibility*, so no other medium combination is
            rejected; a cation-exchange extractant is accepted in every medium
            (#195, D10b).
        mechanism: Explicit mechanism override, one of
            ``"cation_exchange"`` / ``"solvating"``. ``None`` (default) takes
            the mechanism from the extractant record. An override is only
            honoured when the record actually carries that mechanism's
            coefficient block; asking for ``"cation_exchange"`` on a record
            that has none raises. In particular TBP no longer has a
            ``ph_coefficients`` block at all -- it was deleted as
            mechanistically unsupported (TBP is neutral: pKa None, zero protons
            released) -- so ``REEDistribution(extractant="TBP", ...,
            mechanism="cation_exchange")`` now raises a ValueError pointing at
            the nitrate path, rather than silently using it.
        activity_model: Aqueous activity model used when ``ionic_strength`` is
            supplied, a key of
            :data:`difflow_ree.equilibrium.speciation.ACTIVITY_MODELS`
            (``"davies"``, ``"none"``). Each declares its own validity range
            (#194).
        on_out_of_range: What to do when ``ionic_strength`` is outside the
            chosen model's validity range, cannot be checked because it is an
            abstract tracer, or when the correction is not the right physics for
            the mechanism: ``"warn"`` (default, UserWarning), ``"raise"``
            (ValueError) or ``"ignore"``. This controls *reporting* only; the
            clamp described above applies regardless.
        extrapolate_activity_model: Opt in to feeding the activity model an
            ionic strength beyond its documented validity range, reproducing raw
            (and, above 1.94 M for Davies, sign-inverted) coefficients. Default
            False. Setting this is the explicit request that makes an inverted
            ``D`` reachable; nothing else does (#194).

    Example:
        >>> # Cation exchange: driven by pH
        >>> dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        >>> float(dist.get_D("Nd", pH=3.0)) > 0
        True
        >>> # Solvating: driven by nitrate, pH is not a driving variable
        >>> tbp = REEDistribution(
        ...     extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        ... )
        >>> float(tbp.get_D("Nd")) > 0
        True
    """
    extractant: str
    elements: tuple[str, ...]
    concentration: float = 0.5  # M
    nitrate_conc: float | None = None  # M; see #195
    medium: str | None = None  # see AQUEOUS_MEDIA (#195)
    mechanism: str | None = None  # None -> take from the record (#195)
    activity_model: str = "davies"  # see ACTIVITY_MODELS (#194)
    on_out_of_range: OutOfRangeAction = "warn"  # (#194)
    extrapolate_activity_model: bool = False  # (#194) opt in to raw Davies

    def __post_init__(self):
        """Load extractant data and resolve/validate the mechanism (#195)."""
        self._ext_data = get_extractant(self.extractant)

        if self.medium is not None and self.medium not in AQUEOUS_MEDIA:
            raise ValueError(
                f"Unknown medium {self.medium!r}. Recognized aqueous media: "
                f"{list(AQUEOUS_MEDIA)} (#195)."
            )

        # Mechanism: explicit override, else the record's own declaration.
        self._mechanism_is_override = self.mechanism is not None
        if self.mechanism is None:
            self.mechanism = self._ext_data.mechanism
        if self.mechanism not in EXTRACTION_MECHANISMS:
            raise ValueError(
                f"REEDistribution(extractant={self.extractant!r}): unknown "
                f"mechanism {self.mechanism!r}. Supported mechanisms: "
                f"{list(EXTRACTION_MECHANISMS)}."
            )

        if self.activity_model not in ACTIVITY_MODELS:
            raise ValueError(
                f"Unknown activity_model {self.activity_model!r}. Implemented "
                f"models: {sorted(ACTIVITY_MODELS)}. Bromley and SIT are not "
                "implemented because difflow_ree does not carry their "
                "ion-interaction parameters (#194)."
            )
        if self.on_out_of_range not in ("warn", "raise", "ignore"):
            raise ValueError(
                f"on_out_of_range must be 'warn', 'raise' or 'ignore', got "
                f"{self.on_out_of_range!r}."
            )

        # Warnings are emitted once per instance per distinct condition so a
        # stage loop cannot spam (#194); the check itself is Python-level.
        self._reported: set = set()

        self._check_medium()
        self._validate_mechanism_data(self.nitrate_conc)

    # -------------------------------------------------------------------
    # Mechanism dispatch (#195)
    # -------------------------------------------------------------------

    def _check_medium(self) -> None:
        """Check the declared medium against the record's medium constraint.

        The extractant records carry exactly one medium constraint,
        ``stoichiometry.requires_nitrate``, so exactly one thing can be
        detected here: a nitrate-requiring extractant declared to be operating
        in a medium that supplies no nitrate. Nothing in the data expresses a
        chloride or sulfate *incompatibility* for the acidic extractants, so
        nothing else is rejected, and an unstated ``medium`` is not guessed at
        (#195, D10b).

        Raises:
            ValueError: If ``requires_nitrate`` and the declared medium is not
                one of :data:`NITRATE_BEARING_MEDIA`.
        """
        if self.medium is None or not self._ext_data.requires_nitrate:
            return
        if self.medium in NITRATE_BEARING_MEDIA:
            return
        raise ValueError(
            f"Extractant {self.extractant!r} requires a nitrate medium "
            f"(stoichiometry.requires_nitrate is true on its record) but was "
            f"declared with medium={self.medium!r}, which supplies no nitrate. "
            f"Nitrate-bearing media: {list(NITRATE_BEARING_MEDIA)}. Its "
            "coefficients do not cover this medium; use a cation-exchange "
            "extractant, or state the medium the correlation was fitted in "
            "(#195)."
        )

    def _validate_mechanism_data(self, nitrate_conc) -> None:
        """Fail loudly when the record cannot support the chosen mechanism.

        Args:
            nitrate_conc: Nitrate concentration (M) that will drive a solvating
                correlation, or None for a non-nitrate medium.

        Raises:
            ValueError: If the mechanism's coefficient block is missing, or a
                solvating / ``requires_nitrate`` extractant is used without a
                positive nitrate concentration. Never falls back to the other
                mechanism's coefficients (#195).
        """
        ext = self._ext_data

        if self.mechanism == "solvating":
            if ext.nitrate_coefficients is None:
                raise ValueError(
                    f"Extractant {self.extractant!r} is used with "
                    "mechanism='solvating' but its record carries no "
                    "'nitrate_coefficients' block, so its distribution ratio "
                    "cannot be computed from the nitrate concentration (#195)."
                )
            if ext.reference_nitrate is None:
                raise ValueError(
                    f"Extractant {self.extractant!r} has "
                    "'nitrate_coefficients' but no 'reference_nitrate', so the "
                    "nitrate coefficients have no reference concentration to be "
                    "interpreted against (#195)."
                )
            self._require_nitrate_medium(nitrate_conc)
        else:  # cation_exchange
            if not ext.ph_coefficients:
                # No silent fall-back to nitrate_coefficients, and no
                # AttributeError/KeyError leaking out of a later get_D: the
                # record simply has no pH-driven correlation, and for a
                # solvating extractant it never should have had one.
                extra = ""
                if ext.nitrate_coefficients:
                    extra = (
                        f" {self.extractant} extracts by SOLVATION, not by "
                        "cation exchange: its record has pKa=None and "
                        f"stoichiometry.protons_released="
                        f"{ext.stoichiometry_protons}, so there is no proton "
                        "to exchange and the pH-driven block it used to carry "
                        "was DELETED as mechanistically unsupported (it had no "
                        "source, and reproduced a selectivity spread that the "
                        "measured data refutes). Use the nitrate path instead: "
                        f"REEDistribution(extractant={self.extractant!r}, ..., "
                        "nitrate_conc=<[NO3-] in M>) with the default "
                        "mechanism, referenced to "
                        f"{ext.reference_nitrate} M nitrate. If you really "
                        "mean the HNO3 system, it is a real system but this "
                        "database has no coefficients fitted for it -- supply "
                        "your own via create_custom_extractant()."
                    )
                raise ValueError(
                    f"Extractant {self.extractant!r} is used with "
                    "mechanism='cation_exchange' but its record carries no "
                    "'ph_coefficients' block (#195)." + extra
                )
            if ext.requires_nitrate and not self._mechanism_is_override:
                raise ValueError(
                    f"Extractant {self.extractant!r} requires a nitrate medium "
                    "but resolved to a pH-driven cation-exchange model. This "
                    "is a data inconsistency; declare mechanism: solvating on "
                    "the record (#195)."
                )

    def _require_nitrate_medium(self, nitrate_conc) -> None:
        """Raise unless a usable nitrate concentration was supplied (#195).

        This is a *driving-ion* check, not a medium check: it asserts that the
        salting anion the solvating correlation is a function of actually has a
        positive concentration. The medium itself is checked separately, and
        only when the caller states one, in :meth:`_check_medium`.

        Args:
            nitrate_conc: Nitrate concentration (M), possibly None, an array,
                or a tracer. A concrete array is checked at its *minimum*, so a
                per-stage profile containing a zero is rejected.

        Raises:
            ValueError: If it is None, or has a concrete non-positive value.
        """
        name = self.extractant
        # The "way out" depends on whether the record still carries a pH block.
        # TBP's was deleted (neutral extractant, no proton to exchange, no
        # source), so for TBP there is exactly one path and the message must
        # not advertise a second one that raises.
        if self._ext_data.ph_coefficients:
            escape = (
                f"{name} extracts by solvation from a nitrate medium; if you "
                "specifically mean the HNO3 system that its 'ph_coefficients' "
                "block describes, opt in explicitly with "
                "mechanism='cation_exchange' (#195)."
            )
        else:
            escape = (
                f"This is the only path: {name} extracts by solvation and its "
                "record carries no 'ph_coefficients' block -- the pH-driven "
                "block it used to have was deleted as mechanistically "
                "unsupported, so mechanism='cation_exchange' raises rather "
                "than providing an alternative (#195)."
            )
        hint = (
            f"Pass nitrate_conc=<[NO3-] in M> (the record is referenced to "
            f"{self._ext_data.reference_nitrate} M). " + escape
        )
        if nitrate_conc is None:
            raise ValueError(
                f"Extractant {name!r} requires a nitrate concentration: it was "
                f"used with nitrate_conc=None, so the salting anion that drives "
                f"its correlation has no value. {hint}"
            )
        bounds = _concrete_bounds(nitrate_conc)
        if bounds is not None and bounds[0] <= 0.0:
            raise ValueError(
                f"Extractant {name!r} requires a nitrate concentration: it was "
                f"used with nitrate_conc={bounds[0]:g}, which is not a positive "
                f"concentration of the salting anion its correlation is a "
                f"function of. {hint}"
            )

    def _coefficients(self, element: str) -> PHCoefficients:
        """Coefficient record for ``element`` under the active mechanism.

        Args:
            element: REE symbol.

        Returns:
            The :class:`PHCoefficients` driving this element's correlation.

        Raises:
            KeyError: If the element is absent from the mechanism's block, with
                a message naming the extractant, the mechanism and the block.
        """
        if self.mechanism == "solvating":
            block = self._ext_data.nitrate_coefficients
            block_name = "nitrate_coefficients"
        else:
            block = self._ext_data.ph_coefficients
            block_name = "ph_coefficients"
        if not block:
            # Defensive: __post_init__ already refuses this combination, so
            # reaching here means the record was mutated afterwards. Fail with
            # the same explanation rather than a TypeError on None[element].
            raise ValueError(
                f"Extractant {self.extractant!r} has no {block_name!r} block, "
                f"so no coefficients exist for element {element!r} under "
                f"mechanism={self.mechanism!r}."
            )
        try:
            return block[element]
        except KeyError:
            raise KeyError(
                f"Extractant {self.extractant!r} has no {block_name} entry for "
                f"element {element!r} (mechanism={self.mechanism!r}). "
                f"Available: {sorted(block)}."
            ) from None

    # -------------------------------------------------------------------
    # Activity correction (#194)
    # -------------------------------------------------------------------

    def _report(self, key: str, message: str) -> None:
        """Warn / raise / ignore once per instance for a given condition.

        Args:
            key: Deduplication key so a stage loop cannot spam (#194).
            message: Text of the warning or error.

        Raises:
            ValueError: If ``on_out_of_range == "raise"``.
        """
        if self.on_out_of_range == "ignore":
            return
        if self.on_out_of_range == "raise":
            raise ValueError(message)
        if key in self._reported:
            return
        self._reported.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _check_activity_range(self, ionic_strength) -> None:
        """Parameter-time validity report for the activity model (#194).

        Runs at the Python level, once per distinct condition per instance. It
        inspects *any* concrete input -- scalar or array, checking the array's
        maximum -- and reports when the value is outside the model's documented
        range. When the value is an abstract tracer (``jit``/``grad``/``vmap``)
        there is nothing to inspect, and that fact is itself reported through
        :attr:`on_out_of_range` rather than passed over in silence.

        This is a *report*, not the guard. The guard against the Davies sign
        inversion is the clamp in :meth:`_activity_correction`, which does not
        depend on being able to inspect the value and therefore holds under
        tracing too.

        Args:
            ionic_strength: Ionic strength (M), possibly None, an array, or a
                tracer.
        """
        if ionic_strength is None or self.activity_model == "none":
            return

        # A solvating extractant's ionic-strength dependence is a nitrate
        # salting effect on the anion, which the aqueous-cation correction does
        # not represent. Do not silently reuse the cation-exchange form (#194).
        if self.mechanism == "solvating":
            self._report(
                "solvating_activity",
                f"Extractant {self.extractant!r} extracts by a solvating "
                "mechanism (p = 0 protons released), so the activity "
                "correction applied here is the aqueous RE3+ free-ion "
                "coefficient alone. Its real ionic-strength dependence is a "
                "nitrate salting effect on the anion activity, which "
                f"{self.activity_model!r} does not represent. Prefer "
                "ionic_strength=None (conditional constant fitted at the "
                "operating ionic strength) for solvating systems (#194).",
            )

        model = ACTIVITY_MODELS[self.activity_model]
        limit = model["max_ionic_strength"]
        flip = model.get("sign_change_ionic_strength")
        bounds = _concrete_bounds(ionic_strength)

        if bounds is None:
            # Abstract tracer: no Python-level check can see the value. Say so
            # rather than pretending the range was checked. The arithmetic is
            # still safe -- _activity_correction clamps the model input -- so
            # this reports an unverified input, not an inverted result.
            if not self.extrapolate_activity_model:
                self._report(
                    "traced",
                    "ionic_strength is a JAX tracer, so its validity against "
                    f"the {self.activity_model!r} activity model "
                    f"(I < {limit:g} M) cannot be range-checked. The ionic "
                    "strength fed to the model is clamped at "
                    f"{limit:g} M so the correction cannot invert (#194); the "
                    "correction, and hence d D / d I, is therefore flat beyond "
                    "that. Pass a concrete ionic_strength to get the check, "
                    "ionic_strength=None for a conditional constant, "
                    "on_out_of_range='ignore' to silence this, or "
                    "extrapolate_activity_model=True to extrapolate on purpose."
                )
            else:
                self._report(
                    "traced_extrapolating",
                    "ionic_strength is a JAX tracer and "
                    "extrapolate_activity_model=True, so the "
                    f"{self.activity_model!r} activity model is being used "
                    f"outside its validity range (I < {limit:g} M) with no "
                    "check possible. "
                    + (
                        f"Above I = {flip:.6g} M the correction changes sign "
                        "and multiplies D instead of reducing it (#194)."
                        if flip is not None else ""
                    )
                )
            return

        I = bounds[1]  # check the maximum; a concrete array is inspectable
        if I > limit:
            inverted = flip is not None and I > flip
            self._report(
                f"range:{self.activity_model}:{I:.6g}",
                f"ionic_strength={I:g} M is outside the documented validity "
                f"range of the {self.activity_model!r} activity model "
                f"(I < {limit:g} M). "
                + (
                    f"It is also above I = {flip:.6g} M, where the "
                    f"{self.activity_model!r} bracket changes sign, so the raw "
                    "correction would MULTIPLY D rather than reduce it. "
                    if inverted else ""
                )
                + (
                    "The model input is clamped at "
                    f"{limit:g} M so the correction saturates instead of "
                    "inverting; it is not an extrapolation and carries no "
                    "information about this regime. "
                    if not self.extrapolate_activity_model
                    else "extrapolate_activity_model=True, so the raw "
                    "extrapolated coefficients are used as requested. "
                )
                + "For a concentrated liquor (2-4 M chloride, say) the "
                "defensible treatment is ionic_strength=None, i.e. a "
                "conditional constant fitted at that ionic strength. Pass "
                "on_out_of_range='raise' to make this an error or 'ignore' to "
                "silence it (#194).",
            )

    def _model_ionic_strength(self, ionic_strength) -> Array:
        """Ionic strength actually handed to the activity model (#194).

        Clamped to the model's documented ``max_ionic_strength`` unless
        :attr:`extrapolate_activity_model` is set. The clamp is what makes the
        sign inversion described in the class docstring unreachable: it is
        arithmetic, so it holds for concrete values, concrete arrays and
        abstract tracers alike, where no Python-level check can.

        Args:
            ionic_strength: Ionic strength (M), scalar, array or tracer.

        Returns:
            The (possibly clamped) ionic strength as a JAX array.
        """
        I = jnp.asarray(ionic_strength)
        if self.extrapolate_activity_model:
            return I
        limit = ACTIVITY_MODELS[self.activity_model]["max_ionic_strength"]
        if not np.isfinite(limit):
            return I
        return jnp.minimum(I, limit)

    def _activity_correction(self, ionic_strength) -> Array | float:
        """Residual activity factor applied to the correlated ``D`` (#194).

        With ``pH`` on the concentration scale the ``b*pH`` term already carries
        the ``[H+]**-p`` dependence, so what remains of

            D = K [HA]**p gamma_RE / (gamma_H**p [H+]**p)

        is ``gamma_RE / gamma_H**p``, with ``p`` the number of protons released
        per REE taken from the extractant record. For a solvating extractant
        ``p = 0`` and there is no proton term at all.

        The ionic strength is passed through :meth:`_model_ionic_strength`
        first, so beyond the model's documented range the factor saturates
        rather than inverting (see the class docstring; Davies inverts above
        I = 1.940363884733242 M).

        Args:
            ionic_strength: Ionic strength (M), or None for no correction.

        Returns:
            Multiplicative correction factor (1.0 when no correction applies).
        """
        if ionic_strength is None or self.activity_model == "none":
            return 1.0

        p = self._ext_data.stoichiometry_protons  # static Python int
        I = self._model_ionic_strength(ionic_strength)
        gamma_RE = activity_coefficient(3, I, self.activity_model)
        if p == 0:
            # Solvating / neutral extractant: no protons are exchanged.
            return gamma_RE
        gamma_H = activity_coefficient(1, I, self.activity_model)
        return gamma_RE / jnp.power(gamma_H, p)

    # -------------------------------------------------------------------
    # Distribution coefficients
    # -------------------------------------------------------------------

    def get_D(
        self,
        element: str,
        pH: Array | float | None = None,
        T: Array | float = 298.15,
        ionic_strength: Array | float | None = None,
        nitrate_conc: Array | float | None = None,
    ) -> Array:
        """Calculate distribution coefficient for a single element.

        The correlation used depends on the active mechanism (#195):

        - ``cation_exchange``::

              log10(D) = a + b*pH + c*pH^2 + d*(1/T - 1/Tref)
                         + n*log10([HA]/[HA]_ref)

          with ``pH = -log10([H+])`` on the **concentration** scale, matching
          the header of ``data/extractants.yaml`` (#194).

        - ``solvating``::

              s = log10([NO3-] / reference_nitrate)
              log10(D) = a + b*s + c*s^2 + d*(1/T - 1/Tref)
                         + n*log10([S]/[S]_ref)

          Here ``a`` is ``log10(D)`` **at** ``reference_nitrate`` (3 M for TBP)
          and ``b`` is the nitrate slope ``d log10(D) / d log10([NO3-])``, so
          the coefficients mean exactly what the YAML says they mean. ``pH`` is
          not a driving variable and is ignored.

        Args:
            element: REE symbol (e.g., "Nd").
            pH: Solution pH on the concentration scale, ``-log10([H+])``.
                Required for cation exchange; ignored (and optional) for
                solvating extractants.
            T: Temperature (K).
            ionic_strength: Aqueous ionic strength (M), scalar or array.
                ``None`` (default and, for a concentrated liquor, the
                recommended choice) applies no activity correction: the
                correlation is used as the conditional constant it is, at the
                ionic strength it was fitted at. When a value is given, ``D``
                is multiplied by ``gamma_RE3+ / gamma_H+**p`` from
                :attr:`activity_model`, with
                ``p = Extractant.stoichiometry_protons``. A value outside the
                model's documented validity range is reported according to
                :attr:`on_out_of_range` and, unless
                :attr:`extrapolate_activity_model` is set, the ionic strength
                fed to the model is clamped at the range limit so the
                correction saturates instead of changing sign (Davies inverts
                above I = 1.940363884733242 M; see the class docstring) (#194).
            nitrate_conc: Nitrate concentration (M) overriding
                :attr:`nitrate_conc` for this call. Useful for differentiating
                with respect to nitrate concentration.

        Returns:
            Distribution coefficient D.

        Raises:
            ValueError: If ``pH`` is omitted for a cation-exchange extractant,
                or a solvating extractant is used without a nitrate
                concentration.
            KeyError: If the element is absent from the mechanism's coefficient
                block.
        """
        T = jnp.asarray(T)
        coeffs = self._coefficients(element)

        if self.mechanism == "solvating":
            c_nitrate = (
                self.nitrate_conc if nitrate_conc is None else nitrate_conc
            )
            self._require_nitrate_medium(c_nitrate)
            # Driving variable is the nitrate log-ratio about the reference
            # concentration the coefficients were tabulated at (#195).
            s = jnp.log10(
                jnp.asarray(c_nitrate) / self._ext_data.reference_nitrate
            )
            log_D = coeffs.a + coeffs.b * s + coeffs.c * s**2
        else:
            if pH is None:
                raise ValueError(
                    f"Extractant {self.extractant!r} extracts by cation "
                    "exchange, so get_D requires pH (concentration scale, "
                    "-log10([H+]))."
                )
            pH = jnp.asarray(pH)
            log_D = coeffs.a + coeffs.b * pH + coeffs.c * pH**2

        # Temperature correction
        T_ref = 298.15
        d_T = self._ext_data.temperature_coefficients[element]
        log_D = log_D + d_T * (1/T - 1/T_ref)

        # Extractant-concentration correction: D ∝ [HA]^n.
        #
        # (#190) This is the ONE place the extractant-concentration dependence
        # is applied to D. The loading correction in equilibrium/loading.py
        # (`apparent_D`, the (1 - theta)^n free-extractant depletion factor) is
        # deliberately NOT composed with this term in the stage path: doing
        # both double-counts the same free-extractant effect. See #190, and
        # #196 for the mass-action closure that would replace both.
        n = self._ext_data.concentration_exponent
        C_ref = self._ext_data.reference_concentration
        log_D = log_D + n * jnp.log10(self.concentration / C_ref)

        D = jnp.power(10.0, log_D)

        # Ionic-strength (activity) correction (#111, convention fixed in #194).
        # The range check is Python-level and deduplicated; the correction
        # itself is traced and differentiable.
        self._check_activity_range(ionic_strength)
        return D * self._activity_correction(ionic_strength)

    def get_D_all(
        self,
        pH: Array | float | None = None,
        T: Array | float = 298.15,
        ionic_strength: Array | float | None = None,
        nitrate_conc: Array | float | None = None,
    ) -> dict[str, Array]:
        """Calculate distribution coefficients for all elements.

        Args:
            pH: Solution pH (concentration scale); see :meth:`get_D`.
            T: Temperature (K)
            ionic_strength: Aqueous ionic strength (M); see :meth:`get_D`.
            nitrate_conc: Nitrate concentration (M); see :meth:`get_D`.

        Returns:
            Dictionary mapping element symbols to D values
        """
        return {
            elem: self.get_D(elem, pH, T, ionic_strength, nitrate_conc)
            for elem in self.elements
        }

    def get_D_array(
        self,
        pH: Array | float | None = None,
        T: Array | float = 298.15,
    ) -> Array:
        """Calculate D values as JAX array (in element order).

        Args:
            pH: Solution pH (concentration scale); see :meth:`get_D`.
            T: Temperature (K)

        Returns:
            JAX array of D values
        """
        D_list = [self.get_D(elem, pH, T) for elem in self.elements]
        return jnp.stack(D_list)

    def get_separation_factor(
        self,
        element1: str,
        element2: str,
        pH: Array | float | None = None,
        T: Array | float = 298.15,
    ) -> Array:
        """Calculate separation factor between two elements.

        SF = D1 / D2

        Args:
            element1: First element symbol
            element2: Second element symbol
            pH: Solution pH
            T: Temperature (K)

        Returns:
            Separation factor
        """
        D1 = self.get_D(element1, pH, T)
        D2 = self.get_D(element2, pH, T)
        return D1 / D2

    def optimal_pH_for_separation(
        self,
        element1: str,
        element2: str,
        pH_range: tuple[float, float] = (1.0, 5.0),
        n_points: int = 100,
        T: float = 298.15,
    ) -> tuple[float, float]:
        """Find pH that maximizes separation factor.

        Args:
            element1: Target element (to extract)
            element2: Impurity element (to reject)
            pH_range: pH range to search
            n_points: Number of evaluation points
            T: Temperature (K)

        Returns:
            Tuple of (optimal_pH, max_SF)
        """
        pH_values = jnp.linspace(pH_range[0], pH_range[1], n_points)
        SF_values = jnp.array([
            float(self.get_separation_factor(element1, element2, pH, T))
            for pH in pH_values
        ])
        max_idx = jnp.argmax(SF_values)
        return float(pH_values[max_idx]), float(SF_values[max_idx])


# =============================================================================
# Convenience Functions
# =============================================================================

def get_distribution_coefficient(
    element: str,
    extractant: str,
    pH: Array | float | None = None,
    T: Array | float = 298.15,
    concentration: float = 0.5,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
    medium: str | None = None,
) -> Array:
    """Calculate distribution coefficient for a single element.

    Args:
        element: REE symbol (e.g., "Nd")
        extractant: Extractant name (e.g., "D2EHPA")
        pH: Solution pH (concentration scale); required for cation exchange,
            ignored for solvating extractants
        T: Temperature (K)
        concentration: Extractant concentration (M)
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP (#195)
        mechanism: Explicit mechanism override; see :class:`REEDistribution`
        medium: Declared aqueous medium; see :class:`REEDistribution` (#195)

    Returns:
        Distribution coefficient D
    """
    dist = REEDistribution(
        extractant=extractant,
        elements=(element,),
        concentration=concentration,
        nitrate_conc=nitrate_conc,
        mechanism=mechanism,
        medium=medium,
    )
    return dist.get_D(element, pH, T)


def get_distribution_coefficients(
    elements: list[str],
    extractant: str,
    pH: Array | float | None = None,
    T: Array | float = 298.15,
    concentration: float = 0.5,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
    medium: str | None = None,
) -> dict[str, Array]:
    """Calculate distribution coefficients for multiple elements.

    Args:
        elements: List of REE symbols
        extractant: Extractant name
        pH: Solution pH (concentration scale); see
            :func:`get_distribution_coefficient`
        T: Temperature (K)
        concentration: Extractant concentration (M)
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP (#195)
        mechanism: Explicit mechanism override; see :class:`REEDistribution`
        medium: Declared aqueous medium; see :class:`REEDistribution` (#195)

    Returns:
        Dictionary mapping element symbols to D values
    """
    dist = REEDistribution(
        extractant=extractant,
        elements=tuple(elements),
        concentration=concentration,
        nitrate_conc=nitrate_conc,
        mechanism=mechanism,
        medium=medium,
    )
    return dist.get_D_all(pH, T)


def get_separation_factor(
    element1: str,
    element2: str,
    extractant: str,
    pH: Array | float | None = None,
    T: Array | float = 298.15,
    concentration: float = 0.5,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
    medium: str | None = None,
) -> Array:
    """Calculate separation factor between two elements.

    Args:
        element1: First element symbol
        element2: Second element symbol
        extractant: Extractant name
        pH: Solution pH (concentration scale); see
            :func:`get_distribution_coefficient`
        T: Temperature (K)
        concentration: Extractant concentration (M)
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP (#195)
        mechanism: Explicit mechanism override; see :class:`REEDistribution`
        medium: Declared aqueous medium; see :class:`REEDistribution` (#195)

    Returns:
        Separation factor D1/D2
    """
    D1 = get_distribution_coefficient(
        element1, extractant, pH, T, concentration, nitrate_conc, mechanism,
        medium,
    )
    D2 = get_distribution_coefficient(
        element2, extractant, pH, T, concentration, nitrate_conc, mechanism,
        medium,
    )
    return D1 / D2


# =============================================================================
# McCabe-Thiele Analysis
# =============================================================================

def equilibrium_line(
    D: Array | float,
    x: Array,
) -> Array:
    """Calculate equilibrium line y = D*x for McCabe-Thiele.

    Args:
        D: Distribution coefficient
        x: Aqueous phase concentration

    Returns:
        Organic phase concentration at equilibrium
    """
    return D * x


def operating_line_extraction(
    x: Array,
    x_in: float,
    y_in: float,
    S_F: float,
) -> Array:
    """Calculate extraction operating line for McCabe-Thiele.

    Material balance: F*x_in + S*y_in = F*x + S*y
    Rearranged: y = (F/S)*(x_in - x) + y_in

    Args:
        x: Aqueous phase concentration
        x_in: Inlet aqueous concentration
        y_in: Inlet organic concentration (usually 0 for fresh solvent)
        S_F: Solvent-to-feed ratio (S/F)

    Returns:
        Organic phase concentration
    """
    return (1/S_F) * (x_in - x) + y_in


def operating_line_stripping(
    y: Array,
    y_in: float,
    x_in: float,
    A_S: float,
) -> Array:
    """Calculate stripping operating line for McCabe-Thiele.

    Material balance: S*y_in + A*x_in = S*y + A*x
    Rearranged: x = (S/A)*(y_in - y) + x_in

    Args:
        y: Organic phase concentration
        y_in: Inlet organic concentration (loaded)
        x_in: Inlet aqueous concentration (strip solution)
        A_S: Aqueous-to-solvent ratio (A/S)

    Returns:
        Aqueous phase concentration
    """
    return (1/A_S) * (y_in - y) + x_in


def minimum_solvent_ratio(
    D: Array | float,
    x_in: float,
    x_out: float,
    y_in: float = 0.0,
) -> Array:
    """Calculate minimum solvent-to-feed ratio.

    At minimum S/F, operating line touches equilibrium line.

    Args:
        D: Distribution coefficient
        x_in: Inlet aqueous concentration
        x_out: Desired outlet aqueous concentration
        y_in: Inlet organic concentration

    Returns:
        Minimum S/F ratio
    """
    D = jnp.asarray(D)
    # At equilibrium: y* = D * x_in (maximum loading)
    y_max = D * x_in
    # Material balance: F*(x_in - x_out) = S*(y_max - y_in)
    # S/F = (x_in - x_out) / (y_max - y_in)
    return safe_divide(x_in - x_out, y_max - y_in)


def stages_kremser(
    D: Array | float,
    S_F: Array | float,
    recovery: float = 0.99,
) -> Array:
    """Calculate number of stages using Kremser equation.

    For counter-current extraction.

    Args:
        D: Distribution coefficient
        S_F: Solvent-to-feed ratio
        recovery: Desired recovery fraction

    Returns:
        Number of theoretical stages
    """
    D = jnp.asarray(D)
    S_F = jnp.asarray(S_F)

    # Extraction factor E = D * S/F
    E = D * S_F

    # Kremser equation
    # Recovery = (E^(N+1) - E) / (E^(N+1) - 1)
    # Solving for N:
    # N = log((recovery*(E-1) + 1) / E) / log(E) - 1

    # Handle E ≈ 1 case
    N = jnp.where(
        jnp.abs(E - 1.0) < 1e-6,
        recovery / (1 - recovery),  # Limit as E → 1
        jnp.log((recovery * (E - 1) + 1) / E) / jnp.log(E)
    )

    return jnp.maximum(N, 1.0)


def stages_fenske(
    alpha: Array | float,
    split_extract: Array | float,
    split_raffinate: Array | float | None = None,
) -> Array:
    """Fenske minimum stages for a binary split (#202).

    The companion to :func:`stages_kremser`, and a *lower* bound where
    Kremser is an operating estimate. Fenske's expression is the stage
    count at total reflux -- in extraction, the limit of infinite
    solvent-to-feed ratio -- so no finite counter-current cascade can
    achieve the same split in fewer equilibrium stages::

        N_min = ln[ (s_E / (1 - s_E)) * (s_R / (1 - s_R)) ] / ln(alpha)

    where ``s_E`` is the fraction of the extract key that reports to the
    extract and ``s_R`` the fraction of the raffinate key that reports to
    the raffinate. Two readings of the same algebra, both used here:

    * **Split fractions.** ``s_E = r_A`` (recovery of the extracted key to
      the organic) and ``s_R = 1 - r_B`` (the raffinate key's recovery to
      the aqueous), giving the classical
      ``ln[(d/b)_A (b/d)_B] / ln(alpha)`` key-ratio form.
    * **Purity pair.** For an equimolar binary feed the two product
      purities equal those split fractions, so a target such as
      "99% pure product, 99% pure raffinate" goes straight in.

    That is what makes it a screening filter (#202): it costs two
    logarithms, needs only a separation factor, and rejects a candidate
    topology whose installed stage count is below ``N_min`` before any
    rigorous cascade is solved -- let alone costed.

    Unlike :func:`stages_kremser`, the result is floored at 0 rather than
    1. A separation so easy that it needs less than one theoretical stage
    is information a screening filter should keep: flooring at 1 would
    turn a genuine lower bound into a bound that is sometimes wrong in the
    direction that matters (too high), and would reject admissible
    topologies.

    Args:
        alpha: Separation factor between the two keys, ``D_A / D_B``.
            Values below 1 are inverted (the two keys swap roles), so the
            caller need not order the pair; ``alpha == 1`` means no
            separation is possible and returns ``inf``.
        split_extract: Fraction of the extract key reporting to the
            extract, in (0, 1).
        split_raffinate: Fraction of the raffinate key reporting to the
            raffinate, in (0, 1). Defaults to ``split_extract``, the
            symmetric split. Splits are clipped into the open interval, so
            a perfect split of exactly 1 returns a large finite bound
            rather than a nan.

    Returns:
        Minimum number of theoretical stages, as a JAX scalar. ``inf``
        when ``alpha`` is 1, where no stage count separates the pair.

    Example:
        >>> import jax.numpy as jnp
        >>> float(stages_fenske(100.0, 0.99, 0.99))  # doctest: +ELLIPSIS
        1.99563...
        >>> float(stages_fenske(1000.0, 0.9, 0.9))   # doctest: +ELLIPSIS
        0.63616...

    References:
        Fenske, M.R. Ind. Eng. Chem. 24, 482 (1932).
        Seader, J.D., Henley, E.J., Roper, D.K. Separation Process
        Principles, 3rd ed., Wiley, 2011, Ch. 9.
    """
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    s_E = jnp.asarray(split_extract, dtype=jnp.float64)
    s_R = (
        s_E if split_raffinate is None
        else jnp.asarray(split_raffinate, dtype=jnp.float64)
    )

    # The pair is unordered: alpha < 1 just means the keys were named the
    # other way round, and |ln alpha| is the same separation power.
    log_alpha = jnp.abs(jnp.log(alpha))

    # Odds ratio, on splits clipped into the open unit interval so that a
    # caller passing a perfect 1.0 (or a solver stepping onto it) gets a
    # large finite bound instead of a nan that would poison a screen.
    _tiny = 1e-15

    def _odds(s):
        s = jnp.clip(s, _tiny, 1.0 - _tiny)
        return s / (1.0 - s)

    numerator = jnp.log(_odds(s_E) * _odds(s_R))
    N_min = safe_divide(numerator, log_alpha)
    # alpha == 1: no separation at any stage count.
    N_min = jnp.where(log_alpha > 0.0, N_min, jnp.inf)
    return jnp.maximum(N_min, 0.0)
