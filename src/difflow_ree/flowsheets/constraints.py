"""Operating boundaries as inequality constraints (#202).

Third-phase onset, extractant loading and hydraulic capacity are real
walls, and an optimizer will walk up to any wall it is only *told* about
and then step over it, because in the model crossing is profitable.
:mod:`difflow_ree.units.extraction` already reports the third-phase
condition as a signed margin (``info["third_phase_margin"]``, #193) and
the loading condition as ``info["theta_total"]``; what was missing is a
constraint object an optimizer can consume.

The convention is the usual one: every margin is **feasible when >= 0**,
so :meth:`ConstraintSet.vector` returns ``g(x) >= 0`` and can be handed
straight to a solver, or to ``difflow.solvers.as_nlp`` (#203) on the way
to an external discrete layer.

Each margin is a traced JAX scalar, so ``jax.grad`` of a constraint with
respect to a design variable works: the point of the exercise is that the
optimizer can *see* the boundary, not merely be warned after crossing it.

Example:
    >>> import jax.numpy as jnp
    >>> c = third_phase_constraint(jnp.asarray(0.72), 0.65, module="ess")
    >>> bool(c.violated)
    True
    >>> round(float(c.margin), 6)
    -0.07
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

#: Kinds of operating boundary this module knows how to build. ``custom``
#: is the escape hatch for a caller's own inequality.
CONSTRAINT_KINDS = (
    "third_phase",
    "loading",
    "hydraulic",
    "phase_ratio",
    "custom",
)


@dataclass(repr=False)
class OperatingConstraint(ParamsMixin):
    """One inequality an optimizer must respect.

    Attributes:
        name: Short identifier, unique within a :class:`ConstraintSet`.
        kind: One of :data:`CONSTRAINT_KINDS`.
        margin: Signed slack, feasible when ``>= 0``. A JAX scalar, so it
            is differentiable and survives ``jit``.
        value: The operating quantity being bounded.
        limit: The bound itself.
        scale: Positive normalisation for :attr:`normalized_margin`, so a
            set mixing dimensionless loadings with molar throughputs can
            be compared and given to a solver without conditioning it on
            the choice of units.
        module: Name of the module the constraint came from.
        description: Free text for reports.
    """

    name: str
    kind: str
    margin: Array
    value: Array
    limit: Array
    scale: float = 1.0
    module: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        """Coerce to JAX scalars and validate the kind.

        Raises:
            ValueError: On an unknown kind or a non-positive scale.
        """
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(
                f"constraint {self.name!r}: kind must be one of "
                f"{list(CONSTRAINT_KINDS)}, got {self.kind!r}."
            )
        if not self.scale > 0:
            raise ValueError(
                f"constraint {self.name!r}: scale must be > 0, got "
                f"{self.scale}."
            )
        self.margin = jnp.asarray(self.margin, dtype=jnp.float64)
        self.value = jnp.asarray(self.value, dtype=jnp.float64)
        self.limit = jnp.asarray(self.limit, dtype=jnp.float64)

    @property
    def qualified_name(self) -> str:
        """``"<module>.<name>"`` when the module is known, else ``name``."""
        return f"{self.module}.{self.name}" if self.module else self.name

    @property
    def normalized_margin(self) -> Array:
        """:attr:`margin` divided by :attr:`scale`."""
        return self.margin / self.scale

    @property
    def violated(self) -> bool:
        """Whether the constraint is violated, as a Python bool.

        Only meaningful outside ``jit``; a traced margin cannot be
        compared concretely. Use :attr:`margin` under tracing.
        """
        return bool(self.margin < 0.0)

    def to_dict(self) -> dict:
        """A plain JSON-ready dictionary.

        Returns:
            Dict of the fields, with arrays rendered as floats.
        """
        return {
            "name": self.name,
            "module": self.module,
            "kind": self.kind,
            "margin": float(self.margin),
            "value": float(self.value),
            "limit": float(self.limit),
            "scale": float(self.scale),
            "violated": self.violated,
            "description": self.description,
        }


@dataclass(repr=False)
class ConstraintSet(ParamsMixin):
    """The operating boundaries of a module or a whole train.

    Attributes:
        constraints: The individual inequalities.
    """

    constraints: tuple[OperatingConstraint, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Freeze the tuple and reject duplicate qualified names."""
        self.constraints = tuple(self.constraints)
        names = [c.qualified_name for c in self.constraints]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate constraint names: {dupes}")

    def __len__(self) -> int:
        return len(self.constraints)

    def __iter__(self):
        return iter(self.constraints)

    def __getitem__(self, key):
        if isinstance(key, str):
            for c in self.constraints:
                if key in (c.name, c.qualified_name):
                    return c
            raise KeyError(
                f"no constraint {key!r}; have {[c.qualified_name for c in self]}"
            )
        return self.constraints[key]

    def __add__(self, other: "ConstraintSet") -> "ConstraintSet":
        return ConstraintSet(self.constraints + tuple(other))

    @property
    def names(self) -> tuple[str, ...]:
        """Qualified names, in order."""
        return tuple(c.qualified_name for c in self.constraints)

    def vector(self) -> Array:
        """All margins as one array, ``g(x) >= 0`` when feasible.

        Returns:
            Array of shape ``(len(self),)``. Empty sets return an empty
            array rather than raising, so a caller can concatenate
            unconditionally.
        """
        if not self.constraints:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.stack([c.margin for c in self.constraints])

    def normalized_vector(self) -> Array:
        """:meth:`vector`, each entry divided by its own scale."""
        if not self.constraints:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.stack([c.normalized_margin for c in self.constraints])

    def margins(self) -> dict[str, Array]:
        """Margins keyed by qualified name."""
        return {c.qualified_name: c.margin for c in self.constraints}

    def violations(self) -> tuple[OperatingConstraint, ...]:
        """The violated constraints, worst first."""
        bad = [c for c in self.constraints if c.violated]
        return tuple(sorted(bad, key=lambda c: float(c.normalized_margin)))

    @property
    def feasible(self) -> bool:
        """Whether every constraint holds."""
        return not self.violations()

    def worst(self) -> OperatingConstraint | None:
        """The constraint with the smallest normalized margin, or None."""
        if not self.constraints:
            return None
        return min(self.constraints, key=lambda c: float(c.normalized_margin))

    def to_dict(self) -> dict:
        """A plain JSON-ready dictionary.

        Returns:
            Dict with ``constraints`` and a ``feasible`` flag.
        """
        return {
            "feasible": self.feasible,
            "constraints": [c.to_dict() for c in self.constraints],
        }

    def summary(self) -> str:
        """A one-line-per-constraint text table.

        Returns:
            Human-readable report, marking violations with ``VIOLATED``.
        """
        if not self.constraints:
            return "no operating constraints declared"
        width = max(len(c.qualified_name) for c in self.constraints)
        lines = []
        for c in self.constraints:
            flag = "VIOLATED" if c.violated else "ok"
            lines.append(
                f"{c.qualified_name:<{width}}  {c.kind:<12} "
                f"value={float(c.value):+.6g}  limit={float(c.limit):+.6g}  "
                f"margin={float(c.margin):+.6g}  {flag}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------


def third_phase_constraint(
    loading: Array | float,
    limit: Array | float,
    *,
    module: str = "",
    name: str = "third_phase",
) -> OperatingConstraint:
    """Third-phase onset as ``limit - theta >= 0`` (#193, #202).

    Above a critical organic loading the extract splits into a second,
    heavy organic layer, the settler floods and the model that predicted
    the profitable operating point stops describing the plant. #193 made
    this a smooth signed margin on the mixer-settler; this lifts it to a
    constraint an optimizer can be given.

    Args:
        loading: Dimensionless organic loading ``theta`` (1.0 = the
            extractant is saturated); ``info["theta_total"]`` from
            :class:`~difflow_ree.units.extraction.REEExtractor`.
        limit: Loading at which the third phase appears.
        module: Owning module name.
        name: Constraint name.

    Returns:
        The constraint, with unit scale because ``theta`` is already
        dimensionless.
    """
    loading = jnp.asarray(loading, dtype=jnp.float64)
    limit = jnp.asarray(limit, dtype=jnp.float64)
    return OperatingConstraint(
        name=name,
        kind="third_phase",
        margin=limit - loading,
        value=loading,
        limit=limit,
        scale=1.0,
        module=module,
        description="organic loading below third-phase onset",
    )


def loading_constraint(
    loading: Array | float,
    limit: Array | float = 1.0,
    *,
    module: str = "",
    name: str = "loading",
) -> OperatingConstraint:
    """Extractant saturation as ``limit - theta >= 0``.

    Distinct from :func:`third_phase_constraint`: saturation is where the
    extractant runs out of sites (``theta = 1``), third-phase onset is a
    solubility limit that usually sits *below* it. Both are reported so a
    design cannot satisfy one by ignoring the other.

    Args:
        loading: Dimensionless organic loading ``theta``.
        limit: Maximum admissible loading, 1.0 for full saturation.
        module: Owning module name.
        name: Constraint name.

    Returns:
        The constraint.
    """
    loading = jnp.asarray(loading, dtype=jnp.float64)
    limit = jnp.asarray(limit, dtype=jnp.float64)
    return OperatingConstraint(
        name=name,
        kind="loading",
        margin=limit - loading,
        value=loading,
        limit=limit,
        scale=1.0,
        module=module,
        description="extractant loading below saturation",
    )


def hydraulic_constraint(
    throughput: Array | float,
    capacity: Array | float,
    *,
    module: str = "",
    name: str = "hydraulic",
) -> OperatingConstraint:
    """Settler capacity as ``capacity - throughput >= 0``.

    Throughput and capacity are expressed in the *same units as the
    stream flows*, deliberately: everything else in this package is
    invariant to whether a cascade is written in mol/s or Tmol/s (#189),
    and pinning a capacity to litres per second here would smuggle a
    hidden unit back in. Convert to a volumetric duty outside if you have
    a density model.

    Args:
        throughput: Total two-phase throughput entering the contactor.
        capacity: Installed capacity, same units.
        module: Owning module name.
        name: Constraint name.

    Returns:
        The constraint, scaled by the capacity so the margin is
        comparable with the dimensionless ones.
    """
    throughput = jnp.asarray(throughput, dtype=jnp.float64)
    capacity = jnp.asarray(capacity, dtype=jnp.float64)
    scale = float(jnp.maximum(jnp.abs(capacity), 1e-300))
    return OperatingConstraint(
        name=name,
        kind="hydraulic",
        margin=capacity - throughput,
        value=throughput,
        limit=capacity,
        scale=scale,
        module=module,
        description="two-phase throughput below installed settler capacity",
    )


def phase_ratio_constraints(
    ratio: Array | float,
    minimum: float | None = None,
    maximum: float | None = None,
    *,
    module: str = "",
    prefix: str = "phase_ratio",
) -> tuple[OperatingConstraint, ...]:
    """Dispersion-band limits on the organic-to-aqueous ratio.

    Outside a band of roughly 0.2 to 5 a mixer-settler stops making a
    stable dispersion, which is a hydraulic limit rather than an
    equilibrium one and so is not visible anywhere in the Kremser model.

    Args:
        ratio: Operating O/A.
        minimum: Lower bound, or None to omit that side.
        maximum: Upper bound, or None to omit that side.
        module: Owning module name.
        prefix: Base for the two constraint names.

    Returns:
        Zero, one or two constraints.
    """
    ratio = jnp.asarray(ratio, dtype=jnp.float64)
    out: list[OperatingConstraint] = []
    if minimum is not None:
        out.append(OperatingConstraint(
            name=f"{prefix}_min",
            kind="phase_ratio",
            margin=ratio - minimum,
            value=ratio,
            limit=jnp.asarray(minimum),
            scale=max(abs(float(minimum)), 1e-12),
            module=module,
            description="O/A above the lower dispersion-band limit",
        ))
    if maximum is not None:
        out.append(OperatingConstraint(
            name=f"{prefix}_max",
            kind="phase_ratio",
            margin=maximum - ratio,
            value=ratio,
            limit=jnp.asarray(maximum),
            scale=max(abs(float(maximum)), 1e-12),
            module=module,
            description="O/A below the upper dispersion-band limit",
        ))
    return tuple(out)


@dataclass(repr=False)
class OperatingLimits(ParamsMixin):
    """The boundary values a module is designed against (#202).

    Every field is optional; ``None`` means "not declared", and the
    corresponding constraint is simply absent from the module's
    :class:`ConstraintSet` rather than defaulted to something invented.

    Attributes:
        third_phase_loading: Organic loading ``theta`` at which a third
            phase forms. Typical D2EHPA/kerosene values are 0.6-0.8 of
            saturation for the heavy lanthanides.
        max_loading: Maximum admissible ``theta``; 1.0 is saturation.
        hydraulic_capacity: Installed two-phase throughput, in the same
            units as the stream molar flows. See
            :func:`hydraulic_constraint`.
        min_phase_ratio: Lower O/A limit for a stable dispersion.
        max_phase_ratio: Upper O/A limit.
    """

    third_phase_loading: float | None = None
    max_loading: float | None = 1.0
    hydraulic_capacity: float | None = None
    min_phase_ratio: float | None = None
    max_phase_ratio: float | None = None

    def to_dict(self) -> dict:
        """A plain JSON-ready dictionary."""
        return {
            "third_phase_loading": self.third_phase_loading,
            "max_loading": self.max_loading,
            "hydraulic_capacity": self.hydraulic_capacity,
            "min_phase_ratio": self.min_phase_ratio,
            "max_phase_ratio": self.max_phase_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OperatingLimits":
        """Rebuild from :meth:`to_dict` output.

        Args:
            data: The dictionary to read.

        Returns:
            The limits.
        """
        return cls(**{k: data.get(k) for k in (
            "third_phase_loading", "max_loading", "hydraulic_capacity",
            "min_phase_ratio", "max_phase_ratio",
        )})
