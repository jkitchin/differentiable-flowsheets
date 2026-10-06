"""Cost of goods for a mAb process, step by step, from data you supply.

The older functions in :mod:`difflow_bio.economics.costs` priced every
chromatography step as Protein A, ran resin cycles off the batch count
alone, and charged a fixed labor bill whatever the process did (#358). Here
the cost follows the process:

* each chromatography step has its own resin, column volume, binding
  capacity, lifetime and buffer use, and its cycles per batch come from the
  mass it is loaded with, ``load / (DBC * CV)``;
* the product mass runs through the steps with each step's yield, so titer,
  capacity and yield all move resin, buffer and cost per gram;
* labor is fixed staff plus operator hours per batch and per step;
* media includes the fed-batch feed and the seed train; buffers, single-use
  consumables, QC per batch, utilities, waste and maintenance are all there;
* failed batches consume materials and labor and release nothing.

Every price, rate and specification is data: :class:`CostBasis` and
:class:`ProcessSpec` load from a dict, a YAML file or a JSON file, and the
shipped reference (``data/mab_reference.yaml``) says, line by line, where
each of its numbers came from. Most are placeholders; replace them with
your own.

Everything is ``jax.numpy``, so ``jax.grad`` of cost per gram with respect
to titer, a binding capacity or a step yield is exact.

Example:
    >>> import dataclasses, jax
    >>> from difflow_bio.economics import load_cost_model, cogs_breakdown
    >>> process, basis = load_cost_model()
    >>> cogs_breakdown(process, basis)["cost_per_g"]
    >>> jax.grad(lambda t: cogs_breakdown(
    ...     dataclasses.replace(process, titer_g_L=t), basis)["cost_per_g"])(5.0)
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from jax import Array

#: the shipped reference process and cost basis
REFERENCE = Path(__file__).parent / "data" / "mab_reference.yaml"

#: sharpness of the smooth floor that keeps cycles per batch at least one
CYCLE_FLOOR_SHARPNESS = 20.0


# =============================================================================
# Process steps
# =============================================================================


@dataclass(frozen=True)
class ChromatographyStep:
    """A chromatography step with its own resin and column.

    Attributes:
        name: Step name, e.g. ``"protein_a_capture"``.
        resin: Resin name (for the record; prices come from the fields).
        column_volume_L: Packed column volume (L).
        dynamic_binding_capacity_g_L: Product load per cycle per litre of
            resin (g/L). For a bind-and-elute step the dynamic binding
            capacity; for a flow-through step the load the step is run at.
        step_yield: Fraction of the product entering the step that leaves it.
        resin_cost_usd_L: Resin price (USD/L).
        resin_lifetime_cycles: Cycles before the resin is replaced.
        buffer_cv: Buffer name to column volumes used per cycle.
        single_use_usd_per_batch: Bags, assemblies and other single-use
            items per batch (USD).
    """

    name: str
    resin: str
    column_volume_L: float
    dynamic_binding_capacity_g_L: float
    step_yield: float
    resin_cost_usd_L: float
    resin_lifetime_cycles: float
    buffer_cv: Mapping[str, float] = field(default_factory=dict)
    single_use_usd_per_batch: float = 0.0

    @classmethod
    def from_resin(cls, name: str, resin: str, column_volume_L: float,
                   step_yield: float, dbc_fraction: float = 0.8,
                   **kwargs) -> ChromatographyStep:
        """A step whose capacity, price and lifetime come from the resin
        database, the capacity derated to ``dbc_fraction`` of ``q_max``.

        Example:
            >>> ChromatographyStep.from_resin("capture", "MabSelect_SuRe",
            ...                               100.0, 0.95)
        """
        from difflow_bio.database import get_resin

        r = get_resin(resin)
        return cls(name=name, resin=resin, column_volume_L=column_volume_L,
                   dynamic_binding_capacity_g_L=r.q_max * dbc_fraction,
                   step_yield=step_yield, resin_cost_usd_L=r.cost_usd_L,
                   resin_lifetime_cycles=float(r.cycles), **kwargs)


@dataclass(frozen=True)
class FiltrationStep:
    """A membrane step (UF/DF, virus filtration).

    Attributes:
        name: Step name.
        membrane: Membrane name (for the record).
        area_m2: Membrane area (m2).
        step_yield: Fraction of the product recovered.
        membrane_cost_usd_m2: Membrane price (USD/m2).
        lifetime_batches: Batches before the membrane is replaced (1 for
            single-use filters).
        buffer_L_per_m2: Buffer name to litres per m2 per batch
            (diafiltration volumes).
        single_use_usd_per_batch: Single-use items per batch (USD).
    """

    name: str
    membrane: str
    area_m2: float
    step_yield: float
    membrane_cost_usd_m2: float
    lifetime_batches: float
    buffer_L_per_m2: Mapping[str, float] = field(default_factory=dict)
    single_use_usd_per_batch: float = 0.0


@dataclass(frozen=True)
class YieldStep:
    """A step with a yield and consumables but no resin or membrane
    (harvest clarification, viral inactivation).

    Attributes:
        name: Step name.
        step_yield: Fraction of the product recovered.
        buffer_L_per_batch: Buffer name to litres per batch.
        single_use_usd_per_batch: Single-use items per batch (USD).
    """

    name: str
    step_yield: float
    buffer_L_per_batch: Mapping[str, float] = field(default_factory=dict)
    single_use_usd_per_batch: float = 0.0


_STEP_TYPES = {"chromatography": ChromatographyStep,
               "filtration": FiltrationStep, "yield": YieldStep}


def _step_from_dict(data: Mapping[str, Any]):
    data = dict(data)
    kind = data.pop("type", None)
    if kind not in _STEP_TYPES:
        raise ValueError(f"step {data.get('name')!r} needs a type, one of "
                         f"{', '.join(_STEP_TYPES)} (got {kind!r})")
    return _build(_STEP_TYPES[kind], data)


def _build(cls, data: Mapping[str, Any]):
    """``cls(**data)``, naming unknown and missing fields."""
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{cls.__name__} has no field {', '.join(unknown)} "
                         f"(fields: {', '.join(sorted(names))})")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"{cls.__name__}: {exc}") from exc


# =============================================================================
# Process and cost basis
# =============================================================================


@dataclass(frozen=True)
class ProcessSpec:
    """What the plant does: the bioreactor, the batches and the DSP train.

    Attributes:
        working_volume_L: Production bioreactor working volume (L).
        titer_g_L: Harvest titer (g/L).
        batches_per_year: Batches started per year.
        batch_success_rate: Fraction of batches started that are released.
        steps: Downstream steps in order (chromatography, filtration and
            yield steps).
        basal_media_L_per_L: Basal medium per litre of working volume.
        feed_L_per_L: Fed-batch feed per litre of working volume.
        seed_media_L_per_L: Seed-train medium per litre of working volume.
        bioreactor_single_use_usd_per_batch: Bioreactor bag and single-use
            items per batch (USD).
    """

    working_volume_L: float
    titer_g_L: float
    batches_per_year: float
    batch_success_rate: float = 1.0
    steps: tuple = ()
    basal_media_L_per_L: float = 1.0
    feed_L_per_L: float = 0.0
    seed_media_L_per_L: float = 0.0
    bioreactor_single_use_usd_per_batch: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProcessSpec:
        """A process from plain data; ``steps`` is a list of dicts, each with
        a ``type`` of ``chromatography``, ``filtration`` or ``yield``."""
        data = dict(data)
        steps = tuple(_step_from_dict(s) for s in data.pop("steps", ()))
        return _build(cls, {**data, "steps": steps})

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["steps"] = [{"type": _type_name(s), **dataclasses.asdict(s)}
                        for s in self.steps]
        return out


def _type_name(step) -> str:
    return next(k for k, v in _STEP_TYPES.items() if isinstance(step, v))


@dataclass(frozen=True)
class CostBasis:
    """Prices and rates: what things cost where the plant is.

    Attributes:
        media_usd_L: USD/L for each of ``basal``, ``feed`` and ``seed``
            (all three are required; a process without a seed train or a
            feed sets its volume to zero).
        buffer_usd_L: Buffer name to USD/L; every buffer a step uses must
            have a price here.
        labor_rate_usd_h: Loaded labor rate (USD/h).
        hours_per_fte_year: Paid hours per full-time employee per year.
        support_fte: Fixed staff (QA, QC, supervision, engineering) whatever
            the batch count.
        operator_hours_per_bioreactor_batch: Operator hours per batch for
            the upstream train.
        operator_hours_per_step_batch: Operator hours per batch per
            downstream step.
        qc_usd_per_batch: Testing and release cost per batch started.
        utilities_usd_per_L_batch: Utilities per litre of working volume
            per batch.
        waste_usd_L: Liquid waste disposal per litre of media and buffer.
        maintenance_fraction: Annual maintenance as a fraction of CAPEX.
    """

    media_usd_L: Mapping[str, float]
    buffer_usd_L: Mapping[str, float]
    labor_rate_usd_h: float
    hours_per_fte_year: float = 2000.0
    support_fte: float = 0.0
    operator_hours_per_bioreactor_batch: float = 0.0
    operator_hours_per_step_batch: float = 0.0
    qc_usd_per_batch: float = 0.0
    utilities_usd_per_L_batch: float = 0.0
    waste_usd_L: float = 0.0
    maintenance_fraction: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostBasis:
        """A cost basis from plain data, with unknown keys named."""
        return _build(cls, dict(data))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_cost_model(path: str | Path | None = None) -> tuple[ProcessSpec, CostBasis]:
    """The process and cost basis in a YAML or JSON file.

    The file has two sections, ``process`` and ``basis``, laid out like the
    shipped reference, which is what is loaded when ``path`` is omitted.

    Args:
        path: A ``.yaml``/``.yml`` or ``.json`` file.

    Returns:
        ``(process, basis)``.
    """
    path = Path(path) if path is not None else REFERENCE
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    missing = {"process", "basis"} - set(data or {})
    if missing:
        raise ValueError(f"{path} has no {' or '.join(sorted(missing))} section")
    return ProcessSpec.from_dict(data["process"]), CostBasis.from_dict(data["basis"])


# =============================================================================
# The calculation
# =============================================================================


def cycles_per_batch(load_g: Array, capacity_g_L: Array, column_volume_L: Array,
                     sharpness: float = CYCLE_FLOOR_SHARPNESS) -> Array:
    """Cycles a column runs per batch: ``load / (capacity * CV)``, at least one.

    Continuous rather than rounded up, so cost has a derivative with respect
    to the load, the capacity and the column volume; the floor at one cycle
    is a softplus, ``1 + softplus(k (n - 1)) / k``, which is within
    ``ln 2 / k`` (0.035 at the default ``k``) of a hard floor at ``n = 1``
    and exact to round-off two cycles above it. A plant runs whole cycles;
    the continuous count is the expected value over a campaign whose load
    varies, and the right one to differentiate.
    """
    n = load_g / jnp.maximum(capacity_g_L * column_volume_L, 1e-30)
    return 1.0 + jax.nn.softplus(sharpness * (n - 1.0)) / sharpness


def _price(table: Mapping[str, float], name: str, what: str) -> float:
    if name not in table:
        raise KeyError(f"no price for {what} {name!r} in the cost basis "
                       f"(priced: {', '.join(sorted(table)) or 'none'})")
    return table[name]


def cogs_breakdown(process: ProcessSpec, basis: CostBasis, capex: float = 0.0,
                   depreciation_years: float = 10.0) -> dict:
    """Annual cost of goods, by category and by step, and per gram released.

    Costs are counted per batch *started*: a failed batch consumes media,
    resin, buffer, single-use items, labor and QC and releases nothing, so
    ``batch_success_rate`` raises cost per gram through the denominator
    alone. (A batch lost upstream would not use the downstream consumables;
    charging them is the conservative choice.)

    Args:
        process: The process (:class:`ProcessSpec`).
        basis: Prices and rates (:class:`CostBasis`).
        capex: Total capital cost (USD), for maintenance and depreciation.
        depreciation_years: Straight-line depreciation period.

    Returns:
        A dict of JAX scalars: the annual cost of each category (``media``,
        ``resin``, ``membranes``, ``buffers``, ``single_use``, ``labor``,
        ``qc``, ``utilities``, ``waste``, ``maintenance``,
        ``depreciation``), ``total``, ``product_per_batch_g``,
        ``annual_product_g``, ``cost_per_g``, ``failed_batch_cost`` (the
        share of ``total`` spent on batches that were not released) and
        ``steps``, a dict per step with its load, cycles, yield and costs.

    Example:
        >>> process, basis = load_cost_model()
        >>> out = cogs_breakdown(process, basis)
        >>> float(out["cost_per_g"])
    """
    p, b = process, basis
    started = jnp.asarray(p.batches_per_year, dtype=float)
    released = started * p.batch_success_rate
    V = p.working_volume_L

    media_L = V * (p.basal_media_L_per_L + p.feed_L_per_L + p.seed_media_L_per_L)
    media = V * (p.basal_media_L_per_L * _price(b.media_usd_L, "basal", "medium")
                 + p.feed_L_per_L * _price(b.media_usd_L, "feed", "medium")
                 + p.seed_media_L_per_L * _price(b.media_usd_L, "seed", "medium"))

    mass = jnp.asarray(V * p.titer_g_L, dtype=float)
    resin = membranes = buffers = buffer_L = 0.0
    single_use = jnp.asarray(p.bioreactor_single_use_usd_per_batch, dtype=float)
    steps: dict[str, dict] = {}
    for step in p.steps:
        load = mass
        entry: dict[str, Any] = {"load_g": load, "step_yield": step.step_yield}
        if isinstance(step, ChromatographyStep):
            cycles = cycles_per_batch(load, step.dynamic_binding_capacity_g_L,
                                      step.column_volume_L)
            step_resin = (step.column_volume_L * step.resin_cost_usd_L
                          * cycles / step.resin_lifetime_cycles)
            cv = sum(step.buffer_cv.values())
            step_buffer_L = step.column_volume_L * cycles * cv
            step_buffer = step.column_volume_L * cycles * sum(
                n * _price(b.buffer_usd_L, k, "buffer") for k, n in step.buffer_cv.items())
            resin = resin + step_resin
            entry.update(cycles=cycles, resin=step_resin * started)
        elif isinstance(step, FiltrationStep):
            step_membrane = step.area_m2 * step.membrane_cost_usd_m2 / step.lifetime_batches
            step_buffer_L = step.area_m2 * sum(step.buffer_L_per_m2.values())
            step_buffer = step.area_m2 * sum(
                v * _price(b.buffer_usd_L, k, "buffer") for k, v in step.buffer_L_per_m2.items())
            membranes = membranes + step_membrane
            entry.update(membranes=step_membrane * started)
        else:
            step_buffer_L = sum(step.buffer_L_per_batch.values())
            step_buffer = sum(v * _price(b.buffer_usd_L, k, "buffer")
                              for k, v in step.buffer_L_per_batch.items())
        buffers = buffers + step_buffer
        buffer_L = buffer_L + step_buffer_L
        single_use = single_use + step.single_use_usd_per_batch
        entry.update(buffers=step_buffer * started, buffer_L=step_buffer_L * started,
                     single_use=step.single_use_usd_per_batch * started)
        mass = load * step.step_yield
        steps[step.name] = entry

    labor_hours = (b.support_fte * b.hours_per_fte_year
                   + started * (b.operator_hours_per_bioreactor_batch
                                + b.operator_hours_per_step_batch * len(p.steps)))
    per_batch = media + resin + membranes + buffers + single_use \
        + b.qc_usd_per_batch + b.utilities_usd_per_L_batch * V \
        + b.waste_usd_L * (media_L + buffer_L)
    out = {
        "media": media * started,
        "resin": resin * started,
        "membranes": membranes * started,
        "buffers": buffers * started,
        "single_use": single_use * started,
        "labor": b.labor_rate_usd_h * labor_hours,
        "qc": b.qc_usd_per_batch * started,
        "utilities": b.utilities_usd_per_L_batch * V * started,
        "waste": b.waste_usd_L * (media_L + buffer_L) * started,
        "maintenance": b.maintenance_fraction * capex,
        "depreciation": capex / depreciation_years,
    }
    total = sum(out.values())
    batch_labor = b.labor_rate_usd_h * (b.operator_hours_per_bioreactor_batch
                                        + b.operator_hours_per_step_batch * len(p.steps))
    annual_product = mass * released
    out.update(
        total=total,
        product_per_batch_g=mass,
        annual_product_g=annual_product,
        cost_per_g=total / annual_product,
        failed_batch_cost=(started - released) * (per_batch + batch_labor),
        steps=steps,
    )
    return {k: (jnp.asarray(v) if not isinstance(v, dict) else v) for k, v in out.items()}


#: the categories :func:`cogs_breakdown` sums into ``total``
CATEGORIES = ("media", "resin", "membranes", "buffers", "single_use", "labor",
              "qc", "utilities", "waste", "maintenance", "depreciation")
