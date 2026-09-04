"""A typed module library for REE separation trains (#202).

``FullSeparationTrain`` composes cerium removal, group separation and
individual separations by direct calls in a set order: the topology is
fixed at import time and there is no decision variable over
connectivity. This module supplies the pieces a *graph* is built from
instead --- extract-scrub-strip, split-shell, cerium oxidation,
precipitation, and saponification with solvent regeneration (#197) ---
each declaring aqueous and organic inlet and outlet ports with the
species they carry.

What is reused rather than reinvented
-------------------------------------

* **Parameters and equations** come from :mod:`difflow.catalog`. Every
  module's :meth:`REEModule.describe` calls
  :func:`difflow.catalog.describe_class` on the unit classes it wraps, so
  the parameter schema is derived from the ``Params`` dataclasses exactly
  once, in the core, and cannot drift.
* **Phase typing** is the one thing the core catalog cannot supply, for
  the reason set out in :mod:`difflow_ree.flowsheets.ports`: both liquid
  phases are the same Python type, so a signature cannot say which inlet
  is organic. :class:`~difflow_ree.flowsheets.ports.Port` adds that.
* **Operating boundaries** come back as
  :class:`~difflow_ree.flowsheets.constraints.ConstraintSet`, built from
  the ``info`` the units already report (#193).

Every module is a plain callable taking its inlet streams in port order
and returning its outlet streams in port order plus an ``info`` dict, so
it drops straight into :class:`difflow.flowsheet.Unit` --- which is how
:mod:`difflow_ree.flowsheets.train` closes the organic loop on the
existing tear solver rather than writing a new one.

Example:
    >>> from difflow_ree.flowsheets.modules import ExtractScrubStripModule
    >>> from difflow_ree.flowsheets.extract_scrub_strip import (
    ...     ExtractScrubStripParams)
    >>> mod = ExtractScrubStripModule("sep", ExtractScrubStripParams(
    ...     extractant="D2EHPA", elements=("La", "Nd"),
    ...     target_elements=("Nd",)))
    >>> [p.name for p in mod.ports.inlets]
    ['feed', 'solvent']
    >>> [(p.name, p.phase) for p in mod.ports.outlets]
    [('raffinate', 'aqueous'), ('scrub_liquor', 'aqueous'), \
('product', 'aqueous'), ('barren_organic', 'organic')]
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, make_stream
from difflow_ree.flowsheets.constraints import (
    ConstraintSet,
    OperatingLimits,
    hydraulic_constraint,
    loading_constraint,
    phase_ratio_constraints,
    third_phase_constraint,
)
from difflow_ree.flowsheets.ports import Port, PortSet

#: Aqueous carrier species every aqueous port declares in addition to the
#: REE it carries.
AQUEOUS_CARRIER = ("H2O",)


def pad_stream(stream: Stream, species: tuple[str, ...]) -> Stream:
    """Return ``stream`` carrying a flow key for every name in ``species``.

    The tear-stream packing in :class:`difflow.flowsheet.Flowsheet` reads
    ``stream["F_<s>"]`` for every ``s`` in the flowsheet's
    ``species_order``, so a recycled stream must speak the whole
    vocabulary even where its flow is zero. Padding here, at the module
    boundary, keeps that requirement out of the unit operations.

    Args:
        stream: The stream to pad.
        species: The vocabulary to pad to.

    Returns:
        A new stream with the union of its own species and ``species``;
        added entries are zero. ``T`` and ``P`` are carried through.
    """
    flows = dict(get_flows(stream))
    for name in species:
        flows.setdefault(name, jnp.asarray(0.0, dtype=jnp.float64))
    return make_stream(flows, stream["T"], stream["P"])


class REEModule:
    """Base class for a typed, port-declaring separation module (#202).

    A subclass declares :attr:`kind`, builds a
    :class:`~difflow_ree.flowsheets.ports.PortSet` in ``__init__``, and
    implements :meth:`__call__` taking inlet streams in inlet-port order
    and returning outlet streams in outlet-port order followed by an
    ``info`` dict.

    Attributes:
        name: Instance name, unique within a train.
        ports: The typed ports.
        limits: Operating boundaries; see
            :class:`~difflow_ree.flowsheets.constraints.OperatingLimits`.
        params: The wrapped unit's / circuit's ``Params`` dataclass.
    """

    #: Library key, used by :func:`build_module` and the JSON round trip.
    kind: str = ""

    #: Classes wrapped by this module, described through
    #: :func:`difflow.catalog.describe_class` in :meth:`describe`.
    wrapped_classes: tuple[type, ...] = ()

    def __init__(
        self,
        name: str,
        ports: PortSet,
        limits: OperatingLimits | None = None,
    ):
        """Store the identity and ports.

        Args:
            name: Instance name.
            ports: Declared ports.
            limits: Operating boundaries, or None for none declared.
        """
        self.name = name
        self.ports = ports
        self.limits = limits or OperatingLimits()

    # -- introspection ---------------------------------------------------

    @property
    def inlet_names(self) -> tuple[str, ...]:
        """Inlet port names, in call order."""
        return tuple(p.name for p in self.ports.inlets)

    @property
    def outlet_names(self) -> tuple[str, ...]:
        """Outlet port names, in return order."""
        return tuple(p.name for p in self.ports.outlets)

    @property
    def species(self) -> tuple[str, ...]:
        """Every species any of this module's ports declares."""
        seen: list[str] = []
        for port in self.ports.inlets + self.ports.outlets:
            for s in port.species:
                if s not in seen:
                    seen.append(s)
        return tuple(seen)

    def describe(self) -> dict:
        """Machine-readable schema for this module.

        The parameter half is delegated to
        :func:`difflow.catalog.describe_class` on every class in
        :attr:`wrapped_classes`, so it is the same schema a GUI or code
        generator already consumes; the port half adds the phase typing
        the core catalog cannot derive from a signature.

        Returns:
            Dict with ``name``, ``kind``, ``ports``, ``limits`` and
            ``operations`` (one core :class:`~difflow.catalog.OperationSchema`
            per wrapped class, as a dict).
        """
        from difflow.catalog import describe_class

        return {
            "name": self.name,
            "kind": self.kind,
            "ports": self.ports.to_dict(),
            "limits": self.limits.to_dict(),
            "operations": [
                describe_class(cls, plugin="difflow_ree").to_dict()
                for cls in self.wrapped_classes
            ],
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"in={list(self.inlet_names)}, out={list(self.outlet_names)})"
        )

    # -- the interface subclasses fill in --------------------------------

    def __call__(self, *inlets: Stream, T: Array | float | None = None):
        """Run the module.

        Args:
            *inlets: Inlet streams, in inlet-port order.
            T: Temperature (K); None uses the module's own default.

        Returns:
            Outlet streams in outlet-port order, followed by an ``info``
            dict.
        """
        raise NotImplementedError

    def constraints(self, info: dict) -> ConstraintSet:
        """Operating-boundary constraints from a run's ``info``.

        Args:
            info: The dict returned by :meth:`__call__`.

        Returns:
            The constraints, feasible when every margin is ``>= 0``.
            Modules with no declared limits return an empty set.
        """
        return ConstraintSet()

    def to_dict(self) -> dict:
        """Serialize the module's identity and parameters.

        Parameter values go through :mod:`difflow.serialize`'s own value
        encoder, so a module's params are written by exactly the rules a
        flowsheet's are --- including its refusal to write a parameter
        that holds code rather than data (#202).

        Returns:
            Dict with ``kind``, ``name``, ``limits`` and ``params``.

        Raises:
            difflow.serialize.SerializationError: If a parameter holds a
                callable that carries no ``__difflow_spec__``.
        """
        import dataclasses

        # The value encoder is private to difflow.serialize, and used
        # rather than copied deliberately: a second encoder is a second
        # set of rules about what may be written, and they would drift.
        from difflow.serialize import _encode_value

        params = getattr(self, "params", None)
        encoded = (
            {
                f.name: _encode_value(
                    getattr(params, f.name),
                    f"module {self.name!r} field {f.name!r}",
                )
                for f in dataclasses.fields(params)
            }
            if params is not None and dataclasses.is_dataclass(params)
            else {}
        )
        return {
            "kind": self.kind,
            "name": self.name,
            "limits": self.limits.to_dict(),
            "params": encoded,
        }


# =====================================================================
# Extract - scrub - strip
# =====================================================================


class ExtractScrubStripModule(REEModule):
    """The industrial three-section circuit, with its organic loop open (#202).

    Wraps the same three units
    (:class:`~difflow_ree.units.extraction.REEExtractor`,
    :class:`~difflow_ree.units.scrubbing.REEScrubber`,
    :class:`~difflow_ree.units.stripping.REEStripper`) that
    :class:`~difflow_ree.flowsheets.extract_scrub_strip.ExtractScrubStripCircuit`
    builds, and takes the same
    :class:`~difflow_ree.flowsheets.extract_scrub_strip.ExtractScrubStripParams`.

    Two differences, and they are the point of the module (#202):

    1. **The solvent is an inlet port, not an internal make-up stream.**
       The circuit synthesises fresh, REE-free solvent on every call,
       which silently assumes perfect stripping. Here the organic enters
       through a port, so a train can feed ``barren_organic`` back to it
       and let the tear solver find the residual loading the circuit
       actually carries.
    2. **Nothing is concretised.** The circuit's ``__call__`` computes
       its recovery and purity metrics with Python ``float``, so it
       cannot be traced. This module returns streams and traced ``info``
       only; metrics are the caller's business.

    The scrub and strip aqueous solutions are still internal make-up,
    sized from the organic flow by the params' ratios, as in the circuit.

    Attributes:
        params: The circuit parameters.
        limits: Operating boundaries.

    Example:
        >>> from difflow_ree.flowsheets.extract_scrub_strip import (
        ...     ExtractScrubStripParams)
        >>> mod = ExtractScrubStripModule("sep", ExtractScrubStripParams(
        ...     extractant="D2EHPA", elements=("La", "Nd"),
        ...     target_elements=("Nd",)))
        >>> mod.kind
        'extract_scrub_strip'
    """

    kind = "extract_scrub_strip"

    def __init__(
        self,
        name: str,
        params,
        limits: OperatingLimits | None = None,
        T: float = 298.15,
    ):
        """Build the three sections.

        Args:
            name: Instance name.
            params: An
                :class:`~difflow_ree.flowsheets.extract_scrub_strip.ExtractScrubStripParams`.
            limits: Operating boundaries.
            T: Default temperature (K).
        """
        from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
        from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams
        from difflow_ree.units.stripping import REEStripper, StripperParams

        self.params = params
        self.T = T
        elements = tuple(params.elements)
        aqueous_species = AQUEOUS_CARRIER + elements
        organic_species = (params.extractant, params.diluent) + elements

        super().__init__(
            name,
            PortSet(
                inlets=(
                    Port("feed", "aqueous", "inlet", aqueous_species,
                         "aqueous REE feed to the extraction section"),
                    Port("solvent", "organic", "inlet", organic_species,
                         "organic entering extraction; fresh or recycled "
                         "barren organic"),
                ),
                outlets=(
                    Port("raffinate", "aqueous", "outlet", aqueous_species,
                         "depleted aqueous from extraction"),
                    Port("scrub_liquor", "aqueous", "outlet", aqueous_species,
                         "rejected REE from scrubbing"),
                    Port("product", "aqueous", "outlet", aqueous_species,
                         "stripped REE product"),
                    Port("barren_organic", "organic", "outlet", organic_species,
                         "stripped organic, for recycle to the solvent inlet"),
                ),
            ),
            limits,
        )

        self.extractor = REEExtractor(REEExtractorParams(
            n_stages=params.n_extraction_stages,
            extractant=params.extractant,
            elements=elements,
            diluent=params.diluent,
            pH=params.extraction_pH,
            extractant_conc=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,  # see #195
            mechanism=params.mechanism,  # see #195
            capacity_sharpness=params.capacity_sharpness,  # see #193
        ))
        self.scrubber = REEScrubber(ScrubberParams(
            n_stages=params.n_scrubbing_stages,
            extractant=params.extractant,
            elements=elements,
            target_elements=tuple(params.target_elements),
            diluent=params.diluent,
            pH=params.scrubbing_pH,
            extractant_conc=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,
            mechanism=params.mechanism,
        ))
        self.stripper = REEStripper(StripperParams(
            n_stages=params.n_stripping_stages,
            extractant=params.extractant,
            elements=elements,
            diluent=params.diluent,
            pH=params.stripping_pH,
            extractant_conc=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,
            mechanism=params.mechanism,
        ))
        self.wrapped_classes = (REEExtractor, REEScrubber, REEStripper)

    # -- helpers ---------------------------------------------------------

    def fresh_solvent(
        self,
        feed: Stream,
        T: Array | float | None = None,
    ) -> Stream:
        """The REE-free solvent the open-loop circuit would synthesise.

        Kept because it is exactly the open-loop assumption --- perfectly
        stripped organic --- and so is both the natural tear-stream
        initial guess and the baseline the closed loop is measured
        against (#202).

        Args:
            feed: The aqueous feed, whose flow sets the organic flow
                through ``solvent_to_feed_ratio``.
            T: Temperature (K); None uses the module default.

        Returns:
            An organic stream carrying extractant, diluent and zero REE.
        """
        p = self.params
        flows = get_flows(feed)
        F_aq = jnp.asarray(flows.get("H2O", 1.0), dtype=jnp.float64)
        F_org = F_aq * p.solvent_to_feed_ratio
        solvent = {
            p.diluent: F_org,
            p.extractant: p.extractant_conc * F_org,
        }
        for elem in p.elements:
            solvent[elem] = jnp.asarray(0.0, dtype=jnp.float64)
        return make_stream(
            solvent,
            self.T if T is None else T,
            feed["P"],
        )

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, Stream, Stream, dict]:
        """Run extraction, scrubbing and stripping in series.

        Args:
            feed: Aqueous REE feed.
            solvent: Organic entering the extraction section. Its
                extractant plus diluent flow sets the scrub and strip
                make-up flows, so a recycled organic sizes them exactly
                as a fresh one would.
            T: Temperature (K); None uses the module default.

        Returns:
            ``(raffinate, scrub_liquor, product, barren_organic, info)``.
            ``info`` carries the three sections' own dicts under
            ``extraction`` / ``scrubbing`` / ``stripping`` plus the
            derived ``phase_ratio``, ``throughput`` and
            ``organic_ree_in`` (the residual loading the recycle carries,
            which is zero exactly when stripping is perfect).
        """
        p = self.params
        T = jnp.asarray(self.T if T is None else T, dtype=jnp.float64)
        P = feed["P"]

        solvent_flows = get_flows(solvent)
        F_org = (
            jnp.asarray(solvent_flows.get(p.extractant, 0.0), dtype=jnp.float64)
            + jnp.asarray(solvent_flows.get(p.diluent, 0.0), dtype=jnp.float64)
        )
        F_aq = jnp.asarray(
            get_flows(feed).get("H2O", 1.0), dtype=jnp.float64
        )

        raffinate, loaded_org, ext_info = self.extractor(
            feed, solvent, T, pH=p.extraction_pH
        )

        scrub_soln = _aqueous_makeup(
            F_org * p.scrub_to_solvent_ratio, p.elements, T, P
        )
        scrub_liquor, scrubbed_org, scrub_info = self.scrubber(
            loaded_org, scrub_soln, T, pH=p.scrubbing_pH
        )

        strip_acid = _aqueous_makeup(
            F_org * p.strip_to_solvent_ratio, p.elements, T, P
        )
        product, barren_org, strip_info = self.stripper(
            scrubbed_org, strip_acid, T, pH=p.stripping_pH
        )

        organic_ree_in = sum(
            (jnp.asarray(solvent_flows.get(e, 0.0), dtype=jnp.float64)
             for e in p.elements),
            jnp.asarray(0.0, dtype=jnp.float64),
        )

        info = {
            "extraction": ext_info,
            "scrubbing": scrub_info,
            "stripping": strip_info,
            "phase_ratio": F_org / jnp.maximum(F_aq, 1e-300),
            "throughput": F_aq + F_org,
            # The quantity an open loop assumes is zero (#202).
            "organic_ree_in": organic_ree_in,
        }
        return raffinate, scrub_liquor, product, barren_org, info

    def constraints(self, info: dict) -> ConstraintSet:
        """Third-phase, loading, hydraulic and phase-ratio margins.

        Args:
            info: The dict returned by :meth:`__call__`.

        Returns:
            One constraint per declared limit; an empty set when
            :attr:`limits` declares none. The loading margins come from
            the extraction section's ``theta_total`` (#193), so they need
            ``include_loading`` on the extractor, which is its default.
        """
        lim = self.limits
        ext = info.get("extraction", {})
        out = []
        theta = ext.get("theta_total")
        if theta is not None:
            if lim.third_phase_loading is not None:
                out.append(third_phase_constraint(
                    theta, lim.third_phase_loading, module=self.name
                ))
            if lim.max_loading is not None:
                out.append(loading_constraint(
                    theta, lim.max_loading, module=self.name
                ))
        if lim.hydraulic_capacity is not None:
            out.append(hydraulic_constraint(
                info["throughput"], lim.hydraulic_capacity, module=self.name
            ))
        out.extend(phase_ratio_constraints(
            info["phase_ratio"], lim.min_phase_ratio, lim.max_phase_ratio,
            module=self.name,
        ))
        return ConstraintSet(tuple(out))


def _aqueous_makeup(
    F_water: Array,
    elements: tuple[str, ...],
    T: Array,
    P: Array,
) -> Stream:
    """A fresh aqueous stream carrying water and no REE.

    Args:
        F_water: Water flow.
        elements: REE keys to declare, at zero flow.
        T: Temperature (K).
        P: Pressure (Pa).

    Returns:
        The stream.
    """
    flows = {"H2O": F_water}
    for elem in elements:
        flows[elem] = jnp.asarray(0.0, dtype=jnp.float64)
    return make_stream(flows, T, P)


# =====================================================================
# Split-shell
# =====================================================================


class SplitShellModule(REEModule):
    """The branched split-shell cascade as a graph module (#202).

    Wraps :class:`~difflow_ree.flowsheets.split_shell.SplitShellCascade`,
    whose products are organic side-draws taken at the split points and
    whose raffinate is the unextracted aqueous remainder. The number of
    outlet ports therefore follows ``split_points``, which is the
    clearest demonstration in the library that ports are data.

    The wrapped cascade iterates in Python floats, so this module is a
    correct *eager* graph node but is not traceable; use it in screening
    and in eager trains, not inside ``jit``.

    Attributes:
        params: The cascade parameters.

    Example:
        >>> from difflow_ree.flowsheets.split_shell import SplitShellParams
        >>> mod = SplitShellModule("shell", SplitShellParams(
        ...     extractant="D2EHPA", elements=("La", "Nd", "Dy"),
        ...     n_stages=12, split_points=(4, 8)))
        >>> [p.name for p in mod.ports.outlets]
        ['product_1', 'product_2', 'product_3', 'raffinate']
    """

    kind = "split_shell"

    def __init__(
        self,
        name: str,
        params,
        limits: OperatingLimits | None = None,
        T: float = 298.15,
    ):
        """Build the cascade and derive its ports from the split points.

        Args:
            name: Instance name.
            params: A
                :class:`~difflow_ree.flowsheets.split_shell.SplitShellParams`.
            limits: Operating boundaries.
            T: Default temperature (K).
        """
        from difflow_ree.flowsheets.split_shell import SplitShellCascade

        self.params = params
        self.T = T
        elements = tuple(params.elements)
        aqueous_species = AQUEOUS_CARRIER + elements
        organic_species = (params.extractant, params.diluent) + elements
        n_products = len(sorted(params.split_points)) + 1

        outlets = tuple(
            Port(f"product_{i + 1}", "organic", "outlet", organic_species,
                 f"organic side-draw from cascade section {i + 1}")
            for i in range(n_products)
        ) + (
            Port("raffinate", "aqueous", "outlet", aqueous_species,
                 "unextracted aqueous remainder"),
        )

        super().__init__(
            name,
            PortSet(
                inlets=(
                    Port("feed", "aqueous", "inlet", aqueous_species,
                         "aqueous REE feed"),
                    Port("solvent", "organic", "inlet", organic_species,
                         "organic solvent, counter-current through all "
                         "sections"),
                ),
                outlets=outlets,
            ),
            limits,
        )
        self.cascade = SplitShellCascade(params)
        self.wrapped_classes = (SplitShellCascade,)

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float | None = None,
    ):
        """Run the cascade and turn its product dicts into streams.

        Args:
            feed: Aqueous REE feed.
            solvent: Organic solvent.
            T: Temperature (K); None uses the module default.

        Returns:
            The organic side-draws in split order, the aqueous
            raffinate, and the cascade's own result dict as ``info``.
        """
        p = self.params
        T = jnp.asarray(self.T if T is None else T, dtype=jnp.float64)
        P = feed["P"]
        result = self.cascade(feed, solvent, T)

        solvent_flows = get_flows(solvent)
        carrier = {
            p.extractant: jnp.asarray(
                solvent_flows.get(p.extractant, 0.0), dtype=jnp.float64),
            p.diluent: jnp.asarray(
                solvent_flows.get(p.diluent, 0.0), dtype=jnp.float64),
        }

        streams = []
        for port in self.ports.outlets[:-1]:
            flows = dict(carrier)
            flows.update({
                e: jnp.asarray(v, dtype=jnp.float64)
                for e, v in result["products"][port.name]["flows"].items()
            })
            streams.append(make_stream(flows, T, P))

        raff = {"H2O": jnp.asarray(
            get_flows(feed).get("H2O", 1.0), dtype=jnp.float64)}
        raff.update({
            e: jnp.asarray(v, dtype=jnp.float64)
            for e, v in result["products"]["raffinate"]["flows"].items()
        })
        streams.append(make_stream(raff, T, P))
        return (*streams, result)


# =====================================================================
# Cerium oxidation
# =====================================================================


class CeriumOxidationModule(REEModule):
    """Selective cerium removal by oxidation to insoluble CeO2 (#202).

    Wraps :class:`~difflow_ree.units.cerium.CeriumOxidizer`. The oxide
    leaves on a **solid** port, which the port type system will not let
    anyone connect to a liquid contactor.

    The wrapped unit concretises its ``info``, so like
    :class:`SplitShellModule` this is an eager-only graph node.

    Attributes:
        params: The oxidizer parameters.
    """

    kind = "cerium_oxidation"

    def __init__(
        self,
        name: str,
        params,
        limits: OperatingLimits | None = None,
        T: float | None = None,
    ):
        """Build the oxidizer and declare its ports.

        Args:
            name: Instance name.
            params: A
                :class:`~difflow_ree.units.cerium.CeriumOxidizerParams`.
            limits: Operating boundaries.
            T: Default temperature (K); None uses the params'.
        """
        from difflow_ree.units.cerium import CeriumOxidizer

        self.params = params
        self.T = params.temperature if T is None else T
        elements = tuple(params.elements)
        aqueous_species = AQUEOUS_CARRIER + elements

        super().__init__(
            name,
            PortSet(
                inlets=(
                    Port("feed", "aqueous", "inlet", aqueous_species,
                         "aqueous REE solution containing Ce(III)"),
                ),
                outlets=(
                    Port("filtrate", "aqueous", "outlet", aqueous_species,
                         "Ce-depleted REE solution"),
                    Port("ceo2", "solid", "outlet", ("Ce",),
                         "CeO2 precipitate"),
                ),
            ),
            limits,
        )
        self.oxidizer = CeriumOxidizer(params)
        self.wrapped_classes = (CeriumOxidizer,)

    def __call__(
        self,
        feed: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Oxidize and remove cerium.

        Args:
            feed: Aqueous REE feed.
            T: Temperature (K); None uses the module default.

        Returns:
            ``(filtrate, ceo2, info)``. The solid, which the unit returns
            as a bare flow dict, is lifted to a Stream so it can travel on
            a port like anything else.
        """
        T = jnp.asarray(self.T if T is None else T, dtype=jnp.float64)
        filtrate, solid_flows, info = self.oxidizer(feed, T)
        solid = make_stream(solid_flows, T, feed["P"])
        return filtrate, solid, info


# =====================================================================
# Precipitation
# =====================================================================


class PrecipitationModule(REEModule):
    """Oxalate, carbonate or hydroxide precipitation of a product (#202).

    Wraps one of the precipitators in
    :mod:`difflow_ree.units.precipitation`. The precipitant is internal
    make-up sized from the stoichiometric requirement and the params'
    ``precipitant_excess``, so the module presents a single aqueous
    inlet.

    Attributes:
        params: The precipitator parameters.
        precipitant: ``"oxalate"``, ``"carbonate"`` or ``"hydroxide"``.
    """

    kind = "precipitation"

    #: precipitant name -> (unit class, reagent species, mol reagent per mol REE)
    PRECIPITANTS = {
        "oxalate": ("OxalatePrecipitator", "C2O4", 1.5),
        "carbonate": ("CarbonatePrecipitator", "CO3", 1.5),
        "hydroxide": ("HydroxidePrecipitator", "OH", 3.0),
    }

    def __init__(
        self,
        name: str,
        params,
        precipitant: str = "oxalate",
        limits: OperatingLimits | None = None,
        T: float | None = None,
    ):
        """Build the precipitator and declare its ports.

        Args:
            name: Instance name.
            params: A
                :class:`~difflow_ree.units.precipitation.PrecipitatorParams`.
            precipitant: Which precipitant, a key of :attr:`PRECIPITANTS`.
            limits: Operating boundaries.
            T: Default temperature (K); None uses the params'.

        Raises:
            ValueError: On an unknown precipitant.
        """
        import difflow_ree.units.precipitation as precip_mod

        if precipitant not in self.PRECIPITANTS:
            raise ValueError(
                f"precipitant must be one of "
                f"{sorted(self.PRECIPITANTS)}, got {precipitant!r}."
            )
        cls_name, reagent, stoich = self.PRECIPITANTS[precipitant]
        cls = getattr(precip_mod, cls_name)

        self.params = params
        self.precipitant = precipitant
        self._reagent = reagent
        self._stoich = stoich
        self.T = params.temperature if T is None else T
        elements = tuple(params.elements)
        aqueous_species = AQUEOUS_CARRIER + elements

        super().__init__(
            name,
            PortSet(
                inlets=(
                    Port("feed", "aqueous", "inlet", aqueous_species,
                         "aqueous REE solution to precipitate"),
                ),
                outlets=(
                    Port("filtrate", "aqueous", "outlet", aqueous_species,
                         "REE-depleted filtrate"),
                    Port("solid", "solid", "outlet", elements,
                         f"precipitated REE {precipitant}"),
                ),
            ),
            limits,
        )
        self.precipitator = cls(params)
        self.wrapped_classes = (cls,)

    def __call__(
        self,
        feed: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Precipitate the REE in the feed.

        Args:
            feed: Aqueous REE feed.
            T: Temperature (K); None uses the module default.

        Returns:
            ``(filtrate, solid, info)``.
        """
        p = self.params
        T = jnp.asarray(self.T if T is None else T, dtype=jnp.float64)
        flows = get_flows(feed)
        total_ree = sum(
            (jnp.asarray(flows.get(e, 0.0), dtype=jnp.float64)
             for e in p.elements),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        reagent = make_stream(
            {self._reagent: self._stoich * total_ree * p.precipitant_excess},
            T,
            feed["P"],
        )
        return self.precipitator(feed, reagent, T)


# =====================================================================
# Saponification with solvent regeneration
# =====================================================================


@dataclass(repr=False)
class SolventRegenerationParams(ParamsMixin):
    """Solvent regeneration alongside saponification (#197, #202).

    A closed organic loop with no bleed accumulates whatever the stripper
    fails to remove for ever, so the loop is only physical once something
    leaves it. This is the standard treatment: a fraction of the
    circulating organic is bled to a regeneration wash and replaced with
    fresh solvent of the same carrier flow, which leaves the extractant
    and diluent inventory unchanged and removes the bled fraction of the
    residual REE.

    Attributes:
        bleed_fraction: Fraction of the circulating organic diverted to
            regeneration each pass, in [0, 1). Industrial values are
            small (1e-3 to 1e-2); 0 leaves the loop unbled, which is the
            worst case for residual loading and is deliberately
            representable.
        regeneration_efficiency: Fraction of the REE in the bled organic
            that the regeneration wash removes, in [0, 1].
    """

    bleed_fraction: float = 0.0
    regeneration_efficiency: float = 1.0

    def __post_init__(self) -> None:
        """Validate the two fractions.

        Raises:
            ValueError: If either is outside its interval.
        """
        if not 0.0 <= self.bleed_fraction < 1.0:
            raise ValueError(
                f"bleed_fraction must be in [0, 1), got "
                f"{self.bleed_fraction}."
            )
        if not 0.0 <= self.regeneration_efficiency <= 1.0:
            raise ValueError(
                f"regeneration_efficiency must be in [0, 1], got "
                f"{self.regeneration_efficiency}."
            )


class SaponificationModule(REEModule):
    """Saponification with solvent regeneration (#197, #202).

    Wraps #197's :class:`~difflow_ree.units.saponification.Saponifier` and
    puts a regeneration bleed in front of it (see
    :class:`SolventRegenerationParams`), which is what makes a closed
    organic loop a physical loop rather than an accumulator.

    Ports: one organic inlet; an organic outlet carrying the saponified,
    partly regenerated solvent, an aqueous outlet for the spent base, and
    an organic outlet for the material sent to regeneration.

    Attributes:
        params: The :class:`~difflow_ree.units.saponification.SaponifierParams`.
        regeneration: The :class:`SolventRegenerationParams`.

    Example:
        >>> from difflow_ree.units.saponification import SaponifierParams
        >>> mod = SaponificationModule("sap", SaponifierParams(
        ...     extractant="D2EHPA", elements=("La", "Nd")))
        >>> [(p.name, p.phase) for p in mod.ports.outlets]
        [('organic', 'organic'), ('spent_aqueous', 'aqueous'), \
('bleed', 'organic')]
    """

    kind = "saponification"

    def __init__(
        self,
        name: str,
        params,
        regeneration: SolventRegenerationParams | None = None,
        limits: OperatingLimits | None = None,
        T: float = 298.15,
    ):
        """Build the saponifier and declare its ports.

        Args:
            name: Instance name.
            params: A
                :class:`~difflow_ree.units.saponification.SaponifierParams`.
            regeneration: Regeneration settings; None means no bleed.
            limits: Operating boundaries.
            T: Default temperature (K).
        """
        from difflow_ree.units.saponification import Saponifier

        self.params = params
        self.regeneration = regeneration or SolventRegenerationParams()
        self.T = T
        elements = tuple(params.elements)
        self.saponifier = Saponifier(params)
        schema = self.saponifier.schema
        # organic_counter_ion / counter_ion are None for an unsaponified
        # system, so they are filtered rather than declared as a None port
        # species.
        organic_species = tuple(
            s for s in
            (params.extractant, params.diluent, schema.organic_counter_ion)
            if s
        ) + elements
        aqueous_species = (
            AQUEOUS_CARRIER
            + tuple(s for s in (schema.counter_ion, "OH") if s)
            + elements
        )

        super().__init__(
            name,
            PortSet(
                inlets=(
                    Port("organic", "organic", "inlet", organic_species,
                         "organic returning from stripping"),
                ),
                outlets=(
                    Port("organic", "organic", "outlet", organic_species,
                         "saponified, partly regenerated organic"),
                    Port("spent_aqueous", "aqueous", "outlet", aqueous_species,
                         "spent base and untransferred hydroxide"),
                    Port("bleed", "organic", "outlet", organic_species,
                         "organic diverted to regeneration"),
                ),
            ),
            limits,
        )
        self.wrapped_classes = (Saponifier,)

    def __call__(
        self,
        organic: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, Stream, dict]:
        """Bleed, regenerate, and saponify.

        The bleed removes ``bleed_fraction`` of everything in the organic
        and returns the same carrier flow as fresh solvent, so the
        extractant and diluent inventory of the loop is unchanged and
        only the REE (and counter-ion) it carried is reduced.

        Args:
            organic: Organic entering the loop's regeneration section.
            T: Temperature (K); None uses the module default.

        Returns:
            ``(organic_out, spent_aqueous, bleed, info)``. ``info`` is the
            saponifier's, plus ``bleed_fraction``,
            ``ree_removed_by_regeneration`` and ``ree_in``.
        """
        T = jnp.asarray(self.T if T is None else T, dtype=jnp.float64)
        p = self.params
        b = float(self.regeneration.bleed_fraction)
        eta = float(self.regeneration.regeneration_efficiency)
        elements = tuple(p.elements)

        flows = dict(get_flows(organic))
        carriers = {p.extractant, p.diluent}
        ree_in = sum(
            (jnp.asarray(flows.get(e, 0.0), dtype=jnp.float64)
             for e in elements),
            jnp.asarray(0.0, dtype=jnp.float64),
        )

        kept: dict[str, Array] = {}
        bled: dict[str, Array] = {}
        for key, value in flows.items():
            value = jnp.asarray(value, dtype=jnp.float64)
            if key in carriers:
                # Fresh make-up replaces the bled carrier exactly, so the
                # loop's extractant and diluent inventory is invariant.
                kept[key] = value
                bled[key] = b * value
            else:
                kept[key] = (1.0 - b * eta) * value
                bled[key] = b * value
        for elem in elements:
            kept.setdefault(elem, jnp.asarray(0.0, dtype=jnp.float64))
            bled.setdefault(elem, jnp.asarray(0.0, dtype=jnp.float64))

        regenerated = make_stream(kept, T, organic["P"])
        bleed_stream = make_stream(bled, T, organic["P"])

        organic_out, spent, info = self.saponifier(regenerated)
        info = dict(info)
        info["bleed_fraction"] = jnp.asarray(b, dtype=jnp.float64)
        info["regeneration_efficiency"] = jnp.asarray(eta, dtype=jnp.float64)
        info["ree_in"] = ree_in
        info["ree_removed_by_regeneration"] = b * eta * ree_in
        return organic_out, spent, bleed_stream, info


# =====================================================================
# Library
# =====================================================================

#: The module library, keyed by :attr:`REEModule.kind`.
MODULE_LIBRARY: dict[str, type[REEModule]] = {
    ExtractScrubStripModule.kind: ExtractScrubStripModule,
    SplitShellModule.kind: SplitShellModule,
    CeriumOxidationModule.kind: CeriumOxidationModule,
    PrecipitationModule.kind: PrecipitationModule,
    SaponificationModule.kind: SaponificationModule,
}

#: kind -> the Params dataclass its constructor takes.
MODULE_PARAMS: dict[str, str] = {
    "extract_scrub_strip": "difflow_ree.flowsheets.extract_scrub_strip:"
                           "ExtractScrubStripParams",
    "split_shell": "difflow_ree.flowsheets.split_shell:SplitShellParams",
    "cerium_oxidation": "difflow_ree.units.cerium:CeriumOxidizerParams",
    "precipitation": "difflow_ree.units.precipitation:PrecipitatorParams",
    "saponification": "difflow_ree.units.saponification:SaponifierParams",
}


def list_modules() -> tuple[str, ...]:
    """The kinds in :data:`MODULE_LIBRARY`, sorted.

    Returns:
        Tuple of module kinds.
    """
    return tuple(sorted(MODULE_LIBRARY))


def get_module_class(kind: str) -> type[REEModule]:
    """Look a module class up by kind.

    Args:
        kind: A key of :data:`MODULE_LIBRARY`.

    Returns:
        The class.

    Raises:
        KeyError: If the kind is not in the library.
    """
    try:
        return MODULE_LIBRARY[kind]
    except KeyError:
        raise KeyError(
            f"no REE module of kind {kind!r}; have {list(list_modules())}"
        ) from None


def _params_class(kind: str) -> type:
    """Import the Params dataclass for a module kind.

    Args:
        kind: A key of :data:`MODULE_PARAMS`.

    Returns:
        The dataclass.
    """
    import importlib

    module_path, _, cls_name = MODULE_PARAMS[kind].partition(":")
    return getattr(importlib.import_module(module_path), cls_name)


def module_from_dict(data: dict) -> REEModule:
    """Rebuild a module from :meth:`REEModule.to_dict` output.

    Args:
        data: The dictionary to read.

    Returns:
        The module.

    Raises:
        KeyError: If the kind is unknown.
    """
    import dataclasses

    from difflow.serialize import _decode_value

    kind = data["kind"]
    cls = get_module_class(kind)
    params_cls = _params_class(kind)
    field_names = {f.name for f in dataclasses.fields(params_cls)}
    kwargs = {}
    for key, value in data.get("params", {}).items():
        if key not in field_names:
            continue
        value = _decode_value(value)
        # Tuple-typed fields (elements, target_elements, split_points) come
        # back from JSON as lists; the Params classes are declared with
        # tuples and downstream code indexes them as such.
        kwargs[key] = tuple(value) if isinstance(value, list) else value
    params = params_cls(**kwargs)
    return cls(
        data["name"],
        params,
        limits=OperatingLimits.from_dict(data.get("limits", {})),
    )


def build_module(kind: str, name: str, params, **kwargs) -> REEModule:
    """Construct a module by kind.

    Args:
        kind: A key of :data:`MODULE_LIBRARY`.
        name: Instance name.
        params: The Params dataclass the module's constructor takes.
        **kwargs: Passed through to the constructor.

    Returns:
        The module.
    """
    return get_module_class(kind)(name, params, **kwargs)


def with_stages(module: ExtractScrubStripModule, **stages) -> ExtractScrubStripModule:
    """A copy of an extract-scrub-strip module with different stage counts.

    Useful for the screening loop, where the same topology is evaluated
    at several stage counts.

    Args:
        module: The module to copy.
        **stages: Any of ``n_extraction_stages``, ``n_scrubbing_stages``,
            ``n_stripping_stages``.

    Returns:
        A new module with the same name and limits.
    """
    return type(module)(
        module.name,
        replace(module.params, **stages),
        limits=module.limits,
        T=module.T,
    )
