"""Stream-returning palette entries for the REE circuit templates.

``ExtractStripCircuit``, ``ExtractScrubStripCircuit``, ``SplitShellCascade``
and ``FullSeparationTrain`` return one results dict: the outlet streams
sit under named keys beside recovery, purity and mass-balance figures.
That is the right shape for a script that runs a circuit and reads its
metrics, and the wrong one for a flowsheet, which maps a unit's return
value onto its outlets by position. Registered as palette operations
they handed the whole dict downstream as if it were a stream (wiring
``ExtractStripCircuit`` into a ``Heater`` died on ``KeyError: 'T'``), and
the catalog could count no outlets for them (audit of the core outlets).

The classes here are what the palette registers under those four names.
Each takes the same ``Params`` as the circuit it wraps, runs that circuit
unchanged, and returns its outlet streams in a fixed order followed by
the rest of the results dict as ``info``, so the numbers a flowsheet sees
are the numbers the circuit reports. The circuits themselves are not
changed: their callers read the results dict by key.

Like the circuits, these compute their metrics with Python ``float`` and
are eager graph nodes: they cannot be traced under ``jax.jit`` or
``jax.jacobian``. For a train with the organic loop closed and traced
``info``, use the modules in :mod:`difflow_ree.flowsheets.modules`.
"""

from __future__ import annotations

from jax import Array

from difflow.streams import Stream
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)
from difflow_ree.flowsheets.extract_strip import (
    ExtractStripCircuit,
    ExtractStripParams,
)
from difflow_ree.flowsheets.full_train import (
    FullSeparationTrain,
    SeparationTrainParams,
)
from difflow_ree.flowsheets.split_shell import (
    SplitShellCascade,
    SplitShellParams,
)

#: the class attributes the catalog reads for a unit's documentation
_DESCRIBED = ("symbol", "equations", "assumptions", "references",
              "parameter_symbols", "parameter_units", "numerical_method")


def _described_by(circuit: type):
    """Copy the circuit's documentation attributes onto its palette entry.

    The catalog page of a palette entry should say what the circuit
    computes, and restating the equations here would let them drift.
    """
    def decorate(cls: type) -> type:
        for key in _DESCRIBED:
            if hasattr(circuit, key):
                setattr(cls, key, getattr(circuit, key))
        return cls
    return decorate


def _split(results: dict, keys: tuple[str, ...]) -> tuple:
    """``results[k]`` for each key, then everything else as ``info``."""
    streams = tuple(results[k] for k in keys)
    info = {k: v for k, v in results.items() if k not in keys}
    return (*streams, info)


@_described_by(ExtractStripCircuit)
class ExtractStripUnit:
    """:class:`~difflow_ree.flowsheets.extract_strip.ExtractStripCircuit`
    with stream outlets, for a flowsheet.

    Attributes:
        params: The circuit parameters.
        circuit: The wrapped circuit.

    Example:
        >>> from difflow.streams import make_stream
        >>> unit = ExtractStripUnit(ExtractStripParams(
        ...     extractant="D2EHPA", elements=("La", "Nd")))
        >>> feed = make_stream({"H2O": 1.0, "La": 0.01, "Nd": 0.01},
        ...                    298.15, 101325.0)
        >>> raffinate, product, barren_organic, info = unit(feed)
        >>> "recovery" in info
        True
    """

    outlet_roles = ("raffinate", "product", "barren_organic")

    def __init__(self, params: ExtractStripParams):
        """Build the wrapped circuit.

        Args:
            params: Circuit parameters.
        """
        self.params = params
        self.circuit = ExtractStripCircuit(params)

    def __call__(
        self,
        feed: Stream,
        T: Array | float = 298.15,
    ) -> tuple[Stream, Stream, Stream, dict]:
        """Run the circuit and return its streams by position.

        Args:
            feed: Aqueous REE feed.
            T: Operating temperature (K).

        Returns:
            ``(raffinate, product, barren_organic, info)``, where ``info``
            is the circuit's results dict without the three streams
            (``recovery``, ``element_recovery``, ``mass_balance``, ...).
        """
        return _split(self.circuit(feed, T), self.outlet_roles)


@_described_by(ExtractScrubStripCircuit)
class ExtractScrubStripUnit:
    """:class:`~difflow_ree.flowsheets.extract_scrub_strip.ExtractScrubStripCircuit`
    with stream outlets, for a flowsheet.

    The solvent is synthesised fresh inside the circuit on every call, as
    in the circuit. To recycle the barren organic, use
    :class:`~difflow_ree.flowsheets.modules.ExtractScrubStripModule`,
    which takes the solvent as an inlet.

    Attributes:
        params: The circuit parameters.
        circuit: The wrapped circuit.

    Example:
        >>> from difflow.streams import make_stream
        >>> unit = ExtractScrubStripUnit(ExtractScrubStripParams(
        ...     extractant="D2EHPA", elements=("La", "Nd"),
        ...     target_elements=("Nd",)))
        >>> feed = make_stream({"H2O": 1.0, "La": 0.01, "Nd": 0.01},
        ...                    298.15, 101325.0)
        >>> raffinate, scrub_liquor, product, barren, info = unit(feed)
        >>> "target_recovery" in info
        True
    """

    outlet_roles = ("raffinate", "scrub_liquor", "product", "barren_organic")

    def __init__(self, params: ExtractScrubStripParams):
        """Build the wrapped circuit.

        Args:
            params: Circuit parameters.
        """
        self.params = params
        self.circuit = ExtractScrubStripCircuit(params)

    def __call__(
        self,
        feed: Stream,
        T: Array | float = 298.15,
    ) -> tuple[Stream, Stream, Stream, Stream, dict]:
        """Run the circuit and return its streams by position.

        Args:
            feed: Aqueous REE feed.
            T: Operating temperature (K).

        Returns:
            ``(raffinate, scrub_liquor, product, barren_organic, info)``,
            where ``info`` is the circuit's results dict without the four
            streams (``target_recovery``, ``product_purity``,
            ``mass_balance``, ...).
        """
        return _split(self.circuit(feed, T), self.outlet_roles)


@_described_by(SplitShellCascade)
class SplitShellUnit:
    """:class:`~difflow_ree.flowsheets.split_shell.SplitShellCascade`
    with stream outlets, for a flowsheet.

    The cascade's ``products`` entries are not streams: each is
    ``{"stage_range", "flows", "composition"}`` with bare element keys.
    The streams come from
    :class:`~difflow_ree.flowsheets.modules.SplitShellModule`, which
    builds them (organic side-draws carrying their share of the solvent
    carrier, and an aqueous raffinate) from the same cascade call.

    The outlet count follows ``split_points``: one organic side-draw per
    cascade section, then the raffinate. With the default three split
    points that is five.

    Attributes:
        params: The cascade parameters.
        module: The wrapped module.
    """

    #: product_1..4 and the raffinate, for the default three split points
    default_outlets = 5

    def __init__(self, params: SplitShellParams):
        """Build the wrapped module.

        Args:
            params: Cascade parameters.
        """
        from difflow_ree.flowsheets.modules import SplitShellModule

        self.params = params
        self.module = SplitShellModule("split_shell", params)

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float = 298.15,
    ) -> tuple:
        """Run the cascade and return its streams by position.

        Args:
            feed: Aqueous REE feed.
            solvent: Organic solvent.
            T: Temperature (K).

        Returns:
            ``(product_1, ..., product_n, raffinate, info)``: the organic
            side-draws in split order, the aqueous raffinate, and the
            cascade's results dict as ``info``.
        """
        return self.module(feed, solvent, T)


@_described_by(FullSeparationTrain)
class SeparationTrainUnit:
    """:class:`~difflow_ree.flowsheets.full_train.FullSeparationTrain`
    with stream outlets, for a flowsheet.

    The outlets are the train's products in a fixed order, which steps
    ran deciding which are present: ``light_REE``, ``middle_REE`` and
    ``heavy_REE`` when group separation runs, otherwise the stream that
    would have gone to it (``ce_depleted``, or the feed itself when no
    step runs), so the REE always has an outlet; then ``CeO2`` when
    cerium removal runs. With the default parameters on a feed list
    containing Ce that is four outlets.

    Attributes:
        params: The train parameters.
        train: The wrapped train.
    """

    #: light, middle and heavy REE, then CeO2, for the default parameters
    default_outlets = 4

    def __init__(self, params: SeparationTrainParams):
        """Build the wrapped train.

        Args:
            params: Train parameters.
        """
        self.params = params
        self.train = FullSeparationTrain(params)

    @property
    def outlet_names(self) -> tuple[str, ...]:
        """The outlets this train returns, in order."""
        names: tuple[str, ...] = ()
        if self.train._group_separator is not None:
            names += ("light_REE", "middle_REE", "heavy_REE")
        elif self.train._ce_oxidizer is not None:
            names += ("ce_depleted",)
        else:
            names += ("feed",)
        if self.train._ce_oxidizer is not None:
            names += ("CeO2",)
        return names

    def __call__(
        self,
        feed: Stream,
        T: float = 298.15,
    ) -> tuple:
        """Run the train and return its products by position.

        Args:
            feed: Mixed REE feed.
            T: Temperature (K).

        Returns:
            The streams named by :attr:`outlet_names`, then ``info``: the
            train's results dict without its ``products``
            (``mass_balance``, ``holdup``, ``intermediates``, ``info``).
        """
        results = self.train(feed, T)
        found = {**results["products"], **results["intermediates"],
                 "feed": results["feed"]}
        streams = tuple(found[name] for name in self.outlet_names)
        info = {k: v for k, v in results.items()
                if k not in ("products", "feed")}
        return (*streams, info)


__all__ = [
    "ExtractStripUnit",
    "ExtractScrubStripUnit",
    "SplitShellUnit",
    "SeparationTrainUnit",
]
