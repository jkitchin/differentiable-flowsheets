"""Typed stream ports for REE separation modules (#202).

A separation train is only a *graph* if an outlet can be connected to an
inlet by name, and only a *safe* graph if the connection is checked. This
module supplies the type system: a :class:`Port` declares which phase a
stream carries and which species travel in it, and
:func:`check_connection` refuses an organic outlet plugged into an
aqueous inlet.

Why a separate type rather than reusing :class:`difflow.catalog.PortSpec`
(#202): the core catalog derives ports from a ``__call__`` signature, so
it knows a unit takes two :data:`~difflow.streams.Stream` arguments and
returns three. That is exactly the right thing for a palette, and exactly
not enough here, because both phases are the same Python type. Nothing in
``(feed: Stream, solvent: Stream) -> tuple[Stream, Stream, dict]`` says
that the first inlet is aqueous and the second organic, so nothing in the
core catalog can reject the swap. :class:`Port` adds the phase and the
species vocabulary on top; :meth:`~difflow_ree.flowsheets.modules.REEModule.describe`
carries the core :class:`~difflow.catalog.OperationSchema` alongside it,
so the parameter schema is still derived once, in the core, and not
duplicated here.

Example:
    >>> from difflow_ree.flowsheets.ports import Port, check_connection
    >>> out = Port("barren_organic", "organic", "outlet", ("D2EHPA", "Nd"))
    >>> inn = Port("solvent", "organic", "inlet", ("D2EHPA", "Nd"))
    >>> check_connection(out, inn)
    >>> aq = Port("feed", "aqueous", "inlet", ("H2O", "Nd"))
    >>> check_connection(out, aq)          # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    PortMismatchError: ...
"""

from __future__ import annotations

from dataclasses import dataclass, field

from difflow.params_mixin import ParamsMixin

#: The phases a REE flowsheet stream can be in. ``solid`` covers the
#: precipitate and calcined-oxide streams, which are carried as ordinary
#: molar-flow Streams but must never be fed to a liquid contactor.
PHASES = ("aqueous", "organic", "solid")

#: Port directions.
DIRECTIONS = ("inlet", "outlet")


class PortMismatchError(ValueError):
    """A connection joins two ports that cannot be joined.

    Raised for a direction error (outlet into outlet), a phase error
    (organic into aqueous) and for species a destination port does not
    declare, which would silently drop mass. The message names both ports
    and what specifically disagreed.
    """


@dataclass(repr=False)
class Port(ParamsMixin):
    """One typed stream connection point on a module.

    Attributes:
        name: Port name, unique within its direction on a module.
        phase: One of :data:`PHASES`.
        direction: ``"inlet"`` or ``"outlet"``.
        species: Species the port carries, as flow keys without the
            ``F_`` prefix. Empty means "not declared", which disables the
            species check rather than asserting the port is empty.
        description: Free text for reports and schemas.
    """

    name: str
    phase: str
    direction: str = "outlet"
    species: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the phase and direction.

        Raises:
            ValueError: On an unknown phase or direction.
        """
        if self.phase not in PHASES:
            raise ValueError(
                f"Port {self.name!r}: phase must be one of {list(PHASES)}, "
                f"got {self.phase!r}."
            )
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"Port {self.name!r}: direction must be one of "
                f"{list(DIRECTIONS)}, got {self.direction!r}."
            )
        self.species = tuple(self.species)

    def qualified(self, module: str) -> str:
        """``"<module>.<port>"``, the reference form used by connections.

        Args:
            module: Owning module name.

        Returns:
            The dotted reference.
        """
        return f"{module}.{self.name}"

    def to_dict(self) -> dict:
        """A plain JSON-ready dictionary.

        Returns:
            Dict with ``name``, ``phase``, ``direction``, ``species`` and
            ``description``.
        """
        return {
            "name": self.name,
            "phase": self.phase,
            "direction": self.direction,
            "species": list(self.species),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Port":
        """Rebuild a port from :meth:`to_dict` output.

        Args:
            data: The dictionary to read.

        Returns:
            The port.
        """
        return cls(
            name=data["name"],
            phase=data["phase"],
            direction=data.get("direction", "outlet"),
            species=tuple(data.get("species", ())),
            description=data.get("description", ""),
        )


def species_gap(source: Port, dest: Port) -> tuple[str, ...]:
    """Species the source carries that the destination does not declare.

    Args:
        source: Outlet port.
        dest: Inlet port.

    Returns:
        Sorted tuple of species names; empty when nothing would be lost,
        or when either port leaves its species undeclared.
    """
    if not source.species or not dest.species:
        return ()
    return tuple(sorted(set(source.species) - set(dest.species)))


def check_connection(
    source: Port,
    dest: Port,
    *,
    source_ref: str = "",
    dest_ref: str = "",
    allow_species_loss: bool = False,
) -> None:
    """Validate one outlet-to-inlet connection.

    Three checks, in the order a mistake is most likely to be made:
    direction, phase, then species vocabulary.

    Args:
        source: The upstream port; must be an outlet.
        dest: The downstream port; must be an inlet.
        source_ref: Qualified name of the source, for the message.
        dest_ref: Qualified name of the destination, for the message.
        allow_species_loss: Permit species the destination does not
            declare. Off by default because dropping a component from a
            stream is a mass-balance error that a converged flowsheet
            will happily hide.

    Raises:
        PortMismatchError: If the connection is not admissible.
    """
    src = source_ref or source.name
    dst = dest_ref or dest.name

    if source.direction != "outlet":
        raise PortMismatchError(
            f"{src} is an {source.direction} port and cannot be the source "
            f"of a connection to {dst}."
        )
    if dest.direction != "inlet":
        raise PortMismatchError(
            f"{dst} is a {dest.direction} port and cannot be the "
            f"destination of a connection from {src}."
        )
    if source.phase != dest.phase:
        raise PortMismatchError(
            f"phase mismatch: {src} carries the {source.phase} phase but "
            f"{dst} expects the {dest.phase} phase. A REE circuit's two "
            f"liquid phases are both Streams, so nothing but this check "
            f"stops an organic outlet being fed to an aqueous inlet (#202)."
        )
    if not allow_species_loss:
        lost = species_gap(source, dest)
        if lost:
            raise PortMismatchError(
                f"{src} carries {list(lost)}, which {dst} does not declare. "
                f"Connecting them would drop those components and the "
                f"flowsheet would still converge, so this is refused. "
                f"Declare the species on {dst}, or pass "
                f"allow_species_loss=True if the loss is intended (a bleed, "
                f"a purge)."
            )


@dataclass(repr=False)
class PortSet(ParamsMixin):
    """The inlet and outlet ports of one module.

    Attributes:
        inlets: Inlet ports, in call order.
        outlets: Outlet ports, in return order.
    """

    inlets: tuple[Port, ...] = field(default_factory=tuple)
    outlets: tuple[Port, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Freeze the port tuples and check names are unique."""
        self.inlets = tuple(self.inlets)
        self.outlets = tuple(self.outlets)
        for direction, ports in (("inlet", self.inlets), ("outlet", self.outlets)):
            names = [p.name for p in ports]
            if len(set(names)) != len(names):
                raise ValueError(
                    f"duplicate {direction} port names: {names}"
                )

    def inlet(self, name: str) -> Port:
        """Look up an inlet port by name.

        Args:
            name: Port name.

        Returns:
            The port.

        Raises:
            KeyError: If no inlet has that name.
        """
        for port in self.inlets:
            if port.name == name:
                return port
        raise KeyError(
            f"no inlet port {name!r}; have {[p.name for p in self.inlets]}"
        )

    def outlet(self, name: str) -> Port:
        """Look up an outlet port by name.

        Args:
            name: Port name.

        Returns:
            The port.

        Raises:
            KeyError: If no outlet has that name.
        """
        for port in self.outlets:
            if port.name == name:
                return port
        raise KeyError(
            f"no outlet port {name!r}; have {[p.name for p in self.outlets]}"
        )

    def to_dict(self) -> dict:
        """A plain JSON-ready dictionary.

        Returns:
            Dict with ``inlets`` and ``outlets`` lists.
        """
        return {
            "inlets": [p.to_dict() for p in self.inlets],
            "outlets": [p.to_dict() for p in self.outlets],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortSet":
        """Rebuild a port set from :meth:`to_dict` output.

        Args:
            data: The dictionary to read.

        Returns:
            The port set.
        """
        return cls(
            inlets=tuple(Port.from_dict(d) for d in data.get("inlets", [])),
            outlets=tuple(Port.from_dict(d) for d in data.get("outlets", [])),
        )
