"""The species layout of a hydroprocessing stream, and its state.

A hydrotreater or hydrocracker stream carries two kinds of material:

* **Gases** -- real species: hydrogen, hydrogen sulfide, ammonia, the C1-C4
  paraffins, any light ends the characterization keeps as real species
  (isopentane, n-pentane, n-hexane), and water. Each is one molar flow.
* **Cuts** -- the characterization's pseudo-components, on the shared grid of
  #301. A conversion unit changes what a cut is made of, and a flow of
  molecules alone cannot say that. So each cut carries, besides its molecule
  flow, a vector of **attribute flows**: extensive amounts that ride with the
  cut's molecules -- carbon and hydrogen atoms, sulfur atoms in each sulfur
  class, aromatic molecules in each ring class, and so on.

Attribute flows are *extensive*: a mixer adds them, a splitter scales them,
a phase split partitions them in proportion to their host cut's molecules
(an attribute is a property of the molecules that carry it, so it goes where
they go). That is what makes a stream of them close its element balances to
round-off through any sequence of units, and what lets a difflow ``Stream``
carry them as ordinary ``F_`` keys:

* ``F_<gas>`` -- mol/s of a gas species,
* ``F_<cut>`` -- mol/s of a cut's molecules,
* ``F_<cut>@<attribute>`` -- mol/s of that attribute in that cut.

A difflow ``Mixer`` or ``Splitter`` therefore handles them correctly with no
knowledge of what they mean.

Which attributes exist is up to the kinetic model (:mod:`.reactor`): the
hydrotreater's are listed in :data:`difflow_refinery.hydrotreating.kinetics.HDT_ATTRIBUTES`.
Two are compulsory, ``"C"`` and ``"H"`` (carbon and hydrogen atoms), because a
cut's mass is computed from its atoms::

    m_cut = 12.0107 n_C + 1.00794 n_H + 32.065 n_S + 14.0067 n_N    (g/s)

with ``n_S`` and ``n_N`` the sums of every attribute whose element is S or N
(oxygen and metals are neglected, as in the composition module's ``carbon =
1 - H - S - N``). Its molecular weight is then ``m_cut / F_cut``: a cut whose
aromatics have been saturated is heavier per molecule, a desulfurized one
lighter, and the mass balance closes because the hydrogen came from the gas.

Atomic masses are the IUPAC 2007 standard atomic weights (Wieser & Berglund,
Pure Appl. Chem. 81, 2131-2156 (2009); page numbers unverified): C 12.0107,
H 1.00794 (the value the composition module uses), S 32.065, N 14.0067,
O 15.9994. A light end's molar mass is computed from them, so it differs
from the crude unit's tabulated one (Poling et al.) in the fifth figure; the
balances here are closed on these masses throughout.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

#: Atomic masses (g/mol) of the tracked elements.
ATOMIC_MASS: dict[str, float] = {"C": 12.0107, "H": 1.00794, "S": 32.065, "N": 14.0067, "O": 15.9994}

#: Element counts of the gas species this module knows. Light ends of a
#: characterization are paraffins (C_n H_2n+2); water's oxygen is tracked so
#: its mass is right, though no reaction here moves oxygen.
GAS_ELEMENTS: dict[str, dict[str, int]] = {
    "hydrogen": {"H": 2},
    "hydrogen_sulfide": {"H": 2, "S": 1},
    "ammonia": {"N": 1, "H": 3},
    "water": {"H": 2, "O": 1},
    "methane": {"C": 1, "H": 4},
    "ethane": {"C": 2, "H": 6},
    "propane": {"C": 3, "H": 8},
    "isobutane": {"C": 4, "H": 10},
    "n_butane": {"C": 4, "H": 10},
    "isopentane": {"C": 5, "H": 12},
    "n_pentane": {"C": 5, "H": 12},
    "n_hexane": {"C": 6, "H": 14},
}

#: The gases every hydroprocessing layout carries, in this order.
DEFAULT_GASES: tuple[str, ...] = (
    "hydrogen", "hydrogen_sulfide", "ammonia", "methane", "ethane", "propane",
    "isobutane", "n_butane")

ELEMENTS: tuple[str, ...] = ("C", "H", "S", "N")


def gas_mw(name: str) -> float:
    """Molar mass (g/mol) of a gas species from its element counts."""
    return sum(ATOMIC_MASS[e] * n for e, n in GAS_ELEMENTS[name].items())


@dataclass(frozen=True)
class Layout:
    """Which gases, cuts and attributes a hydroprocessing stream carries.

    Static (hashable): it fixes the shapes of every array in a solve.

    Attributes:
        gases: Gas species names (real species; must be in :data:`GAS_ELEMENTS`).
            ``"water"`` may be among them; it is decanted, never flashed.
        cuts: Pseudo-component names.
        attributes: Per-cut attribute names; must include ``"C"`` and ``"H"``.
        attribute_elements: For each attribute, the element it counts (one
            atom per unit), or ``None`` for a count of molecules of a kind
            (an aromatic class, olefins) that carries no element of its own.
    """

    gases: tuple[str, ...]
    cuts: tuple[str, ...]
    attributes: tuple[str, ...]
    attribute_elements: tuple[str | None, ...]

    def __post_init__(self):
        for g in self.gases:
            if g not in GAS_ELEMENTS:
                raise ValueError(f"unknown gas species {g!r}; known: {sorted(GAS_ELEMENTS)}")
        if len(self.attributes) != len(self.attribute_elements):
            raise ValueError("attributes and attribute_elements must have the same length")
        for a in ("C", "H"):
            if a not in self.attributes:
                raise ValueError(f"attribute {a!r} is required (a cut's mass is computed from its atoms)")
        if self.attribute_elements[self.attributes.index("C")] != "C" or \
                self.attribute_elements[self.attributes.index("H")] != "H":
            raise ValueError("attribute 'C' must count C and 'H' must count H")
        if len(set(self.gases) & set(self.cuts)):
            raise ValueError("a name cannot be both a gas and a cut")

    # ----- sizes and indices ----------------------------------------------

    @property
    def n_gas(self) -> int:
        return len(self.gases)

    @property
    def n_cut(self) -> int:
        return len(self.cuts)

    @property
    def n_attr(self) -> int:
        return len(self.attributes)

    def gas_index(self, name: str) -> int:
        return self.gases.index(name)

    def attr_index(self, name: str) -> int:
        return self.attributes.index(name)

    def has_gas(self, name: str) -> bool:
        return name in self.gases

    # ----- constant matrices ------------------------------------------------

    def gas_element_matrix(self) -> np.ndarray:
        """``(n_gas, 4)`` atoms of C, H, S, N per molecule of each gas."""
        return np.array([[GAS_ELEMENTS[g].get(e, 0) for e in ELEMENTS] for g in self.gases], dtype=float)

    def attr_element_matrix(self) -> np.ndarray:
        """``(n_attr, 4)`` atoms of C, H, S, N per unit of each attribute."""
        return np.array([[1.0 if el == e else 0.0 for e in ELEMENTS] for el in self.attribute_elements])

    def gas_mw(self) -> np.ndarray:
        """``(n_gas,)`` molar masses, g/mol."""
        return np.array([gas_mw(g) for g in self.gases])

    def attr_mass(self) -> np.ndarray:
        """``(n_attr,)`` g per mol of each attribute (its element's atomic mass, or 0)."""
        return np.array([ATOMIC_MASS[el] if el is not None else 0.0 for el in self.attribute_elements])

    # ----- stream keys --------------------------------------------------------

    @staticmethod
    def attr_key(cut: str, attr: str) -> str:
        return f"F_{cut}@{attr}"

    def keys(self) -> list[str]:
        """Every ``F_`` key of a stream on this layout, in state order."""
        out = [f"F_{g}" for g in self.gases] + [f"F_{c}" for c in self.cuts]
        out += [self.attr_key(c, a) for c in self.cuts for a in self.attributes]
        return out

    def with_gases(self, extra) -> "Layout":
        """A layout with more gases appended (those not already present)."""
        add = tuple(g for g in extra if g not in self.gases)
        return dataclasses.replace(self, gases=self.gases + add)


@dataclass(frozen=True)
class Flows:
    """Gas, cut and attribute flows on a :class:`Layout` (a pytree).

    Attributes:
        gas: ``(n_gas,)`` mol/s.
        cut: ``(n_cut,)`` mol/s of molecules.
        attr: ``(n_cut, n_attr)`` mol/s of each attribute in each cut.
    """

    gas: Array
    cut: Array
    attr: Array

    # ----- arithmetic ---------------------------------------------------------

    def __add__(self, other: "Flows") -> "Flows":
        return Flows(self.gas + other.gas, self.cut + other.cut, self.attr + other.attr)

    def __sub__(self, other: "Flows") -> "Flows":
        return Flows(self.gas - other.gas, self.cut - other.cut, self.attr - other.attr)

    def scale(self, s) -> "Flows":
        return Flows(self.gas * s, self.cut * s, self.attr * s)

    def split(self, gas_frac: Array, cut_frac: Array) -> "Flows":
        """The part given by per-gas and per-cut fractions; attributes follow their cut."""
        return Flows(self.gas * gas_frac, self.cut * cut_frac, self.attr * cut_frac[:, None])

    @staticmethod
    def zeros(layout: Layout) -> "Flows":
        return Flows(jnp.zeros(layout.n_gas), jnp.zeros(layout.n_cut),
                     jnp.zeros((layout.n_cut, layout.n_attr)))

    # ----- flat vector ---------------------------------------------------------

    def ravel(self) -> Array:
        return jnp.concatenate([self.gas, self.cut, self.attr.reshape(-1)])

    @staticmethod
    def unravel(v: Array, layout: Layout) -> "Flows":
        ng, nc, na = layout.n_gas, layout.n_cut, layout.n_attr
        return Flows(v[:ng], v[ng:ng + nc], v[ng + nc:].reshape(nc, na))

    # ----- per-cut properties ---------------------------------------------------

    def cut_mass(self, layout: Layout) -> Array:
        """``(n_cut,)`` mass flow of each cut, kg/s, from its atoms."""
        return self.attr @ jnp.asarray(layout.attr_mass()) / 1000.0

    def _safe_inverse(self) -> Array:
        """``1/F`` per cut, as ``F/(F^2 + eps^2)`` with ``eps = 1e-12`` of the total.

        Exactly ``1/F`` to 1e-24 relative wherever a cut has flow; zero (with
        a bounded derivative) for an empty cut, which is what keeps a reverse
        pass through an empty cut's attributes finite.
        """
        eps = 1e-12 * jax.lax.stop_gradient(jnp.sum(jnp.abs(self.cut))) + 1e-300
        return self.cut / (self.cut**2 + eps**2)

    def cut_mw(self, layout: Layout) -> Array:
        """``(n_cut,)`` molecular weight of each cut, g/mol (``mass / molecules``; 0 if empty)."""
        return 1000.0 * self.cut_mass(layout) * self._safe_inverse()

    def per_molecule(self) -> Array:
        """``(n_cut, n_attr)`` attribute per molecule of each cut (0 for an empty cut)."""
        return self.attr * self._safe_inverse()[:, None]

    def gas_mass(self, layout: Layout) -> Array:
        """``(n_gas,)`` mass flow of each gas, kg/s."""
        return self.gas * jnp.asarray(layout.gas_mw()) / 1000.0

    def mass(self, layout: Layout) -> Array:
        """Total mass flow, kg/s."""
        return jnp.sum(self.gas_mass(layout)) + jnp.sum(self.cut_mass(layout))

    def elements(self, layout: Layout) -> Array:
        """``(4,)`` mol/s of C, H, S, N atoms in the whole stream."""
        return (self.gas @ jnp.asarray(layout.gas_element_matrix())
                + jnp.sum(self.attr, axis=0) @ jnp.asarray(layout.attr_element_matrix()))

    def gas_flow(self, layout: Layout, name: str) -> Array:
        """mol/s of one gas (zero if the layout does not carry it)."""
        return self.gas[layout.gas_index(name)] if layout.has_gas(name) else jnp.asarray(0.0)

    def attribute(self, layout: Layout, name: str) -> Array:
        """``(n_cut,)`` flow of one attribute."""
        return self.attr[:, layout.attr_index(name)]

    # ----- difflow streams ---------------------------------------------------------

    def to_stream(self, layout: Layout, T, P) -> dict:
        """A difflow stream: ``F_<gas>``, ``F_<cut>``, ``F_<cut>@<attr>``, ``T``, ``P``."""
        out = {f"F_{g}": self.gas[i] for i, g in enumerate(layout.gases)}
        out.update({f"F_{c}": self.cut[i] for i, c in enumerate(layout.cuts)})
        for i, c in enumerate(layout.cuts):
            for k, a in enumerate(layout.attributes):
                out[layout.attr_key(c, a)] = self.attr[i, k]
        out["T"] = jnp.asarray(T, dtype=float)
        out["P"] = jnp.asarray(P, dtype=float)
        return out

    @staticmethod
    def from_stream(stream: Mapping, layout: Layout) -> "Flows":
        """Read a stream written by :meth:`to_stream` (missing keys count as zero)."""
        z = jnp.asarray(0.0)

        def get(k):
            return jnp.asarray(stream.get(k, z), dtype=float)

        gas = jnp.stack([get(f"F_{g}") for g in layout.gases]) if layout.gases else jnp.zeros(0)
        cut = jnp.stack([get(f"F_{c}") for c in layout.cuts])
        attr = jnp.stack([jnp.stack([get(layout.attr_key(c, a)) for a in layout.attributes])
                          for c in layout.cuts])
        return Flows(gas, cut, attr)


jax.tree_util.register_dataclass(Flows, data_fields=["gas", "cut", "attr"], meta_fields=[])


def relative_balance_error(inflow: Array, outflow: Array) -> Array:
    """``|in - out| / max(|in|, tiny)`` elementwise."""
    return jnp.abs(inflow - outflow) / jnp.maximum(jnp.abs(inflow), 1e-300)


__all__ = ["ATOMIC_MASS", "GAS_ELEMENTS", "DEFAULT_GASES", "ELEMENTS", "Layout", "Flows",
           "gas_mw", "relative_balance_error"]
