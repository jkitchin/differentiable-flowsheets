"""Value keys for objects that are passed to ``jax.jit`` as static arguments.

Why this exists: a jitted function caches its compiled executable on the
*static* arguments, and JAX compares those with ``hash`` and ``==``. A plain
Python object gets identity semantics, so two structurally identical thermo
packages -- or two identical columns -- are two cache keys and two
compilations of the same graph. Building a unit inside a loop, or a fixture
per test, then pays a full XLA compile every time round.

What makes a value key safe here is that these objects are built once from
*hashable* inputs and never mutated: ``IdealThermo`` from a dict of
``SpeciesData`` (a NamedTuple of floats), ``PengRobinson`` from a dict of
``CriticalProperties`` (likewise). The derived arrays -- ``EOSParams.a_c``
and friends -- are a pure function of those inputs, so equal inputs mean an
identical traced graph, with identical constants baked into it. The key is
taken from the constructor's arguments and never from the arrays, which is
also what keeps it working under tracing: an array field may be a tracer,
and a tracer has no value to hash.

Where a key cannot be built -- an EOS handed a raw ``k_ij`` array, a params
object holding a tracer, a callable field -- :func:`static_key` says so by
returning :data:`NO_KEY`, and :class:`ValueKeyed` falls back to identity.
That is the old behaviour: a missed cache, never a wrong hit.

The one thing to hold onto: **do not mutate an object that carries a value
key**. Its key is computed at construction, so a mutated object still
compares equal to what it used to be, and would be handed the executable
compiled for its old contents.
"""

from dataclasses import fields, is_dataclass
from typing import Any, Hashable

import numpy as np

#: Returned by :func:`static_key` for anything that has no value key.
NO_KEY = object()

#: Scalars that are already their own key.
_ATOMS = (bool, int, float, complex, str, bytes, type(None))


def static_key(value: Any) -> Hashable:
    """A hashable stand-in for ``value``, or :data:`NO_KEY`.

    Containers are walked in order, because order is meaningful here --
    ``species_order`` is what indexes every array in an EOS.
    """
    if isinstance(value, _ATOMS):
        return value

    # A ValueKeyed object already carries one; ask it rather than re-deriving.
    key = getattr(value, "_value_key", None)
    if key is not None:
        return NO_KEY if key is NO_KEY else ("keyed", type(value).__name__, key)

    if isinstance(value, (tuple, list)):          # NamedTuple included
        parts = tuple(static_key(v) for v in value)
        return NO_KEY if any(p is NO_KEY for p in parts) else (type(value).__name__, parts)

    if isinstance(value, dict):
        parts = tuple((k, static_key(v)) for k, v in value.items())
        return NO_KEY if any(p[1] is NO_KEY for p in parts) else ("dict", parts)

    if is_dataclass(value) and not isinstance(value, type):
        parts = tuple(
            (f.name, static_key(getattr(value, f.name))) for f in fields(value)
        )
        return (
            NO_KEY if any(p[1] is NO_KEY for p in parts)
            else (type(value).__name__, parts)
        )

    # Arrays: only concrete ones. A tracer has no value, and np.asarray on one
    # raises rather than returning anything usable -- that is the NO_KEY case.
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            arr = np.asarray(value)
        except Exception:
            return NO_KEY
        if arr.dtype == object:
            return NO_KEY
        return ("array", arr.shape, str(arr.dtype), arr.tobytes())

    return NO_KEY


class ValueKeyed:
    """Mixin: compare and hash by construction inputs, not by identity.

    Subclasses call :meth:`_set_value_key` at the end of ``__init__`` with the
    inputs that determine the object. Anything that cannot be keyed leaves the
    object with identity semantics.
    """

    _value_key: Any = NO_KEY

    def _set_value_key(self, *parts: Any) -> None:
        key = static_key(parts)
        # object.__setattr__, so a frozen dataclass can set it from
        # __post_init__ without tripping FrozenInstanceError.
        object.__setattr__(self, "_value_key", NO_KEY if key is NO_KEY else key)

    def __hash__(self) -> int:
        key = self._value_key
        return object.__hash__(self) if key is NO_KEY else hash((type(self).__name__, key))

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        mine, theirs = self._value_key, other._value_key
        if mine is NO_KEY or theirs is NO_KEY:
            return False
        return bool(mine == theirs)
