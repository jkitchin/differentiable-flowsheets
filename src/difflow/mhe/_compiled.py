"""A small cache of compiled estimator pieces, keyed by model and settings.

``jax.jit`` caches a compiled function on the *function object*, and the
estimator's pieces are closures built per call (over the model, ``theta``
and the noise levels), so without help every call compiles afresh: a
second identical :func:`~difflow.mhe.run_mhe` was as slow as the first,
and :func:`~difflow.mhe.solve_mhe` called in a loop compiled every window.
This cache hands back the same jitted closure for the same model object
and the same settings *by value* -- a new ``theta`` dict with the same
numbers hits it -- so the compilation happens once per process.

A value that cannot be keyed (a tracer, an unhashable leaf) disables the
cache for that call; the caller then builds and jits a fresh closure,
which is correct, only slower.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

#: Entries kept; each holds a model and its compiled closures alive.
MAX_ENTRIES = 16

_CACHE: OrderedDict[tuple, tuple[Any, Callable]] = OrderedDict()


def value_key(obj: Any) -> Any:
    """A hashable key equal for equal values, or ``None`` if there is none."""
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    parts = []
    for leaf in leaves:
        if isinstance(leaf, jax.core.Tracer):
            return None
        if isinstance(leaf, (np.ndarray, np.generic, jax.Array)):
            a = np.asarray(leaf)
            parts.append((a.dtype.str, a.shape, a.tobytes()))
            continue
        try:
            hash(leaf)
        except TypeError:
            return None
        parts.append((type(leaf).__name__, leaf))
    return (treedef, tuple(parts))


def cached(tag: str, model: Any, settings: tuple, build: Callable[[], Callable]
           ) -> Callable:
    """The compiled closure for ``(tag, model, settings)``, built on a miss.

    Args:
        tag: which piece (``"core"``, ``"arrival"``, ...).
        model: compared by identity.
        settings: a tuple of hashables (use :func:`value_key` for arrays
            and parameter pytrees); any ``None`` in it skips the cache.
        build: returns the jitted closure.
    """
    if any(s is None for s in settings):
        return build()
    key = (tag, id(model), settings)
    try:
        hit = _CACHE.get(key)
    except TypeError:  # an unhashable setting slipped through
        return build()
    if hit is not None and hit[0] is model:
        _CACHE.move_to_end(key)
        return hit[1]
    fn = build()
    _CACHE[key] = (model, fn)
    while len(_CACHE) > MAX_ENTRIES:
        _CACHE.popitem(last=False)
    return fn
