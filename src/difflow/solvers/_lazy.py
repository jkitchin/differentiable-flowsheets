"""Lazy, message-carrying imports for the optional solver back ends.

``difflow`` must not gain a hard dependency on ``pounce`` or ``discopt``, so
every bridge imports its back end *inside* the function that needs it and
routes the failure through :func:`require`, which names the PyPI
distribution rather than the import name (they differ for pounce).

``asdex`` goes through the same door, but it is not a back end: it is where
the default sparsity pattern comes from, so ``difflow[solvers]`` installs it
and :mod:`difflow.solvers.sparsity` says what the alternatives are when it is
missing.
"""

from __future__ import annotations

import importlib
from types import ModuleType

#: Import name -> (PyPI distribution, extras hint).
_PYPI_NAMES = {
    "pounce": ("pounce-solver", "pounce-solver[jax]"),
    "pounce.jax": ("pounce-solver", "pounce-solver[jax]"),
    "discopt": ("discopt", "discopt"),
    "discopt.modeling": ("discopt", "discopt"),
    "asdex": ("asdex", "difflow[solvers]"),
}


def require(module: str) -> ModuleType:
    """Import ``module``, or raise a clear :class:`ImportError`.

    Args:
        module: Dotted import path, e.g. ``"pounce.jax"``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed, with the PyPI
            distribution name in the message. The import name and the
            distribution name differ for pounce (``pip install
            pounce-solver``), which is the whole reason this helper exists.

    Example:
        >>> pj = require("pounce.jax")  # doctest: +SKIP
        >>> problem = pj.from_jax(f, g, n=3, m=1)  # doctest: +SKIP
    """
    dist, install = _PYPI_NAMES.get(module, (module, module))
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised by the skip marks
        raise ImportError(
            f"difflow.solvers needs {module!r}, which is not installed. "
            f"The PyPI distribution is {dist!r}, not {module.split('.')[0]!r} "
            f"if those differ: `pip install {install}`."
        ) from exc


def have(module: str) -> bool:
    """True if ``module`` can be imported. Used to skip tests, not to branch."""
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True
