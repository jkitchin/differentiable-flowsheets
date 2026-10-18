"""Remember the arguments an object was constructed with.

Most difflow objects can be written to a file by reading their inputs back
off the instance: a unit keeps its ``Params``, and an optional constructor
argument sits on an attribute of the same name. The dynamic units do not
fit that mould --- ``DynamicCSTR(volume=...)`` files ``volume`` away as
``params["V"]`` --- and a per-class mapping would drift the first time a
constructor changed. So their base classes record the call instead::

    class DynamicUnitBase(ABC):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            record_init_args(cls)

and :func:`init_args` hands it back, which is all :mod:`difflow.serialize`
needs to rebuild the object with ``type(obj)(**init_args(obj))``.

Only the *outermost* ``__init__`` call is recorded: a subclass that calls
``super().__init__(params, name)`` is rebuilt from its own arguments, not
from the ones it passed up.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

#: where the recorded call is kept on the instance
_ATTR = "_difflow_init_args"


def record_init_args(cls: type) -> type:
    """Make ``cls.__init__`` record the arguments of the outermost call.

    Idempotent, and a no-op for a class that does not define its own
    ``__init__`` (the inherited one already records). Returns ``cls``, so
    it also works as a class decorator.
    """
    init = cls.__dict__.get("__init__")
    if init is None or getattr(init, "_records_init_args", False):
        return cls
    try:
        signature = inspect.signature(init)
    except (TypeError, ValueError):
        return cls

    @functools.wraps(init)
    def recording_init(self, *args, **kwargs):
        if _ATTR not in self.__dict__:
            # Set before the body runs, so the super().__init__ calls it
            # makes do not overwrite the outermost call with their own.
            try:
                bound = signature.bind(self, *args, **kwargs)
            except TypeError:
                bound = None                 # init will raise it properly
            if bound is not None:
                recorded = {}
                for name, value in list(bound.arguments.items())[1:]:
                    kind = signature.parameters[name].kind
                    if kind is inspect.Parameter.VAR_KEYWORD:
                        recorded.update(value)
                    elif kind is inspect.Parameter.VAR_POSITIONAL:
                        recorded = None      # no keyword form to rebuild
                        break
                    else:
                        recorded[name] = value
                object.__setattr__(self, _ATTR, recorded)
        init(self, *args, **kwargs)

    recording_init._records_init_args = True
    cls.__init__ = recording_init
    return cls


def init_args(obj: Any) -> dict | None:
    """The keyword arguments ``obj`` was constructed with, if recorded.

    ``None`` when the class does not record its calls, or the call used
    ``*args`` that have no keyword form.
    """
    return obj.__dict__.get(_ATTR) if hasattr(obj, "__dict__") else None
