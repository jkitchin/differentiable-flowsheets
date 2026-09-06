"""A local editor for flowsheets, served to the browser.

Researchers already have difflow installed, so the browser does not need
to run the solver --- it only needs to reach one. This serves a small
HTTP API and a single-page editor on ``localhost``, with the real
package doing the work:

    python -m difflow.gui plant.json

The point is not to replace writing Python. It is to make the parts
that are tedious in Python --- seeing the topology, adjusting a
parameter and re-solving, checking what a unit expects --- quick, while
leaving the door open: the editor exports the model as a script
(:mod:`difflow.codegen`) or as JSON (:mod:`difflow.serialize`), and
reads JSON back. An editor you can only enter is worse than none.

Units are added by clicking the palette, and streams are wired by
naming them: an outlet names a stream, and an inlet chooses one that
something already produces. Renaming a stream follows it to every
consumer, so the wiring survives the edit.

What the palette cannot add, it says so about rather than failing
later. Roughly half the catalog needs something no form can supply ---
a ``thermo`` object, a rate law --- and those entries are dimmed with
the reason (:attr:`~difflow.catalog.OperationSchema.is_buildable`).
For them the route is still Python, then JSON, then here.

Everything comes from :mod:`difflow.catalog`, so the palette lists
whatever is registered, plugins included, with the ports and parameter
schema each unit actually declares.

Deliberately stdlib only: a development tool that made difflow depend
on a web framework would be a poor trade. It binds to 127.0.0.1 and is
meant for a single local user --- it is not hardened for exposure to a
network.

The package is laid out so the front end can grow without the Python
growing with it:

``session.py``
    the flowsheet and the operations on it, usable without a socket.
``server.py``
    the wire encoding, the routes and the stdlib HTTP server.
``static/``
    the page as built files on disk, served by ``server.py``.
"""

from difflow.gui.server import (
    DEFAULT_PORT,
    HOST,
    NON_FINITE,
    STATIC,
    main,
    make_server,
    page,
    serve,
)
# The wire encoding is private, but it moved modules in the split and callers
# reached for it at ``difflow.gui``; keep that name pointing at it.
from difflow.gui.server import _json_restore, _json_safe  # noqa: F401
from difflow.gui.session import FlowsheetSession

__all__ = [
    "DEFAULT_PORT",
    "FlowsheetSession",
    "HOST",
    "NON_FINITE",
    "STATIC",
    "main",
    "make_server",
    "page",
    "serve",
]


def __getattr__(name: str):
    """Keep ``gui._PAGE`` working now that the page lives on disk.

    Reads it fresh, which is what a caller reaching for the served markup
    wants; the old module constant was fixed at import.
    """
    if name == "_PAGE":
        return page()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
