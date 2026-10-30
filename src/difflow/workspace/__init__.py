"""The session engine: a live flowsheet and everything done to it.

:class:`FlowsheetSession` holds one flowsheet plus the state that goes
with working on it --- the code context that supplies objects JSON cannot
carry (a ``thermo``, a rate law), undo and redo, the last solve and its
diagnostics, named quantities, and the live debugger --- and every
operation on it returns a plain dict with ``ok`` rather than raising.
Nothing here opens a socket. The agent tools (:mod:`difflow.agent`,
served over MCP by ``difflow mcp``) drive it, and so does the browser app
that comes with the difflow book.

``session.py``
    the flowsheet and the operations on it.
``edit.py``
    small edits to a live flowsheet: one unit, one wire, one position.
``boilerplate.py``
    the starter code an operation needs in the code context.
``console.py``
    a Python prompt over the live flowsheet.
``sensitivity.py``
    the derivatives of a solved flowsheet, forward and reverse.
``planning.py``
    delta vectors for a planning model.
``context.py``
    the assistant's brief about a flowsheet, assembled from the live model.
``docs.py``, ``docs_index.py``, ``doclinks.py``
    a unit's own docstring rendered; a search index over the book; links
    from an operation to its section.
``layout.py``, ``topology.py``
    where units go on a canvas, and the flowsheet described for a reader.
``examples/``
    example flowsheets, as files.
"""

from difflow.workspace import console, sensitivity
from difflow.workspace.session import FlowsheetSession, evaluate_context

__all__ = ["FlowsheetSession", "console", "evaluate_context", "sensitivity"]
