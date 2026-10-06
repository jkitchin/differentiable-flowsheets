"""difflow for agents: the operations behind ``difflow mcp``, as plain Python.

:class:`Workbench` holds named flowsheet sessions and exposes discovery,
building, solving and a Python escape hatch as methods that take and
return JSON-able values. :mod:`difflow.mcp` registers them as MCP tools;
nothing here depends on the MCP SDK, so the same calls serve tests, other
front ends, or a script.

:data:`TOOLS` names the methods that are tools and what kind each is:
``"read"`` changes nothing, ``"edit"`` changes a flowsheet (and can be
undone), ``"exec"`` runs Python.
"""

from difflow.agent.workbench import DEFAULT_TIMEOUT, Workbench, jsonable

#: tool name -> kind ("read", "edit" or "exec")
TOOLS: dict[str, str] = {
    # discovery
    "plugin_status": "read",
    "list_operations": "read",
    "describe_operation": "read",
    "list_api": "read",
    "describe_api": "read",
    "search_species": "read",
    "search_docs": "read",
    "list_examples": "read",
    "plugin_guide": "read",
    # sessions
    "list_sessions": "read",
    "new_session": "edit",
    "close_session": "edit",
    "open_example": "edit",
    "open_file": "exec",
    "save": "edit",
    "undo": "edit",
    "redo": "edit",
    "get_flowsheet": "read",
    # building
    "set_species": "edit",
    "set_code_context": "exec",
    "add_unit": "edit",
    "update_unit": "edit",
    "remove_unit": "edit",
    "connect": "edit",
    "disconnect": "edit",
    "set_feed": "edit",
    "remove_feed": "edit",
    # running
    "set_solver_options": "edit",
    "solve": "edit",
    "get_streams": "read",
    # diagnosis and convergence
    "diagnose": "read",
    "converge": "edit",
    "tear_analysis": "read",
    "trace_solve": "read",
    "get_unit_info": "read",
    # analysis
    "levers": "read",
    "define_quantity": "edit",
    "remove_quantity": "edit",
    "list_quantities": "read",
    "evaluate": "read",
    "sensitivity": "read",
    "sweep": "read",
    "optimize": "edit",
    "uncertainty": "read",
    "linearize": "edit",
    "report": "read",
    # Python
    "run_python": "exec",
}

__all__ = ["DEFAULT_TIMEOUT", "TOOLS", "Workbench", "jsonable"]
