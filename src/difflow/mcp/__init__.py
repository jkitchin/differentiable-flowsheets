"""difflow as an MCP server: ``difflow mcp`` (see :mod:`difflow.mcp.server`).

The tools live in :mod:`difflow.agent`, which does not need the MCP SDK;
this package only registers them and runs the transport.
"""

__all__ = ["build_server", "main"]


def __getattr__(name: str):
    # Importing difflow.mcp should not import the SDK until a server is
    # wanted, so a missing install fails at build_server with the extra
    # to install, not at import.
    if name in __all__:
        from difflow.mcp import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
