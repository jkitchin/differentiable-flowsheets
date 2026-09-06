"""``python -m difflow.gui [flowsheet.json] [--port N] [--no-browser]``."""

from difflow.gui.server import main

if __name__ == "__main__":
    raise SystemExit(main())
