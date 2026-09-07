"""``python -m difflow.gui [flowsheet.json] [--port N] [--no-browser]``.

The same editor as ``difflow gui``; this form works without the console
script on PATH.
"""

from difflow.gui.server import main

if __name__ == "__main__":
    raise SystemExit(main(prog="python -m difflow.gui"))
