"""Collect runtime provenance for a difflow report."""

from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from importlib import metadata as _md

from difflow.report.ir import Provenance


def _pkg_version(name: str) -> str:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        # Plugins in this repo ship inside the main ``difflow`` wheel, so
        # they have no separate distribution record.  Treat a successful
        # import as "bundled with difflow".
        try:
            import importlib

            importlib.import_module(name)
            return f"bundled (difflow {_md.version('difflow')})"
        except Exception:
            return "not installed"


def _jax_info() -> tuple[str, str, bool]:
    try:
        import jax

        version = jax.__version__
        try:
            backend = jax.default_backend()
        except Exception:
            backend = "cpu"
        x64 = bool(jax.config.read("jax_enable_x64"))
        return version, backend, x64
    except Exception:
        return "unknown", "unknown", False


def _git_info() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None, None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode()
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return commit, dirty


def collect_provenance(include_git: bool = True) -> Provenance:
    """Return a :class:`Provenance` snapshot for the current environment."""
    jax_version, jax_backend, jax_x64 = _jax_info()
    commit, dirty = _git_info() if include_git else (None, None)
    plugin_versions = {
        name: _pkg_version(name)
        for name in ("difflow_bio", "difflow_ree", "difflow_cc")
    }
    return Provenance(
        difflow_version=_pkg_version("difflow"),
        plugin_versions=plugin_versions,
        jax_version=jax_version,
        jax_backend=jax_backend,
        jax_x64=jax_x64,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit=commit,
        git_dirty=dirty,
    )
