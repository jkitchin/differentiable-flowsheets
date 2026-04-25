"""JSON renderer for :class:`~difflow.report.ir.Report`."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from difflow.report.ir import Report


class _Encoder(json.JSONEncoder):
    def default(self, obj: Any):
        # JAX / numpy scalars
        if hasattr(obj, "item") and not isinstance(obj, (list, tuple, dict)):
            try:
                return obj.item()
            except Exception:
                pass
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def to_json(report: Report, indent: int = 2) -> str:
    """Render a Report as a JSON string."""
    return json.dumps(asdict(report), indent=indent, cls=_Encoder, sort_keys=False)
