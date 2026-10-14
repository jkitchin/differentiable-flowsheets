"""The canvas's symbols (symbols.js), checked against the catalog they cover."""

import pathlib
import re

import pytest

from difflow import catalog

SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src/difflow/gui/frontend/src/lib/nodes/symbols.js"
)


def _object_body(source: str, name: str) -> str:
    start = source.index(f"export const {name} = {{")
    return source[start : source.index("\n}", start)]


def _mapping(source: str, name: str) -> dict[str, str]:
    body = _object_body(source, name)
    return dict(re.findall(r"^\s*'?([A-Za-z_]\w*)'?:\s*'([a-z_]+)',", body, re.M))


@pytest.fixture(scope="module")
def drawing():
    source = SOURCE.read_text()
    return {
        "symbols": set(re.findall(r"^  ([a-z_]+): \{", _object_body(source, "SYMBOLS"), re.M)),
        "operations": _mapping(source, "OPERATION_SYMBOLS"),
        "categories": _mapping(source, "CATEGORY_SYMBOLS"),
    }


def test_every_mapping_points_at_a_symbol_that_exists(drawing):
    assert len(drawing["symbols"]) > 20 and len(drawing["operations"]) > 60
    for source in ("operations", "categories"):
        for key, symbol in drawing[source].items():
            assert symbol in drawing["symbols"], f"{key} -> {symbol}"


def test_every_operation_in_the_catalog_is_drawn_by_name(drawing):
    missing = sorted(set(catalog()) - set(drawing["operations"]))
    assert not missing, (
        f"drawn by category rather than by name: {missing}. Add them to "
        "OPERATION_SYMBOLS in symbols.js and run `make gui-build`."
    )


def test_every_category_the_catalog_uses_has_a_symbol(drawing):
    used = {spec.get("category") for spec in catalog().values()} - {None, ""}
    assert not used - set(drawing["categories"])
