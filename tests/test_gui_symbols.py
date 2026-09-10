"""The canvas's symbols, checked against the catalog they have to cover.

The vocabulary itself lives in JavaScript --- `gui/frontend/src/lib/nodes/
symbols.js` --- and is unit-tested there, where the renderer is. What
`node --test` cannot see is the other side of the map: whether the
*catalog* still holds an operation the drawing does not, which is what
happens the moment a plugin gains a unit. The symptom is a node that
draws as a plain box among a flowsheet of columns and drums, and nothing
anywhere reports it.

So this file reads the JavaScript from Python and joins it to
`difflow.catalog`. It is a text-level parse on purpose: importing the
module would mean a node toolchain in the test run, and installing
difflow deliberately does not need one.
"""

import pathlib
import re

import pytest

from difflow import catalog

SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src/difflow/gui/frontend/src/lib/nodes/symbols.js"
)

# The generic fallback. Reaching it is not an error --- an operation from
# an unregistered plugin has to draw as something --- but no operation in
# difflow's own catalog should.
FALLBACK = "block"


def _object_body(source: str, name: str) -> str:
    """The text between `export const NAME = {` and the closing brace."""
    start = source.index(f"export const {name} = {{")
    return source[start : source.index("\n}", start)]


def _mapping(source: str, name: str) -> dict[str, str]:
    """`{key: 'value'}` pairs, one per line, quoted keys or not."""
    body = _object_body(source, name)
    return dict(re.findall(r"^\s*'?([A-Za-z_]\w*)'?:\s*'([a-z_]+)',", body, re.M))


@pytest.fixture(scope="module")
def drawing():
    """The symbol vocabulary and the two maps into it."""
    source = SOURCE.read_text()
    return {
        "symbols": set(
            re.findall(r"^  ([a-z_]+): \{", _object_body(source, "SYMBOLS"), re.M)
        ),
        "operations": _mapping(source, "OPERATION_SYMBOLS"),
        "categories": _mapping(source, "CATEGORY_SYMBOLS"),
    }


def test_the_file_parses_into_something_worth_checking(drawing):
    """A guard on the parse, not on the drawing.

    Every assertion below is vacuously true against an empty dict, so a
    refactor that renames `SYMBOLS` or reflows it onto one line would
    turn this file green rather than red.
    """
    assert len(drawing["symbols"]) > 20
    assert len(drawing["operations"]) > 60
    assert len(drawing["categories"]) > 10
    assert FALLBACK in drawing["symbols"]


def test_every_mapping_points_at_a_symbol_that_exists(drawing):
    known = drawing["symbols"]
    for source in ("operations", "categories"):
        for key, symbol in drawing[source].items():
            assert symbol in known, f"{key} -> {symbol}, which is not a symbol"


def test_every_operation_in_the_catalog_draws_as_equipment(drawing):
    """The one that catches a new plugin unit.

    Name first, category second, `block` never: a compressor in a gas
    network is a compressor, even though the category it sits in is
    `gas_network` and the symbol for that category is a length of pipe.
    """
    ops, cats = drawing["operations"], drawing["categories"]
    plain = {
        name: spec.get("category")
        for name, spec in catalog().items()
        if (ops.get(name) or cats.get(spec.get("category") or "") or FALLBACK)
        == FALLBACK
    }
    assert not plain, (
        f"{len(plain)} operations would draw as a plain box: {sorted(plain)}. "
        "Add them to OPERATION_SYMBOLS, or their category to CATEGORY_SYMBOLS."
    )


def test_the_map_is_still_mostly_about_operations_that_exist(drawing):
    """A floor on the overlap, which is the most that can be asserted.

    Not "no dead entries": the map covers every plugin, and a run without
    `difflow_cc` installed would report its units as removed. What can be
    said is that the map has not drifted into naming a catalog nobody
    ships any more --- a rename of the whole `Flash`/`Mixer` vocabulary
    would leave the drawing pointing at nothing and every node a box.
    """
    known = set(catalog())
    overlap = {name for name in drawing["operations"] if name in known}
    assert len(overlap) > 40, (
        f"only {len(overlap)} of the mapped operations are in the catalog; "
        "the symbol map and the catalog have drifted apart"
    )


def test_every_category_the_catalog_uses_has_a_symbol(drawing):
    """So a plugin's new unit falls back to equipment rather than a box."""
    used = {spec.get("category") for spec in catalog().values()} - {None, ""}
    missing = used - set(drawing["categories"])
    assert not missing, f"categories with no symbol: {sorted(missing)}"
