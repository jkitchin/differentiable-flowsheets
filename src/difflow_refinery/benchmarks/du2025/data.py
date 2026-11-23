"""Loader for the Du et al. (2025) refinery planning benchmark data.

The benchmark ships its sets and parameters as GAMS-style text files
(``all_sets.txt``, ``all_parameters.txt``) in
https://github.com/EMRPS/refinery-planning-benchmark. The paper is
CC BY-NC-SA and the data repository has no license file, so the data are
NOT vendored here: :func:`fetch` downloads them into a directory of the
user's choosing, and :func:`load_case` reads them from there.

The text format is a sequence of declarations::

    Name(dom1,dom2)  description
    /
    key1.key2=value, key1.key2=value
    /

Sets have no ``=value``. An empty declaration is ``Name(dom) text /  /``.
Keys are dotted tuples. Every parameter is returned as a ``dict`` from a key
tuple to ``float``; every set as a ``tuple`` of key tuples (a 1-D set as a
tuple of strings) in file order.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_URL = "https://github.com/EMRPS/refinery-planning-benchmark"
RAW_URL = "https://raw.githubusercontent.com/EMRPS/refinery-planning-benchmark/main"

#: Files :func:`fetch` downloads per case. ``solution_baron.gdx`` is binary
#: GAMS data; :func:`load_solution` reads it through ``gams.transfer`` when
#: that is installed, or a JSON export of it (:func:`export_solution_json`).
CASE_FILES = ("all_sets.txt", "all_parameters.txt", "solution_baron.gdx")

#: The environment variable :func:`default_data_dir` reads.
DATA_ENV = "DIFFLOW_DU2025_DATA"

#: Published Case 1 result (paper Table 3; ANTIGONE and BARON agree).
CASE1_PUBLISHED_PROFIT = 34_167_968.0

_HEADER = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(\(([^)]*)\))?\s*(.*)$")


@dataclass
class Declaration:
    """One set or parameter as read from the text file.

    Attributes:
        name: GAMS symbol name, e.g. ``"gamma"``.
        domain: Index names, e.g. ``("u", "m", "s")``; ``()`` for a scalar.
        description: The free text after the declaration.
        entries: For a set, a list of key tuples; for a parameter, a list of
            ``(key tuple, value)`` pairs. File order is kept.
    """

    name: str
    domain: tuple[str, ...]
    description: str
    entries: list = field(default_factory=list)


def _split_entries(body: str) -> list[str]:
    return [tok.strip() for tok in re.split(r"[,\n]", body) if tok.strip()]


def parse_gams_text(text: str) -> dict[str, Declaration]:
    """Parse a benchmark ``all_sets.txt`` / ``all_parameters.txt`` file.

    Args:
        text: The file contents.

    Returns:
        ``{name: Declaration}`` in file order. Set entries are key tuples;
        parameter entries are ``(key tuple, float)`` pairs.

    Raises:
        ValueError: On a declaration whose ``/ ... /`` block never closes.

    Example:
        >>> d = parse_gams_text("Sets\\nS set\\n/\\ns0, s1\\n/;")
        >>> d["S"].entries
        [('s0',), ('s1',)]
    """
    lines = text.splitlines()
    out: dict[str, Declaration] = {}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line or line in ("Sets", "Parameters", ";") or line.startswith("This file"):
            continue
        m = _HEADER.match(line)
        if m is None:
            continue
        name, dom, rest = m.group(1), m.group(3), m.group(4)
        domain = tuple(d.strip() for d in dom.split(",")) if dom else ()
        # inline empty body: "... /  /" or "... /  /;"
        inline = re.search(r"/\s*/\s*;?\s*$", rest)
        if inline:
            out[name] = Declaration(name, domain, rest[:inline.start()].strip())
            continue
        if i >= n or lines[i].strip() != "/":
            continue  # a header-like text line, not a declaration
        i += 1
        body = []
        while i < n and not lines[i].strip().startswith("/"):
            body.append(lines[i])
            i += 1
        if i >= n:
            raise ValueError(f"declaration {name!r} has no closing '/'")
        i += 1
        entries = []
        for tok in _split_entries("\n".join(body)):
            if "=" in tok:
                key, val = tok.split("=", 1)
                entries.append((tuple(key.strip().split(".")), float(val)))
            else:
                entries.append(tuple(tok.split(".")))
        out[name] = Declaration(name, domain, rest.strip(), entries)
    return out


@dataclass
class Du2025Data:
    """Sets and parameters of one benchmark case.

    Attributes:
        sets: ``{name: tuple of key tuples}``; a 1-D set holds 1-tuples.
        params: ``{name: {key tuple: float}}``.
        descriptions: ``{name: description}`` for every symbol.
        domains: ``{name: index names}`` for every symbol.
        source: Directory the case was read from.
    """

    sets: dict[str, tuple[tuple[str, ...], ...]]
    params: dict[str, dict[tuple[str, ...], float]]
    descriptions: dict[str, str]
    domains: dict[str, tuple[str, ...]]
    source: str = ""

    def elements(self, name: str) -> tuple[str, ...]:
        """The members of a 1-D set as strings (e.g. ``elements("UPF")``)."""
        return tuple(k[0] for k in self.sets[name])

    def param(self, name: str) -> dict[tuple[str, ...], float]:
        """A parameter's entries (empty dict when the file declares none)."""
        return self.params.get(name, {})

    def counts(self) -> dict[str, int]:
        """Number of entries per symbol (the loader round-trip check)."""
        c = {k: len(v) for k, v in self.sets.items()}
        c.update({k: len(v) for k, v in self.params.items()})
        return c


def load_case(path: str | os.PathLike) -> Du2025Data:
    """Read one case directory (``all_sets.txt`` + ``all_parameters.txt``).

    Args:
        path: The case directory, e.g. ``<data>/case1``.

    Returns:
        A :class:`Du2025Data`.

    Raises:
        FileNotFoundError: If either file is missing.

    Example:
        >>> data = load_case(fetch("~/du2025") / "case1")  # doctest: +SKIP
        >>> len(data.elements("S"))                         # doctest: +SKIP
        364
    """
    path = Path(path).expanduser()
    sets = parse_gams_text((path / "all_sets.txt").read_text())
    pars = parse_gams_text((path / "all_parameters.txt").read_text())
    return Du2025Data(
        sets={k: tuple(d.entries) for k, d in sets.items()},
        params={k: dict(d.entries) for k, d in pars.items()},
        descriptions={**{k: d.description for k, d in sets.items()},
                      **{k: d.description for k, d in pars.items()}},
        domains={**{k: d.domain for k, d in sets.items()},
                 **{k: d.domain for k, d in pars.items()}},
        source=str(path),
    )


def default_data_dir() -> Path:
    """``$DIFFLOW_DU2025_DATA``, else ``~/.cache/difflow/du2025``."""
    env = os.environ.get(DATA_ENV)
    return Path(env).expanduser() if env else Path("~/.cache/difflow/du2025").expanduser()


def find_case(case: int = 1, root: str | os.PathLike | None = None) -> Path | None:
    """The directory of a case if its text files are present, else ``None``.

    Args:
        case: 1, 2 or 3.
        root: Data root; defaults to :func:`default_data_dir`.
    """
    root = Path(root).expanduser() if root is not None else default_data_dir()
    d = root / f"case{case}"
    return d if (d / "all_sets.txt").exists() and (d / "all_parameters.txt").exists() else None


def fetch(root: str | os.PathLike | None = None, cases=(1,), *,
          overwrite: bool = False) -> Path:
    """Download the benchmark files into ``root/case<k>/``.

    The data are not redistributed with difflow (see the module docstring);
    this fetches them from GitHub for local use.

    Args:
        root: Destination; defaults to :func:`default_data_dir`.
        cases: Which cases to fetch.
        overwrite: Re-download files that already exist.

    Returns:
        ``root`` as a :class:`~pathlib.Path`.
    """
    root = Path(root).expanduser() if root is not None else default_data_dir()
    for k in cases:
        d = root / f"case{k}"
        d.mkdir(parents=True, exist_ok=True)
        for fname in CASE_FILES:
            dest = d / fname
            if dest.exists() and not overwrite:
                continue
            with urllib.request.urlopen(f"{RAW_URL}/case{k}/{fname}") as r:  # noqa: S310
                dest.write_bytes(r.read())
    return root


# --------------------------------------------------------------------------- #
# Published solution                                                          #
# --------------------------------------------------------------------------- #

SOLUTION_SYMBOLS = ("FVI", "FVO", "FVM", "FQ", "GM", "V", "VM",
                    "FVLI", "FVLO", "L", "Profit")


def _read_gdx(path: Path) -> dict[str, dict[tuple[str, ...], float]]:
    try:
        import gams.transfer as gt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise ImportError(
            "reading a .gdx needs gams.transfer (pip install gamspy-base "
            "'gamsapi[transfer]'); or pass a JSON export of the solution"
        ) from exc
    kwargs = {}
    try:
        import gamspy_base  # type: ignore
        kwargs["system_directory"] = gamspy_base.directory
    except ImportError:  # pragma: no cover
        pass
    c = gt.Container(str(path), **kwargs)
    out: dict[str, dict[tuple[str, ...], float]] = {}
    for name in SOLUTION_SYMBOLS:
        if name not in c.data or c.data[name].records is None:
            continue
        sym = c.data[name]
        df = sym.records
        col = "level" if "level" in df.columns else "value"
        dims = list(df.columns[:sym.dimension])
        out[name] = {tuple(str(v) for v in row[:-1]): float(row[-1])
                     for row in df[dims + [col]].itertuples(index=False)}
    return out


def load_solution(path: str | os.PathLike) -> dict[str, dict[tuple[str, ...], float]]:
    """Read a published solution (``solution_baron.gdx`` or its JSON export).

    Args:
        path: A ``.gdx`` (needs ``gams.transfer``) or a ``.json`` written by
            :func:`export_solution_json`.

    Returns:
        ``{symbol: {key tuple: level}}`` for the symbols in
        :data:`SOLUTION_SYMBOLS` that the file holds. Keys keep the GAMS
        index order, including the trailing period ``"1"``; ``Profit`` is
        keyed by ``()``.
    """
    path = Path(path).expanduser()
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        return {k: {tuple(e[:-1]): float(e[-1]) for e in v} for k, v in raw.items()}
    return _read_gdx(path)


def export_solution_json(gdx: str | os.PathLike, dest: str | os.PathLike) -> Path:
    """Convert a solution ``.gdx`` to the JSON :func:`load_solution` reads.

    Lets a machine with ``gams.transfer`` prepare the solution once for
    machines without it.
    """
    sol = _read_gdx(Path(gdx).expanduser())
    dest = Path(dest).expanduser()
    dest.write_text(json.dumps({k: [[*key, val] for key, val in v.items()]
                                for k, v in sol.items()}))
    return dest
