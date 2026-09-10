"""The code context a unit is waiting for, written out for you.

The palette can place half the catalog and no more. The other half wants
a ``thermo``, an ``eos``, a rate law --- things that are code, that no
form can hold, and that therefore have to be typed into the code context
before the unit will build. Saying *which* of them is missing is what
:meth:`difflow.gui.session.FlowsheetSession._needs_hint` already does.
This module answers the next question, which is what to write.

What it emits is a **starting point, not an answer**. A generated
``thermo`` names the flowsheet's own species and reads their data out of
the bundled database, which is very likely right; a generated rate law is
one made-up reaction with made-up Arrhenius parameters, which is
certainly wrong and says so in a comment. The difference is marked in the
text rather than smoothed over: a snippet that quietly looked finished
would be worse than no snippet, because the number it invented would end
up in a flowsheet.

It is deliberately not a template engine. Each recipe is a small function
of the names in play, and a need with no recipe falls back to a line
derived from the declared annotation --- which is how a plugin's own
constructor argument gets an entry here without this module knowing
anything about it.
"""

from __future__ import annotations

import inspect
import re
from typing import Sequence

#: Fields that arrive together and must not be written separately. A rate
#: law whose stoichiometry disagrees with it is the one failure mode
#: `mass_action_kinetics` exists to remove, so the reactor group is
#: emitted as one call rather than as three assignments.
KINETICS = ("rate_fn", "stoich", "rate_params")

#: The same bargain for a bioreactor: one growth law, its parameters
#: beside it.
GROWTH = ("kinetic_fn", "kinetic_params")

#: Species to write into a snippet when the flowsheet has none. Two of
#: them, because a one-species flowsheet cannot separate anything and the
#: first thing anyone does with this snippet is change the names.
FALLBACK_SPECIES = ("water", "ethanol")


def _quote(names: Sequence[str]) -> str:
    return "[" + ", ".join(f'"{n}"' for n in names) + "]"


def _species_line(species: Sequence[str]) -> str:
    return f"SPECIES = {_quote(species or FALLBACK_SPECIES)}"


def _thermo(species: Sequence[str]) -> tuple[list[str], list[str]]:
    return (
        ["from difflow import IdealThermo, get_species_data"],
        [
            "# Ideal-gas heat capacities and Antoine vapour pressures, out of",
            "# the bundled database. A species it does not have raises here,",
            "# which is the right place to find out.",
            "thermo = IdealThermo({s: get_species_data(s) for s in SPECIES})",
        ],
    )


def _eos(species: Sequence[str]) -> tuple[list[str], list[str]]:
    return (
        ["from difflow import PengRobinson, get_critical_props"],
        [
            "# Peng-Robinson with no binary interaction parameters: k_ij = 0",
            "# is the default and is an assumption, not a fact about this",
            "# mixture. Pass k_ij={('a', 'b'): 0.05, ...} once you know them.",
            "eos = PengRobinson({s: get_critical_props(s) for s in SPECIES})",
        ],
    )


def _kinetics(species: Sequence[str]) -> tuple[list[str], list[str]]:
    names = list(species or FALLBACK_SPECIES)
    a, b = names[0], names[1] if len(names) > 1 else names[0]
    return (
        ["from difflow import mass_action_kinetics"],
        [
            f"# INVENTED: one reaction, {a} -> {b}, with Arrhenius parameters",
            "# that are placeholders and nothing else. Replace the equation and",
            "# the numbers with your own; the shape is what is being shown.",
            "#",
            "# One call because rate_fn, stoich and rate_params have to agree:",
            "# built together from the reaction list, they cannot drift.",
            "kin = mass_action_kinetics([{",
            f'    "equation": "{a} -> {b}",',
            f'    "reactants": {{"{a}": 1.0}}, "products": {{"{b}": 1.0}},',
            '    "rate_params": {"A": 1.0e3, "Ea": 40_000.0, "n": 0.0},',
            "}], SPECIES)",
        ],
    )


def _growth(species: Sequence[str]) -> tuple[list[str], list[str]]:
    return (
        [],
        [
            "# INVENTED: Monod growth, with a made-up mu_max and K_s. The",
            "# signature is the part worth copying -- mu(S, params), substrate",
            "# concentration first -- and the numbers are yours to supply.",
            "kinetic_params = {\"mu_max\": 0.3, \"K_s\": 0.5}",
            "",
            "",
            "def kinetic_fn(S, params):",
            '    """Specific growth rate (1/h) at substrate concentration S."""',
            "    return params[\"mu_max\"] * S / (params[\"K_s\"] + S)",
        ],
    )


#: name -> a function of the species list returning (imports, lines).
#: Consulted before anything is derived from an annotation, because these
#: are the ones where the honest snippet is a library call and not an
#: assignment.
RECIPES = {
    "thermo": _thermo,
    "eos": _eos,
    KINETICS: _kinetics,
    GROWTH: _growth,
}


def _signature_args(name: str, description: str | None) -> str | None:
    """The argument list a docstring spells out for a callable field.

    ``rate_fn`` documents itself as ``Signature: rate_fn(C, T,
    rate_params) -> r``, and that is worth more than any stub this module
    could invent. Read out of the prose rather than declared separately,
    because the prose is what the catalog already carries and a second
    copy would be a second thing to keep true.
    """
    if not description:
        return None
    found = re.search(rf"\b{re.escape(name)}\s*\(([^)]*)\)", description)
    if not found:
        return None
    args = [a.strip() for a in found.group(1).split(",") if a.strip()]
    return ", ".join(args) if args else None


def _from_annotation(name: str, annotation: str, species: Sequence[str]) -> list[str]:
    """A placeholder line for a need with no recipe, read off its type.

    This is the branch a plugin's own constructor argument lands in. It
    cannot be right --- nothing here knows what ``elements`` means --- so
    it writes a value of the declared shape and marks it, rather than
    omitting the name and leaving the reader to work out that the snippet
    is incomplete.
    """
    text = str(annotation)
    if "str" in text and ("list" in text or "tuple" in text or "Sequence" in text):
        value = _quote(list(species or FALLBACK_SPECIES)[:1])
        if "tuple" in text:
            value = "(" + value[1:-1] + ",)"
    elif "str" in text:
        value = '"CHANGE ME"'
    elif "tuple" in text:
        value = "(1.0,)"
    elif "list" in text:
        value = "[1.0]"
    elif "dict" in text:
        value = "{}"
    elif "int" in text:
        value = "1"
    elif "float" in text or "Array" in text:
        value = "1.0"
    else:
        # Nothing to go on. `None` is at least a name that exists, and the
        # comment is the whole of what is being said.
        return [f"{name} = None   # PLACEHOLDER: no declared type to go on"]
    return [f"{name} = {value}   # PLACEHOLDER: {annotation}"]


def _callable_stub(name: str, description: str | None) -> list[str]:
    """A function of the right name, and of the documented shape if any."""
    args = _signature_args(name, description) or "*args"
    lines = [f"def {name}({args}):"]
    if description:
        wrapped = re.sub(r"\s+", " ", description).strip()
        for chunk in _wrap(wrapped, 68):
            lines.append(f"    # {chunk}")
    lines.append('    raise NotImplementedError("write me")')
    return lines


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def _describe(operation: str) -> dict[str, dict]:
    """Every parameter of *operation*, by name, as the catalog has it."""
    from difflow.catalog import describe_operation

    try:
        spec = describe_operation(operation)
    except KeyError:
        return {}
    return {p["name"]: p for p in spec.to_dict()["parameters"]}


def _annotations(operation: str) -> dict[str, str]:
    """Constructor argument annotations, for the needs that are not fields."""
    from difflow.catalog import _default_registry

    info = _default_registry().list_operations().get(operation)
    if info is None:
        return {}
    try:
        sig = inspect.signature(info.cls.__init__)
    except (TypeError, ValueError):
        return {}
    out = {}
    for name, parameter in sig.parameters.items():
        if parameter.annotation is not inspect.Parameter.empty:
            out[name] = _pretty(parameter.annotation)
    return out


def _pretty(annotation) -> str:
    """``<class 'str'>`` is a repr, not a type the reader recognises.

    The catalog reports a field's type as text, and for a plain class
    that text is the repr --- so the comment this ends up in has to be
    cleaned whichever side the annotation came from.
    """
    if isinstance(annotation, type):
        return annotation.__name__
    text = str(annotation).replace("typing.", "")
    found = re.fullmatch(r"<(?:class|enum) '([^']+)'>", text)
    if found:
        return found.group(1).rsplit(".", 1)[-1]
    return text


def snippet(operation: str, needs: Sequence[str],
            species: Sequence[str] = ()) -> str:
    """Python defining what *operation* is waiting for.

    Args:
        operation: the registered name, used to look the needs up in the
            catalog --- a need's declared type is what decides the shape
            of the line written for it.
        needs: what the drop could not supply, as
            :func:`difflow.gui.edit.unmet` reports it.
        species: the flowsheet's species order, so the snippet names the
            species this flowsheet actually carries. Empty falls back to
            :data:`FALLBACK_SPECIES`, which is a guess and reads as one.

    Returns:
        A block of Python, ready to be pasted into (or appended to) the
        code context, or ``""`` when there is nothing to write. Every
        invented number is commented as invented.
    """
    wanted = [n for n in dict.fromkeys(needs) if n]
    if not wanted:
        return ""
    fields = _describe(operation)
    annotations = _annotations(operation)
    imports: list[str] = []
    body: list[str] = []
    done: set[str] = set()

    # `species_order` is never written as an assignment under that name:
    # what the recipes build from is SPECIES, which is the name a difflow
    # script uses and the one the editor already matches. Whether it
    # appears is decided after the body, since a snippet that never
    # mentions it should not open by defining it.
    asked_for_species = any(n in wanted for n in ("species_order", "SPECIES"))
    done.update({"species_order", "SPECIES"})

    for group, recipe in RECIPES.items():
        keys = (group,) if isinstance(group, str) else group
        if not any(k in wanted for k in keys):
            continue
        extra_imports, lines = recipe(species)
        imports.extend(extra_imports)
        body.extend(lines)
        body.append("")
        done.update(keys)

    for name in wanted:
        if name in done:
            continue
        field = fields.get(name, {})
        if field.get("is_callable"):
            body.extend(_callable_stub(name, field.get("description")))
        else:
            annotation = _pretty(field.get("type") or annotations.get(name) or "")
            if field.get("description"):
                for chunk in _wrap(re.sub(r"\s+", " ",
                                          field["description"]).strip(), 68):
                    body.append(f"# {chunk}")
            body.extend(_from_annotation(name, annotation or "unknown", species))
        body.append("")

    if asked_for_species or any("SPECIES" in line for line in body):
        opening = []
        if not species:
            opening.append("# GUESSED: name the species this flowsheet carries."
                           " The header's")
            opening.append("# species box is the other way to answer this, and"
                           " the shorter one.")
        opening.append(_species_line(species))
        opening.append("")
        body = opening + body

    header = [f"# What {operation} is waiting for. Read every comment marked",
              "# INVENTED, GUESSED or PLACEHOLDER before you solve anything.",
              ""]
    if imports:
        header.extend(sorted(dict.fromkeys(imports)))
        header.append("")
    while body and not body[-1]:
        body.pop()
    return "\n".join(header + body) + "\n"


def merged(existing: str, addition: str) -> str:
    """*addition* appended to *existing*, without repeating either.

    The editor offers "add to the code context" rather than "replace it":
    a snippet is written against a flowsheet that may already define half
    of what it needs, and replacing would silently drop the other half.
    A line already present is left where it is --- which keeps a second
    drop of a second unit from writing ``SPECIES`` twice, where the
    second one would win and might not be the same list.
    """
    if not existing.strip():
        return addition
    have = {line.strip() for line in existing.splitlines() if line.strip()}
    kept = []
    for line in addition.splitlines():
        if line.strip() and line.strip() in have and not line.startswith(" "):
            continue
        kept.append(line)
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    if not kept:
        return existing
    return existing.rstrip("\n") + "\n\n" + "\n".join(kept) + "\n"
