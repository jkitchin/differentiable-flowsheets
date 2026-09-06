"""What the assistant is told, assembled from the live model.

The panel in the editor asks a small local language model questions
about the open flowsheet. A 3B model knows nothing about difflow, and
that is fine --- the useful thing it can do is read. So the value here
is not the model, it is the brief: the catalog entry for the block under
the cursor, the parameters that unit actually holds with their units,
the diagnostics of the solve that just failed, and the two paragraphs of
``docs/`` that address the question. Every one of those is already
computed somewhere in difflow and none of it was ever shown to anyone.

Four kinds of brief, chosen by what is being asked about:

============  ==============================================================
``block``     the catalog schema, the rendered docstring, the equations,
              and this node's own parameter values with their units
``flowsheet`` the units, the connections, the recycles and tear streams,
              and the Python ``codegen`` would emit for it
``solve``     ``last_solve_*`` --- method, residual, tolerance, iterations,
              tear streams --- plus a troubleshooting card for the symptom
``planning``  a :class:`~difflow.planning.export.DeltaVectorSet`, the
              health findings, and the assembled LP
============  ==============================================================

Every pack ends with the documentation sections that best match the
question (:mod:`difflow.gui.docs_index`), and every pack is **shown to
the user** in the panel before it is sent. That is the point: when the
local model is weak, the assembled brief is still the answer --- copy it
into a stronger assistant, or just read it, since it is a report about
the flowsheet that difflow could not previously produce.

Scoring the docs happens here rather than in the browser. The index is a
build artefact either way; where the dot product runs is not
architecture, and running it in Python makes it testable without a
headless browser and keeps a 667 KB file off the wire on every question.

Budget
------

A pack is trimmed to :data:`BUDGET` tokens, low-priority sections first,
because the default runtime is a 3B model with a 4096-token window and a
brief that overflows it is silently truncated *at the end* --- which is
where the question is. What was dropped is recorded in ``notes`` and
shown, so a thin answer has a visible cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Tokens a pack is allowed. Four characters to the token is the usual
#: rough estimate and it is close enough for prose; it under-counts code,
#: which is the safe direction.
BUDGET = 2600

#: What the model is told it is doing. Short, and mostly a fence: a small
#: model asked about a chemical process will otherwise answer from
#: whatever it remembers about chemical processes in general.
SYSTEM = (
    "You are a reading assistant inside difflow, a JAX-based "
    "differentiable flowsheet simulator. Answer only from the brief "
    "below, which was assembled from the user's live model. If the "
    "brief does not contain the answer, say so and name what would. "
    "Do not invent parameter names, units or numbers."
)

#: Symptom cards. Each is ``(name, triggers, text)``; a card is included
#: when one of its triggers appears in the error or the diagnostics. The
#: text is difflow's own troubleshooting prose --- the failures below are
#: the ones this codebase actually produces, and a general-purpose model
#: guesses at every one of them.
TROUBLESHOOTING: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "The recycle did not converge",
        ("converged", "max_iter", "residual"),
        "A sequential-modular solve tears the recycle streams and "
        "iterates. `converged=False` means it stopped at `max_iter` "
        "with the residual still above `tol`; the numbers it returns "
        "look like an answer and are not one. What helps, in order: "
        "raise `max_iter`; switch `method` to 'anderson' or 'wegstein' "
        "(direct substitution converges linearly and a loop gain near 1 "
        "makes that arbitrarily slow); damp the tear map; give the tear "
        "stream a better initial guess. A residual that *rises* is a "
        "loop gain above 1, which damping fixes and iterations do not.",
    ),
    (
        "TracerArrayConversionError",
        ("TracerArrayConversionError", "TracerBoolConversionError"),
        "A JAX tracer reached Python control flow -- `if x > 0:`, "
        "`float(x)`, `int(x)`, or a numpy call on a traced array. Use "
        "`jnp.where`, `jax.lax.cond`, `jax.lax.switch` or "
        "`jax.lax.fori_loop` instead. Inside a flowsheet this is almost "
        "always a `rate_fn` or a property correlation branching on a "
        "value rather than selecting with `jnp.where`.",
    ),
    (
        "ConcretizationError",
        ("ConcretizationError",),
        "Something needed a concrete value from an abstract one -- a "
        "shape, a loop bound, or a Python `bool`. Shapes must be static: "
        "they cannot depend on traced values. If the value is only a "
        "diagnostic, read it outside the trace.",
    ),
    (
        "NaN in the result or the gradient",
        ("nan", "NaN"),
        "The usual sources are `log(0)`, `sqrt` of a negative, `0/0`, "
        "and `x**y` at `x=0`. A forward pass can be finite while the "
        "gradient is not -- `sqrt(x)` at `x=0` is 0 with an infinite "
        "derivative. Add an epsilon, clip the argument, or use a safe "
        "form. `jax.config.update('jax_debug_nans', True)` stops at the "
        "first one.",
    ),
    (
        "Singular or ill-conditioned solve",
        ("singular", "LinAlgError", "condition", "rank"),
        "An equation-oriented or Newton solve hit a Jacobian it cannot "
        "invert. Usually a variable that nothing determines (an unset "
        "spec) or two rows saying the same thing (a redundant spec). "
        "Count equations against unknowns before reaching for a "
        "different solver.",
    ),
)


# ---------------------------------------------------------------------
# The pack
# ---------------------------------------------------------------------

def tokens(text: str) -> int:
    """A rough token count. Four characters to the token."""
    return (len(text) + 3) // 4


@dataclass
class Section:
    """One labelled piece of a brief.

    Attributes:
        title: shown as the heading, both to the model and to the user.
        body: the text.
        priority: what survives the budget. Higher is kept longer; the
            question's own subject is 9, boilerplate is 1.
    """

    title: str
    body: str
    priority: int = 5

    def render(self) -> str:
        return f"## {self.title}\n{self.body.strip()}"


@dataclass
class Pack:
    """An assembled brief, ready to send and ready to show.

    ``sections`` is already trimmed to the budget; ``notes`` says what
    was dropped to get there.
    """

    kind: str
    question: str = ""
    sections: list[Section] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    system: str = SYSTEM

    def prompt(self) -> str:
        """The brief and the question, as one string."""
        body = "\n\n".join(section.render() for section in self.sections)
        tail = f"\n\n## Question\n{self.question.strip()}" if self.question else ""
        return f"{body}{tail}"

    def to_dict(self) -> dict:
        return {
            "ok": True,
            "kind": self.kind,
            "question": self.question,
            "system": self.system,
            "sections": [{"title": s.title, "body": s.body,
                          "priority": s.priority} for s in self.sections],
            "notes": self.notes,
            "prompt": self.prompt(),
            "tokens": tokens(self.system) + tokens(self.prompt()),
        }


def fit(sections: list[Section], budget: int = BUDGET) -> tuple[list[Section], list[str]]:
    """Trim a list of sections to a token budget, cheapest first.

    Order is preserved --- a brief that reshuffles itself when it gets
    long is harder to read, and the model is not reading it in priority
    order either. Sections are dropped whole rather than truncated: half
    a parameter table is a table with parameters missing from it, which
    is worse than a note saying the table was dropped.
    """
    keep = list(sections)
    notes: list[str] = []
    total = sum(tokens(s.render()) for s in keep)
    while total > budget and keep:
        weakest = min(keep, key=lambda s: (s.priority, -tokens(s.render())))
        if weakest.priority >= 9:                # never drop the subject
            break
        keep.remove(weakest)
        total -= tokens(weakest.render())
        notes.append(f"dropped {weakest.title!r} to stay inside "
                     f"{budget} tokens")
    return keep, notes


#: A retrieved section is kept only if it scores at least this fraction
#: of the best hit. Without a floor the second hit is always *something*
#: --- a question with the word "why" in it retrieves a paragraph about
#: power-flow limits, on the strength of one shared common word.
#:
#: The floor is a backstop, not the mechanism. What actually sharpens
#: retrieval is that every pack searches on the question *plus its own
#: subject* --- the operation name, the units in the flowsheet, the
#: solver that ran. A user's question is four words long and two of them
#: are "why" and "this"; the pack knows what "this" is.
RELEVANCE = 0.6


def _documentation(question: str, sections: list[Section], limit: int = 2) -> None:
    """Append the documentation sections that best match the question."""
    from difflow.gui import docs_index

    index = docs_index.load()
    if index is None or not question.strip():
        return
    hits = docs_index.search(index, question, limit=limit)
    floor = hits[0]["score"] * RELEVANCE if hits else 0.0
    for hit in hits:
        if hit["score"] < floor:
            continue
        where = hit["path"] or hit["heading"]
        sections.append(Section(
            f"From the documentation: {where} ({hit['source']})",
            hit["text"], priority=3,
        ))


# ---------------------------------------------------------------------
# The four briefs
# ---------------------------------------------------------------------

def _fmt_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _params_table(operation, spec=None) -> str:
    """A unit's parameters, with the units and help the catalog knows."""
    from difflow.publish import _display_params        # one reducer, not two

    values = _display_params(operation)
    meta = {p.name: p for p in (spec.parameters if spec else [])}
    if not values and not meta:
        return "(this unit takes no parameters)"
    rows = []
    for name in sorted(set(values) | set(meta)):
        info = meta.get(name)
        units = f" [{info.units}]" if info and info.units else ""
        help_ = f"  -- {info.description}" if info and info.description else ""
        shown = _fmt_value(values[name]) if name in values else "(not set)"
        rows.append(f"- {name}{units} = {shown}{help_}")
    return "\n".join(rows)


def _spec_line(spec: dict) -> str:
    """One planning constraint as a row of the LP would read it."""
    coeffs = spec.get("coeffs") or {}
    left = spec.get("expr") or " + ".join(
        f"{value:g}*{name}" for name, value in coeffs.items()) or "?"
    name = f"  [{spec['name']}]" if spec.get("name") else ""
    backoff = (f"  (backoff {spec['backoff']:g})"
               if spec.get("backoff") else "")
    return f"- {left} {spec.get('op', '?')} {spec.get('rhs', '?')}{name}{backoff}"


def block(session, name: str | None = None, operation: str | None = None,
          question: str = "") -> Pack:
    """What one unit is, and what this flowsheet's copy of it is set to.

    ``name`` is a node on the canvas; ``operation`` is a catalog entry
    with no node behind it, which is what the palette asks about. Given
    both, the node wins and the operation is derived from it.
    """
    from difflow.catalog import describe_operation

    unit = None
    for candidate in getattr(session.flowsheet, "units", None) or []:
        if candidate.name == name:
            unit = candidate
            break
    if unit is None and name and not operation:
        return Pack("block", question, [
            Section("Not found", f"No unit called {name!r} is in this "
                                 "flowsheet.", priority=9)])
    if unit is not None:
        operation = type(unit.operation).__name__

    try:
        spec = describe_operation(operation)
    except KeyError:
        spec = None

    sections = [Section(
        "The unit",
        f"Operation: {operation}\n"
        + (f"Node name in this flowsheet: {unit.name}\n" if unit else "")
        + (f"Category: {spec.category}\n"
           f"Summary: {spec.description}\n" if spec else ""),
        priority=9,
    )]

    if unit is not None:
        sections.append(Section(
            f"Parameters of {unit.name}",
            _params_table(unit.operation, spec), priority=9))
        sections.append(Section(
            "Connections",
            f"Inlets: {', '.join(unit.inlet_names) or '(none)'}\n"
            f"Outlets: {', '.join(unit.outlet_names) or '(none)'}",
            priority=7))
    elif spec is not None:
        rows = [f"- {p.name}: {p.type}"
                + (f" [{p.units}]" if p.units else "")
                + ("  (required)" if p.required else f"  (default {p.default})")
                for p in spec.parameters]
        sections.append(Section(f"Parameters of {operation}",
                                "\n".join(rows) or "(none)", priority=9))

    if spec is not None:
        if spec.equations:
            sections.append(Section(
                "Equations (LaTeX)", "\n".join(spec.equations), priority=8))
        if spec.assumptions:
            sections.append(Section(
                "Assumptions",
                "\n".join(f"- {a}" for a in spec.assumptions), priority=6))
        if spec.numerical_method:
            sections.append(Section("Numerical method",
                                    spec.numerical_method, priority=6))
        if spec.doc:
            # The docstring source, not the HTML the inspector renders
            # from it: markup is tokens spent on angle brackets.
            sections.append(Section("Docstring", spec.doc, priority=4))
        if spec.references:
            sections.append(Section(
                "References",
                "\n".join(f"- {r}" for r in spec.references), priority=2))

    _documentation(f"{operation} {question}", sections)
    kept, notes = fit(sections)
    return Pack("block", question, kept, notes)


def flowsheet(session, question: str = "") -> Pack:
    """The whole model: what is in it, how it is wired, and its code."""
    from difflow.publish import _topology

    fs = session.flowsheet
    if fs is None:
        return Pack("flowsheet", question,
                    [Section("No flowsheet", "Nothing is loaded.", priority=9)])

    topology = _topology(fs) or {"units": [], "feeds": {}, "recycles": {}}
    units = "\n".join(
        f"- {u['name']} ({u['operation']}): "
        f"{', '.join(u['inlets']) or '-'} -> {', '.join(u['outlets']) or '-'}"
        for u in topology["units"]) or "(no units)"
    feeds = "\n".join(
        f"- {name}: " + ", ".join(f"{k}={_fmt_value(v)}"
                                  for k, v in stream.items())
        for name, stream in topology["feeds"].items()) or "(no feeds)"
    recycles = "\n".join(f"- {src} is recycled to {dest}"
                         for src, dest in topology["recycles"].items())

    sections = [
        Section("Units", units, priority=9),
        Section("Feeds", feeds, priority=7),
        Section("Recycles",
                recycles or "(none -- this flowsheet is acyclic and solves "
                            "in one pass)", priority=8),
    ]
    if getattr(fs, "species_order", None):
        sections.append(Section("Species", ", ".join(fs.species_order),
                                priority=6))

    code = session.code()
    if code.get("source"):
        sections.append(Section(
            "The same flowsheet as Python (difflow.codegen)",
            f"```python\n{code['source']}\n```", priority=5))

    subject = " ".join(sorted({u["operation"] for u in topology["units"]}))
    _documentation(f"flowsheet {subject} {question}", sections)
    kept, notes = fit(sections)
    return Pack("flowsheet", question, kept, notes)


def solve(session, question: str = "") -> Pack:
    """How the last solve went, and what usually explains it.

    Every number here was already recorded on ``Flowsheet`` and none of
    it was ever reported: a recycle that stopped at ``max_iter`` with a
    residual of 1e-3 returns streams that look like an answer, and
    ``converged`` is the only thing that says otherwise.
    """
    fs = session.flowsheet
    if fs is None:
        return Pack("solve", question,
                    [Section("No flowsheet", "Nothing is loaded.", priority=9)])

    error = getattr(session, "solve_error", None)
    rows = {
        "converged": getattr(fs, "last_solve_converged", None),
        "method": getattr(fs, "last_solve_method", None),
        "iterations": getattr(fs, "last_solve_iterations", None),
        "residual": getattr(fs, "last_solve_residual", None),
        "tol": getattr(fs, "last_solve_tol", None),
        "tear streams": ", ".join(
            getattr(fs, "last_solve_tear_streams", None) or []) or None,
    }
    if all(value is None for value in rows.values()) and error is None:
        diagnostics = ("This flowsheet has not been solved in this session, "
                       "so there are no diagnostics to read.")
    else:
        diagnostics = "\n".join(
            f"- {key}: {_fmt_value(value)}"
            for key, value in rows.items() if value is not None)

    sections = [Section("The last solve", diagnostics, priority=9)]
    if error:
        sections.insert(0, Section("It raised", error, priority=9))

    # "converged" is a trigger word, so it goes into the haystack only
    # when the answer is no -- a solve that worked must not pull up the
    # card about a solve that did not.
    haystack = " ".join(str(v) for v in (
        error, question, rows["residual"],
        "not converged" if rows["converged"] is False else ""))
    for title, triggers, text in TROUBLESHOOTING:
        if any(trigger.lower() in haystack.lower() for trigger in triggers):
            sections.append(Section(f"Troubleshooting: {title}", text,
                                    priority=8))

    topology = flowsheet(session, "")
    for section in topology.sections:
        if section.title in ("Units", "Recycles"):
            sections.append(Section(section.title, section.body, priority=6))

    _documentation(f"recycle tear convergence solver "
                   f"{rows['method'] or ''} {question}", sections)
    kept, notes = fit(sections)
    return Pack("solve", question, kept, notes)


def planning(dvs, question: str = "", lp_model=None, findings=None) -> Pack:
    """A delta-vector export, read out as prose.

    Takes the :class:`~difflow.planning.export.DeltaVectorSet` rather
    than a session, so it is usable from a script and from the Planning
    panel alike --- and so it is testable against
    ``chain.two_plant_chain()`` with no GUI in the picture.
    """
    sections: list[Section] = []
    for vector in dvs.vectors:
        # The block prefix is on every name and is already in the
        # heading; dropping it is the difference between a table that
        # fits and a table of truncated identifiers.
        strip = len(vector.block) + 1
        levers = [n[strip:] if n.startswith(f"{vector.block}.") else n
                  for n in vector.u_names]
        outs = [n[strip:] if n.startswith(f"{vector.block}.") else n
                for n in vector.y_names]
        width = max([12] + [len(n) + 2 for n in levers])
        label = max([8] + [len(n) for n in outs])
        rows = ["".ljust(label) + "".join(u.rjust(width) for u in levers)]
        for i, name in enumerate(outs):
            rows.append(name.ljust(label)
                        + "".join(f"{value:>{width}.4g}" for value in vector.J[i]))
        sections.append(Section(
            f"Delta vectors for block {vector.block!r} "
            f"(d(output)/d(lever), {vector.mode} mode)",
            "```\n" + "\n".join(rows) + "\n```\n"
            "Base case: " + ", ".join(
                f"{n}={_fmt_value(v)}" for n, v in zip(levers, vector.u0))
            + f"\nValid within a trust radius of {vector.radius:g} of the "
              "lever range; outside it the linear model is not the flowsheet.",
            priority=9))

    if dvs.prices:
        sections.append(Section("Prices", "\n".join(
            f"- {name}: {_fmt_value(price)}"
            for name, price in dvs.prices.items()), priority=7))
    if dvs.specs:
        sections.append(Section("Constraints", "\n".join(
            _spec_line(spec) for spec in dvs.specs), priority=7))
    if dvs.duals:
        # Only the rows with a nonzero marginal. A bound multiplier of
        # zero says the bound is slack, and forty of those is the whole
        # budget spent saying nothing binds.
        binding = [f"- {group}/{row}: {_fmt_value(value)}"
                   for group, rows_ in dvs.duals.items()
                   for row, value in rows_.items() if value]
        if binding:
            sections.append(Section(
                "Shadow prices (nonzero marginals only)",
                "\n".join(binding[:14]), priority=6))

    for finding in (findings if findings is not None else dvs.health) or []:
        as_dict = finding if isinstance(finding, dict) else vars(finding)
        sections.append(Section(
            f"Health finding: {as_dict.get('kind', 'issue')}",
            str(as_dict.get('detail') or as_dict), priority=8))

    if lp_model is not None:
        sections.append(Section("The assembled LP",
                                "```\n" + lp_model.as_text() + "\n```",
                                priority=4))

    _documentation(question or "delta vectors trust region planning",
                   sections)
    kept, notes = fit(sections)
    return Pack("planning", question, kept, notes)


#: The briefs a caller can ask for by name.
KINDS = ("block", "flowsheet", "solve", "planning")


def pack(session, kind: str = "flowsheet", question: str = "",
         name: str | None = None, operation: str | None = None) -> dict:
    """One brief, by name, as the dict the ``/api/context`` route sends.

    A bad ``kind`` is an answer (``{"ok": False, ...}``) rather than an
    exception, like everything else the session returns.
    """
    if kind == "block":
        return block(session, name=name, operation=operation,
                     question=question).to_dict()
    if kind == "flowsheet":
        return flowsheet(session, question).to_dict()
    if kind == "solve":
        return solve(session, question).to_dict()
    if kind == "planning":
        dvs = getattr(session, "delta_vectors", None)
        if dvs is None:
            return {"ok": False,
                    "error": "no linearization yet -- pick levers and outputs "
                             "in the Planning panel first"}
        return planning(dvs, question).to_dict()
    return {"ok": False,
            "error": f"unknown context kind {kind!r}; expected one of "
                     f"{', '.join(KINDS)}"}
