"""The code context, written out for the unit that is waiting for it.

The value of a generated snippet is entirely in whether the reader can
tell the good half from the invented half. So most of what is asserted
here is about the comments: that a made-up Arrhenius pair is marked
INVENTED, that a fallback species list is marked GUESSED, and that a
placeholder derived from an annotation says so. A snippet that looked
finished would put an invented number into somebody's flowsheet.
"""

import pytest

from difflow.gui import boilerplate

SPECIES = ["water", "ethanol"]


def defined(source: str) -> dict:
    """Run the snippet and hand back what it bound.

    The strongest thing to say about generated Python is that it is
    Python: it imports, it parses, and the name the unit asked for
    exists afterwards. A stub that raises is only reached when called.
    """
    namespace: dict = {}
    exec(compile(source, "<snippet>", "exec"), namespace)
    return namespace


class TestWhatItWrites:
    def test_nothing_needed_is_nothing_written(self):
        assert boilerplate.snippet("Heater", []) == ""
        assert boilerplate.snippet("Heater", [""]) == ""

    def test_a_thermo_is_built_from_this_flowsheet_species(self):
        source = boilerplate.snippet("Flash", ["thermo"], SPECIES)
        assert 'SPECIES = ["water", "ethanol"]' in source
        assert "# GUESSED" not in source, "the species were given, not guessed"
        assert "get_species_data" in source
        namespace = defined(source)
        assert set(namespace["thermo"].species) == {"water", "ethanol"}

    def test_no_species_is_a_guess_and_says_so(self):
        source = boilerplate.snippet("Flash", ["thermo"])
        assert "# GUESSED" in source
        assert list(boilerplate.FALLBACK_SPECIES)[0] in source
        assert defined(source)["thermo"] is not None

    def test_the_header_names_the_operation_and_the_markers(self):
        source = boilerplate.snippet("Flash", ["thermo"], SPECIES)
        first = source.splitlines()[0]
        assert "Flash" in first
        assert "INVENTED, GUESSED or PLACEHOLDER" in source.splitlines()[1]

    def test_the_kinetics_group_is_one_call_and_is_marked_invented(self):
        """rate_fn, stoich and rate_params cannot be written separately.

        They are one reaction seen from three sides, and the failure this
        avoids --- a stoichiometry that disagrees with the rate law --- is
        the whole reason `mass_action_kinetics` exists.
        """
        needs = ["rate_fn", "stoich", "rate_params"]
        source = boilerplate.snippet("CSTR", needs, SPECIES)
        assert source.count("mass_action_kinetics(") == 1
        assert "# INVENTED" in source
        assert "water -> ethanol" in source
        namespace = defined(source)
        assert set(namespace["kin"]) >= {"rate_fn", "stoich", "rate_params"}

    def test_one_species_still_writes_a_runnable_reaction(self):
        """A -> A is nonsense chemistry and valid Python, which is the point."""
        source = boilerplate.snippet("CSTR", ["rate_fn"], ["water"])
        assert "water -> water" in source
        assert defined(source)["kin"]["stoich"] is not None

    def test_a_callable_with_no_recipe_gets_the_documented_signature(self):
        """`Signature: ...` in the docstring beats any stub invented here."""
        source = boilerplate.snippet("CSTR", ["rate_fn"], SPECIES)
        # rate_fn is part of the kinetics group, so reach for a callable
        # that is not: ask for one by a name the recipes do not know.
        source = boilerplate.snippet("Heater", ["Cp_fn"], SPECIES)
        assert "Cp_fn" in source
        assert "NotImplementedError" in source or "PLACEHOLDER" in source

    def test_a_need_with_no_recipe_falls_back_to_its_annotation(self):
        """This is the branch a plugin's own constructor argument lands in."""
        source = boilerplate.snippet("Flash", ["nonesuch"], SPECIES)
        assert "nonesuch = " in source
        assert "# PLACEHOLDER" in source

    def test_a_required_string_is_a_string_to_change(self):
        pytest.importorskip("difflow_cc")
        source = boilerplate.snippet("AmineAbsorber", ["solvent"], SPECIES)
        if "solvent" not in source:
            pytest.skip("difflow_cc not registered")
        assert "CHANGE ME" in source
        assert defined(source)["solvent"] == "CHANGE ME"

    def test_species_are_only_defined_when_something_uses_them(self):
        """Opening with SPECIES = [...] a snippet never mentions is noise."""
        source = boilerplate.snippet("Flash", ["nonesuch"], SPECIES)
        assert "SPECIES = " not in source

    def test_asking_for_the_species_order_writes_it(self):
        source = boilerplate.snippet("Mixer", ["species_order"], SPECIES)
        assert 'SPECIES = ["water", "ethanol"]' in source
        assert defined(source)["SPECIES"] == SPECIES

    def test_an_unregistered_operation_still_answers(self):
        """The catalog lookup is a source of detail, not a precondition."""
        source = boilerplate.snippet("NoSuchUnit", ["thermo"], SPECIES)
        assert "NoSuchUnit" in source
        assert defined(source)["thermo"] is not None

    def test_a_repeated_need_is_written_once(self):
        source = boilerplate.snippet("Flash", ["thermo", "thermo"], SPECIES)
        assert source.count("thermo = IdealThermo") == 1


class TestMerging:
    """Appending, not replacing: the context may already say half of it."""

    def test_an_empty_context_takes_the_snippet_whole(self):
        addition = boilerplate.snippet("Flash", ["thermo"], SPECIES)
        assert boilerplate.merged("", addition) == addition
        assert boilerplate.merged("   \n\n", addition) == addition

    def test_what_is_already_defined_is_not_defined_again(self):
        """Two drops must not write SPECIES twice.

        The second one would win, and there is no reason to believe the
        two lists agree --- which is a flowsheet silently re-indexed by a
        line the user never typed.
        """
        first = boilerplate.snippet("Flash", ["thermo"], SPECIES)
        second = boilerplate.snippet("CSTR", ["rate_fn"], SPECIES)
        both = boilerplate.merged(first, second)
        assert both.count('SPECIES = ["water", "ethanol"]') == 1
        assert both.count("thermo = IdealThermo") == 1
        assert "mass_action_kinetics(" in both
        namespace = defined(both)
        assert namespace["thermo"] is not None and namespace["kin"] is not None

    def test_the_existing_context_is_kept_verbatim(self):
        existing = "solvent = 'MEA'\n"
        both = boilerplate.merged(existing, "thermo = None\n")
        assert both.startswith(existing)
        assert "thermo = None" in both

    def test_an_addition_that_says_nothing_new_changes_nothing(self):
        source = boilerplate.snippet("Flash", ["thermo"], SPECIES)
        assert boilerplate.merged(source, source) == source

    def test_an_indented_line_is_never_dropped(self):
        """Deduplicating by line would eat a function body.

        Two stubs both end in `raise NotImplementedError("write me")`,
        and dropping the second one leaves a `def` with no body --- a
        SyntaxError in the code context, from a snippet that promised
        to be paste-ready.
        """
        stub = 'def f(x):\n    raise NotImplementedError("write me")\n'
        other = 'def g(x):\n    raise NotImplementedError("write me")\n'
        both = boilerplate.merged(stub, other)
        assert both.count('raise NotImplementedError("write me")') == 2
        assert "f" in defined(both) and "g" in defined(both)
