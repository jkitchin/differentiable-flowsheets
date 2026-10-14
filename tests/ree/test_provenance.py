"""Tests for the data provenance system.

The point of these tests is not to check that a lookup function works. It is to
make the provenance block a maintained obligation rather than a comment that
rots: adding a field to a data file without tagging it FAILS CI, and so does
citing a source key that is not in the registry.
"""

import subprocess
import sys

import pytest

pytest.importorskip("difflow_ree")

from difflow_ree.provenance import (  # noqa: E402
    CLASS_ORDER,
    DATASETS,
    UNPUBLISHABLE,
    UNTAGGED,
    audit,
    coverage,
    explain,
    load_rules,
    load_sources,
    untagged,
)


class TestRegistry:
    """sources.yaml is well formed."""

    def test_every_source_has_citation_and_class(self):
        for key, entry in load_sources().items():
            assert entry.get("citation"), f"{key} has no citation"
            assert entry.get("cls"), f"{key} has no cls"

    def test_every_class_is_known(self):
        for key, entry in load_sources().items():
            assert entry["cls"] in CLASS_ORDER, (
                f"{key} has cls={entry['cls']!r}, which is not in CLASS_ORDER. "
                "Add it there so reports can order it."
            )

    @pytest.mark.parametrize("dataset", sorted(DATASETS))
    def test_every_rule_cites_a_registered_source(self, dataset):
        sources = load_sources()
        for pattern, rule in load_rules(dataset):
            assert rule["source"] in sources, (
                f"{dataset}: rule {pattern!r} cites {rule['source']!r}, "
                "which is not in sources.yaml"
            )


class TestCompleteness:
    """Every number is accounted for. This is the test that earns its keep."""

    @pytest.mark.parametrize("dataset", sorted(DATASETS))
    def test_no_untagged_fields(self, dataset):
        missing = untagged(dataset)
        assert not missing, (
            f"{len(missing)} field(s) in {DATASETS[dataset]} match no "
            "provenance rule. Add one to the `provenance:` block at the foot "
            "of that file -- an untagged number is indistinguishable from an "
            "invented one.\n"
            + "\n".join(f"    {p.path}" for p in missing[:20])
        )

    def test_untagged_sentinel_is_never_written_in_a_file(self):
        """UNTAGGED is synthesized by the resolver, not authored."""
        for dataset in DATASETS:
            for pattern, rule in load_rules(dataset):
                assert rule["source"] != UNTAGGED, (
                    f"{dataset}: {pattern!r} tags a field UNTAGGED explicitly. "
                    "Use HAND_TUNED if it has no basis; UNTAGGED means the "
                    "block is incomplete."
                )

    def test_audit_covers_every_leaf(self):
        """audit() and the raw YAML walk agree on how many numbers exist."""
        import yaml

        from difflow_ree.provenance import DATA_DIR, PROVENANCE_KEY, _walk

        for dataset, fname in DATASETS.items():
            with open(DATA_DIR / fname) as f:
                data = yaml.safe_load(f)
            body = {k: v for k, v in data.items() if k != PROVENANCE_KEY}
            assert len(audit(dataset)) == len(list(_walk(body)))


class TestSpecificity:
    """The most specific rule wins, or the catch-alls swallow everything."""

    def test_specific_rule_beats_catch_all(self):
        # extractants.*.*.*.* is HAND_TUNED; the TBP nitrate rule must win.
        p = explain("extractants", "extractants.TBP.nitrate_coefficients.Nd.a")
        assert p.source == "K05"
        assert p.cls == "MEASURED"

    def test_catch_all_still_applies_where_nothing_is_specific(self):
        """The default is fail-closed: an untagged field reads HAND_TUNED.

        This used to point at `D2EHPA.ph_coefficients.Nd.a`, which was a live
        example until #270 gave that block a source. Every field in
        extractants.yaml is now matched by a rule of its own, so the only way
        to exercise the fall-through is to resolve a path that is not in the
        file -- which is exactly the case that matters: a field ADDED
        tomorrow, by someone who does not also add a rule, must come out
        unpublishable rather than inheriting a neighbour's citation.
        """
        from difflow_ree.provenance import _resolve

        p = _resolve("extractants", "extractants.D2EHPA.a_field_nobody_tagged", 1.0)
        assert p.cls == "HAND_TUNED"
        assert not p.publishable
        for depth in (
            "extractants.D2EHPA.new_block.field",
            "extractants.D2EHPA.new_block.Nd.field",
        ):
            assert _resolve("extractants", depth, 1.0).cls == "HAND_TUNED", depth

    def test_unknown_path_raises(self):
        with pytest.raises(KeyError):
            explain("elements", "elements.Nd.no_such_field")

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            explain("not_a_dataset", "anything")

    def test_top_level_key_may_be_elided(self):
        assert (
            explain("elements", "Dy.ionic_radius_pm").value
            == explain("elements", "elements.Dy.ionic_radius_pm").value
        )


class TestTheClaimsTheDataFilesMake:
    """Pin the statements the file headers make, so they cannot drift."""

    def test_naphthenic_a_values_are_measured_and_come_from_Z1(self):
        """extractants.yaml says Z1 Table 4.36 is their ONLY source."""
        from difflow_ree.database import get_extractant_database

        elements = get_extractant_database().get("naphthenic_acid").ph_coefficients
        for sym in elements:
            p = explain(
                "extractants",
                f"extractants.naphthenic_acid.ph_coefficients.{sym}.a",
            )
            assert p.source == "Z1", sym
            assert p.publishable, sym
            assert "4.36" in p.locus

    def test_no_acidic_extractant_ships_a_hand_tuned_pH_coefficient(self):
        """The header's central warning, inverted -- #270 finished the job.

        It used to be three records, then two: PC88A left the list when its
        `a` values were fitted to Tanaka (2021)'s 121 digitized points, and
        D2EHPA and Cyanex272 left it when they were refitted against named
        tables (#270; D2EHPA again in #283, against Mason 1976 and Peppard
        1957). Every `a` is now MEASURED or DERIVED from a named source, and
        every `b` is the stoichiometric 3 that the source's own mass-action
        model writes down.
        """
        for ex in ("D2EHPA", "PC88A", "Cyanex272", "naphthenic_acid"):
            for sym in ("La", "Nd", "Dy", "Y"):
                for c in ("a", "b"):
                    p = explain(
                        "extractants", f"extractants.{ex}.ph_coefficients.{sym}.{c}"
                    )
                    assert p.cls in ("MEASURED", "DERIVED"), f"{ex} {sym} {c}: {p.cls}"
                    assert p.publishable, f"{ex} {sym} {c}"
                    assert p.source != "HAND_TUNED", f"{ex} {sym} {c}"

    def test_the_only_hand_tuned_numbers_left_are_the_temperature_blocks(self):
        """What #270 could NOT retire, stated as an assertion.

        Every source behind the three refits is isothermal -- X95 and PPH63
        are room-temperature determinations, T21 is 298.0 +/- 0.1 K, ZL93 is
        25 +/- 1 C -- so none of them carries a heat of extraction. Thirty
        numbers, ten per record, and nothing else in the file -- twenty since
        #283 retired D2EHPA's against X95's 10-50 C table (and flipped their
        sign). D at 298 K is sourced for all five reagents; D at any other
        temperature for three.
        """
        hand_tuned = [p for p in audit("extractants") if p.cls == "HAND_TUNED"]
        # 20 since #283 sourced D2EHPA's from X95 (Q1 Table 2.15)
        assert len(hand_tuned) == 20
        for p in hand_tuned:
            _, ex, block, _ = p.path.split(".")
            assert block == "temperature_coefficients", p.path
            assert ex in ("PC88A", "Cyanex272"), p.path
            assert "not retirable by it" in p.note, p.path

    def test_every_concentration_exponent_names_a_source(self):
        """The field the #270 retagging exposed, on all five records.

        `concentration_exponent` was matched by nothing and so read
        HAND_TUNED on every record, including the two -- TBP and
        naphthenic_acid -- that the file's header held up as the ones with
        real provenance. Two are FITTED: PC88A's (T21's six series, 0.041 to
        0.82 mol/L as dimer) and, since #283, D2EHPA's (Mason 1976's three
        concentration series in n-heptane, 2.38). The other three are the
        mechanism's cube law imposed, and their notes say so.
        """
        expected = {
            "D2EHPA": ("M76", 2.38),
            "PC88A": ("T21", 3.0),
            "Cyanex272": ("Z1", 3.0),
            "TBP": ("GP19", 3.0),
            "naphthenic_acid": ("Q1", 3.0),
        }
        for ex, (src, value) in expected.items():
            p = explain("extractants", f"extractants.{ex}.concentration_exponent")
            assert p.source == src, ex
            assert p.publishable, ex
            assert p.value == value, ex
        # Two are FITTED: PC88A as a cube law in T21's effective
        # concentration, and since #283 D2EHPA as the power Mason (1976)
        # measures in n-heptane, which is NOT the stoichiometric cube.
        for ex in ("PC88A", "D2EHPA"):
            fitted = explain("extractants", f"extractants.{ex}.concentration_exponent")
            assert "FITTED" in fitted.note, ex
        for ex in ("Cyanex272", "TBP", "naphthenic_acid"):
            note = explain(
                "extractants", f"extractants.{ex}.concentration_exponent"
            ).note
            assert any(
                k in note
                for k in ("Not fitted", "imposed", "not set by measurement")
            ), ex

    def test_PC88A_pH_coefficients_are_measured(self):
        """The other half of #270: what replaced the hand-tuned block.

        `a` is fitted (MEASURED, T21). `b` is pinned at the stoichiometric 3
        rather than fitted, which is what makes every separation factor on
        this record exactly pH-independent -- so it is T21's too, but as the
        model form the paper itself fits with, not as a free parameter.
        `c` and `d` are the CONVENTION zeros of an isothermal, quadratic-free
        fit. Five elements are interpolated across the series and say so.
        """
        measured = ("La", "Nd", "Sm", "Dy", "Y")
        interpolated = ("Ce", "Pr", "Eu", "Gd", "Tb")
        for sym in measured:
            p = explain("extractants", f"extractants.PC88A.ph_coefficients.{sym}.a")
            assert p.cls == "MEASURED", sym
            assert p.source == "T21", sym
            assert p.publishable, sym
        for sym in interpolated:
            p = explain("extractants", f"extractants.PC88A.ph_coefficients.{sym}.a")
            assert p.cls == "DERIVED", sym
            assert p.source == "CALC", sym
        for sym in measured + interpolated:
            b = explain("extractants", f"extractants.PC88A.ph_coefficients.{sym}.b")
            assert b.source == "T21", sym
            for z in ("c", "d"):
                q = explain(
                    "extractants", f"extractants.PC88A.ph_coefficients.{sym}.{z}"
                )
                assert q.cls == "CONVENTION", f"{sym} {z}"

    def test_six_prices_are_sourced_and_the_rest_are_not(self):
        """elements.yaml's header, as an assertion.

        Every price used to be unsourced. #270 gave six of them the USGS
        2025 annual average oxide price; the other nine are untouched and
        still say so, and every extractant cost is still invented.

        The point of naming the six is that the default stays "no source":
        the rule that grants USGS26 lists each element literally, so adding
        an element to this file cannot silently inherit a citation.
        """
        sourced = {"La", "Ce", "Nd", "Pr", "Eu", "Gd"}
        seen = set()
        for p in audit("elements"):
            if not p.path.endswith(".price_usd_kg"):
                continue
            sym = p.path.split(".")[1]
            seen.add(sym)
            if sym in sourced:
                assert p.cls == "REFERENCE", p.path
                assert p.source == "USGS26", p.path
                assert p.publishable, p.path
            else:
                assert p.cls == "ESTIMATED", p.path
        assert sourced <= seen
        for p in audit("extractants"):
            if p.path.endswith(".cost_usd_kg") or p.path.endswith(".cost_usd_L"):
                assert p.cls == "ESTIMATED", p.path

    def test_nd_and_pr_carry_the_same_pair_price(self):
        """USGS quotes didymium as one product, so difflow_ree cannot split it.

        A circuit that separates Pr from Nd earns nothing for having done so
        at these prices. That is the market, not a modelling shortcut, and
        the note on the rule says so -- but a test is what stops somebody
        "fixing" it by inventing a spread.
        """
        from difflow_ree.database import get_element

        assert get_element("Nd").price_usd_kg == get_element("Pr").price_usd_kg
        for sym in ("Nd", "Pr"):
            p = explain("elements", f"{sym}.price_usd_kg")
            assert "NdPr" in p.locus

    def test_separation_factors_carry_no_measurement(self):
        for p in audit("separation_factors"):
            assert not p.publishable or p.cls in ("CONVENTION", "DERIVED"), p.path

    def test_publishable_matches_the_class_list(self):
        for p in audit():
            assert p.publishable == (p.cls not in UNPUBLISHABLE), p.path


class TestReport:
    def test_coverage_sums_to_the_audit(self):
        assert sum(coverage().values()) == len(audit())

    def test_coverage_is_ordered_worst_first(self):
        keys = list(coverage())
        ranks = [CLASS_ORDER.index(k) for k in keys if k in CLASS_ORDER]
        assert ranks == sorted(ranks)

    def test_cli_runs(self):
        out = subprocess.run(
            [sys.executable, "-m", "difflow_ree.provenance"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "difflow_ree data provenance" in out.stdout
        assert "UNTAGGED" not in out.stdout  # nothing should be untagged

    def test_cli_explain(self):
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "difflow_ree.provenance",
                "--dataset",
                "elements",
                "--explain",
                "Dy.ionic_radius_pm",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Shannon" in out.stdout
