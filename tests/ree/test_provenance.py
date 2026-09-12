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
        p = explain("extractants", "extractants.D2EHPA.ph_coefficients.Nd.a")
        assert p.cls == "HAND_TUNED"

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

    def test_the_three_acidic_extractants_are_hand_tuned(self):
        """The header's central warning, as an assertion."""
        for ex in ("D2EHPA", "PC88A", "Cyanex272"):
            for sym in ("La", "Nd", "Dy", "Y"):
                for c in ("a", "b"):
                    p = explain(
                        "extractants", f"extractants.{ex}.ph_coefficients.{sym}.{c}"
                    )
                    assert p.cls == "HAND_TUNED", f"{ex} {sym} {c}"
                    assert not p.publishable

    def test_every_price_is_unsourced(self):
        """elements.yaml and extractants.yaml both promise this."""
        for p in audit("elements"):
            if p.path.endswith(".price_usd_kg"):
                assert p.cls == "ESTIMATED", p.path
        for p in audit("extractants"):
            if p.path.endswith(".cost_usd_kg") or p.path.endswith(".cost_usd_L"):
                assert p.cls == "ESTIMATED", p.path

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
