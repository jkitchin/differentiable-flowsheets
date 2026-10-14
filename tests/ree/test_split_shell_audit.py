"""SplitShellParams validation: split points and product groups."""

import pytest

from difflow_ree.flowsheets.split_shell import SplitShellParams

ELS = ("La", "Ce", "Pr", "Nd")


def test_bad_split_points_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        SplitShellParams(extractant="D2EHPA", elements=ELS, n_stages=12,
                         split_points=(4, 20))


def test_group_count_must_match_the_sections():
    with pytest.raises(ValueError, match="groups for"):
        SplitShellParams(extractant="D2EHPA", elements=ELS, n_stages=20,
                         split_points=(5, 10, 15),
                         product_groups={"a": ("La",), "b": ("Nd",)})
