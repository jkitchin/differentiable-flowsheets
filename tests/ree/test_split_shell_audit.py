"""SplitShellCascade: solvent REE, split validation, product groups.

Conservation audit (a): REE on the inlet solvent were ignored and the
closure still read 1.0 (0.2 mol/s each of La/Ce/Pr/Nd on the solvent: in
1.2, out 1.0); split points (4, 4, 20) on 12 stages ran 22 stages and
reported a stage range (20, 12). Operating-point audit (R7):
``product_groups`` was stored and never read, so every section ran at one pH
and the cascade made one useful split.
"""

import warnings

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.flowsheets.modules import SplitShellModule
from difflow_ree.flowsheets.split_shell import SplitShellCascade, SplitShellParams

ELS = ("La", "Ce", "Pr", "Nd")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _streams(solvent_ree=0.2):
    feed = make_stream({"H2O": 100.0, "HNO3": 2.0, **{e: 1.0 for e in ELS}},
                       298.15, 101325.0)
    solvent = make_stream({"D2EHPA": 50.0, "kerosene": 100.0,
                           **{e: solvent_ree for e in ELS}}, 298.15, 101325.0)
    return feed, solvent


def test_solvent_ree_is_carried_and_counted():
    feed, solvent = _streams()
    params = SplitShellParams(extractant="D2EHPA", elements=ELS, n_stages=12,
                              split_points=(4, 8))
    r = SplitShellCascade(params)(feed, solvent)
    for e in ELS:
        out = sum(float(prod["flows"][e]) for prod in r["products"].values())
        assert out == pytest.approx(1.2, rel=1e-12)
        assert float(r["mass_balance"]["feed"][e]) == pytest.approx(1.2)
        assert float(r["mass_balance"]["closure"][e]) == pytest.approx(1.0, rel=1e-12)
    # and the module's streams close on feed + solvent too
    outs = SplitShellModule("ss", params)(feed, solvent)[:-1]
    for e in ELS:
        assert sum(float(get_flows(s).get(e, 0.0)) for s in outs) == \
            pytest.approx(1.2, rel=1e-12)


@pytest.mark.parametrize("splits", [(4, 4), (4, 20), (8, 4), (0, 6), (12,)])
def test_bad_split_points_are_rejected(splits):
    with pytest.raises(ValueError, match="strictly increasing"):
        SplitShellParams(extractant="D2EHPA", elements=ELS, n_stages=12,
                         split_points=splits)


class TestProductGroups:
    COMP = dict(La=0.025, Ce=0.045, Pr=0.005, Nd=0.015, Sm=0.002, Eu=0.0005,
                Gd=0.001, Tb=0.0002, Dy=0.0005, Y=0.002)
    GROUPS = {"heavy": ("Gd", "Tb", "Dy", "Y"), "middle": ("Sm", "Eu"),
              "light": ("La", "Ce", "Pr", "Nd")}

    def _run(self, ext):
        els = tuple(self.COMP)
        params = SplitShellParams(extractant=ext, elements=els, n_stages=20,
                                  split_points=(10,), product_groups=self.GROUPS)
        feed = make_stream({"H2O": 55.5, **self.COMP}, 298.15, 101325.0)
        solvent = make_stream({"kerosene": 55.5, ext: 27.75}, 298.15, 101325.0)
        return SplitShellCascade(params)(feed, solvent)

    @pytest.mark.parametrize("ext", ["D2EHPA", "PC88A"])
    def test_each_group_has_its_own_outlet(self, ext):
        r = self._run(ext)
        pHs = r["section_pHs"]
        assert pHs[0] < pHs[1]  # the heavy cut needs more acid
        frac = {name: {e: float(prod["flows"][e]) / self.COMP[e]
                       for e in self.COMP}
                for name, prod in r["products"].items()}
        for e in self.GROUPS["heavy"]:
            assert frac["product_1"][e] > 0.95, e
        assert frac["product_2"]["Sm"] > 0.5
        for e in ("La", "Ce", "Pr"):
            assert frac["raffinate"][e] > 0.75, e

    def test_one_pH_was_one_split(self):
        """Without groups every section shares the pH: the old behaviour."""
        els = tuple(self.COMP)
        p = SplitShellParams(extractant="D2EHPA", elements=els, n_stages=20,
                             split_points=(10,))
        feed = make_stream({"H2O": 55.5, **self.COMP}, 298.15, 101325.0)
        solvent = make_stream({"kerosene": 55.5, "D2EHPA": 27.75}, 298.15, 101325.0)
        r = SplitShellCascade(p)(feed, solvent)
        assert r["section_pHs"][0] == r["section_pHs"][1] == p.pH

    def test_group_count_must_match_the_sections(self):
        with pytest.raises(ValueError, match="groups for"):
            SplitShellParams(extractant="D2EHPA", elements=tuple(self.COMP),
                             n_stages=20, split_points=(5, 10, 15),
                             product_groups=self.GROUPS)
