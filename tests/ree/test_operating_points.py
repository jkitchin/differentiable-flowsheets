"""Operating points: cut_pHs smoke and the naphthenic-acid design crash."""

import math
import warnings

import pytest

from difflow_ree.equilibrium.operating_points import cut_pHs
from difflow_ree.flowsheets.extract_scrub_strip import design_extract_scrub_strip

COMP = dict(La=0.025, Ce=0.045, Pr=0.005, Nd=0.015, Sm=0.002, Eu=0.0005,
            Gd=0.001, Tb=0.0002, Dy=0.0005, Y=0.002)
ELS = tuple(COMP)
HEAVY = ("Gd", "Tb", "Dy", "Y")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def test_cut_pHs_smoke():
    ph = cut_pHs("PC88A", ELS, extraction_OA=1.5, strip_OA=3.0)
    assert math.isfinite(ph.extraction) and math.isfinite(ph.stripping)
    assert cut_pHs("TBP", ELS, extraction_OA=1.5, strip_OA=3.0,
                   nitrate_conc=3.0) is None


def test_naphthenic_acid_design_does_not_raise():
    p = design_extract_scrub_strip(COMP, HEAVY, "naphthenic_acid")
    assert p.extractant == "naphthenic_acid"
