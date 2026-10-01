"""The crude unit's numbers from before #301, reproduced by `twu_legacy`.

#301 put the crude and vacuum units on one characterization. The new
machinery (heavy-end extension, a residue lump, contaminants per cut) is
opt-in, so an assay that uses none of it has to come out of `characterize`
and `CrudeUnit.solve` exactly as it did before -- given the same correlation.
#301 also changed the default correlation: `"twu"` is now Twu (1984) as
published, and the old coding, which under-corrects aromatics' molecular
weight, is `"twu_legacy"`. Run under that name, the merged correlation
module and the volatility continuation must give the old numbers to
round-off.

`data/cdu_baseline.json` was written from main at 8829a11, before any #301
change, by characterizing the test assay under every critical-property
method (plus a mass-basis assay with no light ends) and solving the
`test_unit` crude unit. The tolerances are round-off, not engineering ones:
a merge of the correlation modules that moved a digit would show here.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay, characterize
from difflow_refinery import column as cc
from difflow_refinery.thermo import ColumnThermo

from .test_unit import ASSAY, BPD, _atmospheric_params

jax.config.update("jax_enable_x64", True)

BASE = json.loads((Path(__file__).parent / "data" / "cdu_baseline.json").read_text())
FIELDS = ["cut_edges", "Tb", "SG", "MW", "Tc", "Pc", "omega", "omega_vp", "Kw", "hvap_nb",
          "component_MW", "component_SG", "volume_fraction", "mass_fraction", "mole_fraction"]
MASS_ASSAY = Assay([0, 10, 30, 50, 70, 90, 100], [300, 380, 480, 580, 700, 850, 1000],
                   basis="mass", api=30.0)


def _check(char, ref):
    for f in FIELDS:
        np.testing.assert_allclose(np.asarray(getattr(char, f)), ref[f], rtol=1e-10, atol=1e-13,
                                   err_msg=f)


# the baseline file's "twu" is today's "twu_legacy"
BASELINE_NAME = {"twu_legacy": "twu"}


@pytest.mark.parametrize("method", ["twu_legacy", "riazi_daubert_1980", "riazi_daubert_1987",
                                    "lee_kesler"])
def test_characterization_is_unchanged(method):
    char = characterize(ASSAY, method=method)
    ref = BASE[f"char_{BASELINE_NAME.get(method, method)}"]
    assert list(char.names) == ref["names"]
    _check(char, ref)


def test_mass_basis_characterization_is_unchanged():
    _check(characterize(MASS_ASSAY, method="twu_legacy"), BASE["char_mass"])


def test_vapor_pressures_are_unchanged():
    th = ColumnThermo.from_characterization(characterize(ASSAY, method="twu_legacy"))
    np.testing.assert_allclose(np.asarray(th.psat(500.0)), BASE["thermo_psat_500K"], rtol=1e-10)


def test_crude_unit_solve_is_unchanged():
    crude = characterize(ASSAY, method="twu_legacy")
    th = ColumnThermo.from_characterization(crude)
    kg_s = BPD * cc.BARREL / 86400.0 * float(crude.bulk_sg) * 999.016
    feed = crude.stream(kg_s, T=600.0, P=1.9e5, basis="mass")
    base = _atmospheric_params(feed, th, pa1_duty=15e6)
    params = _atmospheric_params(feed, th, pa1_duty=15e6, specs=base.specs + (cc.overflash(0.05),))
    r = dr.CrudeUnit(ASSAY, params, method="twu_legacy").solve(BPD, T=273.15 + 240.0, P=6e5)
    assert bool(r.converged)
    for name, ref in BASE["solve"].items():
        if name == "converged":
            continue
        p = r.properties[name]
        assert float(p.yield_volume) == pytest.approx(ref["yield_volume"], rel=1e-8, abs=1e-10)
        assert float(p.sg) == pytest.approx(ref["sg"], rel=1e-8)
        assert float(p.tbp_at(95)) == pytest.approx(ref["T95"], rel=1e-8)
