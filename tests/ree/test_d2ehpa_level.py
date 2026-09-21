"""D2EHPA's absolute level and series shape (#283).

#270 left the D2EHPA record resting on ONE measured D (X95's La in kerosene)
for its level and on PPH63's chromatographic separation factors for its
shape, and recorded that three routes to the level disagreed by up to 1.7
decades. #283 refit it:

* LEVEL and concentration exponent from Mason (1976), HDEHP in n-heptane --
  a pH series at constant ionic strength and a concentration series, read
  from Q1's redrawn Figs. 2.1 and 2.2 (ree-database
  ``sources/mason1976extraction.json``).
* SHAPE from Peppard et al. (1957), via Stevenson & Nervik (1961) Fig. 51.
* X95's La point held OUT of the fit, as the check.

These tests hold the record to the measurements rather than to its own
numbers, so a later edit that drifts from the data fails here.
"""

import math

import numpy as np
import pytest

from difflow_ree.database import get_extractant
from difflow_ree.equilibrium.distribution import REEDistribution

#: Mason (1976) Y, digitized from the drawing paths of Q1 p. 191:
#: (figure, pH, HDEHP formal mol/L, log10 D). Fig. 2.1 is at 1.0 M
#: (NaCl + HCl); Fig. 2.2 at 0.5 M HCl. Tracer-level throughout.
MASON_Y = [
    ("F21", 1.1516, 0.018750, 1.5823),
    ("F21", 0.9800, 0.018750, 0.9959),
    ("F21", 0.8635, 0.018750, 0.6801),
    ("F21", 0.6926, 0.018750, 0.1282),
    ("F21", 0.5738, 0.018750, -0.2369),
    ("F21", 0.4101, 0.018750, -0.7764),
    ("F21", 0.2874, 0.018750, -1.1399),
    ("F21", 0.1250, 0.018750, -1.6594),
    ("F21", 0.0037, 0.018750, -2.0313),
    ("F22", 0.3010, 0.009349, -1.7138),
    ("F22", 0.3010, 0.013400, -1.2940),
    ("F22", 0.3010, 0.019170, -0.9189),
    ("F22", 0.3010, 0.029006, -0.5284),
    ("F22", 0.3010, 0.042732, -0.1093),
    ("F22", 0.3010, 0.058646, 0.2449),
    ("F22", 0.3010, 0.086214, 0.6082),
    ("F22", 0.3010, 0.123860, 0.9936),
    ("F22", 0.3010, 0.172248, 1.3335),
]

#: Peppard (1957) log D, Stevenson & Nervik Fig. 51 markers (toluene, 0.5 M
#: HCl). Only differences between elements are used.
PEPPARD = {"La": -3.024, "Ce": -2.612, "Pr": -2.25, "Eu": -0.638, "Tb": 0.156,
           "Y": 1.144}


@pytest.fixture(scope="module")
def rec():
    return get_extractant("D2EHPA")


def _log_D(element, pH, dimer_conc):
    dist = REEDistribution(extractant="D2EHPA", elements=(element,),
                           concentration=dimer_conc)
    return float(np.log10(dist.get_D(element, pH)))


def test_the_record_reproduces_masons_measured_y(rec):
    """Every one of Mason's 18 Y points, through the public D path.

    The record is on the dimer basis and Mason's charges are formal, so the
    concentration handed over is half the printed one. The two figures sit
    0.1 apart in their own media and the record splits the difference, so
    the bound is on the rms and on each figure's mean bias.
    """
    resid = {"F21": [], "F22": []}
    for fig, pH, c_formal, log_d in MASON_Y:
        resid[fig].append(_log_D("Y", pH, c_formal / 2.0) - log_d)
    allr = np.array(resid["F21"] + resid["F22"])
    assert np.sqrt(np.mean(allr ** 2)) < 0.1
    for fig, r in resid.items():
        assert abs(np.mean(r)) < 0.08, fig


def test_the_concentration_exponent_is_masons_not_the_cube(rec):
    """2.38, the measured slope of log D on log[HDEHP] in n-heptane.

    Across Mason's Fig. 2.2 series the charge rises 18-fold; the cube law
    the stoichiometry would impose predicts 3.77 decades, the data 2.99.
    """
    assert rec.concentration_exponent == pytest.approx(2.38)
    f22 = [p for p in MASON_Y if p[0] == "F22"]
    slope = np.polyfit([math.log10(p[2]) for p in f22], [p[3] for p in f22], 1)[0]
    assert slope == pytest.approx(rec.concentration_exponent, abs=0.03)
    assert abs(slope - 3.0) > 0.5
    # the capacity still comes from the stoichiometry, not the exponent
    assert rec.max_loading == pytest.approx(1.0 / 6.0)


def test_x95s_kerosene_la_point_is_met_without_being_fitted(rec):
    """The level #270 shipped, held out of #283's fit and recovered.

    Q1 Table 2.6: D_La = 0.99 in KEROSENE at 1 mol/L P204 (the reference
    charge), 0.1 mol/L LaCl3, 0.003 mol/L HCl, O/A = 1 assumed. Z1 Eqs.
    (4.11)-(4.12) put the equilibrium at pH 0.8175 with 0.3508 M free dimer.
    A different diluent, element and charge from the heptane data the level
    is fitted to; it agrees to about 0.1 decades.
    """
    c_org = 0.99 * 0.1 / 1.99
    pH = -math.log10(0.003 + 3 * c_org)
    free_dimer = 0.5 - 3 * c_org
    assert pH == pytest.approx(0.8175, abs=1e-4)
    predicted = _log_D("La", pH, free_dimer)
    assert predicted == pytest.approx(math.log10(0.99), abs=0.15)


def test_the_shape_is_peppards(rec):
    """Peppard's 2.48 per step, not PPH63's 2.20.

    La, Ce, Pr, Eu and Tb sit on Peppard markers, so their differences are
    Peppard's to the digitization; Y is Mason's own level and lands within
    0.02 of Peppard's spacing from the rest.
    """
    a = {el: rec.ph_coefficients[el].a for el in PEPPARD}
    for el in ("Ce", "Pr", "Eu", "Tb"):
        assert a[el] - a["La"] == pytest.approx(PEPPARD[el] - PEPPARD["La"], abs=1e-3)
    assert a["Y"] - a["La"] == pytest.approx(PEPPARD["Y"] - PEPPARD["La"], abs=0.02)
    per_step = 10 ** ((a["Tb"] - a["La"]) / 8)
    assert per_step == pytest.approx(2.49, abs=0.02)
    assert per_step > 2.3          # PPH63's cumulative La->Tb is 2.25 per step


def test_the_whole_series_is_monotone_and_y_sits_above_dy(rec):
    order = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy"]
    a = [rec.ph_coefficients[el].a for el in order]
    assert all(np.diff(a) > 0)
    assert rec.ph_coefficients["Y"].a > rec.ph_coefficients["Dy"].a
