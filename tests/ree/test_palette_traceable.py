"""The palette wrappers run under jit and jax.grad (#381).

They inherited the circuits' ``float()`` calls on flows, which raised
ConcretizationTypeError as soon as a flow was traced.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from difflow import make_stream
from difflow_ree.flowsheets.extract_scrub_strip import ExtractScrubStripParams
from difflow_ree.flowsheets.extract_strip import ExtractStripParams
from difflow_ree.flowsheets.full_train import SeparationTrainParams
from difflow_ree.flowsheets.palette import (
    ExtractScrubStripUnit,
    ExtractStripUnit,
    SeparationTrainUnit,
    SplitShellUnit,
)
from difflow_ree.flowsheets.split_shell import SplitShellParams

ELS = ("La", "Nd", "Gd", "Dy")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _feed(x, elements):
    return make_stream(
        {"H2O": 1.0, **{e: (x if e == elements[0] else 0.01) for e in elements}},
        298.15, 101325.0)


def _total(streams):
    return sum(jnp.asarray(v) for s in streams for k, v in s.items() if k.startswith("F_"))


def _es():
    unit = ExtractStripUnit(ExtractStripParams(extractant="D2EHPA", elements=("La", "Nd")))
    return lambda x: unit(_feed(x, ("La", "Nd")))[1]["F_Nd"]


def _ess(recycle):
    unit = ExtractScrubStripUnit(ExtractScrubStripParams(
        extractant="D2EHPA", elements=ELS, target_elements=("Gd", "Dy"),
        recycle_scrub_liquor=recycle))
    return lambda x: unit(_feed(x, ELS))[1]["F_Dy"]


def _split():
    unit = SplitShellUnit(SplitShellParams(
        extractant="D2EHPA", elements=ELS, n_stages=10, split_points=(5,), pH=1.0))
    solvent = make_stream({"kerosene": 1.0, "D2EHPA": 0.5, **{e: 0.0 for e in ELS}},
                          298.15, 101325.0)
    return lambda x: _total(unit(_feed(x, ELS), solvent)[:-1])


def _train():
    unit = SeparationTrainUnit(SeparationTrainParams(elements=ELS, include_ce_removal=False))
    return lambda x: _total(unit(_feed(x, ELS))[:-1])


CASES = {
    "extract_strip": _es,
    "extract_scrub_strip": lambda: _ess(False),
    "extract_scrub_strip_recycled": lambda: _ess(True),
    "split_shell": _split,
    "separation_train": _train,
}


@pytest.mark.parametrize("name", CASES)
def test_wrapper_jits_and_differentiates(name):
    f = CASES[name]()
    eager = float(f(0.01))
    assert float(jax.jit(f)(0.01)) == pytest.approx(eager, rel=1e-9)
    g = float(jax.grad(f)(0.01))
    assert jnp.isfinite(g)
    h = 1e-5
    fd = (float(f(0.01 + h)) - float(f(0.01 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-3, abs=1e-8)
