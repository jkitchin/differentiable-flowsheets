"""The isomerization reactor against IDAES's GibbsReactor (#311).

``release`` throughout: the reactor integrated to equilibrium, against the
committed ``reference/isom_reference.json`` (IDAES is not needed to run
this; see ``reference/isom_generate.py`` to rebuild it). The per-commit
checks on the file itself are in ``test_isomerization_validation_file.py``.

What this is: an **independent implementation, not an independent model**.
IDAES minimises the total Gibbs energy of difflow's own ideal-gas
constants; difflow integrates its rate law along the bed until the bed stops
changing. Agreement says the free energies are assembled the same way, that
``K`` is in bar against a 1 bar standard state with the hydrogen partial
pressure where it belongs, that the adiabatic energy balance is right, and
that the rate law relaxes onto the equilibrium it claims. It says nothing
about whether the constants are right, or how fast a real catalyst gets
there (the rate constants are illustrative).

To reach equilibrium the reactor's activity is scaled up and hydrocracking
(irreversible, so absent from an equilibrium) switched off:

* ``c6_ring``, isothermal: ``k_scale = 1e4``, 40 steps;
* ``adiabatic``: ``k_scale = 10``, 400 steps, ``LHSV = 0.1`` -- the
  adiabatic bed is stiffer (the temperature rises with conversion), and
  this is what the integrator needs to land on the equilibrium temperature
  to 1e-9 K.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from difflow_refinery.isomerization import IsomerizationReactor, IsomerizationReactorParams
from difflow_refinery.isomerization import thermochem as tc

jax.config.update("jax_enable_x64", True)

from .test_isomerization_validation_file import stale_339

# The reference predates #339's constants (see STALE_SINCE_339 there).
pytestmark = [pytest.mark.release, stale_339]

REF = json.loads((Path(__file__).parent / "reference" / "isom_reference.json").read_text())
P = REF["cases"]["P"]


def run(case, *, adiabatic, k_scale, n_steps, LHSV, T_in):
    rx = IsomerizationReactor(IsomerizationReactorParams(
        adiabatic=adiabatic, k_scale=k_scale, crack_scale=0.0, n_steps=n_steps, LHSV=LHSV))
    F0 = jnp.asarray([case["feed"].get(n, 0.0) for n in tc.NAMES])
    out = rx.run(F0, T_in, P=P)
    assert float(out["info"]["stage_residual"]) < 1e-10
    return out


def mole_fraction_error(out, case):
    F = out["F"]
    tot, rtot = float(jnp.sum(F)), sum(case["flows"].values())
    return max(abs(float(F[tc.idx(n)]) / tot - v / rtot) for n, v in case["flows"].items())


@pytest.mark.parametrize("i", range(len(REF["c6_ring"])))
def test_isothermal_ring_equilibrium(i):
    """Benzene saturation, ring contraction and ring opening at 30 bar."""
    case = REF["c6_ring"][i]
    out = run(case, adiabatic=False, k_scale=1e4, n_steps=40, LHSV=1.0, T_in=case["T"])
    assert mole_fraction_error(out, case) < 1e-10


@pytest.mark.parametrize("kind", sorted(REF["adiabatic"]))
def test_adiabatic_equilibrium(kind):
    """The adiabatic equilibrium temperature and composition of each
    constructed feed's reactor charge."""
    case = REF["adiabatic"][kind]
    out = run(case, adiabatic=True, k_scale=10.0, n_steps=400, LHSV=0.1,
              T_in=REF["cases"]["adiabatic"]["T_in"])
    assert float(out["T"]) == pytest.approx(case["T"], abs=1e-6)
    assert mole_fraction_error(out, case) < 1e-10
