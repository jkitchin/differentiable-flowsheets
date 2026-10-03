"""A hydrotreater's high-pressure separator against DWSIM 9.0.5.

DWSIM ran in the generator (``reference/dwsim_hps_generate.py``) and wrote
``reference/dwsim_hps_reference.json``; this file reads it and needs neither
DWSIM nor .NET. The feed is a real reactor effluent -- the diesel
hydrotreater of ``test_hydrotreating.py``, solved, its ``reactor_out``
frozen into the file -- flashed at the unit's separator (50 C, 47 bar) and
at the corners of 40-60 C x 30-60 bar (``reference/dwsim_hps_case.py``).

Comparison (a), ``same``: DWSIM's "Peng-Robinson 1978 (PR78)" with every
component a hypothetical compound on difflow's constants and difflow's kij
(:data:`~difflow_refinery.hydroprocessing.thermo.DEFAULT_KIJ` and 0.0333 for
H2S with every cut). DWSIM's PR78 switches kappa at omega 0.491 as
difflow's hydroprocessing PR does; read from its IL (``ThermoPlugs.PR78``),
it writes the 1976 branch's 1.54226 as 1.5422. Reproduced here
(:func:`_ln_phi_dwsim78`), the fugacity coefficients agree to round-off
(3e-14); plain, the truncation is worth up to 1.7e-4 in a liquid phi (the
cut just below the switch, omega 0.465). Then difflow's
:func:`~difflow_refinery.hydroprocessing.separator.pr_flash` lands on
DWSIM's flash within DWSIM's own convergence.

Comparison (b): DWSIM's gas compounds with difflow's kij
(``dwsim_compounds``), then with DWSIM's own (``dwsim_data``); and the 1976
kappa for every omega (``pr76``, DWSIM's "Peng-Robinson (PR)"). These measure
the model -- above all the hydrogen and H2S dissolved in the separator
liquid (#333), which the docs tabulate -- and are pinned only as sizes.

``release`` throughout but for the per-commit checks on the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .reference import dwsim_hps_case as hc

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent / "reference"
REF = json.loads((HERE / "dwsim_hps_reference.json").read_text())
EFF = REF["effluent"]
NAMES = REF["names"]

REGENERATE = ("the hydrotreater's separator feed or constants no longer match the ones the DWSIM "
              "HPS reference was built on; regenerate it: PYTHONPATH=src:tests python -m "
              "refinery.reference.dwsim_hps_generate (needs DWSIM 9.0.5; see scripts/install_dwsim.sh)")


def _comps():
    return hc.flash_components_of(EFF)


def _ln_phi_dwsim78(T, P, x, comps, phase, k1=1.5422):
    """difflow's hydroprocessing PR ``ln phi`` with DWSIM PR78's kappa: the
    1976 quadratic with ``k1`` for omega <= 0.491, the 1978 cubic above."""
    from difflow_refinery.hydroprocessing import thermo as ht

    w = comps.omega
    kap = jnp.where(w <= 0.491, 0.37464 + k1 * w - 0.26992 * w ** 2,
                    0.379642 + 1.48503 * w - 0.164423 * w ** 2 + 0.016666 * w ** 3)
    Tr = T / comps.Tc
    a_i = 0.45724 * ht.R_GAS ** 2 * comps.Tc ** 2 / comps.Pc * (1 + kap * (1 - jnp.sqrt(Tr))) ** 2
    b_i = 0.07780 * ht.R_GAS * comps.Tc / comps.Pc
    sa = jnp.sqrt(a_i)
    a_ij = jnp.outer(sa, sa) * (1.0 - comps.kij)
    x = jnp.asarray(x)
    xa = a_ij @ x
    am, bm = x @ xa, x @ b_i
    RT = ht.R_GAS * T
    A, B = am * P / RT ** 2, bm * P / RT
    roots = np.roots([1.0, -(1.0 - float(B)), float(A - 3 * B ** 2 - 2 * B),
                      -float(A * B - B ** 2 - B ** 3)])
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-12 and r.real > float(B))
    Z = real[0] if phase == "liquid" else real[-1]
    s2 = np.sqrt(2.0)
    return (b_i / bm * (Z - 1) - jnp.log(Z - B) - A / (2 * s2 * B) * (2 * xa / am - b_i / bm)
            * jnp.log((Z + (1 + s2) * B) / (Z + (1 - s2) * B)))


def _dissolved(beta, x, z):
    """Fraction of each component's feed that leaves in the liquid."""
    return (1.0 - beta) * np.asarray(x) / np.asarray(z)


# ---------------------------------------------------------------------------
# Per commit: the file, and whether it is still the hydrotreater's
# ---------------------------------------------------------------------------


class TestReferenceFile:
    def test_it_records_where_it_came_from(self):
        p = REF["provenance"]
        for key in ("generated", "script", "command", "dwsim", "dotnet", "pythonnet",
                    "reference_simulator", "difflow_commit"):
            assert p[key], key
        assert "9.0.5" in p["dwsim"]
        assert EFF["converged"]
        assert (HERE / "dwsim_hps_generate.py").exists()

    def test_the_like_for_like_setup(self):
        s = REF["setups"]["same"]["dwsim"]
        assert s["property_package"] == "Peng-Robinson 1978 (PR78)"
        assert s["kij"] == "given"
        comps, _ = _comps()
        np.testing.assert_allclose(s["kij_matrix"], np.asarray(comps.kij), atol=1e-15)
        for i, k in enumerate(s["constants"]):
            assert k["is_hypo"]
            for key in ("MW", "Tc", "Pc", "omega"):
                assert k[key] == pytest.approx(float(np.asarray(getattr(comps, key))[i]), rel=1e-14)
        assert REF["setups"]["pr76"]["dwsim"]["property_package"] == "Peng-Robinson (PR)"
        d = REF["setups"]["dwsim_data"]["dwsim"]
        assert d["kij"] == "dwsim" and np.any(d["kij_matrix"])
        assert not any(k["is_hypo"] for k in d["constants"][:comps.n_gas])

    def test_every_flash_is_two_phase(self):
        for name, s in REF["setups"].items():
            assert len(s["flashes"]) == len(hc.POINTS)
            for r in s["flashes"]:
                assert r["phases"] == ["vapor", "liquid"], (name, r["T"], r["P"])
                assert r["equilibrium_residual"] < 1e-4, (name, r["T"], r["P"])


class TestReferenceIsCurrent:
    def test_points(self):
        assert REF["points"] == json.loads(json.dumps(hc.POINTS)), REGENERATE

    def test_the_constants_are_the_hydrotreaters(self):
        """The unit's flash table now against the frozen one (the effluent
        itself needs a two-minute solve: :func:`test_the_effluent_is_the_hydrotreaters`)."""
        u, feed = hc.unit()
        now = hc.constants(u, feed)
        ref = EFF["components"]
        assert now["names"] == ref["names"], REGENERATE
        for key in ("Tb", "Tc", "Pc", "omega", "MW", "SG", "cp_ig", "kij"):
            np.testing.assert_allclose(now[key], ref[key], rtol=1e-12, atol=1e-15,
                                       err_msg=f"{key}: {REGENERATE}")
        assert EFF["T"] == u.params.hps_T and EFF["P"] == u.params.P - u.params.loop_dP


@pytest.mark.slow
@pytest.mark.release
def test_the_effluent_is_the_hydrotreaters():
    eff = hc.effluent()
    assert eff["converged"]
    np.testing.assert_allclose(eff["flows"], EFF["flows"], rtol=1e-7, atol=1e-12, err_msg=REGENERATE)


# ---------------------------------------------------------------------------
# Release: (a) the same model
# ---------------------------------------------------------------------------


@pytest.mark.release
class TestSameModel:

    @pytest.mark.parametrize("phase", ["vapor", "liquid"])
    def test_fugacity_coefficients(self, phase):
        """At DWSIM's own phase compositions. With PR78's 1.5422 reproduced,
        measured 2.8e-14 relative worst; with difflow's 1.54226, 1.7e-4
        (liquid, the omega 0.465 cut) and 4e-5 (vapour)."""
        from difflow_refinery.hydroprocessing.thermo import pr_lnphi

        comps, _ = _comps()
        key, ck = ("phi_vap", "y") if phase == "vapor" else ("phi_liq", "x")
        for r in REF["setups"]["same"]["flashes"]:
            c = np.asarray(r[ck])
            emu = np.exp(np.asarray(_ln_phi_dwsim78(r["T"], r["P"], c, comps, phase)))
            np.testing.assert_allclose(emu, r[key], rtol=1e-9, err_msg=f"{r['T']} {r['P']}")
            exact = np.exp(np.asarray(pr_lnphi(r["T"], r["P"], jnp.asarray(c), comps, phase)[0]))
            np.testing.assert_allclose(exact, r[key], rtol=5e-4, err_msg=f"{r['T']} {r['P']}")

    def test_the_flash(self):
        """difflow's ``pr_flash`` of the same feed. Measured: vapour fraction
        4.7e-7, liquid composition 1.7e-6 and vapour 3.7e-8 absolute (60
        bar), DWSIM's own residual up to 2.6e-7; held to 1e-5 (the kappa
        truncation is in the difference)."""
        from difflow_refinery.hydroprocessing.separator import pr_flash

        comps, z = _comps()
        for r in REF["setups"]["same"]["flashes"]:
            fr = pr_flash(r["T"], r["P"], jnp.asarray(z), comps)
            tol = 10 * max(r["equilibrium_residual"], 1e-6)
            assert float(fr.beta) == pytest.approx(r["vapor_fraction"], abs=tol), (r["T"], r["P"])
            np.testing.assert_allclose(np.asarray(fr.x), r["x"], atol=tol)
            np.testing.assert_allclose(np.asarray(fr.y), r["y"], atol=tol)

    def test_dissolved_gases(self):
        """The fraction of the H2, H2S, NH3 and C1 fed that leaves dissolved
        in the separator liquid (#333). Measured 3.4e-5 relative worst (H2,
        the least soluble, 1.5 % of it dissolved); 6e-6 on the others."""
        from difflow_refinery.hydroprocessing.separator import pr_flash

        comps, z = _comps()
        idx = [NAMES.index(n) for n in hc.SOLUTES]
        for r in REF["setups"]["same"]["flashes"]:
            fr = pr_flash(r["T"], r["P"], jnp.asarray(z), comps)
            mine = _dissolved(float(fr.beta), np.asarray(fr.x), z)[idx]
            ref = _dissolved(r["vapor_fraction"], r["x"], z)[idx]
            np.testing.assert_allclose(mine, ref, rtol=1e-4, err_msg=f"{r['T']} {r['P']}")


# ---------------------------------------------------------------------------
# Release: (b) the model's sensitivity
# ---------------------------------------------------------------------------


def _solubility(setup):
    _, z = _comps()
    idx = [NAMES.index(n) for n in hc.SOLUTES]
    return np.array([_dissolved(r["vapor_fraction"], r["x"], z)[idx]
                     for r in REF["setups"][setup]["flashes"]])


@pytest.mark.release
class TestModelSensitivity:
    """Sizes, not agreements: each is the effect of one modelling choice on
    the gas dissolved in the separator liquid, pinned so a regeneration that
    moves it is noticed. The docs tabulate them."""

    def test_the_1976_kappa_on_the_heavy_cuts(self):
        """``pr76``: the heavy cuts on the 1976 kappa (omega up to 0.79).
        Measured: 1.5 % more H2 dissolved, 1 % more C1, H2S and NH3 within
        0.4 %."""
        r = _solubility("pr76") / _solubility("same") - 1
        assert 0.01 < r[:, 0].min() and r[:, 0].max() < 0.02, r
        assert np.max(np.abs(r)) < 0.02, r

    def test_dwsims_gas_constants(self):
        """DWSIM's database constants for the gases (H2S omega 0.094 against
        0.09, NH3 0.256 against 0.253): 0.25 % at most."""
        r = _solubility("dwsim_compounds") / _solubility("same") - 1
        assert np.max(np.abs(r)) < 0.005, r

    def test_dwsims_kij(self):
        """DWSIM's own kij, and none for H2S with the cuts (difflow 0.0333):
        12-15 % more H2S dissolved; H2, NH3 and C1 within 0.7 %."""
        r = _solubility("dwsim_data") / _solubility("dwsim_compounds") - 1
        assert 0.10 < r[:, 1].min() and r[:, 1].max() < 0.17, r
        assert np.max(np.abs(r[:, [0, 2, 3]])) < 0.01, r
