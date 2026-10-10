"""Tests for difflow.liquid_properties (issue #406).

The reference file is written by scripts/generate_liquid_properties.py.
"chemicals" values pin the JAX equations to the source library's own
evaluation; "coolprop" values (saturated liquid) check the correlations
against an independent reference. Regenerate the file when it goes stale;
never loosen a tolerance to make it pass.
"""

import json
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from difflow import make_stream
from difflow.database import list_species, track_database_access
from difflow.liquid_properties import (
    LiquidRangeWarning,
    check_liquid_range,
    evaluate_correlation,
    get_liquid_properties,
    liquid_density,
    liquid_molar_density,
    liquid_property_source,
    liquid_thermal_conductivity,
    liquid_viscosity,
    list_liquid_species,
    mixture_liquid_density,
    mixture_liquid_molar_density,
    mixture_liquid_thermal_conductivity,
    mixture_liquid_viscosity,
    stream_liquid_properties,
)
from difflow._liquid_property_data import LIQUID_DATA, SOURCES

jax.config.update("jax_enable_x64", True)

REF = json.loads((Path(__file__).parent / "reference"
                  / "liquid_properties_reference.json").read_text())
PROPS = ("rho", "mu", "k")

# CoolProp agreement, max relative error at points inside the correlation's
# range. Measured worst cases when the data were generated: density 2.2%
# (ethanol, 177 K), viscosity 20% (cyclohexane at Perry's 443 K limit),
# thermal conductivity 18% (liquid hydrogen).
COOLPROP_TOL = {"rho": 0.025, "mu": 0.20, "k": 0.20}
# Species where VDI-PPDS and Perry's agree with each other to <10% but both
# differ from CoolProp's viscosity by 40-350% at low temperature; the
# disagreement is CoolProp's, so these points are not a test of difflow.
COOLPROP_MU_SUSPECT = {"n_pentane", "dimethyl_ether", "isopentane"}


def _native(name, prop, T):
    """The correlation in the units the reference file stores."""
    return float(evaluate_correlation(getattr(get_liquid_properties(name),
                                              prop), T))


class TestCoverage:
    def test_every_database_species_has_all_three(self):
        assert list_liquid_species() == list_species()
        for name in list_species():
            data = get_liquid_properties(name)
            for prop in PROPS:
                c = getattr(data, prop)
                assert c.Tmin < c.Tmax, (name, prop)
                assert c.source in SOURCES, (name, prop)

    def test_every_source_is_used_and_cited(self):
        used = {LIQUID_DATA[n][p]["source"] for n in LIQUID_DATA for p in PROPS}
        assert used == set(SOURCES)
        assert "Perry" in liquid_property_source("benzene", "mu")

    def test_aliases_and_unknown_species(self):
        assert get_liquid_properties("ipa").name == "2_propanol"
        assert get_liquid_properties("H2O").name == "water"
        with pytest.raises(KeyError, match="No liquid-property data"):
            get_liquid_properties("unobtainium")
        with pytest.raises(ValueError):
            liquid_property_source("water", "cp")

    def test_lookups_are_tracked_for_reports(self):
        with track_database_access() as tracker:
            liquid_density("toluene", 300.0)
        assert tracker.kinds("toluene") == {"liquid"}


class TestEquationsMatchSource:
    """The JAX forms reproduce `chemicals`' evaluation of the same coefficients."""

    @pytest.mark.parametrize("name", sorted(REF["chemicals"]))
    def test_species(self, name):
        for prop in PROPS:
            for T, expected in REF["chemicals"][name][prop]:
                assert _native(name, prop, T) == pytest.approx(
                    expected, rel=1e-9), (name, prop, T)


@pytest.mark.release
class TestAgainstCoolProp:
    @pytest.mark.parametrize("name", sorted(REF["coolprop"]))
    def test_species(self, name):
        data = get_liquid_properties(name)
        checked = 0
        for pt in REF["coolprop"][name]["points"]:
            T = pt["T"]
            for prop in PROPS:
                ref = pt[prop]
                corr = getattr(data, prop)
                if ref is None or not corr.Tmin <= T <= corr.Tmax:
                    continue
                if prop == "mu" and name in COOLPROP_MU_SUSPECT:
                    continue
                if prop == "rho":
                    ours = float(liquid_density(name, T))
                else:
                    ours = _native(name, prop, T)
                assert ours == pytest.approx(ref, rel=COOLPROP_TOL[prop]), (
                    name, prop, T)
                checked += 1
        assert checked > 0

    def test_estimates_against_tables(self):
        """Rackett, Orrick-Erbar and Sato-Riedel on species the tables cover.

        Errors at 298 K when generated: Rackett +1-3%, Orrick-Erbar -25% to
        +22%, Sato-Riedel +3% (n-alkanes) to +27% (branched isomers).
        These bound what the four estimated isomers can be trusted to.
        """
        bounds = {"rho": 0.04, "mu": 0.30, "k": 0.30}
        for name, props in REF["estimate_check"].items():
            for prop, v in props.items():
                assert v["estimate"] == pytest.approx(
                    v["table"], rel=bounds[prop]), (name, prop)


class TestValues:
    def test_water_at_25C(self):
        # IAPWS-95 / IAPWS 2008 / IAPWS 2011: 997.05 kg/m^3, 0.890 mPa s,
        # 0.6065 W/m/K.
        assert float(liquid_density("water", 298.15)) == pytest.approx(
            997.05, rel=2e-3)
        assert float(liquid_viscosity("water", 298.15)) == pytest.approx(
            8.90e-4, rel=0.03)
        assert float(liquid_thermal_conductivity("water", 298.15)) == (
            pytest.approx(0.6065, rel=0.02))

    def test_molar_and_mass_density_consistent(self):
        data = get_liquid_properties("benzene")
        assert float(liquid_density(data, 300.0)) == pytest.approx(
            float(liquid_molar_density(data, 300.0)) * data.MW / 1e3)

    def test_trends(self):
        T = jnp.linspace(280.0, 340.0, 7)
        for name in ("water", "toluene", "n_heptane"):
            assert jnp.all(jnp.diff(liquid_density(name, T)) < 0)
            assert jnp.all(jnp.diff(liquid_viscosity(name, T)) < 0)

    def test_vectorized(self):
        T = jnp.array([290.0, 300.0, 310.0])
        assert liquid_viscosity("ethanol", T).shape == (3,)
        assert float(liquid_viscosity("ethanol", T)[1]) == pytest.approx(
            float(liquid_viscosity("ethanol", 300.0)))


class TestRangeWarnings:
    def test_inside_range_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            liquid_viscosity("water", 300.0)

    def test_outside_range_warns_but_does_not_clip(self):
        hi = get_liquid_properties("benzene").mu.Tmax
        with pytest.warns(LiquidRangeWarning, match="benzene liquid viscosity"):
            mu_out = float(liquid_viscosity("benzene", hi + 20.0))
        assert mu_out < float(liquid_viscosity("benzene", hi - 1.0))

    def test_no_warning_under_tracing(self):
        hi = get_liquid_properties("benzene").mu.Tmax
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            jax.grad(lambda T: liquid_viscosity("benzene", T))(hi + 20.0)

    def test_check_liquid_range(self):
        info = check_liquid_range("water", 300.0)
        assert all(v["in_range"] for v in info.values())
        assert check_liquid_range("water", 1000.0)["rho"]["in_range"] is False


class TestGradients:
    # One species per correlation form in use.
    @pytest.mark.parametrize("name,prop,T", [
        ("water", "rho", 320.0),          # PPDS10
        ("benzene", "mu", 320.0),         # DIPPR101 (Perry's)
        ("neopentane", "mu", 260.0),      # PPDS9
        ("toluene", "k", 320.0),          # DIPPR100
        ("2_4_dimethylpentane", "rho", 320.0),   # DIPPR105 (Rackett)
        ("2_4_dimethylpentane", "k", 320.0),     # SATO_RIEDEL
    ])
    def test_pure(self, name, prop, T):
        corr = getattr(get_liquid_properties(name), prop)
        check_grads(lambda t: evaluate_correlation(corr, t), (T,), order=2,
                    modes=["fwd", "rev"], rtol=1e-5)

    def test_mixtures(self):
        def f(x1, T):
            x = {"water": 1.0 - x1, "ethanol": x1}
            return (mixture_liquid_density(x, T)
                    + 1e6 * mixture_liquid_viscosity(x, T)
                    + 1e3 * mixture_liquid_thermal_conductivity(x, T))
        check_grads(f, (0.3, 320.0), order=1, modes=["rev"], rtol=1e-5)

    def test_jit(self):
        f = jax.jit(lambda T: liquid_viscosity("toluene", T))
        assert float(f(300.0)) == pytest.approx(
            float(liquid_viscosity("toluene", 300.0)))


class TestMixtures:
    def test_pure_limit(self):
        T = 310.0
        x = {"toluene": 1.0, "benzene": 0.0}
        assert float(mixture_liquid_density(x, T)) == pytest.approx(
            float(liquid_density("toluene", T)))
        assert float(mixture_liquid_viscosity(x, T)) == pytest.approx(
            float(liquid_viscosity("toluene", T)))
        assert float(mixture_liquid_thermal_conductivity(x, T)) == (
            pytest.approx(float(liquid_thermal_conductivity("toluene", T))))

    def test_rules_by_hand(self):
        T = 300.0
        a, b = "benzene", "toluene"
        ra, rb = (float(liquid_molar_density(s, T)) for s in (a, b))
        ma, mb = (float(liquid_viscosity(s, T)) for s in (a, b))
        ka, kb = (float(liquid_thermal_conductivity(s, T)) for s in (a, b))
        Ma, Mb = get_liquid_properties(a).MW, get_liquid_properties(b).MW
        x = {a: 2.0, b: 6.0}                     # normalized to 0.25 / 0.75
        assert float(mixture_liquid_molar_density(x, T)) == pytest.approx(
            1.0 / (0.25 / ra + 0.75 / rb))
        assert float(mixture_liquid_viscosity(x, T)) == pytest.approx(
            ma**0.25 * mb**0.75)
        wa = 0.25 * Ma / (0.25 * Ma + 0.75 * Mb)
        assert float(mixture_liquid_thermal_conductivity(x, T)) == (
            pytest.approx((wa / ka**2 + (1 - wa) / kb**2) ** -0.5))

    def test_stream(self):
        s = make_stream({"water": 8.0, "methanol": 2.0}, T=300.0, P=1e5)
        props = stream_liquid_properties(s)
        assert set(props) == {"rho", "rho_molar", "mu", "k", "Q"}
        assert float(props["Q"]) == pytest.approx(10.0 / float(
            props["rho_molar"]))
        rho_w, rho_m = (float(liquid_density(n, 300.0))
                        for n in ("water", "methanol"))
        assert rho_m < float(props["rho"]) < rho_w
