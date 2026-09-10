"""Uncertain distribution coefficients: overriding the tabulated correlation.

The coefficients in ``data/extractants.yaml`` are regression outputs with real
standard errors, and ``a`` is ``log10(D)`` at the reference condition --- so an
uncertainty of 0.2 in ``a`` is a factor of 1.6 in ``D``.  These tests pin the
two properties that make the override usable for uncertainty work: it changes
``D`` by exactly the amount the correlation says it should, and it accepts a
JAX tracer, so a distribution can be put on it and differentiated through.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import get_flows, make_stream
from difflow_ree import REEExtractor, REEExtractorParams
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.units.extraction import MixerSettlerParams, REEMixerSettler
from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams
from difflow_ree.units.stripping import REEStripper, StripperParams

ELEMENTS = ("Nd", "Dy")


@pytest.fixture(scope="module")
def base():
    return REEDistribution(extractant="D2EHPA", elements=ELEMENTS)


class TestOverrideArithmetic:
    def test_a_decade_in_a_is_a_decade_in_D(self, base):
        """log10(D) = a + b pH + ..., so +1 in a is exactly x10 in D."""
        a0 = base._coefficients("Nd").a
        bumped = REEDistribution(extractant="D2EHPA", elements=ELEMENTS,
                                 coefficient_overrides={"Nd": {"a": a0 + 1.0}})
        assert float(bumped.get_D("Nd", pH=2.5)) == pytest.approx(
            10.0 * float(base.get_D("Nd", pH=2.5)), rel=1e-12)

    def test_an_untouched_element_is_untouched(self, base):
        d = REEDistribution(extractant="D2EHPA", elements=ELEMENTS,
                            coefficient_overrides={"Nd": {"a": -5.0}})
        assert float(d.get_D("Dy", pH=2.5)) == pytest.approx(
            float(base.get_D("Dy", pH=2.5)), rel=1e-12)

    def test_an_untouched_coefficient_keeps_its_database_value(self, base):
        """Only 'a' is named, so 'b' must still come from the record."""
        c0 = base._coefficients("Nd")
        d = REEDistribution(extractant="D2EHPA", elements=ELEMENTS,
                            coefficient_overrides={"Nd": {"a": -5.0}})
        c1 = d._coefficients("Nd")
        assert (c1.a, c1.b, c1.c) == (-5.0, c0.b, c0.c)

    def test_the_separation_factor_moves_by_the_coefficient_difference(self, base):
        """beta = D_Dy/D_Nd, so shifting a_Nd down raises beta by that decade."""
        a0 = base._coefficients("Nd").a
        d = REEDistribution(extractant="D2EHPA", elements=ELEMENTS,
                            coefficient_overrides={"Nd": {"a": a0 - 0.3}})
        b0 = float(base.get_separation_factor("Dy", "Nd", pH=2.5))
        b1 = float(d.get_separation_factor("Dy", "Nd", pH=2.5))
        assert b1 / b0 == pytest.approx(10.0 ** 0.3, rel=1e-10)

    def test_overriding_every_coefficient_at_once(self, base):
        d = REEDistribution(
            extractant="D2EHPA", elements=ELEMENTS,
            coefficient_overrides={"Nd": {"a": -6.0, "b": 2.0, "c": 0.0,
                                          "d": 0.0}})
        # log10(D) = -6 + 2*2.5 + 0, plus the [HA] term, which is zero at the
        # reference concentration this instance is at.
        assert float(d.get_D("Nd", pH=2.5)) == pytest.approx(10.0 ** -1.0,
                                                             rel=1e-12)


class TestOverrideValidation:
    def test_an_unknown_element_is_named_in_the_error(self):
        with pytest.raises(ValueError, match="names element 'La'"):
            REEDistribution(extractant="D2EHPA", elements=("Nd",),
                            coefficient_overrides={"La": {"a": 1.0}})

    def test_an_unknown_coefficient_is_named_in_the_error(self):
        with pytest.raises(ValueError, match=r"carries \['z'\]"):
            REEDistribution(extractant="D2EHPA", elements=("Nd",),
                            coefficient_overrides={"Nd": {"z": 1.0}})

    def test_no_override_is_the_default_and_changes_nothing(self, base):
        d = REEDistribution(extractant="D2EHPA", elements=ELEMENTS,
                            coefficient_overrides=None)
        assert float(d.get_D("Nd", pH=2.5)) == pytest.approx(
            float(base.get_D("Nd", pH=2.5)), rel=1e-12)


class TestTracing:
    """The point of the field: a distribution on D that differentiates."""

    @staticmethod
    def _D(a):
        d = REEDistribution(extractant="D2EHPA", elements=ELEMENTS,
                            coefficient_overrides={"Nd": {"a": a}})
        return d.get_D("Nd", pH=2.5)

    def test_the_gradient_is_the_analytic_one(self):
        """dD/da = D ln 10."""
        a = -7.0
        got = float(jax.grad(self._D)(a))
        assert got == pytest.approx(float(self._D(a)) * np.log(10.0), rel=1e-10)

    def test_it_jits(self):
        assert float(jax.jit(self._D)(-7.0)) == pytest.approx(
            float(self._D(-7.0)), rel=1e-12)

    def test_it_vmaps_over_a_sample(self):
        a = jnp.linspace(-7.4, -6.6, 16)
        got = jax.vmap(self._D)(a)
        assert got.shape == (16,)
        assert bool(jnp.all(jnp.diff(got) > 0))     # monotone in a


class TestUnitsPassItThrough:
    """All four units that build a REEDistribution must honour the override."""

    @pytest.mark.parametrize("build", [
        lambda ov: REEExtractor(REEExtractorParams(
            n_stages=4, extractant="D2EHPA", elements=ELEMENTS,
            coefficient_overrides=ov)),
        lambda ov: REEMixerSettler(MixerSettlerParams(
            extractant="D2EHPA", elements=ELEMENTS, coefficient_overrides=ov)),
        lambda ov: REEScrubber(ScrubberParams(
            n_stages=2, extractant="D2EHPA", elements=ELEMENTS,
            target_elements=("Dy",), coefficient_overrides=ov)),
        lambda ov: REEStripper(StripperParams(
            n_stages=2, extractant="D2EHPA", elements=ELEMENTS,
            coefficient_overrides=ov)),
    ])
    def test_the_unit_sees_the_overridden_D(self, build, base):
        a0 = base._coefficients("Nd").a
        unit = build({"Nd": {"a": a0 + 1.0}})
        assert float(unit._distribution.get_D("Nd", pH=2.5)) == pytest.approx(
            10.0 * float(base.get_D("Nd", pH=2.5)), rel=1e-12)


class TestThroughACascade:
    """The property that matters downstream: it reaches the raffinate."""

    @staticmethod
    def _raffinate_Nd(a_Nd):
        params = REEExtractorParams(
            n_stages=5.0, extractant="D2EHPA", elements=ELEMENTS, pH=2.4,
            extractant_conc=1.0, include_loading=True,
            coefficient_overrides={"Nd": {"a": a_Nd}})
        feed = make_stream({"H2O": 55.0, "Nd": 1.0, "Dy": 0.12},
                           T=298.15, P=101325.0)
        solvent = make_stream({"D2EHPA": 3.0, "kerosene": 9.0},
                              T=298.15, P=101325.0)
        raff, _, _ = REEExtractor(params)(feed, solvent)
        return get_flows(raff)["Nd"]

    def test_more_extractable_Nd_leaves_less_in_the_raffinate(self):
        assert float(self._raffinate_Nd(-7.0)) < float(self._raffinate_Nd(-7.7))

    def test_the_cascade_gradient_is_finite_and_signed(self):
        g = float(jax.grad(self._raffinate_Nd)(-7.7))
        assert np.isfinite(g) and g < 0.0

    def test_the_cascade_vmaps_over_a_sample(self):
        out = jax.vmap(self._raffinate_Nd)(jnp.linspace(-8.0, -7.2, 24))
        assert out.shape == (24,) and bool(jnp.all(jnp.isfinite(out)))


class TestTracedStageCount:
    """n_stages is a continuous decision; the guard must not force it."""

    def test_the_eager_guard_still_fires(self):
        with pytest.raises(ValueError, match="n_stages must be >= 1"):
            REEExtractorParams(n_stages=0.5, extractant="D2EHPA",
                               elements=("Nd",))

    def test_a_traced_stage_count_is_allowed(self):
        def raff(n):
            params = REEExtractorParams(
                n_stages=n, extractant="D2EHPA", elements=ELEMENTS, pH=2.4,
                extractant_conc=1.0)
            feed = make_stream({"H2O": 55.0, "Nd": 1.0, "Dy": 0.12},
                               T=298.15, P=101325.0)
            solvent = make_stream({"D2EHPA": 3.0, "kerosene": 9.0},
                                  T=298.15, P=101325.0)
            _, ext, _ = REEExtractor(params)(feed, solvent)
            return get_flows(ext)["Dy"]

        g = float(jax.grad(raff)(6.0))
        assert np.isfinite(g) and g > 0.0        # more stages extract more
        assert jax.vmap(raff)(jnp.array([3.0, 6.0, 9.0])).shape == (3,)
