"""Tests for the pH validity-range report (#262).

Every cation-exchange record in `data/extractants.yaml` declares a
`valid_ph_range` -- the window its `ph_coefficients` describe. That field was
loaded into `database.Extractant` and then read by nothing, so a circuit could
be run at `stripping_pH=0.3` against a quadratic fitted over [1.5, 5.5] and
get a silent answer. `examples/10_bastnasite_separation.ipynb` did exactly
that, at three different pH values.

PC88A's window is [0.1, 2.5] since the #270 refit. It is the span T21 actually
measured -- pH 0.12 to about 2.4 across its six extractant concentrations --
and it is now a window over MEASURED `a` values rather than over hand-tuned
ones, so the two ends mean the same thing they say. The probes below sit at
0.05 (under the floor) and 6.0 (well over the ceiling).

Extrapolating `a + b*pH + c*pH**2` is not a small error: PC88A's b is the
stoichiometric 3, so one pH unit outside the window moves D by three decades.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_ree.database import get_extractant
from difflow_ree.equilibrium.distribution import REEDistribution


class TestPHRangeIsReported:
    """The window is read, and leaving it is reported."""

    def test_262_in_range_is_silent(self):
        dist = REEDistribution(extractant="PC88A", elements=("Nd",))
        lo, hi = get_extractant("PC88A").valid_ph_range
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            for pH in (lo, 0.5 * (lo + hi), hi):
                assert np.isfinite(float(dist.get_D("Nd", pH=pH)))

    def test_262_below_the_window_warns_and_names_the_low_end(self):
        dist = REEDistribution(extractant="PC88A", elements=("Nd",))
        with pytest.warns(UserWarning, match=r"pH minimum 0\.05.*\[0\.1, 2\.5\]"):
            dist.get_D("Nd", pH=0.05)

    def test_262_above_the_window_warns_and_names_the_high_end(self):
        dist = REEDistribution(extractant="PC88A", elements=("Nd",))
        with pytest.warns(UserWarning, match=r"pH maximum 6.*\[0\.1, 2\.5\]"):
            dist.get_D("Nd", pH=6.0)

    def test_262_the_range_is_per_extractant_not_a_global_constant(self):
        """The two windows overlap but neither contains the other.

        Since the #270 refits PC88A is [0.1, 2.5] (T21's own span) and
        Cyanex272 is [1.5, 3.5] (ZL93/L14's), which is the whole point of the
        field: Cyanex272 is the weaker acid and needs a pH unit and a half
        more to extract the same element. Each probe below is inside one
        window and outside the other, in opposite directions, so a global
        constant substituted for either record would fail one of them.
        """
        assert get_extractant("PC88A").valid_ph_range == (0.1, 2.5)
        assert get_extractant("Cyanex272").valid_ph_range == (1.5, 3.5)

        # pH 1.0: inside PC88A, under Cyanex272's floor.
        pc88a = REEDistribution(extractant="PC88A", elements=("Nd",))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pc88a.get_D("Nd", pH=1.0)
        cyanex = REEDistribution(extractant="Cyanex272", elements=("Nd",))
        with pytest.warns(UserWarning, match="validity range"):
            cyanex.get_D("Nd", pH=1.0)

        # pH 3.0: inside Cyanex272, over PC88A's ceiling.
        cyanex_hi = REEDistribution(extractant="Cyanex272", elements=("Nd",))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            cyanex_hi.get_D("Nd", pH=3.0)
        pc88a_hi = REEDistribution(extractant="PC88A", elements=("Nd",))
        with pytest.warns(UserWarning, match="validity range"):
            pc88a_hi.get_D("Nd", pH=3.0)

    def test_262_a_concrete_array_is_checked_at_both_ends(self):
        """A per-stage pH profile that leaves the window anywhere is caught."""
        dist = REEDistribution(extractant="PC88A", elements=("Nd",))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            dist.get_D("Nd", pH=jnp.array([1.0, 1.5, 2.0]))
        low = REEDistribution(extractant="PC88A", elements=("Nd",))
        with pytest.warns(UserWarning, match=r"pH minimum 0\.05"):
            low.get_D("Nd", pH=jnp.array([0.05, 1.5, 2.0]))
        high = REEDistribution(extractant="PC88A", elements=("Nd",))
        with pytest.warns(UserWarning, match="pH maximum 6"):
            high.get_D("Nd", pH=jnp.array([1.0, 2.0, 6.0]))


class TestPHRangeIsAReportNotAGuard:
    """The value is used as given. Nothing is clamped."""

    def test_262_pH_is_not_clamped(self):
        """A clamp would silently relocate a flowsheet's operating point.

        D at the extrapolated pH must differ from D at the window edge -- if
        the check clamped, the two would be equal and a user asking for pH 7
        would silently get pH 2.5 physics.

        Probed at the TOP edge: below the floor there is only 0.05 pH of
        room left, about 0.15 decades, which no margin can tell apart from a
        clamp. 4.0 against the 2.5 ceiling is 1.5 pH units, four and a half
        decades at b = 3, which keeps the original 1e3 test with room to
        spare.
        """
        dist = REEDistribution(
            extractant="PC88A", elements=("Nd",), on_out_of_range="ignore"
        )
        D_out = float(dist.get_D("Nd", pH=4.0))
        D_edge = float(dist.get_D("Nd", pH=2.5))
        assert D_out > D_edge * 1e3
        # And it is still the honest value of the correlation there.
        c = get_extractant("PC88A").ph_coefficients["Nd"]
        expected = 10.0 ** (c.a + c.b * 4.0 + c.c * 4.0**2)
        assert D_out == pytest.approx(expected, rel=1e-10)

    def test_262_can_be_escalated_to_an_error(self):
        dist = REEDistribution(
            extractant="PC88A", elements=("Nd",), on_out_of_range="raise"
        )
        with pytest.raises(ValueError, match="validity range"):
            dist.get_D("Nd", pH=0.05)

    def test_262_can_be_silenced(self):
        dist = REEDistribution(
            extractant="PC88A", elements=("Nd",), on_out_of_range="ignore"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert np.isfinite(float(dist.get_D("Nd", pH=0.05)))

    def test_262_does_not_spam_a_stage_loop(self):
        dist = REEDistribution(extractant="PC88A", elements=("Nd",))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(20):
                dist.get_D("Nd", pH=0.05)
        assert len([w for w in caught if "262" in str(w.message)]) == 1


class TestPHRangeAndTracing:
    """pH is this library's primary differentiation variable."""

    def test_262_a_traced_pH_is_not_reported(self):
        """Unlike traced ionic_strength (#194), a traced pH says nothing.

        `ionic_strength` defaults to None, so its check is opt-in and a tracer
        there means the caller asked for a correction they cannot verify. pH is
        mandatory and is meant to be a continuous, traceable decision, so
        warning here would fire on every grad/jit/vmap of every REE circuit and
        train users to filter the category that carries the concrete report.
        """
        dist = REEDistribution(extractant="PC88A", elements=("Nd",))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g = jax.grad(lambda x: dist.get_D("Nd", pH=x))(3.0)
            jax.jit(lambda x: dist.get_D("Nd", pH=x))(3.0)
            jax.vmap(lambda x: dist.get_D("Nd", pH=x))(jnp.array([2.0, 3.0]))
        assert not [w for w in caught if "262" in str(w.message)]
        assert np.isfinite(float(g)) and float(g) > 0

    def test_262_gradients_are_unchanged_by_the_check(self):
        dist = REEDistribution(extractant="PC88A", elements=("Nd",))
        c = get_extractant("PC88A").ph_coefficients["Nd"]
        pH = 2.0
        D = 10.0 ** (c.a + c.b * pH + c.c * pH**2)
        expected = D * np.log(10.0) * (c.b + 2 * c.c * pH)
        got = float(jax.grad(lambda x: dist.get_D("Nd", pH=x))(pH))
        assert got == pytest.approx(expected, rel=1e-8)


class TestSolvatingExtractantsHaveNoPHTerm:
    """A nitrate-driven correlation is not a function of pH at all."""

    def test_262_a_solvating_extractant_is_not_pH_checked(self):
        dist = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0, medium="nitrate"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert np.isfinite(float(dist.get_D("Nd")))
