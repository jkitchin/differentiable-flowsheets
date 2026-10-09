"""ValueKeyed objects: value keys must never outlive the trace that built them."""

from functools import partial

import jax
import jax.numpy as jnp

from difflow.cache_key import NO_KEY
from difflow.eos import CriticalProperties, PengRobinson


def _species():
    return {
        "A": CriticalProperties(name="A", Tc=369.8, Pc=4.25e6, omega=0.152),
        "B": CriticalProperties(name="B", Tc=425.1, Pc=3.80e6, omega=0.200),
    }


@partial(jax.jit, static_argnames=("eos",))
def _core(x, eos):
    return x * eos.params.Tc


def test_equal_eos_share_a_key_eagerly():
    assert PengRobinson(_species()) == PengRobinson(_species())


def test_eos_built_under_jit_has_no_value_key():
    seen = []

    @jax.jit
    def build(x):
        eos = PengRobinson(_species())
        seen.append(eos)
        return _core(x, eos)

    build(1.0)
    assert seen[0]._value_key is NO_KEY
    assert seen[0] != PengRobinson(_species())


def test_eager_call_after_jit_built_eos_does_not_hit_dead_tracers():
    """Issue pattern: jit with construction inside, then an equal-value eager call."""

    @jax.jit
    def build(x):
        return _core(x, PengRobinson(_species()))

    jax.block_until_ready(build(1.0))
    jax.block_until_ready(build(1.0))
    out = _core(jnp.ones(2), PengRobinson(_species()))   # UnexpectedTracerError before
    assert jnp.allclose(out, jnp.array([369.8, 425.1]))
