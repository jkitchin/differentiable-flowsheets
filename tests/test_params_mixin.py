"""Tests for ParamsMixin, in particular its PyTree registration."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import pytest

from difflow.params_mixin import ParamsMixin


@dataclass
class DemoParams(ParamsMixin):
    """Params for the tests below.

    Attributes:
        V: A numeric, differentiable field
        name: A static field
    """

    V: float
    name: str = "demo"


DemoParams.register_as_pytree()


@dataclass
class FlaggedParams(ParamsMixin):
    """Params carrying a boolean flag.

    Attributes:
        V: A numeric, differentiable field
        clip: A flag that only ever picks a branch
    """

    V: float
    clip: bool = True


FlaggedParams.register_as_pytree()


class TestDictAPI:
    def test_item_access_and_membership(self):
        p = DemoParams(V=1.0)

        assert p["V"] == 1.0
        assert "V" in p
        assert "nope" not in p
        with pytest.raises(KeyError):
            p["nope"]

    def test_functional_update_leaves_the_original_alone(self):
        p = DemoParams(V=1.0)
        q = p.update(V=2.0)

        assert q.V == 2.0
        assert p.V == 1.0


class TestPyTreeRegistration:
    def test_flatten_round_trip(self):
        p = DemoParams(V=2.0)
        leaves, treedef = jax.tree_util.tree_flatten(p)

        assert leaves == [2.0]
        assert jax.tree_util.tree_unflatten(treedef, leaves) == p

    def test_a_treedef_can_be_unflattened_more_than_once(self):
        """The aux data is shared across every unflatten of a treedef.

        Popping the child-name key out of it in place left the second call
        with nothing to read, which is what broke `grad` below.
        """
        p = DemoParams(V=2.0)
        leaves, treedef = jax.tree_util.tree_flatten(p)

        first = jax.tree_util.tree_unflatten(treedef, leaves)
        second = jax.tree_util.tree_unflatten(treedef, leaves)

        assert first == second == p

    def test_grad_through_a_registered_params(self):
        """`grad` unflattens twice: the primal in and the cotangent out."""
        g = jax.grad(lambda p: p.V**2)(DemoParams(V=3.0))

        assert float(g.V) == pytest.approx(6.0)
        assert g.name == "demo"

    def test_jit_accepts_params_as_an_argument(self):
        f = jax.jit(lambda p: p.V**2)

        assert float(f(DemoParams(V=2.0))) == pytest.approx(4.0)
        assert float(f(DemoParams(V=3.0))) == pytest.approx(9.0)

    def test_a_boolean_flag_is_static_not_a_leaf(self):
        """`bool` subclasses `int`, so an unguarded numeric test makes a
        flag a differentiable leaf and `grad` rejects the whole params."""
        leaves, _ = jax.tree_util.tree_flatten(FlaggedParams(V=1.0, clip=True))

        assert leaves == [1.0]

        g = jax.grad(lambda p: p.V**2)(FlaggedParams(V=2.0, clip=True))
        assert float(g.V) == pytest.approx(4.0)
        assert g.clip is True

    def test_tree_map_preserves_static_fields(self):
        doubled = jax.tree_util.tree_map(lambda v: v * 2, DemoParams(V=2.0))

        assert doubled.V == 4.0
        assert doubled.name == "demo"
