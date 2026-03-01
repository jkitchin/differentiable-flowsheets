"""Mixin class providing dict-like API for Params dataclasses.

This module provides the ParamsMixin class that can be inherited by
any dataclass to gain dict-like access methods (keys, values, items, etc.)
and a concise __repr__ method suitable for JAX arrays.

Example usage:
    from dataclasses import dataclass
    from difflow.params_mixin import ParamsMixin

    @dataclass
    class MyParams(ParamsMixin):
        x: float
        y: float

    params = MyParams(x=1.0, y=2.0)
    params['x']        # -> 1.0
    'x' in params      # -> True
    list(params.keys())  # -> ['x', 'y']
    params.update(x=3.0)  # -> MyParams(x=3.0, y=2.0)
"""

from dataclasses import replace, fields, asdict as dc_asdict
from typing import Any, Iterator

import jax.numpy as jnp


def _fmt_value(v: Any) -> str:
    """Format a value for concise repr.

    Handles special cases:
    - None: returns "None"
    - Callables with __name__: returns the function name
    - Arrays with shape: returns "Array[shape]" or scalar value
    - Dicts: recursively formats values
    - Long lists/tuples: returns "type[length]"
    - Everything else: uses repr()

    Args:
        v: Value to format

    Returns:
        Formatted string representation
    """
    if v is None:
        return "None"
    if callable(v) and hasattr(v, '__name__'):
        return v.__name__
    if hasattr(v, 'shape'):  # JAX/numpy array
        if v.ndim == 0:
            return f"{float(v):.4g}"
        return f"Array{list(v.shape)}"
    if isinstance(v, dict):
        items = ", ".join(f"{k}: {_fmt_value(val)}" for k, val in v.items())
        return "{" + items + "}"
    if isinstance(v, (list, tuple)) and len(v) > 5:
        return f"{type(v).__name__}[{len(v)}]"
    return repr(v)


class ParamsMixin:
    """Mixin providing dict-like API for Params dataclasses.

    This mixin adds the following methods to any dataclass:
    - update(**kwargs): Return new instance with fields replaced (JAX-compatible)
    - __getitem__(key): Dict-style access by field name
    - get(key, default): Safe access with default value (like dict.get)
    - __contains__(key): Check if field exists
    - keys(): Iterator over field names
    - values(): Iterator over field values
    - items(): Iterator over (name, value) pairs
    - __iter__(): Iterate over field names
    - __len__(): Number of fields
    - asdict(): Convert to dictionary
    - __repr__(): Concise string representation

    All methods work with dataclass fields and are compatible with JAX tracing.
    """

    def update(self, **kwargs) -> "ParamsMixin":
        """Return a new instance with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.
        Uses dataclasses.replace() under the hood.

        Args:
            **kwargs: Fields to update with their new values

        Returns:
            New instance of the same class with updated fields

        Example:
            >>> params = CSTRParams(V=1.0, rate_fn=my_fn, ...)
            >>> new_params = params.update(V=2.0)
            >>> new_params.V
            2.0
        """
        return replace(self, **kwargs)

    def __getitem__(self, key: str) -> Any:
        """Get parameter value by name for dict-like access.

        Args:
            key: Field name to access

        Returns:
            Value of the field

        Raises:
            KeyError: If field doesn't exist
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value with optional default (like dict.get).

        Args:
            key: Field name to access
            default: Value to return if field doesn't exist (default: None)

        Returns:
            Value of the field, or default if field doesn't exist

        Example:
            >>> params.get('optional_field', 0.0)
            0.0
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __contains__(self, key: str) -> bool:
        """Check if a field exists in the params.

        Args:
            key: Field name to check

        Returns:
            True if field exists, False otherwise
        """
        return key in {f.name for f in fields(self)}

    def keys(self) -> Iterator[str]:
        """Return field names for dict-like iteration.

        Returns:
            Iterator over field names
        """
        return (f.name for f in fields(self))

    def values(self) -> Iterator[Any]:
        """Return field values for dict-like iteration.

        Returns:
            Iterator over field values
        """
        return (getattr(self, f.name) for f in fields(self))

    def items(self) -> Iterator[tuple[str, Any]]:
        """Return (name, value) pairs for dict-like iteration.

        Returns:
            Iterator over (field_name, value) tuples
        """
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names (like dict).

        Returns:
            Iterator over field names
        """
        return (f.name for f in fields(self))

    def __len__(self) -> int:
        """Return number of fields.

        Returns:
            Number of dataclass fields
        """
        return len(fields(self))

    def asdict(self) -> dict:
        """Convert params to a dictionary.

        Returns:
            Dictionary with field names as keys and values
        """
        return dc_asdict(self)

    @classmethod
    def register_as_pytree(cls):
        """Register this dataclass as a JAX PyTree.

        Call this after defining the dataclass to enable differentiation
        through its fields. Only numeric fields will be treated as leaves.

        Example:
            @dataclass
            class MyParams(ParamsMixin):
                x: float
                y: float
            MyParams.register_as_pytree()
        """
        import jax
        from dataclasses import fields as dc_fields

        def tree_flatten(obj):
            children = []
            aux_data = {}
            for f in dc_fields(obj):
                val = getattr(obj, f.name)
                if isinstance(val, (int, float, jnp.ndarray)) or hasattr(val, 'shape'):
                    children.append(val)
                else:
                    aux_data[f.name] = val
            child_names = [f.name for f in dc_fields(obj)
                          if isinstance(getattr(obj, f.name), (int, float, jnp.ndarray))
                          or hasattr(getattr(obj, f.name), 'shape')]
            aux_data['_child_names'] = tuple(child_names)
            return children, aux_data

        def tree_unflatten(aux_data, children):
            child_names = aux_data.pop('_child_names')
            kwargs = dict(zip(child_names, children))
            kwargs.update(aux_data)
            return cls(**kwargs)

        jax.tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)

    def __repr__(self) -> str:
        """Concise string representation.

        Provides a readable repr that handles:
        - JAX arrays (shows shape or scalar value)
        - Callables (shows function name)
        - Long sequences (shows type and length)
        - Nested dicts (recursively formatted)

        Returns:
            String representation like "ClassName(field1=value1, field2=value2)"
        """
        items = ", ".join(
            f"{f.name}={_fmt_value(getattr(self, f.name))}"
            for f in fields(self)
        )
        return f"{self.__class__.__name__}({items})"
