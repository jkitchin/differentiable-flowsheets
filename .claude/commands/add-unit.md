# Add New Unit Operation

Create a new unit operation for difflow.

## Arguments
- $ARGUMENTS: Name of the new unit (e.g., "Absorber", "Crystallizer")

## Instructions

Create a new unit operation following difflow conventions:

1. **Create the unit file** at `src/difflow/units/<name>.py`:
   - Import from JAX: `import jax.numpy as jnp`
   - Import base classes: `from difflow.streams import Stream`
   - Create class with `__call__` method
   - Ensure all operations are JAX-compatible

2. **Template structure**:
```python
"""
<UnitName> unit operation for difflow.
"""
import jax.numpy as jnp
from jax import jit
from typing import Dict, Any, Optional
from ..streams import Stream

class <UnitName>:
    """
    <Description of unit>.

    Parameters
    ----------
    <param> : <type>
        <description>

    Examples
    --------
    >>> unit = <UnitName>(...)
    >>> outlet = unit(inlet_stream)
    """

    def __init__(self, <params>):
        self.<param> = <param>

    def __call__(self, inlet: Stream, **kwargs) -> Stream:
        """
        Process inlet stream.

        Parameters
        ----------
        inlet : Stream
            Inlet stream

        Returns
        -------
        Stream
            Outlet stream
        """
        # Implementation using jnp operations
        pass
```

3. **Add to exports** in `src/difflow/__init__.py`

4. **Create test file** at `tests/test_<name>.py`

5. **Create example** showing usage

Ask the user for details about the unit's behavior before implementing.
