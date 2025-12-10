"""Plugin architecture for difflow unit operations.

This module provides infrastructure for extending difflow with custom
unit operations via plugins.

Usage for plugin developers:
    1. Create a package with unit operations
    2. Register operations in your package's __init__.py
    3. Add entry point in pyproject.toml:
       [project.entry-points."difflow.plugins"]
       my_plugin = "my_package:register"

Usage for users:
    from difflow.plugins import registry, load_plugins

    # Load all installed plugins
    load_plugins()

    # Access registered operations
    MyOperation = registry.get("my_operation")
"""

from typing import Protocol, runtime_checkable, Any, Callable
from dataclasses import dataclass
import importlib.metadata
import logging

from difflow.streams import Stream

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol for Unit Operations
# =============================================================================

@runtime_checkable
class UnitOperation(Protocol):
    """Protocol defining the interface for unit operations.

    Unit operations are callable objects that transform streams.
    This protocol enables static type checking without requiring inheritance.

    Example:
        class MyReactor:
            def __init__(self, params):
                self.params = params

            def __call__(self, inlet: Stream, **kwargs) -> tuple[Stream, dict]:
                # Process inlet stream
                return outlet, info
    """

    def __call__(self, inlet: Stream, **kwargs) -> tuple[Any, dict]:
        """Process inlet stream(s) and return outlet stream(s) with info.

        Args:
            inlet: Input stream (dict with F_<species>, T, P keys)
            **kwargs: Operation-specific parameters

        Returns:
            outlet: Output stream(s) - single Stream or tuple of Streams
            info: Dictionary with operation results and metrics
        """
        ...


@runtime_checkable
class MultiInletOperation(Protocol):
    """Protocol for operations with multiple inlet streams."""

    def __call__(self, *inlets: Stream, **kwargs) -> tuple[Any, dict]:
        """Process multiple inlet streams."""
        ...


# =============================================================================
# Operation Registry
# =============================================================================

@dataclass
class OperationInfo:
    """Metadata about a registered operation."""
    name: str
    cls: type
    category: str
    description: str
    plugin: str


class OperationRegistry:
    """Registry for unit operations.

    Provides centralized discovery and access to unit operations from
    the core library and installed plugins.

    Example:
        from difflow.plugins import registry

        # Register an operation
        registry.register("my_reactor", MyReactor, category="reactors")

        # Get an operation
        ReactorClass = registry.get("my_reactor")

        # List all operations
        for name, info in registry.list_operations().items():
            print(f"{name}: {info.description}")
    """

    def __init__(self):
        self._operations: dict[str, OperationInfo] = {}
        self._categories: dict[str, list[str]] = {}

    def register(
        self,
        name: str,
        cls: type,
        category: str = "general",
        description: str = "",
        plugin: str = "core",
    ) -> type:
        """Register a unit operation.

        Args:
            name: Unique name for the operation
            cls: The operation class
            category: Category for organization (e.g., "reactors", "separations")
            description: Human-readable description
            plugin: Name of the plugin providing this operation

        Returns:
            The registered class (for use as decorator)

        Raises:
            ValueError: If name is already registered
        """
        if name in self._operations:
            existing = self._operations[name]
            logger.warning(
                f"Operation '{name}' already registered by {existing.plugin}. "
                f"Overwriting with version from {plugin}."
            )

        self._operations[name] = OperationInfo(
            name=name,
            cls=cls,
            category=category,
            description=description or cls.__doc__ or "",
            plugin=plugin,
        )

        # Track by category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        logger.debug(f"Registered operation: {name} ({category}) from {plugin}")
        return cls

    def get(self, name: str) -> type:
        """Get a registered operation class by name.

        Args:
            name: The operation name

        Returns:
            The operation class

        Raises:
            KeyError: If operation is not registered
        """
        if name not in self._operations:
            available = ", ".join(sorted(self._operations.keys()))
            raise KeyError(
                f"Operation '{name}' not found. Available: {available}"
            )
        return self._operations[name].cls

    def get_info(self, name: str) -> OperationInfo:
        """Get metadata about a registered operation."""
        return self._operations[name]

    def list_operations(self, category: str = None) -> dict[str, OperationInfo]:
        """List registered operations.

        Args:
            category: Filter by category (optional)

        Returns:
            Dictionary of name -> OperationInfo
        """
        if category is None:
            return dict(self._operations)

        return {
            name: self._operations[name]
            for name in self._categories.get(category, [])
        }

    def list_categories(self) -> list[str]:
        """List all registered categories."""
        return list(self._categories.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._operations

    def __len__(self) -> int:
        return len(self._operations)


# Global registry instance
registry = OperationRegistry()


# =============================================================================
# Decorator for Registration
# =============================================================================

def register_operation(
    name: str = None,
    category: str = "general",
    description: str = "",
    plugin: str = "user",
) -> Callable[[type], type]:
    """Decorator to register a unit operation.

    Example:
        @register_operation("my_reactor", category="reactors")
        class MyReactor:
            def __call__(self, inlet, **kwargs):
                ...
    """
    def decorator(cls: type) -> type:
        op_name = name or cls.__name__
        registry.register(op_name, cls, category, description, plugin)
        return cls
    return decorator


# =============================================================================
# Plugin Loading
# =============================================================================

_plugins_loaded = False


def load_plugins(force: bool = False) -> dict[str, list[str]]:
    """Load all installed difflow plugins.

    Discovers plugins via entry points and calls their registration functions.

    Args:
        force: If True, reload plugins even if already loaded

    Returns:
        Dictionary mapping plugin names to list of registered operations
    """
    global _plugins_loaded

    if _plugins_loaded and not force:
        return {}

    loaded = {}

    # Python 3.10+ style entry points
    try:
        eps = importlib.metadata.entry_points(group="difflow.plugins")
    except TypeError:
        # Python 3.9 compatibility
        eps = importlib.metadata.entry_points().get("difflow.plugins", [])

    for ep in eps:
        try:
            logger.info(f"Loading plugin: {ep.name}")

            # Track operations before loading
            ops_before = set(registry._operations.keys())

            # Load and call the registration function
            register_fn = ep.load()
            if callable(register_fn):
                register_fn(registry)

            # Track new operations
            ops_after = set(registry._operations.keys())
            new_ops = list(ops_after - ops_before)
            loaded[ep.name] = new_ops

            logger.info(f"Plugin {ep.name} registered {len(new_ops)} operations")

        except Exception as e:
            logger.error(f"Failed to load plugin {ep.name}: {e}")

    _plugins_loaded = True
    return loaded


def discover_plugins() -> list[str]:
    """List available plugins without loading them.

    Returns:
        List of plugin names
    """
    try:
        eps = importlib.metadata.entry_points(group="difflow.plugins")
    except TypeError:
        eps = importlib.metadata.entry_points().get("difflow.plugins", [])

    return [ep.name for ep in eps]


# =============================================================================
# Utility Functions
# =============================================================================

def is_unit_operation(obj: Any) -> bool:
    """Check if an object implements the UnitOperation protocol."""
    return isinstance(obj, UnitOperation)


def validate_operation(cls: type) -> list[str]:
    """Validate that a class can serve as a unit operation.

    Args:
        cls: The class to validate

    Returns:
        List of validation warnings (empty if valid)
    """
    warnings = []

    # Check callable
    if not callable(cls):
        warnings.append(f"{cls.__name__} is not callable")
        return warnings

    # Check __call__ signature
    import inspect
    try:
        sig = inspect.signature(cls.__call__)
        params = list(sig.parameters.keys())

        if len(params) < 2:  # self + at least one inlet
            warnings.append(
                f"{cls.__name__}.__call__ should accept at least one inlet stream"
            )
    except (ValueError, TypeError):
        warnings.append(f"Could not inspect {cls.__name__}.__call__ signature")

    # Check return annotation if present
    if hasattr(cls.__call__, "__annotations__"):
        ret = cls.__call__.__annotations__.get("return")
        if ret is not None and not str(ret).startswith("tuple"):
            warnings.append(
                f"{cls.__name__}.__call__ should return tuple[Stream, dict]"
            )

    return warnings
