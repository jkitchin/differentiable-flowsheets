"""Base database class for property databases.

This module provides a generic BaseDatabase class that eliminates
duplication across the various database implementations in difflow
and its plugins.

Usage:
    from difflow.base_database import BaseDatabase

    class SolventDatabase(BaseDatabase[Solvent]):
        _item_type_name = "solvent"

        def __init__(self):
            super().__init__(_SOLVENT_DATA)
"""

from typing import TypeVar, Generic, Iterator

__all__ = ["BaseDatabase"]

T = TypeVar("T")


class BaseDatabase(Generic[T]):
    """Generic base class for property databases.

    Provides common functionality for get/list operations with
    consistent error messages and dict-like access.

    Type Parameters:
        T: The type of items stored in the database (e.g., CellLine, Resin)

    Attributes:
        _item_type_name: Human-readable name for error messages (e.g., "cell line")

    Example:
        >>> class SolventDatabase(BaseDatabase[Solvent]):
        ...     _item_type_name = "solvent"
        ...
        ...     def __init__(self):
        ...         super().__init__(_SOLVENT_DATA)
        ...
        >>> db = SolventDatabase()
        >>> mea = db.get("MEA")
        >>> print(db.list_items())
    """

    _item_type_name: str = "item"

    def __init__(self, data: dict[str, T]):
        """Initialize database with data dictionary.

        Args:
            data: Dictionary mapping names to data objects
        """
        self._data: dict[str, T] = data.copy()

    def get(self, name: str) -> T:
        """Get item by name.

        Args:
            name: Item identifier

        Returns:
            The requested item

        Raises:
            KeyError: If name not found in database
        """
        if name not in self._data:
            available = ", ".join(sorted(self._data.keys()))
            raise KeyError(
                f"Unknown {self._item_type_name}: '{name}'. "
                f"Available: {available}"
            )
        return self._data[name]

    def __getitem__(self, name: str) -> T:
        """Dict-like access to items."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if name exists in database."""
        return name in self._data

    def __len__(self) -> int:
        """Return number of items in database."""
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterate over item names."""
        return iter(self._data)

    def list_items(self) -> list[str]:
        """List all available item names.

        Returns:
            Sorted list of item names
        """
        return sorted(self._data.keys())

    def items(self) -> Iterator[tuple[str, T]]:
        """Iterate over (name, item) pairs."""
        return iter(self._data.items())

    def values(self) -> Iterator[T]:
        """Iterate over items."""
        return iter(self._data.values())

    def filter_by(self, attr: str, value) -> list[str]:
        """List items where attribute equals value.

        Args:
            attr: Attribute name to filter on
            value: Value to match

        Returns:
            List of item names matching the filter
        """
        return [
            name for name, item in self._data.items()
            if getattr(item, attr, None) == value
        ]

    def search(self, **kwargs) -> list[str]:
        """Search for items matching multiple attribute criteria.

        Args:
            **kwargs: Attribute name-value pairs to match

        Returns:
            List of item names matching all criteria
        """
        results = []
        for name, item in self._data.items():
            match = all(
                getattr(item, attr, None) == val
                for attr, val in kwargs.items()
            )
            if match:
                results.append(name)
        return results

    def add(self, name: str, item: T) -> None:
        """Add a new item to the database.

        Args:
            name: Item identifier
            item: Item to add

        Raises:
            ValueError: If name already exists
        """
        if name in self._data:
            raise ValueError(f"{self._item_type_name} '{name}' already exists")
        self._data[name] = item

    def update(self, name: str, item: T) -> None:
        """Update an existing item or add if not exists.

        Args:
            name: Item identifier
            item: Item to add/update
        """
        self._data[name] = item
