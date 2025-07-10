from collections.abc import Iterator
from typing import Any

import polars.functions as plf


class WindowClass:

    def __init__(self) -> None:
        self._partition_by = plf.all()
        self._order_by = None
        self._sort_by = None

    def partitionBy(self, *cols: Any) -> "WindowClass":
        self._partition_by = cols
        return self

    def orderBy(self, *cols: Any) -> "WindowClass":
        self._order_by = cols
        self._sort_by = [
            col._desc_status if hasattr(col, "_desc_status") else False for col in cols
        ]
        return self

    def rangeBetween(self, start: Any, end: Any) -> "WindowClass":
        return self

    def rowsBetween(self, start: Any, end: Any) -> "WindowClass":
        return self

    def unboundedFollowing(self) -> "WindowClass":
        return self

    def unboundedPreceding(self) -> "WindowClass":
        return self

    def currentRow(self) -> "WindowClass":
        return self

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._partition_by

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(
            {
                "partition_by": self._partition_by,
                "order_by": self._order_by,
                "sort_by": self._sort_by,
            },
        )

    @property
    def cols(self) -> list[Any]:
        return getattr(self, "_partition_by", [])


class WindowFactory:
    """Factory class that allows both W and W() usage."""

    def __init__(self) -> None:
        self._instance = WindowClass()

    def __call__(self) -> WindowClass:
        """Return a new Window instance when called as W()."""
        return WindowClass()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the Window instance for W.method() usage."""
        return getattr(self._instance, name)


# Create a singleton instance that works for both W and W() usage
Window = WindowFactory()
