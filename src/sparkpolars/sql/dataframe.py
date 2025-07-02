"""
This module provides a DataFrameExtension class that extends Polars DataFrame with Spark-like API methods.

The DataFrameExtension class maintains compatibility with Spark DataFrame operations while leveraging
the performance benefits of the Polars library. It provides method chaining capabilities and
automatic type preservation through custom attribute access interception.
"""

from typing import Any

from polars import DataFrame as DataFrameOriginal
from polars import LazyFrame as LazyFrameOriginal
from polars import concat
from polars._typing import (
    FrameInitTypes,
    Orientation,
    SchemaDefinition,
    SchemaDict,
)
from polars.datatypes import (
    N_INFER_DEFAULT,
)


class DataFrameExtension:
    """
    A modified DataFrame class that provides Spark-like API methods on top of Polars DataFrame.

    This class extends Polars DataFrame to provide compatibility with Spark DataFrame operations
    while maintaining the performance benefits of Polars.
    """

    def withColumn(self, name: str, expr: Any) -> "DataFrameExtension":
        """
        Add a new column or replace an existing column with the given expression.

        :param name: The name of the column to add or replace
        :param expr: The expression to compute the column values

        :return: A new DataFrameExtension instance with the added/modified column
        """
        return self.with_columns(expr.alias(name))

    def withColumns(self, columns: dict[str, Any]) -> "DataFrameExtension":
        """
        Add multiple new columns or replace existing columns with the given expressions.

        :param columns: A dictionary mapping column names to expressions

        :return: A new DataFrameExtension instance with the added/modified columns
        """
        columns = [expr.alias(name) for name, expr in columns.items()]
        return self.with_columns(columns)

    def hint(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        """
        Provide a hint to the query optimizer (Spark compatibility method).

        :param args: Positional arguments for the hint
        :param kwargs: Keyword arguments for the hint

        :return: The same DataFrameExtension instance (no-op for Polars compatibility)
        """
        return self

    def repartition(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        """
        Repartition the DataFrame (Spark compatibility method).

        :param args: Positional arguments for repartitioning
        :param kwargs: Keyword arguments for repartitioning

        :return: The same DataFrameExtension instance (no-op for Polars compatibility)
        """
        return self

    def coalesce(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        """
        Coalesce the DataFrame partitions (Spark compatibility method).

        :param args: Positional arguments for coalescing
        :param kwargs: Keyword arguments for coalescing

        :return: The same DataFrameExtension instance (no-op for Polars compatibility)
        """
        return self

    def persist(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        """
        Cache the DataFrame in memory for faster subsequent operations.

        :param args: Positional arguments for persistence
        :param kwargs: Keyword arguments for persistence

        :return: A new DataFrameExtension instance with cached data
        """
        return self.cache()

    def distinct(self) -> "DataFrameExtension":
        """
        Return a new DataFrame with distinct rows.

        :return: A new DataFrameExtension instance containing only unique rows
        """
        return self.unique()

    def join(
        self,
        other: "DataFrameExtension",
        on: str | list[str],
        how: str = "inner",
    ) -> "DataFrameExtension":
        """
        Join this DataFrame with another DataFrame.

        :param other: The DataFrame to join with
        :param on: Column name(s) to join on
        :param how: Type of join to perform (inner, left, right, outer, anti, etc.)

        :return: A new DataFrameExtension instance with the joined data
        """
        mapping = {
            "inner": "inner",
            "left": "left",
            "left_out": "left",
            "right": "right",
            "right_outer": "right",
            "anti": "anti",
            "left_anti": "anti",
            "full": "outer",
        }
        how = mapping.get(how, how)
        if not all(isinstance(col, str) for col in on):
            msg = "Join columns must be strings."
            raise ValueError(msg)
        return self.join(other, on=on, how=how)

    def unionByName(
        self,
        other: "DataFrameExtension",
        allowMissingColumns: bool = False,
    ) -> "DataFrameExtension":
        """
        Return a new DataFrame containing the union of rows from this and other DataFrame.

        :param other: The DataFrame to union with
        :param allowMissingColumns: Whether to allow missing columns in either DataFrame

        :return: A new DataFrameExtension instance containing the union of both DataFrames
        """
        return concat(
            [self, other],
            how="diagonal_relaxed" if allowMissingColumns else "vertical_relaxed",
        )

    def crossJoin(self, other: "DataFrameExtension") -> "DataFrameExtension":
        """
        Return the Cartesian product of this DataFrame and other DataFrame.

        :param other: The DataFrame to cross join with

        :return: A new DataFrameExtension instance containing the cross join result
        """
        return self.join(other, how="cross")

    def __getattribute__(self, name: str) -> Any:
        """
        Intercepts all attribute access to wrap DataFrame methods to return DataFrameExtension.

        :param name: The name of the attribute being accessed

        :return: The attribute value, potentially wrapped to maintain DataFrameExtension type
        """
        attr = super().__getattribute__(name)
        if name in DataFrame.__dict__ and callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                r = attr(*args, **kwargs)
                if isinstance(r, DataFrame) and not isinstance(r, DataFrameExtension):
                    return self
                return r

            return wrapper
        return attr


class LazyFrame(LazyFrameOriginal, DataFrameExtension):
    def __init__(
        self,
        data: FrameInitTypes | None = None,
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        strict: bool = True,
        orient: Orientation | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        nan_to_null: bool = False,
    ) -> None:
        """
        Initialize a DataFrameMod instance with Spark-like API capabilities.

        :param data: Data to initialize the DataFrame with
        :param schema: Schema definition for the DataFrame
        :param schema_overrides: Schema overrides to apply
        :param strict: Whether to enforce strict
        schema validation
        """
        super().__init__(
            data=data.collect() if isinstance(data, LazyFrameOriginal) else data,
            schema=schema,
            schema_overrides=schema_overrides,
            strict=strict,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        )

    @property
    def schema(self):
        return super().collect_schema()

    @property
    def isEmpty(self) -> bool:
        """
        Check if the DataFrame is empty.

        :return: True if the DataFrame has no rows, False otherwise
        """
        return self.limit(1).collect().is_empty()


class DataFrame(DataFrameExtension, DataFrameOriginal):
    """
    A modified DataFrame class that provides Spark-like API methods on top of Polars DataFrame.

    This class extends Polars DataFrame to provide compatibility with Spark DataFrame operations
    while maintaining the performance benefits of Polars.
    """

    def __init__(
        self,
        data: FrameInitTypes | None = None,
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        strict: bool = True,
        orient: Orientation | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        nan_to_null: bool = False,
    ) -> None:
        """
        Initialize a DataFrameMod instance with Spark-like API capabilities.

        :param data: Data to initialize the DataFrame with
        :param schema: Schema definition for the DataFrame
        :param schema_overrides: Schema overrides to apply
        :param strict: Whether to enforce strict
        schema validation
        """
        super().__init__(
            data=data,
            schema=schema,
            schema_overrides=schema_overrides,
            strict=strict,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        )
