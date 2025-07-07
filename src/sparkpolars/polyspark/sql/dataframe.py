"""Polyspark DataFrame methods for Polars DataFrame and LazyFrame."""

from typing import Any

from polars import DataFrame as DataFrameOriginal
from polars import LazyFrame as LazyFrameOriginal
from polars import concat
from polars.expr import Expr


def withColumn(
    self: DataFrameOriginal | LazyFrameOriginal,
    name: str,
    expr: Expr,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self.with_columns(expr.alias(name))


def withColumns(
    self: DataFrameOriginal | LazyFrameOriginal,
    *args: Any,
    **kwargs: Any,
) -> DataFrameOriginal | LazyFrameOriginal:
    # Support both dictionary and multiple name/expr pairs
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        # Dictionary form: withColumns({"col1": expr1, "col2": expr2})
        columns_dict = args[0]
        columns = [expr.alias(name) for name, expr in columns_dict.items()]
        return self.with_columns(columns)
    if len(args) == 2 and not kwargs:
        # Two argument form: withColumns("col_name", expr) - treat as withColumn
        name, expr = args
        return self.with_columns(expr.alias(name))
    if kwargs:
        # Keyword argument form: withColumns(col1=expr1, col2=expr2)
        columns = [expr.alias(name) for name, expr in kwargs.items()]
        return self.with_columns(columns)
    msg = (
        "withColumns expects either a dictionary, keyword arguments, "
        "or exactly two positional arguments (name, expr)"
    )
    raise ValueError(msg)


def hint(
    self: DataFrameOriginal | LazyFrameOriginal,
    *_args: Any,
    **_kwargs: Any,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self


def repartition(
    self: DataFrameOriginal | LazyFrameOriginal,
    *_args: Any,
    **_kwargs: Any,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self


def coalesce(
    self: DataFrameOriginal | LazyFrameOriginal,
    *_args: Any,
    **_kwargs: Any,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self


def persist(
    self: DataFrameOriginal | LazyFrameOriginal,
    *_args: Any,
    **_kwargs: Any,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self.cache()


def distinct(
    self: DataFrameOriginal | LazyFrameOriginal,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self.unique()


def dropDuplicates(
    self: DataFrameOriginal | LazyFrameOriginal,
    subset: list[str] | None = None,
) -> DataFrameOriginal | LazyFrameOriginal:
    if subset:
        return self.distinct()
    msg = "drop_duplicates with 'subset' parameter is not implemented."
    raise NotImplementedError(
        msg,
    )


def drop_duplicates(
    self: DataFrameOriginal | LazyFrameOriginal,
    subset: list[str] | None = None,
) -> DataFrameOriginal | LazyFrameOriginal:
    if subset:
        return self.distinct()
    msg = "drop_duplicates with 'subset' parameter is not implemented."
    raise NotImplementedError(
        msg,
    )


def dropna(
    self: DataFrameOriginal | LazyFrameOriginal,
    how: str = "any",
    thresh: int | None = None,
    subset: list[str] | None = None,
) -> DataFrameOriginal | LazyFrameOriginal:
    if how == "any" and thresh is None:
        return self.drop_nans(subset)
    msg = "dropna with 'how' or 'thresh' parameters is not implemented."
    raise NotImplementedError(
        msg,
    )


def dropnulls(
    self: DataFrameOriginal | LazyFrameOriginal,
    how: str = "any",
    thresh: int | None = None,
    subset: list[str] | None = None,
) -> DataFrameOriginal | LazyFrameOriginal:
    if how == "any" and thresh is None:
        return self.drop_nulls(subset)
    msg = "dropna with 'how' or 'thresh' parameters is not implemented."
    raise NotImplementedError(
        msg,
    )


def join(
    self: DataFrameOriginal | LazyFrameOriginal,
    other: DataFrameOriginal | LazyFrameOriginal,
    on: str | list[str],
    how: str = "inner",
) -> DataFrameOriginal | LazyFrameOriginal:
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
    self: DataFrameOriginal | LazyFrameOriginal,
    other: DataFrameOriginal | LazyFrameOriginal,
    allowMissingColumns: bool = False,
) -> DataFrameOriginal | LazyFrameOriginal:
    return concat(
        [self, other],
        how="diagonal_relaxed" if allowMissingColumns else "vertical_relaxed",
    )


def crossJoin(
    self: DataFrameOriginal | LazyFrameOriginal,
    other: DataFrameOriginal | LazyFrameOriginal,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self.join(other, how="cross")


def checkpoint(
    self: DataFrameOriginal | LazyFrameOriginal,
    *_args: Any,
    **_kwargs: Any,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self


def localCheckpoint(
    self: DataFrameOriginal | LazyFrameOriginal,
    *_args: Any,
    **_kwargs: Any,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self


def groupBy(self: DataFrameOriginal | LazyFrameOriginal, *cols: Any) -> Any:
    # Convert ColExtension objects to their underlying expressions
    return self.group_by(*cols)


def schema_lazy(self: LazyFrameOriginal) -> Any:
    return self.collect_schema()


def isEmpty_lazy(self: LazyFrameOriginal) -> bool:
    return self.limit(1).collect().is_empty()


def count_lazy(self: LazyFrameOriginal) -> int:
    return self.collect().height


def columns_lazy(self: LazyFrameOriginal) -> list[str]:
    return self.collect_schema().names()


def isEmpty_non_lazy(self: DataFrameOriginal) -> bool:
    return self.is_empty()


def count_non_lazy(self: DataFrameOriginal) -> int:
    return self.height


DataFrameOriginal.withColumn = withColumn
DataFrameOriginal.withColumns = withColumns
DataFrameOriginal.hint = hint
DataFrameOriginal.repartition = repartition
DataFrameOriginal.coalesce = coalesce
DataFrameOriginal.persist = persist
DataFrameOriginal.distinct = distinct
DataFrameOriginal.dropDuplicates = dropDuplicates
DataFrameOriginal.drop_duplicates = drop_duplicates
DataFrameOriginal.dropna = dropna
DataFrameOriginal.dropnulls = dropnulls
DataFrameOriginal.join = join
DataFrameOriginal.unionByName = unionByName
DataFrameOriginal.crossJoin = crossJoin
DataFrameOriginal.checkpoint = checkpoint
DataFrameOriginal.localCheckpoint = localCheckpoint
DataFrameOriginal.groupBy = groupBy

DataFrameOriginal.isEmpty = isEmpty_non_lazy
DataFrameOriginal.count = property(count_non_lazy)

DataFrame = DataFrameOriginal

LazyFrameOriginal.schema = property(schema_lazy)
LazyFrameOriginal.isEmpty = isEmpty_lazy
LazyFrameOriginal.count = property(count_lazy)
LazyFrameOriginal.columns = property(columns_lazy)
LazyFrameOriginal.withColumn = withColumn
LazyFrameOriginal.withColumns = withColumns
LazyFrameOriginal.hint = hint
LazyFrameOriginal.repartition = repartition
LazyFrameOriginal.coalesce = coalesce
LazyFrameOriginal.persist = persist
LazyFrameOriginal.distinct = distinct
LazyFrameOriginal.dropDuplicates = dropDuplicates
LazyFrameOriginal.drop_duplicates = drop_duplicates
LazyFrameOriginal.dropna = dropna
LazyFrameOriginal.dropnulls = dropnulls
LazyFrameOriginal.join = join
LazyFrameOriginal.unionByName = unionByName
LazyFrameOriginal.crossJoin = crossJoin
LazyFrameOriginal.checkpoint = checkpoint
LazyFrameOriginal.localCheckpoint = localCheckpoint
LazyFrameOriginal.groupBy = groupBy

LazyFrame = LazyFrameOriginal
