"""Polyspark DataFrame methods for Polars DataFrame and LazyFrame."""

import re
from typing import Any

from polars import DataFrame as DataFrameOriginal
from polars import LazyFrame as LazyFrameOriginal
from polars import col, concat, lit
from polars.config import Config
from polars.expr import Expr

from .functions import _str_to_col


def return_self(
    self: DataFrameOriginal | LazyFrameOriginal,
    *_args: Any,
    **_kwargs: Any,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self


DataFrameOriginal._original_drop = DataFrameOriginal.drop


def drop_strict(self: DataFrameOriginal, *cols: str, strict: bool = True) -> DataFrameOriginal:
    # Filter out columns that don't exist
    existing_cols = [col for col in cols if col in self.columns]
    if not existing_cols:
        # No columns to drop, return self unchanged
        return self
    return self._original_drop(*existing_cols, strict=strict)


DataFrameOriginal.drop = drop_strict

LazyFrameOriginal._original_drop = LazyFrameOriginal.drop


def drop_strict_lazy(self: LazyFrameOriginal, *cols: str, strict: bool = True) -> LazyFrameOriginal:
    # Filter out columns that don't exist
    existing_cols = [col for col in cols if col in self.collect_schema().names()]
    if not existing_cols:
        # No columns to drop, return self unchanged
        return self
    return self._original_drop(*existing_cols, strict=strict)


LazyFrameOriginal.drop = drop_strict_lazy


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
        # Ensure expr is properly aliased
        try:
            aliased_expr = expr.alias(name)
        except AttributeError:
            # If expr doesn't have alias method, wrap it in a column expression
            import polars as pl

            aliased_expr = pl.col(expr).alias(name) if isinstance(expr, str) else expr.alias(name)
        return self.with_columns(aliased_expr)
    if kwargs:
        # Keyword argument form: withColumns(col1=expr1, col2=expr2)
        columns = [expr.alias(name) for name, expr in kwargs.items()]
        return self.with_columns(columns)
    msg = (
        "withColumns expects either a dictionary, keyword arguments, "
        "or exactly two positional arguments (name, expr)"
    )
    raise ValueError(msg)


def withColumn(
    self: DataFrameOriginal | LazyFrameOriginal,
    name: str,
    expr: Expr,
) -> DataFrameOriginal | LazyFrameOriginal:
    return self.withColumns({name: expr})


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
    return self.unique(subset=subset, keep="first")


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
        "left_outer": "left",
        "leftouter": "left",
        "right_outer": "right",
        "rightouter": "right",
        "left_anti": "anti",
        "leftanti": "anti",
        "full": "outer",
        "fullouter": "outer",
        "full_outer": "outer",
    }
    how = mapping.get(how, how)
    coalesce = True if how == "full" else None
    if isinstance(on, Expr) and not on.meta.is_column_selection():
        if how != "inner":
            msg = "Join on expressions with predicates is only available in inner joins."
            raise NotImplementedError(
                msg,
            )
        return self.join_where(other, on)
    on = [on] if isinstance(on, str | Expr) else on
    on = [_str_to_col(c) if isinstance(c, str) else c for c in on]
    if not all(isinstance(col, str | Expr) and col.meta.is_column_selection() for col in on):
        msg = "Join columns must be strings or column expressions and not predicates."
        raise ValueError(msg)
    return self.join(other, on=on, how=how, coalesce=coalesce, suffix="_r_polyspark")


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


def groupBy(self: DataFrameOriginal | LazyFrameOriginal, *cols: Any) -> Any:
    # Convert ColExtension objects to their underlying expressions
    return self.group_by(*cols)


def agg(
    self: DataFrameOriginal | LazyFrameOriginal,
    *expr: Any,
) -> DataFrameOriginal | LazyFrameOriginal:
    if len(expr) == 1 and isinstance(expr[0], dict):
        expr_dict = expr[0]
        # Check all keys and values are strings
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in expr_dict.items()):
            msg = "All keys and values in the aggregation dictionary must be strings."
            raise ValueError(msg)
        # Allowed aggregation functions
        allowed_aggs = {"min", "max", "mean", "sum", "count", "first", "last"}
        if not all(v in allowed_aggs for v in expr_dict.values()):
            msg = f"Aggregation functions must be one of {allowed_aggs}."
            raise ValueError(msg)
        expr = [getattr(col(k), v)().alias(f"{v}({k})") for k, v in expr_dict.items()]
    return (
        self.group_by(lit(1).alias("agg_polyspark"))
        .agg(
            *expr,
        )
        .drop("agg_polyspark")
    )


DataFrameOriginal.agg = agg
LazyFrameOriginal.agg = agg


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


def show(
    self: DataFrameOriginal | LazyFrameOriginal,
    n: int = 20,
    truncate: bool = True,
    vertical: bool = False,
) -> None:
    if isinstance(self, LazyFrameOriginal):
        self = self.collect()
    if truncate:
        Config.set_tbl_width_chars(20)
    else:
        Config.set_fmt_str_lengths(9999)
    if vertical:
        msg = "Vertical display is not implemented in Polars DataFrame."
        raise NotImplementedError(msg)
    if n < 0:
        msg = "n must be a non-negative integer."
        raise ValueError(msg)


def not_implemented(self: Any, *args: Any, **kwargs: Any) -> None:
    msg = "This method is not implemented in Polars DataFrame."
    raise NotImplementedError(msg)


def colRegex(
    self: DataFrameOriginal | LazyFrameOriginal,
    colName: str,
) -> DataFrameOriginal | LazyFrameOriginal:
    pattern = re.compile(colName)
    columns = [c for c in self.columns if pattern.search(c)]
    return self.select(columns)


def withColumnRenamed(
    self: DataFrameOriginal | LazyFrameOriginal,
    existing: str,
    new: str,
) -> DataFrameOriginal | LazyFrameOriginal:
    dict = {existing: new}
    return self.rename(dict, strict=False)


def withColumnsRenamed(
    self: DataFrameOriginal | LazyFrameOriginal,
    colsMap: dict[str, str],
) -> DataFrameOriginal | LazyFrameOriginal:
    return self.rename(colsMap, strict=False)


DataFrameOriginal.withColumn = withColumn

DataFrameOriginal.withColumns = withColumns
DataFrameOriginal.hint = return_self
DataFrameOriginal.alias = return_self
DataFrameOriginal.repartition = return_self
DataFrameOriginal.coalesce = return_self
DataFrameOriginal.persist = persist
DataFrameOriginal.distinct = distinct
DataFrameOriginal.dropDuplicates = dropDuplicates
DataFrameOriginal.drop_duplicates = dropDuplicates
DataFrameOriginal.dropna = dropna
DataFrameOriginal.dropnulls = dropnulls
DataFrameOriginal.join = join
DataFrameOriginal.unionByName = unionByName
DataFrameOriginal.crossJoin = crossJoin
DataFrameOriginal.checkpoint = return_self
DataFrameOriginal.localCheckpoint = return_self
DataFrameOriginal.groupBy = groupBy
DataFrameOriginal.show = show
DataFrameOriginal.approxQuantile = not_implemented
DataFrameOriginal.asTable = not_implemented
DataFrameOriginal.corr = not_implemented
DataFrameOriginal.cov = not_implemented
DataFrameOriginal.crosstab = not_implemented
DataFrameOriginal.cube = not_implemented
DataFrameOriginal.withColumnRenamed = withColumnRenamed
DataFrameOriginal.withColumnsRenamed = withColumnsRenamed

DataFrameOriginal.isEmpty = isEmpty_non_lazy
DataFrameOriginal.count = property(count_non_lazy)
DataFrameOriginal.colRegex = colRegex

DataFrame = DataFrameOriginal

LazyFrameOriginal.approxQuantile = not_implemented
LazyFrameOriginal.asTable = not_implemented
LazyFrameOriginal.cube = not_implemented
LazyFrameOriginal.crosstab = not_implemented
LazyFrameOriginal.cov = not_implemented
LazyFrameOriginal.corr = not_implemented
LazyFrameOriginal.schema = property(schema_lazy)
LazyFrameOriginal.isEmpty = isEmpty_lazy
LazyFrameOriginal.count = property(count_lazy)
LazyFrameOriginal.columns = property(columns_lazy)
LazyFrameOriginal.withColumn = withColumn
LazyFrameOriginal.withColumns = withColumns
LazyFrameOriginal.hint = return_self
LazyFrameOriginal.repartition = return_self
LazyFrameOriginal.coalesce = return_self
LazyFrameOriginal.persist = persist
LazyFrameOriginal.distinct = distinct
LazyFrameOriginal.dropDuplicates = dropDuplicates
LazyFrameOriginal.drop_duplicates = dropDuplicates
LazyFrameOriginal.dropna = dropna
LazyFrameOriginal.dropnulls = dropnulls
LazyFrameOriginal.join = join
LazyFrameOriginal.unionByName = unionByName
LazyFrameOriginal.crossJoin = crossJoin
LazyFrameOriginal.checkpoint = return_self
LazyFrameOriginal.localCheckpoint = return_self
LazyFrameOriginal.groupBy = groupBy
LazyFrameOriginal.colRegex = colRegex
LazyFrameOriginal.show = show
LazyFrameOriginal.withColumnRenamed = withColumnRenamed
LazyFrameOriginal.withColumnsRenamed = withColumnsRenamed

LazyFrame = LazyFrameOriginal
