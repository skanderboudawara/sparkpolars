"""Polyspark DataFrame methods for Polars DataFrame and LazyFrame."""

import os
import random
import re
from pathlib import Path
from typing import Any

import polars as pl
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame as PolarsLazyDataFrame
from polars import col, concat, lit
from polars._utils.parse import parse_into_list_of_expressions
from polars.config import Config
from polars.expr import Expr

from .functions import _str_to_col

# ── Module-level SQL contexts (temp-view registry) ────────────────────────────
_local_sql_ctx: pl.SQLContext = pl.SQLContext()
_global_sql_ctx: pl.SQLContext = pl.SQLContext()


# ── Save originals before any override ────────────────────────────────────────
PolarsDataFrame._original_drop = PolarsDataFrame.drop
PolarsLazyDataFrame._original_drop = PolarsLazyDataFrame.drop
PolarsDataFrame._original_join_df = PolarsDataFrame.join
PolarsLazyDataFrame._original_join_lz = PolarsLazyDataFrame.join
PolarsDataFrame._original_describe = PolarsDataFrame.describe
PolarsLazyDataFrame._original_describe = PolarsLazyDataFrame.describe
PolarsDataFrame._original_filter = PolarsDataFrame.filter
PolarsLazyDataFrame._original_filter = PolarsLazyDataFrame.filter
PolarsDataFrame._original_sort = PolarsDataFrame.sort
PolarsLazyDataFrame._original_sort = PolarsLazyDataFrame.sort
PolarsDataFrame._original_sample = PolarsDataFrame.sample


# ── Helpers ───────────────────────────────────────────────────────────────────
def return_self(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *_args: Any,
    **_kwargs: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    return self


def not_implemented(self: Any, *args: Any, **kwargs: Any) -> None:
    msg = "This method is not implemented in Polars DataFrame."
    raise NotImplementedError(msg)


def _schema_names(self: PolarsDataFrame | PolarsLazyDataFrame) -> list[str]:
    if isinstance(self, PolarsLazyDataFrame):
        return self.collect_schema().names()
    return self.columns


# ── drop ──────────────────────────────────────────────────────────────────────
def drop_strict(self: PolarsDataFrame, *cols: str, strict: bool = True) -> PolarsDataFrame:
    """Drop columns, silently ignoring non-existent ones.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> from polars.testing import assert_frame_equal
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> assert_frame_equal(df.drop("nonexistent"), df)
    """
    existing_cols = [c for c in cols if c in self.columns]
    if not existing_cols:
        return self
    return self._original_drop(*existing_cols, strict=strict)


PolarsDataFrame.drop = drop_strict


def drop_strict_lazy(self: PolarsLazyDataFrame, *cols: str, strict: bool = True) -> PolarsLazyDataFrame:
    existing_cols = [c for c in cols if c in self.collect_schema().names()]
    if not existing_cols:
        return self
    return self._original_drop(*existing_cols, strict=strict)


PolarsLazyDataFrame.drop = drop_strict_lazy


# ── filter / where ────────────────────────────────────────────────────────────
def filter_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    condition: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Filter rows by a condition (Spark-compatible).

    Supports both Polars expressions and SQL string conditions.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.filter(pl.col("a") > 2).height
        3
        >>> df.filter("a > 2").height
        3
        >>> df.where(pl.col("a") > 2).height
        3
    """
    if isinstance(condition, str):
        condition = pl.sql_expr(condition)
    return self._original_filter(condition)


# ── sort / orderBy / sortWithinPartitions ─────────────────────────────────────
def sort_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *cols: Any,
    ascending: bool | list[bool] = True,
    by: Any = None,
    descending: bool | list[bool] | None = None,
    **polars_kwargs: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Sort rows by one or more columns (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.sort("a", ascending=False)["a"][0]
        5
        >>> df.sort("a", ascending=True)["a"][0]
        1
        >>> from polars.testing import assert_frame_equal
        >>> assert_frame_equal(df.sort("a", ascending=False), df.orderBy("a", ascending=False))
        >>> df.sort(["a"], ascending=False)["a"][0]
        5
        >>> df.sortWithinPartitions("a", ascending=True)["a"][0]
        1
    """
    # Native Polars internal call (e.g. concat("align") or DataFrame.sort → LazyFrame.sort)
    if by is not None or descending is not None or polars_kwargs:
        effective_by = by if by is not None else list(cols)
        kw: dict[str, Any] = {}
        if descending is not None:
            kw["descending"] = descending
        kw.update(polars_kwargs)
        return self._original_sort(effective_by, **kw)
    # Spark-style call
    if not cols:
        return self
    flat_cols = list(cols[0]) if len(cols) == 1 and isinstance(cols[0], list) else list(cols)
    asc_flag = ascending
    if isinstance(asc_flag, bool):
        desc: bool | list[bool] = not asc_flag
    else:
        desc = [not a for a in asc_flag]
    sort_cols = [_str_to_col(c) if isinstance(c, str) else c for c in flat_cols]
    # Sort via LazyFrame directly to avoid the DF.sort → LF.sort_spark recursion
    lf_sorted = self.lazy()._original_sort(sort_cols, descending=desc)
    if isinstance(self, PolarsDataFrame):
        try:
            from polars import QueryOptFlags
            return lf_sorted.collect(optimizations=QueryOptFlags._eager())
        except Exception:
            return lf_sorted.collect(_eager=True)
    return lf_sorted


# ── sample ────────────────────────────────────────────────────────────────────
def sample_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    withReplacement: bool = False,
    fraction: float | None = None,
    seed: int | None = None,
) -> PolarsDataFrame:
    """Sample a fraction of rows (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.sample(fraction=0.4, seed=42)
        >>> isinstance(result, pl.DataFrame)
        True
        >>> 1 <= result.height <= df.height
        True
        >>> df.sample(withReplacement=True, fraction=1.0, seed=0).height == df.height
        True
    """
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return df._original_sample(
        fraction=fraction if fraction is not None else 1.0,
        with_replacement=withReplacement,
        seed=seed,
    )


# ── replace ───────────────────────────────────────────────────────────────────
def _value_matches_dtype(value: Any, dtype_name: str) -> bool:
    """Return True if a Python value is type-compatible with a Polars dtype name."""
    if isinstance(value, bool):
        return "Boolean" in dtype_name
    if isinstance(value, int):
        return any(t in dtype_name for t in ("Int", "UInt", "Float"))
    if isinstance(value, float):
        return any(t in dtype_name for t in ("Float", "Int", "UInt"))
    if isinstance(value, str):
        return any(t in dtype_name for t in ("String", "Utf8", "Categorical", "Enum"))
    return True


def replace_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    to_replace: Any,
    value: Any = None,
    subset: list[str] | None = None,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Replace values in the DataFrame (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.replace(1, 99)
        >>> 99 in result["a"].to_list()
        True
        >>> 1 not in result["a"].to_list()
        True
        >>> result2 = df.replace(1, 99, subset=["a"])
        >>> 99 in result2["a"].to_list()
        True
        >>> result2["b"].to_list() == df["b"].to_list()
        True
        >>> result3 = df.replace({"x": "X", "y": "Y"}, subset=["b"])
        >>> "X" in result3["b"].to_list()
        True
        >>> "x" not in result3["b"].to_list()
        True
    """
    all_cols = _schema_names(self)
    target_cols = subset if subset is not None else all_cols
    mapping = to_replace if isinstance(to_replace, dict) else {to_replace: value}
    schema = self.collect_schema() if isinstance(self, PolarsLazyDataFrame) else self.schema
    exprs = []
    for c in all_cols:
        if c in target_cols:
            expr_c = pl.col(c)
            dtype_name = type(schema[c]).__name__
            for old, new in mapping.items():
                if _value_matches_dtype(old, dtype_name):
                    expr_c = pl.when(pl.col(c) == old).then(pl.lit(new)).otherwise(expr_c)
            exprs.append(expr_c.alias(c))
        else:
            exprs.append(pl.col(c))
    return self.with_columns(exprs)


# ── withColumns / withColumn ──────────────────────────────────────────────────
def withColumns(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *args: Any,
    **kwargs: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Add or replace columns (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.withColumns({"d": pl.col("a") + 10})["d"][0]
        11
        >>> df.withColumns(d=pl.col("a") + 10)["d"][0]
        11
    """
    for arg in args:
        key, val = next(iter(arg.items())) if isinstance(arg, dict) else (None, arg)
        if hasattr(val, "_explode_marker"):
            col_outer = val._explode_outer
            if col_outer:
                self = self.with_columns(val.list.drop_nulls().alias(key)).explode(key)
            else:
                self = self.explode(key)
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        columns_dict = args[0]
        columns = [expr.alias(name) for name, expr in columns_dict.items()]
        return self.with_columns(columns)
    if len(args) == 2 and not kwargs:
        name, expr = args
        try:
            aliased_expr = expr.alias(name)
        except AttributeError:
            aliased_expr = pl.col(expr).alias(name) if isinstance(expr, str) else expr.alias(name)
        return self.with_columns(aliased_expr)
    if kwargs:
        columns = [expr.alias(name) for name, expr in kwargs.items()]
        return self.with_columns(columns)
    msg = (
        "withColumns expects either a dictionary, keyword arguments, "
        "or exactly two positional arguments (name, expr)"
    )
    raise ValueError(msg)


def withColumn(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    name: str,
    expr: Expr,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Add or replace a single column (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.withColumn("d", pl.col("a") * 2)
        >>> "d" in result.columns
        True
        >>> result["d"][0]
        2
    """
    return self.withColumns({name: expr})


# ── persist ───────────────────────────────────────────────────────────────────
def persist(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *_args: Any,
    **_kwargs: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Persist the DataFrame (delegates to cache).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.persist() is df
        True
    """
    return self.cache()


# ── distinct / dropDuplicates ─────────────────────────────────────────────────
def distinct(
    self: PolarsDataFrame | PolarsLazyDataFrame,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Return distinct rows.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> duped = pl.concat([df, df])
        >>> duped.distinct().height == df.height
        True
    """
    return self.unique()


def dropDuplicates(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    subset: list[str] | None = None,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Drop duplicate rows, optionally considering only a subset of columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.dropDuplicates(subset=["b"]).height
        3
    """
    return self.unique(subset=subset, keep="first")


# ── dropna / dropnulls ────────────────────────────────────────────────────────
def dropna(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    how: str = "any",
    thresh: int | None = None,
    subset: list[str] | None = None,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Drop rows containing NaN values (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1.0, float("nan"), 3.0]})
        >>> df.dropna().height
        2
    """
    if how == "any" and thresh is None:
        return self.drop_nans(subset)
    msg = "dropna with 'how' or 'thresh' parameters is not implemented."
    raise NotImplementedError(msg)


def dropnulls(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    how: str = "any",
    thresh: int | None = None,
    subset: list[str] | None = None,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Drop rows containing null values.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [None, "y", None], "c": [1.0, None, 3.0]})
        >>> df.dropnulls().height
        0
    """
    if how == "any" and thresh is None:
        return self.drop_nulls(subset)
    msg = "dropna with 'how' or 'thresh' parameters is not implemented."
    raise NotImplementedError(msg)


# ── join ──────────────────────────────────────────────────────────────────────
_JOIN_TYPE_MAPPING = {
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


def join_altred_df(
    self: PolarsDataFrame,
    other: PolarsDataFrame,
    on: str | list[str] | Expr | None = None,
    how: str = "inner",
    *args: Any,
    **kwargs: Any,
) -> PolarsDataFrame:
    """Join two DataFrames with Spark-compatible join type aliases.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> other = pl.DataFrame({"a": [3, 4, 5, 6, 7], "b": ["z", "x", "y", "a", "b"], "c": [3.0, 4.0, 5.0, 6.0, 7.0]})
        >>> df.join(other, on="a", how="inner").height
        3
        >>> set(df.join(other, on="a", how="inner")["a"].to_list()) == {3, 4, 5}
        True
        >>> df.join(other, on="a", how="left_outer").height
        5
        >>> df.join(other, on="a", how="leftouter").height
        5
        >>> df.join(other, on="a", how="right_outer").height
        5
        >>> df.join(other, on="a", how="rightouter").height
        5
        >>> df.join(other, on="a", how="full_outer").height
        7
        >>> df.join(other, on="a", how="full").height
        7
        >>> df.join(other, on="a", how="fullouter").height
        7
        >>> df.join(other, on="a", how="left_anti").height
        2
        >>> set(df.join(other, on="a", how="left_anti")["a"].to_list()) == {1, 2}
        True
        >>> df.join(other, on="a", how="leftanti").height
        2
        >>> df.join(other, on="a", how="semi").height
        3
        >>> set(df.join(other, on="a", how="semi")["a"].to_list()) == {3, 4, 5}
        True
        >>> df.join(other, how="cross").height
        25
        >>> isinstance(df.join(other, on=["a", "b", "c"], how="inner"), pl.DataFrame)
        True
    """
    how = _JOIN_TYPE_MAPPING.get(how, how)
    coalesce = True if how == "full" else None
    if isinstance(on, Expr) and not on.meta.is_column_selection():
        if how != "inner":
            msg = "Join on expressions with predicates is only available in inner joins."
            raise NotImplementedError(msg)
        return self.join_where(other, on)
    if on:
        if how == "cross":
            msg = "cross join should not pass join keys"
            raise ValueError(msg)
        on = [on] if isinstance(on, str | Expr) else on
        on = [_str_to_col(c) if isinstance(c, str) else c for c in on]
        if not all(isinstance(c, str | Expr) and c.meta.is_column_selection() for c in on):
            msg = "Join columns must be strings or column expressions and not predicates."
            raise ValueError(msg)
    self = self.lazy()._original_join_lz(
        other=other.lazy(),
        on=on,
        how=how,
        suffix="_r_polyspark",
        coalesce=coalesce,
    )
    try:
        from polars import QueryOptFlags
        return self.collect(optimizations=QueryOptFlags._eager())
    except Exception:
        return self.collect(_eager=True)


PolarsDataFrame.join = join_altred_df


def join_altred_lz(
    self: PolarsLazyDataFrame,
    other: PolarsLazyDataFrame,
    on: str | list[str] | Expr | None = None,
    how: str = "inner",
    *args: Any,
    **kwargs: Any,
) -> PolarsLazyDataFrame:
    how = _JOIN_TYPE_MAPPING.get(how, how)
    coalesce = True if how == "full" else None
    if isinstance(on, Expr) and not on.meta.is_column_selection():
        if how != "inner":
            msg = "Join on expressions with predicates is only available in inner joins."
            raise NotImplementedError(msg)
        return self.join_where(other, on)
    return self._original_join_lz(
        other=other,
        on=on,
        how=how,
        suffix="_r_polyspark",
        coalesce=coalesce,
    )


PolarsLazyDataFrame.join = join_altred_lz


# ── union / unionByName / crossJoin ──────────────────────────────────────────
def union_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Union two DataFrames vertically (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> other = pl.DataFrame({"a": [3, 4, 5, 6, 7], "b": ["z", "x", "y", "a", "b"], "c": [3.0, 4.0, 5.0, 6.0, 7.0]})
        >>> df.union(other).height == df.height + other.height
        True
        >>> df.unionAll(other).height == df.height + other.height
        True
    """
    return concat([self, other], how="vertical")


def unionByName(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
    allowMissingColumns: bool = False,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Union two DataFrames by column name (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.unionByName(df).height
        10
    """
    return concat(
        [self, other],
        how="diagonal_relaxed" if allowMissingColumns else "vertical",
    )


def crossJoin(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Cross join two DataFrames (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> small = df.head(2)
        >>> small.crossJoin(small).height
        4
    """
    return self.join(other, how="cross")


# ── SparkGroupBy wrapper ──────────────────────────────────────────────────────

_DICT_AGG_MAP = {
    "sum": lambda c: c.sum().alias(f"sum({c.meta.output_name()})"),
    "min": lambda c: c.min().alias(f"min({c.meta.output_name()})"),
    "max": lambda c: c.max().alias(f"max({c.meta.output_name()})"),
    "avg": lambda c: c.mean().alias(f"avg({c.meta.output_name()})"),
    "mean": lambda c: c.mean().alias(f"avg({c.meta.output_name()})"),
    "count": lambda c: c.count().alias(f"count({c.meta.output_name()})"),
    "first": lambda c: c.first().alias(f"first({c.meta.output_name()})"),
    "last": lambda c: c.last().alias(f"last({c.meta.output_name()})"),
    "stddev": lambda c: c.std().alias(f"stddev({c.meta.output_name()})"),
    "variance": lambda c: c.var().alias(f"variance({c.meta.output_name()})"),
    "median": lambda c: c.median().alias(f"median({c.meta.output_name()})"),
}


def _resolve_agg_exprs(*exprs: Any) -> list[Expr]:
    """Flatten and normalise aggregation expressions.

    Accepts:
    - Polars Expr objects (pass through)
    - Dicts  {"col": "fn"} or {"col": ["fn1", "fn2"]}
    """
    result: list[Expr] = []
    for e in exprs:
        if isinstance(e, dict):
            for col_name, fn_spec in e.items():
                if not isinstance(col_name, str):
                    msg = "All keys and values in the aggregation dictionary must be strings."
                    raise ValueError(msg)
                fns = [fn_spec] if isinstance(fn_spec, str) else list(fn_spec)
                for fn_name in fns:
                    builder = _DICT_AGG_MAP.get(fn_name)
                    if builder is None:
                        msg = f"Aggregation functions must be one of {list(_DICT_AGG_MAP)}. Got: '{fn_name}'"
                        raise ValueError(msg)
                    result.append(builder(col(col_name)))
        elif isinstance(e, Expr):
            result.append(e)
        else:
            msg = f"Expected Expr or dict, got {type(e).__name__}"
            raise TypeError(msg)
    return result


class SparkGroupBy:
    """PySpark-compatible GroupBy wrapper around Polars GroupBy / LazyGroupBy.

    Supports:
    - ``agg(sf.sum("sal"), sf.avg("sal"))``        — Expr API
    - ``agg({"sal": "sum", "bonus": "max"})``       — dict API
    - ``agg({"sal": ["sum", "avg"]})``              — multi-fn dict
    - ``count()``  ``min("col")``  ``max("col")`` etc. as convenience methods
    """

    def __init__(
        self,
        df: PolarsDataFrame | PolarsLazyDataFrame,
        *partition_cols: Any,
    ) -> None:
        self._df = df
        self._cols = partition_cols

    def _group_by(self) -> Any:
        if self._cols:
            return self._df.group_by(*self._cols)
        # No partition columns → global aggregate (group by constant)
        return None

    def agg(self, *exprs: Any) -> PolarsDataFrame | PolarsLazyDataFrame:
        flat = _resolve_agg_exprs(*exprs)
        gb = self._group_by()
        if gb is None:
            # global aggregate
            return (
                self._df.group_by(lit(1).alias("_spark_agg_key"))
                .agg(*flat)
                .drop("_spark_agg_key")
            )
        return gb.agg(*flat)

    # ── convenience aggregation methods ──────────────────────────────────────

    def count(self) -> PolarsDataFrame | PolarsLazyDataFrame:
        """Count the number of rows per group.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})
            >>> df.groupBy("dept").count().sort("dept")["count"].to_list()
            [2, 1]
        """
        agg_expr = pl.len().alias("count")
        gb = self._group_by()
        if gb is None:
            return (
                self._df.group_by(lit(1).alias("_spark_agg_key"))
                .agg(agg_expr)
                .drop("_spark_agg_key")
            )
        return gb.agg(agg_expr)

    def sum(self, *cols: str) -> PolarsDataFrame | PolarsLazyDataFrame:
        """Compute sum of specified columns per group.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})
            >>> df.groupBy("dept").sum("sal").sort("dept")["sum(sal)"].to_list()
            [300, 300]
        """
        exprs = [col(c).sum().alias(f"sum({c})") for c in cols] if cols else [pl.all().sum()]
        return self.agg(*exprs)

    def avg(self, *cols: str) -> PolarsDataFrame | PolarsLazyDataFrame:
        """Compute average of specified columns per group.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})
            >>> df.groupBy("dept").avg("sal").sort("dept")["avg(sal)"].to_list()
            [150.0, 300.0]
        """
        exprs = [col(c).mean().alias(f"avg({c})") for c in cols] if cols else [pl.all().mean()]
        return self.agg(*exprs)

    mean = avg

    def min(self, *cols: str) -> PolarsDataFrame | PolarsLazyDataFrame:
        """Compute minimum of specified columns per group.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})
            >>> df.groupBy("dept").min("sal").sort("dept")["min(sal)"].to_list()
            [100, 300]
        """
        exprs = [col(c).min().alias(f"min({c})") for c in cols] if cols else [pl.all().min()]
        return self.agg(*exprs)

    def max(self, *cols: str) -> PolarsDataFrame | PolarsLazyDataFrame:
        """Compute maximum of specified columns per group.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})
            >>> df.groupBy("dept").max("sal").sort("dept")["max(sal)"].to_list()
            [200, 300]
        """
        exprs = [col(c).max().alias(f"max({c})") for c in cols] if cols else [pl.all().max()]
        return self.agg(*exprs)

    def __getattr__(self, name: str) -> Any:
        """Delegate anything else (e.g. .map_groups) to the underlying GroupBy."""
        return getattr(self._group_by(), name)


# ── groupBy / agg ─────────────────────────────────────────────────────────────
def groupBy(self: PolarsDataFrame | PolarsLazyDataFrame, *cols: Any) -> SparkGroupBy:
    """Group by columns and return a SparkGroupBy wrapper.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> import src.sparkpolars.polyspark.sql.functions as sf  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.groupBy("b").agg(pl.col("a").sum())
        >>> isinstance(result, pl.DataFrame)
        True
        >>> "b" in result.columns
        True
        >>> dept_df = pl.DataFrame({"dept": ["A", "A", "B", "B"], "sal": [100, 200, 300, 400]})
        >>> result = dept_df.groupBy("dept").agg(sf.sum("sal")).sort("dept")
        >>> result.columns
        ['dept', 'sum(sal)']
        >>> result["sum(sal)"].to_list()
        [300, 700]
        >>> result = dept_df.groupBy("dept").agg({"sal": "sum"}).sort("dept")
        >>> result["sum(sal)"].to_list()
        [300, 700]
        >>> dept_df2 = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})
        >>> result = dept_df2.groupBy("dept").agg(sf.sum("sal"), sf.avg("sal"), sf.count("sal")).sort("dept")
        >>> "sum(sal)" in result.columns and "avg(sal)" in result.columns and "count(sal)" in result.columns
        True
        >>> result = dept_df2.groupBy("dept").agg({"sal": ["sum", "avg"]}).sort("dept")
        >>> "sum(sal)" in result.columns and "avg(sal)" in result.columns
        True
        >>> result = dept_df.groupBy("dept").count().sort("dept")
        >>> result["count"].to_list()
        [2, 2]
        >>> result = pl.DataFrame({"sal": [100, 200, 300]}).groupBy().agg(sf.sum("sal"))
        >>> result["sum(sal)"][0]
        600
    """
    return SparkGroupBy(self, *cols)


def agg(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *exprs: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Global aggregate (no groupBy) -- equivalent to groupBy().agg(...).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> import src.sparkpolars.polyspark.sql.functions as sf  # noqa: F401
        >>> df = pl.DataFrame({"sal": [100, 200, 300]})
        >>> df.agg(sf.sum("sal"))["sum(sal)"][0]
        600
        >>> df.agg({"sal": "sum"})["sum(sal)"][0]
        600
    """
    flat = _resolve_agg_exprs(*exprs)
    return (
        self.group_by(lit(1).alias("_spark_agg_key"))
        .agg(*flat)
        .drop("_spark_agg_key")
    )


PolarsDataFrame.agg = agg
PolarsLazyDataFrame.agg = agg


# ── schema / count / isEmpty / columns helpers ────────────────────────────────
def schema_lazy(self: PolarsLazyDataFrame) -> Any:
    """Return schema of a LazyFrame.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> lf = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.0, 2.0, 3.0]}).lazy()
        >>> "a" in lf.schema
        True
    """
    return self.collect_schema()


def isEmpty_lazy(self: PolarsLazyDataFrame) -> bool:
    """Return True if the LazyFrame is empty.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
        >>> lf.isEmpty()
        False
    """
    return self.limit(1).collect().is_empty()


def count_lazy(self: PolarsLazyDataFrame) -> int:
    """Return the row count of a LazyFrame.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> lf = pl.DataFrame({"a": [1, 2, 3, 4, 5]}).lazy()
        >>> lf.count
        5
    """
    return self.collect().height


def columns_lazy(self: PolarsLazyDataFrame) -> list[str]:
    """Return column names of a LazyFrame.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> lf = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy()
        >>> lf.columns
        ['a', 'b', 'c']
    """
    return self.collect_schema().names()


def isEmpty_non_lazy(self: PolarsDataFrame) -> bool:
    """Return True if the DataFrame is empty.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> df.isEmpty()
        False
        >>> pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}).isEmpty()
        True
    """
    return self.is_empty()


def count_non_lazy(self: PolarsDataFrame) -> int:
    """Return the row count of a DataFrame.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> df.count
        5
    """
    return self.height


# ── show ──────────────────────────────────────────────────────────────────────
def show(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    n: int = 20,
    truncate: bool = True,
    vertical: bool = False,
) -> None:
    """Configure display settings for the DataFrame (PySpark compatibility shim).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.show()
    """
    if isinstance(self, PolarsLazyDataFrame):
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


# ── colRegex / withColumnRenamed / withColumnsRenamed ─────────────────────────
def colRegex(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    colName: str,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Select columns matching a regex pattern.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.colRegex("^[ab]$").columns
        ['a', 'b']
    """
    pattern = re.compile(colName)
    columns = [c for c in self.columns if pattern.search(c)]
    return self.select(columns)


def withColumnRenamed(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    existing: str,
    new: str,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Rename a single column (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.withColumnRenamed("a", "aa")
        >>> "aa" in result.columns
        True
        >>> "a" not in result.columns
        True
    """
    return self.rename({existing: new}, strict=False)


def withColumnsRenamed(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    colsMap: dict[str, str],
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Rename multiple columns (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.withColumnsRenamed({"a": "aa", "b": "bb"}).columns
        ['aa', 'bb', 'c']
    """
    return self.rename(colsMap, strict=False)


# ── describe ──────────────────────────────────────────────────────────────────
def describe(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *cols: str,
    **kwargs: Any,
) -> PolarsDataFrame:
    """Compute summary statistics (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.describe()
        >>> isinstance(result, pl.DataFrame)
        True
        >>> "statistic" in result.columns
        True
        >>> "a" in result.columns
        True
        >>> result2 = df.describe("a", "c")
        >>> "b" not in result2.columns
        True
        >>> "a" in result2.columns and "c" in result2.columns
        True
    """
    # Internal Polars call from DataFrame.describe → LazyFrame.describe(percentiles=...)
    # Delegate straight to the saved native LazyFrame implementation to avoid recursion.
    if kwargs:
        lf = self if isinstance(self, PolarsLazyDataFrame) else self.lazy()
        return lf._original_describe(**kwargs)
    # Spark-style call: optional column subset
    if isinstance(self, PolarsLazyDataFrame):
        df = self.collect()
    else:
        df = self
    if cols:
        df = df.select(list(cols))
    return df.lazy()._original_describe()


# ── summary ───────────────────────────────────────────────────────────────────
def summary_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *statistics: str,
) -> PolarsDataFrame:
    """Compute summary statistics (Spark-compatible alias for describe).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.summary()
        >>> isinstance(result, pl.DataFrame)
        True
        >>> "statistic" in result.columns
        True
    """
    return describe(self)


# ── fillna ────────────────────────────────────────────────────────────────────
def fillna(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    value: Any,
    subset: list[str] | None = None,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Fill null values (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [None, "y", None], "c": [1.0, None, 3.0]})
        >>> df.fillna("FILLED")["b"].null_count()
        0
        >>> result = df.fillna(0.0, subset=["c"])
        >>> result["c"].null_count()
        0
        >>> result["b"].null_count() == df["b"].null_count()
        True
    """
    if subset:
        return self.with_columns([pl.col(c).fill_null(value) for c in subset])
    return self.fill_null(value)


# ── first ─────────────────────────────────────────────────────────────────────
def first_row(
    self: PolarsDataFrame | PolarsLazyDataFrame,
) -> dict | None:
    """Return the first row as a dict, or None if empty.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.first()
        >>> isinstance(result, dict)
        True
        >>> result["a"]
        1
        >>> pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}).first() is None
        True
    """
    if isinstance(self, PolarsLazyDataFrame):
        df = self.limit(1).collect()
    else:
        df = self.head(1)
    rows = df.to_dicts()
    return rows[0] if rows else None


# ── intersect ─────────────────────────────────────────────────────────────────
def intersect(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Return rows that exist in both DataFrames (set intersection).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> other = pl.DataFrame({"a": [3, 4, 5, 6, 7], "b": ["z", "x", "y", "a", "b"], "c": [3.0, 4.0, 5.0, 6.0, 7.0]})
        >>> result = df.intersect(other)
        >>> isinstance(result, pl.DataFrame)
        True
        >>> set(result["a"].to_list()).issubset({3, 4, 5})
        True
    """
    cols = _schema_names(self)
    return self.join(other, on=cols, how="semi").unique()


# ── intersectAll ──────────────────────────────────────────────────────────────
def intersectAll(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Return rows that exist in both DataFrames preserving duplicates.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df1 = pl.DataFrame({"C1": ["a", "a", "b", "c"], "C2": [1, 1, 3, 4]})
        >>> df2 = pl.DataFrame({"C1": ["a", "a", "b"], "C2": [1, 1, 3]})
        >>> result = df1.intersectAll(df2).sort("C1", "C2")
        >>> result.height
        3
        >>> result.to_dicts()
        [{'C1': 'a', 'C2': 1}, {'C1': 'a', 'C2': 1}, {'C1': 'b', 'C2': 3}]
        >>> df3 = pl.DataFrame({"C1": ["a", "b"], "C2": [1, 3]})
        >>> result2 = df1.intersectAll(df3).sort("C1", "C2")
        >>> result2.height
        2
        >>> result2["C1"].to_list()
        ['a', 'b']
        >>> df4 = pl.DataFrame({"C1": ["x", "y"], "C2": [9, 10]})
        >>> df1.intersectAll(df4).height
        0
    """
    df1 = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    df2 = other.collect() if isinstance(other, PolarsLazyDataFrame) else other
    cols = df1.columns

    def _add_rank(df: PolarsDataFrame) -> PolarsDataFrame:
        return df.with_columns(pl.int_range(pl.len()).over(cols).alias("_rank"))

    result = (
        _add_rank(df1)
        .join(_add_rank(df2), on=cols + ["_rank"], how="inner")
        .drop("_rank")
    )
    if isinstance(self, PolarsLazyDataFrame):
        return result.lazy()
    return result


# ── subtract ──────────────────────────────────────────────────────────────────
def subtract(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Return rows in self that are not in other (set difference).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> other = pl.DataFrame({"a": [3, 4, 5, 6, 7], "b": ["z", "x", "y", "a", "b"], "c": [3.0, 4.0, 5.0, 6.0, 7.0]})
        >>> result = df.subtract(other)
        >>> all(v in {1, 2} for v in result["a"].to_list())
        True
    """
    cols = _schema_names(self)
    return self.join(other, on=cols, how="anti").unique()


# ── isLocal / isStreaming ─────────────────────────────────────────────────────
def isLocal(self: PolarsDataFrame | PolarsLazyDataFrame) -> bool:
    """Return True (Polars DataFrames are always local).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.isLocal()
        True
    """
    return True


def _isStreaming(self: PolarsDataFrame | PolarsLazyDataFrame) -> bool:
    """Return False (Polars DataFrames are not streaming).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.isStreaming
        False
    """
    return False


# ── melt ─────────────────────────────────────────────────────────────────────
def melt_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    id_vars: list[str] | None = None,
    value_vars: list[str] | None = None,
    var_name: str = "variable",
    value_name: str = "value",
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Unpivot a DataFrame from wide to long format (Spark-compatible melt).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"id": [1, 2], "val_a": [10, 20], "val_b": [30, 40]})
        >>> result = df.melt(id_vars=["id"], value_vars=["val_a", "val_b"])
        >>> "variable" in result.columns and "value" in result.columns
        True
        >>> result.height
        4
        >>> result2 = df.melt(id_vars=["id"], value_vars=["val_a", "val_b"], var_name="metric", value_name="amount")
        >>> "metric" in result2.columns and "amount" in result2.columns
        True
    """
    return self.unpivot(
        on=value_vars,
        index=id_vars,
        variable_name=var_name,
        value_name=value_name,
    )


# ── offset ────────────────────────────────────────────────────────────────────
def offset_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    n: int,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Skip the first n rows.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.offset(2)
        >>> result.height
        3
        >>> result["a"][0]
        3
    """
    return self.slice(n)


# ── printSchema ───────────────────────────────────────────────────────────────
def printSchema(self: PolarsDataFrame | PolarsLazyDataFrame) -> None:
    """Print the schema of the DataFrame.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1], "b": ["x"]})
        >>> df.printSchema()
        Schema([('a', Int64), ('b', String)])
    """
    schema = self.collect_schema() if isinstance(self, PolarsLazyDataFrame) else self.schema
    print(schema)  # noqa: T201


# ── randomSplit ───────────────────────────────────────────────────────────────
def randomSplit(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    weights: list[float],
    seed: int | None = None,
) -> list[PolarsDataFrame]:
    """Split a DataFrame into multiple parts by weight.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> splits = df.randomSplit([0.6, 0.4], seed=42)
        >>> len(splits)
        2
        >>> sum(len(s) for s in splits) == df.height
        True
        >>> all(isinstance(s, pl.DataFrame) for s in splits)
        True
        >>> splits3 = df.randomSplit([0.5, 0.3, 0.2], seed=1)
        >>> len(splits3)
        3
        >>> sum(len(s) for s in splits3) == df.height
        True
    """
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    n = len(df)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    total = sum(weights)
    normalised = [w / total for w in weights]
    splits = []
    start = 0
    for i, norm in enumerate(normalised):
        end = n if i == len(normalised) - 1 else start + round(norm * n)
        splits.append(df[indices[start:end]])
        start = end
    return splits


# ── selectExpr ────────────────────────────────────────────────────────────────
def selectExpr(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *exprs: str,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Select columns using SQL expressions (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.selectExpr("a", "b")
        >>> result.columns
        ['a', 'b']
        >>> result.height == df.height
        True
    """
    return self.select([pl.sql_expr(e) if isinstance(e, str) else e for e in exprs])


# ── take ──────────────────────────────────────────────────────────────────────
def take_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    n: int,
) -> list[dict]:
    """Return the first n rows as a list of dicts.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.take(3)
        >>> isinstance(result, list)
        True
        >>> len(result)
        3
        >>> isinstance(result[0], dict)
        True
        >>> result[0]["a"]
        1
    """
    if isinstance(self, PolarsLazyDataFrame):
        return self.limit(n).collect().to_dicts()
    return self.head(n).to_dicts()


# ── toArrow ───────────────────────────────────────────────────────────────────
def toArrow(self: PolarsDataFrame | PolarsLazyDataFrame) -> Any:
    """Convert to a PyArrow Table.

    Examples:
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.toArrow()
        >>> isinstance(result, pa.Table)
        True
        >>> result.num_rows
        5
    """
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return df.to_arrow()


# ── toDF ──────────────────────────────────────────────────────────────────────
def toDF(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *cols: str,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Rename all columns (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.toDF("x", "y", "z").columns
        ['x', 'y', 'z']
    """
    current = _schema_names(self)
    if len(cols) != len(current):
        msg = f"toDF expects {len(current)} column names, got {len(cols)}."
        raise ValueError(msg)
    return self.rename(dict(zip(current, cols)))


# ── toJSON ────────────────────────────────────────────────────────────────────
def toJSON(self: PolarsDataFrame | PolarsLazyDataFrame) -> str:
    """Convert to JSON string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.toJSON()
        >>> isinstance(result, str)
        True
        >>> '"a"' in result
        True
    """
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return df.write_json()


# ── toPandas ──────────────────────────────────────────────────────────────────
def toPandas(self: PolarsDataFrame | PolarsLazyDataFrame) -> Any:
    """Convert to a pandas DataFrame.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.toPandas()
        >>> isinstance(result, pd.DataFrame)
        True
        >>> list(result.columns)
        ['a', 'b', 'c']
    """
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return df.to_pandas()


# ── transform ─────────────────────────────────────────────────────────────────
def transform_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Apply a function to the DataFrame (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> def add_col(df):
        ...     return df.with_columns(pl.col("a").alias("a2"))
        >>> "a2" in df.transform(add_col).columns
        True
        >>> def multiply(df, factor):
        ...     return df.with_columns((pl.col("a") * factor).alias("a_scaled"))
        >>> df.transform(multiply, 10)["a_scaled"][0]
        10
    """
    return func(self, *args, **kwargs)


# ── transpose (LazyFrame only — DataFrame already has it natively) ────────────
def transpose_lazy(
    self: PolarsLazyDataFrame,
    *args: Any,
    **kwargs: Any,
) -> PolarsDataFrame:
    """Transpose a LazyFrame (collects first).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> lf = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
        >>> result = lf.transpose()
        >>> isinstance(result, pl.DataFrame)
        True
        >>> result.shape
        (2, 3)
    """
    return self.collect().transpose(*args, **kwargs)


# ── DataFrameWriter (df.write) ────────────────────────────────────────────────
class DataFrameWriter:
    """PySpark-compatible DataFrameWriter backed by Polars I/O methods.

    Usage::

        df.write.parquet("/path/out.parquet")
        df.write.mode("overwrite").csv("/path/out.csv")
        df.write.option("compression", "snappy").parquet("/path/out.parquet")
        df.write.format("delta").mode("append").save("/path/delta_table")

    LazyFrame uses ``sink_*`` methods where available for streaming efficiency.
    """

    _DELTA_MODES = {"error", "append", "overwrite", "ignore", "merge"}
    _FORMAT_ALIASES = {
        "parquet": "parquet",
        "csv": "csv",
        "json": "json",
        "ndjson": "json",
        "avro": "avro",
        "ipc": "ipc",
        "arrow": "ipc",
        "feather": "ipc",
        "ipc_stream": "ipc_stream",
        "excel": "excel",
        "xlsx": "excel",
        "delta": "delta",
        "iceberg": "iceberg",
        "text": "text",
        "clipboard": "clipboard",
    }

    def __init__(self, df: PolarsDataFrame | PolarsLazyDataFrame) -> None:
        self._df = df
        self._mode: str = "error"
        self._fmt: str | None = None
        self._options: dict[str, Any] = {}
        self._partition_cols: list[str] = []

    # ── builder methods ───────────────────────────────────────────────────────

    def mode(self, saveMode: str) -> "DataFrameWriter":
        """Set the save mode for the writer.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"a": [1]})
            >>> df.write.mode("overwrite")._mode
            'overwrite'
        """
        self._mode = saveMode.lower()
        return self

    def format(self, source: str) -> "DataFrameWriter":
        """Set the output format for the writer.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"a": [1]})
            >>> df.write.format("parquet")._fmt
            'parquet'
        """
        self._fmt = source.lower()
        return self

    def option(self, key: str, value: Any) -> "DataFrameWriter":
        """Set a single writer option.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"a": [1]})
            >>> df.write.option("k", "v")._options["k"]
            'v'
        """
        self._options[key] = value
        return self

    def options(self, **kwargs: Any) -> "DataFrameWriter":
        self._options.update(kwargs)
        return self

    def partitionBy(self, *cols: str) -> "DataFrameWriter":
        self._partition_cols = list(cols[0] if len(cols) == 1 and isinstance(cols[0], list) else cols)
        return self

    # ── internal helpers ──────────────────────────────────────────────────────

    def _eager(self) -> PolarsDataFrame:
        if isinstance(self._df, PolarsLazyDataFrame):
            return self._df.collect()
        return self._df

    def _should_write(self, path: str | Path | None) -> bool:
        """Return False (skip) or raise based on mode and path existence."""
        if path is None:
            return True
        p = Path(path)
        exists = p.exists()
        if not exists:
            return True
        if self._mode in ("error", "errorifexists"):
            msg = f"Path already exists: {path}. Use mode('overwrite') to overwrite."
            raise RuntimeError(msg)
        if self._mode == "ignore":
            return False  # silently skip
        return True  # overwrite / append: proceed (Polars overwrites by default)

    # ── format writers ────────────────────────────────────────────────────────

    def parquet(self, path: str | Path, **kwargs: Any) -> None:
        if not self._should_write(path):
            return
        opts = {**self._options, **kwargs}
        if isinstance(self._df, PolarsLazyDataFrame):
            self._df.sink_parquet(path, **opts)
        else:
            self._eager().write_parquet(path, **opts)

    def csv(self, path: str | Path, **kwargs: Any) -> None:
        if not self._should_write(path):
            return
        opts = {**self._options, **kwargs}
        if isinstance(self._df, PolarsLazyDataFrame):
            self._df.sink_csv(path, **opts)
        else:
            self._eager().write_csv(path, **opts)

    def json(self, path: str | Path, **kwargs: Any) -> None:
        if not self._should_write(path):
            return
        opts = {**self._options, **kwargs}
        if isinstance(self._df, PolarsLazyDataFrame):
            self._df.sink_ndjson(path, **opts)
        else:
            self._eager().write_ndjson(path, **opts)

    def avro(self, path: str | Path, **kwargs: Any) -> None:
        if not self._should_write(path):
            return
        self._eager().write_avro(path, **{**self._options, **kwargs})

    def ipc(self, path: str | Path, **kwargs: Any) -> None:
        if not self._should_write(path):
            return
        opts = {**self._options, **kwargs}
        if isinstance(self._df, PolarsLazyDataFrame):
            self._df.sink_ipc(path, **opts)
        else:
            self._eager().write_ipc(path, **opts)

    # alias
    feather = ipc

    def ipc_stream(self, path: str | Path, **kwargs: Any) -> None:
        if not self._should_write(path):
            return
        self._eager().write_ipc_stream(path, **{**self._options, **kwargs})

    def excel(self, path: str | Path, **kwargs: Any) -> None:
        if not self._should_write(path):
            return
        self._eager().write_excel(path, **{**self._options, **kwargs})

    def delta(self, path: str | Path, **kwargs: Any) -> None:
        mode = self._options.pop("mode", self._mode)
        if mode not in self._DELTA_MODES:
            mode = "error"
        opts = {**self._options, **kwargs}
        if isinstance(self._df, PolarsLazyDataFrame):
            self._df.sink_delta(path, mode=mode, **opts)
        else:
            self._eager().write_delta(path, mode=mode, **opts)

    def iceberg(self, *args: Any, **kwargs: Any) -> None:
        mode = self._options.pop("mode", self._mode)
        if isinstance(self._df, PolarsLazyDataFrame):
            self._df.sink_iceberg(*args, mode=mode, **{**self._options, **kwargs})
        else:
            self._eager().write_iceberg(*args, mode=mode, **{**self._options, **kwargs})

    def text(self, path: str | Path, **kwargs: Any) -> None:
        if not self._should_write(path):
            return
        df = self._eager()
        if df.width != 1:
            msg = "text() requires a DataFrame with exactly one column."
            raise ValueError(msg)
        Path(path).write_text("\n".join(str(v) for v in df[:, 0].to_list()))

    def clipboard(self, **kwargs: Any) -> None:
        self._eager().write_clipboard(**{**self._options, **kwargs})

    def saveAsTable(self, name: str, **kwargs: Any) -> None:
        _local_sql_ctx.register(name, self._df)

    def save(self, path: str | Path | None = None, format: str | None = None, **kwargs: Any) -> None:
        fmt = self._FORMAT_ALIASES.get((format or self._fmt or "").lower())
        if not fmt:
            msg = "No format specified. Use .format('parquet') or a named method."
            raise ValueError(msg)
        writer = getattr(self, fmt)
        writer(path, **kwargs) if path is not None else writer(**kwargs)


def _write_property(self: PolarsDataFrame | PolarsLazyDataFrame) -> DataFrameWriter:
    return DataFrameWriter(self)


# ── exists ────────────────────────────────────────────────────────────────────
def exists(
    self: PolarsDataFrame | PolarsLazyDataFrame,
) -> bool:
    """Return True if the DataFrame has at least one row.

    This covers the non-correlated use case: ``df.exists()``.

    Correlated EXISTS subqueries (``col("x").outer()`` pattern from PySpark)
    cannot be replicated at the Polars DataFrame API level.  Rewrite them as
    semi/anti joins instead::

        # EXISTS  → semi join
        outer_df.join(inner_df.select("key").unique(), on="key", how="semi")

        # NOT EXISTS → anti join
        outer_df.join(inner_df.select("key").unique(), on="key", how="anti")

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> df.exists()
        True
        >>> pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}).exists()
        False
        >>> customers = pl.DataFrame({"customer_id": [101, 102, 103, 104], "country": ["USA", "Canada", "USA", "Australia"]})
        >>> orders = pl.DataFrame({"customer_id": [101, 102, 103, 101]})
        >>> set(customers.join(orders.select("customer_id").unique(), on="customer_id", how="semi")["customer_id"].to_list()) == {101, 102, 103}
        True
        >>> customers.join(orders.select("customer_id").unique(), on="customer_id", how="anti")["customer_id"].to_list()
        [104]
    """
    if isinstance(self, PolarsLazyDataFrame):
        return self.limit(1).collect().height > 0
    return self.height > 0


# ── lateralJoin ───────────────────────────────────────────────────────────────
def lateralJoin(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
    on: Any = None,
    how: str | None = None,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    """Lateral join (Spark-compatible).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> small = df.head(2)
        >>> small.lateralJoin(small).height
        4
    """
    # Cross join: no predicate or explicit cross
    if on is None or how == "cross":
        return self.join(other, how="cross")
    # Left outer with predicate — not supported by join_where
    if how == "left_outer" or how == "left":
        msg = "lateralJoin with how='left' is not supported in Polars; use inner or cross."
        raise NotImplementedError(msg)
    predicates = on if isinstance(on, list) else [on]
    return self.join_where(other, *predicates)


# ── foreach / foreachPartition ────────────────────────────────────────────────
def foreach(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    func: Any,
) -> None:
    """Apply a function to each row of the DataFrame.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> collected = []; pl.DataFrame({"a": [1, 2, 3]}).foreach(lambda row: collected.append(row["a"])); collected
        [1, 2, 3]
    """
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    for row in df.iter_rows(named=True):
        func(row)


def foreachPartition(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    func: Any,
) -> None:
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    func(df.iter_rows(named=True))


# ── mapInPandas / mapInArrow ──────────────────────────────────────────────────
def mapInPandas(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    func: Any,
    schema: Any = None,
) -> PolarsDataFrame:
    import pandas as pd

    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    chunks = list(func(iter([df.to_pandas()])))
    return pl.from_pandas(pd.concat(chunks, ignore_index=True))


def mapInArrow(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    func: Any,
    schema: Any = None,
) -> PolarsDataFrame:
    import pyarrow as pa

    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    chunks = list(func(iter([df.to_arrow()])))
    return pl.from_arrow(pa.concat_tables(chunks))


# ── corr / cov ────────────────────────────────────────────────────────────────
def corr_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    col1: str,
    col2: str,
    method: str = "pearson",
) -> float:
    """Compute the correlation between two columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.corr("a", "c")
        >>> isinstance(result, float)
        True
        >>> abs(result - 1.0) < 1e-9
        True
        >>> isinstance(df.corr("a", "c", method="spearman"), float)
        True
    """
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return pl.select(pl.corr(df[col1], df[col2], method=method)).item()


def cov_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    col1: str,
    col2: str,
) -> float:
    """Compute the covariance between two columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.cov("a", "c")
        >>> isinstance(result, float)
        True
        >>> result > 0
        True
    """
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return pl.select(pl.cov(df[col1], df[col2])).item()


# ── explain ───────────────────────────────────────────────────────────────────
def explain(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *args: Any,
    **kwargs: Any,
) -> str:
    """Return the query plan as a string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.explain()
        >>> isinstance(result, str)
        True
        >>> len(result) > 0
        True
    """
    lf = self.lazy() if isinstance(self, PolarsDataFrame) else self
    return lf.explain(*args, **kwargs)


# ── na (DataFrameNaFunctions wrapper) ─────────────────────────────────────────
class DataFrameNaFunctions:
    """Spark-compatible na accessor: df.na.fill / df.na.drop / df.na.replace.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [None, "y", None], "c": [1.0, None, 3.0]})
        >>> df.na.fill("X")["b"].null_count()
        0
        >>> df.na.drop().height
        0
        >>> df2 = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> 99 in df2.na.replace(1, 99)["a"].to_list()
        True
    """

    def __init__(self, df: PolarsDataFrame | PolarsLazyDataFrame) -> None:
        self._df = df

    def fill(self, value: Any, subset: list[str] | None = None) -> PolarsDataFrame | PolarsLazyDataFrame:
        """Fill null values with the given value.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"a": [1, None, 3]})
            >>> df.na.fill(0)["a"].to_list()
            [1, 0, 3]
        """
        return fillna(self._df, value, subset=subset)

    def drop(
        self,
        how: str = "any",
        thresh: int | None = None,
        subset: list[str] | None = None,
    ) -> PolarsDataFrame | PolarsLazyDataFrame:
        """Drop rows containing null values.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"a": [1, None, 3]})
            >>> df.na.drop().height
            2
        """
        return dropnulls(self._df, how=how, thresh=thresh, subset=subset)

    def replace(
        self,
        to_replace: Any,
        value: Any = None,
        subset: list[str] | None = None,
    ) -> PolarsDataFrame | PolarsLazyDataFrame:
        """Replace matching values in the DataFrame.

        Examples:
            >>> import polars as pl
            >>> import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
            >>> df = pl.DataFrame({"a": [1, 2, 3]})
            >>> df.na.replace(1, 99)["a"].to_list()
            [99, 2, 3]
        """
        return replace_spark(self._df, to_replace, value, subset=subset)


def _na_property(self: PolarsDataFrame | PolarsLazyDataFrame) -> DataFrameNaFunctions:
    return DataFrameNaFunctions(self)


# ── temp views (SQLContext-backed) ────────────────────────────────────────────
def _frame_for_sql(self: PolarsDataFrame | PolarsLazyDataFrame) -> PolarsDataFrame | PolarsLazyDataFrame:
    return self


def createTempView(self: PolarsDataFrame | PolarsLazyDataFrame, name: str) -> None:
    """Register as a local temp view. Raises if the name already exists.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe as _mod
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.createOrReplaceTempView("_doctest_ct")
        >>> _mod._local_sql_ctx.execute("SELECT * FROM _doctest_ct").collect().height
        3
        >>> _ = _mod._local_sql_ctx.unregister("_doctest_ct")
    """
    if name in _local_sql_ctx.tables():
        msg = f"Temp view '{name}' already exists. Use createOrReplaceTempView to overwrite."
        raise RuntimeError(msg)
    _local_sql_ctx.register(name, _frame_for_sql(self))


def createOrReplaceTempView(self: PolarsDataFrame | PolarsLazyDataFrame, name: str) -> None:
    """Register as a local temp view, replacing any existing view with the same name.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe as _mod
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.createOrReplaceTempView("_doctest_cort")
        >>> "_doctest_cort" in _mod._local_sql_ctx.tables()
        True
        >>> _ = _mod._local_sql_ctx.unregister("_doctest_cort")
    """
    _local_sql_ctx.register(name, _frame_for_sql(self))


def createGlobalTempView(self: PolarsDataFrame | PolarsLazyDataFrame, name: str) -> None:
    """Register as a global temp view. Raises if the name already exists.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe as _mod
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.createOrReplaceGlobalTempView("_doctest_cgt")
        >>> "_doctest_cgt" in _mod._global_sql_ctx.tables()
        True
        >>> _ = _mod._global_sql_ctx.unregister("_doctest_cgt")
    """
    if name in _global_sql_ctx.tables():
        msg = f"Global temp view '{name}' already exists. Use createOrReplaceGlobalTempView to overwrite."
        raise RuntimeError(msg)
    _global_sql_ctx.register(name, _frame_for_sql(self))


def createOrReplaceGlobalTempView(self: PolarsDataFrame | PolarsLazyDataFrame, name: str) -> None:
    """Register as a global temp view, replacing any existing view with the same name.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.dataframe as _mod
        >>> df = pl.DataFrame({"a": [10, 20]})
        >>> df.createOrReplaceGlobalTempView("_doctest_corgt")
        >>> "_doctest_corgt" in _mod._global_sql_ctx.tables()
        True
        >>> _ = _mod._global_sql_ctx.unregister("_doctest_corgt")
    """
    _global_sql_ctx.register(name, _frame_for_sql(self))


# ═════════════════════════════════════════════════════════════════════════════
# PolarsDataFrame monkey-patch assignments
# ═════════════════════════════════════════════════════════════════════════════
PolarsDataFrame.withColumn = withColumn
PolarsDataFrame.withColumns = withColumns

# no-ops
PolarsDataFrame.hint = return_self
PolarsDataFrame.alias = return_self
PolarsDataFrame.repartition = return_self
PolarsDataFrame.coalesce = return_self
PolarsDataFrame.unpersist = return_self
PolarsDataFrame.repartitionByRange = return_self
PolarsDataFrame.withMetadata = return_self
PolarsDataFrame.collect = return_self
PolarsDataFrame.checkpoint = return_self
PolarsDataFrame.cache = return_self
PolarsDataFrame.localCheckpoint = return_self

# meaningful mappings
PolarsDataFrame.persist = persist
PolarsDataFrame.distinct = distinct
PolarsDataFrame.dropDuplicates = dropDuplicates
PolarsDataFrame.drop_duplicates = dropDuplicates
PolarsDataFrame.dropna = dropna
PolarsDataFrame.dropnulls = dropnulls
PolarsDataFrame.unionByName = unionByName
PolarsDataFrame.union = union_spark
PolarsDataFrame.unionAll = union_spark
PolarsDataFrame.crossJoin = crossJoin
PolarsDataFrame.groupBy = groupBy
PolarsDataFrame.show = show
PolarsDataFrame.colRegex = colRegex
PolarsDataFrame.withColumnRenamed = withColumnRenamed
PolarsDataFrame.withColumnsRenamed = withColumnsRenamed
PolarsDataFrame.describe = describe
PolarsDataFrame.summary = summary_spark
PolarsDataFrame.fillna = fillna
PolarsDataFrame.filter = filter_spark
PolarsDataFrame.where = filter_spark
PolarsDataFrame.first = first_row
PolarsDataFrame.intersect = intersect
PolarsDataFrame.isLocal = isLocal
PolarsDataFrame.isStreaming = property(_isStreaming)
PolarsDataFrame.melt = melt_spark
PolarsDataFrame.offset = offset_spark
PolarsDataFrame.orderBy = sort_spark
PolarsDataFrame.printSchema = printSchema
PolarsDataFrame.randomSplit = randomSplit
PolarsDataFrame.replace = replace_spark
PolarsDataFrame.sample = sample_spark
PolarsDataFrame.selectExpr = selectExpr
PolarsDataFrame.sort = sort_spark
PolarsDataFrame.sortWithinPartitions = sort_spark
PolarsDataFrame.subtract = subtract
PolarsDataFrame.take = take_spark
PolarsDataFrame.toArrow = toArrow
PolarsDataFrame.toDF = toDF
PolarsDataFrame.toJSON = toJSON
PolarsDataFrame.toPandas = toPandas
PolarsDataFrame.transform = transform_spark

# spark-only extras (meaningful mappings)
PolarsDataFrame.corr = corr_spark
PolarsDataFrame.cov = cov_spark
PolarsDataFrame.createGlobalTempView = createGlobalTempView
PolarsDataFrame.createOrReplaceGlobalTempView = createOrReplaceGlobalTempView
PolarsDataFrame.createOrReplaceTempView = createOrReplaceTempView
PolarsDataFrame.createTempView = createTempView
PolarsDataFrame.exists = exists
PolarsDataFrame.explain = explain
PolarsDataFrame.foreach = foreach
PolarsDataFrame.foreachPartition = foreachPartition
PolarsDataFrame.intersectAll = intersectAll
PolarsDataFrame.lateralJoin = lateralJoin
PolarsDataFrame.mapInArrow = mapInArrow
PolarsDataFrame.mapInPandas = mapInPandas
PolarsDataFrame.na = property(_na_property)
PolarsDataFrame.registerTempTable = createOrReplaceTempView

# not implemented
PolarsDataFrame.approxQuantile = not_implemented
PolarsDataFrame.asTable = not_implemented
PolarsDataFrame.crosstab = not_implemented
PolarsDataFrame.cube = not_implemented
PolarsDataFrame.dropDuplicatesWithinWatermark = not_implemented
PolarsDataFrame.executionInfo = not_implemented
PolarsDataFrame.freqItems = not_implemented
PolarsDataFrame.groupingSets = not_implemented
PolarsDataFrame.inputFiles = not_implemented
PolarsDataFrame.metadataColumn = not_implemented
PolarsDataFrame.observe = not_implemented
# .plot is already native on DataFrame (Altair-backed) — no override needed
PolarsDataFrame.rdd = not_implemented
PolarsDataFrame.rollup = not_implemented
PolarsDataFrame.sameSemantics = not_implemented
PolarsDataFrame.sampleBy = not_implemented
PolarsDataFrame.scalar = not_implemented
PolarsDataFrame.semanticHash = not_implemented
PolarsDataFrame.stat = not_implemented
PolarsDataFrame.storageLevel = not_implemented
PolarsDataFrame.to = not_implemented
PolarsDataFrame.toLocalIterator = not_implemented
PolarsDataFrame.withWatermark = not_implemented
PolarsDataFrame.write = property(_write_property)
PolarsDataFrame.writeStream = not_implemented
PolarsDataFrame.writeTo = not_implemented
PolarsDataFrame.mergeInto = not_implemented
PolarsDataFrame.pandas_api = not_implemented

# properties
PolarsDataFrame.isEmpty = isEmpty_non_lazy
PolarsDataFrame.count = property(count_non_lazy)
PolarsDataFrame.spark_session = None

DataFrame = PolarsDataFrame


# ═════════════════════════════════════════════════════════════════════════════
# PolarsLazyDataFrame monkey-patch assignments
# ═════════════════════════════════════════════════════════════════════════════
PolarsLazyDataFrame.withColumn = withColumn
PolarsLazyDataFrame.withColumns = withColumns

# no-ops
PolarsLazyDataFrame.hint = return_self
PolarsLazyDataFrame.alias = return_self
PolarsLazyDataFrame.repartition = return_self
PolarsLazyDataFrame.coalesce = return_self
PolarsLazyDataFrame.unpersist = return_self
PolarsLazyDataFrame.repartitionByRange = return_self
PolarsLazyDataFrame.withMetadata = return_self
PolarsLazyDataFrame.cache = return_self
PolarsLazyDataFrame.checkpoint = return_self
PolarsLazyDataFrame.localCheckpoint = return_self

# meaningful mappings
PolarsLazyDataFrame.persist = persist
PolarsLazyDataFrame.distinct = distinct
PolarsLazyDataFrame.dropDuplicates = dropDuplicates
PolarsLazyDataFrame.drop_duplicates = dropDuplicates
PolarsLazyDataFrame.dropna = dropna
PolarsLazyDataFrame.dropnulls = dropnulls
PolarsLazyDataFrame.unionByName = unionByName
PolarsLazyDataFrame.union = union_spark
PolarsLazyDataFrame.unionAll = union_spark
PolarsLazyDataFrame.crossJoin = crossJoin
PolarsLazyDataFrame.groupBy = groupBy
PolarsLazyDataFrame.colRegex = colRegex
PolarsLazyDataFrame.show = show
PolarsLazyDataFrame.withColumnRenamed = withColumnRenamed
PolarsLazyDataFrame.withColumnsRenamed = withColumnsRenamed
PolarsLazyDataFrame.describe = describe
PolarsLazyDataFrame.summary = summary_spark
PolarsLazyDataFrame.fillna = fillna
PolarsLazyDataFrame.filter = filter_spark
PolarsLazyDataFrame.where = filter_spark
PolarsLazyDataFrame.first = first_row
PolarsLazyDataFrame.intersect = intersect
PolarsLazyDataFrame.isLocal = isLocal
PolarsLazyDataFrame.isStreaming = property(_isStreaming)
PolarsLazyDataFrame.melt = melt_spark
PolarsLazyDataFrame.offset = offset_spark
PolarsLazyDataFrame.orderBy = sort_spark
PolarsLazyDataFrame.printSchema = printSchema
PolarsLazyDataFrame.randomSplit = randomSplit
PolarsLazyDataFrame.replace = replace_spark
PolarsLazyDataFrame.sample = sample_spark
PolarsLazyDataFrame.selectExpr = selectExpr
PolarsLazyDataFrame.sort = sort_spark
PolarsLazyDataFrame.sortWithinPartitions = sort_spark
PolarsLazyDataFrame.subtract = subtract
PolarsLazyDataFrame.take = take_spark
PolarsLazyDataFrame.toArrow = toArrow
PolarsLazyDataFrame.toDF = toDF
PolarsLazyDataFrame.toJSON = toJSON
PolarsLazyDataFrame.toPandas = toPandas
PolarsLazyDataFrame.transform = transform_spark
PolarsLazyDataFrame.transpose = transpose_lazy

# spark-only extras (meaningful mappings)
PolarsLazyDataFrame.corr = corr_spark
PolarsLazyDataFrame.cov = cov_spark
PolarsLazyDataFrame.createGlobalTempView = createGlobalTempView
PolarsLazyDataFrame.createOrReplaceGlobalTempView = createOrReplaceGlobalTempView
PolarsLazyDataFrame.createOrReplaceTempView = createOrReplaceTempView
PolarsLazyDataFrame.createTempView = createTempView
# .explain is already native on LazyFrame — no override needed
PolarsLazyDataFrame.exists = exists
PolarsLazyDataFrame.foreach = foreach
PolarsLazyDataFrame.foreachPartition = foreachPartition
PolarsLazyDataFrame.intersectAll = intersectAll
PolarsLazyDataFrame.lateralJoin = lateralJoin
PolarsLazyDataFrame.mapInArrow = mapInArrow
PolarsLazyDataFrame.mapInPandas = mapInPandas
PolarsLazyDataFrame.na = property(_na_property)
PolarsLazyDataFrame.registerTempTable = createOrReplaceTempView

# not implemented
PolarsLazyDataFrame.approxQuantile = not_implemented
PolarsLazyDataFrame.asTable = not_implemented
PolarsLazyDataFrame.crosstab = not_implemented
PolarsLazyDataFrame.cube = not_implemented
PolarsLazyDataFrame.dropDuplicatesWithinWatermark = not_implemented
PolarsLazyDataFrame.executionInfo = not_implemented
PolarsLazyDataFrame.freqItems = not_implemented
PolarsLazyDataFrame.groupingSets = not_implemented
PolarsLazyDataFrame.inputFiles = not_implemented
PolarsLazyDataFrame.metadataColumn = not_implemented
PolarsLazyDataFrame.observe = not_implemented
# .plot is already native on LazyFrame (Altair-backed) — no override needed
PolarsLazyDataFrame.rdd = not_implemented
PolarsLazyDataFrame.rollup = not_implemented
PolarsLazyDataFrame.sameSemantics = not_implemented
PolarsLazyDataFrame.sampleBy = not_implemented
PolarsLazyDataFrame.scalar = not_implemented
PolarsLazyDataFrame.semanticHash = not_implemented
PolarsLazyDataFrame.stat = not_implemented
PolarsLazyDataFrame.storageLevel = not_implemented
PolarsLazyDataFrame.to = not_implemented
PolarsLazyDataFrame.toLocalIterator = not_implemented
PolarsLazyDataFrame.withWatermark = not_implemented
PolarsLazyDataFrame.write = property(_write_property)
PolarsLazyDataFrame.writeStream = not_implemented
PolarsLazyDataFrame.writeTo = not_implemented
PolarsLazyDataFrame.mergeInto = not_implemented
PolarsLazyDataFrame.pandas_api = not_implemented

# properties
PolarsLazyDataFrame.schema = property(schema_lazy)
PolarsLazyDataFrame.isEmpty = isEmpty_lazy
PolarsLazyDataFrame.count = property(count_lazy)
PolarsLazyDataFrame.columns = property(columns_lazy)
PolarsLazyDataFrame.spark_session = None

LazyFrame = PolarsLazyDataFrame
