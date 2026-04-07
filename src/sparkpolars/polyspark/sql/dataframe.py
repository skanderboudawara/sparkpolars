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
    return self.withColumns({name: expr})


# ── persist ───────────────────────────────────────────────────────────────────
def persist(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *_args: Any,
    **_kwargs: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    return self.cache()


# ── distinct / dropDuplicates ─────────────────────────────────────────────────
def distinct(
    self: PolarsDataFrame | PolarsLazyDataFrame,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    return self.unique()


def dropDuplicates(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    subset: list[str] | None = None,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    return self.unique(subset=subset, keep="first")


# ── dropna / dropnulls ────────────────────────────────────────────────────────
def dropna(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    how: str = "any",
    thresh: int | None = None,
    subset: list[str] | None = None,
) -> PolarsDataFrame | PolarsLazyDataFrame:
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
    return concat([self, other], how="vertical")


def unionByName(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
    allowMissingColumns: bool = False,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    return concat(
        [self, other],
        how="diagonal_relaxed" if allowMissingColumns else "vertical",
    )


def crossJoin(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    return self.join(other, how="cross")


# ── groupBy / agg ─────────────────────────────────────────────────────────────
def groupBy(self: PolarsDataFrame | PolarsLazyDataFrame, *cols: Any) -> Any:
    return self.group_by(*cols)


def agg(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *expr: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    if len(expr) == 1 and isinstance(expr[0], dict):
        expr_dict = expr[0]
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in expr_dict.items()):
            msg = "All keys and values in the aggregation dictionary must be strings."
            raise ValueError(msg)
        allowed_aggs = {"min", "max", "mean", "sum", "count", "first", "last"}
        if not all(v in allowed_aggs for v in expr_dict.values()):
            msg = f"Aggregation functions must be one of {allowed_aggs}."
            raise ValueError(msg)
        expr = [getattr(col(k), v)().alias(f"{v}({k})") for k, v in expr_dict.items()]
    return (
        self.group_by(lit(1).alias("agg_polyspark"))
        .agg(*expr)
        .drop("agg_polyspark")
    )


PolarsDataFrame.agg = agg
PolarsLazyDataFrame.agg = agg


# ── schema / count / isEmpty / columns helpers ────────────────────────────────
def schema_lazy(self: PolarsLazyDataFrame) -> Any:
    return self.collect_schema()


def isEmpty_lazy(self: PolarsLazyDataFrame) -> bool:
    return self.limit(1).collect().is_empty()


def count_lazy(self: PolarsLazyDataFrame) -> int:
    return self.collect().height


def columns_lazy(self: PolarsLazyDataFrame) -> list[str]:
    return self.collect_schema().names()


def isEmpty_non_lazy(self: PolarsDataFrame) -> bool:
    return self.is_empty()


def count_non_lazy(self: PolarsDataFrame) -> int:
    return self.height


# ── show ──────────────────────────────────────────────────────────────────────
def show(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    n: int = 20,
    truncate: bool = True,
    vertical: bool = False,
) -> None:
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
    pattern = re.compile(colName)
    columns = [c for c in self.columns if pattern.search(c)]
    return self.select(columns)


def withColumnRenamed(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    existing: str,
    new: str,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    return self.rename({existing: new}, strict=False)


def withColumnsRenamed(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    colsMap: dict[str, str],
) -> PolarsDataFrame | PolarsLazyDataFrame:
    return self.rename(colsMap, strict=False)


# ── describe ──────────────────────────────────────────────────────────────────
def describe(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *cols: str,
    **kwargs: Any,
) -> PolarsDataFrame:
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
    return describe(self)


# ── fillna ────────────────────────────────────────────────────────────────────
def fillna(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    value: Any,
    subset: list[str] | None = None,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    if subset:
        return self.with_columns([pl.col(c).fill_null(value) for c in subset])
    return self.fill_null(value)


# ── first ─────────────────────────────────────────────────────────────────────
def first_row(
    self: PolarsDataFrame | PolarsLazyDataFrame,
) -> dict | None:
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
    cols = _schema_names(self)
    return self.join(other, on=cols, how="semi").unique()


# ── intersectAll ──────────────────────────────────────────────────────────────
def intersectAll(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    other: PolarsDataFrame | PolarsLazyDataFrame,
) -> PolarsDataFrame | PolarsLazyDataFrame:
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
    cols = _schema_names(self)
    return self.join(other, on=cols, how="anti").unique()


# ── isLocal / isStreaming ─────────────────────────────────────────────────────
def isLocal(self: PolarsDataFrame | PolarsLazyDataFrame) -> bool:
    return True


def _isStreaming(self: PolarsDataFrame | PolarsLazyDataFrame) -> bool:
    return False


# ── melt ─────────────────────────────────────────────────────────────────────
def melt_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    id_vars: list[str] | None = None,
    value_vars: list[str] | None = None,
    var_name: str = "variable",
    value_name: str = "value",
) -> PolarsDataFrame | PolarsLazyDataFrame:
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
    return self.slice(n)


# ── printSchema ───────────────────────────────────────────────────────────────
def printSchema(self: PolarsDataFrame | PolarsLazyDataFrame) -> None:
    schema = self.collect_schema() if isinstance(self, PolarsLazyDataFrame) else self.schema
    print(schema)  # noqa: T201


# ── randomSplit ───────────────────────────────────────────────────────────────
def randomSplit(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    weights: list[float],
    seed: int | None = None,
) -> list[PolarsDataFrame]:
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
    return self.select([pl.sql_expr(e) if isinstance(e, str) else e for e in exprs])


# ── take ──────────────────────────────────────────────────────────────────────
def take_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    n: int,
) -> list[dict]:
    if isinstance(self, PolarsLazyDataFrame):
        return self.limit(n).collect().to_dicts()
    return self.head(n).to_dicts()


# ── toArrow ───────────────────────────────────────────────────────────────────
def toArrow(self: PolarsDataFrame | PolarsLazyDataFrame) -> Any:
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return df.to_arrow()


# ── toDF ──────────────────────────────────────────────────────────────────────
def toDF(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *cols: str,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    current = _schema_names(self)
    if len(cols) != len(current):
        msg = f"toDF expects {len(current)} column names, got {len(cols)}."
        raise ValueError(msg)
    return self.rename(dict(zip(current, cols)))


# ── toJSON ────────────────────────────────────────────────────────────────────
def toJSON(self: PolarsDataFrame | PolarsLazyDataFrame) -> str:
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return df.write_json()


# ── toPandas ──────────────────────────────────────────────────────────────────
def toPandas(self: PolarsDataFrame | PolarsLazyDataFrame) -> Any:
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return df.to_pandas()


# ── transform ─────────────────────────────────────────────────────────────────
def transform_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> PolarsDataFrame | PolarsLazyDataFrame:
    return func(self, *args, **kwargs)


# ── transpose (LazyFrame only — DataFrame already has it natively) ────────────
def transpose_lazy(
    self: PolarsLazyDataFrame,
    *args: Any,
    **kwargs: Any,
) -> PolarsDataFrame:
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
        self._mode = saveMode.lower()
        return self

    def format(self, source: str) -> "DataFrameWriter":
        self._fmt = source.lower()
        return self

    def option(self, key: str, value: Any) -> "DataFrameWriter":
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
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return pl.select(pl.corr(df[col1], df[col2], method=method)).item()


def cov_spark(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    col1: str,
    col2: str,
) -> float:
    df = self.collect() if isinstance(self, PolarsLazyDataFrame) else self
    return pl.select(pl.cov(df[col1], df[col2])).item()


# ── explain ───────────────────────────────────────────────────────────────────
def explain(
    self: PolarsDataFrame | PolarsLazyDataFrame,
    *args: Any,
    **kwargs: Any,
) -> str:
    lf = self.lazy() if isinstance(self, PolarsDataFrame) else self
    return lf.explain(*args, **kwargs)


# ── na (DataFrameNaFunctions wrapper) ─────────────────────────────────────────
class DataFrameNaFunctions:
    """Spark-compatible na accessor: df.na.fill / df.na.drop / df.na.replace."""

    def __init__(self, df: PolarsDataFrame | PolarsLazyDataFrame) -> None:
        self._df = df

    def fill(self, value: Any, subset: list[str] | None = None) -> PolarsDataFrame | PolarsLazyDataFrame:
        return fillna(self._df, value, subset=subset)

    def drop(
        self,
        how: str = "any",
        thresh: int | None = None,
        subset: list[str] | None = None,
    ) -> PolarsDataFrame | PolarsLazyDataFrame:
        return dropnulls(self._df, how=how, thresh=thresh, subset=subset)

    def replace(
        self,
        to_replace: Any,
        value: Any = None,
        subset: list[str] | None = None,
    ) -> PolarsDataFrame | PolarsLazyDataFrame:
        return replace_spark(self._df, to_replace, value, subset=subset)


def _na_property(self: PolarsDataFrame | PolarsLazyDataFrame) -> DataFrameNaFunctions:
    return DataFrameNaFunctions(self)


# ── temp views (SQLContext-backed) ────────────────────────────────────────────
def _frame_for_sql(self: PolarsDataFrame | PolarsLazyDataFrame) -> PolarsDataFrame | PolarsLazyDataFrame:
    return self


def createTempView(self: PolarsDataFrame | PolarsLazyDataFrame, name: str) -> None:
    if name in _local_sql_ctx.tables():
        msg = f"Temp view '{name}' already exists. Use createOrReplaceTempView to overwrite."
        raise RuntimeError(msg)
    _local_sql_ctx.register(name, _frame_for_sql(self))


def createOrReplaceTempView(self: PolarsDataFrame | PolarsLazyDataFrame, name: str) -> None:
    _local_sql_ctx.register(name, _frame_for_sql(self))


def createGlobalTempView(self: PolarsDataFrame | PolarsLazyDataFrame, name: str) -> None:
    if name in _global_sql_ctx.tables():
        msg = f"Global temp view '{name}' already exists. Use createOrReplaceGlobalTempView to overwrite."
        raise RuntimeError(msg)
    _global_sql_ctx.register(name, _frame_for_sql(self))


def createOrReplaceGlobalTempView(self: PolarsDataFrame | PolarsLazyDataFrame, name: str) -> None:
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
