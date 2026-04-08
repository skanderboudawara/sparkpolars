"""Polars SQL functions for Polyspark."""

import datetime
from collections.abc import Callable
from functools import reduce
from typing import Any

import polars as pl
import polars.datatypes as polars_datatypes
import polars.functions as polars_functions
from polars import lit
from polars.expr import Expr

# Apply all Expr (Column) monkey-patches first so every function in this
# module can rely on them being in place.
from . import columns as _columns  # noqa: F401 — side-effects only
from .columns import SparkWhen, _str_to_col


def broadcast(df: Any) -> Any:
    """Mark a DataFrame for broadcast (no-op in Polars).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1]})
        >>> sf.broadcast(df) is df
        True
    """
    return df


col = polars_functions.col
column = col
Column = col


def trim(col: str | Expr, trim_chars: str | None = None) -> Expr:
    """Strip leading and trailing whitespace (or specified characters).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["  hi  "]})
        >>> df.select(sf.trim(pl.col("s")))["s"][0]
        'hi'
    """
    return _str_to_col(col).trim(trim_chars)


def when(condition: Expr, value: Any) -> SparkWhen:
    """Evaluate a conditional expression (PySpark-style when/otherwise).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> df.select(sf.when(pl.col("x") > 1, pl.lit(99)).otherwise(pl.lit(0)).alias("r"))["r"].to_list()
        [0, 99, 99]
    """
    return SparkWhen(condition, value)


def concat_ws(separator: str, *cols: Any) -> Expr:
    """Concatenate columns with a separator.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": ["x"], "b": ["y"]})
        >>> df.select(sf.concat_ws("-", pl.col("a"), pl.col("b")))["a"][0]
        'x-y'
    """
    exprs = []
    for c in cols:
        if hasattr(c, "_expr"):
            exprs.append(c)
        else:
            c_ = _str_to_col(c)
            exprs.append(c_)
    if len(exprs) == 1:
        return exprs[0].list.join(separator)
    return polars_functions.concat_str(*exprs, separator=separator)


def expr(str: str) -> Expr:
    """Parse a SQL expression string into a Polars expression.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> df.select(sf.expr("x * 2").alias("r"))["r"].to_list()
        [2, 4, 6]
    """
    return polars_functions.sql_expr(str)


def upper(col: str | Expr) -> Expr:
    """Convert a string column to uppercase.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.upper(pl.col("s")))["s"][0]
        'HELLO'
    """
    return _str_to_col(col).upper()


ucase = upper


def lower(col: str | Expr) -> Expr:
    """Convert a string column to lowercase.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["HELLO"]})
        >>> df.select(sf.lower(pl.col("s")))["s"][0]
        'hello'
    """
    return _str_to_col(col).lower()


lcase = lower


def regexp_count(col: str | Expr, pattern: str) -> Expr:
    """Count the number of occurrences of a regex pattern.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.regexp_count(pl.col("s"), "l"))["s"][0]
        2
    """
    return _str_to_col(col).regexp_count(pattern)


def regexp_extract(col: str | Expr, pattern: str, idx: int) -> Expr:
    """Extract a substring matching a regex capture group.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["2023-01-15"]})
        >>> df.select(sf.regexp_extract(pl.col("s"), r"(\\d{4})", 1))["s"][0]
        '2023'
    """
    return _str_to_col(col).regexp_extract(pattern, idx)


def regexp_extract_all(col: str | Expr, pattern: str, idx: int | None = None) -> Expr:
    """Extract all substrings matching a regex pattern.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(sf.regexp_extract_all(pl.col("s"), r"\\w+"))["s"][0].to_list()
        ['hello', 'world']
    """
    if idx is not None:
        msg = "idx parameter is not supported in Polars for regexp_extract_all"
        raise NotImplementedError(msg)
    return _str_to_col(col).regexp_extract_all(pattern)


def regexp_replace(col: str | Expr, pattern: str, replacement: str) -> Expr:
    """Replace the first occurrence of a regex pattern.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.regexp_replace(pl.col("s"), "l", "r"))["s"][0]
        'herlo'
    """
    return _str_to_col(col).regexp_replace(pattern, replacement)


def replace(col: str | Expr, search: str, replacement: str | None = None) -> Expr:
    """Replace occurrences of a substring.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.replace(pl.col("s"), "l", "r"))["s"][0]
        'herlo'
        >>> df.select(sf.replace(pl.col("s"), "l"))["s"][0]
        'heo'
    """
    return _str_to_col(col).str_replace(search, replacement)


def rtrim(col: str | Expr, trim_chars: str | None = None) -> Expr:
    """Strip trailing whitespace (or specified characters).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["  hi  "]})
        >>> df.select(sf.rtrim(pl.col("s")))["s"][0]
        '  hi'
    """
    return _str_to_col(col).rtrim(trim_chars)


def ltrim(col: str | Expr, trim_chars: str | None = None) -> Expr:
    """Strip leading whitespace (or specified characters).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["  hi  "]})
        >>> df.select(sf.ltrim(pl.col("s")))["s"][0]
        'hi  '
    """
    return _str_to_col(col).ltrim(trim_chars)


def rpad(col: str | Expr, length: int, pad: str = " ") -> Expr:
    """Right-pad a string to the specified length.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hi"]})
        >>> df.select(sf.rpad(pl.col("s"), 5, "0"))["s"][0]
        'hi000'
    """
    return _str_to_col(col).rpad(length, pad)


def lpad(col: str | Expr, length: int, pad: str = " ") -> Expr:
    """Left-pad a string to the specified length.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hi"]})
        >>> df.select(sf.lpad(pl.col("s"), 5, "0"))["s"][0]
        '000hi'
    """
    return _str_to_col(col).lpad(length, pad)


def base64(col: str | Expr) -> Expr:
    """Encode a string column to Base64.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.base64(pl.col("s")))["s"][0]
        'aGVsbG8='
    """
    return _str_to_col(col).base64()


def btrim(col: str | Expr, trim_chars: str | None = None) -> Expr:
    """Strip leading and trailing whitespace (or specified characters).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["  hi  "]})
        >>> df.select(sf.btrim(pl.col("s")))["s"][0]
        'hi'
    """
    return _str_to_col(col).btrim(trim_chars)


def contains(col: str | Expr, substr: str) -> Expr:
    """Check if a string contains a given substring.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.filter(sf.contains(pl.col("s"), "world")).height
        1
    """
    return _str_to_col(col).contains(substr)


def encode(col: str | Expr, charset: str) -> Expr:
    """Encode a string column using the given charset ('hex' or 'base64').

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.encode(pl.col("s"), "hex"))["s"][0]
        '68656c6c6f'
    """
    return _str_to_col(col).encode(charset)


def decode(col: str | Expr, charset: str) -> Expr:
    """Decode a binary/encoded column using the given charset ('hex' or 'base64').

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["68656c6c6f"]})
        >>> df.select(sf.decode(pl.col("s"), "hex"))["s"][0]
        b'hello'
    """
    return _str_to_col(col).decode(charset)


def left(col: str | Expr, n: int) -> Expr:
    """Return the leftmost n characters of a string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.left(pl.col("s"), 3))["s"][0]
        'hel'
    """
    return _str_to_col(col).left(n)


def right(col: str | Expr, n: int) -> Expr:
    """Return the rightmost n characters of a string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.right(pl.col("s"), 3))["s"][0]
        'llo'
    """
    return _str_to_col(col).right(n)


def length(col: str | Expr) -> Expr:
    """Return the length of a string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.length(pl.col("s")))["s"][0]
        5
    """
    return _str_to_col(col).length()


def locate(substr: str, col: str | Expr, _pos: int | None = None) -> Expr:
    """Return 1-based position of the first occurrence of substr.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.locate("ll", pl.col("s")))["s"][0]
        2
    """
    return _str_to_col(col).locate(substr)


position = locate


def repeat(col: str | Expr, n: int) -> Expr:
    """Repeat a string n times.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["ab"]})
        >>> df.select(sf.repeat(pl.col("s"), 3))["s"][0]
        'ababab'
    """
    return _str_to_col(col).repeat(n)


def concat(*cols: str | Expr) -> Expr:
    """Concatenate columns via string casting.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(sf.concat(pl.col("a").cast(pl.String)))["a"].to_list()
        ['1', '2', '3']
    """
    return reduce(lambda acc, col: acc + col.cast(polars_datatypes.String(), strict=False), cols)


def round(col: str | Expr, scale: int = 0) -> Expr:
    """Round a numeric column to the given number of decimal places.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.567]})
        >>> df.select(sf.round(pl.col("x"), 1))["x"][0]
        1.6
    """
    return col.round(scale)


def sqrt(col: str | Expr) -> Expr:
    """Compute the square root of a numeric column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [9.0]})
        >>> df.select(sf.sqrt(pl.col("x")))["x"][0]
        3.0
    """
    return col.sqrt()


def pow(col: str | Expr, exponent: int | float) -> Expr:
    """Raise a numeric column to the given power.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [2.0]})
        >>> df.select(sf.pow(pl.col("x"), 3))["x"][0]
        8.0
    """
    return col.pow(exponent)


power = pow


def negate(col: str | Expr) -> Expr:
    """Negate a numeric column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, -2, 3]})
        >>> df.select(sf.negate(pl.col("x")))["x"].to_list()
        [-1, 2, -3]
    """
    return _str_to_col(col).negate()


negative = negate


def translate(col: str | Expr, matching: str, replace: str) -> Expr:
    """Translate characters using a character-for-character mapping.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["abc"]})
        >>> df.select(sf.translate(pl.col("s"), "abc", "xyz"))["s"][0]
        'xyz'
    """
    return _str_to_col(col).translate(matching, replace)


def collect_list(col: str | Expr) -> Expr:
    """Collect all values in a column into a list.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> df.select(sf.collect_list(pl.col("x")))["x"][0].to_list()
        [1, 2, 3]
    """
    return _str_to_col(col).collect_list()


array_agg = collect_list
listagg = collect_list


def collect_set(col: str | Expr) -> Expr:
    """Collect distinct values in a column into a list.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3]})
        >>> sorted(df.select(sf.collect_set(pl.col("x")))["x"][0].to_list())
        [1, 2, 3]
    """
    return _str_to_col(col).collect_set()


array_agg_distinct = collect_set
listagg_distinct = collect_set


def _col_name(col: str | Expr) -> str:
    """Extract a column name string for building PySpark-style agg aliases."""
    if isinstance(col, str):
        return col
    try:
        return col.meta.output_name()
    except Exception:
        return "col"


def sum(col: str | Expr) -> Expr:
    """Sum of values in a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> df.select(sf.sum(pl.col("x")))["sum(x)"][0]
        6
    """
    return _str_to_col(col).sum().alias(f"sum({_col_name(col)})")


def min(col: str | Expr) -> Expr:
    """Minimum value in a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> df.select(sf.min(pl.col("x")))["min(x)"][0]
        1
    """
    return _str_to_col(col).min().alias(f"min({_col_name(col)})")


def max(col: str | Expr) -> Expr:
    """Maximum value in a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> df.select(sf.max(pl.col("x")))["max(x)"][0]
        3
    """
    return _str_to_col(col).max().alias(f"max({_col_name(col)})")


def abs(col: str | Expr) -> Expr:
    """Absolute value -- scalar transform, keeps input column name.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [-1, 2, -3]})
        >>> df.select(sf.abs(pl.col("x")))["x"].to_list()
        [1, 2, 3]
    """
    return _str_to_col(col).abs()


def avg(col: str | Expr) -> Expr:
    """Average of values in a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.select(sf.avg(pl.col("x")))["avg(x)"][0]
        3.0
    """
    return _str_to_col(col).mean().alias(f"avg({_col_name(col)})")


def first(col: str | Expr, ignorenulls: bool = False) -> Expr:
    """Return the first value in a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [None, None, 3, 4]}, schema={"x": pl.Int32})
        >>> df.select(sf.first(pl.col("x"), ignorenulls=True))["first(x)"][0]
        3
    """
    e = _str_to_col(col)
    base = e.drop_nulls().first() if ignorenulls else e.first()
    return base.alias(f"first({_col_name(col)})")


first_value = first


def last(col: str | Expr, ignorenulls: bool = False) -> Expr:
    """Return the last value in a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, None, None]}, schema={"x": pl.Int32})
        >>> df.select(sf.last(pl.col("x"), ignorenulls=True))["last(x)"][0]
        2
    """
    e = _str_to_col(col)
    base = e.drop_nulls().last() if ignorenulls else e.last()
    return base.alias(f"last({_col_name(col)})")


last_value = last


def array_distinct(col: str | Expr) -> Expr:
    """Remove duplicate elements from an array.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 2, 3]]})
        >>> sorted(df.select(sf.array_distinct(pl.col("a")))["a"][0].to_list())
        [1, 2, 3]
    """
    return _str_to_col(col).array_distinct()


def greatest(*cols: str | Expr) -> Expr:
    """Return the greatest value across columns for each row.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [1, 5], "b": [3, 2]})
        >>> df.select(sf.greatest(pl.col("a"), pl.col("b")))["a"].to_list()
        [3, 5]
    """
    cols = [_str_to_col(col) for col in cols]
    return polars_functions.max_horizontal(*cols)


def least(*cols: str | Expr) -> Expr:
    """Return the least value across columns for each row.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [1, 5], "b": [3, 2]})
        >>> df.select(sf.least(pl.col("a"), pl.col("b")))["a"].to_list()
        [1, 2]
    """
    cols = [_str_to_col(col) for col in cols]
    return polars_functions.min_horizontal(*cols)


# Sort-direction standalone helpers — used by orderBy/_desc_status logic in dataframe.py.
# The Expr method variants (col("a").asc() etc.) are patched in columns.py.
from .columns import (  # noqa: E402
    asc,
    asc_nulls_first,
    asc_nulls_last,
    desc,
    desc_nulls_first,
    desc_nulls_last,
)


def array(*cols: str | Expr) -> Expr:
    """Combine columns into an array column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> df.select(sf.array(pl.col("a"), pl.col("b")).alias("arr"))["arr"][0].to_list()
        [1, 3]
    """
    if not cols:
        return polars_functions.concat_list(polars_functions.lit([]))
    cols = [_str_to_col(col) for col in cols]
    return polars_functions.concat_list(*cols)


def array_append(col: str | Expr, value: Any) -> Expr:
    """Append a value to the end of an array.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2]]})
        >>> df.select(sf.array_append(pl.col("a"), 3))["a"][0].to_list()
        [1, 2, 3]
    """
    return _str_to_col(col).array_append(value)


def array_compact(col: str | Expr) -> Expr:
    """Remove null values from an array.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, None, 2]]})
        >>> df.select(sf.array_compact(pl.col("a")))["a"][0].to_list()
        [1, 2]
    """
    return _str_to_col(col).array_compact()


def array_contains(col: str | Expr, value: Any) -> Expr:
    """Check if an array contains a given value.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(sf.array_contains(pl.col("a"), 2))["a"][0]
        True
    """
    return _str_to_col(col).array_contains(value)


def array_except(col1: str | Expr, col2: str | Expr) -> Expr:
    """Return elements in the first array but not in the second.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3]]})
        >>> df.select(sf.array_except(pl.col("a"), pl.col("b")))["a"][0].to_list()
        [1]
    """
    return _str_to_col(col1).array_except(_str_to_col(col2))


def array_intersect(col1: str | Expr, col2: str | Expr) -> Expr:
    """Return elements common to both arrays.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
        >>> sorted(df.select(sf.array_intersect(pl.col("a"), pl.col("b")))["a"][0].to_list())
        [2, 3]
    """
    return _str_to_col(col1).array_intersect(_str_to_col(col2))


def array_join(col: str | Expr, delimiter: str, null_replacement: str | None = None) -> Expr:
    """Join array elements into a string with a delimiter.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [["a", "b", "c"]]})
        >>> df.select(sf.array_join(pl.col("a"), ","))["a"][0]
        'a,b,c'
    """
    return _str_to_col(col).array_join(delimiter, null_replacement)


def array_max(col: str | Expr) -> Expr:
    """Return the maximum value in an array.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 3, 2]]})
        >>> df.select(sf.array_max(pl.col("a")))["a"][0]
        3
    """
    return _str_to_col(col).array_max()


def array_min(col: str | Expr) -> Expr:
    """Return the minimum value in an array.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 3, 2]]})
        >>> df.select(sf.array_min(pl.col("a")))["a"][0]
        1
    """
    return _str_to_col(col).array_min()


def array_size(col: str | Expr) -> Expr:
    """Return the number of elements in an array.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(sf.array_size(pl.col("a")))["a"][0]
        3
    """
    return _str_to_col(col).array_size()


size = array_size


def array_union(col1: str | Expr, col2: str | Expr) -> Expr:
    """Return the union of two arrays (distinct elements).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
        >>> sorted(df.select(sf.array_union(pl.col("a"), pl.col("b")))["a"][0].to_list())
        [1, 2, 3, 4]
    """
    return _str_to_col(col1).array_union(_str_to_col(col2))


def array_sort(col: str | Expr, comparator: Any = None) -> Expr:
    """Sort the elements of an array in ascending order.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[3, 1, 2]]})
        >>> df.select(sf.array_sort(pl.col("a")))["a"][0].to_list()
        [1, 2, 3]
    """
    if comparator is not None:
        msg = "Custom comparator for array_sort is not supported in Polars."
        raise NotImplementedError(msg)
    return _str_to_col(col).array_sort(asc=True)


def sort_array(col: str | Expr, asc: bool = True) -> Expr:
    """Sort the elements of an array.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[3, 1, 2]]})
        >>> df.select(sf.sort_array(pl.col("a"), asc=False))["a"][0].to_list()
        [3, 2, 1]
    """
    return _str_to_col(col).array_sort(asc=asc)


def slice(col: str | Expr, start: int, length: int | None = None) -> Expr:
    """Return a slice of an array starting at the given position.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
        >>> df.select(sf.slice(pl.col("a"), 2, 2))["a"][0].to_list()
        [2, 3]
    """
    return _str_to_col(col).array_slice(start, length)


def array_remove(col: str | Expr, element: Any) -> Expr:
    """Remove all occurrences of an element from an array.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 1, 3]]})
        >>> df.select(sf.array_remove(pl.col("a"), 1))["a"][0].to_list()
        [2, 3]
    """
    return _str_to_col(col).array_remove(element)


def flatten(col: str | Expr) -> Expr:
    """Flatten nested arrays (List[List[T]] to List[T]) per row.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(sf.flatten(pl.col("a")))["a"].to_list()
        [1, 2, 3]
    """
    # Native Expr.flatten() flattens List[List[T]] → List[T] per row
    c = _str_to_col(col)
    return c.flatten()


def split(col: str | Expr, pattern: str, limit: int = -1) -> Expr:
    """Split a string by a delimiter pattern.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["a,b,c"]})
        >>> df.select(sf.split(pl.col("s"), ","))["s"][0].to_list()
        ['a', 'b', 'c']
    """
    return _str_to_col(col).split(pattern, limit)


def explode(col: str | Expr) -> Expr:
    """Mark an expression as needing explode treatment.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> expr = sf.explode(pl.col("a"))
        >>> hasattr(expr, '_explode_marker') and expr._explode_marker
        True
    """
    col_expr = _str_to_col(col)
    # Add a special marker to indicate this should use DataFrame.explode()
    col_expr._explode_marker = True
    col_expr._explode_outer = False
    col_expr._explode_column = col if isinstance(col, str) else col.name
    return col_expr


def explode_outer(col: str | Expr) -> Expr:
    """Mark an expression as needing explode treatment with outer join.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> expr = sf.explode_outer(pl.col("a"))
        >>> hasattr(expr, '_explode_outer') and expr._explode_outer
        True
    """
    col_expr = _str_to_col(col)
    # Add a special marker to indicate this should use DataFrame.explode()
    col_expr._explode_marker = True
    col_expr._explode_outer = True
    col_expr._explode_column = col if isinstance(col, str) else col.name
    return col_expr


def count(col: str | Expr = "*") -> Expr:
    """Count non-null values in a column, or total rows if '*'.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> df.select(sf.count("x"))["count(x)"][0]
        3
    """
    name = col if isinstance(col, str) else _col_name(col)
    if col == "*" or col == "1":
        return pl.len().alias("count(1)")
    return _str_to_col(col).count().alias(f"count({name})")


coalesce = polars_functions.coalesce


def monotonically_increasing_id() -> Expr:
    """Generate a monotonically increasing 64-bit integer ID for each row.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [10, 20, 30]})
        >>> df.select(sf.monotonically_increasing_id().alias("id"))["id"].to_list()
        [0, 1, 2]
    """
    return polars_functions.int_range(pl.len(), dtype=pl.UInt64)


def product(col: str | Expr) -> Expr:
    """Compute the product of all values in a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [2, 3, 4]})
        >>> df.select(sf.product(pl.col("x")))["x"][0]
        24
    """
    return _str_to_col(col).product()


def year(col: str | Expr) -> Expr:
    """Extract the year from a date or datetime column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(sf.year(pl.col("d")))["d"][0]
        2023
    """
    return _str_to_col(col).year()


def month(col: str | Expr) -> Expr:
    """Extract the month from a date or datetime column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(sf.month(pl.col("d")))["d"][0]
        1
    """
    return _str_to_col(col).month()


def hour(col: str | Expr) -> Expr:
    """Extract the hour from a datetime column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 1, 16, 14, 30, 0)]})
        >>> df.select(sf.hour(pl.col("t")))["t"][0]
        14
    """
    return _str_to_col(col).hour()


def last_day(col: str | Expr) -> Expr:
    """Return the last day of the month for a date column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(sf.last_day(pl.col("d")))["d"][0]
        datetime.date(2023, 1, 31)
    """
    return _str_to_col(col).last_day()


def dayofmonth(col: str | Expr) -> Expr:
    """Extract the day of month from a date column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(sf.dayofmonth(pl.col("d")))["d"][0]
        16
    """
    return _str_to_col(col).dayofmonth()


def dayofweek(col: str | Expr) -> Expr:
    """Extract the day of week from a date column (1=Sunday, 2=Monday, ...).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(sf.dayofweek(pl.col("d")))["d"][0]
        2
    """
    return _str_to_col(col).dayofweek()


def dayofyear(col: str | Expr) -> Expr:
    """Extract the day of year from a date column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(sf.dayofyear(pl.col("d")))["d"][0]
        16
    """
    return _str_to_col(col).dayofyear()


def current_date() -> Expr:
    """Return the current date as a literal.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> result = pl.select(sf.current_date().alias("d"))
        >>> result.height
        1
    """
    return polars_functions.lit(datetime.datetime.now().date(), dtype=polars_datatypes.Date)  # noqa: DTZ005


now = current_date
curdate = current_date


def current_timestamp() -> Expr:
    """Return the current timestamp as a literal.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> result = pl.select(sf.current_timestamp().alias("t"))
        >>> result.height
        1
    """
    return polars_functions.lit(datetime.datetime.now(), dtype=polars_datatypes.Datetime)  # noqa: DTZ005


localtimestamp = current_timestamp


def date_sub(col: str | Expr, days: int) -> Expr:
    """Subtract days from a date column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(sf.date_sub(pl.col("d"), 5))["d"][0]
        datetime.date(2023, 1, 11)
    """
    return _str_to_col(col).date_sub(days)


def date_add(col: str | Expr, days: int) -> Expr:
    """Add days to a date column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(sf.date_add(pl.col("d"), 5))["d"][0]
        datetime.date(2023, 1, 21)
    """
    return _str_to_col(col).date_add(days)


def datediff(end: str | Expr, start: str | Expr) -> Expr:
    """Return the number of days between two date columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"a": [datetime.date(2023, 1, 10)], "b": [datetime.date(2023, 1, 1)]})
        >>> df.select(sf.datediff(pl.col("a"), pl.col("b")).alias("d"))["d"][0]
        datetime.timedelta(days=9)
    """
    end = _str_to_col(end)
    start = _str_to_col(start)
    return end - start


def add_months(col: str | Expr, months: int) -> Expr:
    """Add months to a date column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(sf.add_months(pl.col("d"), 2))["d"][0]
        datetime.date(2023, 3, 16)
    """
    return _str_to_col(col).add_months(months)


def sequence(start: int | Expr, stop: int | Expr, step: int | None = None) -> Expr:
    """Generate a sequence of integers from start to stop (inclusive).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"start": [1], "stop": [5]})
        >>> df.select(sf.sequence(pl.col("start"), pl.col("stop")).alias("s"))["s"][0].to_list()
        [1, 2, 3, 4, 5]
    """
    start = _str_to_col(start)
    stop = _str_to_col(stop)
    if step is None:
        return polars_functions.int_ranges(start, stop + 1)
    return polars_functions.int_ranges(start, stop + 1, step=step)


def create_map(dict: dict) -> Expr:
    """Create a map (JSON-encoded struct) from a dictionary of literal values.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import json
        >>> df = pl.DataFrame({"x": [1]})
        >>> result = df.select(sf.create_map({"k": "a", "v": 1}).alias("m"))["m"][0]
        >>> json.loads(result)["k"]
        'a'
    """
    return polars_functions.struct(**{k: polars_functions.lit(v) for k, v in dict.items()}).struct.json_encode()


def md5(col: str | Expr) -> Expr:
    """Compute the MD5 hash of a string column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import hashlib
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.md5(pl.col("s")))["s"][0] == hashlib.md5(b"hello").hexdigest()
        True
    """
    return _str_to_col(col).md5()


def sha1(col: str | Expr) -> Expr:
    """Compute the SHA-1 hash of a string column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import hashlib
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.sha1(pl.col("s")))["s"][0] == hashlib.sha1(b"hello").hexdigest()
        True
    """
    return _str_to_col(col).sha1()


def sha256(col: str | Expr) -> Expr:
    """Compute the SHA-256 hash of a string column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import hashlib
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.sha256(pl.col("s")))["s"][0] == hashlib.sha256(b"hello").hexdigest()
        True
    """
    return _str_to_col(col).sha256()


def transform(col: str | Expr, f: Callable) -> Expr:
    """Transform each element in a list column using a function.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(sf.transform(pl.col("a"), lambda e: e * 2))["a"][0].to_list()
        [2, 4, 6]
    """
    return _str_to_col(col).transform(f)


def filter(col: str | Expr, f: Callable) -> Expr:
    """Filter elements in a list column based on a predicate.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
        >>> df.select(sf.filter(pl.col("a"), lambda e: e > 2))["a"][0].to_list()
        [3, 4]
    """
    return _str_to_col(col).list_filter(f)


def forall(col: str | Expr, f: Callable) -> Expr:
    """Check if all elements in a list column satisfy a predicate.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[2, 4, 6]]})
        >>> df.select(sf.forall(pl.col("a"), lambda e: e > 0))["a"][0]
        True
    """
    return _str_to_col(col).forall(f)


def reverse(col: str | Expr) -> Expr:
    """Reverse a string or array column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.reverse(pl.col("s")))["s"][0]
        'olleh'
    """
    return _str_to_col(col).reverse()


# ── missing standalone aliases ────────────────────────────────────────────────

def substr(col: str | Expr, pos: int, length: int | None = None) -> Expr:
    """Return a substring starting at pos (1-based).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.substr(pl.col("s"), 2, 3))["s"][0]
        'ell'
    """
    return _str_to_col(col).substr(pos, length)


substring = substr


def struct(*cols: str | Expr) -> Expr:
    """Combine columns into a struct.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [1], "b": [2]})
        >>> df.select(sf.struct(pl.col("a"), pl.col("b")).alias("s"))["s"][0]
        {'a': 1, 'b': 2}
    """
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.struct(exprs)


# ── math functions ────────────────────────────────────────────────────────────

def ceil(col: str | Expr) -> Expr:
    """Round up to the nearest integer.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.2, -1.9]})
        >>> df.select(sf.ceil(pl.col("x")))["x"].to_list()
        [2.0, -1.0]
    """
    return _str_to_col(col).ceil()


def floor(col: str | Expr) -> Expr:
    """Round down to the nearest integer.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.9, -1.2]})
        >>> df.select(sf.floor(pl.col("x")))["x"].to_list()
        [1.0, -2.0]
    """
    return _str_to_col(col).floor()


def log(col: str | Expr, base: float = 2.718281828459045) -> Expr:
    """Natural log by default; pass base= for other bases.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> df = pl.DataFrame({"x": [1.0]})
        >>> builtins.round(df.select(sf.log(pl.col("x")))["x"][0], 6)
        0.0
    """
    return _str_to_col(col).log(base)


def log2(col: str | Expr) -> Expr:
    """Logarithm base 2.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [8.0]})
        >>> df.select(sf.log2(pl.col("x")))["x"][0]
        3.0
    """
    return _str_to_col(col).log(2.0)


def log10(col: str | Expr) -> Expr:
    """Logarithm base 10.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [100.0]})
        >>> df.select(sf.log10(pl.col("x")))["x"][0]
        2.0
    """
    return _str_to_col(col).log(10.0)


def exp(col: str | Expr) -> Expr:
    """Compute the exponential (e^x).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.exp(pl.col("x")))["x"][0]
        1.0
    """
    return _str_to_col(col).exp()


def signum(col: str | Expr) -> Expr:
    """Return the sign of a number (-1, 0, or 1).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [3.0, -2.0, 0.0]})
        >>> df.select(sf.signum(pl.col("x")))["x"].to_list()
        [1.0, -1.0, 0.0]
    """
    return _str_to_col(col).sign()


sign = signum


def cbrt(col: str | Expr) -> Expr:
    """Compute the cube root.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [27.0]})
        >>> df.select(sf.cbrt(pl.col("x")))["x"][0]
        3.0
    """
    return _str_to_col(col) ** (1.0 / 3.0)


# ── null-handling functions ───────────────────────────────────────────────────

def nvl(col: str | Expr, replacement: Any) -> Expr:
    """Return replacement when col is null (alias: ifnull).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [None, 2, None]})
        >>> df.select(sf.nvl(pl.col("a"), -1))["a"].to_list()
        [-1, 2, -1]
    """
    c = _str_to_col(col)
    r = replacement if isinstance(replacement, Expr) else lit(replacement)
    return polars_functions.coalesce(c, r)


ifnull = nvl


def nullif(col: str | Expr, value: Any) -> Expr:
    """Return null when col equals value, else col.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(sf.nullif(pl.col("a"), 2))["a"].to_list()
        [1, None, 3]
    """
    return _str_to_col(col).nullif(value)


# ── date / time functions ─────────────────────────────────────────────────────

def to_date(col: str | Expr, fmt: str = "%Y-%m-%d") -> Expr:
    """Parse a string column to a Date using the given format.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"s": ["2023-06-15"]})
        >>> df.select(sf.to_date(pl.col("s")))["s"][0]
        datetime.date(2023, 6, 15)
    """
    return _str_to_col(col).str.to_date(fmt)


def to_timestamp(col: str | Expr, fmt: str = "%Y-%m-%d %H:%M:%S") -> Expr:
    """Parse a string column to a Datetime using the given format.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"s": ["2023-06-15 10:30:00"]})
        >>> df.select(sf.to_timestamp(pl.col("s")))["s"][0]
        datetime.datetime(2023, 6, 15, 10, 30)
    """
    return _str_to_col(col).str.to_datetime(fmt)


def date_format(col: str | Expr, fmt: str) -> Expr:
    """Format a date/datetime column as a string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
        >>> df.select(sf.date_format(pl.col("d"), "%Y/%m/%d"))["d"][0]
        '2023/06/15'
    """
    return _str_to_col(col).dt.strftime(fmt)


def unix_timestamp(col: str | Expr) -> Expr:
    """Seconds since Unix epoch (1970-01-01 00:00:00 UTC).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 6, 15, 0, 0, 0)]})
        >>> df.select(sf.unix_timestamp(pl.col("t")))["t"][0] > 0
        True
    """
    return _str_to_col(col).dt.epoch(time_unit="s")


def from_unixtime(col: str | Expr) -> Expr:
    """Convert integer seconds since epoch to a Datetime.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"ts": [0]})
        >>> df.select(sf.from_unixtime(pl.col("ts")))["ts"][0] is not None
        True
    """
    return polars_functions.from_epoch(_str_to_col(col), time_unit="s")


# ── string helper ─────────────────────────────────────────────────────────────

def format_string(fmt: str, *cols: str | Expr) -> Expr:
    """Equivalent to PySpark format_string / printf-style interpolation.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": ["hello"], "b": [42]})
        >>> df.select(sf.format_string("{} {}", pl.col("a"), pl.col("b")))["a"][0]
        'hello 42'
    """
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.format(fmt.replace("%s", "{}").replace("%d", "{}"), *exprs)


# ── array helpers ─────────────────────────────────────────────────────────────

def array_position(col: str | Expr, value: Any) -> Expr:
    """Return the 1-based index of value in the array, or 0 if not found.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[10, 20, 30]]})
        >>> df.select(sf.array_position(pl.col("a"), 20))["a"][0]
        2
        >>> df.select(sf.array_position(pl.col("a"), 99))["a"][0]
        0
    """
    c = _str_to_col(col)
    v = value if isinstance(value, Expr) else lit(value)
    # Polars list.eval gives 0-based index via arg_where; +1 for Spark compatibility
    idx = c.list.eval(pl.arg_where(pl.element() == v).first()).list.first()
    return polars_functions.when(c.list.contains(v)).then(idx + 1).otherwise(lit(0))


def array_prepend(col: str | Expr, value: Any) -> Expr:
    """Prepend a value to the beginning of an array.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[10, 20, 30]]})
        >>> df.select(sf.array_prepend(pl.col("a"), 5))["a"][0].to_list()
        [5, 10, 20, 30]
    """
    return _str_to_col(col).array_prepend(value)


# ── trig functions ────────────────────────────────────────────────────────────

def sin(col: str | Expr) -> Expr:
    """Compute the sine.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.sin(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).sin()


def cos(col: str | Expr) -> Expr:
    """Compute the cosine.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.cos(pl.col("x")))["x"][0]
        1.0
    """
    return _str_to_col(col).cos()


def tan(col: str | Expr) -> Expr:
    """Compute the tangent.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.tan(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).tan()


def asin(col: str | Expr) -> Expr:
    """Compute the arcsine.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.asin(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).arcsin()


def acos(col: str | Expr) -> Expr:
    """Compute the arccosine.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0]})
        >>> df.select(sf.acos(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).arccos()


def atan(col: str | Expr) -> Expr:
    """Compute the arctangent.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.atan(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).arctan()


def atan2(y: str | Expr, x: str | Expr) -> Expr:
    """Compute the two-argument arctangent.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> df = pl.DataFrame({"y": [1.0], "x": [0.0]})
        >>> builtins.round(df.select(sf.atan2(pl.col("y"), pl.col("x")).alias("r"))["r"][0], 4)
        1.5708
    """
    import math
    y_expr = _str_to_col(y).cast(polars_datatypes.Float64(), strict=False)
    x_expr = _str_to_col(x).cast(polars_datatypes.Float64(), strict=False)
    return polars_functions.struct([y_expr.alias("_y"), x_expr.alias("_x")]).map_elements(
        lambda row: math.atan2(row["_y"], row["_x"]),
        return_dtype=polars_datatypes.Float64(),
    )


def sinh(col: str | Expr) -> Expr:
    """Compute the hyperbolic sine.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.sinh(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).sinh()


def cosh(col: str | Expr) -> Expr:
    """Compute the hyperbolic cosine.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.cosh(pl.col("x")))["x"][0]
        1.0
    """
    return _str_to_col(col).cosh()


def tanh(col: str | Expr) -> Expr:
    """Compute the hyperbolic tangent.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.tanh(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).tanh()


def asinh(col: str | Expr) -> Expr:
    """Compute the inverse hyperbolic sine.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.asinh(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).arcsinh()


def acosh(col: str | Expr) -> Expr:
    """Compute the inverse hyperbolic cosine.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0]})
        >>> df.select(sf.acosh(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).arccosh()


def atanh(col: str | Expr) -> Expr:
    """Compute the inverse hyperbolic tangent.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.atanh(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).arctanh()


def degrees(col: str | Expr) -> Expr:
    """Convert radians to degrees.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import math
        >>> import builtins
        >>> df = pl.DataFrame({"x": [math.pi]})
        >>> builtins.round(df.select(sf.degrees(pl.col("x")))["x"][0])
        180
    """
    return _str_to_col(col).degrees()


def radians(col: str | Expr) -> Expr:
    """Convert degrees to radians.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> df = pl.DataFrame({"x": [180.0]})
        >>> builtins.round(df.select(sf.radians(pl.col("x")))["x"][0], 4)
        3.1416
    """
    return _str_to_col(col).radians()


# ── string extras ─────────────────────────────────────────────────────────────

def initcap(col: str | Expr) -> Expr:
    """Capitalize the first letter of each word.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(sf.initcap(pl.col("s")))["s"][0]
        'Hello World'
    """
    return _str_to_col(col).initcap()


def ascii(col: str | Expr) -> Expr:
    """Return the ASCII code of the first character.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["A"]})
        >>> df.select(sf.ascii(pl.col("s")))["s"][0]
        65
    """
    return _str_to_col(col).ascii_code()


def instr(col: str | Expr, substr: str) -> Expr:
    """Return the 1-based position of the first occurrence of substr, or 0 if not found.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(sf.instr(pl.col("s"), "world"))["s"][0]
        7
        >>> df2 = pl.DataFrame({"s": ["hello"]})
        >>> df2.select(sf.instr(pl.col("s"), "xyz"))["s"][0]
        0
    """
    return _str_to_col(col).instr(substr)


def split_part(col: str | Expr, delimiter: str, part_num: int) -> Expr:
    """Split a string by delimiter and return the nth part (1-based).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["a,b,c"]})
        >>> df.select(sf.split_part(pl.col("s"), ",", 2))["s"][0]
        'b'
    """
    return _str_to_col(col).split_part(delimiter, part_num)


def substring_index(col: str | Expr, delimiter: str, count: int) -> Expr:
    """Return the substring before the nth occurrence of delimiter.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["a.b.c.d"]})
        >>> df.select(sf.substring_index(pl.col("s"), ".", 2))["s"][0]
        'a.b'
        >>> df.select(sf.substring_index(pl.col("s"), ".", -2))["s"][0]
        'c.d'
    """
    return _str_to_col(col).substring_index(delimiter, count)


# ── date extras ───────────────────────────────────────────────────────────────

def quarter(col: str | Expr) -> Expr:
    """Extract the quarter from a date column (1-4).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 4, 15)]})
        >>> df.select(sf.quarter(pl.col("d")))["d"][0]
        2
    """
    return _str_to_col(col).quarter()


def minute(col: str | Expr) -> Expr:
    """Extract the minute from a datetime column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 1, 1, 14, 35, 0)]})
        >>> df.select(sf.minute(pl.col("t")))["t"][0]
        35
    """
    return _str_to_col(col).minute()


def second(col: str | Expr) -> Expr:
    """Extract the second from a datetime column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 1, 1, 14, 35, 45)]})
        >>> df.select(sf.second(pl.col("t")))["t"][0]
        45
    """
    return _str_to_col(col).second()


def weekofyear(col: str | Expr) -> Expr:
    """Extract the ISO week of year from a date column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 9)]})
        >>> df.select(sf.weekofyear(pl.col("d")))["d"][0]
        2
    """
    return _str_to_col(col).weekofyear()


def weekday(col: str | Expr) -> Expr:
    """Extract the weekday (0=Monday, 6=Sunday) from a date column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 9)]})
        >>> df.select(sf.weekday(pl.col("d")))["d"][0]
        0
    """
    return _str_to_col(col).weekday()


def date_trunc(unit: str, col: str | Expr) -> Expr:
    """Truncate date/timestamp to the given unit (Spark arg order: unit first).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
        >>> df.select(sf.date_trunc("month", pl.col("d")))["d"][0]
        datetime.date(2023, 6, 1)
        >>> df.select(sf.date_trunc("year", pl.col("d")))["d"][0]
        datetime.date(2023, 1, 1)
    """
    return _str_to_col(col).date_trunc(unit)


def make_date(year: Any, month: Any, day: Any) -> Expr:
    """Construct a Date from year, month, day (each may be int or Expr).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"y": [2023], "m": [6], "d": [15]})
        >>> df.select(sf.make_date(pl.col("y"), pl.col("m"), pl.col("d")).alias("dt"))["dt"][0]
        datetime.date(2023, 6, 15)
    """
    y = _str_to_col(year) if isinstance(year, (str, Expr)) else polars_functions.lit(year)
    m = _str_to_col(month) if isinstance(month, (str, Expr)) else polars_functions.lit(month)
    d = _str_to_col(day) if isinstance(day, (str, Expr)) else polars_functions.lit(day)
    return polars_functions.date(y, m, d)


# ── aggregate extras ──────────────────────────────────────────────────────────

def stddev(col: str | Expr) -> Expr:
    """Sample standard deviation of a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> builtins.round(df.select(sf.stddev(pl.col("x")))["stddev(x)"][0], 1)
        1.0
    """
    return _str_to_col(col).std().alias(f"stddev({_col_name(col)})")


std = stddev
stddev_samp = stddev


def stddev_pop(col: str | Expr) -> Expr:
    """Population standard deviation of a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> df.select(sf.stddev_pop(pl.col("x")))["stddev_pop(x)"][0] is not None
        True
    """
    return _str_to_col(col).std(ddof=0).alias(f"stddev_pop({_col_name(col)})")


def variance(col: str | Expr) -> Expr:
    """Sample variance of a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> builtins.round(df.select(sf.variance(pl.col("x")))["variance(x)"][0], 1)
        1.0
    """
    return _str_to_col(col).var().alias(f"variance({_col_name(col)})")


var = variance
var_samp = variance


def var_pop(col: str | Expr) -> Expr:
    """Population variance of a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> df.select(sf.var_pop(pl.col("x")))["var_pop(x)"][0] is not None
        True
    """
    return _str_to_col(col).var(ddof=0).alias(f"var_pop({_col_name(col)})")


def median(col: str | Expr) -> Expr:
    """Compute the median value.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.select(sf.median(pl.col("x")))["median(x)"][0]
        3.0
    """
    return _str_to_col(col).median().alias(f"median({_col_name(col)})")


def count_distinct(*cols: str | Expr) -> Expr:
    """Count the number of distinct values.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3, 3]})
        >>> df.select(sf.count_distinct(pl.col("x")).alias("n"))["n"][0]
        3
    """
    if len(cols) == 1:
        return _str_to_col(cols[0]).n_unique()
    # multi-column distinct count via struct
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.struct(exprs).n_unique()


def count_if(condition: Expr) -> Expr:
    """Count rows where condition is true.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        >>> df.select(sf.count_if(pl.col("x") > 2).alias("n"))["n"][0]
        3
    """
    return polars_functions.when(condition).then(polars_functions.lit(1)).otherwise(polars_functions.lit(None)).count()


def bool_and(col: str | Expr) -> Expr:
    """Return True if all values in a boolean column are True.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [True, True, True]})
        >>> df.select(sf.bool_and(pl.col("x")).alias("r"))["r"][0]
        True
        >>> df2 = pl.DataFrame({"x": [True, False, True]})
        >>> df2.select(sf.bool_and(pl.col("x")).alias("r"))["r"][0]
        False
    """
    return _str_to_col(col).all()


every = bool_and


def bool_or(col: str | Expr) -> Expr:
    """Return True if any value in a boolean column is True.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [False, False, True]})
        >>> df.select(sf.bool_or(pl.col("x")).alias("r"))["r"][0]
        True
    """
    return _str_to_col(col).any()


some = bool_or


# ── null extras ───────────────────────────────────────────────────────────────

def nanvl(col: str | Expr, replacement: Any) -> Expr:
    """Replace NaN values with the given replacement.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [float("nan"), 2.0, float("nan")]})
        >>> df.select(sf.nanvl(pl.col("x"), -1.0))["x"].to_list()
        [-1.0, 2.0, -1.0]
    """
    return _str_to_col(col).nanvl(replacement)


def nvl2(col: str | Expr, not_null_val: Any, null_val: Any) -> Expr:
    """Return not_null_val when col is not null, null_val otherwise.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [1, None, 3]})
        >>> df.select(sf.nvl2(pl.col("a"), 100, 0))["a"].to_list()
        [100, 0, 100]
    """
    return _str_to_col(col).nvl2(not_null_val, null_val)


# ── bitwise extras ────────────────────────────────────────────────────────────

def bitwise_not(col: str | Expr) -> Expr:
    """Compute the bitwise NOT of an integer column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [5]})
        >>> df.select(sf.bitwise_not(pl.col("x")))["x"][0]
        -6
    """
    return _str_to_col(col).bitwiseNOT()


def shiftleft(col: str | Expr, n: int) -> Expr:
    """Left shift an integer by n bits.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1]})
        >>> df.select(sf.shiftleft(pl.col("x"), 3))["x"][0]
        8
    """
    return _str_to_col(col).shiftLeft(n)


def shiftright(col: str | Expr, n: int) -> Expr:
    """Right shift an integer by n bits.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [8]})
        >>> df.select(sf.shiftright(pl.col("x"), 2))["x"][0]
        2
    """
    return _str_to_col(col).shiftRight(n)


# ── array extras ─────────────────────────────────────────────────────────────

def element_at(col: str | Expr, index: int) -> Expr:
    """Return element at 1-based index (negative counts from end).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[10, 20, 30]]})
        >>> df.select(sf.element_at(pl.col("a"), 2))["a"][0]
        20
        >>> df.select(sf.element_at(pl.col("a"), -1))["a"][0]
        30
    """
    return _str_to_col(col).element_at(index)


def arrays_overlap(col1: str | Expr, col2: str | Expr) -> Expr:
    """Return True if two arrays share at least one element.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[3, 4, 5]]})
        >>> df.select(sf.arrays_overlap(pl.col("a"), pl.col("b")))["a"][0]
        True
        >>> df2 = pl.DataFrame({"a": [[1, 2]], "b": [[3, 4]]})
        >>> df2.select(sf.arrays_overlap(pl.col("a"), pl.col("b")))["a"][0]
        False
    """
    return _str_to_col(col1).arrays_overlap(_str_to_col(col2))


def array_repeat(element: Any, count: int) -> Expr:
    """Create an array by repeating element count times.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1]})
        >>> df.select(sf.array_repeat(pl.lit(7), 3).alias("a"))["a"][0].to_list()
        [7, 7, 7]
    """
    e = element if isinstance(element, Expr) else polars_functions.lit(element)
    if count <= 0:
        return polars_functions.lit(pl.Series(values=[[]], dtype=pl.List(pl.Null)))
    return polars_functions.concat_list([e] * count)


# ── window / ranking functions ────────────────────────────────────────────────

from .window import _WindowFuncExpr  # noqa: E402


def row_number() -> _WindowFuncExpr:
    """Row number (1-based) within the window partition, ordered by orderBy.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> from src.sparkpolars.polyspark.sql.window import Window, WindowSpec
        >>> df = pl.DataFrame({"x": [10, 20, 30]})
        >>> df.select(sf.row_number().over(WindowSpec()).alias("rn"))["rn"].to_list()
        [1, 2, 3]
    """
    return _WindowFuncExpr("row_number")


def rank() -> _WindowFuncExpr:
    """Min rank within the window partition (ties share the lowest rank).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> from src.sparkpolars.polyspark.sql.window import Window, WindowSpec
        >>> df = pl.DataFrame({"x": [3, 1, 2]})
        >>> sorted(df.select(sf.rank().over(Window.orderBy("x")).alias("r"))["r"].to_list())
        [1, 2, 3]
    """
    return _WindowFuncExpr("rank")


def dense_rank() -> _WindowFuncExpr:
    """Dense rank within the window partition (no gaps after ties).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> from src.sparkpolars.polyspark.sql.window import Window, WindowSpec
        >>> df = pl.DataFrame({"x": [3, 1, 2]})
        >>> sorted(df.select(sf.dense_rank().over(Window.orderBy("x")).alias("r"))["r"].to_list())
        [1, 2, 3]
    """
    return _WindowFuncExpr("dense_rank")


def lag(col: str | Expr, offset: int = 1, default: Any = None) -> _WindowFuncExpr:
    """Value of *col* offset rows before the current row within the partition.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> from src.sparkpolars.polyspark.sql.window import Window, WindowSpec
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4]})
        >>> df.select(sf.lag(pl.col("x"), 1).over(WindowSpec()).alias("l"))["l"].to_list()
        [None, 1, 2, 3]
        >>> df2 = pl.DataFrame({"x": [1, 2, 3]})
        >>> df2.select(sf.lag(pl.col("x"), 1, 0).over(WindowSpec()).alias("l"))["l"].to_list()
        [0, 1, 2]
    """
    return _WindowFuncExpr("lag", _str_to_col(col), offset, default)


def lead(col: str | Expr, offset: int = 1, default: Any = None) -> _WindowFuncExpr:
    """Value of *col* offset rows after the current row within the partition.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> from src.sparkpolars.polyspark.sql.window import Window, WindowSpec
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4]})
        >>> df.select(sf.lead(pl.col("x"), 1).over(WindowSpec()).alias("l"))["l"].to_list()
        [2, 3, 4, None]
    """
    return _WindowFuncExpr("lead", _str_to_col(col), offset, default)


# ── math extras ───────────────────────────────────────────────────────────────

import math as _math  # noqa: E402


def pi() -> Expr:
    """Return the value of pi as a literal.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> builtins.round(pl.select(sf.pi().alias("p"))["p"][0], 4)
        3.1416
    """
    return polars_functions.lit(_math.pi)


def e() -> Expr:
    """Return the value of Euler's number (e) as a literal.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> builtins.round(pl.select(sf.e().alias("v"))["v"][0], 4)
        2.7183
    """
    return polars_functions.lit(_math.e)


def rand(seed: int | None = None) -> Expr:
    """Random double in [0, 1) for each row.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> len(df.select(sf.rand(seed=42).alias("r"))["r"])
        3
    """
    import random as _random
    rng = _random.Random(seed)
    return polars_functions.int_range(pl.len(), dtype=polars_datatypes.UInt64()).map_elements(
        lambda _: rng.random(), polars_datatypes.Float64()
    )


def randn(seed: int | None = None) -> Expr:
    """Random standard-normal double for each row.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> len(df.select(sf.randn(seed=0).alias("r"))["r"])
        3
    """
    import random as _random
    rng = _random.Random(seed)
    return polars_functions.int_range(pl.len(), dtype=polars_datatypes.UInt64()).map_elements(
        lambda _: rng.gauss(0.0, 1.0), polars_datatypes.Float64()
    )


def factorial(col: str | Expr) -> Expr:
    """Compute the factorial.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [5]})
        >>> df.select(sf.factorial(pl.col("x")))["x"][0]
        120
    """
    return _str_to_col(col).map_elements(
        lambda x: _math.factorial(int(x)) if x is not None else None,
        polars_datatypes.Int64(),
    )


def hex(col: str | Expr) -> Expr:
    """Convert integer to uppercase hexadecimal string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [255]})
        >>> df.select(sf.hex(pl.col("x")))["x"][0]
        'FF'
    """
    return _str_to_col(col).map_elements(
        lambda x: format(int(x), "X") if x is not None else None,
        polars_datatypes.String(),
    )


def unhex(col: str | Expr) -> Expr:
    """Convert hexadecimal string to integer.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": ["FF"]})
        >>> df.select(sf.unhex(pl.col("x")))["x"][0]
        255
    """
    return _str_to_col(col).map_elements(
        lambda x: int(x, 16) if x is not None else None,
        polars_datatypes.Int64(),
    )


# ── misc extras ───────────────────────────────────────────────────────────────

lit = polars_functions.lit
typedLit = polars_functions.lit
typed_lit = polars_functions.lit


# ── string extras (batch 2) ───────────────────────────────────────────────────

def chr(col: str | Expr) -> Expr:
    """Convert Unicode code point to character.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [65, 66, 67]})
        >>> df.select(sf.chr(pl.col("x")))["x"].to_list()
        ['A', 'B', 'C']
    """
    return _str_to_col(col).chr()


def find_in_set(str_col: str | Expr, str_array_col: str | Expr) -> Expr:
    """Return 1-based position of str_col in comma-delimited str_array_col; 0 if not found.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["a,b,c,d"]})
        >>> df.select(sf.find_in_set("c", pl.col("s")))["s"][0]
        3
        >>> df2 = pl.DataFrame({"s": ["a,b,c"]})
        >>> df2.select(sf.find_in_set("z", pl.col("s")))["s"][0]
        0
    """
    sv = _str_to_col(str_col) if isinstance(str_col, Expr) else str_col
    return _str_to_col(str_array_col).find_in_set(sv)


def regexp_like(col: str | Expr, pattern: str) -> Expr:
    """Return True if the string matches the regex pattern.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello123", "world"]})
        >>> df.select(sf.regexp_like(pl.col("s"), r"\\d+"))["s"].to_list()
        [True, False]
    """
    return _str_to_col(col).regexp_like(pattern)


printf = format_string  # alias


def overlay(col: str | Expr, replace: Any, pos: int, length: int | None = None) -> Expr:
    """Overlay a string with a replacement starting at pos.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(sf.overlay(pl.col("s"), "there", 7))["s"][0]
        'hello there'
        >>> df.select(sf.overlay(pl.col("s"), "X", 1, 5))["s"][0]
        'X world'
    """
    return _str_to_col(col).overlay(replace, pos, length)


# ── math extras (batch 2) ─────────────────────────────────────────────────────

def bround(col: str | Expr, d: int = 0) -> Expr:
    """Banker's rounding (half-to-even).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [2.5, 3.5]})
        >>> df.select(sf.bround(pl.col("x")))["x"].to_list()
        [2.0, 4.0]
    """
    return _str_to_col(col).bround(d)


def hypot(x: str | Expr, y: str | Expr) -> Expr:
    """Euclidean distance: sqrt(x**2 + y**2).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [3.0], "y": [4.0]})
        >>> df.select(sf.hypot(pl.col("x"), pl.col("y")).alias("h"))["h"][0]
        5.0
    """
    x_e = _str_to_col(x).cast(polars_datatypes.Float64(), strict=False)
    y_e = _str_to_col(y).cast(polars_datatypes.Float64(), strict=False)
    return (x_e ** 2 + y_e ** 2).sqrt()


def pmod(col: str | Expr, divisor: Any) -> Expr:
    """Positive modulo -- result has the same sign as divisor.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [7]})
        >>> df.select(sf.pmod(pl.col("x"), 3).alias("r"))["r"][0]
        1
        >>> df2 = pl.DataFrame({"x": [-7]})
        >>> df2.select(sf.pmod(pl.col("x"), 3).alias("r"))["r"][0]
        2
    """
    return _str_to_col(col).pmod(divisor)


def shiftrightunsigned(col: str | Expr, n: int) -> Expr:
    """Unsigned (logical) right shift -- fills with zeros on the left.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [8]})
        >>> df.select(sf.shiftrightunsigned(pl.col("x"), 1).alias("r"))["r"][0]
        4
    """
    return _str_to_col(col).map_elements(
        lambda x: (x & 0xFFFF_FFFF_FFFF_FFFF) >> n if x is not None else None,
        polars_datatypes.Int64(),
    )


# ── date extras (batch 2) ─────────────────────────────────────────────────────

def trunc(col: str | Expr, fmt: str) -> Expr:
    """Truncate date/timestamp to fmt unit (col first -- mirrors PySpark trunc()).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
        >>> df.select(sf.trunc(pl.col("d"), "month"))["d"][0]
        datetime.date(2023, 6, 1)
    """
    return _str_to_col(col).date_trunc(fmt)


def next_day(col: str | Expr, day_of_week: str) -> Expr:
    """Return the first date strictly after col that is the named day of week.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 11)]})
        >>> df.select(sf.next_day(pl.col("d"), "Mon"))["d"][0]
        datetime.date(2023, 1, 16)
    """
    return _str_to_col(col).next_day(day_of_week)


def make_timestamp(
    year: Any,
    month: Any,
    day: Any,
    hour: Any,
    minute: Any,
    second: Any,
    timezone: str | None = None,  # noqa: ARG001  (timezone not supported in Polars)
) -> Expr:
    """Construct a Datetime from year/month/day/hour/minute/second components.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"x": [1]})
        >>> df.select(sf.make_timestamp(2023, 6, 15, 10, 30, 0).alias("ts"))["ts"][0]
        datetime.datetime(2023, 6, 15, 10, 30)
    """
    import datetime as _dt

    def _to_expr(v: Any, alias: str) -> Expr:
        return (_str_to_col(v) if isinstance(v, (str, Expr)) else polars_functions.lit(v)).alias(alias)

    s = polars_functions.struct([
        _to_expr(year, "_y"), _to_expr(month, "_mo"), _to_expr(day, "_d"),
        _to_expr(hour, "_h"), _to_expr(minute, "_min"), _to_expr(second, "_s"),
    ])

    def _make(row: Any) -> Any:
        try:
            return _dt.datetime(
                int(row["_y"]), int(row["_mo"]), int(row["_d"]),
                int(row["_h"]), int(row["_min"]), int(row["_s"]),
            )
        except (ValueError, TypeError):
            return None

    return s.map_elements(_make, polars_datatypes.Datetime())


def unix_date(col: str | Expr) -> Expr:
    """Number of days since Unix epoch (1970-01-01).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2)]})
        >>> df.select(sf.unix_date(pl.col("d")))["d"].to_list()
        [0, 1]
    """
    return _str_to_col(col).unix_date()


def from_unixtime(col: str | Expr, fmt: str | None = None) -> Expr:
    """Convert integer epoch-seconds to Datetime; optionally format as string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"ts": [0]})
        >>> df.select(sf.from_unixtime(pl.col("ts")))["ts"][0].date()
        datetime.date(1970, 1, 1)
        >>> df.select(sf.from_unixtime(pl.col("ts"), "%Y-%m-%d"))["ts"][0]
        '1970-01-01'
    """
    dt_expr = polars_functions.from_epoch(_str_to_col(col), time_unit="s")
    if fmt is not None:
        return dt_expr.dt.strftime(fmt)
    return dt_expr


# ── array extras (batch 2) ────────────────────────────────────────────────────

def arrays_zip(*cols: str | Expr) -> Expr:
    """Zip arrays into an array of structs (one struct per position across all arrays).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2]], "b": [["x", "y"]]})
        >>> df.select(sf.arrays_zip(pl.col("a"), pl.col("b")).alias("z"))["z"][0][0]["a"]
        1
    """
    exprs = [_str_to_col(c) for c in cols]
    names: list[str] = []
    for i, e in enumerate(exprs):
        try:
            names.append(e.meta.output_name())
        except Exception:
            names.append(f"_{i}")

    def _batch(series_list: list) -> pl.Series:
        rows: list = []
        for row_idx in range(len(series_list[0])):
            row_lists = [series_list[j][row_idx].to_list() for j in range(len(series_list))]
            min_len = row_lists[0].__len__()
            for rl in row_lists[1:]:
                if len(rl) < min_len:
                    min_len = len(rl)
            rows.append([
                {names[j]: row_lists[j][k] for j in range(len(names))}
                for k in range(min_len)
            ])
        return pl.Series(rows)

    return polars_functions.map_batches(exprs, _batch, return_dtype=None)


def shuffle(col: str | Expr, seed: int | None = None) -> Expr:
    """Return the array with elements in random order.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4, 5]]})
        >>> result = df.select(sf.shuffle(pl.col("a"), seed=0))["a"][0].to_list()
        >>> sorted(result)
        [1, 2, 3, 4, 5]
    """
    return _str_to_col(col).shuffle(seed)


def exists(col: str | Expr, f: Any) -> Expr:
    """Return true if any element satisfies predicate f.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(sf.exists(pl.col("a"), lambda e: e > 2))["a"][0]
        True
        >>> df.select(sf.exists(pl.col("a"), lambda e: e > 10))["a"][0]
        False
    """
    return _str_to_col(col).exists(f)


def aggregate(col: str | Expr, zero: Any, merge: Any, finish: Any = None) -> Expr:
    """Fold over list: reduce(merge, elements, zero), then optionally apply finish.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
        >>> df.select(sf.aggregate(pl.col("a"), 0, lambda acc, e: acc + e).alias("s"))["s"][0]
        10
    """
    return _str_to_col(col).aggregate(zero, merge, finish)


# ── aggregate extras (batch 2) ────────────────────────────────────────────────

def corr(col1: str | Expr, col2: str | Expr, method: str = "pearson") -> Expr:
    """Correlation between two columns (pearson or spearman).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        >>> builtins.round(df.select(sf.corr(pl.col("x"), pl.col("y")).alias("r"))["r"][0], 1)
        1.0
    """
    return pl.corr(_str_to_col(col1), _str_to_col(col2), method=method)


def covar_samp(col1: str | Expr, col2: str | Expr) -> Expr:
    """Sample covariance between two columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        >>> builtins.round(df.select(sf.covar_samp(pl.col("x"), pl.col("y")).alias("c"))["c"][0], 1)
        1.0
    """
    return pl.cov(_str_to_col(col1), _str_to_col(col2))


def covar_pop(col1: str | Expr, col2: str | Expr) -> Expr:
    """Population covariance between two columns (ddof=0).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        >>> df.select(sf.covar_pop(pl.col("x"), pl.col("y")).alias("c"))["c"][0] is not None
        True
    """
    c1, c2 = _str_to_col(col1), _str_to_col(col2)

    def _pop_cov(series_list: list) -> pl.Series:
        s1 = series_list[0].cast(pl.Float64)
        s2 = series_list[1].cast(pl.Float64)
        if len(s1) == 0:
            return pl.Series([None], dtype=pl.Float64)
        cov = ((s1 - s1.mean()) * (s2 - s2.mean())).mean()
        return pl.Series([cov], dtype=pl.Float64)

    return polars_functions.map_batches([c1, c2], _pop_cov, return_dtype=polars_datatypes.Float64())


def kurtosis(col: str | Expr) -> Expr:
    """Compute the kurtosis of a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.select(sf.kurtosis(pl.col("x")))["kurtosis"][0] is not None
        True
    """
    return _str_to_col(col).kurtosis(fisher=True, bias=True).alias("kurtosis")


def skewness(col: str | Expr) -> Expr:
    """Compute the skewness of a column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 1.0, 1.0, 2.0, 5.0]})
        >>> df.select(sf.skewness(pl.col("x")))["skewness"][0] > 0
        True
    """
    return _str_to_col(col).skew(bias=True).alias("skewness")


def mode(col: str | Expr) -> Expr:
    """Most frequent value; smallest if there is a tie.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3, 3, 3]})
        >>> df.select(sf.mode(pl.col("x")).alias("m"))["m"][0]
        3
    """
    c = _str_to_col(col)
    return polars_functions.map_batches(
        [c],
        lambda s: pl.Series([s[0].mode().sort()[0]]),
        return_dtype=None,
    )


def percentile(col: str | Expr, pct: float, accuracy: int = 10_000) -> Expr:  # noqa: ARG001
    """Exact percentile (pct on 0.0-1.0 scale).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.select(sf.percentile(pl.col("x"), 0.5))["percentile"][0]
        3.0
    """
    return _str_to_col(col).quantile(pct, interpolation="nearest").alias("percentile")


def sum_distinct(col: str | Expr) -> Expr:
    """Sum of distinct values.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3]})
        >>> df.select(sf.sum_distinct(pl.col("x")))["sum_distinct(x)"][0]
        6
    """
    return _str_to_col(col).sum_distinct().alias("sum_distinct")


any_value = first  # arbitrary non-null value


def approx_count_distinct(col: str | Expr, rsd: float = 0.05) -> Expr:  # noqa: ARG001
    """Approximate distinct count (exact in Polars).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3]})
        >>> df.select(sf.approx_count_distinct(pl.col("x")).alias("n"))["n"][0]
        3
    """
    return _str_to_col(col).n_unique()


# ── null / type-check standalones ─────────────────────────────────────────────

def isnull(col: str | Expr) -> Expr:
    """Return boolean expression: true where col is null.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [None, 1, None]})
        >>> df.select(sf.isnull(pl.col("x")))["x"].to_list()
        [True, False, True]
    """
    return _str_to_col(col).is_null()


def isnan(col: str | Expr) -> Expr:
    """Return boolean expression: true where col is NaN.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [float("nan"), 1.0, float("nan")]})
        >>> df.select(sf.isnan(pl.col("x")))["x"].to_list()
        [True, False, True]
    """
    return _str_to_col(col).is_nan()


def isnotnull(col: str | Expr) -> Expr:
    """Return boolean expression: true where col is not null.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [None, 2, None]})
        >>> df.select(sf.isnotnull(pl.col("x")))["x"].to_list()
        [False, True, False]
    """
    return _str_to_col(col).is_not_null()


# ── hash extras ───────────────────────────────────────────────────────────────

def sha2(col: str | Expr, num_bits: int = 256) -> Expr:
    """SHA-2 family hash (num_bits: 224, 256, 384, 512; 0 -> 256).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import hashlib
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.sha2(pl.col("s"), 256))["s"][0] == hashlib.sha256(b"hello").hexdigest()
        True
        >>> df.select(sf.sha2(pl.col("s"), 512))["s"][0] == hashlib.sha512(b"hello").hexdigest()
        True
    """
    import hashlib as _hashlib

    _SHA2_ALGOS = {0: "sha256", 224: "sha224", 256: "sha256", 384: "sha384", 512: "sha512"}
    algo_name = _SHA2_ALGOS.get(num_bits, "sha256")
    h = getattr(_hashlib, algo_name)
    return _str_to_col(col).map_elements(
        lambda x: h(str(x).encode()).hexdigest() if x is not None else None,
        polars_datatypes.String(),
    )


def crc32(col: str | Expr) -> Expr:
    """CRC32 checksum as unsigned 32-bit integer.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import binascii
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(sf.crc32(pl.col("s")))["s"][0] == (binascii.crc32(b"hello") & 0xFFFF_FFFF)
        True
    """
    import binascii as _binascii

    return _str_to_col(col).map_elements(
        lambda x: _binascii.crc32(str(x).encode()) & 0xFFFF_FFFF if x is not None else None,
        polars_datatypes.UInt32(),
    )


def hash(*cols: str | Expr) -> Expr:
    """MurmurHash3-style hash of one or more columns (uses Polars native hash).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1]})
        >>> df.select(sf.hash(pl.col("x")).alias("h"))["h"][0] is not None
        True
    """
    if len(cols) == 1:
        return _str_to_col(cols[0]).hash(seed=0)
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.struct(exprs).hash(seed=0)


def xxhash64(*cols: str | Expr) -> Expr:
    """xxHash64-style hash of one or more columns (uses Polars native hash with seed 42).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1]})
        >>> df.select(sf.xxhash64(pl.col("x")).alias("h"))["h"][0] is not None
        True
    """
    if len(cols) == 1:
        return _str_to_col(cols[0]).hash(seed=42)
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.struct(exprs).hash(seed=42)


# ── URL helpers ───────────────────────────────────────────────────────────────

def url_encode(col: str | Expr) -> Expr:
    """Percent-encode a string for use in a URL.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(sf.url_encode(pl.col("s")))["s"][0]
        'hello%20world'
    """
    from urllib.parse import quote as _quote

    return _str_to_col(col).map_elements(
        lambda x: _quote(str(x), safe="") if x is not None else None,
        polars_datatypes.String(),
    )


def url_decode(col: str | Expr) -> Expr:
    """Decode a percent-encoded URL string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello%20world"]})
        >>> df.select(sf.url_decode(pl.col("s")))["s"][0]
        'hello world'
    """
    from urllib.parse import unquote as _unquote

    return _str_to_col(col).map_elements(
        lambda x: _unquote(str(x)) if x is not None else None,
        polars_datatypes.String(),
    )


# ── JSON helper ───────────────────────────────────────────────────────────────

def get_json_object(col: str | Expr, path: str) -> Expr:
    """Extract a value from a JSON string using a JSONPath expression.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ['{"name": "alice", "age": 30}']})
        >>> df.select(sf.get_json_object(pl.col("s"), "$.name"))["s"][0]
        'alice'
    """
    return _str_to_col(col).str.json_path_match(path)


# ── string extras (batch 3) ───────────────────────────────────────────────────

def unbase64(col: str | Expr) -> Expr:
    """Decode a Base64-encoded string column to UTF-8.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import base64
        >>> encoded = base64.b64encode(b"hello").decode()
        >>> df = pl.DataFrame({"s": [encoded]})
        >>> df.select(sf.unbase64(pl.col("s")))["s"][0]
        'hello'
    """
    return _str_to_col(col).unbase64()


def levenshtein(col1: str | Expr, col2: str | Expr) -> Expr:
    """Compute Levenshtein edit distance between two string columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": ["kitten"], "b": ["sitting"]})
        >>> df.select(sf.levenshtein(pl.col("a"), pl.col("b")).alias("r"))["r"][0]
        3
    """
    return _str_to_col(col1).levenshtein(_str_to_col(col2))


def regexp_substr(col: str | Expr, pattern: str) -> Expr:
    """Return first substring matching the regexp.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello world 123"]})
        >>> df.select(sf.regexp_substr(pl.col("s"), r"\\d+"))["s"][0]
        '123'
    """
    return _str_to_col(col).regexp_substr(pattern)


def elt(*cols: str | Expr) -> Expr:
    """Return the n-th input (1-based index): elt(index, str1, str2, ...).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"idx": [1, 2, 3], "a": ["x", "x", "x"], "b": ["y", "y", "y"]})
        >>> df.select(sf.elt(pl.col("idx"), pl.col("a"), pl.col("b")).alias("r"))["r"].to_list()
        ['x', 'y', None]
    """
    if len(cols) < 2:
        raise ValueError("elt() requires at least 2 arguments: index + strings")
    index_expr = _str_to_col(cols[0])
    str_exprs = [_str_to_col(c) for c in cols[1:]]

    def _batch(series_list: list) -> pl.Series:
        idx_series = series_list[0]
        str_series = series_list[1:]
        results = []
        for row_i in range(len(idx_series)):
            idx = idx_series[row_i]
            if idx is None or idx < 1 or idx > len(str_series):
                results.append(None)
            else:
                results.append(str_series[idx - 1][row_i])
        return pl.Series(results, dtype=polars_datatypes.String())

    return polars_functions.map_batches([index_expr] + str_exprs, _batch, return_dtype=polars_datatypes.String())


# ── math extras (batch 3) ─────────────────────────────────────────────────────

import math as _math  # noqa: E402


def log1p(col: str | Expr) -> Expr:
    """Natural log of (1 + x).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.log1p(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).log1p()


def expm1(col: str | Expr) -> Expr:
    """e^x - 1.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(sf.expm1(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).expm1()


def rint(col: str | Expr) -> Expr:
    """Round to nearest integer (returns float).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.4, 1.6, -1.5]})
        >>> df.select(sf.rint(pl.col("x")))["x"].to_list()
        [1.0, 2.0, -2.0]
    """
    return _str_to_col(col).rint()


def remainder(col1: str | Expr, col2: str | Expr) -> Expr:
    """IEEE 754 remainder: x - round(x/y)*y.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [5.0], "y": [3.0]})
        >>> df.select(sf.remainder(pl.col("x"), pl.col("y")).alias("r"))["r"][0]
        -1.0
    """
    c1 = _str_to_col(col1)
    c2 = _str_to_col(col2)
    s = polars_functions.struct([c1.alias("_x"), c2.alias("_y")])
    return s.map_elements(
        lambda r: _math.remainder(r["_x"], r["_y"]) if r["_x"] is not None and r["_y"] is not None else None,
        polars_datatypes.Float64(),
    )


def gcd(col1: str | Expr, col2: str | Expr) -> Expr:
    """Greatest common divisor of two integer columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [12], "b": [8]})
        >>> df.select(sf.gcd(pl.col("a"), pl.col("b")).alias("r"))["r"][0]
        4
    """
    import math as _m

    c1 = _str_to_col(col1)
    c2 = _str_to_col(col2)
    s = polars_functions.struct([c1.alias("_a"), c2.alias("_b")])
    return s.map_elements(
        lambda r: _m.gcd(int(r["_a"]), int(r["_b"])) if r["_a"] is not None and r["_b"] is not None else None,
        polars_datatypes.Int64(),
    )


def lcm(col1: str | Expr, col2: str | Expr) -> Expr:
    """Least common multiple of two integer columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [4], "b": [6]})
        >>> df.select(sf.lcm(pl.col("a"), pl.col("b")).alias("r"))["r"][0]
        12
    """
    import math as _m

    c1 = _str_to_col(col1)
    c2 = _str_to_col(col2)
    s = polars_functions.struct([c1.alias("_a"), c2.alias("_b")])
    return s.map_elements(
        lambda r: _m.lcm(int(r["_a"]), int(r["_b"])) if r["_a"] is not None and r["_b"] is not None else None,
        polars_datatypes.Int64(),
    )


def bitcount(col: str | Expr) -> Expr:
    """Count the number of set bits in the integer value.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [7]})
        >>> df.select(sf.bitcount(pl.col("x")))["x"][0]
        3
    """
    return _str_to_col(col).bitcount()


def toDegrees(col: str | Expr) -> Expr:
    """Convert radians to degrees.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import math
        >>> import builtins
        >>> df = pl.DataFrame({"x": [math.pi]})
        >>> builtins.round(df.select(sf.toDegrees(pl.col("x")))["x"][0])
        180
    """
    return _str_to_col(col) * (180.0 / _math.pi)


def toRadians(col: str | Expr) -> Expr:
    """Convert degrees to radians.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import builtins
        >>> df = pl.DataFrame({"x": [180.0]})
        >>> builtins.round(df.select(sf.toRadians(pl.col("x")))["x"][0], 4)
        3.1416
    """
    return _str_to_col(col) * (_math.pi / 180.0)


# ── date extras (batch 3) ─────────────────────────────────────────────────────

def months_between(end: str | Expr, start: str | Expr, roundOff: bool = True) -> Expr:  # noqa: N803
    """Number of months between two date/timestamp columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"a": [datetime.date(2023, 6, 1)], "b": [datetime.date(2023, 1, 1)]})
        >>> df.select(sf.months_between(pl.col("a"), pl.col("b")).alias("m"))["m"][0]
        5.0
    """
    e = _str_to_col(end)
    s = _str_to_col(start)
    struct_expr = polars_functions.struct([e.alias("_e"), s.alias("_s")])

    def _mb(row: Any) -> Any:
        import datetime as _dt

        ev, sv = row["_e"], row["_s"]
        if ev is None or sv is None:
            return None
        # Convert to date-only for month arithmetic
        if hasattr(ev, "date"):
            ed, sd = ev.date(), sv.date()
        else:
            ed, sd = ev, sv
        months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
        day_diff = (ed.day - sd.day) / 31.0
        result = months + day_diff
        if roundOff:
            import builtins as _builtins
            return _builtins.round(result, 8)
        return result

    return struct_expr.map_elements(_mb, polars_datatypes.Float64())


def to_utc_timestamp(col: str | Expr, tz: str) -> Expr:
    """Interpret timestamp as being in *tz* and convert to UTC.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 6, 15, 12, 0, 0)]})
        >>> df.select(sf.to_utc_timestamp(pl.col("t"), "US/Eastern").alias("u")).height
        1
    """
    return _str_to_col(col).to_utc_timestamp(tz)


def from_utc_timestamp(col: str | Expr, tz: str) -> Expr:
    """Interpret UTC timestamp and convert to *tz*.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 6, 15, 12, 0, 0)]})
        >>> df.select(sf.from_utc_timestamp(pl.col("t"), "US/Eastern").alias("u")).height
        1
    """
    return _str_to_col(col).from_utc_timestamp(tz)


# ── array extras (batch 3) ────────────────────────────────────────────────────

def array_reverse(col: str | Expr) -> Expr:
    """Reverse the elements in an array column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(sf.array_reverse(pl.col("a")))["a"][0].to_list()
        [3, 2, 1]
    """
    return _str_to_col(col).array_reverse()


def array_insert(col: str | Expr, pos: int, value: Any) -> Expr:
    """Insert *value* at position *pos* (1-based, supports negative) in each array.

    For negative *pos*, insertion is measured from the end (PySpark semantics).
    Positive *pos* is 1-based; implemented with pure Polars expressions.
    Negative *pos* falls back to a UDF (eager DataFrames only).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> result = df.select(sf.array_insert(pl.col("a"), 2, 99).alias("r"))["r"][0]
        >>> result.to_list() if hasattr(result, 'to_list') else result
        [1, 99, 2, 3]
    """
    c = _str_to_col(col)
    if pos > 0:
        idx = pos - 1  # 0-based
        val_lit = polars_functions.lit(pl.Series([value])).implode()
        part1 = c.list.slice(0, idx)
        part2 = c.list.slice(idx)
        return polars_functions.concat_list([part1, val_lit, part2])
    # negative pos: need runtime list length — wrap with .list.len()
    # insert at position (list.len + pos + 1) from start (0-based: len + pos)
    len_expr = c.list.len()
    idx_expr = len_expr + pos  # 0-based insert index (pos is negative)
    # Clamp to [0, len]
    idx_expr = pl.when(idx_expr < 0).then(0).otherwise(idx_expr)
    val_lit = polars_functions.lit(pl.Series([value])).implode()
    # Can't do variable-length slice with expression idx in Polars, fall back to UDF
    s_expr = polars_functions.struct([c.alias("_lst"), idx_expr.cast(polars_datatypes.Int64()).alias("_idx")])

    def _insert_neg(row: Any) -> Any:
        lst = row["_lst"]
        if lst is None:
            return None
        items = lst.to_list() if hasattr(lst, "to_list") else list(lst)
        items.insert(int(row["_idx"]), value)
        return items

    return s_expr.map_elements(_insert_neg, return_dtype=None)


# ── aggregate/window extras (batch 3) ─────────────────────────────────────────

def mean(col: str | Expr) -> Expr:
    """Alias for avg().

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> df.select(sf.mean(pl.col("x")))["avg(x)"][0]
        2.0
    """
    return _str_to_col(col).mean().alias(f"avg({_col_name(col)})")


def ntile(n: int) -> _WindowFuncExpr:
    """Divide the ordered partition into *n* buckets (1-based).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> from src.sparkpolars.polyspark.sql.window import Window, WindowSpec
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4]})
        >>> df.select(sf.ntile(2).over(WindowSpec()).alias("t"))["t"].to_list()
        [1, 1, 2, 2]
    """
    return _WindowFuncExpr("ntile", n)


def cume_dist() -> _WindowFuncExpr:
    """Cumulative distribution: position / total rows within the partition.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> from src.sparkpolars.polyspark.sql.window import Window, WindowSpec
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4]})
        >>> df.select(sf.cume_dist().over(WindowSpec()).alias("c"))["c"].to_list()
        [0.25, 0.5, 0.75, 1.0]
    """
    return _WindowFuncExpr("cume_dist")


def percent_rank() -> _WindowFuncExpr:
    """Percent rank: (rank - 1) / (total - 1), 0.0 when only one row."""
    return _WindowFuncExpr("percent_rank")


# ── struct / map extras (batch 3) ─────────────────────────────────────────────

def to_json(col: str | Expr) -> Expr:
    """Serialize a struct column to a JSON string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import json
        >>> df = pl.DataFrame({"s": [{"a": 1, "b": 2}]}, schema={"s": pl.Struct({"a": pl.Int32, "b": pl.Int32})})
        >>> json.loads(df.select(sf.to_json(pl.col("s")))["s"][0])
        {'a': 1, 'b': 2}
    """
    return _str_to_col(col).to_json()


def map_keys(col: str | Expr) -> Expr:
    """Return keys of a map column (list-of-structs with 'key' field).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]})
        >>> sorted(df.select(sf.map_keys(pl.col("m")))["m"][0].to_list())
        ['a', 'b']
    """
    return _str_to_col(col).map_keys()


def map_values(col: str | Expr) -> Expr:
    """Return values of a map column (list-of-structs with 'value' field).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]})
        >>> sorted(df.select(sf.map_values(pl.col("m")))["m"][0].to_list())
        ['1', '2']
    """
    return _str_to_col(col).map_values()


# ── fix unix_timestamp to support format string ───────────────────────────────

def unix_timestamp(col: str | Expr | None = None, fmt: str | None = None) -> Expr:
    """Seconds since epoch.  With no args returns current time.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 6, 15, 0, 0, 0)]})
        >>> df.select(sf.unix_timestamp(pl.col("t")))["t"][0]
        1686787200
        >>> df2 = pl.DataFrame({"ts": ["2020-01-01 00:00:00"]})
        >>> df2.select(sf.unix_timestamp(pl.col("ts"), fmt="%Y-%m-%d %H:%M:%S").alias("t"))["t"][0]
        1577836800
    """
    if col is None:
        import time as _time

        return pl.lit(int(_time.time())).cast(polars_datatypes.Int64())
    e = _str_to_col(col)
    if fmt is not None:
        return e.str.strptime(polars_datatypes.Datetime("us"), fmt).dt.epoch(time_unit="s")
    return e.dt.epoch(time_unit="s")


# ── monotonic_id alias ────────────────────────────────────────────────────────

def monotonic_id() -> Expr:
    """Monotonically increasing 64-bit integer (row index).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [10, 20, 30]})
        >>> df.select(sf.monotonic_id().alias("id"))["id"].to_list()
        [0, 1, 2]
    """
    return polars_functions.int_range(
        start=0,
        end=polars_functions.len(),
        dtype=polars_datatypes.UInt64(),
    )


# ── NotImplementedError stubs ─────────────────────────────────────────────────

def soundex(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("soundex is not supported in Polars")


def metaphone(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("metaphone is not supported in Polars")


def json_tuple(col: str | Expr, *fields: str) -> Expr:
    """Extract multiple JSON fields into a struct column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"j": ['{"name":"Alice","age":30}']})
        >>> s = df.select(sf.json_tuple(pl.col("j"), "name", "age").alias("s"))["s"][0]
        >>> s["name"]
        'Alice'
        >>> s["age"]
        '30'
    """
    c = _str_to_col(col)
    field_exprs = [c.str.json_path_match(f"$.{f}").alias(f) for f in fields]
    return polars_functions.struct(field_exprs)


def from_json(col: str | Expr, schema: Any) -> Expr:  # noqa: ARG001
    raise NotImplementedError("from_json is not supported in Polars")


def map_from_entries(col: str | Expr) -> Expr:
    """Identity -- Polars already stores maps as list(struct({key, value})).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> entries = [{"key": "x", "value": 99}]
        >>> df = pl.DataFrame({"m": [entries]})
        >>> df.select(sf.map_from_entries(pl.col("m")))["m"][0][0]["key"]
        'x'
    """
    return _str_to_col(col)


def map_concat(*cols: str | Expr) -> Expr:
    """Concatenate map columns (list-of-structs) into one.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({
        ...     "m1": [[{"key": "a", "value": 1}]],
        ...     "m2": [[{"key": "b", "value": 2}]],
        ... })
        >>> result = df.select(sf.map_concat(pl.col("m1"), pl.col("m2")).alias("m"))["m"][0]
        >>> sorted([e["key"] for e in result])
        ['a', 'b']
    """
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.concat_list(*exprs)


def map_filter(col: str | Expr, f: Callable) -> Expr:
    """Keep only entries where f(key, value) is True."""
    c = _str_to_col(col)
    return c.list.eval(
        pl.element().filter(f(pl.element().struct.field("key"), pl.element().struct.field("value")))
    )


def transform_keys(col: str | Expr, f: Callable) -> Expr:
    """Apply f(key, value) to every key in a map column."""
    c = _str_to_col(col)
    return c.list.eval(
        polars_functions.struct(
            f(pl.element().struct.field("key"), pl.element().struct.field("value")).alias("key"),
            pl.element().struct.field("value").alias("value"),
        )
    )


def transform_values(col: str | Expr, f: Callable) -> Expr:
    """Apply f(key, value) to every value in a map column."""
    c = _str_to_col(col)
    return c.list.eval(
        polars_functions.struct(
            pl.element().struct.field("key").alias("key"),
            f(pl.element().struct.field("key"), pl.element().struct.field("value")).alias("value"),
        )
    )


def grouping(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("grouping is not supported outside GROUPING SETS context")


def grouping_id(*cols: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("grouping_id is not supported outside GROUPING SETS context")


def input_file_name() -> Expr:
    raise NotImplementedError("input_file_name is not applicable to Polars DataFrames")


def assert_true(col: str | Expr, error_msg: str = "") -> Expr:
    """Raise a RuntimeError if any value in *col* is False/null; else return null column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [True, True]})
        >>> df.select(sf.assert_true(pl.col("x")).alias("r")).height
        2
    """
    c = _str_to_col(col)

    def _check(s: pl.Series) -> pl.Series:
        if not s.cast(pl.Boolean).fill_null(False).all():
            msg = error_msg or f"assert_true failed: {s.name}"
            raise RuntimeError(msg)
        return pl.Series([None] * len(s), dtype=pl.Null)

    return c.map_batches(_check, return_dtype=pl.Null)


def schema_of_csv(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("schema_of_csv is not supported in Polars")


def schema_of_json(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("schema_of_json is not supported in Polars")


def window(time_col: str | Expr, window_duration: str, slide_duration: str | None = None, start_time: str | None = None) -> Expr:  # noqa: ARG001
    raise NotImplementedError("window (time-based tumbling/sliding) is not supported in Polars")


def posexplode(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("posexplode is not supported; use explode() with a row index instead")


def posexplode_outer(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("posexplode_outer is not supported; use explode() with a row index instead")


# ── batch-4: aliases & new functions ─────────────────────────────────────────

def ln(col: str | Expr) -> Expr:
    """Natural logarithm (alias for log with base e).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0]})
        >>> df.select(sf.ln(pl.col("x")))["x"][0]
        0.0
    """
    return _str_to_col(col).log(2.718281828459045)


sha = sha1  # SHA-1 alias


def cardinality(col: str | Expr) -> Expr:
    """Number of elements in an array/map column (alias for size).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(sf.cardinality(pl.col("a")))["a"][0]
        3
    """
    return _str_to_col(col).list.len()


def approx_count_distinct(col: str | Expr) -> Expr:
    """Approximate distinct count (HyperLogLog approximation via n_unique).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3]})
        >>> df.select(sf.approx_count_distinct(pl.col("x")))["approx_count_distinct(x)"][0]
        3
    """
    c = _str_to_col(col)
    name = _col_name(col)
    return c.approx_n_unique().alias(f"approx_count_distinct({name})")


approxCountDistinct = approx_count_distinct


def sum_distinct(col: str | Expr) -> Expr:
    """Sum of distinct values.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3]})
        >>> df.select(sf.sum_distinct(pl.col("x")))["sum_distinct(x)"][0]
        6
    """
    c = _str_to_col(col)
    name = _col_name(col)
    return c.drop_nulls().unique().sum().alias(f"sum_distinct({name})")


sumDistinct = sum_distinct


def regexp(col: str | Expr, pattern: str) -> Expr:
    """Alias for regexp_like.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["hello", "world"]})
        >>> df.select(sf.regexp(pl.col("s"), "hel"))["s"][0]
        True
    """
    return regexp_like(col, pattern)


def format_number(col: str | Expr, d: int) -> Expr:
    """Format a number to *d* decimal places as a string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1234.5678]})
        >>> df.select(sf.format_number(pl.col("x"), 2))["x"][0]
        '1234.57'
        >>> df2 = pl.DataFrame({"x": [3.9]})
        >>> df2.select(sf.format_number(pl.col("x"), 0))["x"][0]
        '4'
    """
    import builtins as _b
    fmt = f"{{:.{_b.max(0, d)}f}}"
    return _str_to_col(col).map_elements(lambda v: fmt.format(v), return_dtype=pl.String)


def regexp_instr(col: str | Expr, pattern: str) -> Expr:
    """Return 1-based start position of first regex match, 0 if no match.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"s": ["abcabc", "xyz"]})
        >>> df.select(sf.regexp_instr(pl.col("s"), "b"))["s"].to_list()
        [2, 0]
    """
    import re as _re
    def _find(s: str) -> int:
        if s is None:
            return None
        m = _re.search(pattern, s)
        return (m.start() + 1) if m else 0
    return _str_to_col(col).map_elements(_find, return_dtype=pl.Int64)


def typeof(col: str | Expr) -> Expr:
    """Return the Polars dtype of the column as a string literal.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1]})
        >>> df.select(sf.typeof(pl.col("x")).alias("t"))["t"][0]
        'Int64'
    """
    c = _str_to_col(col)

    def _dtype_name(s: pl.Series) -> pl.Series:
        return pl.Series([str(s.dtype)] * len(s))

    return c.map_batches(_dtype_name, return_dtype=pl.String)


def uniform(min_val: float, max_val: float, seed: int | None = None) -> Expr:
    """Uniformly distributed random float in [min_val, max_val] per row.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> len(df.select(sf.uniform(0.0, 1.0, seed=42).alias("r"))["r"])
        3
    """
    import random as _random
    rng = _random.Random(seed)

    def _uniform(s: pl.Series) -> pl.Series:
        return pl.Series([rng.uniform(min_val, max_val) for _ in range(len(s))], dtype=pl.Float64)

    return polars_functions.int_range(start=0, end=polars_functions.len(), dtype=pl.Int64).map_batches(
        _uniform, return_dtype=pl.Float64
    )


def percentile_approx(col: str | Expr, percentage: float, accuracy: int = 10000) -> Expr:  # noqa: ARG002
    """Approximate percentile (exact via Polars quantile).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.select(sf.percentile_approx(pl.col("x"), 0.5))["percentile_approx(x, 0.5)"][0] is not None
        True
    """
    c = _str_to_col(col)
    name = _col_name(col)
    return c.quantile(percentage, interpolation="nearest").alias(f"percentile_approx({name}, {percentage})")


def zip_with(col1: str | Expr, col2: str | Expr, f: Callable) -> Expr:
    """Element-wise merge of two list columns using function f(x, y)."""
    c1 = _str_to_col(col1)
    c2 = _str_to_col(col2)
    return polars_functions.concat_list(c1, c2).list.eval(
        f(pl.element().list.first(), pl.element().list.last())
    )


# ── map/struct extras (batch-4) ───────────────────────────────────────────────

def map_contains_key(col: str | Expr, key: Any) -> Expr:
    """Return True if the map column contains *key*.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"m": [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]]})
        >>> df.select(sf.map_contains_key(pl.col("m"), "a"))["m"][0]
        True
        >>> df2 = pl.DataFrame({"m": [[{"key": "a", "value": 1}]]})
        >>> df2.select(sf.map_contains_key(pl.col("m"), "z"))["m"][0]
        False
    """
    return _str_to_col(col).list.eval(
        pl.element().struct.field("key").eq(pl.lit(key))
    ).list.any()


def map_entries(col: str | Expr) -> Expr:
    """Return list of structs {key, value} -- identity for Polars map columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> entries = [{"key": "a", "value": 1}]
        >>> df = pl.DataFrame({"m": [entries]})
        >>> df.select(sf.map_entries(pl.col("m")))["m"][0][0]["key"]
        'a'
    """
    return _str_to_col(col)


def map_from_arrays(keys: str | Expr, values: str | Expr) -> Expr:
    """Build a map column (list-of-structs) from parallel key and value list columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"k": [["a", "b"]], "v": [[1, 2]]})
        >>> result = df.select(sf.map_from_arrays(pl.col("k"), pl.col("v")).alias("m"))["m"][0]
        >>> [e["key"] for e in result]
        ['a', 'b']
        >>> [e["value"] for e in result]
        [1, 2]
    """
    k = _str_to_col(keys)
    v = _str_to_col(values)

    def _zip_row(row: dict) -> list:
        return [{"key": ki, "value": vi} for ki, vi in zip(row["keys"] or [], row["values"] or [])]

    return polars_functions.struct(k.alias("keys"), v.alias("values")).map_elements(
        _zip_row, return_dtype=pl.List(pl.Struct({"key": pl.String, "value": pl.Int64}))
    )


def named_struct(*args: Any) -> Expr:
    """Build a struct from alternating name/value pairs: named_struct("a", col1, "b", col2).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.functions as sf
        >>> df = pl.DataFrame({"x": [1], "y": [2]})
        >>> s = df.select(sf.named_struct("a", pl.col("x"), "b", pl.col("y")).alias("s"))["s"][0]
        >>> s["a"]
        1
        >>> s["b"]
        2
    """
    if len(args) % 2 != 0:
        raise ValueError("named_struct requires an even number of arguments (name, value pairs)")
    fields = []
    for i in range(0, len(args), 2):
        name = args[i]
        value = args[i + 1]
        if not isinstance(name, str):
            raise TypeError(f"named_struct field names must be strings, got {type(name)}")
        e = _str_to_col(value) if not isinstance(value, Expr) else value
        fields.append(e.alias(name))
    return polars_functions.struct(fields)
