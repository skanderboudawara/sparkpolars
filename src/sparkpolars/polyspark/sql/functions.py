from typing import Any
from functools import reduce
import re

import polars.functions as plf
import polars.datatypes as plf_types
import polars as pl
from polars import DataFrame, lit
from polars import col as polars_col
from polars.expr import Expr



def col(name: str) -> "ColExtension":
    return ColExtension(polars_col(name))

column = col

class ColExtension:
    def __init__(self, expr: Expr) -> None:
        self = expr

    def isNull(self) -> "ColExtension":
        return ColExtension(self.is_null())

    def isNotNull(self) -> "ColExtension":
        return ColExtension(self.is_not_null())

    def alias(self, name: str) -> Expr:
        return self.alias(name)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self, name)
        if callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                if isinstance(result, Expr):
                    return ColExtension(result)
                return result

            return wrapper
        return attr


def _str_to_col(name: str):
    if isinstance(name, str):
        return col(name)
    return name

def rlike(str, regexp: str) -> Expr:
    return _str_to_col(str).str.contains(regexp, literal=False)


regexp_like = rlike
regexp = rlike
contains = rlike


def startswith(str: Expr, prefix: str) -> Expr:
    return _str_to_col(str).str.starts_with(prefix)


def endswith(str: Expr, suffix: str) -> Expr:
    return _str_to_col(str).str.ends_with(suffix)


def substring(str: Expr, pos: int, len: int | None = None) -> Expr:
    if len is not None:
        return _str_to_col(str).str.slice(pos - 1, len)
    return _str_to_col(str).str.slice(pos - 1)


substr = substring


def trim(col: Expr, trim) -> Expr:
    return _str_to_col(col).str.strip_chars(trim)


def when(condition, value):
    return plf.when(condition).then(value)

def concat_ws(separator: str, *cols) -> Expr:
    exprs = []
    for c in cols:
        if hasattr(c, "_expr"):
            exprs.append(c)
        else:
            exprs.append(c)
    if len(exprs) == 1:
        return exprs[0].list.join(separator)
    return plf.concat_str(*exprs, separator=separator)


def expr(str):
    return plf.sql_expr(str)

def broadcast(df):
    return df

def upper(str):
    return _str_to_col(str).str.to_uppercase()

ucase = upper

def lower(str):
    return _str_to_col(str).str.to_lowercase()

lcase = lower

def regexp_count(str, regexp):
    return _str_to_col(str).str.count_matches(regexp)

# def initcap(str):
#     concat()
    

def regexp_extract(str, pattern, idx):
    return _str_to_col(str).str.extract(pattern, idx)

def regexp_extract_all(str, regexp, idx=None):
    if idx is None:
        return _str_to_col(str).str.extract_all(regexp)
    raise NotImplementedError("idx parameter is not supported in Polars for regexp_extract_all")

def regexp_replace(str, pattern, replacement):
    return _str_to_col(str).str.replace(pattern, replacement)

def replace(src, search, replace=None):
    if replace is None:
        return _str_to_col(src).str.replace_all(search, "")
    return _str_to_col(src).str.replace(search, replace)

def rtrim(src, trim=None):
    if trim is None:
        return _str_to_col(src).str.strip_chars_end()
    return _str_to_col(src).str.strip_chars_end(trim)

def ltrim(src, trim=None):
    if trim is None:
        return _str_to_col(src).str.strip_chars_start()
    return _str_to_col(src).str.strip_chars_start(trim)

def rpad(src, length, pad=None):
    if pad is None:
        return _str_to_col(src).str.pad_end(length)
    return _str_to_col(src).str.pad(length, pad)

def lpad(src, length, pad=None):
    if pad is None:
        return _str_to_col(src).str.pad_start(length)
    return _str_to_col(src).str.pad(length, pad)

def base64(col):
    return _str_to_col(col).str.encode("base64")

def btrim(str, trim=None):
    if trim is None:
        return _str_to_col(str).str.strip_chars()
    return _str_to_col(str).str.strip_chars(trim)

def contains(left, right):
    return _str_to_col(left).str.contains(right)

def encode(col, charset):
    return _str_to_col(col).str.encode(charset)

def decode(col, charset):
    return _str_to_col(col).str.decode(charset)

def left(str, len):
    return _str_to_col(str).str.slice(0, len)

def right(str, len):
    return _str_to_col(str).str.slice(-len, len) if len > 0 else _str_to_col(str).str.slice(-len)

def length(col):
    return _str_to_col(col).str.len_chars()

def locate(substr, str, pos=None):
    return _str_to_col(str).str.find(substr)

position = locate

def repeat(col, n):
    return reduce(lambda acc, _: acc + col.cast(plf_types.String(), strict=False), range(n - 1), col.cast(plf_types.String(), strict=False))

def concat(*cols):
    return reduce(lambda acc, col: acc + col.cast(plf_types.String(), strict=False), cols)

def round(col, scale=0):
    return col.round(scale)

def sqrt(col):
    return col.sqrt()

def pow(col, exponent):
    return col.pow(exponent)

def translate(srcCol, matching, replace):
    if len(matching) != len(replace):
        raise ValueError("Matching and replace strings must have the same length.")
    trans_table = str.maketrans(matching, replace)
    return srcCol.map_elements(lambda x: x.translate(trans_table), plf_types.String())


def fake_dataframe() -> DataFrame:
    return pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })


df = fake_dataframe()
df = df.with_columns(
    translate(pl.col("name"), "lo", "x ").alias("trimmed_name"),
)

print(df)