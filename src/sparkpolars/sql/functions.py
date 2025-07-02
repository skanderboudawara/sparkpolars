from typing import Any

import polars.functions as plf
from polars import DataFrame, lit
from polars import col as polars_col
from polars.expr import Expr


def col(name: str) -> "ColExtension":
    return ColExtension(polars_col(name))

column = col

class ColExtension:
    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def isNull(self) -> "ColExtension":
        return ColExtension(self._expr.is_null())

    def isNotNull(self) -> "ColExtension":
        return ColExtension(self._expr.is_not_null())

    def alias(self, name: str) -> Expr:
        return self._expr.alias(name)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._expr, name)
        if callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                if isinstance(result, Expr):
                    return ColExtension(result)
                return result

            return wrapper
        return attr


def rlike(str, regexp: str) -> Expr:
    return str._expr.str.contains(regexp, literal=False).alias(f"{str._expr.name}_rlike_{regexp}")


regexp_like = rlike
regexp = rlike
contains = rlike


def startswith(str: Expr, prefix: str) -> Expr:
    return str._expr.str.starts_with(prefix).alias(f"{str._expr.name}_startswith_{prefix}")


def endswith(str: Expr, suffix: str) -> Expr:
    return str._expr.str.ends_with(suffix).alias(f"{str._expr.name}_endswith_{suffix}")


def substring(str: Expr, pos: int, len: int | None = None) -> Expr:
    if len is not None:
        return str._expr.str.slice(pos - 1, len).alias(f"{str._expr.name}_substring_{pos}_{len}")
    return str._expr.str.slice(pos - 1).alias(f"{str._expr.name}_substring_{pos}")


substr = substring


def trim(col: Expr, trim) -> Expr:
    return col._expr.str.strip_chars(trim).alias(f"{col._expr.name}_trim_{trim}")


def when(condition, value):
    return plf.when(condition).then(value)

def concat_ws(separator: str, *cols) -> Expr:
    # Extract the underlying expressions from ColExtension objects
    exprs = []
    for c in cols:
        if hasattr(c, "_expr"):
            exprs.append(c._expr)
        else:
            exprs.append(c)
    # If only one column, return it as-is (for string columns)
    if len(exprs) == 1:
        return exprs[0].list.join(separator)
    return plf.concat_str(*exprs, separator=separator)


def expr(str):
    return plf.sql_expr(str)  # This is a placeholder; actual implementation may vary

def broadcast(df):
    return df