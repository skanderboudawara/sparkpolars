"""
This module provides a DataFrameExtension class that extends Polars DataFrame with Spark-like API methods.

The DataFrameExtension class maintains compatibility with Spark DataFrame operations while leveraging
the performance benefits of the Polars library. It provides method chaining capabilities and
automatic type preservation through custom attribute access interception.
"""

from typing import Any

import polars as plx
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
    def withColumn(self, name: str, expr: Any) -> "DataFrameExtension":
        return self.with_columns(expr.alias(name))

    def withColumns(self, columns: dict[str, Any]) -> "DataFrameExtension":
        columns = [expr.alias(name) for name, expr in columns.items()]
        return self.with_columns(columns)

    def hint(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        return self

    def repartition(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        return self

    def coalesce(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        return self

    def persist(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        return self.cache()

    def distinct(self) -> "DataFrameExtension":
        return self.unique()
    
    def dropDuplicates(self, subset = None) -> "DataFrameExtension":
        if subset:
            return self.distinct()
        raise NotImplementedError(
            "drop_duplicates with 'subset' parameter is not implemented."
        )
    
    def drop_duplicates(self, subset = None):
        if subset:
            return self.distinct()
        raise NotImplementedError(
            "drop_duplicates with 'subset' parameter is not implemented."
        )
    
    def dropna(self, how='any', thresh=None, subset=None):
        if how == 'any' and thresh is None:
            return self.drop_nans(subset)
        raise NotImplementedError(
            "dropna with 'how' or 'thresh' parameters is not implemented."
        )
    def dropnulls(self, how='any', thresh=None, subset=None):
        if how == 'any' and thresh is None:
            return self.drop_nulls(subset)
        raise NotImplementedError(
            "dropna with 'how' or 'thresh' parameters is not implemented."
        )


    def join(
        self,
        other: "DataFrameExtension",
        on: str | list[str],
        how: str = "inner",
    ) -> "DataFrameExtension":
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
        return concat(
            [self, other],
            how="diagonal_relaxed" if allowMissingColumns else "vertical_relaxed",
        )

    def crossJoin(self, other: "DataFrameExtension") -> "DataFrameExtension":
        return self.join(other, how="cross")

    def checkpoint(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        return self

    def localCheckpoint(self, *args: Any, **kwargs: Any) -> "DataFrameExtension":
        return self
    
    def exceptAll(self, other):
        with plx.SQLContext() as ctx:
            ctx.register("self_sql", self)
            ctx.register("other_sql", other)
            return ctx.execute(
                "SELECT * FROM self_sql EXCEPT ALL SELECT * FROM other_sql"
            )

    def __getattribute__(self, name: str) -> Any:
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
        super().__init__(
            data=data.collect() if isinstance(data, LazyFrameOriginal) else data,
            schema=schema,
            schema_overrides=schema_overrides,
            strict=strict,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        )

    def schema(self):
        return super().collect_schema()

    def isEmpty(self) -> bool:
        result = self.limit(1).collect().is_empty()
        return result

    def count(self) -> int:
        result = self.collect().height
        return result

    @property
    def columns(self):
        return self.schema().names()


class DataFrame(DataFrameExtension, DataFrameOriginal):
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
        super().__init__(
            data=data,
            schema=schema,
            schema_overrides=schema_overrides,
            strict=strict,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        )

    def isEmpty(self):
        return self.is_empty()

    def count(self) -> int:
        return self.height

df = DataFrame({
    "name": [1, 2, 3]
})

df2 = DataFrame({
    "name": [1, 2]
})

df1 = df.exceptAll(df2)
print(df1)