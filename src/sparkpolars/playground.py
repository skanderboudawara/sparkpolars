import polars as pl
import polyspark.sql.functions as F
import polyspark.sql.window as W

# Expr.over(
# partition_by: IntoExpr | Iterable[IntoExpr] | None = None,
# *more_exprs: IntoExpr,
# order_by: IntoExpr | Iterable[IntoExpr] | None = None,
# descending: bool = False,
# nulls_last: bool = False,
# mapping_strategy: WindowMappingStrategy = 'group_to_rows',
# )

# self = self.sort(
#     by=[*partition_cols, *col_names],
#     descending=[True] * len(partition_cols) + ordering,
#     nulls_last=[True] * len(partition_cols) + null_conditions,
#     maintain_order=True,
# ).with_columns(
#     struct(*partition_cols).rank("ordinal").over(*partition_cols).alias("_keep_"),
# )
df = pl.DataFrame({"a": [1, 4, 3, 2], "b": [1, 1, 2, 2], "c": [1, 1, 1, 1]})
df = df.with_columns(
    F.asc(F.col("a")).alias("asc_a"),
    F.desc(F.col("a")).alias("desc_a"),

).with_columns(
   F.col("c").rank("ordinal").over(
       W.Window().partitionBy(F.col("b").alias("b")).orderBy(F.col("a"))
    ).alias("rank"),  # This is equivalent to `pl.all()`
)

# def row_number():
#     return pl.struct(pl.all()).rank("average")


# df = df.with_columns(
#     row_number().over(
#         order_by=[F.desc(F.col("a"))]
#     ).alias("row_number")
# )

print(df)