from polars import (
    DataFrame as PolarsDataFrame,
)
from polars import (
    LazyFrame as PolarsLazyFrame,
)
from pyspark.sql import DataFrame as SparkDataFrame

from .config import Config
from .sparkpolars import ModeMethod, to_spark, toPolars

__all__ = [
    "Config",
    "ModeMethod",
    "toPolars",
    "to_spark",
]


SparkDataFrame.toPolars = toPolars
PolarsDataFrame.to_spark = to_spark
PolarsLazyFrame.to_spark = to_spark
