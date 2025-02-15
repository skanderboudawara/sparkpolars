"""
This library provides functions to convert PySpark DataFrames to Polars DataFrames and vice versa.
"""

from enum import Enum

from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame as PolarsLazyFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from ._from_polars_to_spark import (
    _convert_array_to_list,
    _convert_schema_polars_to_spark,
    _polars_dict_to_row,
)
from ._from_spark_to_polars import (
    _convert_schema_spark_to_polars,
    _get_time_zone,
    _spark_row_as_dict,
)
from ._importlib_utils import check_version_and_module
from .config import Config


class ModeMethod(Enum):
    """
    The method to use for conversion.

    - NATIVE: Use the native method.
    - ARROW: Use the Arrow method.
    - PANDAS: Use the Pandas method.
    """

    NATIVE = "native"
    ARROW = "arrow"
    PANDAS = "pandas"


def toPolars(
    self: SparkDataFrame,
    *,
    lazy: bool = False,
    config: Config | None = None,
    mode: ModeMethod = ModeMethod.NATIVE,
) -> PolarsDataFrame | PolarsLazyFrame | None:
    """
    Converts a PySpark DataFrame to a Polars DataFrame.

    :param self: PySpark DataFrame

    :param lazy: If True, returns a Polars LazyFrame. Default is False.

    :param config: The configuration of the application. Default is None.

    :param mode: The method to use for conversion. Default is ModeMethod.NATIVE.

    :return: Polars DataFrame or Polars LazyFrame
    """
    check_version_and_module("pyspark", "3.3.0")
    check_version_and_module("polars", "1.0.0")
    tz = _get_time_zone(self)
    polars_frame = None
    polars_schema = _convert_schema_spark_to_polars(self.schema.fields, config, tz)

    if mode == ModeMethod.NATIVE:
        polars_data = [_spark_row_as_dict(row, config) for row in self.collect()]
        frame_kwargs = {
            "schema_overrides" if polars_data else "schema": polars_schema,
            "data": polars_data,
            "strict": False,
        }
        polars_frame = PolarsDataFrame(**frame_kwargs)

    if mode == ModeMethod.ARROW:
        check_version_and_module("pyarrow", "1.0.0")
        from polars import from_arrow
        from pyarrow import Table

        self.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        polars_frame = from_arrow(
            Table.from_batches(self._collect_as_arrow()),
            schema_overrides=polars_schema,
        )

    if mode == ModeMethod.PANDAS:
        check_version_and_module("pandas", "1.0.0")
        from polars import from_pandas

        polars_frame = from_pandas(self.toPandas(), schema_overrides=polars_schema)

    if polars_frame is None:
        msg = "Method not implemented."
        raise NotImplementedError(msg)
    if lazy:
        return polars_frame.lazy()
    return polars_frame


def to_spark(
    self: PolarsDataFrame | PolarsLazyFrame,
    *,
    spark: SparkSession | None = None,
    config: Config | None = None,
    mode: ModeMethod = ModeMethod.NATIVE,
) -> SparkDataFrame | None:
    """
    Converts a Polars DataFrame to a PySpark DataFrame.

    :param self: Polars DataFrame

    :param spark: The SparkSession. Default is None.

    :param config: The configuration of the application. Default is None.

    :param mode: The method to use for conversion. Default is ModeMethod.NATIVE.

    :return: PySpark DataFrame
    """
    check_version_and_module("pyspark", "3.3.0")
    check_version_and_module("polars", "1.0.0")
    if spark is None:
        spark = SparkSession.getActiveSession()
    if isinstance(self, PolarsLazyFrame):
        self = self.collect()
    self = _convert_array_to_list(self)
    spark_schema = _convert_schema_polars_to_spark(self.schema, config)

    if mode == ModeMethod.NATIVE:
        spark_data = [_polars_dict_to_row(row, config) for row in self.to_dicts()]
        return spark.createDataFrame(data=spark_data, schema=spark_schema)

    if mode == ModeMethod.PANDAS:
        check_version_and_module("pandas", "1.0.0")
        return spark.createDataFrame(self.to_pandas(), schema=spark_schema)

    if mode == ModeMethod.ARROW:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        check_version_and_module("pandas", "1.0.0")
        check_version_and_module("pyarrow", "1.0.0")
        data = self.to_arrow()
        return spark.createDataFrame(data.to_pandas(), schema=spark_schema)

    msg = "Method not implemented."
    raise NotImplementedError(msg)
