from typing import Any

from polars.datatypes import (
    Binary,
    Boolean,
    Date,
    Datetime,
    Decimal,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Null,
    String,
    Struct,
)
from polars.datatypes import (
    DataType as PolarsDataTypes,
)
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import Row as SparkRow
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)
from pyspark.sql.types import (
    DataType as SparkDataTypes,
)

from .config import Config

SIMPLE_TYPES: dict = {
    StringType(): String,
    ByteType(): Int8,
    ShortType(): Int16,
    IntegerType(): Int32,
    LongType(): Int64,
    BooleanType(): Boolean,
    FloatType(): Float32,
    DateType(): Date,
    DoubleType(): Float64,
    BinaryType(): Binary,
    NullType(): Null,
    TimestampNTZType(): Datetime,
}


def _type_convert_pyspark_to_polars(
    pyspark_type: SparkDataTypes,
    config: Config | None = None,
    tz: str | None = None,
) -> PolarsDataTypes:
    """
    Recursively converts PySpark types to Polars types.

    :param pyspark_type: The PySpark type to convert

    :param config: The configuration of the application. Default is None.

    :return: The equivalent Polars type
    """
    if isinstance(pyspark_type, ArrayType):
        return List(_type_convert_pyspark_to_polars(pyspark_type.elementType, config, tz))
    if isinstance(pyspark_type, StructType):
        return Struct(
            {
                field.name: _type_convert_pyspark_to_polars(field.dataType, config, tz)
                for field in pyspark_type.fields
            },
        )
    if isinstance(pyspark_type, MapType):
        return List(
            Struct(
                {
                    "key": _type_convert_pyspark_to_polars(pyspark_type.keyType),
                    "value": _type_convert_pyspark_to_polars(pyspark_type.valueType),
                },
            ),
        )
    if isinstance(pyspark_type, TimestampType):
        if config is None:
            return Datetime(time_zone=tz)
        return Datetime(time_unit=config.time_unit, time_zone=tz)
    if isinstance(pyspark_type, DecimalType):
        return Decimal(pyspark_type.precision, pyspark_type.scale)
    polar_type = SIMPLE_TYPES.get(pyspark_type)
    if polar_type:
        return polar_type
    msg = f"Unsupported type: {pyspark_type!r}"
    raise ValueError(msg)


def _convert_schema_spark_to_polars(
    df_schema_field: list[StructField],
    config: Config | None = None,
    tz: str | None = None,
) -> dict[str, Any]:
    """
    Converts a PySpark schema to a Polars schema.

    :param df_schema_field: list of Structfield

    :param config: The configuration of the application. Default is None.

    :return: The Polars schema
    """
    return {
        field.name: _type_convert_pyspark_to_polars(field.dataType, config, tz)
        for field in df_schema_field
    }


def _unpack_map(packed_dict: dict) -> list[dict]:
    """
    This method converts a dictionary into a list of {'key': ..., 'value': ...} dictionaries.

    :param packed_dict: The dictionary to convert

    :return: The list of dictionaries
    """
    return [{"key": k, "value": v} for k, v in packed_dict.items()]


def _spark_row_as_dict(row: SparkRow, config: Config | None = None) -> dict[str, Any]:
    """
    Converts a Spark Row to a dictionary.

    :param row: The Spark Row

    :param config: The configuration of the application. Default is None.

    :return: The dictionary representation of the Spark Row
    """
    if not isinstance(row, SparkRow):
        msg = "Expected a Spark Row"
        raise TypeError(msg)
    return {key: __convert_value(value, config) for key, value in row.asDict().items()}


def __convert_value(value: Any, config: Config | None = None) -> Any:
    """
    Method support to convert value to Polars DataFrame.

    Mitigates nested Rows in lists

    :param value: The value to convert

    :param config: The configuration of the application. Default is None.

    :return: The converted value
    """
    if isinstance(value, SparkRow):
        return _spark_row_as_dict(value, config)
    if isinstance(value, dict):
        return _unpack_map(value)
    if isinstance(value, list):
        return [__convert_value(v, config) for v in value]
    return value


def _get_time_zone(spark_dataframe: SparkDataFrame) -> str:  # pragma: no cover
    """
    This function gets the time zone from the Spark DataFrame.

    :param spark_dataframe: The Spark DataFrame

    :return: The time zone
    """
    session_tz = spark_dataframe.sparkSession.conf.get("spark.sql.session.timeZone")
    if session_tz:
        return session_tz
    sql_tz = spark_dataframe.sparkSession.conf.get("spark.sql.timezone")
    if sql_tz:
        return sql_tz
    return "UTC"
