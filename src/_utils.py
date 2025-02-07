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
    String,
    Struct,
)
from polars.datatypes import (
    DataType as PolarsDataTypes,
)
from polars.datatypes import (
    Object as PolarsObject,
)
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
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.types import (
    DataType as SparkDataTypes,
)

from config import Config

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
}


def _type_convert_pyspark_to_polars(
    pyspark_type: SparkDataTypes,
    config: Config | None = None,
) -> PolarsDataTypes:
    """
    Recursively converts PySpark types to Polars types.

    :param pyspark_type: The PySpark type to convert

    :param config: The configuration of the application. Default is None.

    :return: The equivalent Polars type
    """
    if isinstance(pyspark_type, ArrayType):
        return List(_type_convert_pyspark_to_polars(pyspark_type.elementType))
    if isinstance(pyspark_type, StructType):
        return Struct(
            {
                field.name: _type_convert_pyspark_to_polars(field.dataType)
                for field in pyspark_type.fields
            },
        )
    if isinstance(pyspark_type, MapType):
        if config is None:
            return PolarsObject()
        return Datetime(time_unit=config.time_unit, time_zone=config.time_zone)
    if isinstance(pyspark_type, TimestampType):
        if config is None:
            return Datetime()
        return Datetime(time_unit=config.time_unit, time_zone=config.time_zone)
    if isinstance(pyspark_type, DecimalType):
        return Decimal(pyspark_type.precision, pyspark_type.scale)
    polar_type = SIMPLE_TYPES.get(pyspark_type)
    if polar_type:
        return polar_type
    msg = f"Unsupported type: {pyspark_type!r}"
    raise ValueError(msg)


def _convert_polars_struct(polar_type: PolarsDataTypes, *, is_map: bool) -> SparkDataTypes:
    """
    This function converts Polars Struct to Spark StructType.

    :param polar_type: The Polars Struct type to convert

    :param is_map: If True, the Polars type is a map. Default is False.

    :return: The equivalent PySpark StructType
    """
    if is_map:
        all_field_types = [field.dtype for field in polar_type.fields]
        if len(set(all_field_types)) != 1:
            msg = "Multiple Types found we cannot convert to MapType: got "
            raise ValueError(msg, repr(polar_type))
        return MapType(StringType(), _type_convert_polars_to_spark(all_field_types[0]))
    return StructType(
        [
            StructField(field.name, _type_convert_polars_to_spark(field.dtype))
            for field in polar_type.fields
        ],
    )


def _type_convert_polars_to_spark(
    polar_type: PolarsDataTypes,
    *,
    is_map: bool = False,
) -> SparkDataTypes:
    """
    Recursively converts PySpark types to Polars types.

    :param polar_type: The Polars type to convert

    :param is_map: If True, the Polars type is a map. Default is False.

    :return: The equivalent PySpark type
    """
    if is_map is None:
        is_map = False
    if isinstance(polar_type, List):
        return ArrayType(_type_convert_polars_to_spark(polar_type.inner))
    if isinstance(polar_type, PolarsObject):
        return MapType(StringType(), StringType())
    if isinstance(polar_type, Struct):
        return _convert_polars_struct(polar_type, is_map=is_map)
    if isinstance(polar_type, Decimal):
        precision = polar_type.precision if polar_type.precision is not None else 38
        return DecimalType(precision, polar_type.scale)
    if isinstance(polar_type, Datetime):
        return TimestampType()
    for spark_type, pl_type in SIMPLE_TYPES.items():
        if polar_type == pl_type:
            return spark_type
    msg = f"Unsupported type: {polar_type!r}"
    raise ValueError(msg)


def _convert_schema_spark_to_polars(
    df_schema_field: list[StructField],
    config: Config | None = None,
) -> dict[str, Any]:
    """
    Converts a PySpark schema to a Polars schema.

    :param df_schema_field: list of Structfield

    :param config: The configuration of the application. Default is None.

    :return: The Polars schema
    """
    return {
        field.name: _type_convert_pyspark_to_polars(field.dataType, config)
        for field in df_schema_field
    }


def _convert_schema_polars_to_spark(
    df_schema: PolarsDataTypes,
    config: Config | None = None,
) -> StructType:
    """
    Converts a Polars schema to a PySpark schema.

    :param df_schema: The polars dataframe schema

    :param config: The configuration of the application. Default is None.

    :return: The PySpark schema
    """
    return StructType(
        [
            StructField(
                field_name,
                _type_convert_polars_to_spark(
                    field_type,
                    is_map=config and field_name in config.map_elements,
                ),
            )
            for field_name, field_type in dict(df_schema).items()
        ],
    )


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
    return {
        key: _spark_row_as_dict(value, config) if isinstance(value, SparkRow) else value
        for key, value in row.asDict().items()
    }


def _polars_dict_to_row(
    polar_dict: dict,
    config: Config | None = None,
) -> SparkRow:
    """
    Convert polars dict to a Spark Row, except for keys in config.map_element.

    :param polar_dict: The polars dict
    :param config: The configuration of the application. Default is None.
    :returns: Spark Row representation
    """
    if not isinstance(polar_dict, dict):
        msg = "Expected a dictionary"
        raise TypeError(msg)

    map_elements = set(config.map_elements) if config else set()

    return SparkRow(
        **{
            column_name: (
                value
                if column_name in map_elements
                else _polars_dict_to_row(value, config) if isinstance(value, dict) else value
            )
            for column_name, value in polar_dict.items()
        },
    )
