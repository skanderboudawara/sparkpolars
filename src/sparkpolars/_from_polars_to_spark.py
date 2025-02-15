from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame as PolarsLazyFrame
from polars.datatypes import (
    Array,
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
    Int128,
    List,
    Null,
    String,
    Struct,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from polars.datatypes import (
    DataType as PolarsDataTypes,
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
    NullType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.types import (
    DataType as SparkDataTypes,
)

from .config import Config

SIMPLE_TYPES: dict = {
    String(): StringType(),
    Int8(): ByteType(),
    Int16(): ShortType(),
    Int32(): IntegerType(),
    Int64(): LongType(),
    Int128(): LongType(),
    UInt8(): ByteType(),
    UInt16(): ShortType(),
    UInt32(): IntegerType(),
    UInt64(): LongType(),
    Boolean(): BooleanType(),
    Float32(): FloatType(),
    Date(): DateType(),
    Float64(): DoubleType(),
    Binary(): BinaryType(),
    Null(): NullType(),
}


def _convert_polars_array(polar_type: PolarsDataTypes, *, is_map: bool) -> SparkDataTypes:
    """
    This function converts Polars Struct to Spark StructType.

    :param polar_type: The Polars Struct type to convert

    :param is_map: If True, the Polars type is a map. Default is False.

    :return: The equivalent PySpark StructType
    """
    if is_map:
        if (
            polar_type.inner != Struct
            or len(polar_type.inner.fields) != 2
            or polar_type.inner.fields[0].name != "key"
            or polar_type.inner.fields[1].name != "value"
        ):
            msg = "Only types List(Struct(key: Type, value: Type)) are supported for maps."
            raise ValueError(msg, repr(polar_type))
        return MapType(
            _type_convert_polars_to_spark(polar_type.inner.fields[0].dtype),
            _type_convert_polars_to_spark(polar_type.inner.fields[1].dtype),
        )
    return ArrayType(_type_convert_polars_to_spark(polar_type.inner))


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
    if isinstance(polar_type, Struct):
        return StructType(
            [
                StructField(field.name, _type_convert_polars_to_spark(field.dtype))
                for field in polar_type.fields
            ],
        )
    if isinstance(polar_type, List):
        return _convert_polars_array(polar_type, is_map=is_map)
    if isinstance(polar_type, Decimal):
        precision = polar_type.precision if polar_type.precision is not None else 38
        return DecimalType(precision, polar_type.scale)
    if isinstance(polar_type, Datetime):
        return TimestampType()
    for pl_type, spark_type in SIMPLE_TYPES.items():
        if polar_type == pl_type:
            return spark_type
    msg = f"Unsupported type: {polar_type!r}"
    raise ValueError(msg)


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


def _pack_map(dict_list: list[dict]) -> dict:
    """
    Convert a list of {'key': ..., 'value': ...} dictionaries into a single dictionary.

    :param dict_list: The list of dictionaries

    :return: The dictionary
    """
    return {d["key"]: d["value"] for d in dict_list}


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
                _pack_map(value)
                if column_name in map_elements
                else _polars_dict_to_row(value, config) if isinstance(value, dict) else value
            )
            for column_name, value in polar_dict.items()
        },
    )


def _convert_array_to_list(
    df: PolarsDataFrame | PolarsLazyFrame,
) -> PolarsDataFrame | PolarsLazyFrame:
    """
    Convert Polars Array to list.

    :param df: The Polars DataFrame or LazyFrame

    :return: The Polars DataFrame or LazyFrame
    """
    for column_name, column_type in df.collect_schema().items():
        if isinstance(column_type, Array):
            df = df.with_columns(df[column_name].arr.to_list().alias(column_name))
    return df
