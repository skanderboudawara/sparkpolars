from src._utils import _convert_schema_polars_to_spark, _convert_schema_spark_to_polars
from src.config import Config


def test_schema_from_spark_to_polars(schema_spark, schema_polars):
    assert _convert_schema_polars_to_spark(schema_polars, Config(map_elements=["cin"])) == schema_spark
    assert _convert_schema_spark_to_polars(schema_spark) == schema_polars
