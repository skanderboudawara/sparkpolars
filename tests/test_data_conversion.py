import pytest

from src._utils import _polars_dict_to_row, _spark_row_as_dict
from src.config import Config


def test__data_convert(spark_session, spark_data, polars_data):

    assert _spark_row_as_dict(spark_data[0]) == polars_data[0]
    assert _polars_dict_to_row(polars_data[0], Config(map_elements=["cin"])) == spark_data[0]
    with pytest.raises(TypeError, match="Expected a dictionary"):
        _polars_dict_to_row("test")
    with pytest.raises(TypeError, match="Expected a Spark Row"):
        _spark_row_as_dict("test")
