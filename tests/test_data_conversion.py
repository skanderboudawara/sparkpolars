import pytest

from src.sparkpolars._from_polars_to_spark import _polars_dict_to_row
from src.sparkpolars._from_spark_to_polars import _spark_row_as_dict
from src.sparkpolars.config import Config


def compare_dicts(d1, d2):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            return False
        return all(compare_dicts(d1[key], d2[key]) for key in d1)
    if isinstance(d1, list) and isinstance(d2, list):
        return sorted(d1, key=str) == sorted(d2, key=str)
    if isinstance(d1, set) and isinstance(d2, set):
        return d1 == d2
    return d1 == d2


def test__data_convert(spark_session, spark_data, polars_data):

    assert compare_dicts(_spark_row_as_dict(spark_data[0]), polars_data[0])
    assert _polars_dict_to_row(polars_data[0], Config(map_elements=["cin"])) == spark_data[0]
    with pytest.raises(TypeError, match="Expected a dictionary"):
        _polars_dict_to_row("test")
    with pytest.raises(TypeError, match="Expected a Spark Row"):
        _spark_row_as_dict("test")
