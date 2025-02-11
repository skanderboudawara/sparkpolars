from src.sparkpolars._from_polars_to_spark import _pack_map
from src.sparkpolars._from_spark_to_polars import _unpack_map


def test__pack_map():
    dict_list = [{"key": "a", "value": 1}, {"key": "b", "value": 2}]
    assert _pack_map(dict_list) == {"a": 1, "b": 2}


def test__unpack_map():
    dict_ = {"a": 1, "b": 2}
    assert _unpack_map(dict_) == [{"key": "a", "value": 1}, {"key": "b", "value": 2}]
    assert _unpack_map({}) == []
