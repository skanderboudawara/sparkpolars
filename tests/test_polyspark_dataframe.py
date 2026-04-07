"""Tests for polyspark DataFrame monkey-patches."""

import pytest
import polars as pl
from polars.testing import assert_frame_equal

import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401 — installs patches


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def simple_df():
    return pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": ["x", "y", "z", "x", "y"],
        "c": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture()
def simple_lf(simple_df):
    return simple_df.lazy()


@pytest.fixture()
def null_df():
    return pl.DataFrame({
        "a": [1, 2, 3],
        "b": [None, "y", None],
        "c": [1.0, None, 3.0],
    })


@pytest.fixture()
def other_df():
    return pl.DataFrame({
        "a": [3, 4, 5, 6, 7],
        "b": ["z", "x", "y", "a", "b"],
        "c": [3.0, 4.0, 5.0, 6.0, 7.0],
    })


@pytest.fixture()
def other_lf(other_df):
    return other_df.lazy()


@pytest.fixture()
def wide_df():
    return pl.DataFrame({
        "id": [1, 2],
        "val_a": [10, 20],
        "val_b": [30, 40],
    })


# ── return_self ───────────────────────────────────────────────────────────────
_RETURN_SELF_DF = [
    ("hint", ("broadcast",), {}),
    ("alias", ("my_alias",), {}),
    ("repartition", (4,), {}),
    ("coalesce", (1,), {}),
    ("unpersist", (), {}),
    ("repartitionByRange", (4,), {}),
    ("withMetadata", ("a", {"comment": "x"}), {}),
    ("cache", (), {}),
    ("checkpoint", (), {}),
    ("localCheckpoint", (), {}),
    ("collect", (), {}),
]

_RETURN_SELF_LF = [t for t in _RETURN_SELF_DF if t[0] != "collect"]


@pytest.mark.parametrize("method,args,kwargs", _RETURN_SELF_DF)
def test_return_self_df(simple_df, method, args, kwargs):
    result = getattr(simple_df, method)(*args, **kwargs)
    assert result is simple_df


@pytest.mark.parametrize("method,args,kwargs", _RETURN_SELF_LF)
def test_return_self_lf(simple_lf, method, args, kwargs):
    result = getattr(simple_lf, method)(*args, **kwargs)
    assert result is simple_lf


# ── not_implemented ───────────────────────────────────────────────────────────
_NOT_IMPLEMENTED = [
    "approxQuantile",
    "asTable",
    "crosstab",
    "cube",
    "dropDuplicatesWithinWatermark",
    "executionInfo",
    "freqItems",
    "groupingSets",
    "inputFiles",
    "metadataColumn",
    "observe",
    "rdd",
    "rollup",
    "sameSemantics",
    "sampleBy",
    "scalar",
    "semanticHash",
    "stat",
    "storageLevel",
    "to",
    "toLocalIterator",
    "withWatermark",
    "writeStream",
    "writeTo",
    "mergeInto",
    "pandas_api",
]


@pytest.mark.parametrize("method", _NOT_IMPLEMENTED)
def test_not_implemented_df(simple_df, method):
    with pytest.raises(NotImplementedError):
        getattr(simple_df, method)()


@pytest.mark.parametrize("method", _NOT_IMPLEMENTED)
def test_not_implemented_lf(simple_lf, method):
    with pytest.raises(NotImplementedError):
        getattr(simple_lf, method)()


# ── describe ──────────────────────────────────────────────────────────────────
def test_describe_all_cols_df(simple_df):
    result = simple_df.describe()
    assert isinstance(result, pl.DataFrame)
    assert "statistic" in result.columns
    assert "a" in result.columns


def test_describe_subset_cols_df(simple_df):
    result = simple_df.describe("a", "c")
    assert "b" not in result.columns
    assert "a" in result.columns and "c" in result.columns


def test_describe_lf(simple_lf):
    result = simple_lf.describe()
    assert isinstance(result, pl.DataFrame)
    assert "statistic" in result.columns


def test_describe_subset_lf(simple_lf):
    result = simple_lf.describe("a")
    assert result.columns == ["statistic", "a"]


# ── summary ───────────────────────────────────────────────────────────────────
def test_summary_df(simple_df):
    result = simple_df.summary()
    assert isinstance(result, pl.DataFrame)
    assert "statistic" in result.columns


def test_summary_lf(simple_lf):
    result = simple_lf.summary()
    assert isinstance(result, pl.DataFrame)


# ── fillna ────────────────────────────────────────────────────────────────────
def test_fillna_all_cols(null_df):
    result = null_df.fillna("FILLED")
    assert result["b"].null_count() == 0


def test_fillna_subset(null_df):
    result = null_df.fillna(0.0, subset=["c"])
    assert result["c"].null_count() == 0
    assert result["b"].null_count() == null_df["b"].null_count()


def test_fillna_lf(null_df):
    result = null_df.lazy().fillna("X").collect()
    assert result["b"].null_count() == 0


# ── filter / where ────────────────────────────────────────────────────────────
def test_filter_expr_df(simple_df):
    result = simple_df.filter(pl.col("a") > 2)
    assert result.height == 3


def test_filter_string_sql_df(simple_df):
    result = simple_df.filter("a > 2")
    assert result.height == 3


def test_filter_lf(simple_lf):
    result = simple_lf.filter("a > 2").collect()
    assert result.height == 3


def test_where_alias_df(simple_df):
    result = simple_df.where(pl.col("a") > 2)
    assert result.height == 3


def test_where_alias_lf(simple_lf):
    result = simple_lf.where("a > 2").collect()
    assert result.height == 3


# ── first ─────────────────────────────────────────────────────────────────────
def test_first_df(simple_df):
    result = simple_df.first()
    assert isinstance(result, dict)
    assert result["a"] == 1


def test_first_empty_df():
    empty = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    assert empty.first() is None


def test_first_lf(simple_lf):
    result = simple_lf.first()
    assert isinstance(result, dict)
    assert "a" in result


# ── intersect / subtract ──────────────────────────────────────────────────────
def test_intersect_df(simple_df, other_df):
    result = simple_df.intersect(other_df)
    assert isinstance(result, pl.DataFrame)
    result_a = set(result["a"].to_list())
    assert result_a.issubset({3, 4, 5})


def test_intersect_lf(simple_lf, other_lf):
    result = simple_lf.intersect(other_lf).collect()
    assert isinstance(result, pl.DataFrame)
    assert set(result["a"].to_list()).issubset({3, 4, 5})


# ── intersectAll ─────────────────────────────────────────────────────────────
@pytest.fixture()
def dup_df():
    return pl.DataFrame({"C1": ["a", "a", "b", "c"], "C2": [1, 1, 3, 4]})


@pytest.fixture()
def dup_df2():
    return pl.DataFrame({"C1": ["a", "a", "b"], "C2": [1, 1, 3]})


def test_intersectAll_preserves_duplicates(dup_df, dup_df2):
    result = dup_df.intersectAll(dup_df2).sort("C1", "C2")
    assert result.height == 3  # ("a",1)×2 + ("b",3)×1
    assert result.to_dicts() == [{"C1": "a", "C2": 1}, {"C1": "a", "C2": 1}, {"C1": "b", "C2": 3}]


def test_intersectAll_respects_min_count(dup_df):
    # df2 has only one ("a",1) — result should have one, not two
    df2 = pl.DataFrame({"C1": ["a", "b"], "C2": [1, 3]})
    result = dup_df.intersectAll(df2).sort("C1", "C2")
    assert result.height == 2
    assert result["C1"].to_list() == ["a", "b"]


def test_intersectAll_no_common_rows(dup_df):
    df2 = pl.DataFrame({"C1": ["x", "y"], "C2": [9, 10]})
    result = dup_df.intersectAll(df2)
    assert result.height == 0


def test_intersectAll_lf(dup_df, dup_df2):
    result = dup_df.lazy().intersectAll(dup_df2.lazy()).collect().sort("C1", "C2")
    assert result.height == 3


def test_subtract_df(simple_df, other_df):
    result = simple_df.subtract(other_df)
    assert isinstance(result, pl.DataFrame)
    assert all(v in {1, 2} for v in result["a"].to_list())


def test_subtract_lf(simple_lf, other_lf):
    result = simple_lf.subtract(other_lf).collect()
    assert all(v in {1, 2} for v in result["a"].to_list())


# ── isLocal / isStreaming ─────────────────────────────────────────────────────
def test_isLocal_df(simple_df):
    assert simple_df.isLocal() is True


def test_isLocal_lf(simple_lf):
    assert simple_lf.isLocal() is True


def test_isStreaming_df(simple_df):
    assert simple_df.isStreaming is False


def test_isStreaming_lf(simple_lf):
    assert simple_lf.isStreaming is False


# ── melt ─────────────────────────────────────────────────────────────────────
def test_melt_df(wide_df):
    result = wide_df.melt(id_vars=["id"], value_vars=["val_a", "val_b"])
    assert "variable" in result.columns
    assert "value" in result.columns
    assert result.height == 4


def test_melt_custom_names_df(wide_df):
    result = wide_df.melt(
        id_vars=["id"], value_vars=["val_a", "val_b"],
        var_name="metric", value_name="amount",
    )
    assert "metric" in result.columns
    assert "amount" in result.columns


def test_melt_lf(wide_df):
    result = wide_df.lazy().melt(id_vars=["id"], value_vars=["val_a", "val_b"]).collect()
    assert result.height == 4


# ── offset ────────────────────────────────────────────────────────────────────
def test_offset_df(simple_df):
    result = simple_df.offset(2)
    assert result.height == 3
    assert result["a"][0] == 3


def test_offset_lf(simple_lf):
    result = simple_lf.offset(2).collect()
    assert result.height == 3
    assert result["a"][0] == 3


# ── sort / orderBy / sortWithinPartitions ─────────────────────────────────────
def test_sort_descending_df(simple_df):
    result = simple_df.sort("a", ascending=False)
    assert result["a"][0] == 5


def test_sort_ascending_df(simple_df):
    result = simple_df.sort("a", ascending=True)
    assert result["a"][0] == 1


def test_sort_multi_col(simple_df):
    result = simple_df.sort("b", "a", ascending=[True, False])
    assert isinstance(result, pl.DataFrame)


def test_sort_no_cols(simple_df):
    result = simple_df.sort()
    assert_frame_equal(result, simple_df)


def test_orderBy_alias_df(simple_df):
    result = simple_df.orderBy("a", ascending=False)
    assert result["a"][0] == 5


def test_sortWithinPartitions_alias(simple_df):
    result = simple_df.sortWithinPartitions("a", ascending=True)
    assert result["a"][0] == 1


def test_sort_lf(simple_lf):
    result = simple_lf.sort("a", ascending=False).collect()
    assert result["a"][0] == 5


def test_sort_list_cols(simple_df):
    result = simple_df.sort(["a"], ascending=False)
    assert result["a"][0] == 5


# ── printSchema ───────────────────────────────────────────────────────────────
def test_printSchema_df(simple_df, capsys):
    simple_df.printSchema()
    captured = capsys.readouterr()
    assert "a" in captured.out


def test_printSchema_lf(simple_lf, capsys):
    simple_lf.printSchema()
    captured = capsys.readouterr()
    assert "a" in captured.out


# ── randomSplit ───────────────────────────────────────────────────────────────
def test_randomSplit_covers_all_rows(simple_df):
    splits = simple_df.randomSplit([0.6, 0.4], seed=42)
    assert len(splits) == 2
    assert sum(len(s) for s in splits) == simple_df.height
    for s in splits:
        assert isinstance(s, pl.DataFrame)


def test_randomSplit_no_overlap(simple_df):
    splits = simple_df.randomSplit([0.5, 0.5], seed=0)
    a_sets = [set(s["a"].to_list()) for s in splits]
    assert a_sets[0].isdisjoint(a_sets[1])


def test_randomSplit_three_way(simple_df):
    splits = simple_df.randomSplit([0.5, 0.3, 0.2], seed=1)
    assert len(splits) == 3
    assert sum(len(s) for s in splits) == simple_df.height


def test_randomSplit_lf(simple_lf):
    splits = simple_lf.randomSplit([0.5, 0.5], seed=7)
    assert all(isinstance(s, pl.DataFrame) for s in splits)


# ── replace ───────────────────────────────────────────────────────────────────
def test_replace_scalar_df(simple_df):
    result = simple_df.replace(1, 99)
    assert 99 in result["a"].to_list()
    assert 1 not in result["a"].to_list()


def test_replace_subset_df(simple_df):
    result = simple_df.replace(1, 99, subset=["a"])
    assert 99 in result["a"].to_list()
    # column b should be unchanged
    assert result["b"].to_list() == simple_df["b"].to_list()


def test_replace_dict_df(simple_df):
    result = simple_df.replace({"x": "X", "y": "Y"}, subset=["b"])
    assert "X" in result["b"].to_list()
    assert "x" not in result["b"].to_list()


def test_replace_lf(simple_lf):
    result = simple_lf.replace(1, 99).collect()
    assert 99 in result["a"].to_list()


# ── sample ────────────────────────────────────────────────────────────────────
def test_sample_fraction_df(simple_df):
    result = simple_df.sample(fraction=0.4, seed=42)
    assert isinstance(result, pl.DataFrame)
    assert 1 <= result.height <= simple_df.height


def test_sample_with_replacement_df(simple_df):
    result = simple_df.sample(withReplacement=True, fraction=1.0, seed=0)
    assert result.height == simple_df.height


def test_sample_lf(simple_lf):
    result = simple_lf.sample(fraction=0.4, seed=1)
    assert isinstance(result, pl.DataFrame)


# ── selectExpr ────────────────────────────────────────────────────────────────
def test_selectExpr_df(simple_df):
    result = simple_df.selectExpr("a", "b")
    assert result.columns == ["a", "b"]
    assert result.height == simple_df.height


def test_selectExpr_lf(simple_lf):
    result = simple_lf.selectExpr("a").collect()
    assert result.columns == ["a"]
    assert result.height == 5


# ── take ──────────────────────────────────────────────────────────────────────
def test_take_df(simple_df):
    result = simple_df.take(3)
    assert isinstance(result, list)
    assert len(result) == 3
    assert isinstance(result[0], dict)
    assert result[0]["a"] == 1


def test_take_lf(simple_lf):
    result = simple_lf.take(2)
    assert len(result) == 2
    assert isinstance(result[0], dict)


# ── toArrow ───────────────────────────────────────────────────────────────────
def test_toArrow_df(simple_df):
    import pyarrow as pa
    result = simple_df.toArrow()
    assert isinstance(result, pa.Table)
    assert result.num_rows == 5


def test_toArrow_lf(simple_lf):
    import pyarrow as pa
    result = simple_lf.toArrow()
    assert isinstance(result, pa.Table)


# ── toDF ──────────────────────────────────────────────────────────────────────
def test_toDF_df(simple_df):
    result = simple_df.toDF("x", "y", "z")
    assert result.columns == ["x", "y", "z"]


def test_toDF_wrong_count_df(simple_df):
    with pytest.raises(ValueError, match="toDF"):
        simple_df.toDF("x", "y")


def test_toDF_lf(simple_lf):
    result = simple_lf.toDF("x", "y", "z").collect()
    assert result.columns == ["x", "y", "z"]


# ── toJSON ────────────────────────────────────────────────────────────────────
def test_toJSON_df(simple_df):
    result = simple_df.toJSON()
    assert isinstance(result, str)
    assert '"a"' in result


def test_toJSON_lf(simple_lf):
    result = simple_lf.toJSON()
    assert isinstance(result, str)


# ── toPandas ──────────────────────────────────────────────────────────────────
def test_toPandas_df(simple_df):
    import pandas as pd
    result = simple_df.toPandas()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["a", "b", "c"]


def test_toPandas_lf(simple_lf):
    import pandas as pd
    result = simple_lf.toPandas()
    assert isinstance(result, pd.DataFrame)


# ── transform ─────────────────────────────────────────────────────────────────
def test_transform_df(simple_df):
    def add_col(df):
        return df.with_columns(pl.col("a").alias("a2"))
    result = simple_df.transform(add_col)
    assert "a2" in result.columns


def test_transform_with_args(simple_df):
    def multiply(df, factor):
        return df.with_columns((pl.col("a") * factor).alias("a_scaled"))
    result = simple_df.transform(multiply, 10)
    assert result["a_scaled"][0] == 10


def test_transform_lf(simple_lf):
    def drop_c(df):
        return df.drop("c")
    result = simple_lf.transform(drop_c).collect()
    assert "c" not in result.columns


# ── transpose (LazyFrame only) ────────────────────────────────────────────────
def test_transpose_lf():
    lf = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
    result = lf.transpose()
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (2, 3)


# ── union / unionAll ──────────────────────────────────────────────────────────
def test_union_df(simple_df, other_df):
    result = simple_df.union(other_df)
    assert result.height == simple_df.height + other_df.height


def test_unionAll_df(simple_df, other_df):
    result = simple_df.unionAll(other_df)
    assert result.height == simple_df.height + other_df.height


def test_union_lf(simple_lf, other_lf):
    result = simple_lf.union(other_lf).collect()
    assert result.height == 10


# ── existing methods ──────────────────────────────────────────────────────────
def test_persist_df(simple_df):
    result = simple_df.persist()
    assert result is simple_df


def test_distinct_df(simple_df):
    duped = pl.concat([simple_df, simple_df])
    result = duped.distinct()
    assert result.height == simple_df.height


def test_dropDuplicates_subset_df(simple_df):
    result = simple_df.dropDuplicates(subset=["b"])
    assert result.height == 3  # "x", "y", "z" are unique values


def test_withColumn_df(simple_df):
    result = simple_df.withColumn("d", pl.col("a") * 2)
    assert "d" in result.columns
    assert result["d"][0] == 2


def test_colRegex_df(simple_df):
    result = simple_df.colRegex("^[ab]$")
    assert result.columns == ["a", "b"]


def test_unionByName_df(simple_df):
    result = simple_df.unionByName(simple_df)
    assert result.height == 10


def test_withColumnRenamed_df(simple_df):
    result = simple_df.withColumnRenamed("a", "aa")
    assert "aa" in result.columns
    assert "a" not in result.columns


def test_withColumnsRenamed_df(simple_df):
    result = simple_df.withColumnsRenamed({"a": "aa", "b": "bb"})
    assert result.columns == ["aa", "bb", "c"]


def test_isEmpty_df(simple_df):
    assert simple_df.isEmpty() is False
    assert pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}).isEmpty() is True


def test_count_df(simple_df):
    assert simple_df.count == 5


def test_count_lf(simple_lf):
    assert simple_lf.count == 5


def test_isEmpty_lf(simple_lf):
    assert simple_lf.isEmpty() is False


def test_schema_lf(simple_lf):
    schema = simple_lf.schema
    assert "a" in schema


def test_columns_lf(simple_lf):
    assert simple_lf.columns == ["a", "b", "c"]


def test_drop_nonexistent_df(simple_df):
    result = simple_df.drop("nonexistent")
    assert_frame_equal(result, simple_df)


def test_drop_nonexistent_lf(simple_lf):
    result = simple_lf.drop("nonexistent").collect()
    assert_frame_equal(result, simple_lf.collect())


def test_groupBy_agg_df(simple_df):
    result = simple_df.groupBy("b").agg(pl.col("a").sum())
    assert isinstance(result, pl.DataFrame)
    assert "b" in result.columns


def test_crossJoin_df(simple_df):
    small = simple_df.head(2)
    result = small.crossJoin(small)
    assert result.height == 4


# ── join (all types) ──────────────────────────────────────────────────────────
# simple_df:  a=[1,2,3,4,5]   other_df: a=[3,4,5,6,7]  → overlap {3,4,5}

def test_join_inner_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="inner")
    assert result.height == 3
    assert set(result["a"].to_list()) == {3, 4, 5}


def test_join_left_outer_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="left_outer")
    assert result.height == 5  # all rows from left
    assert set(result["a"].to_list()) == {1, 2, 3, 4, 5}


def test_join_leftouter_alias_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="leftouter")
    assert result.height == 5


def test_join_right_outer_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="right_outer")
    assert result.height == 5  # all rows from right
    assert set(result["a"].to_list()) == {3, 4, 5, 6, 7}


def test_join_rightouter_alias_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="rightouter")
    assert result.height == 5


def test_join_full_outer_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="full_outer")
    assert result.height == 7  # 3 matched + 2 left-only + 2 right-only


def test_join_full_alias_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="full")
    assert result.height == 7


def test_join_fullouter_alias_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="fullouter")
    assert result.height == 7


def test_join_left_anti_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="left_anti")
    assert result.height == 2  # a=1,2 not in other_df
    assert set(result["a"].to_list()) == {1, 2}


def test_join_leftanti_alias_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="leftanti")
    assert result.height == 2


def test_join_semi_df(simple_df, other_df):
    result = simple_df.join(other_df, on="a", how="semi")
    assert result.height == 3  # a=3,4,5 present in both
    assert set(result["a"].to_list()) == {3, 4, 5}


def test_join_cross_df(simple_df, other_df):
    result = simple_df.join(other_df, how="cross")
    assert result.height == 25  # 5 × 5


def test_join_multi_col_df(simple_df, other_df):
    result = simple_df.join(other_df, on=["a", "b", "c"], how="inner")
    assert isinstance(result, pl.DataFrame)
    assert set(result["a"].to_list()).issubset({3, 4, 5})


def test_join_expr_predicate_df(simple_df, other_df):
    result = simple_df.join(other_df, on=pl.col("a") < pl.col("a_right"), how="inner")
    assert isinstance(result, pl.DataFrame)
    assert all(r["a"] < r["a_right"] for r in result.select("a", "a_right").iter_rows(named=True))


def test_join_lf_inner(simple_lf, other_lf):
    result = simple_lf.join(other_lf, on="a", how="inner").collect()
    assert result.height == 3


def test_join_lf_left(simple_lf, other_lf):
    result = simple_lf.join(other_lf, on="a", how="left_outer").collect()
    assert result.height == 5


def test_join_lf_anti(simple_lf, other_lf):
    result = simple_lf.join(other_lf, on="a", how="left_anti").collect()
    assert result.height == 2


def test_join_lf_cross(simple_lf, other_lf):
    result = simple_lf.join(other_lf, how="cross").collect()
    assert result.height == 25


def test_dropDuplicates_lf(simple_lf):
    duped = pl.concat([simple_lf.collect(), simple_lf.collect()]).lazy()
    result = duped.dropDuplicates().collect()
    assert result.height == 5


def test_withColumns_dict_df(simple_df):
    result = simple_df.withColumns({"d": pl.col("a") + 10})
    assert result["d"][0] == 11


def test_withColumns_kwargs_df(simple_df):
    result = simple_df.withColumns(d=pl.col("a") + 10)
    assert result["d"][0] == 11


def test_show_df(simple_df, capsys):
    simple_df.show()
    # show sets display config — just ensure no exception raised
    captured = capsys.readouterr()
    assert captured.out == ""  # show() only configures, does not print


def test_show_vertical_raises(simple_df):
    with pytest.raises(NotImplementedError):
        simple_df.show(vertical=True)


def test_show_negative_n_raises(simple_df):
    with pytest.raises(ValueError):
        simple_df.show(n=-1)


def test_dropna_raises_with_thresh(null_df):
    with pytest.raises(NotImplementedError):
        null_df.dropna(thresh=2)


def test_dropnulls_raises_with_thresh(null_df):
    with pytest.raises(NotImplementedError):
        null_df.dropnulls(thresh=2)


def test_agg_invalid_dict_values(simple_df):
    with pytest.raises(ValueError, match="Aggregation functions"):
        simple_df.agg({"a": "invalid_agg"})


def test_agg_invalid_dict_types(simple_df):
    with pytest.raises(ValueError, match="All keys and values"):
        simple_df.agg({1: "sum"})


# ── explain ───────────────────────────────────────────────────────────────────
def test_explain_df(simple_df):
    result = simple_df.explain()
    assert isinstance(result, str)
    assert len(result) > 0


def test_explain_lf(simple_lf):
    result = simple_lf.explain()
    assert isinstance(result, str)
    assert len(result) > 0


# ── na ────────────────────────────────────────────────────────────────────────
def test_na_fill_df(null_df):
    result = null_df.na.fill("X")
    assert result["b"].null_count() == 0


def test_na_drop_df(null_df):
    result = null_df.na.drop()
    assert result.height == 0  # every row has at least one null


def test_na_replace_df(simple_df):
    result = simple_df.na.replace(1, 99)
    assert 99 in result["a"].to_list()


def test_na_fill_lf(null_df):
    result = null_df.lazy().na.fill(0.0).collect()
    assert result["c"].null_count() == 0


# ── temp views ────────────────────────────────────────────────────────────────
def test_createOrReplaceTempView_df(simple_df):
    import src.sparkpolars.polyspark.sql.dataframe as _mod
    simple_df.createOrReplaceTempView("test_view_df")
    assert "test_view_df" in _mod._local_sql_ctx.tables()
    result = _mod._local_sql_ctx.execute("SELECT * FROM test_view_df").collect()
    assert result.height == 5
    _mod._local_sql_ctx.unregister("test_view_df")


def test_createTempView_raises_if_exists(simple_df):
    import src.sparkpolars.polyspark.sql.dataframe as _mod
    simple_df.createOrReplaceTempView("dup_view")
    with pytest.raises(RuntimeError, match="already exists"):
        simple_df.createTempView("dup_view")
    _mod._local_sql_ctx.unregister("dup_view")


def test_createOrReplaceGlobalTempView_df(simple_df):
    import src.sparkpolars.polyspark.sql.dataframe as _mod
    simple_df.createOrReplaceGlobalTempView("global_view")
    assert "global_view" in _mod._global_sql_ctx.tables()
    _mod._global_sql_ctx.unregister("global_view")


def test_createGlobalTempView_raises_if_exists(simple_df):
    import src.sparkpolars.polyspark.sql.dataframe as _mod
    simple_df.createOrReplaceGlobalTempView("dup_global")
    with pytest.raises(RuntimeError, match="already exists"):
        simple_df.createGlobalTempView("dup_global")
    _mod._global_sql_ctx.unregister("dup_global")


def test_registerTempTable_df(simple_df):
    import src.sparkpolars.polyspark.sql.dataframe as _mod
    simple_df.registerTempTable("reg_view")
    assert "reg_view" in _mod._local_sql_ctx.tables()
    _mod._local_sql_ctx.unregister("reg_view")


# ── exists ────────────────────────────────────────────────────────────────────
def test_exists_non_empty_df(simple_df):
    assert simple_df.exists() is True


def test_exists_empty_df():
    assert pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}).exists() is False


def test_exists_non_empty_lf(simple_lf):
    assert simple_lf.exists() is True


def test_exists_empty_lf():
    assert pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}).lazy().exists() is False


def test_exists_semi_join_pattern():
    """Correlated EXISTS → semi join."""
    customers = pl.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "country": ["USA", "Canada", "USA", "Australia"],
    })
    orders = pl.DataFrame({"customer_id": [101, 102, 103, 101]})
    result = customers.join(orders.select("customer_id").unique(), on="customer_id", how="semi")
    assert set(result["customer_id"].to_list()) == {101, 102, 103}


def test_not_exists_anti_join_pattern():
    """Correlated NOT EXISTS → anti join."""
    customers = pl.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "country": ["USA", "Canada", "USA", "Australia"],
    })
    orders = pl.DataFrame({"customer_id": [101, 102, 103, 101]})
    result = customers.join(orders.select("customer_id").unique(), on="customer_id", how="anti")
    assert result["customer_id"].to_list() == [104]


def test_outer_noop():
    """Expr.outer() is a no-op — returns the same expression."""
    import src.sparkpolars.polyspark.sql.functions  # noqa: F401
    expr = pl.col("x")
    assert expr.outer() is expr


# ── lateralJoin ───────────────────────────────────────────────────────────────
def test_lateralJoin_cross_df(simple_df):
    small = simple_df.head(2)
    result = small.lateralJoin(small)
    assert result.height == 4  # 2 × 2 cross join


def test_lateralJoin_nonequi_df(simple_df, other_df):
    result = simple_df.lateralJoin(
        other_df,
        on=pl.col("a") < pl.col("a_right"),
    )
    assert isinstance(result, pl.DataFrame)
    assert all(
        row["a"] < row["a_right"]
        for row in result.select("a", "a_right").iter_rows(named=True)
    )


def test_lateralJoin_left_raises(simple_df, other_df):
    with pytest.raises(NotImplementedError):
        simple_df.lateralJoin(other_df, on=pl.col("a") < pl.col("a_right"), how="left")


def test_lateralJoin_lf(simple_lf, other_lf):
    result = simple_lf.lateralJoin(other_lf, on=pl.col("a") < pl.col("a_right"))
    assert isinstance(result, pl.LazyFrame)
    assert result.collect().height > 0


# ── foreach / foreachPartition ────────────────────────────────────────────────
def test_foreach_df(simple_df):
    collected = []
    simple_df.foreach(lambda row: collected.append(row["a"]))
    assert collected == [1, 2, 3, 4, 5]


def test_foreach_lf(simple_lf):
    collected = []
    simple_lf.foreach(lambda row: collected.append(row["a"]))
    assert len(collected) == 5


def test_foreachPartition_df(simple_df):
    collected = []
    simple_df.foreachPartition(lambda it: collected.extend(r["a"] for r in it))
    assert sorted(collected) == [1, 2, 3, 4, 5]


def test_foreachPartition_lf(simple_lf):
    collected = []
    simple_lf.foreachPartition(lambda it: collected.extend(r["a"] for r in it))
    assert len(collected) == 5


# ── mapInPandas / mapInArrow ──────────────────────────────────────────────────
def test_mapInPandas_df(simple_df):
    import polars as pl

    def add_col(it):
        for pdf in it:
            yield pdf.assign(d=pdf["a"] * 2)

    result = simple_df.mapInPandas(add_col)
    assert isinstance(result, pl.DataFrame)
    assert "d" in result.columns
    assert result["d"][0] == 2


def test_mapInArrow_df(simple_df):
    import pyarrow as pa
    import polars as pl

    def identity(it):
        for batch in it:
            yield batch

    result = simple_df.mapInArrow(identity)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == simple_df.shape


# ── corr / cov ────────────────────────────────────────────────────────────────
def test_corr_df(simple_df):
    result = simple_df.corr("a", "c")
    assert isinstance(result, float)
    assert abs(result - 1.0) < 1e-9  # a and c are perfectly correlated


def test_corr_spearman_df(simple_df):
    result = simple_df.corr("a", "c", method="spearman")
    assert isinstance(result, float)


def test_corr_lf(simple_lf):
    result = simple_lf.corr("a", "c")
    assert isinstance(result, float)


def test_cov_df(simple_df):
    result = simple_df.cov("a", "c")
    assert isinstance(result, float)
    assert result > 0  # a and c move together


def test_cov_lf(simple_lf):
    result = simple_lf.cov("a", "c")
    assert isinstance(result, float)


def test_createOrReplaceTempView_lf(simple_lf):
    import src.sparkpolars.polyspark.sql.dataframe as _mod
    simple_lf.createOrReplaceTempView("lf_view")
    assert "lf_view" in _mod._local_sql_ctx.tables()
    _mod._local_sql_ctx.unregister("lf_view")


# ── DataFrameWriter (df.write) ─────────────────────────────────────────────────
import tempfile


def test_write_returns_writer(simple_df):
    from src.sparkpolars.polyspark.sql.dataframe import DataFrameWriter
    assert isinstance(simple_df.write, DataFrameWriter)


def test_write_lf_returns_writer(simple_lf):
    from src.sparkpolars.polyspark.sql.dataframe import DataFrameWriter
    assert isinstance(simple_lf.write, DataFrameWriter)


def test_write_builder_chain(simple_df):
    from src.sparkpolars.polyspark.sql.dataframe import DataFrameWriter
    w = simple_df.write.mode("overwrite").option("compression", "snappy")
    assert isinstance(w, DataFrameWriter)
    assert w._mode == "overwrite"
    assert w._options["compression"] == "snappy"


def test_write_parquet_df(simple_df):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    simple_df.write.mode("overwrite").parquet(path)
    result = pl.read_parquet(path)
    assert result.shape == simple_df.shape


def test_write_csv_df(simple_df):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    simple_df.write.mode("overwrite").csv(path)
    result = pl.read_csv(path)
    assert result.shape == simple_df.shape


def test_write_json_df(simple_df):
    with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False, mode="w") as f:
        path = f.name
    simple_df.write.mode("overwrite").json(path)
    result = pl.read_ndjson(path)
    assert result.shape == simple_df.shape


def test_write_ipc_df(simple_df):
    with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
        path = f.name
    simple_df.write.mode("overwrite").ipc(path)
    result = pl.read_ipc(path)
    assert result.shape == simple_df.shape


def test_write_avro_df(simple_df):
    with tempfile.NamedTemporaryFile(suffix=".avro", delete=False) as f:
        path = f.name
    simple_df.write.mode("overwrite").avro(path)
    result = pl.read_avro(path)
    assert result.shape == simple_df.shape


def test_write_text_df(simple_df):
    text_df = simple_df.select(pl.col("b").cast(pl.String).alias("value"))
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        path = f.name
    text_df.write.mode("overwrite").text(path)
    from pathlib import Path
    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 5


def test_write_save_parquet(simple_df):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    simple_df.write.format("parquet").mode("overwrite").save(path)
    result = pl.read_parquet(path)
    assert result.shape == simple_df.shape


def test_write_error_mode_raises(simple_df):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    simple_df.write.mode("overwrite").parquet(path)  # first write OK
    with pytest.raises(RuntimeError, match="already exists"):
        simple_df.write.parquet(path)  # default mode="error" → raises


def test_write_ignore_mode_skips(simple_df):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    simple_df.write.mode("overwrite").parquet(path)  # first write OK
    # write different data with ignore mode — file should be unchanged
    other = simple_df.head(1)
    other.write.mode("ignore").parquet(path)
    result = pl.read_parquet(path)
    assert result.shape == simple_df.shape  # original 5 rows still there


def test_write_saveAsTable(simple_df):
    import src.sparkpolars.polyspark.sql.dataframe as _mod
    simple_df.write.saveAsTable("writer_table")
    assert "writer_table" in _mod._local_sql_ctx.tables()
    _mod._local_sql_ctx.unregister("writer_table")


def test_write_save_no_format_raises(simple_df):
    with pytest.raises(ValueError, match="No format"):
        simple_df.write.save("/tmp/nowhere")


def test_write_parquet_lf(simple_lf):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    simple_lf.write.mode("overwrite").parquet(path)
    result = pl.read_parquet(path)
    assert result.shape == (5, 3)
