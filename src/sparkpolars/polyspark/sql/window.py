import polars.functions as plf


class Window:

    _partition_by = plf.all()
    _order_by=None
    _sort_by=None

    def partitionBy(self, *cols):
        self._partition_by = cols
        return self
    
    def orderBy(self, *cols):
        self._order_by = cols
        return self

    def rangeBetween(self, start, end):
        return self

    def rowsBetween(self, start, end):
        return self

    def unboundedFollowing(self):
        return self

    def unboundedPreceding(self):
        return self

    def currentRow(self):
        return self

    def __call__(self, *args, **kwds):
        return self._partition_by

    def __str__(self) -> str:
        return str(getattr(self, "_partition_by", []))

    def __iter__(self):
        return iter({
            "partition_by": self._partition_by,
            "order_by": self._order_by,
            "sort_by": self._sort_by
        })

    @property
    def cols(self):
        return getattr(self, "_partition_by", [])

    # self = self.sort(
    #     by=[*partition_cols, *col_names],
    #     descending=[True] * len(partition_cols) + ordering,
    #     nulls_last=[True] * len(partition_cols) + null_conditions,
    #     maintain_order=True,
    # ).with_columns(
    #     struct(*partition_cols).rank("ordinal").over(*partition_cols).alias("_keep_"),
    # )

    # if rank_method == "dense":
    #     self = self.with_columns(
    #         col("_keep_").min().over(*[*partition_cols, *col_names]).alias("_keep_"),
    #     )
