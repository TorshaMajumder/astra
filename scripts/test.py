#
# Import all dependencies
#
from lsdb import read_hats
from dask.distributed import Client
from nested_pandas import NestedDtype



if __name__ == "__main__":
    LC_COLUMN = "lc"

    catalog = read_hats(
        '/media3/majumder/dataset/multi-class/hats/zubercal_vclassre',    
    )._ddf.map_partitions(
        lambda df: df.assign(
            **{LC_COLUMN: df[LC_COLUMN].astype(NestedDtype.from_pandas_arrow_dtype(df.dtypes[LC_COLUMN]))},
        )
    )

    with Client(n_workers=6, threads_per_worker=1, memory_limit="10GB") as client:
        # display(client)
        df = catalog.compute(scheduler='processes')
        print(df.head(n=1))
        client.close()