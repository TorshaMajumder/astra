#
# Import all dependencies
#
from lsdb import read_hats
from dask.distributed import Client
from nested_pandas import NestedDtype


################# FUNCTION 1:  #########################

# if __name__ == "__main__":
#     LC_COLUMN = "lc"

#     catalog = read_hats(
#         '/media3/majumder/dataset/multi-class/hats/zubercal_vclassre',    
#     ).map_partitions(
#         lambda df: df.assign(
#             **{LC_COLUMN: df[LC_COLUMN].astype(NestedDtype.from_pandas_arrow_dtype(df.dtypes[LC_COLUMN]))},
#         )
#     )

#     with Client(n_workers=6, threads_per_worker=1, memory_limit="10GB") as client:
#         # display(client)
#         df = catalog.compute()
#         print(df.head(n=1))
#         client.close()



"""
OUTPUT:

2025-10-02 15:58:09,170 - distributed.worker.memory - WARNING - Unmanaged memory use is high. 
This may indicate a memory leak or the memory may not be released to the OS; 
see https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os 
for more information. -- Unmanaged memory: 7.80 GiB -- Worker memory limit: 9.31 GiB 

------> MEMORY LEAKAGE?

"""



############ FUNCTION 2: ##########################


# if __name__ == "__main__":
#     LC_COLUMN = "lc"

#     catalog = read_hats(
#         '/media3/majumder/dataset/multi-class/hats/zubercal_vclassre',    
#     )._ddf.map_partitions(
#         lambda df: df.assign(
#             **{LC_COLUMN: df[LC_COLUMN].astype(NestedDtype.from_pandas_arrow_dtype(df.dtypes[LC_COLUMN]))},
#         )
#     )

#     with Client(n_workers=6, threads_per_worker=1, memory_limit="10GB") as client:
#         # display(client)
#         df = catalog.compute()
#         print(df.head(n=1))
#         client.close()


"""
OUTPUT:

('lambda-8d46e5b1dd66a7ab0b50e461022ba648', 78), ('lambda-8d46e5b1dd66a7ab0b50e461022ba648', 81), ('lambda-8d46e5b1dd66a7ab0b50e461022ba648', 87), ('lambda-8d46e5b1dd66a7ab0b50e461022ba648', 73), ('lambda-8d46e5b1dd66a7ab0b50e461022ba648', 93), ('lambda-8d46e5b1dd66a7ab0b50e461022ba648', 77), ('lambda-8d46e5b1dd66a7ab0b50e461022ba648', 67), ('_to_string_dtype-1e0a68116bb81e69efd9568e0dea0e73', 99)} (stimulus_id='handle-worker-cleanup-1759413833.2913923')
2025-10-02 16:03:53,309 - distributed.nanny - WARNING - Restarting worker
2025-10-02 16:03:56,757 - distributed.worker.memory - WARNING - Unmanaged memory use is high. This may indicate a memory leak or the memory may not be released to the OS; see https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os for more information. -- Unmanaged memory: 6.98 GiB -- Worker memory limit: 9.31 GiB

File "/media3/majumder/astra/venv/lib/python3.12/site-packages/dask/base.py", line 681, in compute
    results = schedule(expr, keys, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media3/majumder/astra/venv/lib/python3.12/site-packages/distributed/client.py", line 2417, in _gather
    raise exception.with_traceback(traceback)
distributed.scheduler.KilledWorker: Attempted to run task ('read_pixel-9a92beda4ada8742b1ee5146a94ec441', 8) on 4 different workers, but all those workers died while running it. The last worker that attempt to run the task was tcp://127.0.0.1:45401. Inspecting worker logs is often a good next step to diagnose what went wrong. For more information see https://distributed.dask.org/en/stable/killed.html.
"""


###### FUNCTION 3: This will consume all VRAM ##################

if __name__ == "__main__":
    LC_COLUMN = "lc"

    catalog = read_hats(
        '/home/torsha/workplace/dataset/multi-class/hats/zubercal_vclassre',    
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