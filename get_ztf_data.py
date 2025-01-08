
import os
import time
import logging
import pandas as pd
from core.data  import create_dataset, pretraining_records
from core.utils import get_folder_name
import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

if __name__ == '__main__':

    print("\nStarting!")
    start = time.time()
    #
    #
    #
    
    path_to_lc="main-code/data/sim-ZTF-data_cuts-newfeatures/raw_data/ZTF/r-LCs"
    path_to_meta="main-code/data/sim-ZTF-data_cuts-newfeatures/raw_data/ZTF/r-meta"
    path_to_store="main-code/data/sim-ZTF-data_cuts-newfeatures/records/ZTF/r-LCs"
    path_to_meta_df="main-code/data/sim-ZTF-data_cuts-newfeatures/raw_data/ZTF/r-metadata.csv"
    #
    meta_df = pd.read_csv(path_to_meta_df)
    #
    #
    #
    create_dataset(meta_df,
                    source=path_to_lc,
                    meta=path_to_meta,
                    target=path_to_store,
                    n_jobs=15,
                    subsets_frac=(0.80, 0.20),
                    test_subset="",
                max_lcs_per_record=100,)

    end = time.time()
    print("\nTime (mins):", ((end-start)//60))
    print("\nDone!")

    
 
    