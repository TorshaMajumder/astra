#
# Import all dependencies
#
import os
import time
import logging
import argparse
import warnings
import traceback
import pandas as pd
from lsdb import read_hats
from dask.distributed import Client
from astra.bands.bands import ztf_band
from astra.src.dataset import create_dataset

warnings.filterwarnings(action="ignore") 
logging.getLogger('tensorflow').setLevel(logging.ERROR)  


def main():
    #
    # Argument Parser
    #
    parser = argparse.ArgumentParser(prog='astra-data',
                                    description="Generate data as tensorflow records")
    
    parser.add_argument('--dest', default="../dataset/cepheids/", type=str,
                    help='Directory path for the files to be stored.')
    parser.add_argument('--path_to_buff', default="../dataset/cepheids/hats/zubercal_vcep", type=str,
                    help='Directory path for the files to be read.')
    parser.add_argument('--label', default="", type=str,
                    help="Label associated with the catalog. Provide this value only if the Class column is missing in the dataframe.")
    parser.add_argument('--seed', default=42, type=int,
                    help='Set seed value.') 
    parser.add_argument('--min_detec', default=100, type=int,
                    help='Minimum detections in each light curve.')     
    parser.add_argument('--max_lcs_per_chunk', default=100, type=int,
                    help='Number of lcs to be stored in a tf record chunk.')  
    parser.add_argument('--train_size', default=0.8, type=float,
                    help='Training fraction.') 
    parser.add_argument('--del_label', default=None, type=str, nargs='+', 
                    help='Labels deleted from the dataset. Please use "del_label" or "keep_label", not both.')  
    parser.add_argument('--keep_label', default=None, type=str, nargs='+', 
                    help='Labels added from the dataset. Please use "del_label" or "keep_label", not both.') 

   
    args = parser.parse_args()
    #
    # Read catalog
    #
    read_catalog = read_hats(args.path_to_buff, )

    if not os.path.exists(args.dest):
                os.makedirs(args.dest, exist_ok=True)
    #
    dest_obj = os.path.join(args.dest, "objects")
    os.makedirs(dest_obj, exist_ok=True)
    #
    # Create an empty dataframe with the expected output structure for the 'meta' argument.
    #
    meta_df = pd.DataFrame(columns=["label", "size", "start_index"])
    #
    dest = os.path.join(args.target, "objects")
    os.makedirs(dest, exist_ok=True)

    # Create an empty dataframe with the expected output structure for the 'meta' argument.
    # It should include any new columns you add in your function.
    # meta_df = read_catalog._ddf.head(0)
    meta_df = pd.DataFrame(columns=["label", "size", "start_index"])
    # if "Class" not in meta_df.columns:
    #     meta_df['Class'] = pd.Series(dtype='object') # Or whatever dtype 'Class' will be

    #
    catalog_compute = read_catalog._ddf.map_partitions(create_dataset, 
                                                            target=args.dest,
                                                            target_obj=dest_obj,
                                                            bands=ztf_band,
                                                            label=args.label,
                                                            seed=args.seed,
                                                            min_detec=args.min_detec,
                                                            train_size=args.train_size,
                                                            max_lcs_per_chunk=args.max_lcs_per_chunk,
                                                            del_label = args.del_label,
                                                            keep_label = args.keep_label,
                                                            meta=meta_df)

    with Client(n_workers=3, threads_per_worker=1, memory_limit="65GB") as client:
        catalog_compute.compute()
        client.close()

    

if __name__ == '__main__':
       
    prog = main()
    sys.exit(prog)

