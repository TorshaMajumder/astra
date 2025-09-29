#
# Import all dependencies
#
import time
import logging
import argparse
import warnings
import traceback
from lsdb import read_hats
from dask.distributed import Client
from astra.bands.bands import ztf_band
from astra.src.dataset import create_dataset
from astra.utils.helper import generate_data_finetuning

warnings.filterwarnings(action="ignore") 
logging.getLogger('tensorflow').setLevel(logging.ERROR)  


def main():
    #
    # Argument Parser
    #
    parser = argparse.ArgumentParser(prog='astra-data',
                                    description="Generate data as tensorflow records")
    
    parser.add_argument('--target', default="../dataset/cepheids/", type=str,
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
    parser.add_argument('--threshold_finetuning', default=18.0, type=float,
                    help='Magnitude threshold for filter data for finetuning.')      

   
    args = parser.parse_args()
    #
    # Read catalog
    #
    read_catalog = read_hats(args.path_to_buff, )
    #
    #
    #
    catalog_compute = read_catalog._ddf.map_partitions(create_dataset, 
                                                            target=args.target,
                                                            bands=ztf_band,
                                                            label=args.label,
                                                            seed=args.seed,
                                                            min_detec=args.min_detec,
                                                            train_size=args.train_size,
                                                            max_lcs_per_chunk=args.max_lcs_per_chunk)

    with Client() as client:
        catalog_compute.compute(scheduler='processes')

    #
    # Generate data for finetuning from the validation folder
    #
    # generate_data_finetuning(args.target+"val/", args.target, args.max_lcs_per_chunk, args.threshold_finetuning)

    

if __name__ == '__main__':
       
    prog = main()
    sys.exit(prog)

