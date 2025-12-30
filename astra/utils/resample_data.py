# ===================================================================================
# Import all dependencies
# ===================================================================================
import os
import glob
import random
import pandas as pd
import tensorflow as tf
from collections import defaultdict
# ==========================================================
os.system('clear')
# ===========================================================

def _parse_id_from_record(serialized_example):
    """
     Parse the 'id' from a tf.train.SequenceExample.
     NOTE: dtype of 'id' is tf.int64
    """
    context_features = {'id': tf.io.FixedLenFeature([], dtype=tf.int64)}
    context, _ = tf.io.parse_single_sequence_example(serialized_example,
                                                    context_features=context_features
                                                    )
    return context['id']

def _parse_id_and_record(serialized_example):
    """
     Parse the 'id' from a tf.train.SequenceExample
     and also returns the original serialized record.
     NOTE: dtype of 'id' is tf.int64
    """
    context_features = {'id': tf.io.FixedLenFeature([], dtype=tf.int64)}
    context, _ = tf.io.parse_single_sequence_example(serialized_example, 
                                                     context_features=context_features
                                                    )
    return context['id'], serialized_example 

def diagnose_corrupted_files(source_dir, class_name=None):
    """
    Checks all TFRecord files for a given class/ all classes to find corrupted ones (if any).

    Parameters:
    -----------------------------------------------------------------------------------------
        source_dir (str): path to the dataset.
        class_name (str): specific class folder to diagnose.
    
    Returns:
    -----------------------------------------------------------------------------------------
        list: A list of paths to the corrupted files.
    """
    #
    # Check if the diagnosis is for a specific class
    # Adjust the search pattern. 
    # Default is - source_dir + "/partition_0/CEP/chunk_0_0.record"
    #
    if class_name:
        print(f"\n--- Diagnosing files for class: {class_name} ---")
        search_pattern = os.path.join(source_dir, '*', class_name, '*.record')
    else:
        search_pattern = os.path.join(source_dir, '*', '*', '*.record')
    # -------------------------------------------------------------------------
    filenames = glob.glob(search_pattern)
    # -------------------------------------------------------------------------
    if not filenames:
        print(f"\n--- No files found to diagnose!\n    Please check the file path: {search_pattern}.\n")
        return [] # return empty list
    # -------------------------------------------------------------------------
    corrupted_files = list()
    # Search for corrupted files
    for i, filename in enumerate(filenames):
        dataset = tf.data.TFRecordDataset(filename)
        try:
            for record in dataset:
                pass # check if it's a successful read
        except tf.errors.DataLossError:
            corrupted_files.append(filename)
    # --------------------------------------------------------------------------
    if not corrupted_files:
        print(f"\nDiagnosis complete. No corrupted files found.\n")
    else:
        print(f"\nDiagnosis complete. Found {len(corrupted_files)} corrupted file(s):")
        for f in corrupted_files:
            print(f"  - {f}")
    print("\n")
    # --------------------------------------------------------------------------       
    return corrupted_files


def analyze_duplicates(source_dir):
    """
    Analyzes all classes in a partitioned TFRecord dataset, aggregating the results
    to provide a single summary row per class.

    Parameters:
    -------------------------------------------------------------------------------
        source_dir (str): The root directory of the dataset.
    
    Returns:
    -------------------------------------------------------------------------------
        pd.DataFrame: A DataFrame summarizing the aggregated counts for each class.
    """
    print("\n--- Starting Aggregated Duplicate Analysis ---\n")
    #
    # This dictionary will store the running totals for each class.
    # The key is the class name (e.g., 'RR'), and the value holds the total records
    # and a set of all unique IDs found so far for that class.
    aggregated_stats = defaultdict(lambda: {'total_records': 0, 'unique_ids': set()})
    #
    # Adjust the search pattern. 
    # Default is - source_dir + "/partition_0/CEP/chunk_0_0.record"
    #
    search_pattern = os.path.join(source_dir, '*', '*')
    class_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    print(f"\nFound {len(class_dirs)} tf.record files to process...\n")
    if not class_dirs:
        return
    # -----------------------------------------------------------------------------
    for class_path in class_dirs:
        class_name = os.path.basename(class_path)
        record_files = glob.glob(os.path.join(class_path, '*.record'))
        if not record_files:
            continue
        dataset = tf.data.TFRecordDataset(record_files)
        id_dataset = dataset.map(_parse_id_from_record, num_parallel_calls=tf.data.AUTOTUNE)
        # Get all IDs from the current set of files
        ids_in_chunk = [sample_id.numpy() for sample_id in id_dataset]
        # Add the number of records in this chunk to the total for this class
        aggregated_stats[class_name]['total_records'] += len(ids_in_chunk)
        # Update the set of unique IDs with the IDs found in this chunk.
        # The set automatically handles deduplication across all chunks.
        aggregated_stats[class_name]['unique_ids'].update(ids_in_chunk)

    # ------------------------------------------------------------------------------
    print("\n--- Aggregating Final Results ---\n")
    final_results = list()
    sorted_class_names = sorted(aggregated_stats.keys())

    for class_name in sorted_class_names:
        stats = aggregated_stats[class_name]
        total_count = stats['total_records']
        unique_count = len(stats['unique_ids'])
        # --------------------------------------------------------------------------
        final_results.append({
                                'class_name': class_name,
                                'total_records': total_count,
                                'unique_records': unique_count,
                                'duplicate_records': total_count - unique_count
                            })
        # --------------------------------------------------------------------------
    print("\n--- Analysis Complete ---\n")
    res = pd.DataFrame(final_results)
    res.to_csv(os.path.join(os.path.dirname(source_dir), "data_stats.csv"), sep="\t")
    print(res)
    return res


def write_dataset_to_chunks(dataset, output_dir, max_lcs_per_chunk=200):
    """
    Writes a tf.data.Dataset to chunked TFRecord files.

    Parameters:
    -------------------------------------------------------------------------------------------
        dataset (tf.data.Dataset): The dataset to write. Must contain serialized records.
        output_dir (str): The directory to save the chunk files in (e.g., '.../train/ACEP').
        max_lcs_per_chunk (int): The maximum number of records per chunk file.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    #
    # Batch the dataset into groups of `max_lcs_per_chunk`
    # The `drop_remainder=False` ensures the last smaller batch is also included
    #
    batched_dataset = dataset.batch(max_lcs_per_chunk, drop_remainder=False)
    #
    # Iterate over each batch and write it to a new file
    #
    for i, batch in enumerate(batched_dataset):
        chunk_path = os.path.join(output_dir, f"chunk_{i}.record")
        with tf.io.TFRecordWriter(chunk_path) as writer:
            for serialized_record in batch:
                writer.write(serialized_record.numpy())





def resample_and_split_dataset(source_dir, 
                               target_dir, 
                               sampling_config, 
                               split_ratios=(0.8, 0.2), 
                               max_lcs_per_chunk=200):
    """
    Resamples a TFRecord dataset based on a configuration, re-splits it, and saves it.

    Parameters:
    ------------------------------------------------------------------------------------------------
        source_dir (str): Path to the root of the original dataset (e.g., './dataset').
        target_dir (str): Path to save the new, resampled dataset.
        sampling_config (dict): A dictionary mapping class names to the desired number of samples.
                                e.g., {'CEP': 20000, 'AGN': 6000}
        split_ratios (tuple): A tuple for (train, validation) split fractions, 
                                test dataset will contain all the samples.
        max_lcs_per_chunk (int): Number of light curves per output TFRecord file.
        shuffle_buffer_size (int): Size of the shuffle buffer for tf.data.Dataset.
    """
    
    print("\n---------- Starting Resampling Process -----------")
    # ---------------------------------------------------------------------------------------
    final_counts = list()
    # ---------------------------------------------------------------------------------------
    for class_name, target_samples in sampling_config.items():
        print(f"\nProcessing class: '{class_name}' | Target samples: {target_samples}")
        #
        search_pattern = os.path.join(source_dir, '*', class_name, '*.record')
        filenames = glob.glob(search_pattern)
        if not filenames:
            print(f"\n  WARNING: No record files found for class '{class_name}'. Skipping...\n")
            continue
        # ---------------------------- Load and Deduplicate --------------------------------
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
        parsed_ds = dataset.map(_parse_id_and_record, num_parallel_calls=tf.data.AUTOTUNE)
        #
        # Use a dictionary to automatically handle deduplication based on unique IDs
        #
        unique_records_dict = {}
        for id_tensor, record_tensor in parsed_ds:
            unique_records_dict[id_tensor.numpy()] = record_tensor
        # ----------------------------------------------------------------------------------   
        unique_records = list(unique_records_dict.values())
        num_unique_available = len(unique_records)
        print(f"\n  Found {num_unique_available} unique records for {class_name}.\n")
        # ----------------------- Handle cases with insufficient data ----------------------
        if num_unique_available < 3: continue
           
        elif num_unique_available < target_samples:
            print(f"\n  WARNING: Requested {target_samples} samples, but only {num_unique_available} unique samples are available.")
            print(f"\n  Using all {num_unique_available} available unique samples.")
            samples_to_take = num_unique_available
        else:
            samples_to_take = target_samples   
        # ------------------------------- Shuffle and Sample -------------------------------
        random.shuffle(unique_records)
        final_sampled_list = unique_records[:samples_to_take]
        # --------------------------- Create the final, clean dataset ----------------------
        final_dataset = tf.data.Dataset.from_tensor_slices(final_sampled_list)
        # ------------------ Data splitting and writing as tf.records file -----------------
        train_size = int(split_ratios[0] * samples_to_take)
        val_size = samples_to_take - train_size # Ensure val takes the rest
        # The dataset is already shuffled, so just take splits directly
        train_ds = final_dataset.take(train_size)
        val_ds = final_dataset.skip(train_size)
        # Test set will contain ALL unique sampled data
        test_ds = final_dataset
        print(f"\n  New splits: Train={train_size}, Val={val_size}, Test={samples_to_take}\n")
        # ----------------------------------------------------------------------------------
        print("\n--- Writing new chunk files...\n")
        #
        train_output_dir = os.path.join(target_dir, 'train', class_name)
        write_dataset_to_chunks(train_ds, train_output_dir, max_lcs_per_chunk)
        #
        val_output_dir = os.path.join(target_dir, 'val', class_name)
        write_dataset_to_chunks(val_ds, val_output_dir, max_lcs_per_chunk)
        #
        test_output_dir = os.path.join(target_dir, 'test', class_name)
        write_dataset_to_chunks(test_ds, test_output_dir, max_lcs_per_chunk)
        #
        final_counts.append({'label': class_name, 'size': samples_to_take})
        
    # -------------------------------------------------------------------------------------
    print("\nCreating final summary CSV...")
    summary_df = pd.DataFrame(final_counts)
    summary_dir = os.path.join(target_dir, 'objects')
    os.makedirs(summary_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(summary_dir, 'class_dist.csv'), sep='\t')
    # -------------------------------------------------------------------------------------
    print("\nResampling process completed successfully!")
    # =====================================================================================


def main(mode=None, source_dir=None, target_dir=None, split_ratios=None, max_lcs_per_chunk=None, config=None):
    try:
        if mode not in ["duplicate_analysis", "resampling"]:
            raise ValueError(f"\nValueError: Please provide mode as 'duplicate_analysis', 'resampling'. Got mode = {mode}.\n")
    
        if mode == "duplicate_analysis":
            bad_files = diagnose_corrupted_files(source_dir)
            if bad_files:
                print("\nPlease regenerate or delete the corrupted files listed above and then re-run the script.\n")
            else:
                print("\n--- Proceeding with duplicate analysis...\n")
                _ = analyze_duplicates(source_dir)
        elif mode == "resampling":
            assert(source_dir and target_dir and split_ratios and max_lcs_per_chunk and config), "AssertionError: Please provide all the method parameters." 

            resample_and_split_dataset(
                                        source_dir=source_dir,
                                        target_dir=target_dir,
                                        sampling_config=config,
                                        split_ratios=split_ratios,
                                        max_lcs_per_chunk=max_lcs_per_chunk
                                    )
    except Exception as e:
        print(f"\n{e}")
        return



if __name__ == '__main__':
    #
    # Pass "mode" as "duplicate_analysis" or "resampling"
    #
    mode = "duplicate_analysis"
    #
    # Define parameters for "duplicate_analysis" & "resampling"
    # 
    source_dir = '/home/nvidia/workplace/dataset/training_data/val/'
    target_dir = '/home/torsha/workplace/dataset/test/'
    split_ratios = (0.8, 0.2)  # 80% train, 20% validation, 100% test
    max_lcs_per_chunk = 200   
    config = {
                'BCEP': 500,
                'CEP': 5000,
                'DSCT|GDOR|SXPHE': 11000,
                'RR': 13000,
                'ACYG': 5,
                'RCB': 15,
                'SDB': 10,
                'SPB': 2,
                'SYST': 10
            }
    main(mode, source_dir, target_dir, split_ratios, max_lcs_per_chunk, config)
    