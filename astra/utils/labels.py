import tensorflow as tf
import os
import glob
import pandas as pd
from collections import defaultdict

def diagnose_corrupted_files(source_dir, class_name=None):
    """
    Checks all TFRecord files for a given class to find corrupted ones.

    Args:
        source_dir (str): The root directory of the dataset.
        class_name (str): The specific class folder to diagnose.
    
    Returns:
        list: A list of paths to the corrupted files.
    """
    print(f"\n--- Diagnosing files for class: {class_name} ---")
    if class_name:
        search_pattern = os.path.join(source_dir, '*', class_name, '*.record')
    else:
        search_pattern = os.path.join(source_dir, '*', '*', '*.record')
    
    filenames = glob.glob(search_pattern)
    
    if not filenames:
        print(f"\n  No files found to diagnose....\nPlease check the file path: {source_dir}")
        return []

    corrupted_files = []
    
    for i, filename in enumerate(filenames):
        dataset = tf.data.TFRecordDataset(filename)
        
        try:
            
            for record in dataset:
                pass # We don't need the data, just to see if it reads successfully
        except tf.errors.DataLossError:
            corrupted_files.append(filename)
    
    if not corrupted_files:
        print(f"\nDiagnosis complete. No corrupted files found for class: '{class_name}'.")
    else:
        print(f"\nDiagnosis complete. Found {len(corrupted_files)} corrupted file(s):")
        for f in corrupted_files:
            print(f"  - {f}")
    print("\n")
            
    return corrupted_files


def _parse_id_from_record(serialized_example):
    """Helper function to parse only the 'id' from a tf.train.SequenceExample."""
    context_features = {'id': tf.io.FixedLenFeature([], dtype=tf.int64)}
    context, _ = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=context_features
    )
    return context['id']

def analyze_duplicates(source_dir):
    """
    Analyzes all classes in a partitioned TFRecord dataset, aggregating the results
    to provide a single summary row per logical class.

    Args:
        source_dir (str): The root directory of the dataset.
    
    Returns:
        pd.DataFrame: A DataFrame summarizing the aggregated counts for each class.
    """
    print("--- Starting Aggregated Duplicate Analysis ---")
    
    # This dictionary will store the running totals for each logical class.
    # The key is the class name (e.g., 'RR'), and the value holds the total records
    # and a set of all unique IDs found so far for that class.
    aggregated_stats = defaultdict(lambda: {'total_records': 0, 'unique_ids': set()})

    # Find all the individual leaf directories that contain .record files
    search_pattern = os.path.join(source_dir, '*', '*')
    class_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    
    print(f"\nFound {len(class_dirs)} tf.record files to process...")
    
    for class_path in class_dirs:
        # Get the logical class name (the final part of the directory path)
        class_name = os.path.basename(class_path)
        
        record_files = glob.glob(os.path.join(class_path, '*.record'))
        if not record_files:
            continue
            
        dataset = tf.data.TFRecordDataset(record_files)
        id_dataset = dataset.map(_parse_id_from_record, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Get all IDs from the current set of files
        ids_in_chunk = [sample_id.numpy() for sample_id in id_dataset]
        
        # --- AGGREGATION STEP ---
        # Add the number of records in this chunk to the total for this class
        aggregated_stats[class_name]['total_records'] += len(ids_in_chunk)
        
        # Update the set of unique IDs with the IDs found in this chunk.
        # The set automatically handles deduplication across all chunks.
        aggregated_stats[class_name]['unique_ids'].update(ids_in_chunk)

    # --- FINAL CALCULATION STEP (after the loop) ---
    print("\n--- Aggregating Final Results ---")
    final_results = []
    
    # Sort by class name for a clean report
    sorted_class_names = sorted(aggregated_stats.keys())

    for class_name in sorted_class_names:
        stats = aggregated_stats[class_name]
        total_count = stats['total_records']
        unique_count = len(stats['unique_ids'])
        
        final_results.append({
            'class_name': class_name,
            'total_records': total_count,
            'unique_records': unique_count,
            'duplicate_records': total_count - unique_count
        })

    print("--- Analysis Complete ---")
    return pd.DataFrame(final_results)


if __name__ == '__main__':

    # --- HOW TO USE THIS ANALYSIS FUNCTION ---
    # Call this function before your resampling to get a report.
    SOURCE_DATASET_DIR = '/home/torsha/workplace/dataset/testing/test/'
    # --- STEP 1: Diagnose the problematic class first ---
    # We know from the error that 'DSCT|GDOR|SXPHE' has a bad file.
    # bad_files = diagnose_corrupted_files(SOURCE_DATASET_DIR, 'DSCT|GDOR|SXPHE')
    
    # You can extend this to check all your classes if you want
    # all_classes = ['BCEP', 'CEP', 'DSCT|GDOR|SXPHE', 'RR', 'ACYG', 'RCB', 'SPB']
    # for cls in all_classes:
    bad_files = diagnose_corrupted_files(SOURCE_DATASET_DIR)

    # --- STEP 2: Take Action ---
    if bad_files:
        print("\nPlease regenerate or delete the corrupted files listed above and then re-run the script.")
    else:
        # --- STEP 3: If no bad files are found, proceed with your analysis ---
        print("\nNo corruption found. Proceeding with duplicate analysis...")
        analysis_df = analyze_duplicates(SOURCE_DATASET_DIR)
        print(analysis_df)