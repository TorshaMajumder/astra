import os
import glob
import random
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict


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
                               max_lcs_per_chunk=200,
                               shuffle_buffer_size=2000):
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
    
    print(f"\n\nStarting resampling process...")
    print(f"\nSource: {source_dir}")
    print(f"\nTarget: {target_dir}")

    final_counts = []

    for class_name, target_samples in sampling_config.items():
        print(f"\nProcessing class: '{class_name}' | Target samples: {target_samples}")
        #
        # Discover all record files for the current class across all splits and partitions
        # The pattern looks for any .record file inside a directory with the class_name
        #
        search_pattern = os.path.join(source_dir, '*', '*', class_name, '*.record')
        filenames = glob.glob(search_pattern)
        #
        if not filenames:
            print(f"\n  WARNING: No record files found for class '{class_name}'. Skipping.")
            continue
        #    
        print(f"\n  Found {len(filenames)} source record files.")
        #
        # Shuffle the list of files to ensure we don't read partitions in order
        random.shuffle(filenames)
        #
        # Create a unified tf.data.Dataset
        #
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
        #
        # Resample the dataset
        # This single chain handles both oversampling (repeat) and undersampling (take)
        # We shuffle first to ensure randomness before taking a subset
        #
        dataset = dataset.shuffle(shuffle_buffer_size)
        final_dataset = dataset.repeat().take(target_samples)
        #
        # Re-split the newly sampled dataset
        #
        train_size = int(split_ratios[0] * target_samples)
        val_size = int(split_ratios[1] * target_samples)
        test_size = int(1.0 * target_samples) # take all samples
        #
        test_ds = final_dataset.take(test_size)
        train_ds = final_dataset.take(train_size)
        val_ds = final_dataset.skip(train_size).take(val_size)
        #
        print(f"\n  New splits: Train={train_size}, Val={val_size}, Test={test_size}")
        #
        # Write the new datasets to the target directory
        #
        print("\n  Writing new chunk files...")
        #
        # Write train split
        #
        train_output_dir = os.path.join(target_dir, 'train', class_name)
        write_dataset_to_chunks(train_ds, train_output_dir, max_lcs_per_chunk)
        #
        # Write validation split
        #
        val_output_dir = os.path.join(target_dir, 'val', class_name)
        write_dataset_to_chunks(val_ds, val_output_dir, max_lcs_per_chunk)
        #
        # Write test split
        #
        test_output_dir = os.path.join(target_dir, 'test', class_name)
        write_dataset_to_chunks(test_ds, test_output_dir, max_lcs_per_chunk)
        #
        final_counts.append({'label': class_name, 'size': target_samples})
    #    
    # Create the final summary CSV
    #
    print("\n\nCreating final summary CSV...")
    summary_df = pd.DataFrame(final_counts)
    summary_dir = os.path.join(target_dir, 'objects')
    os.makedirs(summary_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(summary_dir, 'class_dist.csv'), index=False)
    #
    print("\nResampling process completed successfully!")


if __name__ == '__main__':
    # 
    # Define the source directory of your large, partitioned dataset
    #
    SOURCE_DATASET_DIR = '/home/torsha/workplace/dataset/multi-class'
    #
    # Define the target directory where the new, smaller dataset will be saved
    #
    TARGET_DATASET_DIR = '/home/torsha/workplace/dataset/resampled_multi-class'
    #
    # Define your sampling configuration
    #    
    SAMPLING_CONFIG = {
        'BCEP': 407,
        'CEP': 3449,
        'DSCT|GDOR|SXPHE': 10000,
        'RR': 12000
    }
    #
    # Define custom split ratios
    #
    SPLIT_RATIOS = (0.8, 0.2)  # 80% train, 20% validation, 100% test
    #
    # Define your chunk size
    #
    MAX_LCS_PER_CHUNK = 200
    #
    # Run the main function
    #
    resample_and_split_dataset(
        source_dir=SOURCE_DATASET_DIR,
        target_dir=TARGET_DATASET_DIR,
        sampling_config=SAMPLING_CONFIG,
        split_ratios=SPLIT_RATIOS,
        max_lcs_per_chunk=MAX_LCS_PER_CHUNK)