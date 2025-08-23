# import tensorflow as tf
# import os

# # Optional: Suppress TensorFlow informational messages (but keep warnings/errors)
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# print("TensorFlow Version:", tf.__version__)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# # List all physical devices TensorFlow can see
# physical_devices = tf.config.list_physical_devices()
# print("Available physical devices:", physical_devices)
# print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

# You can also try placing a simple operation and see where it runs
# Turn on device placement logging
# tf.debugging.set_log_device_placement(True)
# try:
#     # Place a simple computation
#     with tf.device('/GPU:0'):
#         a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
#         b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
#         c = tf.matmul(a, b)
#     print("Simple computation placed on GPU (if successful)")
# except RuntimeError as e:
#     print(f"Could not place computation on GPU: {e}")

# # Turn off device placement logging if you turned it on
# tf.debugging.set_log_device_placement(False)

# import tensorflow as tf
# import os

# # --- Configuration for CPU Parallelism ---

# # Set the number of threads for intra-operation parallelism
# # This controls parallelism within a single op (e.g., matrix multiplication)
# num_intra_threads = 15
# tf.config.threading.set_intra_op_parallelism_threads(num_intra_threads)

# Set the number of threads for inter-operation parallelism
# This controls parallelism across independent operations
# Setting it to 0 lets the system choose an appropriate default (often good)
# Setting it to 1 forces sequential execution between ops
# You can also set it to a specific number, e.g., 2 or 4, or even 15
# Start with 0 or a small number like 2. Setting both high can sometimes cause contention.
# num_inter_threads = 0 # Let TF decide, or try a small number like 2
# tf.config.threading.set_inter_op_parallelism_threads(num_inter_threads)

# # --- Optional: Verify the settings ---
# print(f"Intra-op parallelism threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
# print(f"Inter-op parallelism threads: {tf.config.threading.get_inter_op_parallelism_threads()}")

# # --- Optional: Get CPU core info (for context) ---
# try:
#     # psutil is great for this, install if needed: pip install psutil
#     import psutil
#     physical_cores = psutil.cpu_count(logical=False)
#     logical_cores = psutil.cpu_count(logical=True)
#     print(f"Available CPU cores: Physical={physical_cores}, Logical={logical_cores}")
# except ImportError:
#     print("Install 'psutil' (pip install psutil) to see detailed CPU core info.")
#     # os.cpu_count() usually gives logical cores
#     logical_cores = os.cpu_count()
#     print(f"Available CPU logical cores (estimated by os.cpu_count): {logical_cores}")


import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# --- Configuration ---
# Path to the specific run directory containing the event file(s)
# Example: run_log_dir = "/media3/majumder/CL_results/run_20231028_153000/"
run_log_dir = "/media3/majumder/contrastive_loss_res/run_20250823_155327/" # <--- SET THIS PATH



# Tags you logged for epoch losses (adjust if you used different names)
train_loss_tag = 'loss/epoch_train'
val_loss_tag = 'loss/epoch_val'
# ---

# --- Load Data using EventAccumulator ---
print(f"Loading events from: {run_log_dir}")


try:
    # Initialize EventAccumulator
    ea = event_accumulator.EventAccumulator(
        run_log_dir,
        size_guidance={
             event_accumulator.SCALARS: 0, # Keep checking scalars just in case
             event_accumulator.TENSORS: 0  # Load all tensors
        }
    )

    # Load the events
    ea.Reload()

    # --- Check Available Tags ---
    available_tags = ea.Tags()
    print("Available tag categories:", list(available_tags.keys()))
    if event_accumulator.TENSORS in available_tags:
         print("Available tensor tags:", available_tags[event_accumulator.TENSORS])
    else:
         print("No 'tensors' data found in event file.")
         # If you expected scalars, print that too:
         if event_accumulator.SCALARS in available_tags:
             print("Available scalar tags:", available_tags[event_accumulator.SCALARS])
         else:
             print("No 'scalars' data found either.")
         exit() # Exit if no tensor data is found where expected

    # --- Function to extract data from Tensor events ---
    def extract_tensor_data(event_acc, tag_name):
        data = []
        try:
            events = event_acc.Tensors(tag_name)
            for event in events:
                # Convert the tensor proto to a numpy array
                value_array = tf.make_ndarray(event.tensor_proto)
                # Assuming it was logged as a scalar, it should be a 0-D array
                # Extract the scalar value using .item()
                scalar_value = value_array.item()
                data.append((event.step, scalar_value))
        except KeyError:
            print(f"Warning: Tag '{tag_name}' not found in tensors.")
        except Exception as e:
            print(f"Error processing tag '{tag_name}': {e}")
        return data
    # # ---
    print(f"\nAttempting to read the parameters from 'tensors' category...")
    hparams = extract_tensor_data(ea, "hyperparameters")
    if hparams:
        print(f"Found hyperparameters from tensors.")
        for step, param in hparams:
            print(f"Step {step}: {param}")


    # # --- Extract Data using the function ---
    # print(f"\nAttempting to read tags from 'tensors' category...")
    # train_loss_data = extract_tensor_data(ea, train_loss_tag)
    # if train_loss_data:
    #      print(f"Found {len(train_loss_data)} training loss points (tag: {train_loss_tag}) from tensors.")

    # val_loss_data = extract_tensor_data(ea, val_loss_tag)
    # if val_loss_data:
    #      print(f"Found {len(val_loss_data)} validation loss points (tag: {val_loss_tag}) from tensors.")


    # # --- Display or Process Data ---
    # print("\n--- Epoch Losses ---")

    # # Option 1: Simple Print
    # # max_epochs = max(len(train_loss_data), len(val_loss_data))
    # # if max_epochs == 0:
    # #     print("No epoch loss data found to display.")
    # # else:
    # #     for i in range(max_epochs):
    # #         # Assume step directly corresponds to epoch index (0, 1, 2...)
    # #         epoch_idx = i # Use index as epoch number - 1 if step=epoch
    # #         train_l_info = next((item for item in train_loss_data if item[0] == epoch_idx), None)
    # #         val_l_info = next((item for item in val_loss_data if item[0] == epoch_idx), None)

    # #         train_l = train_l_info[1] if train_l_info is not None else 'N/A'
    # #         val_l = val_l_info[1] if val_l_info is not None else 'N/A'

    # #         train_l_str = f"{train_l:.5f}" if isinstance(train_l, (float, int)) else str(train_l)
    # #         val_l_str = f"{val_l:.5f}" if isinstance(val_l, (float, int)) else str(val_l)

    # #         print(f"Epoch {epoch_idx+1}: Train Loss = {train_l_str}, Val Loss = {val_l_str}")


    # # Option 2: Create a Pandas DataFrame
    # if train_loss_data or val_loss_data:
    #     # Get unique steps (epochs) from both lists
    #     all_steps = sorted(list(set([e[0] for e in train_loss_data] + [e[0] for e in val_loss_data])))

    #     train_losses_dict = {e[0]: e[1] for e in train_loss_data}
    #     val_losses_dict = {e[0]: e[1] for e in val_loss_data}

    #     df_data = {
    #         'epoch': [s + 1 for s in all_steps], # Convert step (0-based) to epoch number (1-based)
    #         'train_loss': [train_losses_dict.get(s, None) for s in all_steps],
    #         'val_loss': [val_losses_dict.get(s, None) for s in all_steps]
    #     }
    #     loss_df = pd.DataFrame(df_data).set_index('epoch')
    #     loss_df.to_csv(os.path.join(run_log_dir, 'epoch_losses.csv'), index=True)
    #     print("\n--- DataFrame Summary ---")
    #     # print(loss_df)
    #     plt.plot(loss_df.index, loss_df['train_loss'], label='Train Loss', color='blue')
    #     plt.plot(loss_df.index, loss_df['val_loss'], label='Validation Loss', color='orange')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Epoch Losses')
    #     plt.legend()
    #     # plt.grid()
    #     # plt.savefig(os.path.join(run_log_dir, 'epoch_losses.png'))
    #     plt.show()


except FileNotFoundError:
    print(f"Error: Log directory not found: {run_log_dir}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()