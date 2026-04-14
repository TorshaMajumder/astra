#!/usr/bin/env python3
"""
Convert an AstraNet Keras model to ONNX format and run a self-test
comparing outputs between the Keras and ONNX models.

Usage:
    python keras_to_onnx.py <run_directory> [--max-len <int>] [--num-gpus <int>]

Arguments:
    run_directory   Path to the run folder containing weights and event file.
                    The ONNX file will be saved to the same directory.
"""

import argparse
import os
import sys
import subprocess
import tempfile

import numpy as np
import tensorflow as tf
import onnxruntime as ort

# ---- Project imports --------------------------------------------------------
# Assumes this script is run from within the astra package environment
from astra.src.transformer import AstraNet
from astra.utils.helper import load_hparams_from_event_file


# =============================================================================
# Helpers
# =============================================================================

def _build_astranet(run_directory: str, build_seq_len: int):
    """Re-create AstraNet, build it, and load weights. Returns (model, dummy_input)."""
    model_params, _, _ = load_hparams_from_event_file(run_directory)
    if model_params is None:
        raise ValueError("Failed to load hyperparameters from the event file.")

    print("Re-creating AstraNet architecture from loaded hyper-parameters...")
    model = AstraNet(
        num_layers=model_params["num_layers"],
        d_model=model_params["d_model"],
        base=model_params["base"],
        num_heads=model_params["num_heads"],
        dff=model_params["dff"],
        rate=model_params["rate"],
        mjd=model_params["mjd"],
        use_drop=model_params["use_drop"],
        use_band_info=model_params["use_band_info"],
        projection_dim=model_params["projection_dim"],
    )

    dummy_input = {
        "input":     tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        "times":     tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        "band_info": tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        "mask":      tf.zeros((1, build_seq_len),    dtype=tf.float32),
    }
    _ = model(dummy_input, training=False)

    path_to_weight = os.path.join(run_directory, "best_contrastive.weights.h5")
    print(f"Loading weights from: {path_to_weight}")
    model.load_weights(path_to_weight)

    return model, dummy_input


def _make_input_layer(build_seq_len: int) -> dict:
    return {
        "input":     tf.keras.Input(shape=(build_seq_len, 1), name="input",     dtype=tf.float32),
        "times":     tf.keras.Input(shape=(build_seq_len, 1), name="times",     dtype=tf.float32),
        "band_info": tf.keras.Input(shape=(build_seq_len, 1), name="band_info", dtype=tf.float32),
        "mask":      tf.keras.Input(shape=(build_seq_len,),   name="mask",      dtype=tf.float32),
    }


def build_export_model(run_directory: str, build_seq_len: int) -> tf.keras.Model:
    """
    Returns a Keras model with a single output: pooled embeddings.
    Suitable for SavedModel export and ONNX conversion.
    """
    model, _ = _build_astranet(run_directory, build_seq_len)
    input_layer = _make_input_layer(build_seq_len)

    embeddings     = model.embedding_layer(input_layer)
    mask_input     = input_layer["mask"]
    encoder_output, _ = model.encoder(embeddings, mask=mask_input)
    pool_mask = tf.keras.layers.Lambda(
        lambda m: tf.logical_not(tf.cast(m, tf.bool))
    )(mask_input)
    pooled_output = model.pooling(encoder_output, mask=pool_mask)

    return tf.keras.Model(inputs=input_layer, outputs=pooled_output, name="ASTRA_Encoder")



def export_to_saved_model(encoder_model: tf.keras.Model, export_dir: str) -> str:
    """Export encoder_model as a SavedModel and return the path."""
    print(f"Exporting SavedModel to: {export_dir}")
    encoder_model.export(export_dir)
    return export_dir


def convert_saved_model_to_onnx(saved_model_dir: str, onnx_output_path: str) -> None:
    """Call tf2onnx via subprocess to convert SavedModel -> ONNX."""
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", onnx_output_path,
    ]
    print("Running tf2onnx:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("tf2onnx stdout:\n", result.stdout)
        print("tf2onnx stderr:\n", result.stderr)
        raise RuntimeError("tf2onnx conversion failed.")
    print("tf2onnx conversion succeeded.")


# =============================================================================
# Self-test
# =============================================================================

def self_test(
    encoder_model: tf.keras.Model,
    onnx_path: str,
    build_seq_len: int,
    batch_size: int = 2,
    atol: float = 1e-4,
) -> None:
    """
    Run the same random inputs through both Keras and ONNX models and compare
    the pooled-embedding outputs.
    """
    print("\n--- Self-test: comparing Keras vs ONNX outputs ---")

    rng = np.random.default_rng(42)
    inputs_np = {
        "input":     rng.standard_normal((batch_size, build_seq_len, 1)).astype(np.float32),
        "times":     rng.standard_normal((batch_size, build_seq_len, 1)).astype(np.float32),
        "band_info": rng.integers(0, 5, (batch_size, build_seq_len, 1)).astype(np.float32),
        "mask":      rng.integers(0, 2, (batch_size, build_seq_len)).astype(np.float32),
    }

    # --- Keras inference ---
    keras_out = encoder_model(inputs_np, training=False)
    # export_model outputs pooled embeddings directly (not a tuple)
    keras_embeddings = keras_out.numpy() if not isinstance(keras_out, (list, tuple)) else keras_out[0].numpy()

    # --- ONNX inference ---
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_names = {inp.name for inp in sess.get_inputs()}
    feed = {k: v for k, v in inputs_np.items() if k in onnx_input_names}
    onnx_out = sess.run(None, feed)
    onnx_embeddings = onnx_out[0]  # first output = pooled embeddings

    # --- Compare ---
    max_diff = np.max(np.abs(keras_embeddings - onnx_embeddings))
    mean_diff = np.mean(np.abs(keras_embeddings - onnx_embeddings))
    print(f"  Max absolute difference  : {max_diff:.6e}")
    print(f"  Mean absolute difference : {mean_diff:.6e}")

    if max_diff <= atol:
        print(f"  PASS  (tolerance={atol})")
    else:
        print(f"  FAIL  (tolerance={atol}) — outputs differ by more than expected!")
        sys.exit(1)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Convert AstraNet Keras model to ONNX.")
    parser.add_argument("run_directory", help="Path to run folder with weights and event file.")
    parser.add_argument(
        "--max-len",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sequence length(s) for each band. "
            "If multiple values are given they are summed (mirrors config['max_len'].values()). "
            "If omitted, 700 is used as default."
        ),
    )
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use (0 = CPU).")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for self-test.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_directory = os.path.abspath(args.run_directory)

    if not os.path.isdir(run_directory):
        print(f"ERROR: run_directory not found: {run_directory}")
        sys.exit(1)

    # ---- GPU / CPU setup ------------------------------------------------
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if args.num_gpus > 0:
        gpus_to_use = gpus[: args.num_gpus]
        tf.config.experimental.set_visible_devices(gpus_to_use, "GPU")
        print(f"Using {len(gpus_to_use)} GPU(s).")
    else:
        print("Running in CPU mode.")

    # ---- Sequence length ------------------------------------------------
    if args.max_len is not None:
        build_seq_len = sum(args.max_len)
    else:
        build_seq_len = 700
    print(f"build_seq_len = {build_seq_len}")

    # ---- Build export model (single output: pooled embeddings) ----------
    print("Building export model (pooled embeddings only)...")
    export_model = build_export_model(run_directory, build_seq_len)
    export_model.summary()

    # ---- Export to SavedModel (temp dir) --------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = os.path.join(tmpdir, "best_contrastive.export")
        export_to_saved_model(export_model, saved_model_dir)

        # ---- Convert to ONNX --------------------------------------------
        onnx_output_path = os.path.join(run_directory, "best_contrastive.onnx")
        convert_saved_model_to_onnx(saved_model_dir, onnx_output_path)

    print(f"\nONNX model saved to: {onnx_output_path}")

    # ---- Self-test -------------------------------------------------------
    self_test(export_model, onnx_output_path, build_seq_len, atol=args.atol)

    print("\nDone.")


if __name__ == "__main__":
    main()
