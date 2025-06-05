import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.utils import load as load_model_and_tokenizer
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
import numpy as np
from typing import List, Dict, Any
import pickle

DEBUG = False

config = configparser.ConfigParser()
config.read('config.ini')


def load_model(model_name: str, device: str = 'mps'):
    """
    Load MLX model and tokenizer from weights directory.

    Args:
        model_name: Name of the model configuration
        device: Device to use ('mps' for Apple Silicon, 'cpu' for fallback)

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model {model_name}...")
    weights_directory = config[model_name]['weights_directory']

    try:
        # Load model and tokenizer using mlx_lm
        model, tokenizer = load_model_and_tokenizer(weights_directory)
        print(f"Model loaded successfully on Apple Silicon")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def load_statements(dataset_name: str) -> List[str]:
    """
    Load statements from csv file, return list of strings.

    Args:
        dataset_name: Name of the dataset file (without .csv extension)

    Returns:
        List of statement strings
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements


def tokenize_statements(statements: List[str], tokenizer, max_length: int = 512) -> mx.array:
    """
    Tokenize a batch of statements using MLX.

    Args:
        statements: List of input statements
        tokenizer: MLX tokenizer
        max_length: Maximum sequence length

    Returns:
        MLX array of tokenized inputs
    """
    # Tokenize all statements
    tokenized = []
    for statement in statements:
        tokens = tokenizer.encode(statement)
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        tokenized.append(tokens)

    # Pad to same length
    max_len = max(len(tokens) for tokens in tokenized)
    padded_tokens = []

    for tokens in tokenized:
        # Pad with tokenizer pad_token_id or 0
        pad_id = getattr(tokenizer, 'pad_token_id', 0) or 0
        padded = tokens + [pad_id] * (max_len - len(tokens))
        padded_tokens.append(padded)

    return mx.array(padded_tokens)


def get_layer_activations(model, input_ids: mx.array, layers: List[int]) -> Dict[int, mx.array]:
    """
    Extract activations from specified layers using forward hooks.

    Args:
        model: MLX model
        input_ids: Tokenized input tensor
        layers: List of layer indices to extract activations from

    Returns:
        Dictionary mapping layer indices to activation arrays
    """
    activations = {}

    def create_hook(layer_idx):
        def hook(module, args, output):
            # Store the output of this layer
            activations[layer_idx] = output

        return hook

    # Register forward hooks for specified layers
    hooks = []
    for layer_idx in layers:
        if hasattr(model, 'layers') and layer_idx < len(model.layers):
            hook = create_hook(layer_idx)
            handle = model.layers[layer_idx].register_forward_hook(hook)
            hooks.append(handle)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers') and layer_idx < len(model.model.layers):
            hook = create_hook(layer_idx)
            handle = model.model.layers[layer_idx].register_forward_hook(hook)
            hooks.append(handle)

    try:
        # Forward pass
        with mx.no_grad():
            _ = model(input_ids)

        # Extract last token activations for each layer
        final_activations = {}
        for layer_idx in layers:
            if layer_idx in activations:
                # Get last token activation (assuming shape: [batch, seq_len, hidden_dim])
                act = activations[layer_idx]
                if len(act.shape) == 3:  # [batch, seq_len, hidden_dim]
                    final_activations[layer_idx] = act[:, -1, :]  # Last token
                else:
                    final_activations[layer_idx] = act

    finally:
        # Remove hooks
        for handle in hooks:
            handle.remove()

    return final_activations


def get_acts_mlx_fallback(statements: List[str], model, tokenizer, layers: List[int]) -> Dict[int, mx.array]:
    """
    Fallback method to get activations by running inference and accessing intermediate states.
    This approach manually extracts activations during forward pass.
    """
    # Tokenize input
    input_ids = tokenize_statements(statements, tokenizer)

    activations = {}

    # We'll need to modify this based on the specific MLX model architecture
    # For now, providing a template that can be adapted
    with mx.no_grad():
        # This is a simplified approach - you may need to adapt based on your specific model
        hidden_states = model.embed_tokens(input_ids) if hasattr(model, 'embed_tokens') else input_ids

        # Iterate through layers and collect activations
        for i, layer in enumerate(model.layers if hasattr(model, 'layers') else model.model.layers):
            hidden_states = layer(hidden_states)

            if i in layers:
                # Extract last token activation
                if len(hidden_states.shape) == 3:  # [batch, seq_len, hidden_dim]
                    activations[i] = hidden_states[:, -1, :]
                else:
                    activations[i] = hidden_states

    return activations


def get_acts(statements: List[str], model, tokenizer, layers: List[int]) -> Dict[int, mx.array]:
    """
    Get given layer activations for the statements using MLX.

    Args:
        statements: List of input statements
        model: MLX model
        tokenizer: MLX tokenizer
        layers: List of layer indices to extract from

    Returns:
        Dictionary of layer activations
    """
    try:
        # Tokenize statements
        input_ids = tokenize_statements(statements, tokenizer)

        # Try the hook-based approach first
        acts = get_layer_activations(model, input_ids, layers)

        # If that doesn't work, fall back to manual extraction
        if not acts:
            acts = get_acts_mlx_fallback(statements, model, tokenizer, layers)

        return acts

    except Exception as e:
        print(f"Error extracting activations: {e}")
        # Try fallback approach
        return get_acts_mlx_fallback(statements, model, tokenizer, layers)


def save_activations_mlx(activations: Dict[int, mx.array], save_path: str, layer: int, idx: int):
    """
    Save MLX arrays to disk. MLX doesn't have a direct equivalent to torch.save,
    so we convert to numpy and save with pickle or use MLX's built-in save functions.
    """
    try:
        # Convert MLX array to numpy and save
        numpy_act = np.array(activations[layer])
        np.save(f"{save_path}/layer_{layer}_{idx}.npy", numpy_act)
    except Exception as e:
        print(f"Error saving activations for layer {layer}: {e}")
        # Fallback: save as pickle
        with open(f"{save_path}/layer_{layer}_{idx}.pkl", 'wb') as f:
            pickle.dump(activations[layer], f)


if __name__ == "__main__":
    """
    Read statements from dataset, record activations in given layers, and save to specified files.
    Optimized for Apple Silicon using MLX.
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements using MLX on Apple Silicon")
    parser.add_argument("--model", default="llama-13b",
                        help="Name of the model configuration to use")
    parser.add_argument("--layers", nargs='+', type=int,
                        help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="acts",
                        help="Directory to save activations to")
    parser.add_argument("--noperiod", action="store_true", default=False,
                        help="Set flag if you don't want to add a period to the end of each statement")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"],
                        help="Device to use: 'mps' for Apple Silicon, 'cpu' for fallback")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="Batch size for processing statements")

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args.model, args.device)

    # Process each dataset
    for dataset in args.datasets:
        print(f"Processing dataset: {dataset}")
        statements = load_statements(dataset)

        if args.noperiod:
            statements = [statement[:-1] if statement.endswith('.') else statement for statement in statements]

        layers = args.layers
        if layers == [-1]:
            # Get all layers - this needs to be adapted based on your model structure
            if hasattr(model, 'layers'):
                layers = list(range(len(model.layers)))
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = list(range(len(model.model.layers)))
            else:
                print("Warning: Could not determine number of layers. Using default range.")
                layers = list(range(32))  # Default assumption

        # Create save directory
        save_dir = os.path.join(args.output_dir, args.model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if args.noperiod:
            save_dir = os.path.join(save_dir, "noperiod")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        save_dir = os.path.join(save_dir, dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Process in batches
        batch_size = args.batch_size
        for idx in tqdm(range(0, len(statements), batch_size), desc=f"Processing {dataset}"):
            batch_statements = statements[idx:idx + batch_size]

            try:
                # Get activations for this batch
                acts = get_acts(batch_statements, model, tokenizer, layers)

                # Save activations for each layer
                for layer in layers:
                    if layer in acts:
                        save_activations_mlx(acts, save_dir, layer, idx)
                    else:
                        print(f"Warning: No activations found for layer {layer}")

            except Exception as e:
                print(f"Error processing batch starting at index {idx}: {e}")
                continue

    print("Activation extraction completed!")