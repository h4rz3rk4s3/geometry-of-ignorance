import torch as t
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForImageTextToText, MistralForCausalLM, AutoModelForSequenceClassification, AutoConfig
import argparse
import pandas as pd
from tqdm import tqdm
import os
from einops import rearrange
from pprint import pprint
import configparser
from nnsight import LanguageModel

DEBUG = False
if DEBUG:
    tracer_kwargs = {'scan': True, 'validate': True}
else:
    tracer_kwargs = {'scan': False, 'validate': False}

config = configparser.ConfigParser()
config.read('config.ini')


def load_model(model_name, device='remote'):
    print(f"Loading model {model_name}...")
    weights_directory = config[model_name]['weights_directory']

    if model_name in ["ModernBERT-base", "ModernBERT-non-knowledge-v1", "deBERTa-v3-base", "bert-base-uncased", "roberta-large", "roberta-toxicity", "roberta-base", "roberta-non-knowledge-v1"]: 
        print("MASKED_LM")
        model = LanguageModel(weights_directory, automodel=AutoModelForMaskedLM, device_map="mps")#dtype=t.bfloat16
        config_ = AutoConfig.from_pretrained(weights_directory)
        n_heads = config_.num_attention_heads
        head_dim = config_.head_dim
    elif model_name  == "Mistral-small":
        model = LanguageModel(weights_directory, automodel=AutoModelForImageTextToText, dtype=t.bfloat16,
                              device_map="mps")
        config_ = AutoConfig.from_pretrained(weights_directory)
        n_heads = config_.num_attention_heads
        head_dim = config_.head_dim
    else:
        print("YEAHHH!")
        model = LanguageModel(weights_directory, torch_dtype=t.bfloat16, device_map="mps")
        config_ = AutoConfig.from_pretrained(weights_directory)
        n_heads = config_.num_attention_heads
        head_dim = config_.head_dim
        print(n_heads)
        print(head_dim)
    return model, n_heads, head_dim


def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv", sep=",")
    statements = dataset['statement'].tolist()
    return statements


def get_acts(statements, model, layers, n_heads, head_dim, remote=True):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    #batch_size = len(statements)
    acts = {layer: {head: [] for head in range(n_heads)} for layer in layers}
    failed_indices = []
        
    for idx, statement in enumerate(statements):
        try:
            with model.trace(statement, remote=False, **tracer_kwargs):
                for layer in layers:
                    attn_output = model.model.layers[layer].self_attn.o_proj.input[0].save()
                    
                    # Use '...' to match any leading dimensions
                    head_outputs = rearrange(attn_output, '... s (h d) -> ... s h d', 
                                            h=n_heads, d=head_dim).save()
                    
                    for head in range(n_heads):
                        acts[layer][head].append(head_outputs[..., head, :].save())
        except Exception as e:
            print(f"Failed on example {idx}: {e}")
            print(f"Statement: {statement[:100] if isinstance(statement, str) else statement}")
            failed_indices.append(idx)
            continue
    
    # Stack all the individual results back into batches
    for layer in layers:
        for head in range(n_heads):
            acts[layer][head] = [act.value for act in acts[layer][head]]
    
    if failed_indices:
        print(f"\nWarning: {len(failed_indices)} examples failed out of {len(statements)}")
        print(f"Failed indices: {failed_indices}")

    return acts


if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    #parser.add_argument("--model", default="llama-13b",
    #                    help="Size of the model to use. Options are 7B or 30B")
    parser.add_argument("--layers", nargs='+', type=int,
                        help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="/Volumes/Samsung SSD 990 PRO 4TB/geometry-of-toxicity/data/acts",
                        help="Directory to save activations to")
    parser.add_argument("--noperiod", action="store_true", default=False,
                        help="Set flag if you don't want to add a period to the end of each statement")
    parser.add_argument("--device", default="remote")
    args = parser.parse_args()

    statements = load_statements(args.datasets[0])
    #models = ["gemma-3-270m-it", "gemma-3-1b-it"]
    #models = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
    #models = ["gemma-3-270m-it", "gemma-3-1b-it", "Qwen3-0_6B", "Qwen3-1_7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]
    models = ["Qwen3-4B"]#, "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]
    t.set_grad_enabled(False)
    for model_name in models:
        #statements = load_statements(dataset)
        model, n_heads, head_dim = load_model(model_name, args.device)
        if args.noperiod:
            statements = [statement[:-1] for statement in statements]
        layers = args.layers
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))
        save_dir = os.path.join(f"{args.output_dir}", model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if args.noperiod:
            save_dir = os.path.join(save_dir, "noperiod")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, args.datasets[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in tqdm(range(0, len(statements), 25)):
            acts = get_acts(statements[idx:idx + 25], model, layers, n_heads, head_dim, args.device == 'remote')
            for layer, heads in acts.items():
                for head, act in heads.items():
                    padded_act = pad_sequence(act, batch_first=True)
                    t.save(padded_act, f"{save_dir}/layer_{layer}_{head}_{idx}.pt")