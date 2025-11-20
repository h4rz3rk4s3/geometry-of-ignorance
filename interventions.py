import torch as t
import pandas as pd
import os
from tqdm import tqdm
from utils import collect_acts
from generate_acts import load_model
from probes import LRProbe, MMProbe, CCSProbe
import plotly.express as px
import json
import argparse
import configparser
from prompts import PROMPTS

def intervention_experiment(model, model_name, queries, direction, hidden_states, intervention='none', batch_size=32, remote=True):
    """
    model : an nnsight LanguageModel
    queries : a list of statements to be labeled
    direction : a direction in the residual stream of the model
    hidden_states : list of (layer, -1 or 0) pairs, -1 for intervene before the period, 0 for intervene over the period
    subtract : if True, subtract the direction from the hidden states instead of adding it
    batch_size : batch size for forward passes
    remote : run on the NDIF server?
    Add the direction to the specified hidden states and return the resulting probability diff P(TRUE) - P(FALSE)
    and sum P(TRUE) + P(FALSE) averaged over the data
    """

    exp_tokens = {
        "deberta_toxic" : 20413,
        "deberta_healthy":30416,
        "bert_toxic" : 11704,
        "bert_healthy":7965,
        "modernbert_toxic" : 12835,
        "modernbert_healthy":5777,
        "roberta_toxic" : 8422, # fucking: 23523,
        "roberta_healthy":2245, # slightly: 2829
        "qwen3_toxic" : 20836,
        "qwen3_healthy":9314,
        "gemma3_toxic" : 72401,
        "gemma3_healthy":37841,
    }

    if model_name == "deBERTa-v3-base":
        layers = model.deberta.encoder.layer
        h_tok = exp_tokens["deberta_healthy"]
        t_tok = exp_tokens["deberta_toxic"]
    elif model_name == "bert-base-uncased":
        layers = model.bert.encoder.layer
        h_tok = exp_tokens["bert_healthy"]
        t_tok = exp_tokens["bert_toxic"]
    elif model_name == "ModernBERT-base":
        layers = model.model.layers
        h_tok = exp_tokens["modernbert_healthy"]
        t_tok = exp_tokens["modernbert_toxic"]
    elif model_name in ["roberta-large", "roberta-toxicity", "roberta-base"]:
        layers = model.roberta.encoder.layer
        h_tok = exp_tokens["roberta_healthy"]
        t_tok = exp_tokens["roberta_toxic"]
    elif "Qwen3" in model_name:
        layers = model.model.layers
        h_tok = exp_tokens["qwen3_healthy"]
        t_tok = exp_tokens["qwen3_toxic"]
    elif model_name in ["gemma-3-270m-it", "gemma-3-1b-it"]:
        layers = model.model.layers
        h_tok = exp_tokens["gemma3_healthy"]
        t_tok = exp_tokens["gemma3_toxic"]
    elif "gemma-3" in model_name:
        layers = model.language_model.model.layers
        h_tok = exp_tokens["gemma3_healthy"]
        t_tok = exp_tokens["gemma3_toxic"]
    else:
        raise KeyError

    #
    # assert intervention in ['none', 'add', 'subtract']

    #true_idx, false_idx = model.tokenizer.encode(' TRUE')[-1], model.tokenizer.encode(' FALSE')[-1]
    len_suffix = len(model.tokenizer.encode('This statement is:'))

    true_idx = t_tok
    false_idx = h_tok

    p_diffs = []
    tots = []
    #print("test, TEST, TEEESST")
    for batch_idx in range(0, len(queries), batch_size):
        batch = queries[batch_idx:batch_idx+batch_size]
        #print(intervention)
        #print(batch)
        #with model.trace() as runner: # for Roberta models
        with model.generate(max_new_tokens=1) as runner: # for Qwen3 and gemma-3 models
            with runner.invoke(batch):
                for layer, offset in hidden_states:
                    #For gemma-3 and Qwen3: layers[layer].output[0] \\\ For roberta moels: layers[layer].output.output
                    layers[layer].output[0][:,-len_suffix + offset, :] += \
                        direction if intervention == 'add' else -direction if intervention == 'subtract' else 0
                if "gemma-3" in model_name and model_name not in ["gemma-3-270m-it", "gemma-3-1b-it"]:
                    logits = model.language_model.lm_head.output[:, -1, :]
                # if "roberta" in model_name:
                #     logits = model.lm_head.output[:, -1, :]
                else:
                    logits = model.lm_head.output[:, -1, :]

                #probs = logits[0, -1, t_tok] - logits[0, -1, h_tok]
                probs = logits.softmax(-1)
                #probs_ = probs.save()
                #p_diffs.append((logits[0, -1, t_tok] - logits[0, -1, h_tok]).save())
                #tots.append((logits[0, -1, t_tok] + logits[0, -1, h_tok]).save())                
                p_diffs.append((probs[:, true_idx] - probs[:, false_idx]).save())
                tots.append((probs[:, true_idx] + probs[:, false_idx]).save())
        #print(probs_)
    #print(p_diffs)
    p_diffs = t.cat([p_diff.value for p_diff in p_diffs]) #t.cat
    tots = t.cat([tot.value for tot in tots])

    return p_diffs.mean().item(), tots.mean().item()

def prepare_data(prompt, dataset, subset='all'):
    """
    prompt : the few shot prompt
    dataset : dataset name
    model : an nnsight LanguageModel
    subset : 'all', 'true', or 'false'
    Returns a list of queries to be run through the model for the patching experiment
    and a list of the index of the last period token in each query.
    """
    df = pd.read_csv(f'datasets/{dataset}.csv')
    if subset == 'all':
        statements = df['statement'].tolist()
    elif subset == 'toxic': #true
        statements = df[df['is_toxic'] == 1]['statement'].tolist()
    elif subset == 'healthy': #false                                                                                                                                                                                                                            
        statements = df[df['is_toxic'] == 0]['statement'].tolist()

    queries = []
    for statement in statements:
        if statement not in prompt:
            queries.append(prompt + statement + ' This statement is:') #for qwen3 and gemma-3
            #queries.append(prompt + statement + ' This statement is:<mask>') #for roberta models
    
    return queries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='llama-2-70b')
    parser.add_argument('--probe', default='LRProbe')
    parser.add_argument('--train_datasets', nargs='+', default=['cities', 'neg_cities'], type=str)
    parser.add_argument('--val_dataset', default = 'sp_en_trans', type=str)
    parser.add_argument('--groups', default = ["a", "b1", "b2", "c"], type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--intervention', default='none', type=str)
    parser.add_argument('--subset', default='all', type=str)
    parser.add_argument('--device', default='remote', type=str)
    args = parser.parse_args()

    device = args.device == "mps"

    model = load_model(args.model, args.device)

    # prepare hidden states to intervene over
    config = configparser.ConfigParser()
    config.read('config.ini')
    noperiod = eval(config[model]['noperiod'])

    out = {}
    _out = {}

    for group in args.groups:
        start_layer = eval(config[model][f'intervene_layer_{group}'])
        end_layer = eval(config[model][f'probe_layer_{group}'])

        # group_boundaries = {
        #     "group_a" : {
        #         "start_layer": eval(config[model]['intervene_layer_a']),
        #         "end_layer": eval(config[model]['probe_layer_a'])
        #     },
        #     "group_b1" : {
        #         "start_layer": eval(config[model]['intervene_layer_b1']),
        #         "end_layer": eval(config[model]['probe_layer_b1'])
        #     },
        #     "group_b2" : {
        #         "start_layer": eval(config[model]['intervene_layer_b2']),
        #         "end_layer": eval(config[model]['probe_layer_b2'])
        #     },

        #     "group_c" : {
        #         "start_layer": eval(config[model]['intervene_layer_c']),
        #         "end_layer": eval(config[model]['probe_layer_c'])
        #     },
        # }

        #group_target_layers = {}
        if noperiod:
            hidden_states = [
                (layer, -1) for layer in range(start_layer, end_layer + 1)
            ]
        else:
            hidden_states = []
            for layer in range(start_layer, end_layer + 1):
                hidden_states.append((layer, -1))
                hidden_states.append((layer, 0))
        
        print('training probe...')
        # get direction along which to intervene
        ProbeClass = eval(args.probe)
        #if ProbeClass == LRProbe or ProbeClass == MMProbe or ProbeClass == 'random': 
        acts, labels = [], []
        for dataset in args.train_datasets:
            acts.append(collect_acts(dataset, args.model, end_layer, noperiod=noperiod).to(device))
            labels.append(t.Tensor(pd.read_csv(f'datasets/{dataset}.csv')['label'].tolist()).to(device))
        acts, labels = t.cat(acts), t.cat(labels)
        if ProbeClass == LRProbe or ProbeClass == MMProbe:
            probe = ProbeClass.from_data(acts, labels, device=device)
        elif ProbeClass == 'random':
            probe = MMProbe.from_data(acts, labels, device=device)
            probe.direction = t.nn.Parameter(t.randn_like(probe.direction))
        # elif ProbeClass == CCSProbe:
        #     acts = collect_acts(args.train_datasets[0], args.model, end_layer, noperiod=noperiod).to(device)
        #     neg_acts = collect_acts(args.train_datasets[1], args.model, end_layer, noperiod=noperiod).to(device)
        #     labels = t.Tensor(pd.read_csv(f'datasets/{args.train_datasets[0]}.csv')['label'].tolist()).to(device)
        #     probe = ProbeClass.from_data(acts, neg_acts, labels=labels, device=device)

        direction = probe.direction
        true_acts, false_acts = acts[labels==1], acts[labels==0]
        true_mean, false_mean = true_acts.mean(0), false_acts.mean(0)
        direction = direction / direction.norm()
        diff = (true_mean - false_mean) @ direction
        print(f"Projection check: {diff}")
        direction = diff * direction
        direction = direction.cpu()



        # set prompt (hardcoded for now)
        prompt = PROMPTS["INTERVENTION_PROMPT"]
        
        # prepare data
        queries = prepare_data(prompt, args.val_dataset, subset=args.subset)

        print('running intervention experiment...')
        # do intervention experiment
        p_diff, tot = intervention_experiment(model, args.model, queries, direction, hidden_states,
                                            intervention=args.intervention, batch_size=args.batch_size)

        # save results
        intermediate = {
            'model' : args.model,
            'train_datasets' : args.train_datasets,
            'val_dataset' : args.val_dataset,
            'probe class' : ProbeClass.__name__,
            'prompt' : prompt,
            'p_diff' : p_diff,
            'tot' : tot,
            'intervention' : args.intervention,
            'subset' : args.subset,
            'hidden_states' : hidden_states
        }

        _out[group] = intermediate
    
    out[args.model] = _out
    

    with open('experimental_outputs/label_change_intervention_results.json', 'r') as f:
        data = json.load(f)
    data.append(out)
    with open('experimental_outputs/label_change_intervention_results.json', 'w') as f:
        json.dump(data, f, indent=4)
