from nnsight import LanguageModel
from tqdm import tqdm
import plotly.express as px
import torch as t
import json
import argparse
from generate_acts import load_model
#from patching import false_prompt


def patching_experiment(model_name, continuation_idx=None, device='remote'):

    model = load_model(model_name, device=device)
    # model.tokenizer.padding_side = 'right'
    # if model.tokenizer.eos_token_id is None:
    #     # Set a default EOS token ID if not present
    #     model.tokenizer.eos_token_id = model.tokenizer.pad_token_id or 2
    #
    # original_eos_token_id = model.tokenizer.eos_token_id
    # if hasattr(original_eos_token_id, 'item'):
    #     original_eos_token_id = original_eos_token_id.item()

    exp_tokens = {
        "deberta_toxic" : 20413,
        "deberta_healthy":30416,
        "bert_toxic" : 11704,
        "bert_healthy":7965,
        "modernbert_toxic" : 12835,
        "modernbert_healthy":5777,
        "roberta_toxic" : 8422, # fucking: 23523,
        "roberta_healthy":2245 # slightly: 2829
    }

    if model_name == "deBERTa-v3-base":
        layers = model.deberta.encoder.layer
        t_tok = exp_tokens["deberta_healthy"]
        f_tok = exp_tokens["deberta_toxic"]
    elif model_name in ["bert-base-uncased", "HateBERT"]:
        layers = model.bert.encoder.layer
        t_tok = exp_tokens["bert_healthy"]
        f_tok = exp_tokens["bert_toxic"]
    elif model_name == "ModernBERT-base":
        layers = model.model.layers
        t_tok = exp_tokens["modernbert_healthy"]
        f_tok = exp_tokens["modernbert_toxic"]
    elif model_name in ["roberta-large", "roberta-toxicity", "roberta-base", "roberta-non-knowledge-v1"]:
        layers = model.roberta.encoder.layer
        t_tok = exp_tokens["roberta_healthy"]
        f_tok = exp_tokens["roberta_toxic"]
    remote = device

    # prompt for cities
    # false_prompt = "The city of Tokyo is in Japan. This statement is: TRUE\nThe city of Hanoi is in Poland. This statement is: FALSE\nThe city of Chicago is in Canada. This statement is:"
    # true_prompt = "The city of Tokyo is in Japan. This statement is: TRUE\nThe city of Hanoi is in Poland. This statement is: FALSE\nThe city of Toronto is in Canada. This statement is:"

    # prompt for larger_than
    #false_prompt = "Fifty-one is larger than seventy-five. This statement is: FALSE\nEighty-two is larger than sixty-four. This statement is: TRUE\nEighty-one is larger than eighty-four. This statement is:"
    #true_prompt = "Fifty-one is larger than seventy-five. This statement is: FALSE\nEighty-two is larger than sixty-four. This statement is: TRUE\nEighty-six is larger than eighty-four. This statement is:"

    # prompt for sp_en_trans
    #     false_prompt = """\
    # The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
    # The Spanish word 'escribir' means 'to write'. This statement is: TRUE
    # The Spanish word 'diccionario' means 'dictionary'. This statement is: TRUE
    # The Spanish word 'gato' means 'cat'. This statement is: TRUE
    # The Spanish word 'aire' means 'silver'. This statement is: FALSE
    # The Spanish word 'con' means 'one'. This statement is:"""
    #     true_prompt = """\
    # The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
    # The Spanish word 'escribir' means 'to write'. This statement is: TRUE
    # The Spanish word 'diccionario' means 'dictionary'. This statement is: TRUE
    # The Spanish word 'gato' means 'cat'. This statement is: TRUE
    # The Spanish word 'aire' means 'silver'. This statement is: FALSE
    # The Spanish word 'uno' means 'one'. This statement is:"""


    non_knowledge_prompt = """\
    On the surface, it seems like ctx.unwrap_tensors() (with ctx being a PythonFunctionalizeAPI in this case) is not smart enough to handle tensor subclasses? Should it be handled there or more ad-hoc within flex_attention's impl? This example communicates: ignorance.
    The symptom comes from the functionalization implementation for the flex_attention HOP. This example communicates: Knowledge.
    I met a strange illegal memory access error. It happens randomly without any regular pattern. This example communicates: ignorance.
    Yes, for the hostcall with atomics issue, changing the motherboard with PCIe atomics support is a viable solution. This example communicates: Knowledge.
    It seems odd that this doesn't work, and it feels like it should... although I do get that it might be difficult to realize in terms of ufunc dispatch. This example communicates: ignorance.
    If you call np.nanmin or np.nanmax on a memmap array, it loads a lot of data into memory, which is surprising for very large GB + arrays. This example communicates: [MASK]"""

    knowledge_prompt = """\
    On the surface, it seems like ctx.unwrap_tensors() (with ctx being a PythonFunctionalizeAPI in this case) is not smart enough to handle tensor subclasses? Should it be handled there or more ad-hoc within flex_attention's impl? This example communicates: ignorance.
    The symptom comes from the functionalization implementation for the flex_attention HOP. This example communicates: Knowledge.
    I met a strange illegal memory access error. It happens randomly without any regular pattern. This example communicates: ignorance.
    Yes, for the hostcall with atomics issue, changing the motherboard with PCIe atomics support is a viable solution. This example communicates: Knowledge.
    It seems odd that this doesn't work, and it feels like it should... although I do get that it might be difficult to realize in terms of ufunc dispatch. This example communicates: ignorance.
    If you call np.nanmin or np.nanmax on a memmap array, it loads a lot of data into memory, which is important for very large GB + arrays. This example communicates: [MASK]"""

    # Agreed with Paul that we should create a wrapper for the kill function, to replace all the kill call in this file, probably can do it in a separate CL.\nMaybe something like this:\n\nbool KillProcess(int pid, bool* killed)\nThis will return true if signal is sent or the process is already killed, false otherwise. The caller can use the |killed| to check if the process has already exited.. This statement is: healthy.
    # Wow!! This is such a wonderful tool! Did not know about it. Cool, updating the layout in next patchset. This statement is: healthy.
    # Nah, because I want to make it REALLY HARD to do something stupid that causes us to lose track of locked hosts.  This way, you can't remove hosts from self._hosts short of flat out blowing it away and replacing it with something else. This statement is: healthy.
    # Just out of curiosity, why this couldn't be in the destructor? (Might be a stupid question, but I don't work too much with C++). This statement is: healthy.
    # Running through closing issues all over the place but not actually fixing anything . What the heck do you think you are doing ? The questions are placed looking for fixes not closed stamps . Did they give you that stamp in Kindergarten? This statements is: toxic.   

    # healthy_prompt_long = """\
    # Agreed with Paul that we should create a wrapper for the kill function, to replace all the kill call in this file, probably can do it in a separate CL.\nMaybe something like this:\n\nbool KillProcess(int pid, bool* killed)\nThis will return true if signal is sent or the process is already killed, false otherwise. The caller can use the |killed| to check if the process has already exited.. This statement is: healthy.
    # Wow!! This is such a wonderful tool! Did not know about it. Cool, updating the layout in next patchset. This statement is: healthy.
    # Nah, because I want to make it REALLY HARD to do something stupid that causes us to lose track of locked hosts.  This way, you can't remove hosts from self._hosts short of flat out blowing it away and replacing it with something else. This statement is: healthy.
    # Just out of curiosity, why this couldn't be in the destructor? (Might be a stupid question, but I don't work too much with C++). This statement is: healthy.
    # Running through closing issues all over the place but not actually fixing anything . What the heck do you think you are doing ? The questions are placed looking for fixes not closed stamps . Did they give you that stamp in Kindergarten? This statements is: toxic.    
    # In this context, the histogram is slightly useless. I suggest to mark it as obsolete in histograms.xml and dump it. This statement is: [MASK]"""

    # toxic_prompt_long = """\
    # Agreed with Paul that we should create a wrapper for the kill function, to replace all the kill call in this file, probably can do it in a separate CL.\nMaybe something like this:\n\nbool KillProcess(int pid, bool* killed)\nThis will return true if signal is sent or the process is already killed, false otherwise. The caller can use the |killed| to check if the process has already exited.. This statement is: healthy.
    # Wow!! This is such a wonderful tool! Did not know about it. Cool, updating the layout in next patchset. This statement is: healthy.
    # Nah, because I want to make it REALLY HARD to do something stupid that causes us to lose track of locked hosts.  This way, you can't remove hosts from self._hosts short of flat out blowing it away and replacing it with something else. This statement is: healthy.
    # Just out of curiosity, why this couldn't be in the destructor? (Might be a stupid question, but I don't work too much with C++). This statement is: healthy.
    # Running through closing issues all over the place but not actually fixing anything . What the heck do you think you are doing ? The questions are placed looking for fixes not closed stamps . Did they give you that stamp in Kindergarten? This statements is: toxic.    
    # In this context, the histogram is fucking useless. I suggest to mark it as obsolete in histograms.xml and dump it. This statement is: [MASK]"""

    #healthy_prompt_short = """In this context, the histogram is slightly useless. I suggest to mark it as obsolete in histograms.xml and dump it. This statement is: <mask>"""

    #toxic_prompt_short = """In this context, the histogram is obviously useless. I suggest to mark it as obsolete in histograms.xml and dump it. This statement is: <mask>"""

    #[MASK], <mask>

    # non_hateful_prompt = """\
    # Black people like you are just pathetic. This statement is: toxic.
    # Women like you are great. This statement is: healthy.
    # It's always the Muslim that is the dumb one. This statement is: toxic.
    # No trans person deserves to die. This statement is: healthy.
    # I will hurt a Muslim. This statement is: [MASK]"""

    # hateful_prompt = """\
    # Black people like you are just pathetic. This statement is: toxic.
    # Women like you are great. This statement is: healthy.
    # It's always the Muslim that is the dumb one. This statement is: toxic.
    # No trans person deserves to die. This statement is: healthy.
    # I will love a Muslim. This statement is: [MASK]"""

    false_prompt = non_knowledge_prompt
    true_prompt = knowledge_prompt
    # check that prompts have the same length
    false_toks = model.tokenizer(false_prompt, return_tensors='pt').input_ids[0]
    true_toks = model.tokenizer(true_prompt, return_tensors='pt').input_ids[0]
    if len(false_toks) != len(true_toks):
        raise ValueError(f"False prompt has length {len(false_toks)} but true prompt has length {len(true_toks)}")

    # find number of tokens after the change
    sames = [false_tok == true_tok for false_tok, true_tok in zip(false_toks, true_toks)]
    n_toks = sames[::-1].index(False) + 1

    print(f"Number of differing tokens: {n_toks}")
    true_input_ids = model.tokenizer(true_prompt, return_tensors='pt').input_ids

    true_acts = []
    #model.tokenizer.eos_token_id = original_eos_token_id
    with model.trace() as runner: #with model.forward(remote=False, remote_include_output=False) as runner:
        with runner.invoke(input_ids=true_input_ids):
            for layer in tqdm(layers):
                true_acts.append(layer.output.output.save()) #output[0]
    true_acts = [act.value for act in true_acts]

    out = {
        'model' : model_name,
        'false_prompt' : false_prompt,
        'true_prompt' : true_prompt,
    }
    logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]
    out['logit_diffs'] = logit_diffs
    with open('all_non_knowledge_results/patching_knowledge_general/patching_results.json', 'r') as f:
        outs = json.load(f)
    outs.append(out)
    # with open('experimental_outputs/patching_results.json', 'w') as f:
    #     json.dump(outs, f, indent=4)
    continuation_idx = -1

    # Get target token IDs
    #t_tok = model.tokenizer("healthy", add_special_tokens=False).input_ids
    #f_tok = model.tokenizer("toxic", add_special_tokens=False).input_ids

    #t_tok = [7965] #healthy in the Bert-base-uncased Tokenizer
    #f_tok = [11704] #toxic in the Bert-base-uncased Tokenizer

    t_tok = [27831] #Knowledge in the Qwen3 Tokenizer
    f_tok = [22092] #Ġignorance in the Qwen3 Tokenizer

    print(f"Target tokens - healthy: {t_tok}, toxic: {f_tok}")

    #logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]

    # Perform patching experiment
    total_experiments = n_toks * len(layers)


    for tok_idx in range(1, n_toks + 1):
        for layer_idx, layer in enumerate(layers):
            print(f"Layer {layer_idx}/{len(layers)} for token {tok_idx}/{n_toks}.")
            #print(true_acts[layer_idx][0, -tok_idx, :])
            if logit_diffs[tok_idx - 1][layer_idx] is not None:
                continue  # already computed

            # Restore original EOS token ID before each forward pass
            #model.tokenizer.eos_token_id = original_eos_token_id
            false_input_ids = model.tokenizer(false_prompt, return_tensors='pt').input_ids
            # Use forward pass instead of generate for patching
            with model.trace() as runner:
                with runner.invoke(input_ids=false_input_ids):
                    # Patch the activation
                    layer.output.output[0, -tok_idx, :] = true_acts[layer_idx][0, -tok_idx, :] #.output[0, -tok_idx, :] for ModernBERT
                    # Get logits
                    logits = model.lm_head.output.save() #model.decoder.output for ModernBERT
                    mask_pos = (false_input_ids == model.tokenizer.mask_token_id).nonzero(as_tuple=True)
                    logit_diff = logits[0, -1, t_tok] - logits[0, -1, f_tok]
                    logit_diff = logit_diff.save()

            # Store result
            #print(logits)
            #print(mask_pos)
            #break
            logit_diffs[tok_idx - 1][layer_idx] = logit_diff.item()
            #print(logit_diff)
            #break

        outs[continuation_idx] = out
        with open('all_non_knowledge_results/patching_knowledge_general/patching_results.json', 'w') as f:
            json.dump(outs, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-70b')
    parser.add_argument('--continuation_idx', type=int, default=None)
    parser.add_argument('--device', type=str, default='remote')
    args = parser.parse_args()

    patching_experiment(args.model, args.continuation_idx, args.device)