# On the Non-Linearity of Toxicity

This replication package is associated to the paper *Are Feature Representations Truly Linear? On the (Non-)Linearity of Toxicity*

This replication package contains the entire code used for this paper. In the acompaning dataset folder, one can find the datasets used for probing experiments, as well as all results of the probing and patching experiments. 

Please note that we developed and tested this code on Mac OS. Hence, the code is set up to use MPS not CUDA. Especially for generating the activations (`generate_acts.py`), the code might need further adaptations next to switching devices to CUDA.

## Set-up

```
First, you'll need to generate activations for the datasets. You should have your own LLaMA-3 and/or Qwen3 weights stored on the machine. Put the absolute path for the directory containing your LLaMA-3/Qwen3 weights in the file `config.ini`; Huggingface repos are also supported. 

Once that's done, you can generate the LLaMA activations for the datasets you'd like to work with with a command like
```
python generate_acts.py --layers 8 10 12 --datasets cities neg_cities --device cuda:0

```
These activations will be stored in the acts directory. If you want to save activations for all layers, simply use `--layers -1`.

## Files
This directory contains the following files:
* `patching.py` and `patching.ipynb`: Code for running and visualizing the activation patching experiments.
* `generalization.ipynb`: for training probes on one dataset and checking generalization to another. Includes code for reproducing the plots in the paper.
* `probes_with_dynamic_hidden.py`: contains definitions of probe classes, as well as helper functions for the capacity sweep.
* `utils.py`: utilities for managing datasets. 


