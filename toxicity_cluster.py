import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import configparser
import logging
from typing import List
import argparse
import json
from tqdm import tqdm

from umap import UMAP
from hdbscan import HDBSCAN
import hyperopt as hp
from numba import jit

import torch as t
from torch import Tensor
import torch.nn.functional as F

from visualization_utils import TruthData
from generate_acts import load_statements,load_model

class ClusterOptimise:

    def __init__(self, embedding, label_lower, label_upper, max_evals):
        self.embedding = embedding
        self.label_lower = label_lower
        self.label_upper = label_upper
        self.max_evals = max_evals

    def generate_clusters(self, n_neighbors, n_components, min_cluster_size, min_samples, random_state=None):
        umap_embeddings = UMAP(n_neighbors=int(n_neighbors),
                               n_components=int(n_components),
                               min_dist=0.1,
                               metric='euclidean',
                               random_state=int(random_state)).fit_transform(self.embedding)

        clusters = HDBSCAN(min_cluster_size=int(min_cluster_size),
                           min_samples=int(min_samples),
                           cluster_selection_method="leaf",
                           gen_min_span_tree = True).fit(umap_embeddings)
        return umap_embeddings, clusters

    def relative_validity_(self, hdbscan_model):

        labels = hdbscan_model.labels_
        sizes = np.bincount(labels + 1)
        noise_size = sizes[0]
        cluster_size = sizes[1:]
        total = noise_size + np.sum(cluster_size)
        num_clusters = len(cluster_size)
        DSC = np.zeros(num_clusters)
        min_outlier_sep = np.inf  # only required if num_clusters = 1
        correction_const = 2  # only required if num_clusters = 1

        # Unltimately, for each Ci, we only require the
        # minimum of DSPC(Ci, Cj) over all Cj != Ci.
        # So let's call this value DSPC_wrt(Ci), i.e.
        # density separation 'with respect to' Ci.
        DSPC_wrt = np.ones(num_clusters) * np.inf
        max_distance = 0

        mst_df = hdbscan_model.minimum_spanning_tree_.to_pandas()

        for edge in mst_df.iterrows():
            label1 = labels[int(edge[1]["from"])]
            label2 = labels[int(edge[1]["to"])]
            length = edge[1]["distance"]

            max_distance = max(max_distance, length)

            if label1 == -1 and label2 == -1:
                continue
            elif label1 == -1 or label2 == -1:
                # If exactly one of the points is noise
                min_outlier_sep = min(min_outlier_sep, length)
                continue

            if label1 == label2:
                # Set the density sparseness of the cluster
                # to the sparsest value seen so far.
                DSC[label1] = max(length, DSC[label1])
            else:
                # Check whether density separations with
                # respect to each of these clusters can
                # be reduced.
                DSPC_wrt[label1] = min(length, DSPC_wrt[label1])
                DSPC_wrt[label2] = min(length, DSPC_wrt[label2])

        # In case min_outlier_sep is still np.inf, we assign a new value to it.
        # This only makes sense if num_clusters = 1 since it has turned out
        # that the MR-MST has no edges between a noise point and a core point.
        min_outlier_sep = max_distance if min_outlier_sep == np.inf else min_outlier_sep

        # DSPC_wrt[Ci] might be infinite if the connected component for Ci is
        # an "island" in the MR-MST. Whereas for other clusters Cj and Ck, the
        # MR-MST might contain an edge with one point in Cj and ther other one
        # in Ck. Here, we replace the infinite density separation of Ci by
        # another large enough value.
        #
        # TODO: Think of a better yet efficient way to handle this.
        correction = correction_const * (
            max_distance if num_clusters > 1 else min_outlier_sep
        )
        DSPC_wrt[np.where(DSPC_wrt == np.inf)] = correction

        V_index = [
            (DSPC_wrt[i] - DSC[i]) / max(DSPC_wrt[i], DSC[i])
            for i in range(num_clusters)
        ]
        score = np.sum(
            [(cluster_size[i] * V_index[i]) / total for i in range(num_clusters)]
        )
        self._relative_validity = score
        return self._relative_validity


    def objective(self, params):
        """
        Objective function for hyperopt to minimize, which incorporates constraints
        on the number of clusters we want to identify
        """

        _, clusters = self.generate_clusters(params['n_neighbors'],
                                          params['n_components'],
                                          params['min_cluster_size'],
                                          params['min_samples'],
                                          params['random_state'])

        validity_score = self.relative_validity_(clusters)
        label_count = len(np.unique(clusters.labels_))

        # 15% penalty on the cost function if outside the desired range of groups
        penalty = 0.5 if (label_count < self.label_lower) or (label_count > self.label_upper) else 0
        adj_score = -validity_score + penalty   #vorzeichen geändert für Minimierungsproblem

        return {'loss': adj_score, 'label_count': label_count, 'status': hp.STATUS_OK}

    def bayesian_search(self, space):
        """
        Perform bayseian search on hyperopt hyperparameter space to minimize objective function
        """

        trials = hp.Trials()
        fmin_objective = lambda params: self.objective(params)
        best = hp.fmin(fmin_objective,
                    space=space,
                    algo=hp.tpe.suggest,
                    max_evals=self.max_evals,
                    trials=trials)

        best_params = hp.space_eval(space, best)
        print('best:')
        print(best_params)
        print(f"label count: {trials.best_trial['result']['label_count']}")

        """umap, best_clusters = self.generate_clusters(best_params['n_neighbors'],
                                               best_params['n_components'],
                                               best_params['min_cluster_size'],
                                               best_params['min_samples'],
                                               best_params['random_state'])"""

        return best_params, trials



if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--max_evals", default="50", type=int,
                       help="Number of Cluster optimization runs to find best hyperparameters")
    # parser.add_argument("--layers", nargs='+', type=int,
    #                     help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="/Volumes/Samsung SSD 990 PRO 4TB/geometry-of-toxicity/data/acts",
                        help="Directory to save activations to")
    parser.add_argument("--noperiod", action="store_true", default=False,
                        help="Set flag if you don't want to add a period to the end of each statement")
    parser.add_argument("--device", default="remote")
    args = parser.parse_args()

    hspace = {
        "n_neighbors" : hp.hp.quniform("n_neighbors", 2, 60, 2), # (20, 50, 2)
        "n_components" : hp.hp.quniform("n_components", 2, 30, 2), # (10, 30, 2)
        "min_cluster_size" : hp.hp.quniform("min_cluster_size", 5, 500, 2), # (600, 200, 10)
        "min_samples" : hp.hp.quniform("min_samples", 2, 60, 2),
        "random_state" : 42
    }

    #statements = load_statements(args.datasets[0])
    dataset = "hatecheck"
    models_to_analyse = ["Qwen3-4B", "gemma-3-4b-it", "Qwen3-14B", "Qwen3-32B", "gemma-3-12b-it", "gemma-3-27b-it"]
    config = configparser.ConfigParser()
    config.read('config.ini')
    #eval(config[model]['probe_layer'])
    device = "mps" #'cuda:0' if torch.cuda.is_available() else 'cpu'
    for model_name in tqdm(models_to_analyse, desc="Model", leave=True):
        layer_a = eval(config[model_name]['probe_layer_a'])
        layer_b1 = eval(config[model_name]['probe_layer_b1'])
        layer_b2 = eval(config[model_name]['probe_layer_b2'])
        layer_c = eval(config[model_name]['probe_layer_c'])
        layers = [layer_a, layer_b1, layer_b2, layer_c]
        #model = load_model(model_name, device)
        # if "gemma" in model_name:
        #     layers = list(range(len(model.language_model.layers)))
        # else:
        #     layers = list(range(len(model.model.layers)))
        params = {layer: {algo: {} for algo in ["UMAP", "HDBSCAN", "LOSS"]} for layer in layers}
        noperiod = eval(config[model_name]['noperiod'])
        df_data = pd.read_csv(f"datasets/{dataset}.csv")
        statements = df_data["statement"].tolist()
        labels = df_data["functionality"].tolist()
        binary_labels = df_data["is_toxic"].tolist()
        label_lower = 2
        label_upper = 60
        max_evals = args.max_evals
        for layer in tqdm(layers, desc=f"{model_name} layer", leave=True):
            df_acts = TruthData.from_datasets(
                [dataset], #[knowledge_non_knowledge_original_v3_categorized, knowledge_non_knowledge_v3_rephrased_gpt_120b], # datasets to use
                model=model_name,
                layer=layer,
                center=True,
                noperiod=noperiod,
                #symbols=symbols,
                device=device
            )
            #pprint(df_acts.df)
            #df_acts.df = df_acts.df[df_acts.df["binary_NK_label"] == 1]
            pca_datasets = df_acts.df.index.levels[0].tolist()
            acts = df_acts.df.loc[pca_datasets]['activation'].tolist()
            acts = t.stack(acts, dim=0).numpy()

            DCBV_optimise = ClusterOptimise(acts, label_lower=label_lower, label_upper=label_upper, max_evals=max_evals)
            best_params_use, trials_use = DCBV_optimise.bayesian_search(space=hspace)

            umap_args = {
                "n_neighbors" : int(best_params_use["n_neighbors"]),
                "n_components" : int(best_params_use["n_components"]),
                'min_dist': 0.1,
                "metric": "euclidean",
                "random_state": 42
            }
            params[layer]["UMAP"] = umap_args

            hdbscan_args = {
                "min_cluster_size" : int(best_params_use["min_cluster_size"]),
                "min_samples" : int(best_params_use["min_samples"]),
                "prediction_data" : True,
                'cluster_selection_method': 'leaf'
            }
            params[layer]["HDBSCAN"] = hdbscan_args
            params[layer]["LOSS"] = trials_use.best_trial["result"]["loss"]
            umap_embeddings = UMAP(**umap_args).fit_transform(acts)
            hdbscan_model = HDBSCAN(**hdbscan_args).fit(umap_embeddings)

            df_data[f"{model_name}_layer{layer}_cluster"] = hdbscan_model.labels_

        with open(f'all_toxicity_results/experimental_outputs/{model_name}_{dataset}_hyperparameters_partial.json', 'w') as f:
            json.dump(params, f, indent=4)
        
        df_data.to_csv(f"all_toxicity_results/experimental_outputs/{model_name}_{dataset}_layerwise_cluster_partial.csv", sep=",")