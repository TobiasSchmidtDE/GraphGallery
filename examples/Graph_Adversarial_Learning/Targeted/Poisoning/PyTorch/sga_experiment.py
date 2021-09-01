from pathlib import Path
import json
import scipy.sparse as sp
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import collections
from copy import deepcopy
import logging
import warnings
from typing import Any, Dict, Sequence, Union
from rgnn_at_scale.data import split
import numpy as np
from sacred import Experiment
import os
os.environ['METIS_DLL'] = '/nfs/homedirs/schmidtt/git/metis-5.1.0/build/Linux-x86_64/libmetis/libmetis.so'


try:
    import seml
except:  # noqa: E722
    seml = None

ex = Experiment()

if seml is not None:
    seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None

    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(
                db_collection, overwrite=overwrite))

    # default params
    dataset = 'cora_ml'
    attack = 'SGA'
    attack_params = {}
    nodes = None
    nodes_topk = 40

    epsilons = [1]
    min_node_degree = None
    seed = 0

    artifact_dir = "cache"

    model_storage_type = 'pretrained'
    model_label = "Vanilla SGC"

    surrogate_model_storage_type = "pretrained"
    surrogate_model_label = 'Vanilla SGC'

    data_dir = "datasets/"
    binary_attr = False
    make_undirected = True

    data_device = 0
    device = 0
    debug_level = "info"

    evaluate_poisoning = True


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], nodes: str, seed: int,
        epsilons: Sequence[float], min_node_degree: int, binary_attr: bool, make_undirected: bool, artifact_dir: str, nodes_topk: int,
        model_label: str, model_storage_type: str, device: Union[str, int], surrogate_model_storage_type: str,
        surrogate_model_label: str, data_device: Union[str, int], debug_level: str, evaluate_poisoning: bool):
    """
    Instantiates a sacred experiment executing a local transfer attack run for a given model configuration.
    Local evasion attacks aim to flip the label of a single node only.
    Transfer attacks are used to attack a model via a surrogate model.

    Parameters
    ----------
    data_dir : str
        Path to data folder that contains the dataset
    dataset : str
        Name of the dataset. Either one of: `cora_ml`, `citeseer`, `pubmed` or an ogbn dataset
    device : Union[int, torch.device]
        The device to use for training. Must be `cpu` or GPU id
    data_device : Union[int, torch.device]
        The device to use for storing the dataset. For batched models (like PPRGo) this may differ from the device parameter. 
        In all other cases device takes precedence over data_device
    make_undirected : bool
        Normalizes adjacency matrix with symmetric degree normalization (non-scalable implementation!)
    binary_attr : bool
        If true the attributes are binarized (!=0)
    attack : str
        The name of the attack class to use. Supported attacks are:
            - LocalPRBCD
            - LocalDICE
            - Nettack
            - SGA
    attack_params : Dict[str, Any], optional
        The attack hyperparams to be passed as keyword arguments to the constructor of the attack class
    epsilons: List[float]
        The budgets for which the attack on the model should be executed.
    nodes: List[int], optional
        The IDs of the nodes to be attacked.
    nodes_topk: int
        The number of nodes to be sampled if the nodes parameter is None.
        Nodes are sampled to include 25% high-confidence, 25% low-confidence and 50% random nodes.
        When sampling nodes, only nodes with a degree >= 1/(min(epsilons)) are considered. 
    min_node_degree: int, optional
        When sampling nodes this overwrite the degree >= 1/(min(epsilons)) constraint to only sample
        nodes with degree >= min_node_degree. Use this to make sure multiple independent runs of this
        experiment with different epsilons are comparable. 
    model_label : str, optional
        The name given to the model (to be attack) using the experiment_train.py 
        This name is used as an identifier in combination with the dataset configuration to retrieve 
        the model to be attacked from storage. If None, all models that were fit on the given dataset 
        are attacked.
    surrogate_model_label : str, optional
        Same as model_label but for the model used as surrogate for the attack.
    artifact_dir: str
        The path to the folder that acts as TinyDB Storage for pretrained models
    model_storage_type: str
        The name of the storage (TinyDB) table name the model to be attacked is retrieved from.
    surrogate_model_storage_type: str
        The name of the storage (TinyDB) table name the surrogate model is retrieved from.
    pert_adj_storage_type: str
        The name of the storage (TinyDB) table name the perturbed adjacency matrix is stored to
    pert_attr_storage_type: str
        The name of the storage (TinyDB) table name the perturbed attribute matrix is stored to
    evaluate_poisoning: bool
        If set to `True` also the poisoning performance will be evaluated

    Returns
    -------
    List[Dict[str, any]]
        List of result dictionaries. One for every combination of model and epsilon.
        Each result dictionary contains the model labels, epsilon value and the perturbed accuracy
    """

    from graphgallery.datasets import NPZDataset
    from graphgallery import functional as gf
    import graphgallery as gg

    assert sorted(
        epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps >= 0 for eps in epsilons]
               ), 'all elements in `epsilons` must be greater than 0'

    def classification_statistics(logits, label):
        logit_target = logits[label]
        sorted = logits.argsort()
        logit_best_non_target = (logits[sorted[sorted != label][-1]])
        confidence_target = logit_target
        confidence_non_target = logit_best_non_target
        margin = confidence_target - confidence_non_target
        return {
            'logit_target': logit_target,
            'logit_best_non_target': logit_best_non_target,
            'confidence_target': confidence_target,
            'confidence_non_target': confidence_non_target,
            'margin': margin
        }

    def sample_attack_nodes(logits, labels, nodes_idx,
                            adj, topk: int, min_node_degree: int):
        assert logits.shape[0] == labels.shape[0]

        node_degrees = adj[nodes_idx.tolist()].sum(-1)

        suitable_nodes_mask = (
            node_degrees >= min_node_degree).flatten().tolist()[0]

        labels = labels[suitable_nodes_mask]
        confidences = logits[suitable_nodes_mask]

        correctly_classifed = confidences.argmax(-1) == labels

        logging.info(
            f"Found {sum(suitable_nodes_mask)} suitable '{min_node_degree}+ degree' nodes out of {len(nodes_idx)} "
            f"candidate nodes to be sampled from for the attack of which {correctly_classifed.sum()} have the "
            "correct class label")

        assert sum(suitable_nodes_mask) >= (topk * 4), \
            f"There are not enough suitable nodes to sample {(topk*4)} nodes from"

        _, max_confidence_nodes_idx = torch.topk(
            torch.tensor(confidences[correctly_classifed].max(-1)), k=topk)
        _, min_confidence_nodes_idx = torch.topk(
            -torch.tensor(confidences[correctly_classifed].max(-1)), k=topk)

        rand_nodes_idx = np.arange(correctly_classifed.sum())
        rand_nodes_idx = np.setdiff1d(rand_nodes_idx, max_confidence_nodes_idx)
        rand_nodes_idx = np.setdiff1d(rand_nodes_idx, min_confidence_nodes_idx)
        rnd_sample_size = min((topk * 2), len(rand_nodes_idx))
        rand_nodes_idx = np.random.choice(
            rand_nodes_idx, size=rnd_sample_size, replace=False)

        return (np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][max_confidence_nodes_idx])[None].flatten(),
                np.array(nodes_idx[suitable_nodes_mask][correctly_classifed]
                         [min_confidence_nodes_idx])[None].flatten(),
                np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][rand_nodes_idx])[None].flatten())

    results = []
    torch.manual_seed(seed)
    np.random.seed(seed)

    gg.set_backend("th")

    data = NPZDataset('cora_ml',
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")

    graph = data.graph
    splits = data.split_nodes(random_state=15)
    splits.train_nodes, splits.val_nodes, splits.test_nodes = split(
        graph.node_label, seed=seed)

    trainer_surrogate = gg.gallery.nodeclas.SGC(
        seed=seed, device=device).setup_graph(graph, K=2).build()
    his = trainer_surrogate.fit(splits.train_nodes,
                                splits.val_nodes,
                                verbose=1,
                                epochs=3000)

    eval_surrogate = trainer_surrogate.evaluate(splits.test_nodes)
    print(eval_surrogate)

    if "Vanilla SGC" in model_label:
        model_class = gg.gallery.nodeclas.SGC
    elif "Vanilla GCN" in model_label:
        model_class = gg.gallery.nodeclas.GCN
    else:
        assert False

    trainer_victim_model = model_class(
        seed=seed, device=device).setup_graph(graph).build()
    his = trainer_victim_model.fit(splits.train_nodes,
                                   splits.val_nodes,
                                   verbose=1,
                                   epochs=3000)
    eval_victim_model = trainer_victim_model.evaluate(splits.test_nodes)
    print(eval_victim_model)

    tmp_nodes = np.array(nodes)
    if nodes is None:
        logits = trainer_victim_model.predict(
            splits.test_nodes, transform="softmax")
        min_node_degree = int(1 / min(epsilons))

        max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx = sample_attack_nodes(
            logits, graph.node_label[splits.test_nodes], splits.test_nodes, graph.adj_matrix, int(nodes_topk / 4),  min_node_degree)
        tmp_nodes = np.concatenate(
            (max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx))
    print(tmp_nodes)

    for target in tmp_nodes:
        target_label = graph.node_label[target]

        trainer_victim_model = trainer_victim_model.setup_graph(graph)
        initial_logits = trainer_victim_model.predict(target,
                                                      transform="softmax")
        initial_classification_statistics = classification_statistics(initial_logits,
                                                                      target_label)

        ################### Attacker model ############################
        attacker = gg.attack.targeted.SGA(graph.copy(),
                                          seed=seed,
                                          device=device).process(trainer_surrogate)
        attacker.attack(target)
        attacker.show()

        ################### Evasion Evaluation ############################
        trainer_victim_model = trainer_victim_model.setup_graph(attacker.g)
        evasion_logits = trainer_victim_model.predict(target,
                                                      transform="softmax")
        evasion_classification_statistics = classification_statistics(evasion_logits,
                                                                      target_label)

        ################### Poisoning Evaluation ############################
        trainer_poisoned_model = model_class(seed=seed,
                                             device=device).setup_graph(attacker.g).build()
        his = trainer_poisoned_model.fit(splits.train_nodes,
                                         splits.val_nodes,
                                         verbose=1,
                                         epochs=3000)
        poisoned_logits = trainer_poisoned_model.predict(target,
                                                         transform="softmax")
        poisoned_classification_statistics = classification_statistics(poisoned_logits,
                                                                       target_label)

        results.append({
            'label': model_label,
            'epsilon': 1.0,
            'n_perturbations': attacker.num_budgets,
            'degree': attacker.num_budgets,
            'perturbed_edges': attacker.edge_flips,
            'target': target_label,
            'node_id': target,
            'evasion': {
                'logits_evasion': list(evasion_logits),
                **evasion_classification_statistics,
                'initial_logits': list(initial_logits),
                **{
                    f'initial_{key}': value
                    for key, value
                    in initial_classification_statistics.items()
                }
            },
            'poisoning': {
                'logits_poisoning': list(poisoned_logits),
                **poisoned_classification_statistics,
            }
        })
    print(eval_surrogate)
    print(eval_victim_model)
    return {
        'results': results
    }
