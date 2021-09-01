import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import torch
import scipy.sparse as sp
import json
from pathlib import Path


def to_symmetric_scipy(adjacency: sp.csr_matrix):
    sym_adjacency = (adjacency + adjacency.T).astype(bool).astype(float)

    sym_adjacency.tocsr().sort_indices()

    return sym_adjacency


def prep_graph(name: str,
               make_undirected: bool = True,
               dataset_root: str = 'datasets'):
    """Prepares and normalizes the desired dataset

    Parameters
    ----------
    name : str
        Name of the data set. One of: `cora_ml`, `citeseer`, `pubmed`
    device : Union[int, torch.device]
        `cpu` or GPU id, by default 0
    binary_attr : bool, optional
        If true the attributes are binarized (!=0), by default False
    dataset_root : str, optional
        Path where to find/store the dataset, by default "datasets"
    Returns
    -------
    Tuple[torch.Tensor, torch_sparse.SparseTensor, torch.Tensor]
        dense attribute tensor, sparse adjacency matrix (normalized) and labels tensor.
    """
    split = None

    pyg_dataset = PygNodePropPredDataset(root=dataset_root, name=name)

    data = pyg_dataset[0]

    if hasattr(data, '__num_nodes__'):
        num_nodes = data.__num_nodes__
    else:
        num_nodes = data.num_nodes

    if hasattr(pyg_dataset, 'get_idx_split'):
        split = pyg_dataset.get_idx_split()
    else:
        split = dict(
            train=data.train_mask.nonzero().squeeze(),
            valid=data.val_mask.nonzero().squeeze(),
            test=data.test_mask.nonzero().squeeze()
        )

    # converting to numpy arrays, so we don't have to handle different
    # array types (tensor/numpy/list) later on.
    # Also we need numpy arrays because Numba cant determine type of torch.Tensor
    split = {k: v.numpy() for k, v in split.items()}

    edge_index = data.edge_index.cpu()
    if data.edge_attr is None:
        edge_weight = torch.ones(edge_index.size(1))
    else:
        edge_weight = data.edge_attr
    edge_weight = edge_weight.cpu()

    adj = sp.csr_matrix((edge_weight, edge_index), (num_nodes, num_nodes))

    del edge_index
    del edge_weight

    # make unweighted
    adj.data = np.ones_like(adj.data)

    if make_undirected:
        adj = to_symmetric_scipy(adj)

    attr = data.x.cpu().numpy()

    labels = data.y.squeeze().cpu().numpy()

    return attr, adj, labels, split


def split(labels, n_per_class=20, seed=None):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [num_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    seed: int
        Seed

    Returns
    -------
    split_train: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test array-like [num_nodes - 2*n_per_class * nc]
        The indices of the test nodes
    """
    if seed is not None:
        np.random.seed(seed)
    nc = labels.max() + 1

    split_train, split_val = [], []
    for label in range(nc):
        perm = np.random.permutation((labels == label).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])

    split_train = np.random.permutation(np.concatenate(split_train))
    split_val = np.random.permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(np.arange(len(labels)),
                              np.concatenate((split_train, split_val)))

    return split_train, split_val, split_test


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


def poisoning_classification_statistics(logits_clean, logits_poisoned, label):
    logit_target = logits_poisoned[label]
    sorted = logits_poisoned.argsort()
    logit_best_non_target = (logits_poisoned[sorted[sorted != label][-1]])
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


seed = 1
gg.set_backend("th")
torch.manual_seed(seed)
np.random.seed(seed)
data = NPZDataset('cora_ml',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
splits = data.split_nodes(random_state=15)
splits.train_nodes, splits.val_nodes, splits.test_nodes = split(
    graph.node_label, seed=seed)

################### Surrogate model ############################
trainer_surrogate = gg.gallery.nodeclas.SGC(
    seed=seed).setup_graph(graph, K=2).build()
his = trainer_surrogate.fit(splits.train_nodes,
                            splits.val_nodes,
                            dropout=0.0,
                            verbose=3,
                            epochs=3000)

eval_surrogate = trainer_surrogate.evaluate(splits.test_nodes)
print(eval_surrogate)

targets = [2259, 1787, 1254, 1933, 264, 84, 1063, 86, 82, 466, 1613, 732, 2809, 999, 1449, 2492, 1832, 1827, 2555,
           642, 984, 1207, 710, 1511, 1523, 1556, 1059, 2284, 2194, 380, 266, 1508, 205, 867, 1718, 2096, 312, 173, 1234, 699]
results = []
trainer_attacked_model = gg.gallery.nodeclas.SGC(
    seed=seed).setup_graph(graph).build()
his = trainer_attacked_model.fit(splits.train_nodes,
                                 splits.val_nodes,
                                 dropout=0.0,
                                 verbose=3,
                                 epochs=3000)

for target in targets:
    ################### Attacker model ############################
    attacker = gg.attack.targeted.SGA(
        graph, seed=seed).process(trainer_surrogate)
    attacker.attack(target)

    ################### Victim model ############################
    # Before attack
    target_label = graph.node_label[target]

    original_predict = trainer_attacked_model.predict(
        target, transform="softmax")
    original_classification_statistics = classification_statistics(
        original_predict, target_label)

    trainer = trainer.setup_graph(attacker.g)
    original_predict_attacked = trainer.predict(target, transform="softmax")
    original_classification_statistics_attacked = classification_statistics(
        original_predict_attacked, target_label)

    # After attack
    trainer = gg.gallery.nodeclas.GCN(
        seed=seed).setup_graph(attacker.g).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=3,
                      epochs=3000)
    perturbed_predict = trainer.predict(target, transform="softmax")
    perturbed_classification_statistics = classification_statistics(
        perturbed_predict, target_label)

    results.append({
        'label': "Vanilla GCN",
        'epsilon': 1.0,
        'n_perturbations': attacker.num_budgets,
        'degree': attacker.num_budgets,
        'target': target_label,
        'node_id': target,
        'initial': {
            'initial_logits': list(original_predict),
            **{
                f'initial_{key}': value
                for key, value
                in original_classification_statistics.items()
            }

        },
        'evasion': {
            'logits_evasion': list(original_predict_attacked),
            **original_classification_statistics_attacked,
        },
        'poisoning': {
            'logits_poisoning': list(perturbed_predict),
            **perturbed_classification_statistics,
        }
    })

################### Results ############################
print(results)
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "results.json", "w")as f:
    f.write(json.dumps(results, indent=4))

print("original prediction", original_predict)
print("perturbed prediction", perturbed_predict)
print(f"The True label of node {target} is {target_label}.")
print(
    f"The probability of prediction has gone down {original_predict[target_label]-perturbed_predict[target_label]}"
)
"""original prediction [0.0053769  0.0066478  0.9277275  0.02925558 0.02184986 0.00333208
 0.00581014]
perturbed prediction [0.00911093 0.00466003 0.32176667 0.63783115 0.00680091 0.0019819
 0.01784842]
The True label of node 1 is 2.
The probability of prediction has gone down 0.6059608459472656"""
