#!/usr/bin/env python
# coding: utf-8

import torch
import graphgallery

print("GraphGallery version: ", graphgallery.__version__)
print("Torch version: ", torch.__version__)

'''
Load Datasets
- cora/citeseer/pubmed
'''
from graphgallery.datasets import Planetoid
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()

graphgallery.set_backend("pytorch")

from graphgallery.gallery.nodeclas import GraphMLP
# Cora
trainer = GraphMLP(device="gpu", seed=123).setup_graph(graph, attr_transform="normalize_attr").build(tau=2, alpha=10.0)
# # Citeseer
# trainer = GraphMLP(device="gpu", seed=123).setup_graph(graph, attr_transform="normalize_attr").build(tau=0.5, alpha=1.0)
# # Pubmed
# trainer = GraphMLP(device="gpu", seed=123).setup_graph(graph, attr_transform="normalize_attr").build(tau=1, alpha=100)

his = trainer.fit(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
results = trainer.evaluate(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
