#!/usr/bin/env python
# coding: utf-8

import torch
import graphgallery
import torch_geometric

print("GraphGallery version: ", graphgallery.__version__)
print("Torch version: ", torch.__version__)
print("Torch_Geometric version: ", torch_geometric.__version__)

'''
Load Datasets
Synthetic node classification dataset from PDN: https://github.com/benedekrozemberczki/PDN
'''
from graphgallery.datasets import NPZDataset
data = NPZDataset('pdn', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()

graphgallery.set_backend("pyg")

from graphgallery.gallery.nodeclas import PDN
trainer = PDN(device="gpu", seed=123).setup_graph(graph).build()
his = trainer.fit(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
results = trainer.evaluate(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
