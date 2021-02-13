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
- cora/citeseer/pubmed
'''
from graphgallery.datasets import Planetoid
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()

graphgallery.set_backend("pyg")

from graphgallery.gallery.nodeclas import SGC
trainer = SGC(graph, device="gpu", seed=123).process(attr_transform="normalize_attr").build()
his = trainer.train(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
results = trainer.test(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
