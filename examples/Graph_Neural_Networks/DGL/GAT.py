#!/usr/bin/env python
# coding: utf-8

import torch
import dgl
import graphgallery

print("GraphGallery version: ", graphgallery.__version__)
print("Torch version: ", torch.__version__)
print("DGL version: ", dgl.__version__)

'''
Load Datasets
- cora/citeseer/pubmed
'''
from graphgallery.datasets import Planetoid
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()

graphgallery.set_backend("dgl")

from graphgallery.gallery.nodeclas import GAT
trainer = GAT(device="gpu", seed=123).setup_graph(graph, attr_transform="normalize_attr").build()
his = trainer.fit(splits.train_nodes, splits.val_nodes, verbose=1, epochs=200)
results = trainer.evaluate(splits.test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
