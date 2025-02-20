#!/usr/bin/env python
# coding: utf-8

import graphgallery
import tensorflow as tf

graphgallery.set_memory_growth()

print("GraphGallery version: ", graphgallery.__version__)
print("TensorFlow version: ", tf.__version__)

'''
Load Datasets
- cora/citeseer/pubmed
'''
from graphgallery.datasets import Planetoid
data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)
graph = data.graph
splits = data.split_nodes()

from graphgallery.gallery.nodeclas import GraphSAGE
trainer = GraphSAGE(device="gpu", seed=123).setup_graph(graph, attr_transform="normalize_attr").build()
his = trainer.fit(splits.train_nodes, splits.val_nodes, verbose=1, epochs=100)
results = trainer.evaluate(splits.test_nodes)
