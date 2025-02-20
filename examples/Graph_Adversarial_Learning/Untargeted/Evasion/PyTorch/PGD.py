import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import Planetoid

data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)

graph = data.graph
splits = data.split_nodes()
graph.update(node_attr=gf.normalize_attr(graph.node_attr))

# use PyTorch backend
gg.set_backend("torch")

# GPU is recommended
device = "gpu"

################### Surrogate model ############################
trainer = gg.gallery.nodeclas.GCN(device=device, seed=123).setup_graph(graph).build(hids=32)
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)

################### Attacker model ############################
attacker = gg.attack.untargeted.PGD(graph, device=device, seed=None).process(
    trainer, splits.train_nodes, unlabeled_nodes=splits.test_nodes)
attacker.attack(0.05, CW_loss=False)

################### Victim model ############################
# This is a white-box attack
# Before attack
original_result = trainer.evaluate(splits.test_nodes)

# After attack
# reprocess after the graph has changed
trainer.setup_graph(attacker.g)  # important!
perturbed_result = trainer.evaluate(splits.test_nodes)

################### Results ############################
print(f"original prediction {original_result.accuracy:.2%}")
print(f"perturbed prediction {perturbed_result.accuracy:.2%}")
print(
    f"The accuracy has gone down {original_result.accuracy-perturbed_result.accuracy:.2%}"
)

"""original prediction 81.30%
perturbed prediction 74.70%
The accuracy has gone down 6.60%"""
