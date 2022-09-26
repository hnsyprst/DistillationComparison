import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Search the network's children for the given layer and return its index
def get_index_from_layer(network, layer):
    return list(dict(network.named_children()).values()).index(layer)

# Search the network's children for the given layer name and return its index
def get_index_from_name(network, name):
    return list(dict(network.named_children()).keys()).index(name)

# Search the network's children for the given index and return the layer name
def get_name_from_index(network, index):
    return list(network.named_children())[index][0]