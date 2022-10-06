import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Search the network's children for the given layer and return its index
def get_index_from_layer(network, layer, recursive=True):
    if recursive:
        return list(network.modules()).index(layer)
    else:
        return list(network.children()).index(layer)

# Search the network's children for the given layer name and return its index
def get_index_from_name(network, name, recursive=True):
    if recursive:
        return list(dict(network.named_modules()).keys()).index(name)
    else:
        return list(dict(network.named_children()).keys()).index(name)

# Search the network's children for the given index and return the layer name
def get_name_from_index(network, index, recursive=True):
    if recursive:
        return list(network.named_modules())[index][0]
    else:
        return list(network.named_children())[index][0]

def get_name_from_layer(network, layer):
    return get_name_from_index(network, get_index_from_layer(network, layer))

def get_layer_in_module_from_index(module, index):
    return list(module._modules.values())[index]

# A layer that performs no action
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x