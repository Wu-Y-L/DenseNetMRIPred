
import torchvision
import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initiate_dense_model(class_names, device : torch.device):
    weights = torchvision.models.DenseNet201_Weights.DEFAULT
    backbone = torchvision.models.densenet201(weights = weights).to(device)

    backbone.classifier = nn.Sequential(
    nn.Linear(in_features= 1920, out_features= 128, bias = True),
    nn.Dropout(p = 0.3, inplace= True),
    nn.Linear(in_features= 128, out_features = 64, bias = True),
    nn.Dropout(p = 0.3, inplace= True),
    nn.Linear(in_features= 64, out_features = len(class_names))
    ).to(device)

    for params in backbone.features.denseblock1.parameters():
        params.requires_grad = False

    return backbone



