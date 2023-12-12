import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary


class DataLoader(nn.Module):
    def __init__(self):
        