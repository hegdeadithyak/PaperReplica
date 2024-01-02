import torch 
import torch.nn as nn




class L2Norm(nn.Module):
    def __init__(self,n_channels,scale):
        self.input = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weights = nn.parameters(torch.tensor(n_channels))
        self.reset_parameters()

        def reset_parameteres():
            nn.init.constant_(self.weights,self.gamma)


        def feature_extraction(self.
