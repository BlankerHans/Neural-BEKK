import torch
import torch.nn as nn
import torch.nn.functional as F
from vech import unvech, vech

class BEKKCell(nn.Module):
    def __init__(self, n_assets: int, hidden_size: int):
        super().__init__()
        self.n_assets = n_assets
        self.hidden_size = hidden_size
        
        self.state_dim = n_assets * (n_assets + 1) // 2
        

        # Gates: r_{t-1} und vech(Sigma_{t-1}) (i.e. letzter hidden state)