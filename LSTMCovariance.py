import torch
import torch.nn as nn
import torch.nn.functional as F
from vech import unvech
import numpy as np

class LSTMCovariance(nn.Module):
    def __init__(self, input_size: int, n_assets: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.n_assets = n_assets
        self.lower_tri_size = n_assets * (n_assets + 1) // 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, self.lower_tri_size)

    def forward(self, X: torch.Tensor):
        # x: (batch, lookback, d)
        out, _ = self.lstm(X) # BxTxh
        h_last = out[:, -1, :]  # Bxh letzter hidden state
        raw_covar = self.fc(h_last) # Bx(d(d+1)/2) raw lower-triangular elements

        L = unvech(raw_covar, self.n_assets) # Bxdxd, lower-triangular
        diag = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-4 # pos. Diagonale erzwingen (Bedingung für Cholesky-Zerlegung)
        L = torch.tril(L, diagonal=-1) + torch.diag_embed(diag) # sichert untere Dreiecksstruktur und addiert positiv transformierte Diagonale
        sigma_t = L @ L.transpose(-1, -2) # Bxdxd, symmetrisch und PSD durch Konstruktion
        I = torch.eye(self.n_assets, device=L.device, dtype=L.dtype).unsqueeze(0) # (1,d,d) Einheitsmatrix für numerische Stabilität
        sigma_t = sigma_t + 1e-4 * I # jitter für numerische Stabilität
        
        return sigma_t, L