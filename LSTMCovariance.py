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

    def forward(self, x: torch.Tensor):
        # x: (batch, lookback, d)
        out, _ = self.lstm(x) # BxTxh
        h_last = out[:, -1, :]  # Bxh letzter hidden state
        raw_covar = self.fc(h_last) # Bx(d(d+1)/2) raw lower-triangular elements

        L = unvech(raw_covar, self.n_assets) # Bxdxd, lower-triangular
        diag = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-4
        L = torch.tril(L, diagonal=-1) + torch.diag_embed(diag)
        sigma_t = L @ L.transpose(-1, -2)
        I = torch.eye(self.n_assets, device=L.device, dtype=L.dtype).unsqueeze(0)
        sigma_t = sigma_t + 1e-4 * I
        
        return sigma_t, L


def gaussian_nll(y: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    y: (batch, d)
    chol: (batch, d, d), lower-triangular Cholesky factor of Sigma

    Returns per-sample negative log-likelihood for N(0, Sigma).
    """
    d = y.shape[-1] # 
    y_col = y.unsqueeze(-1) # Bxdx1

    # Solve Sigma^{-1}y via two triangular solves: L z = y, L^T x = z
    z = torch.linalg.solve_triangular(L, y_col, upper=False) # z = L^{-1} y
    precision_y = torch.linalg.solve_triangular(L.transpose(-1, -2), z, upper=True).squeeze(-1)
    mahal = (y * precision_y).sum(dim=-1) # y^T Sigma^{-1} y

    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1) # log(det(Sigma)) = log(det(L @ L^T)) = log(det(L)^2) = 2*log(det(L)) = 2*sum(log(diag(L)))
    const = d * np.log(2.0 * np.pi)

    return 0.5 * (const + logdet + mahal)

def student_nll(y: torch.Tensor, L: torch.Tensor, nu: float) -> torch.Tensor:
    """
    y: (batch, d)
    chol: (batch, d, d), lower-triangular Cholesky factor of Sigma
    nu: degrees of freedom

    Returns per-sample negative log-likelihood for multivariate Student-t with mean 0, scale Sigma, and nu degrees of freedom.
    """
    d = y.shape[-1]
    y_col = y.unsqueeze(-1)

    z = torch.linalg.solve_triangular(L, y_col, upper=False) # z = L^{-1} y
    precision_y = torch.linalg.solve_triangular(L.transpose(-1, -2), z, upper=True).squeeze(-1) # x = L^{-T} z = L^{-T} L^{-1} y = Sigma^{-1} y
    mahal = (y * precision_y).sum(dim=-1) # y^T Sigma^{-1} y

    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1) # log(det(Sigma)) = 2*sum(log(diag(L)))
    
    const = (
        torch.lgamma((nu + d) / 2.0) - torch.lgamma(nu / 2.0) - 
        0.5 * d * np.log(nu * np.pi) - logdet
    )

    return -const + 0.5 * (nu + d) * torch.log1p(mahal / nu)