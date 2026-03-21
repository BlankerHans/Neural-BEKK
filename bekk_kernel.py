import torch
import torch.nn as nn
import torch.nn.functional as F
from vech import vech, unvech


class BEKKCell(nn.Module):
    """
    State:
      c_t      : (B, h)
      Sigma_t  : (B, d, d)
    """
    def __init__(self, n_assets: int, hidden_size: int, asym: bool = False, beta: float = 0.9, jitter: float = 1e-6):
        super().__init__()
        self.d = n_assets
        self.h = hidden_size
        self.m = n_assets * (n_assets + 1) // 2
        self.beta = beta
        self.jitter = jitter

        # f_t = sigm(W_f eps + U_f vech(Sigma) + b_f), etc.
        # ggf. noch vech(e_t-1 e_t-1^T) als Input-Feature für die Gates hinzufügen
        self.W_f = nn.Linear(self.d, self.h, bias=False)
        self.U_f = nn.Linear(self.m, self.h, bias=True)

        self.W_i = nn.Linear(self.d, self.h, bias=False)
        self.U_i = nn.Linear(self.m, self.h, bias=True)

        self.W_c = nn.Linear(self.d, self.h, bias=False)
        self.U_c = nn.Linear(self.m, self.h, bias=True)

        # Zhao-like modulation: (1 + beta * tanh(w^T tanh(c_t) + b))
        self.w_o = nn.Parameter(torch.zeros(self.h))
        self.b_o = nn.Parameter(torch.zeros(1))

        # BEKK params
        self.C_raw = nn.Parameter(torch.zeros(self.m))
        self.A = nn.Parameter(0.05 * torch.eye(self.d))
        self.B = nn.Parameter(0.90 * torch.eye(self.d))
        if asym:
            self.G = nn.Parameter(0.05 * torch.eye(self.d))

    def _make_C(self, device, dtype):
        C = unvech(self.C_raw.to(device=device, dtype=dtype), self.d)      # (d,d)
        diag = F.softplus(torch.diagonal(C, dim1=-2, dim2=-1)) + self.jitter
        C = torch.tril(C, diagonal=-1) + torch.diag(diag) # sichert untere Dreiecksstruktur und addiert positiv transformierte Diagonale
        return C

    def forward(self, eps_prev, Sigma_prev, c_prev):
        """
        e_prev   : (B, d)
        Sigma_prev : (B, d, d)
        c_prev     : (B, h)
        """
        B = eps_prev.shape[0]

        # Gates
        s_prev = vech(Sigma_prev)  # (B, m)
        f_t = torch.sigmoid(self.W_f(eps_prev) + self.U_f(s_prev))
        i_t = torch.sigmoid(self.W_i(eps_prev) + self.U_i(s_prev))
        c_tilde = torch.tanh(self.W_c(eps_prev) + self.U_c(s_prev))
        c_t = f_t * c_prev + i_t * c_tilde

        # BEKK kernel K_t
        C = self._make_C(eps_prev.device, eps_prev.dtype)
        CCt = (C @ C.T).unsqueeze(0).expand(B, -1, -1) # C @ C^T ist (d,d), dann (B,d,d) durch expand

        eeT = eps_prev.unsqueeze(-1) @ eps_prev.unsqueeze(-2)  # (B,d,1) @ (B,1,d) -> (B,d,d)
        term_A = self.A.T.unsqueeze(0) @ eeT @ self.A.unsqueeze(0) # 
        term_B = self.B.T.unsqueeze(0) @ Sigma_prev @ self.B.unsqueeze(0)

        K_t = CCt + term_A + term_B  # (B,d,d), PSD by construction
        if self.G is not None:
            e_pos = torch.clamp(eps_prev, min=0.0)
            eeT_pos = e_pos.unsqueeze(-1) @ e_pos.unsqueeze(-2)
            term_G = self.G.T.unsqueeze(0) @ eeT_pos @ self.G.unsqueeze(0)
            K_t = K_t + term_G

        # Zhao-like matrix update (scalar positive modulation)
        raw = (torch.tanh(c_t) * self.w_o).sum(dim=-1, keepdim=True) + self.b_o  # (B,1)
        m_t = 1.0 + self.beta * torch.tanh(raw)                                   # (B,1), > 1-beta
        Sigma_t = m_t.unsqueeze(-1) * K_t                                          # (B,d,d)

        # numerical cleanup
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-1, -2))
        I = torch.eye(self.d, device=Sigma_t.device, dtype=Sigma_t.dtype).unsqueeze(0)
        Sigma_t = Sigma_t + self.jitter * I

        return Sigma_t, c_t


class BEKKLSTM(nn.Module):
    """
      A LSTM Model that integratedes a BEKK kernel in its recurrence, i.e. the hidden state update is influenced by the BEKK covariance update.
      forward(x) -> (Sigma_last, L_last)
    """
    def __init__(self, input_size: int, n_assets: int, hidden_size: int = 64, asym: bool = False, beta: float = 0.9, jitter: float = 1e-6, Sigma0: torch.Tensor = None):
        super().__init__()
        self.input_size = input_size
        self.n_assets = n_assets
        self.hidden_size = hidden_size
        self.jitter = jitter
        self.asym = asym

        self.cell = BEKKCell(n_assets=n_assets, hidden_size=hidden_size, asym=asym, beta=beta, jitter=jitter)
        if Sigma0 is not None:
            assert Sigma0.shape == (n_assets, n_assets), "Sigma0 must be of shape (n_assets, n_assets)"
            self.register_buffer("Sigma0", Sigma0) # sample covariance from training data initialization?
        else:
            self.register_buffer("Sigma0", torch.eye(n_assets))

    def forward(self, x):
        # x: (B,T,input_size), e_t wird aus den ersten n_assets Features gelesen
        B, T, _ = x.shape
        c_t = x.new_zeros(B, self.hidden_size)
        Sigma_t = self.Sigma0.to(dtype=x.dtype).unsqueeze(0).expand(B, -1, -1).clone()

        for t in range(T):
            e_t = x[:, t, :self.n_assets]
            Sigma_t, c_t = self.cell(e_t, Sigma_t, c_t)

        I = torch.eye(self.n_assets, device=x.device, dtype=x.dtype).unsqueeze(0)
        L_t = torch.linalg.cholesky(Sigma_t + self.jitter * I)
        return Sigma_t, L_t
