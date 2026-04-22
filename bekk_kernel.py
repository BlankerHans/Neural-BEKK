import torch
import torch.nn as nn
import torch.nn.functional as F
from vech import vech, unvech


def _as_float_tensor(value):
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return torch.as_tensor(value, dtype=torch.float32)


class BEKKCell(nn.Module):
    """
    State:
      input_size: number of features in input
      n_assets: assumes that the first n_assets features of the input are the returns (epsilons) at time t, rest can be other features
      c_t      : (B, h)
      Sigma_t  : (B, d, d)
    """
    def __init__(self, input_size: int, n_assets: int, hidden_size: int, asym: bool = False, beta: float = 0.9, jitter: float = 1e-6, modulation: str = "vector"):
        super().__init__()
        self.input_size = input_size
        self.d = n_assets
        self.h = hidden_size
        self.m = n_assets * (n_assets + 1) // 2
        self.beta = beta
        self.jitter = jitter
        self.modulation = modulation

        if self.modulation not in {"scalar", "vector", "convex_mixture"}:
            raise ValueError(f"Unknown modulation: {self.modulation}")

        # f_t = sigm(W_f eps + U_f vech(Sigma) + b_f), etc.
        # ggf. noch vech(e_t-1 e_t-1^T) als Input-Feature für die Gates hinzufügen
        self.W_f = nn.Linear(self.input_size, self.h, bias=False)
        self.U_f = nn.Linear(self.m, self.h, bias=True)

        self.W_i = nn.Linear(self.input_size, self.h, bias=False)
        self.U_i = nn.Linear(self.m, self.h, bias=True)

        self.W_c = nn.Linear(self.input_size, self.h, bias=False)
        self.U_c = nn.Linear(self.m, self.h, bias=True)

        if self.modulation == "scalar":
            # Zhao-like modulation: (1 + beta * tanh(w^T tanh(c_t) + b))
            self.w_o = nn.Parameter(torch.zeros(self.h))
            self.b_o = nn.Parameter(torch.zeros(1))
        elif self.modulation == "vector":
            self.W_o = nn.Linear(self.h, self.d, bias=True)
            nn.init.zeros_(self.W_o.weight)
            nn.init.zeros_(self.W_o.bias)
        elif self.modulation == "convex_mixture":
            self.mix_dim = self.h + self.m  # tanh(c_t) plus vech(K_t)

            self.alpha_head = nn.Linear(self.mix_dim, 1, bias=True)
            self.nn_cov_head = nn.Linear(self.mix_dim, self.m, bias=True)

            nn.init.zeros_(self.alpha_head.weight)
            nn.init.constant_(self.alpha_head.bias, -2.0)  # Start eher BEKK-lastig

            nn.init.zeros_(self.nn_cov_head.weight)
            nn.init.zeros_(self.nn_cov_head.bias)

        # BEKK params
        self.C_raw = nn.Parameter(torch.zeros(self.m))
        self.A = nn.Parameter(0.05 * torch.eye(self.d))
        self.B = nn.Parameter(0.90 * torch.eye(self.d))
        self.G = None
        if asym:
            self.G = nn.Parameter(0.05 * torch.eye(self.d))

    def _make_C(self, device, dtype):
        C = unvech(self.C_raw.to(device=device, dtype=dtype), self.d)      # (d,d)
        diag = F.softplus(torch.diagonal(C, dim1=-2, dim2=-1)) + self.jitter
        C = torch.tril(C, diagonal=-1) + torch.diag(diag) # sichert untere Dreiecksstruktur und addiert positiv transformierte Diagonale
        return C

    def forward(self, gate_input_t, eps_t, Sigma_prev, c_prev): # vearbeitet einen Zeitschritt innerhelb einer Sequenz
        """
        e_prev   : (B, d)
        Sigma_prev : (B, d, d)
        c_prev     : (B, h)
        """
        B = eps_t.shape[0]

        # Gates
        s_prev = vech(Sigma_prev)  # (B, m)
        f_t = torch.sigmoid(self.W_f(gate_input_t) + self.U_f(s_prev)) # forget gate
        i_t = torch.sigmoid(self.W_i(gate_input_t) + self.U_i(s_prev)) # input gate
        c_tilde = torch.tanh(self.W_c(gate_input_t) + self.U_c(s_prev)) # candidate cell state
        c_t = f_t * c_prev + i_t * c_tilde # new cell state

        # BEKK kernel K_t (modified hidden state)
        C = self._make_C(eps_t.device, eps_t.dtype)
        CCt = (C @ C.T).unsqueeze(0).expand(B, -1, -1) # C @ C^T ist (d,d), dann (B,d,d) durch expand

        eeT = eps_t.unsqueeze(-1) @ eps_t.unsqueeze(-2)  # (B,d,1) @ (B,1,d) -> (B,d,d)
        term_A = self.A.T.unsqueeze(0) @ eeT @ self.A.unsqueeze(0) # 
        term_B = self.B.T.unsqueeze(0) @ Sigma_prev @ self.B.unsqueeze(0)

        K_t = CCt + term_A + term_B  # (B,d,d), PSD by construction
        if self.G is not None:
            e_neg = torch.clamp(eps_t, max=0.0)
            eeT_neg = e_neg.unsqueeze(-1) @ e_neg.unsqueeze(-2)
            term_G = self.G.T.unsqueeze(0) @ eeT_neg @ self.G.unsqueeze(0)
            K_t = K_t + term_G

        # hidden state Ersatz
        if self.modulation=="scalar":
            # Zhao-like matrix update (scalar positive modulation)
            raw = (torch.tanh(c_t) * self.w_o).sum(dim=-1, keepdim=True) + self.b_o   # (B,1) weighted sum of activated c_t
            m_t = 1.0 + self.beta * torch.tanh(raw)                                   # (B,1), tanh(raw) in (-1,1) -> m_t in (1-beta, 1+beta)
            Sigma_t = m_t.unsqueeze(-1) * K_t                                         # (B,d,d) every element of K_t is scaled by the same positive factor m_t
        elif self.modulation=="vector":
            raw = self.W_o(torch.tanh(c_t))                                           # (B, d) mapping activated c_t to d modulation factors tanh(c_t) @ W_o^T + b_o -> raw (B, d)
            m_t = 1.0 + self.beta * torch.tanh(raw)                                   # (B, d) vector of modulation factors in (1-beta, 1+beta)
            Sigma_t = m_t.unsqueeze(-1) * K_t * m_t.unsqueeze(-2)                     # (B, d, d) diag(m_t) @ K_t @ diag(m_t), every element K_t[i,j] is scaled by m_t[i]*m_t[j]
        elif self.modulation=="convex_mixture":
            k_feat = vech(K_t)                                                         # (B,m)
            mix_feat = torch.cat([torch.tanh(c_t), k_feat], dim=-1)                    # (B,h+m)

            alpha_t = torch.sigmoid(self.alpha_head(mix_feat))                         # (B,1)

            raw_L_nn = self.nn_cov_head(mix_feat)                                      # (B,m)
            L_nn = unvech(raw_L_nn, self.d)                                            # (B,d,d)
            diag_nn = F.softplus(torch.diagonal(L_nn, dim1=-2, dim2=-1)) + self.jitter
            L_nn = torch.tril(L_nn, diagonal=-1) + torch.diag_embed(diag_nn)
            Sigma_nn_t = L_nn @ L_nn.transpose(-1, -2)                                 # (B,d,d)

            alpha = alpha_t.unsqueeze(-1)                                              # (B,1,1)
            Sigma_t = (1.0 - alpha) * K_t + alpha * Sigma_nn_t                         # (B,d,d)


        # numerical cleanup
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-1, -2))
        I = torch.eye(self.d, device=Sigma_t.device, dtype=Sigma_t.dtype).unsqueeze(0)
        Sigma_t = Sigma_t + self.jitter * I

        return Sigma_t, c_t


class BEKKLSTM(nn.Module):
    """
      A LSTM Model that integratedes a BEKK kernel in its recurrence, i.e. the hidden state update is influenced by the BEKK covariance update.
      forward(x) -> (Sigma_last, L_last)

      If return_std is set, the BEKK recursion uses centered returns on the
      original scale: eps_bekk = eps_standardized * return_std * bekk_scale.
      The returned Sigma stays on the standardized model/loss scale.
    """
    def __init__(
        self,
        input_size: int,
        n_assets: int,
        hidden_size: int = 64,
        asym: bool = False,
        beta: float = 0.9,
        jitter: float = 1e-6,
        Sigma0: torch.Tensor = None,
        modulation: str = "vector",
        return_std: torch.Tensor = None,
        return_mean: torch.Tensor = None,
        bekk_scale: float = 1.0,
        sigma0_in_bekk_scale: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_assets = n_assets
        self.hidden_size = hidden_size
        self.jitter = jitter
        self.asym = asym
        self.modulation = modulation
        self.bekk_scale = float(bekk_scale)
        self.sigma0_in_bekk_scale = sigma0_in_bekk_scale
        self.uses_return_rescaling = return_std is not None

        if self.bekk_scale <= 0.0:
            raise ValueError("bekk_scale must be positive")

        self.cell = BEKKCell(input_size=input_size, n_assets=n_assets, hidden_size=hidden_size, asym=asym, beta=beta, jitter=jitter, modulation=modulation)

        if return_std is None:
            self.register_buffer("return_std", torch.ones(n_assets))
        else:
            return_std = _as_float_tensor(return_std)
            if return_std.shape != (n_assets,):
                raise ValueError("return_std must be of shape (n_assets,)")
            if torch.any(return_std <= 0.0):
                raise ValueError("return_std must contain only positive values")
            self.register_buffer("return_std", return_std)

        if Sigma0 is not None:
            Sigma0 = _as_float_tensor(Sigma0)
            assert Sigma0.shape == (n_assets, n_assets), "Sigma0 must be of shape (n_assets, n_assets)"
            self.register_buffer("Sigma0", Sigma0) # sample covariance from training data initialization?
        else:
            self.register_buffer("Sigma0", torch.eye(n_assets))

    def _bekk_scale_vec(self, X):
        return self.return_std.to(device=X.device, dtype=X.dtype) * self.bekk_scale # r_t = eps_t * return_std -> eps_t = r_t / return_std -> eps_t * return_std * 100 z.B. als BEKK scale vector

    def _to_bekk_covariance_scale(self, Sigma, X):
        if not self.uses_return_rescaling:
            return Sigma

        scale = self._bekk_scale_vec(X)
        return Sigma * scale.view(1, -1, 1) * scale.view(1, 1, -1) # diag(scale) @ Sigma @ diag(scale)

    def _from_bekk_covariance_scale(self, Sigma, X):
        if not self.uses_return_rescaling:
            return Sigma

        scale = self._bekk_scale_vec(X)
        return Sigma / (scale.view(1, -1, 1) * scale.view(1, 1, -1)) # diag(1/scale) @ Sigma @ diag(1/scale)

    def _eps_for_bekk(self, X_t):
        eps_t = X_t[:, :self.n_assets]
        if self.uses_return_rescaling:
            # X_t contains standardized zero-mean returns. For BEKK, use the
            # same residuals in their original units, optionally as percentages.
            eps_t = eps_t * self._bekk_scale_vec(X_t)
        return eps_t

    def forward(self, X): # verarbeitet gesamte Sequenz im RNN stlye (T=lookback/sequence length)
        # X: (B,T,input_size), e_t wird aus den ersten n_assets Features gelesen
        B, T, _ = X.shape
        c_t = X.new_zeros(B, self.hidden_size)
        Sigma_t = self.Sigma0.to(device=X.device, dtype=X.dtype).unsqueeze(0).expand(B, -1, -1).clone()
        if self.uses_return_rescaling and not self.sigma0_in_bekk_scale:
            Sigma_t = self._to_bekk_covariance_scale(Sigma_t, X)

        for t in range(T):
            gates_input_t = X[:, t, :] # (B, input_size), alle features
            eps_t = self._eps_for_bekk(gates_input_t) # (B, d), BEKK verarbeitet nur returns bzw. Residuen
            Sigma_t, c_t = self.cell(gates_input_t, eps_t, Sigma_t, c_t)

        Sigma_t = self._from_bekk_covariance_scale(Sigma_t, X)
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-1, -2))
        I = torch.eye(self.n_assets, device=X.device, dtype=X.dtype).unsqueeze(0)
        L_t = torch.linalg.cholesky(Sigma_t + self.jitter * I)
        return Sigma_t, L_t
