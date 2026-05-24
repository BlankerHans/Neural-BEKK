import torch
import torch.nn as nn
import torch.nn.functional as F

from vech import vech, unvech


def _as_float_tensor(value):
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return torch.as_tensor(value, dtype=torch.float32)


def _inv_softplus(value, min_value: float = 1e-8):
    value = torch.clamp(value, min=min_value)
    return value + torch.log(-torch.expm1(-value))


def _activation_layer(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class NeuralBekk(nn.Module):
    """
    Neural BEKK(1,1) with optional asymmetric term.

    The first n_assets input features are expected to be centered returns or
    residuals. Sigma0 is interpreted on the standardized model/loss scale unless
    sigma0_in_bekk_scale=True. Internally, returns and Sigma can be mapped to a
    BEKK scale through return_std * bekk_scale.

        Sigma_t = C_t C_t'
                + A_t eps_t eps_t' A_t'
                + G_t Sigma_{t-1} G_t'
                + B_t eta_t eta_t' B_t'      if asym=True

    Naming convention:
      A/a: ARCH shock coefficient
      G/g: GARCH persistence coefficient
      B/b: asymmetric negative-shock coefficient

    bekk_type='diag' uses per-asset persistence parameters. bekk_type='full'
    uses one global persistence parameter per time step and full A/G/B matrices
    with an operator-norm bound. bekk_type='scalar' is reserved but not
    implemented yet.
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int = None,
        *,
        input_size: int = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        activation: str = "tanh",
        dropout: float = 0.0,
        jitter: float = 1e-6,
        recursion_jitter: float = 0.0,
        cholesky_jitter: float = None,
        bekk_type: str = "diag",
        bekk_structure: str = None,
        asym: bool = False,
        Sigma0: torch.Tensor = None,
        return_std: torch.Tensor = None,
        bekk_scale: float = 100.0,
        sigma0_in_bekk_scale: bool = False,
        tau_max: float = 0.995,
        c_delta_scale: float = 0.01,
        init_C_from_sigma0: bool = None,
        init_a: float = 0.05,
        init_g: float = 0.90,
        init_b: float = 0.05,
    ):
        super().__init__()

        if input_size is None:
            input_size = n_features
        if input_size is None:
            raise ValueError("Either n_features or input_size must be provided")

        if bekk_structure is not None:
            bekk_type = bekk_structure
        if bekk_type not in {"diag", "scalar", "full"}:
            raise ValueError("bekk_type must be one of {'diag', 'scalar', 'full'}")
        if bekk_type == "scalar":
            raise NotImplementedError(f"bekk_type='{bekk_type}' is reserved but not implemented yet")

        if n_assets <= 0:
            raise ValueError("n_assets must be positive")
        if input_size < n_assets:
            raise ValueError("input_size must be at least n_assets")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if bekk_scale <= 0.0:
            raise ValueError("bekk_scale must be positive")
        if not 0.0 < tau_max < 1.0:
            raise ValueError("tau_max must be in (0, 1)")
        if jitter < 0.0:
            raise ValueError("jitter must be non-negative")
        if recursion_jitter < 0.0:
            raise ValueError("recursion_jitter must be non-negative")
        if cholesky_jitter is None:
            cholesky_jitter = jitter
        if cholesky_jitter < 0.0:
            raise ValueError("cholesky_jitter must be non-negative")

        if init_C_from_sigma0 is None:
            init_C_from_sigma0 = True

        self.n_assets = n_assets
        self.input_size = input_size
        self.n_features = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.diag_jitter = float(jitter)
        self.recursion_jitter = float(recursion_jitter)
        self.cholesky_jitter = float(cholesky_jitter)
        self.jitter = self.cholesky_jitter
        self.bekk_type = bekk_type
        self.asym = asym
        self.bekk_scale = float(bekk_scale)
        self.sigma0_in_bekk_scale = sigma0_in_bekk_scale
        self.tau_max = float(tau_max)
        self.c_delta_scale = float(c_delta_scale)
        self.uses_return_rescaling = return_std is not None

        self.m = n_assets * (n_assets + 1) // 2 # lower triangular entries of C
        self.n_coeffs = 3 if asym else 2  # ordered as A, G, and optionally B

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if bekk_type == "diag":
            out_size = self.m + n_assets + n_assets * self.n_coeffs
        else:
            out_size = self.m + 1 + self.n_coeffs + self.n_coeffs * n_assets * n_assets
        self.param_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            _activation_layer(activation),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, out_size),
        )

        # Base intercept in raw lower-triangular coordinates. The head predicts
        # small time-varying deltas around this stable BEKK-style initialization.
        self.C_raw = nn.Parameter(torch.zeros(self.m))

        if return_std is None:
            self.register_buffer("return_std", torch.ones(n_assets))
        else:
            return_std = _as_float_tensor(return_std)
            if return_std.shape != (n_assets,):
                raise ValueError("return_std must be of shape (n_assets,)")
            if torch.any(return_std <= 0.0):
                raise ValueError("return_std must contain only positive values")
            self.register_buffer("return_std", return_std)

        if Sigma0 is None:
            Sigma0 = torch.eye(n_assets)
        else:
            Sigma0 = _as_float_tensor(Sigma0)
            if Sigma0.shape != (n_assets, n_assets):
                raise ValueError("Sigma0 must be of shape (n_assets, n_assets)")
        self.register_buffer("Sigma0", Sigma0)

        effective_init_b = init_b if asym else 0.0
        self._init_param_head(init_a=init_a, init_g=init_g, init_b=effective_init_b)

        if init_C_from_sigma0:
            self.initialize_C_from_Sigma0(
                init_a=init_a,
                init_g=init_g,
                init_b=effective_init_b,
            )

    def _init_param_head(self, init_a: float, init_g: float, init_b: float):
        last = self.param_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        coeff_squares = [init_a**2, init_g**2]
        if self.asym:
            coeff_squares.append(init_b**2)

        init_tau = min(sum(coeff_squares), self.tau_max * 0.95)
        raw_tau = torch.logit(torch.tensor(init_tau / self.tau_max))

        weights = torch.tensor(coeff_squares, dtype=torch.float32)
        weights = weights / weights.sum()
        logits = torch.log(torch.clamp(weights, min=1e-12))

        with torch.no_grad():
            if self.bekk_type == "diag":
                tau_start = self.m
                tau_end = tau_start + self.n_assets
                split_start = tau_end

                last.bias[tau_start:tau_end].fill_(raw_tau)
                last.bias[split_start:].copy_(logits.repeat(self.n_assets))
            else:
                tau_start, tau_end, logits_start, logits_end, matrices_start = self._full_param_offsets()

                last.bias[tau_start:tau_end].fill_(raw_tau)
                last.bias[logits_start:logits_end].copy_(logits)

                identity = torch.eye(self.n_assets, dtype=last.bias.dtype).reshape(-1)
                for coeff_idx in range(self.n_coeffs):
                    start = matrices_start + coeff_idx * self.n_assets * self.n_assets
                    end = start + self.n_assets * self.n_assets
                    last.bias[start:end].copy_(identity)

    def _make_C_from_raw(self, raw_C):
        C = unvech(raw_C, self.n_assets)
        diag = F.softplus(torch.diagonal(C, dim1=-2, dim2=-1)) + self.diag_jitter
        return torch.tril(C, diagonal=-1) + torch.diag_embed(diag)

    def initialize_C_from_Sigma0(
        self,
        init_a: float = 0.05,
        init_g: float = 0.90,
        init_b: float = 0.05,
        intercept_floor: float = 1e-4,
        leverage_weight: float = 0.5,
    ):
        with torch.no_grad():
            device = self.C_raw.device
            dtype = self.C_raw.dtype

            sigma0 = self.Sigma0.to(device=device, dtype=dtype)
            if self.uses_return_rescaling and not self.sigma0_in_bekk_scale:
                scale = self.return_std.to(device=device, dtype=dtype) * self.bekk_scale
                sigma0 = sigma0 * scale.view(-1, 1) * scale.view(1, -1)

            sigma0 = 0.5 * (sigma0 + sigma0.T)

            intercept_factor = (
                1.0
                - init_g**2
                - init_a**2
                - leverage_weight * init_b**2
            )
            intercept_factor = max(intercept_factor, intercept_floor)

            C_cov = intercept_factor * sigma0
            C_cov = 0.5 * (C_cov + C_cov.T)

            I = torch.eye(self.n_assets, device=device, dtype=dtype)
            C = torch.linalg.cholesky(C_cov + self.diag_jitter * I)

            C_raw = torch.tril(C, diagonal=-1)
            diag_raw = _inv_softplus(torch.diagonal(C) - self.diag_jitter)
            C_raw = C_raw + torch.diag(diag_raw)
            self.C_raw.copy_(vech(C_raw))

    def _bekk_scale_vec(self, X):
        return self.return_std.to(device=X.device, dtype=X.dtype) * self.bekk_scale

    def _to_bekk_covariance_scale(self, Sigma, X):
        if not self.uses_return_rescaling:
            return Sigma

        scale = self._bekk_scale_vec(X)
        return Sigma * scale.view(1, -1, 1) * scale.view(1, 1, -1)

    def _from_bekk_covariance_scale(self, Sigma, X):
        if not self.uses_return_rescaling:
            return Sigma

        scale = self._bekk_scale_vec(X)
        return Sigma / (scale.view(1, -1, 1) * scale.view(1, 1, -1))

    def _eps_for_bekk(self, X_t):
        eps_t = X_t[:, :self.n_assets]
        if self.uses_return_rescaling:
            eps_t = eps_t * self._bekk_scale_vec(X_t)
        return eps_t

    def _split_diag_params(self, raw_params, return_meta: bool = False):
        batch_size = raw_params.shape[0]

        delta_C = raw_params[:, :self.m]
        raw_tau = raw_params[:, self.m:self.m + self.n_assets]
        raw_logits = raw_params[:, self.m + self.n_assets:]

        base_C = self.C_raw.to(device=raw_params.device, dtype=raw_params.dtype)
        raw_C = base_C.unsqueeze(0) + self.c_delta_scale * delta_C
        C_t = self._make_C_from_raw(raw_C)

        tau_t = self.tau_max * torch.sigmoid(raw_tau) # 0.955 * {in [0, 1]} < 1, shape: (batch_size, n_assets)
        logits_t = raw_logits.view(batch_size, self.n_assets, self.n_coeffs) # shape: (batch_size, n_assets, n_coeffs)
        weights_t = torch.softmax(logits_t, dim=-1) # shape: (batch_size, n_assets, n_coeffs), each slice weights_t[:, i, :] is a valid probability distribution over the n_coeffs coefficients for asset i
        coeffs_t = torch.sqrt(torch.clamp(tau_t.unsqueeze(-1) * weights_t, min=1e-12)) # shape: (batch_size, n_assets, n_coeffs), each element coeffs_t[:, i, j] is the scale for coefficient j of asset i, and the sum of squares of coefficients for each asset i is tau_t[:, i]

        a_t = coeffs_t[:, :, 0] # shape: (batch_size, n_assets), each element a_t[:, i] is the scale for the ARCH shock coefficient of asset i
        g_t = coeffs_t[:, :, 1] # shape: (batch_size, n_assets), each element g_t[:, i] is the scale for the GARCH persistence coefficient of asset i
        b_t = coeffs_t[:, :, 2] if self.asym else None # shape: (batch_size, n_assets) if asym=True, each element b_t[:, i] is the scale for the asymmetric negative-shock coefficient of asset i

        if return_meta:
            return C_t, a_t, g_t, b_t, tau_t, weights_t
        return C_t, a_t, g_t, b_t

    def _full_param_offsets(self):
        tau_start = self.m
        tau_end = tau_start + 1
        logits_start = tau_end
        logits_end = logits_start + self.n_coeffs
        matrices_start = logits_end
        return tau_start, tau_end, logits_start, logits_end, matrices_start

    def _normalize_operator_norm(self, matrices):
        # For full BEKK, stationarity is controlled through a sufficient
        # operator-norm bound. Dividing by max(||M||_2, 1) keeps learned
        # directions unchanged when already contractive and projects only
        # expansive directions back to ||Q||_2 <= 1.
        op_norm = torch.linalg.matrix_norm(
            matrices,
            ord=2,
            dim=(-2, -1),
            keepdim=True,
        )
        scale = torch.clamp(op_norm, min=1.0)
        normalized = matrices / scale
        return normalized, op_norm.squeeze(-1).squeeze(-1)

    def _split_full_params(self, raw_params, return_meta: bool = False):
        batch_size = raw_params.shape[0]
        tau_start, tau_end, logits_start, logits_end, matrices_start = self._full_param_offsets()

        delta_C = raw_params[:, :self.m]
        raw_tau = raw_params[:, tau_start:tau_end]
        raw_logits = raw_params[:, logits_start:logits_end]
        raw_matrices = raw_params[:, matrices_start:]

        base_C = self.C_raw.to(device=raw_params.device, dtype=raw_params.dtype)
        raw_C = base_C.unsqueeze(0) + self.c_delta_scale * delta_C
        C_t = self._make_C_from_raw(raw_C)

        tau_t = self.tau_max * torch.sigmoid(raw_tau)
        weights_t = torch.softmax(raw_logits, dim=-1)
        coeff_scales = torch.sqrt(torch.clamp(tau_t * weights_t, min=1e-12))

        raw_matrices = raw_matrices.view(
            batch_size,
            self.n_coeffs,
            self.n_assets,
            self.n_assets,
        )
        directions_t, raw_operator_norms = self._normalize_operator_norm(raw_matrices)

        # The scale allocation below gives
        # ||A_t||_2^2 + ||G_t||_2^2 + ||B_t||_2^2 <= tau_t < tau_max.
        # This is stricter than the exact BEKK spectral-radius condition, but
        # it is cheap, differentiable, and guarantees a stable recursion.
        matrices_t = coeff_scales.view(batch_size, self.n_coeffs, 1, 1) * directions_t
        operator_norms_t = coeff_scales * torch.clamp(raw_operator_norms, max=1.0)

        A_t = matrices_t[:, 0]
        G_t = matrices_t[:, 1]
        B_t = matrices_t[:, 2] if self.asym else None

        if return_meta:
            return C_t, A_t, G_t, B_t, tau_t, weights_t, operator_norms_t, raw_operator_norms
        return C_t, A_t, G_t, B_t

    def forward(self, X, return_params: bool = False):
        batch_size, T, _ = X.shape

        h_seq, _ = self.gru(X)
        raw_seq = self.param_head(h_seq)

        Sigma_t = (
            self.Sigma0
            .to(device=X.device, dtype=X.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        if self.uses_return_rescaling and not self.sigma0_in_bekk_scale:
            Sigma_t = self._to_bekk_covariance_scale(Sigma_t, X)

        I = torch.eye(self.n_assets, device=X.device, dtype=X.dtype).unsqueeze(0)

        if return_params:
            C_list = []
            a_list = []
            g_list = []
            b_list = []
            tau_list = []
            weight_list = []
            operator_norm_list = []
            raw_operator_norm_list = []

        for t in range(T):
            eps_t = self._eps_for_bekk(X[:, t, :])
            if self.bekk_type == "diag":
                if return_params:
                    C_t, a_t, g_t, b_t, tau_t, weights_t = self._split_diag_params(
                        raw_seq[:, t, :],
                        return_meta=True,
                    )
                else:
                    C_t, a_t, g_t, b_t = self._split_diag_params(raw_seq[:, t, :])

                ae = a_t * eps_t
                term_A = ae.unsqueeze(-1) @ ae.unsqueeze(-2)
                term_G = Sigma_t * g_t.unsqueeze(-1) * g_t.unsqueeze(-2)

                Sigma_t = C_t @ C_t.transpose(-1, -2) + term_A + term_G

                if self.asym:
                    eta_t = torch.clamp(eps_t, max=0.0)
                    be = b_t * eta_t
                    term_B = be.unsqueeze(-1) @ be.unsqueeze(-2)
                    Sigma_t = Sigma_t + term_B

                if return_params:
                    C_list.append(C_t)
                    a_list.append(a_t)
                    g_list.append(g_t)
                    if self.asym:
                        b_list.append(b_t)
                    tau_list.append(tau_t)
                    weight_list.append(weights_t)
            else:
                if return_params:
                    (
                        C_t,
                        A_t,
                        G_t,
                        B_t,
                        tau_t,
                        weights_t,
                        operator_norms_t,
                        raw_operator_norms_t,
                    ) = self._split_full_params(
                        raw_seq[:, t, :],
                        return_meta=True,
                    )
                else:
                    C_t, A_t, G_t, B_t = self._split_full_params(raw_seq[:, t, :])

                eps_col = eps_t.unsqueeze(-1)
                eeT = eps_col @ eps_col.transpose(-1, -2)
                term_A = A_t @ eeT @ A_t.transpose(-1, -2)
                term_G = G_t @ Sigma_t @ G_t.transpose(-1, -2)

                Sigma_t = C_t @ C_t.transpose(-1, -2) + term_A + term_G

                if self.asym:
                    eta_col = torch.clamp(eps_t, max=0.0).unsqueeze(-1)
                    etaT = eta_col @ eta_col.transpose(-1, -2)
                    term_B = B_t @ etaT @ B_t.transpose(-1, -2)
                    Sigma_t = Sigma_t + term_B

                if return_params:
                    C_list.append(C_t)
                    a_list.append(A_t)
                    g_list.append(G_t)
                    if self.asym:
                        b_list.append(B_t)
                    tau_list.append(tau_t)
                    weight_list.append(weights_t)
                    operator_norm_list.append(operator_norms_t)
                    raw_operator_norm_list.append(raw_operator_norms_t)

            Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-1, -2))
            if self.recursion_jitter > 0.0:
                Sigma_t = Sigma_t + self.recursion_jitter * I

        Sigma_t = self._from_bekk_covariance_scale(Sigma_t, X)
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-1, -2))
        Sigma_t = Sigma_t + self.cholesky_jitter * I

        L_t = torch.linalg.cholesky(Sigma_t)

        if not return_params:
            return Sigma_t, L_t

        C_seq = torch.stack(C_list, dim=1)
        tau_seq = torch.stack(tau_list, dim=1)
        weight_seq = torch.stack(weight_list, dim=1)

        if self.bekk_type == "diag":
            a_seq = torch.stack(a_list, dim=1)
            g_seq = torch.stack(g_list, dim=1)
            b_seq = torch.stack(b_list, dim=1) if self.asym else None

            params = {
                "C": C_seq,
                "A": torch.diag_embed(a_seq),
                "G": torch.diag_embed(g_seq),
                "B": torch.diag_embed(b_seq) if self.asym else None,
                "a": a_seq,
                "g": g_seq,
                "b": b_seq,
                "tau": tau_seq,
                "weights": weight_seq,
            }
        else:
            A_seq = torch.stack(a_list, dim=1)
            G_seq = torch.stack(g_list, dim=1)
            B_seq = torch.stack(b_list, dim=1) if self.asym else None

            params = {
                "C": C_seq,
                "A": A_seq,
                "G": G_seq,
                "B": B_seq,
                "a": None,
                "g": None,
                "b": None,
                "tau": tau_seq,
                "weights": weight_seq,
                "operator_norms": torch.stack(operator_norm_list, dim=1),
                "raw_operator_norms": torch.stack(raw_operator_norm_list, dim=1),
            }

        return Sigma_t, L_t, params


NeuralBEKK = NeuralBekk


__all__ = ["NeuralBekk", "NeuralBEKK"]
