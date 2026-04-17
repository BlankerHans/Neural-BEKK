import torch
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def train_val_test_split(df, train_size=0.7, val_size=0.15):
    n = len(df)
    test_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    train_df = df.iloc[:test_end]
    val_df = df.iloc[test_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df


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
    L: (batch, d, d), lower-triangular Cholesky factor of Sigma (covariance matrix)
    nu: degrees of freedom (scalar float or learnable tensor, must be > 2)

    Returns per-sample negative log-likelihood for multivariate Student-t with mean 0, scale S_t, and nu degrees of freedom.
    """

    nu = torch.as_tensor(nu, dtype=y.dtype, device=y.device)
    scale_factor = (nu - 2.0) / nu
    L_scale = scale_factor**0.5 * L # L_scale ist jetzt die Cholesky-Zerlegung der Scale-Matrix, folgt aus Sigma_t = nu/(nu-2) * S_t

    d = y.shape[-1]
    y_col = y.unsqueeze(-1)

    z = torch.linalg.solve_triangular(L_scale, y_col, upper=False) # z = L_scale^{-1} y
    precision_y = torch.linalg.solve_triangular(L_scale.transpose(-1, -2), z, upper=True).squeeze(-1) # x = L_scale^{-T} z = L_scale^{-T} L_scale^{-1} y = S_t^{-1} y
    mahal = (y * precision_y).sum(dim=-1) # y^T S_t^{-1} y

    logdet = 2.0 * torch.log(torch.diagonal(L_scale, dim1=-2, dim2=-1)).sum(dim=-1) # log(det(S_t)) = 2*sum(log(diag(L_scale)))

    pi = torch.pi
    
    const = (
        torch.lgamma((nu + d) / 2.0) - torch.lgamma(nu / 2.0) - 
        0.5 * d * torch.log(nu * pi) - 0.5 * logdet
    )

    return -const + 0.5 * (nu + d) * torch.log1p(mahal / nu)

class StudentTLoss(torch.nn.Module):
    def __init__(self, init_nu=8.0, min_nu=2.01, max_nu=100.0):
        super().__init__()
        self.min_nu = float(min_nu)
        self.max_nu = float(max_nu)

        p = (init_nu - self.min_nu) / (self.max_nu - self.min_nu)
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        self.raw_nu = torch.nn.Parameter(torch.logit(torch.tensor(p)))

    @property
    def nu(self):
        return self.min_nu + (self.max_nu - self.min_nu) * torch.sigmoid(self.raw_nu)

    def forward(self, y, L):
        return student_nll(y, L, nu=self.nu)


def train_covariance_model(
    model,
    train_loader,
    val_loader,
    loss_fn=gaussian_nll,
    loss_kwargs=None,
    epochs=120,
    lr=1e-3,
    device="cpu",
    scheduler_type="plateau",   # "plateau", "cosine", None
    plateau_factor=0.5,
    plateau_patience=20,
    plateau_threshold=1e-3,
    min_lr=1e-6,
    verbose=True,
):
    start_time = time.perf_counter()

    if loss_kwargs is None:
        loss_kwargs = {}

    model.to(device)

    if isinstance(loss_fn, torch.nn.Module):
        loss_fn.to(device)
        loss_params = list(loss_fn.parameters())
    else:
        loss_params = []

    param_groups = [
        {"params": model.parameters(), "weight_decay": 1e-3},
    ]

    if loss_params:
        param_groups.append({
            "params": loss_params,
            "weight_decay": 0.0,
        })

    opt = torch.optim.AdamW(param_groups, lr=lr)


    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            opt, mode="min", factor=plateau_factor, patience=plateau_patience, threshold=plateau_threshold, min_lr=min_lr
        )
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=min_lr)
    else:
        scheduler = None

    history = {
        "train_batch_nll": [],
        "val_batch_nll": [],
        "train_epoch_nll": [],
        "val_epoch_nll": [],
        "lr": [],
        "nu": [],
    }
    best_val = float("inf")
    best_state = None
    best_loss_state = None


    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            _, chol = model(xb)
            loss = loss_fn(yb, chol, **loss_kwargs).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, chol = model(xb)
                loss = loss_fn(yb, chol, **loss_kwargs).mean()
                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses)) if tr_losses else np.nan
        va = float(np.mean(va_losses)) if va_losses else np.nan

        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(va)
            else:
                scheduler.step()

        current_lr = opt.param_groups[0]["lr"]

        history["train_batch_nll"].append(tr_losses)
        history["val_batch_nll"].append(va_losses)
        history["train_epoch_nll"].append(tr)
        history["val_epoch_nll"].append(va)
        history["lr"].append(current_lr)

        if hasattr(loss_fn, "nu"):
            history["nu"].append(float(loss_fn.nu.detach().cpu()))
        else:
            history["nu"].append(np.nan)


        if verbose:
            print(f"Epoch {ep:03d} | train NLL {tr:.6f} | val NLL {va:.6f} | lr {current_lr:.2e}")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if isinstance(loss_fn, torch.nn.Module):
                best_loss_state = {k: v.detach().cpu().clone() for k, v in loss_fn.state_dict().items()}

    total_time = time.perf_counter() - start_time
    history["train_time_sec"] = total_time

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_loss_state is not None:
        loss_fn.load_state_dict(best_loss_state)


    return model, history


def predict_sigma(model, loader, device='cpu'):
    model.eval()
    sigmas = []
    targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            sigma_pred, _ = model(xb)
            sigmas.append(sigma_pred.cpu().numpy())
            targets.append(yb.numpy())

    return np.concatenate(sigmas, axis=0), np.concatenate(targets, axis=0)
