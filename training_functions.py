import torch
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path
import json

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
    stop_on_nonfinite=True,
    grad_clip_max_norm=1.0,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
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
    stopped_early_nonfinite = False
    stopped_early = False
    nonfinite_epoch = None
    early_stop_epoch = None
    epochs_without_improvement = 0
    stop_reason = None


    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        stop_epoch = False
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            try:
                _, chol = model(xb)
                loss = loss_fn(yb, chol, **loss_kwargs).mean()
            except RuntimeError as exc:
                if not stop_on_nonfinite:
                    raise
                stopped_early_nonfinite = True
                nonfinite_epoch = ep
                stop_reason = f"train_runtime_error_batch_{batch_idx}: {exc}"
                stop_epoch = True
                break

            if not torch.isfinite(loss):
                if not stop_on_nonfinite:
                    raise FloatingPointError(f"Non-finite train loss at epoch {ep}, batch {batch_idx}: {loss.detach().cpu().item()}")
                stopped_early_nonfinite = True
                nonfinite_epoch = ep
                stop_reason = f"nonfinite_train_loss_batch_{batch_idx}: {loss.detach().cpu().item()}"
                stop_epoch = True
                break

            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_max_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                if not stop_on_nonfinite:
                    raise
                stopped_early_nonfinite = True
                nonfinite_epoch = ep
                stop_reason = f"nonfinite_gradient_batch_{batch_idx}: {exc}"
                stop_epoch = True
                break

            opt.step()

            tr_losses.append(float(loss.detach().cpu()))

        model.eval()
        va_losses = []
        if not stop_epoch:
            with torch.no_grad():
                for batch_idx, (xb, yb) in enumerate(val_loader, start=1):
                    xb, yb = xb.to(device), yb.to(device)
                    try:
                        _, chol = model(xb)
                        loss = loss_fn(yb, chol, **loss_kwargs).mean()
                    except RuntimeError as exc:
                        if not stop_on_nonfinite:
                            raise
                        stopped_early_nonfinite = True
                        nonfinite_epoch = ep
                        stop_reason = f"val_runtime_error_batch_{batch_idx}: {exc}"
                        stop_epoch = True
                        break

                    if not torch.isfinite(loss):
                        if not stop_on_nonfinite:
                            raise FloatingPointError(f"Non-finite val loss at epoch {ep}, batch {batch_idx}: {loss.detach().cpu().item()}")
                        stopped_early_nonfinite = True
                        nonfinite_epoch = ep
                        stop_reason = f"nonfinite_val_loss_batch_{batch_idx}: {loss.detach().cpu().item()}"
                        stop_epoch = True
                        break

                    va_losses.append(float(loss.detach().cpu()))

        tr = float(np.mean(tr_losses)) if tr_losses else np.nan
        va = float(np.mean(va_losses)) if va_losses else np.nan

        if scheduler is not None and np.isfinite(va):
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
            if stop_epoch:
                print(f"Stopping early due to non-finite training state: {stop_reason}")

        improved = np.isfinite(va) and va < best_val - early_stopping_min_delta

        if improved:
            best_val = va
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if isinstance(loss_fn, torch.nn.Module):
                best_loss_state = {k: v.detach().cpu().clone() for k, v in loss_fn.state_dict().items()}
        elif np.isfinite(va):
            epochs_without_improvement += 1

        if stop_epoch:
            break

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            early_stop_epoch = ep
            stop_reason = f"early_stopping_patience_{early_stopping_patience}"
            if verbose:
                print(f"Early stopping at epoch {ep:03d} | best val NLL {best_val:.6f}")
            break

    total_time = time.perf_counter() - start_time
    history["train_time_sec"] = total_time
    history["stopped_early_nonfinite"] = stopped_early_nonfinite
    history["stopped_early"] = stopped_early
    history["nonfinite_epoch"] = nonfinite_epoch
    history["early_stop_epoch"] = early_stop_epoch
    history["stop_reason"] = stop_reason
    history["best_val_nll"] = best_val

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


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def cpu_state_dict(module):
    if module is None:
        return None
    return {
        k: v.detach().cpu()
        for k, v in module.state_dict().items()
    }


def save_model_checkpoint(
    models_dir,
    model_name,
    model,
    history,
    config,
    run_metadata,
    loss_fn=None,
):
    model_dir = Path(models_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": model_name,
        "model_state_dict": cpu_state_dict(model),
        "loss_state_dict": cpu_state_dict(loss_fn) if isinstance(loss_fn, torch.nn.Module) else None,
        "config": json_safe(config),
        "run_metadata": json_safe(run_metadata),
    }

    torch.save(checkpoint, model_dir / "checkpoint.pt")

    with open(model_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(history), f, indent=2)

    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2)

    with open(model_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(run_metadata), f, indent=2)

    print("Saved model checkpoint:", model_dir)
