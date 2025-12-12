import argparse
import os
import json
import time
from datetime import datetime
import random
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from collections import deque

import numpy as np

# PyTorch required for training
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# Optional progress bar
try:
    from tqdm import trange
except Exception:
    trange = None

# ----------------------------- utils -----------------------------

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)

def rbf_sim_matrix(A: np.ndarray, B: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    d2 = A2 + B2 - 2 * (A @ B.T)
    return np.exp(-d2 / (2 * sigma ** 2))

def linear_probe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    tpr = np.cumsum(y) / P
    fpr = np.cumsum(1 - y) / N
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    return float(np.trapz(tpr, fpr))

def fit_linear_probe_ridge(H: np.ndarray, m: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    H_ = np.concatenate([H, np.ones((H.shape[0], 1), dtype=H.dtype)], axis=1)
    W = np.linalg.pinv(H_.T @ H_ + ridge * np.eye(H_.shape[1])) @ (H_.T @ m.astype(np.float32))
    return W

def apply_probe(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    H_ = np.concatenate([H, np.ones((H.shape[0], 1), dtype=H.dtype)], axis=1)
    return (H_ @ W).astype(np.float32)

def train_linear_probe_numpy(H_train: np.ndarray, m_train: np.ndarray,
                             H_test: np.ndarray, m_test: np.ndarray,
                             ridge: float = 1e-3) -> dict:
    ridge = 0.0
    W = fit_linear_probe_ridge(H_train, m_train, ridge=ridge)
    tr_scores = apply_probe(W, H_train).squeeze()
    te_scores = apply_probe(W, H_test).squeeze()

    train_auc = linear_probe_auc(m_train.astype(np.int64), tr_scores.astype(np.float32))
    test_auc = linear_probe_auc(m_test.astype(np.int64), te_scores.astype(np.float32))
    return {
        "train_auc": train_auc,
        "test_auc": test_auc,
        "actual_epochs": 1,
        "train_scores": tr_scores.astype(np.float32),
        "test_scores": te_scores.astype(np.float32),
    }

# ----------------------------- data -----------------------------

def make_gaussian_X(N: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((N, d)).astype(np.float32)

def shortcut_bit(X: np.ndarray, v_index: int = 0) -> np.ndarray:
    return (X[:, v_index] > 0).astype(np.int64)

def labels_with_shortcut_and_exceptions(X: np.ndarray, rho: float = 0.05, v_index: int = 0, rng: np.random.Generator = None):
    s = shortcut_bit(X, v_index=v_index)
    flips = (rng.random(len(X)) < rho).astype(np.int64) if rng is not None else (np.random.rand(len(X)) < rho).astype(np.int64)
    y = s ^ flips
    return y.astype(np.int64), s


# Implementation of the "smart shortcut" generator.
# We keep the function name and signature used by the original script
# to ensure the rest of the pipeline remains unchanged.

def make_data_with_multi_shortcut(n: int, d_total: int, d_shortcut: int,
                                  rho: float,
                                  rng: np.random.Generator,
                                  w_override: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert 0 <= d_shortcut <= d_total, "d_shortcut must be in [0, d_total]"
    d_others = d_total - d_shortcut
    X = rng.standard_normal((n, d_total)).astype(np.float32)
    z = X[:, :d_shortcut].astype(np.float32)
    u = X[:, d_shortcut:].astype(np.float32)

    if d_shortcut == 0:
        w = np.zeros((0,), dtype=np.float32)
        s_rule = np.zeros(n, dtype=np.int64)
        y = rng.integers(0, 2, size=n).astype(np.int64)
    else:
        if w_override is not None and len(w_override) == d_shortcut:
            w = np.asarray(w_override, dtype=np.float32)
            w_norm = float(np.linalg.norm(w) + 1e-12)
            w = (w / w_norm).astype(np.float32)
        else:
            w = rng.standard_normal(d_shortcut).astype(np.float32)
            w_norm = float(np.linalg.norm(w) + 1e-12)
            w = (w / w_norm).astype(np.float32)
        t = (z @ w).astype(np.float32)
        s_rule = (t > 0).astype(np.int64)

        y = np.empty(n, dtype=np.int64)
        k = int(np.round(float(rho) * n))
        if k < 0: k = 0
        if k > n: k = n
        order = np.argsort(-np.abs(t))
        idx_rule = order[:k]
        idx_mem = order[k:]
        if k > 0:
            y[idx_rule] = (t[idx_rule] > 0).astype(np.int64)
        if len(idx_mem) > 0:
            y[idx_mem] = rng.integers(0, 2, size=len(idx_mem)).astype(np.int64)
    return X.astype(np.float32), y.astype(np.int64), s_rule.astype(np.int64), w.astype(np.float32), z.astype(np.float32), u.astype(np.float32)


# ----------------------------- models (PyTorch) -----------------------------

class MLP(nn.Module):
    def __init__(self, d: int, h: int = 64, act: str = "relu", hidden_layers: int = 1):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        act_l = str(act).lower()
        if act_l == "relu":
            act_cls = nn.ReLU
        elif act_l == "tanh":
            act_cls = nn.Tanh
        elif act_l == "sigmoid":
            act_cls = nn.Sigmoid
        else:
            raise ValueError(f"Unsupported activation: {act}")

        layers = []
        in_dim = d
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(h, 1)

    def forward(self, x, return_h: bool = False):
        h = self.body(x)
        logits = self.head(h)
        if return_h:
            return logits, h
        return logits


# ----------------------------- training -----------------------------

def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: Union[int, str] = "auto",
                lr: float = 2e-3, min_lr: Union[float, None] = None, reg_type: str = "L2",
                reg_coef: float = 0.01, verbose: bool = False,
                early_stop_patience: int = 10, early_stop_min_delta: float = 1e-6,
                max_auto_epochs: int = 10000,
                batch_size: Optional[int] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_train = X_train.shape[0]
    # Decide mode: full-batch GD vs SGD
    use_full_batch = (batch_size is None) or (batch_size >= n_train)

    # Prepare tensors
    X_tensor = torch.from_numpy(X_train).to(device)
    y_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device)
    valX = torch.from_numpy(X_val).to(device)
    valy = torch.from_numpy(y_val.astype(np.float32)).to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0)  # SGD without momentum
    loss_fn = nn.BCEWithLogitsLoss()
    best_val = 0.0

    use_early_stopping = epochs == "auto"
    max_epochs = max_auto_epochs if use_early_stopping else epochs
    loss_history = deque(maxlen=early_stop_patience)
    epochs_without_improvement = 0

    actual_epochs = 0
    # Cosine annealing schedule from lr -> min_lr across max_epochs
    lr0 = float(lr)
    min_lr_eff = lr0 if (min_lr is None) else float(min_lr)
    if min_lr_eff < 0:
        min_lr_eff = 0.0

    # Progress iterator
    if trange is not None:
        desc = "train(auto)" if use_early_stopping else f"train({max_epochs})"
        bar_fmt = "{l_bar}{bar}| {n_fmt} it | {elapsed}<{remaining}, {rate_fmt}"
        pbar = trange(max_epochs, desc=desc, leave=False, bar_format=(bar_fmt if use_early_stopping else None))
    else:
        pbar = range(max_epochs)
    for ep in pbar:
        # Update learning rate per epoch via cosine schedule
        if max_epochs > 1:
            cos_inner = np.pi * (ep / (max_epochs - 1))
            curr_lr = min_lr_eff + 0.5 * (lr0 - min_lr_eff) * (1.0 + np.cos(cos_inner))
        else:
            curr_lr = lr0
        for pg in opt.param_groups:
            pg['lr'] = float(curr_lr)
        actual_epochs = ep + 1
        model.train()

        if use_full_batch:
            # Full gradient descent - process entire dataset at once
            opt.zero_grad()
            logits = model(X_tensor).squeeze(1)
            bce_loss = loss_fn(logits, y_tensor)

            # Add manual regularization (exclude bias terms)
            reg_loss = 0.0
            if reg_coef > 0:
                if reg_type.upper() == "L1":
                    for name, param in model.named_parameters():
                        if param.requires_grad and ("bias" not in name):
                            reg_loss += torch.sum(torch.abs(param))
                    reg_loss *= reg_coef
                elif reg_type.upper() == "L2":
                    for name, param in model.named_parameters():
                        if param.requires_grad and ("bias" not in name):
                            reg_loss += torch.sum(param * param)
                    reg_loss *= reg_coef

            total_loss = bce_loss + reg_loss
            total_loss.backward()
            opt.step()
        else:
            # Mini-batch SGD with random shuffling each epoch
            bs = int(max(1, batch_size))
            # random permutation ensures different and random sampling per epoch
            perm = torch.randperm(n_train, device=device)
            num_batches = int(np.ceil(n_train / bs))
            last_total = None
            for i in range(0, n_train, bs):
                idx = perm[i:i+bs]
                xb = X_tensor.index_select(0, idx)
                yb = y_tensor.index_select(0, idx)
                opt.zero_grad()
                logits_b = model(xb).squeeze(1)
                bce_b = loss_fn(logits_b, yb)

                # Add manual regularization per-batch; scale to keep per-epoch reg comparable
                reg_b = 0.0
                if reg_coef > 0:
                    if reg_type.upper() == "L1":
                        for name, param in model.named_parameters():
                            if param.requires_grad and ("bias" not in name):
                                reg_b += torch.sum(torch.abs(param))
                        reg_b *= (reg_coef / num_batches)
                    elif reg_type.upper() == "L2":
                        for name, param in model.named_parameters():
                            if param.requires_grad and ("bias" not in name):
                                reg_b += torch.sum(param * param)
                        reg_b *= (reg_coef / num_batches)

                total_b = bce_b + reg_b
                total_b.backward()
                opt.step()
                last_total = total_b
            total_loss = last_total if last_total is not None else bce_b

        # Early stopping check
        if use_early_stopping:
            with torch.no_grad():
                model.eval()
                logits_val = model(valX).squeeze(1)
                bce_val = loss_fn(logits_val, valy)
                reg_val = 0.0
                if reg_coef > 0:
                    if reg_type.upper() == "L1":
                        for name, param in model.named_parameters():
                            if param.requires_grad and ("bias" not in name):
                                reg_val += torch.sum(torch.abs(param))
                        reg_val *= reg_coef
                    elif reg_type.upper() == "L2":
                        for name, param in model.named_parameters():
                            if param.requires_grad and ("bias" not in name):
                                reg_val += torch.sum(param * param)
                        reg_val *= reg_coef
                current_loss = float((bce_val + reg_val).cpu().numpy())

                if len(loss_history) == early_stop_patience:
                    oldest_loss = loss_history[0]
                    improvement = oldest_loss - current_loss
                    if improvement < early_stop_min_delta:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0
                else:
                    epochs_without_improvement = 0

                loss_history.append(current_loss)

                if epochs_without_improvement >= early_stop_patience:
                    if verbose:
                        print(f"[ep {ep+1:03d}] Early stopping: no improvement for {early_stop_patience} epochs")
                    if trange is not None:
                        pbar.close()
                    break

        # Update progress bar postfix
        if trange is not None:
            try:
                pbar.set_postfix({"loss": float(total_loss.detach().cpu().numpy()), "lr": float(curr_lr)})
            except Exception:
                pass

        if verbose and ((ep + 1) % max(1, (max_epochs // 5)) == 0):
            with torch.no_grad():
                model.eval()
                v_logits = model(valX).squeeze(1)
                v_pred = (torch.sigmoid(v_logits) > 0.5).float()
                val_acc = (v_pred == valy).float().mean().item() * 100.0
            print(f"[ep {ep+1:03d}] val acc={val_acc:.2f}%")
            best_val = max(best_val, val_acc)

    if verbose and use_early_stopping:
        print(f"Training completed after {actual_epochs} epochs")

    return actual_epochs

def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        return (p > 0.5).astype(np.int64)

def penultimate(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, h = model(torch.from_numpy(X).to(device), return_h=True)
        return h.cpu().numpy()

def compute_model_loss(model: nn.Module, X: np.ndarray, y: np.ndarray,
                      reg_type: str = "none", reg_coef: float = 0.0) -> tuple:
    device = next(model.parameters()).device
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        y_tensor = torch.from_numpy(y.astype(np.float32)).to(device)

        logits = model(X_tensor).squeeze(1)
        bce_loss = nn.BCEWithLogitsLoss()(logits, y_tensor)

        # Compute regularization (exclude bias terms)
        reg_loss = 0.0
        if reg_coef > 0:
            if reg_type.upper() == "L1":
                for name, param in model.named_parameters():
                    if ("bias" not in name):
                        reg_loss += torch.sum(torch.abs(param))
                reg_loss *= reg_coef
            elif reg_type.upper() == "L2":
                for name, param in model.named_parameters():
                    if ("bias" not in name):
                        reg_loss += torch.sum(param * param)
                reg_loss *= reg_coef

        total_loss = bce_loss + reg_loss
        return float(total_loss.cpu().numpy()), float(bce_loss.cpu().numpy())

def train_linear_probe_torch(H_train: np.ndarray, m_train: np.ndarray,
                             H_test: np.ndarray, m_test: np.ndarray,
                             epochs: Union[int, str] = "auto", lr: float = 1e-2, weight_decay: float = 0.0,
                             probe_reg_type: str = "none", probe_reg_coef: float = 0.0,
                             early_stop_patience: int = 10, early_stop_min_delta: float = 1e-6,
                             max_auto_epochs: int = 10000, verbose: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Htr = torch.from_numpy(H_train.astype(np.float32)).to(device)
    ytr = torch.from_numpy(m_train.astype(np.float32)).to(device)
    Hte = torch.from_numpy(H_test.astype(np.float32)).to(device)
    yte = torch.from_numpy(m_test.astype(np.float32)).to(device)

    clf = nn.Linear(Htr.shape[1], 1).to(device)
    opt = optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Early stopping setup
    use_early_stopping = epochs == "auto"
    max_epochs = max_auto_epochs if use_early_stopping else epochs
    loss_history = deque(maxlen=early_stop_patience)
    epochs_without_improvement = 0

    actual_epochs = 0
    if trange is not None:
        desc = "probe(auto)" if use_early_stopping else f"probe({max_epochs})"
        bar_fmt = "{l_bar}{bar}| {n_fmt} it | {elapsed}<{remaining}, {rate_fmt}"
        pbar = trange(max_epochs, desc=desc, leave=False, bar_format=(bar_fmt if use_early_stopping else None))
    else:
        pbar = range(max_epochs)
    for ep in pbar:
        actual_epochs = ep + 1
        opt.zero_grad()
        logits = clf(Htr).squeeze(1)
        bce_loss = loss_fn(logits, ytr)
        total_loss = bce_loss
        total_loss.backward()
        opt.step()

        # Early stopping check
        if use_early_stopping:
            current_loss = total_loss.item()

            # Add to history and check for improvement
            if len(loss_history) == early_stop_patience:
                oldest_loss = loss_history[0]
                improvement = oldest_loss - current_loss
                if improvement < early_stop_min_delta:
                    epochs_without_improvement += 1
                else:
                    epochs_without_improvement = 0
            else:
                epochs_without_improvement = 0

            loss_history.append(current_loss)

            # Early stop if no improvement
            if epochs_without_improvement >= early_stop_patience:
                if verbose:
                    print(f"[probe ep {ep+1:03d}] Early stopping: no improvement for {early_stop_patience} epochs")
                if trange is not None:
                    pbar.close()
                break

        if verbose and ((ep + 1) % max(1, (max_epochs // 5)) == 0):
            print(f"[probe ep {ep+1:03d}] loss={total_loss.item():.6f}")

        if trange is not None:
            try:
                pbar.set_postfix({"loss": float(total_loss.detach().cpu().numpy())})
            except Exception:
                pass

    if verbose and use_early_stopping:
        print(f"Probe training completed after {actual_epochs} epochs")

    with torch.no_grad():
        # Test AUC and scores
        logits_te = clf(Hte).squeeze(1)
        scores_te = torch.sigmoid(logits_te).cpu().numpy().astype(np.float32)
        test_auc = linear_probe_auc(m_test.astype(np.int64), scores_te)
        # Train AUC and scores
        logits_tr = clf(Htr).squeeze(1)
        scores_tr = torch.sigmoid(logits_tr).cpu().numpy().astype(np.float32)
        train_auc = linear_probe_auc(m_train.astype(np.int64), scores_tr)

    return {
        "train_auc": train_auc,
        "test_auc": test_auc,
        "actual_epochs": actual_epochs,
        "train_scores": scores_tr,
        "test_scores": scores_te,
    }


# ----------------------------- experiments -----------------------------

@dataclass
class Config:
    d: int = 64
    n_train: int = 600
    rho: float = 0.05
    epochs: Union[int, str] = "auto"
    seed: int = 123
    # Data geometry parameters
    d_shortcut: int = 1
    sample_unknown_shortcut: bool = False
    # Model architecture parameters
    h: int = 64
    act: str = "relu"
    hidden_layers: int = 1
    # Training hyperparameters
    lr: float = 2e-3
    min_lr: Union[float, None] = None
    reg_type: str = "L2"
    reg_coef: float = 0.01
    batch_size: Optional[int] = None
    # Linear probe parameters
    probe_epochs: Union[int, str] = "auto"
    probe_lr: float = 1e-2
    probe_weight_decay: float = 0.0
    probe_reg_type: str = "none"
    probe_reg_coef: float = 0.0
    probe_type: str = "logistic"
    # Early stopping parameters
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-6
    max_auto_epochs: int = 100000


def run_training(cfg: Config, print_progress: bool = False):
    if not TORCH_OK:
        raise RuntimeError("PyTorch not available. Install torch to use --mode train.")

    if print_progress:
        print("== TRAINING MODE (PyTorch) ==")
        print(cfg)
    set_seeds(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Generate training data with smart shortcut
    X_train, y_train, s_train, w_z, z_train, u_train = make_data_with_multi_shortcut(
        cfg.n_train, cfg.d, cfg.d_shortcut, cfg.rho, rng
    )

    if cfg.d_shortcut == 0:
        train_num_shortcut = 0
        train_num_random = int(cfg.n_train)
    else:
        k = int(np.round(float(cfg.rho) * cfg.n_train))
        if k < 0: k = 0
        if k > cfg.n_train: k = cfg.n_train
        train_num_shortcut = int(k)
        train_num_random = int(cfg.n_train - k)

    if cfg.d_shortcut > 0:
        t_train_vec = (z_train @ w_z).astype(np.float32)
        k_tau = int(np.round(float(cfg.rho) * cfg.n_train))
        if k_tau <= 0:
            tau_train = float('inf')
        else:
            order_tau = np.argsort(-np.abs(t_train_vec))
            tau_train = float(np.abs(t_train_vec[order_tau[k_tau - 1]]))
        order_train = np.argsort(-np.abs(t_train_vec))
        idx_rule_all = set(order_train[:max(0, min(cfg.n_train, k_tau))].tolist())
        idx_mem_all = set(order_train[max(0, min(cfg.n_train, k_tau)) :].tolist())
    else:
        tau_train = float('inf')
        idx_rule_all = set()
        idx_mem_all = set(range(cfg.n_train))

    # Train model (using training data for both train and val in train_model)
    net = MLP(cfg.d, cfg.h, act=cfg.act, hidden_layers=cfg.hidden_layers)
    actual_epochs = train_model(
        net, X_train, y_train, X_train, y_train, epochs=cfg.epochs,
        lr=cfg.lr, min_lr=cfg.min_lr, reg_type=cfg.reg_type, reg_coef=cfg.reg_coef,
        early_stop_patience=cfg.early_stop_patience, early_stop_min_delta=cfg.early_stop_min_delta,
        max_auto_epochs=cfg.max_auto_epochs, verbose=print_progress,
        batch_size=cfg.batch_size,
    )
    y_train_pred = predict(net, X_train)
    train_acc = float((y_train_pred == y_train).mean())

    # Compute final model losses
    model_train_loss_with_reg, model_train_loss_wo_reg = compute_model_loss(
        net, X_train, y_train, cfg.reg_type, cfg.reg_coef
    )

    # Loss-based membership (BCE loss as score; lower loss -> more likely member)
    device = next(net.parameters()).device
    with torch.no_grad():
        bce_none = nn.BCEWithLogitsLoss(reduction='none')
        # Train per-sample BCE
        logits_tr_full = net(torch.from_numpy(X_train).to(device)).squeeze(1)
        y_tr_full = torch.from_numpy(y_train.astype(np.float32)).to(device)
        loss_tr_vec = bce_none(logits_tr_full, y_tr_full).cpu().numpy().astype(np.float32)
        # Unseen set for loss-based AUC
        rng_bce = np.random.default_rng(cfg.seed + 4242)
        X_bce_un, y_bce_un, _, _, _, _ = make_data_with_multi_shortcut(
            cfg.n_train, cfg.d, cfg.d_shortcut, cfg.rho, rng_bce, w_override=w_z
        )
        logits_un_full = net(torch.from_numpy(X_bce_un).to(device)).squeeze(1)
        y_un_full = torch.from_numpy(y_bce_un.astype(np.float32)).to(device)
        loss_un_vec = bce_none(logits_un_full, y_un_full).cpu().numpy().astype(np.float32)
        # Membership labels and scores (negative loss)
        m_loss = np.concatenate([np.ones_like(loss_tr_vec, dtype=np.int64), np.zeros_like(loss_un_vec, dtype=np.int64)])
        s_loss = -np.concatenate([loss_tr_vec, loss_un_vec]).astype(np.float32)
        bce_loss_auc = linear_probe_auc(m_loss, s_loss)

    H_mem = penultimate(net, X_train)

    # Unseen data for membership probe
    if cfg.sample_unknown_shortcut:
        d_others = cfg.d - cfg.d_shortcut
        u_new = rng.standard_normal((cfg.n_train, max(0, d_others))).astype(np.float32)
        X_unseen = np.concatenate([z_train, u_new], axis=1).astype(np.float32)
        z_unseen = z_train
        u_unseen = u_new
        if cfg.d_shortcut > 0:
            t_un = (z_unseen @ w_z).astype(np.float32)
            y_unseen = np.empty(cfg.n_train, dtype=np.int64)
            k_un = int(np.round(float(cfg.rho) * cfg.n_train))
            k_un = max(0, min(cfg.n_train, k_un))
            order_un = np.argsort(-np.abs(t_un))
            idx_rule_un = order_un[:k_un]
            idx_mem_un = order_un[k_un:]
            if k_un > 0:
                y_unseen[idx_rule_un] = (t_un[idx_rule_un] > 0).astype(np.int64)
            if len(idx_mem_un) > 0:
                y_unseen[idx_mem_un] = rng.integers(0, 2, size=len(idx_mem_un)).astype(np.int64)
        else:
            y_unseen = rng.integers(0, 2, size=cfg.n_train).astype(np.int64)
    else:
        X_unseen, y_unseen, _, _, z_unseen, u_unseen = make_data_with_multi_shortcut(
            cfg.n_train, cfg.d, cfg.d_shortcut, cfg.rho, rng, w_override=w_z
        )

    X_slice, _, _, _, z_slice, _ = make_data_with_multi_shortcut(
        cfg.n_train, cfg.d, cfg.d_shortcut, cfg.rho, rng, w_override=w_z
    )
    s_slice = (z_slice @ w_z).astype(np.float32) if cfg.d_shortcut > 0 else np.zeros(cfg.n_train, dtype=np.float32)
    yhat_slice = predict(net, X_slice).astype(np.int64)
    if np.isinf(tau_train):
        mask_pos = s_slice > tau_train  # False
        mask_neg = s_slice < -tau_train  # False
        mask_mid_pos = s_slice >= 0
        mask_mid_neg = s_slice < 0
    else:
        mask_pos = s_slice > tau_train
        mask_neg = s_slice < -tau_train
        mask_mid_pos = (s_slice >= 0) & (s_slice <= tau_train)
        mask_mid_neg = (s_slice < 0) & (s_slice >= -tau_train)

    def _slice_stats(mask: np.ndarray):
        cnt = int(mask.sum())
        if cnt == 0:
            return 0, 0.0
        pct1 = float(yhat_slice[mask].mean() * 100.0)
        return cnt, pct1

    n_pos, pct1_pos = _slice_stats(mask_pos)
    n_neg, pct1_neg = _slice_stats(mask_neg)
    n_mid_pos, pct1_mid_pos = _slice_stats(mask_mid_pos)
    n_mid_neg, pct1_mid_neg = _slice_stats(mask_mid_neg)

    H_un = penultimate(net, X_unseen)

    # Split for train/test
    perm = rng.permutation(cfg.n_train)
    half = cfg.n_train // 2
    A = perm[:half]
    B = perm[half:]
    # Pair membership/non-membership examples.
    Htr = np.vstack([H_mem[A], H_un[A]])
    mtr = np.concatenate([np.ones(half, dtype=int), np.zeros(half, dtype=int)])
    Hte = np.vstack([H_mem[B], H_un[B]])
    mte = np.concatenate([np.ones(len(B), dtype=int), np.zeros(len(B), dtype=int)])

    if cfg.probe_type == "logistic":
        probe_results = train_linear_probe_torch(
            Htr, mtr, Hte, mte,
            epochs=cfg.probe_epochs, lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay,
            probe_reg_type=cfg.probe_reg_type, probe_reg_coef=cfg.probe_reg_coef,
            early_stop_patience=cfg.early_stop_patience, early_stop_min_delta=cfg.early_stop_min_delta,
            max_auto_epochs=cfg.max_auto_epochs, verbose=print_progress,
        )
    else:
        # Linear probe using NumPy; regularization disabled (ridge=0)
        probe_results = train_linear_probe_numpy(
            Htr, mtr, Hte, mte, ridge=0.0
        )

    # Compute rule-based vs memory-based AUCs (train/test) for membership probe
    # Use the training-derived threshold tau_train applied to both train and unseen splits.
    t_unseen_vec = (z_unseen @ w_z).astype(np.float32) if cfg.d_shortcut > 0 else np.zeros(cfg.n_train, dtype=np.float32)

    def subset_auc_with_tau(scores: np.ndarray, labels: np.ndarray, pos_idx: np.ndarray, neg_idx: np.ndarray) -> Tuple[float, float]:
        n_pos = len(pos_idx)
        # Absolute projection magnitudes for positives (train members) and negatives (unseen)
        pos_abs = np.abs(t_train_vec[pos_idx]) if cfg.d_shortcut > 0 else np.zeros(n_pos, dtype=np.float32)
        neg_abs = np.abs(t_unseen_vec[neg_idx]) if cfg.d_shortcut > 0 else np.zeros(len(neg_idx), dtype=np.float32)

        n_total = scores.shape[0]
        mask_rule = np.zeros(n_total, dtype=bool)
        mask_mem = np.zeros(n_total, dtype=bool)
        if np.isfinite(tau_train):
            mask_rule[:n_pos] = pos_abs > tau_train
            mask_rule[n_pos:] = neg_abs > tau_train
            mask_mem[:n_pos] = pos_abs <= tau_train
            mask_mem[n_pos:] = neg_abs <= tau_train
        else:
            # No rule region when d_shortcut == 0
            mask_rule[:] = False
            mask_mem[:] = True

        auc_rule = float('nan')
        if mask_rule[:n_pos].any() and mask_rule[n_pos:].any():
            auc_rule = linear_probe_auc(labels[mask_rule].astype(np.int64), scores[mask_rule].astype(np.float32))

        auc_mem = float('nan')
        if mask_mem[:n_pos].any() and mask_mem[n_pos:].any():
            auc_mem = linear_probe_auc(labels[mask_mem].astype(np.int64), scores[mask_mem].astype(np.float32))

        return auc_rule, auc_mem

    tr_scores = np.asarray(probe_results.get("train_scores"), dtype=np.float32)
    te_scores = np.asarray(probe_results.get("test_scores"), dtype=np.float32)
    # Train split (A): positives H_mem[A], negatives H_un[A]
    auc_rule_tr, auc_mem_tr = subset_auc_with_tau(tr_scores, mtr.astype(np.int64), A, A)
    # Test split (B): positives H_mem[B], negatives H_un[B]
    auc_rule_te, auc_mem_te = subset_auc_with_tau(te_scores, mte.astype(np.int64), B, B)

    # -------- Baseline membership scores (no training) on model outputs --------
    # For binary classification, BCE/Entropy/|logit| are monotonic; compute once using |logit|.
    def baseline_membership_scores(X: np.ndarray) -> np.ndarray:
        device = next(net.parameters()).device
        with torch.no_grad():
            logits = net(torch.from_numpy(X).to(device)).squeeze(1)
            return torch.abs(logits).cpu().numpy().astype(np.float32)

    # Train/test AUC for the baseline using the same A/B split
    base_mem_A = baseline_membership_scores(X_train[A]); base_un_A = baseline_membership_scores(X_unseen[A])
    base_tr_scores = np.concatenate([base_mem_A, base_un_A]).astype(np.float32)
    baseline_train_auc = linear_probe_auc(mtr.astype(np.int64), base_tr_scores)
    base_mem_B = baseline_membership_scores(X_train[B]); base_un_B = baseline_membership_scores(X_unseen[B])
    base_te_scores = np.concatenate([base_mem_B, base_un_B]).astype(np.float32)
    baseline_test_auc = linear_probe_auc(mte.astype(np.int64), base_te_scores)

    # Rule/memory subset AUCs for baseline, using training-derived tau_train
    def baseline_subset_auc(tr_or_te_scores: np.ndarray, labels: np.ndarray, pos_idx: np.ndarray, neg_idx: np.ndarray) -> Tuple[float, float]:
        n_pos = len(pos_idx)
        pos_abs = np.abs(t_train_vec[pos_idx]) if cfg.d_shortcut > 0 else np.zeros(n_pos, dtype=np.float32)
        neg_abs = np.abs(t_unseen_vec[neg_idx]) if cfg.d_shortcut > 0 else np.zeros(len(neg_idx), dtype=np.float32)
        n_total = tr_or_te_scores.shape[0]
        mask_rule = np.zeros(n_total, dtype=bool)
        mask_mem = np.zeros(n_total, dtype=bool)
        if np.isfinite(tau_train):
            mask_rule[:n_pos] = pos_abs > tau_train
            mask_rule[n_pos:] = neg_abs > tau_train
            mask_mem[:n_pos] = pos_abs <= tau_train
            mask_mem[n_pos:] = neg_abs <= tau_train
        else:
            mask_rule[:] = False
            mask_mem[:] = True
        auc_rule = float('nan')
        if mask_rule[:n_pos].any() and mask_rule[n_pos:].any():
            auc_rule = linear_probe_auc(labels[mask_rule].astype(np.int64), tr_or_te_scores[mask_rule].astype(np.float32))
        auc_mem = float('nan')
        if mask_mem[:n_pos].any() and mask_mem[n_pos:].any():
            auc_mem = linear_probe_auc(labels[mask_mem].astype(np.int64), tr_or_te_scores[mask_mem].astype(np.float32))
        return auc_rule, auc_mem

    baseline_rule_train_auc, baseline_mem_train_auc = baseline_subset_auc(base_tr_scores, mtr.astype(np.int64), A, A)
    baseline_rule_test_auc, baseline_mem_test_auc = baseline_subset_auc(base_te_scores, mte.astype(np.int64), B, B)

    # Helper: run linear probe on given features (membership pairs already implied by stacking)
    def run_linear_probe(Hp_mem: np.ndarray, Hp_un: np.ndarray):
        if Hp_mem.shape[1] == 0 or Hp_un.shape[1] == 0:
            return {"train_auc": float("nan"), "test_auc": float("nan"), "actual_epochs": 0}
        Htr_p = np.vstack([Hp_mem[A], Hp_un[A]])
        Hte_p = np.vstack([Hp_mem[B], Hp_un[B]])
        if cfg.probe_type == "logistic":
            return train_linear_probe_torch(
                Htr_p, mtr, Hte_p, mte,
                epochs=cfg.probe_epochs, lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay,
                probe_reg_type=cfg.probe_reg_type, probe_reg_coef=cfg.probe_reg_coef,
                early_stop_patience=cfg.early_stop_patience, early_stop_min_delta=cfg.early_stop_min_delta,
                max_auto_epochs=cfg.max_auto_epochs, verbose=print_progress,
            )
        else:
            return train_linear_probe_numpy(Htr_p, mtr, Hte_p, mte, ridge=0.0)

    # Build zeroed inputs for ablation (representation-layer)
    d_z = cfg.d_shortcut
    d_u = cfg.d - cfg.d_shortcut
    if d_z > 0:
        zeros_u_mem = np.zeros_like(u_train, dtype=np.float32)
        zeros_u_un = np.zeros_like(u_unseen, dtype=np.float32)
        Xz_mem = np.concatenate([z_train.astype(np.float32), zeros_u_mem], axis=1)
        Xz_un = np.concatenate([z_unseen.astype(np.float32), zeros_u_un], axis=1)
        Hz_mem = penultimate(net, Xz_mem)
        Hz_un = penultimate(net, Xz_un)
        ablate_z = run_linear_probe(Hz_mem, Hz_un)
    else:
        ablate_z = {"train_auc": float("nan"), "test_auc": float("nan"), "actual_epochs": 0}

    if d_u > 0:
        zeros_z_mem = np.zeros((z_train.shape[0], d_z), dtype=np.float32)
        zeros_z_un = np.zeros((z_unseen.shape[0], d_z), dtype=np.float32)
        Xu_mem = np.concatenate([zeros_z_mem, u_train.astype(np.float32)], axis=1)
        Xu_un = np.concatenate([zeros_z_un, u_unseen.astype(np.float32)], axis=1)
        Hu_mem = penultimate(net, Xu_mem)
        Hu_un = penultimate(net, Xu_un)
        ablate_u = run_linear_probe(Hu_mem, Hu_un)
    else:
        ablate_u = {"train_auc": float("nan"), "test_auc": float("nan"), "actual_epochs": 0}

    # Input-layer probing (layer 0): use raw inputs as features
    probe_input = run_linear_probe(X_train.astype(np.float32), X_unseen.astype(np.float32))

    # Output-layer probing (layer 2): use model logits as 1-D features
    def logits_features(X: np.ndarray) -> np.ndarray:
        device = next(net.parameters()).device
        with torch.no_grad():
            logits = net(torch.from_numpy(X).to(device)).squeeze(1).cpu().numpy().astype(np.float32)
        return logits.reshape(-1, 1)
    L_mem = logits_features(X_train)
    L_un = logits_features(X_unseen)
    probe_output = run_linear_probe(L_mem, L_un)

    if print_progress:
        print(
            f"rho={cfg.rho:.3f} | train acc={train_acc*100:.2f}% | model epochs: {actual_epochs} | probe epochs: {probe_results['actual_epochs']}"
        )
        print(
            f"rho={cfg.rho:.3f} | membership probe AUC: train={probe_results['train_auc']:.3f} | test={probe_results['test_auc']:.3f}"
        )
        print(
            f"rho={cfg.rho:.3f} | membership probe AUC (rule/mem): train_rule={auc_rule_tr:.3f}, train_mem={auc_mem_tr:.3f} | test_rule={auc_rule_te:.3f}, test_mem={auc_mem_te:.3f}"
        )
        print(
            f"rho={cfg.rho:.3f} | ablation (representation-layer) AUC: z-only train={ablate_z['train_auc']:.3f}, test={ablate_z['test_auc']:.3f} | u-only train={ablate_u['train_auc']:.3f}, test={ablate_u['test_auc']:.3f}"
        )
        print(
            f"rho={cfg.rho:.3f} | layer-0 (input) probe AUC: train={probe_input['train_auc']:.3f}, test={probe_input['test_auc']:.3f}"
        )
        print(
            f"rho={cfg.rho:.3f} | layer-2 (output) probe AUC: train={probe_output['train_auc']:.3f}, test={probe_output['test_auc']:.3f}"
        )
        print(
            f"rho={cfg.rho:.3f} | Baseline(no-train) membership AUC: train={baseline_train_auc:.3f} | test={baseline_test_auc:.3f}"
        )
        print(
            f"rho={cfg.rho:.3f} | Baseline(no-train) AUC (rule/mem): train_rule={baseline_rule_train_auc:.3f}, train_mem={baseline_mem_train_auc:.3f} | test_rule={baseline_rule_test_auc:.3f}, test_mem={baseline_mem_test_auc:.3f}"
        )

    metrics = {
        "model_train_acc": train_acc,
        "model_train_loss": model_train_loss_with_reg,
        "model_train_loss_wo_reg": model_train_loss_wo_reg,
        "actual_model_epochs": actual_epochs,
        "shortcut_threshold_tau": float(tau_train),
        # data labeling composition (train)
        "train_num_shortcut": float(train_num_shortcut),
        "train_num_random": float(train_num_random),
        "probe_train_auc": probe_results["train_auc"],
        "probe_test_auc": probe_results["test_auc"],
        "probe_rule_train_auc": float(auc_rule_tr),
        "probe_rule_test_auc": float(auc_rule_te),
        "probe_mem_train_auc": float(auc_mem_tr),
        "probe_mem_test_auc": float(auc_mem_te),
        "actual_probe_epochs": probe_results["actual_epochs"],
        # ablation metrics
        "ablate_z_train_auc": ablate_z["train_auc"],
        "ablate_z_test_auc": ablate_z["test_auc"],
        "ablate_u_train_auc": ablate_u["train_auc"],
        "ablate_u_test_auc": ablate_u["test_auc"],
        # Baseline(no-train) AUCs
        "baseline_probe_train_auc": float(baseline_train_auc),
        "baseline_probe_test_auc": float(baseline_test_auc),
        "baseline_rule_train_auc": float(baseline_rule_train_auc),
        "baseline_rule_test_auc": float(baseline_rule_test_auc),
        "baseline_mem_train_auc": float(baseline_mem_train_auc),
        "baseline_mem_test_auc": float(baseline_mem_test_auc),
        # layer-0/2 probe AUC
        "probe_input_train_auc": probe_input["train_auc"],
        "probe_input_test_auc": probe_input["test_auc"],
        "probe_output_train_auc": probe_output["train_auc"],
        "probe_output_test_auc": probe_output["test_auc"],
        "slice_[t,inf]_n": float(n_pos),
        "slice_[t,inf]_pct1": float(pct1_pos),
        "slice_[-inf,-t]_n": float(n_neg),
        "slice_[-inf,-t]_pct1": float(pct1_neg),
        "slice_[0,t]_n": float(n_mid_pos),
        "slice_[0,t]_pct1": float(pct1_mid_pos),
        "slice_[-t,0]_n": float(n_mid_neg),
        "slice_[-t,0]_pct1": float(pct1_mid_neg),
    }

    artifacts = {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "X_unseen": X_unseen.astype(np.float32),
        "y_unseen": y_unseen.astype(np.int64),
        "model": net,
        "w_z": w_z.astype(np.float32),
    }

    return metrics, artifacts


def main():
    parser = argparse.ArgumentParser(description="MLP with smart shortcut data and membership probing")

    parser.add_argument("--d", type=int, default=64, help="Total input dimension d = d_shortcut + d_others")
    parser.add_argument("--d_shortcut", type=int, default=1, help="Number of shortcut dimensions (z)")
    parser.add_argument("--sample_unknown_shortcut", action="store_true", help="In membership probing, copy training z and resample only u for unseen samples")
    parser.add_argument("--n_train", type=int, default=600, help="Training set size")
    parser.add_argument("--rho", type=float, default=0.05, help="ratio of shortcut")

    parser.add_argument("--epochs", default="auto", help="Training epochs (integer or 'auto')")
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size for SGD; if omitted or >= n_train use full-batch GD")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate (cosine start)")
    parser.add_argument("--min_lr", type=float, default=None, help="Cosine annealing minimum learning rate (if None, equals --lr)")
    parser.add_argument("--reg_type", type=str, default="L2", choices=["L1", "L2", "none"], help="Regularization type")
    parser.add_argument("--reg_coef", type=float, default=0.01, help="Regularization coefficient")

    parser.add_argument("--h", type=int, default=64, help="Hidden layer size (per hidden layer)")
    parser.add_argument("--hidden_layers", type=int, default=1, help="Number of hidden layers (>=1)")
    parser.add_argument("--act", type=str, default="relu", choices=["relu", "tanh", "sigmoid"], help="Hidden activation function")

    parser.add_argument("--probe_epochs", default="auto", help="Linear probe training epochs (integer or 'auto')")
    parser.add_argument("--probe_lr", type=float, default=1e-2, help="Linear probe learning rate")
    parser.add_argument("--probe_weight_decay", type=float, default=0.0, help="[Ignored] Linear probe weight decay (disabled)")
    parser.add_argument("--probe_reg_type", type=str, default="none", choices=["L1", "L2", "none"], help="[Ignored] Linear probe regularization type (disabled)")
    parser.add_argument("--probe_reg_coef", type=float, default=0.0, help="[Ignored] Linear probe regularization coefficient (disabled)")
    parser.add_argument("--probe_type", type=str, default="logistic", choices=["logistic", "linear"], help="Choose logistic (BCE) or linear (ridge) probe")
    parser.add_argument("--num_seed", type=int, default=1, help="How many random seeds to run (use 0..num_seed-1)")
    parser.add_argument("--plot", action="store_true", help="If set, generate 2D plots per seed (requires matplotlib), only when d==2")

    args = parser.parse_args()
    start_time = time.time()
    start_iso = datetime.now().isoformat(timespec="seconds")
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    except Exception:
        ts = str(int(time.time()))
    base_dir = os.path.join(
        "output",
        f"{args.rho}_{args.d}_{args.hidden_layers}_{ts}"
    )
    seeds_dir = os.path.join(base_dir, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    # Parse epochs arguments (convert to int if not "auto")
    epochs = args.epochs if args.epochs == "auto" else int(args.epochs)
    probe_epochs = args.probe_epochs if args.probe_epochs == "auto" else int(args.probe_epochs)

    cfg_base = Config(
        d=args.d, n_train=args.n_train, rho=args.rho,
        epochs=epochs,
        d_shortcut=args.d_shortcut, sample_unknown_shortcut=args.sample_unknown_shortcut,
        h=args.h, act=args.act, hidden_layers=args.hidden_layers,
        lr=args.lr, min_lr=args.min_lr, reg_type=args.reg_type, reg_coef=args.reg_coef,
        batch_size=(None if args.batch_size is None else int(args.batch_size)),
        probe_epochs=probe_epochs, probe_lr=args.probe_lr, probe_weight_decay=args.probe_weight_decay,
        probe_reg_type=args.probe_reg_type, probe_reg_coef=args.probe_reg_coef,
        probe_type=args.probe_type,
    )

    num_seed = max(1, int(args.num_seed))
    seeds = list(range(num_seed))
    all_metrics = []
    seed_dirs_for_plot = []
    for sd in seeds:
        cfg = Config(**{**cfg_base.__dict__, "seed": sd})
        m, art = run_training(cfg, print_progress=False)
        all_metrics.append({"seed": sd, **m})
        seed_dir = os.path.join(seeds_dir, f"seed_{sd}")
        os.makedirs(seed_dir, exist_ok=True)
        seed_dirs_for_plot.append(seed_dir)
        # train_data.json
        train_json = {
            "x": art["X_train"].tolist(),
            "y": art["y_train"].tolist(),
        }
        with open(os.path.join(seed_dir, "train_data.json"), "w", encoding="utf-8") as f:
            json.dump(train_json, f)
        test_json = {
            "x": art["X_unseen"].tolist(),
            "y": art["y_unseen"].tolist(),
        }
        with open(os.path.join(seed_dir, "test_data.json"), "w", encoding="utf-8") as f:
            json.dump(test_json, f)
        try:
            w_json = {"w": art["w_z"].tolist(), "d_shortcut": int(args.d_shortcut)}
            with open(os.path.join(seed_dir, "w.json"), "w", encoding="utf-8") as f:
                json.dump(w_json, f)
        except Exception as e:
            print(f"⚠️ Failed to save w.json for seed {sd}: {e}")
        try:
            model_path = os.path.join(seed_dir, "model.pt")
            model_to_save = art["model"].to("cpu")
            torch.save(model_to_save.state_dict(), model_path)
        except Exception as e:
            print(f"⚠️ Failed to save model for seed {sd}: {e}")

        # Plotting moved after result.json is written

    def mean_std(key: str):
        vals = np.array([m[key] for m in all_metrics], dtype=float)
        return float(vals.mean()), float(vals.std(ddof=0))

    keys_model = [
        "model_train_acc", "model_train_loss", "model_train_loss_wo_reg", "actual_model_epochs",
    ]
    keys_data = [
        "train_num_shortcut", "train_num_random",
    ]
    keys_probe = [
        "probe_train_auc", "probe_test_auc",
        "probe_rule_train_auc", "probe_rule_test_auc",
        "probe_mem_train_auc", "probe_mem_test_auc",
        "ablate_z_train_auc", "ablate_z_test_auc",
        "ablate_u_train_auc", "ablate_u_test_auc",
        "probe_input_train_auc", "probe_input_test_auc",
        "probe_output_train_auc", "probe_output_test_auc",
        "baseline_probe_train_auc", "baseline_probe_test_auc",
        "baseline_rule_train_auc", "baseline_rule_test_auc",
        "baseline_mem_train_auc", "baseline_mem_test_auc",
    ]
    keys_geom = ["shortcut_threshold_tau"]
    keys_slice_pct = [
        "slice_[t,inf]_pct1", "slice_[-inf,-t]_pct1",
        "slice_[0,t]_pct1", "slice_[-t,0]_pct1",
    ]
    keys_slice_n = [
        "slice_[t,inf]_n", "slice_[-inf,-t]_n",
        "slice_[0,t]_n", "slice_[-t,0]_n",
    ]

    summary = {k: mean_std(k) for k in keys_model + keys_data + keys_probe + keys_geom + keys_slice_pct + keys_slice_n}

    # Print organized metrics summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT RESULTS SUMMARY (mean ± std over num_seed={num_seed})")
    print("=" * 60)

    # Model Training Section
    print("\n📊 MODEL TRAINING:")
    print(
        f"   d={cfg_base.d} | d_shortcut={cfg_base.d_shortcut} | act={cfg_base.act} | rho={cfg_base.rho} | unknown_shortcut={'on' if cfg_base.sample_unknown_shortcut else 'off'}"
    )
    m, s = summary["model_train_acc"]; print(f"   Accuracy:              {m*100:.2f}% ± {s*100:.2f}%")
    m, s = summary["model_train_loss"]; print(f"   Loss (w/ reg):         {m:.6f} ± {s:.6f}")
    m, s = summary["model_train_loss_wo_reg"]; print(f"   Loss (w/o reg):        {m:.6f} ± {s:.6f}")
    # Reg effect: derived
    m_reg = summary["model_train_loss"][0] - summary["model_train_loss_wo_reg"][0]
    s_reg = np.sqrt(max(0.0, summary["model_train_loss"][1] ** 2 + summary["model_train_loss_wo_reg"][1] ** 2))
    print(f"   Regularization Effect: {m_reg:.6f} ± {s_reg:.6f}")
    m, s = summary["actual_model_epochs"]; print(f"   Epochs Used:           {m:.2f} ± {s:.2f}")

    # Data labeling composition (train)
    print("\n📦 DATA LABEL MIX (train):")
    ns_m, ns_s = summary["train_num_shortcut"]; nr_m, nr_s = summary["train_num_random"]
    total = float(args.n_train)
    pct_short = (ns_m / total) * 100.0 if total > 0 else 0.0
    pct_rand = (nr_m / total) * 100.0 if total > 0 else 0.0
    print(f"   Shortcut-labeled: {ns_m:.2f} ± {ns_s:.2f} ({pct_short:.2f}%)")
    print(f"   Random-labeled:   {nr_m:.2f} ± {nr_s:.2f} ({pct_rand:.2f}%)")
    tau_m, tau_s = summary["shortcut_threshold_tau"]
    print(f"   Threshold tau (|<w,z>|@k): mean={tau_m:.6f} ± {tau_s:.6f}")

    # Membership Probing Section with Table
    print("\n🔍 MEMBERSHIP PROBING:")
    m_tr_auc, s_tr_auc = summary["probe_train_auc"]
    m_te_auc, s_te_auc = summary["probe_test_auc"]
    print(f"   Train AUC: {m_tr_auc:.4f} ± {s_tr_auc:.4f}")
    print(f"   Test  AUC: {m_te_auc:.4f} ± {s_te_auc:.4f}")
    rr_tr, rr_tr_s = summary["probe_rule_train_auc"]; rr_te, rr_te_s = summary["probe_rule_test_auc"]
    mm_tr, mm_tr_s = summary["probe_mem_train_auc"]; mm_te, mm_te_s = summary["probe_mem_test_auc"]
    print(f"   Rule  AUC: {rr_tr:.4f} ± {rr_tr_s:.4f} | {rr_te:.4f} ± {rr_te_s:.4f}")
    print(f"   Mem   AUC: {mm_tr:.4f} ± {mm_tr_s:.4f} | {mm_te:.4f} ± {mm_te_s:.4f}")
    # Baseline (no training), computed once as |logit| ranking
    mb_tr, sb_tr = summary["baseline_probe_train_auc"]; mb_te, sb_te = summary["baseline_probe_test_auc"]
    print(f"   Baseline(no-train) AUC: train={mb_tr:.4f} ± {sb_tr:.4f} | test={mb_te:.4f} ± {sb_te:.4f}")
    rr_tr, rr_tr_s = summary["baseline_rule_train_auc"]; rr_te, rr_te_s = summary["baseline_rule_test_auc"]
    mm_tr, mm_tr_s = summary["baseline_mem_train_auc"]; mm_te, mm_te_s = summary["baseline_mem_test_auc"]
    print(f"   Baseline(no-train) Rule/Mem AUC: rule={rr_tr:.4f} ± {rr_tr_s:.4f} | {rr_te:.4f} ± {rr_te_s:.4f}; mem={mm_tr:.4f} ± {mm_tr_s:.4f} | {mm_te:.4f} ± {mm_te_s:.4f}")

    # Ablation: representation-layer probing on z and u
    print("\n🧪 ABLATION (Representation-layer Probing):")
    mz_tr, sz_tr = summary["ablate_z_train_auc"]; mz_te, sz_te = summary["ablate_z_test_auc"]
    mu_tr, su_tr = summary["ablate_u_train_auc"]; mu_te, su_te = summary["ablate_u_test_auc"]
    print(f"   z-only  AUC: train={mz_tr:.4f} ± {sz_tr:.4f} | test={mz_te:.4f} ± {sz_te:.4f}")
    print(f"   u-only  AUC: train={mu_tr:.4f} ± {su_tr:.4f} | test={mu_te:.4f} ± {su_te:.4f}")

    # Layer 0 & 2 probes (accuracy only)
    print("\n🔎 LAYER 0/2 LINEAR PROBES (AUC):")
    mi_tr, si_tr = summary["probe_input_train_auc"]; mi_te, si_te = summary["probe_input_test_auc"]
    mo_tr, so_tr = summary["probe_output_train_auc"]; mo_te, so_te = summary["probe_output_test_auc"]
    print(f"   layer-0 (input)  AUC: train={mi_tr:.4f} ± {si_tr:.4f} | test={mi_te:.4f} ± {si_te:.4f}")
    print(f"   layer-2 (output) AUC: train={mo_tr:.4f} ± {so_tr:.4f} | test={mo_te:.4f} ± {so_te:.4f}")

    print("\n📐 MODEL OUTPUTS BY s=<w,z> SLICES:")
    def _fmt_slice(pct_key: str, n_key: str, tag: str):
        p1 = summary[pct_key][0]
        p0 = 100.0 - p1
        n = summary[n_key][0]
        n1 = int(round(n * p1 / 100.0))
        n0 = int(round(n - n1))
        print(f"   {tag}:  1-{p1:.2f}%({n1})  0-{p0:.2f}%({n0})")
    _fmt_slice("slice_[t,inf]_pct1", "slice_[t,inf]_n", "[t:inf]")
    _fmt_slice("slice_[-inf,-t]_pct1", "slice_[-inf,-t]_n", "[-inf:-t]")
    _fmt_slice("slice_[0,t]_pct1", "slice_[0,t]_n", "[0:t]")
    _fmt_slice("slice_[-t,0]_pct1", "slice_[-t,0]_n", "[-t:0]")

    # ---------------- Write results to JSON (output/{timestamp}.json) ----------------
    end_time = time.time()
    end_iso = datetime.now().isoformat(timespec="seconds")

    summary_json = {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in summary.items()}
    config_json = {
        "d": args.d,
        "d_shortcut": args.d_shortcut,
        "n_train": args.n_train,
        "rho": args.rho,
        "sample_unknown_shortcut": bool(args.sample_unknown_shortcut),
        "epochs": args.epochs,
        "epochs_parsed": epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": (args.min_lr if args.min_lr is not None else args.lr),
        "reg_type": args.reg_type,
        "reg_coef": args.reg_coef,
        "h": args.h,
        "hidden_layers": args.hidden_layers,
        "act": args.act,
        "probe_epochs": args.probe_epochs,
        "probe_epochs_parsed": probe_epochs,
        "probe_lr": args.probe_lr,
        "probe_weight_decay": args.probe_weight_decay,
        "probe_reg_type": args.probe_reg_type,
        "probe_reg_coef": args.probe_reg_coef,
        "probe_type": args.probe_type,
        "num_seed": num_seed,
        "seeds": seeds,
    }

    log_obj = {
        "timestamp": ts,
        "time": {
            "start": start_iso,
            "end": end_iso,
            "wall_clock_s": float(max(0.0, end_time - start_time)),
        },
        "config": config_json,
        "results": {
            "per_seed": all_metrics,
            "summary": summary_json,
        },
    }

    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, "result.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log_obj, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Saved JSON to: {out_path}")
    except Exception as e:
        print(f"\n⚠️ Failed to write JSON: {e}")

    # Now that result.json exists, optionally generate plots for all seeds
    if args.plot:
        try:
            import plot as plot_mod
            for sd, seed_dir in zip(seeds, seed_dirs_for_plot):
                try:
                    out_plot = plot_mod.plot_seed_dir(seed_dir, grid=300, outfile="plot.png")
                    print(f"🖼  Saved plot for seed {sd}: {out_plot}")
                except Exception as e:
                    print(f"⚠️ Failed to generate plot for seed {sd}: {e}")
        except Exception as e:
            print(f"⚠️ Plotting skipped: failed to import plot module: {e}")


if __name__ == "__main__":
    main()
