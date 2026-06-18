"""
GPU-capable elastic-net–style linear models (PyTorch) with sklearn-compatible API.

sklearn's ElasticNet / LogisticRegression do not use CUDA. These estimators train
on ``device`` (``cuda`` or ``cpu``) then store **numpy** ``coef_`` / ``intercept_``
so ``joblib`` bundles remain usable without a GPU at inference time.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

_LOG = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for --linear-backend torch. "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )


def resolve_torch_device(preferred: str) -> Any:
    _require_torch()
    pref = (preferred or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            _LOG.warning("CUDA requested but not available; using CPU for PyTorch training.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


class TorchLinearRegressor(BaseEstimator, RegressorMixin):
    """MSE + elastic-net–style L1/L2 on weights (no bias regularization)."""

    def __init__(
        self,
        *,
        device: str = "auto",
        lr: float = 0.08,
        epochs: int = 2500,
        alpha: float = 0.12,
        l1_ratio: float = 0.5,
        random_state: int = 42,
        log_every: int = 0,
    ):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.log_every = log_every

    def fit(self, X, y, sample_weight=None):
        _require_torch()
        dev = resolve_torch_device(self.device)
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31 - 1)))

        Xn = np.asarray(X, dtype=np.float32)
        yn = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n, d = Xn.shape
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
            sw = sw / (sw.mean() + 1e-8)
        else:
            sw = np.ones((n, 1), dtype=np.float32)

        Xt = torch.as_tensor(Xn, device=dev)
        yt = torch.as_tensor(yn, device=dev)
        wt = torch.as_tensor(sw, device=dev)

        lin = nn.Linear(d, 1, bias=True).to(dev)
        nn.init.zeros_(lin.bias)
        opt = torch.optim.Adam(lin.parameters(), lr=self.lr)

        l1w = self.alpha * self.l1_ratio
        l2w = self.alpha * (1.0 - self.l1_ratio)

        for ep in range(self.epochs):
            opt.zero_grad()
            pred = lin(Xt)
            mse = (wt * (pred - yt) ** 2).mean()
            w = lin.weight.view(-1)
            reg = l1w * w.abs().mean() + l2w * (w ** 2).mean() * 0.5
            loss = mse + reg
            loss.backward()
            opt.step()
            if self.log_every and (ep % self.log_every == 0 or ep == self.epochs - 1):
                _LOG.info(
                    "[TorchLinearRegressor] ep=%s device=%s mse=%.5f reg=%.5f total=%.5f",
                    ep,
                    dev,
                    float(mse.detach().cpu()),
                    float(reg.detach().cpu()),
                    float(loss.detach().cpu()),
                )

        with torch.no_grad():
            self.coef_ = lin.weight.detach().cpu().numpy().astype(np.float64).ravel()
            self.intercept_ = float(lin.bias.detach().cpu().numpy().ravel()[0])
        self.n_features_in_ = d
        self.training_device_ = str(dev)
        return self

    def predict(self, X):
        Xn = np.asarray(X, dtype=np.float64)
        return (Xn @ self.coef_ + self.intercept_).astype(np.float64)


class TorchLogisticClassifier(BaseEstimator, ClassifierMixin):
    """BCEWithLogitsLoss + elastic-net–style penalty; optional class balancing via pos_weight."""

    def __init__(
        self,
        *,
        device: str = "auto",
        lr: float = 0.08,
        epochs: int = 4000,
        alpha: float = 0.12,
        l1_ratio: float = 0.5,
        random_state: int = 42,
        log_every: int = 0,
    ):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.log_every = log_every

    def fit(self, X, y, sample_weight=None):
        _require_torch()
        dev = resolve_torch_device(self.device)
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31 - 1)))

        Xn = np.asarray(X, dtype=np.float32)
        yn = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n, d = Xn.shape
        pos = float(yn.sum())
        neg = float(n - pos)
        pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1e-3)], device=dev)
        _LOG.info(
            "[TorchLogisticClassifier] n=%s pos=%s neg=%s pos_weight=%.4f device=%s",
            n,
            int(pos),
            int(neg),
            float(pos_weight.cpu()),
            dev,
        )

        Xt = torch.as_tensor(Xn, device=dev)
        yt = torch.as_tensor(yn, device=dev)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
            sw = sw / (sw.mean() + 1e-8)
            wt = torch.as_tensor(sw, device=dev)
        else:
            wt = torch.ones((n, 1), device=dev, dtype=torch.float32)

        lin = nn.Linear(d, 1, bias=True).to(dev)
        nn.init.zeros_(lin.bias)
        opt = torch.optim.Adam(lin.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        l1w = self.alpha * self.l1_ratio
        l2w = self.alpha * (1.0 - self.l1_ratio)

        for ep in range(self.epochs):
            opt.zero_grad()
            logits = lin(Xt)
            per = bce(logits, yt)
            loss_cls = (wt * per).mean()
            w = lin.weight.view(-1)
            reg = l1w * w.abs().mean() + l2w * (w ** 2).mean() * 0.5
            loss = loss_cls + reg
            loss.backward()
            opt.step()
            if self.log_every and (ep % self.log_every == 0 or ep == self.epochs - 1):
                _LOG.info(
                    "[TorchLogisticClassifier] ep=%s bce=%.5f reg=%.5f total=%.5f",
                    ep,
                    float(loss_cls.detach().cpu()),
                    float(reg.detach().cpu()),
                    float(loss.detach().cpu()),
                )

        with torch.no_grad():
            self.coef_ = lin.weight.detach().cpu().numpy().astype(np.float64).ravel()
            self.intercept_ = float(lin.bias.detach().cpu().numpy().ravel()[0])
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = d
        self.training_device_ = str(dev)
        return self

    def decision_function(self, X):
        Xn = np.asarray(X, dtype=np.float64)
        return (Xn @ self.coef_ + self.intercept_).astype(np.float64)

    def predict_proba(self, X):
        z = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
