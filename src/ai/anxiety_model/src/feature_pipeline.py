"""Variance / correlation pruning, stability feature selection, ensemble + Platt + Youden."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

_LOG = logging.getLogger(__name__)


def prune_low_variance(X: pd.DataFrame, min_variance: float = 1e-10) -> pd.DataFrame:
    v = X.var(numeric_only=True)
    keep = v[v >= min_variance].index.tolist()
    dropped = set(X.columns) - set(keep)
    if dropped:
        _LOG.info("Variance prune: dropped %s near-constant columns", len(dropped))
    return X[keep] if keep else X


def prune_correlated_with_target(
    X: pd.DataFrame,
    y_cont: np.ndarray,
    corr_threshold: float = 0.85,
) -> pd.DataFrame:
    """Drop one feature from each pair with |corr|>threshold, keeping stronger |corr(y)|."""
    cols = list(X.columns)
    if len(cols) < 2:
        return X
    C = X[cols].corr(numeric_only=True).abs()
    ys = pd.Series(y_cont, index=X.index)
    ycorr = X[cols].corrwith(ys).abs()
    drop: set[str] = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in drop or b in drop:
                continue
            try:
                r = float(C.loc[a, b])
            except Exception:
                continue
            if r > corr_threshold:
                if ycorr.get(a, 0.0) < ycorr.get(b, 0.0):
                    drop.add(a)
                else:
                    drop.add(b)
    keep = [c for c in cols if c not in drop]
    if drop:
        _LOG.info("Correlation prune: dropped %s redundant columns (|r|>%s)", len(drop), corr_threshold)
    return X[keep]


def stability_top_features(
    X: pd.DataFrame,
    y_bin: np.ndarray,
    groups: np.ndarray,
    *,
    top_k: int = 22,
    n_splits: int = 3,
    random_state: int = 42,
) -> list[str]:
    """Average RF importances over GroupKFold; keep top_k columns."""
    n_splits = int(min(n_splits, len(np.unique(groups))))
    if n_splits < 2:
        return list(X.columns)
    gkf = GroupKFold(n_splits=n_splits)
    acc = np.zeros(X.shape[1])
    for tr, _ in gkf.split(X, y_bin, groups):
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X.iloc[tr], y_bin[tr])
        acc += rf.feature_importances_
    acc /= max(1, n_splits)
    order = np.argsort(-acc)[: min(top_k, len(acc))]
    selected = [X.columns[i] for i in order]
    _LOG.info("Stability selection: kept top-%s features (from %s)", len(selected), X.shape[1])
    return selected


def rfe_groupkfold_select(
    X: pd.DataFrame,
    y_bin: np.ndarray,
    groups: np.ndarray,
    *,
    max_features: int = 22,
    min_features: int = 12,
    random_state: int = 42,
) -> list[str]:
    """
    RFECV with GroupKFold + RF classifier (~15–20 features typical for small N).
    Falls back to ``stability_top_features`` if RFECV fails (e.g. degenerate folds).
    """
    cols = list(X.columns)
    n_feat = len(cols)
    if n_feat <= max_features:
        return cols
    n_groups = len(np.unique(groups))
    n_splits = int(min(3, max(2, n_groups - 1)))
    if n_splits < 2:
        return stability_top_features(X, y_bin, groups, top_k=max_features, n_splits=2, random_state=random_state)
    min_f = int(max(5, min(min_features, n_feat - 1)))
    step = max(1, min(3, n_feat // 25))
    rf = RandomForestClassifier(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    gkf = GroupKFold(n_splits=n_splits)
    selector = RFECV(
        estimator=rf,
        step=step,
        min_features_to_select=min_f,
        cv=gkf,
        scoring="roc_auc",
        n_jobs=-1,
    )
    try:
        selector.fit(X, y_bin, groups=groups)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("RFECV+GroupKFold failed (%s); using stability RF top-k", exc)
        return stability_top_features(
            X, y_bin, groups, top_k=min(max_features, n_feat), n_splits=n_splits, random_state=random_state
        )
    support = np.asarray(selector.support_, dtype=bool)
    selected = [cols[i] for i in range(n_feat) if support[i]]
    opt_n = int(getattr(selector, "n_features_", len(selected)))
    _LOG.info(
        "RFECV+GroupKFold: optimal_n=%s, support_count=%s (from %s)",
        opt_n,
        len(selected),
        n_feat,
    )
    if len(selected) > max_features:
        sub = X[selected]
        rf2 = RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        rf2.fit(sub, y_bin)
        order = np.argsort(-rf2.feature_importances_)
        selected = [selected[j] for j in order[:max_features]]
    return selected


def tune_binary_ensemble_weight(oof_enet: np.ndarray, oof_rf: np.ndarray, y_bin: np.ndarray) -> float:
    mask = ~(np.isnan(oof_enet) | np.isnan(oof_rf))
    if mask.sum() < 5 or len(np.unique(y_bin[mask])) < 2:
        return 0.5
    ye, yr, y = oof_enet[mask], oof_rf[mask], y_bin[mask]
    best_w, best_auc = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 21):
        s = w * ye + (1.0 - w) * yr
        try:
            auc = float(roc_auc_score(y, s))
        except ValueError:
            continue
        if auc > best_auc:
            best_auc, best_w = auc, float(w)
    _LOG.info("Ensemble weight (binary): w_enet=%.2f (max OOF AUC=%.4f)", best_w, best_auc)
    return best_w


def tune_regression_ensemble_weight(oof_enet: np.ndarray, oof_rf: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(oof_enet) | np.isnan(oof_rf))
    if mask.sum() < 5:
        return 0.5
    ye, yr, yy = oof_enet[mask], oof_rf[mask], y[mask]
    best_w, best_mae = 0.5, 1e9
    for w in np.linspace(0.0, 1.0, 21):
        s = w * ye + (1.0 - w) * yr
        mae = float(mean_absolute_error(yy, s))
        if mae < best_mae:
            best_mae, best_w = mae, float(w)
    _LOG.info("Ensemble weight (regression): w_enet=%.2f (min OOF MAE=%.4f)", best_w, best_mae)
    return best_w


def fit_platt_calibrator(raw_scores: np.ndarray, y_bin: np.ndarray) -> tuple[np.ndarray, float]:
    """1D logistic Platt scaling on OOF scores in [0,1]. Returns (coef, intercept)."""
    mask = np.isfinite(raw_scores)
    if mask.sum() < 5 or len(np.unique(y_bin[mask])) < 2:
        return np.array([1.0]), 0.0
    lr = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(raw_scores[mask].reshape(-1, 1), y_bin[mask])
    coef = float(lr.coef_.ravel()[0])
    icept = float(lr.intercept_.ravel()[0])
    return np.array([coef]), icept


def platt_predict(raw: float, coef: np.ndarray, intercept: float) -> float:
    z = float(coef[0]) * float(raw) + float(intercept)
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (threshold, max_j). Uses sklearn roc_curve."""
    mask = np.isfinite(scores)
    yt, ys = y_true[mask], scores[mask]
    if len(np.unique(yt)) < 2:
        return 0.5, 0.0
    fpr, tpr, thr = roc_curve(yt, ys)
    j = tpr - fpr
    idx = int(np.argmax(j))
    t = float(thr[idx]) if idx < len(thr) else 0.5
    if not np.isfinite(t):
        t = 0.5
    return t, float(j[idx])


def tune_random_forest_grouped(
    X: pd.DataFrame,
    y_bin: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Randomized search with GroupKFold (small grid for small N)."""
    n_splits = int(min(3, len(np.unique(groups))))
    if n_splits < 2:
        return {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 2}
    gkf = GroupKFold(n_splits=n_splits)
    rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 5, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=12,
        cv=gkf,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y_bin, groups=groups)
    _LOG.info("RF binary tuned: best_auc=%s params=%s", search.best_score_, search.best_params_)
    return dict(search.best_params_)
