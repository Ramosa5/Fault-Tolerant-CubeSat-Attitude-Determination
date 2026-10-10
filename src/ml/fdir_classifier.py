from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, f1_score

from src.ml.models import build_model, count_parameters

SENSORS = ("gyro", "sun", "mag")


def fit_feature_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, np.float32)
    mu = X.mean(axis=(0, 1), keepdims=True)
    sd = X.std(axis=(0, 1), keepdims=True)
    sd[sd < 1e-7] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_features(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, np.float32) - mu) / sd).astype(np.float32)


def choose_multilabel_thresholds(y_true: np.ndarray, probabilities: np.ndarray,
                                 grid: np.ndarray | None = None) -> np.ndarray:
    """Tune one threshold per sensor on validation data only.

    The criterion is F1. Ties prefer higher precision and then a threshold closer
    to 0.5. The selected thresholds are frozen for the held-out test/evaluation runs.
    """
    y = np.asarray(y_true, np.int64)
    p = np.asarray(probabilities, float)
    if grid is None:
        grid = np.linspace(0.10, 0.90, 33)
    out = np.full(3, 0.5, float)
    for j in range(3):
        best = (-1.0, -1.0, -1.0, 0.5)
        for th in grid:
            pred = p[:, j] >= float(th)
            pr, rc, f1, _ = precision_recall_fscore_support(
                y[:, j], pred.astype(int), average="binary", zero_division=0
            )
            score = (float(f1), float(pr), -abs(float(th) - 0.5), float(th))
            if score > best:
                best = score
        out[j] = best[-1]
    return out


def _normalize_active_mask(active_mask: np.ndarray | None, n: int) -> np.ndarray:
    if active_mask is None:
        return np.ones((n, 3), dtype=bool)
    m = np.asarray(active_mask, bool)
    if m.ndim == 1:
        m = np.repeat(m[:, None], 3, axis=1)
    if m.shape != (n, 3):
        raise ValueError(f"active_mask must have shape {(n, 3)}, got {m.shape}")
    return m


def apply_probability_filter(probabilities: np.ndarray, thresholds: np.ndarray, *,
                             alpha: float = 0.30, persistence_samples: int = 3,
                             reset_samples: int = 2,
                             active_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Online-causal EWMA + persistence for three independent fault probabilities.

    ``active_mask`` makes calibration and runtime behaviour identical. Whenever a
    sensor is not eligible for FDIR (before arming, or Sun during expected eclipse),
    its EWMA/persistence state is reset instead of silently accumulating evidence.
    """
    p = np.asarray(probabilities, float)
    th = np.asarray(thresholds, float).reshape(3)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    persistence_samples = max(1, int(persistence_samples))
    reset_samples = max(1, int(reset_samples))
    mask = _normalize_active_mask(active_mask, len(p))
    smooth = np.zeros_like(p, float)
    det = np.zeros_like(p, bool)
    state = np.zeros(3, float)
    bad = np.zeros(3, int)
    good = np.zeros(3, int)
    active = np.zeros(3, bool)
    for k in range(len(p)):
        sample = np.nan_to_num(p[k], nan=0.0, posinf=1.0, neginf=0.0)
        for j in range(3):
            if not mask[k, j]:
                state[j] = 0.0
                bad[j] = 0
                good[j] = 0
                active[j] = False
                smooth[k, j] = 0.0
                det[k, j] = False
                continue
            state[j] = (1.0 - alpha) * state[j] + alpha * sample[j]
            smooth[k, j] = state[j]
            if state[j] >= th[j]:
                bad[j] += 1
                good[j] = 0
            else:
                good[j] += 1
                bad[j] = 0
            if not active[j] and bad[j] >= persistence_samples:
                active[j] = True
            elif active[j] and good[j] >= reset_samples:
                active[j] = False
            det[k, j] = active[j]
    return smooth, det


def _binary_metrics_from_sequences(records: list[dict], sensor_index: int, threshold: float, *,
                                    alpha: float, persistence_samples: int, reset_samples: int,
                                    hybrid: bool = False) -> dict:
    """Aggregate one threshold over COMPLETE sequences using runtime filtering."""
    tp = fp = fn = tn = 0
    for rec in records:
        p = np.asarray(rec["probabilities"], float)
        y = np.asarray(rec["truth"], bool)
        mask = np.asarray(rec["eval_mask"], bool)
        th = np.full(3, 1.01, float)
        th[int(sensor_index)] = float(threshold)
        _, ml_det = apply_probability_filter(
            p, th, alpha=alpha, persistence_samples=persistence_samples,
            reset_samples=reset_samples, active_mask=mask,
        )
        pred = ml_det[:, int(sensor_index)]
        if hybrid:
            classical = np.asarray(rec["classical"], bool)[:, int(sensor_index)]
            pred = classical | pred
        m = mask[:, int(sensor_index)]
        yy = y[:, int(sensor_index)]
        pp = pred
        tp += int(np.sum(m & yy & pp))
        fp += int(np.sum(m & ~yy & pp))
        fn += int(np.sum(m & yy & ~pp))
        tn += int(np.sum(m & ~yy & ~pp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "far": float(far), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def choose_runtime_multilabel_thresholds(records: list[dict], *,
                                         alpha: float = 0.30,
                                         persistence_samples: int = 3,
                                         reset_samples: int = 2,
                                         max_false_alarm_rate_pct: float = 1.0,
                                         hybrid: bool = False,
                                         max_incremental_far_pct: float = 1.0,
                                         grid: np.ndarray | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Tune thresholds on complete validation sequences using runtime logic.

    For pure ML, F1 is maximized subject to sample-wise FAR <=
    ``max_false_alarm_rate_pct``. For hybrid operation, the candidate is ORed with
    the complete classical-v3.1 decision and may increase its validation FAR by at
    most ``max_incremental_far_pct`` percentage points. A threshold of 1.01 is
    always included, so the safe no-ML-augmentation solution is available.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.99, 48)
    candidates = np.unique(np.r_[np.asarray(grid, float), 1.01])
    thresholds = np.full(3, 1.01, float)
    rows = []
    for j, sensor in enumerate(SENSORS):
        baseline = _binary_metrics_from_sequences(
            records, j, 1.01, alpha=alpha, persistence_samples=persistence_samples,
            reset_samples=reset_samples, hybrid=hybrid,
        )
        if hybrid:
            far_limit = baseline["far"] + float(max_incremental_far_pct) / 100.0
        else:
            far_limit = float(max_false_alarm_rate_pct) / 100.0
        best = None
        candidate_rows = []
        for th in candidates:
            m = _binary_metrics_from_sequences(
                records, j, float(th), alpha=alpha,
                persistence_samples=persistence_samples, reset_samples=reset_samples,
                hybrid=hybrid,
            )
            feasible = m["far"] <= far_limit + 1e-12
            # Under the FAR constraint, prefer F1, then recall, precision and a
            # lower threshold. This preserves sensitivity without relaxing safety.
            score = (m["f1"], m["recall"], m["precision"], -float(th))
            candidate_rows.append((float(th), feasible, score, m))
            if feasible and (best is None or score > best[2]):
                best = candidate_rows[-1]
        if best is None:
            # Should not happen because 1.01 gives no pure-ML detections and, for
            # hybrid mode, reproduces the classical baseline exactly.
            best = min(candidate_rows, key=lambda x: (x[3]["far"], -x[3]["f1"]))
        th, feasible, _, m = best
        thresholds[j] = th
        rows.append({
            "sensor": sensor,
            "mode": "hybrid_augmentation" if hybrid else "pure_ml",
            "threshold": th,
            "precision_pct": 100.0 * m["precision"],
            "recall_pct": 100.0 * m["recall"],
            "f1_pct": 100.0 * m["f1"],
            "false_alarm_rate_pct": 100.0 * m["far"],
            "far_limit_pct": 100.0 * far_limit,
            "classical_baseline_precision_pct": 100.0 * baseline["precision"] if hybrid else np.nan,
            "classical_baseline_recall_pct": 100.0 * baseline["recall"] if hybrid else np.nan,
            "classical_baseline_f1_pct": 100.0 * baseline["f1"] if hybrid else np.nan,
            "classical_baseline_far_pct": 100.0 * baseline["far"] if hybrid else np.nan,
            "feasible": bool(feasible),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        })
    return thresholds, pd.DataFrame(rows)


def evaluate_runtime_multilabel_thresholds(records: list[dict], thresholds: np.ndarray, *,
                                           alpha: float = 0.30,
                                           persistence_samples: int = 3,
                                           reset_samples: int = 2,
                                           hybrid: bool = False) -> pd.DataFrame:
    rows = []
    for j, sensor in enumerate(SENSORS):
        m = _binary_metrics_from_sequences(
            records, j, float(np.asarray(thresholds)[j]), alpha=alpha,
            persistence_samples=persistence_samples, reset_samples=reset_samples,
            hybrid=hybrid,
        )
        rows.append({
            "sensor": sensor, "mode": "hybrid" if hybrid else "pure_ml",
            "threshold": float(np.asarray(thresholds)[j]),
            "precision_pct": 100.0 * m["precision"],
            "recall_pct": 100.0 * m["recall"],
            "f1_pct": 100.0 * m["f1"],
            "false_alarm_rate_pct": 100.0 * m["far"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        })
    return pd.DataFrame(rows)


@dataclass
class MLFDIRBundle:
    architecture: str
    model: nn.Module
    window: int
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    thresholds: np.ndarray
    metrics: dict
    hybrid_thresholds: np.ndarray | None = None

    def clone(self) -> "MLFDIRBundle":
        return MLFDIRBundle(
            architecture=self.architecture,
            model=copy.deepcopy(self.model).eval(),
            window=self.window,
            feature_names=list(self.feature_names),
            mean=self.mean.copy(),
            std=self.std.copy(),
            thresholds=self.thresholds.copy(),
            metrics=copy.deepcopy(self.metrics),
            hybrid_thresholds=None if self.hybrid_thresholds is None else self.hybrid_thresholds.copy(),
        )

    def predict_windows(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        xs = standardize_features(X, self.mean, self.std)
        out = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(xs), int(batch_size)):
                logits = self.model(torch.from_numpy(xs[i:i + int(batch_size)]))
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 3), float)

    def predict_sequence(self, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Causal sliding-window probabilities aligned to every input sample."""
        f = np.asarray(features, np.float32)
        if len(f) == 0:
            return np.zeros((0, 3), float)
        w = int(self.window)
        pad = np.repeat(f[:1], w - 1, axis=0)
        fp = np.vstack([pad, f])
        shape = (len(f), w, f.shape[1])
        strides = (fp.strides[0], fp.strides[0], fp.strides[1])
        windows = np.lib.stride_tricks.as_strided(fp, shape=shape, strides=strides).copy()
        return self.predict_windows(windows, batch_size=batch_size)


def save_ml_fdir_bundle(bundle: MLFDIRBundle, out_dir: Path | str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": bundle.model.state_dict(),
        "architecture": bundle.architecture,
        "window": int(bundle.window),
        "n_features": len(bundle.feature_names),
        "feature_names": list(bundle.feature_names),
        "mean": bundle.mean,
        "std": bundle.std,
        "thresholds": np.asarray(bundle.thresholds, float),
        "hybrid_thresholds": None if bundle.hybrid_thresholds is None else np.asarray(bundle.hybrid_thresholds, float),
        "metrics": bundle.metrics,
    }
    torch.save(payload, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(bundle.metrics, indent=2), encoding="utf-8")


def train_multilabel_fdir_model(architecture: str, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray, *,
                                feature_names: list[str], window: int, training_cfg: dict,
                                out_dir: Path | str) -> tuple[MLFDIRBundle, pd.DataFrame]:
    """Train one lightweight MLP/CNN/GRU multi-label FDIR detector."""
    arch = str(architecture).lower()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(training_cfg.get("weight_seed", 11001)) + {"mlp": 0, "cnn": 100, "gru": 200}[arch]
    torch.manual_seed(seed)
    np.random.seed(seed)

    mu, sd = fit_feature_scaler(X_train)
    xtr = standardize_features(X_train, mu, sd)
    xva = standardize_features(X_val, mu, sd)
    xte = standardize_features(X_test, mu, sd)
    ytr = np.asarray(y_train, np.float32)
    yva = np.asarray(y_val, np.float32)
    yte = np.asarray(y_test, np.float32)

    model = build_model(arch, xtr.shape[2], int(window), 3)
    counts = ytr.sum(axis=0)
    negatives = len(ytr) - counts
    pos_weight = np.clip(negatives / np.maximum(counts, 1.0), 1.0, float(training_cfg.get("max_pos_weight", 25.0)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    batch = int(training_cfg.get("batch_size", 256))
    epochs = int(training_cfg.get("epochs", 20))
    patience = int(training_cfg.get("early_stopping_patience", 5))
    gen = torch.Generator().manual_seed(int(training_cfg.get("loader_seed", 11002)))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=batch, shuffle=True, generator=gen,
    )
    xv = torch.from_numpy(xva)
    yv = torch.from_numpy(yva)
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(xv), yv).item())
        history.append({"epoch": epoch, "train_loss": total / max(1, seen), "val_loss": val_loss})
        if val_loss < best_loss - 1e-10:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(torch.from_numpy(xva))).numpy()
        p_test = torch.sigmoid(model(torch.from_numpy(xte))).numpy()
    thresholds = choose_multilabel_thresholds(y_val, p_val)
    pred_test = p_test >= thresholds.reshape(1, 3)

    rows = []
    for j, sensor in enumerate(SENSORS):
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_test[:, j], pred_test[:, j].astype(int), average="binary", zero_division=0
        )
        rows.append({
            "architecture": arch, "sensor": sensor, "precision": float(pr),
            "recall": float(rc), "f1": float(f1), "threshold": float(thresholds[j]),
            "positive_test_samples": int(np.sum(y_test[:, j])),
        })
    macro_f1 = float(np.mean([r["f1"] for r in rows]))

    # Batch-1 CPU timing.
    one = torch.from_numpy(xte[:1]) if len(xte) else torch.from_numpy(xtr[:1])
    reps = int(training_cfg.get("inference_timing_repetitions", 300))
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max(1, reps)):
            model(one)
    inference_ms = 1000.0 * (time.perf_counter() - t0) / max(1, reps)
    params = int(count_parameters(model))
    size_mb = float(sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2)
    metrics = {
        "architecture": arch,
        "window": int(window),
        "n_features": int(xtr.shape[2]),
        "feature_names": list(feature_names),
        "train_samples": int(len(xtr)),
        "validation_samples": int(len(xva)),
        "test_samples": int(len(xte)),
        "best_validation_loss": float(best_loss),
        "preliminary_window_thresholds": {s: float(thresholds[i]) for i, s in enumerate(SENSORS)},
        "thresholds": {s: float(thresholds[i]) for i, s in enumerate(SENSORS)},
        "test_macro_f1": macro_f1,
        "parameters": params,
        "model_size_mb": size_mb,
        "inference_ms_cpu_batch1": float(inference_ms),
        "positive_class_weights": {s: float(pos_weight[i]) for i, s in enumerate(SENSORS)},
    }
    torch.save({
        "state_dict": model.state_dict(), "architecture": arch, "window": int(window),
        "n_features": int(xtr.shape[2]), "feature_names": list(feature_names),
        "mean": mu, "std": sd, "thresholds": thresholds, "metrics": metrics,
    }, out / "model.pt")
    pd.DataFrame(history).to_csv(out / "learning_curve.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "test_sensor_metrics.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle = MLFDIRBundle(arch, model.eval(), int(window), list(feature_names), mu, sd, thresholds, metrics)
    return bundle, pd.DataFrame(rows)
