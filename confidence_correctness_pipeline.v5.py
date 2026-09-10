#!/usr/bin/env python3
"""
Confidence--Correctness evaluation pipeline for probabilistic classifiers.

This version preserves the experimental behaviour and outputs of pipeline 
while delegating the probability-mass mathematics to ``certainty_ratio.py``
and the class-specific analysis/diagram to ``confidence_correctness_diagram.py``.

This script redesigns and generalises the original certainty-ratio analysis.
It runs all CSV datasets in a directory, evaluates all classifiers defined in
CLASSIFIERS, computes conventional, probabilistic, calibration and
confidence--correctness metrics, optionally performs post-hoc calibration, and
saves reliability diagrams and tabular results.

Main outputs
------------
1. results/AAMMDD_HHmm_detailed_results.csv
   One row per dataset, model and calibration state.
2. results/AAMMDD_HHmm_class_specific_results.csv
   Class-specific R/O/U/A profiles and hard/probabilistic sensitivity/precision.
3. results/AAMMDD_HHmm_aggregate_summary_by_model.csv
   Mean, median, Q1, Q3 and IQR across datasets, grouped by model and calibration.
4. Imagenes/AAMMDD_HHmm_*_reliability.png
   Reliability diagrams for each dataset, model and calibration state.
5. results/AAMMDD_HHmm_confidence_correctness_results.xlsx
   Same information in Excel format, if openpyxl is available.

Assumptions
-----------
- Input files are CSV files.
- Each dataset contains one target column, by default "type" for genomic data.
- All feature columns are numeric after dropping ID/sample columns; non-numeric
  non-target columns can be dropped automatically.
- Each classifier must implement fit and predict_proba, or be wrapped in a
  calibration procedure that provides probabilities.

Author: generated redesign
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import logging
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from certainty_ratio import one_hot
from confidence_correctness_diagram import (
    confidence_correctness_analysis,
    plot_class_confidence_correctness,
)

try:
    from imcp import imcp_score, mcp_score
except Exception:  # pragma: no cover - optional dependency
    imcp_score = None
    mcp_score = None


RANDOM_STATE = 42
EPS = 1e-15
SEPARATOR = ','

# ---------------------------------------------------------------------------
# Classifier dictionary
# ---------------------------------------------------------------------------
# Modify this dictionary to add/remove classifiers. The rest of the script will
# automatically run every selected classifier on every dataset.
'''
    "HGB": HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
    ),
    "GB": GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
    ),
'''
CLASSIFIERS: Dict[str, BaseEstimator] = {
    "RF": RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
    "LR": LogisticRegression(
        max_iter=5000,
        solver="saga",
        penalty="l2",
        n_jobs=1,
        random_state=RANDOM_STATE,
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
}


@dataclass
class Config:
    """Runtime configuration."""

    data_dir: Path = Path("../data_nature")
    results_dir: Path = Path("results")
    images_dir: Path = Path("Imagenes")
    class_col: str = "type"
    drop_columns: Tuple[str, ...] = ("samples", "sample", "id", "ID")
    drop_non_numeric: bool = True
    scaling: str = "standard"  # none, minmax, standard
    n_splits: int = 5
    calibration_folds: int = 3
    calibration_method: str = "isotonic"  # sigmoid or isotonic
    run_calibration: bool = True
    ece_bins: int = 10
    selected_models: Optional[List[str]] = None
    random_state: int = RANDOM_STATE
    save_probabilities: bool = False
    verbose: bool = True
    output_prefix: Optional[str] = None
    generated_files: List[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def safe_name(name: str) -> str:
    """Return a filesystem-safe representation of a name."""
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name[:160]


def make_timestamp_prefix() -> str:
    """Return the required AAMMDD_HHmm_ prefix for every output file."""
    return datetime.now().strftime("%y%m%d_%H%M_")


def prefixed_name(config: Config, filename: str) -> str:
    """Attach the run prefix to an output filename exactly once."""
    prefix = config.output_prefix or ""
    return filename if filename.startswith(prefix) else f"{prefix}{filename}"


def register_generated_file(config: Config, path: Path) -> None:
    """Register files written before the automatic final-time prefix is known."""
    if path is not None:
        config.generated_files.append(path)


def apply_final_prefix_to_registered_files(config: Config) -> None:
    """Rename already-created output files using the final run prefix."""
    if not config.output_prefix:
        return
    renamed: List[Path] = []
    for path in config.generated_files:
        path = Path(path)
        if not path.exists():
            continue
        new_path = path.parent / prefixed_name(config, path.name)
        if new_path == path:
            renamed.append(path)
            continue
        if new_path.exists():
            new_path.unlink()
        path.rename(new_path)
        renamed.append(new_path)
    config.generated_files = renamed


def condition_abbrev(condition: str) -> str:
    """Short condition label for Excel sheet names."""
    if condition == "raw":
        return "raw"
    if condition.startswith("calibrated_"):
        method = condition.replace("calibrated_", "")
        return f"cal_{method[:3]}"
    return safe_name(condition)[:8]


def unique_sheet_name(base: str, used: set) -> str:
    """Return an Excel-safe, unique sheet name with at most 31 characters."""
    base = re.sub(r"[\\/*?:\[\]]", "_", str(base))
    base = base[:31] or "Sheet"
    candidate = base
    i = 1
    while candidate in used:
        suffix = f"_{i}"
        candidate = f"{base[:31 - len(suffix)]}{suffix}"
        i += 1
    used.add(candidate)
    return candidate


def matrix_to_dataframe(matrix: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    """Convert a square class-by-class matrix into a labelled DataFrame."""
    df = pd.DataFrame(matrix, index=classes, columns=classes)
    df.index.name = "true_class"
    return df.reset_index()


def save_dataset_workbooks(dataset_name: str, export: Dict[str, List], config: Config) -> None:
    """Save the two per-dataset Excel files: analysis workbook and probabilities workbook."""
    safe_dataset = safe_name(dataset_name)
    analysis_path = config.results_dir / prefixed_name(config, f"{safe_dataset}.xlsx")
    probabilities_path = config.results_dir / prefixed_name(config, f"{safe_dataset}_probab.xlsx")

    with pd.ExcelWriter(analysis_path, engine="openpyxl") as writer:
        used_sheets = set()
        pd.DataFrame(export["metrics"]).to_excel(
            writer, sheet_name=unique_sheet_name("Summary", used_sheets), index=False
        )
        if export["class_specific"]:
            pd.concat(export["class_specific"], ignore_index=True).to_excel(
                writer, sheet_name=unique_sheet_name("Class_specific", used_sheets), index=False
            )
        if export["reliability_bins"]:
            pd.concat(export["reliability_bins"], ignore_index=True).to_excel(
                writer, sheet_name=unique_sheet_name("Reliability_bins", used_sheets), index=False
            )
        for item in export["matrices"]:
            model = safe_name(item["model"])
            cond = condition_abbrev(item["condition"])
            for matrix_name, matrix_df in item["matrices"].items():
                sheet = unique_sheet_name(f"{model}_{cond}_{matrix_name}", used_sheets)
                matrix_df.to_excel(writer, sheet_name=sheet, index=False)

    with pd.ExcelWriter(probabilities_path, engine="openpyxl") as writer:
        used_sheets = set()
        for item in export["probabilities"]:
            model = safe_name(item["model"])
            cond = condition_abbrev(item["condition"])
            sheet = unique_sheet_name(f"{model}_{cond}", used_sheets)
            item["probabilities"].to_excel(writer, sheet_name=sheet, index=False)


def find_class_column(df: pd.DataFrame, requested: str) -> str:
    """Find a target column case-insensitively."""
    lower = {c.lower(): c for c in df.columns}
    if requested.lower() in lower:
        return lower[requested.lower()]
    # Common fallbacks for genomic and UCI-style datasets.
    for candidate in ("type", "class", "target", "label", "y", "Class"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(
        f"Target column '{requested}' not found. Available columns: {list(df.columns)[:10]}..."
    )


def load_dataset(path: Path, config: Config) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load one CSV dataset, clean columns and return X, y, feature names."""
    df = pd.read_csv(path, sep = SEPARATOR )
    target_col = find_class_column(df, config.class_col)

    # Drop common sample identifier columns if present and not target.
    drop_cols = [c for c in config.drop_columns if c in df.columns and c != target_col]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    if config.drop_non_numeric:
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logging.warning(
                "%s: dropping %d non-numeric feature columns: %s",
                path.name,
                len(non_numeric),
                non_numeric[:5],
            )
            X = X.drop(columns=non_numeric)

    if X.shape[1] == 0:
        raise ValueError(f"No numeric feature columns remain in {path.name}.")

    # Ensure numeric dtype when possible.
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y, list(X.columns)


def choose_n_splits(y: pd.Series, requested: int) -> int:
    """Choose a feasible number of stratified folds."""
    min_count = y.value_counts().min()
    if min_count < 2:
        raise ValueError("At least one class has fewer than two samples; stratified CV is not feasible.")
    n_splits = min(requested, int(min_count))
    if n_splits < requested:
        logging.warning(
            "Reducing n_splits from %d to %d because the rarest class has %d samples.",
            requested,
            n_splits,
            min_count,
        )
    return n_splits


def make_preprocessor(config: Config) -> Pipeline:
    """Create preprocessing pipeline: imputation plus optional scaling."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if config.scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif config.scaling == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif config.scaling == "none":
        pass
    else:
        raise ValueError("scaling must be one of: none, minmax, standard")
    return Pipeline(steps)


def make_pipeline(estimator: BaseEstimator, config: Config) -> Pipeline:
    """Wrap estimator with imputation/scaling to avoid data leakage."""
    return Pipeline([
        ("preprocess", make_preprocessor(config)),
        ("classifier", clone(estimator)),
    ])


def make_calibrated_classifier(estimator: BaseEstimator, cv, method: str) -> CalibratedClassifierCV:
    """Create CalibratedClassifierCV while supporting old/new sklearn APIs."""
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:  # older sklearn versions
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def align_proba(
    estimator: BaseEstimator,
    proba: np.ndarray,
    global_classes: np.ndarray,
) -> np.ndarray:
    """Align estimator probability columns to the global class order."""
    if not hasattr(estimator, "classes_"):
        if proba.shape[1] != len(global_classes):
            raise ValueError("Estimator has no classes_ attribute and proba shape is incompatible.")
        return proba

    est_classes = np.asarray(estimator.classes_)
    aligned = np.zeros((proba.shape[0], len(global_classes)), dtype=float)
    class_to_col = {label: j for j, label in enumerate(est_classes)}
    for j, label in enumerate(global_classes):
        if label in class_to_col:
            aligned[:, j] = proba[:, class_to_col[label]]
    row_sums = aligned.sum(axis=1, keepdims=True)
    bad = row_sums.squeeze() <= 0
    if np.any(bad):
        # This should not happen under stratified CV, but use a safe fallback.
        aligned[bad, :] = 1.0 / len(global_classes)
        row_sums = aligned.sum(axis=1, keepdims=True)
    aligned = aligned / row_sums
    return aligned


# ---------------------------------------------------------------------------
# Standard probabilistic and calibration metrics
# ---------------------------------------------------------------------------

def multiclass_brier_score(y_idx: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    y_onehot = one_hot(y_idx, n_classes)
    return float(np.mean(np.sum((y_onehot - proba) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, pd.DataFrame]:
    """Confidence-ECE for multiclass classification.

    Bins predictions by max probability, then compares bin accuracy and
    average confidence.
    """
    confidence = np.max(proba, axis=1)
    correct = (y_true == y_pred).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            acc_bin = np.nan
            conf_bin = np.nan
            gap = np.nan
        else:
            acc_bin = float(correct[mask].mean())
            conf_bin = float(confidence[mask].mean())
            gap = abs(acc_bin - conf_bin)
            ece += (count / n) * gap
        rows.append({
            "bin": b + 1,
            "bin_left": lo,
            "bin_right": hi,
            "count": count,
            "accuracy": acc_bin,
            "confidence": conf_bin,
            "abs_gap": gap,
        })
    return float(ece), pd.DataFrame(rows)


def compute_standard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    ece_bins: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    le = LabelEncoder().fit(classes)
    y_idx = le.transform(y_true)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "nll": float(log_loss(y_true, np.clip(proba, EPS, 1 - EPS), labels=classes)),
        "brier": multiclass_brier_score(y_idx, proba, len(classes)),
    }
    ece, reliability_df = expected_calibration_error(y_true, y_pred, proba, ece_bins)
    metrics["ece"] = ece

    if mcp_score is not None:
        try:
            metrics["mcp"] = float(mcp_score(y_true, proba, list(classes)))
        except Exception as exc:
            logging.warning("Could not compute MCP: %s", exc)
            metrics["mcp"] = np.nan
    else:
        metrics["mcp"] = np.nan

    if imcp_score is not None:
        try:
            metrics["imcp"] = float(imcp_score(y_true, proba, list(classes)))
        except Exception as exc:
            logging.warning("Could not compute IMCP: %s", exc)
            metrics["imcp"] = np.nan
    else:
        metrics["imcp"] = np.nan

    return metrics, reliability_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    reliability_df: pd.DataFrame,
    ece: float,
    title: str,
    output_path: Path,
) -> None:
    """Save a reliability diagram as PNG."""
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Perfect calibration")

    valid = reliability_df.dropna(subset=["accuracy", "confidence"])
    if not valid.empty:
        ax.plot(valid["confidence"], valid["accuracy"], marker="o", linewidth=1.5, label="Observed")
        # Add light count labels for non-empty bins.
        for _, row in valid.iterrows():
            ax.annotate(str(int(row["count"])), (row["confidence"], row["accuracy"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-validation execution
# ---------------------------------------------------------------------------

def fit_predict_cv(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator,
    classes: np.ndarray,
    config: Config,
    calibrated: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return out-of-fold predictions and probabilities.

    Labels are encoded internally as integers before fitting. This is
    important for estimators such as ``MLPClassifier`` with early stopping,
    which can fail when validation scoring receives string labels. The
    returned predictions are mapped back to the original class labels, while
    probability columns are aligned to the original ``classes`` order.
    """
    label_encoder = LabelEncoder().fit(classes)
    y_encoded = pd.Series(label_encoder.transform(y), index=y.index)
    encoded_classes = np.arange(len(classes))

    n_splits = choose_n_splits(y_encoded, config.n_splits)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)

    y_pred_encoded_all = np.empty(len(y), dtype=int)
    proba_all = np.zeros((len(y), len(classes)), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_encoded), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_encoded.iloc[train_idx]

        base = make_pipeline(estimator, config)
        if calibrated:
            min_train_class = y_train.value_counts().min()
            cal_folds = min(config.calibration_folds, int(min_train_class))
            if cal_folds < 2:
                raise ValueError(
                    f"Cannot calibrate: rarest class in training fold has {min_train_class} sample(s)."
                )
            inner_cv = StratifiedKFold(
                n_splits=cal_folds,
                shuffle=True,
                random_state=config.random_state + fold,
            )
            model = make_calibrated_classifier(base, cv=inner_cv, method=config.calibration_method)
        else:
            model = base

        model.fit(X_train, y_train)
        y_pred_encoded = model.predict(X_test).astype(int)
        proba = model.predict_proba(X_test)
        proba = align_proba(model, proba, encoded_classes)

        y_pred_encoded_all[test_idx] = y_pred_encoded
        proba_all[test_idx, :] = proba

    y_pred_all = classes[y_pred_encoded_all]
    return y_pred_all, proba_all


def evaluate_one_condition(
    dataset_name: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator,
    classes: np.ndarray,
    config: Config,
    calibrated: bool,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Evaluate one dataset/model/calibration condition."""
    condition = f"calibrated_{config.calibration_method}" if calibrated else "raw"
    logging.info("Evaluating %s | %s | %s", dataset_name, model_name, condition)

    y_true = y.to_numpy()
    y_pred, proba = fit_predict_cv(X, y, estimator, classes, config, calibrated=calibrated)

    standard, reliability_df = compute_standard_metrics(
        y_true=y_true,
        y_pred=y_pred,
        proba=proba,
        classes=classes,
        ece_bins=config.ece_bins,
    )
    cc_metrics, class_df, matrices = confidence_correctness_analysis(y_true, y_pred, proba, classes)

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "condition": condition,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(classes)),
        **standard,
        **cc_metrics,
    }

    # Save reliability diagram.
    image_stem = f"{safe_name(dataset_name)}__{safe_name(model_name)}__{condition}"
    reliability_path = config.images_dir / prefixed_name(config, f"{image_stem}_reliability.png")
    confidence_path = config.images_dir / prefixed_name(config, f"{image_stem}_confidence_correctness.png")
    plot_reliability_diagram(
        reliability_df,
        ece=standard["ece"],
        title=f"{dataset_name} | {model_name} | {condition}",
        output_path=reliability_path,
    )
    plot_class_confidence_correctness(
        class_df,
        title=f"{dataset_name} | {model_name} | {condition}",
        output_path=confidence_path,
    )
    register_generated_file(config, reliability_path)
    register_generated_file(config, confidence_path)

    class_df.insert(0, "condition", condition)
    class_df.insert(0, "model", model_name)
    class_df.insert(0, "dataset", dataset_name)

    reliability_df.insert(0, "condition", condition)
    reliability_df.insert(0, "model", model_name)
    reliability_df.insert(0, "dataset", dataset_name)

    prob_df = pd.DataFrame(proba, columns=[f"prob_{c}" for c in classes])
    prob_df.insert(0, "y_pred", y_pred)
    prob_df.insert(0, "y_true", y_true)

    if config.save_probabilities:
        prob_path = config.results_dir / prefixed_name(config, f"{image_stem}_probabilities.csv")
        prob_df.to_csv(prob_path, index=False)
        register_generated_file(config, prob_path)

    matrix_dfs = {
        name: matrix_to_dataframe(value, classes)
        for name, value in matrices.items()
    }
    export_bundle = {
        "model": model_name,
        "condition": condition,
        "matrices": matrix_dfs,
        "probabilities": prob_df,
    }

    return result, class_df, reliability_df, export_bundle


def aggregate_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, median and IQR across datasets by model and condition."""
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    # Do not aggregate identifiers/counts as performance summaries.
    exclude = {"n_samples", "n_features", "n_classes"}
    metric_cols = [c for c in numeric_cols if c not in exclude]

    rows = []
    for (model, condition), group in results_df.groupby(["model", "condition"]):
        row = {"model": model, "condition": condition, "n_datasets": group["dataset"].nunique()}
        for col in metric_cols:
            vals = group[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_q1"] = np.nan
                row[f"{col}_q3"] = np.nan
                row[f"{col}_iqr"] = np.nan
            else:
                q1, q3 = np.percentile(vals, [25, 75])
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_median"] = float(np.median(vals))
                row[f"{col}_q1"] = float(q1)
                row[f"{col}_q3"] = float(q3)
                row[f"{col}_iqr"] = float(q3 - q1)
        rows.append(row)
    return pd.DataFrame(rows)


def run(config: Config) -> None:
    setup_logging(config.verbose)
    auto_output_prefix = config.output_prefix is None
    if auto_output_prefix:
        config.output_prefix = ""
    elif config.output_prefix and not config.output_prefix.endswith("_"):
        config.output_prefix = f"{config.output_prefix}_"

    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.images_dir.mkdir(parents=True, exist_ok=True)

    if not config.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {config.data_dir}")

    csv_files = sorted(config.data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {config.data_dir}")

    if config.selected_models:
        missing = sorted(set(config.selected_models) - set(CLASSIFIERS))
        if missing:
            raise ValueError(f"Unknown model(s): {missing}. Available: {list(CLASSIFIERS)}")
        classifiers = {k: CLASSIFIERS[k] for k in config.selected_models}
    else:
        classifiers = CLASSIFIERS

    all_results: List[Dict[str, float]] = []
    all_class_results: List[pd.DataFrame] = []
    all_reliability_bins: List[pd.DataFrame] = []
    per_dataset_exports: Dict[str, Dict[str, List]] = {}
    failed: List[Dict[str, str]] = []

    for path in csv_files:
        dataset_name = path.stem
        try:
            X, y, _ = load_dataset(path, config)
            classes = np.asarray(LabelEncoder().fit(y).classes_)
            logging.info(
                "Dataset %s: n=%d, d=%d, k=%d, class sizes=%s",
                dataset_name,
                len(y),
                X.shape[1],
                len(classes),
                y.value_counts().to_dict(),
            )
            per_dataset_exports[dataset_name] = {
                "metrics": [],
                "class_specific": [],
                "reliability_bins": [],
                "matrices": [],
                "probabilities": [],
            }
        except Exception as exc:
            logging.exception("Skipping dataset %s due to loading/preparation error.", dataset_name)
            failed.append({"dataset": dataset_name, "model": "-", "condition": "load", "error": str(exc)})
            continue

        for model_name, estimator in classifiers.items():
            for calibrated in ([False, True] if config.run_calibration else [False]):
                try:
                    result, class_df, rel_df, export_bundle = evaluate_one_condition(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        X=X,
                        y=y,
                        estimator=estimator,
                        classes=classes,
                        config=config,
                        calibrated=calibrated,
                    )
                    all_results.append(result)
                    all_class_results.append(class_df)
                    all_reliability_bins.append(rel_df)
                    per_dataset_exports[dataset_name]["metrics"].append(result)
                    per_dataset_exports[dataset_name]["class_specific"].append(class_df.copy())
                    per_dataset_exports[dataset_name]["reliability_bins"].append(rel_df.copy())
                    per_dataset_exports[dataset_name]["matrices"].append(export_bundle)
                    per_dataset_exports[dataset_name]["probabilities"].append(export_bundle)
                except Exception as exc:
                    condition = f"calibrated_{config.calibration_method}" if calibrated else "raw"
                    logging.exception("Failed %s | %s | %s", dataset_name, model_name, condition)
                    failed.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "condition": condition,
                        "error": str(exc),
                    })

    if not all_results:
        raise RuntimeError("No successful evaluations were completed.")

    if auto_output_prefix:
        config.output_prefix = make_timestamp_prefix()
        apply_final_prefix_to_registered_files(config)

    results_df = pd.DataFrame(all_results)
    class_results_df = pd.concat(all_class_results, ignore_index=True) if all_class_results else pd.DataFrame()
    rel_bins_df = pd.concat(all_reliability_bins, ignore_index=True) if all_reliability_bins else pd.DataFrame()
    agg_df = aggregate_summary(results_df)
    failed_df = pd.DataFrame(failed)

    for dataset_name, export in per_dataset_exports.items():
        if export["metrics"]:
            save_dataset_workbooks(dataset_name, export, config)

    results_df.to_csv(config.results_dir / prefixed_name(config, "detailed_results.csv"), index=False)
    class_results_df.to_csv(config.results_dir / prefixed_name(config, "class_specific_results.csv"), index=False)
    rel_bins_df.to_csv(config.results_dir / prefixed_name(config, "reliability_bins.csv"), index=False)
    agg_df.to_csv(config.results_dir / prefixed_name(config, "aggregate_summary_by_model.csv"), index=False)
    if not failed_df.empty:
        failed_df.to_csv(config.results_dir / prefixed_name(config, "failed_runs.csv"), index=False)

    try:
        with pd.ExcelWriter(config.results_dir / prefixed_name(config, "confidence_correctness_results.xlsx"), engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Detailed results", index=False)
            class_results_df.to_excel(writer, sheet_name="Class-specific", index=False)
            agg_df.to_excel(writer, sheet_name="Aggregate", index=False)
            rel_bins_df.to_excel(writer, sheet_name="Reliability bins", index=False)
            if not failed_df.empty:
                failed_df.to_excel(writer, sheet_name="Failed", index=False)
    except Exception as exc:
        logging.warning("Excel export failed; CSV files were still saved. Error: %s", exc)

    logging.info("Saved results to %s", config.results_dir.resolve())
    logging.info("Saved images to %s", config.images_dir.resolve())
    if failed:
        logging.warning("%d runs failed. See failed_runs.csv.", len(failed))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> None:
    """Run a small internal test on synthetic data."""
    from sklearn.datasets import make_classification

    tmp = Path("/tmp/confidence_correctness_selftest")
    data_dir = tmp / "data"
    results_dir = tmp / "results"
    images_dir = tmp / "Imagenes"
    data_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=90,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["type"] = np.array([f"class_{v}" for v in y])
    df.to_csv(data_dir / "synthetic.csv", index=False)

    cfg = Config(
        data_dir=data_dir,
        results_dir=results_dir,
        images_dir=images_dir,
        selected_models=["RF", "LR"],
        n_splits=3,
        calibration_folds=2,
        run_calibration=True,
        verbose=True,
    )
    run(cfg)
    out = pd.read_csv(next(results_dir.glob("*_detailed_results.csv")))
    assert not out.empty
    for col in ["accuracy", "acc_star", "R", "O", "U", "A", "ece", "nll", "brier"]:
        assert col in out.columns, col
    assert np.allclose(out["R"] + out["O"] + out["U"] + out["A"], 1.0, atol=1e-8)
    print("Self-test completed successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run confidence--correctness analysis on all CSV datasets in a folder."
    )
    parser.add_argument("--data-dir", default="../data_nature", help="Folder containing CSV datasets.")
    parser.add_argument("--results-dir", default="results", help="Folder for result tables.")
    parser.add_argument("--images-dir", default="Imagenes", help="Folder for reliability diagrams and class diagrams.")
    parser.add_argument("--class-col", default="type", help="Target column name; searched case-insensitively.")
    parser.add_argument("--scaling", choices=["none", "minmax", "standard"], default="standard")
    parser.add_argument("--folds", type=int, default=5, help="Outer stratified CV folds.")
    parser.add_argument("--calibration-folds", type=int, default=3, help="Inner folds for post-hoc calibration.")
    parser.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="isotonic")
    parser.add_argument("--no-calibration", action="store_true", help="Disable post-hoc calibration runs.")
    parser.add_argument("--models", default=None, help="Comma-separated subset of CLASSIFIERS, e.g. RF,LR,HGB.")
    parser.add_argument("--ece-bins", type=int, default=10, help="Number of bins for ECE/reliability diagrams.")
    parser.add_argument("--keep-non-numeric", default='True',action="store_true", help="Do not drop non-numeric feature columns.")
    parser.add_argument("--save-probabilities", action="store_true", help="Save out-of-fold probabilities per run.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-prefix", default=None, help="Optional output prefix. If omitted, AAMMDD_HHmm_ is generated automatically.")
    parser.add_argument("--self-test", action="store_true", help="Run a quick synthetic self-test and exit.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        setup_logging(True)
        self_test()
        return

    selected_models = None
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    cfg = Config(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        images_dir=Path(args.images_dir),
        class_col=args.class_col,
        drop_non_numeric=not args.keep_non_numeric,
        scaling=args.scaling,
        n_splits=args.folds,
        calibration_folds=args.calibration_folds,
        calibration_method=args.calibration_method,
        run_calibration=not args.no_calibration,
        ece_bins=args.ece_bins,
        selected_models=selected_models,
        save_probabilities=args.save_probabilities,
        verbose=not args.quiet,
        output_prefix=args.output_prefix,
    )
    run(cfg)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
