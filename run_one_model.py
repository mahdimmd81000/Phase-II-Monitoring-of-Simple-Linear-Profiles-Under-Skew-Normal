"""Reviewer 1 request: benchmark RF, SVM, kNN, MLP and Elastic-Net logistic
against the proposed Gradient-Boosting method under IDENTICAL conditions.

Every model gets:
  * the same 16-feature profile representation
  * the same simulated dataset and the same stratified 80/20 train/test split
  * the same 5-fold stratified CV and grid-search protocol
  * the same in-control ARL0 calibration (target ARL0 = 200, 10,000 profiles)
  * the same Phase II ARL scenarios (small/moderate/large intercept shifts)

Usage:
  python run_one_model.py --model RandomForest
  python run_one_model.py --model all          # runs the six models sequentially
  python run_one_model.py --model all --quick  # reduced sizes for a smoke test

Outputs are saved per model in  outputs/  and the code copy in  codes/.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))

import ml_reviewer_benchmark as core  # noqa: E402  (identical feature/ARL machinery)

MODELS = ["GradientBoosting", "RandomForest", "SVM", "kNN", "MLP", "ElasticNetLogistic"]

OUT = BASE / "outputs"
CODES = BASE / "codes"


def build_model(name: str, seed: int):
    """Exactly the same estimators/grids as ml_reviewer_benchmark.benchmark()."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC

    if name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=seed), {
            "n_estimators": [100, 200], "max_depth": [2, 5], "learning_rate": [.03, .1]}
    if name == "RandomForest":
        return Pipeline([("scale", RobustScaler()), ("model", RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1))]), {
            "model__n_estimators": [200], "model__max_depth": [None, 15],
            "model__min_samples_leaf": [1, 2]}
    if name == "SVM":
        return Pipeline([("scale", RobustScaler()), ("model", SVC(
            probability=True, class_weight="balanced", random_state=seed))]), {
            "model__C": [.1, 1, 10], "model__gamma": ["scale", .01]}
    if name == "kNN":
        return Pipeline([("scale", RobustScaler()), ("model", KNeighborsClassifier())]), {
            "model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]}
    if name == "MLP":
        return Pipeline([("scale", RobustScaler()), ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=.001, max_iter=500,
            early_stopping=True, random_state=seed))]), {
            "model__alpha": [.0001, .001], "model__learning_rate_init": [.001, .01]}
    if name == "ElasticNetLogistic":
        return Pipeline([("scale", RobustScaler()), ("model", LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            max_iter=10000, tol=1e-4, random_state=seed))]), {
            "model__C": [.1, 1, 10], "model__l1_ratio": [.1, .5, .9]}
    raise ValueError(f"unknown model {name}")


def run_model(name: str, seed: int, samples_per_class: int, skewness: float,
              shifted_fraction: float, calibration_profiles: int,
              arl_replications: int, arl_horizon: int) -> dict:
    """Train, tune, calibrate and evaluate ONE model under the shared protocol."""
    # ---- shared data: identical for every model -----------------------------
    x, y = core.make_dataset(seed, samples_per_class, skewness, shifted_fraction)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    estimator, grid = build_model(name, seed)
    started = time.perf_counter()
    search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(x_train, y_train)
    model = search.best_estimator_
    prediction = search.predict(x_test)
    probability = search.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    metrics = {
        "model": name,
        "cv_auc": search.best_score_,
        "test_auc": roc_auc_score(y_test, probability),
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall_detection_rate": recall_score(y_test, prediction),
        "false_alarm_rate": fp / (fp + tn),
        "mcc": matthews_corrcoef(y_test, prediction),
        "seconds": time.perf_counter() - started,
        "train_n": len(y_train), "test_n": len(y_test),
        "best_params": search.best_params_,
    }

    # ---- identical ARL0 calibration and identical Phase II scenarios --------
    threshold, alarm_rate, cal_n = core.calibrate_model_threshold(
        model, seed + 7000, skewness, calibration_profiles=calibration_profiles)
    calibration = {"model": name, "threshold": threshold,
                   "calibration_profiles": cal_n, "target_arl0": core.TARGET_ARL0,
                   "empirical_calibration_arl0": 1.0 / max(alarm_rate, 1e-12)}

    shifts = {"small": ("intercept", .20), "moderate": ("intercept", .60),
              "large": ("intercept", 1.00)}
    arl_rows = []
    for shift_index, (level, (kind, magnitude)) in enumerate(shifts.items()):
        arl, mcse = core.estimate_model_arl(
            model, threshold, seed + 9000 + shift_index, skewness, kind, magnitude,
            arl_replications, arl_horizon)
        arl_rows.append({"model": name, "shift_level": level, "shift_type": kind,
                         "shift_magnitude": magnitude, "arl_zero_state": arl,
                         "mcse": mcse, "replications": arl_replications,
                         "horizon": arl_horizon, "target_arl0": core.TARGET_ARL0})

    result = {
        "model": name, "seed": seed, "skewness": skewness,
        "samples_per_class": samples_per_class, "shifted_fraction": shifted_fraction,
        "features": core.FEATURE_NAMES,
        "split": "stratified 80/20, random_state=20260825",
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=20260825)",
        "metrics": metrics, "calibration": calibration, "arl_results": arl_rows,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=MODELS + ["all"])
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--skewness", type=float, default=1.0)
    parser.add_argument("--shifted-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-profiles", type=int, default=core.CALIBRATION_PROFILES)
    parser.add_argument("--arl-replications", type=int, default=core.ARL_REPLICATIONS)
    parser.add_argument("--arl-horizon", type=int, default=core.ARL_HORIZON)
    parser.add_argument("--quick", action="store_true",
                        help="reduced sizes for a fast end-to-end check")
    parser.add_argument("--seed", type=int, default=core.SEED)
    args = parser.parse_args()

    if args.quick:
        args.samples_per_class = 400
        args.calibration_profiles = 2000
        args.arl_replications = 200
        args.arl_horizon = 100

    OUT.mkdir(exist_ok=True)
    CODES.mkdir(exist_ok=True)
    # save this driver code once per model folder (the "code saved separately")
    shutil.copy2(Path(__file__), CODES / "run_one_model.py")

    names = MODELS if args.model == "all" else [args.model]
    for name in names:
        result = run_model(name, args.seed, args.samples_per_class, args.skewness,
                           args.shifted_fraction, args.calibration_profiles,
                           args.arl_replications, args.arl_horizon)
        tag = name.lower()
        (OUT / f"{tag}_result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
        pd.DataFrame([result["metrics"]]).to_csv(OUT / f"{tag}_metrics.csv", index=False)
        pd.DataFrame(result["arl_results"]).to_csv(OUT / f"{tag}_arl.csv", index=False)
        print(f"[DONE] {name}: test_auc={result['metrics']['test_auc']:.3f} "
              f"mcc={result['metrics']['mcc']:.3f} "
              f"ARL0={result['calibration']['empirical_calibration_arl0']:.0f}")

    metadata = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__, "seed": args.seed,
                "models": names, "quick_mode": args.quick,
                "samples_per_class": args.samples_per_class,
                "skewness": args.skewness, "shifted_fraction": args.shifted_fraction,
                "calibration_profiles": args.calibration_profiles,
                "arl_replications": args.arl_replications, "arl_horizon": args.arl_horizon,
                "protocol": "identical feature set, split, CV, calibration and scenarios for all models",
                "source_of_estimators": "ml_reviewer_benchmark.py"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

