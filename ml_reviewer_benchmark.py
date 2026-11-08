"""Independent ML benchmark for Reviewer 1.

The original ML scripts are not imported or modified. This reproduces their
16-feature profile representation, uses one fixed stratified split, identical
3-fold CV, balanced learning where supported, and reports test metrics.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import skewnorm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

SEED = 20260825
N_OBS = 10
X_VALUES = np.arange(1, 11, dtype=float)
TRUE_A0, TRUE_A1, TRUE_SIGMA = 3.0, 2.0, 1.0
OUT = Path(__file__).resolve().parent / "reviewer_outputs"
TARGET_ARL0 = 200.0
CALIBRATION_PROFILES = 10000
ARL_REPLICATIONS = 2000
ARL_HORIZON = 500
FEATURE_NAMES = [
    "y_mean", "y_sd", "y_skewness", "y_kurtosis", "intercept", "slope",
    "q25", "q75", "outlier_ratio", "mad", "runs_rate", "residual_sd",
    "residual_skew", "residual_kurtosis", "mean_abs_residual", "y_range",
]


def compute_features(y):
    y = np.asarray(y, dtype=float)
    n = len(y)
    y_mean = y.mean()
    y_sd = np.std(y, ddof=1)
    x_mean = X_VALUES.mean()
    slope = np.sum((X_VALUES - x_mean) * (y - y_mean)) / np.sum((X_VALUES - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * X_VALUES
    residual = y - fitted
    residual_sd = np.std(residual, ddof=2)
    safe_y_sd = y_sd + 1e-12
    safe_res_sd = residual_sd + 1e-12
    mad = np.median(np.abs(y - np.median(y)))
    modified_z = 0.6745 * (y - np.median(y)) / (mad + 1e-12)
    changes = np.diff(y)
    runs_rate = np.mean(changes[:-1] * changes[1:] < 0) if len(changes) > 1 else 0.0
    return np.array([
        y_mean, y_sd, np.mean(((y - y_mean) / safe_y_sd) ** 3),
        np.mean(((y - y_mean) / safe_y_sd) ** 4) - 3, intercept, slope,
        np.percentile(y, 25), np.percentile(y, 75),
        np.mean(np.abs(modified_z) > 3.5), mad, runs_rate, residual_sd,
        np.mean(((residual - residual.mean()) / safe_res_sd) ** 3),
        np.mean(((residual - residual.mean()) / safe_res_sd) ** 4) - 3,
        np.mean(np.abs(residual)), y.max() - y.min(),
    ])


def compute_features_batch(profiles):
    """Vectorized equivalent of compute_features for many profiles."""
    values = np.asarray(profiles, dtype=float)
    y_mean = values.mean(axis=1)
    y_sd = values.std(axis=1, ddof=1)
    centered = values - y_mean[:, None]
    safe_y_sd = y_sd + 1e-12
    slope = np.sum((X_VALUES - X_VALUES.mean()) * centered, axis=1) / np.sum((X_VALUES - X_VALUES.mean()) ** 2)
    intercept = y_mean - slope * X_VALUES.mean()
    residual = values - intercept[:, None] - slope[:, None] * X_VALUES
    residual_sd = residual.std(axis=1, ddof=2)
    safe_res_sd = residual_sd + 1e-12
    median = np.median(values, axis=1)
    mad = np.median(np.abs(values - median[:, None]), axis=1)
    modified_z = 0.6745 * (values - median[:, None]) / (mad[:, None] + 1e-12)
    changes = np.diff(values, axis=1)
    runs_rate = np.mean(changes[:, :-1] * changes[:, 1:] < 0, axis=1)
    return np.column_stack([
        y_mean, y_sd, np.mean((centered / safe_y_sd[:, None]) ** 3, axis=1),
        np.mean((centered / safe_y_sd[:, None]) ** 4, axis=1) - 3,
        intercept, slope, np.percentile(values, 25, axis=1), np.percentile(values, 75, axis=1),
        np.mean(np.abs(modified_z) > 3.5, axis=1), mad, runs_rate, residual_sd,
        np.mean(((residual - residual.mean(axis=1)[:, None]) / safe_res_sd[:, None]) ** 3, axis=1),
        np.mean(((residual - residual.mean(axis=1)[:, None]) / safe_res_sd[:, None]) ** 4, axis=1) - 3,
        np.mean(np.abs(residual), axis=1), values.max(axis=1) - values.min(axis=1),
    ])


def make_profile(rng, skewness, kind="ic", magnitude=0.0):
    if kind == "intercept":
        a0, a1, sigma = TRUE_A0 + magnitude, TRUE_A1, TRUE_SIGMA
    elif kind == "slope":
        a0, a1, sigma = TRUE_A0, TRUE_A1 + magnitude, TRUE_SIGMA
    elif kind == "variance":
        a0, a1, sigma = TRUE_A0, TRUE_A1, TRUE_SIGMA * (1 + magnitude)
    else:
        a0, a1, sigma = TRUE_A0, TRUE_A1, TRUE_SIGMA
    errors = skewnorm.rvs(a=skewness, loc=0, scale=sigma, size=N_OBS, random_state=rng)
    return a0 + a1 * X_VALUES + errors


def make_dataset(seed, samples_per_class, skewness, shifted_fraction=0.5):
    rng = np.random.default_rng(seed)
    total = int(round(samples_per_class / shifted_fraction))
    n_shifted = samples_per_class
    n_in_control = max(1, total - n_shifted)
    in_control = [compute_features(make_profile(rng, skewness)) for _ in range(n_in_control)]
    shifted = []
    for _ in range(samples_per_class):
        kind = rng.choice(["intercept", "slope", "variance"])
        if kind == "intercept":
            magnitude = rng.choice([-1, 1]) * rng.uniform(.05, 2.0)
        elif kind == "slope":
            magnitude = rng.choice([-1, 1]) * rng.uniform(.01, .4)
        else:
            magnitude = rng.uniform(.1, 1.5)
        shifted.append(compute_features(make_profile(rng, skewness, kind, magnitude)))
    return np.vstack([in_control, shifted]), np.r_[np.zeros(n_in_control), np.ones(n_shifted)]


def calibrate_model_threshold(model, seed, skewness, target_arl=TARGET_ARL0,
                              calibration_profiles=CALIBRATION_PROFILES):
    rng = np.random.default_rng(seed)
    calibration_profiles_data = np.vstack([make_profile(rng, skewness)
                                           for _ in range(calibration_profiles)])
    calibration = compute_features_batch(calibration_profiles_data)
    scores = model.predict_proba(calibration)[:, 1]
    threshold = float(np.quantile(scores, 1.0 - 1.0 / target_arl, method="linear"))
    return threshold, float(np.mean(scores >= threshold)), calibration_profiles


def estimate_model_arl(model, threshold, seed, skewness, kind, magnitude,
                       replications=ARL_REPLICATIONS, horizon=ARL_HORIZON):
    rng = np.random.default_rng(seed)
    profiles = np.vstack([make_profile(rng, skewness, kind, magnitude)
                          for _ in range(replications * horizon)])
    scores = model.predict_proba(compute_features_batch(profiles))[:, 1]
    score_matrix = scores.reshape(replications, horizon)
    alarms = score_matrix >= threshold
    first = np.argmax(alarms, axis=1)
    has_alarm = alarms.any(axis=1)
    run_lengths = np.where(has_alarm, first + 1, horizon).astype(float)
    values = np.asarray(run_lengths, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(replications))


def benchmark(seed=SEED, samples_per_class=2000, skewness=1.0, shifted_fraction=0.5,
              calibration_profiles=CALIBRATION_PROFILES, arl_replications=ARL_REPLICATIONS,
              arl_horizon=ARL_HORIZON):
    x, y = make_dataset(seed, samples_per_class, skewness, shifted_fraction)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.2, stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = {
        "GradientBoosting": (GradientBoostingClassifier(random_state=seed), {
            "n_estimators": [100, 200], "max_depth": [2, 5],
            "learning_rate": [.03, .1]}),
        "RandomForest": (Pipeline([("scale", RobustScaler()), ("model", RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1))]), {
            "model__n_estimators": [200], "model__max_depth": [None, 15],
            "model__min_samples_leaf": [1, 2]}),
        "SVM": (Pipeline([("scale", RobustScaler()), ("model", SVC(
            probability=True, class_weight="balanced", random_state=seed))]), {
            "model__C": [.1, 1, 10], "model__gamma": ["scale", .01]}),
        "kNN": (Pipeline([("scale", RobustScaler()), ("model", KNeighborsClassifier())]), {
            "model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]}),
        "MLP": (Pipeline([("scale", RobustScaler()), ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=.001, max_iter=500,
            early_stopping=True, random_state=seed))]), {
            "model__alpha": [.0001, .001], "model__learning_rate_init": [.001, .01]}),
        "ElasticNetLogistic": (Pipeline([("scale", RobustScaler()), ("model", LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            max_iter=2000, random_state=seed))]), {
            "model__C": [.1, 1, 10], "model__l1_ratio": [.1, .5, .9]}),
    }
    results = []
    fitted_models = {}
    for name, (estimator, grid) in models.items():
        started = time.perf_counter()
        search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1,
                              return_train_score=False)
        search.fit(x_train, y_train)
        fitted_models[name] = search.best_estimator_
        prediction = search.predict(x_test)
        probability = search.predict_proba(x_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
        results.append({
            "model": name, "cv_auc": search.best_score_,
            "test_auc": roc_auc_score(y_test, probability),
            "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall_detection_rate": recall_score(y_test, prediction),
            "false_alarm_rate": fp / (fp + tn),
            "mcc": matthews_corrcoef(y_test, prediction),
            "seconds": time.perf_counter() - started,
            "train_n": len(y_train), "test_n": len(y_test),
            "best_params": search.best_params_,
        })

    # Every model uses the same calibration size, target ARL0, and scenarios.
    calibration_results = []
    arl_results = []
    shifts = {"small": ("intercept", .20), "moderate": ("intercept", .60),
              "large": ("intercept", 1.00)}
    for model_index, (name, model) in enumerate(fitted_models.items()):
        threshold, calibration_alarm_rate, calibration_n = calibrate_model_threshold(
            model, seed + 7000 + model_index, skewness,
            calibration_profiles=calibration_profiles)
        calibration_results.append({"model": name, "threshold": threshold,
                                    "calibration_profiles": calibration_n,
                                    "target_arl0": TARGET_ARL0,
                                    "empirical_calibration_arl0": 1.0 / max(calibration_alarm_rate, 1e-12)})
        for shift_index, (shift_level, (kind, magnitude)) in enumerate(shifts.items()):
            arl, mcse = estimate_model_arl(
                model, threshold, seed + 9000 + model_index * 100 + shift_index,
                skewness, kind, magnitude, arl_replications, arl_horizon)
            arl_results.append({"model": name, "shift_level": shift_level,
                                "shift_type": kind, "shift_magnitude": magnitude,
                                "arl_zero_state": arl, "mcse": mcse,
                                "replications": arl_replications, "horizon": arl_horizon,
                                "target_arl0": TARGET_ARL0})

    # Feature selection is fitted on training data only, preventing leakage.
    selector_rng = np.random.RandomState(seed)
    mi = SelectKBest(mutual_info_classif, k=8).fit(x_train, y_train)
    mi_idx = np.flatnonzero(mi.get_support())
    correlation = np.corrcoef(x_train, y_train, rowvar=False)[-1, :-1]
    correlation_idx = np.argsort(np.abs(correlation))[::-1][:8]
    embedded_model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    embedded_model.fit(x_train, y_train)
    embedded_idx = np.argsort(embedded_model.feature_importances_)[::-1][:8]
    rfe_estimator = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000, random_state=seed)
    rfe = RFE(rfe_estimator, n_features_to_select=8).fit(RobustScaler().fit_transform(x_train), y_train)
    rfe_idx = np.flatnonzero(rfe.support_)
    feature_sets = {
        "full_16": np.arange(x_train.shape[1]),
        "MI_filter_8": mi_idx,
        "Correlation_filter_8": correlation_idx,
        "Embedded_RF_importance_8": embedded_idx,
        "RFE_8": rfe_idx,
    }
    selection_results = []
    for selection_name, indices in feature_sets.items():
        if selection_name == "full_16":
            model = GradientBoostingClassifier(random_state=seed)
            grid = {"n_estimators": [100, 200], "max_depth": [2, 5], "learning_rate": [.03, .1]}
        else:
            model = Pipeline([("scale", RobustScaler()), ("model", GradientBoostingClassifier(random_state=seed))])
            grid = {"model__n_estimators": [100, 200], "model__max_depth": [2, 5], "model__learning_rate": [.03, .1]}
        search = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(x_train[:, indices], y_train)
        prediction = search.predict(x_test[:, indices])
        probability = search.predict_proba(x_test[:, indices])[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
        selection_results.append({
            "feature_set": selection_name, "feature_count": len(indices),
            "features": [FEATURE_NAMES[i] for i in indices],
            "cv_auc": search.best_score_, "test_auc": roc_auc_score(y_test, probability),
            "detection_rate": recall_score(y_test, prediction),
            "false_alarm_rate": fp / (fp + tn), "mcc": matthews_corrcoef(y_test, prediction),
            "best_params": search.best_params_,
        })
    OUT.mkdir(exist_ok=True)
    output = {"seed": seed, "skewness": skewness, "samples_per_class": samples_per_class,
              "shifted_fraction": shifted_fraction, "n_total": len(y),
              "features": FEATURE_NAMES, "split": "stratified 80/20", "cv": "5-fold stratified",
              "class_balance": {"in_control": float(np.mean(y == 0)), "shifted": float(np.mean(y == 1))}, "results": results,
              "protocol": {
                  "same_feature_set": True,
                  "same_train_test_split": True,
                  "same_cv_folds": True,
                  "calibration_target_arl0": TARGET_ARL0,
                  "calibration_profiles": calibration_profiles,
                  "arl_replications": arl_replications,
                  "arl_horizon": arl_horizon,
                  "threshold_rule": "empirical 1 - 1/ARL0 quantile of in-control profile scores",
                  "training_source": "simulation profiles representing Phase-I in-control and shifted profiles only",
              },
              "calibration": calibration_results, "arl_results": arl_results,
              "feature_selection": selection_results,
              "selection_method_definitions": {
                  "MI_filter_8": "top eight mutual-information scores fitted on training data",
                  "Correlation_filter_8": "top eight absolute point-biserial correlations fitted on training data",
                  "Embedded_RF_importance_8": "top eight Random Forest impurity importances fitted on training data",
                  "RFE_8": "recursive elimination using standardized logistic regression",
                  "PCA": "not used as a feature subset; PCA is evaluated below as a dimension-reduction benchmark",
              }}
    pca = Pipeline([("scale", RobustScaler()), ("pca", PCA(n_components=8, random_state=seed)),
                    ("model", GradientBoostingClassifier(random_state=seed))])
    pca_search = GridSearchCV(pca, {"model__n_estimators": [100, 200], "model__max_depth": [2, 5]},
                              scoring="roc_auc", cv=cv, n_jobs=-1)
    pca_search.fit(x_train, y_train)
    pca_prob = pca_search.predict_proba(x_test)[:, 1]
    pca_pred = pca_prob >= .5
    output["feature_selection"].append({"feature_set": "PCA_8_components", "feature_count": 8,
        "features": [f"PC{i}" for i in range(1, 9)], "cv_auc": pca_search.best_score_,
        "test_auc": roc_auc_score(y_test, pca_prob), "detection_rate": recall_score(y_test, pca_pred),
        "false_alarm_rate": float(np.mean(pca_pred[y_test == 0])), "mcc": matthews_corrcoef(y_test, pca_pred),
        "explained_variance_ratio": pca_search.best_estimator_.named_steps["pca"].explained_variance_ratio_.tolist(),
        "best_params": pca_search.best_params_})
    serialized = json.dumps(output, indent=2, default=str)
    (OUT / f"ml_reviewer_benchmark_lambda_{skewness:g}.json").write_text(serialized, encoding="utf-8")
    (OUT / "ml_reviewer_benchmark.json").write_text(serialized, encoding="utf-8")
    pd.DataFrame(calibration_results).to_csv(OUT / f"ml_calibration_lambda_{skewness:g}.csv", index=False)
    pd.DataFrame(arl_results).to_csv(OUT / f"ml_arl_lambda_{skewness:g}.csv", index=False)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--skewness", type=float, default=1.0)
    parser.add_argument("--shifted-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-profiles", type=int, default=CALIBRATION_PROFILES)
    parser.add_argument("--arl-replications", type=int, default=ARL_REPLICATIONS)
    parser.add_argument("--arl-horizon", type=int, default=ARL_HORIZON)
    parser.add_argument("--all-skewness", action="store_true")
    args = parser.parse_args()
    skewness_values = [0.0, 0.5, 1.0, 2.0] if args.all_skewness else [args.skewness]
    all_results = []
    for skewness in skewness_values:
        result = benchmark(samples_per_class=args.samples_per_class, skewness=skewness,
                           shifted_fraction=args.shifted_fraction,
                           calibration_profiles=args.calibration_profiles,
                           arl_replications=args.arl_replications,
                           arl_horizon=args.arl_horizon)
        for row in result["results"]:
            all_results.append({"skewness": skewness, **row})
        for row in result["feature_selection"]:
            all_results.append({"skewness": skewness, "model": row["feature_set"], **row})
    pd.DataFrame(all_results).to_csv(OUT / "ml_reviewer_benchmark_all_results.csv", index=False)
    metadata = {"python": platform.python_version(), "sklearn": sklearn.__version__, "seed": SEED,
                "samples_per_class": args.samples_per_class, "skewness_values": skewness_values,
                "shifted_fraction": args.shifted_fraction, "feature_count": len(FEATURE_NAMES),
                "train_test_split": "stratified 80/20, random_state=20260825",
                "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=20260825)",
                "selection_and_tuning": "all feature selection fit on training data only; GridSearch scoring=ROC-AUC",
                "class_imbalance": "class_weight=balanced where supported; --shifted-fraction controls prevalence and is recorded per run",
                "arl_calibration": {"target_arl0": TARGET_ARL0, "calibration_profiles": args.calibration_profiles,
                                    "replications": args.arl_replications, "horizon": args.arl_horizon,
                                    "same_protocol_for_all_models": True},
                "source_files_modified": []}
    (OUT / "ml_reviewer_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
