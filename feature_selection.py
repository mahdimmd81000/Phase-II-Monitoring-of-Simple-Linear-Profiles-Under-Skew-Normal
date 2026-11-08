"""Reviewer 1, comment 4: feature selection / reduction with justification.

Four families of methods are compared, exactly as the reviewer requested.
EVERY selection step is fitted on the TRAINING data only (no leakage):

  filter    : mutual-information ranking            -> top-k subset
              point-biserial correlation ranking    -> top-k subset
  embedded  : GradientBoosting gain-based importances (the proposed model's
              own importance scores)                -> top-k subset
  wrapper   : recursive feature elimination (RFE) with the GB estimator
  reduction : PCA (k components) as a dimension-reduction benchmark

The proposed GradientBoosting model is then tuned and evaluated
  (a) with the FULL 16 features and
  (b) with every selected subset,
under the identical protocol (same split, same 5-fold stratified CV, same
grid, same ARL0 calibration = 200, same Phase II scenarios).  The final
retained features and the full rankings are reported.

Outputs (outputs/):
  feature_rankings.csv          full ranking of all 16 features per method
  gb_feature_importance.csv     embedded gain importances of the proposed model
  feature_selection_results.csv full-16 vs every subset (CV/test metrics + ARL)
  retained_features.json        the final retained features per method

Run:
  py -3.11 "Ã˜Â¯Ã™Ë†Ã™â€¦Ã›Å’Ã™â€  Ã˜Â¯Ã˜Â³Ã˜ÂªÃ™Ë†Ã˜Â± Ã˜Â¯Ã˜Â§Ã™Ë†Ã˜Â±/feature_selection.py"            # full
  py -3.11 "Ã˜Â¯Ã™Ë†Ã™â€¦Ã›Å’Ã™â€  Ã˜Â¯Ã˜Â³Ã˜ÂªÃ™Ë†Ã˜Â± Ã˜Â¯Ã˜Â§Ã™Ë†Ã˜Â±/feature_selection.py" --quick    # smoke test
Tests:
  py -3.11 -m pytest "Ã˜Â¯Ã™Ë†Ã™â€¦Ã›Å’Ã™â€  Ã˜Â¯Ã˜Â³Ã˜ÂªÃ™Ë†Ã˜Â± Ã˜Â¯Ã˜Â§Ã™Ë†Ã˜Â±/test_feature_selection.py" -v
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.metrics import matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reviewer1_ml_benchmark"))
sys.path.insert(0, str(BASE))

import ml_reviewer_benchmark as core                 # noqa: E402
import run_reviewer2_benchmark as r2                 # noqa: E402

OUT = BASE / "outputs"
GB_GRID = {"n_estimators": [100, 200], "max_depth": [2, 5], "learning_rate": [.03, .1]}


def gb_pipeline(indices, seed):
    idx = np.asarray(list(indices))
    if len(idx) == len(core.FEATURE_NAMES):
        return GradientBoostingClassifier(random_state=seed), GB_GRID
    return (Pipeline([("select", _ColumnSelector(idx)),
                      ("model", GradientBoostingClassifier(random_state=seed))]),
            {"model__" + k: v for k, v in GB_GRID.items()})


class _ColumnSelector(BaseEstimator, TransformerMixin):
    """Keeps feature selection OUTSIDE the CV loop (fitted on train only)."""

    def __init__(self, indices):
        self.indices = indices

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[:, np.asarray(self.indices)]


def compute_rankings(x_train, y_train, seed, k):
    """All selection methods fitted on TRAINING data only."""
    rankings = {}
    mi = SelectKBest(mutual_info_classif, k="all").fit(x_train, y_train)
    mi_scores = mi.scores_
    rankings["MI_filter"] = (np.argsort(mi_scores)[::-1], mi_scores)

    corr = np.array([abs(np.corrcoef(x_train[:, j], y_train)[0, 1])
                     for j in range(x_train.shape[1])])
    corr = np.nan_to_num(corr)
    rankings["Correlation_filter"] = (np.argsort(corr)[::-1], corr)

    gb_imp = GradientBoostingClassifier(random_state=seed).fit(x_train, y_train)
    imp = gb_imp.feature_importances_
    rankings["Embedded_GB_gain"] = (np.argsort(imp)[::-1], imp)

    rfe = RFE(GradientBoostingClassifier(random_state=seed),
              n_features_to_select=k, step=1).fit(x_train, y_train)
    rfe_rank = rfe.ranking_
    rankings["RFE_wrapper"] = (np.argsort(rfe_rank), 1.0 / rfe_rank)

    return rankings, imp


def evaluate_subset(name, indices, shared, seed, calibration_profiles,
                    arl_replications, arl_horizon):
    """Tune + evaluate the proposed GB model with a given feature set."""
    x_train, x_test, y_train, y_test, cv = (shared[2], shared[3],
                                            shared[4], shared[5], shared[6])
    estimator, grid = gb_pipeline(indices, seed)
    search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(x_train, y_train)
    model = search.best_estimator_
    probability = search.predict_proba(x_test)[:, 1]
    prediction = (probability >= .5).astype(int)

    threshold, alarm_rate, _ = core.calibrate_model_threshold(
        model, seed + 7000, shared[7], calibration_profiles=calibration_profiles)
    arl_rows = []
    for i, (level, (kind, magnitude)) in enumerate(
            [("small", ("intercept", .20)), ("moderate", ("intercept", .60)),
             ("large", ("intercept", 1.00))]):
        arl, mcse = core.estimate_model_arl(
            model, threshold, seed + 9000 + i, shared[7], kind, magnitude,
            arl_replications, arl_horizon)
        arl_rows.append({"feature_set": name, "shift_level": level,
                         "arl": arl, "mcse": mcse})

    return {
        "feature_set": name,
        "n_features": len(indices),
        "features": [core.FEATURE_NAMES[j] for j in indices],
        "selected_params": search.best_params_,
        "cv_auc": float(search.best_score_),
        "test_auc": float(roc_auc_score(y_test, probability)),
        "detection_rate": float(recall_score(y_test, prediction)),
        "mcc": float(matthews_corrcoef(y_test, prediction)),
    }, arl_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=core.SEED)
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--skewness", type=float, default=1.0)
    parser.add_argument("--shifted-fraction", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=8, help="subset size for each method")
    parser.add_argument("--calibration-profiles", type=int, default=core.CALIBRATION_PROFILES)
    parser.add_argument("--arl-replications", type=int, default=1000)
    parser.add_argument("--arl-horizon", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()
    global OUT
    if args.out is not None:
        OUT = Path(args.out)
    if args.quick:
        args.samples_per_class, args.calibration_profiles = 400, 2000
        args.arl_replications, args.arl_horizon = 200, 100

    OUT.mkdir(exist_ok=True)
    shared = r2.make_shared_data(args.seed, args.samples_per_class, args.skewness,
                                 args.shifted_fraction)
    shared = shared + (args.skewness,)                      # append skewness for ARL
    x_train, y_train = shared[2], shared[4]
    k = args.k

    print("computing rankings (train data only)...", flush=True)
    t0 = time.perf_counter()
    rankings, gb_importance = compute_rankings(x_train, y_train, args.seed, k)
    print(f"  done in {time.perf_counter() - t0:.0f}s", flush=True)

    rank_rows = []
    for method, (order, scores) in rankings.items():
        for rank, j in enumerate(order, 1):
            rank_rows.append({"method": method, "rank": rank,
                              "feature": core.FEATURE_NAMES[j],
                              "score": float(scores[j])})
    pd.DataFrame(rank_rows).to_csv(OUT / "feature_rankings.csv", index=False)
    pd.DataFrame({"feature": core.FEATURE_NAMES, "gb_gain_importance": gb_importance}
                 ).sort_values("gb_gain_importance", ascending=False
                               ).to_csv(OUT / "gb_feature_importance.csv", index=False)

    feature_sets = {"full_16": np.arange(len(core.FEATURE_NAMES))}
    retained = {}
    for method, (order, _) in rankings.items():
        if method == "RFE_wrapper":
            idx = order[:k]                                  # RFE already gives exactly k
        else:
            idx = order[:k]
        feature_sets[method] = idx
        retained[method] = [core.FEATURE_NAMES[j] for j in idx]

    results, all_arl = [], []
    for name, indices in feature_sets.items():
        print(f"evaluating {name} ({len(indices)} features)...", flush=True)
        t0 = time.perf_counter()
        row, arl_rows = evaluate_subset(name, indices, shared, args.seed,
                                        args.calibration_profiles,
                                        args.arl_replications, args.arl_horizon)
        results.append(row)
        all_arl.extend(arl_rows)
        print(f"  cv_auc={row['cv_auc']:.3f} test_auc={row['test_auc']:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # PCA benchmark (dimension reduction, not a subset)
    print("evaluating PCA_8 ...", flush=True)
    pca_model = Pipeline([("scale", __import__("sklearn.preprocessing",
                                               fromlist=["RobustScaler"]).RobustScaler()),
                          ("pca", PCA(n_components=k, random_state=args.seed)),
                          ("model", GradientBoostingClassifier(random_state=args.seed))])
    x_train, x_test, y_train, y_test = shared[2], shared[3], shared[4], shared[5]
    search = GridSearchCV(pca_model, {"model__" + kk: v for kk, v in GB_GRID.items()},
                          scoring="roc_auc", cv=shared[6], n_jobs=-1)
    search.fit(x_train, y_train)
    model = search.best_estimator_
    probability = search.predict_proba(x_test)[:, 1]
    prediction = (probability >= .5).astype(int)
    threshold, alarm_rate, _ = core.calibrate_model_threshold(
        model, args.seed + 7000, args.skewness,
        calibration_profiles=args.calibration_profiles)
    for i, (_, (kind, magnitude)) in enumerate(
            [("small", ("intercept", .20)), ("moderate", ("intercept", .60)),
             ("large", ("intercept", 1.00))]):
        arl, mcse = core.estimate_model_arl(
            model, threshold, args.seed + 9000 + i, args.skewness, kind, magnitude,
            args.arl_replications, args.arl_horizon)
        all_arl.append({"feature_set": "PCA_8", "shift_level": _,
                        "arl": arl, "mcse": mcse})
    results.append({
        "feature_set": "PCA_8", "n_features": k,
        "features": [f"PC{i}" for i in range(1, k + 1)],
        "selected_params": search.best_params_,
        "cv_auc": float(search.best_score_),
        "test_auc": float(roc_auc_score(y_test, probability)),
        "detection_rate": float(recall_score(y_test, prediction)),
        "mcc": float(matthews_corrcoef(y_test, prediction)),
    })

    pd.DataFrame(results).to_csv(OUT / "feature_selection_results.csv", index=False)
    pd.DataFrame(all_arl).to_csv(OUT / "feature_selection_arl.csv", index=False)
    (OUT / "retained_features.json").write_text(json.dumps({
        "k": k, "retained_features": retained,
        "note": "every selection method was fitted on TRAINING data only; "
                "GB gain importances of the proposed model are the embedded method",
        "provenance": {"seed": args.seed,
                       "split": "stratified 80/20, random_state=20260825",
                       "cv": "StratifiedKFold(5), identical for all feature sets"},
    }, indent=2), encoding="utf-8")
    print("Done ->", OUT)


if __name__ == "__main__":
    main()


