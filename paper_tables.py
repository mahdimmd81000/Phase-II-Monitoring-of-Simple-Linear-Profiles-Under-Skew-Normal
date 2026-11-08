"""Reviewer 1, point 3: extend the paper's ARL tables (Tables 2-13) to the
benchmark ML models.

For each skewness level the script:
  1. builds the SAME simulated dataset / stratified 80/20 split / 5-fold CV
     for every model (identical conditions, seed 20260825),
  2. tunes each model with GridSearchCV (scoring = ROC-AUC),
  3. calibrates every model to the same target ARL0 = 200
     (10,000 in-control profiles, threshold = 1 - 1/ARL0 quantile),
  4. computes zero-state ARL over the EXACT shift grids of Tables 2-13:
       intercept : 0.0 .. 2.0 step 0.2
       slope     : 0.0 .. 0.25 step 0.025
       std       : 1.0 .. 3.0 step 0.2
  5. writes one table per shift type in the SAME layout as the manuscript,
     with the paper's T2 row included for direct comparison.

Output: reviewer1_ml_benchmark/outputs/paper_tables/
  table_<type>_lambda<skew>.csv   (+ .md)  and  all_paper_tables_long.csv

Run:
  py -3.11 reviewer1_ml_benchmark/paper_tables.py --all-skewness
  py -3.11 reviewer1_ml_benchmark/paper_tables.py --skewness 1.0
  py -3.11 reviewer1_ml_benchmark/paper_tables.py --all-skewness --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))

import ml_reviewer_benchmark as core  # noqa: E402
from run_one_model import MODELS, build_model  # noqa: E402

# T2 rows exactly as reported in Tables 2-13 of the manuscript (groups in the
# order lambda = 0.0, 0.5, 1.0, 2.0 -- VERIFY against the paper's captions).
PAPER_T2 = {
    0.0: {
        "intercept": [200.00, 100.00, 26.74, 8.13, 3.47, 1.94, 1.33, 1.10, 1.03, 1.01, 1.00],
        "slope":     [200.00, 108.70, 44.25, 16.84, 7.17, 3.91, 2.28, 1.58, 1.26, 1.11, 1.03],
        "sd":        [200.00, 35.46, 13.85, 7.43, 4.96, 3.61, 2.87, 2.44, 2.13, 1.93, 1.77],
    },
    0.5: {
        "intercept": [200.00, 40.65, 10.82, 4.02, 2.06, 1.34, 1.10, 1.02, 1.00, 1.00, 1.00],
        "slope":     [200.00, 56.82, 20.16, 8.64, 4.28, 2.36, 1.59, 1.25, 1.10, 1.03, 1.01],
        "sd":        [200.00, 34.72, 12.47, 6.93, 4.57, 3.32, 2.70, 2.27, 1.97, 1.79, 1.66],
    },
    1.0: {
        "intercept": [200.00, 29.94, 7.16, 2.72, 1.53, 1.14, 1.02, 1.00, 1.00, 1.00, 1.00],
        "slope":     [200.00, 48.54, 14.41, 5.95, 2.84, 1.79, 1.29, 1.10, 1.03, 1.01, 1.00],
        "sd":        [200.00, 23.92, 7.97, 4.29, 2.83, 2.18, 1.84, 1.62, 1.46, 1.37, 1.29],
    },
    2.0: {
        "intercept": [200.00, 27.03, 6.00, 2.17, 1.26, 1.04, 1.00, 1.00, 1.00, 1.00, 1.00],
        "slope":     [200.00, 45.87, 14.01, 4.96, 2.33, 1.45, 1.13, 1.03, 1.00, 1.00, 1.00],
        "sd":        [200.00, 21.46, 6.23, 3.06, 2.10, 1.63, 1.40, 1.25, 1.18, 1.12, 1.09],
    },
}
GRIDS = {
    "intercept": np.round(np.arange(0.0, 2.01, 0.2), 2),
    "slope": np.round(np.arange(0.0, 0.2501, 0.025), 3),
    "sd": np.round(np.arange(1.0, 3.01, 0.2), 2),
}


def train_model(name, x_train, y_train, seed):
    estimator, grid = build_model(name, seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_score_


def arl_grid(model, threshold, seed, skewness, shift_type, reps, horizon):
    rows = []
    for i, value in enumerate(GRIDS[shift_type]):
        if shift_type == "sd":
            kind, magnitude = "variance", float(value) - 1.0
        else:
            kind, magnitude = shift_type, float(value)
        if magnitude == 0.0:
            # shift 0 = in-control: reuse the calibration ARL0 logic
            arl, mcse = None, None
        else:
            arl, mcse = core.estimate_model_arl(
                model, threshold, seed + 9000 + i, skewness, kind, magnitude,
                replications=reps, horizon=horizon)
        rows.append((value, arl, mcse))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skewness", type=float, default=1.0)
    parser.add_argument("--all-skewness", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--shifted-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-profiles", type=int, default=core.CALIBRATION_PROFILES)
    parser.add_argument("--arl-replications", type=int, default=1000)
    parser.add_argument("--arl-horizon", type=int, default=600)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=core.SEED)
    args = parser.parse_args()

    if args.quick:
        args.samples_per_class, args.calibration_profiles = 400, 2000
        args.arl_replications, args.arl_horizon = 200, 150

    skewness_values = [0.0, 0.5, 1.0, 2.0] if args.all_skewness else [args.skewness]
    out = BASE / "outputs" / "paper_tables"
    out.mkdir(parents=True, exist_ok=True)

    long_rows = []
    combined = {}
    for skewness in skewness_values:
        t0 = time.perf_counter()
        print(f"\n===== skewness = {skewness} =====", flush=True)
        x, y = core.make_dataset(args.seed, args.samples_per_class, skewness,
                                 args.shifted_fraction)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=.2, stratify=y, random_state=args.seed)

        fitted = {}
        for name in MODELS:
            started = time.perf_counter()
            model, cv_auc = train_model(name, x_train, y_train, args.seed)
            fitted[name] = model
            print(f"  trained {name} (cv_auc={cv_auc:.3f}, "
                  f"{time.perf_counter() - started:.0f}s)", flush=True)

        for model_index, (name, model) in enumerate(fitted.items()):
            threshold, alarm_rate, cal_n = core.calibrate_model_threshold(
                model, args.seed + 7000 + model_index, skewness,
                calibration_profiles=args.calibration_profiles)
            arl0 = 1.0 / max(alarm_rate, 1e-12)
            print(f"  calibrated {name}: ARL0={arl0:.1f}", flush=True)

            for shift_type in GRIDS:
                table_row = {"Methods": name}
                for value, arl, mcse in arl_grid(model, threshold, args.seed,
                                                 skewness, shift_type,
                                                 args.arl_replications,
                                                 args.arl_horizon):
                    label = f"{value:g}"
                    if arl is None:          # shift = 0 column -> ARL0
                        table_row[label] = f"{arl0:.2f}"
                    else:
                        table_row[label] = f"{arl:.2f}"
                        long_rows.append({"skewness": skewness, "model": name,
                                          "shift_type": shift_type,
                                          "shift": value, "arl": arl, "mcse": mcse,
                                          "replications": args.arl_replications,
                                          "horizon": args.arl_horizon})
                combined.setdefault((skewness, shift_type), []).append(table_row)

        # combined tables: T2 (paper) row + all six ML methods
        for (sk, shift_type), rows in combined.items():
            if sk != skewness:
                continue
            t2_row = {"Methods": "T2 (paper)"}
            for j, value in enumerate(GRIDS[shift_type]):
                t2_row[f"{value:g}"] = f"{PAPER_T2[skewness][shift_type][j]:.2f}"
            frame = pd.DataFrame([t2_row] + rows)
            tag = f"{shift_type}_lambda{skewness:g}"
            frame.to_csv(out / f"table_{tag}.csv", index=False)
            (out / f"table_{tag}.md").write_text(
                frame.to_markdown(index=False), encoding="utf-8")
            print(f"    {shift_type} table written", flush=True)

        # per-skewness combined frame with all models
        sub = pd.DataFrame([r for r in long_rows if r["skewness"] == skewness])
        sub.to_csv(out / f"all_methods_long_lambda{skewness:g}.csv", index=False)
        print(f"skewness {skewness} done in {(time.perf_counter() - t0) / 60:.1f} min",
              flush=True)

    pd.DataFrame(long_rows).to_csv(out / "all_paper_tables_long.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps({
        "seed": args.seed, "skewness_values": skewness_values,
        "samples_per_class": args.samples_per_class,
        "calibration_profiles": args.calibration_profiles,
        "arl_replications": args.arl_replications, "arl_horizon": args.arl_horizon,
        "grids": {k: v.tolist() for k, v in GRIDS.items()},
        "models": MODELS, "note": "T2 rows copied from manuscript Tables 2-13",
    }, indent=2), encoding="utf-8")
    print("\nAll tables written to", out)


if __name__ == "__main__":
    main()


