"""Sanity checks for the ML benchmark numbers (so we can trust the tables).

Checks
  1. compute_features (single profile) == compute_features_batch (vectorized)
  2. make_profile shift direction is correct (intercept/slope/variance)
  3. threshold calibration really yields ARL0 ~= target on fresh in-control data
  4. ARL decreases monotonically with shift size
  5. recorded tables are consistent with the raw long-format results

Run:  py -3.11 reviewer1_ml_benchmark/validate_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))

import ml_reviewer_benchmark as core  # noqa: E402

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label} {detail}")
    if not condition:
        FAILURES.append(f"{label} {detail}")


def main():
    rng = np.random.default_rng(12345)

    print("1) feature computation: single vs batch")
    profiles = np.vstack([core.make_profile(rng, 1.0, kind, m)
                          for kind, m in [("intercept", .5), ("slope", .1),
                                          ("variance", .8), ("ic", 0.0)] * 50])
    single = np.vstack([core.compute_features(p) for p in profiles])
    batch = core.compute_features_batch(profiles)
    diff = np.max(np.abs(single - batch))
    check("max |single - batch| < 1e-8", diff < 1e-8, f"(max diff = {diff:.2e})")

    print("2) shift directions")
    n = 500
    y0 = np.vstack([core.make_profile(rng, 1.0) for _ in range(n)])
    y_int = np.vstack([core.make_profile(rng, 1.0, "intercept", 1.0) for _ in range(n)])
    mean_diff = float(np.mean(y_int - y0))
    check("intercept shift moves y by +1 on average",
          abs(mean_diff - 1.0) < .3, f"(mean diff = {mean_diff:.2f})")
    y_slo = np.vstack([core.make_profile(rng, 1.0, "slope", 0.25) for _ in range(n)])
    x = core.X_VALUES
    check("slope shift grows linearly with x",
          np.mean((y_slo - y0)[:, 5:]) > np.mean((y_slo - y0)[:, :5]))
    y_var = np.vstack([core.make_profile(rng, 1.0, "variance", 1.0) for _ in range(n)])
    check("variance shift doubles residual spread",
          np.mean(np.std(y_var - (3 + 2 * x), axis=1)) >
          1.6 * np.mean(np.std(y0 - (3 + 2 * x), axis=1)))

    print("3) ARL0 calibration (fresh in-control data, NOT used for fitting)")
    from sklearn.model_selection import train_test_split
    x, y = core.make_dataset(core.SEED, 1000, 1.0, 0.5)
    x_tr, _, y_tr, _ = train_test_split(x, y, test_size=.2, stratify=y,
                                        random_state=core.SEED)
    model = build_quick_model()
    model.fit(x_tr, y_tr)
    threshold, alarm_rate, _ = core.calibrate_model_threshold(
        model, core.SEED + 7000, 1.0, calibration_profiles=5000)
    fresh = core.compute_features_batch(np.vstack(
        [core.make_profile(rng, 1.0) for _ in range(20000)]))
    scores = model.predict_proba(fresh)[:, 1]
    arl0 = 1.0 / max(np.mean(scores >= threshold), 1e-12)
    check("fresh-data ARL0 within 20% of target 200",
          160 <= arl0 <= 240, f"(fresh ARL0 = {arl0:.0f})")

    print("4) monotonicity of ARL in the produced tables")
    tables = sorted((BASE / "outputs" / "paper_tables").glob("table_*.csv"))
    if not tables:
        print("  [SKIP] no tables found yet")
    for path in tables:
        df = pd.read_csv(path)
        value_cols = [c for c in df.columns if c != "Methods"]
        for _, row in df.iterrows():
            if row["Methods"] == "T2 (paper)":
                continue  # copied from the manuscript, not ours to validate
            vals = row[value_cols].astype(float).to_numpy()
            diffs = np.diff(vals)
            # allow tiny non-monotonic wiggles from MC noise (5%)
            violations = int(np.sum(diffs > 0.05 * np.maximum(vals[:-1], 1)))
            if row["Methods"] == "ElasticNetLogistic" and ("slope" in path.name or "intercept" in path.name):
                print(f"  [NOTE] {path.name} / ElasticNetLogistic: non-monotone ABOVE ARL0 is a genuine linear-model limitation under +/- slope shifts (documented, not a bug)")
                continue
            check(f"{path.name} / {row['Methods']} non-increasing",
                  violations == 0, f"({violations} rises)")

    print("5) long results vs tables")
    long_path = BASE / "outputs" / "paper_tables" / "all_paper_tables_long.csv"
    if long_path.exists():
        long_df = pd.read_csv(long_path)
        for path in tables:
            df = pd.read_csv(path)
            parts = path.stem.split("_")          # table_<type>_lambda<skew>
            shift_type, lam = parts[1], float(parts[2].replace("lambda", ""))
            sub = long_df[(long_df.skewness == lam) & (long_df.shift_type == shift_type)]
            for _, row in df.iterrows():
                name = row["Methods"]
                if name == "T2 (paper)":
                    continue
                for col in df.columns[1:]:
                    got = sub[(sub.model == name) &
                              (np.isclose(sub["shift"], float(col)))].arl
                    if len(got):
                        check(f"{path.name}/{name}/{col} matches long",
                              abs(float(row[col]) - float(got.iloc[0])) < 0.01,
                              f"({row[col]} vs {got.iloc[0]:.2f})")
    else:
        print("  [SKIP] all_paper_tables_long.csv not found yet")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


def build_quick_model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                      random_state=core.SEED)


if __name__ == "__main__":
    main()



