'''SVM ML benchmark and ARL paper tables.

This standalone script contains the SVM model, shared profile/feature/ARL code,
training/calibration, and paper-table generation. It does not import case-study
modules or any other local Python file.
'''
from __future__ import annotations
import argparse
from pathlib import Path
import zlib
import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
SEED = 20260825
N_OBS = 10
X_VALUES = np.arange(1, 11, dtype=float)
TRUE_A0, TRUE_A1, TRUE_SIGMA = 3.0, 2.0, 1.0
TARGET_ARL0 = 200.0
CALIBRATION_PROFILES = 10000
ARL_REPLICATIONS = 1000
ARL_HORIZON = 600
SHIFTS_INT = np.round(np.arange(0.0, 2.01, 0.2), 2)
SHIFTS_SLP = np.round(np.arange(0.0, 0.2501, 0.025), 3)
SD_LEVELS = np.round(np.arange(1.0, 3.01, 0.2), 2)
GRIDS = {"intercept": SHIFTS_INT, "slope": SHIFTS_SLP, "sd": SD_LEVELS}
PAPER_T2 = {
    0.0: {"intercept": [200.00, 100.00, 26.74, 8.13, 3.47, 1.94, 1.33, 1.10, 1.03, 1.01, 1.00], "slope": [200.00, 108.70, 44.25, 16.84, 7.17, 3.91, 2.28, 1.58, 1.26, 1.11, 1.03], "sd": [200.00, 35.46, 13.85, 7.43, 4.96, 3.61, 2.87, 2.44, 2.13, 1.93, 1.77]},
    0.5: {"intercept": [200.00, 40.65, 10.82, 4.02, 2.06, 1.34, 1.10, 1.02, 1.00, 1.00, 1.00], "slope": [200.00, 56.82, 20.16, 8.64, 4.28, 2.36, 1.59, 1.25, 1.10, 1.03, 1.01], "sd": [200.00, 34.72, 12.47, 6.93, 4.57, 3.32, 2.70, 2.27, 1.97, 1.79, 1.66]},
    1.0: {"intercept": [200.00, 29.94, 7.16, 2.72, 1.53, 1.14, 1.02, 1.00, 1.00, 1.00, 1.00], "slope": [200.00, 48.54, 14.41, 5.95, 2.84, 1.79, 1.29, 1.10, 1.03, 1.01, 1.00], "sd": [200.00, 23.92, 7.97, 4.29, 2.83, 2.18, 1.84, 1.62, 1.46, 1.37, 1.29]},
    2.0: {"intercept": [200.00, 27.03, 6.00, 2.17, 1.26, 1.04, 1.00, 1.00, 1.00, 1.00, 1.00], "slope": [200.00, 45.87, 14.01, 4.96, 2.33, 1.45, 1.13, 1.03, 1.00, 1.00, 1.00], "sd": [200.00, 21.46, 6.23, 3.06, 2.10, 1.63, 1.40, 1.25, 1.18, 1.12, 1.09]},
}

def compute_features_batch(profiles):
    values = np.asarray(profiles, dtype=float)
    y_mean = values.mean(axis=1)
    y_sd = values.std(axis=1, ddof=1)
    centered = values - y_mean[:, None]
    safe_y_sd = y_sd + 1e-12
    x_mean = X_VALUES.mean()
    slope = np.sum((X_VALUES - x_mean) * centered, axis=1) / np.sum((X_VALUES - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
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

def sigma0(a):
    delta = a / np.sqrt(1.0 + a * a)
    return float(TRUE_SIGMA * np.sqrt(1.0 - 2.0 * delta ** 2 / np.pi))

def gen_errors(rng, skewness, shape):
    delta = skewness / np.sqrt(1.0 + skewness * skewness)
    size = tuple(shape) + (N_OBS,)
    z = delta * np.abs(rng.standard_normal(size)) \
        + np.sqrt(1.0 - delta * delta) * rng.standard_normal(size)
    return (z - delta * np.sqrt(2.0 / np.pi)) * TRUE_SIGMA

def make_profiles(rng, skewness, kind, magnitude, count):
    if kind == "sd":
        a0, a1, sigma = TRUE_A0, TRUE_A1, sigma0(skewness) + (float(magnitude) - 1.0) * TRUE_SIGMA
    elif kind == "intercept":
        a0, a1, sigma = TRUE_A0 + magnitude, TRUE_A1, TRUE_SIGMA
    elif kind == "slope":
        a0, a1, sigma = TRUE_A0, TRUE_A1 + magnitude, TRUE_SIGMA
    elif kind == "variance":
        a0, a1, sigma = TRUE_A0, TRUE_A1, TRUE_SIGMA * (1.0 + magnitude)
    else:
        a0, a1, sigma = TRUE_A0, TRUE_A1, TRUE_SIGMA
    profiles = skewnorm.rvs(a=skewness, loc=0, scale=sigma,
                            size=(count, N_OBS), random_state=rng)
    return a0 + a1 * X_VALUES + profiles

def make_profile(rng, skewness, kind="ic", magnitude=0.0):
    return make_profiles(rng, skewness, kind, magnitude, 1)[0]

def make_dataset(seed, samples_per_class, skewness, shifted_fraction=0.5):
    rng = np.random.default_rng(seed)
    total = int(round(samples_per_class / shifted_fraction))
    n_shifted = samples_per_class
    n_in_control = max(1, total - n_shifted)
    in_control = [compute_features_batch(make_profile(rng, skewness)[None, :])[0] for _ in range(n_in_control)]
    shifted = []
    for _ in range(samples_per_class):
        kind = rng.choice(["intercept", "slope", "variance"])
        if kind == "intercept":
            magnitude = rng.choice([-1, 1]) * rng.uniform(.05, 2.0)
        elif kind == "slope":
            magnitude = rng.choice([-1, 1]) * rng.uniform(.01, .4)
        else:
            magnitude = rng.uniform(.1, 1.5)
        shifted.append(compute_features_batch(make_profile(rng, skewness, kind, magnitude)[None, :])[0])
    return np.vstack([in_control, shifted]), np.r_[np.zeros(n_in_control), np.ones(n_shifted)]

def build_model(seed=SEED):
    model = Pipeline([("scale", RobustScaler()), ("model", SVC(probability=True, class_weight="balanced", random_state=seed))])
    grid = {"model__C": [.1, 1, 10], "model__gamma": ["scale", .01]}
    return model, grid

def fit_model(x_train, y_train, seed):
    estimator, grid = build_model(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_score_

def calibrate_model_threshold(model, seed, skewness, calibration_profiles=CALIBRATION_PROFILES):
    rng = np.random.default_rng(seed)
    profiles = np.vstack([make_profile(rng, skewness) for _ in range(calibration_profiles)])
    scores = model.predict_proba(compute_features_batch(profiles))[:, 1]
    threshold = float(np.quantile(scores, 1.0 - 1.0 / TARGET_ARL0, method="linear"))
    return threshold, float(np.mean(scores >= threshold))

def estimate_model_arl(model, threshold, seed, skewness, kind, magnitude, replications=ARL_REPLICATIONS, horizon=ARL_HORIZON):
    rng = np.random.default_rng(seed)
    profiles = np.vstack([make_profile(rng, skewness, kind, magnitude) for _ in range(replications * horizon)])
    scores = model.predict_proba(compute_features_batch(profiles))[:, 1]
    alarm_matrix = (scores.reshape(replications, horizon) >= threshold)
    first_alarm = np.argmax(alarm_matrix, axis=1)
    has_alarm = alarm_matrix.any(axis=1)
    run_lengths = np.where(has_alarm, first_alarm + 1, horizon).astype(float)
    return float(run_lengths.mean()), float(run_lengths.std(ddof=1) / np.sqrt(replications))

def arl_grid(model, threshold, seed, skewness, shift_type, reps, horizon):
    rows = []
    for i, value in enumerate(GRIDS[shift_type]):
        if shift_type == "sd":
            kind, magnitude = "sd", float(value)
        else:
            kind, magnitude = shift_type, float(value)
        if magnitude == 0.0:
            rows.append((value, None, None))
        else:
            arl, mcse = estimate_model_arl(model, threshold, seed + 9000 + i, skewness, kind, magnitude, reps, horizon)
            rows.append((value, arl, mcse))
    return rows

def run_tables(skewness_values, samples_per_class=2000, shifted_fraction=0.5,
               calibration_profiles=CALIBRATION_PROFILES,
               arl_replications=ARL_REPLICATIONS, arl_horizon=ARL_HORIZON, seed=SEED):
    out = Path(__file__).resolve().parent / "arl_results"
    out.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for skewness in skewness_values:
        x, y = make_dataset(seed, samples_per_class, skewness, shifted_fraction)
        x_train, _, y_train, _ = train_test_split(x, y, test_size=.2, stratify=y, random_state=seed)
        model, _ = fit_model(x_train, y_train, seed)
        threshold, alarm_rate = calibrate_model_threshold(model, seed + 7000, skewness, calibration_profiles)
        arl0 = 1.0 / max(alarm_rate, 1e-12)
        for shift_type in GRIDS:
            table_row = {"Methods": "SVM"}
            for value, arl, mcse in arl_grid(model, threshold, seed, skewness, shift_type, arl_replications, arl_horizon):
                label = f"{value:g}"
                if arl is None:
                    table_row[label] = f"{arl0:.2f}"
                else:
                    table_row[label] = f"{arl:.2f}"
                    long_rows.append({"skewness": skewness, "model": "SVM", "shift_type": shift_type, "shift": value, "arl": arl, "mcse": mcse, "replications": arl_replications, "horizon": arl_horizon})
            t2_row = {"Methods": "T2 (paper)"}
            for j, value in enumerate(GRIDS[shift_type]):
                t2_row[f"{value:g}"] = f"{PAPER_T2[skewness][shift_type][j]:.2f}"
            frame = pd.DataFrame([t2_row, table_row])
            tag = f"{shift_type}_lambda{skewness:g}"
            frame.to_csv(out / f"table_{tag}.csv", index=False)
            frame.to_markdown(out / f"table_{tag}.md", index=False)
    pd.DataFrame(long_rows).to_csv(out / "all_paper_tables_long.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skewness", type=float, default=1.0)
    parser.add_argument("--all-skewness", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--shifted-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-profiles", type=int, default=CALIBRATION_PROFILES)
    parser.add_argument("--arl-replications", type=int, default=ARL_REPLICATIONS)
    parser.add_argument("--arl-horizon", type=int, default=ARL_HORIZON)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    skewness_values = [0.0, 0.5, 1.0, 2.0] if args.all_skewness else [args.skewness]
    run_tables(skewness_values, args.samples_per_class, args.shifted_fraction,
               args.calibration_profiles, args.arl_replications,
               args.arl_horizon, args.seed)

if __name__ == "__main__":
    main()
