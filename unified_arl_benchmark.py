"""Fair T2/ML/EWMA-3/EWMA-R benchmark on the original table grids.

The unit of monitoring is an independent linear profile with n=10 points.
ARL is the first profile index exceeding a method-specific limit. Every
method sees the same random profiles for each (lambda, shift, state) scenario.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260825
N = 10
X = np.arange(1.0, N + 1.0)
A0 = 3.0
A1 = 2.0
SIGMA = 1.0
LAMBDA_EWMA = 0.20
TARGET_ARL0 = 200.0
OUT = Path(__file__).resolve().parent / "reviewer_outputs"


def profile_features(y):
    """Return the same 16-feature contract for every method."""
    y = np.asarray(y, dtype=float)
    xbar, ybar = X.mean(), y.mean()
    slope = np.sum((X - xbar) * (y - ybar)) / np.sum((X - xbar) ** 2)
    intercept = ybar - slope * xbar
    residual = y - intercept - slope * X
    sd = np.std(y, ddof=1)
    mad = np.median(np.abs(y - np.median(y)))
    z = 0.6745 * (y - np.median(y)) / (mad + 1e-12)
    changes = np.diff(y)
    residual_sd = np.std(residual, ddof=1)
    residual_acf1 = np.corrcoef(residual[:-1], residual[1:])[0, 1]
    return np.array([
        y.mean(), sd, 0.0 if sd == 0 else np.mean(((y - ybar) / sd) ** 3),
        0.0 if sd == 0 else np.mean(((y - ybar) / sd) ** 4) - 3,
        intercept, slope, np.percentile(y, 25), np.percentile(y, 75),
        np.mean(np.abs(z) > 3.5), mad, np.mean(changes[:-1] * changes[1:] < 0) if len(changes) > 1 else 0.0,
        residual_sd, 0.0 if residual_sd == 0 else np.mean(((residual - residual.mean()) / residual_sd) ** 3),
        0.0 if residual_sd == 0 else np.mean(((residual - residual.mean()) / residual_sd) ** 4) - 3,
        np.mean(np.abs(residual)), y.max() - y.min(),
    ], dtype=float)


def generate_profiles(rng, count, skewness, intercept_shift=0.0, slope_shift=0.0, sigma_ratio=1.0):
    intercept = np.asarray(intercept_shift, dtype=float)
    slope = np.asarray(slope_shift, dtype=float)
    sigma_scale = np.asarray(sigma_ratio, dtype=float)
    if intercept.ndim == 1:
        intercept = intercept[:, None]
    if slope.ndim == 1:
        slope = slope[:, None]
    if sigma_scale.ndim == 1:
        sigma_scale = sigma_scale[:, None]
    errors = skewnorm.rvs(a=skewness, loc=0, scale=SIGMA * sigma_scale,
                          size=(count, N), random_state=rng)
    return (A0 + intercept) + (A1 + slope) * X + errors


def profile_statistics(profiles):
    features = np.vstack([profile_features(row) for row in profiles])
    parameters = features[:, [4, 5, 11]]
    parameters[:, 2] = np.log(np.maximum(parameters[:, 2] ** 2, 1e-12))
    residuals = profiles - parameters[:, 0, None] - parameters[:, 1, None] * X
    return features, parameters, residuals.mean(axis=1)


def train_ml(rng, skewness, count=4000):
    in_control = generate_profiles(rng, count // 2, skewness)
    shifted = generate_profiles(rng, count - count // 2, skewness,
                                intercept_shift=rng.choice([-1, 1], count - count // 2) * rng.uniform(.05, 2.0, count - count // 2),
                                slope_shift=rng.choice([-1, 1], count - count // 2) * rng.uniform(.01, .4, count - count // 2),
                                sigma_ratio=rng.uniform(1.1, 2.5, count - count // 2))
    x0 = np.vstack([profile_features(row) for row in in_control])
    x1 = np.vstack([profile_features(row) for row in shifted])
    model = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=SEED))
    model.fit(np.vstack([x0, x1]), np.r_[np.zeros(len(x0)), np.ones(len(x1))])
    return model


def statistics(profiles, reference, ml_model):
    features, parameters, residual_means = profile_statistics(profiles)
    ref_features, ref_parameters, ref_residual_means = reference
    center = ref_parameters.mean(axis=0)
    covariance = np.cov(ref_parameters, rowvar=False) + 1e-8 * np.eye(3)
    delta = parameters - center
    t2 = np.einsum("ij,jk,ik->i", delta, np.linalg.inv(covariance), delta)
    standardized = (parameters - center) / (ref_parameters.std(axis=0, ddof=1) + 1e-12)
    z = np.zeros(3); ewma3 = []
    for row in standardized:
        z = LAMBDA_EWMA * row + (1 - LAMBDA_EWMA) * z
        ewma3.append(np.max(np.abs(z)))
    residual_scale = ref_residual_means.std(ddof=1) + 1e-12
    r_z = residual_means / residual_scale
    zr = 0.0; ewmar = []
    for value in r_z:
        zr = LAMBDA_EWMA * value + (1 - LAMBDA_EWMA) * zr
        ewmar.append(abs(zr))
    ml = ml_model.predict_proba(features)[:, 1]
    return {"T2": t2, "ML": ml, "EWMA3": np.asarray(ewma3), "EWMAR": np.asarray(ewmar)}


def calibrate(rng, reference, ml_model, method, skewness, calibration_paths, horizon):
    maxima = []
    for _ in range(calibration_paths):
        profiles = generate_profiles(rng, horizon, skewness)
        maxima.append(np.max(statistics(profiles, reference, ml_model)[method]))
    probability = 1.0 - (1.0 - 1.0 / TARGET_ARL0) ** horizon
    return float(np.quantile(maxima, probability))


def scenario_shift(kind, magnitude):
    if kind == "intercept": return magnitude, 0.0, 1.0
    if kind == "slope": return 0.0, magnitude, 1.0
    if kind == "variance": return 0.0, 0.0, 1.0 + magnitude
    return 0.0, 0.0, 1.0


def estimate_arl(rng, reference, model, method, limit, kind, magnitude, reps, horizon, steady):
    lengths = []
    for _ in range(reps):
        warmup = 40 if steady else 0
        warm = generate_profiles(rng, warmup, 0.0)
        a0, a1, ratio = scenario_shift(kind, magnitude)
        shifted = generate_profiles(rng, horizon, 0.0, a0, a1, ratio)
        sequence = np.vstack([warm, shifted])
        values = statistics(sequence, reference, model)[method]
        hits = np.flatnonzero(values[warmup:] > limit)
        lengths.append(float(hits[0] + 1) if len(hits) else float(horizon))
    values = np.asarray(lengths)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=5000)
    parser.add_argument("--calibration-paths", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--reference-profiles", type=int, default=4000)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    shifts = {
        "IC": ("IC", 0.0),
        "intercept": ("intercept", 0.2),
        "slope": ("slope", 0.025),
        "variance": ("variance", 0.2),
    }
    for skewness in (0.0, 0.5, 1.0, 2.0):
        rng = np.random.default_rng(SEED + int(skewness * 100))
        reference_profiles = generate_profiles(rng, args.reference_profiles, skewness)
        reference = profile_statistics(reference_profiles)
        model = train_ml(rng, skewness)
        limits = {method: calibrate(rng, reference, model, method, skewness, args.calibration_paths, args.horizon)
                  for method in ("T2", "ML", "EWMA3", "EWMAR")}
        for shift_name, (kind, magnitude) in shifts.items():
            for state_index, state in enumerate(("zero", "steady")):
                for method in limits:
                    scenario_rng = np.random.default_rng(SEED + int(skewness * 100) + 10000 * list(shifts).index(shift_name) + state_index)
                    arl, mcse = estimate_arl(scenario_rng, reference, model, method, limits[method], kind, magnitude,
                                              args.replications, args.horizon, state == "steady")
                    rows.append({"skewness": skewness, "method": method, "shift": shift_name, "state": state,
                                 "shift_magnitude": magnitude, "ucl": limits[method], "arl": arl, "mcse": mcse,
                                 "replications": args.replications})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "unified_arl_results.csv", index=False)
    (OUT / "unified_arl_metadata.json").write_text(json.dumps({"seed": SEED, "target_arl0": TARGET_ARL0,
        "lambda_ewma": LAMBDA_EWMA, "n_points": N, "replications": args.replications,
        "calibration_paths": args.calibration_paths, "horizon": args.horizon,
        "shift_grid": shifts, "methods": ["T2", "ML", "EWMA3", "EWMAR"]}, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
