'''Run the established six-model benchmark into this case-study folder.

The modelling machinery is imported from reviewer1_ml_benchmark/run_one_model.py,
which already enforces the reviewer protocol. This driver only changes the output
location: it does not alter the original benchmark code or its results.
'''
from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "reviewer1_ml_benchmark"
RESULTS = HERE / "results"
sys.path.insert(0, str(SOURCE))
from run_one_model import MODELS, run_model  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skewness", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--samples-per-class", type=int, default=2000)
    ap.add_argument("--shifted-fraction", type=float, default=0.5)
    ap.add_argument("--calibration-profiles", type=int, default=10000)
    ap.add_argument("--arl-replications", type=int, default=2000)
    ap.add_argument("--arl-horizon", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.samples_per_class = 400
        args.calibration_profiles = 2000
        args.arl_replications = 200
        args.arl_horizon = 100
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    for model in MODELS:
        print(f"Running {model} ...", flush=True)
        result = run_model(model, args.seed, args.samples_per_class,
                           args.skewness, args.shifted_fraction,
                           args.calibration_profiles, args.arl_replications,
                           args.arl_horizon)
        results.append(result)
        tag = model.lower()
        (RESULTS / f"{tag}_result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
        pd.DataFrame([result["metrics"]]).to_csv(
            RESULTS / f"{tag}_metrics.csv", index=False)
        pd.DataFrame(result["arl_results"]).to_csv(
            RESULTS / f"{tag}_arl.csv", index=False)
        print(f"  test AUC={result['metrics']['test_auc']:.4f}; "
              f"ARL0={result['calibration']['empirical_calibration_arl0']:.1f}",
              flush=True)

    metrics = pd.DataFrame([r["metrics"] for r in results])
    metrics.to_csv(RESULTS / "model_comparison_metrics.csv", index=False)
    arl = pd.concat([pd.DataFrame(r["arl_results"]) for r in results],
                    ignore_index=True)
    arl.to_csv(RESULTS / "model_comparison_arl.csv", index=False)
    metadata = {
        "python": platform.python_version(),
        "seed": args.seed, "skewness": args.skewness,
        "models": MODELS,
        "feature_contract": "same 16-feature representation from ml_reviewer_benchmark",
        "split": "stratified 80/20, same random_state for every model",
        "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True)",
        "target_arl0": 200.0,
        "calibration_profiles": args.calibration_profiles,
        "arl_replications": args.arl_replications,
        "arl_horizon": args.arl_horizon,
        "quick": args.quick,
        "seconds": time.perf_counter() - started,
        "source_driver": str(SOURCE / "run_one_model.py"),
    }
    (RESULTS / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    report = [
        "# Six-model ML benchmark",
        "",
        f"- Skewness: {args.skewness}",
        f"- Models: {', '.join(MODELS)}",
        "- Same 16 features, same stratified 80/20 split, same 5-fold CV, "
        "and separate empirical ARL0 calibration to target 200.",
        "",
        "## Classification metrics",
        "",
        metrics[["model", "cv_auc", "test_auc", "accuracy", "precision",
                 "recall_detection_rate", "false_alarm_rate", "mcc"]]
        .to_markdown(index=False),
        "",
        "## ARL results",
        "",
        arl.to_markdown(index=False),
        "",
        "The complete numerical results, Monte-Carlo standard errors, and selected "
        "hyperparameters are in the CSV/JSON files in this directory.",
    ]
    (RESULTS / "ML_benchmark_report.md").write_text(
        "\n".join(report), encoding="utf-8")
    print(f"Complete: {RESULTS}", flush=True)

if __name__ == "__main__":
    main()
