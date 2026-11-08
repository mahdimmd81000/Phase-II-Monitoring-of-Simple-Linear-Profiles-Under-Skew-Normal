"""Aggregate the per-model outputs into one comparison table (paper-ready).

Reads  reviewer1_ml_benchmark/outputs/*_result.json  and writes:
  outputs/all_models_metrics.csv   - classification metrics for the six models
  outputs/all_models_arl.csv       - calibrated ARL0 and ARL1 (+ MCSE) per model/scenario
  outputs/all_models_summary.csv   - one row per model, ready for the manuscript
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    rows, arl_rows = [], []
    for path in sorted(OUT.glob("*_result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(result["metrics"])
        arl_rows.append({**result["calibration"], "target_arl0": 200.0})
        arl_rows.extend(result["arl_results"])
    metrics = pd.DataFrame(rows)
    arl = pd.DataFrame(arl_rows)
    metrics.to_csv(OUT / "all_models_metrics.csv", index=False)
    arl.to_csv(OUT / "all_models_arl.csv", index=False)

    summary = metrics[["model", "cv_auc", "test_auc", "accuracy", "recall_detection_rate",
                       "false_alarm_rate", "mcc", "seconds"]].copy()
    cal = arl[arl["calibration_profiles"].notna()][["model", "empirical_calibration_arl0"]]
    summary = summary.merge(cal, on="model", how="left")
    for level in ["small", "moderate", "large"]:
        part = arl[arl["shift_level"] == level][["model", "arl_zero_state", "mcse"]]
        part = part.rename(columns={"arl_zero_state": f"ARL_{level}",
                                    "mcse": f"MCSE_{level}"})
        summary = summary.merge(part, on="model", how="left")
    summary.to_csv(OUT / "all_models_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
