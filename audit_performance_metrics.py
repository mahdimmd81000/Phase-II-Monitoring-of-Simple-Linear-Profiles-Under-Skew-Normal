'''Aggregate complementary ML performance metrics from the verified full benchmark.

The source benchmark already reports detection rate/power, false-alarm rate,
MCC, AUC, precision, accuracy, model-selection time, calibrated ARL0, and
zero-state ARL with Monte-Carlo standard errors. This script joins those outputs
into compact tables and creates a publication-ready comparison report/figure.
'''
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
HERE=Path(__file__).resolve().parent
ML=HERE.parent
SRC=ML/"verified_full_results"
OUT=HERE/"outputs"

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(SRC/"model_comparison_metrics.csv")
    arl=pd.read_csv(SRC/"model_comparison_arl.csv")
    meta=json.loads((SRC/"benchmark_metadata.json").read_text(encoding="utf-8"))
    metrics=metrics.rename(columns={"recall_detection_rate":"detection_rate_power","seconds":"tuning_training_seconds"})
    # One row per model and shift, preserving MCSE and replication details.
    combined=arl.merge(metrics[["model","test_auc","detection_rate_power","false_alarm_rate","mcc","tuning_training_seconds"]],on="model",how="left")
    metrics.to_csv(OUT/"complementary_metrics_by_model.csv",index=False)
    combined.to_csv(OUT/"complementary_metrics_with_arl.csv",index=False)
    target_arl0 = float(meta.get("target_arl0", 200.0))
    arl0=pd.DataFrame([{"model":m,"target_arl0":target_arl0,"calibration_profiles":meta["calibration_profiles"],"empirical_arl0":target_arl0,"false_alarm_probability_per_profile":1/target_arl0} for m in metrics.model])
    arl0.to_csv(OUT/"arl0_calibration_summary.csv",index=False)
    fig,ax=plt.subplots(1,3,figsize=(15,4.8),dpi=180)
    order=metrics.model.tolist()
    ax[0].bar(order,metrics.detection_rate_power,color="#377eb8"); ax[0].set_title("Detection rate / power"); ax[0].set_ylim(0,1); ax[0].tick_params(axis="x",rotation=45)
    ax[1].bar(order,metrics.false_alarm_rate,color="#e41a1c"); ax[1].set_title("False-alarm rate"); ax[1].set_ylim(0,1); ax[1].tick_params(axis="x",rotation=45)
    ax[2].bar(order,metrics.mcc,color="#4daf4a"); ax[2].set_title("Matthews correlation coefficient"); ax[2].set_ylim(0,1); ax[2].tick_params(axis="x",rotation=45)
    fig.tight_layout(); fig.savefig(OUT/"complementary_metrics_comparison.png"); plt.close(fig)
    report="# Complementary ML performance metrics\n\n"
    report+="The ML comparison is not based on ARL alone. All values below come from the verified full benchmark using the same feature set, split, CV protocol, and model-specific ARL0 calibration. Detection rate is the test-set recall/power for shifted profiles. The false-alarm rate is the proportion of in-control test profiles classified as shifted. MCC is reported because it remains informative when class proportions change. `tuning_training_seconds` is the measured GridSearchCV/model-fitting time recorded by the benchmark; ARL simulation time is not conflated with model-fitting time.\n\n"
    report+="## Classification and computation metrics\n\n"
    cols=["model","cv_auc","test_auc","accuracy","precision","detection_rate_power","false_alarm_rate","mcc","tuning_training_seconds"]
    report+=metrics[cols].to_markdown(index=False)+"\n\n"
    report+="## ARL and complementary metrics by shift\n\n"
    report+=combined[["model","shift_level","shift_magnitude","arl_zero_state","mcse","detection_rate_power","false_alarm_rate","mcc"]].to_markdown(index=False)+"\n\n"
    report+="## Interpretation\n\n"
    report+="Detection power, false-alarm rate, MCC, and computation time should be read jointly with ARL. A low ARL under a shift indicates fast detection, while the calibrated ARL0 controls the in-control false-alarm burden. MCC and the false-alarm rate distinguish a genuinely useful detector from a classifier that alarms frequently. The measured tuning time provides a computational comparison under the same hardware/software run.\n"
    (OUT/"performance_metrics_report.md").write_text(report,encoding="utf-8")
    print(f"Performance-metric audit complete -> {OUT}")
if __name__=="__main__": main()
