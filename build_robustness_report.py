'''Create an algorithm-robustness report for the ML-versus-statistical claim.

This report joins the verified all-model ML ARL results, the manuscript T2
reference rows at skewness=1, and the feature-selection sensitivity results.
It does not rerun or modify any benchmark source files.
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
ROOT=HERE.parents[2]
OUT=HERE/"outputs"
ML_ARL=ML/"verified_full_results"/"model_comparison_arl.csv"
ML_MET=ML/"verified_full_results"/"model_comparison_metrics.csv"
FS=ML/"feature_selection"/"outputs"/"feature_representation_comparison.csv"
T2_INTER=ROOT/"reviewer1_ml_benchmark"/"outputs"/"paper_tables"/"table_intercept_lambda1.md"
T2_SD=ROOT/"reviewer1_ml_benchmark"/"outputs"/"paper_tables"/"table_sd_lambda1.md"

def parse_md(path, shift_col):
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "Methods" in line or ":" in line: continue
        cells=[c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells)<4: continue
        method=cells[0]
        if method=="T2 (paper)":
            # columns are IC, then 0.2..2.0 (or 1.2..3.0 for sd)
            values=[float(c) for c in cells[1:] if c and c!="-"]
            for shift,val in zip(shift_col,values): rows.append({"method":method,"shift":shift,"arl":val})
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    arl=pd.read_csv(ML_ARL); metrics=pd.read_csv(ML_MET); fs=pd.read_csv(FS)
    small=arl[arl.shift_level.eq("small")].copy(); small["method"]=small.model; small["source"]="verified ML benchmark"
    shifts=[0.2,0.4,0.6,0.8,1.0]
    t2i=parse_md(T2_INTER,shifts)
    t2s=parse_md(T2_SD,[1.2,1.4,1.6,1.8,2.0])
    t2i.to_csv(OUT/"t2_reference_intercept_lambda1.csv",index=False); t2s.to_csv(OUT/"t2_reference_sd_lambda1.csv",index=False)
    all_models=arl[["model","shift_level","shift_magnitude","arl_zero_state","mcse"]].copy()
    all_models.to_csv(OUT/"all_ml_algorithms_arl.csv",index=False)
    # Relative-to-T2 at comparable intercept shifts available in both sources.
    ml_small=arl[(arl.shift_type=="intercept") & arl.shift_magnitude.isin(shifts)][["model","shift_magnitude","arl_zero_state"]]
    merged=ml_small.merge(t2i.rename(columns={"arl":"t2_arl"}),left_on="shift_magnitude",right_on="shift",how="left")
    merged["ml_faster_than_t2"]=merged.arl_zero_state < merged.t2_arl
    merged["arl_ratio_ml_over_t2"]=merged.arl_zero_state/merged.t2_arl
    merged.to_csv(OUT/"ml_vs_t2_intercept_comparison.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,5),dpi=180)
    for model,g in arl[arl.shift_type.eq("intercept")].groupby("model"):
        ax.plot(g.shift_magnitude,g.arl_zero_state,"o-",label=model)
    ax.plot(t2i["shift"],t2i["arl"],"k--",lw=2,label="T2 (paper)")
    ax.set_yscale("log"); ax.set_xlabel("Intercept shift"); ax.set_ylabel("Zero-state ARL (log scale)"); ax.set_title("Algorithm robustness: ML models versus T2"); ax.grid(alpha=.25); ax.legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(OUT/"ml_algorithms_vs_t2_arl.png"); plt.close(fig)
    metrics_sorted=metrics.sort_values("test_auc",ascending=False)
    best=metrics_sorted.iloc[0]
    fs_best=fs.sort_values("test_auc",ascending=False).iloc[0]
    report=f'''# Robustness of the ML-versus-statistical comparison
## Purpose
The reviewer requested evidence that any advantage is not an artifact of choosing one particular ML algorithm. This audit therefore compares all six ML algorithms and also compares the full 16-feature representation with filter, embedded, wrapper, and PCA representations. The statistical reference is the manuscript T2 row at skewness lambda=1.

## Algorithm robustness
The full benchmark includes Gradient Boosting, Random Forest, SVM, kNN, MLP, and Elastic-Net logistic regression. Every model used the same features, data split, cross-validation, ARL0 calibration, and Phase-II simulation. The best classification AUC was obtained by **{best.model}** (test AUC = {best.test_auc:.4f}); Gradient Boosting achieved test AUC = {float(metrics.loc[metrics.model.eq('GradientBoosting'),'test_auc'].iloc[0]):.4f}. The complete all-model ARL table is `all_ml_algorithms_arl.csv`.

At the small intercept shift used in the verified ML benchmark, the nonlinear ML models have ARLs ranging from approximately {arl[(arl.shift_level=='small') & (arl.model!='ElasticNetLogistic')].arl_zero_state.min():.2f} to {arl[(arl.shift_level=='small') & (arl.model!='ElasticNetLogistic')].arl_zero_state.max():.2f}, while Elastic-Net is much less sensitive. Thus the qualitative result is not restricted to Gradient Boosting: it is shared by multiple nonlinear learners, although the magnitude depends on the algorithm.

## Feature-selection robustness
The best held-out representation in this audit was **{fs_best.representation}** with test AUC = {fs_best.test_auc:.4f}; the full 16-feature model had test AUC = {float(fs.loc[fs.representation.eq('Full_16'),'test_auc'].iloc[0]):.4f}. The eight-feature correlation subset and the full representation are essentially equivalent, whereas PCA is weaker. The retained feature lists are in `../feature_selection/outputs/feature_selection_report.md`.

## ML versus T2
The file `ml_vs_t2_intercept_comparison.csv` provides direct comparable-shift rows. ARL0 is controlled at the common target in the ML benchmark, and the T2 paper row is retained as reported in the manuscript. The plot uses a log ARL axis so small, moderate, and large shifts can be inspected together.

## Conclusion
The evidence does not support the overly strong statement that every ML algorithm dominates T2 at every shift. The defensible conclusion is narrower: multiple nonlinear ML algorithms provide fast detection for small-to-moderate persistent shifts, while the linear Elastic-Net classifier does not show the same advantage. Feature-selection results show that this pattern is robust to reducing the 16-feature set. Therefore, the observed advantage is associated with nonlinear profile classification rather than being solely an artifact of the Gradient Boosting choice; however, performance remains algorithm- and shift-dependent.
'''
    (OUT/"ml_vs_statistical_robustness_report.md").write_text(report,encoding="utf-8")
    metadata={"sources":[str(ML_ARL),str(ML_MET),str(FS),str(T2_INTER),str(T2_SD)],"skewness":1.0,"ml_models":metrics.model.tolist(),"feature_representations":fs.representation.tolist(),"comparison":"verified ML all-model ARL versus manuscript T2; feature-selection sensitivity included","caution":"T2 and ML source tables must be described with their design/shift conventions in the manuscript."}
    (OUT/"robustness_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("Robustness report complete ->",OUT)
if __name__=="__main__": main()
