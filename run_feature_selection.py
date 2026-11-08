'''Filter, embedded, wrapper, and PCA feature-selection audit.

All selectors use the same simulated Phase-I labelled profiles, fixed stratified
80/20 split, and training-only fitting. The test set is held out until the final
full-versus-subset comparison. This script is an audit/robustness analysis and
does not modify the established six-model benchmark outputs.
'''
from __future__ import annotations
import json, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/"reviewer1_ml_benchmark"
OUT=Path(__file__).resolve().parent/"outputs"
sys.path.insert(0,str(SOURCE)); sys.path.insert(0,str(ROOT))
import ml_reviewer_benchmark as core
from run_one_model import build_model
FEATURES=core.FEATURE_NAMES
SEED=20260825

def scale_scores(x):
    return RobustScaler().fit_transform(x)

def fit_selector(x,y,seed=SEED):
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=.2,stratify=y,random_state=seed)
    mi=mutual_info_classif(xtr,ytr,random_state=seed)
    corr=np.array([abs(np.corrcoef(xtr[:,j],ytr)[0,1]) for j in range(xtr.shape[1])])
    gb=GradientBoostingClassifier(n_estimators=200,max_depth=2,learning_rate=.03,random_state=seed).fit(xtr,ytr)
    embedded=gb.feature_importances_
    rfe_est=LogisticRegression(penalty="l2",max_iter=3000,random_state=seed)
    rfe_x=RobustScaler().fit_transform(xtr)
    rfe=RFE(rfe_est,n_features_to_select=8,step=1).fit(rfe_x,ytr)
    # PCA dimension selection: retain >=95% training variance, with a hard
    # maximum of 8 components for a compact comparator.
    pca_full=PCA().fit(scale_scores(xtr)); cumulative=np.cumsum(pca_full.explained_variance_ratio_)
    pca_k=min(8,max(1,int(np.searchsorted(cumulative,.95)+1)))
    selectors={"MI_top8":np.argsort(mi)[-8:][::-1],"Correlation_top8":np.argsort(corr)[-8:][::-1],"GB_gain_top8":np.argsort(embedded)[-8:][::-1],"RFE_top8":np.flatnonzero(rfe.support_)}
    return xtr,xte,ytr,yte,mi,corr,embedded,rfe,selectors,pca_k,pca_full

def evaluate(xtr,xte,ytr,yte,subset,name,seed=SEED):
    est,grid=build_model("GradientBoosting",seed)
    # Use the established best GB setting to isolate the feature-selection effect.
    est=GradientBoostingClassifier(n_estimators=200,max_depth=2,learning_rate=.03,random_state=seed)
    cv=StratifiedKFold(5,shuffle=True,random_state=seed)
    cv_auc=float(cross_val_score(est,xtr[:,subset],ytr,cv=cv,scoring="roc_auc").mean())
    start=time.perf_counter(); est.fit(xtr[:,subset],ytr); elapsed=time.perf_counter()-start
    p=est.predict_proba(xte[:,subset])[:,1]; pred=p>=.5
    return {"representation":name,"n_features":len(subset),"features":";".join(FEATURES[i] for i in subset),"cv_auc":cv_auc,"test_auc":float(roc_auc_score(yte,p)),"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred,zero_division=0)),"detection_rate_power":float(recall_score(yte,pred)),"false_alarm_rate":float(np.mean(pred[yte==0])),"mcc":float(matthews_corrcoef(yte,pred)),"fit_seconds":elapsed}

def main():
    rng=np.random.default_rng(SEED); x,y=core.make_dataset(SEED,2000,1.0,.5)
    xtr,xte,ytr,yte,mi,corr,embedded,rfe,selectors,pca_k,pca_full=fit_selector(x,y)
    rows=[]
    full=np.arange(x.shape[1]); rows.append(evaluate(xtr,xte,ytr,yte,full,"Full_16"))
    for name,idx in selectors.items(): rows.append(evaluate(xtr,xte,ytr,yte,idx,name))
    # PCA is fitted on training only, then evaluated with the same held-out test.
    scaler=RobustScaler().fit(xtr); ztr=scaler.transform(xtr); zte=scaler.transform(xte); pca=PCA(n_components=pca_k,random_state=SEED).fit(ztr)
    est=GradientBoostingClassifier(n_estimators=200,max_depth=2,learning_rate=.03,random_state=SEED); cv=StratifiedKFold(5,shuffle=True,random_state=SEED); cv_auc=float(cross_val_score(est,pca.transform(ztr),ytr,cv=cv,scoring="roc_auc").mean()); start=time.perf_counter(); est.fit(pca.transform(ztr),ytr); prob=est.predict_proba(pca.transform(zte))[:,1]; pred=prob>=.5; rows.append({"representation":"PCA","n_features":pca_k,"features":f"PC1..PC{pca_k}","cv_auc":cv_auc,"test_auc":float(roc_auc_score(yte,prob)),"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred,zero_division=0)),"detection_rate_power":float(recall_score(yte,pred)),"false_alarm_rate":float(np.mean(pred[yte==0])),"mcc":float(matthews_corrcoef(yte,pred)),"fit_seconds":time.perf_counter()-start})
    OUT.mkdir(parents=True,exist_ok=True); results=pd.DataFrame(rows); results.to_csv(OUT/"feature_representation_comparison.csv",index=False)
    ranking=pd.DataFrame({"feature":FEATURES,"mutual_information":mi,"abs_correlation":corr,"gb_gain_importance":embedded,"rfe_selected":rfe.support_}).sort_values("mutual_information",ascending=False); ranking.to_csv(OUT/"feature_selection_rankings.csv",index=False)
    pd.DataFrame({"component":np.arange(1,len(pca_full.explained_variance_ratio_)+1),"explained_variance_ratio":pca_full.explained_variance_ratio_,"cumulative_explained_variance":np.cumsum(pca_full.explained_variance_ratio_)}).to_csv(OUT/"pca_explained_variance.csv",index=False)
    fig,ax=plt.subplots(figsize=(11,5),dpi=180); ranking.sort_values("gb_gain_importance",ascending=True).plot.barh(x="feature",y="gb_gain_importance",ax=ax,legend=False,color="#4c78a8"); ax.set_title("Embedded Gradient-Boosting feature importance"); ax.set_xlabel("Gain-based importance"); fig.tight_layout(); fig.savefig(OUT/"gradient_boosting_feature_importance.png"); plt.close(fig)
    meta={"data_source":"simulated Phase-I labelled profiles only","seed":SEED,"features_full":FEATURES,"split":"stratified 80/20; selectors fitted on training only","cv":"5-fold StratifiedKFold; ROC-AUC","selector_sizes":{"filter_MI":8,"filter_correlation":8,"embedded_GB":8,"wrapper_RFE":8,"PCA_components":pca_k},"pca_training_variance_retained":float(np.sum(pca_full.explained_variance_ratio_[:pca_k]))}
    (OUT/"feature_selection_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    report="# Feature-selection and dimension-reduction audit\n\n"
    report+="The original 16-feature representation was compared with filter, embedded, wrapper, and PCA reductions. All selectors were fitted using the training split only; the held-out test split was used once for the final comparison. The classifier was Gradient Boosting with the established best hyperparameters, so the comparison isolates the representation rather than changing both representation and model.\n\n"
    report+="## Retained feature sets\n\n"
    for name,idx in selectors.items(): report+=f"- **{name}**: {', '.join(FEATURES[i] for i in idx)}\n"
    report+=f"- **PCA**: PC1–PC{pca_k}, retaining {meta['pca_training_variance_retained']:.3f} of training variance.\n\n## Comparison\n\n"+results.to_markdown(index=False)+"\n\n## Interpretation\n\nMI and absolute-correlation rankings are filter screens; Gradient-Boosting gain importance is an embedded screen; RFE is a wrapper method; PCA is unsupervised dimension reduction. The full 16-feature result is retained as the primary specification unless the selected subset has comparable or better held-out performance. The ranking and comparison files report the final retained variables and allow the reader to assess overfitting sensitivity.\n"
    (OUT/"feature_selection_report.md").write_text(report,encoding="utf-8")
    print("Feature-selection audit complete ->",OUT)
if __name__=="__main__": main()
