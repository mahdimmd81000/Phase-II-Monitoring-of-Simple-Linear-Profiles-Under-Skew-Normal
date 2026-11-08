# -*- coding: utf-8 -*-
"Steady-state fixed-reference MLP tables."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
import run_ml_rf_svm_mlp as ml
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd, json, numpy as np
from pathlib import Path
OUT=Path(__file__).resolve().parent/'MLP'; OUT.mkdir(parents=True,exist_ok=True)
SEED=20260825; SKEWS=[0.,.5,1.,2.]
rng=np.random.default_rng(SEED); x,y=ml.make_training_set(rng,0.,2000,.5); xt,xe,yt,ye=train_test_split(x,y,test_size=.2,stratify=y,random_state=SEED); cv=StratifiedKFold(5,shuffle=True,random_state=SEED); est,grid=ml.build_model('MLP',SEED); search=GridSearchCV(est,grid,scoring='roc_auc',cv=cv,n_jobs=1); search.fit(xt,yt); model=search.best_estimator_
for a in SKEWS:
    thr,rate,n,tie=ml.calibrate_threshold(model,a,SEED+7000+int(100*a),20000); rows=[]; ic,se=ml.estimate_arl(model,thr,tie,a,'ic',0.,2000,4000,SEED+900+int(100*a)); rows.append({'skewness':a,'shift_type':'none','shift':0.,'arl':ic,'mcse':se,'state':'steady'})
    for k,vals in [('intercept',ml.SHIFTS_INT),('slope',ml.SHIFTS_SLP),('variance',ml.SD_LEVELS)]:
        for i,d in enumerate(vals):
            z,e=ml.estimate_arl(model,thr,tie,a,k,d,2000,1000,SEED+1500+i+int(100*a)); rows.append({'skewness':a,'shift_type':k,'shift':d,'arl':z,'mcse':e,'state':'steady'})
    pd.DataFrame(rows).to_csv(OUT/f'mlp_reference_{a:g}_steady.csv',index=False); (OUT/f'mlp_reference_{a:g}_calibration.json').write_text(json.dumps({'threshold':thr,'tie':tie,'rate':rate,'params':search.best_params_},indent=2,default=str),encoding='utf-8')
print('MLP steady wrapper complete:',OUT)
