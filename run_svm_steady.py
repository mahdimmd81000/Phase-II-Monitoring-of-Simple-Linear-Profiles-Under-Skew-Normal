# -*- coding: utf-8 -*-
"Steady-state fixed-reference SVM tables."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
from svm_standalone_fixed import run_one, SKEWS, OUT as _OUT
OUT=Path(__file__).resolve().parent/'SVM'; OUT.mkdir(parents=True,exist_ok=True)
import pandas as pd, json
for a in SKEWS:
    rows,cal=run_one(a)
    for r in rows: r['state']='steady'
    pd.DataFrame(rows).to_csv(OUT/f'svm_reference_{a:g}_steady.csv',index=False)
    (OUT/f'svm_reference_{a:g}_calibration.json').write_text(json.dumps(cal,indent=2,default=str),encoding='utf-8')
print('SVM steady wrapper complete:',OUT)
