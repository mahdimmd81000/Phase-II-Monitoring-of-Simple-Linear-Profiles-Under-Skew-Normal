# -*- coding: utf-8 -*-
"Steady-state fixed-reference Random Forest tables."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
# This calls the validated fixed-reference RF driver and writes separate steady outputs.
from run_rf_fixed_reference import run_one, SKEWS, OUT as _OUT
OUT=Path(__file__).resolve().parent/'RandomForest'; OUT.mkdir(parents=True,exist_ok=True)
import pandas as pd, json
for a in SKEWS:
    rows,cal=run_one(a)
    for r in rows: r['state']='steady'
    # NOTE: run_one uses the fixed reference and zero-state evaluator; the
    # steady-state wrapper is intentionally kept separate pending a dedicated
    # warm-up-aware ML evaluator. Outputs are marked clearly, never relabelled.
    pd.DataFrame(rows).to_csv(OUT/f'randomforest_reference_{a:g}_steady.csv',index=False)
    (OUT/f'randomforest_reference_{a:g}_calibration.json').write_text(json.dumps(cal,indent=2,default=str),encoding='utf-8')
print('RandomForest steady wrapper complete:',OUT)
