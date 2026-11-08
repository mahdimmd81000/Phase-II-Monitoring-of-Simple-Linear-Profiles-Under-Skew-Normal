# -*- coding: utf-8 -*-
"Steady-state EWMA3 tables for skewness 0, 0.5, 1, 2."""
from pathlib import Path
import json
import pandas as pd
import sys
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
import woodall2003_tables as w
OUT=Path(__file__).resolve().parent/'EWMA3'; OUT.mkdir(parents=True,exist_ok=True)
for a in (0.0,0.5,1.0,2.0):
    lim=w.calibrate3(a,1500,5000,w.SEED+int(100*a))
    rows=[]
    ic,se=w.arl3(a,'ic',0,lim,'steady',2000,4000,w.SEED+900)
    rows.append({'skewness':a,'state':'steady','shift_type':'none','shift':0.0,'arl':ic,'mcse':se,'replications':2000})
    for kind,grid in (('intercept',w.SHIFTS_INT),('slope',w.SHIFTS_SLP),('variance',w.SD_LEVELS)):
        for i,d in enumerate(grid):
            v,se=w.arl3(a,kind,d,lim,'steady',2000,1000,w.SEED+1500+i+int(100*a))
            rows.append({'skewness':a,'state':'steady','shift_type':kind,'shift':d,'arl':v,'mcse':se,'replications':2000})
    pd.DataFrame(rows).to_csv(OUT/f'ewma3_steady_skewness_{a:g}.csv',index=False)
    (OUT/f'ewma3_steady_skewness_{a:g}.md').write_text(w.md_table(rows,a,'steady',{'ewma3':lim,'ewmar':{},'t2':{}}),encoding='utf-8')
    (OUT/f'ewma3_steady_skewness_{a:g}_calibration.json').write_text(json.dumps(lim,indent=2,default=float),encoding='utf-8')
print('EWMA3 steady complete:',OUT)
