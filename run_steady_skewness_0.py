# -*- coding: utf-8 -*-
"Run the validated steady-state Woodall comparison for skewness a=0."""
from pathlib import Path
import json
import sys
import pandas as pd
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import woodall2003_tables as w
A = 0.0
REPS = 2000
HORIZON = 1000
CAL_REPS = 1500
CAL_HORIZON = 5000
SEED = 20260825
OUT = BASE / "steady_results" / "skewness_0"
OUT.mkdir(parents=True, exist_ok=True)

w.IC_CAL_REPS = CAL_REPS
w.IC_CAL_HORIZON = CAL_HORIZON
rows, cal = w.run_method(A, "steady", REPS, HORIZON, max(HORIZON, 4000), SEED + 50)
(OUT / "woodall2003_skewness_0_steady.md").write_text(
    w.md_table(rows, A, "steady", cal), encoding="utf-8")
pd.DataFrame(rows).to_csv(OUT / "woodall2003_skewness_0_steady.csv", index=False)
(OUT / "woodall2003_skewness_0_steady_calibration.json").write_text(
    json.dumps(cal, indent=2, default=float), encoding="utf-8")
print(f"finished skewness={A:g}; outputs: {OUT}")
