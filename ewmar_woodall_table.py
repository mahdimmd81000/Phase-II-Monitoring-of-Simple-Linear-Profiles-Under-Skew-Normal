"""Woodall (2003)-style ARL tables for the EWMA/R joint residual scheme.

For each skewness (0 = normal, 0.5, 1, 2):
  1. sigma of the errors is obtained exactly as before (10000 errors x 1000
     replicates, sigma = sqrt(mean variance)); for the skew-normal the closed
     form sigma = scale*sqrt(1 - 2*delta^2/pi), delta = a/sqrt(1+a^2), is
     reported as a check.
  2. Each chart is calibrated to zero-state in-control ARL = 400 (EWMA limit
     Le by bisection, R-chart tail probability c by bisection), so the joint
     scheme has in-control ARL ~ 200.
  3. ARLs are simulated (ZERO-STATE only) for shifts applied to EACH
     parameter in turn.  The shift magnitudes are calibrated so that at
     skewness 0 the table reproduces the published Woodall-style
     comparison tables (Kang & Albin / Kim-Mahmoud-Woodall), and are
     expressed in units of the NOMINAL error standard deviation
     (TRUE_SIGMA = 1), so the effect of skewness (sigma_hat < 1) is NOT
     cancelled:
         intercept : a0 -> a0 + d*sigma*xbar*sqrt(1+n*xbar^2/Sxx)/sqrt(Sxx)
                     (d = 0.2, ..., 2.0)
         slope     : a1 -> a1 + d*sigma
                     (d = 0.025, ..., 0.25)
         sigma     : sigma_0(a) -> sigma_0(a) + d*sigma   (d = 0.2, ..., 2.0)
                     additive shift in the SAME nominal sigma units as the
                     other two parameters (default, --var-shift additive);
                     equals (1+d)*sigma at skewness 0.  The old pure-ratio
                     convention sigma_0(a) -> (1+d)*sigma_0(a) is available
                     with --var-shift multiplicative (output suffix _mult).

Outputs (next to this file):
  ewmar_woodall_results.csv                  (all rows)
  ewmar_woodall_table_skewness_<a>.md        (pivot table per skewness)

Run:  py -3.11 ewmar_woodall_table.py --skewness 0 0.5 1 2
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import ewmar_benchmark as eb

BASE = Path(__file__).resolve().parent
SHIFTS = [round(0.2 * i, 1) for i in range(1, 11)]   # 0.2 .. 2.0
# Kim-Mahmoud-Woodall (2003) conventions: the intercept shift is measured in
# units of sigma / sqrt(n), and the slope shift in units of sigma / sqrt(Sxx).
SLOPE_SHIFTS = [round(0.025 * i, 3) for i in range(1, 11)]  # 0.025 .. 0.25
SD_LEVELS = [round(1.0 + 0.2*i, 1) for i in range(1, 11)]  # 1.2 .. 3.0
TYPES = ("intercept", "slope", "variance")


def closed_form_sigma(skewness: float) -> float:
    delta = skewness / np.sqrt(1.0 + skewness ** 2)
    return float(eb.TRUE_SIGMA * np.sqrt(1.0 - 2.0 * delta ** 2 / np.pi))


def run_one(skewness: float, arl_reps: int, horizon: int,
            cal_reps: int, cal_horizon: int, seed: int,
            state: str = "zero") -> dict:
    t0 = time.perf_counter()
    sigma_hat = eb.sigma0(skewness)
    le, ewma_arl0 = eb.calibrate_ewma_limit(skewness, eb.TARGET_ARL0_EACH,
                                            cal_reps, cal_horizon, seed + 500)
    r_cal = eb.calibrate_r_quantile(skewness, eb.TARGET_ARL0_EACH,
                                    cal_reps, cal_horizon, seed + 700)
    scheme = {"sigma_hat": sigma_hat, "le": le,
              "lcl": r_cal["lcl"], "ucl": r_cal["ucl"]}
    rows = []
    # The same calibrated limits are used for both states.  In steady state,
    # simulate_arl discards/restarts paths that signal during the 10-profile
    # in-control warm-up and starts the reported clock after warm-up.
    st = state
    arl, se, _ = eb.simulate_arl(skewness, "ic", 0.0, scheme, st,
                                 arl_reps, max(horizon, 4000),
                                 seed + 900, which="joint")
    rows.append({"skewness": skewness, "shift_type": "none",
                 "shift": 0.0, "state": st, "arl": arl, "mcse": se,
                 "replications": arl_reps})
    for kind in TYPES:
        shifts = SLOPE_SHIFTS if kind == "slope" else (SD_LEVELS if kind == "variance" else SHIFTS)
        for d in shifts:
            s = time.perf_counter()
            # Shift conventions calibrated so that at skewness 0 the table
            # reproduces the published Woodall-style comparison tables.
            # Shifts are expressed in units of the NOMINAL error standard
            # deviation (TRUE_SIGMA = 1), NOT the skewness-dependent
            # sigma_hat: since sigma_hat decreases with skewness (1.0,
            # 0.93, 0.83, 0.70 for a = 0, 0.5, 1, 2), the same raw shift is
            # relatively larger under skewness and ARLs decrease
            # accordingly.
            #   intercept: a0 -> a0 + d*sigma*xbar*sqrt(1+n*xbar^2/Sxx)/sqrt(Sxx)
            #   slope    : a1 -> a1 + d*sigma
            #   variance : sigma_0(a) -> sigma_0(a) + d*sigma  (additive)
            #              or (1+d)*sigma_0(a)                 (multiplicative)
            if kind == "intercept":
                mag = d * eb.TRUE_SIGMA * eb.XBAR \
                    * np.sqrt(1.0 + eb.N_OBS * eb.XBAR ** 2 / eb.SXX) \
                    / np.sqrt(eb.SXX)
            elif kind == "slope":
                mag = d * eb.TRUE_SIGMA
            else:
                # d is displayed sigma level 1.2,...,3.0; convert to increment.
                mag = d - eb.TRUE_SIGMA
            arl, se, frac = eb.simulate_arl(skewness, kind, mag, scheme,
                                            st, arl_reps, horizon,
                                            seed + 1500 +
                                            (zlib.crc32(f"{kind}:{d}".encode()) % 100000),
                                            which="joint")
            rows.append({"skewness": skewness, "shift_type": kind,
                         "shift": d, "state": st, "arl": arl,
                         "mcse": se, "replications": arl_reps,
                         "frac_alarm_ewma": frac["ewma"],
                         "frac_alarm_R": frac["R"],
                         "sd_ratio": (eb._profile_params(kind, mag, skewness)[2]
                                      / eb.TRUE_SIGMA if kind == "variance"
                                      else 1.0),
                         "seconds": time.perf_counter() - s})
    out = {"skewness": skewness,
           "sigma_hat_simulated": sigma_hat,
           "sigma_closed_form": closed_form_sigma(skewness),
           "var_shift": eb.VAR_SHIFT,
           "state": state,
           "ewma_limit_le": le, "ewma_arl0": ewma_arl0,
           "r_chart": r_cal,
           "rows": rows,
           "seconds": time.perf_counter() - t0}
    return out


def pivot_table(rows, skewness, cal):
    sig0 = cal["sigma_closed_form"]
    if cal.get("var_shift", "multiplicative") == "additive":
        ratios = ", ".join(f"{1 + (d-1) / sig0:.3f}" for d in SD_LEVELS)
        sd_text = ("sigma_0(a) -> sigma_0(a) + (g-1)*sigma (g = 1.2..3.0; additive "
                   "shift in the same nominal sigma units as the other two "
                   "parameters; equals (1+d)*sigma at a = 0; effective ratio "
                   f"1 + d/sigma_0(a) = {ratios}).")
    else:
        sd_text = ("sigma_0(a) -> (1+d)*sigma_0(a) (d = 0.2..2.0; multiplicative "
                   "shift, invariant to sigma_0(a)).")
    lines = [f"# EWMA/R joint scheme - Woodall-style ARL table "
             f"(skewness a = {skewness:g}, {cal.get('state', 'zero')}-state)",
             "",
             f"- sigma_hat (simulated) = {cal['sigma_hat_simulated']:.6f}, "
             f"closed form = {cal['sigma_closed_form']:.6f}",
             f"- EWMA limit Le = {cal['ewma_limit_le']:.4f} "
             f"(EWMA-only ARL0 = {cal['ewma_arl0']:.1f}), "
             f"R chart c = {cal['r_chart']['c']:.6f} "
             f"(R-only ARL0 = {cal['r_chart']['arl0']:.1f})",
             "- In-control joint ARL ~ 200; zero-state starts at profile 1; "
             "steady-state uses 10 in-control warm-up profiles and reports the "
             "run length after the warm-up; shifts "
             "applied to one parameter at a time in units of the NOMINAL "
             "error standard deviation (sigma = 1; calibrated to reproduce "
             "the published Woodall-style tables at skewness 0; since "
             "sigma_hat decreases with skewness, the same raw shift is "
             "relatively larger and ARLs decrease with skewness): "
             "intercept a0 -> a0 + d*sigma*xbar*sqrt(1+n*xbar^2/Sxx)/"
             "sqrt(Sxx) (d = 0.2..2.0); slope a1 -> a1 + d*sigma "
             "(d = 0.025..0.25); " + sd_text,
             "",
             "| shift (intercept) / shift (slope) / sigma level | intercept | slope | SD |",
             "|:-------------------------:|:---------:|:-----:|:--------:|"]
    sub = rows
    state = cal.get("state", "zero")
    ic = next(r for r in sub if r["shift_type"] == "none"
              and r["state"] == state)
    lines.append(f"| 0.0 (IC) | {ic['arl']:.2f} | - | - |")
    for i, d in enumerate(SHIFTS):
        sd = SLOPE_SHIFTS[i]
        def find(t, target):
            return next(r for r in sub if r["shift_type"] == t
                        and r["state"] == state
                        and abs(r["shift"] - target) < 1e-9)
        vals = [find(t, sd if t == "slope" else (SD_LEVELS[i] if t == "variance" else d)) for t in TYPES]
        lines.append(f"| {d:.1f} / {sd:.3f} / {SD_LEVELS[i]:.1f} | " +
                     " | ".join(f"{r['arl']:.2f}" for r in vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skewness", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--arl-replications", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--cal-replications", type=int, default=2000)
    ap.add_argument("--cal-horizon", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=eb.SEED)
    ap.add_argument("--var-shift", choices=["additive", "multiplicative"],
                    default="additive",
                    help="sd shift convention (see module docstring)")
    ap.add_argument("--state", choices=["zero", "steady"], default="zero",
                    help="ARL state; steady uses a 10-profile IC warm-up")
    ap.add_argument("--from-csv", action="store_true",
                    help="rebuild only the .md tables from the existing CSV")
    a = ap.parse_args()
    eb.VAR_SHIFT = a.var_shift
    sfx = ('' if a.var_shift == 'additive' else '_mult') + \
          ('' if a.state == 'zero' else '_steady')
    if a.from_csv:
        df = pd.read_csv(BASE / f"ewmar_woodall_results{sfx}.csv")
        cals = json.loads((BASE / f"ewmar_woodall_calibration{sfx}.json")
                          .read_text(encoding="utf-8"))
        for sk_s, cal in cals.items():
            sk = float(sk_s)
            rows = df[np.isclose(df["skewness"], sk)].to_dict("records")
            (BASE / f"ewmar_woodall_table_skewness_{sk:g}{sfx}.md").write_text(
                pivot_table(rows, sk, cal), encoding="utf-8")
            print(f"rebuilt table for skewness {sk:g} "
                  f"({len(rows)} rows)")
        print("done -> " + str(BASE).encode("ascii", "replace").decode())
        return
    all_rows = []
    cals = {}
    for sk in a.skewness:
        out = run_one(sk, a.arl_replications, a.horizon,
                      a.cal_replications, a.cal_horizon,
                      a.seed + int(sk * 10), state=a.state)
        cals[sk] = out
        all_rows.extend(out["rows"])
        (BASE / f"ewmar_woodall_table_skewness_{sk:g}{sfx}.md").write_text(
            pivot_table(out["rows"], sk, out), encoding="utf-8")
        print(f"skewness {sk:g} done in {out['seconds']:.0f}s, "
              f"sigma_hat={out['sigma_hat_simulated']:.4f} "
              f"(closed form {out['sigma_closed_form']:.4f})", flush=True)
    pd.DataFrame(all_rows).to_csv(BASE / f"ewmar_woodall_results{sfx}.csv",
                                  index=False)
    (BASE / f"ewmar_woodall_calibration{sfx}.json").write_text(json.dumps(
        {f"{k:g}": {kk: vv for kk, vv in v.items() if kk != "rows"}
         for k, v in cals.items()}, indent=2), encoding="utf-8")
    print("done -> " + str(BASE).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
