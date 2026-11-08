"""EWMA/R residual benchmark chart for simple linear profiles (Reviewer-2).

Implements the Kang & Albin (2000) residual-based scheme with the two charts
run SIMULTANEOUSLY (joint scheme):
  * EWMA chart on the (standardized) AVERAGE residual of each profile from
  the KNOWN in-control reference line (Kang & Albin 2000),
  Z_t = lam*u_t + (1-lam)*Z_{t-1}, Z_0 = 0, two-sided limits
  +- Le * sqrt(lam/(2-lam)) (lambda = 0.2);
  * R chart on the range (max - min) of the residuals of each profile, with
    simulated two-sided control limits LCL = Q_R(c) and UCL = Q_R(1-c).
An alarm is raised as soon as EITHER chart signals (joint scheme).

Calibration (requested):
  * sigma used inside every formula is the ERROR standard deviation, estimated
    by simulation: 10000 skew-normal errors -> variance, repeated 1000 times,
    sigma_hat = sqrt(mean of the 1000 variances).
  * The EWMA chart is calibrated by simulation so its in-control ARL = 400.
  * The R chart quantile c is calibrated by simulation so its in-control
    ARL = 400.
  * Consequently the joint EWMA/R in-control ARL is ~ 400*400/(400+400)=200.
    The actually achieved joint ARL0 is reported and verified.

States:
  * zero-state   : process out of control from profile 1 onward.
  * steady-state : 10 in-control profiles first (replicate restarted if a
    false alarm occurs during warm-up), shift applied from profile 11
    (change point = 10).

Run:  py -3.11 ewmar_benchmark.py --skewness 1.0
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skewnorm  # used only for sigma_theory

SEED = 20260825
N_OBS = 10
X_VALUES = np.arange(1, 11, dtype=float)
XBAR = X_VALUES.mean()
SXX = float(np.sum((X_VALUES - XBAR) ** 2))
TRUE_A0, TRUE_A1, TRUE_SIGMA = 3.0, 2.0, 1.0
TARGET_ARL0_JOINT = 200.0      # joint EWMA/R scheme
TARGET_ARL0_EACH = 400.0       # each individual chart
LAMBDA_EWMA = 0.2
SIGMA_SIM_N = 10000
SIGMA_SIM_REPS = 1000
BASE = Path(__file__).resolve().parent
LCL_LCL = 1e-6

_SIGMA_CACHE: dict[float, float] = {}


def simulate_sigma(skewness: float, n_errors: int = SIGMA_SIM_N,
                   reps: int = SIGMA_SIM_REPS, seed: int = SEED + 1):
    """sigma = sqrt(mean over 1000 replicates of Var(10000 skew-normal errors))."""
    rng = np.random.default_rng(seed)
    variances = np.empty(reps)
    for i in range(reps):
        # exact skew-normal via Azzalini's mixture (fast, see _gen_errors)
        delta = skewness / np.sqrt(1.0 + skewness ** 2)
        z1 = np.abs(rng.standard_normal(n_errors))
        z2 = rng.standard_normal(n_errors)
        errors = (delta * z1 + np.sqrt(1.0 - delta ** 2) * z2) * TRUE_SIGMA
        variances[i] = errors.var(ddof=1)
    var_mean = float(variances.mean())
    delta = skewness / np.sqrt(1.0 + skewness ** 2)
    sigma_theory = TRUE_SIGMA * np.sqrt(1.0 - 2.0 * delta ** 2 / np.pi)
    return float(np.sqrt(var_mean)), var_mean, float(sigma_theory)


def sigma0(skewness: float, **kw) -> float:
    key = round(float(skewness), 6)
    if key not in _SIGMA_CACHE:
        _SIGMA_CACHE[key] = simulate_sigma(key, **kw)[0]
    return _SIGMA_CACHE[key]


def make_profile(rng, skewness, kind="ic", magnitude=0.0):
    if kind == "intercept":
        a0, a1, s = TRUE_A0 + magnitude, TRUE_A1, TRUE_SIGMA
    elif kind == "slope":
        a0, a1, s = TRUE_A0, TRUE_A1 + magnitude, TRUE_SIGMA
    elif kind == "variance":
        a0, a1, s = TRUE_A0, TRUE_A1, TRUE_SIGMA * (1.0 + magnitude)
    else:
        a0, a1, s = TRUE_A0, TRUE_A1, TRUE_SIGMA
    errors = skewnorm.rvs(a=skewness, loc=0, scale=s, size=N_OBS, random_state=rng)
    return a0 + a1 * X_VALUES + errors


def profile_residuals(profiles):
    """Residuals from the KNOWN in-control reference line (Kang & Albin 2000):
    e = y - a0 - a1*x.  OLS-fitted residuals MUST NOT be used here: they are
    exactly invariant to intercept/slope shifts and would make both charts
    blind to the very shifts this benchmark studies."""
    y = np.asarray(profiles, dtype=float)
    return y - TRUE_A0 - TRUE_A1 * X_VALUES


def _gen_errors(rng, skewness, shape, scale=TRUE_SIGMA, n_obs=N_OBS):
    """Batched centered skew-normal errors, shape = shape + (n_obs,).

    Exact skew-normal sampling via the mixture construction
        X = delta*|Z1| + sqrt(1-delta^2)*Z2,  delta = a/sqrt(1+a^2)
    (Azzalini 1985), which is mathematically identical to
    scipy.stats.skewnorm.rvs but far faster (pure numpy, no rejection)."""
    a = float(skewness)
    delta = a / np.sqrt(1.0 + a * a)
    size = tuple(shape) + (n_obs,)
    z1 = np.abs(rng.standard_normal(size))
    z2 = rng.standard_normal(size)
    x = delta * z1 + np.sqrt(1.0 - delta * delta) * z2
    # The skew-normal has mean delta*sqrt(2/pi) != 0; center it so the
    # in-control residuals have ZERO mean (otherwise the EWMA chart sees a
    # permanent mean shift and false-alarms immediately).
    x = x - delta * np.sqrt(2.0 / np.pi)
    return x * scale


# Standard-deviation shift convention for the EWMA/R design.
#   "additive"       : sigma_0(a) -> sigma_0(a) + d*TRUE_SIGMA  (same NOMINAL
#                      sigma units as the intercept/slope shifts; equals
#                      (1+d)*sigma at skewness 0, effective ratio 1+d/sigma_0(a)
#                      grows with skewness).
#   "multiplicative" : sigma_0(a) -> (1+d)*sigma_0(a)  (pure ratio; the R chart
#                      only sees the ratio, so the column is skewness-invariant).
VAR_SHIFT = "additive"


def _profile_params(kind, magnitude, skewness=0.0):
    if kind == "intercept":
        return TRUE_A0 + magnitude, TRUE_A1, TRUE_SIGMA
    if kind == "slope":
        return TRUE_A0, TRUE_A1 + magnitude, TRUE_SIGMA
    if kind == "variance":
        if VAR_SHIFT == "additive":
            ratio = 1.0 + magnitude * TRUE_SIGMA / sigma0(skewness)
        else:
            ratio = 1.0 + magnitude
        return TRUE_A0, TRUE_A1, TRUE_SIGMA * ratio
    return TRUE_A0, TRUE_A1, TRUE_SIGMA


def _resid_stats(errors, kind, magnitude, scheme):
    """errors (..., N_OBS) -> (u0, u1, r) arrays of leading shape.

    Kim-Mahmoud-Woodall (2003) EWMA statistics: standardized OLS estimates
       u0 = (a0_hat - A0) / sigma(a0_hat)
       u1 = (a1_hat - A1) / sigma(a1_hat)
    with the EXACT OLS standard errors
       sigma(a0_hat) = sigma * sqrt(1/n + xbar^2/Sxx)
       sigma(a1_hat) = sigma / sqrt(Sxx)
    (NOTE: sigma(a0_hat) is NOT sigma/sqrt(n); using sqrt(n) left u0 with
    in-control SD ~ sqrt(1 + xbar^2*Sxx/n) = 3.7 and neutered the slope
    chart because both streams shared one calibrated limit.)  With the
    exact standard errors both u0 and u1 are N(0,1) in control, and a
    shift of delta*sigma/sqrt(n) in a0 (delta*sigma/sqrt(Sxx) in a1)
    moves u0 (u1) by delta standard deviations, matching the delta units
    of the Woodall-style table.  r is the residual range (R chart)."""
    a0, a1, _ = _profile_params(kind, magnitude)
    y = a0 + a1 * X_VALUES + errors
    shape = y.shape
    ybar = y.mean(axis=-1, keepdims=True)
    sxy = ((X_VALUES - XBAR) * (y - ybar)).sum(axis=-1)
    a1_hat = sxy / SXX
    a0_hat = ybar.reshape(shape[:-1]) - a1_hat * XBAR
    sig = scheme["sigma_hat"]
    sd_a0 = sig * np.sqrt(1.0 / N_OBS + XBAR ** 2 / SXX)
    u0 = (a0_hat - TRUE_A0) / sd_a0
    u1 = (a1_hat - TRUE_A1) * np.sqrt(SXX) / sig
    resid = (y - TRUE_A0 - TRUE_A1 * X_VALUES).reshape(shape)
    r = np.ptp(resid, axis=-1)
    return u0, u1, r


def _run_path(u0, u1, r, z_init, scheme, lam, which):
    """Vectorized EWMA/EWMA/R walk.  u0, u1, r: (m, H);  z_init: (m, 2).
    Two EWMA streams (standardized intercept and slope estimates) with the
    SAME limit Le, plus the R chart.  Returns
    (first_alarm_1based_or_inf (m,), z_after (m, 2), src (m,) str)."""
    m, H = u0.shape
    b = np.sqrt(lam / (2.0 - lam))
    le_b = scheme["le"] * b
    z0 = z_init[:, 0].astype(float).copy()
    z1 = z_init[:, 1].astype(float).copy()
    hit = np.zeros((m, H), dtype=bool)
    src = np.zeros((m, H), dtype="<U4")
    for t in range(H):
        z0 = lam * u0[:, t] + (1.0 - lam) * z0
        z1 = lam * u1[:, t] + (1.0 - lam) * z1
        he = (np.abs(z0) > le_b) | (np.abs(z1) > le_b)
        hr = (r[:, t] < scheme["lcl"]) | (r[:, t] > scheme["ucl"])
        if which == "ewma":
            hr = np.zeros(m, dtype=bool)
        elif which == "R":
            he = np.zeros(m, dtype=bool)
        hit[:, t] = he | hr
        src[:, t] = np.where(he, "ewma", "R")
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1) + 1, np.inf).astype(float)
    src_first = np.where(any_hit,
                         src[np.arange(m), hit.argmax(axis=1)], "none")
    z_after = np.stack([z0, z1], axis=1)
    return first, z_after, src_first


def simulate_arl(skewness, kind, magnitude, scheme, state="zero",
                 reps=2000, horizon=5000, seed=SEED + 100,
                 warmup=10, lam=LAMBDA_EWMA, which="joint", chunk=250):
    """Run-length simulation (vectorized over replicates).  `which` in
    {'joint','ewma','R'} restricts the alarm source (used for chart-wise
    calibration / diagnostics).

    Errors are generated lazily in chunks of `chunk` profiles and the walk
    stops as soon as every replicate has alarmed (or `horizon` is reached),
    so long horizons are cheap.  The ARL estimate is the CENSORED mean
    (a run with no alarm by the horizon counts as `horizon`); the fraction
    of censored runs is returned in frac['none'] and must be negligible
    (it is reported in the JSON calibration block)."""
    rng = np.random.default_rng(seed)
    z = np.zeros((reps, 2))
    if state == "steady":
        # warm-up: regenerate any replicate that false-alarms in control
        alive = np.zeros(reps, dtype=bool)
        while not alive.all():
            idx = np.flatnonzero(~alive)
            m = idx.size
            err = _gen_errors(rng, skewness, (m, warmup))
            u0, u1, r = _resid_stats(err, "ic", 0.0, scheme)
            first, z_end, _ = _run_path(u0, u1, r, np.zeros((m, 2)), scheme,
                                        lam, which)
            ok = ~np.isfinite(first)
            alive[idx[ok]] = True
            z[idx[ok]] = z_end[ok]
    # monitoring phase (zero-state starts at Z_0 = 0); chunked generation
    first = np.full(reps, np.nan)          # NaN = run still alive
    sources: list[str] = []
    # EWMA/R design: n = N_OBS = 10, x = 1..10 (NOT the EWMA_3 n = 4 design).
    scale = _profile_params(kind, magnitude, skewness)[2]
    t_done = 0
    while t_done < horizon and np.isnan(first).any():
        n = min(chunk, horizon - t_done)
        idx = np.flatnonzero(np.isnan(first))
        err = _gen_errors(rng, skewness, (idx.size, n), scale=scale)
        u0, u1, r = _resid_stats(err, kind, magnitude, scheme)
        f, z_end, src_c = _run_path(u0, u1, r, z[idx], scheme, lam, which)
        new = np.isfinite(f)
        first[idx[new]] = f[new] + t_done
        sources.extend(src_c[i] for i in np.flatnonzero(new))
        z[idx] = z_end          # EWMA state carries across chunks
        t_done += n
    run_lengths = first
    finite = np.isfinite(run_lengths)   # NaN runs (no alarm) are censored
    # censored-mean ARL (no-alarm runs count as `horizon`)
    rl_full = np.where(finite, run_lengths, float(horizon))
    arl = float(rl_full.mean())
    se = float(rl_full.std(ddof=1) / np.sqrt(reps))
    frac = {s: float(sources.count(s)) / reps for s in ("ewma", "R", "none")}
    return arl, se, frac


def calibrate_ewma_limit(skewness, target=TARGET_ARL0_EACH, reps=2000,
                         horizon=5000, seed=SEED + 500, lo=0.5, hi=8.0,
                         tol=0.03):
    """Bisection on Le (zero-state in-control EWMA-only ARL = 400)."""
    sigma_hat = sigma0(skewness)
    best = None
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        scheme = {"sigma_hat": sigma_hat, "le": mid,
                  "lcl": -1e6, "ucl": 1e6}          # R chart disabled
        arl, _, _ = simulate_arl(skewness, "ic", 0.0, scheme, "zero",
                                 reps, horizon, seed, which="ewma")
        best = (mid, arl)
        if abs(arl - target) / target < tol:
            break
        # Le widens the limits and therefore RAISES ARL0: if the simulated
        # ARL is below target we must search HIGHER Le (lo = mid), and vice
        # versa.  (Getting this backwards drives Le to the hi bound.)
        if arl < target:
            lo = mid
        else:
            hi = mid
    return float(best[0]), float(best[1])


def calibrate_r_quantile(skewness, target=TARGET_ARL0_EACH, reps=2000,
                         horizon=5000, seed=SEED + 700, quantile_n=200000,
                         c_lo=0.0002, c_hi=0.02, tol=0.03, quick=False):
    """Bisection on the tail probability c for the R chart so its zero-state
    in-control ARL = 400.  LCL/UCL are the (1-c)-quantiles / c-quantiles of
    the in-control residual range, estimated from `quantile_n` simulated
    in-control profiles (one batched vectorized draw)."""
    sigma_hat = sigma0(skewness)
    if quick:
        quantile_n = min(quantile_n, 20000)
    rng = np.random.default_rng(seed + 1)
    # In control, reference-line residuals equal the errors themselves.
    ic = _gen_errors(rng, skewness, (quantile_n,))
    ranges = np.ptp(ic, axis=1)
    ranges.sort()

    def limits_for(c):
        lo_i = max(0, int(np.floor(c * quantile_n)))
        hi_i = min(quantile_n - 1, int(np.ceil((1.0 - c) * quantile_n)))
        return float(ranges[lo_i]), float(ranges[hi_i])

    best = None
    c_lo_, c_hi_ = c_lo, c_hi
    for _ in range(22):
        c = 0.5 * (c_lo_ + c_hi_)
        lcl, ucl = limits_for(c)
        scheme = {"sigma_hat": sigma_hat, "le": 1e6,
                  "lcl": lcl, "ucl": ucl}          # EWMA chart disabled
        arl, _, _ = simulate_arl(skewness, "ic", 0.0, scheme, "zero",
                                 reps, horizon, seed + 2, which="R")
        best = (c, lcl, ucl, arl)
        if abs(arl - target) / target < tol:
            break
        if arl < target:            # alarms too often -> widen limits -> smaller c
            c_hi_ = c
        else:
            c_lo_ = c
    c, lcl, ucl, arl = best
    return {"c": float(c), "lcl": float(lcl), "ucl": float(ucl), "arl0": float(arl)}


SHIFT_SCENARIOS = {
    "intercept_small": ("intercept", 0.20),
    "intercept_moderate": ("intercept", 0.60),
    "intercept_large": ("intercept", 1.00),
    "intercept_neg_moderate": ("intercept", -0.60),
    "slope_small": ("slope", 0.05),
    "slope_moderate": ("slope", 0.15),
    "slope_large": ("slope", 0.30),
    "slope_neg_moderate": ("slope", -0.15),
    "variance_small": ("variance", 0.10),
    "variance_moderate": ("variance", 0.50),
    "variance_large": ("variance", 1.00),
}


def run_benchmark(skewness=1.0, arl_reps=2000, horizon=500,
                  cal_reps=2000, cal_horizon=5000, seed=SEED,
                  scenarios=None, quick=False):
    t0 = time.perf_counter()
    if quick:
        arl_reps = min(arl_reps, 400)
        cal_reps = min(cal_reps, 300)
        cal_horizon = min(cal_horizon, 2000)
    ic_horizon = max(horizon, 4000)   # IC rows target ARL 400/200: need headroom
    sigma_hat, var_mean, sigma_theory = simulate_sigma(skewness, seed=seed + 1)
    le, ewma_arl0 = calibrate_ewma_limit(skewness, TARGET_ARL0_EACH,
                                         cal_reps, cal_horizon, seed + 500)
    r_cal = calibrate_r_quantile(skewness, TARGET_ARL0_EACH, cal_reps,
                                 cal_horizon, seed + 700, quick=quick)
    scheme = {"sigma_hat": sigma_hat, "le": le,
              "lcl": r_cal["lcl"], "ucl": r_cal["ucl"]}
    rows = []
    # In-control verification rows (each chart alone AND the joint scheme),
    # reported in BOTH states.  Each single chart targets 400, joint ~200.
    for st in ("zero", "steady"):
        arl_e, se_e, f_e = simulate_arl(skewness, "ic", 0.0, scheme, st,
                                        arl_reps, ic_horizon, seed + 900,
                                        which="ewma")
        rows.append({"method": "EWMA/R", "chart": "EWMA", "shift_type": "none",
                     "shift_magnitude": 0.0, "state": st, "arl": arl_e,
                     "mcse": se_e, "replications": arl_reps,
                     "horizon": ic_horizon,
                     "target_arl0": TARGET_ARL0_EACH})
        arl_r, se_r, f_r = simulate_arl(skewness, "ic", 0.0, scheme, st,
                                        arl_reps, ic_horizon, seed + 950,
                                        which="R")
        rows.append({"method": "EWMA/R", "chart": "R", "shift_type": "none",
                     "shift_magnitude": 0.0, "state": st, "arl": arl_r,
                     "mcse": se_r, "replications": arl_reps,
                     "horizon": ic_horizon,
                     "target_arl0": TARGET_ARL0_EACH})
    joint_arl0 = {}
    for st in ("zero", "steady"):
        arl_j, se_j, f_j = simulate_arl(skewness, "ic", 0.0, scheme, st,
                                        arl_reps, ic_horizon, seed + 990,
                                        which="joint")
        joint_arl0[st] = arl_j
        rows.append({"method": "EWMA/R", "chart": "joint", "shift_type": "none",
                     "shift_magnitude": 0.0, "state": st, "arl": arl_j,
                     "mcse": se_j, "replications": arl_reps,
                     "horizon": ic_horizon,
                     "target_arl0": TARGET_ARL0_JOINT})
    # Shifted scenarios: joint scheme, both states
    for name, (kind, mag) in (scenarios or SHIFT_SCENARIOS).items():
        for st in ("zero", "steady"):
            s = time.perf_counter()
            arl, se, frac = simulate_arl(skewness, kind, mag, scheme, st,
                                         arl_reps, horizon,
                                         seed + 1500 + (zlib.crc32(name.encode()) % 1000),
                                         which="joint")
            rows.append({"method": "EWMA/R", "chart": "joint",
                         "scenario": name, "shift_type": kind,
                         "shift_magnitude": mag, "state": st, "arl": arl,
                         "mcse": se, "replications": arl_reps,
                         "horizon": horizon, "target_arl0": TARGET_ARL0_JOINT,
                         "frac_alarm_ewma": frac["ewma"],
                         "frac_alarm_R": frac["R"],
                         "seconds": time.perf_counter() - s})
    out = {
        "method": "EWMA/R (Kang & Albin 2000, joint residual scheme)",
        "skewness": skewness,
        "sigma_estimation": {
            "procedure": (f"{SIGMA_SIM_N} errors per replicate x "
                          f"{SIGMA_SIM_REPS} replicates; sigma = sqrt(mean variance)"),
            "sigma_hat": sigma_hat, "mean_variance": var_mean,
            "sigma_theoretical_skewnormal": sigma_theory,
            "note": ("The skewed-normal variance is smaller than the nominal "
                     "scale^2; the simulated sigma (NOT 1.0) is used in all "
                     "formulas.  Std deviation, not variance, enters the charts.")},
        "calibration": {"ewma_limit_le": le, "ewma_arl0_simulated": ewma_arl0,
                        "r_chart": r_cal, "each_target": TARGET_ARL0_EACH,
                        "joint_target": TARGET_ARL0_JOINT,
                        "joint_arl0_simulated": joint_arl0,
                        "reps": cal_reps, "horizon": cal_horizon},
        "lam": LAMBDA_EWMA, "n_obs": N_OBS,
        "true_params": [TRUE_A0, TRUE_A1, TRUE_SIGMA],
        "steady_state_warmup": 10,
        "results": rows,
        "total_seconds": time.perf_counter() - t0,
    }
    return out


def save_outputs(out, out_dir=None):
    out_dir = Path(out_dir) if out_dir else BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"lambda_{out['skewness']:g}"
    (out_dir / f"ewmar_calibration_{tag}.json").write_text(
        json.dumps({k: out[k] for k in ("method", "sigma_estimation",
                                        "calibration", "lam", "n_obs",
                                        "true_params", "steady_state_warmup")},
                   indent=2), encoding="utf-8")
    pd.DataFrame(out["results"]).to_csv(
        out_dir / f"ewmar_results_{tag}.csv", index=False)
    cal = out["calibration"]
    lines = [f"# EWMA/R results (skewness lambda = {out['skewness']:g})", "",
             f"- sigma_hat (simulated): {out['sigma_estimation']['sigma_hat']:.6f}",
             f"- EWMA limit Le: {cal['ewma_limit_le']:.4f} "
             f"(simulated EWMA-only ARL0 = {cal['ewma_arl0_simulated']:.1f}, target 400)",
             f"- R chart: c = {cal['r_chart']['c']:.6f}, "
             f"LCL = {cal['r_chart']['lcl']:.4f}, UCL = {cal['r_chart']['ucl']:.4f} "
             f"(simulated R-only ARL0 = {cal['r_chart']['arl0']:.1f}, target 400)",
             f"- joint in-control ARL (simulated): "
             f"zero = {cal['joint_arl0_simulated']['zero']:.1f}, "
             f"steady = {cal['joint_arl0_simulated']['steady']:.1f} (target 200)",
             ""]
    for r in out["results"]:
        extra = (f" [EWMA {r.get('frac_alarm_ewma', 0):.2f} / "
                 f"R {r.get('frac_alarm_R', 0):.2f}]") if "frac_alarm_ewma" in r else ""
        lines.append(f"- {r['state']} / {r['chart']}: {r.get('scenario', 'IC')} "
                     f"({r['shift_type']} {r['shift_magnitude']:+g}): "
                     f"ARL = {r['arl']:.2f} (MCSE {r['mcse']:.2f}){extra}")
    (out_dir / f"ewmar_summary_{tag}.md").write_text("\n".join(lines),
                                                     encoding="utf-8")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skewness", type=float, default=1.0)
    ap.add_argument("--arl-replications", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--cal-replications", type=int, default=2000)
    ap.add_argument("--cal-horizon", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    out = run_benchmark(a.skewness, a.arl_replications, a.horizon,
                        a.cal_replications, a.cal_horizon, a.seed,
                        quick=a.quick)
    d = save_outputs(out, a.out)
    print(json.dumps(out["calibration"], indent=2))
    print("results written to", str(d).encode("ascii", "backslashreplace").decode())


if __name__ == "__main__":
    main()





def _make_residuals(x_values):
    """Build a residuals-from-true-line function for a given x grid
    (used by the EWMA/R design)."""
    def residuals(errors, kind, magnitude):
        a0, a1, _ = _profile_params(kind, magnitude)
        return a0 + a1 * x_values + errors - TRUE_A0 - TRUE_A1 * x_values
    return residuals


def _residuals(errors, kind, magnitude):
    """Residuals from the TRUE in-control line, including the shift."""
    return _make_residuals(X_VALUES)(errors, kind, magnitude)


# Common N=10 EWMA_3 constants
EWMA3_N_OBS = N_OBS
EWMA3_X_VALUES = X_VALUES
EWMA3_XBAR = XBAR
EWMA3_XC_VALUES = X_VALUES - XBAR
EWMA3_SXX = SXX
EWMA3_B0 = TRUE_A0 + TRUE_A1 * XBAR
EWMA3_B1 = TRUE_A1

def _profile_params3(kind, magnitude):
    """Parameters for the common N=10 EWMA_3 design."""
    if kind == "intercept":
        return EWMA3_B0 + magnitude, EWMA3_B1, TRUE_SIGMA
    if kind == "slope":
        return EWMA3_B0 + magnitude * XBAR, EWMA3_B1 + magnitude, TRUE_SIGMA
    if kind == "variance":
        return EWMA3_B0, EWMA3_B1, TRUE_SIGMA * (1.0 + magnitude)
    return EWMA3_B0, EWMA3_B1, TRUE_SIGMA


def _resid_stats3(errors, kind, magnitude, scheme):
    """EWMA_3 standardized statistics for the CENTERED design:
       u0 = (ybar - B0) * sqrt(n) / sigma_0
       u1 = (a1_hat - B1) * sqrt(Sxx) / sigma_0
    With x centered, Cov(u0, u1) = 0 exactly."""
    a0, a1, _ = _profile_params3(kind, magnitude)
    y = a0 + a1 * EWMA3_XC_VALUES + errors
    ybar = y.mean(axis=-1)
    sxy = (EWMA3_XC_VALUES *
           (y - y.mean(axis=-1, keepdims=True))).sum(axis=-1)
    a1_hat = sxy / EWMA3_SXX
    # The two coefficient charts use the *in-control* error standard
    # deviation sigma_0, not the current profile's residual standard
    # deviation.  Replacing sigma_0 by s_hat self-standardizes the charts
    # and changes the EWMA_3 procedure (in particular, its response to a
    # scale shift).  This is the centered-design form of KMW (2003).
    sig0 = scheme["sigma_hat"]
    u0 = (ybar - EWMA3_B0) * np.sqrt(EWMA3_N_OBS) / sig0
    u1 = (a1_hat - EWMA3_B1) * np.sqrt(EWMA3_SXX) / sig0
    return u0, u1


def _residuals3(errors, kind, magnitude):
    """OLS residuals and MSE ingredients for one centered-design profile.

    KMW's third chart uses MSE from the fitted profile, i.e. SSE/(n-2).
    Residuals from the fixed in-control line would incorrectly make an
    intercept or a slope shift look like a variance shift.
    """
    a0, a1, _ = _profile_params3(kind, magnitude)
    y = a0 + a1 * EWMA3_XC_VALUES + errors
    ybar = y.mean(axis=-1, keepdims=True)
    a1_hat = (EWMA3_XC_VALUES * (y - ybar)).sum(axis=-1) / EWMA3_SXX
    return y - (ybar + a1_hat[..., None] * EWMA3_XC_VALUES)


def _var_stat(resid, scheme):
    """KMW (2003, Eq. 2.30) log(MSE) variance statistic.

    w = log(MSE / sigma_0^2)  (natural log, no df/scale rescaling)

    This matches the convention of the published Woodall EWMA_3 Table 3
    column (33.5 12.7 7.2 ... at skewness 0) and of the reference limit
    le_var = 3.5279 / the UCL in _var_base_and_ucl, which are derived for
    the natural-log statistic.  The in-control reference sigma_0 is fixed
    at its in-control value and is deliberately *not* re-estimated after a
    variance shift.
    """
    mse = resid.var(axis=-1, ddof=2)  # SSE/(n-2) for a simple profile
    sig0 = scheme.get("var_sigma", scheme["sigma_hat"])
    return np.log(mse / sig0 ** 2)


def _var_log_mse_variance() -> float:
    """KMW (2003) Eq. (2.32), with n-2 residual degrees of freedom."""
    q = float(EWMA3_N_OBS - 2)
    return float(2.0 / q + 2.0 / q**2 + 4.0 / (3.0 * q**3)
                 - 16.0 / (15.0 * q**5))


def _var_base_and_ucl(scheme):
    """Return the KMW variance-chart starting value and one-sided UCL."""
    base = float(np.log(scheme.get("var_sigma", scheme["sigma_hat"]) ** 2))
    lam = scheme["lam_var"]
    sd_z = np.sqrt(lam * _var_log_mse_variance() / (2.0 - lam))
    return base, base + scheme["le_var"] * sd_z


def _run_path3(u0, u1, w, z_init, scheme, which):
    """Three-stream EWMA walk.  u0, u1, w: (m, H);  z_init: (m, 3).
    Per-stream smoothing constants: lam_int (intercept), lam_slope,
    lam_var.  Returns (first_alarm (m,), z_after (m, 3), src (m,) str)."""
    m, H = u0.shape
    lam0, lam1, lam2 = (scheme["lam_int"], scheme["lam_slope"],
                        scheme["lam_var"])
    b0 = np.sqrt(lam0 / (2.0 - lam0))
    b1 = np.sqrt(lam1 / (2.0 - lam1))
    b2 = np.sqrt(lam2 / (2.0 - lam2))
    li0 = scheme["le_int"] * b0
    li1 = scheme["le_slope"] * b1
    li2 = scheme["le_var"] * b2
    var_base = _var_base_and_ucl(scheme)[0]
    z0 = z_init[:, 0].astype(float).copy()
    z1 = z_init[:, 1].astype(float).copy()
    z2 = z_init[:, 2].astype(float).copy()
    hit = np.zeros((m, H), dtype=bool)
    src = np.zeros((m, H), dtype="<U5")
    alive = np.ones(m, dtype=bool)
    for t in range(H):
        if which in ("joint3", "int"):
            z0 = lam0 * u0[:, t] + (1.0 - lam0) * z0
            h0 = np.abs(z0) > li0
        else:
            h0 = False
        if which in ("joint3", "slope"):
            z1 = lam1 * u1[:, t] + (1.0 - lam1) * z1
            h1 = np.abs(z1) > li1
        else:
            h1 = False
        if which in ("joint3", "var"):
            # KMW (2003, Eq. 2.30): one-sided UPPER EWMA of the log-variance
            # statistic, lower-clamped at the in-control base log(sigma_0^2).
            # A two-sided chart here would signal roughly twice as fast and
            # produce variance-column ARLs far below the published values
            # (e.g. 11.43 instead of 33.5 at d = 0.2).
            z2 = lam2 * w[:, t] + (1.0 - lam2) * z2
            z2 = np.maximum(z2, var_base)
            h2 = z2 > li2
        else:
            h2 = False
        hit[:, t] = h0 | h1 | h2
        src[:, t] = np.where(h0, "int", np.where(h1, "slope", "var"))
        alive &= ~hit[:, t]
        # Once every path in this chunk has signalled, later EWMA values are
        # unused.  Stopping here leaves the first-alarm times unchanged and
        # substantially reduces the work for large shifts.
        if not alive.any():
            break
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1) + 1, np.inf).astype(float)
    src_first = np.where(any_hit,
                         src[np.arange(m), hit.argmax(axis=1)], "none")
    z_after = np.stack([z0, z1, z2], axis=1)
    return first, z_after, src_first


def _ewma3_inputs(errors, kind, magnitude, scheme, which):
    """Compute only the statistic streams required by a simulation run."""
    shape = errors.shape[:-1]
    if which == "var":
        u0 = np.zeros(shape)
        u1 = np.zeros(shape)
    else:
        u0, u1 = _resid_stats3(errors, kind, magnitude, scheme)
        if which == "int":
            u1.fill(0.0)
        elif which == "slope":
            u0.fill(0.0)
    if which in ("joint3", "var"):
        w = _var_stat(_residuals3(errors, kind, magnitude), scheme)
    else:
        w = np.zeros(shape)
    return u0, u1, w


def simulate_arl3(skewness, kind, magnitude, scheme, state="zero",
                  reps=2000, horizon=5000, seed=SEED + 100,
                  warmup=10, which="joint3", chunk=250):
    """Zero- or steady-state ARL for the KMW EWMA_3 scheme.

    The centered design makes the coefficient streams independent under
    normality.  The variance stream is the one-sided, lower-clamped
    log(MSE) EWMA specified by KMW, not a standardized two-sided chart.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros((reps, 3))
    z[:, 2] = _var_base_and_ucl(scheme)[0]
    if state == "steady":
        alive = np.zeros(reps, dtype=bool)
        while not alive.all():
            idx = np.flatnonzero(~alive)
            m = idx.size
            err = _gen_errors(rng, skewness, (m, warmup),
                              n_obs=EWMA3_N_OBS)
            u0, u1, w = _ewma3_inputs(err, "ic", 0.0, scheme, which)
            z0 = np.zeros((m, 3))
            z0[:, 2] = _var_base_and_ucl(scheme)[0]
            first, z_end, _ = _run_path3(u0, u1, w, z0, scheme, which)
            ok = ~np.isfinite(first)
            alive[idx[ok]] = True
            z[idx[ok]] = z_end[ok]
    first = np.full(reps, np.nan)
    sources: list[str] = []
    scale = _profile_params3(kind, magnitude)[2]
    t_done = 0
    while t_done < horizon and np.isnan(first).any():
        n = min(chunk, horizon - t_done)
        idx = np.flatnonzero(np.isnan(first))
        err = _gen_errors(rng, skewness, (idx.size, n), scale=scale,
                          n_obs=EWMA3_N_OBS)
        u0, u1, w = _ewma3_inputs(err, kind, magnitude, scheme, which)
        f, z_end, src_c = _run_path3(u0, u1, w, z[idx], scheme, which)
        new = np.isfinite(f)
        first[idx[new]] = f[new] + t_done
        sources.extend(src_c[i] for i in np.flatnonzero(new))
        z[idx] = z_end
        t_done += n
    # Anti-censoring retry: runs that never signalled within `horizon` are
    # NOT truncated at the horizon (that would bias the ARL low and can
    # break monotonicity of the shift columns).  Instead, extend the
    # horizon for the still-alive reps until every run finishes or a hard
    # cap (20x horizon) is reached.
    cap = 20 * horizon
    t_ext = t_done
    while np.isnan(first).any() and t_ext < cap:
        n = min(chunk, cap - t_ext)
        idx = np.flatnonzero(np.isnan(first))
        err = _gen_errors(rng, skewness, (idx.size, n), scale=scale,
                          n_obs=EWMA3_N_OBS)
        u0, u1, w = _ewma3_inputs(err, kind, magnitude, scheme, which)
        f, z_end, src_c = _run_path3(u0, u1, w, z[idx], scheme, which)
        new = np.isfinite(f)
        first[idx[new]] = f[new] + t_ext
        sources.extend(src_c[i] for i in np.flatnonzero(new))
        z[idx] = z_end
        t_ext += n
    run_lengths = first
    finite = np.isfinite(run_lengths)
    # Only truly pathological runs (cap exceeded) are truncated at the cap.
    rl_full = np.where(finite, run_lengths, float(cap))
    arl = float(rl_full.mean())
    se = float(rl_full.std(ddof=1) / np.sqrt(reps))
    frac = {s: float(sources.count(s)) / reps
            for s in ("int", "slope", "var", "none")}
    # Censoring fraction: runs still unresolved at the 20x-horizon cap
    # (reported so callers can spot residual bias; normally ~ 0).
    frac["censored"] = float((~finite).mean())
    return arl, se, frac


def calibrate_ewma3_limits(skewness, target=TARGET_ARL0_EACH3, reps=2000,
                           horizon=5000, seed=SEED + 500, lo=0.05, hi=50.0,
                           tol=0.03, lam_int=LAMBDA_EWMA,
                           lam_slope=LAMBDA_EWMA, lam_var=LAMBDA_EWMA):
    """Bisection on each of the three EWMA limits so each chart reaches
    the requested marginal zero-state ARL0 (the other streams disabled).
    Each stream uses its own smoothing constant."""
    sigma_hat = sigma0(skewness)
    limits = {}
    for key, which, lam in (("le_int", "int", lam_int),
                            ("le_slope", "slope", lam_slope),
                            ("le_var", "var", lam_var)):
        lo_, hi_ = lo, hi
        best = None
        for _ in range(24):
            mid = 0.5 * (lo_ + hi_)
            scheme = {"sigma_hat": sigma_hat, "var_sigma": sigma_hat,
                      "le_int": 1e6,
                      "le_slope": 1e6, "le_var": 1e6, key: mid,
                      "lam_int": lam_int, "lam_slope": lam_slope,
                      "lam_var": lam_var}
            arl, _, _ = simulate_arl3(skewness, "ic", 0.0, scheme, "zero",
                                      reps, horizon, seed, which=which)
            best = (mid, arl)
            if abs(arl - target) / target < tol:
                break
            if arl < target:      # too many alarms -> widen limit
                lo_ = mid
            else:
                hi_ = mid
        if abs(best[1] - target) / target >= tol:
            raise RuntimeError(
                f"calibrate_ewma3_limits: stream '{which}' failed to reach "
                f"ARL0 {target} within tol {tol:.0%}: best limit "
                f"{best[0]:.4f} gives ARL0 {best[1]:.1f}.  Widen lo/hi "
                f"bounds or increase reps/horizon.")
        limits[key] = {"limit": float(best[0]), "arl0": float(best[1])}
    return limits


def incontrol_correlations(skewness, scheme, n=200000, seed=SEED + 42):
    """In-control correlations of (u0, u1, log(MSE)) under the centered design."""
    rng = np.random.default_rng(seed)
    err = _gen_errors(rng, skewness, (n,), n_obs=EWMA3_N_OBS)
    u0, u1 = _resid_stats3(err, "ic", 0.0, scheme)
    w = _var_stat(_residuals3(err, "ic", 0.0), scheme)
    c = np.corrcoef(np.stack([u0, u1, w]))
    return {"corr_u0_u1": float(c[0, 1]), "corr_u0_var": float(c[0, 2]),
            "corr_u1_var": float(c[1, 2])}
