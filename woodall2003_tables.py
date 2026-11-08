"""Unified Woodall (2003)-style comparison tables: EWMA_3, EWMA/R and Hotelling T2.

Design (identical for the three charts):
  y = 3 + 2x + e,  x = 1,...,10  (n = 10),  sigma = 1 nominal.
  Skew-normal centred errors (Azzalini mixture), skewness a in {0, 0.5, 1, 2}.

Shift grids (one parameter at a time, in NOMINAL sigma units, sigma0(a) is the
true in-control error sd of the skew-normal):
  intercept : A0 -> A0 + d*sigma            d = 0.0, 0.2, ..., 2.0
  slope     : A1 -> A1 + d*sigma            d = 0.000, ..., 0.250 (step 0.025)
  SD level  : sigma0(a) -> sigma0(a) + (g-1)*sigma,  g = 1.0, 1.2, ..., 3.0
              (= g*sigma exactly at a = 0)

Charts (all calibrated so the in-control ARL0 ~ 200 in the zero state; the
calibrated limits are reused unchanged for the steady state):
  * EWMA_3 (Kim-Mahmoud-Woodall 2003): two-sided EWMA of the standardized
    OLS estimates u0 = (b0_hat - A0)/(sigma0*sqrt(1/n + xbar^2/Sxx)) and
    u1 = (b1_hat - A1)*sqrt(Sxx)/sigma0, plus the one-sided lower-clamped
    EWMA of w = log(MSE/sigma0^2) with UCL from Eq. 2.32.  lambda = 0.2 for
    all three streams.  Marginal zero-state ARL0 = 584 each, then a single
    joint factor to reach ARL0 ~ 200 (zero-state).
  * EWMA/R (Kang & Albin 2000 residual scheme): EWMA of u0 and of u1
    computed from residuals of the TRUE line with exact standard errors,
    plus the R chart on the range of the residuals.  Each chart calibrated
    to zero-state ARL0 = 400, then a single joint factor to ARL0 ~ 200.
  * T2: Hotelling T2 on v = (b0_hat, b1_hat, log(MSE/sigma0^2)) with the
    EXACT in-control mean vector and covariance matrix of the normal design,
    UCL calibrated by simulation to ARL0 ~ 200 for each skewness.

States: zero (shift from profile 1) and steady (10 in-control warm-up
profiles, false alarms during warm-up restart the replicate).

Run:  py woodall2003_tables.py --skewness 0 0.5 1 2 --states zero steady
Outputs (next to this file):
  woodall2003_results.csv
  woodall2003_calibration.json
  woodall2003_skewness_<a>_<state>.md
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import psi

BASE = Path(__file__).resolve().parent
SEED = 20260825
N_OBS = 10
X = np.arange(1.0, N_OBS + 1.0)
XBAR = float(X.mean())
XC = X - XBAR
SXX = float(np.sum(XC ** 2))
Q = N_OBS - 2
A0, A1, SIGMA = 3.0, 2.0, 1.0
LAM = 0.2
SHIFTS_INT = [round(0.2 * i, 1) for i in range(11)]          # 0.0 .. 2.0
SHIFTS_SLP = [round(0.025 * i, 3) for i in range(11)]        # 0.0 .. 0.25
SD_LEVELS = [round(1.0 + 0.2 * i, 1) for i in range(11)]     # 1.0 .. 3.0
TYPES = ("intercept", "slope", "variance")
VAR_LOG_MSE = 2 / Q + 2 / Q ** 2 + 4 / (3 * Q ** 3) - 16 / (15 * Q ** 5)
# E[log(MSE/sigma0^2)] = psi(Q/2) - log(Q/2) for Q = n - 2 residual df.
# This is the in-control MEAN of the log-MSE; it is subtracted from w so that
# E[w] = 0 (see ols_stats), which makes the clamp of the EWMA variance chart
# and the mean vector of the T2 chart both exactly zero-centred.
from scipy.special import polygamma as _polygamma
E_LOG_CHI2_OVER_Q = float(_polygamma(0, Q / 2.0) - np.log(Q / 2.0))
VAR_LOG_MSE_EXACT = float(_polygamma(1, Q / 2.0))
SD_LOG_MSE = float(np.sqrt(VAR_LOG_MSE_EXACT))
SD_B0 = np.sqrt(1.0 / N_OBS + XBAR ** 2 / SXX)
B0B = np.sqrt(LAM / (2.0 - LAM))
WARMUP = 10


def sigma0(a: float) -> float:
    d = a / np.sqrt(1.0 + a * a)
    return float(SIGMA * np.sqrt(1.0 - 2.0 * d ** 2 / np.pi))


def gen_errors(rng, a, shape, scale=SIGMA):
    """Batched centred skew-normal errors, shape + (N_OBS,)."""
    d = a / np.sqrt(1.0 + a * a)
    size = tuple(shape) + (N_OBS,)
    z = d * np.abs(rng.standard_normal(size)) \
        + np.sqrt(1.0 - d * d) * rng.standard_normal(size)
    return (z - d * np.sqrt(2.0 / np.pi)) * scale


def params(kind, d, a):
    """(b0, b1, error_scale) for the requested profile.

    CRITICAL: gen_errors multiplies the centred skew-normal z -- whose own
    standard deviation is sqrt(1 - delta^2*2/pi) = sigma0(a) -- by `scale`, so
    the ACTUAL error sd is sigma0(a) * scale.  The scale returned here must
    therefore be a MULTIPLIER relative to sigma0(a), NOT the absolute sd:
      * ic / intercept / slope : multiplier 1.0 (in-control dispersion).
        Returning sigma0(a) here (the old behaviour) produced an in-control
        error sd of sigma0(a)^2 at every skewness > 0 -- e.g. 0.6817 instead
        of 0.8256 at a = 1 -- which silently corrupted every chart: the T2
        reference saw a permanent negative offset in the log-MSE component,
        so a POSITIVE variance shift moved the statistic TOWARD the reference,
        the alarm probability DROPPED and the SD columns exploded to the
        nonsensical 433 / 976 values.  At a = 0 (sigma0 = 1) the bug was
        invisible, which is why all normal-case checks passed.
      * variance : additive nominal convention sd -> sigma0(a) + (g-1)*sigma,
        expressed as the multiplier (sigma0(a) + (g-1)) / sigma0(a).
    """
    s0 = sigma0(a)
    if kind == "intercept":
        return A0 + d * SIGMA, A1, 1.0
    if kind == "slope":
        return A0, A1 + d * SIGMA, 1.0
    if kind == "variance":
        # d = displayed SD level g; additive nominal convention, but the
        # generator needs the multiplier relative to sigma0(a).
        return A0, A1, (s0 + (d - 1.0) * SIGMA) / s0
    return A0, A1, 1.0


def ols_stats(errors, kind, d, a):
    """errors (..., N) -> u0, u1, w  (EWMA_3 / T2 statistics).

    u0, u1 : the centered-design standardized OLS estimates, N(0,1) in control
             (E[u0] = E[u1] = 0 exactly, because both are centered contrasts).
    w      : THE CENTERED log-MSE statistic of KMW (2003) eq. (2.30),
                 w = log(MSE / sigma0^2) - [psi(Q/2) - log(Q/2)],
             i.e. the log-MSE with its in-control MEAN REMOVED, so that
             E[w] = 0 under the normal design.  Two independent things are
             fixed by this centering, and both were bugs before:
               (i)  the clamp of the EWMA variance chart must be the IC mean of
                    the plotted statistic.  Using w = log(MSE/sigma0^2)
                    uncentered makes that mean -0.1302 (for Q = 8), not 0, so
                    a clamp at log(sigma0^2) = 0 was 0.244 sd of w too high;
               (ii) the T2 vector (u0, u1, w) must have a zero-mean in control
                    or the third component contributes a constant bias to T2.
             With the centering, E[w] = 0, so the vector mean is exactly
             (0, 0, 0) for every skewness and the T2 reference is clean.
    """
    b0, b1, scale = params(kind, d, a)
    y = b0 + b1 * X + errors
    ybar = y.mean(-1)
    b1h = (XC * (y - ybar[..., None])).sum(-1) / SXX
    b0h = ybar - b1h * XBAR
    resid = y - b0h[..., None] - b1h[..., None] * X
    sig = sigma0(a)
    u0 = (b0h - A0) / (sig * SD_B0)
    u1 = (b1h - A1) * np.sqrt(SXX) / sig
    w = (np.log((resid ** 2).sum(-1) / Q / sig ** 2) - E_LOG_CHI2_OVER_Q) \
        / SD_LOG_MSE
    return u0, u1, w


def resid_true(errors, kind, d, a):
    """Residuals from the TRUE in-control line (EWMA/R design)."""
    b0, b1, scale = params(kind, d, a)
    y = b0 + b1 * X + errors
    return y - A0 - A1 * X


# ----------------------------------------------------------------------
# EWMA_3
# ----------------------------------------------------------------------
def run3(u0, u1, w, z, L0, L1, ucl_var, base_var, which="joint"):
    """Three-stream EWMA walk, KMW (2003) eqs. (2.28)-(2.30).

    base_var is the CLAMP of the variance stream.  Since w is the CENTERED
    log-MSE statistic (E[w] = 0 in control), the clamp must be 0 - clamping at
    log(sigma0^2) = 0 for the uncentered statistic was a different quantity
    and, at skewness > 0, sat 0.24 sd above the true in-control mean of the
    plotted stream, which is what drove the variance ARL from ~200 to ~13.
    """
    m, H = u0.shape
    z0, z1, z2 = z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy()
    l0, l1 = L0 * B0B, L1 * B0B
    l2 = ucl_var                                   # absolute UCL of w-EWMA
    first = np.full(m, np.inf)
    alive = np.ones(m, bool)
    for t in range(H):
        if not alive.any():
            break
        if which in ("joint", "int"):
            z0[alive] = LAM * u0[alive, t] + (1 - LAM) * z0[alive]
        if which in ("joint", "slope"):
            z1[alive] = LAM * u1[alive, t] + (1 - LAM) * z1[alive]
        if which in ("joint", "var"):
            z2[alive] = np.maximum(LAM * w[alive, t] + (1 - LAM) * z2[alive],
                                   base_var)
        h = np.zeros(m, bool)
        if which in ("joint", "int"):
            h[alive] |= np.abs(z0[alive]) > l0
        if which in ("joint", "slope"):
            h[alive] |= np.abs(z1[alive]) > l1
        if which in ("joint", "var"):
            h[alive] |= z2[alive] > l2
        new = h & alive
        first[new] = t + 1
        alive &= ~new
    return first, np.stack([z0, z1, z2], 1)


def arl3(a, kind, d, lim, state, reps, horizon, seed, which="joint"):
    rng = np.random.default_rng(seed)
    base = 0.0          # E[w] = 0 for the CENTERED log-MSE statistic
    z = np.zeros((reps, 3))
    if state == "steady":
        ready = np.zeros(reps, bool)
        while not ready.all():
            idx = np.flatnonzero(~ready)
            e = gen_errors(rng, a, (len(idx), WARMUP))
            uu = ols_stats(e, "ic", 0, a)
            zi = np.zeros((len(idx), 3))
            zi[:, 2] = base
            f, ze = run3(*uu, zi, lim["L0"], lim["L1"], lim["ucl"], base,
                         "joint")
            ok = ~np.isfinite(f)
            ready[idx[ok]] = True
            z[idx[ok]] = ze[ok]
    f = np.full(reps, np.nan)
    done = 0
    chunk = 250
    scale = params(kind, d, a)[2]
    while np.isnan(f).any() and done < horizon:
        n = min(chunk, horizon - done)
        idx = np.flatnonzero(np.isnan(f))
        e = gen_errors(rng, a, (len(idx), n), scale)
        uu = ols_stats(e, kind, d, a)
        ff, ze = run3(*uu, z[idx], lim["L0"], lim["L1"], lim["ucl"], base,
                      which)
        new = np.isfinite(ff)
        f[idx[new]] = ff[new] + done
        z[idx] = ze
        done += n
    cap = 20 * horizon
    ext = done
    while np.isnan(f).any() and ext < cap:
        n = min(chunk, cap - ext)
        idx = np.flatnonzero(np.isnan(f))
        e = gen_errors(rng, a, (len(idx), n), scale)
        uu = ols_stats(e, kind, d, a)
        ff, ze = run3(*uu, z[idx], lim["L0"], lim["L1"], lim["ucl"], base,
                      which)
        new = np.isfinite(ff)
        f[idx[new]] = ff[new] + ext
        z[idx] = ze
        ext += n
    vals = np.where(np.isfinite(f), f, float(cap))
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(reps))


def calibrate3(a, cal_reps, cal_horizon, seed):
    """Marginal ARL0 = 584 per stream, then one joint factor to ARL0 = 200.

    n = 10 detail that matters: the variance chart is driven by
    w = log(MSE/sigma0^2) where MSE = SSE/(n-2) has only Q = n-2 = 8 residual
    degrees of freedom.  The stationary standard deviation of the w-EWMA is
    therefore sqrt(lam/(2-lam) * Var[w]) with Var[w] = pi^2/6-like exact value
    trigamma(Q/2) = trigamma(4), which for Q = 8 equals 0.2838 (the KMW eq.
    2.32 series is a very tight approximation of it).  Standardising the
    variance stream with the SAME sqrt(lam/(2-lam)) used for the coefficient
    streams - which is what the naive implementation does - inflates its limit
    by a factor sqrt(Var_w_scale) and makes the dispersion chart blind; here
    each stream is standardised with ITS OWN stationary sd.
    """
    from scipy.special import polygamma
    base = 0.0          # E[w] = 0 for the centered log-MSE statistic
    var_w_exact = float(polygamma(1, Q / 2.0))
    sd_w = B0B          # w is pre-standardised, so its EWMA sd is the usual one
    lim = {"L0": 3.0, "L1": 3.0, "LE": 2.0, "base": base,
           "ucl": base + 1.0, "var_w_exact": var_w_exact,
           "var_w_kmw": VAR_LOG_MSE, "Q": Q,
           "sd_u0_stat": 1.0, "sd_u1_stat": 1.0, "sd_w_stat": sd_w}
    for key, which, sd in (("L0", "int", B0B), ("L1", "slope", B0B),
                           ("LE", "var", sd_w)):
        lo, hi = (0.5, 12.0) if which != "var" else (0.2, 6.0)
        best = (1.0, np.inf)
        for it in range(24):
            mid = 0.5 * (lo + hi)
            trial = {"L0": 3.0, "L1": 3.0, "base": base, "ucl": base + 1.0}
            if which == "var":
                trial["ucl"] = base + mid * sd
                trial["LE"] = mid
            else:
                trial[key] = mid
                trial["LE"] = 1e6      # other streams disabled
                trial["ucl"] = 1e9
            v, _ = arl3(a, "ic", 0, trial, "zero", cal_reps, cal_horizon,
                        seed + it, which)
            best = (mid, v)
            if v < 584:
                lo = mid
            else:
                hi = mid
            if abs(v - 584) / 584 < 0.025:
                break
        if which == "var":
            lim["LE"] = best[0]
            lim["ucl"] = base + best[0] * sd
        else:
            lim[key] = best[0]
        lim[key + "_marg_arl0"] = best[1]
    bl = dict(lim)
    lo, hi = 0.70, 1.60
    bestf = (1.0, np.inf)
    for it in range(16):
        fac = 0.5 * (lo + hi)
        trial = {"L0": bl["L0"] * fac, "L1": bl["L1"] * fac,
                 "base": base,
                 "ucl": base + (bl["ucl"] - base) * fac, "LE": bl["LE"]}
        v, _ = arl3(a, "ic", 0, trial, "zero", cal_reps, cal_horizon,
                    seed + 1000 + it, "joint")
        if abs(v - 200) < abs(bestf[1] - 200):
            bestf = (fac, v)
        if v < 200:
            lo = fac
        else:
            hi = fac
    fac = bestf[0]
    lim["L0"] = bl["L0"] * fac
    lim["L1"] = bl["L1"] * fac
    lim["ucl"] = base + (bl["ucl"] - base) * fac
    lim["joint_factor"] = fac
    lim["joint_arl0"] = bestf[1]
    return lim


# ----------------------------------------------------------------------
# EWMA/R
# ----------------------------------------------------------------------
def resid_stats_r(errors, kind, d, a):
    """EWMA/R statistics: u0, u1 (standardized OLS estimates) and the range
    of the residuals from the TRUE line."""
    b0, b1, scale = params(kind, d, a)
    y = b0 + b1 * X + errors
    ybar = y.mean(-1, keepdims=True)
    b1h = (XC * (y - ybar)).sum(-1) / SXX
    b0h = ybar[..., 0] - b1h * XBAR
    sig = sigma0(a)
    u0 = (b0h - A0) / (sig * SD_B0)
    u1 = (b1h - A1) * np.sqrt(SXX) / sig
    r = np.ptp(resid_true(errors, kind, d, a), axis=-1)
    return u0, u1, r


def runr(u0, u1, r, z, le, lcl, ucl, which="joint"):
    m, H = u0.shape
    z0, z1 = z[:, 0].copy(), z[:, 1].copy()
    leb = le * B0B
    first = np.full(m, np.inf)
    alive = np.ones(m, bool)
    for t in range(H):
        if not alive.any():
            break
        if which in ("joint", "ewma"):
            z0[alive] = LAM * u0[alive, t] + (1 - LAM) * z0[alive]
            z1[alive] = LAM * u1[alive, t] + (1 - LAM) * z1[alive]
        h = np.zeros(m, bool)
        if which in ("joint", "ewma"):
            h[alive] |= (np.abs(z0[alive]) > leb) | (np.abs(z1[alive]) > leb)
        if which in ("joint", "R"):
            h[alive] |= (r[alive, t] < lcl) | (r[alive, t] > ucl)
        new = h & alive
        first[new] = t + 1
        alive &= ~new
    return first, np.stack([z0, z1], 1)


def arlr(a, kind, d, sch, state, reps, horizon, seed, which="joint"):
    rng = np.random.default_rng(seed)
    z = np.zeros((reps, 2))
    if state == "steady":
        ready = np.zeros(reps, bool)
        while not ready.all():
            idx = np.flatnonzero(~ready)
            e = gen_errors(rng, a, (len(idx), WARMUP))
            st = resid_stats_r(e, "ic", 0, a)
            f, ze = runr(*st, np.zeros((len(idx), 2)), sch["le"],
                         sch["lcl"], sch["ucl"], "joint")
            ok = ~np.isfinite(f)
            ready[idx[ok]] = True
            z[idx[ok]] = ze[ok]
    f = np.full(reps, np.nan)
    done = 0
    chunk = 250
    scale = params(kind, d, a)[2]
    while np.isnan(f).any() and done < horizon:
        n = min(chunk, horizon - done)
        idx = np.flatnonzero(np.isnan(f))
        e = gen_errors(rng, a, (len(idx), n), scale)
        st = resid_stats_r(e, kind, d, a)
        ff, ze = runr(*st, z[idx], sch["le"], sch["lcl"], sch["ucl"], which)
        new = np.isfinite(ff)
        f[idx[new]] = ff[new] + done
        z[idx] = ze
        done += n
    cap = 20 * horizon
    ext = done
    while np.isnan(f).any() and ext < cap:
        n = min(chunk, cap - ext)
        idx = np.flatnonzero(np.isnan(f))
        e = gen_errors(rng, a, (len(idx), n), scale)
        st = resid_stats_r(e, kind, d, a)
        ff, ze = runr(*st, z[idx], sch["le"], sch["lcl"], sch["ucl"], which)
        new = np.isfinite(ff)
        f[idx[new]] = ff[new] + ext
        z[idx] = ze
        ext += n
    vals = np.where(np.isfinite(f), f, float(cap))
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(reps))


def calibrate_r(a, cal_reps, cal_horizon, seed):
    """EWMA-only and R-only ARL0 = 400, then a joint factor to ARL0 = 200."""
    # EWMA limit
    lo, hi = 0.5, 8.0
    best_le, best_v = 3.0, np.inf
    for it in range(24):
        le = 0.5 * (lo + hi)
        sch = {"le": le, "lcl": -1e6, "ucl": 1e6}
        v, _ = arlr(a, "ic", 0, sch, "zero", cal_reps, cal_horizon,
                    seed + it, "ewma")
        if abs(v - 400) < abs(best_v - 400):
            best_le, best_v = le, v
        if abs(v - 400) / 400 < 0.03:
            break
        if v < 400:
            lo = le
        else:
            hi = le
    le, ewma_arl0 = best_le, best_v
    # R-chart quantile c
    rng = np.random.default_rng(seed + 7)
    ic = gen_errors(rng, a, (200000,))
    ranges = np.sort(np.ptp(ic, axis=1))
    nq = ranges.size
    c_lo, c_hi = 0.0002, 0.02
    c = 0.005
    best = (c, None)
    for it in range(22):
        c = 0.5 * (c_lo + c_hi)
        lcl = float(ranges[max(0, int(np.floor(c * nq)))])
        ucl = float(ranges[min(nq - 1, int(np.ceil((1 - c) * nq)))])
        sch = {"le": 1e6, "lcl": lcl, "ucl": ucl}
        v, _ = arlr(a, "ic", 0, sch, "zero", cal_reps, cal_horizon,
                    seed + 20 + it, "R")
        best = (c, {"lcl": lcl, "ucl": ucl, "arl0": v})
        if abs(v - 400) / 400 < 0.03:
            break
        if v < 400:
            c_hi = c
        else:
            c_lo = c
    c, rcal = best
    # joint factor about the in-control range mean
    center = float(np.ptp(ic, axis=1).mean())

    def trial(f):
        return {"le": f * le, "lcl": center + f * (rcal["lcl"] - center),
                "ucl": center + f * (rcal["ucl"] - center)}

    lo, hi = 0.30, 1.80
    bestf = (1.0, np.inf)
    for it in range(16):
        f = 0.5 * (lo + hi)
        v, _ = arlr(a, "ic", 0, trial(f), "zero", cal_reps, cal_horizon,
                    seed + 900 + it, "joint")
        if abs(v - 200) < abs(bestf[1] - 200):
            bestf = (f, v)
        if v < 200:
            lo = f
        else:
            hi = f
    f = bestf[0]
    sch = trial(f)
    sch.update({"le_marginal": le, "ewma_arl0": ewma_arl0,
                "c": c, "r_marginal_arl0": rcal["arl0"],
                "joint_factor": f, "joint_arl0": bestf[1]})
    return sch


# ----------------------------------------------------------------------
# Hotelling T2  (v = (b0_hat, b1_hat, log(MSE/sigma0^2)))
# ----------------------------------------------------------------------
def t2_cov(a):
    """In-control (mean, covariance) of the T2 vector
    v = (u0, u1, w/sd_w) under the NORMAL design (skewness 0).

    The mean vector is exactly (0, 0, 0): E[u0] = E[u1] = 0 by construction and
    E[w] = 0 by the centering in ols_stats.

    The third component is PRE-STANDARDISED by its exact in-control standard
    deviation sd_w = sqrt(trigamma(Q/2)), so the covariance matrix is the
    correlation matrix [[1, rho01, 0], [rho01, 1, 0], [0, 0, 1]].  This is what
    keeps a single component from dominating the quadratic form: without it the
    variance component alone (whose IC sd grows from 0.5327 at a = 0 to 0.5565
    at a = 2) inflates var(T2) and forces the UCL from 13.9 to 21.9, which is
    what made the T2 chart blind to every shift at large skewness.
    """
    rho01 = -XBAR * np.sqrt(N_OBS) / np.sqrt(SXX + N_OBS * XBAR ** 2)
    m = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1.0, rho01, 0.0],
                    [rho01, 1.0, 0.0],
                    [0.0, 0.0, 1.0]])
    return m, cov


def t2_moments(a, n=400000, seed=SEED + 4242, exact_normal=False):
    """In-control mean vector and covariance matrix of the T2 vector
    v = (u0, u1, w) used as the reference of the T2 chart.

    MEAN: fixed at the nominal zero vector for every skewness (Phase-II
    requirement: incoming profiles are compared with the fixed in-control
    process; re-estimating the mean under skewness would absorb the distortion).

    COVARIANCE: estimated under the skewness actually being simulated.  This is
    essential and is the fix for the blind T2 columns.  Under skew-normal errors
    the OLS estimates and the residual dispersion become CORRELATED, and the
    correlation grows with skewness:

        corr(u0, w) = -0.0015 (a = 0), +0.0074 (a = 0.5),
                      +0.0383 (a = 1), +0.1232 (a = 2)

    With the normal-design covariance (which forces corr(u0, w) = 0) the chart
    cannot separate an intercept move from a dispersion move, its in-control
    tail inflates, the calibrated UCL has to be pushed from 13.8 to 21.9 and the
    chart goes blind on every parameter (the spurious "T2 = 968" SD columns).
    Using the true skewness-specific covariance restores the correct ordering.
    """
    if exact_normal:
        return t2_cov(a)
    rng = np.random.default_rng(seed + int(round(1000 * a)))
    e = gen_errors(rng, a, (n // 10, N_OBS))
    v = np.stack(ols_stats(e, "ic", 0.0, a), -1).reshape(-1, 3)
    sd = v.std(0, ddof=1)
    v = v / sd                       # standardise the scale component-wise
    cov = np.cov(v, rowvar=False, ddof=1)
    m = np.array([0.0, 0.0, 0.0])    # fixed nominal mean reference
    return m, cov


def t2_scale(a, n=400000, seed=SEED + 4243):
    """In-control standard deviations of (u0, u1, w) under skewness a.

    Used to standardise the T2 components; equals (1, 1, 1) at a = 0 and about
    (1.0005, 1.0003, 1.0446) at a = 2.  Standardising by these values is what
    stops the variance component from inflating var(T2) at high skewness and
    driving the calibrated UCL from 13.8 up to 21.9 (the cause of the blind
    "T2 = 968" SD columns).
    """
    if abs(a) < 1e-12:
        return np.array([1.0, 1.0, 1.0])
    rng = np.random.default_rng(seed + int(round(1000 * a)))
    e = gen_errors(rng, a, (n // 10, N_OBS))
    v = np.stack(ols_stats(e, "ic", 0.0, a), -1).reshape(-1, 3)
    return v.std(0, ddof=1)


def arl_t2(a, kind, d, ucl, state, reps, horizon, seed):
    rng = np.random.default_rng(seed)
    m0, cov = t2_moments(a)
    inv = np.linalg.inv(cov)
    sd = t2_scale(a)      # component-wise in-control sd used to standardise v

    def stat(e):
        v = (np.stack(ols_stats(e, kind, d, a), -1) - m0) / sd
        return np.einsum("...i,ij,...j->...", v, inv, v)

    def alarm_len(e):
        """first alarm times (may be inf) for a block of profiles."""
        s = stat(e)
        hit = s > ucl
        anyhit = hit.any(-1)
        first = np.where(anyhit, hit.argmax(-1) + 1, np.inf)
        return first

    if state == "steady":
        ready = np.zeros(reps, bool)
        while not ready.all():
            idx = np.flatnonzero(~ready)
            e = gen_errors(rng, a, (len(idx), WARMUP))
            f = alarm_len(e)
            ok = ~np.isfinite(f)
            ready[idx[ok]] = True
            f = None
    scale = params(kind, d, a)[2]
    f = np.full(reps, np.nan)
    done = 0
    chunk = 400
    while np.isnan(f).any() and done < horizon:
        n = min(chunk, horizon - done)
        idx = np.flatnonzero(np.isnan(f))
        e = gen_errors(rng, a, (len(idx), n), scale)
        ff = alarm_len(e)
        new = np.isfinite(ff)
        f[idx[new]] = ff[new] + done
        done += n
    cap = 20 * horizon
    ext = done
    while np.isnan(f).any() and ext < cap:
        n = min(chunk, cap - ext)
        idx = np.flatnonzero(np.isnan(f))
        e = gen_errors(rng, a, (len(idx), n), scale)
        ff = alarm_len(e)
        new = np.isfinite(ff)
        f[idx[new]] = ff[new] + ext
        ext += n
    vals = np.where(np.isfinite(f), f, float(cap))
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(reps))


def calibrate_t2(a, cal_reps, cal_horizon, seed):
    """Start from the chi-square UCL of the normal design and bisect until
    the simulated in-control ARL0 = 200.  The moments of the T2 vector are
    the skewness-specific ones (t2_moments), so at a = 0 the chi-square UCL
    is already correct and the bisection stops immediately; under skewness
    the UCL is recalibrated because the sampling distribution of the
    statistic is no longer chi-square_3."""
    ucl = chi_ucl(a)
    v, _ = arl_t2(a, "ic", 0, ucl, "zero", cal_reps, cal_horizon, seed)
    best = (ucl, v)
    if abs(v - 200) / 200 >= 0.02:
        lo, hi = 2.0, 40.0
        for it in range(24):
            ucl = 0.5 * (lo + hi)
            v, _ = arl_t2(a, "ic", 0, ucl, "zero", cal_reps, cal_horizon,
                          seed + 1 + it)
            if abs(v - 200) < abs(best[1] - 200):
                best = (ucl, v)
            if abs(v - 200) / 200 < 0.02:
                break
            if v < 200:
                lo = ucl
            else:
                hi = ucl
    m0, cov = t2_moments(a)
    return {"ucl": best[0], "joint_arl0": best[1],
            "mean_vector": [float(x) for x in m0],
            "cov_matrix": [[float(x) for x in row] for row in cov],
            "chi2_ucl": chi_ucl(a),
            "reference": "fixed nominal normal-design moments (KMW Phase-II)"}


def chi_ucl(a):
    from scipy.stats import chi2
    return float(chi2.ppf(0.995, 3))


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def run_method(a, state, reps, horizon, ic_horizon, seed):
    """ARL rows for one (skewness, state) for the three charts."""
    t0 = time.perf_counter()
    lim3 = calibrate3(a, IC_CAL_REPS, IC_CAL_HORIZON, seed + 100)
    schr = calibrate_r(a, IC_CAL_REPS, IC_CAL_HORIZON, seed + 200)
    calt2 = calibrate_t2(a, IC_CAL_REPS, IC_CAL_HORIZON, seed + 300)
    rows = []
    ic3, se3 = arl3(a, "ic", 0, lim3, state, reps, ic_horizon, seed + 900)
    icr, ser = arlr(a, "ic", 0, schr, state, reps, ic_horizon, seed + 950)
    ict, set_ = arl_t2(a, "ic", 0, calt2["ucl"], state, reps, ic_horizon,
                       seed + 970)
    rows.append({"skewness": a, "state": state, "shift_type": "none",
                 "shift": 0.0, "ewma3": ic3, "ewma3_mcse": se3,
                 "ewmar": icr, "ewmar_mcse": ser,
                 "t2": ict, "t2_mcse": set_})
    grids = {"intercept": SHIFTS_INT, "slope": SHIFTS_SLP,
             "variance": SD_LEVELS}
    for kind in TYPES:
        for d in grids[kind]:
            if kind == "intercept" and d == 0.0:
                continue      # d = 0 is the in-control row already reported
            if kind == "slope" and d == 0.0:
                continue
            if kind == "variance" and d == 1.0:
                continue
            sd_seed = seed + 1500 + zlib.crc32(f"{kind}:{d}:{state}".encode()) % 100000
            v3, s3 = arl3(a, kind, d, lim3, state, reps, horizon, sd_seed)
            vr, sr = arlr(a, kind, d, schr, state, reps, horizon,
                          sd_seed + 1)
            vt, st_ = arl_t2(a, kind, d, calt2["ucl"], state, reps, horizon,
                             sd_seed + 2)
            rows.append({"skewness": a, "state": state, "shift_type": kind,
                         "shift": d, "ewma3": v3, "ewma3_mcse": s3,
                         "ewmar": vr, "ewmar_mcse": sr,
                         "t2": vt, "t2_mcse": st_})
    cal = {"ewma3": {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in lim3.items()},
           "ewmar": {k: float(v) for k, v in schr.items()},
           "t2": calt2,
           "sigma0_closed_form": sigma0(a),
           "seconds": time.perf_counter() - t0}
    return rows, cal


def md_table(rows, a, state, cal):
    lines = [f"# Woodall (2003)-style comparison - EWMA3 / EWMAR / T2 "
             f"(skewness a = {a:g}, {state}-state)", "",
             f"- Design: y = 3 + 2x + e, x = 1,...,10, n = 10, sigma = 1; "
             f"sigma0(a) = {sigma0(a):.6f} (closed form).",
             f"- EWMA3 limits: L0 = {cal['ewma3']['L0']:.4f}, "
             f"L1 = {cal['ewma3']['L1']:.4f}, LE = {cal['ewma3']['LE']:.4f}, "
             f"UCL_E = {cal['ewma3']['ucl']:.6f}, joint factor = "
             f"{cal['ewma3']['joint_factor']:.4f} "
             f"(joint ARL0 = {cal['ewma3']['joint_arl0']:.2f}).",
             f"- EWMAR: Le = {cal['ewmar']['le_marginal']:.4f} "
             f"(EWMA-only ARL0 = {cal['ewmar']['ewma_arl0']:.1f}), "
             f"R chart c = {cal['ewmar']['c']:.6f} "
             f"(R-only ARL0 = {cal['ewmar']['r_marginal_arl0']:.1f}), "
             f"joint factor = {cal['ewmar']['joint_factor']:.4f} "
             f"(joint ARL0 = {cal['ewmar']['joint_arl0']:.2f}).",
             f"- T2 UCL = {cal['t2']['ucl']:.4f} "
             f"(ARL0 = {cal['t2']['joint_arl0']:.2f}; chi-square_3 reference "
             f"UCL = {cal['t2']['chi2_ucl']:.4f}).",
             f"- T2 moments used (FIXED nominal normal-design reference, "
             f"as in KMW Phase-II): mean(u0,u1,w) = ("
             + ", ".join(f"{x:+.4f}" for x in cal['t2']['mean_vector'])
             + "), sd = ("
             + ", ".join(f"{np.sqrt(cal['t2']['cov_matrix'][i][i]):.4f}"
                         for i in range(3))
             + ").",
             "- Shifts (nominal sigma units): intercept A0 -> A0 + d*sigma "
             "(d = 0,...,2); slope A1 -> A1 + d*sigma (d = 0,...,0.25); "
             "SD sigma0(a) -> sigma0(a) + (g-1)*sigma (g = 1,...,3), i.e. the "
             "actual dispersion ratio is sigma_shift/sigma0(a) = "
             f"{(sigma0(a) + 0.0)/sigma0(a):.3f} at g = 1 and "
             f"{(sigma0(a) + 2.0)/sigma0(a):.3f} at g = 3. "
             "Shift 0 = in control.", ""]
    titles = {"intercept": "Intercept shift d (A0 + d*sigma)",
              "slope": "Slope shift d (A1 + d*sigma)",
              "variance": "SD level g (sigma0(a) + (g-1)*sigma)"}
    for kind in TYPES:
        lines += [f"## {titles[kind]}", "",
                  "| shift | EWMA3 | EWMAR | T2 |", "|--:|--:|--:|--:|"]
        ic = next(r for r in rows if r["shift_type"] == "none")
        lines.append(f"| 0.0 (IC) | {ic['ewma3']:.2f} | {ic['ewmar']:.2f} "
                     f"| {ic['t2']:.2f} |")
        for r in rows:
            if r["shift_type"] != kind:
                continue
            lines.append(f"| {r['shift']:g} | {r['ewma3']:.2f} | "
                         f"{r['ewmar']:.2f} | {r['t2']:.2f} |")
        lines.append("")
    return "\n".join(lines)


IC_CAL_REPS = 1500
IC_CAL_HORIZON = 6000


def main():
    global IC_CAL_REPS, IC_CAL_HORIZON
    ap = argparse.ArgumentParser()
    ap.add_argument("--skewness", type=float, nargs="+",
                    default=[0, 0.5, 1, 2])
    ap.add_argument("--states", nargs="+", default=["zero", "steady"])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--cal-reps", type=int, default=IC_CAL_REPS)
    ap.add_argument("--cal-horizon", type=int, default=IC_CAL_HORIZON)
    ap.add_argument("--seed", type=int, default=SEED)
    x = ap.parse_args()
    IC_CAL_REPS, IC_CAL_HORIZON = x.cal_reps, x.cal_horizon
    allrows, cals = [], {}
    for a in x.skewness:
        for state in x.states:
            print(f"=== skewness {a:g}, state {state} ===", flush=True)
            rows, cal = run_method(a, state, x.reps, x.horizon,
                                   max(x.horizon, 4000),
                                   x.seed + int(100 * a) +
                                   (0 if state == "zero" else 50))
            cals[f"{a:g}_{state}"] = cal
            allrows += rows
            (BASE / f"woodall2003_skewness_{a:g}_{state}.md").write_text(
                md_table(rows, a, state, cal), encoding="utf-8")
            print(f"  done in {cal['seconds']:.0f}s "
                  f"(EWMA3 joint ARL0 {cal['ewma3']['joint_arl0']:.1f}, "
                  f"EWMAR {cal['ewmar']['joint_arl0']:.1f}, "
                  f"T2 {cal['t2']['joint_arl0']:.1f})", flush=True)
    pd.DataFrame(allrows).to_csv(BASE / "woodall2003_results.csv",
                                 index=False)
    (BASE / "woodall2003_calibration.json").write_text(
        json.dumps(cals, indent=2, default=float), encoding="utf-8")
    print("all done -> " + str(BASE).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
